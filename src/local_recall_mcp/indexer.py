"""Incremental semantic index over local markdown/text files.

Embeddings are served by a local Ollama instance; the index is three flat
files (manifest.json, chunks.json, vectors.npy) — no external database.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import numpy as np

from . import csv_source

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]
SectionRule = tuple[str, str]  # (lowercase substring of a heading, section type)

@dataclass(frozen=True)
class SourceSpec:
    """One entry of the 'sources' config. type='csv' enables row-level chunking."""
    base: Path
    pattern: str
    type: str = "text"          # "text" | "csv"
    encoding: str = "utf-8"
    skip_rows: int = 0          # lines to skip before the CSV header row
    template: str | None = None # optional row-rendering template, e.g. "{date} {store} {amount}"


def _normalize_sources(sources: list) -> list[SourceSpec]:
    """Accept both SourceSpec and legacy (Path, pattern) tuples."""
    return [
        s if isinstance(s, SourceSpec) else SourceSpec(base=Path(s[0]), pattern=str(s[1]))
        for s in sources
    ]


HEADER_PATTERN = re.compile(r"(?=^#{2,3} .+$)", re.MULTILINE)
MIN_CHUNK_CHARS = 50
MAX_CHUNK_DISPLAY = 600  # cap chunk body in results to save agent context
EMBED_BATCH_SIZE = 32  # keeps the first full build under Ollama's timeout

# Defaults match the session-log convention used by /save-session in
# everything-claude-code ("What WORKED", "Decisions Made", ...).
# Override per-source-layout via the `section_rules` config key.
DEFAULT_SECTION_RULES: list[SectionRule] = [
    ("what worked", "worked"),
    ("what did not work", "failed"),
    ("decisions made", "decision"),
    ("blockers", "blocker"),
]


def normalize_section_type(title: str, rules: list[SectionRule] = DEFAULT_SECTION_RULES) -> str:
    t = title.lower()
    for key, stype in rules:
        if key in t:
            return stype
    return "other"


def chunk_file(path: Path, rules: list[SectionRule] = DEFAULT_SECTION_RULES) -> list[dict[str, Any]]:
    """Split a file on ##/### headings; files without headings become one chunk."""
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for sec in HEADER_PATTERN.split(text):
        sec = sec.strip()
        if len(sec) < MIN_CHUNK_CHARS:
            continue
        title = sec.split("\n")[0].lstrip("#").strip()
        chunks.append({
            "title": title,
            "content": sec,
            "section_type": normalize_section_type(title, rules),
        })
    return chunks


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _spec_fingerprint(spec: SourceSpec) -> str:
    """Chunking-relevant spec fields, folded into the manifest hash so that
    config changes (template, encoding, skip_rows) re-index affected files."""
    raw = repr((spec.type, spec.encoding, spec.skip_rows, spec.template))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def scan_sources(sources: list) -> dict[str, tuple[str, SourceSpec]]:
    """Map absolute path -> (SHA-256, source spec) for every matching file.

    Unreadable files and missing directories are silently skipped.
    """
    result: dict[str, tuple[str, SourceSpec]] = {}
    for spec in _normalize_sources(sources):
        if not spec.base.exists():
            continue
        for p in sorted(spec.base.glob(spec.pattern)):
            if not p.is_file():
                continue
            try:
                result[str(p.resolve())] = (f"{_file_hash(p)}:{_spec_fingerprint(spec)}", spec)
            except OSError:
                continue
    return result


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default  # corrupted -> fall back to a full rebuild


async def sync_index(
    index_dir: Path,
    sources: list[tuple[Path, str]],
    embed_fn: EmbedFn,
    rules: list[SectionRule] = DEFAULT_SECTION_RULES,
) -> dict[str, int]:
    """Re-embed only files whose hash changed; purge chunks of deleted files."""
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = index_dir / "manifest.json"
    chunks_p = index_dir / "chunks.json"
    vectors_p = index_dir / "vectors.npy"

    manifest: dict[str, str] = _load_json(manifest_p, {})
    chunks: list[dict[str, Any]] = _load_json(chunks_p, [])
    try:
        vectors = np.load(vectors_p) if vectors_p.exists() else np.zeros((0, 0), dtype=np.float32)
    except (OSError, ValueError, EOFError):
        # physically corrupted vectors.npy -> discard manifest, full rebuild
        manifest, chunks = {}, []
        vectors = np.zeros((0, 0), dtype=np.float32)
    if vectors.shape[0] != len(chunks):
        # inconsistent state -> full rebuild
        manifest, chunks = {}, []
        vectors = np.zeros((0, 0), dtype=np.float32)

    current = scan_sources(sources)
    current_hashes = {p: h for p, (h, _) in current.items()}
    changed = [p for p, h in current_hashes.items() if manifest.get(p) != h]
    removed = [p for p in manifest if p not in current_hashes]
    stats = {"added_or_updated": len(changed), "removed": len(removed), "total_chunks": len(chunks)}
    if not changed and not removed:
        return stats

    # Reuse embeddings of chunks whose text is unchanged (e.g. append-only CSVs):
    # map old chunk-body hash -> its vector before anything is dropped.
    old_vec_by_hash: dict[str, np.ndarray] = {}
    if chunks and vectors.shape[0] == len(chunks):
        for i, c in enumerate(chunks):
            old_vec_by_hash.setdefault(_text_hash(c["content"]), vectors[i])

    drop = set(changed) | set(removed)
    keep = [i for i, c in enumerate(chunks) if c["path"] not in drop]
    chunks = [chunks[i] for i in keep]
    vectors = vectors[keep] if vectors.shape[0] else vectors

    new_chunks: list[dict[str, Any]] = []
    for p in changed:
        spec = current[p][1]
        if spec.type == "csv":
            file_chunks = csv_source.chunk_csv_file(
                Path(p),
                encoding=spec.encoding,
                skip_rows=spec.skip_rows,
                template=spec.template,
            )
        else:
            file_chunks = chunk_file(Path(p), rules)
        for c in file_chunks:
            c["path"] = p
            c["source"] = Path(p).name
            new_chunks.append(c)
    if new_chunks:
        pending = [c for c in new_chunks if _text_hash(c["content"]) not in old_vec_by_hash]
        embedded = iter(await embed_fn([c["content"] for c in pending]) if pending else [])
        rows = [
            old_vec_by_hash[h] if (h := _text_hash(c["content"])) in old_vec_by_hash
            else np.asarray(next(embedded), dtype=np.float32)
            for c in new_chunks
        ]
        new_vecs = np.vstack(rows).astype(np.float32)
        vectors = new_vecs if vectors.shape[0] == 0 else np.vstack([vectors, new_vecs])
        chunks.extend(new_chunks)

    manifest_p.write_text(json.dumps(current_hashes, ensure_ascii=False), encoding="utf-8")
    chunks_p.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(vectors_p, vectors if vectors.shape[0] else np.zeros((0, 0), dtype=np.float32))
    stats["total_chunks"] = len(chunks)
    return stats


def _cosine_top_k(query_vec: np.ndarray, vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    scores = v @ q
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


async def search_memory(
    query: str,
    top_k: int = 5,
    section_filter: str | None = None,
    *,
    index_dir: Path,
    sources: list[tuple[Path, str]],
    embed_fn: EmbedFn,
    rules: list[SectionRule] = DEFAULT_SECTION_RULES,
) -> str:
    stats = await sync_index(index_dir, sources, embed_fn, rules)
    chunks: list[dict[str, Any]] = _load_json(index_dir / "chunks.json", [])
    if not chunks:
        return "The index is empty: no searchable files found in the configured sources."
    vectors = np.load(index_dir / "vectors.npy")

    if section_filter:
        idx = [i for i, c in enumerate(chunks) if c["section_type"] == section_filter]
        if not idx:
            available = sorted({c["section_type"] for c in chunks})
            return (
                f"No chunks with section_type={section_filter}. "
                f"Available types: {', '.join(available)}."
            )
        chunks = [chunks[i] for i in idx]
        vectors = vectors[idx]

    q_vec = np.asarray((await embed_fn([query]))[0], dtype=np.float32)
    hits = _cosine_top_k(q_vec, vectors, top_k)

    lines = [
        f"{len(hits)} results "
        f"({stats['total_chunks']} chunks total / {stats['added_or_updated']} files re-indexed / "
        f"{stats['removed']} files removed)"
    ]
    for rank, (i, score) in enumerate(hits, 1):
        c = chunks[i]
        body = c["content"][:MAX_CHUNK_DISPLAY]
        lines.append(
            f"\n[{rank}] score={score:.3f} | {c['source']} | {c['section_type']} | {c['title']}\n{body}"
        )
    return "\n".join(lines)


def make_ollama_embed(base_url: str, model: str, timeout: int) -> EmbedFn:
    async def _embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[i:i + EMBED_BATCH_SIZE]
                resp = await client.post(
                    f"{base_url}/api/embed",
                    json={"model": model, "input": batch},
                )
                if resp.status_code == 404:
                    raise ValueError(
                        f"Embedding model '{model}' not found. Run: ollama pull {model}"
                    )
                resp.raise_for_status()
                embs = resp.json().get("embeddings")
                if not embs:
                    raise ValueError(f"Failed to get embeddings: {resp.text[:200]}")
                out.extend(embs)
        return out
    return _embed
