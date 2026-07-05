"""Unit tests for the indexer — a deterministic fake embed function keeps them offline."""

import asyncio
import hashlib as _hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_recall_mcp import indexer

SESSION_SAMPLE = """# Session: 2026-06-27

## What We Are Building

An MCP server for local LLM access. Phase 1 implemented the ask tool with detailed notes here.

## What WORKED (with evidence)

- The ask tool worked end to end. Confirmed a full response round-trip with evidence recorded.

## What Did NOT Work (and why)

- Writing mcpServers directly into settings.json failed schema validation. Use the CLI registration command instead.

## Decisions Made

- Registered the server at global scope so every project can use it, instead of per-project scope.

## Blockers & Open Questions

- The code-focused model gives inaccurate general-knowledge answers. Need to consider model switching.
"""


async def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 8-dim fake vectors: same text -> same vector."""
    out = []
    for t in texts:
        h = _hashlib.sha256(t.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:8]])
    return out


LONG_A = "## Section A\n\n" + "Body of A, long enough for the incremental-update test. " * 5
LONG_B = "## Section B\n\n" + "Body of B, long enough for the deletion test. " * 5


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "sample-session.tmp"
        self.path.write_text(SESSION_SAMPLE, encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_chunk_count_and_keys(self):
        chunks = indexer.chunk_file(self.path)
        self.assertEqual(len(chunks), 5)
        for c in chunks:
            self.assertIn("title", c)
            self.assertIn("content", c)
            self.assertIn("section_type", c)

    def test_section_type_normalization(self):
        chunks = indexer.chunk_file(self.path)
        types_by_title = {c["title"]: c["section_type"] for c in chunks}
        self.assertEqual(types_by_title["What WORKED (with evidence)"], "worked")
        self.assertEqual(types_by_title["What Did NOT Work (and why)"], "failed")
        self.assertEqual(types_by_title["Decisions Made"], "decision")
        self.assertEqual(types_by_title["Blockers & Open Questions"], "blocker")
        self.assertEqual(types_by_title["What We Are Building"], "other")

    def test_custom_section_rules(self):
        rules = [("lessons", "lesson")]
        p = Path(self.tmpdir.name) / "custom.md"
        p.write_text(
            "## Lessons Learned\n\n" + "A lesson repeated to pass the minimum length. " * 3,
            encoding="utf-8",
        )
        chunks = indexer.chunk_file(p, rules)
        self.assertEqual(chunks[0]["section_type"], "lesson")

    def test_short_chunks_dropped(self):
        p = Path(self.tmpdir.name) / "short.md"
        p.write_text("## A\n\ntiny\n\n## B\n\n" + "A long enough body. " * 20, encoding="utf-8")
        chunks = indexer.chunk_file(p)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["title"], "B")

    def test_file_without_headings_is_one_chunk(self):
        p = Path(self.tmpdir.name) / "plain.md"
        p.write_text("Just a flat note without any headings. " * 3, encoding="utf-8")
        chunks = indexer.chunk_file(p)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_type"], "other")


class TestSyncIndex(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.src = Path(self.tmpdir.name) / "src"
        self.src.mkdir()
        self.index = Path(self.tmpdir.name) / "index"
        self.sources = [(self.src, "*.md")]

    def tearDown(self):
        self.tmpdir.cleanup()

    def _sync(self):
        return asyncio.run(indexer.sync_index(self.index, self.sources, fake_embed))

    def test_initial_build(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        stats = self._sync()
        self.assertEqual(stats["added_or_updated"], 1)
        self.assertEqual(stats["total_chunks"], 1)
        vectors = np.load(self.index / "vectors.npy")
        self.assertEqual(vectors.shape[0], 1)

    def test_no_change_no_reembed(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        self._sync()
        stats = self._sync()
        self.assertEqual(stats["added_or_updated"], 0)
        self.assertEqual(stats["removed"], 0)

    def test_file_added(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        self._sync()
        (self.src / "b.md").write_text(LONG_B, encoding="utf-8")
        stats = self._sync()
        self.assertEqual(stats["added_or_updated"], 1)
        self.assertEqual(stats["total_chunks"], 2)

    def test_file_removed_purges_chunks(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        (self.src / "b.md").write_text(LONG_B, encoding="utf-8")
        self._sync()
        (self.src / "b.md").unlink()
        stats = self._sync()
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(stats["total_chunks"], 1)
        chunks = json.loads((self.index / "chunks.json").read_text(encoding="utf-8"))
        self.assertNotIn("b.md", {c["source"] for c in chunks})

    def test_vectors_stay_aligned_with_chunks(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        (self.src / "b.md").write_text(LONG_B, encoding="utf-8")
        self._sync()
        (self.src / "a.md").unlink()
        self._sync()
        chunks = json.loads((self.index / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(self.index / "vectors.npy")
        self.assertEqual(len(chunks), vectors.shape[0])
        expected = asyncio.run(fake_embed([chunks[0]["content"]]))[0]
        np.testing.assert_allclose(vectors[0], np.asarray(expected, dtype=np.float32), rtol=1e-5)

    def test_sourcespec_sources_work(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        spec_sources = [indexer.SourceSpec(base=self.src, pattern="*.md")]
        stats = asyncio.run(indexer.sync_index(self.index, spec_sources, fake_embed))
        self.assertEqual(stats["added_or_updated"], 1)
        self.assertEqual(stats["total_chunks"], 1)

    def test_corrupted_vectors_triggers_full_rebuild(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        self._sync()
        (self.index / "vectors.npy").write_bytes(b"broken")
        stats = self._sync()
        self.assertEqual(stats["added_or_updated"], 1)
        vectors = np.load(self.index / "vectors.npy")
        self.assertEqual(vectors.shape[0], 1)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.src = Path(self.tmpdir.name) / "src"
        self.src.mkdir()
        self.index = Path(self.tmpdir.name) / "index"
        self.sources = [(self.src, "*.tmp")]
        (self.src / "s1.tmp").write_text(SESSION_SAMPLE, encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _search(self, query, **kw):
        return asyncio.run(indexer.search_memory(
            query, index_dir=self.index, sources=self.sources, embed_fn=fake_embed, **kw
        ))

    def test_exact_content_query_hits_itself(self):
        # fake_embed is deterministic, so querying with a chunk's own body scores 1.0
        chunks = indexer.chunk_file(self.src / "s1.tmp")
        failed_chunk = next(c for c in chunks if c["section_type"] == "failed")
        result = self._search(failed_chunk["content"], top_k=1)
        self.assertIn("settings.json", result)
        self.assertIn("score=1.000", result)

    def test_section_filter_only_returns_that_type(self):
        chunks = indexer.chunk_file(self.src / "s1.tmp")
        decision_chunk = next(c for c in chunks if c["section_type"] == "decision")
        result = self._search(decision_chunk["content"], top_k=5, section_filter="decision")
        self.assertIn("global scope", result)
        self.assertNotIn("settings.json", result)

    def test_unknown_filter_lists_available_types(self):
        result = self._search("anything", section_filter="nonexistent")
        self.assertIn("No chunks with section_type=nonexistent", result)
        self.assertIn("decision", result)

    def test_empty_index_message(self):
        empty_src = Path(self.tmpdir.name) / "empty"
        empty_src.mkdir()
        result = asyncio.run(indexer.search_memory(
            "anything", index_dir=Path(self.tmpdir.name) / "idx2",
            sources=[(empty_src, "*.md")], embed_fn=fake_embed,
        ))
        self.assertIn("index is empty", result)


def make_counting_embed():
    """fake_embed wrapper that records every batch passed to it."""
    calls: list[list[str]] = []

    async def _embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return await fake_embed(texts)

    return _embed, calls


CSV_V1 = "item,price\napple,120\nbanana,80\n"
CSV_V2 = "item,price\napple,120\nbanana,80\ncherry,300\n"


class TestCsvSyncAndEmbedReuse(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.src = Path(self.tmpdir.name) / "src"
        self.src.mkdir()
        self.index = Path(self.tmpdir.name) / "index"
        self.csv_spec = indexer.SourceSpec(base=self.src, pattern="*.csv", type="csv")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_csv_rows_become_chunks(self):
        (self.src / "b.csv").write_text(CSV_V1, encoding="utf-8")
        stats = asyncio.run(indexer.sync_index(self.index, [self.csv_spec], fake_embed))
        self.assertEqual(stats["total_chunks"], 2)
        chunks = json.loads((self.index / "chunks.json").read_text(encoding="utf-8"))
        self.assertEqual({c["section_type"] for c in chunks}, {"csv"})

    def test_append_embeds_only_new_rows(self):
        (self.src / "b.csv").write_text(CSV_V1, encoding="utf-8")
        embed, calls = make_counting_embed()
        asyncio.run(indexer.sync_index(self.index, [self.csv_spec], embed))
        self.assertEqual(sum(len(b) for b in calls), 2)  # 初回は全行
        (self.src / "b.csv").write_text(CSV_V2, encoding="utf-8")
        asyncio.run(indexer.sync_index(self.index, [self.csv_spec], embed))
        appended = [t for b in calls[1:] for t in b]
        self.assertEqual(len(appended), 1)  # 追記1行だけ埋め込み
        self.assertIn("cherry", appended[0])
        vectors = np.load(self.index / "vectors.npy")
        self.assertEqual(vectors.shape[0], 3)

    def test_md_edit_reuses_unchanged_sections(self):
        (self.src / "a.md").write_text(LONG_A + "\n" + LONG_B, encoding="utf-8")
        md_spec = indexer.SourceSpec(base=self.src, pattern="*.md")
        embed, calls = make_counting_embed()
        asyncio.run(indexer.sync_index(self.index, [md_spec], embed))
        self.assertEqual(sum(len(b) for b in calls), 2)  # Section A + B
        changed_b = "## Section B\n\n" + "Rewritten body of B for the reuse test. " * 5
        (self.src / "a.md").write_text(LONG_A + "\n" + changed_b, encoding="utf-8")
        asyncio.run(indexer.sync_index(self.index, [md_spec], embed))
        re_embedded = [t for b in calls[1:] for t in b]
        self.assertEqual(len(re_embedded), 1)  # Section A はベクトル再利用
        self.assertIn("Rewritten", re_embedded[0])

    def test_mixed_md_and_csv_sources(self):
        (self.src / "a.md").write_text(LONG_A, encoding="utf-8")
        (self.src / "b.csv").write_text(CSV_V1, encoding="utf-8")
        md_spec = indexer.SourceSpec(base=self.src, pattern="*.md")
        stats = asyncio.run(indexer.sync_index(self.index, [md_spec, self.csv_spec], fake_embed))
        self.assertEqual(stats["total_chunks"], 3)


if __name__ == "__main__":
    unittest.main()
