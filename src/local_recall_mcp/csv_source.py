"""Row-level chunking for CSV sources: one record = one searchable chunk.

Generic by design — column names, encodings, and preamble layouts are all
declared in the user's config; nothing vendor-specific lives here.
"""

import csv
from pathlib import Path
from typing import Any

MAX_TITLE_CHARS = 60


class _BlankOnMissing(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render_row(row: dict[str, Any], template: str | None) -> str:
    """Render one CSV row as natural text for embedding.

    Without a template, mirrors the LangChain CSVLoader convention:
    "column: value | column: value" for every non-empty column.
    """
    clean = {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None and v is not None}
    if template:
        return template.format_map(_BlankOnMissing(clean)).strip()
    return " | ".join(f"{k}: {v}" for k, v in clean.items() if k and v)


def chunk_csv_file(
    path: Path,
    *,
    encoding: str = "utf-8",
    skip_rows: int = 0,
    template: str | None = None,
) -> list[dict[str, Any]]:
    """One chunk per data row. No minimum-length filter: single transactions
    are short by nature. The file stem is appended as retrieval context."""
    chunks: list[dict[str, Any]] = []
    with open(path, newline="", encoding=encoding, errors="replace") as f:
        for _ in range(skip_rows):
            f.readline()
        for row in csv.DictReader(f):
            text = render_row(row, template)
            if not text:
                continue
            chunks.append({
                "title": text[:MAX_TITLE_CHARS],
                "content": f"{text}（{path.stem}）",
                "section_type": "csv",
            })
    return chunks
