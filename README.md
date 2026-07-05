# local-recall-mcp

**Fully local long-term memory for AI agents.** Semantic search over your notes and session logs from any MCP client — embeddings served by [Ollama](https://ollama.com), so nothing ever leaves your machine.

Your agent forgets everything between sessions. Your session logs and notes already contain the answers — what worked, what failed, what you decided and why. `local-recall-mcp` turns those files into a searchable memory the agent can query before repeating old mistakes.

- 🔒 **100% local** — no cloud APIs, no keys, no telemetry. Ollama does the embeddings
- 🪶 **One tool, tiny footprint** — a single `search_memory` tool, so it barely costs any agent context
- ⚡ **Incremental indexing** — SHA-256 manifest re-embeds only changed files, purges deleted ones, and self-heals from a corrupted index
- 🏷️ **Section-type filtering** — map your headings (e.g. `What Did NOT Work`) to types like `failed`, then search only past failures
- 📦 **No database** — the whole index is three flat files (`manifest.json`, `chunks.json`, `vectors.npy`)

## Quickstart

**1. Get Ollama and the embedding model** (~1.2 GB, multilingual):

```bash
ollama pull bge-m3
```

**2. Create a config** at `~/.local-recall/config.yaml`:

```yaml
ollama:
  base_url: http://localhost:11434
  embed_model: bge-m3
  embed_timeout: 300

index_dir: ~/.local-recall/index

sources:
  - path: ~/notes
    pattern: "**/*.md"
```

**3. Register the server** with your MCP client. For Claude Code:

```bash
claude mcp add recall -- uvx local-recall-mcp
```

For Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "recall": {
      "command": "uvx",
      "args": ["local-recall-mcp"]
    }
  }
}
```

**4. Ask your agent** things like *"search memory for how we fixed the MCP connection issue"*. The first query builds the index; later queries re-embed only what changed.

## Presets

Ready-made configs in [`configs/`](configs/):

| Preset | What it indexes |
| --- | --- |
| [`claude-code.yaml`](configs/claude-code.yaml) | Claude Code session logs (`/save-session` output) and auto-memory files, with `worked` / `failed` / `decision` / `blocker` filters |
| [`obsidian.yaml`](configs/obsidian.yaml) | An Obsidian vault (or any folder of markdown notes) |
| [`budget-csv.yaml`](configs/budget-csv.yaml) | Credit-card / bank statement CSVs, one searchable chunk per transaction |

Copy one to `~/.local-recall/config.yaml`, or point the server at it directly:

```bash
claude mcp add recall -- uvx local-recall-mcp --config /path/to/claude-code.yaml
```

The config path can also be set via the `LOCAL_RECALL_CONFIG` environment variable.

## Configuration reference

```yaml
ollama:
  base_url: http://localhost:11434   # your Ollama endpoint
  embed_model: bge-m3                # any Ollama embedding model
  embed_timeout: 300                 # seconds; first full build is the slow one

index_dir: ~/.local-recall/index     # where the three index files live

sources:                             # any number of directories
  - path: ~/notes
    pattern: "**/*.md"               # glob, relative to path
  - path: ~/.claude/sessions
    pattern: "*.tmp"

section_rules:                       # optional heading -> type mapping
  - contains: "what worked"          # case-insensitive substring of a ##/### heading
    type: worked
  - contains: "what did not work"
    type: failed
```

Files are chunked on `##`/`###` headings; files without headings become a single chunk. Each chunk gets a `section_type` from the first matching rule (`other` if none match), and the `search_memory` tool accepts a `section_filter` to narrow results to one type — the killer use case being *"only show me past failures before I try this again."*

## CSV sources

Any CSV becomes searchable row by row — bank statements, card statements,
order-history exports. One record = one chunk, so *"when did I start paying
Anthropic?"* finds the exact transaction.

```yaml
sources:
  - path: ~/Documents/statements
    pattern: "*.csv"
    type: csv
    encoding: cp932   # optional, default utf-8
    skip_rows: 4      # optional, lines before the header row
    template: "{date} {store} {amount}"   # optional
```

Without `template`, rows render as `column: value | column: value`. CSV chunks
get `section_type: csv`, so `section_filter: "csv"` narrows results to
transactions only.

### Scale

- Unchanged rows are never re-embedded: appending 50 rows to a 20k-row CSV
  embeds only the 50 new rows (chunk-level embedding reuse).
- Practical ceiling is roughly 50k chunks (~200 MB of vectors, sub-100ms
  brute-force search). Beyond that, split your sources.
- Aggregation ("total spent in May") is out of scope: semantic search recalls
  records, it does not compute.

## How it works

```
sources (*.md, *.tmp, ...)          ~/.local-recall/index/
        │  SHA-256 per file          ├── manifest.json   path -> hash
        ▼                            ├── chunks.json     title/content/type
   diff vs manifest ──► re-embed ──► └── vectors.npy     float32 matrix
   (changed files only)   (Ollama /api/embed, batched)

query ──► embed ──► cosine top-k over vectors ──► chunks, capped at 600 chars each
```

No vector database, no background daemon. Sync happens lazily on each search call and is a no-op when nothing changed. A corrupted or misaligned index triggers a full rebuild automatically.

## Non-goals

Kept deliberately small — these are out of scope for v0.x:

- Embedding providers other than Ollama (local-first is the point)
- External vector databases (flat files comfortably handle tens of thousands of chunks)
- Reranking or hybrid search (cosine similarity only)
- Parsers beyond markdown/plain text/CSV (no PDF, no HTML, no xlsx, no JSON)
- Aggregation over CSV data (recall, not arithmetic)
- Any GUI

If you need one of these, open an issue describing the use case — real demand is what justifies scope.

## Development

```bash
git clone https://github.com/Chikoku-NEKO/local-recall-mcp
cd local-recall-mcp
pip install -e .
python -m unittest discover -s tests
```

Tests run offline against a deterministic fake embedding function.

## License

[MIT](LICENSE)
