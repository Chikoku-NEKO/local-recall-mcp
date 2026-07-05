"""MCP server exposing a single tool: search_memory."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import mcp.types as types
import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import indexer

DEFAULT_CONFIG_PATH = Path.home() / ".local-recall" / "config.yaml"

DEFAULT_TOOL_DESCRIPTION = (
    "Semantic search over your local notes and session logs "
    "(fully local: embeddings via Ollama, nothing leaves your machine). "
    "Use it before design, implementation, or debugging work to recall "
    "past decisions, solutions, and failure patterns."
)

SAMPLE_CONFIG = """\
ollama:
  base_url: http://localhost:11434
  embed_model: bge-m3
  embed_timeout: 300   # seconds; the first full build can take a while

index_dir: ~/.local-recall/index

sources:
  - path: ~/notes
    pattern: "**/*.md"

# CSV sources are searched row by row (one record = one chunk):
# sources:
#   - path: ~/Documents/statements
#     pattern: "*.csv"
#     type: csv
#     encoding: cp932     # e.g. Japanese card statements
#     skip_rows: 4        # preamble lines before the header row
#     template: "{date} {store} {amount}"   # optional; default renders "column: value | ..."

# Optional: map heading substrings to filterable section types.
# section_rules:
#   - contains: "what worked"
#     type: worked
"""


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not cfg.get("sources"):
        raise ValueError(f"No 'sources' defined in {path}")
    return cfg


def parse_sources(cfg: dict[str, Any]) -> list[indexer.SourceSpec]:
    specs = []
    for s in cfg["sources"]:
        stype = str(s.get("type", "text"))
        if stype not in ("text", "csv"):
            raise ValueError(f"Unknown source type '{stype}' (expected 'text' or 'csv')")
        specs.append(indexer.SourceSpec(
            base=Path(str(s["path"])).expanduser(),
            pattern=str(s.get("pattern", "**/*.md" if stype == "text" else "*.csv")),
            type=stype,
            encoding=str(s.get("encoding", "utf-8")),
            skip_rows=int(s.get("skip_rows", 0)),
            template=(str(s["template"]) if s.get("template") else None),
        ))
    return specs


def parse_rules(cfg: dict[str, Any]) -> list[indexer.SectionRule]:
    raw = cfg.get("section_rules")
    if raw is None:
        return indexer.DEFAULT_SECTION_RULES
    return [(str(r["contains"]).lower(), str(r["type"])) for r in raw]


def resolve_tool_description(cfg: dict[str, Any]) -> str:
    return str((cfg.get("tool") or {}).get("description") or DEFAULT_TOOL_DESCRIPTION)


def build_server(cfg: dict[str, Any]) -> Server:
    ollama_cfg = cfg.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    embed_model = ollama_cfg.get("embed_model", "bge-m3")
    embed_timeout = int(ollama_cfg.get("embed_timeout", 300))
    index_dir = Path(str(cfg.get("index_dir", "~/.local-recall/index"))).expanduser()
    sources = parse_sources(cfg)
    rules = parse_rules(cfg)
    section_types = sorted({stype for _, stype in rules})
    if any(s.type == "csv" for s in sources):
        section_types = sorted(set(section_types) | {"csv"})
    tool_description = resolve_tool_description(cfg)

    server = Server("local-recall-mcp")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        filter_schema: dict[str, Any] = {
            "type": "string",
            "description": "Only return chunks of this section type (optional).",
        }
        if section_types:
            filter_schema["enum"] = section_types
        return [
            types.Tool(
                name="search_memory",
                description=tool_description,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language query, e.g. 'how did we fix the MCP connection issue'",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to return (default 5).",
                        },
                        "section_filter": filter_schema,
                    },
                    "required": ["query"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name != "search_memory":
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
        query = str(arguments.get("query", "")).strip()
        if not query:
            return [types.TextContent(type="text", text="'query' is empty. Provide a search query.")]
        top_k = int(arguments.get("top_k", 5) or 5)
        section_filter = arguments.get("section_filter") or None
        embed_fn = indexer.make_ollama_embed(base_url, embed_model, embed_timeout)
        try:
            text = await indexer.search_memory(
                query,
                top_k=top_k,
                section_filter=section_filter,
                index_dir=index_dir,
                sources=sources,
                embed_fn=embed_fn,
                rules=rules,
            )
        except ValueError as e:
            text = str(e)
        except httpx.ConnectError:
            text = f"Cannot reach Ollama at {base_url}. Is 'ollama serve' running?"
        except httpx.TimeoutException:
            text = (
                f"Embedding timed out after {embed_timeout}s. The first full index build "
                "can be slow; raise 'ollama.embed_timeout' in your config and retry."
            )
        return [types.TextContent(type="text", text=text)]

    return server


async def run(cfg: dict[str, Any]) -> None:
    server = build_server(cfg)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="local-recall-mcp",
        description="Fully local semantic memory for AI agents over MCP.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config.yaml (default: $LOCAL_RECALL_CONFIG or {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    config_path = args.config or Path(
        os.environ.get("LOCAL_RECALL_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    config_path = config_path.expanduser()
    if not config_path.exists():
        sys.stderr.write(
            f"Config not found: {config_path}\n"
            f"Create it first, for example:\n\n{SAMPLE_CONFIG}\n"
        )
        sys.exit(1)
    cfg = load_config(config_path)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    cli()
