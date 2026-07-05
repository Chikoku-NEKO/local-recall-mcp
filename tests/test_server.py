"""Unit tests for config parsing in server.py — no MCP runtime needed."""

import unittest

from local_recall_mcp import server


class TestParseSources(unittest.TestCase):
    def test_default_entry_is_text_type(self):
        cfg = {"sources": [{"path": "~/notes", "pattern": "**/*.md"}]}
        (spec,) = server.parse_sources(cfg)
        self.assertEqual(spec.type, "text")
        self.assertEqual(spec.encoding, "utf-8")
        self.assertEqual(spec.skip_rows, 0)
        self.assertIsNone(spec.template)

    def test_csv_entry_carries_all_fields(self):
        cfg = {"sources": [{
            "path": "~/statements", "pattern": "*.csv", "type": "csv",
            "encoding": "cp932", "skip_rows": 4, "template": "{date} {store}",
        }]}
        (spec,) = server.parse_sources(cfg)
        self.assertEqual(
            (spec.type, spec.encoding, spec.skip_rows, spec.template),
            ("csv", "cp932", 4, "{date} {store}"),
        )
        self.assertEqual(spec.pattern, "*.csv")

    def test_invalid_type_raises(self):
        cfg = {"sources": [{"path": "~/x", "type": "xlsx"}]}
        with self.assertRaises(ValueError):
            server.parse_sources(cfg)


class TestToolDescription(unittest.TestCase):
    def test_default_when_unset(self):
        self.assertEqual(server.resolve_tool_description({}), server.DEFAULT_TOOL_DESCRIPTION)

    def test_override_from_config(self):
        cfg = {"tool": {"description": "第二の脳：作業前に必ず検索"}}
        self.assertEqual(server.resolve_tool_description(cfg), "第二の脳：作業前に必ず検索")


if __name__ == "__main__":
    unittest.main()
