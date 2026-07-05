"""Unit tests for row-level CSV chunking — all offline, no Ollama."""

import tempfile
import unittest
from pathlib import Path

from local_recall_mcp import csv_source

UC_SAMPLE = (
    "カード名,ＵＣカードＰＲＩＺＥ\n"
    "お支払日,2026年06月05日\n"
    "当月お支払い額合計,0000135924\n"
    "\n"
    "ご利用区分,ご利用日,ご利用者区分,ご利用店,ポイント対象,今回回数,ご利用金額,今回のお支払い金額,備考\n"
    "１回払い,2026/04/15,本人,ＡＮＴＨＲＯＰＩＣ,対象,1,3000,3000,\n"
    "１回払い,2026/04/20,本人,セブンイレブン,対象,1,580,580,\n"
)

SIMPLE = "item,price\napple,120\nbanana,80\n"


class TestRenderRow(unittest.TestCase):
    def test_default_render_joins_nonempty_columns(self):
        row = {"item": "apple", "price": "120", "note": ""}
        self.assertEqual(csv_source.render_row(row, None), "item: apple | price: 120")

    def test_template_render(self):
        row = {"date": "2026/04/15", "store": "ANTHROPIC", "amount": "3000"}
        out = csv_source.render_row(row, "{date} {store} {amount}円")
        self.assertEqual(out, "2026/04/15 ANTHROPIC 3000円")

    def test_template_missing_column_becomes_blank(self):
        out = csv_source.render_row({"a": "x"}, "{a}-{nonexistent}-end")
        self.assertEqual(out, "x--end")


class TestChunkCsvFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_one_row_one_chunk_short_rows_kept(self):
        p = self.dir / "simple.csv"
        p.write_text(SIMPLE, encoding="utf-8")
        chunks = csv_source.chunk_csv_file(p)
        self.assertEqual(len(chunks), 2)  # 50字未満でも捨てない
        self.assertIn("item: apple", chunks[0]["content"])
        self.assertEqual(chunks[0]["section_type"], "csv")
        self.assertIn("simple", chunks[0]["content"])  # ファイルstemの文脈付与

    def test_cp932_skip_rows_and_template(self):
        p = self.dir / "UC_2606.csv"
        p.write_bytes(UC_SAMPLE.encode("cp932"))
        chunks = csv_source.chunk_csv_file(
            p, encoding="cp932", skip_rows=4,
            template="{ご利用日} {ご利用店} {ご利用金額}円（{ご利用区分}）",
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["content"], "2026/04/15 ＡＮＴＨＲＯＰＩＣ 3000円（１回払い）（UC_2606）")
        self.assertIn("セブンイレブン", chunks[1]["content"])

    def test_cp932_default_render_no_mojibake(self):
        p = self.dir / "UC_2606.csv"
        p.write_bytes(UC_SAMPLE.encode("cp932"))
        chunks = csv_source.chunk_csv_file(p, encoding="cp932", skip_rows=4)
        self.assertIn("ご利用店: ＡＮＴＨＲＯＰＩＣ", chunks[0]["content"])

    def test_skip_rows_beyond_eof_returns_empty(self):
        p = self.dir / "tiny.csv"
        p.write_text("a,b\n", encoding="utf-8")
        self.assertEqual(csv_source.chunk_csv_file(p, skip_rows=99), [])

    def test_empty_rows_dropped(self):
        p = self.dir / "gaps.csv"
        p.write_text("item,price\napple,120\n,\n", encoding="utf-8")
        chunks = csv_source.chunk_csv_file(p)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
