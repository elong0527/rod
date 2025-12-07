import shutil
import tempfile
import unittest
from pathlib import Path

import polars as pl
from rtflite import RTFDocument

from csrlite.common.rtf import create_rtf_table_n_pct


class TestRTF(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.output_file = Path(self.test_dir) / "test.rtf"
        self.df = pl.DataFrame(
            {
                "Col1": ["A", "B"],
                "Col2": [1, 2],
            }
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_create_rtf_table_basic(self) -> None:
        title = "Test Table"
        footnote = "Test Footnote"

        source = "Test Source"

        doc = create_rtf_table_n_pct(
            df=self.df,
            col_header_1=["Column 1", "Column 2"],
            col_header_2=None,
            col_widths=[1.5, 1.0],
            title=title,
            footnote=footnote,
            source=source,
        )

        self.assertIsInstance(doc, RTFDocument)

        # Verify writing
        doc.write_rtf(str(self.output_file))
        self.assertTrue(self.output_file.exists())

        content = self.output_file.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("Test Table", content)

        self.assertIn("Column 1", content)
        # Note: RTF content checking is fuzzy, but simple strings often appear.

    def test_create_rtf_table_nested_headers(self) -> None:
        doc = create_rtf_table_n_pct(
            df=self.df,
            col_header_1=["Group", "Value"],
            col_header_2=["", "N (%)"],
            col_widths=[1, 1],
            title="Nested Header Table",
            footnote=None,
            source=None,
        )

        doc.write_rtf(str(self.output_file))
        self.assertTrue(self.output_file.exists())
