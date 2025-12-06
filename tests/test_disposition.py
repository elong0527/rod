# pyre-strict
import unittest
from pathlib import Path

import polars as pl

from csrlite.common.plan import load_plan
from csrlite.disposition.disposition import (
    disposition,
    disposition_ard,
    disposition_df,
    disposition_rtf,
    study_plan_to_disposition_summary,
)


class TestDispositionArd(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data for disposition analysis."""
        self.population_data = pl.DataFrame(
            {
                "USUBJID": ["01", "02", "03", "04", "05", "06"],
                "TRT01A": [
                    "Treatment A",
                    "Treatment A",
                    "Treatment B",
                    "Treatment B",
                    "Treatment B",
                    "Treatment A",
                ],
                "SAFFL": ["Y", "Y", "Y", "Y", "Y", "Y"],
            }
        )

        self.observation_data = pl.DataFrame(
            {
                "USUBJID": ["01", "03", "04"],
                "DSDECOD": ["Completed", "Withdrawn", "Screening Failure"],
                "DSTERM": ["Study completed", "Adverse event", "Failed screening"],
            }
        )

    def test_disposition_ard_basic(self) -> None:
        """Test basic ARD generation for disposition."""
        variables = [
            ("SAFFL == 'Y'", "Enrolled"),
            ("DSDECOD == 'Completed'", "Completed"),
        ]

        ard = disposition_ard(
            population=self.population_data,
            observation=self.observation_data,
            population_filter=None,
            observation_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            variables=variables,
            total=True,
            missing_group="error",
        )

        # Check that ARD has expected columns
        self.assertIn("__index__", ard.columns)
        self.assertIn("__group__", ard.columns)
        self.assertIn("__value__", ard.columns)

        # Check that we have results for all groups
        groups = ard["__group__"].unique().to_list()
        self.assertIn("Treatment A", groups)
        self.assertIn("Treatment B", groups)
        self.assertIn("Total", groups)

    def test_disposition_ard_no_group(self) -> None:
        """Test ARD generation without group variable."""
        variables = [
            ("SAFFL == 'Y'", "Enrolled"),
        ]

        ard = disposition_ard(
            population=self.population_data,
            observation=self.observation_data,
            population_filter=None,
            observation_filter=None,
            id=("USUBJID", "Subject ID"),
            group=None,  # No grouping
            variables=variables,
            total=True,
            missing_group="error",
        )

        # When no group is specified, __all__ is used
        self.assertIn("__group__", ard.columns)
        groups = ard["__group__"].unique().to_list()
        self.assertIn("All Subjects", groups)

    def test_disposition_ard_with_filters(self) -> None:
        """Test ARD generation with population and observation filters."""
        variables = [
            ("SAFFL == 'Y'", "Safety Population"),
        ]

        ard = disposition_ard(
            population=self.population_data,
            observation=self.observation_data,
            population_filter="TRT01A == 'Treatment A'",
            observation_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            variables=variables,
            total=False,
            missing_group="error",
        )

        # Should only have Treatment A group
        groups = ard["__group__"].unique().to_list()
        self.assertEqual(len(groups), 1)
        self.assertIn("Treatment A", groups)

    def test_disposition_ard_indentation(self) -> None:
        """Test ARD generation with indented labels."""
        variables = [
            ("SAFFL == 'Y'", "Enrolled"),
            ("DSDECOD == 'Completed'", "    Completed"),
            ("DSDECOD == 'Withdrawn'", "    Withdrawn"),
        ]

        ard = disposition_ard(
            population=self.population_data,
            observation=self.observation_data,
            population_filter=None,
            observation_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            variables=variables,
            total=True,
            missing_group="error",
        )

        # Check that labels are preserved including indentation
        labels = ard["__index__"].unique().to_list()
        self.assertIn("Enrolled", labels)
        self.assertIn("    Completed", labels)
        self.assertIn("    Withdrawn", labels)


class TestDispositionDf(unittest.TestCase):
    def test_disposition_df_basic(self) -> None:
        """Test transformation of ARD to display format."""
        ard = pl.DataFrame(
            {
                "__index__": ["Enrolled", "Enrolled", "Completed", "Completed"],
                "__group__": ["Treatment A", "Treatment B", "Treatment A", "Treatment B"],
                "__value__": ["3 (100%)", "3 (100%)", "1 (33.3%)", "2 (66.7%)"],
            }
        )

        df = disposition_df(ard)

        # Check columns
        self.assertIn("Disposition Status", df.columns)
        self.assertIn("Treatment A", df.columns)
        self.assertIn("Treatment B", df.columns)

        # Check shape
        self.assertEqual(df.height, 2)  # Two rows: Enrolled, Completed

    def test_disposition_df_preserves_order(self) -> None:
        """Test that row order is preserved from ARD."""
        # Create ARD with Enum to enforce order
        var_labels = ["Enrolled", "Completed", "Discontinued"]
        ard = pl.DataFrame(
            {
                "__index__": var_labels * 2,
                "__group__": ["Grp A"] * 3 + ["Grp B"] * 3,
                "__value__": ["1", "2", "3", "4", "5", "6"],
            }
        ).with_columns(pl.col("__index__").cast(pl.Enum(var_labels)))

        df = disposition_df(ard)

        # Check that rows are in expected order
        status_col = df["Disposition Status"].to_list()
        self.assertEqual(status_col, var_labels)


class TestDispositionRtf(unittest.TestCase):
    def test_disposition_rtf_basic(self) -> None:
        """Test RTF generation from display dataframe."""
        df = pl.DataFrame(
            {
                "Disposition Status": ["Enrolled", "Completed"],
                "Treatment A": ["3 (100%)", "1 (33.3%)"],
                "Treatment B": ["3 (100%)", "2 (66.7%)"],
                "Total": ["6 (100%)", "3 (50.0%)"],
            }
        )

        rtf_doc = disposition_rtf(
            df=df,
            title=["Disposition of Participants", "Safety Population"],
            footnote=["Percentages based on enrolled participants."],
            source=None,
            col_rel_width=None,
        )

        # Check that RTF document was created
        self.assertIsNotNone(rtf_doc)

    def test_disposition_rtf_custom_widths(self) -> None:
        """Test RTF generation with custom column widths."""
        df = pl.DataFrame(
            {
                "Disposition Status": ["Enrolled"],
                "Treatment A": ["3 (100%)"],
            }
        )

        rtf_doc = disposition_rtf(
            df=df,
            title=["Test Title"],
            footnote=None,
            source=None,
            col_rel_width=[3.0, 1.5],
        )

        self.assertIsNotNone(rtf_doc)


class TestDispositionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and output directory."""
        self.population_data = pl.DataFrame(
            {
                "USUBJID": ["01", "02", "03", "04", "05", "06"],
                "TRT01A": [
                    "Treatment A",
                    "Treatment A",
                    "Treatment B",
                    "Treatment B",
                    "Treatment B",
                    "Treatment A",
                ],
                "SAFFL": ["Y", "Y", "Y", "Y", "Y", "Y"],
            }
        )

        self.observation_data = pl.DataFrame(
            {
                "USUBJID": ["01", "03", "04"],
                "DSDECOD": ["Completed", "Withdrawn", "Screening Failure"],
            }
        )

        self.output_dir = Path("tests/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up test output files."""
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.rtf"):
                file.unlink()

    def test_disposition_full_pipeline(self) -> None:
        """Test complete disposition pipeline."""
        variables = [
            ("SAFFL == 'Y'", "Enrolled"),
            ("DSDECOD == 'Completed'", "    Completed"),
        ]

        output_file = str(self.output_dir / "test_disposition.rtf")

        result_path = disposition(
            population=self.population_data,
            observation=self.observation_data,
            population_filter=None,
            observation_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            variables=variables,
            title=["Disposition of Participants"],
            footnote=["Test footnote"],
            source=None,
            output_file=output_file,
            total=True,
            missing_group="error",
        )

        # Check that file was created
        self.assertTrue(Path(result_path).exists())
        self.assertEqual(result_path, output_file)

    def test_disposition_no_total(self) -> None:
        """Test disposition without total column."""
        variables = [("SAFFL == 'Y'", "Enrolled")]
        output_file = str(self.output_dir / "test_disposition_no_total.rtf")

        result_path = disposition(
            population=self.population_data,
            observation=self.observation_data,
            population_filter=None,
            observation_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            variables=variables,
            title=["Test"],
            footnote=None,
            source=None,
            output_file=output_file,
            total=False,
        )

        self.assertTrue(Path(result_path).exists())


class TestStudyPlanToDisposition(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test environment."""
        self.output_dir = Path("studies/xyz123/rtf")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up test output files."""
        if self.output_dir.exists():
            for file in self.output_dir.glob("disposition_*.rtf"):
                file.unlink()

    def test_study_plan_to_disposition_summary(self) -> None:
        """Test generating disposition tables from StudyPlan."""
        # Load the study plan
        study_plan = load_plan("studies/xyz123/yaml/plan_ds_xyz123.yaml")

        # Generate disposition tables
        rtf_files = study_plan_to_disposition_summary(study_plan)

        # Check that files were generated
        self.assertIsInstance(rtf_files, list)
        self.assertGreater(len(rtf_files), 0)

        # Check that all files exist
        for rtf_file in rtf_files:
            self.assertTrue(Path(rtf_file).exists(), f"File {rtf_file} should exist")


if __name__ == "__main__":
    unittest.main()
