# pyre-strict
import unittest
from pathlib import Path
from typing import Callable

import polars as pl

from csrlite.common.parse import StudyPlanParser
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
                "USUBJID": ["01", "02", "03", "04", "05", "06", "07", "08"],
                "TRT01A": [
                    "Treatment A",
                    "Treatment A",
                    "Treatment B",
                    "Treatment B",
                    "Treatment B",
                    "Treatment A",
                    "Treatment A",
                    "Treatment B",
                ],
                "SAFFL": ["Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y"],
                "EOSSTT": [
                    "Completed",
                    "Completed",
                    "Discontinued",
                    "Discontinued",
                    "Completed",
                    "Completed",
                    "Ongoing",  # New case: Ongoing
                    "Discontinued",  # New case: Discontinued with Null Reason
                ],
                "DCSREAS": [
                    None,
                    None,
                    "Withdrawn",
                    "Screening Failure",
                    None,
                    None,
                    None,  # Ongoing has Null reason
                    None,  # Discontinued has Null reason (Missing)
                ],
            }
        )
        discontinued_reason = pl.Series(
            "DCSREAS",
            [
                None,  # 01
                None,  # 02
                "Withdrawn",  # 03
                "Screening Failure",  # 04
                None,  # 05
                None,  # 06
                None,  # 07
                "Adverse Event",  # 08 - Was None, now explicitly Adverse Event to pass validation
            ],
        )
        self.population_data = self.population_data.with_columns(
            discontinued_reason.alias("DCSREAS")
        ).with_columns(
            pl.col("EOSSTT").cast(pl.Categorical),
            pl.col("DCSREAS").cast(pl.Categorical),
        )

    def test_disposition_ard_basic(self) -> None:
        """Test basic ARD generation for disposition."""

        ard = disposition_ard(
            population=self.population_data,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            dist_reason_term=("DCSREAS", "Discontinued"),
            ds_term=("EOSSTT", "Status"),
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

        # Verify specific values
        # Check "Ongoing" presence (Subject 07 in Treatment A)
        ongoing_row = ard.filter(
            (pl.col("__index__") == "Ongoing") & (pl.col("__group__") == "Treatment A")
        )
        self.assertFalse(ongoing_row.is_empty(), "Ongoing row should exist for Treatment A")

    def test_disposition_ard_validation_error(self) -> None:
        """Test that invalid data raises validation errors."""
        # 1. Invalid Status
        invalid_status_data = self.population_data.clone().with_columns(
            pl.when(pl.col("USUBJID") == "01")
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("EOSSTT"))
            .alias("EOSSTT")
        )

        with self.assertRaisesRegex(ValueError, "Invalid disposition statuses"):
            disposition_ard(
                population=invalid_status_data,
                population_filter=None,
                id=("USUBJID", "Subject ID"),
                group=("TRT01A", "Treatment"),
                dist_reason_term=("DCSREAS", "Discontinued"),
                ds_term=("EOSSTT", "Status"),
                total=True,
                missing_group="error",
            )

        # 2. Inconsistent Data (Completed with Mismatched Reason)
        inconsistent_data = self.population_data.clone().with_columns(
            pl.when(pl.col("USUBJID") == "01")  # Subject 01 is Completed
            .then(pl.lit("Adverse Event"))  # Mismatched Reason
            .otherwise(pl.col("DCSREAS"))
            .alias("DCSREAS")
        )

        with self.assertRaisesRegex(ValueError, "mismatched discontinuation reason"):
            disposition_ard(
                population=inconsistent_data,
                population_filter=None,
                id=("USUBJID", "Subject ID"),
                group=("TRT01A", "Treatment"),
                dist_reason_term=("DCSREAS", "Discontinued"),
                ds_term=("EOSSTT", "Status"),
                total=True,
                missing_group="error",
            )

        # 3. Valid Redundancy (Completed with Reason="Completed" is ALLOWED)
        redundant_data = self.population_data.clone().with_columns(
            pl.when(pl.col("USUBJID") == "01")  # Subject 01 is Completed
            .then(pl.lit("Completed"))  # Matches Status
            .otherwise(pl.col("DCSREAS"))
            .alias("DCSREAS")
        )
        # Should NOT raise error
        disposition_ard(
            population=redundant_data,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            dist_reason_term=("DCSREAS", "Discontinued"),
            ds_term=("EOSSTT", "Status"),
            total=True,
            missing_group="error",
        )

        # 4. Invalid Discontinued (Discontinued with Null Reason)
        # 4. Invalid Discontinued (Discontinued with Null Reason)

        # Test Case for Invalid Discontinued (Null)
        invalid_disc_null = self.population_data.clone().with_columns(
            pl.when(pl.col("USUBJID") == "03")  # Subject 03 is Discontinued/Withdrawn
            .then(None)  # Make Reason Null
            .otherwise(pl.col("DCSREAS"))
            .alias("DCSREAS")
        )
        with self.assertRaisesRegex(ValueError, "missing or invalid discontinuation reason"):
            disposition_ard(
                population=invalid_disc_null,
                population_filter=None,
                id=("USUBJID", "Subject ID"),
                group=("TRT01A", "Treatment"),
                dist_reason_term=("DCSREAS", "Discontinued"),
                ds_term=("EOSSTT", "Status"),
                total=True,
                missing_group="error",
            )

        # 5. Invalid Discontinued (Discontinued with Reason="Completed")
        invalid_disc_comp = self.population_data.clone().with_columns(
            pl.when(pl.col("USUBJID") == "03")  # Subject 03 is Discontinued
            .then(pl.lit("Completed"))  # Invalid Reason
            .otherwise(pl.col("DCSREAS"))
            .alias("DCSREAS")
        )
        with self.assertRaisesRegex(ValueError, "missing or invalid discontinuation reason"):
            disposition_ard(
                population=invalid_disc_comp,
                population_filter=None,
                id=("USUBJID", "Subject ID"),
                group=("TRT01A", "Treatment"),
                dist_reason_term=("DCSREAS", "Discontinued"),
                ds_term=("EOSSTT", "Status"),
                total=True,
                missing_group="error",
            )

    def test_disposition_ard_no_group(self) -> None:
        """Test ARD generation without group variable."""

        ard = disposition_ard(
            population=self.population_data,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=None,  # No grouping
            dist_reason_term=("DCSREAS", "Discontinued"),
            ds_term=("EOSSTT", "Status"),
            total=True,
            missing_group="error",
        )

        # When no group is specified, Overall is used
        self.assertIn("__group__", ard.columns)
        groups = ard["__group__"].unique().to_list()
        self.assertIn("Overall", groups)

    def test_disposition_ard_with_filters(self) -> None:
        """Test ARD generation with population and observation filters."""

        ard = disposition_ard(
            population=self.population_data,
            population_filter="TRT01A == 'Treatment A'",
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            dist_reason_term=("DCSREAS", "Discontinued"),
            ds_term=("EOSSTT", "Status"),
            total=False,
            missing_group="error",
        )

        # Should only have Treatment A group
        groups = ard["__group__"].unique().to_list()
        self.assertEqual(len(groups), 1)
        self.assertIn("Treatment A", groups)


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
        self.assertIn("Term", df.columns)
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
        status_col = df["Term"].to_list()
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
                "EOSSTT": [
                    "Completed",
                    "Completed",
                    "Discontinued",
                    "Discontinued",
                    "Completed",
                    "Completed",
                ],
                "DCSREAS": [
                    None,
                    None,
                    "Withdrawn",
                    "Screening Failure",
                    None,
                    None,
                ],
            }
        ).with_columns(
            pl.col("EOSSTT").cast(pl.Categorical),
            pl.col("DCSREAS").cast(pl.Categorical),
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

        output_file = str(self.output_dir / "test_disposition.rtf")

        result_path = disposition(
            population=self.population_data,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            dist_reason_term=("DCSREAS", "Discontinued"),
            ds_term=("EOSSTT", "Status"),
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
        output_file = str(self.output_dir / "test_disposition_no_total.rtf")

        result_path = disposition(
            population=self.population_data,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            dist_reason_term=("DCSREAS", "Discontinued"),
            ds_term=("EOSSTT", "Status"),
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
        study_plan = load_plan("studies/xyz123/yaml/plan_xyz123.yaml")

        original_get_datasets: Callable[[StudyPlanParser, str], tuple[pl.DataFrame]] = (
            StudyPlanParser.get_datasets
        )

        # Generate disposition tables
        with unittest.mock.patch(
            "csrlite.common.parse.StudyPlanParser.get_datasets", autospec=True
        ) as mock_get:
            # Create a side effect to load real data then clean it
            def get_clean_datasets(self: StudyPlanParser, name: str) -> tuple[pl.DataFrame]:
                dfs = original_get_datasets(self, name)
                if name == "adsl":
                    df = dfs[0]
                    # Clean data: values where EOSSTT in {Completed, Ongoing}
                    # should have DCREASCD = None
                    # to satisfy validation rules
                    clean_df = df.with_columns(
                        pl.when(pl.col("EOSSTT").is_in(["Completed", "Ongoing"]))
                        .then(None)
                        .otherwise(pl.col("DCSREAS"))
                        .alias("DCSREAS")
                    )
                    return (clean_df,)
                return dfs

            mock_get.side_effect = get_clean_datasets
            rtf_files = study_plan_to_disposition_summary(study_plan)

        # Check that files were generated
        self.assertIsInstance(rtf_files, list)
        self.assertGreater(len(rtf_files), 0)

        # Check that all files exist
        for rtf_file in rtf_files:
            self.assertTrue(Path(rtf_file).exists(), f"File {rtf_file} should exist")


if __name__ == "__main__":
    unittest.main()
