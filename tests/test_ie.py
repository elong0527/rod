# pyre-strict
import unittest
from pathlib import Path
from typing import Callable

import polars as pl

from csrlite.common.parse import StudyPlanParser
from csrlite.common.plan import load_plan
from csrlite.ie.ie import (
    ie_summary,
    ie_summary_ard,
    ie_summary_df,
    ie_summary_rtf,
    study_plan_to_ie_summary,
)


class TestIeArd(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data for IE analysis."""
        self.population_data = pl.DataFrame(
            {
                "USUBJID": ["01", "02", "03", "04", "05"],
                "TRT01A": [
                    "Treatment A",
                    "Treatment A",
                    "Treatment B",
                    "Treatment B",
                    "Treatment B",
                ],
                "SAFFL": ["Y", "Y", "Y", "Y", "Y"],
            }
        )
        self.observation_data = pl.DataFrame(
            {
                "USUBJID": ["02", "03", "03"],
                "PARAM": [
                    "Criterion 1 not met", 
                    "Criterion 1 not met", 
                    "Criterion 2 not met"
                ],
                "PARAMCAT": [
                    "Inclusion",
                    "Inclusion",
                    "Exclusion"
                ],
                "AFLAG": ["Y", "Y", "Y"],
            }
        )
        self.observation_data_mixed = pl.DataFrame(
             {
                "USUBJID": ["01", "02"],
                "PARAM": ["C1", "C2"],
                "PARAMCAT": ["Inclusion", "Exclusion"],
                "AFLAG": ["N", "Y"], # 01 met criteria (N), 02 failed (Y)
            }           
        )

    def test_ie_summary_ard_basic(self) -> None:
        """Test basic ARD generation for IE."""

        ard = ie_summary_ard(
            population=self.population_data,
            observation=self.observation_data,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            total=True,
            missing_group="error",
        )

        # Check columns
        self.assertIn("__index__", ard.columns)
        self.assertIn("__group__", ard.columns)
        self.assertIn("__value__", ard.columns)

        # Check values
        # Treatment A: 2 subjects (01, 02). 02 has failure.
        # Treatment B: 3 subjects (03, 04, 05). 03 has 2 failures.
        
        # Enrolled row (now Total Subjects Screened)
        enrolled_trt_a = ard.filter(
            (pl.col("__index__") == "Total Subjects Screened") & (pl.col("__group__") == "Treatment A")
        )["__value__"].item()
        self.assertEqual(enrolled_trt_a, "2")
        
        # Check Header: "Inclusion Criteria Not Met"
        incl_header_trt_a = ard.filter(
            (pl.col("__index__") == "Inclusion Criteria Not Met") & (pl.col("__group__") == "Treatment A")
        )["__value__"].item()
        self.assertEqual(incl_header_trt_a, "1 ( 50.0)")

        # Check Criterion 1 (now indented)
        c1_trt_a = ard.filter(
            (pl.col("__index__") == "    Criterion 1 not met") & (pl.col("__group__") == "Treatment A")
        )["__value__"].item()
        self.assertEqual(c1_trt_a, "1 ( 50.0)") # Subject 02

        c1_trt_b = ard.filter(
            (pl.col("__index__") == "    Criterion 1 not met") & (pl.col("__group__") == "Treatment B")
        )["__value__"].item()
        self.assertEqual(c1_trt_b, "1 ( 33.3)") # Subject 03
        
        # Check Header: "Exclusion Criteria Met"
        excl_header_trt_b = ard.filter(
            (pl.col("__index__") == "Exclusion Criteria Met") & (pl.col("__group__") == "Treatment B")
        )["__value__"].item()
        self.assertEqual(excl_header_trt_b, "1 ( 33.3)")

    def test_ie_summary_ard_filters(self) -> None:
        """Test ARD with AFLAG filtering."""
        ard = ie_summary_ard(
            population=self.population_data,
            observation=self.observation_data_mixed,
            population_filter=None,
            id=("USUBJID", "Subject ID"),
            group=("TRT01A", "Treatment"),
            total=True,
            missing_group="error",
        )
        
        # Only subject 02 (Treatment A) matches AFLAG=Y
        
        c2_trt_a = ard.filter(
             (pl.col("__index__") == "    C2") & (pl.col("__group__") == "Treatment A")
        )["__value__"].item()
        self.assertEqual(c2_trt_a, "1 ( 50.0)")
        
        # C1 should NOT be present or be 0? 
        # count_subject_with_observation returns rows only for found items usually, 
        # unless configured otherwise. But wait, if it's 0 it might strictly not appear 
        # depending on implementation of count utility.
        # But here C1 has AFLAG=N, so it gets filtered out before counting.
        
        c1_rows = ard.filter(pl.col("__index__") == "C1")
        self.assertTrue(c1_rows.is_empty())


class TestIePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path("tests/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Mock data (same as above)
        self.pop = pl.DataFrame({"USUBJID": ["01"], "TRT": ["A"]})
        self.obs = pl.DataFrame({"USUBJID": ["01"], "PARAM": ["C1"], "PARAMCAT": ["Inclusion"], "AFLAG": ["Y"]})

    def tearDown(self) -> None:
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.rtf"):
                file.unlink()

    def test_ie_pipeline_integration(self) -> None:
        output_file = str(self.output_dir / "test_ie.rtf")
        ie_summary(
            population=self.pop,
            observation=self.obs,
            population_filter=None,
            id=("USUBJID", "ID"),
            group=("TRT", "Group"),
            title=["Test"],
            footnote=None,
            source=None,
            output_file=output_file
        )
        self.assertTrue(Path(output_file).exists())


class TestStudyPlanToIe(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path("studies/xyz123/rtf")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.output_dir.exists():
            for file in self.output_dir.glob("ie_summary_*.rtf"):
                file.unlink()

    def test_study_plan_to_ie_summary(self) -> None:
        """Test generating IE tables from StudyPlan."""
        # We need a real plan file.
        # Assuming the plan has been written to studies/xyz123/yaml/plan_ie_xyz123.yaml
        # And assuming adie.parquet and adsl.parquet exist in data/
        
        # We'll use the one we just created in the plan phase
        plan_path = "studies/xyz123/yaml/plan_ie_xyz123.yaml"
        if not Path(plan_path).exists():
            self.skipTest("Plan file not found")
            
        study_plan = load_plan(plan_path)
        
        rtf_files = study_plan_to_ie_summary(study_plan)
        
        self.assertGreater(len(rtf_files), 0)
        for f in rtf_files:
            self.assertTrue(Path(f).exists())

if __name__ == "__main__":
    unittest.main()
