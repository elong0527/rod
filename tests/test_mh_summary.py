# pyre-strict
import os
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from csrlite.common.plan import StudyPlan
from csrlite.mh.mh_summary import mh_summary, mh_summary_ard, mh_summary_df, study_plan_to_mh_summary

# -----------------------------------------------------------------------------
# Test Data
# -----------------------------------------------------------------------------


@pytest.fixture
def adsl_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "USUBJID": ["01", "02", "03", "04", "05"],
            "TRT01A": ["A", "A", "B", "B", "A"],
            "SAFFL": ["Y", "Y", "Y", "Y", "Y"],
        }
    )


@pytest.fixture
def admh_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "USUBJID": ["01", "01", "03", "04"],
            "MHBODSYS": ["SOC1", "SOC2", "SOC1", "SOC1"],
            "MHDECOD": ["PT1", "PT2", "PT1", "PT3"],
            "MHOCCUR": ["Y", "Y", "Y", "Y"],
        }
    )


# -----------------------------------------------------------------------------
# Tests for MH Summary
# -----------------------------------------------------------------------------


def test_mh_summary_ard(adsl_data: pl.DataFrame, admh_data: pl.DataFrame) -> None:
    """Test ARD generation."""
    
    # 01 (A): SOC1/PT1, SOC2/PT2
    # 02 (A): None
    # 03 (B): SOC1/PT1
    # 04 (B): SOC1/PT3
    # 05 (A): None
    
    # Group A: 3 subs. Subjects with MH: 01.
    # Group B: 2 subs. Subjects with MH: 03, 04.
    
    ard = mh_summary_ard(
        population=adsl_data,
        observation=admh_data,
        population_filter=None,
        observation_filter=None,
        group_col="TRT01A",
        id_col="USUBJID",
        variables=[("MHBODSYS", "SOC"), ("MHDECOD", "PT")],
    )
    
    # Check Structure
    assert "label" in ard.columns
    assert "indent" in ard.columns
    assert "count_A" in ard.columns
    assert "count_B" in ard.columns
    
    # 1. Any MH
    # A: 01 has MH. 1/3
    # B: 03, 04 have MH. 2/2
    row0 = ard.filter(pl.col("label") == "Any Medical History").row(0, named=True)
    assert row0["count_A"] == 1
    assert row0["count_B"] == 2
    
    # 2. SOC1
    # A: 01. (1)
    # B: 03, 04. (2)
    row_soc1 = ard.filter(pl.col("label") == "SOC1").row(0, named=True)
    assert row_soc1["count_A"] == 1
    assert row_soc1["count_B"] == 2
    
    # 3. PT1 (under SOC1)
    # A: 01 (1)
    # B: 03 (1)
    row_pt1 = ard.filter(pl.col("label") == "PT1").row(0, named=True)
    assert row_pt1["count_A"] == 1
    assert row_pt1["count_B"] == 1
    
    # 4. PT3 (under SOC1)
    # A: 0 (0)
    # B: 04 (1)
    row_pt3 = ard.filter(pl.col("label") == "PT3").row(0, named=True)
    assert row_pt3["count_A"] == 0
    assert row_pt3["count_B"] == 1


def test_mh_summary_df(adsl_data: pl.DataFrame, admh_data: pl.DataFrame) -> None:
    """Test display dataframe formatting."""
    ard = mh_summary_ard(
        population=adsl_data,
        observation=admh_data,
        population_filter=None,
        observation_filter=None,
        group_col="TRT01A",
        id_col="USUBJID",
        variables=[],
    )
    
    df = mh_summary_df(ard)
    
    assert "Medical History" in df.columns
    assert "A" in df.columns
    assert "B" in df.columns
    
    # Check format "n (pct)"
    val = df.select("A").row(0)[0]
    assert "(" in val
    assert ")" in val


def test_mh_summary_rtf(adsl_data: pl.DataFrame, admh_data: pl.DataFrame, tmp_path: Any) -> None:
    """Test RTF output."""
    output_path = tmp_path / "test_mh_summary.rtf"
    
    mh_summary(
        population=adsl_data,
        observation=admh_data,
        output_file=str(output_path),
        title=["MH Summary"],
    )
    
    assert output_path.exists()


def test_study_plan_to_mh_summary(tmp_path: Any) -> None:
    """Test study plan integration."""
    mock_plan = MagicMock(spec=StudyPlan)
    mock_plan.output_dir = str(tmp_path)
    mock_plan.study_data = {"plans": [{"analysis": "mh_summary", "population": "saffl"}]}
    
    mock_expander = MagicMock()
    mock_expander.expand_plan.return_value = [{"analysis": "mh_summary", "population": "saffl"}]
    mock_expander.create_analysis_spec.return_value = {
        "analysis": "mh_summary",
        "population": "saffl",
    }
    mock_plan.expander = mock_expander
    
    with patch("csrlite.mh.mh_summary.StudyPlanParser") as MockParser:
        parser_instance = MockParser.return_value
        
        adsl_mock = pl.DataFrame({"USUBJID": ["1"], "TRT01A": ["A"]})
        admh_mock = pl.DataFrame({"USUBJID": ["1"], "MHBODSYS": ["S"], "MHDECOD": ["P"], "MHOCCUR": ["Y"]})
        
        parser_instance.get_population_data.return_value = (adsl_mock, "TRT01A")
        parser_instance.get_datasets.return_value = (admh_mock,)
        
        # We need to ensure apply_common_filters returns logic
        with patch("csrlite.mh.mh_summary.apply_common_filters") as mock_apply:
            mock_apply.return_value = (adsl_mock, admh_mock)
            
            generated = study_plan_to_mh_summary(mock_plan)
            
            assert len(generated) == 1
            assert "mh_summary_saffl_trt01a.rtf" in generated[0]

