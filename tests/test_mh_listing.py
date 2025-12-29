# pyre-strict
import os
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from csrlite.common.plan import StudyPlan
from csrlite.mh.mh_listing import mh_listing, mh_listing_df, study_plan_to_mh_listing

@pytest.fixture
def adsl_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "USUBJID": ["01-001", "01-002", "01-003"],
            "TRT01A": ["Drug A", "Placebo", "Drug A"],
            "AGE": [45, 52, 38],
            "SEX": ["M", "F", "M"],
            "SAFFL": ["Y", "Y", "Y"],
        }
    )

@pytest.fixture
def admh_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "USUBJID": ["01-001", "01-001", "01-002"],
            "MHSEQ": [1, 2, 1],
            "MHBODSYS": ["Infections", "Cardiac", "Nervous"],
            "MHDECOD": ["Flu", "Hypertension", "Headache"],
            "MHSTDTC": ["2023-01-01", "2020-05-15", "2022-11-20"],
            "MHENRTPT": ["RESOLVED", "ONGOING", "RESOLVED"],
            "MHOCCUR": ["Y", "Y", "Y"],
        }
    )

def test_mh_listing_df(adsl_data: pl.DataFrame, admh_data: pl.DataFrame) -> None:
    """Test dataframe creation for listing."""
    df = mh_listing_df(
        population=adsl_data,
        observation=admh_data,
        population_filter=None,
        observation_filter=None,
        id_col="USUBJID",
        pop_cols=None,
        obs_cols=None,
        sort_cols=None,
    )
    assert df.height == 3
    row1 = df.filter((pl.col("USUBJID") == "01-001") & (pl.col("MHSEQ") == 1)).row(0, named=True)
    assert row1["MHDECOD"] == "Flu"

def test_mh_listing_df_sorting(adsl_data: pl.DataFrame, admh_data: pl.DataFrame) -> None:
    """Test sorting in listing."""
    df = mh_listing_df(
        population=adsl_data,
        observation=admh_data,
        population_filter=None,
        observation_filter=None,
        id_col="USUBJID",
        pop_cols=None,
        obs_cols=None,
        sort_cols=["USUBJID", "MHSTDTC"],
    )
    dates = df.filter(pl.col("USUBJID") == "01-001")["MHSTDTC"].to_list()
    assert dates == ["2020-05-15", "2023-01-01"]

def test_mh_listing_rtf(adsl_data: pl.DataFrame, admh_data: pl.DataFrame, tmp_path: Any) -> None:
    """Test RTF generation."""
    output_path = tmp_path / "test_mh_listing.rtf"
    mh_listing(
        population=adsl_data,
        observation=admh_data,
        output_file=str(output_path),
        title=["MH Listing"],
    )
    assert output_path.exists()

def test_mh_listing_missing_obs_data(adsl_data: pl.DataFrame) -> None:
    """Test error when obs data is missing (None passed)."""
    # Passing None to observation usually triggers the check if apply_common_filters preserves None
    # Current apply_common_filters impl logic: if obs is None, returns None.
    # So this should raise ValueError in mh_listing_df
    
    # We need to trick the type checker? No, python is dynamic.
    with pytest.raises(ValueError, match="Observation data is missing"):
        mh_listing_df(
            population=adsl_data,
            observation=None, # pyre-ignore
            population_filter=None,
            observation_filter=None,
            id_col="USUBJID",
            pop_cols=None,
            obs_cols=None,
            sort_cols=None,
        )

@patch("csrlite.mh.mh_listing.mh_listing")
def test_study_plan_to_mh_listing(mock_mh_listing: MagicMock, tmp_path: Any) -> None:
    """Test study plan integration using mocks."""
    
    mock_plan = MagicMock(spec=StudyPlan)
    mock_plan.output_dir = str(tmp_path)
    mock_plan.study_data = {"plans": [{"analysis": "mh_listing", "population": "saffl"}]}

    mock_expander = MagicMock()
    mock_expander.expand_plan.return_value = [{"analysis": "mh_listing", "population": "saffl"}]
    mock_expander.create_analysis_spec.return_value = {
        "analysis": "mh_listing",
        "population": "saffl",
    }
    mock_plan.expander = mock_expander

    with patch("csrlite.mh.mh_listing.StudyPlanParser") as MockParser:
        parser_instance = MockParser.return_value

        adsl_mock = pl.DataFrame({"USUBJID": ["001"], "TRT01A": ["A"], "SAFFL": ["Y"]})
        admh_mock = pl.DataFrame({"USUBJID": ["001"], "MHDECOD": ["Flu"], "MHOCCUR": ["Y"]})
        
        parser_instance.get_population_data.return_value = (adsl_mock, "TRT01A")
        parser_instance.get_datasets.return_value = (admh_mock,)

        # We don't strictly need to patch apply_common_filters if we mock mh_listing
        # because logic flow: study_plan_to_mh_listing -> parser -> mh_listing(args)
        
        generated = study_plan_to_mh_listing(mock_plan)
        
        assert len(generated) == 1
        assert "mh_listing_saffl.rtf" in generated[0]
        
        # Verify call
        mock_mh_listing.assert_called_once()
        _, kwargs = mock_mh_listing.call_args
        assert kwargs["output_file"].endswith("mh_listing_saffl.rtf")

def test_study_plan_to_mh_listing_defaults(tmp_path: Any) -> None:
    """Test default generation (no explicit plan defaults to nothing if none found)."""
    mock_plan = MagicMock(spec=StudyPlan)
    mock_plan.study_data = {"plans": []}
    
    # Mock expander explicitly
    mock_expander = MagicMock()
    mock_expander.expand_plan.return_value = []
    mock_plan.expander = mock_expander
    
    with patch("csrlite.mh.mh_listing.StudyPlanParser"):
        generated = study_plan_to_mh_listing(mock_plan)
        assert len(generated) == 0
