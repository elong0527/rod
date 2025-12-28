# pyre-strict
import os

# -----------------------------------------------------------------------------
# Test Data
# -----------------------------------------------------------------------------
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from csrlite.common.plan import StudyPlan
from csrlite.ie.ie_listing import ie_listing_df, ie_listing_rtf, study_plan_to_ie_listing


@pytest.fixture
def adsl_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "USUBJID": ["01-001", "01-002", "01-003"],
            "DCSREAS": ["Completed", "Adverse Event", "Lack of Efficacy"],
            "DISCONFL": ["", "Y", "Y"],
            "OTHER": [1, 2, 3],
        }
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_ie_listing_df(adsl_data: pl.DataFrame) -> None:
    """Test dataframe creation for listing."""
    df = ie_listing_df(adsl_data)

    # Validation
    assert df.columns == ["USUBJID", "DCSREAS"]
    assert df.height == 3
    # Check content
    usubjid = df["USUBJID"].to_list()
    assert "01-001" in usubjid


def test_ie_listing_rtf(adsl_data: pl.DataFrame, tmp_path: Any) -> None:
    """Test RTF generation."""
    df = ie_listing_df(adsl_data)
    output_path = tmp_path / "test_listing.rtf"

    ie_listing_rtf(df, str(output_path), title="Test Listing")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_study_plan_to_ie_listing(tmp_path: Any) -> None:
    """Test the full study plan integration."""

    # Mock StudyPlan
    mock_plan = MagicMock(spec=StudyPlan)
    mock_plan.output_dir = str(tmp_path)
    mock_plan.study_data = {"plans": [{"analysis": "ie_listing", "population": "discon"}]}

    # Mock Expander
    mock_expander = MagicMock()
    mock_expander.expand_plan.return_value = [{"analysis": "ie_listing", "population": "discon"}]
    mock_expander.create_analysis_spec.return_value = {
        "analysis": "ie_listing",
        "population": "discon",
    }
    mock_plan.expander = mock_expander

    # Mock Data Loading/Filtering
    # We need to mock StudyPlanParser
    with patch("csrlite.ie.ie_listing.StudyPlanParser") as MockParser:
        parser_instance = MockParser.return_value

        # Mock get_datasets
        adsl_mock = pl.DataFrame({"USUBJID": ["001"], "DCSREAS": ["AE"], "DISCONFL": ["Y"]})
        parser_instance.get_datasets.return_value = (adsl_mock,)

        # Mock get_population_filter
        parser_instance.get_population_filter.return_value = "adsl:disconfl == 'Y'"

        # Mock apply_common_filters
        # Since we import it in ie.py, we should patch where it is used or imported
        with patch("csrlite.ie.ie_listing.apply_common_filters") as mock_apply:
            mock_apply.return_value = (adsl_mock, None)

            generated_files = study_plan_to_ie_listing(mock_plan)

            assert len(generated_files) == 1
            assert "ie_listing_discon.rtf" in generated_files[0]
            assert os.path.exists(generated_files[0])


def test_study_plan_to_ie_listing_population_error(tmp_path: Any) -> None:
    """Test error handling when population loading fails."""
    mock_plan = MagicMock(spec=StudyPlan)
    mock_plan.output_dir = str(tmp_path)
    mock_plan.study_data = {"plans": [{"analysis": "ie_listing", "population": "discon"}]}

    mock_expander = MagicMock()
    mock_expander.expand_plan.return_value = [{"analysis": "ie_listing", "population": "discon"}]
    mock_expander.create_analysis_spec.return_value = {
        "analysis": "ie_listing",
        "population": "discon",
    }
    mock_plan.expander = mock_expander

    with patch("csrlite.ie.ie_listing.StudyPlanParser") as MockParser:
        parser_instance = MockParser.return_value
        # Mock get_datasets (ADSL) to fail
        parser_instance.get_datasets.side_effect = ValueError("ADSL not found")

        generated = study_plan_to_ie_listing(mock_plan)

        assert len(generated) == 0


def test_study_plan_to_ie_listing_default_generation(tmp_path: Any) -> None:
    """Test default generation when no explicit listing plan is present."""
    mock_plan = MagicMock(spec=StudyPlan)
    mock_plan.output_dir = str(tmp_path)
    # No "ie_listing" in plans
    mock_plan.study_data = {"plans": []}

    # Empty expansion
    mock_expander = MagicMock()
    mock_expander.expand_plan.return_value = []
    mock_plan.expander = mock_expander

    with patch("csrlite.ie.ie_listing.StudyPlanParser") as MockParser:
        parser_instance = MockParser.return_value

        # Mock ADSL success
        adsl_mock = pl.DataFrame({"USUBJID": ["001"], "DCSREAS": ["AE"]})
        parser_instance.get_datasets.return_value = (adsl_mock,)
        parser_instance.get_population_filter.return_value = None

        with patch("csrlite.ie.ie_listing.apply_common_filters") as mock_apply:
            mock_apply.return_value = (adsl_mock, None)

            # Should default to creating one listing for "enrolled"
            generated = study_plan_to_ie_listing(mock_plan)

            assert len(generated) == 1
            assert "ie_listing_enrolled.rtf" in generated[0]
