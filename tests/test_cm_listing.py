# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from csrlite.cm.cm_listing import (
    cm_listing,
    cm_listing_ard,
    cm_listing_rtf,
    study_plan_to_cm_listing,
)


class TestCmListing(unittest.TestCase):
    def setUp(self) -> None:
        self.adsl = pl.DataFrame(
            {
                "USUBJID": ["1", "2"],
                "TRT01P": ["A", "B"],
                "SEX": ["M", "F"],
            }
        )
        self.adcm = pl.DataFrame(
            {
                "USUBJID": ["1", "1"],
                "CMTRT": ["Aspirin", "Tylenol"],
                "CMSEQ": [1, 2],
                "ASTDT": ["2023-01-01", "2023-01-02"],
            }
        )
        self.id = ("USUBJID", "Subject ID")

    def test_cm_listing_ard(self) -> None:
        ard = cm_listing_ard(
            population=self.adsl,
            observation=self.adcm,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            population_columns=[("SEX", "Sex")],
            observation_columns=[("CMTRT", "Medication"), ("ASTDT", "Start Date")],
            sort_columns=["USUBJID", "CMSEQ"],
        )

        # Subjects: 1 has 2 events. 2 has 0 events.
        # ARD should have 2 rows for User 1.
        self.assertEqual(ard.height, 2)

        # Check columns
        self.assertIn("SEX", ard.columns)
        self.assertIn("CMTRT", ard.columns)
        self.assertIn("__index__", ard.columns)

    @patch("csrlite.cm.cm_listing.RTFDocument")
    def test_cm_listing_rtf(self, mock_rtf_doc_cls: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_rtf_doc_cls.return_value = mock_doc

        df = pl.DataFrame({"A": [1], "B": [2]})
        res = cm_listing_rtf(
            df=df,
            column_labels={"A": "Label A"},
            title=["Title"],
            footnote=None,
            source=None,
        )

        self.assertEqual(res, mock_doc)
        mock_rtf_doc_cls.assert_called_once()

    @patch("csrlite.cm.cm_listing.RTFDocument")
    def test_cm_listing_integration(self, mock_rtf_doc_cls: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_rtf_doc_cls.return_value = mock_doc
        output_file = "test.rtf"

        res = cm_listing(
            population=self.adsl,
            observation=self.adcm,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            title=["Title"],
            footnote=None,
            source=None,
            output_file=output_file,
        )

        self.assertEqual(res, output_file)
        mock_doc.write_rtf.assert_called_with(output_file)

    @patch("csrlite.cm.cm_listing.cm_listing")
    def test_study_plan_to_cm_listing(self, mock_cm_listing: MagicMock) -> None:
        mock_cm_listing.return_value = "path.rtf"

        mock_plan = MagicMock()
        mock_plan.output_dir = "out"

        plan_df = pl.DataFrame(
            {"analysis": ["cm_listing"], "population": ["pop1"], "group": ["group1"]}
        )
        mock_plan.get_plan_df.return_value = plan_df

        mock_plan.datasets = {"adsl": self.adsl, "adcm": self.adcm}

        mock_kw_pop = MagicMock()
        mock_kw_pop.filter = "1=1"
        mock_kw_pop.label = "Pop"

        mock_kw_group = MagicMock()
        mock_kw_group.variable = "adsl:TRT01P"
        mock_kw_group.group_label = ["A"]

        mock_plan.keywords.get_population.return_value = mock_kw_pop
        mock_plan.keywords.get_group.return_value = mock_kw_group
        mock_plan.keywords.populations.get.return_value = mock_kw_pop
        
        # Mock observation keyword
        mock_plan.keywords.observations.get.return_value = None

        res = study_plan_to_cm_listing(mock_plan)

        self.assertEqual(res, ["path.rtf"])
        mock_cm_listing.assert_called_once()
