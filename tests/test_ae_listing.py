# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from csrlite.ae.ae_listing import (
    ae_listing,
    ae_listing_ard,
    ae_listing_rtf,
    study_plan_to_ae_listing,
)


class TestAeListing(unittest.TestCase):
    def setUp(self) -> None:
        self.adsl = pl.DataFrame(
            {
                "USUBJID": ["1", "2"],
                "TRT01P": ["A", "B"],
                "SEX": ["M", "F"],
            }
        )
        self.adae = pl.DataFrame(
            {
                "USUBJID": ["1", "1"],
                "AEDECOD": ["Headache", "Nausea"],
                "AESER": ["N", "N"],
                "ASTDY": [1, 5],
            }
        )
        self.id = ("USUBJID", "Subject ID")

    def test_ae_listing_ard(self) -> None:
        ard = ae_listing_ard(
            population=self.adsl,
            observation=self.adae,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            population_columns=[("SEX", "Sex")],
            observation_columns=[("AEDECOD", "AE Term"), ("ASTDY", "Day")],
            sort_columns=["USUBJID", "ASTDY"],
        )

        # Subjects: 1 has 2 events. 2 has 0 events.
        # ARD should have 2 rows for User 1.
        # User 2 absent from ADAE, so should not be in listing?
        # Logic says: left join result (observation) to population?
        # Code: result = observation_filtered.join(population_subset, left)
        # So only subjects WITH events are kept.

        self.assertEqual(ard.height, 2)

        # Check columns
        self.assertIn("SEX", ard.columns)
        self.assertIn("AEDECOD", ard.columns)
        self.assertIn("__index__", ard.columns)

        # Check sorting
        days = ard["ASTDY"].to_list()
        self.assertEqual(days, [1, 5])

    def test_ae_listing_ard_page_by(self) -> None:
        ard = ae_listing_ard(
            population=self.adsl,
            observation=self.adae,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            population_columns=[("SEX", "Sex")],
            page_by=["SEX"],
        )
        # Check __index__ formation
        # Should contain "Sex = M" for User 1
        idx = ard["__index__"][0]
        self.assertIn("Sex", idx)
        self.assertIn("M", idx)
        self.assertFalse("SEX" in ard.columns)  # Dropped from main cols

    @patch("csrlite.ae.ae_listing.RTFDocument")
    def test_ae_listing_rtf(self, mock_rtf_doc_cls: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_rtf_doc_cls.return_value = mock_doc

        df = pl.DataFrame({"A": [1], "B": [2]})
        res = ae_listing_rtf(
            df=df,
            column_labels={"A": "Label A"},
            title=["Title"],
            footnote=None,
            source=None,
        )

        self.assertEqual(res, mock_doc)
        mock_rtf_doc_cls.assert_called_once()

    @patch("csrlite.ae.ae_listing.RTFDocument")
    def test_ae_listing_integration(self, mock_rtf_doc_cls: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_rtf_doc_cls.return_value = mock_doc
        output_file = "test.rtf"

        res = ae_listing(
            population=self.adsl,
            observation=self.adae,
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

    @patch("csrlite.ae.ae_listing.ae_listing")
    def test_study_plan_to_ae_listing(self, mock_ae_listing: MagicMock) -> None:
        mock_ae_listing.return_value = "path.rtf"

        mock_plan = MagicMock()
        mock_plan.output_dir = "out"

        plan_df = pl.DataFrame(
            {"analysis": ["ae_listing"], "population": ["pop1"], "group": ["group1"]}
        )
        mock_plan.get_plan_df.return_value = plan_df

        mock_plan.datasets = {"adsl": self.adsl, "adae": self.adae}

        mock_kw_pop = MagicMock()
        mock_kw_pop.filter = "1=1"
        mock_kw_pop.label = "Pop"

        mock_kw_group = MagicMock()
        mock_kw_group.variable = "adsl:TRT01P"
        mock_kw_group.group_label = ["A"]

        mock_plan.keywords.get_population.return_value = mock_kw_pop
        mock_plan.keywords.get_group.return_value = mock_kw_group
        mock_plan.keywords.populations.get.return_value = mock_kw_pop

        # Mock default behavior for observation filter (None)
        mock_plan.keywords.get_observation.return_value = None

        res = study_plan_to_ae_listing(mock_plan)

        self.assertEqual(res, ["path.rtf"])
        mock_ae_listing.assert_called_once()
