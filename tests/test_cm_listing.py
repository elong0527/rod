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

    def test_study_plan_to_cm_listing_missing_group(self) -> None:
        mock_plan = MagicMock()
        mock_plan.output_dir = "out"

        # Plan without group
        plan_df = pl.DataFrame(
            {"analysis": ["cm_listing"], "population": ["pop1"], "group": [None]}
        )
        mock_plan.get_plan_df.return_value = plan_df

        with self.assertRaises(ValueError) as cm:
            study_plan_to_cm_listing(mock_plan)

        self.assertIn("Group not specified", str(cm.exception))

    def test_cm_listing_custom_columns(self) -> None:
        # Test overriding columns
        ard = cm_listing_ard(
            population=self.adsl,
            observation=self.adcm,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            population_columns=[("SEX", "Test Sex")],
            observation_columns=[("ASTDT", "Test Date")],
            sort_columns=["USUBJID"],
        )

        # Check if custom labels/cols are used
        self.assertIn("SEX", ard.columns)
        self.assertIn("ASTDT", ard.columns)
        self.assertNotIn("CMTRT", ard.columns)  # Should be excluded as we overrode it

        # The index row for population should use "Test Sex" label
        # (Though ARD generation puts labels in __index__ but for listing it's column based?
        # Actually cm_listing_ard returns joined DF.
        pass

    def test_cm_listing_ard_defaults(self) -> None:
        # Test usage of defaults (None for columns)
        ard = cm_listing_ard(
            population=self.adsl,
            observation=self.adcm,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            population_columns=None,  # Should just select ID
            observation_columns=None,  # Should just select ID
            sort_columns=None,
        )

        # Check columns
        self.assertIn("USUBJID", ard.columns)
        # Should contain __index__
        self.assertIn("__index__", ard.columns)

        # Should not contain other cols unless joined (but pop cols is None)
        # However cm_listing_ard logic for None obs cols is just ID.
        self.assertNotIn("CMTRT", ard.columns)
        self.assertNotIn("TRT01P", ard.columns)  # No join happened

    def test_cm_listing_ard_page_by(self) -> None:
        # Test page_by logic
        ard = cm_listing_ard(
            population=self.adsl,
            observation=self.adcm,
            population_filter=None,
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            population_columns=[("TRT01P", "Treatment")],
            observation_columns=[("CMTRT", "Medication")],
            sort_columns=None,
            page_by=["TRT01P"],  # Valid page_by col
        )

        # __index__ should contain page info
        # User 1 is in Group A.
        # index should contain 'Treatment = A'
        user1_rows = ard.filter(pl.col("USUBJID") == "1")
        index_val = user1_rows["__index__"][0]
        self.assertIn("Treatment = A", index_val)

        # page_by columns should be removed from main cols (except ID)
        self.assertNotIn("TRT01P", ard.columns)

    @patch("csrlite.cm.cm_listing.RTFDocument")
    def test_cm_listing_rtf_custom_width(self, mock_rtf_doc_cls: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_rtf_doc_cls.return_value = mock_doc

        df = pl.DataFrame({"A": [1], "B": [2]})
        cm_listing_rtf(
            df=df,
            column_labels={"A": "Label A"},
            title=["Title"],
            footnote=None,
            source=None,
            col_rel_width=[2.0, 1.0],
        )

        # Verify call args
        args, kwargs = mock_rtf_doc_cls.call_args
        rtf_body = kwargs["rtf_body"]
        self.assertEqual(rtf_body.col_rel_width, [2.0, 1.0])

    @patch("csrlite.cm.cm_listing.cm_listing")
    def test_study_plan_to_cm_listing_parameter(self, mock_cm_listing: MagicMock) -> None:
        mock_cm_listing.return_value = "path.rtf"

        mock_plan = MagicMock()
        mock_plan.output_dir = "out"

        # Plan with parameter
        plan_df = pl.DataFrame(
            {
                "analysis": ["cm_listing"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["group1"],
            }
        )
        mock_plan.get_plan_df.return_value = plan_df
        mock_plan.datasets = {"adsl": self.adsl, "adcm": self.adcm}

        mock_kw_pop = MagicMock()
        mock_kw_pop.filter = "1=1"

        mock_kw_obs = MagicMock()
        mock_kw_obs.filter = "1=1"

        mock_kw_param = MagicMock()
        mock_kw_param.filter = "PARAM='X'"  # Should be used
        mock_kw_param.label = "Param Label"

        mock_kw_group = MagicMock()
        mock_kw_group.variable = "adsl:TRT01P"
        mock_kw_group.group_label = ["A"]

        mock_plan.keywords.get_population.return_value = mock_kw_pop
        mock_plan.keywords.get_observation.return_value = mock_kw_obs
        mock_plan.keywords.get_parameter.return_value = mock_kw_param
        mock_plan.keywords.get_group.return_value = mock_kw_group

        mock_plan.keywords.populations.get.return_value = mock_kw_pop
        mock_plan.keywords.observations.get.return_value = mock_kw_obs

        study_plan_to_cm_listing(mock_plan)

        mock_cm_listing.assert_called_once()
        args, kwargs = mock_cm_listing.call_args
        self.assertEqual(kwargs["parameter_filter"], "PARAM='X'")

    @patch("csrlite.cm.cm_listing.cm_listing")
    def test_cm_listing_title_missing_labels(self, mock_cm_listing: MagicMock) -> None:
        mock_cm_listing.return_value = "path.rtf"

        mock_plan = MagicMock()
        mock_plan.output_dir = "out"

        # Plan with parameter
        plan_df = pl.DataFrame(
            {
                "analysis": ["cm_listing"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["group1"],
            }
        )
        mock_plan.get_plan_df.return_value = plan_df
        mock_plan.datasets = {"adsl": self.adsl, "adcm": self.adcm}

        mock_kw_pop = MagicMock()
        mock_kw_pop.filter = "1=1"
        mock_kw_pop.label = None  # Missing label

        mock_kw_obs = MagicMock()
        mock_kw_obs.filter = "1=1"
        mock_kw_obs.label = None  # Missing label

        mock_kw_param = MagicMock()
        mock_kw_param.filter = "PARAM='X'"
        mock_kw_param.label = "Param Label"

        mock_kw_group = MagicMock()
        mock_kw_group.variable = "adsl:TRT01P"
        mock_kw_group.group_label = ["A"]

        # Setup getting keywords
        mock_plan.keywords.get_population.return_value = mock_kw_pop
        mock_plan.keywords.get_observation.return_value = mock_kw_obs
        mock_plan.keywords.get_parameter.return_value = mock_kw_param
        mock_plan.keywords.get_group.return_value = mock_kw_group

        mock_plan.keywords.populations.get.return_value = mock_kw_pop
        mock_plan.keywords.observations.get.return_value = mock_kw_obs

        study_plan_to_cm_listing(mock_plan)

        # Check title
        args, kwargs = mock_cm_listing.call_args
        title = kwargs["title"]
        # Base title is "Listing of Concomitant Medications"
        self.assertEqual(len(title), 1)
        self.assertEqual(title[0], "Listing of Concomitant Medications")
