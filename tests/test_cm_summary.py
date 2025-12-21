# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from csrlite.cm.cm_summary import (
    cm_summary,
    cm_summary_ard,
    cm_summary_df,
    cm_summary_rtf,
    study_plan_to_cm_summary,
)


class TestCmSummary(unittest.TestCase):
    def setUp(self) -> None:
        # Create dummy data
        self.adsl = pl.DataFrame(
            {
                "USUBJID": ["1", "2", "3", "4"],
                "TRT01P": ["A", "A", "B", "B"],
                "SAFFL": ["Y", "Y", "Y", "Y"],
            }
        )
        self.adcm = pl.DataFrame(
            {
                "USUBJID": ["1", "3"],
                "CMDECOD": ["Drug1", "Drug2"],
                "CONFL": ["Y", "Y"],
            }
        )
        self.id = ("USUBJID", "Subject ID")
        self.group = ("TRT01P", "Treatment")

    def test_cm_summary_ard(self) -> None:
        variables = [("1=1", "Any Medication")]

        ard = cm_summary_ard(
            population=self.adsl,
            observation=self.adcm,
            population_filter="SAFFL = 'Y'",
            observation_filter=None,
            id=self.id,
            group=self.group,
            variables=variables,
            total=True,
            missing_group="error",
        )

        # Check structure
        self.assertIn("__index__", ard.columns)
        self.assertIn("__group__", ard.columns)
        self.assertIn("__value__", ard.columns)

        # Confirm we have population rows
        pop_rows = ard.filter(pl.col("__index__") == "Participants in population")
        self.assertGreater(pop_rows.height, 0)

        # Confirm we have variable rows
        var_rows = ard.filter(pl.col("__index__") == "Any Medication")
        self.assertGreater(var_rows.height, 0)

        # Check specific value for Any Medication in group A (Subject 1 has it)
        # Subject 1 is in Group A. Total in Group A is 2.
        # So 1/2 (50.0%)
        val_a = var_rows.filter(pl.col("__group__") == "A").select("__value__").item()
        self.assertEqual(val_a, "1 ( 50.0)")

        # Check Total column exists
        self.assertFalse(ard.filter(pl.col("__group__") == "Total").is_empty())

    def test_cm_summary_df(self) -> None:
        # create a minimal ARD
        ard = pl.DataFrame(
            {
                "__index__": ["Row1", "Row1"],
                "__group__": ["A", "B"],
                "__value__": ["1", "2"],
            }
        )
        df = cm_summary_df(ard)
        self.assertIn("A", df.columns)
        self.assertIn("B", df.columns)
        self.assertEqual(df.filter(pl.col("__index__") == "Row1")["A"][0], "1")

    @patch("csrlite.cm.cm_summary.create_rtf_table_n_pct")
    def test_cm_summary_rtf(self, mock_create_table: MagicMock) -> None:
        df = pl.DataFrame(
            {
                "__index__": ["Row1"],
                "A": ["1 (50%)"],
                "B": ["0 (0%)"],
            }
        )
        mock_doc = MagicMock()
        mock_create_table.return_value = mock_doc

        res = cm_summary_rtf(
            df=df,
            title=["Title"],
            footnote=["Footnote"],
            source=["Source"],
        )

        self.assertEqual(res, mock_doc)
        mock_create_table.assert_called_once()

    @patch("csrlite.cm.cm_summary.create_rtf_table_n_pct")
    def test_cm_summary_integration(self, mock_create_table: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_create_table.return_value = mock_doc

        output_file = "test_output.rtf"
        variables = [("1=1", "Any Medication")]

        res = cm_summary(
            population=self.adsl,
            observation=self.adcm,
            population_filter="SAFFL = 'Y'",
            observation_filter=None,
            id=self.id,
            group=self.group,
            variables=variables,
            title=["Title"],
            footnote=None,
            source=None,
            output_file=output_file,
        )

        self.assertEqual(res, output_file)
        mock_doc.write_rtf.assert_called_with(output_file)

    @patch("csrlite.cm.cm_summary.cm_summary")
    def test_study_plan_to_cm_summary(self, mock_cm_summary: MagicMock) -> None:
        mock_cm_summary.return_value = "path/to/output.rtf"

        # Mock StudyPlan
        mock_plan = MagicMock()
        mock_plan.output_dir = "outputs"

        # Mock Plan DF
        plan_df = pl.DataFrame(
            {
                "analysis": ["cm_summary"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["group1"],
            }
        )
        mock_plan.get_plan_df.return_value = plan_df

        # Mock datasets
        mock_plan.datasets = {"adsl": self.adsl, "adcm": self.adcm}

        # Mock keywords
        mock_kw_pop = MagicMock()
        mock_kw_pop.filter = "filter1"
        mock_kw_pop.label = "Pop Label"

        mock_kw_obs = MagicMock()
        mock_kw_obs.filter = "filter2"
        mock_kw_obs.label = "Obs Label"

        mock_kw_param = MagicMock()
        mock_kw_param.filter = "filter3"
        mock_kw_param.label = "Param Label"
        mock_kw_param.indent = 0

        mock_kw_group = MagicMock()
        mock_kw_group.variable = "adsl:TRT01P"
        mock_kw_group.group_label = ["A", "B"]

        # Configure lookups
        mock_plan.keywords.get_population.return_value = mock_kw_pop
        mock_plan.keywords.get_observation.return_value = mock_kw_obs
        mock_plan.keywords.get_parameter.return_value = mock_kw_param
        mock_plan.keywords.get_group.return_value = mock_kw_group

        # Dictionary access for titles
        mock_plan.keywords.populations.get.return_value = mock_kw_pop
        mock_plan.keywords.observations.get.return_value = mock_kw_obs

        # Run
        res = study_plan_to_cm_summary(mock_plan)

        self.assertEqual(res, ["path/to/output.rtf"])
        mock_cm_summary.assert_called_once()
