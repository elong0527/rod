# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from csrlite.ae.ae_specific import (
    ae_specific,
    ae_specific_ard,
    ae_specific_df,
    ae_specific_rtf,
    study_plan_to_ae_specific,
)


class TestAeSpecific(unittest.TestCase):
    def setUp(self) -> None:
        # Dummy data
        self.adsl = pl.DataFrame(
            {
                "USUBJID": ["1", "2", "3", "4"],
                "TRT01P": ["A", "A", "B", "B"],
                "SAFFL": ["Y", "Y", "Y", "Y"],
            }
        )
        self.adae = pl.DataFrame(
            {
                "USUBJID": ["1", "1", "3"],
                "AEDECOD": ["Headache", "Nausea", "Headache"],
                "AESOC": ["Nervous", "Gastro", "Nervous"],
                "AESER": ["N", "N", "Y"],
            }
        )
        self.id = ("USUBJID", "Subject ID")
        self.group = ("TRT01P", "Treatment")
        self.ae_term = ("AEDECOD", "Term")

    def test_ae_specific_ard(self) -> None:
        ard = ae_specific_ard(
            population=self.adsl,
            observation=self.adae,
            population_filter="SAFFL = 'Y'",
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            group=self.group,
            ae_term=self.ae_term,
            total=True,
            missing_group="error",
        )

        # Check structure
        self.assertIn("__index__", ard.columns)
        self.assertIn("__group__", ard.columns)
        self.assertIn("__value__", ard.columns)

        # Check for Summary Rows "with one or more"
        # Group A has User 1 (2 events), User 2 (0 events). Total 2.
        # User 1 has event. So 1/2 (50.0%)
        # But wait, User 1 has 2 events, but counting SUBJECTS with events.
        ard = ard.with_columns(pl.col("__index__").cast(pl.Utf8))
        with_rows = ard.filter(pl.col("__index__").str.contains("with one or more"))
        val_a = with_rows.filter(pl.col("__group__") == "A").select("__value__").item()
        self.assertEqual(val_a, "1 ( 50.0)")

        # Group B has User 3 (1 event), User 4 (0 events). Total 2.
        # User 3 has event. So 1/2 (50.0%)
        val_b = with_rows.filter(pl.col("__group__") == "B").select("__value__").item()
        self.assertEqual(val_b, "1 ( 50.0)")

        # Check for Specific Term "Headache"
        # User 1 (Group A) and User 3 (Group B) have Headache.
        headache_rows = ard.filter(pl.col("__index__").str.to_lowercase() == "headache")
        self.assertFalse(headache_rows.is_empty())

        # Check formatting logic (First letter upper, rest lower)
        # "Headache" -> "Headache"
        # "Nausea" -> "Nausea"

    def test_ae_specific_df(self) -> None:
        ard = pl.DataFrame(
            {
                "__index__": ["Term1", "Term1"],
                "__group__": ["A", "B"],
                "__value__": ["1", "2"],
            }
        )
        df = ae_specific_df(ard)
        self.assertIn("Term", df.columns)
        self.assertIn("A", df.columns)
        self.assertIn("B", df.columns)
        self.assertEqual(df["Term"][0], "Term1")

    @patch("csrlite.ae.ae_specific.create_rtf_table_n_pct")
    def test_ae_specific_rtf(self, mock_create_table: MagicMock) -> None:
        df = pl.DataFrame(
            {
                "Term": ["Term1"],
                "A": ["1 (50%)"],
                "B": ["0 (0%)"],
            }
        )
        mock_doc = MagicMock()
        mock_create_table.return_value = mock_doc

        res = ae_specific_rtf(
            df=df,
            title=["Title"],
            footnote=["Footnote"],
            source=["Source"],
        )

        self.assertEqual(res, mock_doc)
        mock_create_table.assert_called_once()

    @patch("csrlite.ae.ae_specific.create_rtf_table_n_pct")
    def test_ae_specific_integration(self, mock_create_table: MagicMock) -> None:
        mock_doc = MagicMock()
        mock_create_table.return_value = mock_doc
        output_file = "test_output.rtf"

        res = ae_specific(
            population=self.adsl,
            observation=self.adae,
            population_filter="SAFFL = 'Y'",
            observation_filter=None,
            parameter_filter=None,
            id=self.id,
            group=self.group,
            ae_term=self.ae_term,
            title=["Title"],
            footnote=None,
            source=None,
            output_file=output_file,
        )

        self.assertEqual(res, output_file)
        mock_doc.write_rtf.assert_called_with(output_file)

    @patch("csrlite.ae.ae_specific.ae_specific")
    def test_study_plan_to_ae_specific(self, mock_ae_specific: MagicMock) -> None:
        mock_ae_specific.return_value = "path/to/output.rtf"

        mock_plan = MagicMock()
        mock_plan.output_dir = "outputs"

        # Mock Plan DF
        plan_df = pl.DataFrame(
            {
                "analysis": ["ae_specific"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["group1"],
            }
        )
        mock_plan.get_plan_df.return_value = plan_df

        mock_plan.datasets = {"adsl": self.adsl, "adae": self.adae}

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

        mock_plan.keywords.get_population.return_value = mock_kw_pop
        mock_plan.keywords.get_observation.return_value = mock_kw_obs
        mock_plan.keywords.get_parameter.return_value = mock_kw_param
        mock_plan.keywords.get_group.return_value = mock_kw_group

        mock_plan.keywords.populations.get.return_value = mock_kw_pop
        mock_plan.keywords.observations.get.return_value = mock_kw_obs

        res = study_plan_to_ae_specific(mock_plan)

        self.assertEqual(res, ["path/to/output.rtf"])
        mock_ae_specific.assert_called_once()
