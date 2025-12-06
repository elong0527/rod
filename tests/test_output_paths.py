# pyre-strict
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl

from csrlite.ae.ae_listing import study_plan_to_ae_listing
from csrlite.ae.ae_specific import study_plan_to_ae_specific
from csrlite.ae.ae_summary import study_plan_to_ae_summary
from csrlite.disposition.disposition import study_plan_to_disposition_summary


class TestOutputPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_plan = MagicMock()
        self.mock_plan.output_dir = "custom/output/dir"

        # Mock datasets
        self.mock_plan.datasets = {
            "adsl": pl.DataFrame({"USUBJID": ["1"], "TRT01A": ["A"]}),
            "adae": pl.DataFrame({"USUBJID": ["1"], "AEDECOD": ["Headache"]}),
            "ds": pl.DataFrame({"USUBJID": ["1"], "DSDECOD": ["Completed"]}),
        }

        # Mock keyword objects
        mock_pop = MagicMock()
        mock_pop.filter = ""
        mock_pop.label = "Safety Population"

        mock_obs = MagicMock()
        mock_obs.filter = ""
        mock_obs.label = "Treatment Emergent"

        mock_group = MagicMock()
        mock_group.variable = "adsl:TRT01A"
        mock_group.group_label = ["Treatment A"]

        mock_param = MagicMock()
        mock_param.filter = ""
        mock_param.label = "Any AE"
        mock_param.indent = 0

        # Mock keyword resolution
        self.mock_plan.keywords.get_population.return_value = mock_pop
        self.mock_plan.keywords.get_observation.return_value = mock_obs
        self.mock_plan.keywords.get_group.return_value = mock_group
        self.mock_plan.keywords.get_parameter.return_value = mock_param

        # Mock dictionaries for direct access (used in title generation)
        self.mock_plan.keywords.populations = {"pop1": mock_pop}
        self.mock_plan.keywords.observations = {"obs1": mock_obs}

    @patch("csrlite.ae.ae_listing.ae_listing")
    def test_ae_listing_output_path(self, mock_ae_listing: MagicMock) -> None:
        mock_ae_listing.side_effect = lambda **kwargs: kwargs["output_file"]

        self.mock_plan.get_plan_df.return_value = pl.DataFrame(
            {
                "analysis": ["ae_listing"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["grp1"],
                "name": ["test_analysis"],
            }
        )

        output_files = study_plan_to_ae_listing(self.mock_plan)

        self.assertEqual(len(output_files), 1)
        self.assertEqual(
            Path(output_files[0]),
            Path("custom/output/dir/ae_listing_pop1_obs1_param1.rtf"),
        )

    @patch("csrlite.ae.ae_summary.ae_summary")
    def test_ae_summary_output_path(self, mock_ae_summary: MagicMock) -> None:
        mock_ae_summary.side_effect = lambda **kwargs: kwargs["output_file"]

        self.mock_plan.get_plan_df.return_value = pl.DataFrame(
            {
                "analysis": ["ae_summary"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["grp1"],
                "name": ["test_analysis"],
            }
        )

        output_files = study_plan_to_ae_summary(self.mock_plan)

        self.assertEqual(len(output_files), 1)
        self.assertEqual(
            Path(output_files[0]),
            Path("custom/output/dir/ae_summary_pop1_obs1_param1.rtf"),
        )

    @patch("csrlite.ae.ae_specific.ae_specific")
    def test_ae_specific_output_path(self, mock_ae_specific: MagicMock) -> None:
        mock_ae_specific.side_effect = lambda **kwargs: kwargs["output_file"]

        self.mock_plan.get_plan_df.return_value = pl.DataFrame(
            {
                "analysis": ["ae_specific"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["grp1"],
                "name": ["test_analysis"],
            }
        )

        output_files = study_plan_to_ae_specific(self.mock_plan)

        self.assertEqual(len(output_files), 1)
        self.assertEqual(
            Path(output_files[0]),
            Path("custom/output/dir/ae_specific_pop1_obs1_param1.rtf"),
        )

    @patch("csrlite.disposition.disposition.disposition")
    def test_disposition_output_path(self, mock_disposition: MagicMock) -> None:
        mock_disposition.side_effect = lambda **kwargs: kwargs["output_file"]

        self.mock_plan.get_plan_df.return_value = pl.DataFrame(
            {
                "analysis": ["disposition_summary"],
                "population": ["pop1"],
                "observation": ["obs1"],
                "parameter": ["param1"],
                "group": ["grp1"],
                "name": ["test_analysis"],
            }
        )

        output_files = study_plan_to_disposition_summary(self.mock_plan)

        self.assertEqual(len(output_files), 1)
        self.assertEqual(
            Path(output_files[0]),
            Path("custom/output/dir/disposition_summary_pop1_grp1.rtf"),
        )
