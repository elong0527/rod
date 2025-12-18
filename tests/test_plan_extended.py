import unittest
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import polars as pl
import yaml

from csrlite.common.plan import KeywordRegistry, StudyPlan, load_plan


class TestPlanExtended(unittest.TestCase):
    def setUp(self):
        self.study_data = {
            "study": {"name": "Test Study", "output": "output"},
            "data": [{"name": "adsl", "path": "adsl.parquet"}],
            "population": [{"name": "safety", "label": "Safety Set", "filter": "SAFFL == 'Y'"}],
            "observation": [
                {"name": "ae_obs", "label": "Adverse Events", "filter": "AESER == 'Y'"}
            ],
            "parameter": [{"name": "any_ae", "label": "Any AE", "filter": "True"}],
            "group": [{"name": "treatment", "variable": "TRT01P", "label": ["A", "B"]}],
            "plans": [{"analysis": "ae_summary", "population": "safety", "group": "treatment"}],
        }

    @patch("csrlite.common.plan.pl.read_parquet")
    def test_get_dfs(self, mock_read):
        mock_read.return_value = pl.DataFrame()
        plan = StudyPlan(self.study_data)

        # Test get_dataset_df
        df_ds = plan.get_dataset_df()
        self.assertIsNotNone(df_ds)
        self.assertEqual(df_ds.item(0, "name"), "adsl")
        self.assertEqual(df_ds.item(0, "path"), "adsl.parquet")

        # Test get_population_df
        df_pop = plan.get_population_df()
        self.assertIsNotNone(df_pop)
        self.assertEqual(df_pop.item(0, "name"), "safety")

        # Test get_observation_df
        df_obs = plan.get_observation_df()
        self.assertIsNotNone(df_obs)
        self.assertEqual(df_obs.item(0, "name"), "ae_obs")

        # Test get_parameter_df
        df_param = plan.get_parameter_df()
        self.assertIsNotNone(df_param)
        self.assertEqual(df_param.item(0, "name"), "any_ae")

        # Test get_group_df
        df_group = plan.get_group_df()
        self.assertIsNotNone(df_group)
        self.assertEqual(df_group.item(0, "name"), "treatment")

        # Test get_plan_df
        df_plan = plan.get_plan_df()
        self.assertIsNotNone(df_plan)
        self.assertEqual(len(df_plan), 1)

    @patch("csrlite.common.plan.pl.read_parquet")
    @patch("csrlite.common.plan.logger")
    def test_print(self, mock_logger, mock_read):
        mock_read.return_value = pl.DataFrame()
        plan = StudyPlan(self.study_data)

        plan.print()

        # Verify logger was called multiple times
        self.assertTrue(mock_logger.info.call_count >= 6)

    @patch("csrlite.common.plan.pl.read_parquet")
    def test_str(self, mock_read):
        mock_read.return_value = pl.DataFrame()
        plan = StudyPlan(self.study_data)
        s = str(plan)
        self.assertIn("StudyPlan(study='Test Study'", s)
        self.assertIn("plans=1", s)

    @patch("csrlite.common.plan.pl.read_parquet")
    def test_empty_plan(self, mock_read):
        empty_data = {"study": {"name": "Empty"}}
        plan = StudyPlan(empty_data)

        # Test None returns for empty registries
        self.assertIsNone(plan.get_dataset_df())
        self.assertIsNone(plan.get_population_df())
        self.assertIsNone(plan.get_observation_df())
        self.assertIsNone(plan.get_parameter_df())
        self.assertIsNone(plan.get_group_df())

        # Test _generate_title (technically private, but useful to cover if used by expander)
        # We need a dummy plan to pass to it
        from csrlite.common.plan import AnalysisPlan

        dummy_plan = AnalysisPlan(
            analysis="test", population="pop", observation=None, parameter=None
        )
        title = plan.expander._generate_title(dummy_plan)
        self.assertIn("Test", title)

    def test_keyword_registry_group_label_handling(self):
        # Case 1: label is list, group_label missing -> auto-migrated
        data = {"group": [{"name": "g1", "variable": "v1", "label": ["A", "B"]}]}
        reg = KeywordRegistry()
        reg.load_from_dict(data)
        self.assertEqual(reg.groups["g1"].group_label, ["A", "B"])

        # Case 2: group_label explicitly provided
        data2 = {"group": [{"name": "g2", "variable": "v2", "group_label": ["X", "Y"]}]}
        reg2 = KeywordRegistry()
        reg2.load_from_dict(data2)
        self.assertEqual(reg2.groups["g2"].group_label, ["X", "Y"])

    def test_load_plan_wrapper(self):
        import os

        tmp_name = ""
        try:
            with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.dump(self.study_data, tmp)
                tmp_name = tmp.name

            with patch("csrlite.common.plan.pl.read_parquet") as mock_read:
                mock_read.return_value = pl.DataFrame()
                plan = load_plan(tmp_name)
                self.assertIsNotNone(plan)
                self.assertEqual(plan.study_data["study"]["name"], "Test Study")
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
