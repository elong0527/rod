# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

from csrlite.common.plan import KeywordRegistry, PlanExpander, StudyPlan


class TestPlan(unittest.TestCase):
    def test_plan_expander_simple(self) -> None:
        registry = KeywordRegistry()
        expander = PlanExpander(registry)

        plan_data = {
            "analysis": "ae_summary",
            "population": "pop1",
            "observation": "obs1",
            "parameter": "param1",
            "group": "grp1",
        }

        plans = expander.expand_plan(plan_data)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].analysis, "ae_summary")
        self.assertEqual(plans[0].population, "pop1")
        self.assertEqual(plans[0].observation, "obs1")
        self.assertEqual(plans[0].parameter, "param1")

    def test_plan_expander_multiple(self) -> None:
        registry = KeywordRegistry()
        expander = PlanExpander(registry)

        plan_data = {
            "analysis": "ae_summary",
            "population": ["pop1", "pop2"],
            "observation": ["obs1"],
            "parameter": ["param1", "param2"],
            "group": "grp1",
        }

        plans = expander.expand_plan(plan_data)

        # 2 pops * 1 obs * 2 params = 4 plans
        self.assertEqual(len(plans), 4)

        ids = {p.id for p in plans}
        self.assertIn("ae_summary_pop1_obs1_param1", ids)
        self.assertIn("ae_summary_pop2_obs1_param2", ids)

    def test_study_plan_init(self) -> None:
        study_data = {
            "study": {"name": "Test Study"},
            "data": [{"name": "adsl", "path": "adsl.parquet"}],
            "plans": [],
        }

        with patch("csrlite.common.plan.pl.read_parquet") as mock_read:
            mock_read.return_value = MagicMock()

            plan = StudyPlan(study_data)

            self.assertIn("adsl", plan.datasets)
            self.assertIsNotNone(plan.keywords.get_data_source("adsl"))
