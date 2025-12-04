# pyre-strict
import unittest
import polars as pl

from tlfyaml.utils import apply_common_filters


class TestUtils(unittest.TestCase):
    def test_apply_common_filters_no_filters(self):
        pop = pl.DataFrame({"id": [1, 2, 3], "group": ["A", "B", "A"]})
        obs = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

        res_pop, res_obs = apply_common_filters(pop, obs, None, None, None)

        self.assertTrue(res_pop.equals(pop))
        self.assertTrue(res_obs.equals(obs))

    def test_apply_common_filters_population_filter(self):
        pop = pl.DataFrame({"id": [1, 2, 3], "group": ["A", "B", "A"]})
        obs = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

        # Filter for group A
        res_pop, res_obs = apply_common_filters(pop, obs, "group == 'A'", None, None)

        expected_pop = pop.filter(pl.col("group") == "A")
        self.assertTrue(res_pop.equals(expected_pop))
        self.assertTrue(
            res_obs.equals(obs)
        )  # Observation not filtered by population filter directly in this function

    def test_apply_common_filters_observation_filter(self):
        pop = pl.DataFrame({"id": [1, 2, 3]})
        obs = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

        res_pop, res_obs = apply_common_filters(pop, obs, None, "val > 15", None)

        expected_obs = obs.filter(pl.col("val") > 15)
        self.assertTrue(res_pop.equals(pop))
        self.assertTrue(res_obs.equals(expected_obs))

    def test_apply_common_filters_parameter_filter(self):
        pop = pl.DataFrame({"id": [1, 2, 3]})
        obs = pl.DataFrame({"id": [1, 2, 3], "param": ["X", "Y", "X"]})

        res_pop, res_obs = apply_common_filters(pop, obs, None, None, "param == 'X'")

        expected_obs = obs.filter(pl.col("param") == "X")
        self.assertTrue(res_pop.equals(pop))
        self.assertTrue(res_obs.equals(expected_obs))

    def test_apply_common_filters_all_filters(self):
        pop = pl.DataFrame({"id": [1, 2, 3], "group": ["A", "B", "A"]})
        obs = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30], "param": ["X", "Y", "X"]})

        res_pop, res_obs = apply_common_filters(pop, obs, "group == 'A'", "val > 0", "param == 'X'")

        expected_pop = pop.filter(pl.col("group") == "A")
        expected_obs = obs.filter((pl.col("val") > 0) & (pl.col("param") == "X"))

        self.assertTrue(res_pop.equals(expected_pop))
        self.assertTrue(res_obs.equals(expected_obs))
