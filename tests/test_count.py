# pyre-strict
import unittest

import polars as pl

from csrlite.common.count import count_subject, count_subject_with_observation


class TestCountSubject(unittest.TestCase):
    def setUp(self) -> None:
        self.population_data = pl.DataFrame(
            {"USUBJID": ["01", "02", "03", "04", "05"], "TRT01A": ["A", "A", "B", "B", "B"]}
        )
        self.observation_data = pl.DataFrame(
            {
                "USUBJID": ["01", "01", "03", "04"],
                "AESOC": ["Infection", "Headache", "Infection", "Headache"],
            }
        )

    def test_count_subject(self) -> None:
        result = count_subject(
            population=self.population_data, id="USUBJID", group="TRT01A", total=True
        )

        # Expected: A=2, B=3, Total=5
        self.assertEqual(result.filter(pl.col("TRT01A") == "A")["n_subj_pop"][0], 2)
        self.assertEqual(result.filter(pl.col("TRT01A") == "B")["n_subj_pop"][0], 3)
        self.assertEqual(result.filter(pl.col("TRT01A") == "Total")["n_subj_pop"][0], 5)

    def test_count_subject_no_total(self) -> None:
        result = count_subject(
            population=self.population_data, id="USUBJID", group="TRT01A", total=False
        )

        self.assertNotIn("Total", result["TRT01A"].to_list())
        self.assertEqual(result.height, 2)

    def test_count_subject_missing_group_error(self) -> None:
        pop_missing = pl.DataFrame({"USUBJID": ["01", "02"], "TRT01A": ["A", None]})

        with self.assertRaisesRegex(ValueError, "Missing values found"):
            count_subject(pop_missing, "USUBJID", "TRT01A", missing_group="error")

    def test_count_subject_duplicate_id_error(self) -> None:
        pop_dup = pl.DataFrame({"USUBJID": ["01", "01"], "TRT01A": ["A", "B"]})

        with self.assertRaisesRegex(ValueError, "not unique"):
            count_subject(pop_dup, "USUBJID", "TRT01A")

    def test_count_subject_with_observation(self) -> None:
        result = count_subject_with_observation(
            population=self.population_data,
            observation=self.observation_data,
            id="USUBJID",
            group="TRT01A",
            variable="AESOC",
            total=True,
        )

        # Check structure
        self.assertIn("n_obs", result.columns)
        self.assertIn("n_subj", result.columns)
        self.assertIn("pct_subj_fmt", result.columns)

        # Check specific values
        # Group A: 2 subjects. Obs: 01 (Infection, Headache).
        # Infection: 1 subj (50%), Headache: 1 subj (50%)

        row_a_inf = result.filter((pl.col("TRT01A") == "A") & (pl.col("AESOC") == "Infection"))
        self.assertEqual(row_a_inf["n_subj"][0], 1)
        self.assertEqual(row_a_inf["pct_subj"][0], 50.0)

        # Group B: 3 subjects. Obs: 03 (Infection), 04 (Headache).
        # Infection: 1 subj (33.3%), Headache: 1 subj (33.3%)
        row_b_inf = result.filter((pl.col("TRT01A") == "B") & (pl.col("AESOC") == "Infection"))
        self.assertEqual(row_b_inf["n_subj"][0], 1)
        self.assertLess(abs(row_b_inf["pct_subj"][0] - 33.3), 0.1)

    def test_count_subject_with_observation_missing_id_in_pop(self) -> None:
        obs_bad = pl.DataFrame(
            {
                "USUBJID": ["99"],  # Not in pop
                "AESOC": ["X"],
            }
        )

        with self.assertRaisesRegex(ValueError, "not present in the population"):
            count_subject_with_observation(
                population=self.population_data,
                observation=obs_bad,
                id="USUBJID",
                group="TRT01A",
                variable="AESOC",
            )
