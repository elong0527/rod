# pyre-strict
import unittest
from unittest.mock import MagicMock

import polars as pl

from csrlite.common.parse import (
    StudyPlanParser,
    apply_filter_sql,
    parse_filter_to_sql,
    parse_parameter,
)


class TestParse(unittest.TestCase):
    def test_parse_filter_to_sql_simple(self) -> None:
        self.assertEqual(parse_filter_to_sql("adsl:saffl == 'Y'"), "SAFFL = 'Y'")

    def test_parse_filter_to_sql_and(self) -> None:
        res = parse_filter_to_sql("adsl:saffl == 'Y' and adsl:sex == 'M'")
        self.assertEqual(res, "SAFFL = 'Y' AND SEX = 'M'")

    def test_parse_filter_to_sql_in(self) -> None:
        res = parse_filter_to_sql("adae:aerel in ['A', 'B']")
        self.assertEqual(res, "AEREL IN ('A', 'B')")

    def test_parse_filter_to_sql_empty(self) -> None:
        self.assertEqual(parse_filter_to_sql(""), "1=1")

    def test_parse_parameter_single(self) -> None:
        self.assertEqual(parse_parameter("param1"), ["param1"])

    def test_parse_parameter_multiple(self) -> None:
        self.assertEqual(parse_parameter("p1;p2; p3"), ["p1", "p2", "p3"])

    def test_parse_parameter_empty(self) -> None:
        self.assertEqual(parse_parameter(""), [])


class TestApplyFilterSql(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "A": [1, 2, 3],
                "B": ["x", "y", "z"],
            }
        )

    def test_apply_filter_sql_basic(self) -> None:
        res = apply_filter_sql(self.df, "df:a > 1")
        self.assertTrue(isinstance(res, pl.DataFrame))
        self.assertEqual(res.height, 2)
        self.assertEqual(res["A"].to_list(), [2, 3])

    def test_apply_filter_sql_empty(self) -> None:
        res = apply_filter_sql(self.df, "")
        self.assertEqual(res.height, 3)

    def test_apply_filter_fallback(self) -> None:
        # Mock pl.sql_expr to raise an exception, forcing fallback
        original_sql_expr = pl.sql_expr
        try:
            pl.sql_expr = MagicMock(side_effect=Exception("SQL Error"))

            # This should trigger the fallback path which uses _parse_filter_expr
            # We use a simple filter that _parse_filter_expr can handle
            # "b == 'y'" -> pl.col('B') == 'y'
            res = apply_filter_sql(self.df, "df:b == 'y'")

            self.assertEqual(res.height, 1)
            self.assertEqual(res["B"][0], "y")

            # Also test 'in' operator in fallback
            res_in = apply_filter_sql(self.df, "df:a in [1, 2]")
            self.assertEqual(res_in.height, 2)

        finally:
            pl.sql_expr = original_sql_expr


class TestStudyPlanParser(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_plan = MagicMock()
        self.parser = StudyPlanParser(self.mock_plan)

    def test_get_population_filter(self) -> None:
        mock_pop = MagicMock()
        mock_pop.filter = "adsl:saffl == 'Y'"
        self.mock_plan.keywords.get_population.return_value = mock_pop

        res = self.parser.get_population_filter("saffl")
        self.assertEqual(res, "SAFFL = 'Y'")
        self.mock_plan.keywords.get_population.assert_called_with("saffl")

    def test_get_population_filter_not_found(self) -> None:
        self.mock_plan.keywords.get_population.return_value = None
        with self.assertRaises(ValueError):
            self.parser.get_population_filter("missing")

    def test_get_observation_filter(self) -> None:
        mock_obs = MagicMock()
        mock_obs.filter = "adae:rel == 'Y'"
        self.mock_plan.keywords.get_observation.return_value = mock_obs

        res = self.parser.get_observation_filter("obs")
        self.assertEqual(res, "REL = 'Y'")

    def test_get_observation_filter_none(self) -> None:
        res = self.parser.get_observation_filter(None)
        self.assertIsNone(res)

    def test_get_observation_filter_not_found(self) -> None:
        self.mock_plan.keywords.get_observation.return_value = None
        res = self.parser.get_observation_filter("missing")
        self.assertIsNone(res)

    def test_get_parameter_info(self) -> None:
        mock_p1 = MagicMock()
        mock_p1.filter = "p1 > 0"
        mock_p1.label = "Param 1"
        mock_p1.indent = 0

        mock_p2 = MagicMock()
        mock_p2.filter = "p2 < 0"
        mock_p2.label = None
        mock_p2.indent = 1

        mock_map: dict[str, MagicMock] = {"p1": mock_p1, "p2": mock_p2}

        def get_param(name: str) -> MagicMock | None:
            return mock_map.get(name)

        self.mock_plan.keywords.get_parameter.side_effect = get_param

        names, filters, labels, indents = self.parser.get_parameter_info("p1;p2")
        self.assertEqual(names, ["p1", "p2"])
        # filters converted to SQL
        self.assertEqual(filters, ["P1 > 0", "P2 < 0"])
        self.assertEqual(labels, ["Param 1", "p2"])
        self.assertEqual(indents, [0, 1])

    def test_get_parameter_info_not_found(self) -> None:
        self.mock_plan.keywords.get_parameter.return_value = None
        with self.assertRaises(ValueError):
            self.parser.get_parameter_info("missing")

    def test_get_single_parameter_info(self) -> None:
        mock_p = MagicMock()
        mock_p.filter = "x == 1"
        mock_p.label = "Label"
        self.mock_plan.keywords.get_parameter.return_value = mock_p

        param_filter, label = self.parser.get_single_parameter_info("param")
        self.assertEqual(param_filter, "X = 1")
        self.assertEqual(label, "Label")

    def test_get_single_parameter_info_not_found(self) -> None:
        self.mock_plan.keywords.get_parameter.return_value = None
        with self.assertRaises(ValueError):
            self.parser.get_single_parameter_info("missing")

    def test_get_group_info(self) -> None:
        mock_grp = MagicMock()
        mock_grp.variable = "adsl:trt01p"
        mock_grp.group_label = ["A", "B"]
        self.mock_plan.keywords.get_group.return_value = mock_grp

        var, labels = self.parser.get_group_info("treatment")
        self.assertEqual(var, "TRT01P")
        self.assertEqual(labels, ["A", "B"])

    def test_get_group_info_not_found(self) -> None:
        self.mock_plan.keywords.get_group.return_value = None
        with self.assertRaises(ValueError):
            self.parser.get_group_info("missing")

    def test_get_datasets(self) -> None:
        d1 = pl.DataFrame({"a": [1]})
        d2 = pl.DataFrame({"b": [2]})
        self.mock_plan.datasets = {"d1": d1, "d2": d2}

        res = self.parser.get_datasets("d1", "d2")
        self.assertEqual(len(res), 2)
        # Using polars assert_frame_equal would be better but this is simple enough
        self.assertEqual(res[0]["a"][0], 1)
        self.assertEqual(res[1]["b"][0], 2)

    def test_get_datasets_not_found(self) -> None:
        self.mock_plan.datasets = {}
        with self.assertRaises(ValueError):
            self.parser.get_datasets("missing")

    def test_get_population_data(self) -> None:
        # Integrated test with mocked parts
        df = pl.DataFrame({"USUBJID": [1, 2], "SAFFL": ["Y", "N"], "TRT": ["A", "B"]})
        self.mock_plan.datasets = {"adsl": df}

        # Mock population filter
        mock_pop = MagicMock()
        mock_pop.filter = "adsl:saffl == 'Y'"
        self.mock_plan.keywords.get_population.return_value = mock_pop

        # Mock group info
        mock_grp = MagicMock()
        mock_grp.variable = "adsl:trt"
        mock_grp.group_label = ["A"]
        self.mock_plan.keywords.get_group.return_value = mock_grp

        # Execute
        pop_df, grp_var = self.parser.get_population_data("saffl", "treatment")

        # Verify
        self.assertEqual(grp_var, "TRT")
        self.assertEqual(pop_df.height, 1)
        self.assertEqual(pop_df["USUBJID"][0], 1)
