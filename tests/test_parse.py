# pyre-strict
import unittest

from tlfyaml.parse import parse_filter_to_sql, parse_parameter


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
        self.assertEqual(parse_filter_to_sql(None), "1=1")

    def test_parse_parameter_single(self) -> None:
        self.assertEqual(parse_parameter("param1"), ["param1"])

    def test_parse_parameter_multiple(self) -> None:
        self.assertEqual(parse_parameter("p1;p2; p3"), ["p1", "p2", "p3"])

    def test_parse_parameter_empty(self) -> None:
        self.assertEqual(parse_parameter(""), [])
        self.assertEqual(parse_parameter(None), [])
