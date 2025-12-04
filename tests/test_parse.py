# pyre-strict
from tlfyaml.parse import parse_filter_to_sql, parse_parameter


def test_parse_filter_to_sql_simple():
    assert parse_filter_to_sql("adsl:saffl == 'Y'") == "SAFFL = 'Y'"

def test_parse_filter_to_sql_and():
    res = parse_filter_to_sql("adsl:saffl == 'Y' and adsl:sex == 'M'")
    assert res == "SAFFL = 'Y' AND SEX = 'M'"

def test_parse_filter_to_sql_in():
    res = parse_filter_to_sql("adae:aerel in ['A', 'B']")
    assert res == "AEREL IN ('A', 'B')"

def test_parse_filter_to_sql_empty():
    assert parse_filter_to_sql("") == "1=1"
    assert parse_filter_to_sql(None) == "1=1"

def test_parse_parameter_single():
    assert parse_parameter("param1") == ["param1"]

def test_parse_parameter_multiple():
    assert parse_parameter("p1;p2; p3") == ["p1", "p2", "p3"]

def test_parse_parameter_empty():
    assert parse_parameter("") == []
    assert parse_parameter(None) == []
