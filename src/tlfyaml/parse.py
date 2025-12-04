# pyre-strict
"""
StudyPlan Parsing Utilities

This module provides utilities for parsing and extracting information from StudyPlan objects,
including filter conversion, parameter parsing, and keyword resolution.
"""

import re
from typing import Any

import polars as pl

from .plan import StudyPlan


def parse_filter_to_sql(filter_str: str) -> str:
    """
    Parse custom filter syntax to SQL WHERE clause.

    Converts:
    - "adsl:saffl == 'Y'" -> "SAFFL = 'Y'"
    - "adae:trtemfl == 'Y' and adae:aeser == 'Y'" -> "TRTEMFL = 'Y' AND AESER = 'Y'"
    - "adae:aerel in ['A', 'B']" -> "AEREL IN ('A', 'B')"

    Args:
        filter_str: Custom filter string with dataset:column format

    Returns:
        SQL WHERE clause string
    """
    if not filter_str or filter_str.strip() == "":
        return "1=1"  # Always true

    # Remove dataset prefixes (adsl:, adae:)
    sql = re.sub(r"\w+:", "", filter_str)

    # Convert Python syntax to SQL
    sql = sql.replace("==", "=")  # Python equality to SQL
    sql = sql.replace(" and ", " AND ")  # Python to SQL
    sql = sql.replace(" and ", " AND ")  # Python to SQL
    sql = sql.replace(" or ", " OR ")  # Python to SQL
    sql = sql.replace(" in ", " IN ")  # Python to SQL

    # Convert Python list syntax to SQL IN: ['A', 'B'] -> ('A', 'B')
    sql = sql.replace("[", "(").replace("]", ")")

    # Uppercase column names (assuming ADaM standard)
    # Match word boundaries before operators
    sql = re.sub(
        r"\b([a-z]\w*)\b(?=\s*[=<>!]|\s+IN)", lambda m: m.group(1).upper(), sql, flags=re.IGNORECASE
    )

    return sql


def apply_filter_sql(df: pl.DataFrame, filter_str: str) -> pl.DataFrame:
    """
    Apply filter using pl.sql_expr() - simpler and faster than SQLContext.

    Args:
        df: DataFrame to filter
        filter_str: Custom filter string

    Returns:
        Filtered DataFrame
    """
    if not filter_str or filter_str.strip() == "":
        return df

    where_clause = parse_filter_to_sql(filter_str)

    try:
        # Use pl.sql_expr() - much simpler and faster!
        return df.filter(pl.sql_expr(where_clause))
    except Exception as e:
        # Fallback to manual parsing if SQL fails
        print(f"Warning: SQL filter failed ({e}), using fallback method")
        return df.filter(_parse_filter_expr(filter_str))


def _parse_filter_expr(filter_str: str) -> Any:
    """
    Fallback filter parser using Polars expressions.
    Used if SQL parsing fails.

    Args:
        filter_str: Filter string

    Returns:
        Polars expression
    """
    if not filter_str or filter_str.strip() == "":
        return pl.lit(True)

    # Remove dataset prefixes
    filter_str = re.sub(r"\w+:", "", filter_str)

    # Handle 'in' operator: column in ['A', 'B'] -> pl.col(column).is_in(['A', 'B'])
    in_pattern = r"(\w+)\s+in\s+\[([^\]]+)\]"

    def replace_in(match: re.Match) -> str:
        col = match.group(1).upper()
        values = match.group(2)
        return f"(pl.col('{col}').is_in([{values}]))"

    filter_str = re.sub(in_pattern, replace_in, filter_str)

    # Handle equality/inequality
    eq_pattern = r"(\w+)\s*(==|!=|>|<|>=|<=)\s*'([^']+)'"

    def replace_eq(match: re.Match) -> str:
        col = match.group(1).upper()
        op = match.group(2)
        val = match.group(3)
        return f"(pl.col('{col}') {op} '{val}')"

    filter_str = re.sub(eq_pattern, replace_eq, filter_str)

    # Replace 'and'/'or'
    filter_str = filter_str.replace(" and ", " & ")
    filter_str = filter_str.replace(" or ", " | ")

    return eval(filter_str)


def parse_parameter(parameter_str: str) -> list[str]:
    """
    Parse semicolon-separated parameter string.

    Args:
        parameter_str: Single parameter or semicolon-separated (e.g., "any;rel;ser")

    Returns:
        List of parameter names
    """
    if not parameter_str:
        return []
    if ";" in parameter_str:
        return [p.strip() for p in parameter_str.split(";")]
    return [parameter_str]


class StudyPlanParser:
    """
    Parser class for extracting and resolving information from StudyPlan objects.

    This class provides methods to extract filters, labels, and other configuration
    from StudyPlan keywords and convert them to analysis-ready formats.
    """

    def __init__(self, study_plan: StudyPlan):
        """
        Initialize parser with a StudyPlan object.

        Args:
            study_plan: StudyPlan object with loaded datasets and keywords
        """
        self.study_plan = study_plan

    def get_population_filter(self, population: str) -> str:
        """
        Get population filter as SQL WHERE clause.

        Args:
            population: Population keyword name

        Returns:
            SQL WHERE clause string

        Raises:
            ValueError: If population keyword not found
        """
        pop = self.study_plan.keywords.get_population(population)
        if pop is None:
            raise ValueError(f"Population '{population}' not found")
        return parse_filter_to_sql(pop.filter)

    def get_observation_filter(self, observation: str | None) -> str | None:
        """
        Get observation filter as SQL WHERE clause.

        Args:
            observation: Optional observation keyword name

        Returns:
            SQL WHERE clause string or None if observation not specified
        """
        if not observation:
            return None
        obs = self.study_plan.keywords.get_observation(observation)
        if obs:
            return parse_filter_to_sql(obs.filter)
        return None

    def get_parameter_info(
        self, parameter: str
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Get parameter names, filters, and labels.

        Args:
            parameter: Parameter keyword, can be semicolon-separated (e.g., "any;rel;ser")

        Returns:
            Tuple of (parameter_names, parameter_filters, parameter_labels)

        Raises:
            ValueError: If any parameter keyword not found
        """
        param_names = parse_parameter(parameter)
        param_labels = []
        param_filters = []

        for param_name in param_names:
            param = self.study_plan.keywords.get_parameter(param_name)
            if param is None:
                raise ValueError(f"Parameter '{param_name}' not found")
            param_filters.append(parse_filter_to_sql(param.filter))
            param_labels.append(param.label or param_name)

        return param_names, param_filters, param_labels

    def get_single_parameter_info(self, parameter: str) -> tuple[str, str]:
        """
        Get single parameter filter and label (NOT semicolon-separated).

        Args:
            parameter: Single parameter keyword name

        Returns:
            Tuple of (parameter_filter, parameter_label)

        Raises:
            ValueError: If parameter keyword not found
        """
        param = self.study_plan.keywords.get_parameter(parameter)
        if param is None:
            raise ValueError(f"Parameter '{parameter}' not found")
        return parse_filter_to_sql(param.filter), param.label or parameter

    def get_group_info(self, group: str) -> tuple[str, list[str]]:
        """
        Get group variable name and labels.

        Args:
            group: Group keyword name

        Returns:
            Tuple of (group_variable, group_labels)

        Raises:
            ValueError: If group keyword not found
        """
        grp = self.study_plan.keywords.get_group(group)
        if grp is None:
            raise ValueError(f"Group '{group}' not found")

        group_var = grp.variable.split(":")[-1].upper()
        group_labels = grp.group_label if grp.group_label else []

        return group_var, group_labels

    def get_datasets(self, *dataset_names: str) -> tuple[pl.DataFrame, ...]:
        """
        Get multiple datasets from StudyPlan.

        Args:
            *dataset_names: Names of datasets to retrieve (e.g., "adsl", "adae")

        Returns:
            Tuple of DataFrames in the order requested

        Raises:
            ValueError: If any dataset not found
        """
        datasets = []
        for name in dataset_names:
            ds = self.study_plan.datasets.get(name)
            if ds is None:
                raise ValueError(f"Dataset '{name}' not found in study plan")
            datasets.append(ds)
        return tuple(datasets)

    def get_population_data(
        self, population: str, group: str
    ) -> tuple[pl.DataFrame, str]:
        """
        Get filtered population dataset and group variable.

        Args:
            population: Population keyword name
            group: Group keyword name

        Returns:
            Tuple of (filtered_adsl, group_variable)
        """
        # Get ADSL dataset
        adsl, = self.get_datasets("adsl")

        # Apply population filter
        pop_filter = self.get_population_filter(population)
        adsl_pop = apply_filter_sql(adsl, pop_filter)

        # Get group variable
        group_var, _ = self.get_group_info(group)

        return adsl_pop, group_var
