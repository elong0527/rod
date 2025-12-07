# pyre-strict
"""
Adverse Event (AE) Analysis Functions

This module provides a three-step pipeline for AE summary analysis:
- ae_summary_ard: Generate Analysis Results Data (ARD) in long format
- ae_summary_df: Transform ARD to wide display format
- ae_summary_rtf: Generate formatted RTF output
- ae_summary: Complete pipeline wrapper
- study_plan_to_ae_summary: Batch generation from StudyPlan

Uses Polars native SQL capabilities for data manipulation, count.py utilities for subject counting,
and parse.py utilities for StudyPlan parsing.
"""

from pathlib import Path

import polars as pl
from rtflite import RTFDocument

from ..common.count import count_subject, count_subject_with_observation
from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_table_n_pct
from ..common.utils import apply_common_filters


def study_plan_to_ae_summary(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate AE summary RTF outputs for all analyses defined in StudyPlan.

    This function reads the expanded plan from StudyPlan and generates
    an RTF table for each analysis specification automatically.

    Args:
        study_plan: StudyPlan object with loaded datasets and analysis specifications

    Returns:
        list[str]: List of paths to generated RTF files
    """

    # Meta data
    analysis = "ae_summary"
    analysis_label = "Analysis of Adverse Event Summary"
    output_dir = study_plan.output_dir
    footnote = ["Every participant is counted a single time for each applicable row and column."]
    source = None

    population_df_name = "adsl"
    observation_df_name = "adae"

    id = ("USUBJID", "Subject ID")
    total = True
    missing_group = "error"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan DataFrame
    plan_df = study_plan.get_plan_df()

    # Filter for AE summary analyses
    ae_plans = plan_df.filter(pl.col("analysis") == analysis)

    rtf_files = []

    # Generate RTF for each analysis
    for row in ae_plans.iter_rows(named=True):
        population = row["population"]
        observation = row.get("observation")
        parameter = row["parameter"]
        group = row.get("group")

        # Validate group is specified
        if group is None:
            raise ValueError(
                f"Group not specified in YAML "
                f"population={population}, observation={observation}, parameter={parameter}. "
                "Please add group to your YAML plan."
            )

        # Get datasets using parser
        population_df, observation_df = parser.get_datasets(population_df_name, observation_df_name)

        # Get filters and configuration using parser
        population_filter = parser.get_population_filter(population)
        param_names, param_filters, param_labels, _ = parser.get_parameter_info(
            parameter
        )  # Ignore indent for AE
        obs_filter = parser.get_observation_filter(observation)
        group_var_name, group_labels = parser.get_group_info(group)

        # Build variables as list of tuples [(filter, label)]
        variables_list = list(zip(param_filters, param_labels))

        # Build group tuple (variable_name, label)
        group_var_label = group_labels[0] if group_labels else group_var_name
        group_tuple = (group_var_name, group_var_label)

        # Build title with population and observation context
        title_parts = [analysis_label]
        if observation:
            obs_kw = study_plan.keywords.observations.get(observation)
            if obs_kw and obs_kw.label:
                title_parts.append(obs_kw.label)

        pop_kw = study_plan.keywords.populations.get(population)
        if pop_kw and pop_kw.label:
            title_parts.append(pop_kw.label)

        # Build output filename
        filename = f"{analysis}_{population}"
        if observation:
            filename += f"_{observation}"
        filename += f"_{parameter.replace(';', '_')}.rtf"
        output_file = str(Path(output_dir) / filename)

        # Generate RTF using the new ae_summary signature
        rtf_path = ae_summary(
            population=population_df,
            observation=observation_df,
            population_filter=population_filter,
            observation_filter=obs_filter,
            id=id,
            group=group_tuple,
            variables=variables_list,
            title=title_parts,
            footnote=footnote,
            source=source,
            output_file=output_file,
            total=total,
            missing_group=missing_group,
        )

        rtf_files.append(rtf_path)

    return rtf_files


def ae_summary(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str],
    variables: list[tuple[str, str]],
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    output_file: str,
    total: bool = True,
    col_rel_width: list[float] | None = None,
    missing_group: str = "error",
) -> str:
    """
    Complete AE summary pipeline wrapper.

    This function orchestrates the three-step pipeline:
    1. ae_summary_ard: Generate Analysis Results Data
    2. ae_summary_df: Transform to display format
    3. ae_summary_rtf: Generate RTF output and write to file

    Args:
        population: Population DataFrame (subject-level data, e.g., ADSL)
        observation: Observation DataFrame (event data, e.g., ADAE)
        population_filter: SQL WHERE clause for population (can be None)
        observation_filter: SQL WHERE clause for observation (can be None)
        id: Tuple (variable_name, label) for ID column
        group: Tuple (variable_name, label) for grouping variable
        variables: List of tuples [(filter, label)] for analysis variables
        title: Title for RTF output as list of strings
        footnote: Optional footnote for RTF output as list of strings
        source: Optional source for RTF output as list of strings
        output_file: File path to write RTF output
        total: Whether to include total column (default: True)
        col_rel_width: Optional column widths for RTF output
        missing_group: How to handle missing group values (default: "error")

    Returns:
        str: Path to the generated RTF file
    """
    # Step 1: Generate ARD
    ard = ae_summary_ard(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        id=id,
        group=group,
        variables=variables,
        total=total,
        missing_group=missing_group,
    )

    # Step 2: Transform to display format
    df = ae_summary_df(ard)

    # Step 3: Generate RTF and write to file
    rtf_doc = ae_summary_rtf(
        df=df,
        title=title,
        footnote=footnote,
        source=source,
        col_rel_width=col_rel_width,
    )
    rtf_doc.write_rtf(output_file)

    return output_file


def ae_summary_ard(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str],
    variables: list[tuple[str, str]],
    total: bool,
    missing_group: str,
) -> pl.DataFrame:
    """
    Generate Analysis Results Data (ARD) for AE summary analysis.

    Creates a long-format DataFrame with standardized structure (__index__, __group__, __value__)
    containing population counts and observation statistics for each analysis variable.

    Args:
        population: Population DataFrame (subject-level data, e.g., ADSL)
        observation: Observation DataFrame (event data, e.g., ADAE)
        population_filter: SQL WHERE clause for population (can be None)
        observation_filter: SQL WHERE clause for observation (can be None)
        id: Tuple (variable_name, label) for ID column
        group: Tuple (variable_name, label) for grouping variable
        variables: List of tuples [(filter, label)] for analysis variables
        total: Whether to include total column in counts
        missing_group: How to handle missing group values: "error", "ignore", or "fill"

    Returns:
        pl.DataFrame: Long-format ARD with columns __index__, __group__, __value__
    """
    # Extract group variable name (label is in tuple but not needed separately)
    pop_var_name = "Participants in population"
    id_var_name, id_var_label = id
    group_var_name, group_var_label = group

    # Apply common filters (parameter_filter is handled inside the loop, so None here)
    population_filtered, observation_to_filter = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
    )

    assert observation_to_filter is not None

    # Filter observation data to include only subjects in the filtered population
    # Process all variables in the list
    observation_filtered_list = []
    for variable_filter, variable_label in variables:
        obs_filtered = (
            observation_to_filter.filter(
                pl.col(id_var_name).is_in(population_filtered[id_var_name].to_list())
            )
            .filter(pl.sql_expr(variable_filter))
            .with_columns(pl.lit(variable_label).alias("__index__"))
        )

        observation_filtered_list.append(obs_filtered)

    # Concatenate all filtered observations
    observation_filtered = pl.concat(observation_filtered_list)

    # Population
    n_pop = count_subject(
        population=population_filtered,
        id=id_var_name,
        group=group_var_name,
        total=total,
        missing_group=missing_group,
    )

    n_pop = n_pop.select(
        pl.lit(pop_var_name).alias("__index__"),
        pl.col(group_var_name).alias("__group__"),
        pl.col("n_subj_pop").cast(pl.String).alias("__value__"),
    )

    # Empty row with same structure as n_pop but with empty strings
    n_empty = n_pop.select(
        pl.lit("").alias("__index__"), pl.col("__group__"), pl.lit("").alias("__value__")
    )

    # Observation
    n_obs = count_subject_with_observation(
        population=population_filtered,
        observation=observation_filtered,
        id=id_var_name,
        group=group_var_name,
        total=total,
        variable="__index__",
        missing_group=missing_group,
    )

    n_obs = n_obs.select(
        pl.col("__index__"),
        pl.col(group_var_name).alias("__group__"),
        pl.col("n_pct_subj_fmt").alias("__value__"),
    )

    res = pl.concat([n_pop, n_empty, n_obs])

    # Convert __index__ to ordered Enum based on appearance
    # Build the ordered categories list: population name, empty string, then variable labels
    variable_labels = [label for _, label in variables]
    ordered_categories = [pop_var_name, ""] + variable_labels

    res = res.with_columns(pl.col("__index__").cast(pl.Enum(ordered_categories))).sort(
        "__index__", "__group__"
    )

    return res


def ae_summary_df(ard: pl.DataFrame) -> pl.DataFrame:
    """
    Transform AE summary ARD (Analysis Results Data) into display-ready DataFrame.

    Converts the long-format ARD with __index__, __group__, and __value__ columns
    into a wide-format display table where groups become columns.

    Args:
        ard: Analysis Results Data DataFrame with __index__, __group__, __value__ columns

    Returns:
        pl.DataFrame: Wide-format display table with groups as columns
    """
    # Pivot from long to wide format: __group__ values become columns
    df_wide = ard.pivot(index="__index__", on="__group__", values="__value__")

    return df_wide


def ae_summary_rtf(
    df: pl.DataFrame,
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    col_rel_width: list[float] | None = None,
) -> RTFDocument:
    """
    Generate RTF table from AE summary display DataFrame.

    Creates a formatted RTF table with two-level column headers showing
    treatment groups with "n (%)" values.

    Args:
        df: Display DataFrame from ae_summary_df (wide format with __index__ column)
        title: Title(s) for the table as list of strings
        footnote: Optional footnote(s) as list of strings
        source: Optional source note(s) as list of strings
        col_rel_width: Optional list of relative column widths. If None, auto-calculated
                       as [n_cols-1, 1, 1, 1, ...] where n_cols is total column count

    Returns:
        RTFDocument: RTF document object that can be written to file
    """

    # Rename __index__ to empty string for display
    df_rtf = df.rename({"__index__": ""})

    # Calculate number of columns
    n_cols = len(df_rtf.columns)

    # Build first-level column headers (use actual column names)
    col_header_1 = list(df_rtf.columns)

    # Build second-level column headers (empty for first, "n (%)" for groups)
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

    # Calculate column widths - auto-calculate if not provided
    if col_rel_width is None:
        col_widths = [float(n_cols - 1)] + [1.0] * (n_cols - 1)
    else:
        col_widths = col_rel_width

    return create_rtf_table_n_pct(
        df=df_rtf,
        col_header_1=col_header_1,
        col_header_2=col_header_2,
        col_widths=col_widths,
        title=title,
        footnote=footnote,
        source=source,
    )
