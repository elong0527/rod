# pyre-strict
"""
Adverse Event (AE) Specific Analysis Functions

This module provides functions for AE specific analysis showing detailed event listings
organized by System Organ Class (SOC) and Preferred Term (PT), following metalite.ae patterns.

The three-step pipeline:
- ae_specific_ard: Generate Analysis Results Data with SOC/PT hierarchy
- ae_specific_df: Transform to display format
- ae_specific_rtf: Generate formatted RTF output
- ae_specific: Complete pipeline wrapper
- study_plan_to_ae_specific: Batch generation from StudyPlan

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
from .ae_utils import get_ae_parameter_row_labels, get_ae_parameter_title


def ae_specific_ard(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    parameter_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str],
    ae_term: tuple[str, str],
    total: bool = True,
    missing_group: str = "error",
    n_with_label: str = "    with one or more adverse events",
    n_without_label: str = "    with no adverse events",
) -> pl.DataFrame:
    """
    Generate Analysis Results Data (ARD) for AE specific analysis.

    Creates a long-format DataFrame showing the number and percentage
    of subjects experiencing specific adverse events.

    Args:
        population: Population DataFrame (subject-level data, e.g., ADSL)
        observation: Observation DataFrame (event data, e.g., ADAE)
        population_filter: SQL WHERE clause for population (can be None)
        observation_filter: SQL WHERE clause for observation (can be None)
        parameter_filter: SQL WHERE clause for parameter filtering (can be None)
        id: Tuple (variable_name, label) for ID column
        group: Tuple (variable_name, label) for grouping variable
        ae_term: Tuple (variable_name, label) for AE term column
        total: Whether to include total column in counts
        missing_group: How to handle missing group values: "error", "ignore", or "fill"
        n_with_label: Label for "with one or more" row (dynamic based on parameter)
        n_without_label: Label for "with no" row (dynamic based on parameter)

    Returns:
        pl.DataFrame: Long-format ARD with columns __index__, __group__, __value__
    """
    # Extract variable names
    pop_var_name = "Participants in population"
    id_var_name, id_var_label = id
    group_var_name, group_var_label = group
    ae_term_var_name, ae_term_var_label = ae_term

    # Apply common filters
    population_filtered, observation_to_filter = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        parameter_filter=parameter_filter,
    )

    assert observation_to_filter is not None

    # Filter observation to include only subjects in filtered population
    observation_filtered = observation_to_filter.filter(
        pl.col(id_var_name).is_in(population_filtered[id_var_name].to_list())
    ).with_columns(pl.col(ae_term_var_name).alias("__index__"))

    # Note: We'll extract categories from concatenated result later for both __index__ and __group__

    # Population counts - keep original for denominator calculations
    n_pop_counts = count_subject(
        population=population_filtered,
        id=id_var_name,
        group=group_var_name,
        total=total,
        missing_group=missing_group,
    )

    # Transform population counts for display
    n_pop = n_pop_counts.select(
        pl.lit(pop_var_name).alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_subj_pop").cast(pl.String).alias("__value__"),
    )

    # Empty separator row
    n_empty = n_pop.select(
        pl.lit("").alias("__index__"), pl.col("__group__"), pl.lit("").alias("__value__")
    )

    # Summary rows: "with one or more" and "with no" adverse events
    # Count subjects with at least one event
    subjects_with_events = observation_filtered.select(id_var_name).unique()

    # Get population with event indicator
    pop_with_indicator = population_filtered.with_columns(
        pl.col(id_var_name).is_in(subjects_with_events[id_var_name]).alias("__has_event__")
    )

    # Count subjects with and without events using count_subject_with_observation
    event_counts = count_subject_with_observation(
        population=population_filtered,
        observation=pop_with_indicator,
        id=id_var_name,
        group=group_var_name,
        variable="__has_event__",
        total=total,
        missing_group=missing_group,
    )

    # Extract 'with' counts
    n_with = event_counts.filter(pl.col("__has_event__") == "true").select(
        [
            pl.lit(n_with_label).alias("__index__"),
            pl.col(group_var_name).cast(pl.String).alias("__group__"),
            pl.col("n_pct_subj_fmt").alias("__value__"),
        ]
    )

    # Extract 'without' counts
    n_without = event_counts.filter(pl.col("__has_event__") == "false").select(
        [
            pl.lit(n_without_label).alias("__index__"),
            pl.col(group_var_name).cast(pl.String).alias("__group__"),
            pl.col("n_pct_subj_fmt").alias("__value__"),
        ]
    )

    # AE term counts
    n_index = count_subject_with_observation(
        population=population_filtered,
        observation=observation_filtered,
        id=id_var_name,
        group=group_var_name,
        total=total,
        variable="__index__",
        missing_group=missing_group,
    )

    n_index = n_index.select(
        (
            pl.col("__index__").cast(pl.String).str.slice(0, 1).str.to_uppercase()
            + pl.col("__index__").cast(pl.String).str.slice(1).str.to_lowercase()
        ).alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_pct_subj_fmt").alias("__value__"),
    )

    # Concatenate all parts
    parts = [n_pop, n_with, n_without, n_empty, n_index]

    res = pl.concat(parts)

    # Extract unique categories from concatenated result in order of appearance
    index_categories = res.select("__index__").unique(maintain_order=True).to_series().to_list()
    group_categories = res.select("__group__").unique(maintain_order=True).to_series().to_list()

    # Convert to Enum types for proper categorical ordering and sorting
    res = res.with_columns(
        [
            pl.col("__index__").cast(pl.Enum(index_categories)),
            pl.col("__group__").cast(pl.Enum(group_categories)),
        ]
    )

    # Sort by index and group using categorical ordering
    res = res.sort("__index__", "__group__")

    return res


def ae_specific_df(ard: pl.DataFrame) -> pl.DataFrame:
    """
    Transform AE specific ARD to display-ready DataFrame.

    Converts the long-format ARD with __index__, __group__, __value__ columns
    into a wide-format display table where groups become columns.

    Args:
        ard: Analysis Results Data DataFrame with __index__, __group__, __value__ columns

    Returns:
        pl.DataFrame: Wide-format display table with index rows and groups as columns
    """
    # Pivot from long to wide format
    df_wide = ard.pivot(index="__index__", on="__group__", values="__value__")

    # Rename __index__ to display column name
    df_wide = df_wide.rename({"__index__": "Term"}).select(pl.col("Term"), pl.exclude("Term"))

    return df_wide


def ae_specific_rtf(
    df: pl.DataFrame,
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    col_rel_width: list[float] | None = None,
) -> RTFDocument:
    """
    Generate RTF table from AE specific display DataFrame.

    Creates a formatted RTF table with two-level column headers showing
    treatment groups with "n (%)" values.

    Args:
        df: Display DataFrame from ae_specific_df (wide format)
        title: Title(s) for the table as list of strings
        footnote: Optional footnote(s) as list of strings
        source: Optional source note(s) as list of strings
        col_rel_width: Optional list of relative column widths. If None, auto-calculated
                       as [n_cols-1, 1, 1, 1, ...] where n_cols is total column count

    Returns:
        RTFDocument: RTF document object that can be written to file
    """

    # Rename Term to empty string for display
    df_rtf = df.rename({"Term": ""})

    # Calculate number of columns
    n_cols = len(df_rtf.columns)

    # Build first-level column headers
    col_header_1 = list(df_rtf.columns)

    # Build second-level column headers
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

    # Calculate column widths
    if col_rel_width is None:
        col_widths = [n_cols / 1.5] + [1] * (n_cols - 1)
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


def ae_specific(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    parameter_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str],
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    output_file: str,
    ae_term: tuple[str, str],
    total: bool = True,
    col_rel_width: list[float] | None = None,
    missing_group: str = "error",
    n_with_label: str = "    with one or more adverse events",
    n_without_label: str = "    with no adverse events",
) -> str:
    """
    Complete AE specific pipeline wrapper.

    This function orchestrates the three-step pipeline:
    1. ae_specific_ard: Generate Analysis Results Data
    2. ae_specific_df: Transform to display format
    3. ae_specific_rtf: Generate RTF output and write to file

    Args:
        population: Population DataFrame (subject-level data, e.g., ADSL)
        observation: Observation DataFrame (event data, e.g., ADAE)
        population_filter: SQL WHERE clause for population (can be None)
        observation_filter: SQL WHERE clause for observation (can be None)
        parameter_filter: SQL WHERE clause for parameter filtering (can be None)
        id: Tuple (variable_name, label) for ID column
        group: Tuple (variable_name, label) for grouping variable
        title: Title for RTF output as list of strings
        footnote: Optional footnote for RTF output as list of strings
        source: Optional source for RTF output as list of strings
        output_file: File path to write RTF output
        ae_term: Tuple (variable_name, label) for AE term column
                 (default: ("AEDECOD", "Adverse Event"))
        total: Whether to include total column (default: True)
        col_rel_width: Optional column widths for RTF output
        missing_group: How to handle missing group values (default: "error")
        n_with_label: Label for "with one or more" row (dynamic based on parameter)
        n_without_label: Label for "with no" row (dynamic based on parameter)

    Returns:
        str: Path to the generated RTF file
    """
    # Step 1: Generate ARD
    ard = ae_specific_ard(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        parameter_filter=parameter_filter,
        id=id,
        group=group,
        ae_term=ae_term,
        total=total,
        missing_group=missing_group,
        n_with_label=n_with_label,
        n_without_label=n_without_label,
    )

    # Step 2: Transform to display format
    df = ae_specific_df(ard)

    # Step 3: Generate RTF and write to file
    rtf_doc = ae_specific_rtf(
        df=df,
        title=title,
        footnote=footnote,
        source=source,
        col_rel_width=col_rel_width,
    )
    rtf_doc.write_rtf(output_file)

    return output_file


def study_plan_to_ae_specific(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate AE specific RTF outputs for all analyses defined in StudyPlan.

    This function reads the expanded plan from StudyPlan and generates
    an RTF table for each ae_specific analysis specification automatically.

    Args:
        study_plan: StudyPlan object with loaded datasets and analysis specifications

    Returns:
        list[str]: List of paths to generated RTF files
    """

    # Meta data
    analysis = "ae_specific"
    output_dir = study_plan.output_dir
    footnote = ["Every participant is counted a single time for each applicable row and column."]
    source = None

    population_df_name = "adsl"
    observation_df_name = "adae"

    id = ("USUBJID", "Subject ID")
    ae_term = ("AEDECOD", "Adverse Event")
    total = True
    missing_group = "error"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan DataFrame
    plan_df = study_plan.get_plan_df()

    # Filter for AE specific analyses
    ae_plans = plan_df.filter(pl.col("analysis") == analysis)

    rtf_files = []

    # Generate RTF for each analysis
    for row in ae_plans.iter_rows(named=True):
        population = row["population"]
        observation = row.get("observation")
        parameter = row.get("parameter")
        group = row.get("group")

        # Validate group is specified
        if group is None:
            raise ValueError(
                f"Group not specified in YAML for analysis: population={population}, "
                f"observation={observation}, parameter={parameter}. "
                f"Please add group to your YAML plan."
            )

        # Get datasets using parser
        population_df, observation_df = parser.get_datasets(population_df_name, observation_df_name)

        # Get filters and configuration using parser
        population_filter = parser.get_population_filter(population)
        obs_filter = parser.get_observation_filter(observation)

        # Get parameter filter if parameter is specified
        parameter_filter = None
        if parameter:
            param_names, param_filters, param_labels, _ = parser.get_parameter_info(
                parameter
            )  # Ignore indent for AE
            # For ae_specific, use the first (and typically only) filter
            parameter_filter = param_filters[0] if param_filters else None

        group_var_name, group_labels = parser.get_group_info(group)

        # Build group tuple
        group_var_label = group_labels[0] if group_labels else group_var_name
        group_tuple = (group_var_name, group_var_label)

        # Build dynamic title and row labels based on parameter
        param = study_plan.keywords.get_parameter(parameter) if parameter else None
        dynamic_title = get_ae_parameter_title(param)
        n_with_label, n_without_label = get_ae_parameter_row_labels(param)

        # Build title with population and observation context
        title_parts = [dynamic_title]
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
        if parameter:
            filename += f"_{parameter.replace(';', '_')}"
        filename += ".rtf"
        output_file = str(Path(output_dir) / filename)

        # Generate RTF
        rtf_path = ae_specific(
            population=population_df,
            observation=observation_df,
            population_filter=population_filter,
            observation_filter=obs_filter,
            parameter_filter=parameter_filter,
            id=id,
            group=group_tuple,
            ae_term=ae_term,
            title=title_parts,
            footnote=footnote,
            source=source,
            output_file=output_file,
            total=total,
            missing_group=missing_group,
            n_with_label=n_with_label,
            n_without_label=n_without_label,
        )

        rtf_files.append(rtf_path)

    return rtf_files
