# pyre-strict
"""
Concomitant Medications (CM) Summary Functions

This module provides a three-step pipeline for CM summary analysis:
- cm_summary_ard: Generate Analysis Results Data (ARD) in long format
- cm_summary_df: Transform ARD to wide display format
- cm_summary_rtf: Generate formatted RTF output
- cm_summary: Complete pipeline wrapper
- study_plan_to_cm_summary: Batch generation from StudyPlan

Applications:
- Summary of Concomitant Medications
- Summary of Prior Medications
"""

from pathlib import Path

import polars as pl
from rtflite import RTFDocument

from ..common.count import count_subject, count_subject_with_observation
from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_table_n_pct
from ..common.utils import apply_common_filters


def study_plan_to_cm_summary(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate CM summary RTF outputs for all analyses defined in StudyPlan.

    Args:
        study_plan: StudyPlan object with loaded datasets and analysis specifications

    Returns:
        list[str]: List of paths to generated RTF files
    """

    # Meta data
    analysis = "cm_summary"
    analysis_label = "Summary of Concomitant Medications"
    output_dir = study_plan.output_dir
    footnote = ["Every participant is counted a single time for each applicable row and column."]
    source = None

    population_df_name = "adsl"
    observation_df_name = "adcm"

    id = ("USUBJID", "Subject ID")
    total = True
    missing_group = "error"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan DataFrame
    plan_df = study_plan.get_plan_df()

    # Filter for CM summary analyses
    cm_plans = plan_df.filter(pl.col("analysis") == analysis)

    rtf_files = []

    # Generate RTF for each analysis
    for row in cm_plans.iter_rows(named=True):
        population = row["population"]
        observation = row.get("observation")
        parameter = row.get("parameter")
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

        # Handle parameters (variables to summarize)
        if parameter:
            param_names, param_filters, param_labels, _ = parser.get_parameter_info(parameter)
        else:
            # Default to summarizing "Any Medication" if no parameter specified
            # But usually cm_summary needs parameters defining what to count
            # Use a default generic filter if none provided
            param_filters = ["1=1"]
            param_labels = ["Any Medication"]

        obs_filter = parser.get_observation_filter(observation)
        group_var_name, group_labels = parser.get_group_info(group)

        # Build variables as list of tuples [(filter, label)]
        variables_list = list(zip(param_filters, param_labels))

        # Build group tuple (variable_name, label)
        group_var_label = group_labels[0] if group_labels else group_var_name
        group_tuple = (group_var_name, group_var_label)

        # Build title
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
        if parameter:
            filename += f"_{parameter.replace(';', '_')}"
        filename += ".rtf"
        output_file = str(Path(output_dir) / filename)

        # Generate RTF
        rtf_path = cm_summary(
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


def cm_summary_ard(
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
    Generate Analysis Results Data (ARD) for CM summary analysis.
    """
    # Reuse the same logic logic as ae_summary_ard since it's generic counting
    # But checking if we should duplicate code or import?
    # For now, duplication allows independence (e.g. if CM specific logic is needed later)

    pop_var_name = "Participants in population"
    id_var_name, id_var_label = id
    group_var_name, group_var_label = group

    population_filtered, observation_to_filter = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
    )

    assert observation_to_filter is not None

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

    if observation_filtered_list:
        observation_filtered = pl.concat(observation_filtered_list)
    else:
        # Handle case with no variables (empty df with correct schema)
        observation_filtered = observation_to_filter.clear().with_columns(
            pl.lit("").alias("__index__")
        )

    # Population counts
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

    n_empty = n_pop.select(
        pl.lit("").alias("__index__"), pl.col("__group__"), pl.lit("").alias("__value__")
    )

    # Observation counts
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

    variable_labels = [label for _, label in variables]
    ordered_categories = [pop_var_name, ""] + variable_labels

    # Ensure all categories are present in Enum
    res = res.with_columns(pl.col("__index__").cast(pl.Enum(ordered_categories))).sort(
        "__index__", "__group__"
    )

    return res


def cm_summary_df(ard: pl.DataFrame) -> pl.DataFrame:
    """Transform CM summary ARD into display-ready DataFrame."""
    df_wide = ard.pivot(index="__index__", on="__group__", values="__value__")
    return df_wide


def cm_summary_rtf(
    df: pl.DataFrame,
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    col_rel_width: list[float] | None = None,
) -> RTFDocument:
    """Generate RTF table from CM summary display DataFrame."""
    df_rtf = df.rename({"__index__": ""})
    n_cols = len(df_rtf.columns)
    col_header_1 = list(df_rtf.columns)
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

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


def cm_summary(
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
    """Complete CM summary pipeline wrapper."""
    ard = cm_summary_ard(
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

    df = cm_summary_df(ard)

    rtf_doc = cm_summary_rtf(
        df=df,
        title=title,
        footnote=footnote,
        source=source,
        col_rel_width=col_rel_width,
    )
    rtf_doc.write_rtf(output_file)

    return output_file
