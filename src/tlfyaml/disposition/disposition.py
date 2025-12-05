"""
Disposition Table 1.1 Analysis Functions

This module provides a pipeline for Disposition Table 1.1 summary analysis:
- disposition_ard: Generate Analysis Results Data (ARD)
- disposition_df: Transform ARD to display format
- disposition_rtf: Generate formatted RTF output
- disposition: Complete pipeline wrapper
- study_plan_to_disposition: Batch generation from StudyPlan
"""

from pathlib import Path

import polars as pl
from rtflite import RTFDocument

from ..ae.ae_utils import create_ae_rtf_table
from ..common.count import count_subject_with_observation
from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.utils import apply_common_filters


def study_plan_to_disposition(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate Disposition Table 1.1 RTF outputs for all analyses defined in StudyPlan.
    """
    # Meta data
    analysis_type = "disposition_table_1_1"
    output_dir = "studies/xyz123/rtf"
    footnote = ["Percentages are based on the number of enrolled participants."]
    source = None

    population_df_name = "adsl"
    observation_df_name = "ds"  # As per plan_ds_xyz123.yaml

    id = ("USUBJID", "Subject ID")
    total = True
    missing_group = "error"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan DataFrame
    plan_df = study_plan.get_plan_df()

    # Filter for disposition analyses
    disp_plans = plan_df.filter(pl.col("analysis") == analysis_type)

    rtf_files = []

    for row in disp_plans.iter_rows(named=True):
        population = row["population"]
        observation = row.get("observation")
        parameter = row["parameter"]
        group = row.get("group")
        title_text = row.get(
            "title", "Disposition of Participants"
        )  # Allow title override from plan if we supported it in parser, else default

        # Get datasets
        population_df, observation_df = parser.get_datasets(population_df_name, observation_df_name)

        # Get filters
        population_filter = parser.get_population_filter(population)
        obs_filter = parser.get_observation_filter(observation)

        # Get parameters with indent levels
        param_names, param_filters, param_labels, param_indents = parser.get_parameter_info(
            parameter
        )

        # Apply indentation to labels
        indented_labels = []
        for label, indent_level in zip(param_labels, param_indents):
            indent_str = "    " * indent_level  # 4 spaces per indent level
            indented_labels.append(f"{indent_str}{label}")

        variables_list = list(zip(param_filters, indented_labels))

        # Get group info (optional)
        if group is not None:
            group_var_name, group_labels = parser.get_group_info(group)
            group_var_label = group_labels[0] if group_labels else group_var_name
            group_tuple = (group_var_name, group_var_label)
        else:
            # When no group specified, use a dummy group column for overall counts
            group_tuple = None

        # Build title
        title_parts = [title_text]
        pop_kw = study_plan.keywords.populations.get(population)
        if pop_kw and pop_kw.label:
            title_parts.append(pop_kw.label)

        # Build output filename
        group_suffix = f"_{group}" if group else ""
        filename = f"{analysis_type}_{population}{group_suffix}.rtf"
        output_file = str(Path(output_dir) / filename)

        rtf_path = disposition(
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


def disposition(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str] | None,
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
    Complete Disposition Table 1.1 pipeline wrapper.
    """
    # Step 1: Generate ARD
    ard = disposition_ard(
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
    df = disposition_df(ard)

    # Step 3: Generate RTF
    rtf_doc = disposition_rtf(
        df=df,
        title=title,
        footnote=footnote,
        source=source,
        col_rel_width=col_rel_width,
    )
    rtf_doc.write_rtf(output_file)

    return output_file


def disposition_ard(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str] | None,
    variables: list[tuple[str, str]],
    total: bool,
    missing_group: str,
) -> pl.DataFrame:
    """
    Generate ARD for Disposition Table 1.1.
    """
    id_var_name, _ = id

    # Handle optional group
    if group is not None:
        group_var_name, _ = group
    else:
        # Create a dummy group column for overall counts
        group_var_name = "__all__"
        population = population.with_columns(pl.lit("All Subjects").alias(group_var_name))
        observation = observation.with_columns(pl.lit("All Subjects").alias(group_var_name))
        total = False  # No need for total column when there's only one group

    # Apply common filters
    population_filtered, observation_to_filter = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
    )

    # For each parameter, we create an "observation" dataset and use
    # count_subject_with_observation. This approach works for both ADSL-based
    # filters (e.g., "Enrolled") and DS-based filters (e.g., "Discontinued")

    results = []

    for var_filter, var_label in variables:
        # Try to apply the filter to population first, then observation
        # This handles both ADSL-based and DS-based parameter filters
        try:
            target_obs = population_filtered.filter(pl.sql_expr(var_filter))
        except Exception:
            target_obs = observation_to_filter.filter(pl.sql_expr(var_filter))

        # Add the parameter label as a variable for counting
        target_obs = target_obs.with_columns(pl.lit(var_label).alias("__index__"))

        # Use count_subject_with_observation to get n (%) for each group
        counts = count_subject_with_observation(
            population=population_filtered,
            observation=target_obs,
            id=id_var_name,
            group=group_var_name,
            variable="__index__",
            total=total,
            missing_group=missing_group,
        )

        results.append(
            counts.select(
                pl.col("__index__"),
                pl.col(group_var_name).alias("__group__"),
                pl.col("n_pct_subj_fmt").alias("__value__"),
            )
        )

    # Combine all results
    ard = pl.concat(results)

    # Sort by the order of variables in the list
    # Create an Enum for __index__
    var_labels = [label for _, label in variables]
    ard = ard.with_columns(pl.col("__index__").cast(pl.Enum(var_labels))).sort(
        "__index__", "__group__"
    )

    return ard


def disposition_df(ard: pl.DataFrame) -> pl.DataFrame:
    """
    Transform ARD to display format.
    """
    # Pivot
    df_wide = ard.pivot(index="__index__", on="__group__", values="__value__")

    # Rename index
    df_wide = df_wide.rename({"__index__": "Disposition Status"})

    return df_wide


def disposition_rtf(
    df: pl.DataFrame,
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    col_rel_width: list[float] | None = None,
) -> RTFDocument:
    """
    Generate RTF.
    """
    # Reuse generic table creation
    # Columns: Disposition Status, Group 1, Group 2, ... Total

    n_cols = len(df.columns)
    col_header_1 = list(df.columns)
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

    if col_rel_width is None:
        col_widths = [2.5] + [1] * (n_cols - 1)
    else:
        col_widths = col_rel_width

    return create_ae_rtf_table(
        df=df,
        col_header_1=col_header_1,
        col_header_2=col_header_2,
        col_widths=col_widths,
        title=title,
        footnote=footnote,
        source=source,
    )
