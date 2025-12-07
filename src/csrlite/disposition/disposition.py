# pyre-strict
"""
Disposition Table 1.1 Analysis Functions

This module provides a pipeline for Disposition Table 1.1 summary analysis:
- disposition_ard: Generate Analysis Results Data (ARD)
- disposition_df: Transform ARD to display format
- disposition_rtf: Generate formatted RTF output
- disposition: Complete pipeline wrapper
- study_plan_to_disposition_summary: Batch generation from StudyPlan
"""

from pathlib import Path

import polars as pl
from rtflite import RTFDocument


from ..common.count import count_subject, count_subject_with_observation
from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan

from ..common.rtf import create_rtf_table_n_pct
from ..common.utils import apply_common_filters


def study_plan_to_disposition_summary(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate Disposition Summary Table outputs for all analyses defined in StudyPlan.
    """
    # Meta data
    analysis_type = "disposition_summary"
    output_dir = study_plan.output_dir
    title = "Disposition of Participants"
    footnote = ["Percentages are based on the number of enrolled participants."]
    source = None

    population_df_name = "adsl"

    id = ("USUBJID", "Subject ID")
    ds_term = ("EOSSTT", "Disposition Status")
    dist_reason_term = ("DCREASCD", "Discontinued Reason")

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
        group = row.get("group")
        title_text = title

        # Get datasets
        (population_df,) = parser.get_datasets(population_df_name)

        # Get filters
        population_filter = parser.get_population_filter(population)

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
            population_filter=population_filter,
            id=id,
            group=group_tuple,
            ds_term=ds_term,
            dist_reason_term=dist_reason_term,
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
    population_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str] | None,
    ds_term: tuple[str, str],
    dist_reason_term: tuple[str, str],
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    output_file: str,
    total: bool = True,
    col_rel_width: list[float] | None = None,
    missing_group: str = "error",
) -> str:
    """
    Complete Disposition Summary Table pipeline wrapper.
    """
    # Step 1: Generate ARD
    ard = disposition_ard(
        population=population,
        population_filter=population_filter,
        id=id,
        group=group,
        ds_term=ds_term,
        dist_reason_term=dist_reason_term,
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


def _validate_disposition_data(df: pl.DataFrame, ds_var: str, reason_var: str) -> None:
    """
    Validate disposition data integrity.

    Rules:
    1. ds_var must be {Completed, Ongoing, Discontinued} and non-null.
    2. If ds_var is Completed/Ongoing, reason_var must be the same as ds_var or null.
    3. If ds_var is Discontinued, reason_var must be non-null and not Completed/Ongoing.
    """
    # Rule 1: Valid Statuses
    valid_statuses = ["Completed", "Ongoing", "Discontinued"]
    if df[ds_var].is_null().any():
        raise ValueError(f"Found null values in disposition status column '{ds_var}'")

    invalid_status = df.filter(~pl.col(ds_var).is_in(valid_statuses))
    if not invalid_status.is_empty():
        bad_values = invalid_status[ds_var].unique().to_list()
        raise ValueError(
            f"Invalid disposition statuses found: {bad_values}. Must be one of {valid_statuses}"
        )

    # Rule 2: Completed/Ongoing implies Reason is Null OR equal to Status
    inconsistent_completed = df.filter(
        (pl.col(ds_var).is_in(["Completed", "Ongoing"]))
        & (~pl.col(reason_var).is_null())
        & (pl.col(reason_var) != pl.col(ds_var))
    )
    if not inconsistent_completed.is_empty():
        raise ValueError(
            f"Found subjects with status 'Completed' or 'Ongoing' with mismatched "
            f"discontinuation reason in '{reason_var}'. Reason must be Null or match Status."
        )

    # Rule 3: Discontinued implies Reason is NOT Null AND NOT {Completed, Ongoing}
    invalid_discontinued = df.filter(
        (pl.col(ds_var) == "Discontinued")
        & ((pl.col(reason_var).is_null()) | (pl.col(reason_var).is_in(["Completed", "Ongoing"])))
    )
    if not invalid_discontinued.is_empty():
        raise ValueError(
            f"Found subjects with status 'Discontinued' but missing or invalid "
            f"discontinuation reason in '{reason_var}'"
        )


def disposition_ard(
    population: pl.DataFrame,
    population_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str] | None,
    ds_term: tuple[str, str],
    dist_reason_term: tuple[str, str],
    total: bool,
    missing_group: str,
    pop_var_name: str = "Enrolled",
) -> pl.DataFrame:
    """
    Generate ARD for Summary Table.
    """
    # Unpack variables
    ds_var_name, _ = ds_term
    dist_reason_var_name, _ = dist_reason_term
    id_var_name, _ = id

    # Validate Data
    _validate_disposition_data(population, ds_var_name, dist_reason_var_name)

    # Apply common filters
    population_filtered, _ = apply_common_filters(
        population=population,
        observation=None,
        population_filter=population_filter,
        observation_filter=None,
    )

    if group:
        group_var_name, _ = group
    else:
        # Create dummy group for overall analysis
        group_var_name = "Overall"
        total = False
        population_filtered = population_filtered.with_columns(
            pl.lit("Overall").alias(group_var_name)
        )

    # Enrolled Subjects
    n_pop_counts = count_subject(
        population=population_filtered,
        id=id_var_name,
        group=group_var_name,
        total=total,
        missing_group=missing_group,
    )

    n_pop = n_pop_counts.select(
        pl.lit(pop_var_name).alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_subj_pop").cast(pl.String).alias("__value__"),
    )

    # Disposition Status Counts (Completed, Ongoing, Discontinued, etc.)
    # We count subjects by their status in ds_var_name
    n_status_counts = count_subject_with_observation(
        population=population_filtered,
        observation=population_filtered,
        id=id_var_name,
        group=group_var_name,
        variable=ds_var_name,
        total=total,
        missing_group=missing_group,
    )

    n_status = n_status_counts.select(
        pl.col(ds_var_name).cast(pl.String).alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_pct_subj_fmt").cast(pl.String).alias("__value__"),
    )

    # Discontinued Subjects by Reason
    # We filter where ds_var_name == "Discontinued"
    filtered_for_reason = population_filtered.filter(pl.col(ds_var_name) == "Discontinued")

    n_dist_reason_counts = count_subject_with_observation(
        population=population_filtered,
        observation=filtered_for_reason,
        id=id_var_name,
        group=group_var_name,
        variable=dist_reason_var_name,
        total=total,
        missing_group=missing_group,
    )

    n_dist_reason = n_dist_reason_counts.select(
        (pl.lit("    ") + pl.col(dist_reason_var_name).cast(pl.String)).alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_pct_subj_fmt").cast(pl.String).alias("__value__"),
    ).sort("__index__")

    # Missing Subjects
    n_missing_counts = count_subject(
        population=filtered_for_reason.filter(pl.col(dist_reason_var_name).is_null()),
        id=id_var_name,
        group=group_var_name,
        total=total,
        missing_group=missing_group,
    )

    n_missing = n_missing_counts.select(
        pl.lit("Missing").alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_subj_pop").cast(pl.String).alias("__value__"),
    )

    return pl.concat([n_pop, n_status, n_dist_reason, n_missing])


def disposition_df(ard: pl.DataFrame) -> pl.DataFrame:
    """
    Transform ARD to display format.
    """
    # Pivot
    # Pivot from long to wide format
    df_wide = ard.pivot(index="__index__", on="__group__", values="__value__")

    # Rename __index__ to display column name
    df_wide = df_wide.rename({"__index__": "Term"}).select(pl.col("Term"), pl.exclude("Term"))

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

    return create_rtf_table_n_pct(
        df=df,
        col_header_1=col_header_1,
        col_header_2=col_header_2,
        col_widths=col_widths,
        title=title,
        footnote=footnote,
        source=source,
    )
