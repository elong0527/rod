# pyre-strict
"""
Inclusion/Exclusion (IE) Table Analysis Functions

This module provides a pipeline for IE summary analysis:
- ie_ard: Generate Analysis Results Data (ARD)
- ie_df: Transform ARD to display format
- ie_rtf: Generate formatted RTF output
- study_plan_to_ie_summary: Batch generation from StudyPlan
"""

from pathlib import Path
from typing import Any

import polars as pl

from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_listing, create_rtf_table_n_pct
from ..common.utils import apply_common_filters


def study_plan_to_ie_summary(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate IE Summary Table outputs for all analyses defined in StudyPlan.
    """
    # Meta data
    analysis_type = "ie_summary"
    output_dir = study_plan.output_dir
    title = "Summary of Protocol Deviations (Inclusion/Exclusion)"
    # footnote = ["Percentages are based on the number of enrolled participants."]

    # Defaults
    criteria_df_name = "adie"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan (Manually expansion to avoid AttributeError)
    plans = study_plan.study_data.get("plans", [])
    all_specs = []
    for plan_data in plans:
        expanded = study_plan.expander.expand_plan(plan_data)
        for p in expanded:
            all_specs.append(study_plan.expander.create_analysis_spec(p))

    plan_df = pl.DataFrame(all_specs)

    ie_plans = plan_df.filter(pl.col("analysis") == analysis_type)

    generated_files = []

    # Iterate over analyses
    for analysis in ie_plans.iter_rows(named=True):
        # Load data
        # Note: IE analysis needs both ADSL (for population/group) and ADIE (for criteria)
        pop_name = analysis.get("population", "enrolled")
        group_kw = analysis.get("group")  # Can be None

        try:
            if group_kw:
                # Load Filtered Population (ADSL) with Group
                adsl, group_col = parser.get_population_data(pop_name, group_kw)
                group_col = group_col.upper()
                grp_suffix = group_col
            else:
                # Load Filtered Population (ADSL) without Group
                # Manual load + filter since get_population_data requires group
                (adsl_raw,) = parser.get_datasets("adsl")
                pop_filter = parser.get_population_filter(pop_name)

                adsl, _ = apply_common_filters(
                    population=adsl_raw,
                    observation=None,
                    population_filter=pop_filter,
                    observation_filter=None,
                )

                group_col = None
                grp_suffix = "total"

        except ValueError as e:
            print(f"Error loading population: {e}")
            continue

        # Load ADIE
        try:
            (adie,) = parser.get_datasets(criteria_df_name)
        except ValueError as e:
            print(f"Error loading datasets: {e}")
            continue

        # Output filename
        filename = f"{analysis_type}_{pop_name}_{grp_suffix}.rtf".lower()
        output_path = f"{output_dir}/{filename}"

        # Generate ARD
        ard = ie_ard(adsl=adsl, adie=adie, group_col=group_col)

        # Generate DF
        df = ie_df(ard)

        # Generate RTF
        ie_rtf(df, output_path, title=title)

        generated_files.append(output_path)

    return generated_files


def ie_ard(adsl: pl.DataFrame, adie: pl.DataFrame, group_col: str | None = None) -> pl.DataFrame:
    """
    Generate Analysis Results Data (ARD) for IE Table.

    Structure:
    - Total Screening Failures
    - Exclusion Criteria Met
      - [Detail]
    - Inclusion Criteria Not Met
      - [Detail]
    """
    # If group_col is None, create a dummy group column
    actual_group_col: str = group_col if group_col else "Total"

    # 1. Prepare Data
    # Join ADIE to ADSL to get treatment group info
    df_joined: pl.DataFrame = adie.join(
        adsl.select(["USUBJID"] + ([group_col] if group_col else [])), on="USUBJID", how="inner"
    )

    if not group_col:
        # Add dummy Total column
        df_joined = df_joined.with_columns(pl.lit("Total").alias("Total"))

    # Define hierarchy
    results: list[dict[str, Any]] = []

    # Get distinct groups
    groups: list[str]
    if group_col:
        groups_raw: list[str | None] = sorted(adsl.select(group_col).unique().to_series().to_list())
        groups = [g for g in groups_raw if g is not None]
    else:
        groups = ["Total"]

    # Helper to calculate n and pct (pct of what? usually pct of failures? or pct of screened?)
    # Usually IE table % is based on Total Screening Failures.
    # Let's count Total Screening Failures per Group first.

    # Total Screening Failures (Subjects present in ADIE)
    # Note: A subject can match multiple criteria.
    total_failures_by_group = df_joined.group_by(actual_group_col).agg(
        pl.col("USUBJID").n_unique().alias("count")
    )

    total_failures_map: dict[str, int] = {
        row[actual_group_col]: row["count"] for row in total_failures_by_group.iter_rows(named=True)
    }

    # Helper for row generation
    def add_row(
        label: str, filter_expr: pl.Expr | None = None, is_header: bool = False, indent: int = 0
    ) -> None:
        row_data: dict[str, Any] = {"label": label, "indent": indent, "is_header": is_header}

        for g in groups:
            # Filter data for this group
            g_df = df_joined.filter(pl.col(actual_group_col) == g)

            if filter_expr is not None:
                # Filter specific criteria
                g_df = g_df.filter(filter_expr)

            n = g_df.select("USUBJID").n_unique()

            # Pct based on total failures in that group?
            denom = total_failures_map.get(g, 0)
            pct = (n / denom * 100) if denom > 0 else 0.0

            row_data[f"count_{g}"] = n
            row_data[f"pct_{g}"] = pct

        results.append(row_data)

    # 1. Total Screening Failures
    add_row("Total Screening Failures")

    # 2. Exclusion Criteria Met
    excl_expr = pl.col("PARAMCAT") == "EXCLUSION CRITERIA MET"
    add_row("Exclusion Criteria Met", excl_expr, is_header=True, indent=1)

    # Details for Exclusion
    excl_params = (
        df_joined.filter(excl_expr).select("PARAM").unique().sort("PARAM").to_series().to_list()
    )
    for param in excl_params:
        add_row(param, excl_expr & (pl.col("PARAM") == param), indent=2)

    # 3. Inclusion Criteria Not Met
    incl_expr = pl.col("PARAMCAT") == "INCLUSION CRITERIA NOT MET"
    add_row("Inclusion Criteria Not Met", incl_expr, is_header=True, indent=1)

    # Details for Inclusion
    incl_params = (
        df_joined.filter(incl_expr).select("PARAM").unique().sort("PARAM").to_series().to_list()
    )
    for param in incl_params:
        add_row(param, incl_expr & (pl.col("PARAM") == param), indent=2)

    return pl.DataFrame(results)


def ie_df(ard: pl.DataFrame) -> pl.DataFrame:
    """Transform ARD to display DataFrame."""
    # Find group columns
    cols = ard.columns
    group_cols = [c for c in cols if c.startswith("count_")]
    groups = [c.replace("count_", "") for c in group_cols]

    # Create valid Polars expressions for selecting columns
    # Apply indentation: 3 spaces per indent level
    # Note: Using \u00A0 (NBSP) might be safer for RTF if spaces get collapsed,
    # but regular spaces usually work in table cells. Let's start with regular spaces.

    select_exprs = [
        (pl.lit("   ").repeat_by(pl.col("indent")).list.join("") + pl.col("label")).alias(
            "Criteria"
        )
    ]

    for g in groups:
        # Format n (%)
        # We need to construct the string.
        # Polars string formatting
        # format: "{n} ({pct:.1f})"

        # Note: Polars doesn't have f-string strictly in expressions like python
        # We use strict casting and concatenation

        col_n = pl.col(f"count_{g}")
        col_pct = pl.col(f"pct_{g}")

        fmt = (
            col_n.cast(pl.Utf8)
            + " ("
            + col_pct.map_elements(lambda x: f"{x:.1f}", return_dtype=pl.Utf8)
            + ")"
        ).alias(g)

        select_exprs.append(fmt)

    return ard.select(select_exprs)


def ie_rtf(df: pl.DataFrame, output_path: str, title: str = "") -> None:
    """Generate RTF."""

    # Rename Criteria column to empty string for display if needed or keep as is?
    # Usually "Criteria".

    # Calculate number of columns
    n_cols = len(df.columns)

    # Build first-level column headers (use actual column names)
    col_header_1 = list(df.columns)

    # Build second-level column headers (empty for first, "n (%)" for groups)
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

    # Calculate column widths - auto-calculate
    # [n_cols-1, 1, 1, 1, ...]
    col_widths = [float(n_cols - 1)] + [1.0] * (n_cols - 1)

    rtf_doc = create_rtf_table_n_pct(
        df=df,
        col_header_1=col_header_1,
        col_header_2=col_header_2,
        col_widths=col_widths,
        title=[title],
        footnote=None,
        source=None,
    )

    rtf_doc.write_rtf(output_path)


def study_plan_to_ie_listing(
    study_plan: StudyPlan,
) -> list[str]:
    """
    Generate IE Listing outputs.
    """
    # Meta data
    analysis_type = "ie_listing"
    output_dir = study_plan.output_dir
    title = "Listing of Protocol Deviations"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan (Manually expansion to avoid AttributeError)
    plans = study_plan.study_data.get("plans", [])
    all_specs = []
    for plan_data in plans:
        expanded = study_plan.expander.expand_plan(plan_data)
        for p in expanded:
            all_specs.append(study_plan.expander.create_analysis_spec(p))

    plan_df = pl.DataFrame(all_specs)

    # Filter for ie_listing if user specified it, otherwise we might trigger it manually
    # But for now let's assume if this function is called, we want to run for "ie_listing"
    # If the user only asked for "ie_listing", we might need to add it to the plan or just run it blindly for all?
    # Usually we filter by analysis_type.
    listing_plans = plan_df.filter(pl.col("analysis") == analysis_type)

    # If no specific listing plans are found, maybe we should just generate one default one for the whole study?
    # But sticking to the pattern:
    generated_files = []

    # If listing_plans is empty, let's create a default one to ensure we generate something for the user
    # since they explicitly asked for it.
    if listing_plans.height == 0:
        # Create a dummy row to force generation
        listing_plans = pl.DataFrame([{"population": "enrolled", "analysis": analysis_type}])

    for analysis in listing_plans.iter_rows(named=True):
        # Load ADSL
        pop_name = analysis.get("population", "enrolled")

        try:
            # Load Filtered Population (ADSL) without Group (Listings usually don't group by columns like tables)
            # But if they did, we'd handle it. For now, just load raw ADSL.
            (adsl_raw,) = parser.get_datasets("adsl")
            # We could filter, but user just said "show USUBJID and DCSREAS from adsl".
            # Applying population filter if possible.
            pop_filter = parser.get_population_filter(pop_name)

            adsl, _ = apply_common_filters(
                population=adsl_raw,
                observation=None,
                population_filter=pop_filter,
                observation_filter=None,
            )

        except ValueError as e:
            print(f"Error loading population: {e}")
            continue

        # Output filename
        filename = f"{analysis_type}_{pop_name}.rtf".lower()
        output_path = f"{output_dir}/{filename}"

        # Generate DF
        df = ie_listing_df(adsl)

        # Generate RTF
        ie_listing_rtf(df, output_path, title=title)

        generated_files.append(output_path)

    return generated_files


def ie_listing_df(adsl: pl.DataFrame) -> pl.DataFrame:
    """Select columns for Listing."""
    # Check if DCSREAS exists
    cols = ["USUBJID", "DCSREAS"]
    available = [c for c in cols if c in adsl.columns]
    return adsl.select(available)


def ie_listing_rtf(df: pl.DataFrame, output_path: str, title: str = "") -> None:
    """Generate RTF Listing."""

    col_widths = [1.5, 3.5]  # Approximate ratio

    rtf_doc = create_rtf_listing(
        df=df,
        col_header=list(df.columns),
        col_widths=col_widths,
        title=[title],
        footnote=None,
        source=None,
    )

    rtf_doc.write_rtf(output_path)
