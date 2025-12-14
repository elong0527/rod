# pyre-strict
"""
Inclusion/Exclusion (IE) Analysis Functions

This module provides a pipeline for Inclusion/Exclusion Table summary analysis:
- ie_summary_ard: Generate Analysis Results Data (ARD)
- ie_summary_df: Transform ARD to display format
- ie_summary_rtf: Generate formatted RTF output
- ie_summary: Complete pipeline wrapper
- study_plan_to_ie_summary: Batch generation from StudyPlan
"""

from pathlib import Path

import polars as pl
from rtflite import RTFDocument

from ..common.count import count_subject, count_subject_with_observation
from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_table_n_pct
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
    title = "Summary of Inclusion/Exclusion Criteria Exceptions"
    footnote = ["Percentages are based on the number of enrolled participants."]
    source = None

    population_df_name = "adsl"
    observation_df_name = "adie"

    id = ("USUBJID", "Subject ID")

    # Standard IE terms usually don't vary by protocol, but we'll use defaults
    # For IE, we count subjects who met criteria (e.g. AFLAG='Y')

    total = True
    missing_group = "error"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan DataFrame
    plan_df = study_plan.get_plan_df()

    # Filter for ie analyses
    ie_plans = plan_df.filter(pl.col("analysis") == analysis_type)

    rtf_files = []

    for row in ie_plans.iter_rows(named=True):
        population = row["population"]
        group = row.get("group")
        title_text = title

        # Get datasets
        population_df, observation_df = parser.get_datasets(population_df_name, observation_df_name)

        # Get filters
        population_filter = parser.get_population_filter(population)

        # Get group info (optional)
        if group is not None:
            group_var_name, group_labels = parser.get_group_info(group)
            group_var_label = group_labels[0] if group_labels else group_var_name
            group_tuple = (group_var_name, group_var_label)
        else:
            # When no group specified, use a dummy group column for overall counts
            group_tuple = ("Overall", "Overall")

        # Build title
        title_parts = [title_text]
        pop_kw = study_plan.keywords.populations.get(population)
        if pop_kw and pop_kw.label:
            title_parts.append(pop_kw.label)

        # Build output filename
        group_suffix = f"_{group}" if group else ""
        filename = f"{analysis_type}_{population}{group_suffix}.rtf"
        output_file = str(Path(output_dir) / filename)

        rtf_path = ie_summary(
            population=population_df,
            observation=observation_df,
            population_filter=population_filter,
            id=id,
            group=group_tuple,
            title=title_parts,
            footnote=footnote,
            source=source,
            output_file=output_file,
            total=total,
            missing_group=missing_group,
        )
        rtf_files.append(rtf_path)

    return rtf_files


def ie_summary(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str],
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    output_file: str,
    total: bool = True,
    col_rel_width: list[float] | None = None,
    missing_group: str = "error",
) -> str:
    """
    Complete IE Summary Table pipeline wrapper.
    """
    # Step 1: Generate ARD
    ard = ie_summary_ard(
        population=population,
        observation=observation,
        population_filter=population_filter,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    # Step 2: Transform to display format
    df = ie_summary_df(ard)

    # Step 3: Generate RTF
    rtf_doc = ie_summary_rtf(
        df=df,
        title=title,
        footnote=footnote,
        source=source,
        col_rel_width=col_rel_width,
    )
    rtf_doc.write_rtf(output_file)

    return output_file


def ie_summary_ard(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    id: tuple[str, str],
    group: tuple[str, str],
    total: bool,
    missing_group: str,
    pop_var_name: str = "Total Subjects Screened",
) -> pl.DataFrame:
    """
    Generate ARD for IE Summary Table.
    """
    id_var_name, _ = id
    group_var_name, _ = group

    # Apply common filters
    population_filtered, observation_to_filter = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=None,
    )

    assert observation_to_filter is not None

    # Identify screen failures in observation not in population
    # Only if they have AFLAG='Y'
    obs_failures = observation_to_filter.filter(pl.col("AFLAG") == "Y")

    # Get IDs of failures not in population
    failure_ids_not_in_pop = (
        obs_failures.filter(~pl.col(id_var_name).is_in(population_filtered[id_var_name]))
        .select(id_var_name)
        .unique()
    )

    if failure_ids_not_in_pop.height > 0:
        # We need to construct a population dataframe for these subjects
        # We need the group variable.
        # Check if group_var matches TRT01A (common case) and ADIE has TRT01P
        group_col_source = group_var_name
        if group_var_name not in obs_failures.columns:
            if group_var_name == "TRT01A" and "TRT01P" in obs_failures.columns:
                group_col_source = "TRT01P"
            else:
                # Fallback: Can't determine group, maybe use 'Missing' or error
                # For now, let's try to find it or errors will occur later
                pass

        # Select unique subjects and their group from failures
        # Note: A subject might have multiple failures, so multiple rows.
        # We need unique ID and Group. Assuming Group is constant for ID.
        failures_pop = (
            obs_failures.filter(~pl.col(id_var_name).is_in(population_filtered[id_var_name]))
            .select([id_var_name, pl.col(group_col_source).alias(group_var_name)])
            .unique(subset=[id_var_name])
        )

        # Combine
        # We only need ID and Group for counting
        pop_core = population_filtered.select([id_var_name, group_var_name])
        population_combined = pl.concat([pop_core, failures_pop], how="diagonal")
    else:
        population_combined = population_filtered

    # Update population_filtered to be the combined one for downstream counting
    population_filtered = population_combined

    # Filter for criteria failures (AFLAG = 'Y')
    # and subjects present in filtered (combined) population
    observation_filtered = obs_failures.filter(
        pl.col(id_var_name).is_in(population_filtered[id_var_name])
    )

    if group_var_name == "Overall":
        if "Overall" not in population_filtered.columns:
            population_filtered = population_filtered.with_columns(
                pl.lit("Overall").alias("Overall")
            )
        total = False

    # 1. Total Subjects Screened
    n_pop_counts = count_subject(
        population=population_filtered,
        id=id_var_name,
        group=group_var_name,
        total=total,
        missing_group=missing_group,
    )

    n_pop_formatted = n_pop_counts.select(
        pl.lit(pop_var_name).alias("__index__"),
        pl.col(group_var_name).cast(pl.String).alias("__group__"),
        pl.col("n_subj_pop").cast(pl.String).alias("__value__"),
    )
    # ------------------------------------------------------------------
    # 2. Iterate over Category + Detail Rows
    # ------------------------------------------------------------------
    # The categories we want, in specific order
    categories = [
        ("Inclusion Criteria Not Met", "Inclusion"),
        ("Exclusion Criteria Met", "Exclusion"),
    ]

    cat_blocks = []

    for display_cat_name, search_string in categories:
        # Filter observations for this category
        obs_cat_all = observation_filtered.filter(
            pl.col("PARAMCAT").str.to_lowercase().str.contains(search_string.lower())
        )

        # If no observations for this category, we might still want 
        # to show the header '0' if total is needed.
        # But typically if empty we show 0.

        # --- Header Row (with counts) ---
        # We create a dummy variable to count "Subjects with ANY failure in this category"
        # The dummy columns allow count_subject_with_observation to treat the header as a variable.
        if obs_cat_all.height > 0:
            obs_for_header = obs_cat_all.with_columns(
                pl.lit(display_cat_name).alias("HeaderVariable")
            )
            n_header = count_subject_with_observation(
                population=population_filtered,
                observation=obs_for_header,
                id=id_var_name,
                group=group_var_name,
                variable="HeaderVariable",
                total=total,
                missing_group=missing_group,
            )

            # The result has __variable__="Inclusion Criteria Not Met" (from our dummy).
            # We rename __variable__ to __index__.
            # We do NOT want indentation for the header.
            n_header = n_header.with_columns(pl.col("__variable__").alias("__index__"))
            cat_blocks.append(n_header)  # Add header to blocks
        else:
            # If empty, create 0 rows? Or just Skip?
            # User expectation: "Inclusion Criteria Not Met" 0/N ...
            # TODO: If we want 0s, we need to construct it manually or pass empty obs.
            # count_subject_with_observation handles empty obs if variables provided.
            # But here variable values come FROM data usually.
             # For now assume if no data, no header needed.
             # (or handled by count's 0 filling if we pass explicit var list?)
            continue

        # --- Detail Rows (Indented) ---
        n_detail = count_subject_with_observation(
            population=population_filtered,
            observation=obs_cat_all,
            id=id_var_name,
            group=group_var_name,
            variable="PARAM",  # Use "PARAM" for detail rows
            total=total,
            missing_group=missing_group,
        )

        # Add indentation to detail rows
        n_detail = n_detail.with_columns(
            (pl.lit("    ") + pl.col("__variable__")).alias("__index__")
        )

        # Concatenate Header + Details
        cat_blocks.append(n_detail)  # Add details to blocks

    if not cat_blocks:
        return n_pop_formatted.select(
            [
                pl.col(c)
                for c in n_pop_formatted.columns
                if c in ["__index__", "__group__", "__value__"]
            ]
        )

    # Combine everything: Population -> Inclusion Block -> Exclusion Block
    # Note: We skipped "Any Eligibility Criteria" as requested.

    # We need to conform columns

    final_dfs = [n_pop_formatted] + cat_blocks

    # Standardize columns before concat
    # We need: __index__, __group__ (renamed from group_var), __value__ (renamed from n_pct)

    standardized_dfs = []
    for df in final_dfs:
        # Map value column
        val_col = "__value__" if "__value__" in df.columns else "n_pct_subj_fmt"
        # Map group column
        grp_col = "__group__" if "__group__" in df.columns else group_var_name

        df_std = df.select(
            pl.col("__index__"),
            pl.col(grp_col).cast(pl.String).alias("__group__"),
            pl.col(val_col).alias("__value__"),
        )
        standardized_dfs.append(df_std)

    ard = pl.concat(standardized_dfs)

    return ard


def ie_summary_df(ard: pl.DataFrame) -> pl.DataFrame:
    """
    Transform ARD to display format.
    """
    # Capture the order of terms from the ARD (which is already sorted)
    # We want to preserve this order in the pivot
    terms_order = ard["__index__"].unique(maintain_order=True).to_list()

    # Cast to Enum to enforce order
    ard_ordered = ard.with_columns(pl.col("__index__").cast(pl.Enum(terms_order)))

    # Pivot
    df_wide = ard_ordered.pivot(index="__index__", on="__group__", values="__value__")

    # Rename __index__ to display column name
    df_wide = df_wide.rename({"__index__": "Criteria"}).select(
        pl.col("Criteria"), pl.exclude("Criteria")
    )

    # Sanitize strings to avoid RTF encoding errors (e.g. smart quotes, non-ascii chars)
    # rtflite might not handle utf-8 well on all platforms/configs
    # We must handle Enum columns (Criteria) too
    sanitized_cols = []
    for c in df_wide.columns:
        dtype = df_wide[c].dtype
        if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
            col_expr = (
                pl.col(c)
                .cast(pl.String)
                .map_elements(
                    lambda s: s.encode("ascii", "ignore").decode("ascii") if s else s,
                    return_dtype=pl.String,
                )
            )
            sanitized_cols.append(col_expr)
        else:
            sanitized_cols.append(pl.col(c))

    df_wide = df_wide.select(sanitized_cols)

    return df_wide


def ie_summary_rtf(
    df: pl.DataFrame,
    title: list[str],
    footnote: list[str] | None,
    source: list[str] | None,
    col_rel_width: list[float] | None = None,
) -> RTFDocument:
    """
    Generate RTF.
    """

    # Sanitize metadata
    def safe_str(s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii")

    safe_title = [safe_str(t) for t in title]

    safe_footnote = None
    if footnote:
        safe_footnote = [safe_str(f) for f in footnote]

    safe_source = None
    if source:
        safe_source = [safe_str(s) for s in source]

    n_cols = len(df.columns)
    col_header_1 = [""] + [safe_str(c) for c in df.columns[1:]]
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

    if col_rel_width is None:
        col_widths = [3.0] + [1] * (n_cols - 1)
    else:
        col_widths = col_rel_width

    return create_rtf_table_n_pct(
        df=df,
        col_header_1=col_header_1,
        col_header_2=col_header_2,
        col_widths=col_widths,
        title=safe_title,
        footnote=safe_footnote,
        source=safe_source,
    )
