# pyre-strict
"""
Protocol Deviation (PD) Listing Functions

This module provides functions for generating detailed PD listings showing individual
protocol deviation records with key details like category, term, and coded term.

The two-step pipeline:
- pd_listing_ard: Filter, select, sort, and rename columns (returns display-ready data)
- pd_listing_rtf: Generate formatted RTF output
- pd_listing: Complete pipeline wrapper
- study_plan_to_pd_listing: Batch generation from StudyPlan

Uses Polars native SQL capabilities for data manipulation and parse.py utilities
for StudyPlan parsing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from rtflite import RTFBody, RTFColumnHeader, RTFDocument, RTFFootnote, RTFPage, RTFSource, RTFTitle

from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.utils import apply_common_filters


def pd_listing_ard(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: Optional[str],
    observation_filter: Optional[str],
    id: Tuple[str, str],
    population_columns: Optional[List[Tuple[str, str]]] = None,
    observation_columns: Optional[List[Tuple[str, str]]] = None,
    sort_columns: Optional[List[str]] = None,
    page_by: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Generate Analysis Results Data (ARD) for PD listing.

    Filters and joins population and observation data, then selects relevant columns.

    Args:
        population: Population DataFrame (subject-level data, e.g., ADSL)
        observation: Observation DataFrame (event data, e.g., ADPD)
        population_filter: SQL WHERE clause for population (can be None)
        observation_filter: SQL WHERE clause for observation (can be None)
        id: Tuple (variable_name, label) for ID column
        population_columns: List of tuples (variable_name, label) from population
                            (e.g., [("SEX", "Sex"), ("RACE", "Race")])
        observation_columns: List of tuples (variable_name, label) from observation
                             (e.g., [("DVCAT", "Category")])
        sort_columns: List of column names to sort by. If None, sorts by id column.
        page_by: List of column names to page by.

    Returns:
        pl.DataFrame: Filtered and joined records with selected columns
    """
    id_var_name, id_var_label = id

    # Apply common filters
    # Note: parameter_filter is None for PD as it usually doesn't have a PARAM column like AE
    population_filtered, observation_to_filter = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        parameter_filter=None,
    )

    assert observation_to_filter is not None

    # Filter observation to include only subjects in filtered population
    observation_filtered = observation_to_filter.filter(
        pl.col(id_var_name).is_in(population_filtered[id_var_name].to_list())
    )

    # Determine which observation columns to select
    if observation_columns is None:
        # Default: select id column only
        obs_cols = [id_var_name]
    else:
        # Extract variable names from tuples
        obs_col_names = [var_name for var_name, _ in observation_columns]
        # Ensure id is included
        obs_cols = [id_var_name] + [col for col in obs_col_names if col != id_var_name]

    # Select available observation columns
    obs_cols_available = [col for col in obs_cols if col in observation_filtered.columns]
    result = observation_filtered.select(obs_cols_available)

    # Join with population to add population columns
    if population_columns is not None:
        # Extract variable names from tuples
        pop_col_names = [var_name for var_name, _ in population_columns]
        # Select id + requested population columns
        pop_cols = [id_var_name] + [col for col in pop_col_names if col != id_var_name]
        pop_cols_available = [col for col in pop_cols if col in population_filtered.columns]
        population_subset = population_filtered.select(pop_cols_available)

        # Left join to preserve all observation records
        result = result.join(population_subset, on=id_var_name, how="left")

    # Create __index__ column for pagination
    # Default to using the id column as the index
    if id_var_name in result.columns:
        result = result.with_columns(
            (pl.lit(f"{id_var_label} = ") + pl.col(id_var_name).cast(pl.Utf8)).alias("__index__")
        )

    # Use page_by columns if provided and they exist
    existing_page_by_cols = [col for col in page_by if col in result.columns] if page_by else []

    if existing_page_by_cols:
        # Create a mapping from column name to label
        column_labels = {id_var_name: id_var_label}
        if population_columns:
            for var_name, var_label in population_columns:
                column_labels[var_name] = var_label

        # Ensure the order of labels matches the order of columns in page_by
        index_expressions = []
        for col_name in existing_page_by_cols:
            label = column_labels.get(col_name, col_name)
            index_expressions.append(pl.lit(f"{label} = ") + pl.col(col_name).cast(pl.Utf8))

        result = result.with_columns(
            pl.concat_str(index_expressions, separator=", ").alias("__index__")
        )

        page_by_remove = [col for col in (page_by or []) if col != id_var_name]
        result = result.drop(page_by_remove)

    if "__index__" in result.columns:
        # Get all columns except __index__
        other_columns = [col for col in result.columns if col != "__index__"]
        # Reorder to have __index__ first
        result = result.select(["__index__"] + other_columns)

    # Sort by specified columns or default to id column
    if sort_columns is None:
        # Default: sort by id column if it exists in result
        if id_var_name in result.columns:
            result = result.sort(id_var_name)
    else:
        # Sort by specified columns that exist in result
        cols_to_sort = [col for col in sort_columns if col in result.columns]
        if cols_to_sort:
            result = result.sort(cols_to_sort)

    return result


def pd_listing_rtf(
    df: pl.DataFrame,
    column_labels: Dict[str, str],
    title: List[str],
    footnote: Optional[List[str]],
    source: Optional[List[str]],
    col_rel_width: Optional[List[float]] = None,
    group_by: Optional[List[str]] = None,
    page_by: Optional[List[str]] = None,
    orientation: str = "landscape",
) -> RTFDocument:
    """
    Generate RTF table from PD listing display DataFrame.

    Creates a formatted RTF table with column headers and optional section grouping/pagination.

    Args:
        df: Display DataFrame from pd_listing_ard
        column_labels: Dictionary mapping column names to display labels
        title: Title(s) for the table as list of strings
        footnote: Optional footnote(s) as list of strings
        source: Optional source note(s) as list of strings
        col_rel_width: Optional list of relative column widths. If None, auto-calculated
                       as equal widths for all columns
        group_by: Optional list of column names to group by for section headers within pages.
                  Should only contain population columns (e.g., ["TRT01A", "USUBJID"])
        page_by: Optional list of column names to trigger new pages when values change.
                 Should only contain population columns (e.g., ["TRT01A"])
        orientation: Page orientation ("portrait" or "landscape"), default is "landscape"

    Returns:
        RTFDocument: RTF document object that can be written to file
    """
    # Calculate number of columns
    n_cols = len(df.columns)

    # Build column headers using labels
    col_header = [column_labels.get(col, col) for col in df.columns]

    # Calculate column widths
    if col_rel_width is None:
        col_widths = [1.0] * n_cols
    else:
        col_widths = col_rel_width

    # Normalize title, footnote, source to lists
    title_list = title
    footnote_list: List[str] = footnote or []
    source_list: List[str] = source or []

    # Build RTF document
    rtf_components: Dict[str, Any] = {
        "df": df,
        "rtf_page": RTFPage(orientation=orientation),
        "rtf_title": RTFTitle(text=title_list),
        "rtf_column_header": [
            RTFColumnHeader(
                text=col_header[1:],
                col_rel_width=col_widths[1:],
                text_justification=["l"] + ["l"] * (n_cols - 1),  # Default left align for PD
            ),
        ],
        "rtf_body": RTFBody(
            col_rel_width=col_widths,
            text_justification=["l"] * n_cols,
            border_left=["single"],
            border_top=["single"] + [""] * (n_cols - 1),
            border_bottom=["single"] + [""] * (n_cols - 1),
            group_by=group_by,
            page_by=page_by,
        ),
    }

    # Add optional footnote
    if footnote_list:
        rtf_components["rtf_footnote"] = RTFFootnote(text=footnote_list)

    # Add optional source
    if source_list:
        rtf_components["rtf_source"] = RTFSource(text=source_list)

    # Create RTF document
    doc = RTFDocument(**rtf_components)

    return doc


def pd_listing(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: Optional[str],
    observation_filter: Optional[str],
    id: Tuple[str, str],
    title: List[str],
    footnote: Optional[List[str]],
    source: Optional[List[str]],
    output_file: str,
    population_columns: Optional[List[Tuple[str, str]]] = None,
    observation_columns: Optional[List[Tuple[str, str]]] = None,
    sort_columns: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    page_by: Optional[List[str]] = None,
    col_rel_width: Optional[List[float]] = None,
    orientation: str = "landscape",
) -> str:
    """
    Complete PD listing pipeline wrapper.

    This function orchestrates the two-step pipeline:
    1. pd_listing_ard: Filter, join, select, and sort columns
    2. pd_listing_rtf: Generate RTF output with optional grouping/pagination

    Args:
        population: Population DataFrame (subject-level data, e.g., ADSL)
        observation: Observation DataFrame (event data, e.g., ADPD)
        population_filter: SQL WHERE clause for population (can be None)
        observation_filter: SQL WHERE clause for observation (can be None)
        id: Tuple (variable_name, label) for ID column
        title: Title for RTF output as list of strings
        footnote: Optional footnote for RTF output as list of strings
        source: Optional source for RTF output as list of strings
        output_file: File path to write RTF output
        population_columns: Optional list of tuples (variable_name, label) from population
        observation_columns: Optional list of tuples (variable_name, label) from observation
        sort_columns: Optional list of column names to sort by. If None, sorts by id column.
        group_by: Optional list of column names to group by for section headers
                  (population columns only)
        page_by: Optional list of column names to trigger new pages (population columns only)
        col_rel_width: Optional column widths for RTF output
        orientation: Page orientation ("portrait" or "landscape"), default is "landscape"
    """
    # Step 1: Generate ARD (includes filtering, joining, and selecting)
    df = pd_listing_ard(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        id=id,
        population_columns=population_columns,
        observation_columns=observation_columns,
        sort_columns=sort_columns,
        page_by=page_by,
    )

    # Build column labels from tuples
    id_var_name, id_var_label = id
    column_labels = {id_var_name: id_var_label}

    # Add observation column labels
    if observation_columns is not None:
        for var_name, var_label in observation_columns:
            column_labels[var_name] = var_label

    # Add population column labels
    if population_columns is not None:
        for var_name, var_label in population_columns:
            column_labels[var_name] = var_label

    # Set __index__ header to empty string
    column_labels["__index__"] = ""

    # Step 2: Generate RTF and write to file
    rtf_doc = pd_listing_rtf(
        df=df,
        column_labels=column_labels,
        title=title,
        footnote=footnote,
        source=source,
        col_rel_width=col_rel_width,
        group_by=group_by,
        page_by=["__index__"],
        orientation=orientation,
    )
    rtf_doc.write_rtf(output_file)

    return output_file


def study_plan_to_pd_listing(
    study_plan: StudyPlan,
) -> List[str]:
    """
    Generate PD listing RTF outputs for all analyses defined in StudyPlan.

    This function reads the expanded plan from StudyPlan and generates
    an RTF listing for each pd_listing analysis specification automatically.

    Args:
        study_plan: StudyPlan object with loaded datasets and analysis specifications

    Returns:
        List[str]: List of paths to generated RTF files
    """

    # Meta data
    analysis = "pd_listing"
    output_dir = study_plan.output_dir
    # Adjusted column widths for PD
    col_rel_width = [1.0, 1.0, 1.0, 3.0, 2.0]
    footnote = None
    source = None

    population_df_name = "adsl"
    observation_df_name = "adpd"

    id = ("USUBJID", "Subject ID")
    # Column configuration with labels - easy to customize
    # Population columns (demographics) - group variable will be added dynamically
    population_columns_base: List[
        Tuple[str, str]
    ] = []  # Often empty for PD listing, maybe just ID and treatment

    # Observation columns (event details)
    # Using the columns identified from adpd.parquet
    observation_columns_base = [
        ("DVCAT", "Category"),
        ("DVTERM", "Term"),
        ("DVDECOD", "Coded Term"),
    ]

    # Sorting configuration
    sort_columns = ["TRT01A", "USUBJID", "DVCAT", "DVTERM"]
    page_by = ["USUBJID", "TRT01A"]
    group_by = ["USUBJID"]

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get expanded plan DataFrame
    plan_df = study_plan.get_plan_df()

    # Filter for PD listing analyses
    pd_plans = plan_df.filter(pl.col("analysis") == analysis)

    rtf_files = []

    # Generate RTF for each analysis
    for row in pd_plans.iter_rows(named=True):
        population = row["population"]
        observation = row.get("observation")
        group = row.get("group")

        # Validate group is specified
        if group is None:
            raise ValueError(
                f"Group not specified in YAML for analysis: population={population}, "
                f"observation={observation}. "
                "Please add group to your YAML plan."
            )

        # Get datasets using parser
        population_df, observation_df = parser.get_datasets(population_df_name, observation_df_name)

        # Get filters using parser
        population_filter = parser.get_population_filter(population)
        obs_filter = parser.get_observation_filter(observation)

        # Get group variable name from YAML
        group_var_name, group_labels = parser.get_group_info(group)

        # Determine group variable label
        group_var_label = group_labels[0] if group_labels else "Treatment"

        # Build columns dynamically from base configuration with labels
        population_columns = population_columns_base + [(group_var_name, group_var_label)]
        observation_columns = observation_columns_base

        # Build title with population and observation context
        title_parts = ["Listing of Protocol Deviations"]

        pop_kw = study_plan.keywords.populations.get(population)
        if pop_kw and pop_kw.label:
            title_parts.append(pop_kw.label)

        # Build output filename
        filename = f"{analysis}_{population}"
        if observation:
            filename += f"_{observation}"
        filename += ".rtf"
        output_file = str(Path(output_dir) / filename)

        # Generate RTF
        rtf_path = pd_listing(
            population=population_df,
            observation=observation_df,
            population_filter=population_filter,
            observation_filter=obs_filter,
            id=id,
            title=title_parts,
            footnote=footnote,
            source=source,
            output_file=output_file,
            population_columns=population_columns,
            observation_columns=observation_columns,
            sort_columns=sort_columns,
            col_rel_width=col_rel_width,
            group_by=group_by,
            page_by=page_by,
        )

        rtf_files.append(rtf_path)

    return rtf_files
