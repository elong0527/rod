# pyre-strict
"""
Medical History (MH) Listing Analysis Functions
"""

from pathlib import Path

import polars as pl

from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_listing
from ..common.utils import apply_common_filters


def mh_listing(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None = "SAFFL = 'Y'",
    observation_filter: str | None = "MHOCCUR = 'Y'",
    id: tuple[str, str] = ("USUBJID", "Subject ID"),
    title: list[str] | None = None,
    footnote: list[str] | None = None,
    source: list[str] | None = None,
    output_file: str = "mh_listing.rtf",
    population_columns: list[tuple[str, str]] | None = None,
    observation_columns: list[tuple[str, str]] | None = None,
    sort_columns: list[str] | None = None,
) -> str:
    """
    Generate Medical History Listing.
    """
    if title is None:
        title = ["Listing of Medical History"]

    # Generate DF
    df = mh_listing_df(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        id_col=id[0],
        pop_cols=population_columns,
        obs_cols=observation_columns,
        sort_cols=sort_columns,
    )

    # Generate RTF
    mh_listing_rtf(df=df, output_path=output_file, title=title, footnote=footnote, source=source)

    return output_file


def mh_listing_df(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    id_col: str,
    pop_cols: list[tuple[str, str]] | None,
    obs_cols: list[tuple[str, str]] | None,
    sort_cols: list[str] | None,
) -> pl.DataFrame:
    # Defaults
    if pop_cols is None:
        # Default interesting cols from ADSL
        pop_cols = [("TRT01A", "Treatment"), ("AGE", "Age"), ("SEX", "Sex")]

    if obs_cols is None:
        # Default from ADMH
        obs_cols = [
            ("MHSEQ", "Seq"),
            ("MHBODSYS", "System Organ Class"),
            ("MHDECOD", "Preferred Term"),
            ("MHSTDTC", "Start Date"),
            ("MHENRTPT", "Status"),
        ]

    # Apply filters
    adsl, adq = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
    )

    if adq is None:
        raise ValueError("Observation data is missing")

    # Join
    # Select specific columns from ADSL
    pop_col_names = [c[0] for c in pop_cols]
    # Ensure ID is there
    if id_col not in pop_col_names:
        pop_col_names = [id_col] + pop_col_names

    adsl_sub = adsl.select(pop_col_names)

    joined = adq.join(adsl_sub, on=id_col, how="inner")

    # Sort
    if sort_cols:
        # Check if cols exist
        valid_sorts = [c for c in sort_cols if c in joined.columns]
        if valid_sorts:
            joined = joined.sort(valid_sorts)

    # Select display columns (id + pop + obs)
    display_cols = [id_col] + [c[0] for c in pop_cols if c[0] != id_col] + [c[0] for c in obs_cols]
    final_df = joined.select([c for c in display_cols if c in joined.columns])

    # Rename for display?
    # Usually listing keeps raw names or we Map them.
    # The create_rtf_listing function takes col_header list.

    return final_df


def mh_listing_rtf(
    df: pl.DataFrame,
    output_path: str,
    title: list[str] | str,
    footnote: list[str] | None,
    source: list[str] | None,
) -> None:
    if df.is_empty():
        return

    # Generate headers from predefined mapping or current logic?
    # Here we just use column names for simplicity or we could pass headers.
    # We didn't output headers from mh_listing_df.
    # Let's assume the order is maintained.

    headers = list(df.columns)

    # Approximate widths
    # ID: 1, TRT: 1.5, AGE: 0.5, SEX: 0.5, SEQ: 0.5, SOC: 2, PT: 2, DATE: 1, STATUS: 1
    # Total ~ 10 units?
    # Simple uniform distribution or weighted?
    n_cols = len(headers)
    col_widths = [1.0] * n_cols

    rtf_doc = create_rtf_listing(
        df=df,
        col_header=headers,
        col_widths=col_widths,
        title=title,
        footnote=footnote,
        source=source,
    )

    rtf_doc.write_rtf(output_path)


def study_plan_to_mh_listing(study_plan: StudyPlan) -> list[str]:
    """
    Batch generate MH listings.
    """
    analysis_type = "mh_listing"
    output_dir = study_plan.output_dir

    parser = StudyPlanParser(study_plan)

    plans = study_plan.study_data.get("plans", [])
    all_specs = []
    for plan_data in plans:
        expanded = study_plan.expander.expand_plan(plan_data)
        for p in expanded:
            all_specs.append(study_plan.expander.create_analysis_spec(p))

    plan_df = pl.DataFrame(all_specs)

    if "analysis" in plan_df.columns:
        mh_plans = plan_df.filter(pl.col("analysis") == analysis_type)
    else:
        mh_plans = pl.DataFrame()

    generated_files = []

    for analysis in mh_plans.iter_rows(named=True):
        pop_name = analysis.get("population", "enrolled")

        try:
            # Load Population
            adsl, _ = parser.get_population_data(pop_name, "trt01a")  # dummy group

            (admh,) = parser.get_datasets("admh")

            filename = f"{analysis_type}_{pop_name}.rtf".lower()
            output_path = f"{output_dir}/{filename}"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            mh_listing(
                population=adsl,
                observation=admh,
                population_filter=None,
                observation_filter=None,  # Show all?
                output_file=output_path,
                title=["Listing of Medical History", f"({pop_name} Population)"],
                source=["Source: ADSL, ADMH"],
            )

            generated_files.append(output_path)

        except Exception as e:
            print(f"Error generating MH listing: {e}")
            continue

    return generated_files
