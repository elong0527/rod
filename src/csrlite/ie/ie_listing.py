# pyre-strict
"""
Inclusion/Exclusion (IE) Listing Analysis Functions
"""

from pathlib import Path


import polars as pl

from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_listing
from ..common.utils import apply_common_filters


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

    if "analysis" in plan_df.columns:
        listing_plans = plan_df.filter(pl.col("analysis") == analysis_type)
    else:
        listing_plans = pl.DataFrame()

    generated_files = []

    # If listing_plans is empty, create a dummy row to force generation
    if listing_plans.height == 0:
        listing_plans = pl.DataFrame([{"population": "enrolled", "analysis": analysis_type}])

    for analysis in listing_plans.iter_rows(named=True):
        # Load ADSL
        pop_name = analysis.get("population", "enrolled")

        try:
            (adsl_raw,) = parser.get_datasets("adsl")
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


def ie_listing_rtf(df: pl.DataFrame, output_path: str, title: str | list[str] = "") -> None:
    """Generate RTF Listing."""
    col_widths = [1.5, 3.5]  # Approximate ratio

    rtf_doc = create_rtf_listing(
        df=df,
        col_header=list(df.columns),
        col_widths=col_widths,
        title=title,
        footnote=[],
        source=[],
    )

    rtf_doc.write_rtf(output_path)
