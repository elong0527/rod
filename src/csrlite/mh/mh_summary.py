# pyre-strict
"""
Medical History (MH) Summary Analysis Functions
"""

from pathlib import Path
from typing import Any

import polars as pl

from ..common.parse import StudyPlanParser
from ..common.plan import StudyPlan
from ..common.rtf import create_rtf_table_n_pct
from ..common.utils import apply_common_filters


def mh_summary(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None = "SAFFL = 'Y'",
    observation_filter: str | None = "MHOCCUR = 'Y'",
    id: tuple[str, str] = ("USUBJID", "Subject ID"),
    group: tuple[str, str] = ("TRT01A", "Treatment"),
    variables: list[tuple[str, str]] | None = None,
    title: list[str] | None = None,
    footnote: list[str] | None = None,
    source: list[str] | None = None,
    output_file: str = "mh_summary.rtf",
) -> str:
    """
    Generate Medical History Summary Table.
    """
    if title is None:
        title = ["Summary of Medical History by Body System and Preferred Term"]

    if variables is None:
        # Default hierarchy: Body System -> Preferred Term
        variables = [("MHBODSYS", "System Organ Class"), ("MHDECOD", "Preferred Term")]

    # Generate ARD
    ard = mh_summary_ard(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
        group_col=group[0],
        id_col=id[0],
        variables=variables,
    )

    # Transform to Display DF
    df = mh_summary_df(ard)

    # Generate RTF
    mh_summary_rtf(df=df, output_path=output_file, title=title, footnote=footnote, source=source)

    return output_file


def mh_summary_ard(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    group_col: str,
    id_col: str,
    variables: list[tuple[str, str]],
) -> pl.DataFrame:
    """
    Generate ARD for MH Summary.
    Hierarchy is often Body System -> Preferred Term.
    """

    # Apply filters
    adsl, adq = apply_common_filters(
        population=population,
        observation=observation,
        population_filter=population_filter,
        observation_filter=observation_filter,
    )

    if adq is None:
        # Should not happen as we passed observation df
        raise ValueError("Observation data is missing")

    # This summary usually nests MHDECOD under MHBODSYS
    # Structure:
    # Any Medical History (1=1)
    #   Body System 1
    #     Term A
    #     Term B

    # We can reuse count_subject_with_observation but it handles list of flexible conditions.
    # For nested structure, we might need manual construction or nested calls.

    # Let's assume standard 2-level nesting: MHBODSYS -> MHDECOD
    # Check if variables match this pattern

    # Identify the hierarchy columns
    # If standard usage: variables=[("MHBODSYS", "SOC"), ("MHDECOD", "PT")]

    # We will build a list of (filter_expr, label, indent_level, is_header)

    specs = []

    # 1. Overall "Any Medical History"
    specs.append(
        {"filter": pl.lit(True), "label": "Any Medical History", "indent": 0, "is_header": False}
    )

    # Get distinct Body Systems
    bodsys_list = adq.select("MHBODSYS").unique().sort("MHBODSYS").to_series().to_list()

    for sys in bodsys_list:
        if sys is None:
            continue

        # Add Body System Row
        specs.append(
            {
                "filter": pl.col("MHBODSYS") == sys,
                "label": sys,
                "indent": 1,
                "is_header": False,  # It has counts
            }
        )

        # Get distinct Terms within this System
        terms = (
            adq.filter(pl.col("MHBODSYS") == sys)
            .select("MHDECOD")
            .unique()
            .sort("MHDECOD")
            .to_series()
            .to_list()
        )

        for term in terms:
            if term is None:
                continue
            specs.append(
                {
                    "filter": (pl.col("MHBODSYS") == sys) & (pl.col("MHDECOD") == term),
                    "label": term,
                    "indent": 2,
                    "is_header": False,
                }
            )

    # Now calculate counts for each spec
    results = []

    # Get total population counts by group
    pop_counts = adsl.group_by(group_col).count().sort(group_col)
    groups = pop_counts.select(group_col).to_series().to_list()

    # Pre-calculate totals map
    pop_totals = {row[group_col]: row["count"] for row in pop_counts.iter_rows(named=True)}

    # Helper to calculate row
    def calc_row(spec: dict[str, Any]) -> dict[str, Any]:
        row_res = {"label": spec["label"], "indent": spec["indent"], "is_header": spec["is_header"]}

        # Filter observation data based on spec string/expr
        # Note: count_subject_with_observation logic handles join.
        # We can simulate logic here.

        # 1. Filter ADQ based on criteria
        filtered_obs = adq.filter(spec["filter"])

        # 2. Join with ADSL to get groups (inner join to count only subjects in population)
        # But we already filtered ADSL (population).

        subset = filtered_obs.join(adsl.select([id_col, group_col]), on=id_col, how="inner")

        # 3. Group by Group Col
        counts = subset.select(id_col, group_col).unique().group_by(group_col).count()
        counts_map = {row[group_col]: row["count"] for row in counts.iter_rows(named=True)}

        for g in groups:
            n = counts_map.get(g, 0)
            denom = pop_totals.get(g, 0)
            pct = (n / denom * 100.0) if denom > 0 else 0.0
            row_res[f"count_{g}"] = n
            row_res[f"pct_{g}"] = pct

        return row_res

    for spec in specs:
        results.append(calc_row(spec))

    return pl.DataFrame(results)


def mh_summary_df(ard: pl.DataFrame) -> pl.DataFrame:
    """
    Transform ARD to Display DataFrame.
    """
    if ard.is_empty():
        return pl.DataFrame()

    # Identify group columns
    cols = ard.columns
    group_cols = [c for c in cols if c.startswith("count_")]
    groups = [c.replace("count_", "") for c in group_cols]

    select_exprs = [
        (pl.lit("   ").repeat_by(pl.col("indent")).list.join("") + pl.col("label")).alias(
            "Medical History"
        )
    ]

    for g in groups:
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


def mh_summary_rtf(
    df: pl.DataFrame,
    output_path: str,
    title: list[str] | str,
    footnote: list[str] | None,
    source: list[str] | None,
) -> None:
    """
    Generate RTF document.
    """
    if df.is_empty():
        # Handle empty case?
        return

    n_cols = len(df.columns)
    col_width_first = 2.5
    remaining_width = 7.0  # Approx page width
    col_width_others = remaining_width / (n_cols - 1)
    col_widths = [col_width_first] + [col_width_others] * (n_cols - 1)

    col_header_1 = list(df.columns)
    col_header_2 = [""] + ["n (%)"] * (n_cols - 1)

    rtf_doc = create_rtf_table_n_pct(
        df=df,
        col_header_1=col_header_1,
        col_header_2=col_header_2,
        col_widths=col_widths,
        title=title,
        footnote=footnote,
        source=source,
    )

    rtf_doc.write_rtf(output_path)


def study_plan_to_mh_summary(study_plan: StudyPlan) -> list[str]:
    """
    Batch generate MH summaries from study plan.
    """
    analysis_type = "mh_summary"
    output_dir = study_plan.output_dir

    # Initialize parser
    parser = StudyPlanParser(study_plan)

    # Get plans
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
        group_kw = analysis.get("group", "trt01a")  # specific key?

        try:
            # Load Population
            adsl, group_col = parser.get_population_data(pop_name, group_kw)

            # Load MH Data
            # Note: Assuming 'admh' is the dataset name
            (admh,) = parser.get_datasets("admh")

            filename = f"{analysis_type}_{pop_name}_{group_kw}.rtf".lower()
            output_path = f"{output_dir}/{filename}"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            mh_summary(
                population=adsl,
                observation=admh,
                population_filter=None,  # Already filtered by parser
                observation_filter="MHOCCUR = 'Y'",
                group=(group_col, group_col),  # Use actual col name
                output_file=output_path,
                title=[
                    "Summary of Medical History by System Organ Class and Preferred Term",
                    f"({pop_name} Population)",
                ],
                source=["Source: ADSL, ADMH"],
            )

            generated_files.append(output_path)

        except Exception as e:
            print(f"Error generating MH summary: {e}")
            continue

    return generated_files
