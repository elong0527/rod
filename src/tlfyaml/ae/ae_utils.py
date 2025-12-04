# pyre-strict
from typing import Any

import polars as pl
from rtflite import RTFBody, RTFColumnHeader, RTFDocument, RTFFootnote, RTFPage, RTFSource, RTFTitle


def get_ae_parameter_title(param: Any, prefix: str = "Participants With") -> str:
    """
    Extract title from parameter for ae_* title generation.

    Args:
        param: Parameter object with terms attribute
        prefix: Prefix for the title (e.g. "Participants With", "Listing of Participants With")

    Returns:
        Title string for the analysis
    """
    default_suffix = "Adverse Events"

    if not param:
        return f"{prefix} {default_suffix}"

    # Check for terms attribute
    if hasattr(param, "terms") and param.terms and isinstance(param.terms, dict):
        terms = param.terms

        # Preprocess to empty strings (avoiding None)
        before = terms.get("before", "").title()
        after = terms.get("after", "").title()

        # Build title and clean up extra spaces
        title = f"{prefix} {before} {default_suffix} {after}"
        return " ".join(title.split())  # Remove extra spaces

    # Fallback to default
    return f"{prefix} {default_suffix}"


def get_ae_parameter_row_labels(param: Any) -> tuple[str, str]:
    """
    Generate n_with and n_without row labels based on parameter terms.

    Returns:
        Tuple of (n_with_label, n_without_label)
    """
    # Default labels
    default_with = "    with one or more adverse events"
    default_without = "    with no adverse events"

    if not param or not hasattr(param, "terms") or not param.terms:
        return (default_with, default_without)

    terms = param.terms
    before = terms.get("before", "").lower()
    after = terms.get("after", "").lower()

    # Build dynamic labels with leading indentation
    with_label = f"with one or more {before} adverse events {after}"
    without_label = f"with no {before} adverse events {after}"

    # Clean up extra spaces and add back the 4-space indentation
    with_label = "    " + " ".join(with_label.split())
    without_label = "    " + " ".join(without_label.split())

    return (with_label, without_label)


def create_ae_rtf_table(
    df: pl.DataFrame,
    col_header_1: list[str],
    col_header_2: list[str] | None,
    col_widths: list[float] | None,
    title: list[str] | str,
    footnote: list[str] | str | None,
    source: list[str] | str | None,
    borders_2: bool = True,
    orientation: str = "landscape",
) -> RTFDocument:
    """
    Create a standardized RTF table document with 1 or 2 header rows.
    """
    n_cols = len(df.columns)

    # Calculate column widths if None - simple default
    if col_widths is None:
        col_widths = [1] * n_cols

    # Normalize metadata
    title_list = [title] if isinstance(title, str) else title
    footnote_list = [footnote] if isinstance(footnote, str) else (footnote or [])
    source_list = [source] if isinstance(source, str) else (source or [])

    headers = [
        RTFColumnHeader(
            text=col_header_1,
            col_rel_width=col_widths,
            text_justification=["l"] + ["c"] * (n_cols - 1),
        )
    ]

    if col_header_2:
        h2_kwargs = {
            "text": col_header_2,
            "col_rel_width": col_widths,
            "text_justification": ["l"] + ["c"] * (n_cols - 1),
        }
        if borders_2:
            h2_kwargs["border_left"] = ["single"]
            h2_kwargs["border_top"] = [""]

        headers.append(RTFColumnHeader(**h2_kwargs))

    rtf_components: dict[str, Any] = {
        "df": df,
        "rtf_page": RTFPage(orientation=orientation),
        "rtf_title": RTFTitle(text=title_list),
        "rtf_column_header": headers,
        "rtf_body": RTFBody(
            col_rel_width=col_widths,
            text_justification=["l"] + ["c"] * (n_cols - 1),
            border_left=["single"] * n_cols,
        ),
    }

    if footnote_list:
        rtf_components["rtf_footnote"] = RTFFootnote(text=footnote_list)

    if source_list:
        rtf_components["rtf_source"] = RTFSource(text=source_list)

    return RTFDocument(**rtf_components)
