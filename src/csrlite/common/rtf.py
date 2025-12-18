# pyre-strict
from typing import Any

import polars as pl
from rtflite import RTFBody, RTFColumnHeader, RTFDocument, RTFFootnote, RTFPage, RTFSource, RTFTitle


def create_rtf_table_n_pct(
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

    Args:
        df: Polars DataFrame containing the table data.
        col_header_1: List of strings for the first header row.
        col_header_2: Optional list of strings for the second header row.
        col_widths: Optional list of relative column widths. Defaults to equal widths.
        title: Title string or list of title strings.
        footnote: Footnote string or list of footnote strings.
        source: Source string or list of source strings.
        borders_2: Whether to show borders for the second header row. Defaults to True.
        orientation: Page orientation, "landscape" or "portrait". Defaults to "landscape".

    Returns:
        RTFDocument object.
    """
    n_cols = len(df.columns)

    # Calculate column widths if None - simple default
    if col_widths is None:
        col_widths = [1.0] * n_cols

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


def create_rtf_listing(
    df: pl.DataFrame,
    col_header: list[str],
    col_widths: list[float] | None,
    title: list[str] | str,
    footnote: list[str] | str | None,
    source: list[str] | str | None,
    orientation: str = "landscape",
) -> RTFDocument:
    """
    Create a standardized RTF listing document.
    """
    n_cols = len(df.columns)

    # Calculate column widths if None
    if col_widths is None:
        col_widths = [1.0] * n_cols

    # Normalize metadata
    title_list = [title] if isinstance(title, str) else title
    footnote_list = [footnote] if isinstance(footnote, str) else (footnote or [])
    source_list = [source] if isinstance(source, str) else (source or [])

    headers = [
        RTFColumnHeader(
            text=col_header,
            col_rel_width=col_widths,
            text_justification=["l"] * n_cols,
        )
    ]

    rtf_components: dict[str, Any] = {
        "df": df,
        "rtf_page": RTFPage(orientation=orientation),
        "rtf_title": RTFTitle(text=title_list),
        "rtf_column_header": headers,
        "rtf_body": RTFBody(
            col_rel_width=col_widths,
            text_justification=["l"] * n_cols,
            border_left=["single"] * n_cols,
        ),
    }

    if footnote_list:
        rtf_components["rtf_footnote"] = RTFFootnote(text=footnote_list)

    if source_list:
        rtf_components["rtf_source"] = RTFSource(text=source_list)

    return RTFDocument(**rtf_components)
