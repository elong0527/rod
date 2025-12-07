from typing import Any


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
