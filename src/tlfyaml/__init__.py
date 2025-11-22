"""
TLF YAML Framework for Clinical Trial Reporting

A simple, clean framework for generating Tables, Listings, and Figures (TLFs)
for clinical trial regulatory submissions using YAML configuration files with
template inheritance and keyword-driven analysis specifications.

Key Features:
- Template inheritance (organization -> TA -> study)
- Keyword-driven analysis specifications
- Metalite R package pattern compatibility
- SQL-like filtering for platform agnosticism
- Plan expansion via Cartesian products
- Self-contained YAML configurations

Example Usage:
    from tlfyaml import PlanLoader

    loader = PlanLoader('config/yaml/study_abc123.yaml')
    summary = loader.expand()

    print(f"Study: {summary['study']['name']}")
    print(f"Analyses: {summary['individual_analyses']}")
"""

from .plan import (
    # Core classes
    PlanLoader,
    StudyPlan,
    KeywordRegistry,
    PlanExpander,

    # Data structures
    Keyword,
    Population,
    Observation,
    Parameter,
    Group,
    DataSource,
    AnalysisPlan,
)

__version__ = "0.1.0"
__author__ = "TLF YAML Framework Contributors"

# Main exports for common usage
__all__ = [
    # Primary user interface
    "PlanLoader",
    "StudyPlan",

    # Core functionality
    "KeywordRegistry",
    "PlanExpander",

    # Data models
    "Keyword",
    "Population",
    "Observation",
    "Parameter",
    "Group",
    "DataSource",
    "AnalysisPlan",
]

# Convenience function for quick expansion
def expand_plan(plan_path: str) -> dict:
    """Quick function to load and expand a plan file."""
    loader = PlanLoader(plan_path)
    return loader.expand()