import logging
import sys

from .common.config import config

# Configure logging
logging.basicConfig(
    level=config.logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("csrlite")

from .ae.ae_listing import (
    # AE listing functions
    ae_listing,
    study_plan_to_ae_listing,
)
from .ae.ae_specific import (
    # AE specific functions
    ae_specific,
    study_plan_to_ae_specific,
)
from .ae.ae_summary import (
    # AE summary functions
    ae_summary,
    study_plan_to_ae_summary,
)
from .common.count import (
    count_subject,
    count_subject_with_observation,
)
from .common.parse import (
    StudyPlanParser,
    parse_filter_to_sql,
)
from .common.plan import (
    # Core classes
    load_plan,
)
from .disposition.disposition import study_plan_to_disposition_summary

# Main exports for common usage
__all__ = [
    # Primary user interface
    "load_plan",
    # AE analysis (direct pipeline wrappers)
    "ae_summary",
    "ae_specific",
    "ae_listing",
    # AE analysis (StudyPlan integration)
    "study_plan_to_ae_summary",
    "study_plan_to_ae_specific",
    "study_plan_to_ae_listing",
    # Disposition analysis
    "study_plan_to_disposition_summary",
    # Count functions
    "count_subject",
    "count_subject_with_observation",
    # Parse utilities
    "StudyPlanParser",
    "parse_filter_to_sql",
]
