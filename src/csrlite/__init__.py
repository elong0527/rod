import logging
import sys

from .ae.ae_listing import (  # AE listing functions
    ae_listing,
    study_plan_to_ae_listing,
)
from .ae.ae_specific import (  # AE specific functions
    ae_specific,
    study_plan_to_ae_specific,
)
from .ae.ae_summary import (  # AE summary functions
    ae_summary,
    study_plan_to_ae_summary,
)
from .cm.cm_listing import (  # CM listing functions
    cm_listing,
    study_plan_to_cm_listing,
)
from .cm.cm_summary import (
    cm_summary,
    study_plan_to_cm_summary,
)
from .common.config import config
from .common.count import (
    count_subject,
    count_subject_with_observation,
)
from .common.parse import (
    StudyPlanParser,
    parse_filter_to_sql,
)
from .common.plan import (  # Core classes
    load_plan,
)
from .disposition.disposition import study_plan_to_disposition_summary
from .ie.ie_listing import (
    ie_listing_df,
    ie_listing_rtf,
    study_plan_to_ie_listing,
)
from .ie.ie_summary import (
    ie_ard,
    ie_df,
    ie_rtf,
    study_plan_to_ie_summary,
)
from .mh.mh_listing import (
    mh_listing,
    study_plan_to_mh_listing,
)
from .mh.mh_summary import (
    mh_summary,
    study_plan_to_mh_summary,
)
from .pd.pd_listing import (
    pd_listing,
    study_plan_to_pd_listing,
)

# Configure logging
logging.basicConfig(
    level=config.logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("csrlite")

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
    # CM analysis
    "cm_listing",
    "study_plan_to_cm_listing",
    "cm_summary",
    "study_plan_to_cm_summary",
    # Disposition analysis
    "study_plan_to_disposition_summary",
    # Count functions
    "count_subject",
    "count_subject_with_observation",
    # Parse utilities
    "StudyPlanParser",
    "parse_filter_to_sql",
    # IE analysis
    "ie_ard",
    "ie_df",
    "ie_rtf",
    "study_plan_to_ie_summary",
    "ie_listing_df",
    "ie_listing_rtf",
    "study_plan_to_ie_listing",
    # PD analysis
    "pd_listing",
    "study_plan_to_pd_listing",
    # MH analysis
    "mh_listing",
    "study_plan_to_mh_listing",
    "mh_summary",
    "study_plan_to_mh_summary",
]
