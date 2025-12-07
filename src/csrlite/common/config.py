# pyre-strict
"""
Central configuration for csrlite.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class CsrLiteConfig(BaseModel):
    """
    Global configuration for csrlite library.
    """

    # Column Name Defaults
    id_col: str = Field(default="USUBJID", description="Subject Identifier Column")
    group_col: Optional[str] = Field(default=None, description="Treatment Group Column")
    
    # Missing Value Handling
    missing_str: str = Field(default="__missing__", description="String to represent missing string values")
    
    # Logging
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Default logging level"
    )

    model_config = ConfigDict(validate_assignment=True)


# Global configuration instance
config = CsrLiteConfig()
