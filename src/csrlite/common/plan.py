# pyre-strict
"""
Clean, simple TLF plan system.
This module provides a straightforward implementation for clinical TLF generation
using YAML plans with template inheritance and keyword resolution.
"""

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .yaml_loader import YamlInheritanceLoader

logger: logging.Logger = logging.getLogger(__name__)


class Keyword(BaseModel):
    """Base keyword definition."""

    name: str
    label: Optional[str] = None
    description: Optional[str] = None


class Population(Keyword):
    """Population definition with filter."""

    filter: str = ""


class Observation(Keyword):
    """Observation/timepoint definition with filter."""

    filter: str = ""


class Parameter(Keyword):
    """Parameter definition with filter."""

    filter: str = ""
    terms: Optional[Dict[str, str]] = None
    indent: int = 0


class Group(Keyword):
    """Treatment group definition."""

    variable: str = ""
    level: List[str] = Field(default_factory=list)
    group_label: List[str] = Field(default_factory=list)

    # Allow label to be excluded if it conflicts or handled manually

    # pyre-ignore[56]
    @field_validator("group_label", mode="before")
    @classmethod
    def set_group_label(cls, v: Any, info: Any) -> Any:
        # If group_label is missing, fallback to 'label' field if present in input data
        # Note: Pydantic V2 validation context doesn't easily give access to other fields input
        # unless using model_validator. But here we can rely on standard defaulting or
        # fix it at the registry level like before.
        # Actually, let's keep it simple: if not provided, it's empty.
        # The original code did:
        # if "group_label" not in item_data: item_data["group_label"] = item_data.get("label", [])
        return v or []


class DataSource(BaseModel):
    """Data source definition."""

    name: str
    path: str
    dataframe: Optional[pl.DataFrame] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnalysisPlan(BaseModel):
    """Individual analysis plan specification."""

    analysis: str
    population: str
    observation: Optional[str] = None
    group: Optional[str] = None
    parameter: Optional[str] = None

    @property
    def id(self) -> str:
        """Generate unique analysis ID."""
        parts = [self.analysis, self.population]
        if self.observation:
            parts.append(self.observation)
        if self.parameter:
            parts.append(self.parameter)
        return "_".join(parts)


class KeywordRegistry(BaseModel):
    """Registry for managing keywords."""

    populations: Dict[str, Population] = Field(default_factory=dict)
    observations: Dict[str, Observation] = Field(default_factory=dict)
    parameters: Dict[str, Parameter] = Field(default_factory=dict)
    groups: Dict[str, Group] = Field(default_factory=dict)
    data_sources: Dict[str, DataSource] = Field(default_factory=dict)

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load keywords from a dictionary."""
        # We manually load so we can handle the dict-to-list-of-models transformation
        # and the specific logic for defaults.

        for item in data.get("population", []):
            pop_item = Population(**item)
            self.populations[pop_item.name] = pop_item

        for item in data.get("observation", []):
            obs_item = Observation(**item)
            self.observations[obs_item.name] = obs_item

        for item in data.get("parameter", []):
            param_item = Parameter(**item)
            self.parameters[param_item.name] = param_item

        for item in data.get("group", []):
            # Special handling for Group where 'label' might be a list (for group_label)
            # but Keyword.label expects a string.
            if "label" in item and isinstance(item["label"], list):
                if "group_label" not in item:
                    item["group_label"] = item["label"]
                # Remove label from item to avoid validation error on Keyword.label
                # or set it to a joined string if a label is really needed
                del item["label"]

            group_item = Group(**item)
            self.groups[group_item.name] = group_item

        for item in data.get("data", []):
            ds_item = DataSource(**item)
            self.data_sources[ds_item.name] = ds_item

    def get_population(self, name: str) -> Optional[Population]:
        return self.populations.get(name)

    def get_observation(self, name: str) -> Optional[Observation]:
        return self.observations.get(name)

    def get_parameter(self, name: str) -> Optional[Parameter]:
        return self.parameters.get(name)

    def get_group(self, name: str) -> Optional[Group]:
        return self.groups.get(name)

    def get_data_source(self, name: str) -> Optional[DataSource]:
        return self.data_sources.get(name)


class PlanExpander:
    """Expands condensed plans into individual analysis specifications."""

    def __init__(self, keywords: KeywordRegistry) -> None:
        self.keywords = keywords

    def expand_plan(self, plan_data: Dict[str, Any]) -> List[AnalysisPlan]:
        """Expand a single condensed plan into individual plans."""
        analysis = plan_data["analysis"]
        populations = self._to_list(plan_data.get("population", []))
        observations: List[Any] = self._to_list(plan_data.get("observation")) or [None]
        parameters: List[Any] = self._parse_parameters(plan_data.get("parameter")) or [None]
        group = plan_data.get("group")

        expanded_plans = [
            AnalysisPlan(
                analysis=analysis, population=pop, observation=obs, group=group, parameter=param
            )
            for pop, obs, param in itertools.product(populations, observations, parameters)
        ]
        return expanded_plans

    def create_analysis_spec(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """Create a summary analysis specification with keywords."""
        spec = {
            "analysis": plan.analysis,
            "population": plan.population,
            "observation": plan.observation,
            "parameter": plan.parameter,
            "group": plan.group,
        }
        return spec

    def _to_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def _parse_parameters(self, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]  # Keep semicolon-separated values as single parameter
        return list(value)

    def _generate_title(self, plan: AnalysisPlan) -> str:
        parts = [plan.analysis.replace("_", " ").title()]
        if (pop := self.keywords.get_population(plan.population)) and pop.label:
            parts.append(f"- {pop.label}")
        if plan.observation:
            obs = self.keywords.get_observation(plan.observation)
            if obs and obs.label:
                parts.append(f"- {obs.label}")
        if plan.parameter:
            param = self.keywords.get_parameter(plan.parameter)
            if param and param.label:
                parts.append(f"- {param.label}")
        return " ".join(parts)


class StudyPlan:
    """Main study plan."""

    def __init__(self, study_data: Dict[str, Any], base_path: Optional[Path] = None) -> None:
        self.study_data = study_data
        self.base_path: Path = base_path or Path(".")
        self.datasets: Dict[str, pl.DataFrame] = {}
        self.keywords = KeywordRegistry()
        self.expander = PlanExpander(self.keywords)
        self.keywords.load_from_dict(self.study_data)
        self.load_datasets()

    @property
    def output_dir(self) -> str:
        """Get output directory from study configuration."""
        study_config = self.study_data.get("study", {})
        return cast(str, study_config.get("output", "."))

    def load_datasets(self) -> None:
        """Load datasets from paths specified in data_sources."""
        for name, data_source in self.keywords.data_sources.items():
            try:
                # Ensure the path is relative to the base_path of the plan
                path = self.base_path / data_source.path
                df = pl.read_parquet(path)
                self.datasets[name] = df
                data_source.dataframe = df
                logger.info(f"Successfully loaded dataset '{name}' from '{path}'")
            except Exception as e:
                logger.warning(
                    f"Could not load dataset '{name}' from '{data_source.path}'. Reason: {e}"
                )

    def get_plan_df(self) -> pl.DataFrame:
        """Expand all condensed plans into a DataFrame of detailed specifications."""
        all_specs = [
            self.expander.create_analysis_spec(plan)
            for plan_data in self.study_data.get("plans", [])
            for plan in self.expander.expand_plan(plan_data)
        ]
        return pl.DataFrame(all_specs)

    def get_dataset_df(self) -> Optional[pl.DataFrame]:
        """Get a DataFrame of data sources."""
        if not self.keywords.data_sources:
            return None
        return pl.DataFrame(
            [
                {"name": name, "path": ds.path, "loaded": name in self.datasets}
                for name, ds in self.keywords.data_sources.items()
            ]
        )

    def get_population_df(self) -> Optional[pl.DataFrame]:
        """Get a DataFrame of analysis populations."""
        if not self.keywords.populations:
            return None
        return pl.DataFrame(
            [
                {"name": name, "label": pop.label, "filter": pop.filter}
                for name, pop in self.keywords.populations.items()
            ]
        )

    def get_observation_df(self) -> Optional[pl.DataFrame]:
        """Get a DataFrame of analysis observations."""
        if not self.keywords.observations:
            return None
        return pl.DataFrame(
            [
                {"name": name, "label": obs.label, "filter": obs.filter}
                for name, obs in self.keywords.observations.items()
            ]
        )

    def get_parameter_df(self) -> Optional[pl.DataFrame]:
        """Get a DataFrame of analysis parameters."""
        if not self.keywords.parameters:
            return None
        return pl.DataFrame(
            [
                {"name": name, "label": param.label, "filter": param.filter}
                for name, param in self.keywords.parameters.items()
            ]
        )

    def get_group_df(self) -> Optional[pl.DataFrame]:
        """Get a DataFrame of analysis groups."""
        if not self.keywords.groups:
            return None
        return pl.DataFrame(
            [
                {
                    "name": name,
                    "variable": group.variable,
                    "levels": str(group.level),
                    "labels": str(group.group_label),
                }
                for name, group in self.keywords.groups.items()
            ]
        )

    def print(self) -> None:
        """Print comprehensive study plan information using Polars DataFrames."""
        logger.info("ADaM Metadata:")

        if (df := self.get_dataset_df()) is not None:
            logger.info(f"\nData Sources:\n{df}")

        if (df := self.get_population_df()) is not None:
            logger.info(f"\nAnalysis Population Type:\n{df}")

        if (df := self.get_observation_df()) is not None:
            logger.info(f"\nAnalysis Observation Type:\n{df}")

        if (df := self.get_parameter_df()) is not None:
            logger.info(f"\nAnalysis Parameter Type:\n{df}")

        if (df := self.get_group_df()) is not None:
            logger.info(f"\nAnalysis Groups:\n{df}")

        if (df := self.get_plan_df()) is not None:
            logger.info(f"\nAnalysis Plans:\n{df}")

    def __str__(self) -> str:
        study_name = self.study_data.get("study", {}).get("name", "Unknown")
        condensed_plans = len(self.study_data.get("plans", []))
        individual_analyses = len(self.get_plan_df())
        return (
            f"StudyPlan(study='{study_name}', plans={condensed_plans}, "
            f"analyses={individual_analyses})"
        )


def load_plan(plan_path: str) -> StudyPlan:
    """
    Loads a study plan from a YAML file, resolving template inheritance.
    """
    path = Path(plan_path)
    base_path = path.parent
    loader = YamlInheritanceLoader(base_path)
    study_data = loader.load(path.name)
    return StudyPlan(study_data, base_path)
