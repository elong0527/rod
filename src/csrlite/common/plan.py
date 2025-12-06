# pyre-strict
"""
Clean, simple TLF plan system.
This module provides a straightforward implementation for clinical TLF generation
using YAML plans with template inheritance and keyword resolution.
"""

import itertools
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import polars as pl

from .yaml_loader import YamlInheritanceLoader


@dataclass
class Keyword:
    """Base keyword definition."""

    name: str
    label: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Population(Keyword):
    """Population definition with filter."""

    filter: str = ""


@dataclass
class Observation(Keyword):
    """Observation/timepoint definition with filter."""

    filter: str = ""


@dataclass
class Parameter(Keyword):
    """Parameter definition with filter.

    The terms field supports dynamic title generation:
    - terms.before: "serious" → "Serious Adverse Events"
    - terms.after: "resulting in death" → "Adverse Events Resulting in Death"
    """

    filter: str = ""
    terms: Optional[Dict[str, str]] = None
    indent: int = 0  # Indentation level for hierarchical display


@dataclass
class Group(Keyword):
    """Treatment group definition."""

    variable: str = ""
    level: List[str] = field(default_factory=list)
    group_label: List[str] = field(default_factory=list)


@dataclass
class DataSource:
    """Data source definition."""

    name: str
    path: str
    dataframe: Optional[pl.DataFrame] = None


@dataclass
class AnalysisPlan:
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


class KeywordRegistry:
    """Registry for managing keywords."""

    def __init__(self) -> None:
        self.populations: Dict[str, Population] = {}
        self.observations: Dict[str, Observation] = {}
        self.parameters: Dict[str, Parameter] = {}
        self.groups: Dict[str, Group] = {}
        self.data_sources: Dict[str, DataSource] = {}

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load keywords from a dictionary."""
        self._load_keyword_type(data, "population", Population, self.populations)
        self._load_keyword_type(data, "observation", Observation, self.observations)
        self._load_keyword_type(data, "parameter", Parameter, self.parameters)
        self._load_keyword_type(data, "group", Group, self.groups)
        self._load_keyword_type(data, "data", DataSource, self.data_sources)

    def _load_keyword_type(
        self, data: Dict[str, Any], key: str, keyword_class: Any, target_dict: Dict[str, Any]
    ) -> None:
        """Generic method to load a type of keyword."""
        for item_data in data.get(key, []):
            if keyword_class == Group and "group_label" not in item_data:
                item_data["group_label"] = item_data.get("label", [])

            expected_fields = {f.name for f in fields(keyword_class) if f.init}
            filtered_data = {k: v for k, v in item_data.items() if k in expected_fields}

            instance = keyword_class(**filtered_data)
            target_dict[instance.name] = instance

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
                print(f"Successfully loaded dataset '{name}' from '{path}'")
            except Exception as e:
                print(
                    f"Warning: Could not load dataset '{name}' from '{data_source.path}'. "
                    f"Reason: {e}"
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
        print("ADaM Metadata:")

        if (df := self.get_dataset_df()) is not None:
            print("\nData Sources:")
            print(df)

        if (df := self.get_population_df()) is not None:
            print("\nAnalysis Population Type:")
            print(df)

        if (df := self.get_observation_df()) is not None:
            print("\nAnalysis Observation Type:")
            print(df)

        if (df := self.get_parameter_df()) is not None:
            print("\nAnalysis Parameter Type:")
            print(df)

        if (df := self.get_group_df()) is not None:
            print("\nAnalysis Groups:")
            print(df)

        if (df := self.get_plan_df()) is not None:
            print("\nAnalysis Plans:")
            print(df)

    def __str__(self) -> str:
        study_name = self.study_data.get("study", Dict[str, Any]()).get("name", "Unknown")
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
