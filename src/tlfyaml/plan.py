"""
Clean, simple TLF plan system.

This module provides a straightforward implementation for clinical TLF generation
using YAML plans with template inheritance and keyword resolution.
"""

import yaml
from typing import Any
from pathlib import Path
import itertools
from dataclasses import dataclass, field


@dataclass
class Keyword:
    """Base keyword definition."""
    name: str
    label: str | None = None
    description: str | None = None


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
    """Parameter definition with filter."""
    filter: str = ""


@dataclass
class Group(Keyword):
    """Treatment group definition."""
    variable: str = ""
    level: list[str] = field(default_factory=list)
    group_label: list[str] = field(default_factory=list)


@dataclass
class DataSource:
    """Data source definition."""
    name: str
    path: str


@dataclass
class AnalysisPlan:
    """Individual analysis plan specification."""
    analysis: str
    population: str
    observation: str | None = None
    group: str | None = None
    parameter: str | None = None

    @property
    def id(self) -> str:
        """Generate unique analysis ID."""
        parts = [self.analysis, self.population]
        if self.observation:
            parts.append(self.observation)
        if self.parameter:
            parts.append(self.parameter)
        return '_'.join(parts)


class KeywordRegistry:
    """Registry for managing keywords with template inheritance."""

    def __init__(self):
        self.populations: dict[str, Population] = {}
        self.observations: dict[str, Observation] = {}
        self.parameters: dict[str, Parameter] = {}
        self.groups: dict[str, Group] = {}
        self.data_sources: dict[str, DataSource] = {}

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load keywords from dictionary data with inheritance support."""
        # Load populations (study-specific can override template)
        for pop_data in data.get('population', []):
            # Check if population already exists (from template)
            existing_pop = self.populations.get(pop_data['name'])

            pop = Population(
                name=pop_data['name'],
                # Use study values if provided, otherwise keep template values
                label=pop_data.get('label') or (existing_pop.label if existing_pop else None),
                description=pop_data.get('description') or (existing_pop.description if existing_pop else None),
                filter=pop_data.get('filter') or (existing_pop.filter if existing_pop else '')
            )
            # Override/add population
            self.populations[pop.name] = pop

        # Load observations (study-specific can override template)
        for obs_data in data.get('observation', []):
            # Check if observation already exists (from template)
            existing_obs = self.observations.get(obs_data['name'])

            obs = Observation(
                name=obs_data['name'],
                # Use study values if provided, otherwise keep template values
                label=obs_data.get('label') or (existing_obs.label if existing_obs else None),
                description=obs_data.get('description') or (existing_obs.description if existing_obs else None),
                filter=obs_data.get('filter') or (existing_obs.filter if existing_obs else '')
            )
            # Override/add observation
            self.observations[obs.name] = obs

        # Load parameters (study-specific can override template)
        for param_data in data.get('parameter', []):
            # Check if parameter already exists (from template)
            existing_param = self.parameters.get(param_data['name'])

            param = Parameter(
                name=param_data['name'],
                # Use study label if provided, otherwise keep template label
                label=param_data.get('label') or (existing_param.label if existing_param else None),
                # Use study description if provided, otherwise keep template description
                description=param_data.get('description') or (existing_param.description if existing_param else None),
                # Use study filter if provided, otherwise keep template filter
                filter=param_data.get('filter') or (existing_param.filter if existing_param else '')
            )
            # Override/add parameter
            self.parameters[param.name] = param

        # Load groups (study-specific can override template)
        for group_data in data.get('group', []):
            group = Group(
                name=group_data['name'],
                label=group_data.get('label'),
                description=group_data.get('description'),
                variable=group_data.get('variable', ''),
                level=group_data.get('level', []),
                group_label=group_data.get('group_label', group_data.get('label', []))
            )
            # Override any existing group with same name
            self.groups[group.name] = group

        # Load data sources (study-specific can override template)
        for data_src in data.get('data', []):
            ds = DataSource(
                name=data_src['name'],
                path=data_src['path']
            )
            # Override any existing data source with same name
            self.data_sources[ds.name] = ds

    def get_population(self, name: str) -> Population | None:
        """Get population by name."""
        return self.populations.get(name)

    def get_observation(self, name: str) -> Observation | None:
        """Get observation by name."""
        return self.observations.get(name)

    def get_parameter(self, name: str) -> Parameter | None:
        """Get parameter by name."""
        return self.parameters.get(name)

    def get_group(self, name: str) -> Group | None:
        """Get group by name."""
        return self.groups.get(name)

    def get_data_source(self, name: str) -> DataSource | None:
        """Get data source by name."""
        return self.data_sources.get(name)


class PlanExpander:
    """Expands condensed plans into individual analysis specifications."""

    def __init__(self, keywords: KeywordRegistry):
        self.keywords = keywords

    def expand_plan(self, plan_data: dict[str, Any]) -> list[AnalysisPlan]:
        """Expand a single condensed plan into individual plans."""
        analysis = plan_data['analysis']

        # Normalize inputs to lists
        populations = self._to_list(plan_data.get('population', []))
        observations = self._to_list(plan_data.get('observation')) or [None]
        parameters = self._parse_parameters(plan_data.get('parameter')) or [None]
        group = plan_data.get('group')

        # Generate Cartesian product
        expanded_plans = []
        for pop, obs, param in itertools.product(populations, observations, parameters):
            plan = AnalysisPlan(
                analysis=analysis,
                population=pop,
                observation=obs,
                group=group,
                parameter=param
            )
            expanded_plans.append(plan)

        return expanded_plans

    def create_analysis_spec(self, plan: AnalysisPlan) -> dict[str, Any]:
        """Create detailed analysis specification with resolved keywords."""
        spec = {
            'id': plan.id,
            'analysis': plan.analysis,
            'title': self._generate_title(plan)
        }

        # Add population info
        pop = self.keywords.get_population(plan.population)
        if pop:
            spec['population'] = {
                'keyword': plan.population,
                'label': pop.label or plan.population,
                'filter': pop.filter
            }

        # Add observation info
        if plan.observation:
            obs = self.keywords.get_observation(plan.observation)
            if obs:
                spec['observation'] = {
                    'keyword': plan.observation,
                    'label': obs.label or plan.observation,
                    'filter': obs.filter
                }

        # Add parameter info
        if plan.parameter:
            param = self.keywords.get_parameter(plan.parameter)
            if param:
                spec['parameter'] = {
                    'keyword': plan.parameter,
                    'label': param.label or plan.parameter,
                    'filter': param.filter
                }

        # Add group info
        if plan.group:
            group = self.keywords.get_group(plan.group)
            if group:
                spec['group'] = {
                    'keyword': plan.group,
                    'variable': group.variable,
                    'levels': group.level,
                    'labels': group.group_label
                }

        return spec

    def _to_list(self, value: Any) -> list[str]:
        """Convert value to list."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def _parse_parameters(self, value: Any) -> list[str] | None:
        """Parse parameter field (handles semicolon-separated)."""
        if value is None:
            return None
        if isinstance(value, str):
            return [p.strip() for p in value.split(';')]
        return list(value)

    def _generate_title(self, plan: AnalysisPlan) -> str:
        """Generate analysis title."""
        parts = [plan.analysis.replace('_', ' ').title()]

        pop = self.keywords.get_population(plan.population)
        if pop and pop.label:
            parts.append(f"- {pop.label}")

        if plan.observation:
            obs = self.keywords.get_observation(plan.observation)
            if obs and obs.label:
                parts.append(f"- {obs.label}")

        if plan.parameter:
            param = self.keywords.get_parameter(plan.parameter)
            if param and param.label:
                parts.append(f"- {param.label}")

        return ' '.join(parts)


class StudyPlan:
    """Main study plan with template inheritance."""

    def __init__(self, study_data: dict[str, Any], base_path: Path = None):
        self.study_data = study_data
        self.base_path = base_path or Path('.')

        # Initialize components
        self.keywords = KeywordRegistry()
        self.expander = PlanExpander(self.keywords)

        # Load templates and study data
        self._load_templates()
        self._load_study_keywords()

    def _load_templates(self) -> None:
        """Load template files if specified."""
        templates = self.study_data.get('study', {}).get('template', [])
        if isinstance(templates, str):
            templates = [templates]

        # Load templates in order (earlier templates can be overridden by later ones)
        for template_file in templates:
            template_path = self.base_path / template_file
            if template_path.exists():
                with open(template_path, 'r') as f:
                    template_data = yaml.safe_load(f) or {}
                    # Load template keywords first
                    self.keywords.load_from_dict(template_data)
            else:
                print(f"Warning: Template file not found: {template_path}")

    def _load_study_keywords(self) -> None:
        """Load study-specific keyword definitions."""
        self.keywords.load_from_dict(self.study_data)

    def expand_all_plans(self) -> list[dict[str, Any]]:
        """Expand all condensed plans into detailed specifications."""
        all_specs = []

        for plan_data in self.study_data.get('plans', []):
            # Expand to individual plans
            expanded_plans = self.expander.expand_plan(plan_data)

            # Create detailed specifications
            for plan in expanded_plans:
                spec = self.expander.create_analysis_spec(plan)
                all_specs.append(spec)

        return all_specs

    def get_summary(self) -> dict[str, Any]:
        """Get summary of study plan."""
        expanded = self.expand_all_plans()

        return {
            'study': self.study_data.get('study', {}),
            'templates': self.study_data.get('study', {}).get('template', []),
            'keyword_counts': {
                'populations': len(self.keywords.populations),
                'observations': len(self.keywords.observations),
                'parameters': len(self.keywords.parameters),
                'groups': len(self.keywords.groups),
                'data_sources': len(self.keywords.data_sources)
            },
            'condensed_plans': len(self.study_data.get('plans', [])),
            'individual_analyses': len(expanded),
            'analyses': expanded
        }


class PlanLoader:
    """Simple loader for study plans."""

    def __init__(self, plan_path: str):
        """Initialize loader with direct path to plan file."""
        self.plan_path = Path(plan_path)
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Plan file not found: {self.plan_path}")

        # Load and create study plan immediately
        with open(self.plan_path, 'r') as f:
            study_data = yaml.safe_load(f)

        self.study_plan = StudyPlan(study_data, self.plan_path.parent)

    def expand(self) -> dict[str, Any]:
        """Return expanded plan summary."""
        return self.study_plan.get_summary()

    def load(self) -> StudyPlan:
        """Return the loaded study plan."""
        return self.study_plan


