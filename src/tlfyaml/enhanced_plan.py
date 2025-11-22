"""
Enhanced plan system with template support and keyword resolution.

This module handles the new design where plans include:
- Template references for inheritance
- Keyword definitions within the plan file
- Data source mappings
- Enhanced metadata with labels, filters, and descriptions
"""

import yaml
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import itertools
from pydantic import BaseModel, Field, field_validator, ConfigDict


class KeywordDefinition(BaseModel):
    """Base class for keyword definitions."""
    name: str = Field(..., description="Keyword identifier")
    label: Optional[str] = Field(None, description="Display label")
    description: Optional[str] = Field(None, description="Description")


class PopulationKeyword(KeywordDefinition):
    """Population keyword definition."""
    filter: str = Field(..., description="Filter expression for population")


class ObservationKeyword(KeywordDefinition):
    """Observation keyword definition."""
    filter: str = Field(..., description="Filter expression for observation")


class GroupKeyword(KeywordDefinition):
    """Group/treatment keyword definition."""
    model_config = ConfigDict(populate_by_name=True)

    variable: str = Field(..., description="Variable name for grouping")
    level: Optional[List[str]] = Field(None, description="Possible values")
    group_label: Optional[List[str]] = Field(None, description="Display labels for levels", alias="label")


class ParameterKeyword(KeywordDefinition):
    """Parameter keyword definition."""
    filter: str = Field(..., description="Filter expression for parameter")


class DataSource(BaseModel):
    """Data source definition."""
    name: str = Field(..., description="Data source identifier")
    path: str = Field(..., description="Path to data file")


class EnhancedPlan(BaseModel):
    """Enhanced plan specification with keyword support."""
    analysis: str = Field(..., description="Analysis type")
    population: Union[str, List[str]] = Field(..., description="Population identifier(s)")
    observation: Optional[Union[str, List[str]]] = Field(None, description="Observation identifier(s)")
    group: Optional[str] = Field(None, description="Grouping variable")
    parameter: Optional[Union[str, List[str]]] = Field(None, description="Parameter identifier(s)")

    @field_validator('population')
    @classmethod
    def normalize_population(cls, v):
        """Normalize population to list format."""
        return [v] if isinstance(v, str) else v

    @field_validator('observation')
    @classmethod
    def normalize_observation(cls, v):
        """Normalize observation to list format."""
        if v is None:
            return None
        return [v] if isinstance(v, str) else v

    @field_validator('parameter')
    @classmethod
    def parse_parameter(cls, v):
        """Parse parameter - handle semicolon-separated strings and lists."""
        if v is None:
            return None
        if isinstance(v, str):
            return [p.strip() for p in v.split(';')]
        return v

    def expand(self, keyword_resolver: 'KeywordResolver') -> List[Dict[str, Any]]:
        """Expand this plan into individual analysis specifications."""
        expanded = []

        populations = self.population
        observations = self.observation or [None]
        parameters = self.parameter or [None]

        for pop, obs, param in itertools.product(populations, observations, parameters):
            # Resolve keywords to get full definitions
            population_def = keyword_resolver.resolve_population(pop)
            observation_def = keyword_resolver.resolve_observation(obs) if obs else None
            parameter_def = keyword_resolver.resolve_parameter(param) if param else None
            group_def = keyword_resolver.resolve_group(self.group) if self.group else None

            spec = {
                'id': self._generate_id(pop, obs, param),
                'analysis': self.analysis,
                'population': {
                    'keyword': pop,
                    'label': population_def.label if population_def else pop,
                    'filter': population_def.filter if population_def else None
                },
                'title': self._generate_title(population_def, observation_def, parameter_def)
            }

            if obs and observation_def:
                spec['observation'] = {
                    'keyword': obs,
                    'label': observation_def.label,
                    'filter': observation_def.filter
                }

            if param and parameter_def:
                spec['parameter'] = {
                    'keyword': param,
                    'label': parameter_def.label,
                    'filter': parameter_def.filter
                }

            if self.group and group_def:
                spec['group'] = {
                    'keyword': self.group,
                    'variable': group_def.variable,
                    'levels': group_def.level,
                    'labels': group_def.group_label
                }

            expanded.append(spec)

        return expanded

    def _generate_id(self, population: str, observation: Optional[str], parameter: Optional[str]) -> str:
        """Generate unique ID for analysis."""
        parts = [self.analysis, population]
        if observation:
            parts.append(observation)
        if parameter:
            parts.append(parameter)
        return '_'.join(parts)

    def _generate_title(self, population_def: Optional[PopulationKeyword],
                       observation_def: Optional[ObservationKeyword],
                       parameter_def: Optional[ParameterKeyword]) -> str:
        """Generate title from keyword definitions."""
        parts = []

        # Add analysis type (capitalized)
        parts.append(self.analysis.replace('_', ' ').title())

        # Add population
        if population_def and population_def.label:
            parts.append(f"- {population_def.label}")

        # Add observation
        if observation_def and observation_def.label:
            parts.append(f"- {observation_def.label}")

        # Add parameter
        if parameter_def and parameter_def.label:
            parts.append(f"- {parameter_def.label}")

        return ' '.join(parts)


class KeywordResolver:
    """Resolves keywords from plan file definitions."""

    def __init__(self, populations: List[PopulationKeyword],
                 observations: List[ObservationKeyword],
                 parameters: List[ParameterKeyword],
                 groups: List[GroupKeyword]):
        self.populations = {p.name: p for p in populations}
        self.observations = {o.name: o for o in observations}
        self.parameters = {p.name: p for p in parameters}
        self.groups = {g.name: g for g in groups}

    def resolve_population(self, keyword: str) -> Optional[PopulationKeyword]:
        """Resolve population keyword."""
        return self.populations.get(keyword)

    def resolve_observation(self, keyword: str) -> Optional[ObservationKeyword]:
        """Resolve observation keyword."""
        return self.observations.get(keyword)

    def resolve_parameter(self, keyword: str) -> Optional[ParameterKeyword]:
        """Resolve parameter keyword."""
        return self.parameters.get(keyword)

    def resolve_group(self, keyword: str) -> Optional[GroupKeyword]:
        """Resolve group keyword."""
        return self.groups.get(keyword)


class EnhancedStudyPlan(BaseModel):
    """Enhanced study plan with templates and keyword definitions."""
    study: Dict[str, Any] = Field(..., description="Study metadata")
    plans: List[EnhancedPlan] = Field(..., description="Analysis plans")

    # Keyword definitions
    population: Optional[List[PopulationKeyword]] = Field(None, description="Population definitions")
    observation: Optional[List[ObservationKeyword]] = Field(None, description="Observation definitions")
    parameter: Optional[List[ParameterKeyword]] = Field(None, description="Parameter definitions")
    group: Optional[List[GroupKeyword]] = Field(None, description="Group definitions")
    data: Optional[List[DataSource]] = Field(None, description="Data source definitions")

    def create_keyword_resolver(self) -> KeywordResolver:
        """Create keyword resolver from definitions."""
        return KeywordResolver(
            populations=self.population or [],
            observations=self.observation or [],
            parameters=self.parameter or [],
            groups=self.group or []
        )

    def expand_all(self) -> List[Dict[str, Any]]:
        """Expand all plans into individual specifications."""
        resolver = self.create_keyword_resolver()

        all_expanded = []
        for plan in self.plans:
            expanded = plan.expand(resolver)
            all_expanded.extend(expanded)
        return all_expanded

    def summary(self) -> Dict[str, Any]:
        """Generate summary of plan expansion."""
        expanded = self.expand_all()

        # Process data sources
        data_sources = {}
        if self.data:
            data_sources = {ds.name: ds.path for ds in self.data}

        return {
            'study': self.study,
            'data_sources': data_sources,
            'keyword_counts': {
                'populations': len(self.population or []),
                'observations': len(self.observation or []),
                'parameters': len(self.parameter or []),
                'groups': len(self.group or [])
            },
            'condensed_plans': len(self.plans),
            'individual_analyses': len(expanded),
            'analyses': expanded
        }


class EnhancedPlanLoader:
    """Enhanced loader for plan YAML files with template support."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

    def load_templates(self, template_list: List[str]) -> Dict[str, Any]:
        """Load and merge template files."""
        merged_template = {}

        for template_file in template_list:
            template_path = self.base_path / template_file
            if template_path.exists():
                with open(template_path, 'r') as f:
                    template_data = yaml.safe_load(f)
                    # Merge template data (simple merge for now)
                    merged_template = self._deep_merge(merged_template, template_data)
            else:
                print(f"Warning: Template file not found: {template_path}")

        return merged_template

    def load(self, plan_file: str) -> EnhancedStudyPlan:
        """Load and validate a plan YAML file with template resolution."""
        file_path = self.base_path / plan_file

        if not file_path.exists():
            raise FileNotFoundError(f"Plan file not found: {file_path}")

        with open(file_path, 'r') as f:
            plan_data = yaml.safe_load(f)

        # Handle template inheritance
        if 'template' in plan_data.get('study', {}):
            templates = plan_data['study']['template']
            if isinstance(templates, str):
                templates = [templates]

            template_data = self.load_templates(templates)

            # Merge template data with plan data (plan data takes precedence)
            merged_data = self._deep_merge(template_data, plan_data)
        else:
            merged_data = plan_data

        return EnhancedStudyPlan(**merged_data)

    def expand(self, plan_file: str) -> Dict[str, Any]:
        """Load plan file and return expansion summary."""
        study_plan = self.load(plan_file)
        return study_plan.summary()

    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


def demonstrate_enhanced_system():
    """Demonstrate the enhanced plan system."""
    print("Enhanced TLF Plan System with Templates")
    print("=" * 50)

    loader = EnhancedPlanLoader('examples/yaml')

    try:
        summary = loader.expand('plan_xyz123.yaml')

        print(f"📊 Study: {summary['study']['name']}")
        print(f"   Title: {summary['study']['title']}")
        print(f"   Templates: {summary['study'].get('template', 'None')}")
        print()

        print(f"🔍 Keywords defined:")
        for keyword_type, count in summary['keyword_counts'].items():
            print(f"   • {keyword_type}: {count}")
        print()

        print(f"📈 Plan expansion:")
        print(f"   • Condensed plans: {summary['condensed_plans']}")
        print(f"   • Individual analyses: {summary['individual_analyses']}")
        print()

        print("📋 Sample analyses:")
        for i, analysis in enumerate(summary['analyses'][:5], 1):
            print(f"   {i:2d}. {analysis['id']}")
            print(f"       Title: {analysis['title']}")
            if 'population' in analysis:
                print(f"       Population: {analysis['population']['label']} ({analysis['population']['keyword']})")
            if 'observation' in analysis:
                print(f"       Observation: {analysis['observation']['label']} ({analysis['observation']['keyword']})")

        if len(summary['analyses']) > 5:
            print(f"       ... and {len(summary['analyses']) - 5} more")

        print()
        print("✅ Enhanced System Benefits:")
        print("   • Template inheritance support")
        print("   • Self-contained keyword definitions")
        print("   • Rich metadata with labels and filters")
        print("   • Scalable across organizations")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_enhanced_system()