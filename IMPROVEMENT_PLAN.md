# MVP Improvement Plan

## Executive Summary

This plan addresses critical gaps in the current MVP to make it production-ready for clinical trial analysis. The improvements focus on robustness, testability, and usability while maintaining the core YAML-driven Review-Oriented Development paradigm.

## Priority 1: Critical Foundation (Weeks 1-2)

### 1.1 Custom Exception Hierarchy

**Current Gap**: No custom exceptions; generic Python errors make debugging difficult.

**Implementation**:
```python
# src/tlfyaml/exceptions.py
class TLFYAMLError(Exception):
    """Base exception for tlfyaml package."""
    pass

class YAMLValidationError(TLFYAMLError):
    """Raised when YAML structure is invalid."""
    def __init__(self, file_path: str, message: str, line_number: int = None):
        self.file_path = file_path
        self.line_number = line_number
        super().__init__(f"{file_path}:{line_number or '?'} - {message}")

class TemplateNotFoundError(TLFYAMLError):
    """Raised when a template file cannot be found."""
    pass

class FilterSyntaxError(TLFYAMLError):
    """Raised when filter syntax is invalid."""
    def __init__(self, filter_str: str, position: int, expected: str):
        self.filter_str = filter_str
        self.position = position
        super().__init__(
            f"Invalid filter syntax at position {position}: '{filter_str}'\n"
            f"Expected: {expected}"
        )

class KeywordNotFoundError(TLFYAMLError):
    """Raised when a referenced keyword doesn't exist."""
    def __init__(self, keyword_type: str, keyword_name: str):
        super().__init__(f"{keyword_type} keyword '{keyword_name}' not found in registry")

class DatasetNotFoundError(TLFYAMLError):
    """Raised when a dataset cannot be loaded."""
    pass
```

**Impact**: Better error messages for users, easier debugging, regulatory audit trail.

### 1.2 Fix Missing Methods in StudyPlan

**Current Gap**: `get_summary()` referenced but not implemented; `print()` calls non-existent methods.

**Implementation**:
```python
# In src/tlfyaml/plan.py - StudyPlan class

def get_summary(self) -> Dict[str, Any]:
    """Get comprehensive study plan summary."""
    return {
        'study': self.study_data.get('study', {}),
        'condensed_plans': len(self.study_data.get('plans', [])),
        'individual_analyses': len(self.get_plan_df()),
        'populations': len(self.keywords.populations),
        'observations': len(self.keywords.observations),
        'parameters': len(self.keywords.parameters),
        'groups': len(self.keywords.groups),
        'datasets': len(self.keywords.data_sources),
        'templates': self.study_data.get('study', {}).get('template', [])
    }

def print(self) -> None:
    """Print comprehensive study plan information using Polars DataFrames."""
    summary = self.get_summary()
    print(f"\n{'='*60}")
    print(f"Study Plan: {summary['study'].get('name', 'Unknown')}")
    print(f"{'='*60}")

    if summary['templates']:
        print(f"\nTemplates: {', '.join(summary['templates'])}")

    if (df := self.get_dataset_df()) is not None:
        print("\n📊 Data Sources:")
        print(df)

    if (df := self.get_population_df()) is not None:
        print("\n👥 Analysis Populations:")
        print(df)

    if (df := self.get_observation_df()) is not None:
        print("\n📅 Observation Periods:")
        print(df)

    if (df := self.get_parameter_df()) is not None:
        print("\n📋 Parameters:")
        print(df)

    if (df := self.get_group_df()) is not None:
        print("\n🔗 Groups:")
        print(df)

    if (df := self.get_plan_df()) is not None:
        print("\n🎯 Analysis Plans:")
        print(f"Total individual analyses: {len(df)}")
        print(df.head(10))
        if len(df) > 10:
            print(f"... and {len(df) - 10} more")

    print(f"\n{'='*60}\n")
```

**Impact**: Working demo functionality, better user experience.

### 1.3 Testing Infrastructure

**Current Gap**: No tests directory, no test files, 0% test coverage.

**Implementation**:

Create `tests/` structure:
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── fixtures/                # Test YAML files and data
│   ├── minimal_template.yaml
│   ├── minimal_study.yaml
│   ├── invalid_yaml/
│   └── test_data/
├── test_yaml_loader.py      # Template inheritance tests
├── test_plan.py             # Plan expansion tests
├── test_keywords.py         # Keyword registry tests
├── test_filters.py          # Filter execution tests
└── integration/
    └── test_end_to_end.py   # Full workflow tests
```

**Sample conftest.py**:
```python
import pytest
from pathlib import Path
import polars as pl

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def minimal_template_yaml(tmp_path):
    """Create minimal organization template."""
    template = tmp_path / "org.yaml"
    template.write_text("""
population:
  - name: apat
    label: "All Participants as Treated"
  - name: itt
    label: "Intention-to-Treat"

parameter:
  - name: any
    label: "Any Adverse Event"
""")
    return template

@pytest.fixture
def sample_adsl(tmp_path):
    """Create sample ADSL dataset."""
    df = pl.DataFrame({
        'USUBJID': ['001', '002', '003'],
        'SAFFL': ['Y', 'Y', 'N'],
        'ITTFL': ['Y', 'Y', 'Y'],
        'TRT01A': ['Drug A', 'Placebo', 'Drug A']
    })
    path = tmp_path / "adsl.parquet"
    df.write_parquet(path)
    return path, df
```

**Key Test Cases**:
1. Template inheritance (simple, multi-level, field-level merging)
2. Keyword loading and retrieval
3. Plan expansion (semicolon vs list parameters)
4. Filter syntax validation
5. Dataset loading and error handling
6. YAML validation errors

**Impact**: Confidence in refactoring, regression prevention, documentation through examples.

## Priority 2: Core Functionality (Weeks 3-4)

### 2.1 Filter Parser and Executor

**Current Gap**: Filters are defined as strings but never executed. No way to actually filter data.

**Implementation**:
```python
# src/tlfyaml/filters.py
import re
from typing import Any, Dict, Tuple
import polars as pl
from .exceptions import FilterSyntaxError

class FilterParser:
    """Parse SQL-like filter syntax into Polars expressions."""

    # Grammar: dataset:column operator value [and/or dataset:column operator value]
    FILTER_PATTERN = re.compile(
        r'(\w+):(\w+)\s*(==|!=|>|<|>=|<=|in|not in)\s*(.+?)(?:\s+(and|or)\s+|$)',
        re.IGNORECASE
    )

    def parse(self, filter_str: str) -> pl.Expr:
        """
        Parse filter string into Polars expression.

        Examples:
            "adsl:saffl == 'Y'" -> pl.col('saffl') == 'Y'
            "adae:aeser == 'Y' and adae:aerel in ['RELATED']" -> combined expression
        """
        if not filter_str or not filter_str.strip():
            return pl.lit(True)  # No filter = include all

        matches = list(self.FILTER_PATTERN.finditer(filter_str))
        if not matches:
            raise FilterSyntaxError(filter_str, 0, "dataset:column operator value")

        expressions = []
        combinators = []

        for match in matches:
            dataset, column, operator, value, combinator = match.groups()
            expr = self._create_expression(column, operator, value, filter_str)
            expressions.append(expr)
            if combinator:
                combinators.append(combinator.lower())

        # Combine expressions with and/or
        result = expressions[0]
        for i, expr in enumerate(expressions[1:]):
            if i < len(combinators):
                if combinators[i] == 'and':
                    result = result & expr
                elif combinators[i] == 'or':
                    result = result | expr

        return result

    def _create_expression(self, column: str, operator: str, value_str: str,
                          filter_str: str) -> pl.Expr:
        """Create Polars expression for a single condition."""
        col = pl.col(column)
        value = self._parse_value(value_str.strip())

        operator = operator.lower()
        if operator == '==':
            return col == value
        elif operator == '!=':
            return col != value
        elif operator == '>':
            return col > value
        elif operator == '<':
            return col < value
        elif operator == '>=':
            return col >= value
        elif operator == '<=':
            return col <= value
        elif operator == 'in':
            if not isinstance(value, list):
                raise FilterSyntaxError(filter_str, 0, "list for 'in' operator")
            return col.is_in(value)
        elif operator == 'not in':
            if not isinstance(value, list):
                raise FilterSyntaxError(filter_str, 0, "list for 'not in' operator")
            return ~col.is_in(value)
        else:
            raise FilterSyntaxError(filter_str, 0, f"unknown operator '{operator}'")

    def _parse_value(self, value_str: str) -> Any:
        """Parse value string into Python object."""
        value_str = value_str.strip()

        # List: ['a', 'b']
        if value_str.startswith('[') and value_str.endswith(']'):
            # Simple list parser - could use ast.literal_eval for production
            items = value_str[1:-1].split(',')
            return [self._parse_scalar(item.strip()) for item in items if item.strip()]

        return self._parse_scalar(value_str)

    def _parse_scalar(self, value_str: str) -> Any:
        """Parse scalar value."""
        value_str = value_str.strip()

        # String: 'value' or "value"
        if (value_str.startswith("'") and value_str.endswith("'")) or \
           (value_str.startswith('"') and value_str.endswith('"')):
            return value_str[1:-1]

        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # Null
        if value_str.lower() in ('null', 'none'):
            return None

        # Unquoted string (treat as literal)
        return value_str


class FilterExecutor:
    """Execute filters on datasets."""

    def __init__(self, datasets: Dict[str, pl.DataFrame]):
        self.datasets = datasets
        self.parser = FilterParser()

    def apply_filter(self, filter_str: str, target_dataset: str = None) -> pl.DataFrame:
        """
        Apply filter and return filtered DataFrame.

        Args:
            filter_str: Filter expression like "adsl:saffl == 'Y'"
            target_dataset: If provided, apply filter to this dataset
                           If None, infer from filter string

        Returns:
            Filtered DataFrame
        """
        if not filter_str or not filter_str.strip():
            # No filter - return target dataset unfiltered
            if target_dataset and target_dataset in self.datasets:
                return self.datasets[target_dataset]
            raise ValueError("No filter provided and no target dataset specified")

        # Parse to get dataset references and expression
        dataset_name = self._extract_dataset_name(filter_str)
        if target_dataset and target_dataset != dataset_name:
            raise ValueError(
                f"Filter references dataset '{dataset_name}' but target is '{target_dataset}'"
            )

        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_name}' referenced in filter not found. "
                f"Available: {list(self.datasets.keys())}"
            )

        df = self.datasets[dataset_name]
        expr = self.parser.parse(filter_str)

        return df.filter(expr)

    def _extract_dataset_name(self, filter_str: str) -> str:
        """Extract dataset name from filter string."""
        match = re.match(r'(\w+):', filter_str)
        if not match:
            raise FilterSyntaxError(filter_str, 0, "dataset:column format")
        return match.group(1)
```

**Usage in StudyPlan**:
```python
from .filters import FilterExecutor

class StudyPlan:
    def __init__(self, ...):
        # ... existing code ...
        self.filter_executor = FilterExecutor(self.datasets)

    def get_filtered_population(self, population_name: str) -> pl.DataFrame:
        """Get filtered dataset for a population."""
        pop = self.keywords.get_population(population_name)
        if not pop:
            raise KeywordNotFoundError('population', population_name)

        return self.filter_executor.apply_filter(pop.filter, 'adsl')

    def get_filtered_observations(self, observation_name: str,
                                   dataset_name: str = 'adae') -> pl.DataFrame:
        """Get filtered observations."""
        obs = self.keywords.get_observation(observation_name)
        if not obs:
            raise KeywordNotFoundError('observation', observation_name)

        return self.filter_executor.apply_filter(obs.filter, dataset_name)
```

**Impact**: Core functionality working, can actually filter data, enables TLF generation.

### 2.2 Pydantic Models for YAML Validation

**Current Gap**: No schema validation for YAML structure; errors caught too late.

**Implementation**:
```python
# src/tlfyaml/models.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Any

class KeywordModel(BaseModel):
    """Base model for keywords."""
    name: str = Field(..., pattern=r'^[a-z][a-z0-9_]*$',
                     description="Keyword name (lowercase, alphanumeric + underscore)")
    label: Optional[str] = Field(None, description="Human-readable label")
    description: Optional[str] = Field(None, description="Detailed description")

class PopulationModel(KeywordModel):
    """Population keyword model."""
    filter: str = Field(default="", description="Filter expression for population")

    @field_validator('filter')
    @classmethod
    def validate_filter_syntax(cls, v: str) -> str:
        """Validate filter syntax (basic check)."""
        if v and ':' not in v:
            raise ValueError("Filter must include dataset reference (e.g., 'adsl:saffl')")
        return v

class ObservationModel(KeywordModel):
    """Observation keyword model."""
    filter: str = Field(default="", description="Filter expression for observation period")

class ParameterModel(KeywordModel):
    """Parameter keyword model."""
    filter: str = Field(default="", description="Filter expression for parameter")

class GroupModel(KeywordModel):
    """Group keyword model."""
    variable: str = Field(default="", description="Grouping variable (dataset:column)")
    level: List[str] = Field(default_factory=list, description="Group levels")
    group_label: List[str] = Field(default_factory=list, description="Group labels")

class DataSourceModel(BaseModel):
    """Data source model."""
    name: str = Field(..., pattern=r'^[a-z][a-z0-9_]*$')
    path: str = Field(..., description="Path to dataset file")

class AnalysisPlanModel(BaseModel):
    """Analysis plan model."""
    analysis: str = Field(..., description="Analysis function name")
    population: Union[str, List[str]] = Field(..., description="Population keyword(s)")
    observation: Optional[Union[str, List[str]]] = Field(None)
    parameter: Optional[Union[str, List[str]]] = Field(None)
    group: Optional[str] = Field(None)

class StudyModel(BaseModel):
    """Study configuration model."""
    name: str = Field(..., description="Study identifier")
    title: Optional[str] = Field(None)
    template: Optional[Union[str, List[str]]] = Field(None, description="Template file(s)")

class StudyPlanYAMLModel(BaseModel):
    """Complete study plan YAML model."""
    study: StudyModel
    plans: List[AnalysisPlanModel] = Field(default_factory=list)
    population: List[PopulationModel] = Field(default_factory=list)
    observation: List[ObservationModel] = Field(default_factory=list)
    parameter: List[ParameterModel] = Field(default_factory=list)
    group: List[GroupModel] = Field(default_factory=list)
    data: List[DataSourceModel] = Field(default_factory=list)

    model_config = {
        'extra': 'forbid'  # Fail on unknown fields
    }
```

**Integration**:
```python
# In yaml_loader.py
from .models import StudyPlanYAMLModel
from .exceptions import YAMLValidationError

def load(self, file_name: str) -> Dict[str, Any]:
    """Load and validate YAML file."""
    # ... existing loading code ...

    # Validate with Pydantic
    try:
        validated = StudyPlanYAMLModel(**data)
        return validated.model_dump()
    except ValidationError as e:
        raise YAMLValidationError(
            file_path=str(file_path),
            message=f"YAML validation failed: {e}",
            line_number=None
        )
```

**Impact**: Catch errors early, better error messages, enforce standards.

## Priority 3: Enhanced Usability (Weeks 5-6)

### 3.1 CLI Interface

**Current Gap**: No command-line tool; users must write Python scripts.

**Implementation**:
```python
# src/tlfyaml/cli.py
import click
import polars as pl
from pathlib import Path
from .plan import load_plan
from .exceptions import TLFYAMLError

@click.group()
@click.version_option()
def cli():
    """TLF YAML Framework CLI for clinical trial analysis planning."""
    pass

@cli.command()
@click.argument('plan_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for plan DataFrame')
@click.option('--format', type=click.Choice(['csv', 'json', 'parquet']), default='csv')
def expand(plan_file, output, format):
    """Expand analysis plan and output individual analyses."""
    try:
        study_plan = load_plan(plan_file)
        df = study_plan.get_plan_df()

        if output:
            if format == 'csv':
                df.write_csv(output)
            elif format == 'json':
                df.write_json(output)
            elif format == 'parquet':
                df.write_parquet(output)
            click.echo(f"✓ Plan expanded to {output}")
        else:
            click.echo(df)

    except TLFYAMLError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('plan_file', type=click.Path(exists=True))
def validate(plan_file):
    """Validate YAML plan file."""
    try:
        study_plan = load_plan(plan_file)
        summary = study_plan.get_summary()

        click.echo("✓ YAML structure valid")
        click.echo(f"✓ Study: {summary['study'].get('name')}")
        click.echo(f"✓ Templates: {len(summary.get('templates', []))}")
        click.echo(f"✓ Analyses: {summary['individual_analyses']}")
        click.echo(f"✓ Keywords: {summary['populations']} populations, "
                  f"{summary['parameters']} parameters")

        # Validate datasets load
        for name in study_plan.datasets:
            click.echo(f"✓ Dataset '{name}' loaded successfully")

    except TLFYAMLError as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('plan_file', type=click.Path(exists=True))
def info(plan_file):
    """Display detailed plan information."""
    try:
        study_plan = load_plan(plan_file)
        study_plan.print()
    except TLFYAMLError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('plan_file', type=click.Path(exists=True))
@click.argument('population')
def preview_population(plan_file, population):
    """Preview filtered population data."""
    try:
        study_plan = load_plan(plan_file)
        df = study_plan.get_filtered_population(population)

        click.echo(f"Population: {population}")
        click.echo(f"N = {len(df)}")
        click.echo("\nFirst 10 rows:")
        click.echo(df.head(10))

    except TLFYAMLError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli()
```

**Setup in pyproject.toml**:
```toml
[project.scripts]
tlfyaml = "tlfyaml.cli:cli"
```

**Usage**:
```bash
tlfyaml validate examples/yaml/plan_xyz123.yaml
tlfyaml expand examples/yaml/plan_xyz123.yaml -o plan.csv
tlfyaml info examples/yaml/plan_xyz123.yaml
tlfyaml preview-population examples/yaml/plan_xyz123.yaml apat
```

**Impact**: Better developer experience, easier debugging, scripting support.

### 3.2 Enhanced Keyword Classes

**Current Gap**: Keyword classes are plain dataclasses with no behavior.

**Implementation**:
```python
@dataclass
class Population(Keyword):
    """Population definition with filter."""
    filter: str = ""

    def validate(self) -> bool:
        """Validate population definition."""
        if not self.name:
            raise ValueError("Population must have a name")
        if self.filter and ':' not in self.filter:
            raise ValueError(f"Population '{self.name}' filter must reference a dataset")
        return True

    def get_dataset_reference(self) -> Optional[str]:
        """Extract dataset name from filter."""
        if not self.filter:
            return None
        match = re.match(r'(\w+):', self.filter)
        return match.group(1) if match else None

    def __repr__(self) -> str:
        return f"Population(name='{self.name}', label='{self.label}', filter='{self.filter[:50]}...')"
```

Similar enhancements for Observation, Parameter, Group classes.

**Impact**: Self-validating keywords, better debugging output.

### 3.3 Comprehensive Documentation

**Implementation**:
- Add docstrings with examples to all public methods
- Create usage examples in `examples/` directory
- Add type hints to all function signatures
- Document exceptions in docstrings

**Example**:
```python
def load_plan(plan_path: str) -> StudyPlan:
    """
    Load a study plan from a YAML file with template inheritance.

    This function resolves template references, validates the YAML structure,
    loads referenced datasets, and creates a StudyPlan object ready for use.

    Args:
        plan_path: Path to the study plan YAML file

    Returns:
        StudyPlan object with loaded data and expanded keywords

    Raises:
        YAMLValidationError: If YAML structure is invalid
        TemplateNotFoundError: If referenced template doesn't exist
        DatasetNotFoundError: If referenced dataset cannot be loaded

    Examples:
        >>> from tlfyaml import load_plan
        >>> study_plan = load_plan('examples/yaml/plan_xyz123.yaml')
        >>> df = study_plan.get_plan_df()
        >>> print(f"Study has {len(df)} individual analyses")
        Study has 9 individual analyses

        >>> # Get filtered population
        >>> apat_df = study_plan.get_filtered_population('apat')
        >>> print(f"APAT population: N={len(apat_df)}")
        APAT population: N=254
    """
    # ... implementation ...
```

## Priority 4: Advanced Features (Future)

### 4.1 TLF Generation Engine
- Implement actual table/listing/figure generators
- Support multiple output formats (RTF, HTML, PDF)
- Template-based formatting

### 4.2 Validation Framework
- Cross-validation of filters against actual data
- Orphan keyword detection
- Circular template dependency detection

### 4.3 Performance Optimization
- Lazy loading of large datasets
- Filter result caching
- Parallel plan expansion

### 4.4 Integration Features
- R integration via reticulate
- Export to metalite format
- SAS XPORT import support

## Implementation Strategy

### Phase 1 (Immediate - Week 1-2)
1. Create exception hierarchy
2. Fix missing methods
3. Set up testing infrastructure
4. Write first 20 unit tests

### Phase 2 (Core - Week 3-4)
1. Implement filter parser
2. Implement filter executor
3. Add Pydantic validation
4. Achieve 80% test coverage

### Phase 3 (Polish - Week 5-6)
1. Add CLI interface
2. Enhance keyword classes
3. Complete documentation
4. Integration tests

### Phase 4 (Future)
1. TLF generation
2. Advanced validation
3. Performance tuning
4. External integrations

## Success Metrics

- **Test Coverage**: Achieve minimum 80% coverage
- **Error Handling**: All public methods have proper exception handling
- **Documentation**: All public APIs fully documented with examples
- **Validation**: YAML validation catches 95% of configuration errors before execution
- **Performance**: Plan expansion for 100+ analyses completes in <1 second
- **User Experience**: CLI commands work without reading source code

## Migration Path

All improvements maintain backward compatibility with existing YAML files. The enhancement strategy:

1. **Additive**: New features don't break existing code
2. **Gradual**: Optional validation can be enabled incrementally
3. **Documented**: Migration guides for new features
4. **Tested**: Existing example YAML files used in test suite

## Conclusion

This plan transforms the MVP from a proof-of-concept into a production-ready framework suitable for regulatory submissions in clinical trials. The prioritization ensures critical foundation work happens first while leaving advanced features for future iterations.
