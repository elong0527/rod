# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Review-Oriented Development (ROD) framework for clinical trial analysis and reporting in Biopharma. The project uses hierarchical YAML configurations with template inheritance to specify Tables, Listings, and Figures (TLFs) for clinical studies. The framework is designed to shift focus from writing code to reviewing AI-generated content, with built-in audit trails and regulatory compliance.

## Key Concepts

### YAML-Driven Analysis Planning

The core paradigm is specification-first development where YAML files define **what** to analyze, not **how**:

- **Organization/TA templates** (`examples/yaml/organization.yaml`): Define reusable keywords (populations, parameters, observations) shared across studies
- **Study plans** (`examples/yaml/plan_xyz123.yaml`): Reference templates and add study-specific filters and analysis specifications
- **Inheritance**: Study plans inherit from templates via field-level merging (e.g., template provides `label`, study adds `filter`)

### Architecture Components

1. **YamlInheritanceLoader** (`src/tlfyaml/yaml_loader.py`): Resolves template inheritance chain (organization → TA → study) using deep merging with keyword-level granularity
2. **KeywordRegistry** (`src/tlfyaml/plan.py`): Manages populations, observations, parameters, groups, and data sources as typed dataclasses
3. **PlanExpander** (`src/tlfyaml/plan.py`): Expands condensed YAML plans into individual analysis specifications via Cartesian products
4. **StudyPlan** (`src/tlfyaml/plan.py`): Main interface that loads datasets (Polars DataFrames from Parquet), resolves keywords, and expands plans

### Keyword System

Keywords use an enhanced SQL-like filter syntax:
- `adsl:saffl == 'Y'` means filter ADSL dataset where `saffl` equals 'Y'
- `adae:trtemfl == 'Y' and adae:aerel in ['RELATED', 'PROBABLY RELATED']` for multi-condition filters
- Keywords support: `name`, `label`, `description`, `filter`, `variable`, `level`, `group_label`

### Plan Expansion Logic

Plans use two parameter formats:
- **Semicolon-separated** (`parameter: "any;rel;ser"`): Single combined parameter for one analysis
- **List** (`parameter: ["any", "rel", "ser"]`): Cartesian expansion into multiple analyses

Example:
```yaml
- analysis: "ae_summary"
  population: "apat"
  observation: ["week12", "week24"]
  parameter: "any;rel;ser"  # Single combination
```
Expands to 2 analyses (week12, week24) each using the combined parameter `any;rel;ser`.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync --all-extras --group dev
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src/tlfyaml --cov-report=html

# Specific test
pytest tests/test_yaml_loader.py -v
```

### Documentation
```bash
# Render Quarto book (requires Quarto 1.5.57+)
quarto render docs/

# Preview locally
quarto preview docs/

# The book is published automatically via GitHub Actions to:
# https://elong0527.github.io/demo-tlf-yaml
```

### Code Quality
```bash
# Format code (Black with 100 char line length)
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

## Data Architecture

- **ADaM datasets** in `data/` directory (Parquet format): adsl, adae, adlbc, adlbh, adlbhy, adqsadas, adqscibc, adqsnpix, adtte, advs
- Uses Polars for data processing (not Pandas)
- Datasets are lazy-loaded when StudyPlan is instantiated
- Path resolution: relative to YAML file's directory (e.g., `../../data/adsl.parquet`)

## Quarto Documentation Structure

The `docs/` directory contains a Quarto book with Python code execution:
- `_quarto.yml`: Book configuration (type: book, freeze: auto, cache: true)
- `index.qmd`: ROD framework overview and business benefits
- `inheritance.qmd`: Template inheritance system explanation
- `plan.qmd`: Working example with Python code blocks that load and display study plans
- Uses Jupyter kernel (python3) for executing Python code in `.qmd` files
- Code blocks set `sys.path` to include `src/` for importing `tlfyaml`

## Important Patterns

### Loading a Study Plan
```python
from tlfyaml import load_plan

study_plan = load_plan('../examples/yaml/plan_xyz123.yaml')
df = study_plan.get_plan_df()  # Returns Polars DataFrame
```

### Field-Level Inheritance Example
When `organization.yaml` has `parameter: name: any, label: "Any Adverse Event"` and study plan has `parameter: name: any, filter: "adae:trtemfl == 'Y'"`, the merged result contains both `label` from template and `filter` from study.

### AI Integration Philosophy
- AI generates YAML from natural language requirements
- AI generates code from approved YAML specifications
- Humans review YAML (not implementation code)
- YAML serves as audit trail for regulatory compliance
- Template-driven approach reduces AI hallucinations

## File Conventions

- Python source: `src/tlfyaml/`
- YAML examples: `examples/yaml/`
- Data files: `data/` (Parquet only)
- Documentation: `docs/` (Quarto .qmd files)
- Tests: `tests/` (pytest)
- Project config: `pyproject.toml` (setuptools backend)

## Notes for AI Development

When working with this codebase:
1. Understand that YAML is the source of truth, not Python code
2. Preserve the separation between specification (YAML) and implementation (Python)
3. Maintain backward compatibility with existing YAML files
4. When adding features, consider template inheritance implications
5. Use Polars DataFrame operations, not Pandas
6. Follow the metalite R package patterns for consistency
7. Ensure any new keywords support field-level inheritance
8. Test with both semicolon-separated and list parameter formats
