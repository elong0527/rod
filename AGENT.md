# TLF YAML Framework Agent Documentation

## Overview

This document provides guidance for AI agents working with the TLF YAML Framework for clinical trial reporting. The framework uses hierarchical YAML configurations with inheritance to generate Tables, Listings, and Figures (TLFs) for regulatory submissions.

## Architecture Understanding

### Core Components

```
src/tlfyaml/
├── models/          # Pydantic data models
│   ├── base.py      # Core entities (Variable, Population, Treatment, etc.)
│   ├── tlf.py       # TLF specifications (Table, Listing, Figure)
│   └── config.py    # Configuration models (Org, TA, Study)
├── loaders/         # YAML inheritance system
│   └── yaml_loader.py
└── generators/      # TLF generation engines
    └── tlf_generator.py
```

### Hierarchical Inheritance Pattern

1. **Organization Level** (`org_common.yaml`)
   - Common variables, populations, output formats
   - Shared across all therapeutic areas

2. **Therapeutic Area Level** (`ta_safety.yaml`)
   - TA-specific populations and parameters
   - Inherits from organization level
   - Example: safety tables, efficacy endpoints

3. **Study Level** (`study_xyz123.yaml`)
   - Study-specific treatments, data sources, TLFs
   - Inherits from TA level
   - Protocol-specific configurations

## Key Design Principles

### 1. YAML Anchors for Reusability
```yaml
# Define once
safety_population: &safety
  name: "SAFFL"
  label: "Safety Population Flag"
  filter: "SAFFL == 'Y'"

# Reuse everywhere
tlfs:
  ae_summary:
    population: *safety
```

### 2. SQL-like Filtering
```yaml
population:
  safety:
    filter: "SAFFL == 'Y' AND TRTEMFL == 'Y'"
  # Converts to polars: df.filter((pl.col("SAFFL") == "Y") & (pl.col("TRTEMFL") == "Y"))
```

### 3. Metadata-Driven Configuration
```yaml
tlf_spec:
  type: "table"
  title: "Adverse Events Summary"
  data_source: "adae"
  population: "safety"
  summary_vars: ["any_ae", "ser_ae"]
```

## Working with the Framework

### Loading Configurations
```python
from tlfyaml import YAMLInheritanceLoader, TLFGenerator

# Load with inheritance resolution
loader = YAMLInheritanceLoader(config_base_path="examples/yaml")
config = loader.load_study_config("study_xyz123.yaml")

# Generate TLFs
generator = TLFGenerator(config)
generator.generate_tlf("ae_summary_by_treatment")
```

### Understanding Data Models

#### Core Entities
- **Variable**: Clinical data variable (USUBJID, TRTA, etc.)
- **Population**: Analysis population with filter criteria
- **Treatment**: Treatment group definition with filters
- **DataSource**: ADaM dataset specification

#### TLF Types
- **Table**: Summary tables with grouping and statistics
- **Listing**: Patient-level data listings
- **Figure**: Plots and visualizations

### Common Patterns

#### 1. Adding New TLF
```yaml
tlfs:
  new_analysis:
    type: "table"
    title: "New Analysis Table"
    data_source: "adsl"
    population: "itt"
    group_by: ["TRTA"]
    summary_vars: ["AGE", "SEX"]
    output:
      filename: "t_new_analysis.rtf"
```

#### 2. Creating TA-Specific Templates
```yaml
# In ta_oncology.yaml
oncology_populations:
  evaluable_tumor: &eval_tumor
    name: "EVALFL"
    label: "Evaluable for Tumor Response"
    filter: "EVALFL == 'Y' AND BASELINE_TUMOR IS NOT NULL"

efficacy_template: &eff_template
  type: "table"
  data_source: "adrs"
  population: *eval_tumor
```

#### 3. Study-Specific Customization
```yaml
# Inherit template and customize
tumor_response:
  <<: *eff_template
  title: "Tumor Response Analysis"
  summary_vars: ["COMPLETE_RESPONSE", "PARTIAL_RESPONSE"]
```

## Agent Guidelines

### 1. When Modifying Configurations

**DO:**
- Use anchors (`&`) for reusable definitions
- Follow inheritance hierarchy (org → ta → study)
- Include complete filter expressions
- Validate YAML syntax and structure
- Test configuration loading

**DON'T:**
- Duplicate definitions across files
- Hard-code values that should be configurable
- Break existing inheritance chains
- Create circular references

### 2. Understanding Clinical Context

**Key Clinical Concepts:**
- **Safety Population**: Subjects who received study treatment (`SAFFL == 'Y'`)
- **ITT Population**: All randomized subjects (`ITTFL == 'Y'`)
- **Treatment-Emergent**: Events during treatment period (`TRTEMFL == 'Y'`)
- **Serious AE**: Life-threatening events (`AESER == 'Y'`)

**Common Variables:**
- `USUBJID`: Unique Subject Identifier
- `TRTA`/`TRTP`: Actual/Planned Treatment
- `AESOC`: System Organ Class (AE grouping)
- `AEDECOD`: Preferred Term (specific AE)

### 3. Error Handling

**Common Issues:**
- Missing inheritance files
- Invalid filter syntax
- Circular references in anchors
- Missing required fields in TLF specs

**Debugging:**
- Check inheritance chain resolution
- Validate filter expressions
- Verify data source availability
- Test with minimal configurations

## Extension Points

### Adding New TLF Types
1. Extend `TLFBase` in `models/tlf.py`
2. Add generation logic in `generators/tlf_generator.py`
3. Update configuration schema

### Custom Analysis Parameters
1. Define in therapeutic area YAML
2. Reference in study TLF specifications
3. Implement calculation logic in generator

### New Output Formats
1. Extend `OutputFormat` model
2. Add format-specific generation logic
3. Update template inheritance

## Best Practices for Agents

### 1. Configuration Changes
- Always validate inheritance after changes
- Test with example data before deployment
- Document new patterns and templates
- Follow naming conventions

### 2. Clinical Accuracy
- Understand regulatory requirements
- Validate population definitions
- Ensure traceability of calculations
- Follow CDISC ADaM standards

### 3. Maintainability
- Use descriptive names for anchors
- Comment complex filter logic
- Group related definitions
- Minimize configuration duplication

## Resources

- **Design Document**: `TLF_YAML_Framework_Design.md`
- **Example Configurations**: `examples/yaml/`
- **Demo Script**: `examples/demo.py`
- **Pydantic Models**: `src/tlfyaml/models/`

## Quick Reference

### File Structure
```
examples/yaml/
├── org_common.yaml      # Organization-wide definitions
├── ta_safety.yaml       # Safety analysis templates
└── study_xyz123.yaml    # Study-specific configuration
```

### Key Commands
```bash
# Run demo
uv run python examples/demo.py

# Load configuration
config = loader.load_study_config("study_xyz123.yaml")

# Generate TLF
generator.generate_tlf("ae_summary_by_treatment")
```

This framework provides a robust foundation for clinical TLF generation while maintaining flexibility and regulatory compliance.