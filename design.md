# YAML TLF Framework Design Document

## 1. Executive Summary

This document outlines the design for a hierarchical YAML-based framework for generating Tables, Listings, and Figures (TLFs) in clinical trials. 
The framework is inspired by the metalite R package ecosystem and designed for Python implementation using rtflite, polars, and pydantic.

## 2. Design Principles

### Core Principles (from metalite)
- **Metadata-driven**: Single source of truth for clinical data definitions
- **Separation of concerns**: Isolate analysis logic from data sources
- **Hierarchical inheritance**: Organization → Therapeutic Area → Study
- **SQL-like filters**: Platform-agnostic filtering syntax
- **Self-contained YAML**: No hard-coded assumptions

### Framework Goals
- Support full clinical study reporting (safety focus)
- Generate RTF outputs via rtflite
- Enable both summary tables and patient-level listings
- Provide extensible architecture for custom TLFs

## 3. Hierarchical Architecture

```
Organization Level (Org)
├── Common definitions (USUBJID, standard populations)
├── Therapeutic Area Level (TA)
│   ├── TA-specific populations and parameters
│   ├── Oncology/ Diabetes/ Cardiology specific terms
│   └── Study Level
│       ├── Study-specific treatments and endpoints
│       ├── Protocol-specific populations
│       └── TLF Specifications
```

### Inheritance Model
- **Organization → TA → Study**: Child levels inherit and can override parent definitions
- **Conflict Resolution**: More specific (child) definitions override general (parent) ones
- **Reference System**: Use YAML anchors and references for reusability

## 4. YAML Schema Structure

### 4.1 Organization Level YAML (`org_common.yaml`)
```yaml
# Organization-wide definitions
organization:
  name: "clinical_trials_org"
  version: "1.0"

common_variables:
  subject_id: &subject_id
    name: "USUBJID"
    label: "Unique Subject Identifier"
    type: "string"
    required: true

common_populations: &common_populations
  safety:
    name: "SAFFL"
    label: "Safety Population Flag"
    filter: "SAFFL == 'Y'"

  itt:
    name: "ITTFL"
    label: "Intent-to-Treat Population Flag"
    filter: "ITTFL == 'Y'"

output_formats: &output_formats
  rtf:
    engine: "rtflite"
    page_orientation: "landscape"
    font_family: "Times New Roman"
    font_size: 9
```

### 4.2 Therapeutic Area Level YAML (`ta_safety.yaml`)
```yaml
# Therapeutic Area: Safety Tables
therapeutic_area:
  name: "safety"
  inherits_from: "org_common"

# TA-specific populations
populations:
  <<: *common_populations
  safety_evaluable:
    name: "SAFEVAL"
    label: "Safety Evaluable Population"
    filter: "SAFFL == 'Y' AND DTHFL != 'Y'"

# TA-specific parameters
parameters:
  adverse_events:
    any_ae: &any_ae
      name: "AECAT"
      label: "Any Adverse Event"
      filter: "TRTEMFL == 'Y'"

    serious_ae: &ser_ae
      name: "AESER"
      label: "Serious Adverse Event"
      filter: "AESER == 'Y' AND TRTEMFL == 'Y'"

    related_ae: &rel_ae
      name: "AEREL"
      label: "Treatment Related AE"
      filter: "AEREL IN ('POSSIBLE', 'PROBABLE', 'DEFINITE') AND TRTEMFL == 'Y'"

# TA-specific TLF templates
tlf_templates:
  ae_summary_table: &ae_summary_template
    type: "table"
    category: "safety"
    data_source: "adae"
    population: "safety"
    group_by: ["TRTA"]
    summary_vars: ["any_ae", "ser_ae", "rel_ae"]
    output_format: "rtf"
```

### 4.3 Study Level YAML (`study_xyz123.yaml`)
```yaml
# Study-specific configuration
study:
  name: "XYZ123"
  title: "Phase III Study of Drug X vs Placebo"
  inherits_from: "ta_safety"

# Study-specific data sources
data_sources:
  adsl:
    name: "adsl"
    path: "data/adam_validate/adsl.parquet"
    source: "ADSL"

  adae:
    name: "adae"
    path: "data/adam_validate/adae.parquet"
    source: "ADAE"

# Study-specific treatments
treatments:
  control: &control
    name: "Placebo"
    variable: "TRTA"
    filter: "TRTA == 'Placebo'"

  active: &active
    name: "Drug X 10mg"
    variable: "TRTA"
    filter: "TRTA == 'Drug X 10mg'"

# Study-specific populations (inherit + override)
populations:
  <<: *common_populations
  modified_itt:
    name: "mITT"
    label: "Modified Intent-to-Treat"
    filter: "ITTFL == 'Y' AND RANDFL == 'Y'"

# Study-specific TLF specifications
tlfs:
  ae_summary_by_treatment:
    <<: *ae_summary_template
    title: "Summary of Adverse Events by Treatment Group"
    subtitle: "Safety Population"
    footnotes:
      - "AE = Adverse Event"
      - "Serious AE defined per ICH E2A"
    columns:
      - name: "System Organ Class"
        variable: "AESOC"
      - name: "Preferred Term"
        variable: "AEDECOD"
    treatments: [*control, *active]
    output:
      filename: "t_ae_summary_by_trt.rtf"
      title_page: true

  ae_listing_serious:
    type: "listing"
    title: "Listing of Serious Adverse Events"
    data_source: "adae"
    population: "safety"
    filter: "AESER == 'Y'"
    sort_by: ["USUBJID", "AESTDTC"]
    columns:
      - {name: "Subject", variable: "USUBJID"}
      - {name: "Treatment", variable: "TRTA"}
      - {name: "System Organ Class", variable: "AESOC"}
      - {name: "Preferred Term", variable: "AEDECOD"}
      - {name: "Start Date", variable: "AESTDTC"}
      - {name: "End Date", variable: "AEENDTC"}
      - {name: "Severity", variable: "AESEV"}
      - {name: "Relationship", variable: "AEREL"}
    output:
      filename: "l_ae_serious.rtf"
```

## 5. Pydantic Models

### 5.1 Core Data Models
```python
from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field

class Variable(BaseModel):
    name: str
    label: str
    type: Literal["string", "numeric", "date", "datetime"]
    required: bool = False

class Population(BaseModel):
    name: str
    label: str
    filter: str  # SQL-like syntax

class Parameter(BaseModel):
    name: str
    label: str
    filter: Optional[str] = None

class DataSource(BaseModel):
    name: str
    path: str
    source: str  # ADaM domain name

class Treatment(BaseModel):
    name: str
    variable: str
    filter: str

class Column(BaseModel):
    name: str
    variable: str
    format: Optional[str] = None
    width: Optional[int] = None

class OutputFormat(BaseModel):
    filename: str
    title_page: bool = True
    orientation: Literal["portrait", "landscape"] = "landscape"
    font_size: int = 9

class TLFBase(BaseModel):
    type: Literal["table", "listing", "figure"]
    title: str
    subtitle: Optional[str] = None
    data_source: str
    population: str
    output: OutputFormat

class Table(TLFBase):
    type: Literal["table"] = "table"
    group_by: Optional[List[str]] = None
    summary_vars: List[str]
    footnotes: Optional[List[str]] = None

class Listing(TLFBase):
    type: Literal["listing"] = "listing"
    filter: Optional[str] = None
    sort_by: List[str]
    columns: List[Column]

class StudyConfig(BaseModel):
    study: Dict[str, str]
    inherits_from: str
    data_sources: Dict[str, DataSource]
    treatments: Dict[str, Treatment]
    populations: Dict[str, Population]
    tlfs: Dict[str, Union[Table, Listing]]
```

## 6. Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
1. **Setup project structure**
   ```
   src/
     tlfyaml/
       __init__.py
       models/         # Pydantic models
       loaders/        # YAML loading/inheritance
       generators/     # TLF generation engines
       filters/        # SQL-like filter parsing
       utils/          # Helper functions
   tests/
   examples/
     yaml/           # Example YAML configurations
     data/           # Sample ADaM data
   ```

2. **Implement YAML inheritance system**
   - YAML loader with inheritance resolution
   - Conflict resolution (child overrides parent)
   - Reference/anchor support

3. **Create Pydantic models**
   - Core data models (Variable, Population, etc.)
   - TLF-specific models (Table, Listing, Figure)
   - Validation rules

### Phase 2: Data Processing Engine (Weeks 3-4)
1. **SQL-like filter parser**
   - Parse filter strings to polars expressions
   - Support common SQL operators (==, !=, IN, LIKE, etc.)
   - Validate filters against data schemas

2. **Data loading and validation**
   - polars-based data loading from parquet/CSV
   - Data validation against YAML specifications
   - Population subsetting

### Phase 3: RTF Generation Engine (Weeks 5-6)
1. **rtflite integration**
   - Table generation with rtflite
   - Listing generation
   - RTF formatting and styling

2. **Template processing**
   - Convert YAML TLF specs to rtflite calls
   - Handle grouping, sorting, summarization
   - Apply study-specific formatting

### Phase 4: Safety Tables Implementation (Weeks 7-8)
1. **Implement metalite.ae-inspired tables**
   - AE summary tables by treatment group
   - AE incidence tables by System Organ Class
   - Serious AE listings
   - Related AE summaries

2. **Advanced features**
   - Cross-tabulation support
   - Statistical testing (Chi-square, Fisher's exact)
   - Confidence intervals

## 7. Key Components

### 7.1 YAML Inheritance Loader
```python
class YAMLInheritanceLoader:
    def load_study_config(self, study_yaml_path: str) -> StudyConfig:
        # Load study YAML
        # Resolve inherits_from chain
        # Merge configurations with child override
        # Validate final config with Pydantic
        pass

    def _resolve_inheritance(self, config: dict) -> dict:
        # Recursively resolve inheritance
        pass
```

### 7.2 Filter Engine
```python
class FilterEngine:
    def parse_sql_filter(self, filter_str: str) -> polars.Expr:
        # Convert "TRTEMFL == 'Y'" to polars expression
        pass

    def apply_population_filter(self, df: polars.DataFrame,
                              population: Population) -> polars.DataFrame:
        pass
```

### 7.3 TLF Generator
```python
class TLFGenerator:
    def generate_table(self, table_spec: Table,
                      data: polars.DataFrame) -> str:
        # Generate RTF table using rtflite
        pass

    def generate_listing(self, listing_spec: Listing,
                        data: polars.DataFrame) -> str:
        # Generate RTF listing using rtflite
        pass
```

## 8. Usage Example

```python
from tlfyaml import YAMLInheritanceLoader, TLFGenerator

# Load study configuration
loader = YAMLInheritanceLoader()
config = loader.load_study_config("examples/yaml/study_xyz123.yaml")

# Generate TLF
generator = TLFGenerator(config)
generator.generate_tlf("ae_summary_by_treatment")
generator.generate_tlf("ae_listing_serious")
```

## 9. Success Criteria

**Phase 1 Success**:
- YAML inheritance working correctly
- Pydantic models validate successfully
- Basic project structure established

**Final Success**:
- Generate AE summary table matching metalite.ae output
- Generate serious AE listing
- RTF outputs formatted for regulatory submission
- Hierarchical YAML system working across org→TA→study levels
- SQL-like filters correctly subset data using polars

This design provides a robust, extensible foundation for clinical TLF generation while maintaining the principles and patterns established by the metalite ecosystem.