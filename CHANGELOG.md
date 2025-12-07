# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0]

### Added
- **Centralized Configuration**: Introduced `src/csrlite/common/config.py` with `CsrLiteConfig` Pydantic model for managing global settings (column names, logging levels).
- **Listing**: Added logging configuration in `src/csrlite/__init__.py`.
- **Integration Tests**: Added `tests/integration_test_pipeline.py` covering the full pipeline from plan loading to RTF generation.
- **Extended Testing**: Added `tests/test_plan_extended.py` significantly increasing coverage for `plan.py`.

### Changed
- **Refactored `count.py`**:
    - Decoupled calculation from formatting.
    - Split `count_subject_with_observation` into `count_summary_data` (numeric) and `format_summary_table` (string).
    - Preserved backward compatibility with a wrapper.
- **Enhanced `StudyPlan`**:
    - Migrated `StudyPlan` and related classes (`Keyword`, `Population`, `Observation`, etc.) from `dataclasses` to `Pydantic` models for robust validation.
    - Improved `KeywordRegistry` to better handle nested polymorphism and legacy `group_label` list structures.
- **Dependencies**:
    - Added `structlog` (optional) and standard `logging` integration.
    - Enforced `pydantic>=2.0.0`.

### Fixed
- **Static Analysis**: Resolved all `pyre`, `mypy`, and `ruff` errors, including restoring the correct search path for Pyre.
- **Polars Deprecation**: Fixed `is_in` deprecation warning in `ae_specific.py`.
- **Bug**: Fixed `TypeError` in `StudyPlan.__str__` where `Dict` was instantiated.

### Removed
- Removed manual `print` statements in favor of structured logging.

## [0.1.0]

### Added
- Initial implementation of the csrlite library.