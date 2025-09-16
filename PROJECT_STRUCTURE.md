# RLDK Project Structure Guidelines

## 🏗️ **Overview**

This document defines the proper organization and structure for the RLDK project. It addresses common issues like scattered files, inconsistent organization, and lack of clear boundaries between different types of content.

## 📁 **Directory Structure**

### **Root Level Organization**

```
/workspace/
├── src/                        # Source code (main package)
├── tests/                      # All test files
├── docs/                       # Documentation
├── examples/                   # Example scripts and demos
├── scripts/                   # Utility and maintenance scripts
├── tools/                     # Development tools
├── templates/                 # Template files
├── reference/                 # Reference materials
├── recipes/                   # Usage recipes and patterns
├── assets/                    # Static assets (images, etc.)
├── artifacts/                 # Generated artifacts
├── rldk_reports/             # Generated reports
├── requirements*.txt          # Python dependencies
├── pyproject.toml            # Project configuration
├── Makefile                  # Build automation
├── Dockerfile                # Container configuration
├── .gitignore                # Git ignore rules
├── README.md                 # Main project documentation
├── CONTRIBUTING.md           # Contribution guidelines
├── agents.md                 # AI agent guidelines
└── PROJECT_STRUCTURE.md      # This file
```

### **Source Code Structure (`src/rldk/`)**

```
src/rldk/
├── __init__.py               # Package initialization
├── cli.py                    # Command-line interface
├── tracking/                 # Experiment tracking
│   ├── __init__.py
│   ├── tracker.py
│   ├── config.py
│   ├── dataset_tracker.py
│   ├── model_tracker.py
│   ├── environment_tracker.py
│   ├── seed_tracker.py
│   └── git_tracker.py
├── forensics/                # PPO analysis and debugging
│   ├── __init__.py
│   ├── comprehensive_ppo_forensics.py
│   ├── ppo_scan.py
│   ├── env_audit.py
│   ├── log_scan.py
│   ├── ckpt_diff.py
│   └── advantage_statistics_tracker.py
├── ingest/                   # Data ingestion
│   ├── __init__.py
│   └── ingest.py
├── adapters/                 # Framework adapters
│   ├── __init__.py
│   ├── base.py
│   ├── trl.py
│   ├── openrlhf.py
│   ├── wandb.py
│   └── custom_jsonl.py
├── diff/                     # Run comparison
│   ├── __init__.py
│   └── diff.py
├── determinism/              # Determinism checking
│   ├── __init__.py
│   └── check.py
├── reward/                   # Reward analysis
│   ├── __init__.py
│   ├── health_analysis.py
│   ├── drift.py
│   └── calibration.py
├── evals/                    # Evaluation suites
│   ├── __init__.py
│   ├── suites.py
│   ├── runner.py
│   ├── metrics/
│   └── probes/
├── replay/                   # Seeded replay
│   ├── __init__.py
│   └── replay.py
├── bisect/                   # Git bisect
│   ├── __init__.py
│   └── bisect.py
├── cards/                    # Trust cards
│   ├── __init__.py
│   ├── determinism.py
│   ├── drift.py
│   └── reward.py
├── io/                       # I/O utilities
│   ├── __init__.py
│   ├── event_schema.py
│   ├── writers.py
│   └── readers.py
├── config/                   # Configuration management
│   ├── __init__.py
│   ├── evaluation_config.py
│   ├── forensics_config.py
│   ├── visualization_config.py
│   ├── suite_config.py
│   ├── environments/
│   └── presets/
├── integrations/             # Framework integrations
│   ├── __init__.py
│   ├── trl/
│   └── openrlhf/
└── utils/                    # Utility functions
    ├── __init__.py
    ├── seed.py
    ├── validation.py
    └── helpers.py
```

### **Test Structure (`tests/`)**

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_tracking.py
│   ├── test_forensics.py
│   ├── test_determinism.py
│   ├── test_reward.py
│   ├── test_evals.py
│   └── test_config.py
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_tracking_integration.py
│   ├── test_forensics_integration.py
│   └── test_end_to_end.py
├── e2e/                     # End-to-end tests
│   ├── __init__.py
│   ├── test_full_workflow.py
│   └── test_cli_commands.py
├── data/                    # Test data
│   ├── sample_logs/
│   ├── sample_models/
│   └── sample_datasets/
└── fixtures/                # Test fixtures
    ├── sample_configs.py
    └── mock_data.py
```

### **Documentation Structure (`docs/`)**

```
docs/
├── index.md                 # Main documentation index
├── getting-started/         # Getting started guides
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── reference/               # API reference
│   ├── api.md
│   ├── commands.md
│   └── configuration.md
├── evals/                   # Evaluation documentation
│   ├── data_requirements.md
│   └── suite_descriptions.md
├── tutorials/               # Tutorials and guides
│   ├── basic_usage.md
│   ├── advanced_features.md
│   └── troubleshooting.md
├── architecture/            # Architecture documentation
│   ├── overview.md
│   ├── components.md
│   └── design_decisions.md
└── implementation/          # Implementation summaries
    ├── tracking_system.md
    ├── forensics_analysis.md
    └── evaluation_suites.md
```

### **Examples Structure (`examples/`)**

```
examples/
├── README.md                # Examples overview
├── basic_usage/             # Basic usage examples
│   ├── simple_tracking.py
│   ├── basic_forensics.py
│   └── quick_evaluation.py
├── advanced_features/        # Advanced feature examples
│   ├── custom_adapters.py
│   ├── distributed_training.py
│   └── hyperparameter_tuning.py
├── integrations/            # Framework integration examples
│   ├── trl_integration.py
│   ├── openrlhf_integration.py
│   └── wandb_integration.py
├── demos/                   # Demo scripts
│   ├── comprehensive_ppo_forensics_demo/
│   ├── comprehensive_ppo_monitor_demo/
│   └── enhanced_ppo_scan_demo/
└── notebooks/               # Jupyter notebooks
    ├── rldk_demo.ipynb
    └── analysis_examples.ipynb
```

## 📋 **File Naming Conventions**

### **Python Files**

- **Modules**: `snake_case.py` (e.g., `data_processor.py`)
- **Classes**: `PascalCase` in files (e.g., `class DataProcessor`)
- **Functions**: `snake_case` (e.g., `def process_data()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)

### **Configuration Files**

- **Config classes**: `{domain}_config.py` (e.g., `evaluation_config.py`)
- **Environment configs**: `{environment}.py` (e.g., `production.py`)
- **Preset configs**: `{preset}_preset.py` (e.g., `strict_preset.py`)

### **Test Files**

- **Unit tests**: `test_{module}.py` (e.g., `test_tracking.py`)
- **Integration tests**: `test_{module}_integration.py` (e.g., `test_tracking_integration.py`)
- **E2E tests**: `test_{feature}_e2e.py` (e.g., `test_cli_e2e.py`)

### **Documentation Files**

- **User docs**: `{topic}.md` (e.g., `installation.md`)
- **API docs**: `{module}_api.md` (e.g., `tracking_api.md`)
- **Implementation summaries**: `{feature}_implementation.md` (e.g., `tracking_implementation.md`)

### **Example Files**

- **Examples**: `{purpose}_{feature}.py` (e.g., `basic_tracking.py`)
- **Demos**: `{feature}_demo.py` (e.g., `forensics_demo.py`)

## 🚫 **What NOT to Put in Root**

### **Never Create in Root:**

- **Python files**: Use `src/rldk/` or `examples/`
- **Test files**: Use `tests/`
- **Documentation**: Use `docs/`
- **Configuration**: Use `src/rldk/config/`
- **Example scripts**: Use `examples/`
- **Temporary files**: Use `/tmp/` or add to `.gitignore`
- **Random markdown files**: Use appropriate subdirectories

### **Common Mistakes:**

```bash
# ❌ WRONG: Files in root
touch test_something.py
touch NEW_FEATURE_SUMMARY.md
touch temp_analysis.py
touch config_settings.py

# ✅ CORRECT: Files in proper locations
touch tests/unit/test_something.py
touch docs/implementation/new_feature.md
touch examples/analysis_example.py
touch src/rldk/config/new_settings.py
```

## 🔧 **Configuration Organization**

### **Configuration Hierarchy**

```
src/rldk/config/
├── __init__.py               # Main config exports
├── base_config.py            # Base configuration class
├── evaluation_config.py      # Evaluation parameters
├── forensics_config.py       # Forensics parameters
├── visualization_config.py   # Visualization parameters
├── suite_config.py          # Evaluation suite parameters
├── environments/             # Environment-specific configs
│   ├── __init__.py
│   ├── development.py
│   ├── production.py
│   └── testing.py
└── presets/                 # Configuration presets
    ├── __init__.py
    ├── strict.py
    ├── lenient.py
    ├── research.py
    └── fast.py
```

### **Configuration Usage Pattern**

```python
# ✅ CORRECT: Configuration usage
from ..config import get_eval_config, get_forensics_config

def your_function(config=None):
    if config is None:
        config = get_eval_config()
    
    # Use config parameters
    if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
        # process data
        pass

# ❌ WRONG: Hardcoded values
def your_function():
    if len(data) > 10:  # Hardcoded!
        # process data
        pass
```

## 🧪 **Test Organization**

### **Test Structure Rules**

1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test interactions between components
3. **E2E Tests**: Test complete workflows
4. **Test Data**: Keep in `tests/data/`
5. **Fixtures**: Keep in `tests/fixtures/`

### **Test Naming**

```python
# ✅ CORRECT: Test class naming
class TestDataProcessor:
    def test_process_empty_data(self):
        pass
    
    def test_process_valid_data(self):
        pass
    
    def test_process_invalid_data(self):
        pass

# ✅ CORRECT: Test function naming
def test_data_processor_with_default_config():
    pass

def test_data_processor_with_custom_config():
    pass
```

## 📚 **Documentation Organization**

### **Documentation Types**

1. **User Documentation**: How to use the library
2. **API Documentation**: Reference for all functions/classes
3. **Implementation Documentation**: How things work internally
4. **Tutorial Documentation**: Step-by-step guides
5. **Architecture Documentation**: System design and decisions

### **Documentation Structure**

```markdown
<!-- ✅ CORRECT: Documentation structure -->
# Feature Name

## Overview
Brief description of the feature.

## Usage
Code examples showing how to use the feature.

## Configuration
Configuration options and parameters.

## API Reference
Detailed API documentation.

## Examples
Complete working examples.

## Troubleshooting
Common issues and solutions.
```

## 🎯 **Quality Guidelines**

### **Code Quality**

- **Type Hints**: Always include type hints
- **Docstrings**: Document all public functions/classes
- **Error Handling**: Include proper error handling
- **Configuration**: Use configuration instead of hardcoded values
- **Testing**: Include comprehensive tests

### **File Quality**

- **Single Responsibility**: Each file should have one clear purpose
- **Consistent Naming**: Follow naming conventions
- **Proper Imports**: Use relative imports within packages
- **Clean Structure**: Logical organization of code

### **Documentation Quality**

- **Clear Examples**: Include working code examples
- **Complete Coverage**: Document all public APIs
- **Up-to-date**: Keep documentation current with code
- **User-focused**: Write for the user, not the implementer

## 🔄 **Migration Guidelines**

### **When Restructuring**

1. **Identify Dependencies**: Find all files that import/use the code being moved
2. **Update Imports**: Change all import statements
3. **Update Tests**: Move tests to match new structure
4. **Update Documentation**: Update file paths in documentation
5. **Validate**: Run tests to ensure nothing broke

### **Import Updates**

```python
# Before restructuring
from rldk.utils import helper_function

# After restructuring
from rldk.utils.helpers import helper_function
```

## 🚀 **Quick Reference**

### **File Locations**
- Source code: `src/rldk/`
- Tests: `tests/`
- Examples: `examples/`
- Documentation: `docs/`
- Configuration: `src/rldk/config/`
- Scripts: `scripts/`

### **Naming Conventions**
- Python files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_{module}.py`

### **Configuration Pattern**
```python
from ..config import get_eval_config

def your_function(config=None):
    if config is None:
        config = get_eval_config()
    # Use config.PARAMETER_NAME
```

### **Test Pattern**
```python
# tests/unit/test_your_module.py
import pytest
from src.rldk.your_module import YourClass

class TestYourClass:
    def test_with_default_config(self):
        obj = YourClass()
        result = obj.method(test_data)
        assert result is not None
```

---

**Remember: Consistency is key. Follow the established patterns and structure to maintain a clean, organized codebase.**