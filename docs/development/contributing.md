# Contributing to RLDK

Thank you for your interest in contributing to RLDK! This guide will help you get started with contributing to the project.

## Getting Started

### Development Setup

1. **Fork and Clone the Repository**
```bash
git clone https://github.com/your-username/rldk.git
cd rldk
```

2. **Set Up Development Environment**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -U pip
pip install -e .[dev]
```

3. **Verify Installation**
```bash
# Run tests
pytest

# Run linting
ruff check .
ruff format --check .

# Run CLI smoke tests
rldk --help
rldk evals list-suites
```

### Development Workflow

1. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Your Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
```bash
# Run the acceptance script
./tools/accept.sh

# Or run individual checks
make lint
make test-quick
make cli-smoke
```

4. **Commit and Push**
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

5. **Create a Pull Request**
   - Use the GitHub interface to create a PR
   - Fill out the PR template
   - Wait for review and CI to pass

## Coding Standards

### Python Code Style

We use several tools to maintain code quality:

- **Ruff**: For linting and import sorting
- **Black**: For code formatting (via Ruff)
- **MyPy**: For type checking
- **Pytest**: For testing

### Code Formatting

```bash
# Format code
ruff format .

# Check formatting
ruff format --check .

# Fix linting issues
ruff check --fix .
```

### Type Hints

All new code should include type hints:

```python
from typing import List, Dict, Optional, Union

def process_metrics(
    metrics: Dict[str, float],
    threshold: float = 0.1
) -> List[str]:
    """Process metrics and return anomalies."""
    anomalies: List[str] = []
    
    for metric_name, value in metrics.items():
        if value > threshold:
            anomalies.append(metric_name)
    
    return anomalies
```

### Documentation

All public functions and classes should have docstrings:

```python
def track_experiment(
    name: str,
    config: Dict[str, Any],
    enable_tracking: bool = True
) -> str:
    """Track a machine learning experiment.
    
    Args:
        name: Name of the experiment
        config: Configuration dictionary
        enable_tracking: Whether to enable tracking
        
    Returns:
        Path to the experiment directory
        
    Raises:
        ValueError: If name is empty
        
    Example:
        >>> path = track_experiment("my_exp", {"lr": 0.01})
        >>> print(f"Experiment saved to: {path}")
    """
    if not name:
        raise ValueError("Experiment name cannot be empty")
    
    # Implementation here
    return experiment_path
```

## Testing

### Writing Tests

We use pytest for testing. Tests should be placed in the `tests/` directory:

```python
# tests/test_tracking.py
import pytest
from rldk.tracking import ExperimentTracker, TrackingConfig

def test_experiment_tracker_basic():
    """Test basic experiment tracking functionality."""
    config = TrackingConfig(experiment_name="test_exp")
    tracker = ExperimentTracker(config)
    
    tracker.start_experiment()
    tracker.add_metadata("test_key", "test_value")
    path = tracker.finish_experiment()
    
    assert path is not None
    assert "test_exp" in path

def test_experiment_tracker_with_invalid_config():
    """Test that invalid config raises appropriate error."""
    with pytest.raises(ValueError):
        TrackingConfig(experiment_name="")  # Empty name should fail
```

### Test Categories

We have several types of tests:

1. **Unit Tests** (`tests/unit/`): Test individual functions and classes
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **End-to-End Tests** (`tests/e2e/`): Test complete workflows

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tracking.py

# Run with coverage
pytest --cov=rldk --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run integration tests
pytest tests/integration/
```

### Test Fixtures

Use fixtures for common test setup:

```python
# tests/conftest.py
import pytest
import tempfile
from rldk.tracking import TrackingConfig

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def basic_config(temp_dir):
    """Provide a basic tracking configuration."""
    return TrackingConfig(
        experiment_name="test_experiment",
        output_dir=temp_dir
    )
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality
3. **Documentation**: Improve docs and examples
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize existing code

### Feature Requests

Before implementing a new feature:

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create an Issue**: Describe the feature and its use case
3. **Discuss Design**: Get feedback on the approach
4. **Implement**: Create the feature with tests and docs

### Bug Reports

When reporting bugs:

1. **Use the Issue Template**: Fill out all sections
2. **Provide Reproduction Steps**: Clear steps to reproduce
3. **Include Environment Info**: Python version, OS, dependencies
4. **Add Logs/Errors**: Include relevant error messages

Example bug report:
```markdown
## Bug Description
RLDK tracking fails when using very large datasets

## Steps to Reproduce
1. Create dataset with 10M+ samples
2. Call `tracker.track_dataset(large_dataset, "data")`
3. Observe memory error

## Expected Behavior
Should handle large datasets efficiently

## Environment
- Python: 3.10.0
- RLDK: 0.1.0
- OS: Ubuntu 20.04
- Memory: 16GB

## Error Message
```
MemoryError: Unable to allocate array
```
```

### Pull Request Guidelines

**Before Submitting:**
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] CHANGELOG is updated (if applicable)

**PR Description Should Include:**
- [ ] Clear description of changes
- [ ] Motivation for the changes
- [ ] Testing strategy
- [ ] Breaking changes (if any)

**PR Template:**
```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

## Development Environment

### IDE Setup

**VS Code Configuration** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.*_cache": true
    }
}
```

### Useful Commands

```bash
# Development workflow
make init          # Set up development environment
make lint          # Run linting checks
make test-quick    # Run fast tests
make cli-smoke     # Test CLI commands
make docs-serve    # Serve documentation locally

# Advanced testing
pytest --pdb                    # Debug test failures
pytest --lf                     # Run only last failed tests
pytest -x                       # Stop on first failure
pytest -v tests/test_tracking.py::test_basic  # Run specific test

# Code quality
ruff check --statistics        # Show linting statistics
mypy src/rldk/                 # Type checking
coverage report                # Show test coverage
```

### Debugging

**Using the Debugger:**
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use pytest with pdb
pytest --pdb tests/test_tracking.py
```

**Logging for Development:**
```python
import logging

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Debug information")
    logger.info("General information")
    logger.warning("Warning message")
```

## Architecture Overview

### Project Structure

```
rldk/
├── src/rldk/              # Main package
│   ├── tracking/          # Experiment tracking
│   ├── forensics/         # Training analysis
│   ├── determinism/       # Reproducibility checking
│   ├── evals/            # Evaluation suites
│   ├── integrations/     # Framework integrations
│   └── utils/            # Shared utilities
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example scripts
├── scripts/              # Utility scripts
└── tools/                # Development tools
```

### Key Design Principles

1. **Modularity**: Each component should be independently usable
2. **Extensibility**: Easy to add new adapters, metrics, and integrations
3. **Performance**: Minimal overhead during training
4. **Reliability**: Robust error handling and graceful degradation
5. **Usability**: Clear APIs and comprehensive documentation

### Adding New Features

**New Adapter Example:**
```python
# src/rldk/adapters/my_adapter.py
from .base import BaseAdapter
import pandas as pd

class MyFrameworkAdapter(BaseAdapter):
    """Adapter for MyFramework logs."""
    
    def can_handle(self, source: str) -> bool:
        """Check if this adapter can handle the source."""
        return source.endswith('.myformat')
    
    def ingest(self, source: str) -> pd.DataFrame:
        """Ingest data from MyFramework format."""
        # Implementation here
        return normalized_df
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the ingested data."""
        # Validation logic here
        return True
```

**New Metric Example:**
```python
# src/rldk/evals/metrics/my_metric.py
from typing import Dict, Any
from ..base import BaseMetric

class MyMetric(BaseMetric):
    """Custom evaluation metric."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def evaluate(self, model, tokenizer, data) -> Dict[str, Any]:
        """Evaluate the metric."""
        # Implementation here
        return {
            "score": score,
            "passed": score > self.threshold,
            "details": details
        }
```

## Release Process

### Version Management

We use semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. **Update Version**
   - Update `pyproject.toml`
   - Update `src/rldk/__init__.py`

2. **Update Documentation**
   - Update CHANGELOG.md
   - Update README.md if needed
   - Regenerate API docs

3. **Test Release**
   - Run full test suite
   - Test installation from source
   - Test key workflows

4. **Create Release**
   - Tag release: `git tag v0.1.0`
   - Push tag: `git push origin v0.1.0`
   - Create GitHub release
   - Publish to PyPI

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Documentation**: Comprehensive guides and API reference

### Maintainer Response

We aim to:
- Respond to issues within 48 hours
- Review PRs within 1 week
- Release bug fixes within 2 weeks
- Release new features monthly

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and inclusive in all interactions.

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributor graphs
- Special recognition for significant contributions

Thank you for contributing to RLDK! Your contributions help make RL research more reproducible and reliable.
