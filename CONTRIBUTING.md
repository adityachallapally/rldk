# Contributing to RLDK

Thank you for your interest in contributing to RLDK! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Local Setup](#local-setup)
- [Development Environment](#development-environment)
- [Running Tests](#running-tests)
- [Linting and Code Quality](#linting-and-code-quality)
- [Pull Request Process](#pull-request-process)
- [PR Checklist](#pr-checklist)

## Local Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda package manager

### Installation

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rldk.git
   cd rldk
   ```

2. **Create a virtual environment:**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n rldk python=3.8
   conda activate rldk
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Development Environment

### Project Structure

- `src/rldk/` - Main package source code
- `tests/` - Test files
- `docs/` - Documentation
- `examples/` - Example usage scripts
- `scripts/` - Utility scripts

### IDE Configuration

For the best development experience, we recommend:

- **VS Code**: Install Python extension and configure it to use the project's virtual environment
- **PyCharm**: Open the project and configure the Python interpreter to use the virtual environment

## Running Tests

### All Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rldk

# Run specific test file
pytest tests/test_specific_module.py

# Run tests with verbose output
pytest -v
```

### Test Categories
```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with specific markers
pytest -m "not slow"
```

## Linting and Code Quality

### Code Formatting

```bash
# Format code with black
black src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/

# Format with ruff (alternative to black)
ruff format src/ tests/ examples/
```

### Linting

```bash
# Run ruff linter
ruff check src/ tests/ examples/

# Run mypy type checking
mypy src/

# Run all linting tools
ruff check src/ tests/ examples/ && mypy src/
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically run formatting and linting:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run all checks** locally:
   ```bash
   # Format and lint
   black src/ tests/ examples/
   ruff check src/ tests/ examples/
   mypy src/
   
   # Run tests
   pytest --cov=src/rldk
   ```

### Submitting the PR

1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference any related issues
   - Include screenshots if UI changes
   - Ensure CI passes

## PR Checklist

Before submitting your pull request, please ensure:

### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] All new code is properly documented with docstrings
- [ ] Type hints are provided for new functions/methods
- [ ] No linting errors (run `ruff check` and `mypy`)
- [ ] Code is formatted with `black`

### Testing
- [ ] New functionality has corresponding tests
- [ ] All tests pass locally (`pytest`)
- [ ] Test coverage is maintained or improved
- [ ] Tests are clear and well-documented

### Documentation
- [ ] README.md is updated if needed
- [ ] API documentation is updated for new features
- [ ] Examples are updated if applicable
- [ ] Changelog is updated (if applicable)

### Review Readiness
- [ ] PR title is descriptive and concise
- [ ] PR description explains the changes clearly
- [ ] Related issues are referenced
- [ ] Breaking changes are clearly documented
- [ ] Migration instructions provided if needed

### Final Checks
- [ ] Branch is up to date with main
- [ ] CI/CD pipeline passes
- [ ] No merge conflicts
- [ ] Ready for review

## Getting Help

If you need help or have questions:

1. **Check existing issues** on GitHub
2. **Search documentation** in the `docs/` folder
3. **Create a new issue** with the "question" label
4. **Join discussions** in existing issues

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## License

By contributing to RLDK, you agree that your contributions will be licensed under the MIT License.