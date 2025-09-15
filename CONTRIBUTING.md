# Contributing to RLDK

Thank you for your interest in contributing to RLDK! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository and clone your fork
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **MyPy** for type checking

Run these tools before committing:
```bash
black .
isort .
ruff check .
mypy src/
```

## Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/rldk --cov-report=html
```

## Release Process

### Prerequisites

Before creating a release, ensure you have:

1. **PyPI Account**: Access to the PyPI project for RLDK
2. **Trusted Publishing Setup**: The repository must be configured in PyPI with OIDC settings for trusted publishing
3. **GitHub Pages**: Ensure GitHub Pages is enabled for the repository

### Release Steps

1. **Update Version**: Update the version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.1"  # Update to new version
   ```

2. **Update Changelog**: Add release notes to `CHANGELOG.md` (if it exists) or update the release description

3. **Commit Changes**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 0.1.1"
   ```

4. **Create and Push Tag**:
   ```bash
   git tag -a v0.1.1 -m "Release version 0.1.1"
   git push origin v0.1.1
   ```

5. **Create GitHub Release**:
   - Go to the GitHub repository
   - Click "Releases" → "Create a new release"
   - Choose the tag you just created (e.g., `v0.1.1`)
   - Add a release title and description
   - Click "Publish release"

### Automated Processes

Once you create a GitHub release, the following processes will happen automatically:

1. **PyPI Publishing**: The `publish_pypi.yml` workflow will:
   - Build the package (sdist and wheel)
   - Publish to PyPI using trusted publishing (no API tokens needed)
   - Only triggers on published releases

2. **Documentation Deployment**: The `deploy_docs.yml` workflow will:
   - Build the MkDocs documentation
   - Deploy to GitHub Pages
   - Triggers on pushes to `main` branch and on tags

### Troubleshooting

- **PyPI Publishing Fails**: Ensure the repository is properly configured in PyPI with OIDC settings
- **Documentation Not Updating**: Check that GitHub Pages is enabled and the workflow has the correct permissions
- **Version Conflicts**: Ensure the version in `pyproject.toml` matches the git tag

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes following the code style guidelines
3. Add tests for new functionality
4. Update documentation if needed
5. Run the full test suite
6. Submit a pull request with a clear description

## Questions?

If you have questions about contributing, please:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the documentation in the `docs/` directory

Thank you for contributing to RLDK!