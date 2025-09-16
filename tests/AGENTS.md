# Test Suite Guidelines

These rules apply to all files under `tests/`.

## Structure

- Mirror the `src/rldk/` package layout when adding new tests. Place integration or end-to-end scenarios in the existing `integration/` or `e2e/` folders.
- Use descriptive filenames (`test_<module>_<behavior>.py`). Avoid placing test modules outside of `tests/`.
- Share fixtures via `conftest.py` or `tests/_make_fixtures.py`; keep them reusable and lightweight.

## Style

- Prefer `pytest` idioms such as fixtures and parametrization.
- Mark slow or integration-heavy tests with the configured markers (`slow`, `integration`, `mutation`, etc.) so that they can be deselected.
- Avoid random sleeps. If timing is required, use deterministic synchronization or `pytest`'s retry utilities.

## Validation

- Run targeted `pytest` selections (`pytest tests/path`) when iterating locally.
- Ensure any new fixtures or helpers are documented in `docs/` when they require special setup.
