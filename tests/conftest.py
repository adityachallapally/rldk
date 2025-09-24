"""Pytest configuration and fixtures for RL Debug Kit tests."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:  # pragma: no cover - defensive for older numpy installs
    import numpy as _np

    if not isinstance(getattr(_np, "bool_", bool), type):
        _np.bool_ = bool  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - numpy is an optional dependency
    pass

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add tools directory to Python path for profiler imports
tools_path = project_root / "tools"
if str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.DataFrame({
        "step": range(100),
        "reward_mean": np.random.normal(0.5, 0.2, 100),
        "reward_std": np.random.uniform(0.1, 0.3, 100),
        "tokens_out": np.random.randint(10, 100, 100),
        "repetition_penalty": np.random.uniform(0.8, 1.2, 100),
        "human_preference": np.random.uniform(0, 1, 100),
        "ground_truth": np.random.choice([0, 1], 100),
        "epoch": np.random.randint(0, 10, 100),
        "run_id": ["test_run"] * 100,
    })


@pytest.fixture
def mock_torch():
    """Mock torch module for tests that don't need real PyTorch."""
    with patch.dict('sys.modules', {'torch': MagicMock()}):
        yield


@pytest.fixture
def mock_datasets():
    """Mock datasets module for tests."""
    mock_datasets = MagicMock()
    mock_datasets.Dataset = MagicMock()
    with patch.dict('sys.modules', {'datasets': mock_datasets}):
        yield mock_datasets


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    from rldk.utils.seed import reset_global_seed

    reset_global_seed()

    # Set environment variables for testing
    os.environ["RLDK_TEST_MODE"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    original_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = [str(src_path)]
    if original_pythonpath:
        pythonpath_parts.append(original_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    yield

    # Cleanup after test
    if "RLDK_TEST_MODE" in os.environ:
        del os.environ["RLDK_TEST_MODE"]
    if "WANDB_MODE" in os.environ:
        del os.environ["WANDB_MODE"]
    if original_pythonpath is None:
        del os.environ["PYTHONPATH"]
    else:
        os.environ["PYTHONPATH"] = original_pythonpath

    reset_global_seed()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that run quickly without external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require network access or external services"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests that test complete workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run (e.g., model downloads)"
    )
    config.addinivalue_line(
        "markers", "trl: Tests related to TRL integration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
