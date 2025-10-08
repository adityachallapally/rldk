"""
Comprehensive tests for the tracking system.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Import tracking components
from rldk.tracking import (
    DatasetTracker,
    EnvironmentTracker,
    ExperimentTracker,
    GitTracker,
    ModelTracker,
    SeedTracker,
    TrackingConfig,
)


class TestTrackingConfig:
    """Test TrackingConfig class."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = TrackingConfig(
            experiment_name="test_experiment",
            output_dir=Path("./test_output")
        )

        assert config.experiment_name == "test_experiment"
        assert config.experiment_id is not None
        assert config.output_dir == Path("./test_output")
        assert config.enable_dataset_tracking is True
        assert config.enable_model_tracking is True

    def test_config_accepts_string_paths(self, tmp_path):
        """Test that string paths are converted to Path instances."""
        output_dir = tmp_path / "run_output"
        dataset_cache_dir = tmp_path / "cache_dir"
        git_repo_path = tmp_path

        config = TrackingConfig(
            experiment_name="string_path_experiment",
            output_dir=str(output_dir),
            dataset_cache_dir=str(dataset_cache_dir),
            git_repo_path=str(git_repo_path),
        )

        assert config.output_dir == output_dir
        assert config.dataset_cache_dir == dataset_cache_dir
        assert config.git_repo_path == git_repo_path
        assert output_dir.exists()
        assert dataset_cache_dir.exists()

    def test_config_defaults(self):
        """Test config default values."""
        config = TrackingConfig(experiment_name="test")

        assert config.dataset_checksum_algorithm == "sha256"
        assert config.model_fingerprint_algorithm == "sha256"
        assert config.capture_conda_env is True
        assert config.capture_pip_freeze is True
        assert config.track_numpy_seed is True
        assert config.track_torch_seed is True


class TestDatasetTracker:
    """Test DatasetTracker class."""

    def test_numpy_dataset_tracking(self):
        """Test tracking NumPy arrays."""
        tracker = DatasetTracker()

        # Create test data
        data = np.random.randn(100, 10)

        # Track the dataset
        info = tracker.track_dataset(data, "test_numpy")

        assert info["name"] == "test_numpy"
        assert info["type"] == "ndarray"
        assert info["shape"] == (100, 10)
        assert info["dtype"] == "float64"
        assert "checksum" in info
        assert info["checksum"] is not None

    def test_pandas_dataset_tracking(self):
        """Test tracking Pandas DataFrames."""
        tracker = DatasetTracker()

        # Create test data
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randint(0, 10, 100),
            'C': ['text'] * 100
        })

        # Track the dataset
        info = tracker.track_dataset(data, "test_pandas")

        assert info["name"] == "test_pandas"
        assert info["type"] == "DataFrame"
        assert info["shape"] == (100, 3)
        assert info["columns"] == ['A', 'B', 'C']
        assert "checksum" in info

    def test_torch_dataset_tracking(self):
        """Test tracking PyTorch datasets."""
        tracker = DatasetTracker()

        # Create test data
        data = torch.randn(100, 10)
        targets = torch.randint(0, 5, (100,))
        dataset = torch.utils.data.TensorDataset(data, targets)

        # Track the dataset
        info = tracker.track_dataset(dataset, "test_torch")

        assert info["name"] == "test_torch"
        assert info["type"] == "TensorDataset"
        assert info["num_samples"] == 100
        assert "checksum" in info

    def test_file_dataset_tracking(self):
        """Test tracking file-based datasets."""
        tracker = DatasetTracker()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Hello, World!")

            # Track the file
            info = tracker.track_dataset(test_file, "test_file")

            assert info["name"] == "test_file"
            assert info["path"] == str(test_file.absolute())
            assert info["size_bytes"] > 0
            assert "checksum" in info

    def test_checksum_consistency(self):
        """Test that checksums are consistent for identical data."""
        tracker = DatasetTracker()

        # Create identical datasets
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([1, 2, 3, 4, 5])

        info1 = tracker.track_dataset(data1, "test1")
        info2 = tracker.track_dataset(data2, "test2")

        assert info1["checksum"] == info2["checksum"]


class TestModelTracker:
    """Test ModelTracker class."""

    def test_simple_model_tracking(self):
        """Test tracking a simple PyTorch model."""
        tracker = ModelTracker()

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        # Track the model
        info = tracker.track_model(model, "test_model")

        assert info["name"] == "test_model"
        assert info["type"] == "Sequential"
        assert info["num_parameters"] > 0
        assert info["num_trainable_parameters"] > 0
        assert "architecture_checksum" in info
        assert info["architecture_checksum"] is not None

    def test_model_architecture_info(self):
        """Test model architecture information extraction."""
        tracker = ModelTracker()

        # Create a model with known structure
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        info = tracker.track_model(model, "test_model")

        # Check layer information
        assert len(info["layers"]) > 0
        assert any(layer["type"] == "Linear" for layer in info["layers"])
        assert any(layer["type"] == "ReLU" for layer in info["layers"])

        # Check structure
        assert "Sequential" in info["structure"]
        assert "Linear" in info["structure"]

    def test_model_weights_tracking(self):
        """Test tracking model weights."""
        tracker = ModelTracker()

        # Create a simple model
        model = nn.Linear(5, 3)

        # Track with weights
        info = tracker.track_model(model, "test_model", save_weights=True)

        assert "weights_checksum" in info
        assert "weights_size_bytes" in info
        assert info["weights_size_bytes"] > 0

    def test_architecture_checksum_consistency(self):
        """Test that architecture checksums are consistent for identical models."""
        tracker = ModelTracker()

        # Create identical models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        info1 = tracker.track_model(model1, "model1")
        info2 = tracker.track_model(model2, "model2")

        assert info1["architecture_checksum"] == info2["architecture_checksum"]


class TestEnvironmentTracker:
    """Test EnvironmentTracker class."""

    def test_environment_capture(self):
        """Test environment information capture."""
        tracker = EnvironmentTracker()

        # Capture environment
        info = tracker.capture_environment()

        assert "timestamp" in info
        assert "python_version" in info
        assert "python_executable" in info
        assert "pip" in info
        assert "system" in info
        assert "ml_frameworks" in info
        assert "environment_checksum" in info

    def test_ml_frameworks_capture(self):
        """Test ML framework information capture."""
        tracker = EnvironmentTracker()

        info = tracker.capture_environment()
        frameworks = info["ml_frameworks"]

        # Check that key frameworks are captured
        assert "torch" in frameworks
        assert "numpy" in frameworks

        # Check PyTorch info
        torch_info = frameworks["torch"]
        assert "version" in torch_info
        assert "cuda_available" in torch_info

    def test_system_info_capture(self):
        """Test system information capture."""
        tracker = EnvironmentTracker()

        info = tracker.capture_environment()
        system = info["system"]

        assert "platform" in system
        assert "system" in system
        assert "machine" in system
        assert "python_implementation" in system


class TestSeedTracker:
    """Test SeedTracker class."""

    def test_seed_capture(self):
        """Test seed state capture."""
        tracker = SeedTracker()

        # Capture seeds
        info = tracker.capture_seeds()

        assert "timestamp" in info
        assert "seeds" in info
        assert "seed_checksum" in info

        seeds = info["seeds"]
        assert "python" in seeds
        assert "numpy" in seeds
        assert "torch" in seeds

    def test_seed_setting(self):
        """Test seed setting."""
        tracker = SeedTracker()

        # Set seeds
        seed = 42
        info = tracker.set_seeds(seed)

        assert info["set_seed"] == seed
        assert info["seeds"]["python"]["seed"] == seed
        assert info["seeds"]["numpy"]["seed"] == seed
        assert info["seeds"]["torch"]["seed"] == seed

    def test_reproducible_environment(self):
        """Test creating reproducible environment."""
        tracker = SeedTracker()

        # Create reproducible environment
        seed = 123
        info = tracker.create_reproducible_environment(seed)

        assert info["set_seed"] == seed
        assert "reproducibility_settings" in info
        assert info["reproducibility_settings"]["PYTHONHASHSEED"] == str(seed)

    def test_seed_state_save_load(self):
        """Test saving and loading seed state."""
        tracker = SeedTracker()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set seeds
            tracker.set_seeds(42)

            # Save state
            save_path = Path(temp_dir) / "seed_state.json"
            tracker.save_seed_state(str(save_path))

            # Change seeds
            tracker.set_seeds(100)

            # Load state
            loaded_info = tracker.load_seed_state(str(save_path))

            assert loaded_info["seed_info"]["set_seed"] == 42


class TestGitTracker:
    """Test GitTracker class."""

    def test_git_tracker_initialization(self):
        """Test GitTracker initialization."""
        tracker = GitTracker()

        assert tracker.repo_path == Path.cwd()

    @patch('subprocess.run')
    def test_git_state_capture(self, mock_run):
        """Test Git state capture with mocked subprocess."""
        # Mock subprocess responses
        mock_responses = [
            # git rev-parse HEAD
            MagicMock(returncode=0, stdout="abc123def456\n"),
            # git rev-parse --short HEAD
            MagicMock(returncode=0, stdout="abc123d\n"),
            # git log -1 --pretty=format:%s
            MagicMock(returncode=0, stdout="Test commit message\n"),
            # git log -1 --pretty=format:%an|%ae|%ad
            MagicMock(returncode=0, stdout="Test Author|test@example.com|2023-01-01 12:00:00 +0000\n"),
            # git branch --show-current
            MagicMock(returncode=0, stdout="main\n"),
            # git describe --tags --exact-match
            MagicMock(returncode=1, stdout=""),  # No exact tag match
        ]

        mock_run.side_effect = mock_responses

        tracker = GitTracker()
        info = tracker.capture_git_state()

        assert "commit" in info
        assert info["commit"]["hash"] == "abc123def456"
        assert info["commit"]["short_hash"] == "abc123d"
        assert info["commit"]["message"] == "Test commit message"
        assert info["commit"]["branch"] == "main"

    def test_is_git_repo(self):
        """Test Git repository detection."""
        tracker = GitTracker()

        # This will depend on whether we're in a git repo
        # Just test that the method doesn't crash
        result = tracker.is_git_repo()
        assert isinstance(result, bool)


class TestExperimentTracker:
    """Test ExperimentTracker class."""

    def test_experiment_tracker_initialization(self):
        """Test ExperimentTracker initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="test_experiment",
                output_dir=Path(temp_dir)
            )

            tracker = ExperimentTracker(config)

            assert tracker.experiment_name == "test_experiment"
            assert tracker.experiment_id == config.experiment_id
            assert tracker.output_dir == Path(temp_dir)

    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="test_experiment",
                output_dir=Path(temp_dir),
                save_to_wandb=False  # Disable wandb for testing
            )

            tracker = ExperimentTracker(config)

            # Start experiment
            start_info = tracker.start_experiment()
            assert "experiment_id" in start_info
            assert "environment" in start_info
            assert "git" in start_info
            assert "seeds" in start_info

            # Track a dataset
            data = np.random.randn(100, 10)
            dataset_info = tracker.track_dataset(data, "test_dataset")
            assert dataset_info["name"] == "test_dataset"

            # Track a model
            model = nn.Linear(10, 1)
            model_info = tracker.track_model(model, "test_model")
            assert model_info["name"] == "test_model"

            # Set seeds
            seed_info = tracker.set_seeds(42)
            assert seed_info["set_seed"] == 42

            # Add metadata
            tracker.add_metadata("test_key", "test_value")
            tracker.add_tag("test_tag")
            tracker.set_notes("Test notes")

            # Finish experiment
            summary = tracker.finish_experiment()
            assert summary["experiment_name"] == "test_experiment"
            assert summary["datasets_tracked"] == 1
            assert summary["models_tracked"] == 1
            assert "test_tag" in summary["tags"]

    def test_tracking_data_persistence(self):
        """Test that tracking data is saved to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="test_experiment",
                output_dir=Path(temp_dir),
                save_to_json=True,
                save_to_yaml=True,
                save_to_wandb=False
            )

            tracker = ExperimentTracker(config)
            tracker.start_experiment()

            # Finish to ensure final state is persisted
            tracker.finish_experiment()

            # Check that files are created
            json_files = list(Path(temp_dir).glob("*.json"))
            yaml_files = list(Path(temp_dir).glob("*.yaml"))

            assert len(json_files) > 0
            assert len(yaml_files) > 0

            # Check that latest files exist
            latest_json = Path(temp_dir) / "test_experiment_latest.json"
            latest_yaml = Path(temp_dir) / "test_experiment_latest.yaml"

            assert latest_json.exists()
            assert latest_yaml.exists()

            # Check canonical files for compatibility with standalone scripts
            canonical_json = Path(temp_dir) / "experiment.json"
            canonical_yaml = Path(temp_dir) / "experiment.yaml"

            assert canonical_json.exists()
            assert canonical_yaml.exists()

            # Check file contents
            with open(latest_json) as f:
                data = json.load(f)
                assert data["experiment_name"] == "test_experiment"
                assert "environment" in data
                assert "git" in data
                assert "seeds" in data

            with open(canonical_json) as f:
                canonical_data = json.load(f)
                assert canonical_data["experiment_name"] == "test_experiment"
                assert canonical_data["models"] == data["models"]

    def test_default_wandb_configuration(self):
        """Test that W&B is enabled by default."""
        config = TrackingConfig(experiment_name="test_experiment")

        # Check that W&B is enabled by default
        assert config.save_to_wandb
        assert config.wandb_project == "rldk-experiments"

    def test_tracking_summary(self):
        """Test tracking summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="test_experiment",
                output_dir=Path(temp_dir),
                save_to_wandb=False  # Disable for testing
            )

            tracker = ExperimentTracker(config)
            tracker.start_experiment()

            # Track some data
            data = np.random.randn(50, 5)
            tracker.track_dataset(data, "test_dataset")

            model = nn.Linear(5, 1)
            tracker.track_model(model, "test_model")

            # Get summary
            summary = tracker.get_tracking_summary()

            assert summary["experiment_name"] == "test_experiment"
            assert summary["datasets_tracked"] == 1
            assert summary["models_tracked"] == 1
            assert summary["has_environment"] is True
            assert summary["has_git"] is True
            assert summary["has_seeds"] is True


class TestIntegration:
    """Integration tests for the tracking system."""

    def test_full_ml_pipeline_tracking(self):
        """Test tracking a complete ML pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="ml_pipeline_test",
                output_dir=Path(temp_dir),
                save_to_wandb=False
            )

            tracker = ExperimentTracker(config)

            # Start experiment
            tracker.start_experiment()

            # Create reproducible environment
            tracker.create_reproducible_environment(42)

            # Track training data
            X_train = np.random.randn(1000, 20)
            y_train = np.random.randint(0, 2, 1000)
            tracker.track_dataset(X_train, "X_train", {"split": "train", "features": 20})
            tracker.track_dataset(y_train, "y_train", {"split": "train", "classes": 2})

            # Track test data
            X_test = np.random.randn(200, 20)
            y_test = np.random.randint(0, 2, 200)
            tracker.track_dataset(X_test, "X_test", {"split": "test", "features": 20})
            tracker.track_dataset(y_test, "y_test", {"split": "test", "classes": 2})

            # Create and track model
            model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

            tracker.track_model(model, "classifier", {
                "task": "binary_classification",
                "architecture": "feedforward",
                "layers": 7
            })

            # Simulate training
            torch.optim.Adam(model.parameters())
            nn.BCELoss()

            # Track training metadata
            tracker.add_metadata("optimizer", "Adam")
            tracker.add_metadata("loss_function", "BCELoss")
            tracker.add_metadata("learning_rate", 0.001)
            tracker.add_metadata("batch_size", 32)
            tracker.add_metadata("epochs", 10)

            # Add tags
            tracker.add_tag("binary_classification")
            tracker.add_tag("pytorch")
            tracker.add_tag("test_run")

            # Set notes
            tracker.set_notes("Test ML pipeline with comprehensive tracking")

            # Finish experiment
            summary = tracker.finish_experiment()

            # Verify tracking
            assert summary["datasets_tracked"] == 4
            assert summary["models_tracked"] == 1
            assert "binary_classification" in summary["tags"]
            assert "pytorch" in summary["tags"]
            assert summary["metadata_keys"] == ["optimizer", "loss_function", "learning_rate", "batch_size", "epochs"]

            # Check that all files were created
            assert (Path(temp_dir) / "ml_pipeline_test_latest.json").exists()
            assert (Path(temp_dir) / "ml_pipeline_test_latest.yaml").exists()

    def test_large_model_tracking(self):
        """Test tracking a larger model to verify performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="large_model_test",
                output_dir=Path(temp_dir),
                save_model_weights=False,  # Don't save weights for large models
                save_to_wandb=False
            )

            tracker = ExperimentTracker(config)
            tracker.start_experiment()

            # Create a larger model
            model = nn.Sequential(
                nn.Linear(1000, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

            # Track the model
            model_info = tracker.track_model(model, "large_model")

            # Verify tracking
            assert model_info["num_parameters"] > 100000  # Should have many parameters
            assert "architecture_checksum" in model_info
            assert model_info["architecture_checksum"] is not None

            # Check that architecture file was created
            arch_file = Path(temp_dir) / "large_model_architecture.txt"
            assert arch_file.exists()

            # Verify architecture file content
            with open(arch_file) as f:
                content = f.read()
                assert "Sequential" in content
                assert "Linear" in content
                assert "ReLU" in content

    def test_metric_logging_persists_and_appends(self):
        """Ensure metric logging updates tracking data and persists to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrackingConfig(
                experiment_name="metric_logging_test",
                output_dir=Path(temp_dir),
                save_to_wandb=False,
            )

            tracker = ExperimentTracker(config)

            first_entry = tracker.log_metric("accuracy", 0.91, step=1, timestamp=datetime(2024, 1, 1, 12, 0, 0))
            assert first_entry["name"] == "accuracy"
            assert first_entry["value"] == 0.91
            assert first_entry["step"] == 1
            assert first_entry["timestamp"].startswith("2024-01-01T12:00:00")

            multiple_entries = tracker.log_metrics({"loss": 0.12, "precision": 0.78}, step=2)
            assert len(multiple_entries["metrics"]) == 2
            assert tracker.tracking_data["metrics"][-1]["name"] == "precision"

            canonical_path = Path(temp_dir) / "experiment.json"
            assert canonical_path.exists()

            with canonical_path.open() as fp:
                persisted_data = json.load(fp)

            metric_names = [entry["name"] for entry in persisted_data["metrics"]]
            assert metric_names.count("accuracy") == 1
            assert metric_names.count("loss") == 1
            assert metric_names.count("precision") == 1

            loss_entry = next(entry for entry in persisted_data["metrics"] if entry["name"] == "loss")
            assert loss_entry["step"] == 2
            assert isinstance(loss_entry["timestamp"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
