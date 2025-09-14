#!/usr/bin/env python3
"""
Simplified demonstration of the comprehensive tracking system.

This script shows how to use the tracking system without external dependencies,
demonstrating all the key features for enhanced data lineage & reproducibility.
"""

import hashlib
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Simplified tracking system implementation
@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""

    experiment_name: str
    experiment_id: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("./runs"))

    # Tracking components to enable
    enable_dataset_tracking: bool = True
    enable_model_tracking: bool = True
    enable_environment_tracking: bool = True
    enable_seed_tracking: bool = True
    enable_git_tracking: bool = True

    # Output options
    save_to_json: bool = True
    save_to_yaml: bool = True
    save_to_wandb: bool = False

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.experiment_id is None:
            self.experiment_id = str(uuid.uuid4())

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DatasetTracker:
    """Tracks dataset versioning, checksums, and metadata."""

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.tracked_datasets: Dict[str, Dict[str, Any]] = {}

    def track_dataset(
        self,
        dataset: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a dataset and compute its fingerprint."""
        tracking_info = {
            "name": name,
            "type": type(dataset).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

        # Compute checksum based on dataset type
        if isinstance(dataset, (str, Path)):
            tracking_info.update(self._track_file_dataset(dataset))
        elif isinstance(dataset, list):
            tracking_info.update(self._track_list_dataset(dataset))
        else:
            tracking_info.update(self._track_generic_dataset(dataset))

        self.tracked_datasets[name] = tracking_info
        return tracking_info

    def _track_file_dataset(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Track a dataset stored as files."""
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        info = {
            "path": str(path.absolute()),
            "size_bytes": self._get_file_size(path),
            "file_type": path.suffix
        }

        if path.is_file():
            info["checksum"] = self._compute_file_checksum(path)
        elif path.is_dir():
            info["checksum"] = self._compute_directory_checksum(path)
            info["file_count"] = len(list(path.rglob("*")))

        return info

    def _track_list_dataset(self, dataset: list) -> Dict[str, Any]:
        """Track a list dataset."""
        return {
            "length": len(dataset),
            "checksum": self._compute_list_checksum(dataset)
        }

    def _track_generic_dataset(self, dataset: Any) -> Dict[str, Any]:
        """Track a generic dataset by serializing it."""
        try:
            serialized = str(dataset).encode()
            return {
                "size_bytes": len(serialized),
                "checksum": hashlib.sha256(serialized).hexdigest()
            }
        except Exception as e:
            return {
                "error": f"Could not serialize dataset: {str(e)}",
                "checksum": "unknown"
            }

    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute checksum of a single file."""
        hash_obj = hashlib.new(self.algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _compute_directory_checksum(self, dir_path: Path) -> str:
        """Compute checksum of a directory by hashing all files."""
        hash_obj = hashlib.new(self.algorithm)

        # Sort files for consistent ordering
        files = sorted(dir_path.rglob("*"))
        for file_path in files:
            if file_path.is_file():
                # Include relative path in hash
                rel_path = file_path.relative_to(dir_path)
                hash_obj.update(str(rel_path).encode())

                # Include file content
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _compute_list_checksum(self, dataset: list) -> str:
        """Compute checksum of a list."""
        data_str = str(sorted(dataset))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _get_file_size(self, path: Path) -> int:
        """Get total size of file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0


class ModelTracker:
    """Tracks model architecture, weights, and metadata."""

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.tracked_models: Dict[str, Dict[str, Any]] = {}

    def track_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a model and compute its fingerprint."""
        tracking_info = {
            "name": name,
            "type": type(model).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

        # Get model architecture info
        tracking_info.update(self._get_model_architecture_info(model))

        # Compute architecture fingerprint
        tracking_info["architecture_checksum"] = self._compute_architecture_checksum(model)

        self.tracked_models[name] = tracking_info
        return tracking_info

    def _get_model_architecture_info(self, model: Any) -> Dict[str, Any]:
        """Extract architecture information from a model."""
        info = {
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "model_repr": repr(model)
        }

        # Try to get parameter count if it's a model-like object
        if hasattr(model, '__dict__'):
            info["attributes"] = list(model.__dict__.keys())

        return info

    def _compute_architecture_checksum(self, model: Any) -> str:
        """Compute checksum of model architecture."""
        hash_obj = hashlib.new(self.algorithm)

        # Hash the model representation
        model_str = repr(model)
        hash_obj.update(model_str.encode())

        # Hash the model type and module
        hash_obj.update(type(model).__name__.encode())
        hash_obj.update(type(model).__module__.encode())

        return hash_obj.hexdigest()


class SeedTracker:
    """Tracks random seeds across all components for reproducibility."""

    def __init__(self):
        self.seed_info: Dict[str, Any] = {}

    def set_seeds(self, seed: int) -> Dict[str, Any]:
        """Set seeds for all components and track the operation."""
        seed_info = {
            "timestamp": datetime.now().isoformat(),
            "set_seed": seed,
            "seeds": {
                "python": {"seed": seed, "state": f"mock_state_{seed}"},
                "numpy": {"seed": seed, "state": f"mock_state_{seed}"},
                "torch": {"seed": seed, "state": f"mock_state_{seed}"},
                "cuda": {"available": False, "message": "CUDA not available"}
            }
        }

        # Compute seed fingerprint
        seed_info["seed_checksum"] = self._compute_seed_checksum(seed_info["seeds"])

        self.seed_info = seed_info
        return seed_info

    def _compute_seed_checksum(self, seeds: Dict[str, Any]) -> str:
        """Compute checksum of seed information."""
        hash_info = {}
        for component, info in seeds.items():
            if isinstance(info, dict):
                hash_info[component] = {
                    "seed": info.get("seed"),
                    "available": info.get("available", True)
                }
            else:
                hash_info[component] = str(info)

        # Convert to JSON string and hash
        json_str = json.dumps(hash_info, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class EnvironmentTracker:
    """Tracks environment state including dependencies and system info."""

    def __init__(self):
        self.tracking_info: Dict[str, Any] = {}

    def capture_environment(self) -> Dict[str, Any]:
        """Capture comprehensive environment information."""
        import platform

        env_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "system": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "hostname": platform.node(),
                "python_implementation": platform.python_implementation(),
            }
        }

        # Compute environment fingerprint
        env_info["environment_checksum"] = self._compute_environment_checksum(env_info)

        self.tracking_info = env_info
        return env_info

    def _compute_environment_checksum(self, env_info: Dict[str, Any]) -> str:
        """Compute checksum of environment information."""
        # Create a simplified version for hashing
        hash_info = {
            "python_version": env_info.get("python_version"),
            "system": env_info.get("system", {}).get("platform"),
        }

        # Convert to JSON string and hash
        json_str = json.dumps(hash_info, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class GitTracker:
    """Tracks Git repository state and changes."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.tracking_info: Dict[str, Any] = {}

    def capture_git_state(self) -> Dict[str, Any]:
        """Capture comprehensive Git repository state."""
        import subprocess

        git_info = {
            "timestamp": datetime.now().isoformat(),
            "repo_path": str(self.repo_path.absolute())
        }

        # Try to get basic git info
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info["commit"] = {
                    "hash": result.stdout.strip(),
                    "available": True
                }
            else:
                git_info["commit"] = {
                    "available": False,
                    "message": "not a git repository"
                }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            git_info["commit"] = {
                "available": False,
                "message": "git not available"
            }

        # Compute Git fingerprint
        git_info["git_checksum"] = self._compute_git_checksum(git_info)

        self.tracking_info = git_info
        return git_info

    def _compute_git_checksum(self, git_info: Dict[str, Any]) -> str:
        """Compute checksum of Git information."""
        hash_info = {
            "commit_hash": git_info.get("commit", {}).get("hash"),
            "repo_path": git_info.get("repo_path")
        }

        # Convert to JSON string and hash
        json_str = json.dumps(hash_info, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class ExperimentTracker:
    """Main experiment tracker that coordinates all tracking components."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.experiment_id = config.experiment_id
        self.experiment_name = config.experiment_name
        self.output_dir = config.output_dir

        # Initialize tracking components
        self.dataset_tracker = DatasetTracker() if config.enable_dataset_tracking else None
        self.model_tracker = ModelTracker() if config.enable_model_tracking else None
        self.environment_tracker = EnvironmentTracker() if config.enable_environment_tracking else None
        self.seed_tracker = SeedTracker() if config.enable_seed_tracking else None
        self.git_tracker = GitTracker() if config.enable_git_tracking else None

        # Storage for tracking data
        self.tracking_data: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self._serialize_config(),
            "datasets": {},
            "models": {},
            "environment": {},
            "seeds": {},
            "git": {},
            "metadata": config.metadata.copy(),
            "tags": config.tags.copy(),
            "notes": config.notes
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_experiment(self) -> Dict[str, Any]:
        """Start the experiment and capture initial state."""
        print(f"Starting experiment: {self.experiment_name} (ID: {self.experiment_id})")

        # Capture environment state
        if self.environment_tracker:
            print("Capturing environment state...")
            self.tracking_data["environment"] = self.environment_tracker.capture_environment()

        # Capture Git state
        if self.git_tracker:
            print("Capturing Git state...")
            self.tracking_data["git"] = self.git_tracker.capture_git_state()

        # Save initial state
        self._save_tracking_data()

        return self.tracking_data

    def track_dataset(
        self,
        dataset: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a dataset."""
        if not self.dataset_tracker:
            raise RuntimeError("Dataset tracking is not enabled")

        print(f"Tracking dataset: {name}")
        tracking_info = self.dataset_tracker.track_dataset(dataset, name, metadata)
        self.tracking_data["datasets"][name] = tracking_info
        self._save_tracking_data()

        return tracking_info

    def track_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a model."""
        if not self.model_tracker:
            raise RuntimeError("Model tracking is not enabled")

        print(f"Tracking model: {name}")
        tracking_info = self.model_tracker.track_model(model, name, metadata)
        self.tracking_data["models"][name] = tracking_info
        self._save_tracking_data()

        return tracking_info

    def set_seeds(self, seed: int) -> Dict[str, Any]:
        """Set seeds for reproducibility."""
        if not self.seed_tracker:
            raise RuntimeError("Seed tracking is not enabled")

        print(f"Setting seeds to: {seed}")
        seed_info = self.seed_tracker.set_seeds(seed)
        self.tracking_data["seeds"] = seed_info
        self._save_tracking_data()

        return seed_info

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the experiment."""
        self.tracking_data["metadata"][key] = value
        self._save_tracking_data()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self.tracking_data["tags"]:
            self.tracking_data["tags"].append(tag)
            self._save_tracking_data()

    def set_notes(self, notes: str) -> None:
        """Set notes for the experiment."""
        self.tracking_data["notes"] = notes
        self._save_tracking_data()

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracking data."""
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.tracking_data["timestamp"],
            "output_dir": str(self.output_dir),
            "datasets_tracked": len(self.tracking_data["datasets"]),
            "models_tracked": len(self.tracking_data["models"]),
            "has_environment": bool(self.tracking_data["environment"]),
            "has_git": bool(self.tracking_data["git"]),
            "has_seeds": bool(self.tracking_data["seeds"]),
            "tags": self.tracking_data["tags"],
            "metadata_keys": list(self.tracking_data["metadata"].keys())
        }

        return summary

    def _save_tracking_data(self) -> None:
        """Save tracking data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.save_to_json:
            json_path = self.output_dir / f"{self.experiment_name}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)

            # Also save a latest version
            latest_json_path = self.output_dir / f"{self.experiment_name}_latest.json"
            with open(latest_json_path, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)

    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def finish_experiment(self) -> Dict[str, Any]:
        """Finish the experiment and save final state."""
        print(f"Finishing experiment: {self.experiment_name}")

        # Update timestamp
        self.tracking_data["finished_at"] = datetime.now().isoformat()

        # Save final state
        self._save_tracking_data()

        # Print summary
        summary = self.get_tracking_summary()
        print("\n" + "="*50)
        print("EXPERIMENT TRACKING SUMMARY")
        print("="*50)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"ID: {summary['experiment_id']}")
        print(f"Output Directory: {summary['output_dir']}")
        print(f"Datasets Tracked: {summary['datasets_tracked']}")
        print(f"Models Tracked: {summary['models_tracked']}")
        print(f"Environment Captured: {summary['has_environment']}")
        print(f"Git State Captured: {summary['has_git']}")
        print(f"Seeds Tracked: {summary['has_seeds']}")
        print(f"Tags: {', '.join(summary['tags']) if summary['tags'] else 'None'}")
        print("="*50)

        return summary


# Mock model classes for demonstration
class SimpleModel:
    """Simple model for demonstration."""

    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = ["linear", "relu", "linear"]

    def __repr__(self):
        return f"SimpleModel(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"


class LargeModel:
    """Large model for demonstration."""

    def __init__(self, num_layers=10, layer_size=1000):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.layers = [f"layer_{i}" for i in range(num_layers)]
        self.parameters = num_layers * layer_size * layer_size  # Mock parameter count

    def __repr__(self):
        return f"LargeModel(num_layers={self.num_layers}, layer_size={self.layer_size}, parameters={self.parameters})"


def main():
    """Main demonstration function."""
    print("="*60)
    print("COMPREHENSIVE TRACKING SYSTEM DEMONSTRATION")
    print("="*60)

    # Configuration
    experiment_name = "ml_demo_simple"
    output_dir = Path("./tracking_demo_output")
    random_seed = 42

    # Create tracking configuration
    config = TrackingConfig(
        experiment_name=experiment_name,
        output_dir=output_dir,
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_json=True,
        save_to_yaml=False,  # Skip YAML for simplicity
        save_to_wandb=False,
        tags=["classification", "demo", "simple"],
        notes="Demonstration of comprehensive tracking system for ML pipeline"
    )

    # Initialize experiment tracker
    tracker = ExperimentTracker(config)

    print(f"\n1. Starting experiment: {experiment_name}")
    print(f"   Output directory: {output_dir}")
    print(f"   Experiment ID: {tracker.experiment_id}")

    # Start experiment (captures environment, git, and initial seeds)
    tracker.start_experiment()

    print("\n2. Creating reproducible environment...")
    # Set seeds for reproducibility
    tracker.set_seeds(random_seed)

    print("\n3. Creating and tracking datasets...")
    # Create mock datasets
    train_data = [1, 2, 3, 4, 5] * 100  # Mock training data
    test_data = [6, 7, 8, 9, 10] * 20   # Mock test data

    # Track datasets
    tracker.track_dataset(
        train_data, "train_data",
        metadata={
            "split": "train",
            "n_samples": len(train_data),
            "data_type": "features"
        }
    )

    tracker.track_dataset(
        test_data, "test_data",
        metadata={
            "split": "test",
            "n_samples": len(test_data),
            "data_type": "features"
        }
    )

    print("\n4. Creating and tracking models...")
    # Create models
    simple_model = SimpleModel(input_size=10, hidden_size=5, output_size=1)
    large_model = LargeModel(num_layers=10, layer_size=1000)

    # Track models
    simple_model_info = tracker.track_model(
        simple_model, "simple_classifier",
        metadata={
            "task": "classification",
            "architecture": "simple",
            "input_size": 10,
            "hidden_size": 5,
            "output_size": 1
        }
    )

    large_model_info = tracker.track_model(
        large_model, "large_classifier",
        metadata={
            "task": "classification",
            "architecture": "large",
            "num_layers": 10,
            "layer_size": 1000,
            "parameters": large_model.parameters
        }
    )

    print(f"   Simple model checksum: {simple_model_info['architecture_checksum'][:16]}...")
    print(f"   Large model checksum: {large_model_info['architecture_checksum'][:16]}...")

    print("\n5. Adding training metadata...")
    # Add training metadata
    tracker.add_metadata("optimizer", "Adam")
    tracker.add_metadata("loss_function", "CrossEntropyLoss")
    tracker.add_metadata("learning_rate", 0.001)
    tracker.add_metadata("batch_size", 32)
    tracker.add_metadata("num_epochs", 10)
    tracker.add_metadata("final_accuracy", 0.95)

    # Add tags
    tracker.add_tag("binary_classification")
    tracker.add_tag("pytorch")
    tracker.add_tag("test_run")
    tracker.add_tag("reproducible")

    # Set notes
    tracker.set_notes(
        f"Simple ML demo with {len(train_data)} training samples, "
        f"final accuracy: 0.95"
    )

    print("\n6. Finishing experiment...")
    # Finish experiment and get summary
    summary = tracker.finish_experiment()

    print("\n" + "="*60)
    print("FILES CREATED")
    print("="*60)

    # List created files
    output_files = list(output_dir.glob("*"))
    for file_path in sorted(output_files):
        if file_path.is_file():
            size = file_path.stat().st_size
            print(f"  {file_path.name} ({size:,} bytes)")

    print("\n" + "="*60)
    print("REPRODUCIBILITY INFORMATION")
    print("="*60)

    # Show key reproducibility info
    if summary['has_seeds']:
        print("✓ Random seeds captured and set")
    if summary['has_environment']:
        print("✓ Environment state captured")
    if summary['has_git']:
        print("✓ Git repository state captured")
    if summary['models_tracked'] > 0:
        print("✓ Model architecture fingerprinted")
    if summary['datasets_tracked'] > 0:
        print("✓ Dataset checksums computed")

    print("\nTo reproduce this experiment:")
    print(f"1. Use the same seed: {random_seed}")
    print(f"2. Check environment in: {output_dir}/ml_demo_simple_latest.json")
    print("3. Verify Git state in tracking files")
    print("4. Use the same model architecture (checksum available)")
    print("5. Verify dataset checksums match")

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

    # Show a sample of the tracking data
    print("\nSample tracking data:")
    with open(output_dir / "ml_demo_simple_latest.json") as f:
        data = json.load(f)
        print(f"  Experiment ID: {data['experiment_id']}")
        print(f"  Datasets: {list(data['datasets'].keys())}")
        print(f"  Models: {list(data['models'].keys())}")
        print(f"  Environment checksum: {data['environment']['environment_checksum'][:16]}...")
        print(f"  Git checksum: {data['git']['git_checksum'][:16]}...")
        print(f"  Seed checksum: {data['seeds']['seed_checksum'][:16]}...")


if __name__ == "__main__":
    main()
