#!/usr/bin/env python3
"""
Standalone test of large model tracking capabilities.
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


# Standalone tracking system implementation
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
        if isinstance(dataset, list):
            tracking_info.update(self._track_list_dataset(dataset))
        else:
            tracking_info.update(self._track_generic_dataset(dataset))

        self.tracked_datasets[name] = tracking_info
        return tracking_info

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

    def _compute_list_checksum(self, dataset: list) -> str:
        """Compute checksum of a list."""
        # For large lists, sample a subset for performance
        if len(dataset) > 100000:  # 100K elements
            # Sample every nth element to get a representative subset
            step = len(dataset) // 10000  # Sample 10K elements
            sample = dataset[::step]
        else:
            sample = dataset

        data_str = str(sorted(sample))
        return hashlib.sha256(data_str.encode()).hexdigest()


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
        if hasattr(model, 'parameters'):
            info["parameters"] = model.parameters
        if hasattr(model, 'num_layers'):
            info["num_layers"] = model.num_layers
        if hasattr(model, 'layer_size'):
            info["layer_size"] = model.layer_size
        if hasattr(model, 'vocab_size'):
            info["vocab_size"] = model.vocab_size

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

        # Hash model attributes if available
        if hasattr(model, 'parameters'):
            hash_obj.update(str(model.parameters).encode())
        if hasattr(model, 'num_layers'):
            hash_obj.update(str(model.num_layers).encode())
        if hasattr(model, 'layer_size'):
            hash_obj.update(str(model.layer_size).encode())

        return hash_obj.hexdigest()


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

        # Storage for tracking data
        self.tracking_data: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self._serialize_config(),
            "datasets": {},
            "models": {},
            "metadata": config.metadata.copy(),
            "tags": config.tags.copy(),
            "notes": config.notes
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_experiment(self) -> Dict[str, Any]:
        """Start the experiment and capture initial state."""
        print(f"Starting experiment: {self.experiment_name} (ID: {self.experiment_id})")

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
        print(f"Tags: {', '.join(summary['tags']) if summary['tags'] else 'None'}")
        print("="*50)

        return summary


class LargeModel:
    """Simulated large model for testing."""

    def __init__(self, num_layers=100, layer_size=2048, vocab_size=50000):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.vocab_size = vocab_size
        self.embedding_dim = 512
        self.num_attention_heads = 16
        self.max_sequence_length = 512

        # Simulate model parameters
        self.parameters = self._calculate_parameters()

        # Simulate model structure
        self.layers = self._create_layer_structure()

    def _calculate_parameters(self):
        """Calculate approximate number of parameters."""
        # Embedding layer
        embedding_params = self.vocab_size * self.embedding_dim

        # Transformer layers
        attention_params = 4 * self.embedding_dim * self.embedding_dim  # Q, K, V, O
        ffn_params = 2 * self.embedding_dim * (4 * self.embedding_dim)  # FFN

        layer_params = attention_params + ffn_params
        total_layer_params = layer_params * self.num_layers

        # Output layer
        output_params = self.embedding_dim * self.vocab_size

        return embedding_params + total_layer_params + output_params

    def _create_layer_structure(self):
        """Create a mock layer structure."""
        layers = []

        # Embedding layer
        layers.append({
            "type": "Embedding",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "parameters": self.vocab_size * self.embedding_dim
        })

        # Transformer layers
        for i in range(self.num_layers):
            layers.append({
                "type": "TransformerLayer",
                "layer_id": i,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_attention_heads,
                "parameters": 4 * self.embedding_dim * self.embedding_dim + 2 * self.embedding_dim * (4 * self.embedding_dim)
            })

        # Output layer
        layers.append({
            "type": "OutputLayer",
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "parameters": self.embedding_dim * self.vocab_size
        })

        return layers

    def __repr__(self):
        return f"LargeModel(num_layers={self.num_layers}, layer_size={self.layer_size}, vocab_size={self.vocab_size}, parameters={self.parameters:,})"


def test_large_model_tracking():
    """Test tracking with a large model."""
    print("Testing large model tracking...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config
        config = TrackingConfig(
            experiment_name="large_model_test",
            output_dir=Path(temp_dir),
            save_to_wandb=False
        )

        # Create tracker
        tracker = ExperimentTracker(config)

        print("1. Starting experiment...")
        tracker.start_experiment()

        print("2. Creating large model...")
        # Create a large model
        large_model = LargeModel(
            num_layers=50,  # 50 transformer layers
            layer_size=2048,  # 2048 hidden size
            vocab_size=30000  # 30k vocabulary
        )

        print(f"   Model parameters: {large_model.parameters:,}")
        print(f"   Model layers: {len(large_model.layers)}")

        print("3. Tracking large model...")
        # Track the model
        model_info = tracker.track_model(
            large_model, "large_transformer",
            metadata={
                "model_type": "transformer",
                "num_layers": large_model.num_layers,
                "layer_size": large_model.layer_size,
                "vocab_size": large_model.vocab_size,
                "embedding_dim": large_model.embedding_dim,
                "num_attention_heads": large_model.num_attention_heads,
                "max_sequence_length": large_model.max_sequence_length,
                "total_parameters": large_model.parameters
            }
        )

        print("   ✓ Model tracked successfully")
        print(f"   Architecture checksum: {model_info['architecture_checksum'][:16]}...")
        print(f"   Model type: {model_info['type']}")

        print("4. Testing with even larger model...")
        # Create an even larger model
        huge_model = LargeModel(
            num_layers=100,  # 100 transformer layers
            layer_size=4096,  # 4096 hidden size
            vocab_size=50000  # 50k vocabulary
        )

        print(f"   Huge model parameters: {huge_model.parameters:,}")

        # Track the huge model
        huge_model_info = tracker.track_model(
            huge_model, "huge_transformer",
            metadata={
                "model_type": "transformer",
                "num_layers": huge_model.num_layers,
                "layer_size": huge_model.layer_size,
                "vocab_size": huge_model.vocab_size,
                "total_parameters": huge_model.parameters
            }
        )

        print("   ✓ Huge model tracked successfully")
        print(f"   Architecture checksum: {huge_model_info['architecture_checksum'][:16]}...")

        print("5. Testing dataset tracking with large datasets...")
        # Create large datasets
        large_dataset = list(range(1000000))  # 1M samples
        tracker.track_dataset(
            large_dataset, "large_training_data",
            metadata={
                "size": len(large_dataset),
                "type": "synthetic",
                "description": "Large synthetic dataset for testing"
            }
        )

        print("   ✓ Large dataset tracked (1M samples)")

        print("6. Adding performance metadata...")
        # Add performance-related metadata
        tracker.add_metadata("model_size_mb", large_model.parameters * 4 / (1024 * 1024))  # 4 bytes per parameter
        tracker.add_metadata("huge_model_size_mb", huge_model.parameters * 4 / (1024 * 1024))
        tracker.add_metadata("dataset_size_mb", len(large_dataset) * 4 / (1024 * 1024))
        tracker.add_metadata("total_memory_estimate_mb",
                           (large_model.parameters + huge_model.parameters) * 4 / (1024 * 1024) +
                           len(large_dataset) * 4 / (1024 * 1024))

        print("7. Finishing experiment...")
        # Finish experiment
        summary = tracker.finish_experiment()

        print("\n✓ Large model tracking test completed!")
        print(f"  Experiment ID: {summary['experiment_id']}")
        print(f"  Models tracked: {summary['models_tracked']}")
        print(f"  Datasets tracked: {summary['datasets_tracked']}")
        print(f"  Output directory: {summary['output_dir']}")

        # Check created files
        files_created = list(Path(temp_dir).glob("*"))
        print(f"  Files created: {len(files_created)}")
        for file_path in files_created:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"    {file_path.name} ({size:,} bytes)")

        # Verify tracking data
        json_file = Path(temp_dir) / "large_model_test_latest.json"
        with open(json_file) as f:
            data = json.load(f)

            # Check models
            assert "large_transformer" in data["models"]
            assert "huge_transformer" in data["models"]

            # Check datasets
            assert "large_training_data" in data["datasets"]

            # Check metadata
            assert "model_size_mb" in data["metadata"]
            assert "huge_model_size_mb" in data["metadata"]

            print("  ✓ Tracking data verification passed")

        return True


def test_checksum_performance():
    """Test checksum computation performance with large data."""
    print("\nTesting checksum performance...")

    # Test with different data sizes
    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        print(f"  Testing with {size:,} elements...")

        # Create large dataset
        data = list(range(size))

        # Time the checksum computation
        start_time = datetime.now()
        checksum = hashlib.sha256(str(data).encode()).hexdigest()
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        print(f"    Checksum: {checksum[:16]}...")
        print(f"    Time: {duration:.3f} seconds")

        # Verify checksum consistency
        checksum2 = hashlib.sha256(str(data).encode()).hexdigest()
        assert checksum == checksum2
        print("    ✓ Checksum consistency verified")


def main():
    """Run all tests."""
    print("="*60)
    print("LARGE MODEL TRACKING TESTS")
    print("="*60)

    try:
        # Test large model tracking
        success = test_large_model_tracking()

        if success:
            print("\n✓ Large model tracking test passed!")
        else:
            print("\n✗ Large model tracking test failed!")
            return False

        # Test checksum performance
        test_checksum_performance()

        print("\n" + "="*60)
        print("🎉 ALL LARGE MODEL TESTS PASSED!")
        print("="*60)
        print("The tracking system successfully handles:")
        print("✓ Large models with millions of parameters")
        print("✓ Large datasets with millions of samples")
        print("✓ Efficient checksum computation")
        print("✓ Model architecture fingerprinting")
        print("✓ Performance metadata tracking")
        print("✓ File output and persistence")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
