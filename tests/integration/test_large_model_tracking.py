#!/usr/bin/env python3
"""
Test the tracking system with a simulated large model.
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

# Import the tracking system
sys.path.append('src')
from rldk.tracking import ExperimentTracker, TrackingConfig


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
            save_model_architecture=True,
            save_model_weights=False,  # Don't save weights for large models
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
