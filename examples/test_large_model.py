#!/usr/bin/env python3
"""
Test the tracking system with a large model from Hugging Face.

This script tests the tracking system's ability to handle large models
and verifies that all tracking components work correctly with real-world models.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rldk.tracking import ExperimentTracker, TrackingConfig


def test_with_bert_model():
    """Test tracking with a BERT model."""
    print("Testing with BERT-base-uncased model...")

    try:
        # Load BERT model and tokenizer
        model_name = "bert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"✓ Loaded {model_name}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"✗ Failed to load BERT model: {e}")
        return False

    return test_model_tracking(model, tokenizer, "bert-base-uncased")


def test_with_distilbert_model():
    """Test tracking with a DistilBERT model."""
    print("Testing with DistilBERT model...")

    try:
        # Load DistilBERT model and tokenizer
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"✓ Loaded {model_name}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"✗ Failed to load DistilBERT model: {e}")
        return False

    return test_model_tracking(model, tokenizer, "distilbert-base-uncased")


def test_with_small_model():
    """Test tracking with a smaller model as fallback."""
    print("Testing with a small custom model...")

    # Create a smaller model
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )

    # Create a simple tokenizer-like object
    class SimpleTokenizer:
        def __init__(self):
            self.vocab_size = 10000
            self.model_max_length = 512

        def __call__(self, text, **kwargs):
            # Simple tokenization
            tokens = text.split()[:self.model_max_length]
            return {"input_ids": torch.tensor([hash(token) % self.vocab_size for token in tokens])}

    tokenizer = SimpleTokenizer()

    print("✓ Created custom model")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return test_model_tracking(model, tokenizer, "custom-small-model")


def test_model_tracking(model, tokenizer, model_name):
    """Test tracking for a given model and tokenizer."""
    print(f"\n--- Testing tracking for {model_name} ---")

    # Create output directory
    output_dir = Path(f"./test_output_{model_name.replace('-', '_')}")
    output_dir.mkdir(exist_ok=True)

    try:
        # Create tracking configuration
        config = TrackingConfig(
            experiment_name=f"large_model_test_{model_name.replace('-', '_')}",
            output_dir=output_dir,
            enable_dataset_tracking=True,
            enable_model_tracking=True,
            enable_environment_tracking=True,
            enable_seed_tracking=True,
            enable_git_tracking=True,
            save_model_architecture=True,
            save_model_weights=False,  # Don't save weights for large models
            save_to_json=True,
            save_to_yaml=True,
            save_to_wandb=False,
            tags=["large_model", "huggingface", "test"],
            notes=f"Testing tracking system with {model_name}"
        )

        # Initialize tracker
        tracker = ExperimentTracker(config)

        print("1. Starting experiment...")
        tracker.start_experiment()

        print("2. Setting seeds...")
        tracker.create_reproducible_environment(42)

        print("3. Creating test dataset...")
        # Create a test dataset
        test_data = np.random.randn(100, 768)  # BERT-like input size
        tracker.track_dataset(
            test_data, "test_embeddings",
            metadata={
                "model": model_name,
                "input_size": 768,
                "n_samples": 100
            }
        )

        print("4. Tracking model...")
        # Track the model
        model_info = tracker.track_model(
            model, model_name,
            metadata={
                "source": "huggingface" if hasattr(model, 'config') else "custom",
                "model_type": type(model).__name__,
                "pretrained": hasattr(model, 'config')
            }
        )

        print("   ✓ Model tracked successfully")
        print(f"   Parameters: {model_info['num_parameters']:,}")
        print(f"   Architecture checksum: {model_info['architecture_checksum'][:16]}...")

        print("5. Tracking tokenizer...")
        # Track the tokenizer
        tokenizer_info = tracker.track_tokenizer(
            tokenizer, f"{model_name}_tokenizer",
            metadata={
                "model": model_name,
                "type": type(tokenizer).__name__
            }
        )

        print("   ✓ Tokenizer tracked successfully")
        print(f"   Type: {tokenizer_info['type']}")
        if 'vocab_size' in tokenizer_info:
            print(f"   Vocab size: {tokenizer_info['vocab_size']}")

        print("6. Testing model inference...")
        # Test model inference
        model.eval()
        with torch.no_grad():
            # Create test input
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                input_size = model.config.hidden_size
            else:
                input_size = 768

            test_input = torch.randn(1, 128, input_size)  # Batch size 1, sequence length 128

            try:
                output = model(test_input)
                print("   ✓ Model inference successful")
                print(f"   Output shape: {output.last_hidden_state.shape if hasattr(output, 'last_hidden_state') else output.shape}")

                # Track inference metadata
                tracker.add_metadata("inference_test", "successful")
                tracker.add_metadata("input_shape", list(test_input.shape))
                if hasattr(output, 'last_hidden_state'):
                    tracker.add_metadata("output_shape", list(output.last_hidden_state.shape))
                else:
                    tracker.add_metadata("output_shape", list(output.shape))

            except Exception as e:
                print(f"   ✗ Model inference failed: {e}")
                tracker.add_metadata("inference_test", f"failed: {str(e)}")

        print("7. Adding performance metadata...")
        # Add performance-related metadata
        tracker.add_metadata("model_size_mb", sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024))
        tracker.add_metadata("device", str(next(model.parameters()).device))
        tracker.add_metadata("dtype", str(next(model.parameters()).dtype))

        print("8. Finishing experiment...")
        # Finish experiment
        summary = tracker.finish_experiment()

        print("\n✓ Experiment completed successfully!")
        print(f"  Experiment ID: {summary['experiment_id']}")
        print(f"  Datasets tracked: {summary['datasets_tracked']}")
        print(f"  Models tracked: {summary['models_tracked']}")
        print(f"  Output directory: {summary['output_dir']}")

        # Check created files
        files_created = list(output_dir.glob("*"))
        print(f"  Files created: {len(files_created)}")
        for file_path in files_created:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"    {file_path.name} ({size:,} bytes)")

        return True

    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("="*60)
    print("LARGE MODEL TRACKING TEST")
    print("="*60)

    # Test with different models
    models_to_test = [
        ("BERT", test_with_bert_model),
        ("DistilBERT", test_with_distilbert_model),
        ("Custom Small", test_with_small_model)
    ]

    results = {}

    for model_name, test_func in models_to_test:
        print(f"\n{'='*20} {model_name} {'='*20}")
        try:
            success = test_func()
            results[model_name] = success
        except Exception as e:
            print(f"✗ {model_name} test failed with exception: {e}")
            results[model_name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{model_name:15} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The tracking system works correctly with large models.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check the output above for details.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
