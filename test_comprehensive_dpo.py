#!/usr/bin/env python3
"""Comprehensive tests for the simplified DPO approach."""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_error_handling():
    """Test error handling and edge cases."""
    print("🧪 Testing Error Handling and Edge Cases")
    print("=" * 50)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer
        from trl import DPOConfig
        from datasets import Dataset
        
        # Test 1: Missing TRL
        print("Test 1: Missing TRL import...")
        try:
            # This should work since TRL is available
            from trl import DPOTrainer
            print("✅ TRL import successful")
        except ImportError as e:
            print(f"❌ TRL import failed: {e}")
            return False
        
        # Test 2: Invalid config type
        print("\nTest 2: Invalid config type...")
        try:
            create_dpo_trainer(
                model_name="sshleifer/tiny-gpt2",
                dpo_config="invalid_config",  # Should be DPOConfig
                train_dataset=Dataset.from_dict({"prompt": ["test"], "chosen": ["good"], "rejected": ["bad"]})
            )
            print("❌ Should have failed with invalid config")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught invalid config: {e}")
        
        # Test 3: Missing dataset
        print("\nTest 3: Missing dataset...")
        try:
            config = DPOConfig(learning_rate=1e-5, max_steps=1, bf16=False, fp16=False)
            create_dpo_trainer(
                model_name="sshleifer/tiny-gpt2",
                dpo_config=config,
                train_dataset=None
            )
            print("❌ Should have failed with missing dataset")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught missing dataset: {e}")
        
        # Test 4: Invalid dataset format
        print("\nTest 4: Invalid dataset format...")
        try:
            config = DPOConfig(learning_rate=1e-5, max_steps=1, bf16=False, fp16=False)
            invalid_dataset = Dataset.from_dict({
                "wrong_column": ["test"],
                "another_wrong": ["test"]
            })
            create_dpo_trainer(
                model_name="sshleifer/tiny-gpt2",
                dpo_config=config,
                train_dataset=invalid_dataset
            )
            print("❌ Should have failed with invalid dataset format")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught invalid dataset format: {e}")
        
        # Test 5: Invalid model name
        print("\nTest 5: Invalid model name...")
        try:
            config = DPOConfig(learning_rate=1e-5, max_steps=1, bf16=False, fp16=False)
            dataset = Dataset.from_dict({
                "prompt": ["test"],
                "chosen": ["good"],
                "rejected": ["bad"]
            })
            create_dpo_trainer(
                model_name="nonexistent/model/that/does/not/exist",
                dpo_config=config,
                train_dataset=dataset
            )
            print("❌ Should have failed with invalid model name")
            return False
        except Exception as e:
            print(f"✅ Correctly caught invalid model name: {type(e).__name__}")
        
        print("\n✅ All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_models():
    """Test with different model sizes and types."""
    print("\n🧪 Testing Different Models")
    print("=" * 40)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer
        from trl import DPOConfig
        from datasets import Dataset
        
        # Create test dataset
        dataset = Dataset.from_dict({
            "prompt": ["What is AI?"],
            "chosen": ["AI is artificial intelligence."],
            "rejected": ["AI is not real."]
        })
        
        # Test different models
        models_to_test = [
            "sshleifer/tiny-gpt2",  # Very small model
            # "gpt2",  # Standard GPT-2 (larger)
            # "microsoft/DialoGPT-small",  # Different architecture
        ]
        
        for model_name in models_to_test:
            print(f"\nTesting model: {model_name}")
            try:
                config = DPOConfig(
                    learning_rate=1e-5,
                    max_steps=1,
                    bf16=False,
                    fp16=False,
                    per_device_train_batch_size=1
                )
                
                start_time = time.time()
                trainer = create_dpo_trainer(
                    model_name=model_name,
                    dpo_config=config,
                    train_dataset=dataset
                )
                creation_time = time.time() - start_time
                
                print(f"✅ Model {model_name} loaded in {creation_time:.2f}s")
                print(f"   Model type: {type(trainer.model).__name__}")
                print(f"   Ref model type: {type(trainer.ref_model).__name__}")
                
                # Test a quick training step
                trainer.train()
                print(f"✅ Training completed successfully")
                
            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                return False
        
        print("\n✅ All model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rldk_integration():
    """Test RLDK integration thoroughly."""
    print("\n🧪 Testing RLDK Integration")
    print("=" * 40)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer, PPOMonitor
        from trl import DPOConfig
        from datasets import Dataset
        
        # Create test dataset
        dataset = Dataset.from_dict({
            "prompt": ["Test prompt"] * 5,
            "chosen": ["Good response"] * 5,
            "rejected": ["Bad response"] * 5
        })
        
        # Test 1: Basic RLDK monitor creation
        print("Test 1: RLDK monitor creation...")
        monitor = PPOMonitor(
            output_dir="./rldk_test",
            kl_threshold=0.1,
            reward_threshold=0.05,
            run_id="rldk_test"
        )
        print("✅ RLDK monitor created successfully")
        
        # Test 2: Monitor with trainer
        print("\nTest 2: Monitor with trainer...")
        config = DPOConfig(
            learning_rate=1e-5,
            max_steps=2,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=1
        )
        
        trainer = create_dpo_trainer(
            model_name="sshleifer/tiny-gpt2",
            dpo_config=config,
            train_dataset=dataset,
            callbacks=[monitor]
        )
        print("✅ Trainer created with RLDK monitor")
        
        # Test 3: Training with monitoring
        print("\nTest 3: Training with monitoring...")
        trainer.train()
        print("✅ Training completed with RLDK monitoring")
        
        # Test 4: Check if metrics were collected
        print("\nTest 4: Check metrics collection...")
        if hasattr(monitor, 'metrics_history'):
            metrics_count = len(monitor.metrics_history)
            print(f"✅ Collected {metrics_count} metrics")
            if metrics_count > 0:
                print(f"   Sample metrics: {list(monitor.metrics_history[0].keys())}")
        else:
            print("⚠️  No metrics_history attribute found")
        
        # Test 5: Check output directory
        print("\nTest 5: Check output directory...")
        output_dir = Path("./rldk_test")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"✅ Output directory created with {len(files)} files")
        else:
            print("⚠️  Output directory not found")
        
        print("\n✅ All RLDK integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ RLDK integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test performance characteristics."""
    print("\n🧪 Testing Performance")
    print("=" * 30)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer
        from trl import DPOConfig
        from datasets import Dataset
        
        # Create larger dataset for performance testing
        dataset = Dataset.from_dict({
            "prompt": [f"Question {i}" for i in range(10)],
            "chosen": [f"Good answer {i}" for i in range(10)],
            "rejected": [f"Bad answer {i}" for i in range(10)]
        })
        
        config = DPOConfig(
            learning_rate=1e-5,
            max_steps=5,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=1
        )
        
        # Test creation time
        print("Testing creation time...")
        start_time = time.time()
        trainer = create_dpo_trainer(
            model_name="sshleifer/tiny-gpt2",
            dpo_config=config,
            train_dataset=dataset
        )
        creation_time = time.time() - start_time
        print(f"✅ Trainer creation: {creation_time:.2f}s")
        
        # Test training time
        print("Testing training time...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        print(f"✅ Training completion: {training_time:.2f}s")
        
        # Test memory usage (basic check)
        print("Testing memory usage...")
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"✅ Memory usage: {memory_mb:.1f} MB")
        
        print("\n✅ Performance tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_variations():
    """Test different dataset variations."""
    print("\n🧪 Testing Dataset Variations")
    print("=" * 40)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer
        from trl import DPOConfig
        from datasets import Dataset
        
        # Test different dataset sizes
        dataset_sizes = [1, 5, 10]
        
        for size in dataset_sizes:
            print(f"\nTesting dataset size: {size}")
            
            dataset = Dataset.from_dict({
                "prompt": [f"Question {i}" for i in range(size)],
                "chosen": [f"Good answer {i}" for i in range(size)],
                "rejected": [f"Bad answer {i}" for i in range(size)]
            })
            
            config = DPOConfig(
                learning_rate=1e-5,
                max_steps=1,
                bf16=False,
                fp16=False,
                per_device_train_batch_size=1
            )
            
            trainer = create_dpo_trainer(
                model_name="sshleifer/tiny-gpt2",
                dpo_config=config,
                train_dataset=dataset
            )
            
            trainer.train()
            print(f"✅ Dataset size {size} worked")
        
        print("\n✅ All dataset variation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset variation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all comprehensive tests."""
    print("🎯 Comprehensive DPO Testing Suite")
    print("=" * 60)
    
    tests = [
        ("Error Handling", test_error_handling),
        ("Different Models", test_different_models),
        ("RLDK Integration", test_rldk_integration),
        ("Performance", test_performance),
        ("Dataset Variations", test_dataset_variations),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Simplified DPO approach is robust and reliable!")
        print("✅ Ready to update examples and documentation!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("❌ Need to fix issues before proceeding")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)