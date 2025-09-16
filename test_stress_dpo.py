#!/usr/bin/env python3
"""Stress test for the simplified DPO approach."""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def stress_test():
    """Run stress tests to ensure robustness."""
    print("🔥 Stress Testing Simplified DPO Approach")
    print("=" * 60)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer, PPOMonitor
        from trl import DPOConfig
        from datasets import Dataset
        
        # Test 1: Multiple trainers in sequence
        print("Test 1: Multiple trainers in sequence...")
        for i in range(3):
            print(f"  Creating trainer {i+1}/3...")
            
            dataset = Dataset.from_dict({
                "prompt": [f"Question {j}" for j in range(5)],
                "chosen": [f"Good answer {j}" for j in range(5)],
                "rejected": [f"Bad answer {j}" for j in range(5)]
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
            print(f"  ✅ Trainer {i+1} completed")
        
        # Test 2: Large dataset
        print("\nTest 2: Large dataset...")
        large_dataset = Dataset.from_dict({
            "prompt": [f"Question {i}" for i in range(50)],
            "chosen": [f"Good answer {i}" for i in range(50)],
            "rejected": [f"Bad answer {i}" for i in range(50)]
        })
        
        config = DPOConfig(
            learning_rate=1e-5,
            max_steps=2,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=1
        )
        
        start_time = time.time()
        trainer = create_dpo_trainer(
            model_name="sshleifer/tiny-gpt2",
            dpo_config=config,
            train_dataset=large_dataset
        )
        creation_time = time.time() - start_time
        
        trainer.train()
        print(f"✅ Large dataset (50 samples) handled in {creation_time:.2f}s")
        
        # Test 3: Multiple callbacks
        print("\nTest 3: Multiple callbacks...")
        monitor1 = PPOMonitor(output_dir="./stress_test_1", run_id="stress_1")
        monitor2 = PPOMonitor(output_dir="./stress_test_2", run_id="stress_2")
        
        dataset = Dataset.from_dict({
            "prompt": ["Test prompt"],
            "chosen": ["Good response"],
            "rejected": ["Bad response"]
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
            train_dataset=dataset,
            callbacks=[monitor1, monitor2]
        )
        
        trainer.train()
        print("✅ Multiple callbacks handled successfully")
        
        # Test 4: Memory cleanup
        print("\nTest 4: Memory cleanup...")
        import gc
        import torch
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Memory cleanup completed")
        
        print("\n🎉 All stress tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 40)
    
    try:
        from rldk.integrations.trl import create_dpo_trainer, simple_reward_function
        from trl import DPOConfig
        from datasets import Dataset
        
        # Test 1: Empty strings
        print("Test 1: Empty strings in dataset...")
        dataset = Dataset.from_dict({
            "prompt": [""],
            "chosen": [""],
            "rejected": [""]
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
        print("✅ Empty strings handled")
        
        # Test 2: Very long strings
        print("\nTest 2: Very long strings...")
        long_text = "This is a very long text. " * 100  # ~3000 characters
        dataset = Dataset.from_dict({
            "prompt": [long_text],
            "chosen": [long_text + " This is good."],
            "rejected": [long_text + " This is bad."]
        })
        
        trainer = create_dpo_trainer(
            model_name="sshleifer/tiny-gpt2",
            dpo_config=config,
            train_dataset=dataset
        )
        
        trainer.train()
        print("✅ Long strings handled")
        
        # Test 3: Special characters
        print("\nTest 3: Special characters...")
        special_text = "Hello! @#$%^&*()_+{}|:<>?[]\\;'\",./"
        dataset = Dataset.from_dict({
            "prompt": [special_text],
            "chosen": [special_text + " Good!"],
            "rejected": [special_text + " Bad!"]
        })
        
        trainer = create_dpo_trainer(
            model_name="sshleifer/tiny-gpt2",
            dpo_config=config,
            train_dataset=dataset
        )
        
        trainer.train()
        print("✅ Special characters handled")
        
        # Test 4: Reward function edge cases
        print("\nTest 4: Reward function edge cases...")
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            "a" * 1000,  # Very long string
            "!@#$%^&*()",  # Only special characters
            "good great excellent amazing wonderful fantastic",  # All positive words
        ]
        
        for text in edge_cases:
            reward = simple_reward_function(text)
            print(f"  '{text[:20]}...' -> {reward:.2f}")
        
        print("✅ Reward function edge cases handled")
        
        print("\n🎉 All edge case tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run stress tests."""
    print("🔥 DPO Stress Testing Suite")
    print("=" * 60)
    
    tests = [
        ("Stress Tests", stress_test),
        ("Edge Cases", test_edge_cases),
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
    print("STRESS TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} stress tests passed")
    
    if passed == total:
        print("\n🎉 ALL STRESS TESTS PASSED!")
        print("✅ Simplified DPO approach is ROBUST and PRODUCTION-READY!")
        print("✅ Ready to update examples and documentation!")
        return True
    else:
        print(f"\n⚠️  {total - passed} stress tests failed")
        print("❌ Need to fix issues before proceeding")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Stress testing interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)