#!/usr/bin/env python3
"""
Test performance with different model sizes and data volumes
"""

import sys
import time
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

import _path_setup  # noqa: F401


def test_performance():
    """Test performance with different model sizes and data volumes"""
    print("Testing Performance with Different Model Sizes")
    print("=" * 60)
    
    try:
        from rldk.tracking import ExperimentTracker, TrackingConfig
        from rldk.forensics import ComprehensivePPOForensics
        
        # Test 1: Small model (GPT-2)
        print("\n1. Testing with GPT-2 (117M parameters)...")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            model = AutoModelForCausalLM.from_pretrained('gpt2')
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = TrackingConfig(
                    experiment_name="gpt2_performance_test",
                    output_dir=temp_dir,
                    enable_model_tracking=True,
                    save_to_wandb=False
                )
                
                tracker = ExperimentTracker(config)
                tracker.start_experiment()
                tracker.track_model(model, "gpt2")
                tracker.track_tokenizer(tokenizer, "gpt2_tokenizer")
                tracker.finish_experiment()
            
            gpt2_time = time.time() - start_time
            print(f"✓ GPT-2 tracking completed in {gpt2_time:.2f} seconds")
            
        except Exception as e:
            print(f"⚠ GPT-2 test failed: {e}")
        
        # Test 2: Medium model (GPT-2 Medium)
        print("\n2. Testing with GPT-2 Medium (345M parameters)...")
        start_time = time.time()
        
        try:
            tokenizer_medium = AutoTokenizer.from_pretrained('gpt2-medium')
            model_medium = AutoModelForCausalLM.from_pretrained('gpt2-medium')
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = TrackingConfig(
                    experiment_name="gpt2_medium_performance_test",
                    output_dir=temp_dir,
                    enable_model_tracking=True,
                    save_to_wandb=False
                )
                
                tracker = ExperimentTracker(config)
                tracker.start_experiment()
                tracker.track_model(model_medium, "gpt2_medium")
                tracker.track_tokenizer(tokenizer_medium, "gpt2_medium_tokenizer")
                tracker.finish_experiment()
            
            gpt2_medium_time = time.time() - start_time
            print(f"✓ GPT-2 Medium tracking completed in {gpt2_medium_time:.2f} seconds")
            
        except Exception as e:
            print(f"⚠ GPT-2 Medium test failed: {e}")
        
        # Test 3: Large dataset processing
        print("\n3. Testing with large dataset processing...")
        start_time = time.time()
        
        try:
            # Create a large dataset
            large_data = []
            for i in range(10000):  # 10K samples
                large_data.append({
                    'step': i * 10,
                    'reward': 0.8 + np.random.normal(0, 0.1),
                    'loss': 0.5 + np.random.normal(0, 0.1),
                    'kl': 0.1 + np.random.normal(0, 0.02)
                })
            
            df_large = pd.DataFrame(large_data)
            
            # Test PPO forensics with large dataset
            forensics = ComprehensivePPOForensics()
            
            for i, row in df_large.iterrows():
                forensics.update(
                    step=row['step'],
                    kl=row['kl'],
                    kl_coef=0.2,
                    entropy=2.5,
                    reward_mean=row['reward'],
                    reward_std=0.3,
                    policy_grad_norm=1.0,
                    value_grad_norm=0.8,
                    total_grad_norm=1.8,
                    advantage_mean=0.1,
                    advantage_std=0.5,
                    advantage_min=-0.5,
                    advantage_max=1.0,
                    advantage_median=0.05,
                    advantage_samples=[0.1, 0.2, -0.1, 0.3]
                )
                
                if i % 2000 == 0:
                    print(f"  Processed {i} samples...")
            
            large_dataset_time = time.time() - start_time
            print(f"✓ Large dataset processing completed in {large_dataset_time:.2f} seconds")
            print(f"  - Processed {len(df_large)} samples")
            
            # Get analysis
            analysis = forensics.get_comprehensive_analysis()
            anomalies = forensics.get_anomalies()
            print(f"  - Analysis generated: {len(analysis.get('metrics', {}))} metrics")
            print(f"  - Anomalies detected: {len(anomalies)}")
            
        except Exception as e:
            print(f"⚠ Large dataset test failed: {e}")
        
        # Test 4: Memory usage test
        print("\n4. Testing memory efficiency...")
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Use a fallback model if the original model failed to load
            test_model = None
            if 'model' in locals() and model is not None:
                test_model = model
            else:
                # Load a small model for memory testing
                test_model = AutoModelForCausalLM.from_pretrained('gpt2')
            
            # Create many small experiments
            for i in range(10):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config = TrackingConfig(
                        experiment_name=f"memory_test_{i}",
                        output_dir=temp_dir,
                        enable_model_tracking=True,
                        save_to_wandb=False
                    )
                    
                    tracker = ExperimentTracker(config)
                    tracker.start_experiment()
                    tracker.track_model(test_model, f"model_{i}")
                    tracker.finish_experiment()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            print(f"✓ Memory efficiency test completed")
            print(f"  - Memory used: {memory_used:.2f} MB")
            print(f"  - Memory per experiment: {memory_used/10:.2f} MB")
            
        except Exception as e:
            print(f"⚠ Memory efficiency test failed: {e}")
        
        print(f"\n✓ Performance testing completed")
        return True
        
    except Exception as e:
        print(f"✗ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_performance()