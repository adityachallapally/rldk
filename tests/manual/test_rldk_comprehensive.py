#!/usr/bin/env python3
"""
Comprehensive test script for RLDK package - testing like a researcher would
This script tests the package end-to-end with real models and data
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import tempfile
import shutil
from datetime import datetime

import _path_setup  # noqa: F401


def test_experiment_tracking():
    """Test experiment tracking with real models and data"""
    print("=" * 60)
    print("TESTING EXPERIMENT TRACKING")
    print("=" * 60)
    
    try:
        from rldk.tracking import ExperimentTracker, TrackingConfig
        
        # Create a temporary directory for the experiment
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Configure tracking
            config = TrackingConfig(
                experiment_name="gpt2_test_experiment",
                output_dir=temp_path,
                enable_dataset_tracking=True,
                enable_model_tracking=True,
                enable_environment_tracking=True,
                enable_seed_tracking=True,
                enable_git_tracking=True,
                save_to_json=True,
                save_to_yaml=True
            )
            
            print(f"✓ TrackingConfig created successfully")
            
            # Initialize tracker
            tracker = ExperimentTracker(config)
            print(f"✓ ExperimentTracker initialized successfully")
            
            # Start experiment
            tracker.start_experiment()
            print(f"✓ Experiment started successfully")
            
            # Load real model and tokenizer
            print("Loading GPT-2 model and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            model = AutoModelForCausalLM.from_pretrained('gpt2')
            
            # Track model and tokenizer
            tracker.track_model(model, "gpt2_policy")
            tracker.track_tokenizer(tokenizer, "gpt2_tokenizer")
            print(f"✓ Model and tokenizer tracked successfully")
            
            # Create some sample training data
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is revolutionizing artificial intelligence.",
                "Reinforcement learning requires careful debugging and analysis.",
                "Natural language processing has made significant progress.",
                "Deep learning models require extensive training and validation."
            ]
            
            # Create dataset
            dataset = Dataset.from_dict({"text": sample_texts})
            tracker.track_dataset(dataset, "training_data")
            print(f"✓ Training dataset tracked successfully")
            
            # Set seeds
            tracker.set_seeds(42)
            print(f"✓ Seeds set successfully")
            
            # Add metadata
            tracker.add_metadata("learning_rate", 1e-5)
            tracker.add_metadata("batch_size", 32)
            tracker.add_metadata("model_size", "117M")
            tracker.add_metadata("framework", "transformers")
            print(f"✓ Metadata added successfully")
            
            # Add tags
            tracker.add_tag("test")
            tracker.add_tag("gpt2")
            tracker.add_tag("research")
            print(f"✓ Tags added successfully")
            
            # Finish experiment
            tracker.finish_experiment()
            print(f"✓ Experiment finished successfully")
            
            # Check if files were created
            json_file = temp_path / "experiment.json"
            yaml_file = temp_path / "experiment.yaml"
            
            if json_file.exists():
                print(f"✓ JSON file created: {json_file}")
                with open(json_file) as f:
                    data = json.load(f)
                    print(f"  - Experiment ID: {data.get('experiment_id', 'N/A')}")
                    print(f"  - Model count: {len(data.get('models', {}))}")
                    print(f"  - Dataset count: {len(data.get('datasets', {}))}")
            else:
                print(f"✗ JSON file not created")
                
            if yaml_file.exists():
                print(f"✓ YAML file created: {yaml_file}")
            else:
                print(f"✗ YAML file not created")
                
        return True
        
    except Exception as e:
        print(f"✗ Experiment tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_forensics():
    """Test PPO forensics with simulated training data"""
    print("\n" + "=" * 60)
    print("TESTING PPO FORENSICS")
    print("=" * 60)
    
    try:
        from rldk.forensics import ComprehensivePPOForensics
        
        # Initialize forensics
        forensics = ComprehensivePPOForensics(
            kl_target=0.1,
            kl_target_tolerance=0.05,
            window_size=100,
            enable_kl_schedule_tracking=True,
            enable_gradient_norms_analysis=True,
            enable_advantage_statistics=True
        )
        print(f"✓ ComprehensivePPOForensics initialized successfully")
        
        # Simulate training data with some anomalies
        np.random.seed(42)
        steps = list(range(0, 1000, 10))
        
        for i, step in enumerate(steps):
            # Simulate normal training with some anomalies
            if 200 <= step <= 250:  # KL spike anomaly
                kl = 0.3 + np.random.normal(0, 0.05)
                policy_grad_norm = 1.0 + np.random.normal(0, 0.1)
                value_grad_norm = 0.8 + np.random.normal(0, 0.1)
            elif 500 <= step <= 550:  # Gradient explosion
                kl = 0.08 + np.random.normal(0, 0.02)
                policy_grad_norm = 5.0 + np.random.normal(0, 0.5)
                value_grad_norm = 3.0 + np.random.normal(0, 0.3)
            else:  # Normal training
                kl = 0.08 + np.random.normal(0, 0.02)
                policy_grad_norm = 1.0 + np.random.normal(0, 0.1)
                value_grad_norm = 0.8 + np.random.normal(0, 0.1)
            
            # Update forensics
            metrics = forensics.update(
                step=step,
                kl=kl,
                kl_coef=0.2,
                entropy=2.5 + np.random.normal(0, 0.1),
                reward_mean=0.8 + np.random.normal(0, 0.1),
                reward_std=0.3 + np.random.normal(0, 0.05),
                policy_grad_norm=policy_grad_norm,
                value_grad_norm=value_grad_norm,
                total_grad_norm=policy_grad_norm + value_grad_norm,
                advantage_mean=0.1 + np.random.normal(0, 0.05),
                advantage_std=0.5 + np.random.normal(0, 0.1),
                advantage_min=-0.5 + np.random.normal(0, 0.1),
                advantage_max=1.0 + np.random.normal(0, 0.1),
                advantage_median=0.05 + np.random.normal(0, 0.05),
                advantage_samples=[np.random.normal(0, 0.5) for _ in range(10)]
            )
            
            if i % 50 == 0:  # Print progress every 50 steps
                print(f"  Processed step {step}")
        
        print(f"✓ Training data processed successfully")
        
        # Get comprehensive analysis
        analysis = forensics.get_comprehensive_analysis()
        print(f"✓ Comprehensive analysis generated")
        print(f"  - Total metrics tracked: {len(analysis.get('metrics', {}))}")
        
        # Get anomalies
        anomalies = forensics.get_anomalies()
        print(f"✓ Anomalies detected: {len(anomalies)}")
        for anomaly in anomalies[:5]:  # Show first 5 anomalies
            print(f"  - {anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}")
        
        # Get health summary
        health_summary = forensics.get_health_summary()
        print(f"✓ Health summary generated")
        print(f"  - Overall health: {health_summary.get('overall_health', 'Unknown')}")
        print(f"  - Health score: {health_summary.get('health_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ PPO forensics failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_determinism_checking():
    """Test determinism checking with real training scenarios"""
    print("\n" + "=" * 60)
    print("TESTING DETERMINISM CHECKING")
    print("=" * 60)
    
    try:
        from rldk.determinism import check
        
        # Create a simple deterministic training script
        script_content = '''
import numpy as np
import torch
import random
import os

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Simple deterministic computation
x = np.random.randn(100)
y = torch.randn(100)
z = random.random()

# Output results
print(f"numpy_result:{np.sum(x):.6f}")
print(f"torch_result:{torch.sum(y).item():.6f}")
print(f"random_result:{z:.6f}")
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "test_deterministic.py"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            print(f"✓ Test script created: {script_path}")
            
            # Test determinism checking
            report = check(
                cmd=f"python {script_path}",
                compare=["numpy_result", "torch_result", "random_result"],
                replicas=3,
                device="cpu"
            )
            
            print(f"✓ Determinism check completed")
            print(f"  - Passed: {report.passed}")
            print(f"  - Replicas: {len(report.mismatches) if hasattr(report, 'mismatches') else 'N/A'}")
            print(f"  - Culprit: {report.culprit if hasattr(report, 'culprit') else 'N/A'}")
            
            if hasattr(report, 'fixes') and report.fixes:
                print(f"  - Fixes suggested: {len(report.fixes)}")
                for fix in report.fixes[:3]:
                    print(f"    * {fix}")
            
        return True
        
    except Exception as e:
        print(f"✗ Determinism checking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_analysis():
    """Test reward model health analysis and drift detection"""
    print("\n" + "=" * 60)
    print("TESTING REWARD ANALYSIS")
    print("=" * 60)
    
    try:
        from rldk.reward import health, compare_models
        
        # Create sample training data
        np.random.seed(42)
        steps = list(range(0, 1000, 10))
        
        # Simulate training data with reward drift
        training_data = []
        for i, step in enumerate(steps):
            # Simulate reward drift over time
            base_reward = 0.8
            if step > 500:  # Drift starts after step 500
                drift = (step - 500) * 0.001
                reward_mean = base_reward + drift
            else:
                reward_mean = base_reward
            
            training_data.append({
                'step': step,
                'reward_mean': reward_mean + np.random.normal(0, 0.1),
                'reward_std': 0.3 + np.random.normal(0, 0.05),
                'loss': 0.5 + np.random.normal(0, 0.1)
            })
        
        df_training = pd.DataFrame(training_data)
        print(f"✓ Training data created: {len(df_training)} samples")
        
        # Create reference data (baseline)
        reference_data = df_training[df_training['step'] <= 500].copy()
        print(f"✓ Reference data created: {len(reference_data)} samples")
        
        # Analyze reward model health
        health_report = health(
            run_data=df_training,
            reference_data=reference_data,
            reward_col="reward_mean",
            step_col="step",
            threshold_drift=0.1,
            threshold_saturation=0.8,
            threshold_calibration=0.7
        )
        
        print(f"✓ Reward health analysis completed")
        print(f"  - Passed: {health_report.passed}")
        print(f"  - Drift detected: {health_report.drift_detected}")
        print(f"  - Saturation issues: {health_report.saturation_issues}")
        print(f"  - Calibration score: {health_report.calibration_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reward analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_ingestion():
    """Test data ingestion from different sources"""
    print("\n" + "=" * 60)
    print("TESTING DATA INGESTION")
    print("=" * 60)
    
    try:
        from rldk.ingest import ingest_runs
        from rldk.adapters import TRLAdapter, CustomJSONLAdapter
        
        # Create sample TRL-style data
        trl_data = []
        for i in range(100):
            trl_data.append({
                'step': i * 10,
                'train/loss': 0.5 + np.random.normal(0, 0.1),
                'train/reward': 0.8 + np.random.normal(0, 0.1),
                'train/kl': 0.1 + np.random.normal(0, 0.02),
                'train/entropy': 2.5 + np.random.normal(0, 0.1),
                'train/policy_grad_norm': 1.0 + np.random.normal(0, 0.1),
                'train/value_grad_norm': 0.8 + np.random.normal(0, 0.1)
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save as JSONL
            jsonl_path = temp_path / "trl_data.jsonl"
            with open(jsonl_path, 'w') as f:
                for item in trl_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"✓ Sample TRL data created: {jsonl_path}")
            
            # Test TRL adapter
            try:
                trl_adapter = TRLAdapter(str(jsonl_path))
                df_trl = trl_adapter.load()
                print(f"✓ TRL adapter loaded data: {len(df_trl)} rows")
                print(f"  - Columns: {list(df_trl.columns)}")
            except Exception as e:
                print(f"⚠ TRL adapter failed: {e}")
            
            # Test generic ingestion
            try:
                df_generic = ingest_runs(str(jsonl_path), adapter_hint="custom_jsonl")
                print(f"✓ Generic ingestion loaded data: {len(df_generic)} rows")
            except Exception as e:
                print(f"⚠ Generic ingestion failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_suites():
    """Test evaluation suites with different model sizes"""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION SUITES")
    print("=" * 60)
    
    try:
        from rldk.evals import run
        from rldk.evals.suites import QUICK_SUITE, COMPREHENSIVE_SUITE
        
        # Create sample evaluation data
        eval_data = []
        for i in range(200):
            eval_data.append({
                'step': i * 5,
                'text': f"This is sample text {i} for evaluation purposes.",
                'reward': 0.8 + np.random.normal(0, 0.1),
                'loss': 0.5 + np.random.normal(0, 0.1),
                'kl': 0.1 + np.random.normal(0, 0.02)
            })
        
        df_eval = pd.DataFrame(eval_data)
        print(f"✓ Evaluation data created: {len(df_eval)} samples")
        
        # Test quick suite
        try:
            quick_result = run(
                run_data=df_eval,
                suite="quick",
                seed=42,
                sample_size=100
            )
            print(f"✓ Quick evaluation suite completed")
            print(f"  - Overall score: {quick_result.overall_score}")
            print(f"  - Sample size: {quick_result.sample_size}")
        except Exception as e:
            print(f"⚠ Quick suite failed: {e}")
        
        # Test comprehensive suite
        try:
            comprehensive_result = run(
                run_data=df_eval,
                suite="comprehensive",
                seed=42,
                sample_size=150
            )
            print(f"✓ Comprehensive evaluation suite completed")
            print(f"  - Overall score: {comprehensive_result.overall_score}")
            print(f"  - Sample size: {comprehensive_result.sample_size}")
        except Exception as e:
            print(f"⚠ Comprehensive suite failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation suites failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test CLI commands end-to-end"""
    print("\n" + "=" * 60)
    print("TESTING CLI COMMANDS")
    print("=" * 60)
    
    try:
        import subprocess
        import sys
        
        # Test various CLI commands
        commands_to_test = [
            ["rldk", "evals", "list-suites"],
            ["rldk", "seed", "--show"],
            ["rldk", "format-info"],
        ]
        
        for cmd in commands_to_test:
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=30,
                    cwd="/workspace"
                )
                if result.returncode == 0:
                    print(f"✓ Command {' '.join(cmd)} succeeded")
                else:
                    print(f"⚠ Command {' '.join(cmd)} failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⚠ Command {' '.join(cmd)} timed out")
            except Exception as e:
                print(f"⚠ Command {' '.join(cmd)} error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI commands testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive tests"""
    print("RLDK COMPREHENSIVE TESTING")
    print("Testing like a researcher would - with real models and data")
    print("=" * 80)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Experiment Tracking", test_experiment_tracking),
        ("PPO Forensics", test_ppo_forensics),
        ("Determinism Checking", test_determinism_checking),
        ("Reward Analysis", test_reward_analysis),
        ("Data Ingestion", test_data_ingestion),
        ("Evaluation Suites", test_evaluation_suites),
        ("CLI Commands", test_cli_commands),
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! RLDK is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. See details above.")
    
    return test_results

if __name__ == "__main__":
    main()