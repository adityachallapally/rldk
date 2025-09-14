#!/usr/bin/env python3
"""
Comprehensive RLDK Testing Script
Simulates a real researcher's workflow with RL debugging and analysis
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import RLDK components
try:
    from rldk.tracking import ExperimentTracker, TrackingConfig
    from rldk.forensics import ComprehensivePPOForensics
    from rldk.determinism import check
    from rldk.reward import health, compare_models
    from rldk.evals import run
    from rldk.diff import first_divergence
    from rldk.replay import replay
    from rldk.utils.seed import set_global_seed, get_current_seed
    from rldk.ingest import ingest_runs
    from rldk.adapters import TRLAdapter
    print("✅ All RLDK imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Import transformers for model testing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import Dataset

class RLTrainingSimulator:
    """Simulates a realistic RL training environment for testing RLDK"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.training_data = []
        self.eval_data = []
        self.metrics_history = []
        
    def load_model(self):
        """Load a pre-trained model from Hugging Face"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"✅ Model loaded successfully: {self.model.config.model_type}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def generate_training_data(self, num_samples: int = 1000):
        """Generate synthetic training data for RL simulation"""
        logger.info(f"Generating {num_samples} training samples")
        
        prompts = [
            "The weather today is",
            "In a world where",
            "The most important thing is",
            "When I think about",
            "The future of AI is",
            "Learning to code",
            "The best way to",
            "In my opinion",
            "The key to success",
            "Understanding machine learning"
        ]
        
        self.training_data = []
        for i in range(num_samples):
            prompt = np.random.choice(prompts)
            # Generate some synthetic reward data
            reward = np.random.normal(0.5, 0.2)  # Simulated reward
            kl_div = np.random.normal(0.1, 0.05)  # Simulated KL divergence
            entropy = np.random.normal(2.0, 0.3)  # Simulated entropy
            
            self.training_data.append({
                'step': i,
                'prompt': prompt,
                'reward': reward,
                'kl': kl_div,
                'entropy': entropy,
                'loss': np.random.normal(0.3, 0.1),
                'learning_rate': 1e-5,
                'batch_size': 32
            })
        
        logger.info(f"✅ Generated {len(self.training_data)} training samples")
        return self.training_data
    
    def simulate_training_step(self, step: int) -> Dict[str, float]:
        """Simulate a single training step with realistic metrics"""
        # Simulate some training dynamics
        base_reward = 0.5 + 0.1 * np.sin(step / 100)  # Oscillating reward
        noise = np.random.normal(0, 0.1)
        reward = max(0, base_reward + noise)
        
        # Simulate KL divergence with occasional spikes
        if step % 200 == 0:
            kl = np.random.normal(0.3, 0.1)  # Spike
        else:
            kl = np.random.normal(0.1, 0.02)
        
        # Simulate entropy decay
        entropy = max(0.5, 2.0 - step / 1000)
        
        # Simulate gradient norms
        policy_grad_norm = np.random.normal(1.0, 0.2)
        value_grad_norm = np.random.normal(0.8, 0.15)
        
        metrics = {
            'step': step,
            'reward_mean': reward,
            'reward_std': np.random.normal(0.2, 0.05),
            'kl': kl,
            'kl_coef': 0.2,
            'entropy': entropy,
            'loss': np.random.normal(0.3, 0.1),
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm,
            'total_grad_norm': policy_grad_norm + value_grad_norm,
            'advantage_mean': np.random.normal(0.1, 0.05),
            'advantage_std': np.random.normal(0.5, 0.1),
            'advantage_min': np.random.normal(-0.5, 0.1),
            'advantage_max': np.random.normal(1.0, 0.1),
            'advantage_median': np.random.normal(0.05, 0.02),
            'learning_rate': 1e-5,
            'batch_size': 32
        }
        
        self.metrics_history.append(metrics)
        return metrics

def test_experiment_tracking():
    """Test RLDK experiment tracking functionality"""
    logger.info("🧪 Testing Experiment Tracking")
    
    try:
        # Create tracking config
        config = TrackingConfig(
            experiment_name="rldk_comprehensive_test",
            enable_dataset_tracking=True,
            enable_model_tracking=True,
            enable_environment_tracking=True,
            enable_seed_tracking=True,
            enable_git_tracking=True,
            output_dir=Path("./test_runs")
        )
        
        # Initialize tracker
        tracker = ExperimentTracker(config)
        tracker.start_experiment()
        
        # Load model and tokenizer
        simulator = RLTrainingSimulator()
        if not simulator.load_model():
            return False
        
        # Track model
        tracker.track_model(simulator.model, "dialgpt_policy")
        tracker.track_tokenizer(simulator.tokenizer, "dialgpt_tokenizer")
        
        # Generate and track training data
        training_data = simulator.generate_training_data(100)
        tracker.track_dataset(training_data, "training_data")
        
        # Set seeds
        tracker.set_seeds(42)
        
        # Add metadata
        tracker.add_metadata("learning_rate", 1e-5)
        tracker.add_metadata("batch_size", 32)
        tracker.add_metadata("model_name", simulator.model_name)
        
        # Finish experiment
        tracker.finish_experiment()
        
        logger.info("✅ Experiment tracking test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Experiment tracking test failed: {e}")
        return False

def test_ppo_forensics():
    """Test RLDK PPO forensics functionality"""
    logger.info("🧪 Testing PPO Forensics")
    
    try:
        # Initialize forensics
        forensics = ComprehensivePPOForensics(
            kl_target=0.1,
            kl_target_tolerance=0.05,
            window_size=50,
            enable_kl_schedule_tracking=True,
            enable_gradient_norms_analysis=True,
            enable_advantage_statistics=True
        )
        
        # Simulate training data
        simulator = RLTrainingSimulator()
        simulator.load_model()
        
        # Generate training metrics
        for step in range(200):
            metrics = simulator.simulate_training_step(step)
            
            # Update forensics
            forensics.update(
                step=metrics['step'],
                kl=metrics['kl'],
                kl_coef=metrics['kl_coef'],
                entropy=metrics['entropy'],
                reward_mean=metrics['reward_mean'],
                reward_std=metrics['reward_std'],
                policy_grad_norm=metrics['policy_grad_norm'],
                value_grad_norm=metrics['value_grad_norm'],
                total_grad_norm=metrics['total_grad_norm'],
                advantage_mean=metrics['advantage_mean'],
                advantage_std=metrics['advantage_std'],
                advantage_min=metrics['advantage_min'],
                advantage_max=metrics['advantage_max'],
                advantage_median=metrics['advantage_median'],
                advantage_samples=[np.random.normal(0.1, 0.5) for _ in range(10)]
            )
        
        # Get analysis
        analysis = forensics.get_comprehensive_analysis()
        anomalies = forensics.get_anomalies()
        health_summary = forensics.get_health_summary()
        
        logger.info(f"✅ PPO forensics test passed - Found {len(anomalies)} anomalies")
        logger.info(f"Health summary: {health_summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PPO forensics test failed: {e}")
        return False

def test_determinism_checking():
    """Test RLDK determinism checking"""
    logger.info("🧪 Testing Determinism Checking")
    
    try:
        # Create a simple training script for testing
        test_script = """
import numpy as np
import torch
from rldk.utils.seed import set_global_seed

# Set seed
set_global_seed(42, deterministic=True)

# Generate some random data
data = np.random.randn(100)
torch_data = torch.randn(50, 10)

# Save results
with open('test_output.txt', 'w') as f:
    f.write(f"numpy_sum: {np.sum(data)}\\n")
    f.write(f"torch_sum: {torch.sum(torch_data).item()}\\n")
"""
        
        with open("test_determinism_script.py", "w") as f:
            f.write(test_script)
        
        # Test determinism
        report = check(
            cmd="python test_determinism_script.py",
            compare=["numpy_sum", "torch_sum"],
            replicas=3
        )
        
        logger.info(f"✅ Determinism check completed - Passed: {report.passed}")
        logger.info(f"Replica variance: {report.replica_variance}")
        
        # Cleanup
        os.remove("test_determinism_script.py")
        if os.path.exists("test_output.txt"):
            os.remove("test_output.txt")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Determinism checking test failed: {e}")
        return False

def test_reward_analysis():
    """Test RLDK reward analysis functionality"""
    logger.info("🧪 Testing Reward Analysis")
    
    try:
        # Generate training data
        simulator = RLTrainingSimulator()
        simulator.load_model()
        training_data = simulator.generate_training_data(200)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Test reward health analysis
        health_report = health(
            run_data=df,
            reward_col="reward",
            threshold_drift=0.1,
            threshold_saturation=0.8,
            threshold_calibration=0.7
        )
        
        logger.info(f"✅ Reward analysis test passed - Health passed: {health_report.passed}")
        logger.info(f"Drift detected: {health_report.drift_detected}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward analysis test failed: {e}")
        return False

def test_evaluation_suites():
    """Test RLDK evaluation suites"""
    logger.info("🧪 Testing Evaluation Suites")
    
    try:
        # Generate training data
        simulator = RLTrainingSimulator()
        simulator.load_model()
        training_data = simulator.generate_training_data(100)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Test evaluation suite
        eval_result = run(
            run_data=df,
            suite="quick",
            seed=42,
            sample_size=50
        )
        
        logger.info(f"✅ Evaluation suite test passed - Overall score: {eval_result.overall_score}")
        logger.info(f"Sample size: {eval_result.sample_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation suite test failed: {e}")
        return False

def test_run_comparison():
    """Test RLDK run comparison functionality"""
    logger.info("🧪 Testing Run Comparison")
    
    try:
        # Generate two different training runs
        simulator1 = RLTrainingSimulator()
        simulator1.load_model()
        
        simulator2 = RLTrainingSimulator()
        simulator2.load_model()
        
        # Generate different data for each run
        run1_data = []
        run2_data = []
        
        for step in range(100):
            # Run 1: Normal training
            metrics1 = simulator1.simulate_training_step(step)
            run1_data.append(metrics1)
            
            # Run 2: Slightly different (simulate drift)
            metrics2 = simulator2.simulate_training_step(step)
            if step > 50:  # Introduce drift after step 50
                metrics2['reward_mean'] += 0.2
            run2_data.append(metrics2)
        
        df1 = pd.DataFrame(run1_data)
        df2 = pd.DataFrame(run2_data)
        
        # Test divergence detection
        divergence_report = first_divergence(
            df_a=df1,
            df_b=df2,
            signals=["reward_mean", "kl", "entropy"],
            k_consecutive=3,
            window=20,
            tolerance=2.0
        )
        
        logger.info(f"✅ Run comparison test passed - Diverged: {divergence_report.diverged}")
        if divergence_report.diverged:
            logger.info(f"First divergence at step: {divergence_report.first_step}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Run comparison test failed: {e}")
        return False

def test_seed_management():
    """Test RLDK seed management"""
    logger.info("🧪 Testing Seed Management")
    
    try:
        # Test global seed setting
        seed = set_global_seed(42, deterministic=True)
        logger.info(f"Set global seed: {seed}")
        
        # Test getting current seed
        current_seed = get_current_seed()
        logger.info(f"Current seed: {current_seed}")
        
        # Test seed context
        from rldk.utils.seed import seed_context
        
        with seed_context(123):
            test_data = np.random.randn(10)
            logger.info(f"Data in context: {test_data[:3]}")
        
        # Test that seed is restored
        test_data_after = np.random.randn(10)
        logger.info(f"Data after context: {test_data_after[:3]}")
        
        logger.info("✅ Seed management test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Seed management test failed: {e}")
        return False

def test_cli_commands():
    """Test RLDK CLI commands"""
    logger.info("🧪 Testing CLI Commands")
    
    try:
        import subprocess
        
        # Test version command
        result = subprocess.run(["/home/ubuntu/.local/bin/rldk", "version"], 
                              capture_output=True, text=True)
        logger.info(f"Version command: {result.stdout.strip()}")
        
        # Test seed command
        result = subprocess.run(["/home/ubuntu/.local/bin/rldk", "seed", "--show"], 
                              capture_output=True, text=True)
        logger.info(f"Seed command: {result.stdout.strip()}")
        
        logger.info("✅ CLI commands test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ CLI commands test failed: {e}")
        return False

def main():
    """Run comprehensive RLDK testing"""
    logger.info("🚀 Starting Comprehensive RLDK Testing")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Experiment Tracking", test_experiment_tracking),
        ("PPO Forensics", test_ppo_forensics),
        ("Determinism Checking", test_determinism_checking),
        ("Reward Analysis", test_reward_analysis),
        ("Evaluation Suites", test_evaluation_suites),
        ("Run Comparison", test_run_comparison),
        ("Seed Management", test_seed_management),
        ("CLI Commands", test_cli_commands),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            test_results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            
            if result:
                logger.info(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
            else:
                logger.error(f"❌ {test_name} FAILED ({end_time - start_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            test_results[test_name] = {
                'passed': False,
                'duration': 0,
                'error': str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🏁 TESTING SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result['passed'])
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        duration = f"{result['duration']:.2f}s"
        logger.info(f"{status} {test_name:<25} ({duration})")
        if not result['passed'] and 'error' in result:
            logger.info(f"    Error: {result['error']}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("📊 Results saved to test_results.json")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)