#!/usr/bin/env python3
"""
Real Researcher Workflow Test for RLDK
Simulates how a researcher would actually use RLDK in their daily work
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
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import ComprehensivePPOForensics
from rldk.determinism import check
from rldk.reward import health
from rldk.evals import run
from rldk.diff import first_divergence
from rldk.utils.seed import set_global_seed, get_current_seed
from rldk.ingest import ingest_runs

# Import transformers for model testing
from transformers import AutoTokenizer, AutoModelForCausalLM

class ResearcherWorkflow:
    """Simulates a real researcher's workflow with RLDK"""
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.model = None
        self.tokenizer = None
        self.experiment_dir = Path("./researcher_experiments")
        self.experiment_dir.mkdir(exist_ok=True)
        
    def step1_load_model(self):
        """Step 1: Load a pre-trained model (like a researcher would)"""
        logger.info("🔬 Step 1: Loading pre-trained model from Hugging Face")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"✅ Loaded {self.model_name} successfully")
            logger.info(f"Model type: {self.model.config.model_type}")
            logger.info(f"Model size: {self.model.num_parameters():,} parameters")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def step2_setup_experiment_tracking(self):
        """Step 2: Set up experiment tracking (like a researcher would)"""
        logger.info("🔬 Step 2: Setting up experiment tracking")
        
        try:
            config = TrackingConfig(
                experiment_name="dialgpt_rl_experiment",
                enable_dataset_tracking=True,
                enable_model_tracking=True,
                enable_environment_tracking=True,
                enable_seed_tracking=True,
                enable_git_tracking=True,
                output_dir=self.experiment_dir
            )
            
            self.tracker = ExperimentTracker(config)
            self.tracker.start_experiment()
            
            # Track the model
            self.tracker.track_model(self.model, "dialgpt_policy")
            self.tracker.track_tokenizer(self.tokenizer, "dialgpt_tokenizer")
            
            # Set reproducible seeds
            self.tracker.set_seeds(42)
            
            # Add metadata
            self.tracker.add_metadata("model_name", self.model_name)
            self.tracker.add_metadata("model_size", self.model.num_parameters())
            self.tracker.add_metadata("experiment_type", "rl_training")
            
            logger.info("✅ Experiment tracking set up successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set up tracking: {e}")
            return False
    
    def step3_generate_training_data(self):
        """Step 3: Generate training data (like a researcher would)"""
        logger.info("🔬 Step 3: Generating training data")
        
        try:
            # Generate prompts for training
            prompts = [
                "The weather today is",
                "In a world where AI",
                "The most important thing is",
                "When I think about the future",
                "Learning to code requires",
                "The best way to succeed",
                "In my opinion, technology",
                "The key to understanding",
                "Machine learning is",
                "The future of work"
            ]
            
            self.training_data = []
            for i in range(500):  # Generate 500 training samples
                prompt = np.random.choice(prompts)
                
                # Simulate RL training metrics
                reward = np.random.normal(0.6, 0.15)  # Simulated reward
                kl_div = np.random.normal(0.12, 0.03)  # Simulated KL divergence
                entropy = np.random.normal(1.8, 0.2)  # Simulated entropy
                
                self.training_data.append({
                    'step': i,
                    'prompt': prompt,
                    'reward': reward,
                    'kl': kl_div,
                    'entropy': entropy,
                    'loss': np.random.normal(0.25, 0.08),
                    'learning_rate': 1e-5,
                    'batch_size': 32,
                    'policy_grad_norm': np.random.normal(1.1, 0.3),
                    'value_grad_norm': np.random.normal(0.9, 0.2),
                    'advantage_mean': np.random.normal(0.08, 0.04),
                    'advantage_std': np.random.normal(0.45, 0.1)
                })
            
            # Track the dataset
            self.tracker.track_dataset(self.training_data, "training_data")
            
            logger.info(f"✅ Generated {len(self.training_data)} training samples")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to generate training data: {e}")
            return False
    
    def step4_run_ppo_forensics(self):
        """Step 4: Run PPO forensics analysis (like a researcher would)"""
        logger.info("🔬 Step 4: Running PPO forensics analysis")
        
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
            
            # Process training data
            for data_point in self.training_data:
                forensics.update(
                    step=data_point['step'],
                    kl=data_point['kl'],
                    kl_coef=0.2,
                    entropy=data_point['entropy'],
                    reward_mean=data_point['reward'],
                    reward_std=0.15,
                    policy_grad_norm=data_point['policy_grad_norm'],
                    value_grad_norm=data_point['value_grad_norm'],
                    total_grad_norm=data_point['policy_grad_norm'] + data_point['value_grad_norm'],
                    advantage_mean=data_point['advantage_mean'],
                    advantage_std=data_point['advantage_std'],
                    advantage_min=np.random.normal(-0.4, 0.1),
                    advantage_max=np.random.normal(0.8, 0.1),
                    advantage_median=np.random.normal(0.05, 0.02),
                    advantage_samples=[np.random.normal(0.1, 0.5) for _ in range(10)]
                )
            
            # Get analysis results
            analysis = forensics.get_comprehensive_analysis()
            anomalies = forensics.get_anomalies()
            health_summary = forensics.get_health_summary()
            
            logger.info(f"✅ PPO forensics completed")
            logger.info(f"Found {len(anomalies)} anomalies")
            logger.info(f"Health summary: {health_summary}")
            
            # Save results
            with open(self.experiment_dir / "ppo_forensics_results.json", "w") as f:
                json.dump({
                    "analysis": analysis,
                    "anomalies": anomalies,
                    "health_summary": health_summary
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"❌ PPO forensics failed: {e}")
            return False
    
    def step5_test_determinism(self):
        """Step 5: Test training determinism (like a researcher would)"""
        logger.info("🔬 Step 5: Testing training determinism")
        
        try:
            # Create a simple training script
            training_script = f'''
import numpy as np
import torch
from rldk.utils.seed import set_global_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set seed for reproducibility
set_global_seed(42, deterministic=True)

# Load model
tokenizer = AutoTokenizer.from_pretrained("{self.model_name}")
model = AutoModelForCausalLM.from_pretrained("{self.model_name}")

# Generate some data
data = np.random.randn(100)
torch_data = torch.randn(50, 10)

# Simple forward pass
inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Save results
with open('training_output.txt', 'w') as f:
    f.write(f'numpy_sum: {{np.sum(data)}}\\n')
    f.write(f'torch_sum: {{torch.sum(torch_data).item()}}\\n')
    f.write(f'logits_sum: {{torch.sum(logits).item()}}\\n')
    f.write(f'logits_mean: {{torch.mean(logits).item()}}\\n')
'''
            
            script_path = self.experiment_dir / "test_training_script.py"
            with open(script_path, "w") as f:
                f.write(training_script)
            
            # Test determinism
            report = check(
                cmd=f"python {script_path}",
                compare=["numpy_sum", "torch_sum", "logits_sum", "logits_mean"],
                replicas=3
            )
            
            logger.info(f"✅ Determinism test completed")
            logger.info(f"Passed: {report.passed}")
            logger.info(f"Replica variance: {report.replica_variance}")
            
            # Save results
            with open(self.experiment_dir / "determinism_results.json", "w") as f:
                json.dump({
                    "passed": report.passed,
                    "replica_variance": report.replica_variance,
                    "mismatches": report.mismatches,
                    "fixes": report.fixes
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"❌ Determinism test failed: {e}")
            return False
    
    def step6_reward_analysis(self):
        """Step 6: Analyze reward model health (like a researcher would)"""
        logger.info("🔬 Step 6: Analyzing reward model health")
        
        try:
            # Convert training data to DataFrame
            df = pd.DataFrame(self.training_data)
            
            # Run reward health analysis
            health_report = health(
                run_data=df,
                reward_col="reward",
                threshold_drift=0.1,
                threshold_saturation=0.8,
                threshold_calibration=0.7
            )
            
            logger.info(f"✅ Reward analysis completed")
            logger.info(f"Health passed: {health_report.passed}")
            logger.info(f"Drift detected: {health_report.drift_detected}")
            logger.info(f"Calibration score: {health_report.calibration_score}")
            
            # Save results
            with open(self.experiment_dir / "reward_analysis_results.json", "w") as f:
                json.dump({
                    "passed": health_report.passed,
                    "drift_detected": health_report.drift_detected,
                    "saturation_issues": health_report.saturation_issues,
                    "calibration_score": health_report.calibration_score,
                    "shortcut_signals": health_report.shortcut_signals,
                    "label_leakage_risk": health_report.label_leakage_risk
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"❌ Reward analysis failed: {e}")
            return False
    
    def step7_evaluation_suite(self):
        """Step 7: Run evaluation suite (like a researcher would)"""
        logger.info("🔬 Step 7: Running evaluation suite")
        
        try:
            # Convert training data to DataFrame
            df = pd.DataFrame(self.training_data)
            
            # Run evaluation suite
            eval_result = run(
                run_data=df,
                suite="quick",
                seed=42,
                sample_size=100
            )
            
            logger.info(f"✅ Evaluation suite completed")
            logger.info(f"Suite: {eval_result.suite_name}")
            logger.info(f"Sample size: {eval_result.sample_size}")
            logger.info(f"Scores: {eval_result.scores}")
            
            # Save results
            with open(self.experiment_dir / "evaluation_results.json", "w") as f:
                json.dump({
                    "suite_name": eval_result.suite_name,
                    "sample_size": eval_result.sample_size,
                    "scores": eval_result.scores,
                    "confidence_intervals": eval_result.confidence_intervals,
                    "effect_sizes": eval_result.effect_sizes
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"❌ Evaluation suite failed: {e}")
            return False
    
    def step8_run_comparison(self):
        """Step 8: Compare different training runs (like a researcher would)"""
        logger.info("🔬 Step 8: Comparing training runs")
        
        try:
            # Generate a second run with some differences
            run2_data = []
            for i, data_point in enumerate(self.training_data):
                new_data = data_point.copy()
                # Introduce some drift after step 200
                if i > 200:
                    new_data['reward'] += 0.2
                    new_data['kl'] += 0.05
                run2_data.append(new_data)
            
            df1 = pd.DataFrame(self.training_data)
            df2 = pd.DataFrame(run2_data)
            
            # Test divergence detection
            divergence_report = first_divergence(
                df_a=df1,
                df_b=df2,
                signals=["reward", "kl", "entropy"],
                k_consecutive=5,
                window=30,
                tolerance=2.0
            )
            
            logger.info(f"✅ Run comparison completed")
            logger.info(f"Diverged: {divergence_report.diverged}")
            if divergence_report.diverged:
                logger.info(f"First divergence at step: {divergence_report.first_step}")
                logger.info(f"Tripped signals: {divergence_report.tripped_signals}")
            
            # Save results
            with open(self.experiment_dir / "run_comparison_results.json", "w") as f:
                json.dump({
                    "diverged": divergence_report.diverged,
                    "first_step": divergence_report.first_step,
                    "tripped_signals": divergence_report.tripped_signals,
                    "notes": divergence_report.notes,
                    "suspected_causes": divergence_report.suspected_causes
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"❌ Run comparison failed: {e}")
            return False
    
    def step9_finish_experiment(self):
        """Step 9: Finish experiment and save results (like a researcher would)"""
        logger.info("🔬 Step 9: Finishing experiment")
        
        try:
            # Add final metadata
            self.tracker.add_metadata("total_steps", len(self.training_data))
            self.tracker.add_metadata("experiment_status", "completed")
            self.tracker.add_metadata("final_timestamp", time.time())
            
            # Finish experiment
            self.tracker.finish_experiment()
            
            logger.info("✅ Experiment completed and saved")
            logger.info(f"Results saved to: {self.experiment_dir}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to finish experiment: {e}")
            return False
    
    def run_full_workflow(self):
        """Run the complete researcher workflow"""
        logger.info("🚀 Starting Complete Researcher Workflow")
        logger.info("=" * 60)
        
        steps = [
            ("Load Model", self.step1_load_model),
            ("Setup Tracking", self.step2_setup_experiment_tracking),
            ("Generate Data", self.step3_generate_training_data),
            ("PPO Forensics", self.step4_run_ppo_forensics),
            ("Test Determinism", self.step5_test_determinism),
            ("Reward Analysis", self.step6_reward_analysis),
            ("Evaluation Suite", self.step7_evaluation_suite),
            ("Run Comparison", self.step8_run_comparison),
            ("Finish Experiment", self.step9_finish_experiment),
        ]
        
        results = {}
        start_time = time.time()
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*20} {step_name} {'='*20}")
            step_start = time.time()
            
            try:
                success = step_func()
                step_duration = time.time() - step_start
                results[step_name] = {
                    'success': success,
                    'duration': step_duration
                }
                
                if success:
                    logger.info(f"✅ {step_name} completed in {step_duration:.2f}s")
                else:
                    logger.error(f"❌ {step_name} failed in {step_duration:.2f}s")
                    
            except Exception as e:
                step_duration = time.time() - step_start
                logger.error(f"❌ {step_name} crashed: {e}")
                results[step_name] = {
                    'success': False,
                    'duration': step_duration,
                    'error': str(e)
                }
        
        # Summary
        total_duration = time.time() - start_time
        successful_steps = sum(1 for r in results.values() if r['success'])
        total_steps = len(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("🏁 RESEARCHER WORKFLOW SUMMARY")
        logger.info("=" * 60)
        
        for step_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = f"{result['duration']:.2f}s"
            logger.info(f"{status} {step_name:<20} ({duration})")
            if not result['success'] and 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        logger.info(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully")
        logger.info(f"Total time: {total_duration:.2f}s")
        logger.info(f"Results directory: {self.experiment_dir}")
        
        # Save summary
        with open(self.experiment_dir / "workflow_summary.json", "w") as f:
            json.dump({
                "total_duration": total_duration,
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "results": results
            }, f, indent=2)
        
        return successful_steps == total_steps

def main():
    """Main function"""
    workflow = ResearcherWorkflow()
    success = workflow.run_full_workflow()
    
    if success:
        logger.info("🎉 All tests passed! RLDK is working well for researcher workflows.")
    else:
        logger.error("💥 Some tests failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)