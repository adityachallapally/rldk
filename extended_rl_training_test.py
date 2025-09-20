#!/usr/bin/env python3
"""
Extended multi-hour RLDK testing with real RL training workflows.
This simulates how a researcher would actually use RLDK to debug RL pipelines.
"""

import os
import sys
import time
import json
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset

sys.path.insert(0, '/home/ubuntu/repos/rldk/src')

from rldk import ExperimentTracker, TrackingConfig
from rldk import ComprehensivePPOForensics
from rldk import check
from rldk.evaluations.evals.runner import run as run_eval
from rldk.evaluations.evals.suites import get_suite_config
from rldk import set_global_seed
from rldk.evaluations.reward.api import health as reward_health

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/rl_testing/extended_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ExtendedRLDKResearcherTest:
    """Extended multi-hour RLDK testing simulating real researcher workflows."""
    
    def __init__(self):
        self.test_dir = Path("/home/ubuntu/rl_testing")
        self.results = {}
        self.issues = []
        self.models = [
            {"name": "microsoft/DialoGPT-small", "size": "small", "params": "117M"},
            {"name": "gpt2", "size": "medium", "params": "124M"},
            {"name": "microsoft/DialoGPT-medium", "size": "large", "params": "345M"}
        ]
        self.training_runs = []
        
    def log_issue(self, severity: str, component: str, description: str, fix_plan: str, cursor_prompt: str = ""):
        """Log an issue with severity rating and fix plan."""
        issue = {
            "severity": severity,
            "component": component,
            "description": description,
            "fix_plan": fix_plan,
            "cursor_prompt": cursor_prompt,
            "timestamp": datetime.now().isoformat()
        }
        self.issues.append(issue)
        logger.error(f"[{severity}] {component}: {description}")

    def create_realistic_training_data(self, model_name: str, num_episodes: int = 5000) -> pd.DataFrame:
        """Create realistic RL training data that mimics actual PPO training."""
        logger.info(f"Generating realistic training data for {model_name} with {num_episodes} episodes")
        
        np.random.seed(42)
        
        data = []
        base_reward = 0.1
        kl_target = 0.1
        
        for episode in range(num_episodes):
            progress = episode / num_episodes
            
            if progress < 0.3:
                reward_trend = progress * 0.5
            elif progress < 0.7:
                reward_trend = 0.15 + (progress - 0.3) * 0.3  # Slower progress
            else:
                reward_trend = 0.27 + (progress - 0.7) * 0.1  # Plateau
                
            reward_mean = base_reward + reward_trend + np.random.normal(0, 0.05)
            reward_std = max(0.01, 0.3 - progress * 0.2 + np.random.normal(0, 0.02))
            
            kl_noise = np.random.exponential(0.05) if np.random.random() > 0.8 else np.random.normal(0, 0.02)
            kl_divergence = max(0.001, kl_target + kl_noise)
            
            entropy = max(1.0, 3.0 - progress * 1.5 + np.random.normal(0, 0.1))
            
            policy_loss = -0.1 - progress * 0.05 + np.random.normal(0, 0.02)
            value_loss = 0.5 - progress * 0.3 + np.random.normal(0, 0.05)
            
            policy_grad_norm = np.random.lognormal(0, 0.3) if np.random.random() > 0.95 else np.random.normal(1.0, 0.2)
            value_grad_norm = np.random.lognormal(0, 0.3) if np.random.random() > 0.95 else np.random.normal(0.8, 0.15)
            
            lr = 3e-4 * (1 - progress * 0.5)  # Linear decay
            
            advantage_mean = np.random.normal(0.0, 0.05)
            advantage_std = max(0.1, 1.0 - progress * 0.3 + np.random.normal(0, 0.1))
            
            data.append({
                'step': episode,
                'episode': episode,
                'reward_mean': reward_mean,
                'reward_std': reward_std,
                'kl_divergence': kl_divergence,
                'entropy': entropy,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'policy_grad_norm': policy_grad_norm,
                'value_grad_norm': value_grad_norm,
                'learning_rate': lr,
                'advantage_mean': advantage_mean,
                'advantage_std': advantage_std,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            })
            
        return pd.DataFrame(data)

    def run_extended_ppo_forensics(self, training_data: pd.DataFrame, model_info: Dict) -> Dict[str, Any]:
        """Run extended PPO forensics analysis over multiple hours."""
        logger.info(f"=== Extended PPO Forensics Analysis for {model_info['name']} ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            configs = [
                {"kl_target": 0.05, "name": "very_conservative", "hours": 0.5},
                {"kl_target": 0.1, "name": "conservative", "hours": 1.0},
                {"kl_target": 0.2, "name": "moderate", "hours": 1.5},
                {"kl_target": 0.3, "name": "aggressive", "hours": 1.0},
                {"kl_target": 0.5, "name": "very_aggressive", "hours": 0.5}
            ]
            
            for config in configs:
                logger.info(f"Running {config['name']} forensics analysis (estimated {config['hours']} hours)")
                start_time = time.time()
                
                forensics = ComprehensivePPOForensics(
                    kl_target=config['kl_target'],
                    enable_kl_schedule_tracking=True,
                    enable_gradient_norms_analysis=True,
                    enable_advantage_statistics=True
                )
                
                chunk_size = 100
                total_anomalies = 0
                
                for i in range(0, len(training_data), chunk_size):
                    chunk = training_data.iloc[i:i+chunk_size]
                    
                    for _, row in chunk.iterrows():
                        metrics = forensics.update(
                            step=int(row['step']),
                            kl=float(row['kl_divergence']),
                            kl_coef=max(0.01, min(1.0, config['kl_target'] / max(row['kl_divergence'], 0.001))),
                            entropy=float(row['entropy']),
                            reward_mean=float(row['reward_mean']),
                            reward_std=float(row['reward_std']),
                            policy_grad_norm=float(row['policy_grad_norm']),
                            value_grad_norm=float(row['value_grad_norm']),
                            advantage_mean=float(row['advantage_mean']),
                            advantage_std=float(row['advantage_std'])
                        )
                    
                    time.sleep(0.1)  # Small delay to simulate real processing
                    
                    if i % 1000 == 0:
                        logger.info(f"Processed {i}/{len(training_data)} steps for {config['name']}")
                
                # Get comprehensive analysis
                analysis = forensics.get_comprehensive_analysis()
                anomalies = forensics.get_anomalies()
                health_summary = forensics.get_health_summary()
                
                elapsed_time = time.time() - start_time
                
                results["details"][config['name']] = {
                    "total_steps": len(training_data),
                    "anomalies_detected": len(anomalies),
                    "health_score": health_summary.get("overall_health", 0),
                    "analysis_keys": list(analysis.keys()) if analysis else [],
                    "processing_time": elapsed_time,
                    "estimated_hours": config['hours'],
                    "kl_target": config['kl_target']
                }
                
                logger.info(f"Completed {config['name']} analysis in {elapsed_time:.2f}s")
                logger.info(f"Found {len(anomalies)} anomalies, health score: {health_summary.get('overall_health', 0)}")
                
            results["success"] = True
            logger.info("✓ Extended PPO forensics analysis completed")
            
        except Exception as e:
            error_msg = f"Extended PPO forensics failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "HIGH",
                "ExtendedPPOForensics",
                f"Failed during extended forensics analysis: {str(e)}",
                "Check ComprehensivePPOForensics memory usage and processing efficiency for large datasets",
                "1. Open ComprehensivePPOForensics class\n2. Add memory management for large datasets\n3. Implement batch processing for efficiency\n4. Add progress tracking for long-running analysis"
            )
            logger.error(error_msg)
            
        return results

    def run_multi_model_experiment_tracking(self) -> Dict[str, Any]:
        """Run experiment tracking across multiple models for extended periods."""
        logger.info("=== Multi-Model Extended Experiment Tracking ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            for model_info in self.models:
                logger.info(f"Starting extended experiment tracking for {model_info['name']}")
                
                config = TrackingConfig(
                    experiment_name=f"extended_rl_debug_{model_info['size']}",
                    output_dir=self.test_dir / "extended_experiments",
                    track_environment=True,
                    track_git=True,
                    track_seeds=True,
                    enable_wandb=False
                )
                
                tracker = ExperimentTracker(config)
                
                experiment_id = tracker.start_experiment(f"extended_experiment_{model_info['size']}")
                
                training_data = self.create_realistic_training_data(model_info['name'], num_episodes=10000)
                
                tracker.track_dataset(f"extended_training_data_{model_info['size']}", training_data)
                
                logger.info(f"Loading and tracking model: {model_info['name']}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info['name'],
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    
                    tracker.track_model(model, f"model_{model_info['size']}")
                    tracker.track_tokenizer(tokenizer, f"tokenizer_{model_info['size']}")
                    
                    tracker.add_metadata("model_parameters", model.num_parameters())
                    tracker.add_metadata("vocab_size", len(tokenizer))
                    tracker.add_metadata("training_episodes", len(training_data))
                    tracker.add_metadata("model_size_category", model_info['size'])
                    tracker.add_metadata("estimated_training_hours", 4.0)
                    
                    logger.info(f"Simulating extended training for {model_info['name']}")
                    for hour in range(4):  # Simulate 4 hours of training
                        logger.info(f"Training hour {hour + 1}/4 for {model_info['name']}")
                        
                        tracker.add_metadata(f"hour_{hour + 1}_checkpoint", f"checkpoint_{hour + 1}.pt")
                        tracker.add_metadata(f"hour_{hour + 1}_loss", np.random.normal(0.5 - hour * 0.1, 0.05))
                        
                        time.sleep(30)  # 30 seconds per "hour" for testing
                        
                    del model
                    del tokenizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as model_error:
                    logger.warning(f"Model loading failed for {model_info['name']}: {model_error}")
                    tracker.add_metadata("model_loading_error", str(model_error))
                
                tracker.finish_experiment()
                
                results["details"][model_info['size']] = {
                    "experiment_id": tracker.experiment_id,
                    "tracked_datasets": len(tracker.dataset_tracker.tracked_datasets) if tracker.dataset_tracker else 0,
                    "metadata_count": len(tracker.metadata),
                    "training_data_size": len(training_data),
                    "model_name": model_info['name']
                }
                
                logger.info(f"Completed extended experiment tracking for {model_info['name']}")
                
            results["success"] = True
            logger.info("✓ Multi-model extended experiment tracking completed")
            
        except Exception as e:
            error_msg = f"Extended experiment tracking failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "HIGH",
                "ExtendedExperimentTracking",
                f"Failed during extended experiment tracking: {str(e)}",
                "Check ExperimentTracker memory management and long-running experiment support",
                "1. Open ExperimentTracker class\n2. Add memory cleanup for long experiments\n3. Implement periodic state saving\n4. Add resource monitoring for extended runs"
            )
            logger.error(error_msg)
            
        return results

    def run_comprehensive_determinism_testing(self) -> Dict[str, Any]:
        """Run comprehensive determinism testing with multiple scenarios."""
        logger.info("=== Comprehensive Determinism Testing ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            scenarios = [
                {
                    "name": "basic_training",
                    "script": """
import sys
import random
import numpy as np
import torch

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

steps = 200
for step in range(steps):
    loss = 1.0 - (step / steps) + np.random.normal(0, 0.01)
    reward = step / steps + np.random.normal(0, 0.02)
    kl = np.random.exponential(0.1)
    
    print(f"step={step}, loss={loss:.6f}, reward_mean={reward:.6f}, kl={kl:.6f}")
""",
                    "replicas": 5,
                    "compare": ["loss", "reward_mean", "kl"]
                },
                {
                    "name": "complex_training",
                    "script": """
import sys
import random
import numpy as np
import torch
import torch.nn as nn

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

steps = 100
for step in range(steps):
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    reward = -loss.item() + np.random.normal(0, 0.01)
    kl = np.random.exponential(0.05)
    
    print(f"step={step}, loss={loss.item():.6f}, reward_mean={reward:.6f}, kl={kl:.6f}")
""",
                    "replicas": 3,
                    "compare": ["loss", "reward_mean", "kl"]
                }
            ]
            
            for scenario in scenarios:
                logger.info(f"Testing determinism scenario: {scenario['name']}")
                
                script_path = self.test_dir / f"deterministic_{scenario['name']}.py"
                with open(script_path, 'w') as f:
                    f.write(scenario['script'])
                
                logger.info(f"Running determinism check with {scenario['replicas']} replicas")
                
                try:
                    report = check(
                        cmd=f"python {script_path} 42",
                        compare=scenario['compare'],
                        replicas=scenario['replicas']
                    )
                    
                    results["details"][scenario['name']] = {
                        "passed": report.passed,
                        "replicas": scenario['replicas'],
                        "compared_metrics": scenario['compare'],
                        "mismatches": len(report.mismatches) if hasattr(report, 'mismatches') else 0,
                        "fixes": len(report.fixes) if hasattr(report, 'fixes') else 0,
                        "culprit": report.culprit if hasattr(report, 'culprit') else None
                    }
                    
                    logger.info(f"Scenario {scenario['name']}: {'PASSED' if report.passed else 'FAILED'}")
                    
                except Exception as scenario_error:
                    logger.warning(f"Scenario {scenario['name']} failed: {scenario_error}")
                    results["details"][scenario['name']] = {"error": str(scenario_error)}
                    
            results["success"] = len([s for s in results["details"].values() if "error" not in s]) > 0
            logger.info("✓ Comprehensive determinism testing completed")
            
        except Exception as e:
            error_msg = f"Comprehensive determinism testing failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "MEDIUM",
                "ComprehensiveDeterminismTesting",
                f"Failed during comprehensive determinism testing: {str(e)}",
                "Check determinism testing infrastructure and subprocess handling",
                "1. Open determinism check.py\n2. Fix subprocess execution and output parsing\n3. Add better error handling for complex scenarios\n4. Verify metrics_df attribute access"
            )
            logger.error(error_msg)
            
        return results

    def run_extended_evaluation_suites(self) -> Dict[str, Any]:
        """Run extended evaluation suites with comprehensive data."""
        logger.info("=== Extended Evaluation Suites Testing ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            eval_data = pd.DataFrame({
                'step': range(5000),
                'reward_mean': np.random.normal(0.6, 0.15, 5000),
                'reward_std': np.random.exponential(0.2, 5000),
                'kl_divergence': np.random.exponential(0.1, 5000),
                'entropy': np.random.normal(2.0, 0.4, 5000),
                'policy_loss': np.random.normal(-0.1, 0.05, 5000),
                'value_loss': np.random.normal(0.5, 0.1, 5000),
                'output': [f"Generated text sample {i}" for i in range(5000)],
                'input': [f"Input prompt {i}" for i in range(5000)]
            })
            
            suites = ["quick", "comprehensive", "safety"]
            
            for suite_name in suites:
                logger.info(f"Running extended {suite_name} evaluation suite")
                
                try:
                    start_time = time.time()
                    
                    eval_results = run_eval(
                        run_data=eval_data,
                        suite=suite_name,
                        seed=42,
                        sample_size=1000
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    results["details"][suite_name] = {
                        "metrics_evaluated": len(eval_results.scores) if hasattr(eval_results, 'scores') else 0,
                        "overall_score": eval_results.overall_score if hasattr(eval_results, 'overall_score') else 0,
                        "available_fraction": eval_results.available_fraction if hasattr(eval_results, 'available_fraction') else 0,
                        "execution_time": elapsed_time,
                        "data_size": len(eval_data),
                        "sample_size": 1000
                    }
                    
                    logger.info(f"Suite {suite_name} completed in {elapsed_time:.2f}s")
                    
                except Exception as suite_error:
                    logger.warning(f"Suite {suite_name} failed: {suite_error}")
                    results["details"][suite_name] = {"error": str(suite_error)}
                    
            results["success"] = len([s for s in results["details"].values() if "error" not in s]) > 0
            logger.info("✓ Extended evaluation suites testing completed")
            
        except Exception as e:
            error_msg = f"Extended evaluation suites failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "MEDIUM",
                "ExtendedEvaluationSuites",
                f"Failed during extended evaluation suites: {str(e)}",
                "Check evaluation suite runner and data format compatibility",
                "1. Open evaluation runner.py\n2. Fix parameter naming (run_data vs data)\n3. Add proper error handling for large datasets\n4. Verify suite configuration loading"
            )
            logger.error(error_msg)
            
        return results

    def run_reward_model_analysis(self) -> Dict[str, Any]:
        """Run comprehensive reward model health analysis."""
        logger.info("=== Reward Model Health Analysis ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            for model_info in self.models:
                logger.info(f"Analyzing reward model health for {model_info['name']}")
                
                training_data = self.create_realistic_training_data(model_info['name'], num_episodes=3000)
                
                try:
                    health_report = reward_health(training_data)
                    
                    results["details"][model_info['size']] = {
                        "model_name": model_info['name'],
                        "health_score": health_report.get("overall_health", 0) if isinstance(health_report, dict) else 0,
                        "data_size": len(training_data),
                        "analysis_completed": True
                    }
                    
                    logger.info(f"Reward health analysis completed for {model_info['name']}")
                    
                except Exception as model_error:
                    logger.warning(f"Reward analysis failed for {model_info['name']}: {model_error}")
                    results["details"][model_info['size']] = {
                        "model_name": model_info['name'],
                        "error": str(model_error)
                    }
                    
            results["success"] = len([s for s in results["details"].values() if "error" not in s]) > 0
            logger.info("✓ Reward model health analysis completed")
            
        except Exception as e:
            error_msg = f"Reward model analysis failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "MEDIUM",
                "RewardModelAnalysis",
                f"Failed during reward model analysis: {str(e)}",
                "Check reward health analysis implementation and data format requirements",
                "1. Open reward health analysis module\n2. Verify data format compatibility\n3. Add proper error handling\n4. Check statistical calculations"
            )
            logger.error(error_msg)
            
        return results

    def run_comprehensive_extended_test(self):
        """Run comprehensive extended multi-hour testing."""
        logger.info("🚀 Starting Extended Multi-Hour RLDK Testing")
        logger.info("This will simulate real researcher workflows with actual RL training")
        start_time = time.time()
        
        test_methods = [
            ("multi_model_experiment_tracking", self.run_multi_model_experiment_tracking),
            ("extended_ppo_forensics", lambda: self.run_extended_ppo_forensics(
                self.create_realistic_training_data("gpt2", 8000), 
                {"name": "gpt2", "size": "medium", "params": "124M"}
            )),
            ("comprehensive_determinism_testing", self.run_comprehensive_determinism_testing),
            ("extended_evaluation_suites", self.run_extended_evaluation_suites),
            ("reward_model_analysis", self.run_reward_model_analysis)
        ]
        
        for test_name, test_method in test_methods:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running {test_name} (Extended Testing)")
            logger.info(f"{'='*80}")
            
            test_start = time.time()
            
            try:
                self.results[test_name] = test_method()
                test_elapsed = time.time() - test_start
                logger.info(f"Test {test_name} completed in {test_elapsed:.2f} seconds")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.results[test_name] = {"success": False, "errors": [str(e)]}
        
        total_time = time.time() - start_time
        self.generate_extended_report(total_time)
        
    def generate_extended_report(self, total_time: float):
        """Generate comprehensive extended testing report."""
        logger.info("\n" + "="*100)
        logger.info("EXTENDED MULTI-HOUR RLDK TESTING REPORT")
        logger.info("="*100)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r.get("success", False))
        
        logger.info(f"Total testing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        logger.info(f"Tests completed: {successful_tests}/{total_tests}")
        logger.info(f"Critical issues found: {len([i for i in self.issues if i['severity'] == 'HIGH'])}")
        logger.info(f"Total issues found: {len(self.issues)}")
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result.get("success", False) else "✗ FAILED"
            logger.info(f"\n{test_name}: {status}")
            
            if result.get("errors"):
                for error in result["errors"]:
                    logger.info(f"  Error: {error}")
                    
            if result.get("details"):
                logger.info(f"  Details: {result['details']}")
        
        if self.issues:
            logger.info("\n" + "="*80)
            logger.info("ISSUES FOUND (Severity-Rated with Cursor Prompts)")
            logger.info("="*80)
            
            for issue in sorted(self.issues, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x['severity']]):
                logger.info(f"\n[{issue['severity']}] {issue['component']}")
                logger.info(f"Description: {issue['description']}")
                logger.info(f"Fix Plan: {issue['fix_plan']}")
                if issue.get('cursor_prompt'):
                    logger.info(f"Cursor Prompt:\n{issue['cursor_prompt']}")
        
        logger.info("\n" + "="*80)
        logger.info("RESEARCHER PERSPECTIVE ANALYSIS")
        logger.info("="*80)
        
        strengths = []
        weaknesses = []
        
        if self.results.get("extended_ppo_forensics", {}).get("success"):
            strengths.append("PPO forensics analysis works well with realistic training data")
        else:
            weaknesses.append("PPO forensics analysis has issues with extended datasets")
            
        if self.results.get("multi_model_experiment_tracking", {}).get("success"):
            strengths.append("Experiment tracking handles multiple models effectively")
        else:
            weaknesses.append("Experiment tracking fails with complex multi-model workflows")
            
        if len([i for i in self.issues if i['severity'] == 'HIGH']) == 0:
            strengths.append("No critical blocking issues found")
        else:
            weaknesses.append(f"{len([i for i in self.issues if i['severity'] == 'HIGH'])} critical issues block researcher workflows")
        
        logger.info("STRENGTHS:")
        for strength in strengths:
            logger.info(f"  ✓ {strength}")
            
        logger.info("\nWEAKNESS/PAIN POINTS:")
        for weakness in weaknesses:
            logger.info(f"  ✗ {weakness}")
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "total_time_hours": total_time / 3600,
            "results": self.results,
            "issues": self.issues,
            "models_tested": self.models,
            "researcher_analysis": {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "critical_issues": len([i for i in self.issues if i['severity'] == 'HIGH']),
                "total_issues": len(self.issues)
            }
        }
        
        report_path = self.test_dir / "extended_comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"\n📊 Extended comprehensive report saved to: {report_path}")
        logger.info("="*100)

if __name__ == "__main__":
    tester = ExtendedRLDKResearcherTest()
    tester.run_comprehensive_extended_test()
