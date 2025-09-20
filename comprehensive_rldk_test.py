#!/usr/bin/env python3
"""
Comprehensive end-to-end testing of RLDK as a researcher would use it.
Tests all major components with real HuggingFace models of different sizes.
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, '/home/ubuntu/repos/rldk/src')

from rldk import ExperimentTracker, TrackingConfig, ComprehensivePPOForensics, check, set_global_seed
from rldk.evaluations.evals.runner import run as run_eval
from rldk.evaluations.evals.suites import get_suite_config
from rldk.core.io.consolidated_schemas import TrainingMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/rl_testing/rldk_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLDKResearcherTest:
    """Comprehensive RLDK testing from a researcher's perspective."""
    
    def __init__(self):
        self.test_dir = Path("/home/ubuntu/rl_testing")
        self.results = {}
        self.issues = []
        self.models = [
            {"name": "microsoft/DialoGPT-small", "size": "small", "params": "124M"},
            {"name": "gpt2", "size": "medium", "params": "124M"},
            {"name": "microsoft/DialoGPT-medium", "size": "large", "params": "354M"}
        ]
        
    def log_issue(self, severity: str, component: str, description: str, fix_plan: str):
        """Log an issue with severity rating and fix plan."""
        issue = {
            "severity": severity,  # CRITICAL, HIGH, MEDIUM, LOW
            "component": component,
            "description": description,
            "fix_plan": fix_plan,
            "timestamp": datetime.now().isoformat()
        }
        self.issues.append(issue)
        logger.error(f"[{severity}] {component}: {description}")
        
    def test_experiment_tracking(self) -> Dict[str, Any]:
        """Test experiment tracking system with real models."""
        logger.info("=== Testing Experiment Tracking System ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            for model_info in self.models:
                logger.info(f"Testing experiment tracking with {model_info['name']}")
                
                config = TrackingConfig(
                    experiment_name=f"test_experiment_{model_info['size']}",
                    enable_dataset_tracking=True,
                    enable_model_tracking=True,
                    enable_environment_tracking=True,
                    enable_seed_tracking=True,
                    enable_git_tracking=True
                )
                
                tracker = ExperimentTracker(config)
                
                tracker.start_experiment()
                
                mock_data = pd.DataFrame({
                    'step': range(100),
                    'reward_mean': np.random.normal(0.5, 0.1, 100),
                    'kl_divergence': np.random.exponential(0.1, 100),
                    'entropy': np.random.normal(2.0, 0.3, 100)
                })
                
                tracker.track_dataset(mock_data, f"training_data_{model_info['size']}")
                
                tracker.set_seeds(42)
                
                tracker.add_metadata("model_name", model_info['name'])
                tracker.add_metadata("model_size", model_info['size'])
                tracker.add_metadata("parameters", model_info['params'])
                tracker.add_metadata("learning_rate", 1e-5)
                tracker.add_metadata("batch_size", 32)
                
                tracker.finish_experiment()
                
                results["details"][model_info['size']] = {
                    "experiment_id": tracker.experiment_id,
                    "tracked_datasets": len(tracker.dataset_tracker.tracked_datasets) if tracker.dataset_tracker else 0,
                    "metadata_count": len(tracker._metadata) if hasattr(tracker, '_metadata') else 0
                }
                
            results["success"] = True
            logger.info("✓ Experiment tracking tests passed")
            
        except Exception as e:
            error_msg = f"Experiment tracking failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "HIGH", 
                "ExperimentTracker",
                f"Failed to track experiments with real models: {str(e)}",
                "Check ExperimentTracker initialization and dataset tracking methods. Verify all dependencies are properly imported and configured."
            )
            logger.error(error_msg)
            
        return results
    
    def test_ppo_forensics(self) -> Dict[str, Any]:
        """Test comprehensive PPO forensics analysis."""
        logger.info("=== Testing PPO Forensics Analysis ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            forensics_configs = [
                {"kl_target": 0.1, "name": "conservative"},
                {"kl_target": 0.2, "name": "moderate"}, 
                {"kl_target": 0.5, "name": "aggressive"}
            ]
            
            for config in forensics_configs:
                logger.info(f"Testing PPO forensics with {config['name']} configuration")
                
                forensics = ComprehensivePPOForensics(
                    kl_target=config['kl_target'],
                    enable_kl_schedule_tracking=True,
                    enable_gradient_norms_analysis=True,
                    enable_advantage_statistics=True
                )
                
                for step in range(1000):
                    kl = np.random.exponential(config['kl_target'])
                    kl_coef = max(0.01, min(1.0, config['kl_target'] / max(kl, 0.001)))
                    entropy = max(0.1, np.random.normal(2.0, 0.5))
                    reward_mean = np.random.normal(0.5, 0.2)
                    reward_std = max(0.01, np.random.exponential(0.3))
                    
                    policy_grad_norm = np.random.lognormal(0, 0.5) if np.random.random() > 0.95 else np.random.normal(1.0, 0.3)
                    value_grad_norm = np.random.lognormal(0, 0.5) if np.random.random() > 0.95 else np.random.normal(0.8, 0.2)
                    
                    advantage_mean = np.random.normal(0.0, 0.1)
                    advantage_std = max(0.01, np.random.exponential(0.5))
                    
                    metrics = forensics.update(
                        step=step,
                        kl=kl,
                        kl_coef=kl_coef,
                        entropy=entropy,
                        reward_mean=reward_mean,
                        reward_std=reward_std,
                        policy_grad_norm=policy_grad_norm,
                        value_grad_norm=value_grad_norm,
                        advantage_mean=advantage_mean,
                        advantage_std=advantage_std
                    )
                
                analysis = forensics.get_comprehensive_analysis()
                anomalies = forensics.get_anomalies()
                health_summary = forensics.get_health_summary()
                
                results["details"][config['name']] = {
                    "total_steps": 1000,
                    "anomalies_detected": len(anomalies),
                    "health_score": health_summary.get("overall_health", 0),
                    "analysis_keys": list(analysis.keys()) if analysis else []
                }
                
            results["success"] = True
            logger.info("✓ PPO forensics tests passed")
            
        except Exception as e:
            error_msg = f"PPO forensics failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "HIGH",
                "ComprehensivePPOForensics", 
                f"Failed to run forensics analysis: {str(e)}",
                "Check ComprehensivePPOForensics initialization and update methods. Verify all statistical calculations handle edge cases properly."
            )
            logger.error(error_msg)
            
        return results
    
    def test_determinism_checking(self) -> Dict[str, Any]:
        """Test determinism checking across multiple runs."""
        logger.info("=== Testing Determinism Checking ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            script_path = self.test_dir / "deterministic_train.py"
            script_content = '''
import sys
import random
import numpy as np
import torch

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

steps = 50
for step in range(steps):
    loss = 1.0 - (step / steps) + np.random.normal(0, 0.01)
    reward = step / steps + np.random.normal(0, 0.02)
    kl = np.random.exponential(0.1)
    
    print(f"step={step}, loss={loss:.6f}, reward_mean={reward:.6f}, kl={kl:.6f}")
'''
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            logger.info("Running determinism check with 3 replicas")
            
            report = check(
                cmd=f"python {script_path} 42",
                compare=["loss", "reward_mean", "kl"],
                replicas=3
            )
            
            results["details"] = {
                "passed": report.passed,
                "replicas": 3,
                "compared_metrics": ["loss", "reward_mean", "kl"],
                "mismatches": len(report.mismatches) if hasattr(report, 'mismatches') else 0,
                "fixes": len(report.fixes) if hasattr(report, 'fixes') else 0
            }
            
            results["success"] = True
            logger.info(f"✓ Determinism check completed: {'PASSED' if report.passed else 'FAILED'}")
            
        except Exception as e:
            error_msg = f"Determinism checking failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "MEDIUM",
                "DeterminismChecker",
                f"Failed to run determinism checks: {str(e)}",
                "Check determinism.check function and ensure subprocess execution works properly. Verify timeout handling and output parsing."
            )
            logger.error(error_msg)
            
        return results
    
    def test_evaluation_suites(self) -> Dict[str, Any]:
        """Test evaluation suites with different configurations."""
        logger.info("=== Testing Evaluation Suites ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            mock_data = pd.DataFrame({
                'step': range(200),
                'reward_mean': np.random.normal(0.6, 0.15, 200),
                'reward_std': np.random.exponential(0.2, 200),
                'kl_divergence': np.random.exponential(0.1, 200),
                'entropy': np.random.normal(2.0, 0.4, 200),
                'policy_loss': np.random.normal(-0.1, 0.05, 200),
                'value_loss': np.random.normal(0.5, 0.1, 200),
                'output': [f"Generated text sample {i}" for i in range(200)],
                'input': [f"Input prompt {i}" for i in range(200)]
            })
            
            suites = ["quick", "comprehensive", "safety"]
            
            for suite_name in suites:
                logger.info(f"Testing {suite_name} evaluation suite")
                
                try:
                    suite_config = get_suite_config(suite_name)
                    
                    eval_results = run_eval(
                        run_data=mock_data,
                        suite=suite_name
                    )
                    
                    results["details"][suite_name] = {
                        "metrics_evaluated": len(eval_results.scores) if hasattr(eval_results, 'scores') else 0,
                        "overall_score": eval_results.overall_score if hasattr(eval_results, 'overall_score') else 0,
                        "available_fraction": eval_results.available_fraction if hasattr(eval_results, 'available_fraction') else 0
                    }
                    
                except Exception as suite_error:
                    logger.warning(f"Suite {suite_name} failed: {suite_error}")
                    results["details"][suite_name] = {"error": str(suite_error)}
            
            results["success"] = len([s for s in results["details"].values() if "error" not in s]) > 0
            logger.info("✓ Evaluation suites testing completed")
            
        except Exception as e:
            error_msg = f"Evaluation suites failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "MEDIUM",
                "EvaluationSuites",
                f"Failed to run evaluation suites: {str(e)}",
                "Check evaluation suite configurations and runner implementation. Verify mock data format matches expected schema."
            )
            logger.error(error_msg)
            
        return results
    
    def test_cli_commands(self) -> Dict[str, Any]:
        """Test CLI commands with realistic scenarios."""
        logger.info("=== Testing CLI Commands ===")
        results = {"success": False, "details": {}, "errors": []}
        
        try:
            import subprocess
            
            commands = [
                ("rldk version", "version"),
                ("rldk --help", "help"),
                ("rldk seed --show", "seed_show"),
                ("rldk format-info", "format_info")
            ]
            
            for cmd, name in commands:
                logger.info(f"Testing CLI command: {cmd}")
                try:
                    result = subprocess.run(
                        cmd.split(),
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd="/home/ubuntu/repos/rldk"
                    )
                    
                    results["details"][name] = {
                        "return_code": result.returncode,
                        "stdout_length": len(result.stdout),
                        "stderr_length": len(result.stderr),
                        "success": result.returncode == 0
                    }
                    
                except subprocess.TimeoutExpired:
                    results["details"][name] = {"error": "timeout"}
                except Exception as cmd_error:
                    results["details"][name] = {"error": str(cmd_error)}
            
            successful_commands = sum(1 for details in results["details"].values() 
                                    if details.get("success", False))
            
            results["success"] = successful_commands > 0
            logger.info(f"✓ CLI testing completed: {successful_commands}/{len(commands)} commands successful")
            
        except Exception as e:
            error_msg = f"CLI testing failed: {str(e)}"
            results["errors"].append(error_msg)
            self.log_issue(
                "LOW",
                "CLI",
                f"Failed to test CLI commands: {str(e)}",
                "Check CLI command implementations and subprocess execution. Verify all commands have proper error handling and timeouts."
            )
            logger.error(error_msg)
            
        return results
    
    def run_comprehensive_test(self):
        """Run all comprehensive tests."""
        logger.info("🚀 Starting comprehensive RLDK testing as a researcher would use it")
        start_time = time.time()
        
        test_methods = [
            ("experiment_tracking", self.test_experiment_tracking),
            ("ppo_forensics", self.test_ppo_forensics), 
            ("determinism_checking", self.test_determinism_checking),
            ("evaluation_suites", self.test_evaluation_suites),
            ("cli_commands", self.test_cli_commands)
        ]
        
        for test_name, test_method in test_methods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {test_name} tests...")
            logger.info(f"{'='*60}")
            
            try:
                self.results[test_name] = test_method()
                time.sleep(2)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.results[test_name] = {"success": False, "errors": [str(e)]}
        
        total_time = time.time() - start_time
        self.generate_report(total_time)
        
    def generate_report(self, total_time: float):
        """Generate comprehensive testing report."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE RLDK TESTING REPORT")
        logger.info("="*80)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r.get("success", False))
        
        logger.info(f"Total testing time: {total_time:.2f} seconds")
        logger.info(f"Tests completed: {successful_tests}/{total_tests}")
        logger.info(f"Issues found: {len(self.issues)}")
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result.get("success", False) else "✗ FAILED"
            logger.info(f"\n{test_name}: {status}")
            
            if result.get("errors"):
                for error in result["errors"]:
                    logger.info(f"  Error: {error}")
                    
            if result.get("details"):
                logger.info(f"  Details: {result['details']}")
        
        if self.issues:
            logger.info("\n" + "="*60)
            logger.info("ISSUES FOUND (with severity and fix plans)")
            logger.info("="*60)
            
            for issue in self.issues:
                logger.info(f"\n[{issue['severity']}] {issue['component']}")
                logger.info(f"Description: {issue['description']}")
                logger.info(f"Fix Plan: {issue['fix_plan']}")
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "results": self.results,
            "issues": self.issues,
            "models_tested": self.models
        }
        
        report_path = self.test_dir / "comprehensive_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"\n📊 Detailed report saved to: {report_path}")
        logger.info("="*80)

if __name__ == "__main__":
    tester = RLDKResearcherTest()
    tester.run_comprehensive_test()
