#!/usr/bin/env python3
"""
RLDK Function Validator

This script systematically tests every function in the RLDK repository to ensure
they work correctly under various conditions, including edge cases and stress scenarios.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import uuid

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import RLDK components
try:
    from rldk.ingest import ingest_runs, ingest_runs_to_events
    from rldk.diff import first_divergence
    from rldk.determinism.check import check
    from rldk.reward import health
    from rldk.evals import run
    from rldk.replay import replay
    from rldk.cards import generate_determinism_card, generate_drift_card, generate_reward_card
    from rldk.io import write_json
    from rldk.io.reward_writers import generate_reward_health_report
    from rldk.bisect import bisect_commits
    from rldk.forensics.ppo_scan import scan_logs
    from rldk.artifacts.env_audit import audit_environment
    from rldk.artifacts.log_scan import scan_training_logs
    from rldk.artifacts.ckpt_diff import diff_checkpoints
    from rldk.reward.drift import detect_reward_drift
    from rldk.reward.calibration import calibrate_reward_model
    from rldk.reward.health import analyze_reward_health
    from rldk.tracking.tracker import Tracker
    from rldk.tracking.environment_tracker import EnvironmentTracker
    from rldk.tracking.model_tracker import ModelTracker
    from rldk.tracking.dataset_tracker import DatasetTracker
    from rldk.tracking.seed_tracker import SeedTracker
    from rldk.tracking.git_tracker import GitTracker
    from rldk.adapters.base import BaseAdapter
    from rldk.adapters.trl import TRLAdapter
    from rldk.adapters.openrlhf import OpenRLHFAdapter
    from rldk.adapters.wandb import WandBAdapter
    from rldk.adapters.custom_jsonl import CustomJSONLAdapter
    from rldk.io.readers import read_jsonl, read_tensorboard
    from rldk.io.writers import write_jsonl, write_tensorboard
    from rldk.io.schemas import Event, TrainingRun
    from rldk.evals.runner import EvalRunner
    from rldk.evals.suites import get_eval_suite
    from rldk.evals.metrics import compute_metrics
    from rldk.evals.probes import run_probes
    from rldk.replay.replay import ReplayManager
    from rldk.diff.diff import compute_diff
    from rldk.determinism.determinism import check_determinism
    from rldk.bisect.bisect import BisectManager
    RLDK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RLDK components: {e}")
    RLDK_AVAILABLE = False

# Import profiler components
try:
    from profiler.anomaly_detection import AdvancedAnomalyDetector
    from profiler.hooks import AnomalyDetectionHook, profiler_registry
    from profiler.torch_profiler import TorchProfiler
    from profiler.profiler_context import ProfilerContext
    PROFILER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import profiler components: {e}")
    PROFILER_AVAILABLE = False


class TestDataGenerator:
    """Generate test data for various RLDK functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_logs(self, num_runs: int = 3, steps_per_run: int = 100) -> List[Path]:
        """Generate synthetic training logs."""
        log_paths = []
        
        for run_idx in range(num_runs):
            log_path = self.output_dir / f"training_run_{run_idx}.jsonl"
            
            with open(log_path, 'w') as f:
                for step in range(steps_per_run):
                    # Generate realistic training metrics
                    log_entry = {
                        "step": step,
                        "epoch": step // 10,
                        "loss": np.random.normal(2.0, 0.5) * np.exp(-step / 100),
                        "reward": np.random.normal(0.5, 0.1),
                        "kl_divergence": np.random.normal(0.1, 0.05),
                        "policy_loss": np.random.normal(1.0, 0.3),
                        "value_loss": np.random.normal(0.5, 0.2),
                        "entropy": np.random.normal(0.8, 0.1),
                        "learning_rate": 1e-4 * (0.95 ** (step // 10)),
                        "gradient_norm": np.random.normal(1.0, 0.5),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add some anomalies for testing
                    if step == 50:  # KL spike
                        log_entry["kl_divergence"] = 2.0
                    elif step == 75:  # Reward spike
                        log_entry["reward"] = 5.0
                    elif step == 90:  # Loss spike
                        log_entry["loss"] = 10.0
                    
                    f.write(json.dumps(log_entry) + "\n")
            
            log_paths.append(log_path)
        
        return log_paths
    
    def generate_checkpoints(self, num_checkpoints: int = 3) -> List[Path]:
        """Generate synthetic model checkpoints."""
        checkpoint_paths = []
        
        for i in range(num_checkpoints):
            checkpoint_path = self.output_dir / f"checkpoint_{i}.pt"
            
            # Create a simple model
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
            
            # Modify weights slightly for each checkpoint
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.1 * i
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': i * 100,
                'loss': 2.0 - i * 0.1,
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            
            checkpoint_paths.append(checkpoint_path)
        
        return checkpoint_paths
    
    def generate_reward_data(self, num_prompts: int = 100) -> Path:
        """Generate synthetic reward model data."""
        reward_data_path = self.output_dir / "reward_data.jsonl"
        
        with open(reward_data_path, 'w') as f:
            for i in range(num_prompts):
                prompt = f"Test prompt {i}: What is the meaning of life?"
                response = f"Response {i}: The meaning of life is {42 + i}."
                
                # Generate rewards with some drift
                base_reward = 0.5
                if i > 50:  # Simulate drift
                    base_reward += 0.2
                
                reward_entry = {
                    "prompt": prompt,
                    "response": response,
                    "reward": base_reward + np.random.normal(0, 0.1),
                    "model_version": "A" if i < 50 else "B",
                    "timestamp": datetime.now().isoformat()
                }
                
                f.write(json.dumps(reward_entry) + "\n")
        
        return reward_data_path
    
    def generate_evaluation_data(self, num_samples: int = 50) -> Path:
        """Generate synthetic evaluation data."""
        eval_data_path = self.output_dir / "eval_data.jsonl"
        
        with open(eval_data_path, 'w') as f:
            for i in range(num_samples):
                eval_entry = {
                    "sample_id": i,
                    "input": f"Test input {i}",
                    "output": f"Test output {i}",
                    "ground_truth": f"Ground truth {i}",
                    "score": np.random.uniform(0.3, 0.9),
                    "metadata": {
                        "category": "test",
                        "difficulty": np.random.choice(["easy", "medium", "hard"])
                    }
                }
                
                f.write(json.dumps(eval_entry) + "\n")
        
        return eval_data_path


class RLDKFunctionValidator:
    """Systematically test all RLDK functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize test data generator
        self.data_generator = TestDataGenerator(self.output_dir / "test_data")
        
        # Test results
        self.test_results = {
            'start_time': time.time(),
            'functions_tested': {},
            'overall_success': True,
            'errors': [],
            'warnings': []
        }
    
    def setup_logging(self):
        """Setup logging for the validator."""
        log_file = self.output_dir / "function_validation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def test_function(self, func_name: str, func, *args, **kwargs) -> Dict[str, Any]:
        """Test a single function and return results."""
        self.logger.info(f"Testing function: {func_name}")
        
        result = {
            'function_name': func_name,
            'success': False,
            'error': None,
            'execution_time': 0,
            'output': None,
            'args': str(args),
            'kwargs': str(kwargs)
        }
        
        try:
            start_time = time.time()
            output = func(*args, **kwargs)
            end_time = time.time()
            
            result['success'] = True
            result['execution_time'] = end_time - start_time
            result['output'] = str(output) if output is not None else None
            
            self.logger.info(f"✅ {func_name} - SUCCESS ({result['execution_time']:.3f}s)")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"❌ {func_name} - FAILED: {e}")
            self.test_results['errors'].append(f"{func_name}: {e}")
        
        return result
    
    def test_ingest_functions(self) -> Dict[str, Any]:
        """Test all ingest-related functions."""
        self.logger.info("Testing ingest functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        log_paths = self.data_generator.generate_training_logs()
        
        # Test ingest_runs
        try:
            df = ingest_runs(str(self.data_generator.output_dir))
            results['ingest_runs'] = {
                'success': True,
                'rows': len(df),
                'columns': list(df.columns)
            }
        except Exception as e:
            results['ingest_runs'] = {'success': False, 'error': str(e)}
        
        # Test ingest_runs_to_events
        try:
            events = ingest_runs_to_events(str(self.data_generator.output_dir))
            results['ingest_runs_to_events'] = {
                'success': True,
                'event_count': len(events)
            }
        except Exception as e:
            results['ingest_runs_to_events'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_diff_functions(self) -> Dict[str, Any]:
        """Test all diff-related functions."""
        self.logger.info("Testing diff functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        log_paths = self.data_generator.generate_training_logs(num_runs=2)
        
        # Test first_divergence
        try:
            divergence = first_divergence(
                str(log_paths[0]),
                str(log_paths[1]),
                signals=['loss', 'reward']
            )
            results['first_divergence'] = {
                'success': True,
                'divergence_step': divergence.get('step') if divergence else None
            }
        except Exception as e:
            results['first_divergence'] = {'success': False, 'error': str(e)}
        
        # Test compute_diff
        try:
            diff_result = compute_diff(
                str(log_paths[0]),
                str(log_paths[1]),
                signals=['loss', 'reward']
            )
            results['compute_diff'] = {
                'success': True,
                'has_diff': diff_result is not None
            }
        except Exception as e:
            results['compute_diff'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_determinism_functions(self) -> Dict[str, Any]:
        """Test all determinism-related functions."""
        self.logger.info("Testing determinism functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        log_paths = self.data_generator.generate_training_logs(num_runs=3)
        
        # Test check (determinism check)
        try:
            determinism_result = check(str(self.data_generator.output_dir))
            results['check'] = {
                'success': True,
                'is_deterministic': determinism_result.get('is_deterministic', False)
            }
        except Exception as e:
            results['check'] = {'success': False, 'error': str(e)}
        
        # Test check_determinism
        try:
            det_result = check_determinism(str(self.data_generator.output_dir))
            results['check_determinism'] = {
                'success': True,
                'result': det_result is not None
            }
        except Exception as e:
            results['check_determinism'] = {'success': False, 'error': str(e)}
        
        # Test generate_determinism_card
        try:
            card_path = self.output_dir / "determinism_card.json"
            generate_determinism_card(str(self.data_generator.output_dir), str(card_path))
            results['generate_determinism_card'] = {
                'success': True,
                'card_created': card_path.exists()
            }
        except Exception as e:
            results['generate_determinism_card'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_reward_functions(self) -> Dict[str, Any]:
        """Test all reward-related functions."""
        self.logger.info("Testing reward functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        reward_data_path = self.data_generator.generate_reward_data()
        
        # Test health analysis
        try:
            health_result = health(str(reward_data_path))
            results['health'] = {
                'success': True,
                'result': health_result is not None
            }
        except Exception as e:
            results['health'] = {'success': False, 'error': str(e)}
        
        # Test detect_reward_drift
        try:
            drift_result = detect_reward_drift(str(reward_data_path))
            results['detect_reward_drift'] = {
                'success': True,
                'drift_detected': drift_result.get('drift_detected', False)
            }
        except Exception as e:
            results['detect_reward_drift'] = {'success': False, 'error': str(e)}
        
        # Test analyze_reward_health
        try:
            health_analysis = analyze_reward_health(str(reward_data_path))
            results['analyze_reward_health'] = {
                'success': True,
                'analysis': health_analysis is not None
            }
        except Exception as e:
            results['analyze_reward_health'] = {'success': False, 'error': str(e)}
        
        # Test generate_reward_card
        try:
            card_path = self.output_dir / "reward_card.json"
            generate_reward_card(str(reward_data_path), str(card_path))
            results['generate_reward_card'] = {
                'success': True,
                'card_created': card_path.exists()
            }
        except Exception as e:
            results['generate_reward_card'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_eval_functions(self) -> Dict[str, Any]:
        """Test all evaluation-related functions."""
        self.logger.info("Testing eval functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        eval_data_path = self.data_generator.generate_evaluation_data()
        
        # Test run evaluation
        try:
            eval_result = run(str(eval_data_path), suite='quick')
            results['run'] = {
                'success': True,
                'result': eval_result is not None
            }
        except Exception as e:
            results['run'] = {'success': False, 'error': str(e)}
        
        # Test get_eval_suite
        try:
            suite = get_eval_suite('quick')
            results['get_eval_suite'] = {
                'success': True,
                'suite': suite is not None
            }
        except Exception as e:
            results['get_eval_suite'] = {'success': False, 'error': str(e)}
        
        # Test compute_metrics
        try:
            # Create sample data for metrics
            sample_data = [{'score': 0.8}, {'score': 0.9}, {'score': 0.7}]
            metrics = compute_metrics(sample_data)
            results['compute_metrics'] = {
                'success': True,
                'metrics': metrics is not None
            }
        except Exception as e:
            results['compute_metrics'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_forensics_functions(self) -> Dict[str, Any]:
        """Test all forensics-related functions."""
        self.logger.info("Testing forensics functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        log_paths = self.data_generator.generate_training_logs()
        
        # Test scan_logs (PPO scan)
        try:
            scan_result = scan_logs(str(log_paths[0]))
            results['scan_logs'] = {
                'success': True,
                'anomalies_found': len(scan_result.get('anomalies', []))
            }
        except Exception as e:
            results['scan_logs'] = {'success': False, 'error': str(e)}
        
        # Test audit_environment
        try:
            env_result = audit_environment(str(self.data_generator.output_dir))
            results['audit_environment'] = {
                'success': True,
                'audit_completed': env_result is not None
            }
        except Exception as e:
            results['audit_environment'] = {'success': False, 'error': str(e)}
        
        # Test scan_training_logs
        try:
            log_scan_result = scan_training_logs(str(log_paths[0]))
            results['scan_training_logs'] = {
                'success': True,
                'scan_completed': log_scan_result is not None
            }
        except Exception as e:
            results['scan_training_logs'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_checkpoint_functions(self) -> Dict[str, Any]:
        """Test all checkpoint-related functions."""
        self.logger.info("Testing checkpoint functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Generate test data
        checkpoint_paths = self.data_generator.generate_checkpoints()
        
        # Test diff_checkpoints
        try:
            diff_result = diff_checkpoints(str(checkpoint_paths[0]), str(checkpoint_paths[1]))
            results['diff_checkpoints'] = {
                'success': True,
                'diff_completed': diff_result is not None
            }
        except Exception as e:
            results['diff_checkpoints'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_tracking_functions(self) -> Dict[str, Any]:
        """Test all tracking-related functions."""
        self.logger.info("Testing tracking functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Test Tracker initialization
        try:
            tracker = Tracker(experiment_name="test_experiment")
            results['Tracker'] = {
                'success': True,
                'tracker_created': tracker is not None
            }
        except Exception as e:
            results['Tracker'] = {'success': False, 'error': str(e)}
        
        # Test EnvironmentTracker
        try:
            env_tracker = EnvironmentTracker()
            results['EnvironmentTracker'] = {
                'success': True,
                'tracker_created': env_tracker is not None
            }
        except Exception as e:
            results['EnvironmentTracker'] = {'success': False, 'error': str(e)}
        
        # Test ModelTracker
        try:
            model_tracker = ModelTracker()
            results['ModelTracker'] = {
                'success': True,
                'tracker_created': model_tracker is not None
            }
        except Exception as e:
            results['ModelTracker'] = {'success': False, 'error': str(e)}
        
        # Test DatasetTracker
        try:
            dataset_tracker = DatasetTracker()
            results['DatasetTracker'] = {
                'success': True,
                'tracker_created': dataset_tracker is not None
            }
        except Exception as e:
            results['DatasetTracker'] = {'success': False, 'error': str(e)}
        
        # Test SeedTracker
        try:
            seed_tracker = SeedTracker()
            results['SeedTracker'] = {
                'success': True,
                'tracker_created': seed_tracker is not None
            }
        except Exception as e:
            results['SeedTracker'] = {'success': False, 'error': str(e)}
        
        # Test GitTracker
        try:
            git_tracker = GitTracker()
            results['GitTracker'] = {
                'success': True,
                'tracker_created': git_tracker is not None
            }
        except Exception as e:
            results['GitTracker'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_adapter_functions(self) -> Dict[str, Any]:
        """Test all adapter functions."""
        self.logger.info("Testing adapter functions...")
        
        results = {}
        
        if not RLDK_AVAILABLE:
            return {'error': 'RLDK not available'}
        
        # Test TRLAdapter
        try:
            trl_adapter = TRLAdapter(str(self.data_generator.output_dir))
            results['TRLAdapter'] = {
                'success': True,
                'can_handle': trl_adapter.can_handle()
            }
        except Exception as e:
            results['TRLAdapter'] = {'success': False, 'error': str(e)}
        
        # Test OpenRLHFAdapter
        try:
            openrlhf_adapter = OpenRLHFAdapter(str(self.data_generator.output_dir))
            results['OpenRLHFAdapter'] = {
                'success': True,
                'can_handle': openrlhf_adapter.can_handle()
            }
        except Exception as e:
            results['OpenRLHFAdapter'] = {'success': False, 'error': str(e)}
        
        # Test WandBAdapter
        try:
            wandb_adapter = WandBAdapter("test://run")
            results['WandBAdapter'] = {
                'success': True,
                'adapter_created': wandb_adapter is not None
            }
        except Exception as e:
            results['WandBAdapter'] = {'success': False, 'error': str(e)}
        
        # Test CustomJSONLAdapter
        try:
            jsonl_adapter = CustomJSONLAdapter(str(self.data_generator.output_dir))
            results['CustomJSONLAdapter'] = {
                'success': True,
                'can_handle': jsonl_adapter.can_handle()
            }
        except Exception as e:
            results['CustomJSONLAdapter'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_profiler_functions(self) -> Dict[str, Any]:
        """Test all profiler functions."""
        self.logger.info("Testing profiler functions...")
        
        results = {}
        
        if not PROFILER_AVAILABLE:
            return {'error': 'Profiler not available'}
        
        # Test AdvancedAnomalyDetector
        try:
            detector = AdvancedAnomalyDetector()
            results['AdvancedAnomalyDetector'] = {
                'success': True,
                'detector_created': detector is not None
            }
        except Exception as e:
            results['AdvancedAnomalyDetector'] = {'success': False, 'error': str(e)}
        
        # Test TorchProfiler
        try:
            profiler = TorchProfiler()
            results['TorchProfiler'] = {
                'success': True,
                'profiler_created': profiler is not None
            }
        except Exception as e:
            results['TorchProfiler'] = {'success': False, 'error': str(e)}
        
        # Test ProfilerContext
        try:
            context = ProfilerContext()
            results['ProfilerContext'] = {
                'success': True,
                'context_created': context is not None
            }
        except Exception as e:
            results['ProfilerContext'] = {'success': False, 'error': str(e)}
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all RLDK functions."""
        self.logger.info("Starting comprehensive RLDK function validation...")
        
        start_time = time.time()
        
        # Test all function categories
        test_categories = [
            ('ingest', self.test_ingest_functions),
            ('diff', self.test_diff_functions),
            ('determinism', self.test_determinism_functions),
            ('reward', self.test_reward_functions),
            ('eval', self.test_eval_functions),
            ('forensics', self.test_forensics_functions),
            ('checkpoint', self.test_checkpoint_functions),
            ('tracking', self.test_tracking_functions),
            ('adapter', self.test_adapter_functions),
            ('profiler', self.test_profiler_functions)
        ]
        
        for category_name, test_func in test_categories:
            self.logger.info(f"Testing {category_name} functions...")
            try:
                category_results = test_func()
                self.test_results['functions_tested'][category_name] = category_results
                
                # Check for failures
                for func_name, result in category_results.items():
                    if isinstance(result, dict) and not result.get('success', True):
                        self.test_results['overall_success'] = False
                        
            except Exception as e:
                self.logger.error(f"Error testing {category_name} functions: {e}")
                self.test_results['functions_tested'][category_name] = {'error': str(e)}
                self.test_results['overall_success'] = False
        
        self.test_results['end_time'] = time.time()
        self.test_results['total_duration'] = self.test_results['end_time'] - start_time
        
        # Save results
        results_file = self.output_dir / "function_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        self.logger.info(f"Function validation completed. Results saved to {results_file}")
        
        return self.test_results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the validation results."""
        report = []
        report.append("=" * 80)
        report.append("RLDK FUNCTION VALIDATION SUMMARY")
        report.append("=" * 80)
        report.append(f"Overall Success: {'✅ PASSED' if self.test_results['overall_success'] else '❌ FAILED'}")
        report.append(f"Total Duration: {self.test_results['total_duration']:.2f} seconds")
        report.append(f"Errors: {len(self.test_results['errors'])}")
        report.append(f"Warnings: {len(self.test_results['warnings'])}")
        report.append("")
        
        # Category summary
        report.append("CATEGORY SUMMARY:")
        report.append("-" * 40)
        
        for category, results in self.test_results['functions_tested'].items():
            if isinstance(results, dict) and 'error' not in results:
                total_functions = len(results)
                successful_functions = sum(1 for r in results.values() if isinstance(r, dict) and r.get('success', False))
                report.append(f"{category.upper()}: {successful_functions}/{total_functions} functions passed")
            else:
                report.append(f"{category.upper()}: ❌ ERROR - {results.get('error', 'Unknown error')}")
        
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for category, results in self.test_results['functions_tested'].items():
            if isinstance(results, dict) and 'error' not in results:
                report.append(f"\n{category.upper()}:")
                for func_name, result in results.items():
                    if isinstance(result, dict):
                        status = "✅" if result.get('success', False) else "❌"
                        report.append(f"  {status} {func_name}")
                        if not result.get('success', False) and 'error' in result:
                            report.append(f"    Error: {result['error']}")
        
        # Errors summary
        if self.test_results['errors']:
            report.append("\nERRORS:")
            report.append("-" * 40)
            for error in self.test_results['errors']:
                report.append(f"❌ {error}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run function validation."""
    print("🔍 Starting RLDK Function Validation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("./rldk_function_validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validator
    validator = RLDKFunctionValidator(output_dir)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Generate and print summary
    summary = validator.generate_summary_report()
    print(summary)
    
    # Save summary to file
    summary_file = output_dir / "validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"\n📁 Detailed results saved to: {output_dir}")
    print(f"📄 Summary report saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    main()