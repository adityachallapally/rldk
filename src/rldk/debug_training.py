"""
One-Click Training Debug - Comprehensive debugging for any training run
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess
import shutil
import tempfile

from .universal_monitor import UniversalMonitor
from .anomaly_detector import AnomalyDetector
from .forensics.ppo_scan import PPOScanner
from .reward.drift import RewardDriftDetector
from .artifacts.ckpt_diff import CheckpointDiff
from .artifacts.env_audit import EnvironmentAuditor
from .io.schemas import TrainingMetrics, AnomalyReport, HealthScore, DebugReport

console = Console()

@dataclass
class DebugConfig:
    """Configuration for training debugging"""
    auto_fix: bool = False
    generate_report: bool = True
    create_test_case: bool = True
    comprehensive: bool = True
    save_artifacts: bool = True
    report_dir: str = "rldk_reports"
    test_case_dir: str = "debug_test_cases"

class TrainingDebugger:
    """
    One-Click Training Debug - Comprehensive debugging for any training run
    
    Features:
    - Auto-detects training framework
    - Comprehensive analysis of training logs
    - Automatic issue detection and classification
    - Intelligent fix suggestions
    - Reproducible test case generation
    - Detailed reports with visualizations
    """
    
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        self.console = Console()
        self.monitor = UniversalMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.ppo_scanner = PPOScanner()
        self.drift_detector = RewardDriftDetector()
        self.ckpt_diff = CheckpointDiff()
        self.env_auditor = EnvironmentAuditor()
        
        # Create directories
        os.makedirs(self.config.report_dir, exist_ok=True)
        os.makedirs(self.config.test_case_dir, exist_ok=True)
        
    def debug_training(self, run_path: Union[str, Path], 
                      framework: Optional[str] = None) -> DebugReport:
        """
        Comprehensive debugging of training run
        
        Args:
            run_path: Path to training run (logs, checkpoints, etc.)
            framework: Framework name (auto-detected if None)
            
        Returns:
            DebugReport with comprehensive analysis and recommendations
        """
        run_path = Path(run_path)
        
        if not run_path.exists():
            raise FileNotFoundError(f"Run path does not exist: {run_path}")
        
        self.console.print(f"[bold blue]🔍 Starting comprehensive debug of: {run_path}[/bold blue]")
        
        # Auto-detect framework if not provided
        if framework is None:
            framework = self.monitor.auto_detect_framework(run_path)
            self.console.print(f"[green]Auto-detected framework: {framework}[/green]")
        
        # Initialize report
        report = DebugReport(
            run_path=str(run_path),
            framework=framework,
            timestamp=datetime.now().isoformat(),
            issues=[],
            fixes=[],
            recommendations=[],
            health_score=None,
            details={}
        )
        
        # Run comprehensive analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Step 1: Analyze training logs
            task1 = progress.add_task("Analyzing training logs...", total=None)
            log_analysis = self._analyze_training_logs(run_path, framework)
            progress.update(task1, completed=True)
            
            # Step 2: Detect anomalies
            task2 = progress.add_task("Detecting anomalies...", total=None)
            anomaly_report = self.anomaly_detector.detect_training_anomalies(run_path)
            progress.update(task2, completed=True)
            
            # Step 3: PPO forensics (if applicable)
            task3 = progress.add_task("Running PPO forensics...", total=None)
            ppo_report = self._run_ppo_forensics(run_path, framework)
            progress.update(task3, completed=True)
            
            # Step 4: Checkpoint analysis
            task4 = progress.add_task("Analyzing checkpoints...", total=None)
            ckpt_analysis = self._analyze_checkpoints(run_path)
            progress.update(task4, completed=True)
            
            # Step 5: Environment audit
            task5 = progress.add_task("Auditing environment...", total=None)
            env_audit = self._audit_environment(run_path)
            progress.update(task5, completed=True)
            
            # Step 6: Generate health score
            task6 = progress.add_task("Calculating health score...", total=None)
            health_score = self._calculate_health_score(log_analysis, anomaly_report, ppo_report)
            progress.update(task6, completed=True)
            
            # Step 7: Generate fixes and recommendations
            task7 = progress.add_task("Generating recommendations...", total=None)
            fixes, recommendations = self._generate_fixes_and_recommendations(
                log_analysis, anomaly_report, ppo_report, ckpt_analysis, env_audit
            )
            progress.update(task7, completed=True)
            
            # Step 8: Create test case (if requested)
            if self.config.create_test_case:
                task8 = progress.add_task("Creating reproducible test case...", total=None)
                test_case = self._create_test_case(run_path, framework)
                progress.update(task8, completed=True)
            
            # Step 9: Generate report
            task9 = progress.add_task("Generating comprehensive report...", total=None)
            final_report = self._generate_final_report(
                report, log_analysis, anomaly_report, ppo_report, 
                ckpt_analysis, env_audit, health_score, fixes, recommendations
            )
            progress.update(task9, completed=True)
        
        # Save report
        if self.config.generate_report:
            self._save_report(final_report)
        
        # Apply fixes if requested
        if self.config.auto_fix:
            self._apply_fixes(fixes, run_path)
        
        self.console.print(f"[bold green]✅ Debug complete! Report saved to {self.config.report_dir}[/bold green]")
        
        return final_report
    
    def _analyze_training_logs(self, run_path: Path, framework: str) -> Dict[str, Any]:
        """Analyze training logs for issues"""
        analysis = {
            'framework': framework,
            'log_files': [],
            'metrics': {},
            'issues': [],
            'warnings': []
        }
        
        # Find log files
        log_files = list(run_path.rglob("*.jsonl")) + list(run_path.rglob("*.json"))
        
        if not log_files:
            analysis['issues'].append("No log files found")
            return analysis
        
        analysis['log_files'] = [str(f) for f in log_files]
        
        # Analyze each log file
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    analysis['warnings'].append(f"Empty log file: {log_file}")
                    continue
                
                # Parse and analyze logs
                log_data = []
                for line in lines:
                    try:
                        data = json.loads(line.strip())
                        log_data.append(data)
                    except json.JSONDecodeError:
                        continue
                
                if not log_data:
                    analysis['warnings'].append(f"No valid JSON in log file: {log_file}")
                    continue
                
                # Extract metrics
                metrics = self._extract_metrics(log_data)
                analysis['metrics'][str(log_file)] = metrics
                
                # Check for common issues
                issues = self._check_log_issues(log_data, framework)
                analysis['issues'].extend(issues)
                
            except Exception as e:
                analysis['warnings'].append(f"Error analyzing {log_file}: {e}")
        
        return analysis
    
    def _extract_metrics(self, log_data: List[Dict]) -> Dict[str, Any]:
        """Extract key metrics from log data"""
        metrics = {}
        
        if not log_data:
            return metrics
        
        # Extract numeric metrics
        numeric_fields = ['step', 'loss', 'reward', 'kl_divergence', 'entropy', 
                         'value_loss', 'policy_loss', 'gradient_norm', 'learning_rate']
        
        for field in numeric_fields:
            values = [log.get(field) for log in log_data if isinstance(log.get(field), (int, float))]
            if values:
                metrics[field] = {
                    'values': values,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self._calculate_trend(values)
                }
        
        # Calculate derived metrics
        if 'loss' in metrics and 'reward' in metrics:
            metrics['loss_reward_ratio'] = {
                'mean': metrics['loss']['mean'] / max(metrics['reward']['mean'], 1e-8),
                'trend': self._calculate_trend([l/r for l, r in zip(metrics['loss']['values'], metrics['reward']['values'])])
            }
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend of values"""
        if len(values) < 10:
            return "insufficient_data"
        
        # Use linear regression to determine trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _check_log_issues(self, log_data: List[Dict], framework: str) -> List[str]:
        """Check for common issues in log data"""
        issues = []
        
        if not log_data:
            issues.append("No log data found")
            return issues
        
        # Check for missing steps
        steps = [log.get('step') for log in log_data if isinstance(log.get('step'), int)]
        if len(steps) > 1:
            expected_steps = set(range(min(steps), max(steps) + 1))
            missing_steps = expected_steps - set(steps)
            if missing_steps:
                issues.append(f"Missing training steps: {len(missing_steps)} steps missing")
        
        # Check for loss spikes
        losses = [log.get('loss') for log in log_data if isinstance(log.get('loss'), (int, float))]
        if len(losses) > 10:
            loss_std = np.std(losses)
            loss_mean = np.mean(losses)
            spikes = [i for i, loss in enumerate(losses) if abs(loss - loss_mean) > 3 * loss_std]
            if spikes:
                issues.append(f"Loss spikes detected: {len(spikes)} spikes")
        
        # Check for reward drops
        rewards = [log.get('reward') for log in log_data if isinstance(log.get('reward'), (int, float))]
        if len(rewards) > 10:
            drops = [i for i in range(1, len(rewards)) if rewards[i] < rewards[i-1] * 0.5]
            if drops:
                issues.append(f"Reward drops detected: {len(drops)} drops")
        
        # Framework-specific checks
        if framework == 'trl':
            # Check for TRL-specific issues
            if any('kl_divergence' in log for log in log_data):
                kl_values = [log.get('kl_divergence', 0) for log in log_data]
                if any(kl > 0.1 for kl in kl_values):
                    issues.append("High KL divergence detected")
        
        elif framework == 'openrlhf':
            # Check for OpenRLHF-specific issues
            if any('value_loss' in log for log in log_data):
                value_losses = [log.get('value_loss', 0) for log in log_data]
                if any(vl < 0.01 for vl in value_losses):
                    issues.append("Value function collapse detected")
        
        return issues
    
    def _run_ppo_forensics(self, run_path: Path, framework: str) -> Dict[str, Any]:
        """Run PPO forensics analysis"""
        if framework not in ['trl', 'ppo', 'openrlhf']:
            return {'applicable': False, 'reason': f'PPO forensics not applicable for {framework}'}
        
        try:
            ppo_report = self.ppo_scanner.scan_logs(run_path)
            return {
                'applicable': True,
                'anomalies': ppo_report.anomalies if hasattr(ppo_report, 'anomalies') else [],
                'details': ppo_report.details if hasattr(ppo_report, 'details') else {},
                'recommendations': ppo_report.recommendations if hasattr(ppo_report, 'recommendations') else []
            }
        except Exception as e:
            return {
                'applicable': True,
                'error': str(e),
                'anomalies': [],
                'details': {},
                'recommendations': []
            }
    
    def _analyze_checkpoints(self, run_path: Path) -> Dict[str, Any]:
        """Analyze training checkpoints"""
        analysis = {
            'checkpoints': [],
            'issues': [],
            'recommendations': []
        }
        
        # Find checkpoint files
        ckpt_files = list(run_path.rglob("*.pt")) + list(run_path.rglob("*.pth")) + list(run_path.rglob("*.safetensors"))
        
        if not ckpt_files:
            analysis['issues'].append("No checkpoint files found")
            return analysis
        
        analysis['checkpoints'] = [str(f) for f in ckpt_files]
        
        # Compare consecutive checkpoints
        if len(ckpt_files) >= 2:
            try:
                # Compare last two checkpoints
                ckpt1, ckpt2 = ckpt_files[-2], ckpt_files[-1]
                diff_result = self.ckpt_diff.diff_checkpoints(ckpt1, ckpt2)
                
                analysis['checkpoint_diff'] = {
                    'top_movers': diff_result.get('top_movers', []),
                    'gradient_ratio': diff_result.get('gradient_ratio', 1.0),
                    'parameter_changes': diff_result.get('parameter_changes', {})
                }
                
                # Check for issues
                if diff_result.get('gradient_ratio', 1.0) < 0.1:
                    analysis['issues'].append("Value function collapse detected")
                
                if len(diff_result.get('top_movers', [])) > 10:
                    analysis['issues'].append("Excessive parameter changes detected")
                
            except Exception as e:
                analysis['issues'].append(f"Error comparing checkpoints: {e}")
        
        return analysis
    
    def _audit_environment(self, run_path: Path) -> Dict[str, Any]:
        """Audit training environment for determinism issues"""
        try:
            audit_result = self.env_auditor.audit_environment(run_path)
            return {
                'deterministic': audit_result.get('deterministic', False),
                'issues': audit_result.get('issues', []),
                'fixes': audit_result.get('fixes', []),
                'environment_info': audit_result.get('environment_info', {})
            }
        except Exception as e:
            return {
                'deterministic': False,
                'issues': [f"Error during environment audit: {e}"],
                'fixes': [],
                'environment_info': {}
            }
    
    def _calculate_health_score(self, log_analysis: Dict, anomaly_report: AnomalyReport, 
                               ppo_report: Dict) -> HealthScore:
        """Calculate comprehensive health score"""
        score = 100.0
        issues = []
        
        # Deduct points for issues
        if log_analysis.get('issues'):
            score -= len(log_analysis['issues']) * 10
        
        if anomaly_report.anomalies_detected:
            score -= anomaly_report.anomaly_count * 2
        
        if ppo_report.get('anomalies'):
            score -= len(ppo_report['anomalies']) * 5
        
        # Ensure score is in valid range
        score = max(0.0, min(100.0, score))
        
        # Determine health level
        if score >= 85:
            health_level = "excellent"
        elif score >= 70:
            health_level = "good"
        elif score >= 50:
            health_level = "fair"
        else:
            health_level = "poor"
        
        return HealthScore(
            overall_score=score,
            stability_score=score * 0.3,
            convergence_score=score * 0.3,
            efficiency_score=score * 0.2,
            robustness_score=score * 0.2,
            issues=issues,
            recommendations=[]
        )
    
    def _generate_fixes_and_recommendations(self, log_analysis: Dict, anomaly_report: AnomalyReport,
                                           ppo_report: Dict, ckpt_analysis: Dict, env_audit: Dict) -> Tuple[List[str], List[str]]:
        """Generate fixes and recommendations based on analysis"""
        fixes = []
        recommendations = []
        
        # Log analysis fixes
        if log_analysis.get('issues'):
            for issue in log_analysis['issues']:
                if "Missing training steps" in issue:
                    fixes.append("Check training script for interruptions or errors")
                elif "Loss spikes" in issue:
                    fixes.append("Reduce learning rate or add gradient clipping")
                elif "Reward drops" in issue:
                    fixes.append("Check reward model stability and training data quality")
        
        # Anomaly detection fixes
        if anomaly_report.anomalies_detected:
            fixes.append("Review training parameters and data preprocessing")
            recommendations.append("Monitor training more closely for similar issues")
        
        # PPO forensics fixes
        if ppo_report.get('anomalies'):
            for anomaly in ppo_report['anomalies']:
                if 'KL spike' in str(anomaly):
                    fixes.append("Adjust KL penalty coefficient")
                elif 'gradient' in str(anomaly).lower():
                    fixes.append("Add gradient clipping or reduce learning rate")
        
        # Checkpoint analysis fixes
        if ckpt_analysis.get('issues'):
            for issue in ckpt_analysis['issues']:
                if "Value function collapse" in issue:
                    fixes.append("Check value function initialization and learning rate")
                elif "Excessive parameter changes" in issue:
                    fixes.append("Reduce learning rate or add regularization")
        
        # Environment audit fixes
        if not env_audit.get('deterministic', True):
            fixes.extend(env_audit.get('fixes', []))
        
        # General recommendations
        if len(fixes) > 0:
            recommendations.append("Consider running training with reduced learning rate")
            recommendations.append("Add more comprehensive logging for better debugging")
        
        return fixes, recommendations
    
    def _create_test_case(self, run_path: Path, framework: str) -> Dict[str, Any]:
        """Create reproducible test case"""
        test_case = {
            'framework': framework,
            'run_path': str(run_path),
            'created_at': datetime.now().isoformat(),
            'files': [],
            'config': {},
            'reproduction_steps': []
        }
        
        # Copy essential files
        test_case_dir = Path(self.config.test_case_dir) / f"test_case_{run_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_case_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy log files
        log_files = list(run_path.rglob("*.jsonl")) + list(run_path.rglob("*.json"))
        for log_file in log_files[:5]:  # Limit to first 5 log files
            dest_file = test_case_dir / log_file.name
            shutil.copy2(log_file, dest_file)
            test_case['files'].append(str(dest_file))
        
        # Copy config files
        config_files = list(run_path.rglob("*.yaml")) + list(run_path.rglob("*.yml")) + list(run_path.rglob("*.json"))
        for config_file in config_files[:3]:  # Limit to first 3 config files
            dest_file = test_case_dir / config_file.name
            shutil.copy2(config_file, dest_file)
            test_case['files'].append(str(dest_file))
        
        # Create reproduction script
        repro_script = self._create_reproduction_script(framework, test_case_dir)
        test_case['reproduction_script'] = str(repro_script)
        
        # Save test case metadata
        test_case_file = test_case_dir / "test_case.json"
        with open(test_case_file, 'w') as f:
            json.dump(test_case, f, indent=2)
        
        return test_case
    
    def _create_reproduction_script(self, framework: str, test_case_dir: Path) -> Path:
        """Create reproduction script for test case"""
        script_content = f"""#!/usr/bin/env python3
# Reproduction script for {framework} training
# Generated by RLDK Debug Training

import os
import json
import subprocess
from pathlib import Path

def reproduce_training():
    print("Reproducing training case...")
    
    # Load test case metadata
    with open("test_case.json", "r") as f:
        test_case = json.load(f)
    
    print(f"Framework: {{test_case['framework']}}")
    print(f"Original run path: {{test_case['run_path']}}")
    
    # Add framework-specific reproduction steps here
    if test_case['framework'] == 'trl':
        print("To reproduce TRL training:")
        print("1. Install TRL: pip install trl")
        print("2. Run training with config files in this directory")
    elif test_case['framework'] == 'openrlhf':
        print("To reproduce OpenRLHF training:")
        print("1. Install OpenRLHF: pip install openrlhf")
        print("2. Run training with config files in this directory")
    
    print("\\nTest case files:")
    for file in test_case['files']:
        print(f"  - {{file}}")

if __name__ == "__main__":
    reproduce_training()
"""
        
        script_path = test_case_dir / "reproduce.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _generate_final_report(self, report: DebugReport, log_analysis: Dict, 
                              anomaly_report: AnomalyReport, ppo_report: Dict,
                              ckpt_analysis: Dict, env_audit: Dict, health_score: HealthScore,
                              fixes: List[str], recommendations: List[str]) -> DebugReport:
        """Generate final comprehensive report"""
        
        # Update report with all findings
        report.health_score = health_score
        report.fixes = fixes
        report.recommendations = recommendations
        
        # Collect all issues
        all_issues = []
        all_issues.extend(log_analysis.get('issues', []))
        if anomaly_report.anomalies_detected:
            all_issues.append(f"Anomalies detected: {anomaly_report.anomaly_count}")
        all_issues.extend(ppo_report.get('anomalies', []))
        all_issues.extend(ckpt_analysis.get('issues', []))
        all_issues.extend(env_audit.get('issues', []))
        
        report.issues = all_issues
        
        # Add detailed analysis
        report.details = {
            'log_analysis': log_analysis,
            'anomaly_report': {
                'anomalies_detected': anomaly_report.anomalies_detected,
                'anomaly_count': anomaly_report.anomaly_count,
                'anomaly_ratio': anomaly_report.anomaly_ratio,
                'confidence': anomaly_report.confidence
            },
            'ppo_forensics': ppo_report,
            'checkpoint_analysis': ckpt_analysis,
            'environment_audit': env_audit
        }
        
        return report
    
    def _save_report(self, report: DebugReport):
        """Save debug report to file"""
        report_file = Path(self.config.report_dir) / f"debug_report_{Path(report.run_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert report to dict for JSON serialization
        report_dict = {
            'run_path': report.run_path,
            'framework': report.framework,
            'timestamp': report.timestamp,
            'health_score': {
                'overall_score': report.health_score.overall_score,
                'stability_score': report.health_score.stability_score,
                'convergence_score': report.health_score.convergence_score,
                'efficiency_score': report.health_score.efficiency_score,
                'robustness_score': report.health_score.robustness_score
            } if report.health_score else None,
            'issues': report.issues,
            'fixes': report.fixes,
            'recommendations': report.recommendations,
            'details': report.details
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.console.print(f"[green]Report saved to: {report_file}[/green]")
    
    def _apply_fixes(self, fixes: List[str], run_path: Path):
        """Apply automatic fixes (placeholder for future implementation)"""
        self.console.print("[yellow]Auto-fix feature is not yet implemented. Please apply fixes manually.[/yellow]")
        for fix in fixes:
            self.console.print(f"[blue]Fix: {fix}[/blue]")

# Convenience functions
def debug_training(run_path: Union[str, Path], **kwargs) -> DebugReport:
    """Quick training debugging with default settings"""
    debugger = TrainingDebugger()
    return debugger.debug_training(run_path, **kwargs)

def quick_debug(run_path: Union[str, Path]) -> Dict[str, Any]:
    """Quick debugging without comprehensive analysis"""
    debugger = TrainingDebugger(DebugConfig(comprehensive=False, create_test_case=False))
    return debugger.debug_training(run_path).__dict__