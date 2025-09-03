"""
Universal Training Monitor - Auto-detect and monitor ANY training framework
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import threading
import time
import queue

from .adapters.base import BaseAdapter
from .adapters.trl import TRLAdapter
from .adapters.openrlhf import OpenRLHFAdapter
from .forensics.ppo_scan import PPOScanner
from .reward.drift import RewardDriftDetector
from .io.schemas import TrainingMetrics, AnomalyReport, HealthScore

console = Console()

@dataclass
class MonitoringConfig:
    """Configuration for universal monitoring"""
    frameworks: List[str] = None
    auto_detect: bool = True
    real_time_alerts: bool = True
    alert_thresholds: Dict[str, float] = None
    check_interval: float = 5.0  # seconds
    max_history: int = 1000
    save_reports: bool = True
    report_dir: str = "rldk_reports"
    
    def __post_init__(self):
        if self.frameworks is None:
            self.frameworks = ['trl', 'openrlhf', 'custom', 'ppo', 'dpo']
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'kl_spike': 4.0,  # KL divergence spike threshold
                'reward_drop': 0.5,  # Reward drop threshold
                'gradient_explosion': 10.0,  # Gradient norm threshold
                'value_collapse': 0.1,  # Value function collapse threshold
                'training_instability': 0.3,  # Training instability threshold
            }

class UniversalMonitor:
    """
    Universal Training Monitor - Auto-detect and monitor ANY training framework
    
    Features:
    - Auto-detects training framework from logs
    - Real-time monitoring with alerts
    - Comprehensive anomaly detection
    - Cross-framework comparison
    - Health scoring and recommendations
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.console = Console()
        self.adapters = self._initialize_adapters()
        self.scanner = PPOScanner()
        self.drift_detector = RewardDriftDetector()
        self.monitoring_data = {}
        self.alerts = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Create report directory
        os.makedirs(self.config.report_dir, exist_ok=True)
        
    def _initialize_adapters(self) -> Dict[str, BaseAdapter]:
        """Initialize framework adapters"""
        return {
            'trl': TRLAdapter(),
            'openrlhf': OpenRLHFAdapter(),
            # Add more adapters as needed
        }
    
    def auto_detect_framework(self, log_path: Union[str, Path]) -> str:
        """
        Auto-detect training framework from log files
        
        Args:
            log_path: Path to training logs
            
        Returns:
            Detected framework name
        """
        log_path = Path(log_path)
        
        if not log_path.exists():
            raise FileNotFoundError(f"Log path does not exist: {log_path}")
        
        # Check for framework-specific files
        if (log_path / "trl_logs.jsonl").exists() or "trl" in str(log_path):
            return "trl"
        elif (log_path / "openrlhf_logs.jsonl").exists() or "openrlhf" in str(log_path):
            return "openrlhf"
        elif (log_path / "ppo_logs.jsonl").exists() or "ppo" in str(log_path):
            return "ppo"
        elif (log_path / "dpo_logs.jsonl").exists() or "dpo" in str(log_path):
            return "dpo"
        
        # Check file contents for framework signatures
        for file_path in log_path.rglob("*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        
                        # Check for framework-specific fields
                        if 'trl' in str(data).lower() or 'transformers' in str(data).lower():
                            return "trl"
                        elif 'openrlhf' in str(data).lower():
                            return "openrlhf"
                        elif 'ppo' in str(data).lower() or 'policy' in str(data).lower():
                            return "ppo"
                        elif 'dpo' in str(data).lower() or 'direct_preference' in str(data).lower():
                            return "dpo"
            except:
                continue
        
        return "custom"
    
    def start_monitoring(self, log_path: Union[str, Path], framework: Optional[str] = None):
        """
        Start real-time monitoring of training logs
        
        Args:
            log_path: Path to training logs
            framework: Framework name (auto-detected if None)
        """
        log_path = Path(log_path)
        
        if framework is None and self.config.auto_detect:
            framework = self.auto_detect_framework(log_path)
            self.console.print(f"[green]Auto-detected framework: {framework}[/green]")
        
        self.monitoring_data[log_path] = {
            'framework': framework,
            'log_path': log_path,
            'start_time': datetime.now(),
            'metrics': [],
            'anomalies': [],
            'health_score': None
        }
        
        if self.config.real_time_alerts:
            self._start_monitor_thread(log_path)
        
        self.console.print(f"[green]Started monitoring {framework} training at {log_path}[/green]")
    
    def _start_monitor_thread(self, log_path: Path):
        """Start monitoring thread for real-time alerts"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(log_path,),
                daemon=True
            )
            self.monitor_thread.start()
    
    def _monitor_loop(self, log_path: Path):
        """Main monitoring loop"""
        last_modified = 0
        
        while self.is_monitoring:
            try:
                # Check for new log files
                for file_path in log_path.rglob("*.jsonl"):
                    if file_path.stat().st_mtime > last_modified:
                        last_modified = file_path.stat().st_mtime
                        
                        # Process new logs
                        self._process_new_logs(file_path)
                        
                        # Check for anomalies
                        self._check_anomalies(log_path)
                        
                        # Update health score
                        self._update_health_score(log_path)
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.console.print(f"[red]Monitoring error: {e}[/red]")
                time.sleep(self.config.check_interval)
    
    def _process_new_logs(self, file_path: Path):
        """Process new log entries"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Parse new lines
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    self._analyze_log_entry(data)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            self.console.print(f"[red]Error processing logs: {e}[/red]")
    
    def _analyze_log_entry(self, data: Dict[str, Any]):
        """Analyze individual log entry for anomalies"""
        # Check for KL divergence spikes
        if 'kl_divergence' in data:
            kl_value = data['kl_divergence']
            if kl_value > self.config.alert_thresholds['kl_spike']:
                self._create_alert('kl_spike', f"KL divergence spike: {kl_value}")
        
        # Check for reward drops
        if 'reward' in data:
            reward_value = data['reward']
            if reward_value < self.config.alert_thresholds['reward_drop']:
                self._create_alert('reward_drop', f"Reward drop detected: {reward_value}")
        
        # Check for gradient explosion
        if 'gradient_norm' in data:
            grad_norm = data['gradient_norm']
            if grad_norm > self.config.alert_thresholds['gradient_explosion']:
                self._create_alert('gradient_explosion', f"Gradient explosion: {grad_norm}")
        
        # Check for value function collapse
        if 'value_loss' in data:
            value_loss = data['value_loss']
            if value_loss < self.config.alert_thresholds['value_collapse']:
                self._create_alert('value_collapse', f"Value function collapse: {value_loss}")
    
    def _create_alert(self, alert_type: str, message: str):
        """Create and queue an alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': 'high' if alert_type in ['kl_spike', 'gradient_explosion'] else 'medium'
        }
        
        self.alerts.put(alert)
        
        # Print alert
        color = 'red' if alert['severity'] == 'high' else 'yellow'
        self.console.print(f"[{color}][{alert_type.upper()}] {message}[/{color}]")
    
    def _check_anomalies(self, log_path: Path):
        """Run comprehensive anomaly detection"""
        try:
            # Run PPO forensics
            ppo_report = self.scanner.scan_logs(log_path)
            if ppo_report.anomalies:
                for anomaly in ppo_report.anomalies:
                    self._create_alert('ppo_anomaly', f"PPO anomaly: {anomaly}")
            
            # Check for reward model drift
            # This would require multiple reward models to compare
            # For now, we'll skip this check
            
        except Exception as e:
            self.console.print(f"[red]Error checking anomalies: {e}[/red]")
    
    def _update_health_score(self, log_path: Path):
        """Update training health score"""
        try:
            # Calculate health score based on various metrics
            health_score = self._calculate_health_score(log_path)
            self.monitoring_data[log_path]['health_score'] = health_score
            
        except Exception as e:
            self.console.print(f"[red]Error updating health score: {e}[/red]")
    
    def _calculate_health_score(self, log_path: Path) -> HealthScore:
        """Calculate comprehensive health score"""
        # This is a simplified version - in practice, this would be more sophisticated
        score = 100.0
        issues = []
        
        # Check for various issues and adjust score
        # Implementation would depend on the specific metrics available
        
        return HealthScore(
            overall_score=score,
            stability_score=score,
            convergence_score=score,
            efficiency_score=score,
            robustness_score=score,
            issues=issues,
            recommendations=[]
        )
    
    def stop_monitoring(self, log_path: Optional[Union[str, Path]] = None):
        """Stop monitoring"""
        if log_path is None:
            self.is_monitoring = False
        else:
            log_path = Path(log_path)
            if log_path in self.monitoring_data:
                del self.monitoring_data[log_path]
        
        self.console.print("[yellow]Stopped monitoring[/yellow]")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        status = {
            'monitoring': self.is_monitoring,
            'active_runs': len(self.monitoring_data),
            'alerts': self.alerts.qsize(),
            'runs': {}
        }
        
        for log_path, data in self.monitoring_data.items():
            status['runs'][str(log_path)] = {
                'framework': data['framework'],
                'start_time': data['start_time'].isoformat(),
                'health_score': data['health_score'].overall_score if data['health_score'] else None
            }
        
        return status
    
    def generate_report(self, log_path: Union[str, Path]) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        log_path = Path(log_path)
        
        if log_path not in self.monitoring_data:
            raise ValueError(f"Not monitoring {log_path}")
        
        data = self.monitoring_data[log_path]
        
        # Generate comprehensive report
        report = {
            'framework': data['framework'],
            'monitoring_duration': (datetime.now() - data['start_time']).total_seconds(),
            'health_score': data['health_score'],
            'anomalies': data['anomalies'],
            'recommendations': self._generate_recommendations(data),
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        if self.config.save_reports:
            report_file = self.config.report_dir / f"monitoring_report_{log_path.name}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on monitoring data"""
        recommendations = []
        
        if data['health_score']:
            score = data['health_score'].overall_score
            
            if score < 50:
                recommendations.append("Training appears to be failing. Consider stopping and debugging.")
            elif score < 70:
                recommendations.append("Training shows signs of instability. Monitor closely.")
            elif score < 85:
                recommendations.append("Training is progressing but could be optimized.")
            else:
                recommendations.append("Training is healthy and progressing well.")
        
        return recommendations
    
    def show_dashboard(self):
        """Show real-time monitoring dashboard"""
        with Live(self._create_dashboard(), refresh_per_second=1) as live:
            while self.is_monitoring:
                live.update(self._create_dashboard())
                time.sleep(1)
    
    def _create_dashboard(self) -> Panel:
        """Create monitoring dashboard"""
        table = Table(title="RLDK Universal Monitor Dashboard")
        table.add_column("Run", style="cyan")
        table.add_column("Framework", style="magenta")
        table.add_column("Health Score", style="green")
        table.add_column("Alerts", style="red")
        table.add_column("Status", style="blue")
        
        for log_path, data in self.monitoring_data.items():
            health_score = data['health_score'].overall_score if data['health_score'] else "N/A"
            alerts_count = self.alerts.qsize()
            
            # Determine status color
            if health_score == "N/A":
                status = "Initializing"
                status_style = "yellow"
            elif health_score >= 85:
                status = "Healthy"
                status_style = "green"
            elif health_score >= 70:
                status = "Warning"
                status_style = "yellow"
            else:
                status = "Critical"
                status_style = "red"
            
            table.add_row(
                str(log_path.name),
                data['framework'],
                str(health_score),
                str(alerts_count),
                f"[{status_style}]{status}[/{status_style}]"
            )
        
        return Panel(table, title="[bold blue]RLDK Universal Monitor[/bold blue]")

# Convenience functions for easy usage
def start_monitoring(log_path: Union[str, Path], **kwargs) -> UniversalMonitor:
    """Start monitoring with default configuration"""
    config = MonitoringConfig(**kwargs)
    monitor = UniversalMonitor(config)
    monitor.start_monitoring(log_path)
    return monitor

def quick_debug(log_path: Union[str, Path]) -> Dict[str, Any]:
    """Quick debugging of training logs"""
    monitor = UniversalMonitor()
    framework = monitor.auto_detect_framework(log_path)
    
    # Run comprehensive analysis
    report = {
        'framework': framework,
        'analysis': monitor.scanner.scan_logs(log_path),
        'recommendations': []
    }
    
    return report