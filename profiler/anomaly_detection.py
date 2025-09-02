"""
Advanced Anomaly Detection System for RLHF Training.

This module provides sophisticated detection rules for:
- Gradient explosion/vanishing detection
- Learning rate schedule anomalies
- Batch size impact analysis
- Model convergence tracking
- Reward model calibration drift
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import warnings
from scipy import stats
from scipy.signal import find_peaks
import json
from pathlib import Path


@dataclass
class AnomalyAlert:
    """Represents an anomaly detection alert."""
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'gradient', 'learning_rate', 'batch_size', 'convergence', 'reward_drift'
    message: str
    step: int
    value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientStats:
    """Statistics for gradient analysis."""
    mean: float
    std: float
    max: float
    min: float
    norm: float
    explosion_ratio: float
    vanishing_ratio: float


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence tracking."""
    loss_trend: float
    plateau_detected: bool
    convergence_rate: float
    stability_score: float
    improvement_rate: float


class GradientAnomalyDetector:
    """Detects gradient explosion and vanishing problems."""
    
    def __init__(
        self,
        explosion_threshold: float = 10.0,
        vanishing_threshold: float = 1e-6,
        window_size: int = 100,
        alert_threshold: float = 5.0
    ):
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        self.gradient_history = deque(maxlen=window_size)
        self.norm_history = deque(maxlen=window_size)
        self.alerts = []
    
    def analyze_gradients(self, model: nn.Module, step: int) -> List[AnomalyAlert]:
        """Analyze gradients for anomalies."""
        alerts = []
        
        # Collect gradient statistics
        total_norm = 0.0
        gradient_stats = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                
                grad_stats = GradientStats(
                    mean=param.grad.data.mean().item(),
                    std=param.grad.data.std().item(),
                    max=param.grad.data.max().item(),
                    min=param.grad.data.min().item(),
                    norm=grad_norm,
                    explosion_ratio=grad_norm / (param.data.norm(2).item() + 1e-8),
                    vanishing_ratio=grad_norm / (param.data.norm(2).item() + 1e-8)
                )
                gradient_stats.append((name, grad_stats))
        
        total_norm = total_norm ** 0.5
        
        # Store in history
        self.gradient_history.append(gradient_stats)
        self.norm_history.append(total_norm)
        
        # Check for gradient explosion
        if total_norm > self.explosion_threshold:
            severity = 'critical' if total_norm > self.explosion_threshold * 2 else 'high'
            # Convert gradient stats to serializable format
            serializable_stats = [(name, stats.__dict__) for name, stats in gradient_stats]
            alerts.append(AnomalyAlert(
                severity=severity,
                category='gradient',
                message=f"Gradient explosion detected: norm={total_norm:.4f} > {self.explosion_threshold}",
                step=step,
                value=total_norm,
                threshold=self.explosion_threshold,
                metadata={'gradient_stats': serializable_stats}
            ))
        
        # Check for gradient vanishing
        if total_norm < self.vanishing_threshold:
            # Convert gradient stats to serializable format
            serializable_stats = [(name, stats.__dict__) for name, stats in gradient_stats]
            alerts.append(AnomalyAlert(
                severity='high',
                category='gradient',
                message=f"Gradient vanishing detected: norm={total_norm:.8f} < {self.vanishing_threshold}",
                step=step,
                value=total_norm,
                threshold=self.vanishing_threshold,
                metadata={'gradient_stats': serializable_stats}
            ))
        
        # Check for sudden changes in gradient norms
        if len(self.norm_history) >= 10:
            recent_norms = list(self.norm_history)[-10:]
            norm_std = np.std(recent_norms)
            norm_mean = np.mean(recent_norms)
            
            if norm_std > norm_mean * self.alert_threshold:
                alerts.append(AnomalyAlert(
                    severity='medium',
                    category='gradient',
                    message=f"High gradient variance detected: std={norm_std:.4f} > {norm_mean * self.alert_threshold:.4f}",
                    step=step,
                    value=norm_std,
                    threshold=norm_mean * self.alert_threshold,
                    metadata={'recent_norms': recent_norms}
                ))
        
        return alerts


class LearningRateAnomalyDetector:
    """Detects anomalies in learning rate schedules."""
    
    def __init__(
        self,
        change_threshold: float = 0.5,
        window_size: int = 50,
        min_lr: float = 1e-8,
        max_lr: float = 1.0
    ):
        self.change_threshold = change_threshold
        self.window_size = window_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.lr_history = deque(maxlen=window_size)
        self.alerts = []
    
    def analyze_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> List[AnomalyAlert]:
        """Analyze learning rate for anomalies."""
        alerts = []
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        # Check for out-of-range learning rates
        if current_lr < self.min_lr:
            alerts.append(AnomalyAlert(
                severity='high',
                category='learning_rate',
                message=f"Learning rate too low: {current_lr:.2e} < {self.min_lr:.2e}",
                step=step,
                value=current_lr,
                threshold=self.min_lr,
                metadata={'optimizer_state': optimizer.state_dict()}
            ))
        
        if current_lr > self.max_lr:
            alerts.append(AnomalyAlert(
                severity='high',
                category='learning_rate',
                message=f"Learning rate too high: {current_lr:.2e} > {self.max_lr:.2e}",
                step=step,
                value=current_lr,
                threshold=self.max_lr,
                metadata={'optimizer_state': optimizer.state_dict()}
            ))
        
        # Check for sudden changes in learning rate
        if len(self.lr_history) >= 5:
            recent_lrs = list(self.lr_history)[-5:]
            lr_changes = [abs(recent_lrs[i] - recent_lrs[i-1]) / recent_lrs[i-1] 
                         for i in range(1, len(recent_lrs))]
            
            max_change = max(lr_changes) if lr_changes else 0
            
            if max_change > self.change_threshold:
                alerts.append(AnomalyAlert(
                    severity='medium',
                    category='learning_rate',
                    message=f"Sudden learning rate change detected: {max_change:.2%} > {self.change_threshold:.2%}",
                    step=step,
                    value=max_change,
                    threshold=self.change_threshold,
                    metadata={'recent_lrs': recent_lrs, 'changes': lr_changes}
                ))
        
        return alerts


class BatchSizeImpactAnalyzer:
    """Analyzes the impact of batch size changes on training."""
    
    def __init__(
        self,
        performance_threshold: float = 0.1,
        window_size: int = 20
    ):
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        self.batch_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.alerts = []
    
    def analyze_batch_impact(
        self, 
        batch_size: int, 
        loss: float, 
        step: int
    ) -> List[AnomalyAlert]:
        """Analyze batch size impact on training performance."""
        alerts = []
        
        self.batch_history.append(batch_size)
        self.loss_history.append(loss)
        
        # Check for batch size changes and their impact
        if len(self.batch_history) >= 10:
            recent_batches = list(self.batch_history)[-10:]
            recent_losses = list(self.loss_history)[-10:]
            
            # Detect batch size changes
            batch_changes = [recent_batches[i] != recent_batches[i-1] 
                           for i in range(1, len(recent_batches))]
            
            if any(batch_changes):
                # Analyze performance before and after batch size change
                change_idx = next(i for i, changed in enumerate(batch_changes) if changed)
                
                if change_idx > 2 and change_idx < len(recent_losses) - 2:
                    before_loss = np.mean(recent_losses[:change_idx])
                    after_loss = np.mean(recent_losses[change_idx+1:])
                    
                    performance_change = (after_loss - before_loss) / before_loss
                    
                    if abs(performance_change) > self.performance_threshold:
                        severity = 'high' if abs(performance_change) > self.performance_threshold * 2 else 'medium'
                        alerts.append(AnomalyAlert(
                            severity=severity,
                            category='batch_size',
                            message=f"Batch size change impact: {performance_change:.2%} performance change",
                            step=step,
                            value=performance_change,
                            threshold=self.performance_threshold,
                            metadata={
                                'batch_size_change': recent_batches[change_idx-1] != recent_batches[change_idx],
                                'before_loss': before_loss,
                                'after_loss': after_loss,
                                'change_idx': change_idx
                            }
                        ))
        
        return alerts


class ConvergenceTracker:
    """Tracks model convergence and detects convergence issues."""
    
    def __init__(
        self,
        plateau_threshold: float = 0.001,
        plateau_window: int = 50,
        min_improvement: float = 1e-6
    ):
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_improvement = min_improvement
        
        self.loss_history = deque(maxlen=plateau_window * 2)
        self.alerts = []
    
    def analyze_convergence(self, loss: float, step: int) -> Tuple[ConvergenceMetrics, List[AnomalyAlert]]:
        """Analyze convergence metrics and detect issues."""
        alerts = []
        
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.plateau_window:
            return ConvergenceMetrics(0, False, 0, 0, 0), alerts
        
        recent_losses = list(self.loss_history)[-self.plateau_window:]
        
        # Calculate loss trend
        x = np.arange(len(recent_losses))
        slope, _, r_value, _, _ = stats.linregress(x, recent_losses)
        loss_trend = slope
        
        # Detect plateau
        loss_std = np.std(recent_losses)
        plateau_detected = loss_std < self.plateau_threshold
        
        # Calculate convergence rate
        if len(recent_losses) >= 10:
            early_loss = np.mean(recent_losses[:5])
            late_loss = np.mean(recent_losses[-5:])
            convergence_rate = (early_loss - late_loss) / early_loss
        else:
            convergence_rate = 0
        
        # Calculate stability score
        stability_score = 1.0 / (1.0 + loss_std)
        
        # Calculate improvement rate
        if len(self.loss_history) >= 20:
            old_losses = list(self.loss_history)[-40:-20]
            new_losses = list(self.loss_history)[-20:]
            old_avg = np.mean(old_losses)
            new_avg = np.mean(new_losses)
            improvement_rate = (old_avg - new_avg) / old_avg
        else:
            improvement_rate = 0
        
        metrics = ConvergenceMetrics(
            loss_trend=loss_trend,
            plateau_detected=plateau_detected,
            convergence_rate=convergence_rate,
            stability_score=stability_score,
            improvement_rate=improvement_rate
        )
        
        # Generate alerts
        if plateau_detected and abs(improvement_rate) < self.min_improvement:
            alerts.append(AnomalyAlert(
                severity='medium',
                category='convergence',
                message=f"Training plateau detected: loss_std={loss_std:.6f} < {self.plateau_threshold}",
                step=step,
                value=loss_std,
                threshold=self.plateau_threshold,
                metadata={'metrics': metrics}
            ))
        
        if convergence_rate < -0.1:  # Loss is increasing
            alerts.append(AnomalyAlert(
                severity='high',
                category='convergence',
                message=f"Loss increasing: convergence_rate={convergence_rate:.4f}",
                step=step,
                value=convergence_rate,
                threshold=-0.1,
                metadata={'metrics': metrics}
            ))
        
        return metrics, alerts


class RewardCalibrationDriftDetector:
    """Enhanced reward model calibration drift detection."""
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        calibration_threshold: float = 0.7,
        window_size: int = 100
    ):
        self.drift_threshold = drift_threshold
        self.calibration_threshold = calibration_threshold
        self.window_size = window_size
        
        self.reward_history = deque(maxlen=window_size)
        self.calibration_history = deque(maxlen=window_size)
        self.alerts = []
    
    def analyze_reward_drift(
        self, 
        rewards: np.ndarray, 
        predictions: np.ndarray, 
        step: int
    ) -> List[AnomalyAlert]:
        """Analyze reward model calibration drift."""
        alerts = []
        
        # Calculate calibration metrics
        calibration_score = self._calculate_calibration_score(rewards, predictions)
        self.calibration_history.append(calibration_score)
        
        # Store reward statistics
        reward_stats = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        }
        self.reward_history.append(reward_stats)
        
        # Check calibration quality
        if calibration_score < self.calibration_threshold:
            alerts.append(AnomalyAlert(
                severity='high',
                category='reward_drift',
                message=f"Poor calibration detected: score={calibration_score:.4f} < {self.calibration_threshold}",
                step=step,
                value=calibration_score,
                threshold=self.calibration_threshold,
                metadata={'rewards': rewards, 'predictions': predictions}
            ))
        
        # Check for calibration drift
        if len(self.calibration_history) >= 20:
            recent_calibration = list(self.calibration_history)[-20:]
            calibration_trend = np.polyfit(range(len(recent_calibration)), recent_calibration, 1)[0]
            
            if abs(calibration_trend) > self.drift_threshold:
                alerts.append(AnomalyAlert(
                    severity='medium',
                    category='reward_drift',
                    message=f"Calibration drift detected: trend={calibration_trend:.4f}",
                    step=step,
                    value=calibration_trend,
                    threshold=self.drift_threshold,
                    metadata={'recent_calibration': recent_calibration}
                ))
        
        # Check for reward distribution changes
        if len(self.reward_history) >= 10:
            recent_rewards = list(self.reward_history)[-10:]
            old_rewards = list(self.reward_history)[-20:-10]
            
            if old_rewards:
                old_mean = np.mean([r['mean'] for r in old_rewards])
                new_mean = np.mean([r['mean'] for r in recent_rewards])
                
                mean_change = abs(new_mean - old_mean) / (old_mean + 1e-8)
                
                if mean_change > self.drift_threshold:
                    alerts.append(AnomalyAlert(
                        severity='medium',
                        category='reward_drift',
                        message=f"Reward distribution drift: mean change={mean_change:.4f}",
                        step=step,
                        value=mean_change,
                        threshold=self.drift_threshold,
                        metadata={'old_mean': old_mean, 'new_mean': new_mean}
                    ))
        
        return alerts
    
    def _calculate_calibration_score(self, rewards: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate calibration score using reliability diagram."""
        try:
            # Bin the predictions and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = rewards[in_bin].mean()
                    avg_confidence_in_bin = predictions[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return 1.0 - ece  # Higher is better
        except:
            return 0.5  # Default neutral score


class AdvancedAnomalyDetector:
    """Main anomaly detection system that coordinates all detectors."""
    
    def __init__(
        self,
        output_dir: str = "anomaly_detection",
        save_alerts: bool = True,
        **kwargs
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_alerts = save_alerts
        
        # Initialize detectors
        self.gradient_detector = GradientAnomalyDetector(**kwargs.get('gradient', {}))
        self.lr_detector = LearningRateAnomalyDetector(**kwargs.get('learning_rate', {}))
        self.batch_analyzer = BatchSizeImpactAnalyzer(**kwargs.get('batch_size', {}))
        self.convergence_tracker = ConvergenceTracker(**kwargs.get('convergence', {}))
        self.reward_detector = RewardCalibrationDriftDetector(**kwargs.get('reward_drift', {}))
        
        self.all_alerts = []
        self.step_count = 0
    
    def analyze_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        batch_size: int,
        rewards: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> List[AnomalyAlert]:
        """Analyze a single training step for anomalies."""
        self.step_count += 1
        alerts = []
        
        # Gradient analysis
        gradient_alerts = self.gradient_detector.analyze_gradients(model, self.step_count)
        alerts.extend(gradient_alerts)
        
        # Learning rate analysis
        lr_alerts = self.lr_detector.analyze_learning_rate(optimizer, self.step_count)
        alerts.extend(lr_alerts)
        
        # Batch size impact analysis
        batch_alerts = self.batch_analyzer.analyze_batch_impact(batch_size, loss, self.step_count)
        alerts.extend(batch_alerts)
        
        # Convergence tracking
        convergence_metrics, convergence_alerts = self.convergence_tracker.analyze_convergence(loss, self.step_count)
        alerts.extend(convergence_alerts)
        
        # Reward calibration drift (if available)
        if rewards is not None and predictions is not None:
            reward_alerts = self.reward_detector.analyze_reward_drift(rewards, predictions, self.step_count)
            alerts.extend(reward_alerts)
        
        # Store alerts
        self.all_alerts.extend(alerts)
        
        # Save alerts if enabled
        if self.save_alerts and alerts:
            self._save_alerts(alerts)
        
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all detected anomalies."""
        if not self.all_alerts:
            return {"total_alerts": 0, "by_category": {}, "by_severity": {}}
        
        by_category = {}
        by_severity = {}
        
        for alert in self.all_alerts:
            by_category[alert.category] = by_category.get(alert.category, 0) + 1
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
        
        return {
            "total_alerts": len(self.all_alerts),
            "by_category": by_category,
            "by_severity": by_severity,
            "latest_alerts": self.all_alerts[-10:] if len(self.all_alerts) > 10 else self.all_alerts
        }
    
    def _save_alerts(self, alerts: List[AnomalyAlert]):
        """Save alerts to file."""
        alert_data = []
        for alert in alerts:
            # Convert metadata to JSON-serializable format
            serializable_metadata = self._make_serializable(alert.metadata)
            
            alert_data.append({
                "severity": alert.severity,
                "category": alert.category,
                "message": alert.message,
                "step": alert.step,
                "value": alert.value,
                "threshold": alert.threshold,
                "metadata": serializable_metadata
            })
        
        alert_file = self.output_dir / f"alerts_step_{self.step_count}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (GradientStats, ConvergenceMetrics)):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                # Try to convert to string as last resort
                return str(obj)
            except:
                return None
    
    def save_final_report(self):
        """Save final anomaly detection report."""
        summary = self.get_summary()
        
        # Make summary serializable
        serializable_summary = self._make_serializable(summary)
        
        report_file = self.output_dir / "anomaly_detection_report.json"
        with open(report_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2, default=str)
        
        print(f"Anomaly detection report saved to {report_file}")
        return report_file