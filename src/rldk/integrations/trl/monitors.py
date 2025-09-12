"""Specialized monitors for TRL training components."""

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

try:
    from trl import PPOTrainer
    from trl.trainer.ppo_trainer import PPOTrainer as PPOTrainerClass
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    PPOTrainer = None
    PPOTrainerClass = None

# Import comprehensive PPO forensics
try:
    from rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics
    COMPREHENSIVE_FORENSICS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_FORENSICS_AVAILABLE = False


@dataclass
class PPOMetrics:
    """Container for PPO-specific metrics."""
    
    # Rollout metrics
    rollout_reward_mean: float = 0.0
    rollout_reward_std: float = 0.0
    rollout_reward_min: float = 0.0
    rollout_reward_max: float = 0.0
    rollout_length_mean: float = 0.0
    rollout_length_std: float = 0.0
    
    # Policy metrics
    policy_kl_mean: float = 0.0
    policy_kl_std: float = 0.0
    policy_entropy_mean: float = 0.0
    policy_entropy_std: float = 0.0
    policy_clip_frac: float = 0.0
    policy_loss: float = 0.0
    
    # Value function metrics
    value_loss: float = 0.0
    value_mean: float = 0.0
    value_std: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    
    # Learning metrics
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    clip_ratio: float = 0.0
    
    # Efficiency metrics
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_utilization: float = 0.0
    
    # Health indicators
    policy_collapse_risk: float = 0.0
    reward_hacking_risk: float = 0.0
    training_stability: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }


class PPOMonitor(TrainerCallback):
    """Specialized monitor for PPO training with advanced analytics."""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        kl_threshold: float = 0.1,
        reward_threshold: float = 0.05,
        gradient_threshold: float = 1.0,
        clip_frac_threshold: float = 0.2,
        enable_advanced_analytics: bool = True,
        run_id: Optional[str] = None,
    ):
        """Initialize PPO monitor.
        
        Args:
            output_dir: Directory to save PPO analysis
            kl_threshold: KL divergence alert threshold
            reward_threshold: Reward variance alert threshold
            gradient_threshold: Gradient norm alert threshold
            clip_frac_threshold: Clip fraction alert threshold
            enable_advanced_analytics: Enable advanced PPO analytics
            run_id: Unique identifier for this run
        """
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL is required for PPOMonitor. Install with: pip install trl"
            )
        
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_ppo_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds
        self.kl_threshold = kl_threshold
        self.reward_threshold = reward_threshold
        self.gradient_threshold = gradient_threshold
        self.clip_frac_threshold = clip_frac_threshold
        
        self.enable_advanced_analytics = enable_advanced_analytics
        self.run_id = run_id or f"ppo_run_{int(time.time())}"
        
        # Metrics storage
        self.ppo_metrics_history: List[PPOMetrics] = []
        self.current_ppo_metrics = PPOMetrics()
        
        # Advanced analytics
        self.reward_distribution_history: List[List[float]] = []
        self.kl_divergence_history: List[float] = []
        self.policy_entropy_history: List[float] = []
        
        # Alert system
        self.ppo_alerts: List[Dict[str, Any]] = []
        
        print(f"🎯 PPO Monitor initialized - Run ID: {self.run_id}")
        print(f"📊 PPO thresholds: KL={kl_threshold}, Reward={reward_threshold}, "
              f"Gradient={gradient_threshold}, ClipFrac={clip_frac_threshold}")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor PPO-specific metrics at step end."""
        # Extract PPO metrics from logs
        self._extract_ppo_metrics_from_logs(state)
        
        # Note: We don't store metrics here because on_log is called after on_step_end
        # and contains the actual logged values. We'll store metrics in on_log instead.
        
        # Log PPO metrics
        if state.global_step % 10 == 0:
            self._log_ppo_metrics()
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Extract PPO metrics from training logs."""
        # PPO rollout metrics
        if 'ppo/rewards/mean' in logs:
            self.current_ppo_metrics.rollout_reward_mean = logs['ppo/rewards/mean']
        if 'ppo/rewards/std' in logs:
            self.current_ppo_metrics.rollout_reward_std = logs['ppo/rewards/std']
        if 'ppo/rewards/min' in logs:
            self.current_ppo_metrics.rollout_reward_min = logs['ppo/rewards/min']
        if 'ppo/rewards/max' in logs:
            self.current_ppo_metrics.rollout_reward_max = logs['ppo/rewards/max']
        
        # PPO policy metrics
        if 'ppo/policy/kl_mean' in logs:
            self.current_ppo_metrics.policy_kl_mean = logs['ppo/policy/kl_mean']
            self.kl_divergence_history.append(logs['ppo/policy/kl_mean'])
        if 'ppo/policy/kl_std' in logs:
            self.current_ppo_metrics.policy_kl_std = logs['ppo/policy/kl_std']
        if 'ppo/policy/entropy' in logs:
            self.current_ppo_metrics.policy_entropy_mean = logs['ppo/policy/entropy']
            self.policy_entropy_history.append(logs['ppo/policy/entropy'])
        if 'ppo/policy/clipfrac' in logs:
            self.current_ppo_metrics.policy_clip_frac = logs['ppo/policy/clipfrac']
        # Check both policy loss key variants for backward compatibility
        if 'ppo/policy/policy_loss' in logs:
            self.current_ppo_metrics.policy_loss = logs['ppo/policy/policy_loss']
        elif 'ppo/val/policy_loss' in logs:
            self.current_ppo_metrics.policy_loss = logs['ppo/val/policy_loss']
        
        # PPO value function metrics
        if 'ppo/val/value_loss' in logs:
            self.current_ppo_metrics.value_loss = logs['ppo/val/value_loss']
        if 'ppo/val/mean' in logs:
            self.current_ppo_metrics.value_mean = logs['ppo/val/mean']
        if 'ppo/val/std' in logs:
            self.current_ppo_metrics.value_std = logs['ppo/val/std']
        
        # Learning metrics
        if 'learning_rate' in logs:
            self.current_ppo_metrics.learning_rate = logs['learning_rate']
        if 'grad_norm' in logs:
            self.current_ppo_metrics.gradient_norm = logs['grad_norm']
        
        # Advanced analytics
        if self.enable_advanced_analytics:
            self._analyze_policy_health()
            self._detect_reward_hacking()
            self._monitor_convergence()
        
        # Store metrics AFTER log values are applied
        self.ppo_metrics_history.append(PPOMetrics(**self.current_ppo_metrics.to_dict()))
        
        # Check for PPO-specific alerts AFTER metrics are stored
        self._check_ppo_alerts()
    
    def _extract_ppo_metrics_from_logs(self, state: TrainerState):
        """Extract PPO metrics from trainer state."""
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            
            # Extract rollout length if available
            if 'ppo/rollout/length_mean' in latest_log:
                self.current_ppo_metrics.rollout_length_mean = latest_log['ppo/rollout/length_mean']
            if 'ppo/rollout/length_std' in latest_log:
                self.current_ppo_metrics.rollout_length_std = latest_log['ppo/rollout/length_std']
    
    def _analyze_policy_health(self):
        """Analyze policy health indicators."""
        # Policy collapse detection
        if len(self.kl_divergence_history) > 10:
            recent_kl = self.kl_divergence_history[-10:]
            kl_trend = np.polyfit(range(len(recent_kl)), recent_kl, 1)[0]
            
            # High KL divergence trend indicates potential policy collapse
            if kl_trend > 0.01 and np.mean(recent_kl) > 0.05:
                self.current_ppo_metrics.policy_collapse_risk = min(1.0, np.mean(recent_kl) * 10)
            else:
                self.current_ppo_metrics.policy_collapse_risk = 0.0
        
        # Training stability based on entropy
        if len(self.policy_entropy_history) > 20:
            recent_entropy = self.policy_entropy_history[-20:]
            entropy_std = np.std(recent_entropy)
            self.current_ppo_metrics.training_stability = max(0, 1 - entropy_std)
    
    def _detect_reward_hacking(self):
        """Detect potential reward hacking patterns."""
        if len(self.ppo_metrics_history) > 50:
            recent_rewards = [m.rollout_reward_mean for m in self.ppo_metrics_history[-50:] 
                            if m.rollout_reward_mean != 0]
            
            if len(recent_rewards) > 10:
                # Check for suspicious reward patterns
                reward_std = np.std(recent_rewards)
                reward_mean = np.mean(recent_rewards)
                
                # High variance with low mean might indicate reward hacking
                if reward_std > 2 * abs(reward_mean) and reward_mean < 0:
                    self.current_ppo_metrics.reward_hacking_risk = min(1.0, reward_std / abs(reward_mean))
                else:
                    self.current_ppo_metrics.reward_hacking_risk = 0.0
    
    def _monitor_convergence(self):
        """Monitor training convergence."""
        if len(self.ppo_metrics_history) > 100:
            # Analyze reward trend over last 100 steps
            recent_rewards = [m.rollout_reward_mean for m in self.ppo_metrics_history[-100:] 
                            if m.rollout_reward_mean != 0]
            
            if len(recent_rewards) > 50:
                # Calculate convergence indicator
                reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                reward_variance = np.var(recent_rewards)
                
                # Convergence: low trend and low variance
                convergence_score = max(0, 1 - abs(reward_trend) - reward_variance)
                # Store in a way that can be accessed by the main callback
                self.current_ppo_metrics.training_stability = convergence_score
    
    def _check_ppo_alerts(self):
        """Check for PPO-specific training issues."""
        current = self.current_ppo_metrics
        
        # KL divergence alert
        if current.policy_kl_mean > self.kl_threshold:
            self._add_ppo_alert(
                "high_kl_divergence",
                f"Policy KL divergence {current.policy_kl_mean:.4f} exceeds threshold {self.kl_threshold}"
            )
        
        # Reward variance alert
        if current.rollout_reward_std > self.reward_threshold:
            self._add_ppo_alert(
                "high_reward_variance",
                f"Reward std {current.rollout_reward_std:.4f} exceeds threshold {self.reward_threshold}"
            )
        
        # Gradient norm alert
        if current.gradient_norm > self.gradient_threshold:
            self._add_ppo_alert(
                "high_gradient_norm",
                f"Gradient norm {current.gradient_norm:.4f} exceeds threshold {self.gradient_threshold}"
            )
        
        # Clip fraction alert
        if current.policy_clip_frac > self.clip_frac_threshold:
            self._add_ppo_alert(
                "high_clip_fraction",
                f"Clip fraction {current.policy_clip_frac:.4f} exceeds threshold {self.clip_frac_threshold}"
            )
        
        # Policy collapse alert
        if current.policy_collapse_risk > 0.7:
            self._add_ppo_alert(
                "policy_collapse_risk",
                f"High policy collapse risk: {current.policy_collapse_risk:.3f}"
            )
        
        # Reward hacking alert
        if current.reward_hacking_risk > 0.5:
            self._add_ppo_alert(
                "reward_hacking_risk",
                f"Potential reward hacking detected: {current.reward_hacking_risk:.3f}"
            )
    
    def _add_ppo_alert(self, alert_type: str, message: str):
        """Add a PPO-specific alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "step": len(self.ppo_metrics_history),
            "timestamp": time.time(),
            "severity": "warning",
            "ppo_metrics": self.current_ppo_metrics.to_dict()
        }
        self.ppo_alerts.append(alert)
        print(f"🚨 PPO Alert: {message}")
    
    def _log_ppo_metrics(self):
        """Log PPO-specific metrics."""
        current = self.current_ppo_metrics
        print(f"🎯 PPO Step {len(self.ppo_metrics_history)}: "
              f"Reward={current.rollout_reward_mean:.4f}±{current.rollout_reward_std:.4f}, "
              f"KL={current.policy_kl_mean:.4f}, "
              f"Entropy={current.policy_entropy_mean:.4f}, "
              f"ClipFrac={current.policy_clip_frac:.4f}, "
              f"CollapseRisk={current.policy_collapse_risk:.3f}")
    
    def save_ppo_analysis(self):
        """Save comprehensive PPO analysis."""
        if not self.ppo_metrics_history:
            return
        
        # Save metrics
        df = pd.DataFrame([m.to_dict() for m in self.ppo_metrics_history])
        metrics_path = self.output_dir / f"{self.run_id}_ppo_metrics.csv"
        df.to_csv(metrics_path, index=False)
        
        # Save alerts
        if self.ppo_alerts:
            alerts_path = self.output_dir / f"{self.run_id}_ppo_alerts.json"
            with open(alerts_path, "w") as f:
                json.dump(self.ppo_alerts, f, indent=2)
        
        # Generate PPO summary report
        self._generate_ppo_report()
    
    def _generate_ppo_report(self):
        """Generate comprehensive PPO training report."""
        if not self.ppo_metrics_history:
            return
        
        df = pd.DataFrame([m.to_dict() for m in self.ppo_metrics_history])
        
        report = {
            "run_id": self.run_id,
            "total_steps": len(self.ppo_metrics_history),
            "final_reward_mean": df['rollout_reward_mean'].iloc[-1] if len(df) > 0 else 0,
            "final_reward_std": df['rollout_reward_std'].iloc[-1] if len(df) > 0 else 0,
            "final_kl_mean": df['policy_kl_mean'].iloc[-1] if len(df) > 0 else 0,
            "final_entropy": df['policy_entropy_mean'].iloc[-1] if len(df) > 0 else 0,
            "max_policy_collapse_risk": df['policy_collapse_risk'].max() if len(df) > 0 else 0,
            "max_reward_hacking_risk": df['reward_hacking_risk'].max() if len(df) > 0 else 0,
            "average_training_stability": df['training_stability'].mean() if len(df) > 0 else 0,
            "total_alerts": len(self.ppo_alerts),
            "alert_types": list(set(alert['type'] for alert in self.ppo_alerts))
        }
        
        report_path = self.output_dir / f"{self.run_id}_ppo_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 PPO Report: {report}")


@dataclass
class CheckpointMetrics:
    """Container for checkpoint analysis metrics."""
    
    step: int = 0
    epoch: float = 0.0
    timestamp: float = 0.0
    
    # Model metrics
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Parameter analysis
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    
    # Gradient analysis
    gradient_mean: float = 0.0
    gradient_std: float = 0.0
    gradient_norm: float = 0.0
    
    # Health indicators
    parameter_drift: float = 0.0
    gradient_flow_health: float = 1.0
    model_health_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }


class CheckpointMonitor(TrainerCallback):
    """Monitor for checkpoint analysis and model health."""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        enable_parameter_analysis: bool = True,
        enable_gradient_analysis: bool = True,
        run_id: Optional[str] = None,
        max_parameter_size_mb: int = 50,  # Configurable limit
        max_total_memory_mb: int = 500,   # Total memory limit
    ):
        """Initialize checkpoint monitor."""
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_checkpoint_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_parameter_analysis = enable_parameter_analysis
        self.enable_gradient_analysis = enable_gradient_analysis
        self.run_id = run_id or f"checkpoint_run_{int(time.time())}"
        
        # Memory management settings
        self.max_parameter_size_mb = max_parameter_size_mb
        self.max_total_memory_mb = max_total_memory_mb
        
        # Metrics storage
        self.checkpoint_metrics_history: List[CheckpointMetrics] = []
        self.previous_weights: Optional[Dict[str, torch.Tensor]] = None
        self._current_memory_usage_mb = 0
        
        print(f"💾 Checkpoint Monitor initialized - Run ID: {self.run_id}")
        print(f"📊 Memory limits: {max_parameter_size_mb}MB per param, {max_total_memory_mb}MB total")
    
    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            self._cleanup_weights()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Analyze checkpoint when saved."""
        model = kwargs.get('model')
        if model is None:
            return
        
        # Store current model reference for drift calculation (using weak reference)
        import weakref
        self._current_model = weakref.ref(model) if model is not None else None
        
        # Create checkpoint metrics
        checkpoint_metrics = CheckpointMetrics(
            step=state.global_step,
            epoch=state.epoch,
            timestamp=time.time()
        )
        
        # Analyze model parameters
        if self.enable_parameter_analysis:
            self._analyze_parameters(model, checkpoint_metrics)
        
        # Analyze gradients
        if self.enable_gradient_analysis:
            self._analyze_gradients(model, checkpoint_metrics)
        
        # Calculate health indicators
        self._calculate_health_indicators(checkpoint_metrics)
        
        # Store metrics
        self.checkpoint_metrics_history.append(checkpoint_metrics)
        
        # Update previous weights for next comparison (with memory management)
        new_weights = self._extract_current_weights(model)
        if new_weights is not None:
            # Clean up old weights to prevent memory accumulation
            self._cleanup_weights()
            self.previous_weights = new_weights
        
        # Save checkpoint analysis
        self._save_checkpoint_analysis(checkpoint_metrics, model, state)
        
        print(f"💾 Checkpoint {state.global_step} analyzed - "
              f"Health Score: {checkpoint_metrics.model_health_score:.3f}")
    
    def _analyze_parameters(self, model, metrics: CheckpointMetrics):
        """Analyze model parameters."""
        all_weights = []
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                all_weights.extend(param.data.flatten().cpu().numpy())
            total_params += param.numel()
        
        metrics.total_parameters = total_params
        metrics.trainable_parameters = trainable_params
        
        if all_weights:
            all_weights = np.array(all_weights)
            metrics.weight_mean = float(np.mean(all_weights))
            metrics.weight_std = float(np.std(all_weights))
            metrics.weight_min = float(np.min(all_weights))
            metrics.weight_max = float(np.max(all_weights))
        
        # Estimate model size
        metrics.model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    def _analyze_gradients(self, model, metrics: CheckpointMetrics):
        """Analyze model gradients."""
        all_gradients = []
        total_grad_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                all_gradients.extend(param.grad.data.flatten().cpu().numpy())
        
        metrics.gradient_norm = total_grad_norm ** 0.5
        
        if all_gradients:
            all_gradients = np.array(all_gradients)
            metrics.gradient_mean = float(np.mean(all_gradients))
            metrics.gradient_std = float(np.std(all_gradients))
    
    def _calculate_health_indicators(self, metrics: CheckpointMetrics):
        """Calculate model health indicators."""
        # Parameter drift detection
        if self.previous_weights is not None and len(self.checkpoint_metrics_history) > 0:
            # Calculate actual parameter drift by comparing current weights with previous weights
            # Note: We need the model to extract current weights, but this method doesn't have access to it
            # For now, we'll use a simplified approach based on parameter statistics
            if hasattr(self, '_current_model') and self._current_model is not None:
                model_ref = self._current_model()  # Get model from weak reference
                if model_ref is not None:
                    current_weights = self._extract_current_weights(model_ref)
                    if current_weights is not None:
                        drift_score = self._calculate_weight_drift(self.previous_weights, current_weights)
                        metrics.parameter_drift = drift_score
                    else:
                        metrics.parameter_drift = 0.0
                else:
                    metrics.parameter_drift = 0.0
            else:
                metrics.parameter_drift = 0.0
        else:
            metrics.parameter_drift = 0.0
        
        # Gradient flow health
        if metrics.gradient_norm > 0:
            # Healthy gradient norms are typically between 0.1 and 10
            if 0.1 <= metrics.gradient_norm <= 10:
                metrics.gradient_flow_health = 1.0
            else:
                metrics.gradient_flow_health = max(0, 1 - abs(np.log10(metrics.gradient_norm)))
        
        # Overall model health score
        metrics.model_health_score = (
            metrics.gradient_flow_health * 0.6 +
            (1 - min(1, metrics.parameter_drift)) * 0.4
        )
    
    def _save_checkpoint_analysis(self, metrics: CheckpointMetrics, model, state: TrainerState):
        """Save detailed checkpoint analysis."""
        analysis = {
            "checkpoint_info": metrics.to_dict(),
            "model_info": {
                "total_parameters": metrics.total_parameters,
                "trainable_parameters": metrics.trainable_parameters,
                "model_size_mb": metrics.model_size_mb,
            },
            "timestamp": time.time(),
            "step": state.global_step,
            "epoch": state.epoch
        }
        
        analysis_path = self.output_dir / f"{self.run_id}_checkpoint_{state.global_step}.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
    
    def save_checkpoint_summary(self):
        """Save checkpoint monitoring summary."""
        if not self.checkpoint_metrics_history:
            return
        
        df = pd.DataFrame([m.to_dict() for m in self.checkpoint_metrics_history])
        summary_path = self.output_dir / f"{self.run_id}_checkpoint_summary.csv"
        df.to_csv(summary_path, index=False)
        
        # Generate summary report
        report = {
            "run_id": self.run_id,
            "total_checkpoints": len(self.checkpoint_metrics_history),
            "average_health_score": df['model_health_score'].mean(),
            "min_health_score": df['model_health_score'].min(),
            "max_parameter_drift": df['parameter_drift'].max(),
            "average_gradient_norm": df['gradient_norm'].mean(),
            "model_size_mb": df['model_size_mb'].iloc[-1] if len(df) > 0 else 0,
        }
        
        report_path = self.output_dir / f"{self.run_id}_checkpoint_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Checkpoint Report: {report}")
    
    def _cleanup_weights(self):
        """Clean up stored weights to prevent memory leaks."""
        # Clean up previous weights
        if hasattr(self, 'previous_weights') and self.previous_weights is not None:
            # Explicitly delete tensors to free memory
            for name, tensor in self.previous_weights.items():
                if hasattr(tensor, 'cpu'):
                    tensor.cpu()  # Move to CPU before deletion
                del tensor
            del self.previous_weights
            self.previous_weights = None
        
        # Clean up current model weak reference
        if hasattr(self, '_current_model') and self._current_model is not None:
            # Check if the weak reference is still valid
            model_ref = self._current_model()
            if model_ref is None:
                # Weak reference is dead, clean it up
                del self._current_model
                self._current_model = None
        
        # Clean up checkpoint metrics history if it gets too large
        if hasattr(self, 'checkpoint_metrics_history') and len(self.checkpoint_metrics_history) > 100:
            # Keep only the last 50 checkpoints
            self.checkpoint_metrics_history = self.checkpoint_metrics_history[-50:]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _extract_current_weights(self, model) -> Optional[Dict[str, torch.Tensor]]:
        """Extract current model weights for drift calculation with memory management."""
        if model is None:
            return None
        
        weights = {}
        total_size_mb = 0
        skipped_params = 0
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_size_mb = (param.numel() * param.element_size()) / (1024 * 1024)
                    
                    # Skip parameters that are too large
                    if param_size_mb > self.max_parameter_size_mb:
                        skipped_params += 1
                        continue
                    
                    # Check total memory limit
                    if total_size_mb + param_size_mb > self.max_total_memory_mb:
                        print(f"Warning: Memory limit reached. Skipping remaining parameters.")
                        break
                    
                    # Clone parameter with proper cleanup
                    try:
                        weights[name] = param.data.clone().detach()
                        total_size_mb += param_size_mb
                    except RuntimeError as e:
                        print(f"Warning: Failed to clone parameter {name}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Warning: Could not extract weights due to error: {e}")
            return None
        
        if skipped_params > 0:
            print(f"Info: Skipped {skipped_params} parameters due to size limits")
        
        self._current_memory_usage_mb = total_size_mb
        return weights if weights else None
    
    def _calculate_weight_drift(self, prev_weights: Dict[str, torch.Tensor], 
                               curr_weights: Dict[str, torch.Tensor]) -> float:
        """Calculate parameter drift between two weight sets."""
        if prev_weights is None or curr_weights is None:
            return 0.0
        
        total_drift = 0.0
        total_params = 0
        
        for name in prev_weights:
            if name in curr_weights:
                prev_param = prev_weights[name]
                curr_param = curr_weights[name]
                
                if prev_param.shape == curr_param.shape:
                    # Calculate relative change
                    param_diff = torch.abs(curr_param - prev_param)
                    param_norm = torch.norm(prev_param)
                    
                    if param_norm > 0:
                        relative_drift = torch.norm(param_diff) / param_norm
                        total_drift += relative_drift.item() * prev_param.numel()
                        total_params += prev_param.numel()
        
        if total_params > 0:
            return total_drift / total_params
        else:
            return 0.0


class ComprehensivePPOMonitor(TrainerCallback):
    """Comprehensive PPO monitor with advanced forensics capabilities."""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        kl_target: float = 0.1,
        kl_target_tolerance: float = 0.05,
        enable_kl_schedule_tracking: bool = True,
        enable_gradient_norms_analysis: bool = True,
        enable_advantage_statistics: bool = True,
        log_interval: int = 10,
        run_id: Optional[str] = None,
    ):
        """Initialize comprehensive PPO monitor.
        
        Args:
            output_dir: Directory to save analysis results
            kl_target: Target KL divergence value
            kl_target_tolerance: Tolerance around target for "in range" calculation
            enable_kl_schedule_tracking: Enable KL schedule tracking
            enable_gradient_norms_analysis: Enable gradient norms analysis
            enable_advantage_statistics: Enable advantage statistics tracking
            log_interval: Steps between detailed logging
            run_id: Unique identifier for this run
        """
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL is required for ComprehensivePPOMonitor. Install with: pip install trl"
            )
        
        if not COMPREHENSIVE_FORENSICS_AVAILABLE:
            raise ImportError(
                "Comprehensive PPO forensics is required. Ensure all forensics modules are available."
            )
        
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_comprehensive_ppo_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = log_interval
        self.run_id = run_id or f"comprehensive_ppo_run_{int(time.time())}"
        
        # Initialize comprehensive forensics
        self.forensics = ComprehensivePPOForensics(
            kl_target=kl_target,
            kl_target_tolerance=kl_target_tolerance,
            enable_kl_schedule_tracking=enable_kl_schedule_tracking,
            enable_gradient_norms_analysis=enable_gradient_norms_analysis,
            enable_advantage_statistics=enable_advantage_statistics,
        )
        
        # Metrics storage
        self.comprehensive_metrics_history: List[Dict[str, Any]] = []
        
        print(f"🔍 Comprehensive PPO Monitor initialized - Run ID: {self.run_id}")
        print(f"📊 KL Target: {kl_target}±{kl_target_tolerance}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor PPO training at step end."""
        # Extract metrics from trainer state if available
        self._extract_metrics_from_state(state)
        
        # Log comprehensive metrics at intervals
        if state.global_step % self.log_interval == 0:
            self._log_comprehensive_metrics()
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Extract and analyze PPO metrics from training logs."""
        # Extract basic PPO metrics
        step = state.global_step
        kl = logs.get('ppo/policy/kl_mean', 0.0)
        kl_coef = logs.get('ppo/policy/kl_coef', 1.0)
        entropy = logs.get('ppo/policy/entropy', 0.0)
        reward_mean = logs.get('ppo/rewards/mean', 0.0)
        reward_std = logs.get('ppo/rewards/std', 0.0)
        
        # Extract gradient norms
        policy_grad_norm = logs.get('ppo/policy/grad_norm', None)
        value_grad_norm = logs.get('ppo/val/grad_norm', None)
        total_grad_norm = logs.get('grad_norm', None)
        
        # Extract advantage statistics
        advantage_mean = logs.get('ppo/advantages/mean', None)
        advantage_std = logs.get('ppo/advantages/std', None)
        advantage_min = logs.get('ppo/advantages/min', None)
        advantage_max = logs.get('ppo/advantages/max', None)
        
        # Update comprehensive forensics
        comprehensive_metrics = self.forensics.update(
            step=step,
            kl=kl,
            kl_coef=kl_coef,
            entropy=entropy,
            reward_mean=reward_mean,
            reward_std=reward_std,
            policy_grad_norm=policy_grad_norm,
            value_grad_norm=value_grad_norm,
            total_grad_norm=total_grad_norm,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            advantage_min=advantage_min,
            advantage_max=advantage_max,
        )
        
        # Store metrics
        self.comprehensive_metrics_history.append(comprehensive_metrics.to_dict())
        
        # Check for anomalies
        anomalies = self.forensics.get_anomalies()
        if anomalies:
            self._handle_anomalies(anomalies, step)
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save comprehensive analysis when checkpoint is saved."""
        # Save comprehensive analysis
        analysis_path = self.output_dir / f"{self.run_id}_comprehensive_analysis_step_{state.global_step}.json"
        self.forensics.save_analysis(str(analysis_path))
        
        # Save metrics history
        self._save_metrics_history()
        
        print(f"💾 Comprehensive PPO analysis saved at step {state.global_step}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Generate final comprehensive analysis."""
        print("🏁 Comprehensive PPO Monitor: Training completed")
        
        # Generate final analysis
        final_analysis = self.forensics.get_comprehensive_analysis()
        health_summary = self.forensics.get_health_summary()
        
        # Save final analysis
        final_analysis_path = self.output_dir / f"{self.run_id}_final_comprehensive_analysis.json"
        with open(final_analysis_path, "w") as f:
            json.dump(final_analysis, f, indent=2)
        
        # Save health summary
        health_summary_path = self.output_dir / f"{self.run_id}_health_summary.json"
        with open(health_summary_path, "w") as f:
            json.dump(health_summary, f, indent=2)
        
        # Save metrics history
        self._save_metrics_history()
        
        # Print final summary
        print(f"📋 Final Health Summary: {health_summary}")
        print(f"📁 Comprehensive analysis saved to: {self.output_dir}")
    
    def _extract_metrics_from_state(self, state: TrainerState):
        """Extract additional metrics from trainer state."""
        # Extract basic training state information
        if hasattr(state, 'global_step'):
            self.forensics.current_metrics.step = state.global_step
        
        if hasattr(state, 'epoch'):
            self.forensics.current_metrics.epoch = state.epoch
        
        # Extract learning rate from log history if available
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            
            # Extract learning rate
            if 'learning_rate' in latest_log:
                self.forensics.current_metrics.learning_rate = latest_log['learning_rate']
            
            # Extract loss information
            if 'train_loss' in latest_log:
                self.forensics.current_metrics.loss = latest_log['train_loss']
            
            # Extract gradient norm
            if 'grad_norm' in latest_log:
                self.forensics.current_metrics.grad_norm = latest_log['grad_norm']
            
            # Extract PPO-specific metrics from log history
            if 'ppo/rewards/mean' in latest_log:
                self.forensics.current_metrics.reward_mean = latest_log['ppo/rewards/mean']
            if 'ppo/rewards/std' in latest_log:
                self.forensics.current_metrics.reward_std = latest_log['ppo/rewards/std']
            if 'ppo/policy/kl_mean' in latest_log:
                self.forensics.current_metrics.kl = latest_log['ppo/policy/kl_mean']
            if 'ppo/policy/kl_std' in latest_log:
                self.forensics.current_metrics.kl_std = latest_log['ppo/policy/kl_std']
            if 'ppo/policy/entropy' in latest_log:
                self.forensics.current_metrics.entropy = latest_log['ppo/policy/entropy']
            if 'ppo/policy/clipfrac' in latest_log:
                self.forensics.current_metrics.clip_frac = latest_log['ppo/policy/clipfrac']
            if 'ppo/val/value_loss' in latest_log:
                self.forensics.current_metrics.value_loss = latest_log['ppo/val/value_loss']
            # Check both policy loss key variants for backward compatibility
            if 'ppo/policy/policy_loss' in latest_log:
                self.forensics.current_metrics.policy_loss = latest_log['ppo/policy/policy_loss']
            elif 'ppo/val/policy_loss' in latest_log:
                self.forensics.current_metrics.policy_loss = latest_log['ppo/val/policy_loss']
            if 'ppo/policy/grad_norm' in latest_log:
                self.forensics.current_metrics.policy_grad_norm = latest_log['ppo/policy/grad_norm']
            if 'ppo/val/grad_norm' in latest_log:
                self.forensics.current_metrics.value_grad_norm = latest_log['ppo/val/grad_norm']
            if 'ppo/advantages/mean' in latest_log:
                self.forensics.current_metrics.advantage_mean = latest_log['ppo/advantages/mean']
            if 'ppo/advantages/std' in latest_log:
                self.forensics.current_metrics.advantage_std = latest_log['ppo/advantages/std']
            if 'ppo/policy/kl_coef' in latest_log:
                self.forensics.current_metrics.kl_coef = latest_log['ppo/policy/kl_coef']
            
            # Extract rollout information if available
            if 'ppo/rollout/length_mean' in latest_log:
                self.forensics.current_metrics.rollout_length_mean = latest_log['ppo/rollout/length_mean']
            if 'ppo/rollout/length_std' in latest_log:
                self.forensics.current_metrics.rollout_length_std = latest_log['ppo/rollout/length_std']
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from extracted state information."""
        # Calculate KL divergence if both mean and std are available
        if hasattr(self.forensics.current_metrics, 'kl') and hasattr(self.forensics.current_metrics, 'kl_std'):
            if self.forensics.current_metrics.kl is not None and self.forensics.current_metrics.kl_std is not None:
                # Store KL divergence for trend analysis
                if not hasattr(self, 'kl_divergence_history'):
                    self.kl_divergence_history = []
                self.kl_divergence_history.append(self.forensics.current_metrics.kl)
        
        # Calculate policy entropy trend if available
        if hasattr(self.forensics.current_metrics, 'entropy') and self.forensics.current_metrics.entropy is not None:
            if not hasattr(self, 'policy_entropy_history'):
                self.policy_entropy_history = []
            self.policy_entropy_history.append(self.forensics.current_metrics.entropy)
        
        # Calculate reward trend if available
        if hasattr(self.forensics.current_metrics, 'reward_mean') and self.forensics.current_metrics.reward_mean is not None:
            if not hasattr(self, 'reward_history'):
                self.reward_history = []
            self.reward_history.append(self.forensics.current_metrics.reward_mean)
        
        # Calculate advantage statistics if available
        if hasattr(self.forensics.current_metrics, 'advantage_mean') and hasattr(self.forensics.current_metrics, 'advantage_std'):
            if (self.forensics.current_metrics.advantage_mean is not None and 
                self.forensics.current_metrics.advantage_std is not None):
                if not hasattr(self, 'advantage_history'):
                    self.advantage_history = []
                self.advantage_history.append({
                    'mean': self.forensics.current_metrics.advantage_mean,
                    'std': self.forensics.current_metrics.advantage_std
                })
        
        # Calculate gradient norm ratio if both policy and value gradients are available
        if (hasattr(self.forensics.current_metrics, 'policy_grad_norm') and 
            hasattr(self.forensics.current_metrics, 'value_grad_norm')):
            if (self.forensics.current_metrics.policy_grad_norm is not None and 
                self.forensics.current_metrics.value_grad_norm is not None):
                if self.forensics.current_metrics.value_grad_norm > 0:
                    self.forensics.current_metrics.policy_value_grad_ratio = (
                        self.forensics.current_metrics.policy_grad_norm / self.forensics.current_metrics.value_grad_norm
                    )
    
    def _log_comprehensive_metrics(self):
        """Log comprehensive metrics at intervals."""
        current_metrics = self.forensics.current_metrics
        
        print(f"🔍 Comprehensive PPO Step {current_metrics.step}:")
        print(f"   Overall Health: {current_metrics.overall_health_score:.3f}")
        print(f"   Training Stability: {current_metrics.training_stability_score:.3f}")
        print(f"   Convergence Quality: {current_metrics.convergence_quality_score:.3f}")
        print(f"   KL: {current_metrics.kl:.4f}, KL Coef: {current_metrics.kl_coef:.4f}")
        print(f"   Reward: {current_metrics.reward_mean:.4f}±{current_metrics.reward_std:.4f}")
        
        # Log tracker-specific metrics
        if current_metrics.kl_schedule_metrics:
            kl_metrics = current_metrics.kl_schedule_metrics
            print(f"   KL Health: {kl_metrics.kl_health_score:.3f}, "
                  f"Schedule Health: {kl_metrics.schedule_health_score:.3f}")
        
        if current_metrics.gradient_norms_metrics:
            grad_metrics = current_metrics.gradient_norms_metrics
            print(f"   Gradient Health: {grad_metrics.gradient_health_score:.3f}, "
                  f"Policy/Value Ratio: {grad_metrics.policy_value_ratio:.3f}")
        
        if current_metrics.advantage_statistics_metrics:
            adv_metrics = current_metrics.advantage_statistics_metrics
            print(f"   Advantage Health: {adv_metrics.advantage_health_score:.3f}, "
                  f"Bias: {adv_metrics.advantage_bias:.4f}")
    
    def _handle_anomalies(self, anomalies: List[Dict[str, Any]], step: int):
        """Handle detected anomalies."""
        for anomaly in anomalies:
            severity = anomaly.get("severity", "warning")
            message = anomaly.get("message", "Unknown anomaly")
            tracker = anomaly.get("tracker", "unknown")
            
            if severity == "critical":
                print(f"🚨 CRITICAL ANOMALY [{tracker}]: {message}")
            elif severity == "warning":
                print(f"⚠️  WARNING [{tracker}]: {message}")
    
    def _save_metrics_history(self):
        """Save comprehensive metrics history."""
        if not self.comprehensive_metrics_history:
            return
        
        # Save as CSV
        df = pd.DataFrame(self.comprehensive_metrics_history)
        csv_path = self.output_dir / f"{self.run_id}_comprehensive_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.output_dir / f"{self.run_id}_comprehensive_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self.comprehensive_metrics_history, f, indent=2)
        
        print(f"💾 Comprehensive metrics saved: {csv_path}")
    
    def get_current_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        return self.forensics.get_health_summary()
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get current anomalies."""
        return self.forensics.get_anomalies()