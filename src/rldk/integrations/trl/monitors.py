"""Specialized monitors for TRL training components."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
try:
    from transformers import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    TRAINER_API_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    TRAINER_API_AVAILABLE = False

    class _TrainerAPINotAvailableStub:
        _ERROR = (
            "Transformers trainer callbacks are required for RLDK's TRL monitors. "
            "Install with: pip install 'transformers[torch]'"
        )

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(self._ERROR)

    class TrainerCallback(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

    class TrainerControl(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

    class TrainerState(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

    class TrainingArguments(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

try:  # pragma: no cover - optional dependency for typing compatibility
    import torch
except ImportError:  # pragma: no cover - optional dependency for typing compatibility
    class _TorchStub:
        class nn:
            Module = object

    torch = _TorchStub()  # type: ignore

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

    _LOG_KEY_MAPPING = {
        "ppo/rewards/mean": "rollout_reward_mean",
        "ppo/rewards/std": "rollout_reward_std",
        "ppo/rewards/min": "rollout_reward_min",
        "ppo/rewards/max": "rollout_reward_max",
        "ppo/rollout/length_mean": "rollout_length_mean",
        "ppo/rollout/length_std": "rollout_length_std",
        "ppo/policy/kl_mean": "policy_kl_mean",
        "ppo/policy/kl_std": "policy_kl_std",
        "ppo/policy/entropy": "policy_entropy_mean",
        "ppo/policy/entropy_std": "policy_entropy_std",
        "ppo/policy/clipfrac": "policy_clip_frac",
        "ppo/policy/policy_loss": "policy_loss",
        "ppo/val/policy_loss": "policy_loss",
        "ppo/val/value_loss": "value_loss",
        "ppo/val/value_mean": "value_mean",
        "ppo/val/value_std": "value_std",
        "ppo/val/mean": "value_mean",
        "ppo/val/std": "value_std",
        "ppo/advantages/mean": "advantage_mean",
        "ppo/advantages/std": "advantage_std",
        "ppo/efficiency/tokens_per_second": "tokens_per_second",
        "ppo/efficiency/samples_per_second": "samples_per_second",
        "ppo/resources/gpu_utilization": "gpu_utilization",
        "ppo/health/policy_collapse_risk": "policy_collapse_risk",
        "ppo/health/reward_hacking_risk": "reward_hacking_risk",
        "ppo/health/training_stability": "training_stability",
        "ppo/training/clip_ratio": "clip_ratio",
        "learning_rate": "learning_rate",
        "grad_norm": "gradient_norm",
    }

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
        if not TRAINER_API_AVAILABLE:
            raise ImportError(
                "Transformers trainer callbacks are required for PPOMonitor. "
                "Install with: pip install 'transformers[torch]'"
            )

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
        self._last_logged_step: int = 0

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
        step = getattr(state, "global_step", len(self.ppo_metrics_history))
        self.log_metrics(step=step, metrics=logs)

    def log_metrics(self, step: int | None, metrics: Mapping[str, Any]) -> PPOMetrics:
        """Record PPO metrics outside of the Hugging Face Trainer callbacks.

        This helper allows simulations, tests, and custom training loops to
        feed PPO metrics directly into the monitor while still benefiting from
        alerting and advanced analytics.

        Args:
            step: The global training step associated with the metrics.
            metrics: Mapping of metric values. Keys can either use the
                ``PPOMetrics`` field names or the TRL logging keys
                (e.g. ``"ppo/rewards/mean"``).
        Returns:
            The :class:`PPOMetrics` instance that was stored for ``step``.
        """

        if step is None:
            step = len(self.ppo_metrics_history)

        try:
            step = int(step)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
            raise TypeError("step must be an integer value") from exc

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be provided as a mapping")

        normalized_metrics = self._normalize_metrics_dict(metrics)
        if not normalized_metrics:
            # Nothing to record; keep current metrics unchanged
            return PPOMetrics(**self.current_ppo_metrics.to_dict())

        self._last_logged_step = step

        current_data = self.current_ppo_metrics.to_dict()
        current_data.update(normalized_metrics)
        self.current_ppo_metrics = PPOMetrics(**current_data)

        if "policy_kl_mean" in normalized_metrics:
            self.kl_divergence_history.append(normalized_metrics["policy_kl_mean"])
        if "policy_entropy_mean" in normalized_metrics:
            self.policy_entropy_history.append(normalized_metrics["policy_entropy_mean"])

        if self.enable_advanced_analytics:
            self._analyze_policy_health()
            self._detect_reward_hacking()
            self._monitor_convergence()

        logged_metrics = PPOMetrics(**self.current_ppo_metrics.to_dict())
        self.ppo_metrics_history.append(logged_metrics)
        self._check_ppo_alerts()

        return logged_metrics

    def _normalize_metrics_dict(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Normalize raw metric keys to ``PPOMetrics`` field names."""

        normalized: Dict[str, float] = {}
        valid_fields = self.current_ppo_metrics.__dataclass_fields__.keys()

        for key, value in metrics.items():
            if value is None:
                continue

            normalized_key = self._LOG_KEY_MAPPING.get(key, key)
            if normalized_key not in valid_fields:
                continue

            try:
                normalized[normalized_key] = float(value)
            except (TypeError, ValueError):
                continue

        return normalized

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
        alert_step = getattr(self, "_last_logged_step", len(self.ppo_metrics_history))
        alert = {
            "type": alert_type,
            "message": message,
            "step": alert_step,
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
            "alert_types": list({alert['type'] for alert in self.ppo_alerts})
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
    ):
        """Initialize checkpoint monitor."""
        if not TRAINER_API_AVAILABLE:
            raise ImportError(
                "Transformers trainer callbacks are required for CheckpointMonitor. "
                "Install with: pip install 'transformers[torch]'"
            )

        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL is required for CheckpointMonitor. Install with: pip install trl"
            )
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_checkpoint_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_parameter_analysis = enable_parameter_analysis
        self.enable_gradient_analysis = enable_gradient_analysis
        self.run_id = run_id or f"checkpoint_run_{int(time.time())}"

        # Metrics storage
        self.checkpoint_metrics_history: List[CheckpointMetrics] = []
        self.previous_weights: Optional[Dict[str, "torch.Tensor"]] = None

        print(f"💾 Checkpoint Monitor initialized - Run ID: {self.run_id}")

    def log_checkpoint(
        self,
        step: int | None,
        checkpoint_data: Mapping[str, Any],
        model: torch.nn.Module | None = None,
    ) -> CheckpointMetrics:
        """Record checkpoint metrics from external integrations.

        Args:
            step: Global training step when the checkpoint was produced.
            checkpoint_data: Mapping containing checkpoint statistics.
                Keys should match :class:`CheckpointMetrics` fields when
                available.
            model: Optional model instance whose parameters and gradients
                should be analyzed for additional insights.
        Returns:
            The :class:`CheckpointMetrics` instance that was recorded.
        """

        if step is None:
            step = len(self.checkpoint_metrics_history)

        if not isinstance(checkpoint_data, Mapping):
            raise TypeError("checkpoint_data must be provided as a mapping")

        checkpoint_dict = dict(checkpoint_data)

        try:
            step = int(step)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
            raise TypeError("step must be an integer value") from exc

        epoch_value = checkpoint_dict.get("epoch", 0.0)
        timestamp_value = checkpoint_dict.get("timestamp", time.time())

        try:
            epoch_float = float(epoch_value)
        except (TypeError, ValueError):  # pragma: no cover - fallback when casting fails
            epoch_float = 0.0

        try:
            timestamp_float = float(timestamp_value)
        except (TypeError, ValueError):  # pragma: no cover - fallback when casting fails
            timestamp_float = float(time.time())

        metrics = CheckpointMetrics(
            step=step,
            epoch=epoch_float,
            timestamp=timestamp_float,
        )

        for field_name in metrics.__dataclass_fields__:
            if field_name in {"step", "epoch", "timestamp"}:
                continue
            if field_name in checkpoint_dict and checkpoint_dict[field_name] is not None:
                setattr(metrics, field_name, checkpoint_dict[field_name])

        if model is not None:
            if self.enable_parameter_analysis:
                self._analyze_parameters(model, metrics)
            if self.enable_gradient_analysis:
                self._analyze_gradients(model, metrics)

        self._calculate_health_indicators(metrics)
        self.checkpoint_metrics_history.append(metrics)

        self._save_checkpoint_analysis(
            metrics,
            model,
            step=metrics.step,
            epoch=metrics.epoch,
        )

        print(f"💾 Checkpoint {step} logged - Health Score: {metrics.model_health_score:.3f}")

        return metrics

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Analyze checkpoint when saved."""
        model = kwargs.get('model')
        if model is None:
            return

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

        # Save checkpoint analysis
        self._save_checkpoint_analysis(
            checkpoint_metrics,
            model,
            step=state.global_step,
            epoch=state.epoch,
        )

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
            # This is a simplified drift calculation
            # In practice, you'd compare with the previous checkpoint's weights
            metrics.parameter_drift = 0.0  # Placeholder

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

    def _save_checkpoint_analysis(
        self,
        metrics: CheckpointMetrics,
        model: torch.nn.Module | None,
        *,
        step: int | None = None,
        epoch: float | None = None,
    ):
        """Save detailed checkpoint analysis."""

        checkpoint_step = metrics.step if step is None else int(step)
        checkpoint_epoch = metrics.epoch if epoch is None else float(epoch)

        analysis = {
            "checkpoint_info": metrics.to_dict(),
            "model_info": {
                "total_parameters": metrics.total_parameters,
                "trainable_parameters": metrics.trainable_parameters,
                "model_size_mb": metrics.model_size_mb,
            },
            "timestamp": time.time(),
            "step": checkpoint_step,
            "epoch": checkpoint_epoch
        }

        analysis_path = self.output_dir / f"{self.run_id}_checkpoint_{checkpoint_step}.json"
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
        enable_length_bias_detection: bool = True,
        length_bias_threshold: float = 0.35,
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
            enable_length_bias_detection: Enable length bias detection on collected responses
            length_bias_threshold: Threshold used for severity-driven alerts
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
            enable_length_bias_detection=enable_length_bias_detection,
            length_bias_threshold=length_bias_threshold,
        )

        # Metrics storage
        self.comprehensive_metrics_history: List[Dict[str, Any]] = []

        # Response tracking for length-bias analysis
        self._pending_response_payloads: List[Dict[str, Any]] = []
        self._recent_response_fingerprints: deque[Tuple[Any, ...]] = deque(maxlen=512)
        self._recent_response_fingerprint_set: set[Tuple[Any, ...]] = set()

        print(f"🔍 Comprehensive PPO Monitor initialized - Run ID: {self.run_id}")
        print(f"📊 KL Target: {kl_target}±{kl_target_tolerance}")
        print(f"📁 Output directory: {self.output_dir}")
        if enable_length_bias_detection:
            print(
                f"📏 Length bias threshold: {length_bias_threshold:.3f}"
            )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor PPO training at step end."""
        # Extract metrics from trainer state if available
        self._extract_metrics_from_state(state)

        # Cache response payloads for length bias detection when available
        self._queue_response_payloads(None, kwargs)

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

        # Gather response payloads for length bias detection
        self._queue_response_payloads(logs, kwargs)
        response_payloads: Optional[List[Dict[str, Any]]] = None
        if self._pending_response_payloads:
            response_payloads = list(self._pending_response_payloads)
            self._pending_response_payloads.clear()

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
            response_data=response_payloads,
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

    def _queue_response_payloads(
        self,
        logs: Optional[Mapping[str, Any]],
        callback_kwargs: Optional[Mapping[str, Any]],
    ) -> None:
        """Collect response payloads emitted by the PPO loop for length bias detection."""

        if not getattr(self.forensics, "enable_length_bias_detection", False):
            return

        new_records = self._collect_response_payloads(logs, callback_kwargs)
        for record in new_records:
            fingerprint = self._fingerprint_response_record(record)
            if not self._remember_response_fingerprint(fingerprint):
                continue
            self._pending_response_payloads.append(record)

    def _collect_response_payloads(
        self,
        logs: Optional[Mapping[str, Any]],
        callback_kwargs: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract normalized response payloads from callback inputs."""

        records: List[Dict[str, Any]] = []

        def consume(source: Any) -> None:
            if source is None:
                return
            if isinstance(source, Mapping):
                records.extend(self._extract_records_from_mapping(source, consume))
            elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
                for item in source:
                    consume(item)

        if callback_kwargs:
            consume(callback_kwargs)
        if logs:
            consume(logs)

        return records

    def _extract_records_from_mapping(
        self,
        mapping: Mapping[str, Any],
        consume: Any,
    ) -> List[Dict[str, Any]]:
        """Extract response records from a mapping and recurse into nested payloads."""

        records = self._extract_batch_records(mapping)

        single_record = self._extract_single_record(mapping)
        if single_record:
            records.append(single_record)

        nested_keys = (
            "response_data",
            "responses",
            "response_payloads",
            "ppo_response_data",
            "ppo_responses",
            "ppo_response_payloads",
            "samples",
            "sample_batch",
            "trajectories",
            "episodes",
            "rollouts",
            "rollout_samples",
            "ppo_rollouts",
            "ppo_batch",
            "batch",
            "outputs",
            "details",
        )
        for key in nested_keys:
            nested = mapping.get(key)
            if nested is not None and nested is not mapping:
                consume(nested)

        for value in mapping.values():
            if value is mapping:
                continue
            if isinstance(value, Mapping):
                consume(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                consume(value)

        return records

    def _extract_batch_records(self, mapping: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Extract response records from batch-style payloads."""

        reward_seq = None
        for key in ("rewards", "reward_scores", "scores", "returns"):
            reward_seq = self._as_sequence(mapping.get(key))
            if reward_seq:
                break
        if not reward_seq:
            return []

        response_seq = None
        for key in ("responses", "response_texts", "response", "completions", "outputs"):
            response_seq = self._as_sequence(mapping.get(key))
            if response_seq:
                break

        length_seq = None
        for key in (
            "response_lengths",
            "lengths",
            "response_length",
            "response_tokens",
            "token_counts",
            "tokens_out",
            "output_lengths",
            "output_tokens",
        ):
            length_seq = self._as_sequence(mapping.get(key))
            if length_seq:
                break

        records: List[Dict[str, Any]] = []
        for idx, reward_value in enumerate(reward_seq):
            response_value = None
            if response_seq and idx < len(response_seq):
                response_value = response_seq[idx]
            length_value = None
            if length_seq and idx < len(length_seq):
                length_value = length_seq[idx]

            record = self._build_response_record(reward_value, response_value, length_value)
            if record:
                records.append(record)

        return records

    def _extract_single_record(self, mapping: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a single response record from scalar entries."""

        reward_candidate = self._first_present(
            mapping,
            (
                "reward",
                "response_reward",
                "reward_score",
                "score",
            ),
        )
        response_candidate = self._first_present(
            mapping,
            (
                "response",
                "response_text",
                "completion",
                "output",
            ),
        )
        length_candidate = self._first_present(
            mapping,
            (
                "length",
                "response_length",
                "response_tokens",
                "tokens_out",
                "output_length",
            ),
        )

        if isinstance(response_candidate, Sequence) and not isinstance(response_candidate, (str, bytes)):
            response_candidate = None
        if isinstance(length_candidate, Sequence) and not isinstance(length_candidate, (str, bytes)):
            length_candidate = None

        return self._build_response_record(
            reward_candidate,
            response_candidate,
            length_candidate,
        )

    def _build_response_record(
        self,
        reward: Any,
        response: Any,
        length: Any,
    ) -> Optional[Dict[str, Any]]:
        reward_value = self._to_float(reward)
        if reward_value is None:
            return None

        record: Dict[str, Any] = {"reward": reward_value}

        length_value = self._to_float(length)
        if length_value is not None:
            record["length"] = length_value

        if response is not None:
            record["response"] = str(response)

        if "length" not in record and "response" not in record:
            return None

        return record

    def _as_sequence(self, value: Any) -> Optional[List[Any]]:
        if value is None or isinstance(value, (str, bytes)):
            return None
        if isinstance(value, Mapping):
            return None
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            try:
                value = value.detach().cpu().numpy()
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "tolist"):
            try:
                candidate = value.tolist()
                if isinstance(candidate, (list, tuple)):
                    return list(candidate)
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(value, Sequence):
            return list(value)
        return None

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, np.generic):
            return float(value.item())
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "item"):
            try:
                return float(value.detach().cpu().item())
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        try:
            return float(value)
        except Exception:
            return None

    def _first_present(self, mapping: Mapping[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if value is not None:
                    return value
        return None

    def _fingerprint_response_record(self, record: Mapping[str, Any]) -> Tuple[Any, ...]:
        reward = record.get("reward")
        length = record.get("length")
        response = record.get("response")
        reward_key = None if reward is None else round(float(reward), 6)
        length_key = None if length is None else round(float(length), 3)
        response_key = None if response is None else str(response)[:128]
        return (reward_key, length_key, response_key)

    def _remember_response_fingerprint(self, fingerprint: Tuple[Any, ...]) -> bool:
        if fingerprint in self._recent_response_fingerprint_set:
            return False
        if len(self._recent_response_fingerprints) >= self._recent_response_fingerprints.maxlen:
            oldest = self._recent_response_fingerprints.popleft()
            self._recent_response_fingerprint_set.discard(oldest)
        self._recent_response_fingerprints.append(fingerprint)
        self._recent_response_fingerprint_set.add(fingerprint)
        return True

    def _log_comprehensive_metrics(self):
        """Log comprehensive metrics at intervals."""
        current_metrics = self.forensics.current_metrics

        print(f"🔍 Comprehensive PPO Step {current_metrics.step}:")
        print(f"   Overall Health: {current_metrics.overall_health_score:.3f}")
        print(f"   Training Stability: {current_metrics.training_stability_score:.3f}")
        print(f"   Convergence Quality: {current_metrics.convergence_quality_score:.3f}")
        print(f"   KL: {current_metrics.kl:.4f}, KL Coef: {current_metrics.kl_coef:.4f}")
        print(f"   Reward: {current_metrics.reward_mean:.4f}±{current_metrics.reward_std:.4f}")

        if (
            current_metrics.length_bias_score is not None
            and current_metrics.length_bias_threshold is not None
        ):
            print(
                "   Length bias: "
                f"{current_metrics.length_bias_score:.3f}"
                f" (threshold={current_metrics.length_bias_threshold:.3f})"
            )

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
