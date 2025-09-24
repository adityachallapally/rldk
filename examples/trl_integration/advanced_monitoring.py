"""Advanced monitoring example with custom metrics and alerts."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Import RLDK components
from rldk.integrations.trl import (
    CheckpointMonitor,
    PPOMonitor,
    RLDKCallback,
    RLDKDashboard,
    RLDKMetrics,
)
from rldk.utils.math_utils import safe_divide, safe_rate_calculation

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


class CustomRLDKCallback(RLDKCallback):
    """Custom RLDK callback with additional monitoring capabilities."""

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
            self._fallback_mode = False
        except ImportError:
            self._initialize_without_trl(*args, **kwargs)
            self._fallback_mode = True
        self.custom_metrics_history: List[Dict[str, Any]] = []
        self.performance_benchmarks = {
            "target_reward": 1.0,
            "max_kl_divergence": 0.1,
            "min_entropy": 1.0,
        }

    def on_step_end(self, args, state, control, **kwargs):
        """Enhanced step end monitoring."""
        super().on_step_end(args, state, control, **kwargs)

        # Custom performance analysis
        self._analyze_performance()

        # Custom efficiency metrics
        self._calculate_efficiency_metrics()

    def _analyze_performance(self):
        """Analyze training performance against benchmarks."""
        current = self.current_metrics

        # Performance score calculation
        reward_score = min(1.0, safe_divide(current.reward_mean, self.performance_benchmarks["target_reward"], 0.0))
        kl_score = max(0, 1 - safe_divide(current.kl_mean, self.performance_benchmarks["max_kl_divergence"], 0.0))
        entropy_score = min(1.0, safe_divide(current.entropy_mean, self.performance_benchmarks["min_entropy"], 0.0))

        performance_score = safe_divide(reward_score + kl_score + entropy_score, 3, 0.0)

        # Store custom metrics
        custom_metrics = {
            "step": current.step,
            "performance_score": performance_score,
            "reward_score": reward_score,
            "kl_score": kl_score,
            "entropy_score": entropy_score,
            "efficiency_ratio": self._calculate_efficiency_ratio(),
        }

        self.custom_metrics_history.append(custom_metrics)

        # Performance alerts
        if performance_score < 0.5:
            self._add_alert("low_performance",
                          f"Performance score {performance_score:.3f} is below threshold")

    def _calculate_efficiency_metrics(self):
        """Calculate training efficiency metrics."""
        if len(self.metrics_history) < 2:
            return

        # Calculate tokens per second
        recent_metrics = self.metrics_history[-5:]
        total_tokens = sum(m.tokens_in + m.tokens_out for m in recent_metrics if m.tokens_in and m.tokens_out)
        total_time = sum(m.step_time for m in recent_metrics if m.step_time > 0)

        if total_time > 0:
            self.current_metrics.tokens_per_second = safe_rate_calculation(total_tokens, total_time, 0.0)
        else:
            self.current_metrics.tokens_per_second = 0.0

    def _calculate_efficiency_ratio(self) -> float:
        """Calculate overall training efficiency ratio."""
        if len(self.metrics_history) < 10:
            return 1.0

        # Efficiency based on reward improvement vs time
        recent_rewards = [m.reward_mean for m in self.metrics_history[-10:] if m.reward_mean != 0]
        if len(recent_rewards) < 5:
            return 1.0

        reward_improvement = recent_rewards[-1] - recent_rewards[0]
        time_elapsed = self.metrics_history[-1].wall_time - self.metrics_history[-10].wall_time

        if time_elapsed > 0:
            return max(0, safe_rate_calculation(reward_improvement, time_elapsed, 0.0))

        return 1.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.custom_metrics_history:
            return {}

        df = pd.DataFrame(self.custom_metrics_history)

        return {
            "average_performance_score": df['performance_score'].mean(),
            "best_performance_score": df['performance_score'].max(),
            "average_efficiency_ratio": df['efficiency_ratio'].mean(),
            "total_custom_alerts": len([a for a in self.alerts if a['type'] == 'low_performance']),
            "performance_trend": self._calculate_performance_trend(),
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend over time."""
        if len(self.custom_metrics_history) < 10:
            return "insufficient_data"

        recent_scores = [m['performance_score'] for m in self.custom_metrics_history[-10:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"

    def save_metrics_history(self):
        """Save custom metrics history and call parent method."""
        # Save custom metrics
        if self.custom_metrics_history:
            custom_df = pd.DataFrame(self.custom_metrics_history)
            custom_path = self.output_dir / f"{self.run_id}_custom_metrics.csv"
            custom_df.to_csv(custom_path, index=False)
            print(f"📊 Custom metrics saved to {custom_path}")

        # Call parent method to save standard metrics
        if not getattr(self, "_fallback_mode", False):
            super().save_metrics_history()

    def _initialize_without_trl(
        self,
        output_dir: str | None = None,
        log_interval: int = 10,
        alert_thresholds: Dict[str, float] | None = None,
        enable_checkpoint_analysis: bool = True,
        enable_resource_monitoring: bool = True,
        run_id: str | None = None,
        enable_jsonl_logging: bool = True,
        jsonl_log_interval: int = 1,
        **_: Any,
    ) -> None:
        """Minimal initialization path used when TRL isn't available."""

        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.enable_checkpoint_analysis = enable_checkpoint_analysis
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_jsonl_logging = enable_jsonl_logging
        self.jsonl_log_interval = jsonl_log_interval

        self.alert_thresholds = {
            "kl_divergence": 0.1,
            "clip_fraction": 0.2,
            "gradient_norm": 1.0,
            "reward_std": 0.5,
            "loss_spike": 2.0,
            "memory_usage": 0.9,
        }
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)

        self.metrics_history = []
        self.current_metrics = RLDKMetrics()
        self.step_start_time = time.time()
        self.run_start_time = time.time()

        self.run_id = run_id or f"rldk_run_{int(self.run_start_time)}"
        self.current_metrics.run_id = self.run_id

        self.jsonl_file = None
        self.alerts = []


class AdvancedPPOMonitor(PPOMonitor):
    """Advanced PPO monitor with custom analytics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convergence_analysis = {
            "reward_convergence": False,
            "policy_convergence": False,
            "value_convergence": False,
        }
        self.anomaly_detection_enabled = True

    def _detect_anomalies(self):
        """Detect training anomalies."""
        if not self.anomaly_detection_enabled or len(self.ppo_metrics_history) < 20:
            return

        recent_metrics = self.ppo_metrics_history[-20:]

        # Detect reward anomalies
        rewards = [m.rollout_reward_mean for m in recent_metrics if m.rollout_reward_mean != 0]
        if len(rewards) > 10:
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)

            # Check for outliers
            outliers = [r for r in rewards if abs(r - reward_mean) > 3 * reward_std]
            if len(outliers) > len(rewards) * 0.1:  # More than 10% outliers
                self._add_ppo_alert("reward_anomalies",
                                  f"Detected {len(outliers)} reward outliers in recent steps")

        # Detect policy collapse
        kl_values = [m.policy_kl_mean for m in recent_metrics if m.policy_kl_mean != 0]
        if len(kl_values) > 10:
            kl_trend = np.polyfit(range(len(kl_values)), kl_values, 1)[0]
            if kl_trend > 0.05:  # Rapidly increasing KL divergence
                self._add_ppo_alert("policy_collapse_detected",
                                  f"Policy collapse detected: KL trend {kl_trend:.4f}")

    def _analyze_convergence(self):
        """Enhanced convergence analysis."""
        super()._monitor_convergence()

        if len(self.ppo_metrics_history) < 50:
            return

        recent_metrics = self.ppo_metrics_history[-50:]

        # Reward convergence
        rewards = [m.rollout_reward_mean for m in recent_metrics if m.rollout_reward_mean != 0]
        if len(rewards) > 20:
            reward_variance = np.var(rewards[-20:])
            self.convergence_analysis["reward_convergence"] = reward_variance < 0.01

        # Policy convergence
        kl_values = [m.policy_kl_mean for m in recent_metrics if m.policy_kl_mean != 0]
        if len(kl_values) > 20:
            kl_variance = np.var(kl_values[-20:])
            self.convergence_analysis["policy_convergence"] = kl_variance < 0.001

        # Value convergence
        value_losses = [m.value_loss for m in recent_metrics if m.value_loss != 0]
        if len(value_losses) > 20:
            value_variance = np.var(value_losses[-20:])
            self.convergence_analysis["value_convergence"] = value_variance < 0.01

    def on_step_end(self, args, state, control, **kwargs):
        """Enhanced step end with anomaly detection."""
        super().on_step_end(args, state, control, **kwargs)

        # Note: Advanced analytics are now called in on_log after metrics are populated

    def get_convergence_report(self) -> Dict[str, Any]:
        """Get convergence analysis report."""
        return {
            "convergence_analysis": self.convergence_analysis,
            "all_converged": all(self.convergence_analysis.values()),
            "convergence_percentage": safe_divide(sum(self.convergence_analysis.values()), len(self.convergence_analysis), 0.0) * 100,
        }

    def save_ppo_analysis(self):
        """Save advanced PPO analysis and call parent method."""
        # Save convergence analysis
        convergence_path = self.output_dir / f"{self.run_id}_convergence_analysis.json"
        with open(convergence_path, "w") as f:
            json.dump(self.convergence_analysis, f, indent=2)
        print(f"📊 Convergence analysis saved to {convergence_path}")

        # Call parent method to save standard PPO analysis
        super().save_ppo_analysis()


class AdvancedMonitoringCallback(CustomRLDKCallback):
    """Backward compatible alias exposing the advanced monitoring features."""

    # This subclass exists to preserve the import path used in legacy tooling.
    # All functionality is provided by ``CustomRLDKCallback``.
    def _calculate_tokens_per_second(self) -> float:
        """Compute tokens per second using safe rate calculations for tests."""

        total_tokens = 0.0
        total_time = 0.0

        for metric in getattr(self, "metrics_history", []):
            tokens_in = getattr(metric, "tokens_in", 0) or 0
            tokens_out = getattr(metric, "tokens_out", 0) or 0
            step_time = getattr(metric, "step_time", 0) or 0

            total_tokens += float(tokens_in) + float(tokens_out)
            total_time += float(step_time)

        tokens_per_second = safe_rate_calculation(total_tokens, total_time, 0.0)

        if hasattr(self, "current_metrics"):
            setattr(self.current_metrics, "tokens_per_second", tokens_per_second)

        return tokens_per_second


def test_advanced_monitoring():
    """Test advanced monitoring capabilities."""
    print("🚀 Testing Advanced Monitoring")

    output_dir = "./test_advanced_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize advanced monitors
    custom_callback = CustomRLDKCallback(
        output_dir=output_dir,
        log_interval=2,
        run_id="advanced_test"
    )

    advanced_ppo_monitor = AdvancedPPOMonitor(
        output_dir=output_dir,
        kl_threshold=0.08,  # Stricter threshold
        reward_threshold=0.03,
        run_id="advanced_test"
    )

    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        enable_parameter_analysis=True,
        enable_gradient_analysis=True,
        run_id="advanced_test"
    )

    print("✅ Advanced monitors initialized")

    # Simulate training with various scenarios
    scenarios = [
        {"name": "Normal Training", "reward_trend": 0.01, "kl_trend": 0.001},
        {"name": "Reward Hacking", "reward_trend": 0.1, "kl_trend": 0.05},
        {"name": "Policy Collapse", "reward_trend": -0.05, "kl_trend": 0.1},
        {"name": "Convergence", "reward_trend": 0.001, "kl_trend": 0.0001},
    ]

    for scenario in scenarios:
        print(f"\n🎭 Testing scenario: {scenario['name']}")

        # Simulate training steps for this scenario
        for step in range(10):
            # Generate metrics based on scenario
            base_reward = 0.5 + step * scenario['reward_trend']
            base_kl = 0.05 + step * scenario['kl_trend']

            # Add some noise
            reward_noise = np.random.normal(0, 0.1)
            kl_noise = np.random.normal(0, 0.01)

            fake_logs = {
                'ppo/rewards/mean': max(0, base_reward + reward_noise),
                'ppo/rewards/std': 0.2 + abs(reward_noise),
                'ppo/policy/kl_mean': max(0, base_kl + kl_noise),
                'ppo/policy/entropy': 2.0 - step * 0.05,
                'ppo/policy/clipfrac': 0.1 + abs(kl_noise),
                'ppo/val/value_loss': 0.3 - step * 0.02,
                'learning_rate': 1e-5,
                'grad_norm': 0.5 + abs(kl_noise),
            }

            # Call callbacks
            from transformers import TrainerControl, TrainerState, TrainingArguments

            args = TrainingArguments(output_dir=output_dir)
            state = TrainerState()
            state.global_step = len(custom_callback.metrics_history)
            state.epoch = safe_divide(state.global_step, 10.0, 0.0)
            control = TrainerControl()

            custom_callback.on_step_end(args, state, control)
            custom_callback.on_log(args, state, control, fake_logs)

            advanced_ppo_monitor.log_metrics(step=state.global_step, metrics=fake_logs)

            if step % 3 == 0:
                # Simulate checkpoint
                fake_model = torch.nn.Linear(10, 1)  # Simple model for testing
                custom_callback.on_save(args, state, control, model=fake_model)
                checkpoint_monitor.log_checkpoint(
                    step=state.global_step,
                    checkpoint_data={
                        "epoch": state.epoch,
                        "timestamp": time.time(),
                        "gradient_norm": fake_logs.get('grad_norm', 0.0),
                    },
                    model=fake_model,
                )

        print(f"✅ {scenario['name']} scenario completed")

    # Generate reports
    print("\n📊 Generating Advanced Reports")

    # Performance summary
    performance_summary = custom_callback.get_performance_summary()
    print(f"Performance Summary: {performance_summary}")

    # Convergence report
    convergence_report = advanced_ppo_monitor.get_convergence_report()
    print(f"Convergence Report: {convergence_report}")

    # Save all data
    custom_callback.save_metrics_history()
    advanced_ppo_monitor.save_ppo_analysis()
    checkpoint_monitor.save_checkpoint_summary()

    print("✅ Advanced monitoring test completed")
    return True


def test_dashboard_integration():
    """Test dashboard integration."""
    print("📊 Testing Dashboard Integration")

    try:
        # Initialize dashboard
        dashboard = RLDKDashboard(
            output_dir="./test_advanced_output",
            port=8502,  # Different port to avoid conflicts
            run_id="advanced_test"
        )

        print("✅ Dashboard initialized")

        # Test dashboard app creation
        app_file = dashboard.output_dir / "test_dashboard_app.py"
        dashboard._create_dashboard_app(app_file)

        if app_file.exists():
            print("✅ Dashboard app created successfully")
        else:
            print("❌ Dashboard app creation failed")
            return False

        print("✅ Dashboard integration test passed")
        return True

    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎯 Advanced RLDK TRL Integration Test Suite")
    print("=" * 60)

    # Test advanced monitoring
    success1 = test_advanced_monitoring()
    print()

    # Test dashboard integration
    success2 = test_dashboard_integration()

    if success1 and success2:
        print("\n🎉 All advanced tests passed!")
    else:
        print("\n❌ Some advanced tests failed.")
