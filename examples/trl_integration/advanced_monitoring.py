"""Advanced monitoring example with custom metrics and alerts."""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# Import RLDK components
from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor, RLDKDashboard
from rldk.utils.math_utils import safe_rate, nan_aware_mean

try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


class CustomRLDKCallback(RLDKCallback):
    """Custom RLDK callback with additional monitoring capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics_history: List[Dict[str, Any]] = []
        self.performance_benchmarks = {
            "target_reward": 1.0,
            "max_kl_divergence": 0.1,
            "min_entropy": 1.0,
        }
        # Counters for division by zero tracking
        self.division_counters = {
            "samples_seen": 0,
            "samples_used": 0,
            "zero_denominator_skipped": 0,
            "non_positive_time_skipped": 0,
            "other_skip_reasons": []
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
        reward_score = min(1.0, current.reward_mean / self.performance_benchmarks["target_reward"])
        kl_score = max(0, 1 - current.kl_mean / self.performance_benchmarks["max_kl_divergence"])
        entropy_score = min(1.0, current.entropy_mean / self.performance_benchmarks["min_entropy"])
        
        performance_score = (reward_score + kl_score + entropy_score) / 3
        
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
        """Calculate training efficiency metrics with robust division."""
        if len(self.metrics_history) < 2:
            return
        
        # Calculate tokens per second using robust division
        recent_metrics = self.metrics_history[-5:]
        total_tokens = sum(m.tokens_in + m.tokens_out for m in recent_metrics if m.tokens_in and m.tokens_out)
        total_time = sum(m.step_time for m in recent_metrics if m.step_time > 0)
        
        self.division_counters["samples_seen"] += 1
        
        tokens_per_sec, used, reason = safe_rate(total_tokens, total_time, on_zero="skip")
        if used:
            self.current_metrics.tokens_per_second = tokens_per_sec
            self.division_counters["samples_used"] += 1
        else:
            self.current_metrics.tokens_per_second = 0.0
            if reason == "zero_denominator_skipped":
                self.division_counters["zero_denominator_skipped"] += 1
            elif reason == "non_positive_time_skipped":
                self.division_counters["non_positive_time_skipped"] += 1
            else:
                self.division_counters["other_skip_reasons"].append(reason)
    
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
            efficiency_ratio, used, reason = safe_rate(reward_improvement, time_elapsed, on_zero="skip")
            if used:
                return max(0, efficiency_ratio)
        
        return 1.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.custom_metrics_history:
            return {}
        
        df = pd.DataFrame(self.custom_metrics_history)
        
        return {
            "average_performance_score": nan_aware_mean(df['performance_score'].tolist()),
            "best_performance_score": df['performance_score'].max(),
            "average_efficiency_ratio": nan_aware_mean(df['efficiency_ratio'].tolist()),
            "total_custom_alerts": len([a for a in self.alerts if a['type'] == 'low_performance']),
            "performance_trend": self._calculate_performance_trend(),
            "division_counters": self.division_counters,
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
    
    def get_throughput_metrics(self) -> Dict[str, Any]:
        """Get throughput metrics with division counters."""
        if not hasattr(self.current_metrics, 'tokens_per_second'):
            return {
                "tokens_per_second": 0.0,
                "window_size_used": 0,
                "zero_time_samples_skipped": 0,
                "division_counters": self.division_counters
            }
        
        return {
            "tokens_per_second": getattr(self.current_metrics, 'tokens_per_second', 0.0),
            "window_size_used": self.division_counters["samples_used"],
            "zero_time_samples_skipped": self.division_counters["zero_denominator_skipped"],
            "division_counters": self.division_counters
        }
    
    def save_metrics_history(self):
        """Save custom metrics history and call parent method."""
        # Save custom metrics
        if self.custom_metrics_history:
            custom_df = pd.DataFrame(self.custom_metrics_history)
            custom_path = self.output_dir / f"{self.run_id}_custom_metrics.csv"
            custom_df.to_csv(custom_path, index=False)
            print(f"📊 Custom metrics saved to {custom_path}")
        
        # Call parent method to save standard metrics
        super().save_metrics_history()


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
            "convergence_percentage": sum(self.convergence_analysis.values()) / len(self.convergence_analysis) * 100,
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
            from transformers import TrainerState, TrainerControl, TrainingArguments
            
            args = TrainingArguments(output_dir=output_dir)
            state = TrainerState()
            state.global_step = len(custom_callback.metrics_history)
            state.epoch = state.global_step / 10.0
            control = TrainerControl()
            
            custom_callback.on_step_end(args, state, control)
            custom_callback.on_log(args, state, control, fake_logs)
            
            advanced_ppo_monitor.on_step_end(args, state, control)
            advanced_ppo_monitor.on_log(args, state, control, fake_logs)
            
            if step % 3 == 0:
                # Simulate checkpoint
                fake_model = torch.nn.Linear(10, 1)  # Simple model for testing
                custom_callback.on_save(args, state, control, model=fake_model)
                checkpoint_monitor.on_save(args, state, control, model=fake_model)
        
        print(f"✅ {scenario['name']} scenario completed")
    
    # Generate reports
    print("\n📊 Generating Advanced Reports")
    
    # Performance summary
    performance_summary = custom_callback.get_performance_summary()
    print(f"Performance Summary: {performance_summary}")
    
    # Throughput metrics with counters
    throughput_metrics = custom_callback.get_throughput_metrics()
    print(f"Throughput Metrics: {throughput_metrics}")
    
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