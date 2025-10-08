"""Comprehensive tests for PPO forensics features."""

import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from rldk.forensics.advantage_statistics_tracker import (
    AdvantageStatisticsMetrics,
    AdvantageStatisticsTracker,
)
from rldk.forensics.comprehensive_ppo_forensics import (
    ComprehensivePPOForensics,
    ComprehensivePPOMetrics,
)
from rldk.forensics.gradient_norms_analyzer import (
    GradientNormsAnalyzer,
    GradientNormsMetrics,
)
from rldk.forensics.kl_schedule_tracker import KLScheduleMetrics, KLScheduleTracker


class TestKLScheduleTracker:
    """Test KL schedule tracking functionality."""

    def test_kl_schedule_tracker_initialization(self):
        """Test KL schedule tracker initialization."""
        tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)

        assert tracker.kl_target == 0.1
        assert tracker.kl_target_tolerance == 0.05
        assert tracker.target_range_low == pytest.approx(0.05)
        assert tracker.target_range_high == pytest.approx(0.15)
        assert len(tracker.kl_history) == 0

    def test_kl_schedule_tracker_update(self):
        """Test KL schedule tracker update functionality."""
        tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)

        # Update with normal KL values
        for step in range(20):
            kl = 0.1 + 0.02 * np.sin(step * 0.1)  # Oscillating around target
            kl_coef = 1.0 + 0.1 * np.cos(step * 0.1)  # Oscillating coefficient

            metrics = tracker.update(step, kl, kl_coef)

            assert metrics.current_kl == kl
            assert metrics.current_kl_coef == kl_coef
            assert metrics.kl_target == 0.1

        # Should have enough data for analysis
        assert len(tracker.kl_history) == 20
        assert len(tracker.metrics_history) == 20

    def test_kl_schedule_anomaly_detection(self):
        """Test KL schedule anomaly detection."""
        tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)

        # Create data with KL spike
        for step in range(50):
            if step < 10:
                kl = 0.1  # Normal KL
                kl_coef = 1.0
            else:
                kl = 0.5  # KL spike
                kl_coef = 1.0

            tracker.update(step, kl, kl_coef)

        anomalies = tracker.get_anomalies()

        # Should detect at least one KL-related anomaly
        anomaly_types = {a["type"] for a in anomalies}
        assert any(t in anomaly_types for t in {"kl_trend_anomaly", "target_range_anomaly", "kl_drift_detected"})

    def test_kl_schedule_controller_analysis(self):
        """Test KL controller performance analysis."""
        tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)

        # Create data with poor controller performance
        for step in range(50):
            kl = 0.25 if step % 2 == 0 else 0.0  # Oscillating well outside target
            kl_coef = 1.0  # Controller not responding

            tracker.update(step, kl, kl_coef)

        summary = tracker.get_summary()

        # Should have poor controller performance
        assert summary["controller_responsiveness"] < 0.5
        assert summary["time_in_target_range"] < 0.5

    def test_kl_drift_detection(self):
        """KL drift tracker should raise anomalies when divergence grows."""
        tracker = KLScheduleTracker(
            kl_target=0.1,
            kl_target_tolerance=0.05,
            drift_threshold=0.02,
            drift_window_size=20,
            reference_period=20,
        )

        # Reference period with stable KL
        for step in range(20):
            tracker.update(step, 0.1 + 0.005 * np.sin(step), 1.0)

        # Introduce sustained drift
        for step in range(20, 60):
            tracker.update(step, 0.4 + 0.01 * np.sin(step), 1.2)

        summary = tracker.get_summary()
        assert summary["kl_drift_score"] > 0.0
        assert summary["kl_drift_detected"] is True
        drift_anomalies = [
            anomaly for anomaly in summary["anomalies"] if anomaly["type"].startswith("kl_drift")
        ]
        assert drift_anomalies


class TestGradientNormsAnalyzer:
    """Test gradient norms analysis functionality."""

    def test_gradient_norms_analyzer_initialization(self):
        """Test gradient norms analyzer initialization."""
        analyzer = GradientNormsAnalyzer(
            exploding_threshold=10.0,
            vanishing_threshold=0.001,
            imbalance_threshold=0.1
        )

        assert analyzer.exploding_threshold == 10.0
        assert analyzer.vanishing_threshold == 0.001
        assert analyzer.imbalance_threshold == 0.1
        assert len(analyzer.policy_grad_history) == 0

    def test_gradient_norms_analyzer_update(self):
        """Test gradient norms analyzer update functionality."""
        analyzer = GradientNormsAnalyzer()

        # Update with normal gradient norms
        for step in range(20):
            policy_norm = 0.5 + 0.1 * np.sin(step * 0.1)
            value_norm = 0.3 + 0.05 * np.cos(step * 0.1)

            metrics = analyzer.update(step, policy_norm, value_norm)

            assert metrics.policy_grad_norm == policy_norm
            assert metrics.value_grad_norm == value_norm
            assert metrics.policy_value_ratio == policy_norm / value_norm

        assert len(analyzer.policy_grad_history) == 20
        assert len(analyzer.metrics_history) == 20

    def test_gradient_norms_anomaly_detection(self):
        """Test gradient norms anomaly detection."""
        analyzer = GradientNormsAnalyzer(exploding_threshold=5.0, vanishing_threshold=0.01)

        # Create data with exploding gradients
        for step in range(20):
            if step < 10:
                policy_norm = 0.5
                value_norm = 0.3
            else:
                policy_norm = 10.0  # Exploding gradient
                value_norm = 0.3

            analyzer.update(step, policy_norm, value_norm)

        anomalies = analyzer.get_anomalies()

        # Should detect exploding gradient anomaly
        exploding_anomalies = [a for a in anomalies if a["type"] == "exploding_gradient_anomaly"]
        assert len(exploding_anomalies) > 0

    def test_gradient_norms_balance_analysis(self):
        """Test gradient balance analysis."""
        analyzer = GradientNormsAnalyzer()

        # Create data with imbalanced gradients
        for step in range(20):
            policy_norm = 5.0  # Much higher than value
            value_norm = 0.1

            analyzer.update(step, policy_norm, value_norm)

        summary = analyzer.get_summary()

        # Should have poor gradient balance
        assert summary["gradient_balance"] < 0.5
        assert summary["current_policy_value_ratio"] > 10.0


class TestAdvantageStatisticsTracker:
    """Test advantage statistics tracking functionality."""

    def test_advantage_statistics_tracker_initialization(self):
        """Test advantage statistics tracker initialization."""
        tracker = AdvantageStatisticsTracker(
            bias_threshold=0.1,
            scale_threshold=2.0
        )

        assert tracker.bias_threshold == 0.1
        assert tracker.scale_threshold == 2.0
        assert len(tracker.advantage_mean_history) == 0

    def test_advantage_statistics_tracker_update(self):
        """Test advantage statistics tracker update functionality."""
        tracker = AdvantageStatisticsTracker()

        # Update with normal advantage statistics
        for step in range(20):
            advantage_mean = 0.0 + 0.1 * np.sin(step * 0.1)
            advantage_std = 1.0 + 0.1 * np.cos(step * 0.1)
            advantage_min = advantage_mean - advantage_std
            advantage_max = advantage_mean + advantage_std

            metrics = tracker.update(
                step, advantage_mean, advantage_std,
                advantage_min, advantage_max
            )

            assert metrics.advantage_mean == advantage_mean
            assert metrics.advantage_std == advantage_std
            assert metrics.advantage_min == advantage_min
            assert metrics.advantage_max == advantage_max

        assert len(tracker.advantage_mean_history) == 20
        assert len(tracker.metrics_history) == 20

    def test_advantage_statistics_anomaly_detection(self):
        """Test advantage statistics anomaly detection."""
        tracker = AdvantageStatisticsTracker(bias_threshold=0.05)

        # Create data with high bias
        for step in range(20):
            advantage_mean = 0.2  # High bias
            advantage_std = 1.0

            tracker.update(step, advantage_mean, advantage_std)

        anomalies = tracker.get_anomalies()

        # Should detect bias anomaly
        bias_anomalies = [a for a in anomalies if a["type"] == "advantage_bias_anomaly"]
        assert len(bias_anomalies) > 0

    def test_advantage_distribution_analysis(self):
        """Test advantage distribution analysis."""
        tracker = AdvantageStatisticsTracker()

        # Create data with skewed distribution
        base_samples = np.concatenate([
            np.full(80, -0.5),
            np.full(20, 3.0),
            np.full(20, 4.0),
        ])

        for step in range(20):
            tracker.update(
                step,
                advantage_mean=0.0,
                advantage_std=1.0,
                advantage_samples=base_samples.tolist(),
            )

        summary = tracker.get_summary()

        # Should detect skewed distribution
        assert abs(summary["advantage_skewness"]) > 0.5


class TestComprehensivePPOForensics:
    """Test comprehensive PPO forensics functionality."""

    def test_comprehensive_ppo_forensics_initialization(self):
        """Test comprehensive PPO forensics initialization."""
        forensics = ComprehensivePPOForensics(
            kl_target=0.1,
            kl_target_tolerance=0.05,
            enable_kl_schedule_tracking=True,
            enable_gradient_norms_analysis=True,
            enable_advantage_statistics=True
        )

        assert forensics.kl_target == 0.1
        assert forensics.kl_target_tolerance == 0.05
        assert forensics.kl_schedule_tracker is not None
        assert forensics.gradient_norms_analyzer is not None
        assert forensics.advantage_statistics_tracker is not None

    def test_comprehensive_ppo_forensics_update(self):
        """Test comprehensive PPO forensics update functionality."""
        forensics = ComprehensivePPOForensics()

        # Update with comprehensive data
        for step in range(20):
            metrics = forensics.update(
                step=step,
                kl=0.1 + 0.02 * np.sin(step * 0.1),
                kl_coef=1.0 + 0.1 * np.cos(step * 0.1),
                entropy=2.0 - 0.1 * step,
                reward_mean=0.5 + 0.1 * step,
                reward_std=0.2,
                policy_grad_norm=0.5 + 0.1 * np.sin(step * 0.1),
                value_grad_norm=0.3 + 0.05 * np.cos(step * 0.1),
                advantage_mean=0.0 + 0.05 * np.sin(step * 0.1),
                advantage_std=1.0 + 0.1 * np.cos(step * 0.1)
            )

            assert metrics.step == step
            assert metrics.kl_schedule_metrics is not None
            assert metrics.gradient_norms_metrics is not None
            assert metrics.advantage_statistics_metrics is not None

        assert len(forensics.comprehensive_metrics_history) == 20

    def test_comprehensive_ppo_forensics_anomaly_detection(self):
        """Test comprehensive PPO forensics anomaly detection."""
        forensics = ComprehensivePPOForensics()

        # Create data with multiple anomalies
        for step in range(50):
            if step < 25:
                # Normal data
                kl = 0.1
                policy_grad_norm = 0.5
                advantage_mean = 0.0
            else:
                # Anomalous data
                kl = 0.5  # KL spike
                policy_grad_norm = 10.0  # Exploding gradient
                advantage_mean = 0.3  # High bias

            forensics.update(
                step=step,
                kl=kl,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.5,
                reward_std=0.2,
                policy_grad_norm=policy_grad_norm,
                value_grad_norm=0.3,
                advantage_mean=advantage_mean,
                advantage_std=1.0
            )

        anomalies = forensics.get_anomalies()

        # Should detect multiple types of anomalies
        anomaly_types = {a["type"] for a in anomalies}
        assert len(anomaly_types) > 1  # Should have multiple anomaly types

        # Should have anomalies from different trackers
        tracker_types = {a["tracker"] for a in anomalies}
        assert len(tracker_types) > 1  # Should have anomalies from multiple trackers

    def test_comprehensive_ppo_kl_drift_analysis(self):
        """KL drift analysis should highlight sustained divergence."""
        forensics = ComprehensivePPOForensics(
            enable_kl_drift_tracking=True,
            kl_drift_threshold=0.05,
            kl_drift_window_size=20,
            kl_drift_reference_period=20,
        )

        # Reference period
        for step in range(20):
            forensics.update(
                step=step,
                kl=0.09 + 0.002 * np.sin(step),
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.5,
                reward_std=0.2,
            )

        # Drift period
        for step in range(20, 60):
            forensics.update(
                step=step,
                kl=0.4 + 0.01 * np.sin(step),
                kl_coef=1.4,
                entropy=1.8,
                reward_mean=0.4,
                reward_std=0.25,
            )

        drift_summary = forensics.get_kl_drift_analysis()
        assert drift_summary["detected"] is True
        assert drift_summary["score"] > 0.0
        assert drift_summary["trend"] in {"increasing", "stable", "decreasing"}

    def test_comprehensive_ppo_forensics_analysis(self):
        """Test comprehensive PPO forensics analysis generation."""
        forensics = ComprehensivePPOForensics()

        # Generate some data
        for step in range(30):
            forensics.update(
                step=step,
                kl=0.1 + 0.01 * np.sin(step * 0.1),
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.5,
                reward_std=0.2,
                policy_grad_norm=0.5,
                value_grad_norm=0.3,
                advantage_mean=0.0,
                advantage_std=1.0
            )

        analysis = forensics.get_comprehensive_analysis()

        assert analysis["version"] == "2.0"
        assert analysis["total_steps"] == 30
        assert "trackers" in analysis
        assert "kl_schedule" in analysis["trackers"]
        assert "gradient_norms" in analysis["trackers"]
        assert "advantage_statistics" in analysis["trackers"]
        assert "anomalies" in analysis

    def test_comprehensive_ppo_forensics_health_summary(self):
        """Test comprehensive PPO forensics health summary."""
        forensics = ComprehensivePPOForensics()

        # Generate healthy data
        for step in range(20):
            forensics.update(
                step=step,
                kl=0.1,  # On target
                kl_coef=1.0 + 0.05 * np.sin(step),
                entropy=2.0,
                reward_mean=0.5,
                reward_std=0.2,
                policy_grad_norm=0.5,  # Healthy gradient
                value_grad_norm=0.3,
                advantage_mean=0.0,  # No bias
                advantage_std=1.0
            )

        health_summary = forensics.get_health_summary()

        assert health_summary["status"] in {"healthy", "warning"}
        assert health_summary["overall_health_score"] > 0.7
        assert health_summary["total_anomalies"] <= 1

    def test_comprehensive_ppo_forensics_save_analysis(self):
        """Test comprehensive PPO forensics save functionality."""
        forensics = ComprehensivePPOForensics()

        # Generate some data
        for step in range(10):
            forensics.update(
                step=step,
                kl=0.1,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.5,
                reward_std=0.2,
                policy_grad_norm=0.5,
                value_grad_norm=0.3,
                advantage_mean=0.0,
                advantage_std=1.0
            )

        # Save analysis to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            forensics.save_analysis(temp_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path) as f:
                analysis = json.load(f)

            assert analysis["version"] == "2.0"
            assert analysis["total_steps"] == 10

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestComprehensivePPOForensicsIntegration:
    """Test integration with existing PPO scan functionality."""

    def test_scan_ppo_events_comprehensive(self):
        """Test comprehensive PPO scan integration."""
        forensics = ComprehensivePPOForensics()

        # Create sample events
        events = []
        for step in range(20):
            event = {
                "step": step,
                "kl": 0.1 + 0.02 * np.sin(step * 0.1),
                "kl_coef": 1.0,
                "entropy": 2.0,
                "advantage_mean": 0.0,
                "advantage_std": 1.0,
                "grad_norm_policy": 0.5,
                "grad_norm_value": 0.3,
                "reward_mean": 0.5,
                "reward_std": 0.2,
            }
            events.append(event)

        # Run comprehensive scan
        result = forensics.scan_ppo_events_comprehensive(iter(events))

        # Should have original scan results plus comprehensive analysis
        assert "version" in result
        assert "rules_fired" in result
        assert "comprehensive_analysis" in result
        assert "enhanced_version" in result
        assert result["enhanced_version"] == "2.0"

        # Comprehensive analysis should have all trackers
        comp_analysis = result["comprehensive_analysis"]
        assert "trackers" in comp_analysis
        assert "kl_schedule" in comp_analysis["trackers"]
        assert "gradient_norms" in comp_analysis["trackers"]
        assert "advantage_statistics" in comp_analysis["trackers"]


if __name__ == "__main__":
    pytest.main([__file__])
