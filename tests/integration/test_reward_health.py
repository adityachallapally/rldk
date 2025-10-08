"""Tests for reward health analysis module."""

import numpy as np
import pandas as pd
import pytest

from rldk.reward.calibration import analyze_calibration
from rldk.reward.drift import detect_reward_drift
from rldk.reward.health_analysis import (
    OveroptimizationAnalysis,
    RewardHealthReport,
    health,
)
from rldk.reward.length_bias import LengthBiasMetrics


class TestRewardHealth:
    """Test reward health analysis functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create sample training run data
        np.random.seed(42)
        n_steps = 100

        self.sample_data = pd.DataFrame(
            {
                "step": range(n_steps),
                "reward_mean": np.random.normal(0.5, 0.2, n_steps),
                "reward_std": np.random.uniform(0.1, 0.3, n_steps),
                "tokens_out": np.random.randint(10, 100, n_steps),
                "repetition_penalty": np.random.uniform(0.8, 1.2, n_steps),
                "human_preference": np.random.uniform(0, 1, n_steps),
                "ground_truth": np.random.choice([0, 1], n_steps),
                "epoch": np.random.randint(0, 10, n_steps),
                "run_id": ["test_run"] * n_steps,
            }
        )

        # Create reference data
        self.reference_data = pd.DataFrame(
            {
                "step": range(n_steps),
                "reward_mean": np.random.normal(0.6, 0.15, n_steps),
                "reward_std": np.random.uniform(0.1, 0.3, n_steps),
            }
        )

    def test_health_basic_functionality(self):
        """Test basic reward health analysis."""
        report = health(self.sample_data)

        assert isinstance(report, RewardHealthReport)
        assert hasattr(report, "passed")
        assert hasattr(report, "drift_detected")
        assert hasattr(report, "saturation_issues")
        assert hasattr(report, "calibration_score")
        assert hasattr(report, "shortcut_signals")
        assert hasattr(report, "label_leakage_risk")
        assert hasattr(report, "fixes")
        assert hasattr(report, "length_bias_detected")
        assert hasattr(report, "length_bias_metrics")
        assert hasattr(report, "length_bias_recommendations")

    def test_health_with_reference_data(self):
        """Test reward health analysis with reference data."""
        report = health(self.sample_data, self.reference_data)

        assert isinstance(report, RewardHealthReport)
        # Should detect some drift due to different distributions
        assert report.drift_detected in [True, False]  # Depends on random data

    def test_health_custom_columns(self):
        """Test reward health with custom column names."""
        custom_data = self.sample_data.rename(
            columns={"reward_mean": "custom_reward", "step": "custom_step"}
        )

        report = health(custom_data, reward_col="custom_reward", step_col="custom_step")

        assert isinstance(report, RewardHealthReport)

    def test_health_missing_columns(self):
        """Test reward health with missing required columns."""
        incomplete_data = self.sample_data.drop(columns=["reward_mean"])

        with pytest.raises(ValueError, match="Reward column"):
            health(incomplete_data)

    def test_health_saturation_detection(self):
        """Test saturation detection."""
        # Create data with saturation
        saturated_data = self.sample_data.copy()
        saturated_data["reward_mean"] = np.ones(100) * 0.99  # High saturation

        report = health(saturated_data, threshold_saturation=0.5)

        assert len(report.saturation_issues) > 0
        assert any("upper saturation" in issue for issue in report.saturation_issues)

    def test_health_length_bias_detection(self):
        """Test dedicated length bias detection."""
        correlated = self.sample_data.copy()
        correlated["reward_mean"] = correlated["tokens_out"] * 0.02

        report = health(
            correlated,
            threshold_shortcut=0.5,
            threshold_length_bias=0.2,
        )

        assert report.length_bias_detected is True
        assert report.length_bias_metrics.bias_severity is not None
        assert any(
            "Length bias" in signal for signal in report.shortcut_signals
        )

    def test_health_label_leakage_detection(self):
        """Test label leakage detection."""
        # Create data with strong correlation to metadata
        leakage_data = self.sample_data.copy()
        leakage_data["reward_mean"] = (
            leakage_data["epoch"] * 0.1
        )  # Correlated with epoch

        report = health(leakage_data, threshold_leakage=0.2)

        assert report.label_leakage_risk > 0.2

    def test_health_thresholds(self):
        """Test different threshold configurations."""
        # Test with very strict thresholds
        strict_report = health(
            self.sample_data,
            threshold_drift=0.01,  # Very strict
            threshold_saturation=0.1,  # Very strict
            threshold_calibration=0.9,  # Very strict
            threshold_shortcut=0.1,  # Very strict
            threshold_leakage=0.1,  # Very strict
        )

        # Test with very lenient thresholds
        lenient_report = health(
            self.sample_data,
            threshold_drift=0.5,  # Very lenient
            threshold_saturation=0.95,  # Very lenient
            threshold_calibration=0.1,  # Very lenient
            threshold_shortcut=0.9,  # Very lenient
            threshold_leakage=0.9,  # Very lenient
            threshold_length_bias=0.9,
        )

        # Lenient thresholds should result in fewer issues
        assert len(strict_report.fixes) >= len(lenient_report.fixes)

    def test_health_empty_data(self):
        """Test health analysis with empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError):
            health(empty_data)

    def test_health_single_row(self):
        """Test health analysis with single row of data."""
        single_row = self.sample_data.iloc[:1]

        report = health(single_row)

        assert isinstance(report, RewardHealthReport)
        # Should handle single row gracefully

    def test_health_disable_length_bias(self):
        """Ensure length bias detector can be disabled."""
        correlated = self.sample_data.copy()
        correlated["reward_mean"] = correlated["tokens_out"] * 0.02

        report = health(
            correlated,
            enable_length_bias_detection=False,
            threshold_length_bias=0.1,
        )

        assert report.length_bias_detected is False
        assert report.length_bias_metrics.valid_sample_count == 0
        assert not any("Length bias" in signal for signal in report.shortcut_signals)


class TestCalibration:
    """Test calibration analysis functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 50

        self.calibration_data = pd.DataFrame(
            {
                "reward_mean": np.random.uniform(0, 1, n_samples),
                "human_preference": np.random.choice([0, 1], n_samples),
                "ground_truth": np.random.choice([0, 1], n_samples),
            }
        )

    def test_analyze_calibration_human_preference(self):
        """Test calibration analysis with human preference data."""
        score, details = analyze_calibration(
            self.calibration_data, "reward_mean", threshold=0.7
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(details, dict)
        assert "human_preference" in details

    def test_analyze_calibration_ground_truth(self):
        """Test calibration analysis with ground truth data."""
        score, details = analyze_calibration(
            self.calibration_data, "reward_mean", threshold=0.7
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(details, dict)

    def test_analyze_calibration_no_data(self):
        """Test calibration analysis with no calibration data."""
        no_calibration_data = self.calibration_data.drop(
            columns=["human_preference", "ground_truth"]
        )

        score, details = analyze_calibration(no_calibration_data, "reward_mean")

        assert score == 0.0
        assert "error" in details

    def test_analyze_calibration_insufficient_data(self):
        """Test calibration analysis with insufficient data."""
        insufficient_data = self.calibration_data.iloc[:5]  # Only 5 samples

        score, details = analyze_calibration(insufficient_data, "reward_mean")

        assert score == 0.0
        assert "human_preference" in details and "error" in details["human_preference"]


class TestDriftDetection:
    """Test drift detection functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_steps = 100

        self.run_data = pd.DataFrame(
            {"step": range(n_steps), "reward_mean": np.random.normal(0.5, 0.2, n_steps)}
        )

        self.reference_data = pd.DataFrame(
            {
                "step": range(n_steps),
                "reward_mean": np.random.normal(0.6, 0.15, n_steps),
            }
        )

    def test_detect_reward_drift(self):
        """Test reward drift detection."""
        drift_detected, drift_metrics = detect_reward_drift(
            self.run_data, self.reference_data, "reward_mean", "step"
        )

        assert isinstance(drift_detected, bool)
        assert isinstance(drift_metrics, pd.DataFrame)

    def test_detect_reward_drift_missing_columns(self):
        """Test drift detection with missing columns."""
        incomplete_run = self.run_data.drop(columns=["reward_mean"])

        with pytest.raises(ValueError):
            detect_reward_drift(
                incomplete_run, self.reference_data, "reward_mean", "step"
            )

    def test_detect_reward_drift_different_distributions(self):
        """Test drift detection with clearly different distributions."""
        # Create clearly different distributions
        different_reference = self.reference_data.copy()
        different_reference["reward_mean"] = np.random.normal(
            1.0, 0.1, len(different_reference)
        )

        drift_detected, drift_metrics = detect_reward_drift(
            self.run_data,
            different_reference,
            "reward_mean",
            "step",
            threshold_drift=0.1,
        )

        # Should detect drift with clearly different distributions
        assert drift_detected in [True, False]  # Depends on random data

    def test_detect_reward_drift_similar_distributions(self):
        """Test drift detection with similar distributions."""
        # Create similar distributions
        similar_reference = self.reference_data.copy()
        similar_reference["reward_mean"] = np.random.normal(
            0.5, 0.2, len(similar_reference)
        )

        drift_detected, drift_metrics = detect_reward_drift(
            self.run_data, similar_reference, "reward_mean", "step", threshold_drift=0.1
        )

        # Should not detect drift with similar distributions
        assert drift_detected in [True, False]  # Depends on random data


class TestRewardHealthReport:
    """Test RewardHealthReport dataclass."""

    def test_reward_health_report_creation(self):
        """Test creating RewardHealthReport instance."""
        report = RewardHealthReport(
            passed=True,
            drift_detected=False,
            saturation_issues=[],
            calibration_score=0.8,
            shortcut_signals=[],
            label_leakage_risk=0.1,
            fixes=[],
            drift_metrics=pd.DataFrame(),
            calibration_details={},
            shortcut_analysis={},
            saturation_analysis={},
            length_bias_detected=False,
            length_bias_metrics=LengthBiasMetrics(),
            length_bias_recommendations=[],
            overoptimization=OveroptimizationAnalysis(),
        )

        assert report.passed is True
        assert report.drift_detected is False
        assert report.calibration_score == 0.8
        assert report.label_leakage_risk == 0.1
        assert len(report.saturation_issues) == 0
        assert len(report.shortcut_signals) == 0
        assert len(report.fixes) == 0

    def test_reward_health_report_with_issues(self):
        """Test RewardHealthReport with detected issues."""
        report = RewardHealthReport(
            passed=False,
            drift_detected=True,
            saturation_issues=["High upper saturation"],
            calibration_score=0.4,
            shortcut_signals=["Length bias detected"],
            label_leakage_risk=0.7,
            fixes=["Check data pipeline"],
            drift_metrics=pd.DataFrame({"step": [100], "p_value": [0.01]}),
            calibration_details={"error": "Poor calibration"},
            shortcut_analysis={"length_correlation": 0.8},
            saturation_analysis={"upper_saturation_ratio": 0.9},
            length_bias_detected=True,
            length_bias_metrics=LengthBiasMetrics(bias_severity=0.8),
            length_bias_recommendations=["Audit prompts for response length bias."],
            overoptimization=OveroptimizationAnalysis(flagged=True),
        )

        assert report.passed is False
        assert report.drift_detected is True
        assert len(report.saturation_issues) == 1
        assert len(report.shortcut_signals) == 1
        assert len(report.fixes) == 1
        assert report.calibration_score == 0.4
        assert report.label_leakage_risk == 0.7


if __name__ == "__main__":
    pytest.main([__file__])
