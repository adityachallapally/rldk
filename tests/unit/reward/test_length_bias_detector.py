"""Unit tests for the length bias detector."""

import numpy as np
import pytest

from src.rldk.reward.length_bias import LengthBiasDetector


class TestLengthBiasDetector:
    def test_positive_correlation(self) -> None:
        detector = LengthBiasDetector()
        responses = ["a" * i for i in range(5, 15)]
        rewards = [float(i) for i in range(10)]
        lengths = list(range(5, 15))

        metrics = detector.analyze_length_bias(responses, rewards, lengths)

        assert metrics.pearson_correlation is not None
        assert metrics.pearson_correlation > 0.95
        assert metrics.spearman_correlation is not None
        assert metrics.spearman_correlation > 0.95
        assert any(
            pattern.startswith("Longer responses") for pattern in metrics.optimization_patterns
        )

    def test_negative_correlation(self) -> None:
        detector = LengthBiasDetector()
        responses = ["b" * i for i in range(5, 15)]
        rewards = [float(20 - i) for i in range(10)]
        lengths = list(range(5, 15))

        metrics = detector.analyze_length_bias(responses, rewards, lengths)

        assert metrics.pearson_correlation is not None
        assert metrics.pearson_correlation < -0.95
        assert metrics.spearman_correlation is not None
        assert metrics.spearman_correlation < -0.95

    def test_zero_correlation_with_constant_rewards(self) -> None:
        detector = LengthBiasDetector()
        responses = ["c" * i for i in range(5, 15)]
        rewards = [1.0] * 10
        lengths = list(range(5, 15))

        metrics = detector.analyze_length_bias(responses, rewards, lengths)

        assert metrics.pearson_correlation is None
        assert metrics.spearman_correlation is None

    def test_quartile_metrics_and_variance(self) -> None:
        detector = LengthBiasDetector()
        responses = [f"resp-{i}" for i in range(16)]
        lengths = np.linspace(10, 100, num=16)
        rewards = np.linspace(0.0, 1.5, num=16)

        metrics = detector.analyze_length_bias(responses, rewards, lengths)

        assert set(metrics.quartile_metrics.keys()) == {"q1", "q2", "q3", "q4"}
        assert metrics.quartile_metrics["q4"]["mean_reward"] > metrics.quartile_metrics["q1"]["mean_reward"]
        assert metrics.variance_explained is not None
        assert metrics.variance_explained > 0

    def test_odin_metrics(self) -> None:
        detector = LengthBiasDetector()
        lengths = np.array([10.0, 20.0, 30.0])
        rewards = np.array([1.0, 2.0, 3.0])

        odin = detector.calculate_odin_metrics(lengths, rewards)
        assert pytest.approx(odin["reward_per_token"], rel=1e-6) == 0.1
        assert pytest.approx(odin["efficiency"], rel=1e-6) == 0.1
        assert odin["optimization_flag"] is True

    def test_empty_inputs(self) -> None:
        detector = LengthBiasDetector()
        metrics = detector.analyze_length_bias([], [])
        assert metrics.valid_sample_count == 0
        assert metrics.recommendations

    def test_length_extraction_without_tokenizer(self) -> None:
        detector = LengthBiasDetector()
        responses = ["short", "a much longer response"]
        rewards = [0.0, 1.0]

        metrics = detector.analyze_length_bias(responses, rewards)
        assert metrics.mean_length is not None
        assert metrics.pearson_correlation is not None

    def test_graceful_degradation_without_scipy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        detector = LengthBiasDetector()
        responses = ["x" * i for i in range(5, 15)]
        rewards = [float(i) for i in range(10)]
        lengths = list(range(5, 15))

        from src.rldk.reward import length_bias

        monkeypatch.setattr(length_bias, "pearsonr", None)
        monkeypatch.setattr(length_bias, "spearmanr", None)

        metrics = detector.analyze_length_bias(responses, rewards, lengths)
        assert metrics.pearson_correlation is not None
        assert metrics.pearson_correlation > 0.9
