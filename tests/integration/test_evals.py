"""Tests for evaluation module."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rldk.evals.probes import (
    evaluate_alignment,
    evaluate_hallucination,
    evaluate_harmlessness,
    evaluate_helpfulness,
    evaluate_kl_divergence,
    evaluate_reward_alignment,
)
from rldk.evals.runner import EvalResult, compare_evaluations, run, save_eval_results
from rldk.evals.suites import get_eval_suite, get_suite_info, list_available_suites


class TestEvaluationRunner:
    """Test evaluation runner functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100

        # Generate mock event logs for throughput evaluation
        def generate_mock_events():
            events = []
            base_time = 1000  # Base timestamp in milliseconds
            for i in range(np.random.randint(5, 15)):  # 5-15 events per sample
                event_type = np.random.choice([
                    "token_generated", "generation_complete", "inference_step",
                    "batch_complete", "training_step"
                ])

                if event_type in ["token_generated", "generation_complete", "inference_step"]:
                    events.append({
                        "event_type": event_type,
                        "timestamp": base_time + i * 100 + np.random.randint(0, 50),
                        "token_count": np.random.randint(10, 100)
                    })
                else:  # batch_complete, training_step
                    events.append({
                        "event_type": event_type,
                        "timestamp": base_time + i * 100 + np.random.randint(0, 50),
                        "batch_size": np.random.randint(8, 32),
                        "processing_time": np.random.uniform(0.5, 3.0)
                    })
            return events

        # Generate mock model outputs for toxicity/bias evaluation
        def generate_mock_output():
            outputs = [
                "This is a helpful and informative response about machine learning.",
                "The weather today is sunny and pleasant.",
                "I enjoy reading books and learning new things.",
                "Women are intelligent and capable leaders in their fields.",
                "Men are strong and confident in their abilities.",
                "The woman is a brilliant scientist who made important discoveries.",
                "The man is a caring teacher who helps students learn.",
                "This is educational material about artificial intelligence.",
                "Reading is a great hobby that expands your knowledge.",
                "The person is talented regardless of their background."
            ]
            return np.random.choice(outputs)

        self.sample_data = pd.DataFrame(
            {
                "step": range(n_samples),
                "reward_mean": np.random.normal(0.5, 0.2, n_samples),
                "reward_std": np.random.uniform(0.1, 0.3, n_samples),
                "human_preference": np.random.uniform(0, 1, n_samples),
                "ground_truth": np.random.choice([0, 1], n_samples),
                "tokens_out": np.random.randint(10, 100, n_samples),
                "response_quality": np.random.uniform(0.6, 0.9, n_samples),
                "relevance_score": np.random.uniform(0.5, 0.8, n_samples),
                "safety_score": np.random.uniform(0.7, 0.95, n_samples),
                "factual_accuracy": np.random.uniform(0.6, 0.9, n_samples),
                # Add required columns for comprehensive evaluation
                "events": [json.dumps(generate_mock_events()) for _ in range(n_samples)],
                "output": [generate_mock_output() for _ in range(n_samples)],
                "input": [f"Test input {i}: What is machine learning?" for i in range(n_samples)],
                "model_name": ["test-model"] * n_samples,
                # Add additional metrics for other evaluations
                "confidence_score": np.random.uniform(0.3, 0.9, n_samples),
                "uncertainty_score": np.random.uniform(0.1, 0.5, n_samples),
                "entropy_mean": np.random.uniform(0.5, 2.0, n_samples),
                "kl_mean": np.random.uniform(0.01, 0.5, n_samples),
                "accuracy": np.random.uniform(0.6, 0.95, n_samples),
                "loss": np.random.uniform(0.1, 2.0, n_samples),
                "training_time": np.cumsum(np.random.uniform(0.1, 1.0, n_samples)),
                "memory_usage": np.random.uniform(4.0, 16.0, n_samples),
                "gpu_memory": np.random.uniform(2.0, 12.0, n_samples),
                "inference_time": np.random.uniform(0.05, 0.5, n_samples),
                "throughput": np.random.uniform(100, 1000, n_samples),
                "latency": np.random.uniform(0.01, 0.2, n_samples),
                "batch_time": np.random.uniform(0.1, 2.0, n_samples),
                "batch_size": np.random.randint(8, 32, n_samples),
                "gradient_norm": np.random.uniform(0.1, 5.0, n_samples),
                "flops": np.random.uniform(100, 5000, n_samples),
                "adversarial_score": np.random.uniform(0.3, 0.9, n_samples),
                "noise_score": np.random.uniform(0.4, 0.8, n_samples),
                "perturbation_score": np.random.uniform(0.3, 0.8, n_samples),
                "attack_success_rate": np.random.uniform(0.1, 0.4, n_samples),
                "input_sensitivity": np.random.uniform(0.1, 0.6, n_samples),
                "memory_fragmentation": np.random.uniform(0.0, 0.3, n_samples),
                "cache_hit_rate": np.random.uniform(0.6, 0.95, n_samples),
                "memory_bandwidth": np.random.uniform(0.2, 0.8, n_samples),
                "memory_allocation_count": np.random.randint(100, 5000, n_samples),
                "memory_errors": np.random.randint(0, 3, n_samples),
                "gpu_utilization": np.random.uniform(0.3, 0.9, n_samples),
                "memory_access_time": np.random.uniform(0.0001, 0.01, n_samples),
                "temperature": np.random.uniform(0.8, 1.2, n_samples),
                "reliability_score": np.random.uniform(0.6, 0.9, n_samples),
                "expected_calibration_error": np.random.uniform(0.01, 0.1, n_samples),
            }
        )

        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_basic_functionality(self):
        """Test basic evaluation run."""
        result = run(self.sample_data, suite="quick", seed=42)

        assert isinstance(result, EvalResult)
        assert result.suite_name == "quick"
        assert result.seed == 42
        assert result.sample_size == 50  # Default for quick suite
        assert isinstance(result.scores, dict)
        assert isinstance(result.confidence_intervals, dict)
        assert isinstance(result.effect_sizes, dict)
        assert isinstance(result.raw_results, list)

    def test_run_with_custom_sample_size(self):
        """Test evaluation run with custom sample size."""
        result = run(self.sample_data, suite="quick", seed=42, sample_size=75)

        assert result.sample_size == 75
        assert len(result.raw_results) > 0

    def test_run_with_output_dir(self):
        """Test evaluation run with output directory."""
        output_dir = Path(self.temp_dir) / "eval_output"

        run(self.sample_data, suite="quick", seed=42, output_dir=output_dir)

        assert output_dir.exists()
        assert (output_dir / "eval_card.md").exists()
        assert (output_dir / "eval_results.jsonl").exists()
        assert (output_dir / "eval_summary.json").exists()

    def test_run_unknown_suite(self):
        """Test evaluation run with unknown suite."""
        with pytest.raises(ValueError, match="Unknown evaluation suite"):
            run(self.sample_data, suite="unknown_suite")

    def test_run_empty_data(self):
        """Test evaluation run with empty data."""
        empty_data = pd.DataFrame()

        result = run(empty_data, suite="quick", seed=42)

        assert result.sample_size == 0
        # For empty data, we should have scores but they should be NaN
        assert len(result.scores) > 0
        assert all(pd.isna(score) for score in result.scores.values())

    def test_run_single_row(self):
        """Test evaluation run with single row."""
        single_row = self.sample_data.iloc[:1]

        result = run(single_row, suite="quick", seed=42)

        assert result.sample_size == 1
        assert len(result.raw_results) > 0


class TestEvaluationSuites:
    """Test evaluation suite functionality."""

    def test_get_eval_suite_quick(self):
        """Test getting quick evaluation suite."""
        suite = get_eval_suite("quick")

        assert suite is not None
        assert suite["name"] == "quick"
        assert "evaluations" in suite
        assert "baseline_scores" in suite
        assert suite["default_sample_size"] == 50

    def test_get_eval_suite_comprehensive(self):
        """Test getting comprehensive evaluation suite."""
        suite = get_eval_suite("comprehensive")

        assert suite is not None
        assert suite["name"] == "comprehensive"
        assert suite["default_sample_size"] == 200
        assert len(suite["evaluations"]) > len(get_eval_suite("quick")["evaluations"])

    def test_get_eval_suite_unknown(self):
        """Test getting unknown evaluation suite."""
        suite = get_eval_suite("unknown")

        assert suite is None

    def test_list_available_suites(self):
        """Test listing available evaluation suites."""
        suites = list_available_suites()

        assert isinstance(suites, dict)
        assert "quick" in suites
        assert "comprehensive" in suites
        assert "safety" in suites
        assert "performance" in suites

    def test_get_suite_info(self):
        """Test getting suite information."""
        suite_info = get_suite_info("quick")

        assert suite_info is not None
        assert suite_info["name"] == "quick"
        assert "evaluations" in suite_info
        assert "baseline_scores" in suite_info
        assert suite_info["generates_plots"] is True

    def test_kl_divergence_in_suites(self):
        """Test that KL divergence is included in evaluation suites."""
        # Check that KL divergence is in the quick suite
        quick_suite = get_eval_suite("quick")
        assert quick_suite is not None
        assert "kl_divergence" in quick_suite["evaluations"]
        assert "kl_divergence" in quick_suite["baseline_scores"]

        # Check that KL divergence is in the comprehensive suite
        comprehensive_suite = get_eval_suite("comprehensive")
        assert comprehensive_suite is not None
        assert "kl_divergence" in comprehensive_suite["evaluations"]
        assert "kl_divergence" in comprehensive_suite["baseline_scores"]

        # Check that KL divergence is in the safety suite
        safety_suite = get_eval_suite("safety")
        assert safety_suite is not None
        assert "kl_divergence" in safety_suite["evaluations"]
        assert "kl_divergence" in safety_suite["baseline_scores"]

        # Check that KL divergence is in the performance suite
        performance_suite = get_eval_suite("performance")
        assert performance_suite is not None
        assert "kl_divergence" in performance_suite["evaluations"]
        assert "kl_divergence" in performance_suite["baseline_scores"]

        # Check that KL divergence is in the trust suite
        trust_suite = get_eval_suite("trust")
        assert trust_suite is not None
        assert "kl_divergence" in trust_suite["evaluations"]
        assert "kl_divergence" in trust_suite["baseline_scores"]

    def test_comprehensive_evaluation_with_complete_data(self):
        """Test that all evaluation metrics work with complete data including events and output columns."""
        # Test throughput evaluation with events column
        from rldk.evals.metrics.throughput import evaluate_throughput
        throughput_result = evaluate_throughput(self.sample_data)

        assert "score" in throughput_result
        assert "details" in throughput_result
        assert "method" in throughput_result
        assert throughput_result["method"] == "event_log_analysis"
        assert throughput_result["num_samples"] > 0
        assert 0 <= throughput_result["score"] <= 1

        # Test toxicity evaluation with output column
        from rldk.evals.metrics.toxicity import evaluate_toxicity
        toxicity_result = evaluate_toxicity(self.sample_data)

        assert "score" in toxicity_result
        assert "details" in toxicity_result
        assert "method" in toxicity_result
        assert toxicity_result["method"] == "content_analysis"
        assert toxicity_result["num_samples"] > 0
        assert 0 <= toxicity_result["score"] <= 1

        # Test bias evaluation with output column
        from rldk.evals.metrics.bias import evaluate_bias
        bias_result = evaluate_bias(self.sample_data)

        assert "score" in bias_result
        assert "details" in bias_result
        assert "method" in bias_result
        assert bias_result["method"] == "demographic_analysis"
        assert bias_result["num_samples"] > 0
        assert 0 <= bias_result["score"] <= 1

    def test_all_suite_evaluations_with_complete_data(self):
        """Test that all evaluation suites work with complete data."""
        # Test quick suite
        quick_result = run(self.sample_data, suite="quick", seed=42)
        assert isinstance(quick_result, EvalResult)
        assert quick_result.sample_size > 0
        assert len(quick_result.scores) > 0

        # Verify that throughput, toxicity, bias, and catastrophic forgetting evaluations succeeded
        assert "throughput" in quick_result.scores
        assert "toxicity" in quick_result.scores
        assert "bias" in quick_result.scores
        assert "catastrophic_forgetting" in quick_result.scores

        # Test comprehensive suite
        comprehensive_result = run(self.sample_data, suite="comprehensive", seed=42)
        assert isinstance(comprehensive_result, EvalResult)
        assert comprehensive_result.sample_size > 0
        assert len(comprehensive_result.scores) > 0

        # Verify that all metrics are present
        expected_metrics = [
            "throughput",
            "toxicity",
            "bias",
            "alignment",
            "helpfulness",
            "harmlessness",
            "hallucination",
            "reward_alignment",
            "kl_divergence",
            "catastrophic_forgetting",
        ]
        for metric in expected_metrics:
            assert metric in comprehensive_result.scores

    def test_missing_columns_handling(self):
        """Test that evaluations handle missing columns gracefully."""
        # Test with minimal data (missing events and output columns)
        minimal_data = pd.DataFrame({
            "step": range(10),
            "reward_mean": np.random.normal(0.5, 0.2, 10),
        })

        # Throughput should handle missing events column
        from rldk.evals.metrics.throughput import evaluate_throughput
        throughput_result = evaluate_throughput(minimal_data)
        assert throughput_result["score"] == 0.0
        assert "missing_log_column" in throughput_result["error"]

        # Toxicity should handle missing output column
        from rldk.evals.metrics.toxicity import evaluate_toxicity
        toxicity_result = evaluate_toxicity(minimal_data)
        assert toxicity_result["score"] == 1.0  # High score = high toxicity (bad)
        assert "missing_output_column" in toxicity_result["error"]

        # Bias should handle missing output column
        from rldk.evals.metrics.bias import evaluate_bias
        bias_result = evaluate_bias(minimal_data)
        assert bias_result["score"] == 1.0  # High score = high bias (bad)
        assert "missing_output_column" in bias_result["error"]


class TestEvaluationProbes:
    """Test evaluation probe functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 50

        # Generate mock event logs for throughput evaluation
        def generate_mock_events():
            events = []
            base_time = 1000  # Base timestamp in milliseconds
            for i in range(np.random.randint(5, 15)):  # 5-15 events per sample
                event_type = np.random.choice([
                    "token_generated", "generation_complete", "inference_step",
                    "batch_complete", "training_step"
                ])

                if event_type in ["token_generated", "generation_complete", "inference_step"]:
                    events.append({
                        "event_type": event_type,
                        "timestamp": base_time + i * 100 + np.random.randint(0, 50),
                        "token_count": np.random.randint(10, 100)
                    })
                else:  # batch_complete, training_step
                    events.append({
                        "event_type": event_type,
                        "timestamp": base_time + i * 100 + np.random.randint(0, 50),
                        "batch_size": np.random.randint(8, 32),
                        "processing_time": np.random.uniform(0.5, 3.0)
                    })
            return events

        # Generate mock model outputs for toxicity/bias evaluation
        def generate_mock_output():
            outputs = [
                "This is a helpful and informative response about machine learning.",
                "The weather today is sunny and pleasant.",
                "I enjoy reading books and learning new things.",
                "Women are intelligent and capable leaders in their fields.",
                "Men are strong and confident in their abilities.",
                "The woman is a brilliant scientist who made important discoveries.",
                "The man is a caring teacher who helps students learn.",
                "This is educational material about artificial intelligence.",
                "Reading is a great hobby that expands your knowledge.",
                "The person is talented regardless of their background."
            ]
            return np.random.choice(outputs)

        self.probe_data = pd.DataFrame(
            {
                "step": range(n_samples),
                "reward_mean": np.random.normal(0.5, 0.2, n_samples),
                "human_preference": np.random.uniform(0, 1, n_samples),
                "ground_truth": np.random.choice([0, 1], n_samples),
                "tokens_out": np.random.randint(10, 100, n_samples),
                "response_quality": np.random.uniform(0.6, 0.9, n_samples),
                "safety_score": np.random.uniform(0.7, 0.95, n_samples),
                "factual_accuracy": np.random.uniform(0.6, 0.9, n_samples),
                # Add required columns for comprehensive evaluation
                "events": [json.dumps(generate_mock_events()) for _ in range(n_samples)],
                "output": [generate_mock_output() for _ in range(n_samples)],
                "input": [f"Test input {i}: What is machine learning?" for i in range(n_samples)],
                "model_name": ["test-model"] * n_samples,
            }
        )

    def test_evaluate_alignment(self):
        """Test alignment evaluation."""
        result = evaluate_alignment(self.probe_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert 0 <= result["score"] <= 1

    def test_evaluate_helpfulness(self):
        """Test helpfulness evaluation."""
        result = evaluate_helpfulness(self.probe_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert 0 <= result["score"] <= 1

    def test_evaluate_harmlessness(self):
        """Test harmlessness evaluation."""
        result = evaluate_harmlessness(self.probe_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert 0 <= result["score"] <= 1

    def test_evaluate_hallucination(self):
        """Test hallucination evaluation."""
        result = evaluate_hallucination(self.probe_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert 0 <= result["score"] <= 1
        assert "note" in result  # Should indicate lower is better

    def test_evaluate_reward_alignment(self):
        """Test reward alignment evaluation."""
        result = evaluate_reward_alignment(self.probe_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert 0 <= result["score"] <= 1

    def test_evaluate_kl_divergence(self):
        """Test KL divergence evaluation."""
        result = evaluate_kl_divergence(self.probe_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert 0 <= result["score"] <= 1

    def test_evaluate_with_missing_columns(self):
        """Test evaluation with missing columns."""
        minimal_data = pd.DataFrame({"reward_mean": np.random.normal(0.5, 0.2, 20)})

        # All evaluations should handle missing columns gracefully
        alignment_result = evaluate_alignment(minimal_data, seed=42)
        helpfulness_result = evaluate_helpfulness(minimal_data, seed=42)
        harmlessness_result = evaluate_harmlessness(minimal_data, seed=42)
        hallucination_result = evaluate_hallucination(minimal_data, seed=42)
        reward_alignment_result = evaluate_reward_alignment(minimal_data, seed=42)
        kl_divergence_result = evaluate_kl_divergence(minimal_data, seed=42)

        assert all(
            isinstance(r, dict)
            for r in [
                alignment_result,
                helpfulness_result,
                harmlessness_result,
                hallucination_result,
                reward_alignment_result,
                kl_divergence_result,
            ]
        )
        assert all(
            "score" in r
            for r in [
                alignment_result,
                helpfulness_result,
                harmlessness_result,
                hallucination_result,
                reward_alignment_result,
                kl_divergence_result,
            ]
        )


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_eval_result_creation(self):
        """Test creating EvalResult instance."""
        result = EvalResult(
            suite_name="test_suite",
            scores={"metric1": 0.8, "metric2": 0.6},
            confidence_intervals={"metric1": (0.7, 0.9), "metric2": (0.5, 0.7)},
            effect_sizes={"metric1": 0.5, "metric2": 0.3},
            sample_size=50,
            seed=42,
            metadata={"test": True},
            raw_results=[{"evaluation": "test", "result": 0.8}],
        )

        assert result.suite_name == "test_suite"
        assert result.scores["metric1"] == 0.8
        assert result.confidence_intervals["metric1"] == (0.7, 0.9)
        assert result.effect_sizes["metric1"] == 0.5
        assert result.sample_size == 50
        assert result.seed == 42
        assert result.metadata["test"] is True
        assert len(result.raw_results) == 1


class TestEvaluationOutput:
    """Test evaluation output and file generation."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 50

        # Generate mock event logs for throughput evaluation
        def generate_mock_events():
            events = []
            base_time = 1000  # Base timestamp in milliseconds
            for i in range(np.random.randint(5, 15)):  # 5-15 events per sample
                event_type = np.random.choice([
                    "token_generated", "generation_complete", "inference_step",
                    "batch_complete", "training_step"
                ])

                if event_type in ["token_generated", "generation_complete", "inference_step"]:
                    events.append({
                        "event_type": event_type,
                        "timestamp": base_time + i * 100 + np.random.randint(0, 50),
                        "token_count": np.random.randint(10, 100)
                    })
                else:  # batch_complete, training_step
                    events.append({
                        "event_type": event_type,
                        "timestamp": base_time + i * 100 + np.random.randint(0, 50),
                        "batch_size": np.random.randint(8, 32),
                        "processing_time": np.random.uniform(0.5, 3.0)
                    })
            return events

        # Generate mock model outputs for toxicity/bias evaluation
        def generate_mock_output():
            outputs = [
                "This is a helpful and informative response about machine learning.",
                "The weather today is sunny and pleasant.",
                "I enjoy reading books and learning new things.",
                "Women are intelligent and capable leaders in their fields.",
                "Men are strong and confident in their abilities.",
                "The woman is a brilliant scientist who made important discoveries.",
                "The man is a caring teacher who helps students learn.",
                "This is educational material about artificial intelligence.",
                "Reading is a great hobby that expands your knowledge.",
                "The person is talented regardless of their background."
            ]
            return np.random.choice(outputs)

        self.sample_data = pd.DataFrame(
            {
                "step": range(n_samples),
                "reward_mean": np.random.normal(0.5, 0.2, n_samples),
                "human_preference": np.random.uniform(0, 1, n_samples),
                "ground_truth": np.random.choice([0, 1], n_samples),
                # Add required columns for comprehensive evaluation
                "events": [json.dumps(generate_mock_events()) for _ in range(n_samples)],
                "output": [generate_mock_output() for _ in range(n_samples)],
                "input": [f"Test input {i}: What is machine learning?" for i in range(n_samples)],
                "model_name": ["test-model"] * n_samples,
            }
        )

        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_eval_results(self):
        """Test saving evaluation results."""
        # Create a mock evaluation result
        result = EvalResult(
            suite_name="test_suite",
            scores={"metric1": 0.8},
            confidence_intervals={"metric1": (0.7, 0.9)},
            effect_sizes={"metric1": 0.5},
            sample_size=50,
            seed=42,
            metadata={"test": True},
            raw_results=[{"evaluation": "test", "result": 0.8}],
        )

        output_dir = Path(self.temp_dir) / "eval_output"

        # Save results
        save_eval_results(result, output_dir)

        # Check that files were created
        assert output_dir.exists()
        assert (output_dir / "eval_card.md").exists()
        assert (output_dir / "eval_results.jsonl").exists()
        assert (output_dir / "eval_summary.json").exists()

        # Check markdown content
        with open(output_dir / "eval_card.md") as f:
            content = f.read()
            assert "test_suite" in content
            assert "metric1" in content
            assert "0.800" in content

    def test_compare_evaluations(self):
        """Test comparing multiple evaluation results."""
        # Create mock evaluation results
        result1 = EvalResult(
            suite_name="suite1",
            scores={"metric1": 0.8, "metric2": 0.6},
            confidence_intervals={"metric1": (0.7, 0.9)},
            effect_sizes={"metric1": 0.5},
            sample_size=50,
            seed=42,
            metadata={},
            raw_results=[],
        )

        result2 = EvalResult(
            suite_name="suite2",
            scores={"metric1": 0.9, "metric2": 0.7},
            confidence_intervals={"metric1": (0.8, 1.0)},
            effect_sizes={"metric1": 0.6},
            sample_size=50,
            seed=42,
            metadata={},
            raw_results=[],
        )

        # Compare evaluations
        comparison = compare_evaluations([result1, result2])

        assert comparison["runs_compared"] == 2
        assert "metric1" in comparison["comparisons"]
        assert "metric2" in comparison["comparisons"]

        metric1_comp = comparison["comparisons"]["metric1"]
        assert metric1_comp["mean"] == pytest.approx(0.85, abs=1e-10)
        assert metric1_comp["std"] == pytest.approx(0.05, abs=1e-4)
        assert metric1_comp["min"] == 0.8
        assert metric1_comp["max"] == 0.9

    def test_compare_evaluations_insufficient_data(self):
        """Test comparing evaluations with insufficient data."""
        with pytest.raises(ValueError, match="Need at least 2 evaluation results"):
            compare_evaluations(
                [
                    EvalResult(
                        suite_name="test",
                        scores={},
                        confidence_intervals={},
                        effect_sizes={},
                        sample_size=0,
                        seed=42,
                        metadata={},
                        raw_results=[],
                    )
                ]
            )


if __name__ == "__main__":
    pytest.main([__file__])
