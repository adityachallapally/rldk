"""Tests for evaluation integrity checks."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rldk.evals.integrity import (
    evaluate_answer_leakage,
    evaluate_data_split_integrity,
    evaluate_evaluation_robustness,
    evaluate_prompt_contamination,
)


class TestPromptContamination:
    """Test prompt contamination detection."""

    def setup_method(self):
        """Set up test data."""
        n_samples = 100
        self.sample_data = pd.DataFrame({
            "step": range(n_samples),
            "reward_mean": np.random.normal(0.5, 0.2, n_samples),
            "prompt": [f"Test prompt {i}" for i in range(n_samples)],
            "response": [f"Test response {i}" for i in range(n_samples)],
        })

    def test_basic_functionality(self):
        """Test basic prompt contamination evaluation."""
        result = evaluate_prompt_contamination(self.sample_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "metrics" in result
        assert "sample_size" in result
        assert result["sample_size"] == 100
        assert 0 <= result["score"] <= 1

    def test_duplicate_prompts(self):
        """Test detection of duplicate prompts."""
        # Create data with many duplicates
        duplicate_data = self.sample_data.copy()
        duplicate_data["prompt"] = ["Same prompt"] * 50 + ["Another prompt"] * 50

        result = evaluate_prompt_contamination(duplicate_data, seed=42)

        # Should detect high contamination due to duplicates
        assert result["score"] < 0.8  # Lower score indicates more contamination

        # Check that duplicate metric is present
        metrics = dict(result["metrics"])
        assert "duplicate_prompts" in metrics
        assert metrics["duplicate_prompts"] > 0.4  # High duplicate ratio

    def test_metadata_leakage(self):
        """Test detection of metadata leakage in prompts."""
        # Create data with metadata in prompts
        leakage_data = self.sample_data.copy()
        leakage_data["prompt"] = [
            f"Test prompt for epoch {i} at step {i*10}"
            for i in range(len(leakage_data))
        ]

        result = evaluate_prompt_contamination(leakage_data, seed=42)

        # Should detect contamination due to metadata leakage
        assert result["score"] < 0.9  # Lower score indicates more contamination

    def test_no_prompt_data(self):
        """Test behavior when no prompt data is available."""
        no_prompt_data = pd.DataFrame({
            "step": range(50),
            "reward_mean": np.random.normal(0.5, 0.2, 50),
        })

        result = evaluate_prompt_contamination(no_prompt_data, seed=42)

        assert result["score"] == 0.5  # Neutral score
        assert "no_prompt_data" in result["method"]

    def test_test_patterns(self):
        """Test detection of test-like patterns."""
        # Create data with test patterns
        test_data = self.sample_data.copy()
        test_data["prompt"] = [
            f"Answer the following question: What is {i} + {i}?"
            for i in range(len(test_data))
        ]

        result = evaluate_prompt_contamination(test_data, seed=42)

        # Should detect contamination due to test patterns
        assert result["score"] < 0.9  # Lower score indicates more contamination


class TestAnswerLeakage:
    """Test answer leakage detection."""

    def setup_method(self):
        """Set up test data."""
        n_samples = 100
        self.sample_data = pd.DataFrame({
            "step": range(n_samples),
            "reward_mean": np.random.normal(0.5, 0.2, n_samples),
            "prompt": [f"Test prompt {i}" for i in range(n_samples)],
            "response": [f"Test response {i}" for i in range(n_samples)],
        })

    def test_basic_functionality(self):
        """Test basic answer leakage evaluation."""
        result = evaluate_answer_leakage(self.sample_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "metrics" in result
        assert "sample_size" in result
        assert result["sample_size"] == 100
        assert 0 <= result["score"] <= 1

    def test_direct_answer_leakage(self):
        """Test detection of direct answer leakage."""
        # Create data with direct answer leakage
        leakage_data = self.sample_data.copy()
        leakage_data["prompt"] = [
            "What is 2 + 2? The answer is 4. Please confirm."
            for i in range(len(leakage_data))
        ]
        leakage_data["response"] = [
            "The answer is 4"
            for i in range(len(leakage_data))
        ]

        result = evaluate_answer_leakage(leakage_data, seed=42)

        # Should detect high leakage
        assert result["score"] < 0.8  # Lower score indicates more leakage

    def test_numerical_leakage(self):
        """Test detection of numerical answer leakage."""
        # Create data with numerical leakage
        leakage_data = self.sample_data.copy()
        leakage_data["prompt"] = [
            f"Calculate the sum of {i} and {i+1}"
            for i in range(len(leakage_data))
        ]
        leakage_data["response"] = [
            f"The sum is {i + (i+1)}"
            for i in range(len(leakage_data))
        ]

        result = evaluate_answer_leakage(leakage_data, seed=42)

        # Should detect some numerical leakage
        assert result["score"] < 0.9  # Lower score indicates more leakage

    def test_no_response_data(self):
        """Test behavior when no response data is available."""
        no_response_data = pd.DataFrame({
            "step": range(50),
            "reward_mean": np.random.normal(0.5, 0.2, 50),
        })

        result = evaluate_answer_leakage(no_response_data, seed=42)

        assert result["score"] == 0.5  # Neutral score
        assert "no_response_data" in result["method"]

    def test_partial_answer_leakage(self):
        """Test detection of partial answer leakage."""
        # Create data with partial answer leakage
        leakage_data = self.sample_data.copy()
        leakage_data["prompt"] = [
            "The answer is: What is the capital of France?"
            for i in range(len(leakage_data))
        ]
        leakage_data["response"] = [
            "The capital of France is Paris"
            for i in range(len(leakage_data))
        ]

        result = evaluate_answer_leakage(leakage_data, seed=42)

        # Should detect partial leakage
        assert result["score"] < 0.9  # Lower score indicates more leakage


class TestDataSplitIntegrity:
    """Test data split integrity detection."""

    def setup_method(self):
        """Set up test data."""
        n_samples = 200
        self.sample_data = pd.DataFrame({
            "step": range(n_samples),
            "reward_mean": np.random.normal(0.5, 0.2, n_samples),
            "split": ["train"] * 100 + ["val"] * 50 + ["test"] * 50,
            "prompt": [f"Test prompt {i}" for i in range(n_samples)],
            "response": [f"Test response {i}" for i in range(n_samples)],
        })

    def test_basic_functionality(self):
        """Test basic data split integrity evaluation."""
        result = evaluate_data_split_integrity(self.sample_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "metrics" in result
        assert "sample_size" in result
        assert result["sample_size"] == 200
        assert 0 <= result["score"] <= 1

    def test_cross_split_duplicates(self):
        """Test detection of cross-split duplicates."""
        # Create data with cross-split duplicates
        duplicate_data = self.sample_data.copy()
        duplicate_data["prompt"] = ["Same prompt"] * 200  # All same prompt

        result = evaluate_data_split_integrity(duplicate_data, seed=42)

        # Should detect high contamination due to cross-split duplicates
        assert result["score"] < 0.8  # Lower score indicates more contamination

    def test_unbalanced_splits(self):
        """Test detection of unbalanced splits."""
        # Create data with very unbalanced splits
        unbalanced_data = self.sample_data.copy()
        unbalanced_data["split"] = ["train"] * 190 + ["val"] * 5 + ["test"] * 5

        result = evaluate_data_split_integrity(unbalanced_data, seed=42)

        # Should detect integrity issues due to unbalanced splits
        assert result["score"] < 0.9  # Lower score indicates more issues

    def test_no_split_data(self):
        """Test behavior when no split data is available."""
        no_split_data = pd.DataFrame({
            "step": range(50),
            "reward_mean": np.random.normal(0.5, 0.2, 50),
        })

        result = evaluate_data_split_integrity(no_split_data, seed=42)

        assert result["score"] == 0.5  # Neutral score
        assert "no_split_data" in result["method"]

    def test_temporal_violations(self):
        """Test detection of temporal violations."""
        # Create data with temporal violations
        temporal_data = self.sample_data.copy()
        temporal_data["timestamp"] = pd.date_range("2023-01-01", periods=200, freq="H")
        # Reverse timestamps within some splits
        temporal_data.loc[50:100, "timestamp"] = temporal_data.loc[50:100, "timestamp"].iloc[::-1]

        result = evaluate_data_split_integrity(temporal_data, seed=42)

        # Should detect some temporal violations
        assert result["score"] < 1.0  # Lower score indicates more violations


class TestEvaluationRobustness:
    """Test evaluation robustness detection."""

    def setup_method(self):
        """Set up test data."""
        n_samples = 100
        self.sample_data = pd.DataFrame({
            "step": range(n_samples),
            "reward_mean": np.random.normal(0.5, 0.2, n_samples),
            "reward_std": np.random.uniform(0.1, 0.3, n_samples),
            "accuracy": np.random.uniform(0.6, 0.9, n_samples),
        })

    def test_basic_functionality(self):
        """Test basic evaluation robustness evaluation."""
        result = evaluate_evaluation_robustness(self.sample_data, seed=42)

        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "metrics" in result
        assert "sample_size" in result
        assert result["sample_size"] == 100
        assert 0 <= result["score"] <= 1

    def test_small_sample_size(self):
        """Test detection of small sample size."""
        small_data = self.sample_data.iloc[:5]  # Very small sample

        result = evaluate_evaluation_robustness(small_data, seed=42)

        # Should detect robustness issues due to small sample
        assert result["score"] < 0.8  # Lower score indicates less robust

    def test_high_variance(self):
        """Test detection of high variance."""
        # Create data with high variance
        high_var_data = self.sample_data.copy()
        high_var_data["reward_mean"] = np.random.normal(0.5, 1.0, 100)  # High variance

        result = evaluate_evaluation_robustness(high_var_data, seed=42)

        # Should detect robustness issues due to high variance
        assert result["score"] < 0.9  # Lower score indicates less robust

    def test_systematic_bias(self):
        """Test detection of systematic bias."""
        # Create data with systematic bias
        biased_data = self.sample_data.copy()
        biased_data["reward_mean"] = biased_data["step"] * 0.01  # Correlated with step

        result = evaluate_evaluation_robustness(biased_data, seed=42)

        # Should detect robustness issues due to systematic bias
        assert result["score"] < 0.9  # Lower score indicates less robust

    def test_outliers(self):
        """Test detection of outliers."""
        # Create data with outliers
        outlier_data = self.sample_data.copy()
        outlier_data.loc[0, "reward_mean"] = 10.0  # Extreme outlier

        result = evaluate_evaluation_robustness(outlier_data, seed=42)

        # Should detect robustness issues due to outliers
        assert result["score"] < 1.0  # Lower score indicates less robust


class TestIntegrationWithEvaluationFramework:
    """Test integration with the evaluation framework."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        n_samples = 100
        self.sample_data = pd.DataFrame({
            "step": range(n_samples),
            "reward_mean": np.random.normal(0.5, 0.2, n_samples),
            "prompt": [f"Test prompt {i}" for i in range(n_samples)],
            "response": [f"Test response {i}" for i in range(n_samples)],
            "split": ["train"] * 60 + ["val"] * 20 + ["test"] * 20,
        })

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integrity_suite_import(self):
        """Test that integrity evaluations can be imported and used."""
        from src.rldk.evals.suites import get_eval_suite

        # Test that integrity suite exists
        integrity_suite = get_eval_suite("integrity")
        assert integrity_suite is not None
        assert integrity_suite["name"] == "integrity"

        # Test that integrity evaluations are included
        evaluations = integrity_suite["evaluations"]
        assert "prompt_contamination" in evaluations
        assert "answer_leakage" in evaluations
        assert "data_split_integrity" in evaluations
        assert "evaluation_robustness" in evaluations

    def test_quick_suite_integration(self):
        """Test that integrity checks are integrated into quick suite."""
        from src.rldk.evals.suites import get_eval_suite

        quick_suite = get_eval_suite("quick")
        evaluations = quick_suite["evaluations"]

        # Quick suite should include basic integrity checks
        assert "prompt_contamination" in evaluations
        assert "answer_leakage" in evaluations

    def test_comprehensive_suite_integration(self):
        """Test that integrity checks are integrated into comprehensive suite."""
        from src.rldk.evals.suites import get_eval_suite

        comprehensive_suite = get_eval_suite("comprehensive")
        evaluations = comprehensive_suite["evaluations"]

        # Comprehensive suite should include all integrity checks
        assert "prompt_contamination" in evaluations
        assert "answer_leakage" in evaluations
        assert "data_split_integrity" in evaluations
        assert "evaluation_robustness" in evaluations

    def test_baseline_scores(self):
        """Test that baseline scores are properly set for integrity checks."""
        from src.rldk.evals.suites import get_eval_suite

        integrity_suite = get_eval_suite("integrity")
        baseline_scores = integrity_suite["baseline_scores"]

        # Check that baseline scores are reasonable
        assert baseline_scores["prompt_contamination"] > 0.7  # High baseline (less contamination)
        assert baseline_scores["answer_leakage"] > 0.7  # High baseline (less leakage)
        assert baseline_scores["data_split_integrity"] > 0.8  # Very high baseline (good integrity)
        assert baseline_scores["evaluation_robustness"] > 0.7  # High baseline (more robust)
