"""Tests for toxicity evaluation metrics."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rldk.evals.metrics.toxicity import (
    SimpleToxicityClassifier,
    detect_toxic_patterns,
    evaluate_toxicity,
)


class TestToxicityEvaluation:
    """Test cases for toxicity evaluation."""

    def test_simple_toxicity_classifier_benign_text(self):
        """Test simple toxicity classifier with benign text."""
        classifier = SimpleToxicityClassifier()

        result = classifier.classify_text("This is a helpful and informative response.")

        assert "toxicity_score" in result
        assert "confidence" in result
        assert "toxic_words" in result
        assert "toxic_phrases" in result
        assert "context_score" in result
        assert "mitigation_factor" in result
        assert result["toxicity_score"] >= 0
        assert result["toxicity_score"] <= 1
        assert result["confidence"] >= 0
        assert result["confidence"] <= 1

    def test_simple_toxicity_classifier_toxic_text(self):
        """Test simple toxicity classifier with toxic text."""
        classifier = SimpleToxicityClassifier()

        result = classifier.classify_text("You should kill yourself because you are worthless.")

        assert result["toxicity_score"] > 0
        assert len(result["toxic_words"]) > 0
        assert result["toxic_phrases"] > 0

    def test_simple_toxicity_classifier_empty_text(self):
        """Test simple toxicity classifier with empty text."""
        classifier = SimpleToxicityClassifier()

        result = classifier.classify_text("")

        assert result["toxicity_score"] == 0.0
        assert result["confidence"] == 0.0

    def test_simple_toxicity_classifier_none_text(self):
        """Test simple toxicity classifier with None text."""
        classifier = SimpleToxicityClassifier()

        result = classifier.classify_text(None)

        assert result["toxicity_score"] == 0.0
        assert result["confidence"] == 0.0

    def test_simple_toxicity_classifier_mitigation_words(self):
        """Test simple toxicity classifier with mitigation words."""
        classifier = SimpleToxicityClassifier()

        result = classifier.classify_text("This is not hate speech and should not be considered harmful.")

        assert result["mitigation_factor"] < 1.0  # Should reduce toxicity

    def test_detect_toxic_patterns_benign_text(self):
        """Test toxic pattern detection with benign text."""
        result = detect_toxic_patterns("This is a helpful response about machine learning.")

        assert "patterns" in result
        assert "pattern_score" in result
        assert result["pattern_score"] == 0.0
        assert len(result["patterns"]) == 0

    def test_detect_toxic_patterns_hate_speech(self):
        """Test toxic pattern detection with hate speech."""
        result = detect_toxic_patterns("All women are emotional and should not be leaders.")

        assert result["pattern_score"] > 0
        assert len(result["patterns"]) > 0

    def test_detect_toxic_patterns_threats(self):
        """Test toxic pattern detection with threats."""
        result = detect_toxic_patterns("I will kill you if you disagree with me.")

        assert result["pattern_score"] > 0
        assert len(result["patterns"]) > 0

    def test_detect_toxic_patterns_empty_text(self):
        """Test toxic pattern detection with empty text."""
        result = detect_toxic_patterns("")

        assert result["pattern_score"] == 0.0
        assert len(result["patterns"]) == 0

    def test_evaluate_toxicity_with_valid_data(self):
        """Test toxicity evaluation with valid data."""
        data = pd.DataFrame({
            "output": [
                "This is a helpful response.",
                "You should kill yourself.",
                "This is informative content.",
                "I hate all people from that group."
            ]
        })

        result = evaluate_toxicity(data)

        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "num_samples" in result
        assert "metrics" in result
        assert result["method"] == "content_analysis"
        assert result["num_samples"] == 4
        assert 0 <= result["score"] <= 1

    def test_evaluate_toxicity_missing_output_column(self):
        """Test toxicity evaluation with missing output column."""
        data = pd.DataFrame({
            "input": ["test input"],
            "model_name": ["test-model"]
        })

        result = evaluate_toxicity(data)

        assert result["score"] == 1.0  # High score = high toxicity (bad)
        assert "missing_output_column" in result["error"]
        assert result["num_samples"] == 0

    def test_evaluate_toxicity_insufficient_samples(self):
        """Test toxicity evaluation with insufficient samples."""
        data = pd.DataFrame({
            "output": ["This is a test response."]
        })

        result = evaluate_toxicity(data, min_samples=10)

        assert result["score"] == 1.0
        assert "insufficient_samples" in result["error"]
        assert result["num_samples"] == 1

    def test_evaluate_toxicity_no_toxicity_scores(self):
        """Test toxicity evaluation with no valid toxicity scores."""
        data = pd.DataFrame({
            "output": [None, "", "   "]
        })

        result = evaluate_toxicity(data)

        assert result["score"] == 1.0
        assert "no_toxicity_scores" in result["error"]
        assert result["num_samples"] == 0

    def test_evaluate_toxicity_with_mixed_content(self):
        """Test toxicity evaluation with mixed benign and toxic content."""
        data = pd.DataFrame({
            "output": [
                "This is a helpful and informative response about machine learning.",
                "You should kill yourself because you are worthless.",
                "The weather today is sunny and pleasant.",
                "All people from that group are inferior and should be eliminated.",
                "I enjoy reading books and learning new things."
            ]
        })

        result = evaluate_toxicity(data)

        assert result["score"] > 0
        assert result["num_samples"] == 5
        assert "metrics" in result
        assert "mean_toxicity" in result["metrics"]
        assert "high_toxicity_ratio" in result["metrics"]
        assert "mean_pattern_score" in result["metrics"]

    def test_evaluate_toxicity_confidence_intervals(self):
        """Test that confidence intervals are calculated correctly."""
        data = pd.DataFrame({
            "output": [
                "This is a helpful response.",
                "You should kill yourself.",
                "This is informative content.",
                "I hate all people from that group.",
                "The weather is nice today.",
                "You are worthless and should die.",
                "This is educational material.",
                "All women are inferior to men.",
                "Reading is a great hobby.",
                "You deserve to be harmed."
            ]
        })

        result = evaluate_toxicity(data)

        assert "metrics" in result
        assert "confidence_interval" in result["metrics"]
        ci = result["metrics"]["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert "level" in ci
        assert ci["lower"] <= ci["upper"]
        assert ci["level"] == 0.95

    def test_evaluate_toxicity_toxicity_percentiles(self):
        """Test that toxicity percentiles are calculated correctly."""
        data = pd.DataFrame({
            "output": [
                "This is a helpful response.",
                "You should kill yourself.",
                "This is informative content.",
                "I hate all people from that group.",
                "The weather is nice today.",
                "You are worthless and should die.",
                "This is educational material.",
                "All women are inferior to men.",
                "Reading is a great hobby.",
                "You deserve to be harmed."
            ]
        })

        result = evaluate_toxicity(data)

        assert "metrics" in result
        assert "toxicity_percentiles" in result["metrics"]
        percentiles = result["metrics"]["toxicity_percentiles"]
        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles

    def test_evaluate_toxicity_custom_parameters(self):
        """Test toxicity evaluation with custom parameters."""
        data = pd.DataFrame({
            "output": [
                "This is a helpful response.",
                "You should kill yourself.",
                "This is informative content."
            ]
        })

        result = evaluate_toxicity(
            data,
            output_column="output",
            min_samples=3,
            use_external_classifier=False
        )

        assert result["score"] > 0
        assert result["num_samples"] == 3

    @patch('builtins.__import__')
    def test_evaluate_toxicity_with_external_classifier(self, mock_import):
        """Test toxicity evaluation with external classifier."""
        # Mock Detoxify import and class
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = {"toxicity": 0.3}

        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'detoxify':
                mock_detoxify = MagicMock()
                mock_detoxify.Detoxify.return_value = mock_classifier
                return mock_detoxify
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        data = pd.DataFrame({
            "output": [
                "This is a test response.",
                "You should kill yourself.",
                "This is helpful content."
            ]
        })

        result = evaluate_toxicity(data, use_external_classifier=True)

        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["raw_data"]["classifier_type"] == "external"

    @patch('builtins.__import__')
    def test_evaluate_toxicity_external_classifier_import_error(self, mock_import):
        """Test toxicity evaluation when external classifier import fails."""
        # Mock import error for detoxify
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'detoxify':
                raise ImportError("Detoxify not available")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        data = pd.DataFrame({
            "output": [
                "This is a test response.",
                "You should kill yourself.",
                "This is helpful content."
            ]
        })

        result = evaluate_toxicity(data, use_external_classifier=True)

        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["raw_data"]["classifier_type"] == "simple"

    def test_evaluate_toxicity_with_real_data(self):
        """Test toxicity evaluation with real test data."""
        # Read the test data file
        import os
        test_file = os.path.join(os.path.dirname(__file__), "..", "data", "toxicity_outputs.txt")
        with open(test_file) as f:
            outputs = [line.strip() for line in f if line.strip()]

        data = pd.DataFrame({
            "output": outputs
        })

        result = evaluate_toxicity(data)

        assert result["score"] > 0
        assert result["num_samples"] == len(outputs)
        assert "metrics" in result
        assert "mean_toxicity" in result["metrics"]
        assert "high_toxicity_ratio" in result["metrics"]

    def test_evaluate_toxicity_edge_cases(self):
        """Test toxicity evaluation with edge cases."""
        data = pd.DataFrame({
            "output": [
                None,  # None value
                "",    # Empty string
                "   ", # Whitespace only
                123,   # Non-string value
                "This is a normal response with some potentially concerning words like hate and violence but used in a non-toxic context.",
                "You should not kill yourself because life is valuable."  # Mitigation
            ]
        })

        result = evaluate_toxicity(data)

        assert result["score"] >= 0
        assert result["num_samples"] >= 0
        assert "raw_data" in result
        assert "skipped_samples" in result["raw_data"]

    def test_evaluate_toxicity_high_toxicity_ratio(self):
        """Test that high toxicity ratio is calculated correctly."""
        data = pd.DataFrame({
            "output": [
                "This is a helpful response.",
                "You should kill yourself.",
                "This is informative content.",
                "I hate all people from that group.",
                "The weather is nice today.",
                "You are worthless and should die.",
                "This is educational material.",
                "All women are inferior to men.",
                "Reading is a great hobby.",
                "You deserve to be harmed."
            ]
        })

        result = evaluate_toxicity(data)

        assert "metrics" in result
        assert "high_toxicity_ratio" in result["metrics"]
        high_toxicity_ratio = result["metrics"]["high_toxicity_ratio"]
        assert 0 <= high_toxicity_ratio <= 1


if __name__ == "__main__":
    pytest.main([__file__])
