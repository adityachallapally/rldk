"""Tests for bias evaluation metrics."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rldk.evals.metrics.bias import (
    SimpleSentimentAnalyzer,
    calculate_demographic_bias,
    detect_demographic_mentions,
    detect_stereotype_patterns,
    evaluate_bias,
)


class TestBiasEvaluation:
    """Test cases for bias evaluation."""

    def test_simple_sentiment_analyzer_positive_text(self):
        """Test simple sentiment analyzer with positive text."""
        analyzer = SimpleSentimentAnalyzer()

        result = analyzer.analyze_sentiment("This is a great and wonderful response.")

        assert "sentiment_score" in result
        assert "positive_count" in result
        assert "negative_count" in result
        assert "total_words" in result
        assert result["sentiment_score"] > 0
        assert result["positive_count"] > 0
        assert result["negative_count"] == 0

    def test_simple_sentiment_analyzer_negative_text(self):
        """Test simple sentiment analyzer with negative text."""
        analyzer = SimpleSentimentAnalyzer()

        result = analyzer.analyze_sentiment("This is a terrible and awful response.")

        assert result["sentiment_score"] < 0
        assert result["positive_count"] == 0
        assert result["negative_count"] > 0

    def test_simple_sentiment_analyzer_neutral_text(self):
        """Test simple sentiment analyzer with neutral text."""
        analyzer = SimpleSentimentAnalyzer()

        result = analyzer.analyze_sentiment("This is a neutral response about the topic.")

        assert result["sentiment_score"] == 0
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_simple_sentiment_analyzer_empty_text(self):
        """Test simple sentiment analyzer with empty text."""
        analyzer = SimpleSentimentAnalyzer()

        result = analyzer.analyze_sentiment("")

        assert result["sentiment_score"] == 0.0
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_simple_sentiment_analyzer_none_text(self):
        """Test simple sentiment analyzer with None text."""
        analyzer = SimpleSentimentAnalyzer()

        result = analyzer.analyze_sentiment(None)

        assert result["sentiment_score"] == 0.0
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_detect_demographic_mentions_gender(self):
        """Test demographic mention detection for gender."""
        result = detect_demographic_mentions("The woman is a scientist and the man is a teacher.")

        assert "gender" in result
        assert len(result["gender"]) > 0

    def test_detect_demographic_mentions_race(self):
        """Test demographic mention detection for race."""
        result = detect_demographic_mentions("The Asian person and Black person are both talented.")

        assert "race" in result
        assert len(result["race"]) > 0

    def test_detect_demographic_mentions_age(self):
        """Test demographic mention detection for age."""
        result = detect_demographic_mentions("The young person and old person have different perspectives.")

        assert "age" in result
        assert len(result["age"]) > 0

    def test_detect_demographic_mentions_religion(self):
        """Test demographic mention detection for religion."""
        result = detect_demographic_mentions("The Muslim person and Christian person are both kind.")

        assert "religion" in result
        assert len(result["religion"]) > 0

    def test_detect_demographic_mentions_nationality(self):
        """Test demographic mention detection for nationality."""
        result = detect_demographic_mentions("The American person and British person are both educated.")

        assert "nationality" in result
        assert len(result["nationality"]) > 0

    def test_detect_demographic_mentions_no_mentions(self):
        """Test demographic mention detection with no mentions."""
        result = detect_demographic_mentions("This is a general response about technology.")

        assert len(result) == 0

    def test_detect_demographic_mentions_empty_text(self):
        """Test demographic mention detection with empty text."""
        result = detect_demographic_mentions("")

        assert len(result) == 0

    def test_calculate_demographic_bias(self):
        """Test demographic bias calculation."""
        texts = [
            "The woman is intelligent.",
            "The man is strong.",
            "The woman is emotional.",
            "The man is aggressive."
        ]

        sentiment_scores = [0.1, 0.2, -0.1, -0.2]

        demographic_mentions = [
            {"gender": ["female:woman"]},
            {"gender": ["male:man"]},
            {"gender": ["female:woman"]},
            {"gender": ["male:man"]}
        ]

        result = calculate_demographic_bias(texts, sentiment_scores, demographic_mentions)

        assert "bias_score" in result
        assert "demographic_bias" in result
        assert "num_demographic_groups" in result
        assert result["bias_score"] >= 0
        assert result["num_demographic_groups"] > 0

    def test_calculate_demographic_bias_insufficient_data(self):
        """Test demographic bias calculation with insufficient data."""
        texts = ["The woman is intelligent."]
        sentiment_scores = [0.1]
        demographic_mentions = [{"gender": ["female:woman"]}]

        result = calculate_demographic_bias(texts, sentiment_scores, demographic_mentions)

        assert result["bias_score"] == 0.0
        assert len(result["demographic_bias"]) == 0

    def test_detect_stereotype_patterns_gender_stereotypes(self):
        """Test stereotype pattern detection for gender stereotypes."""
        result = detect_stereotype_patterns("Women are emotional and should not be leaders.")

        assert "stereotype_score" in result
        assert "patterns" in result
        assert result["stereotype_score"] > 0
        assert len(result["patterns"]) > 0

    def test_detect_stereotype_patterns_racial_stereotypes(self):
        """Test stereotype pattern detection for racial stereotypes."""
        result = detect_stereotype_patterns("Black people are dangerous and should be avoided.")

        assert result["stereotype_score"] > 0
        assert len(result["patterns"]) > 0

    def test_detect_stereotype_patterns_age_stereotypes(self):
        """Test stereotype pattern detection for age stereotypes."""
        result = detect_stereotype_patterns("Old people are slow and incompetent.")

        assert result["stereotype_score"] > 0
        assert len(result["patterns"]) > 0

    def test_detect_stereotype_patterns_no_stereotypes(self):
        """Test stereotype pattern detection with no stereotypes."""
        result = detect_stereotype_patterns("This is a helpful response about technology.")

        assert result["stereotype_score"] == 0.0
        assert len(result["patterns"]) == 0

    def test_detect_stereotype_patterns_empty_text(self):
        """Test stereotype pattern detection with empty text."""
        result = detect_stereotype_patterns("")

        assert result["stereotype_score"] == 0.0
        assert len(result["patterns"]) == 0

    def test_evaluate_bias_with_valid_data(self):
        """Test bias evaluation with valid data."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders.",
                "Men are aggressive and should not be trusted."
            ]
        })

        result = evaluate_bias(data)

        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "num_samples" in result
        assert "metrics" in result
        assert result["method"] == "demographic_analysis"
        assert result["num_samples"] == 4
        assert 0 <= result["score"] <= 1

    def test_evaluate_bias_missing_output_column(self):
        """Test bias evaluation with missing output column."""
        data = pd.DataFrame({
            "input": ["test input"],
            "model_name": ["test-model"]
        })

        result = evaluate_bias(data)

        assert result["score"] == 1.0  # High score = high bias (bad)
        assert "missing_output_column" in result["error"]
        assert result["num_samples"] == 0

    def test_evaluate_bias_insufficient_samples(self):
        """Test bias evaluation with insufficient samples."""
        data = pd.DataFrame({
            "output": ["This is a test response."]
        })

        result = evaluate_bias(data, min_samples=10)

        assert result["score"] == 1.0
        assert "insufficient_samples" in result["error"]
        assert result["num_samples"] == 1

    def test_evaluate_bias_no_sentiment_scores(self):
        """Test bias evaluation with no valid sentiment scores."""
        data = pd.DataFrame({
            "output": [None, "", "   "]
        })

        result = evaluate_bias(data)

        assert result["score"] == 1.0
        assert "no_sentiment_scores" in result["error"]
        assert result["num_samples"] == 0

    def test_evaluate_bias_with_mixed_content(self):
        """Test bias evaluation with mixed content."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders in their fields.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be in leadership positions.",
                "Men are aggressive and should not be trusted with sensitive tasks.",
                "The woman is a brilliant scientist who made important discoveries."
            ]
        })

        result = evaluate_bias(data)

        assert result["score"] > 0
        assert result["num_samples"] == 5
        assert "metrics" in result
        assert "demographic_bias_score" in result["metrics"]
        assert "mean_stereotype_score" in result["metrics"]
        assert "sentiment_variance" in result["metrics"]

    def test_evaluate_bias_confidence_intervals(self):
        """Test that confidence intervals are calculated correctly."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders.",
                "Men are aggressive and should not be trusted.",
                "The woman is a brilliant scientist.",
                "The man is a caring teacher.",
                "Women are all the same and lack individual personality.",
                "Men are all the same and lack individual personality.",
                "The woman is a talented artist.",
                "The man is a skilled engineer."
            ]
        })

        result = evaluate_bias(data)

        assert "metrics" in result
        assert "sentiment_confidence_interval" in result["metrics"]
        ci = result["metrics"]["sentiment_confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert "level" in ci
        assert ci["lower"] <= ci["upper"]
        assert ci["level"] == 0.95

    def test_evaluate_bias_custom_parameters(self):
        """Test bias evaluation with custom parameters."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders."
            ]
        })

        result = evaluate_bias(
            data,
            output_column="output",
            min_samples=3,
            use_external_sentiment=False
        )

        assert result["score"] > 0
        assert result["num_samples"] == 3

    @patch('builtins.__import__')
    def test_evaluate_bias_with_external_sentiment(self, mock_import):
        """Test bias evaluation with external sentiment analyzer."""
        # Mock VADER import and class
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.return_value = {"compound": 0.3}

        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'vaderSentiment.vaderSentiment':
                mock_vader = MagicMock()
                mock_vader.SentimentIntensityAnalyzer.return_value = mock_analyzer
                return mock_vader
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders."
            ]
        })

        result = evaluate_bias(data, use_external_sentiment=True)

        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["raw_data"]["sentiment_analyzer_type"] == "external"

    @patch('builtins.__import__')
    def test_evaluate_bias_external_sentiment_import_error(self, mock_import):
        """Test bias evaluation when external sentiment analyzer import fails."""
        # Mock import error for vaderSentiment
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'vaderSentiment.vaderSentiment':
                raise ImportError("VADER not available")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders."
            ]
        })

        result = evaluate_bias(data, use_external_sentiment=True)

        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["raw_data"]["sentiment_analyzer_type"] == "simple"

    def test_evaluate_bias_with_real_data(self):
        """Test bias evaluation with real test data."""
        # Read the test data file
        import os
        test_file = os.path.join(os.path.dirname(__file__), "..", "data", "bias_outputs.txt")
        with open(test_file) as f:
            outputs = [line.strip() for line in f if line.strip()]

        data = pd.DataFrame({
            "output": outputs
        })

        result = evaluate_bias(data)

        assert result["score"] > 0
        assert result["num_samples"] == len(outputs)
        assert "metrics" in result
        assert "demographic_bias_score" in result["metrics"]
        assert "mean_stereotype_score" in result["metrics"]

    def test_evaluate_bias_edge_cases(self):
        """Test bias evaluation with edge cases."""
        data = pd.DataFrame({
            "output": [
                None,  # None value
                "",    # Empty string
                "   ", # Whitespace only
                123,   # Non-string value
                "This is a neutral response without demographic mentions.",
                "The person is talented regardless of their background."  # Neutral
            ]
        })

        result = evaluate_bias(data)

        assert result["score"] >= 0
        assert result["num_samples"] >= 0
        assert "raw_data" in result
        assert "skipped_samples" in result["raw_data"]

    def test_evaluate_bias_demographic_categories_detected(self):
        """Test that demographic categories are correctly detected."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Asian people are hardworking and successful.",
                "Black people are talented and creative.",
                "Muslim people are peaceful and respectful.",
                "Young people are energetic and innovative."
            ]
        })

        result = evaluate_bias(data)

        assert "raw_data" in result
        assert "demographic_categories_detected" in result["raw_data"]
        categories = result["raw_data"]["demographic_categories_detected"]
        assert len(categories) > 0
        assert "gender" in categories
        assert "race" in categories
        assert "religion" in categories
        assert "age" in categories

    def test_evaluate_bias_sentiment_variance(self):
        """Test that sentiment variance is calculated correctly."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders.",
                "Men are aggressive and should not be trusted.",
                "The woman is a brilliant scientist.",
                "The man is a caring teacher.",
                "Women are all the same and lack individual personality.",
                "Men are all the same and lack individual personality.",
                "The woman is a talented artist.",
                "The man is a skilled engineer."
            ]
        })

        result = evaluate_bias(data)

        assert "metrics" in result
        assert "sentiment_variance" in result["metrics"]
        variance = result["metrics"]["sentiment_variance"]
        assert variance >= 0

    def test_evaluate_bias_demographic_bias_details(self):
        """Test that demographic bias details are provided."""
        data = pd.DataFrame({
            "output": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders.",
                "Men are aggressive and should not be trusted.",
                "The woman is a brilliant scientist.",
                "The man is a caring teacher."
            ]
        })

        result = evaluate_bias(data)

        assert "metrics" in result
        assert "demographic_bias_details" in result["metrics"]
        bias_details = result["metrics"]["demographic_bias_details"]
        assert len(bias_details) > 0


if __name__ == "__main__":
    pytest.main([__file__])
