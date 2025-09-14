"""Test division by zero and negative denominator handling."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.utils.error_handling import safe_percentage, safe_rate_calculation
from rldk.utils.math_utils import safe_divide_with_negative_support, try_divide


class TestDivisionHandling:
    """Test division handling functions for consistency."""

    def test_try_divide_normal_case(self):
        """Test normal division case."""
        result = try_divide(10, 2)
        assert result == 5.0

    def test_try_divide_by_zero(self):
        """Test division by zero."""
        result = try_divide(10, 0)
        assert result == 0.0

    def test_try_divide_negative_denominator(self):
        """Test that try_divide skips negative denominators (consistent with safe_percentage and safe_rate_calculation)."""
        result = try_divide(10, -2)
        assert result == 0.0  # Should return fallback, not -5.0

    def test_try_divide_negative_numerator(self):
        """Test division with negative numerator (denominator is positive)."""
        result = try_divide(-10, 2)
        assert result == -5.0

    def test_try_divide_custom_fallback(self):
        """Test division with custom fallback value."""
        result = try_divide(10, 0, fallback=1.0)
        assert result == 1.0

    def test_try_divide_negative_denominator_custom_fallback(self):
        """Test negative denominator with custom fallback."""
        result = try_divide(10, -2, fallback=99.0)
        assert result == 99.0  # Should return fallback, not perform division


class TestConsistencyWithOtherFunctions:
    """Test that try_divide is consistent with safe_percentage and safe_rate_calculation."""

    def test_negative_denominator_consistency(self):
        """Test that all functions handle negative denominators consistently."""
        numerator = 10
        denominator = -2

        # All should skip negative denominators and return fallback
        try_divide_result = try_divide(numerator, denominator)
        safe_percentage_result = safe_percentage(numerator, denominator)
        safe_rate_result = safe_rate_calculation(numerator, denominator)

        assert try_divide_result == 0.0
        assert safe_percentage_result == 0.0
        assert safe_rate_result == 0.0

        # Verify they all return the same fallback behavior
        assert try_divide_result == safe_percentage_result
        assert safe_percentage_result == safe_rate_result

    def test_zero_denominator_consistency(self):
        """Test that all functions handle zero denominators consistently."""
        numerator = 10
        denominator = 0

        try_divide_result = try_divide(numerator, denominator)
        safe_percentage_result = safe_percentage(numerator, denominator)
        safe_rate_result = safe_rate_calculation(numerator, denominator)

        assert try_divide_result == 0.0
        assert safe_percentage_result == 0.0
        assert safe_rate_result == 0.0

    def test_positive_denominator_consistency(self):
        """Test that all functions handle positive denominators consistently."""
        numerator = 10
        denominator = 2

        try_divide_result = try_divide(numerator, denominator)
        safe_percentage_result = safe_percentage(numerator, denominator)
        safe_rate_result = safe_rate_calculation(numerator, denominator)

        assert try_divide_result == 5.0
        assert safe_percentage_result == 500.0  # percentage = (numerator * 100) / denominator
        assert safe_rate_result == 5.0


class TestAlternativeImplementation:
    """Test the alternative implementation that allows negative denominators."""

    def test_safe_divide_with_negative_support_allows_negative(self):
        """Test that safe_divide_with_negative_support allows negative denominators."""
        result = safe_divide_with_negative_support(10, -2)
        assert result == -5.0  # Should perform the division

    def test_safe_divide_with_negative_support_zero_denominator(self):
        """Test that safe_divide_with_negative_support handles zero denominators."""
        result = safe_divide_with_negative_support(10, 0)
        assert result == 0.0  # Should return fallback

    def test_difference_between_implementations(self):
        """Test the difference between the two implementations."""
        numerator = 10
        denominator = -2

        # try_divide should skip negative denominators
        try_divide_result = try_divide(numerator, denominator)

        # safe_divide_with_negative_support should allow negative denominators
        alt_result = safe_divide_with_negative_support(numerator, denominator)

        assert try_divide_result == 0.0  # fallback
        assert alt_result == -5.0  # actual division result


class TestEdgeCases:
    """Test edge cases for division functions."""

    def test_float_inputs(self):
        """Test with float inputs."""
        result = try_divide(10.5, 2.5)
        assert result == 4.2

    def test_mixed_type_inputs(self):
        """Test with mixed int/float inputs."""
        result = try_divide(10, 2.5)
        assert result == 4.0

    def test_very_small_positive_denominator(self):
        """Test with very small positive denominator."""
        result = try_divide(1, 0.0001)
        assert result == 10000.0

    def test_very_small_negative_denominator(self):
        """Test with very small negative denominator."""
        result = try_divide(1, -0.0001)
        assert result == 0.0  # Should return fallback, not -10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
