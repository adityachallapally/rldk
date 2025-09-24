"""Proper test file for KL divergence fixes without hardcoded mocks."""

import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock numpy only when needed using pytest fixtures
@pytest.fixture
def mock_numpy():
    """Mock numpy for testing when not available."""
    try:
        import numpy as np
        return np
    except ImportError:
        # Create a minimal mock for numpy functionality
        mock_np = MagicMock()
        mock_np.array = lambda x, **kwargs: x
        mock_np.isnan = lambda x: str(x) == 'nan'
        mock_np.isinf = lambda x: str(x) in ['inf', '-inf']
        mock_np.std = lambda x, **kwargs: 0.1 if len(x) > 1 else 0.0
        mock_np.sum = sum
        mock_np.polyfit = lambda x, y, deg: [0.0, 0.0]
        mock_np.corrcoef = lambda x, y: [[1.0, 0.0], [0.0, 1.0]]
        mock_np.diff = lambda x: [x[i+1] - x[i] for i in range(len(x)-1)]
        mock_np.mean = lambda x: sum(x) / len(x) if x else 0
        mock_np.max = max
        mock_np.min = min
        mock_np.any = any
        mock_np.linalg = MagicMock()
        mock_np.linalg.LinAlgError = Exception
        return mock_np


class TestKLDivergenceFixes:
    """Test the fixed KL divergence implementation."""

    def test_imports_work(self):
        """Test that the modules can be imported without errors."""
        try:
            from rldk.evals.metrics import calculate_kl_divergence
            from rldk.forensics.kl_schedule_tracker import (
                KLScheduleMetrics,
                KLScheduleTracker,
                _safe_coefficient_value,
                _safe_kl_value,
            )
            assert True  # If we get here, imports work
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_safe_kl_value_processing(self):
        """Test safe KL value processing without external dependencies."""
        try:
            from rldk.forensics.kl_schedule_tracker import _safe_kl_value

            # Test valid inputs
            assert _safe_kl_value(0.5) == 0.5
            assert _safe_kl_value("0.5") == 0.5

            # Test edge cases
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # NaN handling
                result = _safe_kl_value(float('nan'))
                assert result == 0.0
                assert len(w) >= 1
                assert any("NaN KL value detected" in str(warning.message) for warning in w)

                # Infinite handling
                result = _safe_kl_value(float('inf'))
                assert result == 1e6
                assert len(w) >= 1
                assert any("Positive infinity KL value detected" in str(warning.message) for warning in w)

                # Negative handling
                result = _safe_kl_value(-0.1)
                assert result == 0.0
                assert len(w) >= 1
                assert any("Negative KL value" in str(warning.message) for warning in w)

        except ImportError:
            pytest.skip("KL schedule tracker not available")

    def test_safe_coefficient_value_processing(self):
        """Test safe coefficient value processing."""
        try:
            from rldk.forensics.kl_schedule_tracker import _safe_coefficient_value

            # Test valid inputs
            assert _safe_coefficient_value(1.0) == 1.0
            assert _safe_coefficient_value("1.0") == 1.0

            # Test edge cases
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Zero handling
                result = _safe_coefficient_value(0.0)
                assert result == 1.0
                assert len(w) >= 1
                assert any("Non-positive coefficient value" in str(warning.message) for warning in w)

                # Negative handling
                result = _safe_coefficient_value(-0.1)
                assert result == 1.0
                assert len(w) >= 1
                assert any("Non-positive coefficient value" in str(warning.message) for warning in w)

        except ImportError:
            pytest.skip("KL schedule tracker not available")

    def test_kl_schedule_tracker_robustness(self, mock_numpy):
        """Test KL schedule tracker with mocked numpy."""
        try:
            from rldk.forensics import kl_schedule_tracker as tracker_module

            with patch.object(tracker_module, "np", mock_numpy):
                tracker = tracker_module.KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)
                assert tracker.kl_target == 0.1
                assert tracker.kl_target_tolerance == 0.05

                # Test normal updates
                metrics = tracker.update(step=1, kl_value=0.05, kl_coef=1.0)
                assert metrics.current_kl == 0.05
                assert metrics.current_kl_coef == 1.0

                # Test edge case updates
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    metrics = tracker.update(step=2, kl_value=float('nan'), kl_coef=float('nan'))
                    assert metrics.current_kl == 0.0
                    assert metrics.current_kl_coef == 1.0
                    assert len(w) >= 2

        except ImportError:
            pytest.skip("KL schedule tracker not available")

    def test_kl_divergence_calculation_with_numpy(self):
        """Test KL divergence calculation when numpy is available."""
        try:
            import numpy as np

            from rldk.evals.metrics import calculate_kl_divergence

            # Test basic calculation
            p = np.array([0.5, 0.3, 0.2])
            q = np.array([0.4, 0.4, 0.2])

            kl_div = calculate_kl_divergence(p, q)
            assert isinstance(kl_div, float)
            assert kl_div >= 0.0
            assert not np.isnan(kl_div)
            assert not np.isinf(kl_div)

            # Test identical distributions
            p_identical = np.array([0.5, 0.3, 0.2])
            q_identical = np.array([0.5, 0.3, 0.2])

            kl_div_identical = calculate_kl_divergence(p_identical, q_identical)
            assert abs(kl_div_identical) < 1e-6

            # Test error handling
            with pytest.raises(ValueError, match="NaN values"):
                p_nan = np.array([0.5, np.nan, 0.3])
                q_nan = np.array([0.4, 0.4, 0.2])
                calculate_kl_divergence(p_nan, q_nan)

        except ImportError:
            pytest.skip("numpy not available for KL divergence testing")

    def test_numerical_stability(self):
        """Test numerical stability of calculations."""
        try:
            from rldk.forensics.kl_schedule_tracker import (
                _safe_coefficient_value,
                _safe_kl_value,
            )

            # Test repeated calculations with same inputs
            results_kl = []
            results_coef = []

            for _ in range(100):
                results_kl.append(_safe_kl_value(0.5))
                results_coef.append(_safe_coefficient_value(1.0))

            # All results should be identical
            assert all(r == results_kl[0] for r in results_kl)
            assert all(r == results_coef[0] for r in results_coef)

        except ImportError:
            pytest.skip("KL schedule tracker not available")


class TestBackwardCompatibility:
    """Test backward compatibility of the fixes."""

    def test_original_function_signatures(self):
        """Test that original function signatures still work."""
        try:
            from rldk.forensics.kl_schedule_tracker import KLScheduleTracker

            tracker = KLScheduleTracker()

            # Test with original float types
            metrics = tracker.update(step=1, kl_value=0.1, kl_coef=1.0)
            assert isinstance(metrics.current_kl, float)
            assert isinstance(metrics.current_kl_coef, float)

            # Test with new Any types (should still work)
            metrics = tracker.update(step=2, kl_value="0.2", kl_coef="1.5")
            assert isinstance(metrics.current_kl, float)
            assert isinstance(metrics.current_kl_coef, float)

        except ImportError:
            pytest.skip("KL schedule tracker not available")

    def test_default_values_unchanged(self):
        """Test that default values haven't changed."""
        try:
            from rldk.forensics.kl_schedule_tracker import (
                _safe_coefficient_value,
                _safe_kl_value,
            )

            # Test default values
            assert _safe_kl_value(None) == 0.0  # Default for KL
            assert _safe_coefficient_value(None) == 1.0  # Default for coefficient

        except ImportError:
            pytest.skip("KL schedule tracker not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
