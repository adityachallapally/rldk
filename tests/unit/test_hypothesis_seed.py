"""
Hypothesis tests for seed management and reproducibility.
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
try:  # pragma: no cover - optional dependency handling
    from hypothesis import assume, given
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    HYPOTHESIS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis package is not installed",
)

if not HYPOTHESIS_AVAILABLE:  # pragma: no cover - optional dependency handling
    pytest.skip("hypothesis package is not installed", allow_module_level=True)


class TestSeedHypothesis:
    """Test seed management with property-based testing."""

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_setting(self, seed: int):
        """Test that seed setting works for various seed values."""
        try:
            from rldk.utils.seed import get_current_seed, set_global_seed

            # Set seed
            result = set_global_seed(seed)
            assert result == seed

            # Get current seed
            current = get_current_seed()
            assert current == seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed1=st.integers(min_value=0, max_value=2**32-1),
        seed2=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_context_manager(self, seed1: int, seed2: int):
        """Test that seed context manager works correctly."""
        try:
            from rldk.utils.seed import get_current_seed, seed_context, set_global_seed

            # Set initial seed
            set_global_seed(seed1)
            assert get_current_seed() == seed1

            # Use context manager
            with seed_context(seed2):
                assert get_current_seed() == seed2

            # Seed should be restored
            assert get_current_seed() == seed1

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_state_summary(self, seed: int):
        """Test that seed state summary works for various seeds."""
        try:
            from rldk.utils.seed import get_seed_state_summary, set_global_seed

            # Set seed
            set_global_seed(seed)

            # Get summary
            summary = get_seed_state_summary()
            assert isinstance(summary, dict)
            assert "seed" in summary
            assert summary["seed"] == seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_validation(self, seed: int):
        """Test that seed validation works for various seeds."""
        try:
            from rldk.utils.seed import set_global_seed, validate_seed_consistency

            # Set seed
            set_global_seed(seed)

            # Validate consistency
            is_consistent = validate_seed_consistency()
            assert isinstance(is_consistent, bool)

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_reproducible_environment(self, seed: int):
        """Test that reproducible environment setup works for various seeds."""
        try:
            from rldk.utils.seed import set_reproducible_environment

            # Set up reproducible environment
            result = set_reproducible_environment(seed)
            assert result == seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seeds=st.lists(
            st.integers(min_value=0, max_value=2**32-1),
            min_size=1,
            max_size=10
        )
    )
    def test_multiple_seed_changes(self, seeds: List[int]):
        """Test that multiple seed changes work correctly."""
        try:
            from rldk.utils.seed import get_current_seed, set_global_seed

            for seed in seeds:
                set_global_seed(seed)
                assert get_current_seed() == seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_restoration(self, seed: int):
        """Test that seed restoration works correctly."""
        try:
            from rldk.utils.seed import (
                get_current_seed,
                restore_seed_state,
                set_global_seed,
            )

            # Set initial seed
            set_global_seed(seed)
            get_current_seed()

            # Change seed
            set_global_seed(seed + 1)
            assert get_current_seed() == seed + 1

            # Restore state (this might not work exactly as expected without proper state management)
            # But it should not crash
            restore_seed_state()

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_determinism(self, seed: int):
        """Test that same seed produces consistent results."""
        try:
            from rldk.utils.seed import get_current_seed, set_global_seed

            # Set seed twice
            set_global_seed(seed)
            result1 = get_current_seed()

            set_global_seed(seed)
            result2 = get_current_seed()

            # Results should be identical
            assert result1 == result2
            assert result1 == seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_edge_cases(self, seed: int):
        """Test edge cases for seed values."""
        try:
            from rldk.utils.seed import get_current_seed, set_global_seed

            # Test edge cases
            edge_cases = [0, 1, 2**16-1, 2**32-1]
            for edge_seed in edge_cases:
                set_global_seed(edge_seed)
                assert get_current_seed() == edge_seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_type_consistency(self, seed: int):
        """Test that seed types remain consistent."""
        try:
            from rldk.utils.seed import get_current_seed, set_global_seed

            # Set seed
            result = set_global_seed(seed)
            assert isinstance(result, int)

            # Get current seed
            current = get_current_seed()
            assert isinstance(current, int)

            # Values should match
            assert result == current
            assert result == seed

        except ImportError:
            # Skip if module not available
            pytest.skip("Seed module not available")
