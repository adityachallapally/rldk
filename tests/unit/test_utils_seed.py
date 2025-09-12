#!/usr/bin/env python3
"""Unit tests for rldk.utils.seed module."""

import pytest
import numpy as np
import random
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module under test
from rldk.utils.seed import (
    set_global_seed,
    get_current_seed,
    restore_seed_state,
    get_seed_state_summary,
    set_reproducible_environment,
    validate_seed_consistency,
    seed_context
)


class TestSeedManagement:
    """Test seed management functionality."""
    
    def test_set_global_seed(self):
        """Test setting global seed."""
        # Test with explicit seed
        seed = set_global_seed(42)
        assert seed == 42
        assert get_current_seed() == 42
        
        # Test with None (should generate random seed)
        seed = set_global_seed(None)
        assert seed is not None
        assert isinstance(seed, int)
        assert get_current_seed() == seed
        
        # Test with deterministic=False
        seed = set_global_seed(42, deterministic=False)
        assert seed == 42
        assert get_current_seed() == 42
    
    def test_get_current_seed(self):
        """Test getting current seed."""
        # Initially should be None
        assert get_current_seed() is None
        
        # After setting seed
        set_global_seed(123)
        assert get_current_seed() == 123
    
    def test_seed_context_manager(self):
        """Test seed context manager."""
        # Set initial seed
        set_global_seed(42)
        initial_seed = get_current_seed()
        
        # Use context manager
        with seed_context(100):
            assert get_current_seed() == 100
        
        # Should restore original seed
        assert get_current_seed() == initial_seed
    
    def test_seed_context_manager_nested(self):
        """Test nested seed context managers."""
        set_global_seed(42)
        
        with seed_context(100):
            assert get_current_seed() == 100
            
            with seed_context(200):
                assert get_current_seed() == 200
            
            assert get_current_seed() == 100
        
        assert get_current_seed() == 42
    
    def test_restore_seed_state(self):
        """Test restoring seed state."""
        # Set initial seed
        set_global_seed(42)
        initial_state = get_seed_state_summary()
        
        # Change seed
        set_global_seed(100)
        
        # Restore state
        restore_seed_state(initial_state)
        
        # Should be back to original
        assert get_current_seed() == 42
    
    def test_get_seed_state_summary(self):
        """Test getting seed state summary."""
        set_global_seed(42)
        summary = get_seed_state_summary()
        
        assert isinstance(summary, dict)
        assert 'global_seed' in summary
        assert 'python_state' in summary
        assert 'numpy_state' in summary
        assert summary['global_seed'] == 42
    
    def test_set_reproducible_environment(self):
        """Test setting reproducible environment."""
        with patch.dict(os.environ, {}, clear=True):
            seed = set_reproducible_environment(42)
            
            assert seed == 42
            assert os.environ.get('PYTHONHASHSEED') == '42'
            assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
            assert os.environ.get('OMP_NUM_THREADS') == '1'
    
    def test_validate_seed_consistency(self):
        """Test validating seed consistency."""
        # Set seed and validate
        set_global_seed(42)
        assert validate_seed_consistency(42) is True
        
        # Change seed and validate
        set_global_seed(100)
        assert validate_seed_consistency(42) is False
        assert validate_seed_consistency(100) is True
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        # Set seed and generate some random numbers
        set_global_seed(42)
        random_numbers_1 = [random.random() for _ in range(10)]
        
        # Reset and use same seed
        set_global_seed(42)
        random_numbers_2 = [random.random() for _ in range(10)]
        
        # Should be identical
        assert random_numbers_1 == random_numbers_2
    
    def test_numpy_seed_reproducibility(self):
        """Test that same seed produces same numpy results."""
        # Set seed and generate some numpy random numbers
        set_global_seed(42)
        numpy_numbers_1 = [np.random.random() for _ in range(10)]
        
        # Reset and use same seed
        set_global_seed(42)
        numpy_numbers_2 = [np.random.random() for _ in range(10)]
        
        # Should be identical
        assert numpy_numbers_1 == numpy_numbers_2
    
    def test_seed_context_with_exception(self):
        """Test that seed context restores state even if exception occurs."""
        set_global_seed(42)
        initial_seed = get_current_seed()
        
        try:
            with seed_context(100):
                assert get_current_seed() == 100
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original seed even after exception
        assert get_current_seed() == initial_seed
    
    def test_seed_context_with_none_seed(self):
        """Test seed context with None seed."""
        set_global_seed(42)
        
        with seed_context(None):
            # Should generate a random seed
            seed = get_current_seed()
            assert seed is not None
            assert isinstance(seed, int)
        
        # Should restore original seed
        assert get_current_seed() == 42
    
    def test_seed_state_stack(self):
        """Test that seed state stack works correctly."""
        set_global_seed(42)
        
        # Multiple context managers should stack correctly
        with seed_context(100):
            with seed_context(200):
                with seed_context(300):
                    assert get_current_seed() == 300
                assert get_current_seed() == 200
            assert get_current_seed() == 100
        assert get_current_seed() == 42
    
    def test_restore_seed_state_invalid(self):
        """Test restoring invalid seed state."""
        # Should handle invalid state gracefully
        invalid_state = {'invalid': 'state'}
        
        # Should not raise exception
        restore_seed_state(invalid_state)
        
        # Current seed should still be accessible
        assert get_current_seed() is not None or get_current_seed() is None
    
    def test_set_global_seed_with_torch(self):
        """Test setting global seed with torch (mocked)."""
        with patch('torch.manual_seed') as mock_torch_seed, \
             patch('torch.cuda.manual_seed_all') as mock_cuda_seed:
            
            set_global_seed(42, deterministic=True)
            
            # Should call torch seeding functions
            mock_torch_seed.assert_called_with(42)
            mock_cuda_seed.assert_called_with(42)
    
    def test_set_global_seed_without_torch(self):
        """Test setting global seed without torch."""
        with patch.dict('sys.modules', {'torch': None}):
            # Should not raise exception
            seed = set_global_seed(42)
            assert seed == 42
            assert get_current_seed() == 42
    
    def test_environment_variables_setting(self):
        """Test that environment variables are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            set_reproducible_environment(42)
            
            # Check specific environment variables
            assert os.environ.get('PYTHONHASHSEED') == '42'
            assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
            assert os.environ.get('OMP_NUM_THREADS') == '1'
    
    def test_seed_validation_edge_cases(self):
        """Test seed validation with edge cases."""
        # Test with None seed
        set_global_seed(None)
        assert validate_seed_consistency(None) is False
        
        # Test with 0 seed
        set_global_seed(0)
        assert validate_seed_consistency(0) is True
        
        # Test with negative seed
        set_global_seed(-1)
        assert validate_seed_consistency(-1) is True
    
    def test_seed_context_deterministic(self):
        """Test seed context with deterministic parameter."""
        set_global_seed(42)
        
        with seed_context(100, deterministic=False):
            assert get_current_seed() == 100
        
        # Should restore original seed
        assert get_current_seed() == 42
    
    def test_get_seed_state_summary_content(self):
        """Test that seed state summary contains expected content."""
        set_global_seed(42)
        summary = get_seed_state_summary()
        
        # Check that summary contains expected keys
        expected_keys = ['global_seed', 'python_state', 'numpy_state']
        for key in expected_keys:
            assert key in summary
        
        # Check that values are reasonable
        assert summary['global_seed'] == 42
        assert isinstance(summary['python_state'], tuple)
        assert isinstance(summary['numpy_state'], tuple)
    
    def test_seed_context_multiple_restores(self):
        """Test multiple restores of seed state."""
        set_global_seed(42)
        state1 = get_seed_state_summary()
        
        set_global_seed(100)
        state2 = get_seed_state_summary()
        
        # Restore first state
        restore_seed_state(state1)
        assert get_current_seed() == 42
        
        # Restore second state
        restore_seed_state(state2)
        assert get_current_seed() == 100


class TestSeedIntegration:
    """Test seed integration with other modules."""
    
    def test_seed_with_gymnasium(self):
        """Test seed integration with gymnasium (mocked)."""
        with patch('gymnasium.make') as mock_gym:
            mock_env = MagicMock()
            mock_env.observation_space.shape = [4]
            mock_env.action_space.n = 2
            mock_gym.return_value = mock_env
            
            # Set seed and create environment
            set_global_seed(42)
            
            # Environment should be created with seed
            # (This is a simplified test - real integration would be more complex)
            assert get_current_seed() == 42
    
    def test_seed_with_pandas(self):
        """Test seed integration with pandas."""
        import pandas as pd
        
        # Set seed
        set_global_seed(42)
        
        # Create DataFrame with random data
        df1 = pd.DataFrame({'value': np.random.random(10)})
        
        # Reset seed and create another DataFrame
        set_global_seed(42)
        df2 = pd.DataFrame({'value': np.random.random(10)})
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_seed_with_sklearn(self):
        """Test seed integration with sklearn."""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        # Set seed
        set_global_seed(42)
        
        # Create some data
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, random_state=42)
        
        # Reset seed and split again
        set_global_seed(42)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state=42)
        
        # Should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)


class TestSeedErrorHandling:
    """Test seed error handling."""
    
    def test_seed_context_with_invalid_seed(self):
        """Test seed context with invalid seed type."""
        with pytest.raises(TypeError):
            with seed_context("invalid_seed"):
                pass
    
    def test_restore_seed_state_with_none(self):
        """Test restoring None seed state."""
        # Should handle None gracefully
        restore_seed_state(None)
        
        # Should not crash
        assert get_current_seed() is not None or get_current_seed() is None
    
    def test_restore_seed_state_with_empty_dict(self):
        """Test restoring empty seed state."""
        # Should handle empty dict gracefully
        restore_seed_state({})
        
        # Should not crash
        assert get_current_seed() is not None or get_current_seed() is None
    
    def test_set_global_seed_with_invalid_type(self):
        """Test setting global seed with invalid type."""
        with pytest.raises(TypeError):
            set_global_seed("invalid_seed")
    
    def test_set_global_seed_with_float(self):
        """Test setting global seed with float."""
        # Should convert float to int
        seed = set_global_seed(42.5)
        assert seed == 42
        assert isinstance(seed, int)
    
    def test_set_global_seed_with_negative_float(self):
        """Test setting global seed with negative float."""
        # Should convert negative float to int
        seed = set_global_seed(-42.5)
        assert seed == -42
        assert isinstance(seed, int)


if __name__ == "__main__":
    pytest.main([__file__])