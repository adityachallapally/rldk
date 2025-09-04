#!/usr/bin/env python3
"""Test script to verify the statistical function fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rldk.evals.metrics import calculate_confidence_intervals, calculate_effect_sizes
import numpy as np

def test_confidence_intervals():
    """Test confidence interval calculations with edge cases."""
    print("Testing calculate_confidence_intervals...")
    
    # Test 1: Normal case
    scores = [1, 2, 3, 4, 5]
    result = calculate_confidence_intervals(scores)
    print(f"Normal case: {result}")
    assert 'mean' in result
    assert 'lower' in result
    assert 'upper' in result
    assert result['mean'] == 3.0
    
    # Test 2: Insufficient data
    scores = [1]
    result = calculate_confidence_intervals(scores)
    print(f"Insufficient data: {result}")
    assert 'error' in result
    assert result['mean'] == 1.0
    
    # Test 3: Identical scores
    scores = [5, 5, 5, 5]
    result = calculate_confidence_intervals(scores)
    print(f"Identical scores: {result}")
    assert 'note' in result
    assert result['mean'] == 5.0
    assert result['lower'] == 5.0
    assert result['upper'] == 5.0
    
    # Test 4: Empty data
    scores = []
    result = calculate_confidence_intervals(scores)
    print(f"Empty data: {result}")
    assert 'error' in result
    assert result['mean'] is None

def test_effect_sizes():
    """Test effect size calculations with edge cases."""
    print("\nTesting calculate_effect_sizes...")
    
    # Test 1: Normal case
    scores1 = [1, 2, 3, 4, 5]
    scores2 = [2, 3, 4, 5, 6]
    result = calculate_effect_sizes(scores1, scores2)
    print(f"Normal case: {result}")
    assert 'cohens_d' in result
    assert 'mann_whitney_u' in result
    assert 'p_value' in result
    assert 'error' not in result
    
    # Test 2: Identical distributions
    scores1 = [1, 2, 3, 4, 5]
    scores2 = [1, 2, 3, 4, 5]
    result = calculate_effect_sizes(scores1, scores2)
    print(f"Identical distributions: {result}")
    assert result['cohens_d'] is None  # Should be None when pooled_std = 0
    assert 'mann_whitney_u' in result
    assert 'p_value' in result
    
    # Test 3: Insufficient data
    scores1 = [1]
    scores2 = [2]
    result = calculate_effect_sizes(scores1, scores2)
    print(f"Insufficient data: {result}")
    assert 'error' in result
    assert result['cohens_d'] is None
    
    # Test 4: One group has no variance
    scores1 = [1, 1, 1, 1, 1]
    scores2 = [2, 3, 4, 5, 6]
    result = calculate_effect_sizes(scores1, scores2)
    print(f"One group no variance: {result}")
    assert result['cohens_d'] is None  # Should be None when pooled_std = 0
    assert 'mann_whitney_u' in result
    assert 'p_value' in result

if __name__ == "__main__":
    print("Running statistical function tests...")
    test_confidence_intervals()
    test_effect_sizes()
    print("\n✅ All tests passed! Statistical functions are working correctly.")