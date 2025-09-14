#!/usr/bin/env python3
"""
Example usage of the improved RLDK divergence detection system.

This example shows how to use the enhanced divergence detection with:
- More sensitive parameters
- Debug mode for troubleshooting
- Better detection of gradual divergences
"""

import sys
import os

# Add the src directory to the path so we can import rldk
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Note: This example assumes pandas and numpy are available
# In a real environment, you would install them with: pip install pandas numpy

try:
    import pandas as pd
    import numpy as np
    from rldk.diff.diff import first_divergence
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install pandas numpy")
    sys.exit(1)

def create_test_data():
    """Create test data that should trigger divergence detection."""
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run A: Stable training
    run_a = pd.DataFrame({
        'step': range(1000),
        'reward_mean': [0.5 + (i/1000) * 0.3 + np.random.normal(0, 0.1) for i in range(1000)],
        'loss': [0.5 * np.exp(-i/500) + np.random.normal(0, 0.05) for i in range(1000)]
    })

    # Run B: Diverges at step 300
    run_b = pd.DataFrame({
        'step': range(1000),
        'reward_mean': [0.5 + (i/1000) * 0.3 + np.random.normal(0, 0.1) if i < 300 
                       else 0.5 + (i/1000) * 0.3 - (i-300)/100 * 0.2 + np.random.normal(0, 0.15) 
                       for i in range(1000)],
        'loss': [0.5 * np.exp(-i/500) + np.random.normal(0, 0.05) if i < 300
                else 0.5 * np.exp(-i/500) + (i-300)/100 * 0.1 + np.random.normal(0, 0.08)
                for i in range(1000)]
    })
    
    return run_a, run_b

def test_improved_detection():
    """Test the improved divergence detection system."""
    
    print("RLDK Divergence Detection - Improved System Test")
    print("=" * 60)
    
    # Create test data
    run_a, run_b = create_test_data()
    
    print(f"Created test data:")
    print(f"- Run A: {run_a.shape[0]} steps")
    print(f"- Run B: {run_b.shape[0]} steps")
    print(f"- Signals: {list(run_a.columns)}")
    
    # Test with improved parameters and debugging
    print(f"\nTesting with improved parameters:")
    print(f"- k_consecutive: 2 (reduced from 3 for more sensitivity)")
    print(f"- window: 20 (reduced from 50 for more responsiveness)")
    print(f"- tolerance: 1.5 (reduced from 2.0 for more sensitivity)")
    print(f"- debug: True (for detailed analysis)")
    
    result = first_divergence(
        run_a, run_b, 
        signals=['reward_mean', 'loss'], 
        k_consecutive=2, 
        window=20, 
        tolerance=1.5, 
        debug=True
    )
    
    # Display results
    print(f"\nResults:")
    print(f"=" * 30)
    print(f"Diverged: {result.diverged}")
    print(f"First divergence step: {result.first_step}")
    print(f"Tripped signals: {len(result.tripped_signals)}")
    
    if result.notes:
        print(f"\nNotes:")
        for note in result.notes:
            print(f"- {note}")
    
    if result.suspected_causes:
        print(f"\nSuspected causes:")
        for cause in result.suspected_causes:
            print(f"- {cause}")
    
    # Display debug information
    if hasattr(result, 'debug_info') and result.debug_info:
        print(f"\nDebug Information:")
        print(f"=" * 30)
        
        params = result.debug_info.get('parameters', {})
        print(f"Parameters used: {params}")
        
        print(f"\nData analysis:")
        print(f"- Common steps: {result.debug_info.get('common_steps_count', 'N/A')}")
        print(f"- Total steps A: {result.debug_info.get('total_steps_a', 'N/A')}")
        print(f"- Total steps B: {result.debug_info.get('total_steps_b', 'N/A')}")
        
        z_scores = result.debug_info.get('z_scores', {})
        if z_scores:
            print(f"\nZ-score statistics:")
            for signal, stats in z_scores.items():
                print(f"- {signal}:")
                print(f"  Mean: {stats.get('mean', 'N/A'):.3f}")
                print(f"  Std:  {stats.get('std', 'N/A'):.3f}")
                print(f"  Max:  {stats.get('max', 'N/A'):.3f}")
                print(f"  Min:  {stats.get('min', 'N/A'):.3f}")
                print(f"  Valid: {stats.get('valid_count', 'N/A')}")
        
        violations = result.debug_info.get('violations', {})
        if violations:
            print(f"\nViolation counts:")
            for signal, count in violations.items():
                print(f"- {signal}: {count} violations")
    
    # Display divergence details
    if result.details is not None and not result.details.empty:
        print(f"\nDivergence Details:")
        print(f"=" * 30)
        print(result.details.to_string(index=False))
    else:
        print(f"\nNo detailed divergence information available")
    
    # Check actual differences around the expected divergence point
    print(f"\nActual Differences Around Step 300:")
    print(f"=" * 40)
    for step in [295, 300, 305, 310, 320]:
        if step < len(run_a) and step < len(run_b):
            reward_diff = run_a.loc[step, 'reward_mean'] - run_b.loc[step, 'reward_mean']
            loss_diff = run_a.loc[step, 'loss'] - run_b.loc[step, 'loss']
            print(f"Step {step:3d}: reward_diff={reward_diff:7.4f}, loss_diff={loss_diff:7.4f}")
    
    return result

def demonstrate_parameter_sensitivity():
    """Demonstrate how different parameters affect sensitivity."""
    
    print(f"\n" + "=" * 60)
    print("PARAMETER SENSITIVITY DEMONSTRATION")
    print("=" * 60)
    
    run_a, run_b = create_test_data()
    
    # Test different parameter combinations
    test_configs = [
        {"k_consecutive": 1, "window": 10, "tolerance": 1.0, "name": "Very Sensitive"},
        {"k_consecutive": 2, "window": 20, "tolerance": 1.5, "name": "Balanced (Recommended)"},
        {"k_consecutive": 3, "window": 30, "tolerance": 2.0, "name": "Conservative"},
        {"k_consecutive": 5, "window": 50, "tolerance": 2.5, "name": "Very Conservative"},
    ]
    
    for config in test_configs:
        name = config.pop("name")
        print(f"\n{name}:")
        print(f"  k_consecutive={config['k_consecutive']}, window={config['window']}, tolerance={config['tolerance']}")
        
        result = first_divergence(run_a, run_b, signals=['reward_mean', 'loss'], **config)
        print(f"  Result: Diverged={result.diverged}, First step={result.first_step}")

if __name__ == "__main__":
    try:
        result = test_improved_detection()
        demonstrate_parameter_sensitivity()
        
        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if result.diverged:
            print("✅ SUCCESS: Divergence was detected!")
            print(f"   First divergence detected at step {result.first_step}")
            print("   The improved system is working correctly.")
        else:
            print("⚠️  Divergence was not detected.")
            print("   Check the debug information above to understand why.")
            print("   You may need to adjust parameters further or check your data.")
        
        print(f"\nKey improvements made:")
        print(f"1. More sensitive parameters (tolerance: 2.0→1.5, k_consecutive: 3→2)")
        print(f"2. Added relative change detection for gradual divergences")
        print(f"3. Comprehensive debugging and logging")
        print(f"4. Better edge case handling")
        print(f"5. Detailed debug information in reports")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()