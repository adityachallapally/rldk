#!/usr/bin/env python3
"""Test script to verify the fixes for the comprehensive PPO forensics bugs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_metrics_copy_fix():
    """Test that metrics copying works without crashing."""
    print("Testing metrics copy fix...")
    
    try:
        from rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics
        
        # Initialize forensics
        forensics = ComprehensivePPOForensics()
        
        # Update with data that would trigger tracker metrics
        metrics = forensics.update(
            step=0,
            kl=0.1,
            kl_coef=1.0,
            entropy=2.0,
            reward_mean=0.5,
            reward_std=0.2,
            policy_grad_norm=0.5,
            value_grad_norm=0.3,
            advantage_mean=0.0,
            advantage_std=1.0
        )
        
        # This should not crash
        assert metrics is not None
        assert len(forensics.comprehensive_metrics_history) == 1
        
        # Test multiple updates
        for i in range(5):
            forensics.update(
                step=i+1,
                kl=0.1 + 0.01 * i,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.5,
                reward_std=0.2,
                policy_grad_norm=0.5,
                value_grad_norm=0.3,
                advantage_mean=0.0,
                advantage_std=1.0
            )
        
        assert len(forensics.comprehensive_metrics_history) == 6
        print("✅ Metrics copy fix works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Metrics copy fix failed: {e}")
        return False

def test_iterator_exhaustion_fix():
    """Test that iterator exhaustion is fixed."""
    print("Testing iterator exhaustion fix...")
    
    try:
        from rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics
        
        # Create sample events
        events = [
            {"step": 0, "kl": 0.1, "kl_coef": 1.0, "entropy": 2.0, "advantage_mean": 0.0, "advantage_std": 1.0, "grad_norm_policy": 0.5, "grad_norm_value": 0.3},
            {"step": 1, "kl": 0.1, "kl_coef": 1.0, "entropy": 2.0, "advantage_mean": 0.0, "advantage_std": 1.0, "grad_norm_policy": 0.5, "grad_norm_value": 0.3},
            {"step": 2, "kl": 0.1, "kl_coef": 1.0, "entropy": 2.0, "advantage_mean": 0.0, "advantage_std": 1.0, "grad_norm_policy": 0.5, "grad_norm_value": 0.3},
        ]
        
        # Initialize forensics
        forensics = ComprehensivePPOForensics()
        
        # This should work without exhausting the iterator
        result = forensics.scan_ppo_events_comprehensive(iter(events))
        
        # Should have both original scan and comprehensive analysis
        assert "version" in result
        assert "comprehensive_analysis" in result
        assert "enhanced_version" in result
        
        # Comprehensive analysis should have processed the events
        comp_analysis = result["comprehensive_analysis"]
        assert comp_analysis["total_steps"] == 3
        
        print("✅ Iterator exhaustion fix works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Iterator exhaustion fix failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Comprehensive PPO Forensics Fixes")
    print("=" * 50)
    
    tests = [
        test_metrics_copy_fix,
        test_iterator_exhaustion_fix,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All fixes work correctly!")
        return True
    else:
        print("❌ Some fixes failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)