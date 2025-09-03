#!/usr/bin/env python3
"""
Test script for RLDK Ultimate Post-Training Features
"""

import os
import json
import tempfile
from pathlib import Path

def create_test_logs():
    """Create simple test logs"""
    test_data = [
        {"step": 1, "loss": 2.0, "reward": 0.5, "kl_divergence": 0.01},
        {"step": 2, "loss": 1.9, "reward": 0.6, "kl_divergence": 0.02},
        {"step": 3, "loss": 1.8, "reward": 0.7, "kl_divergence": 0.01},
        {"step": 4, "loss": 1.7, "reward": 0.8, "kl_divergence": 0.01},
        {"step": 5, "loss": 1.6, "reward": 0.9, "kl_divergence": 0.01},
    ]
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    log_file = temp_dir / "test_training.jsonl"
    
    with open(log_file, 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
    
    return temp_dir

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from rldk import (
            UniversalMonitor, start_monitoring,
            AnomalyDetector, detect_anomalies, detect_training_anomalies,
            TrainingDebugger, debug_training, quick_debug
        )
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_universal_monitor():
    """Test universal monitor"""
    print("\nTesting Universal Monitor...")
    
    try:
        from rldk import UniversalMonitor
        
        # Create test logs
        test_dir = create_test_logs()
        
        # Test monitor
        monitor = UniversalMonitor()
        framework = monitor.auto_detect_framework(test_dir)
        
        print(f"✅ Auto-detected framework: {framework}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"❌ Universal Monitor test failed: {e}")
        return False

def test_anomaly_detector():
    """Test anomaly detector"""
    print("\nTesting Anomaly Detector...")
    
    try:
        from rldk import detect_training_anomalies
        
        # Create test logs
        test_dir = create_test_logs()
        
        # Test anomaly detection
        report = detect_training_anomalies(test_dir)
        
        print(f"✅ Anomaly detection successful!")
        print(f"   Anomalies detected: {report.anomalies_detected}")
        print(f"   Anomaly count: {report.anomaly_count}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"❌ Anomaly Detector test failed: {e}")
        return False

def test_debug_training():
    """Test debug training"""
    print("\nTesting Debug Training...")
    
    try:
        from rldk import quick_debug
        
        # Create test logs
        test_dir = create_test_logs()
        
        # Test quick debug
        result = quick_debug(test_dir)
        
        print(f"✅ Quick debug successful!")
        print(f"   Framework: {result.get('framework', 'Unknown')}")
        print(f"   Issues found: {len(result.get('issues', []))}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"❌ Debug Training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing RLDK Ultimate Post-Training Features")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_universal_monitor,
        test_anomaly_detector,
        test_debug_training,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RLDK Ultimate Features are working!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)