#!/usr/bin/env python3
"""Test script to demonstrate the monitor functionality."""

import json
import time
import random
from pathlib import Path
from datetime import datetime

# Import the standalone modules
import sys
sys.path.append('/workspace/examples')

from standalone_monitor import MonitorEngine, Event


def create_test_data():
    """Create test JSONL data."""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    test_file = artifacts_dir / "test_data.jsonl"
    
    print("Creating test data...")
    with open(test_file, "w") as f:
        for step in range(20):
            # Create events that will trigger alerts
            kl = 0.1 + (step * 0.02)  # Gradually increasing KL
            reward = random.uniform(-1.0, 1.0)
            grad_norm = random.uniform(0.5, 2.0)
            
            # KL event
            kl_event = {
                "time": datetime.utcnow().isoformat() + "Z",
                "step": step,
                "name": "kl",
                "value": kl,
                "run_id": "test-run"
            }
            f.write(json.dumps(kl_event) + "\n")
            
            # Reward event
            reward_event = {
                "time": datetime.utcnow().isoformat() + "Z",
                "step": step,
                "name": "reward",
                "value": reward,
                "run_id": "test-run"
            }
            f.write(json.dumps(reward_event) + "\n")
            
            # Gradient norm event
            grad_event = {
                "time": datetime.utcnow().isoformat() + "Z",
                "step": step,
                "name": "grad_norm",
                "value": grad_norm,
                "run_id": "test-run"
            }
            f.write(json.dumps(grad_event) + "\n")
    
    print(f"Created test data: {test_file}")
    return test_file


def test_batch_monitoring():
    """Test batch monitoring."""
    print("\n=== Testing Batch Monitoring ===")
    
    # Create test data
    test_file = create_test_data()
    
    # Run batch monitoring
    engine = MonitorEngine("examples/rules.yaml", "artifacts/test_alerts.jsonl")
    
    print("Processing events...")
    with open(test_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                event = Event(data)
                engine.process_event(event)
    
    print("Batch monitoring completed!")
    
    # Check results
    alerts_file = Path("artifacts/test_alerts.jsonl")
    if alerts_file.exists():
        with open(alerts_file, "r") as f:
            alerts = [json.loads(line) for line in f if line.strip()]
        
        print(f"Generated {len(alerts)} alerts:")
        for alert in alerts[:5]:  # Show first 5 alerts
            print(f"  {alert['rule_id']}: {alert['message']}")
        
        if len(alerts) > 5:
            print(f"  ... and {len(alerts) - 5} more alerts")
    
    return len(alerts) if alerts_file.exists() else 0


def test_field_mapping():
    """Test field mapping functionality."""
    print("\n=== Testing Field Mapping ===")
    
    # Create test data with different field names
    test_file = Path("artifacts/test_mapped_data.jsonl")
    
    with open(test_file, "w") as f:
        for step in range(10):
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "s": step,  # step mapped to 's'
                "metric": "kl",  # name mapped to 'metric'
                "v": 0.4 + (step * 0.01),  # value mapped to 'v'
                "run": "mapped-test"
            }
            f.write(json.dumps(event) + "\n")
    
    print(f"Created mapped test data: {test_file}")
    
    # Test with field mapping
    field_map = {"s": "step", "metric": "name", "v": "value"}
    
    engine = MonitorEngine("examples/rules.yaml", "artifacts/mapped_alerts.jsonl")
    
    print("Processing mapped events...")
    with open(test_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Apply field mapping
                mapped_data = {}
                for key, value in data.items():
                    mapped_key = field_map.get(key, key)
                    mapped_data[mapped_key] = value
                
                event = Event(mapped_data)
                engine.process_event(event)
    
    print("Field mapping test completed!")
    
    # Check results
    alerts_file = Path("artifacts/mapped_alerts.jsonl")
    if alerts_file.exists():
        with open(alerts_file, "r") as f:
            alerts = [json.loads(line) for line in f if line.strip()]
        print(f"Generated {len(alerts)} alerts with field mapping")
    
    return len(alerts) if alerts_file.exists() else 0


def main():
    """Main test function."""
    print("RLDK Monitor Core Functionality Test")
    print("=" * 50)
    
    # Test batch monitoring
    batch_alerts = test_batch_monitoring()
    
    # Test field mapping
    mapped_alerts = test_field_mapping()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Batch monitoring: {batch_alerts} alerts generated")
    print(f"Field mapping: {mapped_alerts} alerts generated")
    
    if batch_alerts > 0 and mapped_alerts > 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    print("\nGenerated files:")
    for file in Path("artifacts").glob("*.jsonl"):
        print(f"  {file}")


if __name__ == "__main__":
    main()