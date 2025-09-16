#!/usr/bin/env python3
"""Acceptance tests for monitor actions and robust outputs."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


def test_auto_stop_by_pid():
    """Test auto-stop functionality using PID signaling."""
    print("Testing auto-stop by PID...")
    
    # Start the minimal streaming loop
    loop_process = subprocess.Popen(
        [sys.executable, "examples/minimal_streaming_loop.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Get the PID from the output
    pid_line = loop_process.stdout.readline()
    assert "PID:" in pid_line
    pid = int(pid_line.split(":")[1].strip())
    print(f"Started loop with PID: {pid}")
    
    # Wait a moment for the loop to start
    time.sleep(2)
    
    # Start monitoring with stop action
    monitor_process = subprocess.Popen([
        sys.executable, "-m", "rldk", "monitor",
        "--stream", "artifacts/run.jsonl",
        "--rules", "examples/rules.yaml",
        "--pid", str(pid),
        "--alerts", "artifacts/alerts.jsonl",
        "--summary", "artifacts/summary.txt"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for the loop to be stopped by monitoring
    try:
        loop_process.wait(timeout=30)
        print("Loop process terminated (likely by monitoring)")
    except subprocess.TimeoutExpired:
        print("Loop process did not terminate within timeout")
        loop_process.terminate()
        loop_process.wait()
        assert False, "Loop should have been stopped by monitoring"
    
    # Stop monitoring
    monitor_process.terminate()
    monitor_process.wait()
    
    # Check that alerts were generated
    alerts_file = Path("artifacts/alerts.jsonl")
    assert alerts_file.exists(), "Alerts file should exist"
    
    alerts = []
    with alerts_file.open() as f:
        for line in f:
            alerts.append(json.loads(line.strip()))
    
    print(f"Generated {len(alerts)} alerts")
    assert len(alerts) > 0, "Should have generated alerts"
    
    # Check for stop action
    stop_alerts = [a for a in alerts if a.get("action") == "stop"]
    assert len(stop_alerts) > 0, "Should have stop action alerts"
    
    # Check that stop action was successful
    for alert in stop_alerts:
        if alert.get("action_result"):
            result = alert["action_result"]
            assert result.get("success"), f"Stop action should succeed: {result}"
    
    print("✓ Auto-stop by PID test passed")


def test_sentinel_file_creation():
    """Test sentinel file creation action."""
    print("Testing sentinel file creation...")
    
    # Clean up any existing sentinel file
    sentinel_file = Path("artifacts/low_reward_alert.txt")
    if sentinel_file.exists():
        sentinel_file.unlink()
    
    # Start the minimal streaming loop
    loop_process = subprocess.Popen(
        [sys.executable, "examples/minimal_streaming_loop.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Get the PID from the output
    pid_line = loop_process.stdout.readline()
    pid = int(pid_line.split(":")[1].strip())
    
    # Wait a moment for the loop to start
    time.sleep(2)
    
    # Start monitoring
    monitor_process = subprocess.Popen([
        sys.executable, "-m", "rldk", "monitor",
        "--stream", "artifacts/run.jsonl",
        "--rules", "examples/rules.yaml",
        "--alerts", "artifacts/alerts.jsonl"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for sentinel file to be created
    max_wait = 30
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if sentinel_file.exists():
            print("✓ Sentinel file created")
            break
        time.sleep(0.5)
    else:
        print("✗ Sentinel file was not created within timeout")
        loop_process.terminate()
        monitor_process.terminate()
        assert False, "Sentinel file should have been created"
    
    # Check sentinel file content
    content = sentinel_file.read_text()
    assert "Alert triggered" in content, "Sentinel file should contain alert message"
    
    # Clean up
    loop_process.terminate()
    monitor_process.terminate()
    loop_process.wait()
    monitor_process.wait()
    
    print("✓ Sentinel file creation test passed")


def test_failed_action_logging():
    """Test that failed actions are properly logged."""
    print("Testing failed action logging...")
    
    # Create a rule with a failing HTTP action
    failing_rules = {
        "rules": [
            {
                "id": "failing_http",
                "where": "name == 'kl'",
                "condition": "value > 0.1",
                "window": {"size": 1, "kind": "consecutive"},
                "actions": [
                    {
                        "http": {
                            "url": "http://nonexistent-domain-12345.com/post",
                            "method": "POST",
                            "timeout_sec": 1,
                            "retries": 1
                        }
                    }
                ]
            }
        ]
    }
    
    # Write failing rules to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(failing_rules, f)
        failing_rules_path = f.name
    
    try:
        # Start the minimal streaming loop
        loop_process = subprocess.Popen(
            [sys.executable, "examples/minimal_streaming_loop.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for the loop to start
        time.sleep(2)
        
        # Start monitoring with failing rules
        monitor_process = subprocess.Popen([
            sys.executable, "-m", "rldk", "monitor",
            "--stream", "artifacts/run.jsonl",
            "--rules", failing_rules_path,
            "--alerts", "artifacts/alerts.jsonl"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for some alerts to be generated
        time.sleep(10)
        
        # Stop processes
        loop_process.terminate()
        monitor_process.terminate()
        loop_process.wait()
        monitor_process.wait()
        
        # Check that failed actions are logged
        alerts_file = Path("artifacts/alerts.jsonl")
        if alerts_file.exists():
            alerts = []
            with alerts_file.open() as f:
                for line in f:
                    alerts.append(json.loads(line.strip()))
            
            failed_actions = []
            for alert in alerts:
                if alert.get("action_result") and not alert["action_result"].get("success"):
                    failed_actions.append(alert)
            
            print(f"Found {len(failed_actions)} failed actions")
            assert len(failed_actions) > 0, "Should have logged failed actions"
            
            for alert in failed_actions:
                result = alert["action_result"]
                assert "error" in result, "Failed action should have error message"
                print(f"Failed action error: {result['error']}")
        
        print("✓ Failed action logging test passed")
        
    finally:
        # Clean up temp file
        os.unlink(failing_rules_path)


def test_rolling_windows():
    """Test rolling window functionality."""
    print("Testing rolling windows...")
    
    # Create a rule with rolling window
    rolling_rules = {
        "rules": [
            {
                "id": "rolling_test",
                "where": "name == 'reward'",
                "condition": "mean(value) > 0.0",
                "window": {"size": 5, "kind": "rolling"},
                "actions": [
                    {
                        "warn": {
                            "msg": "Rolling mean reward {mean(value):.3f} > 0.0 at step {step}"
                        }
                    }
                ]
            }
        ]
    }
    
    # Write rolling rules to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(rolling_rules, f)
        rolling_rules_path = f.name
    
    try:
        # Start the minimal streaming loop
        loop_process = subprocess.Popen(
            [sys.executable, "examples/minimal_streaming_loop.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for the loop to start
        time.sleep(2)
        
        # Start monitoring with rolling rules
        monitor_process = subprocess.Popen([
            sys.executable, "-m", "rldk", "monitor",
            "--stream", "artifacts/run.jsonl",
            "--rules", rolling_rules_path,
            "--alerts", "artifacts/alerts.jsonl"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for some alerts to be generated
        time.sleep(15)
        
        # Stop processes
        loop_process.terminate()
        monitor_process.terminate()
        loop_process.wait()
        monitor_process.wait()
        
        # Check that rolling window alerts were generated
        alerts_file = Path("artifacts/alerts.jsonl")
        if alerts_file.exists():
            alerts = []
            with alerts_file.open() as f:
                for line in f:
                    alerts.append(json.loads(line.strip()))
            
            rolling_alerts = [a for a in alerts if a.get("rule_id") == "rolling_test"]
            print(f"Generated {len(rolling_alerts)} rolling window alerts")
            # Rolling windows should trigger more frequently than consecutive windows
            assert len(rolling_alerts) > 0, "Should have generated rolling window alerts"
        
        print("✓ Rolling windows test passed")
        
    finally:
        # Clean up temp file
        os.unlink(rolling_rules_path)


def test_human_summary():
    """Test human-readable summary generation."""
    print("Testing human summary generation...")
    
    # Start the minimal streaming loop
    loop_process = subprocess.Popen(
        [sys.executable, "examples/minimal_streaming_loop.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment for the loop to start
    time.sleep(2)
    
    # Start monitoring with summary output
    monitor_process = subprocess.Popen([
        sys.executable, "-m", "rldk", "monitor",
        "--stream", "artifacts/run.jsonl",
        "--rules", "examples/rules.yaml",
        "--alerts", "artifacts/alerts.jsonl",
        "--summary", "artifacts/summary.txt"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for some alerts to be generated
    time.sleep(10)
    
    # Stop processes
    loop_process.terminate()
    monitor_process.terminate()
    loop_process.wait()
    monitor_process.wait()
    
    # Check that human summary was generated
    summary_file = Path("artifacts/summary.txt")
    assert summary_file.exists(), "Human summary file should exist"
    
    content = summary_file.read_text()
    assert "RLDK Monitoring Summary" in content, "Summary should have header"
    assert "Rules:" in content, "Summary should have rules section"
    assert "Alerts:" in content, "Summary should have alerts section"
    
    print("✓ Human summary test passed")


def main():
    """Run all acceptance tests."""
    print("Running RLDK Monitor Actions Acceptance Tests")
    print("=" * 50)
    
    # Ensure artifacts directory exists
    Path("artifacts").mkdir(exist_ok=True)
    
    try:
        test_auto_stop_by_pid()
        test_sentinel_file_creation()
        test_failed_action_logging()
        test_rolling_windows()
        test_human_summary()
        
        print("\n" + "=" * 50)
        print("✓ All acceptance tests passed!")
        
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()