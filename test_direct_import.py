#!/usr/bin/env python3
"""Direct import test for monitor engine."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import directly from the module
from rldk.monitor.engine import (
    StopAction, SentinelAction, ShellAction, HttpAction, WarnAction,
    load_rules, generate_human_summary
)

def test_action_creation():
    """Test creating action instances."""
    print("Testing action creation...")
    
    # Test StopAction
    stop = StopAction(pid=12345, kill_timeout_sec=10)
    assert stop.pid == 12345
    assert stop.kill_timeout_sec == 10
    print("✓ StopAction created successfully")
    
    # Test SentinelAction
    sentinel = SentinelAction(path="/tmp/test.txt")
    assert sentinel.path == "/tmp/test.txt"
    print("✓ SentinelAction created successfully")
    
    # Test ShellAction
    shell = ShellAction(command="echo test", timeout_sec=30)
    assert shell.command == "echo test"
    assert shell.timeout_sec == 30
    print("✓ ShellAction created successfully")
    
    # Test HttpAction
    http = HttpAction(url="http://example.com", method="POST", timeout_sec=30, retries=3)
    assert http.url == "http://example.com"
    assert http.method == "POST"
    assert http.timeout_sec == 30
    assert http.retries == 3
    print("✓ HttpAction created successfully")
    
    # Test WarnAction
    warn = WarnAction(message_template="Test message")
    assert warn.message_template == "Test message"
    print("✓ WarnAction created successfully")


def test_human_summary():
    """Test human summary generation."""
    print("Testing human summary generation...")
    
    from rldk.monitor.engine import MonitorReport
    
    # Create a mock report
    report = MonitorReport(
        rules={
            "test_rule": {
                "condition": "value > 0.5",
                "window": {"size": 3, "kind": "consecutive"},
                "activations": 2,
                "first_activation": {"step": 10, "name": "test", "value": 0.6},
                "last_activation": {"step": 20, "name": "test", "value": 0.7}
            }
        },
        alerts=[
            {
                "rule_id": "test_rule",
                "action": "warn",
                "name": "test",
                "value": 0.6,
                "step": 10,
                "message": "Test alert"
            }
        ]
    )
    
    summary = generate_human_summary(report)
    assert "RLDK Monitoring Summary" in summary
    assert "test_rule" in summary
    assert "Test alert" in summary
    print("✓ Human summary generation works")


if __name__ == "__main__":
    print("Running Direct Import Tests")
    print("=" * 30)
    
    try:
        test_action_creation()
        test_human_summary()
        print("\n✓ All direct import tests passed!")
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)