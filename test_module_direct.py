#!/usr/bin/env python3
"""Direct module import test."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import the module directly without going through the package
import importlib.util

def load_module_from_path(module_name, file_path):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_engine_module():
    """Test the engine module directly."""
    print("Testing engine module directly...")
    
    # Load the engine module directly
    engine_module = load_module_from_path("engine", "src/rldk/monitor/engine.py")
    
    # Test action classes
    StopAction = engine_module.StopAction
    SentinelAction = engine_module.SentinelAction
    ShellAction = engine_module.ShellAction
    HttpAction = engine_module.HttpAction
    WarnAction = engine_module.WarnAction
    
    # Test StopAction
    stop = StopAction(pid=12345, kill_timeout_sec=10)
    assert stop.pid == 12345
    assert stop.kill_timeout_sec == 10
    print("✓ StopAction works")
    
    # Test SentinelAction
    sentinel = SentinelAction(path="/tmp/test.txt")
    assert sentinel.path == "/tmp/test.txt"
    print("✓ SentinelAction works")
    
    # Test ShellAction
    shell = ShellAction(command="echo test", timeout_sec=30)
    assert shell.command == "echo test"
    assert shell.timeout_sec == 30
    print("✓ ShellAction works")
    
    # Test HttpAction
    http = HttpAction(url="http://example.com", method="POST", timeout_sec=30, retries=3)
    assert http.url == "http://example.com"
    assert http.method == "POST"
    assert http.timeout_sec == 30
    assert http.retries == 3
    print("✓ HttpAction works")
    
    # Test WarnAction
    warn = WarnAction(message_template="Test message")
    assert warn.message_template == "Test message"
    print("✓ WarnAction works")
    
    # Test human summary function
    generate_human_summary = engine_module.generate_human_summary
    MonitorReport = engine_module.MonitorReport
    
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
    print("Running Direct Module Tests")
    print("=" * 30)
    
    try:
        test_engine_module()
        print("\n✓ All direct module tests passed!")
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)