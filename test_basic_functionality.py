#!/usr/bin/env python3
"""Basic functionality test for monitor actions."""

import json
import tempfile
from pathlib import Path

# Test the basic data structures and parsing
def test_action_classes():
    """Test that action classes can be instantiated."""
    import sys
    sys.path.insert(0, 'src')
    
    from rldk.monitor.engine import StopAction, SentinelAction, ShellAction, HttpAction, WarnAction
    
    # Test StopAction
    stop_action = StopAction(pid=12345, kill_timeout_sec=10)
    assert stop_action.pid == 12345
    assert stop_action.kill_timeout_sec == 10
    
    # Test SentinelAction
    sentinel_action = SentinelAction(path="/tmp/test.txt")
    assert sentinel_action.path == "/tmp/test.txt"
    
    # Test ShellAction
    shell_action = ShellAction(command="echo test", timeout_sec=30)
    assert shell_action.command == "echo test"
    assert shell_action.timeout_sec == 30
    
    # Test HttpAction
    http_action = HttpAction(url="http://example.com", method="POST", timeout_sec=30, retries=3)
    assert http_action.url == "http://example.com"
    assert http_action.method == "POST"
    assert http_action.timeout_sec == 30
    assert http_action.retries == 3
    
    print("✓ Action classes test passed")


def test_rules_parsing():
    """Test that rules with new actions can be parsed."""
    import sys
    sys.path.insert(0, 'src')
    
    from rldk.monitor.engine import load_rules
    
    # Create test rules
    test_rules = {
        "rules": [
            {
                "id": "test_stop",
                "condition": "value > 0.5",
                "window": {"size": 1, "kind": "consecutive"},
                "actions": [
                    {"stop": {"pid": 12345, "kill_timeout_sec": 5}}
                ]
            },
            {
                "id": "test_sentinel",
                "condition": "value < 0.0",
                "window": {"size": 1, "kind": "rolling"},
                "actions": [
                    {"sentinel": {"path": "/tmp/alert.txt"}}
                ]
            },
            {
                "id": "test_shell",
                "condition": "value > 1.0",
                "window": {"size": 1, "kind": "consecutive"},
                "actions": [
                    {"shell": {"command": "echo alert", "timeout_sec": 10}}
                ]
            },
            {
                "id": "test_http",
                "condition": "value > 2.0",
                "window": {"size": 1, "kind": "consecutive"},
                "actions": [
                    {
                        "http": {
                            "url": "http://example.com/post",
                            "method": "POST",
                            "timeout_sec": 15,
                            "retries": 2
                        }
                    }
                ]
            }
        ]
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(test_rules, f)
        temp_path = f.name
    
    try:
        # Load rules
        rules = load_rules(temp_path)
        assert len(rules) == 4, f"Expected 4 rules, got {len(rules)}"
        
        # Check that actions were parsed correctly
        for rule in rules:
            assert len(rule.actions) == 1, f"Rule {rule.id} should have 1 action"
            action = rule.actions[0]
            
            if rule.id == "test_stop":
                assert hasattr(action, 'pid'), "Stop action should have pid"
                assert action.pid == 12345
                assert action.kill_timeout_sec == 5
            elif rule.id == "test_sentinel":
                assert hasattr(action, 'path'), "Sentinel action should have path"
                assert action.path == "/tmp/alert.txt"
            elif rule.id == "test_shell":
                assert hasattr(action, 'command'), "Shell action should have command"
                assert action.command == "echo alert"
                assert action.timeout_sec == 10
            elif rule.id == "test_http":
                assert hasattr(action, 'url'), "HTTP action should have url"
                assert action.url == "http://example.com/post"
                assert action.method == "POST"
                assert action.timeout_sec == 15
                assert action.retries == 2
        
        print("✓ Rules parsing test passed")
        
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_window_kinds():
    """Test that both consecutive and rolling window kinds are supported."""
    import sys
    sys.path.insert(0, 'src')
    
    from rldk.monitor.engine import load_rules
    
    # Create test rules with different window kinds
    test_rules = {
        "rules": [
            {
                "id": "consecutive_test",
                "condition": "value > 0.5",
                "window": {"size": 3, "kind": "consecutive"},
                "actions": [{"warn": {"msg": "Consecutive test"}}]
            },
            {
                "id": "rolling_test",
                "condition": "value > 0.5",
                "window": {"size": 3, "kind": "rolling"},
                "actions": [{"warn": {"msg": "Rolling test"}}]
            }
        ]
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(test_rules, f)
        temp_path = f.name
    
    try:
        # Load rules
        rules = load_rules(temp_path)
        assert len(rules) == 2, f"Expected 2 rules, got {len(rules)}"
        
        # Check window kinds
        consecutive_rule = next(r for r in rules if r.id == "consecutive_test")
        rolling_rule = next(r for r in rules if r.id == "rolling_test")
        
        assert consecutive_rule.window_kind == "consecutive"
        assert rolling_rule.window_kind == "rolling"
        
        print("✓ Window kinds test passed")
        
    finally:
        # Clean up
        Path(temp_path).unlink()


def main():
    """Run basic functionality tests."""
    print("Running Basic Functionality Tests")
    print("=" * 40)
    
    try:
        test_action_classes()
        test_rules_parsing()
        test_window_kinds()
        
        print("\n" + "=" * 40)
        print("✓ All basic functionality tests passed!")
        
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())