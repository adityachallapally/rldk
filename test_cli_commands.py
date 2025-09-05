#!/usr/bin/env python3
"""Test script to verify CLI commands are properly integrated."""

import sys
import re
from pathlib import Path

def test_cli_commands():
    """Test that all CLI commands from README are available."""
    print("Testing CLI commands...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Commands from README
    expected_commands = [
        'track',
        'env-audit', 
        'log-scan',
        'diff-ckpt',
        'reward-drift',
        'compare-runs',
        'check-determinism',
        'replay',
        'eval',
        'doctor',
        'card',
        'version'
    ]
    
    for command in expected_commands:
        # Look for the command definition
        pattern = rf'@app\.command\(name="{re.escape(command)}"\)'
        if re.search(pattern, content):
            print(f"✓ {command} command found")
        else:
            print(f"❌ {command} command missing")
            return False
    
    print("✓ All CLI commands from README are available")
    return True

def test_cli_help_text():
    """Test that CLI commands have proper help text."""
    print("Testing CLI help text...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check that commands have help text
    help_pattern = r'help="[^"]*"'
    help_matches = re.findall(help_pattern, content)
    
    if len(help_matches) > 10:  # Should have many help strings
        print(f"✓ Found {len(help_matches)} help strings")
    else:
        print(f"❌ Only found {len(help_matches)} help strings")
        return False
    
    print("✓ CLI commands have proper help text")
    return True

def test_cli_imports():
    """Test that CLI imports are correct."""
    print("Testing CLI imports...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check that forensics functions are imported directly
    forensics_imports = [
        'from rldk.forensics.ckpt_diff import diff_checkpoints',
        'from rldk.forensics.env_audit import audit_environment', 
        'from rldk.forensics.log_scan import scan_logs'
    ]
    
    for import_line in forensics_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    # Check that reward functions are imported directly
    reward_imports = [
        'from rldk.reward.drift import compare_models',
        'from rldk.reward.health_config.exit_codes import raise_on_failure',
        'from rldk.reward.health_config.config import load_config, get_legacy_thresholds'
    ]
    
    for import_line in reward_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    print("✓ CLI imports are correct")
    return True

def main():
    """Run all CLI tests."""
    print("=== RLDK CLI Integration Test ===")
    print()
    
    try:
        if not test_cli_commands():
            return 1
        print()
        
        if not test_cli_help_text():
            return 1
        print()
        
        if not test_cli_imports():
            return 1
        print()
        
        print("✅ All CLI tests passed!")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())