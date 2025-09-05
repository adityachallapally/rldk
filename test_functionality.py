#!/usr/bin/env python3
"""Comprehensive functionality test for RLDK after restructuring."""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add src to path
src_path = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(src_path))

def test_card_functions():
    """Test that card generation functions have correct signatures."""
    print("Testing card generation functions...")
    
    try:
        from rldk.determinism import generate_determinism_card
        from rldk.diff import generate_drift_card  
        from rldk.reward import generate_reward_card
        
        # Check function signatures by inspecting their code
        import inspect
        
        # Determinism card should take 3 args: events, run_path, output_dir
        det_sig = inspect.signature(generate_determinism_card)
        det_params = list(det_sig.parameters.keys())
        expected_det = ['events', 'run_path', 'output_dir']
        if det_params == expected_det:
            print("✓ generate_determinism_card has correct signature")
        else:
            print(f"❌ generate_determinism_card signature wrong: {det_params} != {expected_det}")
            return False
        
        # Drift card should take 5 args: events_a, events_b, run_a, run_b, output_dir
        drift_sig = inspect.signature(generate_drift_card)
        drift_params = list(drift_sig.parameters.keys())
        expected_drift = ['events_a', 'events_b', 'run_a', 'run_b', 'output_dir']
        if drift_params == expected_drift:
            print("✓ generate_drift_card has correct signature")
        else:
            print(f"❌ generate_drift_card signature wrong: {drift_params} != {expected_drift}")
            return False
        
        # Reward card should take 3 args: events, run_path, output_dir (SINGLE RUN!)
        reward_sig = inspect.signature(generate_reward_card)
        reward_params = list(reward_sig.parameters.keys())
        expected_reward = ['events', 'run_path', 'output_dir']
        if reward_params == expected_reward:
            print("✓ generate_reward_card has correct signature (single run)")
        else:
            print(f"❌ generate_reward_card signature wrong: {reward_params} != {expected_reward}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing card functions: {e}")
        return False

def test_cli_help_text():
    """Test that CLI help text is correct."""
    print("Testing CLI help text...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check that reward card help text doesn't mention two runs
    if 'for drift cards only' in content:
        print("✓ Help text correctly specifies drift cards only need two runs")
    else:
        print("❌ Help text doesn't specify drift cards only")
        return False
    
    # Check that reward card logic doesn't require run_b
    if 'if not run_b:' in content and 'reward cards require two runs' in content:
        print("❌ Reward card still incorrectly requires two runs")
        return False
    else:
        print("✓ Reward card correctly uses single run")
    
    return True

def test_import_consistency():
    """Test that all imports are consistent and work."""
    print("Testing import consistency...")
    
    # Test that we can import all the main modules
    modules_to_test = [
        'rldk.tracking',
        'rldk.forensics', 
        'rldk.ingest',
        'rldk.diff',
        'rldk.determinism',
        'rldk.reward',
        'rldk.evals'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"✓ {module_name} imports successfully")
        except Exception as e:
            print(f"❌ {module_name} import failed: {e}")
            return False
    
    return True

def test_moved_functions():
    """Test that moved functions are in the right places."""
    print("Testing moved functions...")
    
    # Check that forensics functions are in the right place
    forensics_path = Path('/workspace/src/rldk/forensics')
    expected_forensics_files = ['env_audit.py', 'log_scan.py', 'ckpt_diff.py']
    
    for file_name in expected_forensics_files:
        file_path = forensics_path / file_name
        if file_path.exists():
            print(f"✓ {file_name} in forensics module")
        else:
            print(f"❌ {file_name} missing from forensics module")
            return False
    
    # Check that replay is in determinism
    replay_path = Path('/workspace/src/rldk/determinism/replay.py')
    if replay_path.exists():
        print("✓ replay.py in determinism module")
    else:
        print("❌ replay.py missing from determinism module")
        return False
    
    # Check that bisect is in diff
    bisect_path = Path('/workspace/src/rldk/diff/bisect.py')
    if bisect_path.exists():
        print("✓ bisect.py in diff module")
    else:
        print("❌ bisect.py missing from diff module")
        return False
    
    return True

def test_cli_commands_exist():
    """Test that all expected CLI commands exist."""
    print("Testing CLI commands...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Commands that should exist
    expected_commands = [
        '@app.command(name="track")',
        '@app.command(name="env-audit")',
        '@app.command(name="log-scan")',
        '@app.command(name="diff-ckpt")',
        '@app.command(name="reward-drift")',
        '@app.command(name="compare-runs")',
        '@app.command(name="check-determinism")',
        '@app.command(name="replay")',
        '@app.command(name="eval")',
        '@app.command(name="doctor")',
        '@app.command(name="card")',
        '@app.command(name="version")',
        '@app.command(name="reward-health")',
        '@app.command(name="reward-health-gate")'
    ]
    
    for command in expected_commands:
        if command in content:
            print(f"✓ {command}")
        else:
            print(f"❌ Missing: {command}")
            return False
    
    return True

def test_no_old_imports():
    """Test that no old imports remain."""
    print("Testing no old imports...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Old imports that should not exist
    old_imports = [
        'from rldk.cli_forensics import',
        'from rldk.cli_reward import',
        'from rldk.artifacts import',
        'from rldk.bisect import',
        'from rldk.replay import',
        'from rldk.cards import'
    ]
    
    for old_import in old_imports:
        if old_import in content:
            print(f"❌ Old import still present: {old_import}")
            return False
        else:
            print(f"✓ Old import removed: {old_import}")
    
    return True

def main():
    """Run all functionality tests."""
    print("=== RLDK Functionality Test ===")
    print()
    
    tests = [
        test_card_functions,
        test_cli_help_text,
        test_import_consistency,
        test_moved_functions,
        test_cli_commands_exist,
        test_no_old_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print()
            else:
                print(f"❌ {test.__name__} failed")
                print()
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✅ All functionality tests passed!")
        return 0
    else:
        print("❌ Some tests failed - there are issues to fix")
        return 1

if __name__ == "__main__":
    sys.exit(main())