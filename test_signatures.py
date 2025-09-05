#!/usr/bin/env python3
"""Test function signatures and logic without requiring dependencies."""

import sys
import re
from pathlib import Path

def test_card_function_signatures():
    """Test card function signatures by checking parameter names."""
    print("Testing card function signatures...")
    
    # Test determinism card - should have 3 params: events, run_path, output_dir
    det_path = Path('/workspace/src/rldk/determinism/determinism.py')
    if det_path.exists():
        with open(det_path) as f:
            content = f.read()
        
        if 'def generate_determinism_card(' in content and 'events: List[Event]' in content and 'run_path: str' in content and 'output_dir: Optional[str]' in content:
            print("✓ generate_determinism_card has correct signature (3 params)")
        else:
            print("❌ generate_determinism_card wrong signature")
            return False
    else:
        print("❌ determinism.py not found")
        return False
    
    # Test drift card - should have 5 params: events_a, events_b, run_a, run_b, output_dir
    drift_path = Path('/workspace/src/rldk/diff/drift.py')
    if drift_path.exists():
        with open(drift_path) as f:
            content = f.read()
        
        if 'def generate_drift_card(' in content and 'events_a' in content and 'events_b' in content and 'run_a' in content and 'run_b' in content and 'output_dir' in content:
            print("✓ generate_drift_card has correct signature (5 params)")
        else:
            print("❌ generate_drift_card wrong signature")
            return False
    else:
        print("❌ drift.py not found")
        return False
    
    # Test reward card - THIS IS THE CRITICAL ONE - should have 3 params: events, run_path, output_dir (SINGLE RUN!)
    reward_path = Path('/workspace/src/rldk/reward/reward.py')
    if reward_path.exists():
        with open(reward_path) as f:
            content = f.read()
        
        if 'def generate_reward_card(' in content and 'events: List[Event]' in content and 'run_path: str' in content and 'output_dir: Optional[str]' in content and 'events_a' not in content and 'events_b' not in content:
            print("✓ generate_reward_card has correct signature (3 params, single run)")
        else:
            print("❌ generate_reward_card wrong signature")
            return False
    else:
        print("❌ reward.py not found")
        return False
    
    return True

def test_cli_card_logic():
    """Test CLI card logic is correct."""
    print("Testing CLI card logic...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check determinism card logic
    if 'card_type == "determinism"' in content:
        if 'events = ingest_runs_to_events(run_a)' in content and 'generate_determinism_card(events, run_a, output_dir)' in content:
            print("✓ Determinism card uses single run correctly")
        else:
            print("❌ Determinism card logic incorrect")
            return False
    else:
        print("❌ Determinism card logic not found")
        return False
    
    # Check drift card logic
    if 'card_type == "drift"' in content:
        if 'if not run_b:' in content and 'events_a = ingest_runs_to_events(run_a)' in content and 'events_b = ingest_runs_to_events(run_b)' in content:
            print("✓ Drift card uses two runs correctly")
        else:
            print("❌ Drift card logic incorrect")
            return False
    else:
        print("❌ Drift card logic not found")
        return False
    
    # Check reward card logic - THIS IS THE CRITICAL ONE
    if 'card_type == "reward"' in content:
        # Should NOT require run_b
        if 'if not run_b:' in content and 'reward cards require two runs' in content:
            print("❌ Reward card still incorrectly requires two runs!")
            return False
        elif 'events = ingest_runs_to_events(run_a)' in content and 'generate_reward_card(events, run_a, output_dir)' in content:
            print("✓ Reward card uses single run correctly")
        else:
            print("❌ Reward card logic incorrect")
            return False
    else:
        print("❌ Reward card logic not found")
        return False
    
    return True

def test_moved_files_content():
    """Test that moved files have correct content."""
    print("Testing moved files content...")
    
    # Test that forensics files were moved correctly
    forensics_files = {
        'env_audit.py': 'def audit_environment',
        'log_scan.py': 'def scan_logs',
        'ckpt_diff.py': 'def diff_checkpoints'
    }
    
    for file_name, expected_function in forensics_files.items():
        file_path = Path(f'/workspace/src/rldk/forensics/{file_name}')
        if file_path.exists():
            with open(file_path) as f:
                content = f.read()
            if expected_function in content:
                print(f"✓ {file_name} has correct function")
            else:
                print(f"❌ {file_name} missing function: {expected_function}")
                return False
        else:
            print(f"❌ {file_name} not found")
            return False
    
    # Test that replay was moved to determinism
    replay_path = Path('/workspace/src/rldk/determinism/replay.py')
    if replay_path.exists():
        with open(replay_path) as f:
            content = f.read()
        if 'def replay(' in content:
            print("✓ replay.py has correct function")
        else:
            print("❌ replay.py missing replay function")
            return False
    else:
        print("❌ replay.py not found")
        return False
    
    # Test that bisect was moved to diff
    bisect_path = Path('/workspace/src/rldk/diff/bisect.py')
    if bisect_path.exists():
        with open(bisect_path) as f:
            content = f.read()
        if 'def bisect_commits(' in content:
            print("✓ bisect.py has correct function")
        else:
            print("❌ bisect.py missing bisect_commits function")
            return False
    else:
        print("❌ bisect.py not found")
        return False
    
    return True

def test_import_statements():
    """Test that import statements are correct."""
    print("Testing import statements...")
    
    # Test main __init__.py
    init_path = Path('/workspace/src/rldk/__init__.py')
    with open(init_path) as f:
        content = f.read()
    
    # Should import from correct modules
    expected_imports = [
        'from .tracking import ExperimentTracker, TrackingConfig',
        'from .ingest import ingest_runs',
        'from .diff import first_divergence',
        'from .determinism import check',
        'from .reward import health, RewardHealthReport',
        'from .evals import run, EvalResult',
        'from .diff import bisect_commits'
    ]
    
    for import_line in expected_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    # Test CLI imports
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Should import directly from modules, not old CLI files
    expected_cli_imports = [
        'from rldk.forensics.ckpt_diff import diff_checkpoints',
        'from rldk.forensics.env_audit import audit_environment',
        'from rldk.forensics.log_scan import scan_logs',
        'from rldk.reward.drift import compare_models'
    ]
    
    for import_line in expected_cli_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    # Should NOT import from old CLI files
    old_imports = [
        'from rldk.cli_forensics import',
        'from rldk.cli_reward import'
    ]
    
    for old_import in old_imports:
        if old_import in content:
            print(f"❌ Old import still present: {old_import}")
            return False
        else:
            print(f"✓ Old import removed: {old_import}")
    
    return True

def main():
    """Run all signature tests."""
    print("=== RLDK Signature and Logic Test ===")
    print()
    
    tests = [
        test_card_function_signatures,
        test_cli_card_logic,
        test_moved_files_content,
        test_import_statements
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
        print("✅ All signature and logic tests passed!")
        return 0
    else:
        print("❌ Some tests failed - there are issues to fix")
        return 1

if __name__ == "__main__":
    sys.exit(main())