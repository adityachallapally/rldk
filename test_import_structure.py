#!/usr/bin/env python3
"""Test script to verify import structure is correct."""

import sys
import re
from pathlib import Path

def test_main_init_imports():
    """Test that main __init__.py has correct imports."""
    print("Testing main __init__.py imports...")
    
    init_path = Path('/workspace/src/rldk/__init__.py')
    with open(init_path) as f:
        content = f.read()
    
    # Check main public API imports
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
    
    # Check __all__ exports
    all_pattern = r'__all__\s*=\s*\[(.*?)\]'
    match = re.search(all_pattern, content, re.DOTALL)
    if match:
        all_content = match.group(1)
        expected_exports = [
            'ExperimentTracker',
            'TrackingConfig',
            'ingest_runs',
            'first_divergence',
            'check',
            'bisect_commits',
            'health',
            'RewardHealthReport',
            'run',
            'EvalResult'
        ]
        
        for export in expected_exports:
            if export in all_content:
                print(f"✓ {export} in __all__")
            else:
                print(f"❌ Missing from __all__: {export}")
                return False
    else:
        print("❌ __all__ not found")
        return False
    
    print("✓ Main __init__.py imports correct")
    return True

def test_module_init_files():
    """Test that module __init__.py files have correct imports."""
    print("Testing module __init__.py files...")
    
    modules = {
        'tracking': ['ExperimentTracker', 'TrackingConfig'],
        'forensics': ['scan_ppo_events', 'audit_environment', 'scan_logs', 'diff_checkpoints'],
        'ingest': ['ingest_runs', 'BaseAdapter', 'TRLAdapter', 'OpenRLHFAdapter', 'WandBAdapter', 'CustomJSONLAdapter'],
        'diff': ['first_divergence', 'bisect_commits', 'BisectResult', 'generate_drift_card'],
        'determinism': ['check', 'replay', 'ReplayReport', 'generate_determinism_card'],
        'reward': ['health', 'RewardHealthReport', 'generate_reward_card'],
        'evals': ['run', 'EvalResult']
    }
    
    for module_name, expected_exports in modules.items():
        init_path = Path(f'/workspace/src/rldk/{module_name}/__init__.py')
        with open(init_path) as f:
            content = f.read()
        
        for export in expected_exports:
            if export in content:
                print(f"✓ {module_name}.{export}")
            else:
                print(f"❌ Missing from {module_name}: {export}")
                return False
    
    print("✓ All module __init__.py files correct")
    return True

def test_import_paths():
    """Test that import paths are correct after restructuring."""
    print("Testing import paths...")
    
    # Check that moved files exist and have some imports
    moved_files = [
        'src/rldk/forensics/env_audit.py',
        'src/rldk/forensics/log_scan.py', 
        'src/rldk/determinism/replay.py',
        'src/rldk/diff/bisect.py'
    ]
    
    for file_path in moved_files:
        full_path = Path(f'/workspace/{file_path}')
        if full_path.exists():
            with open(full_path) as f:
                content = f.read()
            
            # Check that file has some imports
            if 'import' in content:
                print(f"✓ {file_path} has imports")
            else:
                print(f"❌ {file_path} has no imports")
                return False
        else:
            print(f"❌ {file_path} not found")
            return False
    
    print("✓ All import paths correct")
    return True

def test_cli_imports():
    """Test that CLI has correct imports."""
    print("Testing CLI imports...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check that CLI imports are correct
    expected_cli_imports = [
        'from rldk.forensics.ckpt_diff import diff_checkpoints',
        'from rldk.forensics.env_audit import audit_environment',
        'from rldk.forensics.log_scan import scan_logs',
        'from rldk.reward.drift import compare_models',
        'from rldk.determinism import replay',
        'from rldk.diff import bisect_commits'
    ]
    
    for import_line in expected_cli_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing from CLI: {import_line}")
            return False
    
    # Check that old imports are not present
    old_imports = [
        'from rldk.cli_forensics import',
        'from rldk.cli_reward import',
        'from rldk.artifacts import',
        'from rldk.bisect import',
        'from rldk.replay import'
    ]
    
    for old_import in old_imports:
        if old_import in content:
            print(f"❌ Old import still present: {old_import}")
            return False
        else:
            print(f"✓ Old import removed: {old_import}")
    
    print("✓ CLI imports correct")
    return True

def main():
    """Run all import structure tests."""
    print("=== RLDK Import Structure Test ===")
    print()
    
    try:
        if not test_main_init_imports():
            return 1
        print()
        
        if not test_module_init_files():
            return 1
        print()
        
        if not test_import_paths():
            return 1
        print()
        
        if not test_cli_imports():
            return 1
        print()
        
        print("✅ All import structure tests passed!")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())