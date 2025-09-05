#!/usr/bin/env python3
"""Test script to verify rldk package structure after restructuring."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '/workspace/src')

def test_package_structure():
    """Test that the package structure matches the README architecture."""
    print("Testing package structure...")
    
    # Check that core modules exist
    core_modules = ['tracking', 'forensics', 'ingest', 'diff', 'determinism', 'reward', 'evals']
    
    for module in core_modules:
        module_path = Path(f'/workspace/src/rldk/{module}')
        assert module_path.exists(), f"Module {module} not found"
        assert (module_path / '__init__.py').exists(), f"Module {module} missing __init__.py"
        print(f"✓ {module} module exists")
    
    # Check that integrations is separate
    integrations_path = Path('/workspace/integrations')
    assert integrations_path.exists(), "Integrations directory not found"
    print("✓ integrations directory is separate")
    
    # Check that old modules are gone
    old_modules = ['artifacts', 'adapters', 'replay', 'bisect', 'cards']
    for module in old_modules:
        module_path = Path(f'/workspace/src/rldk/{module}')
        assert not module_path.exists(), f"Old module {module} still exists"
        print(f"✓ {module} module removed")
    
    print("✓ Package structure is correct")

def test_imports_structure():
    """Test that imports can be resolved (without executing)."""
    print("Testing import structure...")
    
    # Test main __init__.py
    init_path = Path('/workspace/src/rldk/__init__.py')
    with open(init_path) as f:
        content = f.read()
    
    # Check that main API is exported
    assert 'ExperimentTracker' in content, "ExperimentTracker not in __init__.py"
    assert 'TrackingConfig' in content, "TrackingConfig not in __init__.py"
    print("✓ Main API exported in __init__.py")
    
    # Test that module __init__.py files exist and have content
    modules = ['tracking', 'forensics', 'ingest', 'diff', 'determinism', 'reward', 'evals']
    for module in modules:
        init_path = Path(f'/workspace/src/rldk/{module}/__init__.py')
        with open(init_path) as f:
            content = f.read()
        assert len(content.strip()) > 0, f"Module {module} __init__.py is empty"
        print(f"✓ {module} __init__.py has content")
    
    print("✓ Import structure is correct")

def test_cli_structure():
    """Test that CLI is properly integrated."""
    print("Testing CLI structure...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    assert cli_path.exists(), "Main CLI file not found"
    
    with open(cli_path) as f:
        content = f.read()
    
    # Check that old CLI files are not imported
    assert 'cli_forensics' not in content, "cli_forensics still imported"
    assert 'cli_reward' not in content, "cli_reward still imported"
    print("✓ Old CLI files not imported")
    
    # Check that forensics and reward functions are directly imported
    assert 'from rldk.forensics' in content, "Forensics functions not directly imported"
    assert 'from rldk.reward' in content, "Reward functions not directly imported"
    print("✓ Forensics and reward functions directly imported")
    
    # Check that old CLI files are deleted
    assert not Path('/workspace/src/rldk/cli_forensics.py').exists(), "cli_forensics.py still exists"
    assert not Path('/workspace/src/rldk/cli_reward.py').exists(), "cli_reward.py still exists"
    print("✓ Old CLI files deleted")
    
    print("✓ CLI structure is correct")

def main():
    """Run all tests."""
    print("=== RLDK Package Structure Test ===")
    print()
    
    try:
        test_package_structure()
        print()
        test_imports_structure()
        print()
        test_cli_structure()
        print()
        print("✅ All structure tests passed!")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())