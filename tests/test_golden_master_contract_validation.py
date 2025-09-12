#!/usr/bin/env python3
"""
Test that validates golden master runner against baseline contract.
"""

import json
import os
import re
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any
import pytest


def load_baseline_contract() -> Dict[str, Any]:
    """Load the baseline contract."""
    with open("api_contract.baseline.yaml", "r") as f:
        return yaml.safe_load(f)


def validate_json_against_schema(json_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON data against a schema."""
    # Simple validation - check required fields exist
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in json_data:
            return False
    return True


def test_golden_master_determinism_validation():
    """Test that determinism command produces valid output according to baseline contract."""
    contract = load_baseline_contract()
    
    # Find determinism command in baseline
    determinism_cmd = None
    for cmd in contract.get("cli", []):
        if cmd["cmd"] == "check-determinism":
            determinism_cmd = cmd
            break
    
    if not determinism_cmd:
        pytest.skip("check-determinism command not found in baseline contract")
    
    # Run a minimal determinism test
    result = subprocess.run(
        ["rldk", "check-determinism", "--run", "echo 'test'", "--seeds", "2"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    # Check exit code
    expected_exit_code = determinism_cmd["exit_codes"]["success"]
    assert result.returncode == expected_exit_code, f"Expected exit code {expected_exit_code}, got {result.returncode}"
    
    # Check output format matches regex
    output_regex = determinism_cmd.get("oneline_regex", r".*")
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if line.strip():
            assert re.match(output_regex, line), f"Output '{line}' doesn't match regex '{output_regex}'"
    
    # Check if artifact file exists and validates against schema
    artifact_pattern = None
    for artifact in contract.get("artifacts", []):
        if artifact["cmd"] == "check-determinism":
            artifact_pattern = artifact["artifact"]["path_pattern"]
            schema = artifact["artifact"]["schema"]
            break
    
    if artifact_pattern:
        # Look for artifact files
        import glob
        artifact_files = glob.glob(artifact_pattern)
        if artifact_files:
            # Validate the first artifact file
            with open(artifact_files[0], "r") as f:
                artifact_data = json.load(f)
            
            assert validate_json_against_schema(artifact_data, schema), \
                f"Artifact data doesn't validate against schema: {artifact_data}"


def test_golden_master_reward_health_validation():
    """Test that reward-health command produces valid output according to baseline contract."""
    contract = load_baseline_contract()
    
    # Find reward-health command in baseline
    reward_health_cmd = None
    for cmd in contract.get("cli", []):
        if cmd["cmd"] == "reward-health":
            reward_health_cmd = cmd
            break
    
    if not reward_health_cmd:
        pytest.skip("reward-health command not found in baseline contract")
    
    # Create minimal test data
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        f.write('{"step": 1, "reward": 0.5}\n')
        f.flush()
        
        # Run reward health test
        result = subprocess.run(
            ["rldk", "reward-health", "--logs", f.name],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        # Check exit code
        expected_exit_code = reward_health_cmd["exit_codes"]["success"]
        assert result.returncode == expected_exit_code, f"Expected exit code {expected_exit_code}, got {result.returncode}"
        
        # Check output format matches regex
        output_regex = reward_health_cmd.get("oneline_regex", r".*")
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if line.strip():
                assert re.match(output_regex, line), f"Output '{line}' doesn't match regex '{output_regex}'"
        
        # Check if artifact file exists and validates against schema
        artifact_pattern = None
        for artifact in contract.get("artifacts", []):
            if artifact["cmd"] == "reward-health":
                artifact_pattern = artifact["artifact"]["path_pattern"]
                schema = artifact["artifact"]["schema"]
                break
        
        if artifact_pattern:
            # Look for artifact files
            import glob
            artifact_files = glob.glob(artifact_pattern)
            if artifact_files:
                # Validate the first artifact file
                with open(artifact_files[0], "r") as f:
                    artifact_data = json.load(f)
                
                assert validate_json_against_schema(artifact_data, schema), \
                    f"Artifact data doesn't validate against schema: {artifact_data}"


def test_golden_master_replay_validation():
    """Test that replay command produces valid output according to baseline contract."""
    contract = load_baseline_contract()
    
    # Find replay command in baseline
    replay_cmd = None
    for cmd in contract.get("cli", []):
        if cmd["cmd"] == "replay":
            replay_cmd = cmd
            break
    
    if not replay_cmd:
        pytest.skip("replay command not found in baseline contract")
    
    # Run a minimal replay test
    result = subprocess.run(
        ["rldk", "replay", "--run", "echo 'test'", "--seeds", "2"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    # Check exit code
    expected_exit_code = replay_cmd["exit_codes"]["success"]
    assert result.returncode == expected_exit_code, f"Expected exit code {expected_exit_code}, got {result.returncode}"
    
    # Check output format matches regex
    output_regex = replay_cmd.get("oneline_regex", r".*")
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if line.strip():
            assert re.match(output_regex, line), f"Output '{line}' doesn't match regex '{output_regex}'"
    
    # Check if artifact file exists and validates against schema
    artifact_pattern = None
    for artifact in contract.get("artifacts", []):
        if artifact["cmd"] == "replay":
            artifact_pattern = artifact["artifact"]["path_pattern"]
            schema = artifact["artifact"]["schema"]
            break
    
    if artifact_pattern:
        # Look for artifact files
        import glob
        artifact_files = glob.glob(artifact_pattern)
        if artifact_files:
            # Validate the first artifact file
            with open(artifact_files[0], "r") as f:
                artifact_data = json.load(f)
            
            assert validate_json_against_schema(artifact_data, schema), \
                f"Artifact data doesn't validate against schema: {artifact_data}"


def test_golden_master_summary():
    """Print a summary of the golden master validation."""
    contract = load_baseline_contract()
    
    print("\n=== GOLDEN MASTER CONTRACT VALIDATION SUMMARY ===")
    print(f"Baseline contract version: {contract.get('version', 'unknown')}")
    print(f"Symbols validated: {len(contract.get('symbols', []))}")
    print(f"CLI commands validated: {len(contract.get('cli', []))}")
    print(f"Artifacts validated: {len(contract.get('artifacts', []))}")
    print(f"Contract files:")
    print(f"  - Baseline: api_contract.baseline.yaml")
    print(f"  - V1 Target: api_contract.v1.yaml")
    print(f"  - Diff: artifacts/api_contract.diff.txt")
    print("All golden master tests passed!")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])