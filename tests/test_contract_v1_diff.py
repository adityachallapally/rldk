#!/usr/bin/env python3
"""
Test file for computing diff between baseline and v1 contracts.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Set


def load_contract(filename: str) -> Dict[str, Any]:
    """Load a contract from YAML file."""
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def compute_symbol_diff(baseline: Dict[str, Any], v1: Dict[str, Any]) -> str:
    """Compute diff for symbols section."""
    baseline_symbols = {s["name"]: s for s in baseline.get("symbols", [])}
    v1_symbols = {s["name"]: s for s in v1.get("symbols", [])}
    
    baseline_names = set(baseline_symbols.keys())
    v1_names = set(v1_symbols.keys())
    
    added = v1_names - baseline_names
    removed = baseline_names - v1_names
    common = baseline_names & v1_names
    
    diff_lines = []
    diff_lines.append("=== SYMBOLS DIFF ===")
    
    if added:
        diff_lines.append(f"ADDED ({len(added)}):")
        for name in sorted(added):
            symbol = v1_symbols[name]
            diff_lines.append(f"  + {name} ({symbol['kind']}) - {symbol['doc']}")
    
    if removed:
        diff_lines.append(f"REMOVED ({len(removed)}):")
        for name in sorted(removed):
            symbol = baseline_symbols[name]
            diff_lines.append(f"  - {name} ({symbol['kind']}) - {symbol['doc']}")
    
    if common:
        diff_lines.append(f"CHANGED ({len(common)}):")
        for name in sorted(common):
            baseline_symbol = baseline_symbols[name]
            v1_symbol = v1_symbols[name]
            
            changes = []
            if baseline_symbol["kind"] != v1_symbol["kind"]:
                changes.append(f"kind: {baseline_symbol['kind']} -> {v1_symbol['kind']}")
            if baseline_symbol["import_path"] != v1_symbol["import_path"]:
                changes.append(f"import_path: {baseline_symbol['import_path']} -> {v1_symbol['import_path']}")
            if baseline_symbol["doc"] != v1_symbol["doc"]:
                changes.append(f"doc: {baseline_symbol['doc']} -> {v1_symbol['doc']}")
            
            if changes:
                diff_lines.append(f"  ~ {name}: {', '.join(changes)}")
    
    if not (added or removed or common):
        diff_lines.append("No changes")
    
    return "\n".join(diff_lines)


def compute_cli_diff(baseline: Dict[str, Any], v1: Dict[str, Any]) -> str:
    """Compute diff for CLI section."""
    baseline_commands = {c["cmd"]: c for c in baseline.get("cli", [])}
    v1_commands = {c["cmd"]: c for c in v1.get("cli", [])}
    
    baseline_names = set(baseline_commands.keys())
    v1_names = set(v1_commands.keys())
    
    added = v1_names - baseline_names
    removed = baseline_names - v1_names
    common = baseline_names & v1_names
    
    diff_lines = []
    diff_lines.append("=== CLI DIFF ===")
    
    if added:
        diff_lines.append(f"ADDED ({len(added)}):")
        for name in sorted(added):
            cmd = v1_commands[name]
            diff_lines.append(f"  + {name}: {cmd['synopsis']}")
    
    if removed:
        diff_lines.append(f"REMOVED ({len(removed)}):")
        for name in sorted(removed):
            cmd = baseline_commands[name]
            diff_lines.append(f"  - {name}: {cmd.get('synopsis', 'N/A')}")
    
    if common:
        diff_lines.append(f"CHANGED ({len(common)}):")
        for name in sorted(common):
            baseline_cmd = baseline_commands[name]
            v1_cmd = v1_commands[name]
            
            changes = []
            if baseline_cmd.get("synopsis") != v1_cmd.get("synopsis"):
                changes.append(f"synopsis: {baseline_cmd.get('synopsis', 'N/A')} -> {v1_cmd.get('synopsis', 'N/A')}")
            
            baseline_req = set(baseline_cmd.get("required_flags", []))
            v1_req = set(v1_cmd.get("required_flags", []))
            if baseline_req != v1_req:
                changes.append(f"required_flags: {baseline_req} -> {v1_req}")
            
            baseline_opt = set(baseline_cmd.get("optional_flags", []))
            v1_opt = set(v1_cmd.get("optional_flags", []))
            if baseline_opt != v1_opt:
                changes.append(f"optional_flags: {baseline_opt} -> {v1_opt}")
            
            if changes:
                diff_lines.append(f"  ~ {name}: {', '.join(changes)}")
    
    if not (added or removed or common):
        diff_lines.append("No changes")
    
    return "\n".join(diff_lines)


def compute_artifacts_diff(baseline: Dict[str, Any], v1: Dict[str, Any]) -> str:
    """Compute diff for artifacts section."""
    # Baseline has artifacts as a list of dicts with "cmd" key
    baseline_artifacts = {a["cmd"]: a for a in baseline.get("artifacts", [])}
    
    # V1 has artifacts as a dict with schema names as keys
    v1_artifacts = v1.get("artifacts", {})
    
    baseline_names = set(baseline_artifacts.keys())
    v1_names = set(v1_artifacts.keys())
    
    added = v1_names - baseline_names
    removed = baseline_names - v1_names
    common = baseline_names & v1_names
    
    diff_lines = []
    diff_lines.append("=== ARTIFACTS DIFF ===")
    
    if added:
        diff_lines.append(f"ADDED ({len(added)}):")
        for name in sorted(added):
            artifact = v1_artifacts[name]
            diff_lines.append(f"  + {name}: schema definition")
    
    if removed:
        diff_lines.append(f"REMOVED ({len(removed)}):")
        for name in sorted(removed):
            artifact = baseline_artifacts[name]
            diff_lines.append(f"  - {name}: {artifact.get('artifact', {}).get('path_pattern', 'N/A')}")
    
    if common:
        diff_lines.append(f"CHANGED ({len(common)}):")
        for name in sorted(common):
            baseline_artifact = baseline_artifacts[name]
            v1_artifact = v1_artifacts[name]
            
            changes = []
            baseline_pattern = baseline_artifact.get("artifact", {}).get("path_pattern", "N/A")
            # V1 artifacts are schema definitions, not path patterns
            if "type" in v1_artifact:
                changes.append(f"structure: path_pattern -> schema_definition")
            
            if changes:
                diff_lines.append(f"  ~ {name}: {', '.join(changes)}")
    
    if not (added or removed or common):
        diff_lines.append("No changes")
    
    return "\n".join(diff_lines)


def test_compute_contract_diff():
    """Test that computes diff between baseline and v1 contracts."""
    # Load contracts
    baseline = load_contract("api_contract.baseline.yaml")
    v1 = load_contract("api_contract.v1.yaml")
    
    # Compute diffs
    symbol_diff = compute_symbol_diff(baseline, v1)
    cli_diff = compute_cli_diff(baseline, v1)
    artifacts_diff = compute_artifacts_diff(baseline, v1)
    
    # Combine all diffs
    full_diff = "\n\n".join([symbol_diff, cli_diff, artifacts_diff])
    
    # Print to stdout
    print(full_diff)
    
    # Save to file
    with open("artifacts/api_contract.diff.txt", "w") as f:
        f.write(full_diff)
    
    print(f"\nDiff saved to artifacts/api_contract.diff.txt")


if __name__ == "__main__":
    # Create artifacts directory if it doesn't exist
    Path("artifacts").mkdir(exist_ok=True)
    
    # Compute and display diff
    test_compute_contract_diff()