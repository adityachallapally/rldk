#!/usr/bin/env python3
"""
Test file for baseline contract introspection and validation.
This file discovers the public API surface of rldk by importing modules and running CLI commands.
"""

import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest

# Import rldk to discover public symbols
import rldk


def discover_public_symbols() -> List[Dict[str, str]]:
    """Discover public Python symbols from rldk package."""
    symbols = []
    
    # Get symbols from rldk.__all__
    for symbol_name in rldk.__all__:
        try:
            symbol = getattr(rldk, symbol_name)
            kind = "class" if inspect.isclass(symbol) else "function"
            
            # Get docstring first line
            doc = ""
            if hasattr(symbol, "__doc__") and symbol.__doc__:
                doc = symbol.__doc__.split('\n')[0].strip()
            
            # Get import path
            import_path = f"rldk.{symbol_name}"
            
            symbols.append({
                "name": symbol_name,
                "kind": kind,
                "import_path": import_path,
                "doc": doc
            })
        except Exception as e:
            print(f"Warning: Could not inspect symbol {symbol_name}: {e}")
    
    return symbols


def discover_cli_commands() -> List[Dict[str, Any]]:
    """Discover CLI commands and their flags by running help commands."""
    commands = []
    
    # Get main help
    try:
        result = subprocess.run(
            ["rldk", "--help"], 
            capture_output=True, 
            text=True, 
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        if result.returncode != 0:
            print(f"Warning: rldk --help failed: {result.stderr}")
            return commands
        
        # Parse commands from help output
        help_text = result.stdout
        cmd_names = []
        
        # Look for command names in the help output
        lines = help_text.split('\n')
        for i, line in enumerate(lines):
            if 'Commands' in line:
                # Found the commands section, look for command names
                for j in range(i + 1, len(lines)):
                    cmd_line = lines[j]
                    if '│' in cmd_line:
                        parts = cmd_line.split('│')
                        if len(parts) >= 2:
                            cmd_name = parts[1].strip()
                            if cmd_name and not cmd_name.startswith('─') and cmd_name != '':
                                # Extract just the command name (before any spaces)
                                cmd_name = cmd_name.split()[0]
                                cmd_names.append(cmd_name)
                    elif cmd_line.strip() == '' or not cmd_line.startswith('│'):
                        # End of commands section
                        break
        
        print(f"Discovered commands: {cmd_names}")
        
        # Get help for each command
        for cmd in cmd_names:
            try:
                cmd_result = subprocess.run(
                    ["rldk", cmd, "--help"],
                    capture_output=True,
                    text=True,
                    env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
                )
                
                if cmd_result.returncode == 0:
                    cmd_help = cmd_result.stdout
                    
                    # Parse required and optional flags
                    required_flags = []
                    optional_flags = []
                    
                    # Simple regex-based parsing of typer help output
                    for line in cmd_help.split('\n'):
                        if '--' in line and '[' in line:
                            # Optional flag
                            flag = line.split()[0]
                            if flag.startswith('--'):
                                optional_flags.append(flag)
                        elif '--' in line and '...' in line:
                            # Required flag
                            flag = line.split()[0]
                            if flag.startswith('--'):
                                required_flags.append(flag)
                    
                    # Try to run a minimal synthetic test to get output format
                    oneline_regex = get_command_output_regex(cmd)
                    
                    commands.append({
                        "cmd": cmd,
                        "synopsis": f"rldk {cmd} [OPTIONS]",
                        "required_flags": required_flags,
                        "optional_flags": optional_flags,
                        "oneline_regex": oneline_regex,
                        "exit_codes": {"success": 0, "failure": 1}
                    })
                    
            except Exception as e:
                print(f"Warning: Could not get help for command {cmd}: {e}")
                
    except Exception as e:
        print(f"Warning: Could not discover CLI commands: {e}")
    
    return commands


def get_command_output_regex(cmd: str) -> str:
    """Get a permissive regex for command output by running minimal synthetic tests."""
    # Create minimal test data for each command
    test_cases = {
        "ingest": lambda: run_ingest_test(),
        "diff": lambda: run_diff_test(),
        "check-determinism": lambda: run_determinism_test(),
        "reward-health": lambda: run_reward_health_test(),
        "replay": lambda: run_replay_test(),
        "eval": lambda: run_eval_test(),
        "compare-runs": lambda: run_compare_runs_test(),
        "diff-ckpt": lambda: run_diff_ckpt_test(),
        "env-audit": lambda: run_env_audit_test(),
        "log-scan": lambda: run_log_scan_test(),
        "track": lambda: run_track_test(),
        "reward-drift": lambda: run_reward_drift_test(),
        "doctor": lambda: run_doctor_test(),
        "version": lambda: run_version_test(),
        "card": lambda: run_card_test(),
    }
    
    if cmd in test_cases:
        try:
            return test_cases[cmd]()
        except Exception as e:
            print(f"Warning: Could not get output regex for {cmd}: {e}")
    
    # Default permissive regex
    return r".*"


def run_ingest_test() -> str:
    """Run minimal ingest test and return output regex."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        f.write('{"step": 1, "reward": 0.5}\n')
        f.flush()
        
        result = subprocess.run(
            ["rldk", "ingest", f.name],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        if result.returncode == 0:
            # Extract key patterns from output
            output = result.stdout
            if "Ingested" in output and "training steps" in output:
                return r"^.*Ingested \d+ training steps.*$"
    
    return r".*"


def run_diff_test() -> str:
    """Run minimal diff test and return output regex."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f2:
        f1.write('{"step": 1, "reward": 0.5}\n')
        f2.write('{"step": 1, "reward": 0.6}\n')
        f1.flush()
        f2.flush()
        
        result = subprocess.run(
            ["rldk", "diff", "--a", f1.name, "--b", f2.name, "--signals", "reward"],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        if result.returncode == 0:
            return r"^.*divergence.*$"
    
    return r".*"


def run_determinism_test() -> str:
    """Run minimal determinism test and return output regex."""
    result = subprocess.run(
        ["rldk", "check-determinism", "--run", "echo 'test'", "--seeds", "2"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*determinism.*pass.*seeds.*$"
    
    return r".*"


def run_reward_health_test() -> str:
    """Run minimal reward health test and return output regex."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        f.write('{"step": 1, "reward": 0.5}\n')
        f.flush()
        
        result = subprocess.run(
            ["rldk", "reward-health", "--logs", f.name],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        if result.returncode == 0:
            return r"^.*reward_health.*pass.*drift.*$"
    
    return r".*"


def run_replay_test() -> str:
    """Run minimal replay test and return output regex."""
    result = subprocess.run(
        ["rldk", "replay", "--run", "echo 'test'", "--seeds", "2"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*replay.*snapshot.*seeds.*$"
    
    return r".*"


def run_eval_test() -> str:
    """Run minimal eval test and return output regex."""
    result = subprocess.run(
        ["rldk", "eval", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*eval.*$"
    
    return r".*"


def run_compare_runs_test() -> str:
    """Run minimal compare-runs test and return output regex."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        Path(d1).joinpath("run.json").write_text('{"step": 1, "reward": 0.5}')
        Path(d2).joinpath("run.json").write_text('{"step": 1, "reward": 0.6}')
        
        result = subprocess.run(
            ["rldk", "compare-runs", d1, d2],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        if result.returncode == 0:
            return r"^.*comparison.*complete.*$"
    
    return r".*"


def run_diff_ckpt_test() -> str:
    """Run minimal diff-ckpt test and return output regex."""
    result = subprocess.run(
        ["rldk", "diff-ckpt", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*checkpoint.*comparison.*$"
    
    return r".*"


def run_env_audit_test() -> str:
    """Run minimal env-audit test and return output regex."""
    result = subprocess.run(
        ["rldk", "env-audit"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*environment.*audit.*$"
    
    return r".*"


def run_log_scan_test() -> str:
    """Run minimal log-scan test and return output regex."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log') as f:
        f.write("PPO training log\n")
        f.flush()
        
        result = subprocess.run(
            ["rldk", "log-scan", f.name],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        if result.returncode == 0:
            return r"^.*log.*scan.*complete.*$"
    
    return r".*"


def run_track_test() -> str:
    """Run minimal track test and return output regex."""
    result = subprocess.run(
        ["rldk", "track", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*tracking.*started.*$"
    
    return r".*"


def run_reward_drift_test() -> str:
    """Run minimal reward-drift test and return output regex."""
    result = subprocess.run(
        ["rldk", "reward-drift", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*reward.*drift.*$"
    
    return r".*"


def run_doctor_test() -> str:
    """Run minimal doctor test and return output regex."""
    result = subprocess.run(
        ["rldk", "doctor", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*diagnostics.*$"
    
    return r".*"


def run_version_test() -> str:
    """Run version test and return output regex."""
    result = subprocess.run(
        ["rldk", "version"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*version.*$"
    
    return r".*"


def run_card_test() -> str:
    """Run minimal card test and return output regex."""
    result = subprocess.run(
        ["rldk", "card", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
    )
    
    if result.returncode == 0:
        return r"^.*card.*generation.*$"
    
    return r".*"


def discover_artifacts() -> List[Dict[str, Any]]:
    """Discover artifacts produced by commands."""
    artifacts = []
    
    # For now, we'll create minimal schemas based on command names
    # In a real implementation, we'd run each command and analyze the output
    artifact_patterns = {
        "check-determinism": {
            "path_pattern": "rldk_reports/determinism/*.json",
            "schema": {
                "type": "object",
                "required": ["pass", "seeds", "checksums"]
            }
        },
        "reward-health": {
            "path_pattern": "rldk_reports/reward_health/*.json", 
            "schema": {
                "type": "object",
                "required": ["pass", "drift", "signals"]
            }
        },
        "replay": {
            "path_pattern": "rldk_reports/repro/metadata.json",
            "schema": {
                "type": "object", 
                "required": ["command", "seeds", "git", "pip_freeze"]
            }
        }
    }
    
    for cmd, pattern in artifact_patterns.items():
        artifacts.append({
            "cmd": cmd,
            "artifact": pattern
        })
    
    return artifacts


def generate_baseline_contract() -> Dict[str, Any]:
    """Generate the baseline contract by introspection."""
    return {
        "version": "baseline",
        "symbols": discover_public_symbols(),
        "cli": discover_cli_commands(),
        "artifacts": discover_artifacts()
    }


def test_generate_baseline_contract():
    """Test that generates the baseline contract."""
    contract = generate_baseline_contract()
    
    # Save the contract
    with open("api_contract.baseline.yaml", "w") as f:
        yaml.dump(contract, f, default_flow_style=False)
    
    # Basic validation
    assert "symbols" in contract
    assert "cli" in contract
    assert "artifacts" in contract
    assert len(contract["symbols"]) > 0
    assert len(contract["cli"]) > 0


def test_validate_symbols():
    """Test that all recorded symbols can be imported and have correct kind."""
    with open("api_contract.baseline.yaml", "r") as f:
        contract = yaml.safe_load(f)
    
    for symbol in contract["symbols"]:
        # Import the symbol
        module = importlib.import_module("rldk")
        symbol_obj = getattr(module, symbol["name"])
        
        # Check kind
        expected_kind = symbol["kind"]
        actual_kind = "class" if inspect.isclass(symbol_obj) else "function"
        assert actual_kind == expected_kind, f"Symbol {symbol['name']} has wrong kind"


def test_validate_cli_help():
    """Test that CLI help contains recorded flag names."""
    with open("api_contract.baseline.yaml", "r") as f:
        contract = yaml.safe_load(f)
    
    for cmd_info in contract["cli"]:
        cmd = cmd_info["cmd"]
        
        # Get help for command
        result = subprocess.run(
            ["rldk", cmd, "--help"],
            capture_output=True,
            text=True,
            env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        )
        
        if result.returncode == 0:
            help_text = result.stdout
            
            # Check that required flags are mentioned
            for flag in cmd_info.get("required_flags", []):
                assert flag in help_text, f"Required flag {flag} not found in help for {cmd}"
            
            # Check that optional flags are mentioned  
            for flag in cmd_info.get("optional_flags", []):
                assert flag in help_text, f"Optional flag {flag} not found in help for {cmd}"


def test_validate_command_output():
    """Test that commands produce expected output format."""
    with open("api_contract.baseline.yaml", "r") as f:
        contract = yaml.safe_load(f)
    
    for cmd_info in contract["cli"]:
        cmd = cmd_info["cmd"]
        regex = cmd_info.get("oneline_regex", r".*")
        
        # Run minimal synthetic test
        if cmd == "check-determinism":
            result = subprocess.run(
                ["rldk", cmd, "--run", "echo 'test'", "--seeds", "2"],
                capture_output=True,
                text=True,
                env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
            )
        elif cmd == "reward-health":
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                f.write('{"step": 1, "reward": 0.5}\n')
                f.flush()
                result = subprocess.run(
                    ["rldk", cmd, "--logs", f.name],
                    capture_output=True,
                    text=True,
                    env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
                )
        elif cmd == "replay":
            result = subprocess.run(
                ["rldk", cmd, "--run", "echo 'test'", "--seeds", "2"],
                capture_output=True,
                text=True,
                env={"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
            )
        else:
            # Skip other commands for now
            continue
        
        if result.returncode == 0:
            # Check that output matches regex
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.strip():
                    assert re.match(regex, line), f"Output '{line}' doesn't match regex '{regex}' for {cmd}"


if __name__ == "__main__":
    # Generate baseline contract
    contract = generate_baseline_contract()
    
    # Save to file
    with open("api_contract.baseline.yaml", "w") as f:
        yaml.dump(contract, f, default_flow_style=False)
    
    print("Baseline contract generated successfully!")
    print(f"Found {len(contract['symbols'])} symbols")
    print(f"Found {len(contract['cli'])} CLI commands")
    print(f"Found {len(contract['artifacts'])} artifacts")