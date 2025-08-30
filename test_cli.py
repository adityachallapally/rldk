#!/usr/bin/env python3
"""Test script for CLI debugging."""

import sys
sys.path.insert(0, 'src')

from rldk.cli import app

print("Testing CLI app...")
print(f"App type: {type(app)}")

# Test version command
print("\nTesting version command...")
try:
    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, ['version'])
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {repr(result.stdout)}")
    print(f"Error: {repr(result.stderr)}")
except Exception as e:
    print(f"Error testing CLI: {e}")

# Test help command
print("\nTesting help command...")
try:
    result = runner.invoke(app, ['--help'])
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {repr(result.stdout)}")
    print(f"Error: {repr(result.stderr)}")
except Exception as e:
    print(f"Error testing help: {e}")
