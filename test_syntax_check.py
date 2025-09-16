#!/usr/bin/env python3
"""Syntax check for the monitor engine."""

import ast
import sys

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        # Check for specific classes and functions
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return True, classes, functions
    except SyntaxError as e:
        return False, str(e), []

def main():
    """Check syntax of the monitor engine."""
    print("Checking syntax of monitor engine...")
    
    success, classes, functions = check_syntax("src/rldk/monitor/engine.py")
    
    if success:
        print("✓ Syntax is valid")
        print(f"Found classes: {classes}")
        print(f"Found functions: {len(functions)} functions")
        
        # Check for expected classes
        expected_classes = ["StopAction", "SentinelAction", "ShellAction", "HttpAction", "WarnAction"]
        for cls in expected_classes:
            if cls in classes:
                print(f"✓ Found {cls}")
            else:
                print(f"✗ Missing {cls}")
        
        # Check for expected functions
        expected_functions = ["generate_human_summary", "load_rules"]
        for func in expected_functions:
            if func in functions:
                print(f"✓ Found {func}")
            else:
                print(f"✗ Missing {func}")
        
        print("\n✓ All syntax checks passed!")
        return 0
    else:
        print(f"✗ Syntax error: {classes}")
        return 1

if __name__ == "__main__":
    sys.exit(main())