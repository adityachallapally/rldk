"""Safe deterministic command runner for RLDK determinism checks."""

import os
import shlex
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from ..utils.error_handling import RLDKTimeoutError


def run_with_timeout_subprocess(
    args: list, 
    env: Dict[str, str], 
    timeout: int
) -> subprocess.CompletedProcess:
    """Run a subprocess with timeout and return the result.
    
    Args:
        args: Command arguments as a list
        env: Environment variables dict
        timeout: Timeout in seconds
        
    Returns:
        CompletedProcess result
        
    Raises:
        RLDKTimeoutError: If the command times out
    """
    try:
        result = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired as e:
        raise RLDKTimeoutError(
            f"Command timed out after {timeout} seconds",
            suggestion="Try increasing the timeout or optimizing the command",
            error_code="COMMAND_TIMEOUT",
            details={"timeout_seconds": timeout, "command": shlex.join(args)}
        ) from e


def run_deterministic_command(command: str, seed: int, timeout: int) -> int:
    """Run a command with deterministic seeding and return exit code.
    
    Args:
        command: Command string to execute
        seed: Seed value for deterministic execution
        timeout: Timeout in seconds
        
    Returns:
        Exit code of the command
        
    Raises:
        RLDKTimeoutError: If the command times out
    """
    # Parse command using shlex to handle quotes properly
    try:
        cmd_parts = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command format: {e}") from e
    
    # Check if command targets Python (heuristic)
    is_python_command = (
        len(cmd_parts) > 0 and 
        (cmd_parts[0].startswith("python") or 
         (len(cmd_parts) > 1 and cmd_parts[1].endswith(".py")))
    )
    
    if is_python_command:
        return _run_python_command_safe(command, cmd_parts, seed, timeout)
    else:
        return _run_generic_command(command, cmd_parts, seed, timeout)


def _run_python_command_safe(command: str, cmd_parts: list, seed: int, timeout: int) -> int:
    """Run a Python command with deterministic seeding using a safe approach."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a configuration file with the seed and command info
        config = {
            "seed": seed,
            "command": command,
            "cmd_parts": cmd_parts
        }
        
        config_path = Path(temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Create a safe Python shim that reads from the config file
        shim_content = '''#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import json
import runpy
import subprocess
from pathlib import Path

def main():
    # Read configuration safely
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    seed = config["seed"]
    cmd_parts = config["cmd_parts"]
    
    # Set deterministic environment
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch deterministic settings
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Disable TF32 for better determinism
        if hasattr(torch.backends.cuda, 'matmul.allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # TensorFlow deterministic settings
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass
    
    # Execute the target script safely
    try:
        # Preserve original sys.argv
        original_argv = sys.argv.copy()
        sys.argv = cmd_parts
        
        # Determine the target script
        if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
            script_path = sys.argv[1]
            if os.path.exists(script_path):
                runpy.run_path(script_path, run_name='__main__')
            else:
                print(f"Error: Script not found: {script_path}", file=sys.stderr)
                sys.exit(1)
        else:
            # For python -c commands, use subprocess instead of exec
            if len(sys.argv) > 2 and sys.argv[1] == '-c':
                # Use subprocess to execute the -c command safely
                result = subprocess.run([sys.executable, '-c', sys.argv[2]], 
                                     capture_output=True, text=True)
                print(result.stdout, end='')
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end='')
                sys.exit(result.returncode)
            else:
                print("Error: Invalid Python command format", file=sys.stderr)
                sys.exit(1)
                
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Write shim to temporary file
        shim_path = Path(temp_dir) / "deterministic_shim.py"
        with open(shim_path, 'w') as f:
            f.write(shim_content)
        
        # Make it executable
        shim_path.chmod(0o755)
        
        # Set up environment with PYTHONHASHSEED
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)
        
        # Run the shim with the config file
        result = run_with_timeout_subprocess(
            [sys.executable, str(shim_path), str(config_path)],
            env=env,
            timeout=timeout
        )
        
        return result.returncode


def _run_generic_command(command: str, cmd_parts: list, seed: int, timeout: int) -> int:
    """Run a generic command with PYTHONHASHSEED set.
    
    Note: Deep framework seeding may require user code modifications.
    """
    # Set up environment with PYTHONHASHSEED
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    
    # Run the command
    result = run_with_timeout_subprocess(
        cmd_parts,
        env=env,
        timeout=timeout
    )
    
    return result.returncode