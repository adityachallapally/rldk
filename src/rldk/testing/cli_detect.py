"""Helper for detecting available CLI command variants."""

import subprocess
from typing import Optional


def detect_reward_health_cmd() -> str:
    """Detect available reward health command variant."""
    try:
        result = subprocess.run(
            ["rldk", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if "reward-health" in result.stdout:
            return "rldk reward-health"
        
        result = subprocess.run(
            ["rldk", "reward", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if "reward-health" in result.stdout:
            return "rldk reward reward-health"
        else:
            raise RuntimeError(
                "No reward health command found. Expected 'rldk reward-health' or 'rldk reward reward-health'"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to detect reward health command: {e}")


def detect_reward_drift_cmd() -> str:
    """Detect available reward drift command variant."""
    try:
        result = subprocess.run(
            ["rldk", "reward", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if "reward-drift" in result.stdout:
            return "rldk reward reward-drift"
        elif "drift" in result.stdout:
            return "rldk reward drift"
        else:
            raise RuntimeError(
                "No reward drift command found. Expected 'rldk reward reward-drift' or 'rldk reward drift'"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to detect reward drift command: {e}")
