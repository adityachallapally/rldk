#!/usr/bin/env python3
"""Minimal streaming loop that emits KL metrics for monitor testing."""
import json
import time
import sys
import os
from datetime import datetime, timezone
from pathlib import Path


def emit_event(step: int, name: str, value: float, run_id: str = "test-run"):
    """Emit a canonical JSONL event."""
    event = {
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "step": step,
        "name": name,
        "value": value,
        "run_id": run_id
    }
    print(json.dumps(event))
    sys.stdout.flush()


def main():
    """Run minimal streaming loop with escalating KL values."""
    print(f"PID: {os.getpid()}", file=sys.stderr)  # Print PID for stop action testing
    
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    output_file = artifacts_dir / "run.jsonl"
    
    with output_file.open("w") as f:
        for step in range(100):
            kl_value = 0.1 + (step * 0.01)  # Will exceed 0.35 after step 25
            event = {
                "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "step": step,
                "name": "kl",
                "value": kl_value,
                "run_id": "test-run"
            }
            
            line = json.dumps(event) + "\n"
            f.write(line)
            f.flush()
            print(json.dumps(event))
            sys.stdout.flush()
            
            emit_event(step, "reward", -0.1 + (step * 0.001))
            emit_event(step, "grad_norm", 0.5 + (step * 0.02))
            
            time.sleep(0.1)  # 100ms between steps


if __name__ == "__main__":
    main()
