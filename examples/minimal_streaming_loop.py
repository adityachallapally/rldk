#!/usr/bin/env python3
"""Minimal streaming loop that emits JSONL events with rising KL values."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.emit import EventWriter


def main():
    """Run minimal streaming loop that demonstrates auto-stop behavior."""
    print(f"PID: {os.getpid()}")
    
    output_path = Path("artifacts/run.jsonl")
    output_path.parent.mkdir(exist_ok=True)
    
    with EventWriter(output_path, run_id="minimal-demo") as writer:
        for step in range(1, 101):
            if step < 20:
                kl = 0.1 + (step * 0.01)  # Gradual increase
            elif step < 40:
                kl = 0.3 + ((step - 20) * 0.005)  # Slower increase
            else:
                kl = 0.4 + ((step - 40) * 0.01)  # Faster increase to trigger rule
            
            writer.log(step=step, name="kl", value=kl)
            
            reward = -0.5 + (step * 0.01)  # Improving reward
            writer.log(step=step, name="reward", value=reward)
            
            grad_norm = 1.0 + (step * 0.02)  # Rising gradient norm
            writer.log(step=step, name="grad_norm", value=grad_norm)
            
            print(f"Step {step}: kl={kl:.3f}, reward={reward:.3f}, grad_norm={grad_norm:.3f}")
            
            time.sleep(0.1)
    
    print("Training loop completed")


if __name__ == "__main__":
    main()
