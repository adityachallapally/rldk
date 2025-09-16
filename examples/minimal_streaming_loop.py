#!/usr/bin/env python3
"""Minimal streaming loop that emits kl, reward, grad_norm metrics."""

import os
import sys
import time
import random
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.emit import EventWriter


def main():
    """Run a minimal training loop that emits metrics."""
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Create event writer
    run_id = f"minimal-run-{int(time.time())}"
    writer = EventWriter("artifacts/run.jsonl", run_id=run_id)
    
    print(f"Starting minimal streaming loop (PID: {os.getpid()})")
    print(f"Run ID: {run_id}")
    print("Writing to: artifacts/run.jsonl")
    print("Press Ctrl+C to stop")
    
    try:
        step = 0
        while True:
            # Simulate training metrics
            kl = random.uniform(0.1, 0.5)  # KL divergence
            reward = random.uniform(-1.0, 1.0)  # Reward
            grad_norm = random.uniform(0.5, 2.0)  # Gradient norm
            
            # Emit metrics
            writer.log(step=step, name="kl", value=kl)
            writer.log(step=step, name="reward", value=reward)
            writer.log(step=step, name="grad_norm", value=grad_norm)
            
            print(f"Step {step}: kl={kl:.3f}, reward={reward:.3f}, grad_norm={grad_norm:.3f}")
            
            step += 1
            time.sleep(1)  # Simulate training time
            
    except KeyboardInterrupt:
        print("\nStopping training loop...")
    finally:
        writer.close()
        print("Training loop stopped")


if __name__ == "__main__":
    main()