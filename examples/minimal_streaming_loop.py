#!/usr/bin/env python3
"""
Minimal streaming loop that demonstrates auto-stop functionality.

This script simulates a training loop that emits metrics and can be stopped
by the monitoring system when certain conditions are met.
"""

import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rldk.emit import EventWriter


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Run a minimal training loop that emits metrics."""
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Print PID for monitoring
    pid = os.getpid()
    print(f"Training loop PID: {pid}")
    print(f"Metrics will be written to: {artifacts_dir / 'run.jsonl'}")
    print("Start monitoring with: rldk monitor --stream artifacts/run.jsonl --rules rules.yaml --pid", pid)
    print("Press Ctrl+C to stop manually\n")
    
    # Initialize event writer
    with EventWriter(artifacts_dir / "run.jsonl") as writer:
        run_id = f"run-{int(time.time())}"
        
        for step in range(1, 1001):  # Run for up to 1000 steps
            # Simulate training metrics
            # KL divergence that gradually increases
            kl = 0.1 + (step / 1000.0) * 0.4 + random.uniform(-0.05, 0.05)
            kl = max(0.0, kl)  # Ensure non-negative
            
            # Reward that fluctuates
            reward = 0.5 + 0.3 * random.uniform(-1, 1) + (step / 1000.0) * 0.2
            
            # Gradient norm that occasionally spikes
            grad_norm = 1.0 + random.uniform(-0.2, 0.2)
            if step % 50 == 0:  # Occasional spikes
                grad_norm += random.uniform(2.0, 5.0)
            
            # Loss that generally decreases but has noise
            loss = 2.0 * (0.9 ** (step / 100.0)) + random.uniform(-0.1, 0.1)
            loss = max(0.01, loss)  # Ensure positive
            
            # Emit metrics
            writer.log(
                step=step,
                name="kl",
                value=kl,
                run_id=run_id,
                tags={"phase": "training", "model": "gpt2-small"}
            )
            
            writer.log(
                step=step,
                name="reward",
                value=reward,
                run_id=run_id,
                tags={"phase": "training", "model": "gpt2-small"}
            )
            
            writer.log(
                step=step,
                name="grad_norm",
                value=grad_norm,
                run_id=run_id,
                tags={"phase": "training", "model": "gpt2-small"}
            )
            
            writer.log(
                step=step,
                name="loss",
                value=loss,
                run_id=run_id,
                tags={"phase": "training", "model": "gpt2-small"}
            )
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"Step {step}: KL={kl:.3f}, Reward={reward:.3f}, GradNorm={grad_norm:.3f}, Loss={loss:.3f}")
            
            # Small delay to simulate real training
            time.sleep(0.1)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()