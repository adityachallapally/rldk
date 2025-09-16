#!/usr/bin/env python3
"""Minimal streaming loop that emits metrics and can be stopped by monitoring rules."""

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
    
    # Print PID for monitoring
    print(f"PID: {os.getpid()}")
    
    # Set up output directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Set up event writer
    writer = EventWriter(artifacts_dir / "run.jsonl", run_id="minimal-loop-test")
    
    # Simulate training loop
    step = 0
    kl_value = 0.1
    reward_value = 0.0
    grad_norm = 1.0
    
    print("Starting training loop...")
    print("Metrics will be written to artifacts/run.jsonl")
    print("Use Ctrl+C to stop manually, or let monitoring rules stop the process")
    
    try:
        while True:
            # Simulate metric evolution
            kl_value += random.uniform(-0.05, 0.1)
            kl_value = max(0.0, min(1.0, kl_value))  # Clamp to [0, 1]
            
            reward_value += random.uniform(-0.1, 0.2)
            reward_value = max(-1.0, min(1.0, reward_value))  # Clamp to [-1, 1]
            
            grad_norm += random.uniform(-0.1, 0.2)
            grad_norm = max(0.1, min(5.0, grad_norm))  # Clamp to [0.1, 5.0]
            
            # Emit metrics
            writer.log(step=step, name="kl", value=kl_value)
            writer.log(step=step, name="reward", value=reward_value)
            writer.log(step=step, name="grad_norm", value=grad_norm)
            
            # Print current metrics
            print(f"Step {step}: kl={kl_value:.3f}, reward={reward_value:.3f}, grad_norm={grad_norm:.3f}")
            
            step += 1
            time.sleep(0.5)  # Simulate training time
            
    except KeyboardInterrupt:
        print("\nTraining loop interrupted by user")
    except Exception as exc:
        print(f"Training loop failed: {exc}")
    finally:
        writer.close()
        print("Training loop completed")


if __name__ == "__main__":
    main()