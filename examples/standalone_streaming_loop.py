#!/usr/bin/env python3
"""Standalone minimal streaming loop that emits kl, reward, grad_norm metrics."""

import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path


class EventWriter:
    """Standalone event writer for JSONL events."""
    
    def __init__(self, path, run_id=None):
        self.path = Path(path)
        self.run_id = run_id or f"run-{int(datetime.utcnow().timestamp())}"
        
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file in append mode with line buffering
        self.file = open(self.path, "a", buffering=1)
    
    def log(self, step, name, value, **kwargs):
        """Log an event to JSONL."""
        event = {
            "time": datetime.utcnow().isoformat() + "Z",
            "step": int(step),
            "name": str(name),
            "value": float(value),
            "run_id": self.run_id
        }
        
        # Add optional fields
        if "tags" in kwargs:
            event["tags"] = kwargs["tags"]
        if "meta" in kwargs:
            event["meta"] = kwargs["meta"]
        
        # Write to file
        self.file.write(json.dumps(event) + "\n")
        self.file.flush()
    
    def close(self):
        """Close the file."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
            # Simulate training metrics with increasing KL to trigger alerts
            kl = random.uniform(0.1, 0.5) + (step * 0.01)  # Gradually increasing KL
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