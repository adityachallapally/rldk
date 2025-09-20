#!/usr/bin/env python3
"""Test script for RLDK streaming monitoring."""

import time
import random
import json
import sys
from pathlib import Path

def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "training_metrics.jsonl"
    
    print(f"Streaming training metrics to {output_file}", file=sys.stderr)
    
    # Simulate training for 30 steps
    for step in range(30):
        # Simulate various metrics with some anomalies
        kl = 0.1 + 0.02 * random.random()
        kl_coef = 0.2 - 0.001 * step
        entropy = 2.5 - 0.01 * step + 0.1 * random.random()
        reward_mean = 0.8 + 0.1 * (step / 30) + 0.05 * random.random()
        reward_std = 0.3 + 0.02 * random.random()
        policy_grad_norm = 1.2 + 0.1 * random.random()
        value_grad_norm = 0.8 + 0.05 * random.random()
        learning_rate = 1e-5
        loss = 0.5 - 0.01 * step + 0.02 * random.random()
        
        # Add some anomalies
        if step == 15:
            kl = 0.5  # KL spike
            reward_mean = -0.2  # Reward collapse
        elif step == 25:
            policy_grad_norm = 5.0  # Gradient explosion
        
        # Create events
        events = [
            {"step": step, "name": "kl", "value": kl, "time": time.time()},
            {"step": step, "name": "kl_coef", "value": kl_coef, "time": time.time()},
            {"step": step, "name": "entropy", "value": entropy, "time": time.time()},
            {"step": step, "name": "reward_mean", "value": reward_mean, "time": time.time()},
            {"step": step, "name": "reward_std", "value": reward_std, "time": time.time()},
            {"step": step, "name": "policy_grad_norm", "value": policy_grad_norm, "time": time.time()},
            {"step": step, "name": "value_grad_norm", "value": value_grad_norm, "time": time.time()},
            {"step": step, "name": "learning_rate", "value": learning_rate, "time": time.time()},
            {"step": step, "name": "loss", "value": loss, "time": time.time()}
        ]
        
        # Write events to file
        with open(output_file, "a") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                f.flush()
        
        # Print to stdout for CLI monitoring
        for event in events:
            print(json.dumps(event))
            sys.stdout.flush()
        
        time.sleep(0.1)  # Simulate training delay
    
    print("Training completed", file=sys.stderr)

if __name__ == "__main__":
    main()