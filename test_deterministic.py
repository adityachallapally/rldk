#!/usr/bin/env python3
"""Simple deterministic test script for RLDK determinism checking."""

import argparse
import random
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(args.seed)
    
    # Simulate training steps
    results = []
    for step in range(args.steps):
        # Simulate some computation
        time.sleep(0.01)
        
        # Generate deterministic values based on seed and step
        value = random.random() + step * 0.1
        results.append(value)
        
        # Print metrics in a format that RLDK can parse
        print(f"step={step},loss={value:.6f},reward={value*2:.6f}")
    
    # Print final summary
    print(f"final_loss={results[-1]:.6f}")
    print(f"total_steps={len(results)}")

if __name__ == "__main__":
    main()