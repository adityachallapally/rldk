#!/usr/bin/env python3
"""
Test determinism checking with real training scenarios
"""

import sys
import tempfile
import subprocess
from pathlib import Path

import _path_setup  # noqa: F401


def test_determinism_checking():
    """Test determinism checking with real training scenarios"""
    print("Testing Determinism Checking with Real Scenarios")
    print("=" * 60)
    
    try:
        from rldk.determinism import check
        
        # Create a more realistic deterministic training script
        script_content = '''
import numpy as np
import torch
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Simulate a simple training step
def train_step():
    # Create some random data
    x = np.random.randn(100)
    y = torch.randn(100)
    z = random.random()
    
    # Simple computation
    loss = np.sum(x**2) + torch.sum(y**2).item() + z**2
    
    return loss

# Run training
loss = train_step()

# Output results
print(f"loss:{loss:.6f}")
print(f"numpy_result:{np.sum(np.random.randn(50)):.6f}")
print(f"torch_result:{torch.sum(torch.randn(50)).item():.6f}")
print(f"random_result:{random.random():.6f}")
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "test_deterministic_training.py"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            print(f"✓ Training script created: {script_path}")
            
            # Test determinism checking with more replicas
            report = check(
                cmd=f"python {script_path}",
                compare=["loss", "numpy_result", "torch_result", "random_result"],
                replicas=5,
                device="cpu"
            )
            
            print(f"✓ Determinism check completed")
            print(f"  - Passed: {report.passed}")
            print(f"  - Replicas tested: 5")
            
            if hasattr(report, 'mismatches') and report.mismatches:
                print(f"  - Mismatches found: {len(report.mismatches)}")
                for i, mismatch in enumerate(report.mismatches[:3]):
                    print(f"    {i+1}. {mismatch}")
            else:
                print(f"  - No mismatches found")
            
            if hasattr(report, 'fixes') and report.fixes:
                print(f"  - Fixes suggested: {len(report.fixes)}")
                for fix in report.fixes[:3]:
                    print(f"    * {fix}")
            
            if hasattr(report, 'culprit') and report.culprit:
                print(f"  - Culprit: {report.culprit}")
        
        return True
        
    except Exception as e:
        print(f"✗ Determinism checking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_determinism_checking()