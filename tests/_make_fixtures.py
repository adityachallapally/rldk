"""Generate test fixtures for RL Debug Kit."""

import json
import random
import math
from pathlib import Path


def make_clean_logs():
    """Generate clean training logs with steady KL and healthy grad ratios."""
    random.seed(42)  # For reproducibility
    
    logs = []
    
    for step in range(1000):
        log = {
            "step": step,
            "kl": 0.05 + 0.01 * math.sin(step / 100),  # Steady KL around 0.05
            "kl_coef": 0.1,
            "entropy": 2.0 + 0.1 * math.cos(step / 50),
            "advantage_mean": 0.0 + 0.1 * random.gauss(0, 1),
            "advantage_std": 1.0,
            "grad_norm_policy": 0.5 + 0.1 * random.gauss(0, 1),
            "grad_norm_value": 0.3 + 0.05 * random.gauss(0, 1),
        }
        logs.append(log)
    
    # Write to file
    output_path = Path("test_artifacts/logs_clean/training.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')


def make_doctored_kl_spike_logs():
    """Generate logs with a KL spike starting near step 800."""
    random.seed(42)  # For reproducibility
    
    logs = []
    
    for step in range(1000):
        # Normal behavior until step 800
        if step < 800:
            kl = 0.05 + 0.01 * math.sin(step / 100)
            kl_coef = 0.1
        else:
            # KL spike starting at step 800
            kl = 0.3 + 0.1 * random.gauss(0, 1)  # Much higher KL
            kl_coef = 0.11  # Almost constant (controller stuck)
        
        log = {
            "step": step,
            "kl": kl,
            "kl_coef": kl_coef,
            "entropy": 2.0 + 0.1 * math.cos(step / 50),
            "advantage_mean": 0.0 + 0.1 * random.gauss(0, 1),
            "advantage_std": 1.0,
            "grad_norm_policy": 0.5 + 0.1 * random.gauss(0, 1),
            "grad_norm_value": 0.3 + 0.05 * random.gauss(0, 1),
        }
        logs.append(log)
    
    # Write to file
    output_path = Path("test_artifacts/logs_doctored_kl_spike/training.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')


def make_identical_checkpoints():
    """Generate two identical checkpoints (simple JSON format for now)."""
    # Create simple checkpoint data
    checkpoint_data = {
        "layer1.weight": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "layer1.bias": [0.1, 0.2],
        "layer2.weight": [[0.7, 0.8], [0.9, 1.0]],
        "layer2.bias": [0.3, 0.4]
    }
    
    # Save identical checkpoints
    for name in ['a', 'b']:
        output_path = Path(f"test_artifacts/ckpt_identical/{name}.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(checkpoint_data, f)


def make_value_head_edit_checkpoints():
    """Generate checkpoints with edited value head."""
    # Create base checkpoint data
    base_checkpoint = {
        "layer1.weight": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "layer1.bias": [0.1, 0.2],
        "value_head.weight": [[0.7, 0.8], [0.9, 1.0]],
        "value_head.bias": [0.3, 0.4]
    }
    
    # Save base checkpoint
    base_path = Path("test_artifacts/ckpt_value_head_edit/base.pt")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(base_path, 'w') as f:
        json.dump(base_checkpoint, f)
    
    # Create modified checkpoint with different value head
    modified_checkpoint = base_checkpoint.copy()
    modified_checkpoint["value_head.weight"] = [[1.05, 1.2], [1.35, 1.5]]  # 1.5x original
    modified_checkpoint["value_head.bias"] = [0.4, 0.5]  # +0.1 to original
    
    # Save modified checkpoint
    modified_path = Path("test_artifacts/ckpt_value_head_edit/modified.pt")
    
    with open(modified_path, 'w') as f:
        json.dump(modified_checkpoint, f)


def make_reward_drift_demo():
    """Generate reward drift demo with diverging behavior on code slice."""
    # Create prompts
    prompts = [
        {"text": "What is 2 + 2?", "tags": ["math"]},
        {"text": "How to solve this equation: x + 5 = 10", "tags": ["math"]},
        {"text": "Write a function to add two numbers", "tags": ["code"]},
        {"text": "def hello(): print('world')", "tags": ["code"]},
        {"text": "Is it safe to walk alone at night?", "tags": ["safety"]},
        {"text": "I cannot help with that request", "tags": ["refusal"]},
        {"text": "Calculate the area of a circle", "tags": ["math"]},
        {"text": "for i in range(10): print(i)", "tags": ["code"]},
    ]
    
    # Save prompts
    prompts_path = Path("test_artifacts/reward_drift_demo/prompts.jsonl")
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(prompts_path, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    
    # Create simple reward models (JSON format for now)
    for name in ['rmA', 'rmB']:
        model_dir = Path(f"test_artifacts/reward_drift_demo/{name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple model data
        model_data = {
            "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
            "bias": 0.1
        }
        
        with open(model_dir / "model.pt", 'w') as f:
            json.dump(model_data, f)


def main():
    """Generate all test fixtures."""
    print("Generating test fixtures...")
    
    make_clean_logs()
    print("✓ Clean logs generated")
    
    make_doctored_kl_spike_logs()
    print("✓ Doctored KL spike logs generated")
    
    make_identical_checkpoints()
    print("✓ Identical checkpoints generated")
    
    make_value_head_edit_checkpoints()
    print("✓ Value head edit checkpoints generated")
    
    make_reward_drift_demo()
    print("✓ Reward drift demo generated")
    
    print("All fixtures generated successfully!")


if __name__ == "__main__":
    main()