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

    with open(output_path, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")


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

    with open(output_path, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")


def make_identical_checkpoints():
    """Generate two identical checkpoints as PyTorch state dicts."""
    import torch

    # Create simple checkpoint data as tensors
    checkpoint_data = {
        "layer1.weight": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
        ),
        "layer1.bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "layer2.weight": torch.tensor([[0.7, 0.8], [0.9, 1.0]], dtype=torch.float32),
        "layer2.bias": torch.tensor([0.3, 0.4], dtype=torch.float32),
    }

    # Save identical checkpoints
    for name in ["a", "b"]:
        output_path = Path(f"test_artifacts/ckpt_identical/{name}.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint_data, output_path)


def make_value_head_edit_checkpoints():
    """Generate checkpoints with edited value head."""
    import torch

    # Create base checkpoint data as tensors
    base_checkpoint = {
        "layer1.weight": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
        ),
        "layer1.bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "value_head.weight": torch.tensor(
            [[0.7, 0.8], [0.9, 1.0]], dtype=torch.float32
        ),
        "value_head.bias": torch.tensor([0.3, 0.4], dtype=torch.float32),
    }

    # Save base checkpoint
    base_path = Path("test_artifacts/ckpt_value_head_edit/a.pt")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(base_checkpoint, base_path)

    # Create modified checkpoint with different value head
    modified_checkpoint = base_checkpoint.copy()
    modified_checkpoint["value_head.weight"] = torch.tensor(
        [[1.05, 1.2], [1.35, 1.5]], dtype=torch.float32
    )  # 1.5x original
    modified_checkpoint["value_head.bias"] = torch.tensor(
        [0.4, 0.5], dtype=torch.float32
    )  # +0.1 to original

    # Save modified checkpoint
    modified_path = Path("test_artifacts/ckpt_value_head_edit/b.pt")

    torch.save(modified_checkpoint, modified_path)


def make_reward_drift_demo():
    """Generate reward drift demo with diverging behavior on code slice."""
    import torch

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

    with open(prompts_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    # Create simple reward models as PyTorch files
    for i, name in enumerate(["rmA", "rmB"]):
        model_dir = Path(f"test_artifacts/reward_drift_demo/{name}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create different model data for each model to ensure drift detection
        model_data = {
            "weights": torch.tensor(
                [
                    0.1 + i * 0.1,
                    0.2 + i * 0.1,
                    0.3 + i * 0.1,
                    0.4 + i * 0.1,
                    0.5 + i * 0.1,
                ],
                dtype=torch.float32,
            ),
            "bias": torch.tensor(0.1 + i * 0.2, dtype=torch.float32),
        }

        torch.save(model_data, model_dir / "model.pt")


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
