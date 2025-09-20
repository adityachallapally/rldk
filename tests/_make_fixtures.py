"""Generate test fixtures for RL Debug Kit."""

import argparse
import json
import math
import random
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
    try:
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
    except ImportError:
        print("⚠️  PyTorch not available, skipping checkpoint generation")
        # Don't create dummy files - let tests handle missing files gracefully


def make_value_head_edit_checkpoints():
    """Generate checkpoints with edited value head."""
    try:
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
    except ImportError:
        print("⚠️  PyTorch not available, skipping value head checkpoint generation")
        # Don't create dummy files - let tests handle missing files gracefully


def make_reward_drift_demo():
    """Generate reward drift demo with diverging behavior on code slice."""
    # Create prompts (this part doesn't need torch)
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
    try:
        import torch

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
    except ImportError:
        print("⚠️  PyTorch not available, skipping reward model generation")
        # Don't create dummy files - let tests handle missing files gracefully


def make_grpo_logs(group_size: int = 4, seeds: list[int] | None = None) -> None:
    """Generate synthetic GRPO-style logs with group-normalized rewards."""

    if seeds is None:
        seeds = [1, 2, 3]

    base_dir = Path("test_artifacts/logs_grpo")
    base_dir.mkdir(parents=True, exist_ok=True)

    target_kl = 0.08

    for seed in seeds:
        rng = random.Random(seed)
        run_dir = base_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        events = []
        kl_coef = 0.12 + 0.02 * rng.random()

        for step in range(1000):
            kl_value = 0.07 + 0.015 * math.sin(step / 50.0) + rng.gauss(0, 0.003)

            if 800 <= step <= 804:
                kl_value = 0.32 + rng.gauss(0, 0.02)

            if kl_value > target_kl * 1.15:
                kl_coef *= 1.005
            elif kl_value < target_kl * 0.85:
                kl_coef *= 0.995
            else:
                kl_coef *= 0.999

            kl_coef = max(0.05, min(kl_coef, 0.25))

            raw_rewards = [
                rng.gauss(1.0 + 0.1 * math.sin(step / 40.0), 0.15 + 0.02 * rng.random())
                for _ in range(group_size)
            ]
            group_mean = sum(raw_rewards) / group_size
            variance = sum((value - group_mean) ** 2 for value in raw_rewards) / group_size
            group_std = math.sqrt(max(variance, 1e-6))
            normalized = [(value - group_mean) / group_std for value in raw_rewards]

            reward_mean = sum(normalized) / group_size
            reward_std = math.sqrt(
                sum((value - reward_mean) ** 2 for value in normalized) / group_size
            )

            entropy = 2.2 + 0.05 * math.cos(step / 45.0) + rng.gauss(0, 0.01)
            grad_norm_policy = 0.65 + 0.08 * rng.gauss(0, 1)
            grad_norm_value = 0.45 + 0.06 * rng.gauss(0, 1)

            events.append(
                {
                    "step": step,
                    "kl": round(kl_value, 6),
                    "kl_coef": round(kl_coef, 6),
                    "entropy": round(entropy, 6),
                    "advantage_mean": round(reward_mean + rng.gauss(0, 0.01), 6),
                    "advantage_std": round(max(reward_std + rng.gauss(0, 0.01), 0.05), 6),
                    "grad_norm_policy": round(max(grad_norm_policy, 0.05), 6),
                    "grad_norm_value": round(max(grad_norm_value, 0.05), 6),
                    "reward_mean": round(reward_mean, 6),
                    "reward_std": round(reward_std, 6),
                    "group_size": group_size,
                }
            )

        run_path = run_dir / "run.jsonl"
        with open(run_path, "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        index_path = run_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"files": ["run.jsonl"], "group_size": group_size}, f, indent=2)


def main() -> None:
    """Generate requested test fixtures."""

    parser = argparse.ArgumentParser(description="Generate RLDK test fixtures")
    parser.add_argument(
        "--make",
        nargs="+",
        choices=[
            "all",
            "clean",
            "doctored_kl_spike",
            "identical_checkpoints",
            "value_head_edit",
            "reward_drift",
            "grpo",
        ],
        default=["all"],
        help="Fixtures to generate (default: all)",
    )

    args = parser.parse_args()
    requested = set(args.make)

    if "all" in requested:
        requested = {
            "clean",
            "doctored_kl_spike",
            "identical_checkpoints",
            "value_head_edit",
            "reward_drift",
            "grpo",
        }

    generators = {
        "clean": (make_clean_logs, "Clean logs"),
        "doctored_kl_spike": (make_doctored_kl_spike_logs, "Doctored KL spike logs"),
        "identical_checkpoints": (make_identical_checkpoints, "Identical checkpoints"),
        "value_head_edit": (make_value_head_edit_checkpoints, "Value head edit checkpoints"),
        "reward_drift": (make_reward_drift_demo, "Reward drift demo"),
        "grpo": (make_grpo_logs, "GRPO logs"),
    }

    order = [
        "clean",
        "doctored_kl_spike",
        "identical_checkpoints",
        "value_head_edit",
        "reward_drift",
        "grpo",
    ]

    print("Generating test fixtures...")

    for key in order:
        if key in requested:
            generator, label = generators[key]
            generator()
            print(f"✓ {label} generated")

    print("All requested fixtures generated successfully!")


if __name__ == "__main__":
    main()
