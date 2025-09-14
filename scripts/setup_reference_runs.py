#!/usr/bin/env python3
"""
Setup script to create the missing reference runs for tests.

This script creates the required reference runs that the test suite expects:
- reference/runs/summarization/good
- reference/runs/summarization/tokenizer_changed

It creates minimal but valid training logs that match the expected format.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def compute_sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_minimal_manifest(output_dir: Path, num_samples: int = 5) -> str:
    """Create a minimal dataset manifest for testing."""
    manifest_path = output_dir / "ag_news_manifest.jsonl"

    manifest = []
    for i in range(num_samples):
        prompt = f"This is test prompt {i} for summarization training."
        reference = f"This is test summary {i} for the prompt."

        manifest.append({
            "dataset_id": "ag_news",
            "dataset_revision": "main",
            "dataset_split": "train",
            "original_index": i,
            "prompt_sha256": compute_sha256(prompt),
            "reference_sha256": compute_sha256(reference),
            "prompt": prompt,
            "reference": reference,
            "transforms": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "do_sample": False,
            },
        })

    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    print(f"Created minimal manifest with {len(manifest)} samples: {manifest_path}")
    return str(manifest_path)


def create_training_log_entry(
    step: int,
    sample_id: int,
    prompt_sha256: str,
    pad_direction: str,
    truncate_at: int,
    tokenizer_vocab_sha256: str,
    tokenizer_merges_sha256: str,
    input_ids_sha256: str,
    attention_mask_sha256: str,
    output_text: str,
    loss: float,
    reward_scalar: float,
    kl_to_ref: float
) -> Dict:
    """Create a training log entry with the expected format."""
    return {
        "global_step": step,
        "sample_id": sample_id,
        "prompt_sha256": prompt_sha256,
        "dataset_id": "ag_news",
        "dataset_revision": "main",
        "dataset_split": "train",
        "rng.python": 12345 + step,
        "rng.numpy": 67890 + step,
        "rng.torch": 11111 + step,
        "tokenizer.name": "gpt2",
        "tokenizer.revision": "main",
        "tokenizer.vocab_sha256": tokenizer_vocab_sha256,
        "tokenizer.merges_sha256": tokenizer_merges_sha256,
        "model.name": "gpt2",
        "model.revision": "main",
        "env.python_version": "3.8.0",
        "env.torch_version": "1.9.0",
        "env.platform": "Linux",
        "inputs.input_ids_sha256": input_ids_sha256,
        "inputs.attention_mask_sha256": attention_mask_sha256,
        "inputs.max_length": 512,
        "inputs.pad_direction": pad_direction,
        "inputs.truncate_at": truncate_at,
        "outputs.text": output_text,
        "loss": loss,
        "reward_scalar": reward_scalar,
        "pairwise_logit": None,
        "heuristic_score": None,
        "kl_to_ref": kl_to_ref,
    }


def create_reference_runs():
    """Create the required reference runs for testing."""
    # Create directories
    good_run_dir = Path("reference/runs/summarization/good")
    tokenizer_changed_dir = Path("reference/runs/summarization/tokenizer_changed")

    good_run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_changed_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal manifest
    datasets_dir = Path("reference/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = create_minimal_manifest(datasets_dir)

    # Load manifest for reference
    manifest = []
    with open(manifest_path) as f:
        for line in f:
            manifest.append(json.loads(line.strip()))

    # Create good run training log
    good_log_path = good_run_dir / "training_log.jsonl"
    with open(good_log_path, "w") as f:
        for step, item in enumerate(manifest):
            log_entry = create_training_log_entry(
                step=step,
                sample_id=item["original_index"],
                prompt_sha256=item["prompt_sha256"],
                pad_direction="right",
                truncate_at=512,
                tokenizer_vocab_sha256="vocab_good_123",
                tokenizer_merges_sha256="merges_good_123",
                input_ids_sha256=f"input_good_{step}",
                attention_mask_sha256=f"mask_good_{step}",
                output_text=f"Good summary for step {step}",
                loss=2.5 - step * 0.1,  # Decreasing loss
                reward_scalar=0.15 + step * 0.01,  # Increasing reward
                kl_to_ref=0.1 - step * 0.02  # Decreasing KL divergence
            )
            f.write(json.dumps(log_entry) + "\n")

    print(f"Created good run training log: {good_log_path}")

    # Create tokenizer changed run training log
    tokenizer_changed_log_path = tokenizer_changed_dir / "training_log.jsonl"
    with open(tokenizer_changed_log_path, "w") as f:
        for step, item in enumerate(manifest):
            log_entry = create_training_log_entry(
                step=step,
                sample_id=item["original_index"],
                prompt_sha256=item["prompt_sha256"],
                pad_direction="left",  # Different from good run
                truncate_at=256,  # Different from good run
                tokenizer_vocab_sha256="vocab_changed_456",  # Different from good run
                tokenizer_merges_sha256="merges_changed_456",  # Different from good run
                input_ids_sha256=f"input_changed_{step}",
                attention_mask_sha256=f"mask_changed_{step}",
                output_text=f"Changed summary for step {step}",
                loss=2.8 - step * 0.08,  # Different loss pattern
                reward_scalar=0.12 + step * 0.008,  # Different reward pattern
                kl_to_ref=0.15 - step * 0.015  # Different KL divergence pattern
            )
            f.write(json.dumps(log_entry) + "\n")

    print(f"Created tokenizer changed run training log: {tokenizer_changed_log_path}")

    print("✅ Reference runs created successfully!")
    print(f"   - Good run: {good_run_dir}")
    print(f"   - Tokenizer changed run: {tokenizer_changed_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup reference runs for testing")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing reference runs before creating new ones"
    )

    args = parser.parse_args()

    if args.clean:
        # Clean existing runs
        runs_dir = Path("reference/runs")
        if runs_dir.exists():
            import shutil
            shutil.rmtree(runs_dir)
            print("Cleaned existing reference runs")

    create_reference_runs()


if __name__ == "__main__":
    main()
