#!/usr/bin/env python3
"""
Summarization training script for reference tasks.

Runs ~50 steps with GPT-2 on CPU, logs JSONL with strict schema,
sets all seeds, and computes simple teacher forced loss.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_kl_divergence(logits: torch.Tensor, ref_logits: torch.Tensor) -> float:
    """Compute KL divergence between two logit distributions."""
    try:
        probs = torch.softmax(logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs = probs + eps
        ref_probs = ref_probs + eps

        kl_div = torch.sum(ref_probs * torch.log(ref_probs / probs))
        return kl_div.item()
    except (RuntimeError, ValueError) as e:
        # Log the specific error for debugging
        print(f"Warning: Error computing KL divergence: {e}")
        return float("nan")


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load dataset manifest."""
    manifest = []
    with open(manifest_path) as f:
        for line in f:
            manifest.append(json.loads(line.strip()))
    return manifest


def train_summarization(
    manifest_path: str,
    output_dir: str,
    max_steps: int = 50,
    seed: int = 42,
    model_name: str = "gpt2",
    max_length: int = 512,
    pad_direction: str = "right",
    truncate_at: Optional[int] = None,
):
    """Train summarization model with strict logging."""

    # Set all seeds
    set_all_seeds(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = load_manifest(manifest_path)

    # Load model and tokenizer
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reference model for KL computation if available
    ref_model = None
    try:
        ref_model = GPT2LMHeadModel.from_pretrained("gpt2")
        ref_model.eval()
    except (OSError, ImportError, RuntimeError) as e:
        print(f"Warning: Could not load reference model for KL computation: {e}")

    # Prepare data
    data = []
    for item in manifest:
        prompt = item["prompt"]
        reference = item["reference"]

        # Tokenize
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length" if pad_direction == "right" else False,
            truncation=True if truncate_at else False,
            return_tensors="pt",
        )

        # Apply truncation if specified
        if truncate_at and inputs["input_ids"].shape[1] > truncate_at:
            if pad_direction == "right":
                inputs["input_ids"] = inputs["input_ids"][:, :truncate_at]
                inputs["attention_mask"] = inputs["attention_mask"][:, :truncate_at]
            else:
                inputs["input_ids"] = inputs["input_ids"][:, -truncate_at:]
                inputs["attention_mask"] = inputs["attention_mask"][:, -truncate_at:]

        # Generate reference tokens
        ref_inputs = tokenizer(
            reference,
            max_length=max_length,
            padding="max_length" if pad_direction == "right" else False,
            truncation=True if truncate_at else False,
            return_tensors="pt",
        )

        data.append({"item": item, "inputs": inputs, "ref_inputs": ref_inputs})

    # Training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Log file
    log_path = output_path / "training_log.jsonl"

    with open(log_path, "w") as log_file:
        for step in range(max_steps):
            # Sample batch
            batch_idx = step % len(data)
            batch_data = data[batch_idx]

            inputs = batch_data["inputs"]
            ref_inputs = batch_data["ref_inputs"]
            item = batch_data["item"]

            # Forward pass
            # Ensure input and target have the same length
            input_length = inputs["input_ids"].shape[1]
            target_length = ref_inputs["input_ids"].shape[1]

            if input_length != target_length:
                # Truncate target to match input length
                if input_length < target_length:
                    ref_inputs["input_ids"] = ref_inputs["input_ids"][:, :input_length]
                else:
                    # Pad target to match input length
                    padding_length = input_length - target_length
                    padding = torch.full(
                        (1, padding_length), tokenizer.eos_token_id, dtype=torch.long
                    )
                    ref_inputs["input_ids"] = torch.cat(
                        [ref_inputs["input_ids"], padding], dim=1
                    )

            outputs = model(**inputs, labels=ref_inputs["input_ids"])
            loss = outputs.loss

            # Compute KL divergence to reference if available
            kl_to_ref = float("nan")
            if ref_model is not None:
                with torch.no_grad():
                    ref_outputs = ref_model(**inputs)
                    kl_to_ref = compute_kl_divergence(
                        outputs.logits, ref_outputs.logits
                    )

            # Generate output text
            with torch.no_grad():
                generated = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=32,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            # Compute reward (simple length-based for demo)
            reward_scalar = len(output_text.split()) / 100.0

            # Log step
            log_entry = {
                "global_step": step,
                "sample_id": item["original_index"],
                "prompt_sha256": item["prompt_sha256"],
                "dataset_id": item["dataset_id"],
                "dataset_revision": item["dataset_revision"],
                "dataset_split": item["dataset_split"],
                "rng.python": random.getstate()[1][0],
                "rng.numpy": int(np.random.get_state()[1][0]),
                "rng.torch": int(torch.randint(0, 1000000, (1,)).item()),
                "tokenizer.name": tokenizer.name_or_path,
                "tokenizer.revision": "main",  # Default for local models
                "tokenizer.vocab_sha256": compute_sha256(str(tokenizer.get_vocab())),
                "tokenizer.merges_sha256": compute_sha256(
                    str(tokenizer.merge_file)
                    if hasattr(tokenizer, "merge_file")
                    else "none"
                ),
                "model.name": model.name_or_path,
                "model.revision": "main",  # Default for local models
                "env.python_version": platform.python_version(),
                "env.torch_version": torch.__version__,
                "env.platform": platform.platform(),
                "inputs.input_ids_sha256": compute_sha256(
                    str(inputs["input_ids"].numpy().tobytes())
                ),
                "inputs.attention_mask_sha256": compute_sha256(
                    str(inputs["attention_mask"].numpy().tobytes())
                ),
                "inputs.max_length": max_length,
                "inputs.pad_direction": pad_direction,
                "inputs.truncate_at": truncate_at,
                "outputs.text": output_text,
                "loss": loss.item(),
                "reward_scalar": reward_scalar,
                "pairwise_logit": float("nan"),  # Not applicable for summarization
                "heuristic_score": float("nan"),  # Not applicable for summarization
                "kl_to_ref": kl_to_ref,
            }

            # Write to log file
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(
                    f"Step {step}: Loss = {loss.item():.4f}, Reward = {reward_scalar:.4f}"
                )

    print(f"Training complete! Logs saved to: {log_path}")
    return str(log_path)


def main():
    parser = argparse.ArgumentParser(description="Train summarization model")
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to dataset manifest"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for logs"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Maximum training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--pad-direction",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Padding direction",
    )
    parser.add_argument(
        "--truncate-at",
        type=int,
        default=None,
        help="Truncate sequences at this length",
    )

    args = parser.parse_args()

    train_summarization(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        model_name=args.model,
        max_length=args.max_length,
        pad_direction=args.pad_direction,
        truncate_at=args.truncate_at,
    )


if __name__ == "__main__":
    main()
