#!/usr/bin/env python3
"""
Code fix evaluation script for reference tasks.

Generates outputs per prompt and logs JSONL with the required schema.
Uses MBPP dataset for code generation evaluation.
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


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load dataset manifest."""
    manifest = []
    with open(manifest_path) as f:
        for line in f:
            manifest.append(json.loads(line.strip()))
    return manifest


def compute_code_quality_score(text: str) -> float:
    """Compute a simple code quality score based on content."""
    # Simple heuristic: check for code-like patterns
    code_indicators = [
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "if ",
        "for ",
        "while ",
    ]
    text_lower = text.lower()

    indicator_count = sum(1 for indicator in code_indicators if indicator in text_lower)
    quality_score = min(1.0, indicator_count * 0.2)

    return quality_score


def run_code_fix_evals(
    manifest_path: str,
    output_dir: str,
    seed: int = 42,
    model_name: str = "gpt2",
    max_length: int = 512,
    pad_direction: str = "right",
    truncate_at: Optional[int] = None,
):
    """Run code fix evaluations with strict logging."""

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

    # Evaluation loop
    model.eval()

    # Log file
    log_path = output_path / "code_fix_eval_log.jsonl"

    with open(log_path, "w") as log_file:
        for i, item in enumerate(manifest):
            prompt = item["prompt"]

            # Tokenize
            inputs = tokenizer(
                prompt,
                max_length=max_length,
                padding="max_length" if pad_direction == "right" else False,
                truncation=True,  # Always truncate to avoid position embedding issues
                return_tensors="pt",
            )

            # Apply truncation if specified
            if truncate_at and inputs["input_ids"].shape[1] > truncate_at:
                if pad_direction == "right":
                    inputs["input_ids"] = inputs["input_ids"][:, :truncate_at]
                    inputs["attention_mask"] = inputs["attention_mask"][:, :truncate_at]
                else:
                    inputs["input_ids"] = inputs["input_ids"][:, -truncate_at:]
                    inputs["attention_mask"] = inputs["attention_mask"][
                        :, -truncate_at:
                    ]

            # Generate output
            with torch.no_grad():
                generated = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=128,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            # Compute code quality score
            quality_score = compute_code_quality_score(output_text)

            # Log entry
            log_entry = {
                "global_step": i,
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
                "loss": None,  # Not applicable for evaluation
                "reward_scalar": quality_score,
                "pairwise_logit": float("nan"),  # Not applicable for code generation
                "heuristic_score": quality_score,
                "kl_to_ref": float("nan"),  # Not applicable for code generation
            }

            # Write to log file
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            if i % 10 == 0:
                print(
                    f"Processed {i+1}/{len(manifest)} samples, Quality score: {quality_score:.3f}"
                )

    print(f"Code fix evaluation complete! Logs saved to: {log_path}")
    return str(log_path)


def main():
    parser = argparse.ArgumentParser(description="Run code fix evaluations")
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to dataset manifest"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for logs"
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

    run_code_fix_evals(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        model_name=args.model,
        max_length=args.max_length,
        pad_direction=args.pad_direction,
        truncate_at=args.truncate_at,
    )


if __name__ == "__main__":
    main()
