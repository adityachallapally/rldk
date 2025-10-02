#!/usr/bin/env python3
"""
Test RLDK with full-scale GRPO training using GPT-2 and IMDB dataset.

This script uses real GPT-2 (124M parameters) with the IMDB dataset to test
RLDK's monitoring and forensics capabilities for GRPO algorithm. Uses unstable
hyperparameters to trigger monitoring alerts.
"""

import os

os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from rldk.integrations.trl import EventWriterCallback, create_grpo_config
from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer


def prepare_dataset(tokenizer, num_samples=500):
    """Load and prepare IMDB dataset for GRPO training."""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    prompts = []
    references = []
    accepted = []

    for text in dataset["text"]:
        prompt = text[:100]
        prompts.append(prompt)
        references.append(text)
        accepted.append(True)

    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "reference_response": references,
            "accepted": accepted,
        }
    )


def main():
    output_dir = Path("artifacts/test_full_grpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "gpt2"
    print("=== Testing RLDK with Full GRPO Training ===")
    print(f"Model: {model_name} (GPT-2, 124M parameters)")
    print("Dataset: IMDB (real HuggingFace dataset)")
    print("Steps: 150")
    print(f"Output: {output_dir}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"

    dataset = prepare_dataset(tokenizer, num_samples=500)

    print("Loading model with value head...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    print("Configuring GRPO with intentionally unstable hyperparameters...")
    grpo_config = create_grpo_config(
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_grad_norm=0.1,
        num_train_epochs=1,
        max_steps=150,
        logging_steps=1,
        output_dir=str(output_dir),
        remove_unused_columns=False,
        seed=42,
        run_name="test_full_grpo",
    )

    events_path = output_dir / "events.jsonl"
    callback = EventWriterCallback(events_path, run_id="test_full_grpo")

    print("Creating GRPO trainer with RLDK monitoring...")
    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[callback],
    )

    print("Starting GRPO training for 150 steps...")
    print(f"Monitoring events will be written to: {events_path}")
    print()

    try:
        grpo_trainer.train()

        print("\n=== GRPO Training Completed Successfully ===")
        print(f"Events logged to: {events_path}")
        print("\nNext steps to verify RLDK functionality:")
        print("1. Monitor with GRPO-specific alerts:")
        print(
            f"   rldk monitor --stream {events_path} --rules grpo_safe --alerts {output_dir}/alerts.jsonl"
        )
        print("2. Scan for anomalies:")
        print(f"   rldk forensics log-scan {events_path}")
        print("3. Analyze KL drift:")
        print(f"   rldk forensics kl-drift --run {events_path}")
        print("4. Comprehensive diagnostics:")
        print(f"   rldk forensics doctor {output_dir}")

    except Exception as e:
        print("\n!!! CRITICAL ERROR during GRPO training !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
