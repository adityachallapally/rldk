#!/usr/bin/env python3
"""
Test RLDK with full-scale PPO training using GPT-2 and IMDB dataset.

This script uses real GPT-2 (124M parameters) with the IMDB dataset to test
RLDK's monitoring and forensics capabilities. Intentionally uses unstable
hyperparameters to trigger monitoring alerts.
"""

import os

os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from rldk.integrations.trl import create_ppo_trainer
from trl import PPOConfig


def prepare_dataset(tokenizer, num_samples=500):
    """Load and prepare IMDB dataset for PPO training."""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    prompts = []
    for text in dataset["text"]:
        prompt = text[:100]
        prompts.append(prompt)

    from datasets import Dataset

    raw_dataset = Dataset.from_dict({"query": prompts})

    print("Tokenizing dataset for PPOTrainer...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["query"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None,
        )
        return tokenized

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["query"],
    )

    return tokenized_dataset


def main():
    output_dir = Path("artifacts/test_full_ppo")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "gpt2"
    print("=== Testing RLDK with Full PPO Training ===")
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
    print(f"Dataset prepared with columns: {dataset.column_names}")
    print(f"Sample dataset entry keys: {list(dataset[0].keys())}")

    print("Configuring PPO with intentionally unstable hyperparameters...")
    ppo_config = PPOConfig(
        run_name="test_full_ppo",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        num_ppo_epochs=2,
        max_grad_norm=0.1,
        kl_coef=0.01,
        cliprange=0.2,
        cliprange_value=0.2,
        max_steps=150,
        logging_steps=1,
        report_to=None,
        remove_unused_columns=False,
        seed=42,
        bf16=False,
        fp16=False,
    )

    events_path = output_dir / "events.jsonl"

    print("Creating PPO trainer with RLDK monitoring...")
    ppo_trainer = create_ppo_trainer(
        model_name=model_name,
        ppo_config=ppo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        event_log_path=events_path,
    )

    print("Starting PPO training for 150 steps...")
    print(f"Monitoring events will be written to: {events_path}")
    print()

    try:
        ppo_trainer.train()

        print("\n=== PPO Training Completed Successfully ===")
        print(f"Events logged to: {events_path}")
        print("\nNext steps to verify RLDK functionality:")
        print("1. Monitor with alerts:")
        print(
            f"   rldk monitor --stream {events_path} --rules ppo_safe --alerts {output_dir}/alerts.jsonl"
        )
        print("2. Scan for anomalies:")
        print(f"   rldk forensics log-scan {events_path}")
        print("3. Analyze KL drift:")
        print(f"   rldk forensics kl-drift --run {events_path}")
        print("4. Comprehensive diagnostics:")
        print(f"   rldk forensics doctor {output_dir}")

    except Exception as e:
        print("\n!!! CRITICAL ERROR during PPO training !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
