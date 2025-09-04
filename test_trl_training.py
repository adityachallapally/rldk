#!/usr/bin/env python3
"""Minimal TRL training test with RLDK integration."""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor

def test_trl_training():
    """Test TRL training with RLDK integration."""
    
    # Create output directory
    output_dir = "runs/trl_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting TRL training test with RLDK integration")
    
    # Load model and tokenizer
    model_name = "sshleifer/tiny-gpt2"
    print(f"📥 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Load dataset
    print("📥 Loading IMDB dataset (1% sample)")
    dataset = load_dataset("imdb", split="train[:1%]")
    
    # Prepare dataset for PPO
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("text", "query")
    tokenized_dataset = tokenized_dataset.remove_columns(["label"])
    
    # Create RLDK callbacks
    rldk_callback = RLDKCallback(
        output_dir=output_dir,
        run_id="trl_test_run",
        log_interval=1
    )
    
    ppo_monitor = PPOMonitor(
        output_dir=output_dir,
        run_id="trl_test_run"
    )
    
    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        run_id="trl_test_run"
    )
    
    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        num_ppo_epochs=1,
        max_steps=20,
        logging_steps=1,
        save_steps=10,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
    )
    
    # Create PPO trainer with all required models
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=model,  # Use same model as reference
        tokenizer=tokenizer,
        dataset=tokenized_dataset,
        data_collator=None,  # Use default
    )
    
    print("✅ PPO Trainer created successfully")
    print(f"📊 Training on {len(tokenized_dataset)} samples")
    
    # Add callbacks
    trainer.add_callback(rldk_callback)
    trainer.add_callback(ppo_monitor)
    trainer.add_callback(checkpoint_monitor)
    
    print("🚀 Starting training...")
    
    # Train
    trainer.train()
    
    print("✅ Training completed successfully")
    
    # Check if output files were created
    run_dir = os.path.join(output_dir, "trl_test_run")
    if os.path.exists(run_dir):
        print(f"📁 Run directory created: {run_dir}")
        files = os.listdir(run_dir)
        print(f"📄 Generated files: {files}")
        
        # Check for metrics.jsonl
        metrics_file = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(metrics_file):
            print(f"✅ Metrics file created: {metrics_file}")
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                print(f"📊 Metrics entries: {len(lines)}")
                if lines:
                    print(f"📝 Last entry: {lines[-1].strip()}")
        else:
            print("⚠️  No metrics.jsonl file found")
            
        # Check for alert files
        alert_files = [f for f in files if f.startswith("alerts_step_")]
        if alert_files:
            print(f"⚠️  Alert files found: {alert_files}")
        else:
            print("ℹ️  No alert files found (this is normal for short training)")
    else:
        print("⚠️  No run directory created")
    
    return True

if __name__ == "__main__":
    success = test_trl_training()
    if success:
        print("🎯 TRL training test completed successfully")
    else:
        print("❌ TRL training test failed")