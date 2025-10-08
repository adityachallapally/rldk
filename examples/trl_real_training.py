#!/usr/bin/env python3
"""Real TRL PPO training with RLDK monitoring."""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# Import RLDK utilities
from rldk.integrations.trl import create_ppo_trainer, tokenize_text_column
from rldk.integrations.trl.monitors import PPOMonitor as Monitor

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


class SimpleRewardModel(nn.Module):
    """Simple reward model for PPO training."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Simple reward head
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        
        # Pool hidden states (mean pooling)
        if attention_mask is not None:
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
            
        # Get reward
        reward = self.reward_head(pooled)
        return reward


def create_tiny_dataset():
    """Create a tiny dataset for testing."""
    prompts = [
        "Hello world",
        "Python is",
        "AI can",
        "Machine learning",
        "Deep learning",
    ] * 4  # 20 samples total
    
    responses = [
        "a programming language",
        "helpful for automation", 
        "solve complex problems",
        "uses neural networks",
        "requires lots of data",
    ] * 4
    
    return Dataset.from_dict({
        "prompt": prompts,
        "response": responses,
    })


def run_real_trl_training():
    """Run actual TRL PPO training with RLDK monitoring."""
    if not TRL_AVAILABLE:
        print("❌ TRL not available - cannot run real training")
        return False
    
    print("🚀 Starting REAL TRL PPO Training with RLDK Monitoring")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./artifacts/trl_real"
    os.makedirs(output_dir, exist_ok=True)
    event_log_path = os.path.join(output_dir, "trl_real_training_events.jsonl")
    
    # Use tiny model to keep downloads fast
    model_name = "sshleifer/tiny-gpt2"  # Very small model
    
    print(f"📦 Using tiny model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create policy model with value head
        policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create reference model with value head for compatibility
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

        # Create reward model
        reward_model = SimpleRewardModel(base_model)

        # Create value model (same as policy model for simplicity)
        value_model = policy_model

        print("✅ All models loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Create dataset
    dataset = create_tiny_dataset()
    print(f"📊 Dataset created with {len(dataset)} samples")

    dataset = tokenize_text_column(
        dataset,
        tokenizer,
        text_column="prompt",
        padding=True,
        truncation=True,
        keep_original=False,
        desc="Tokenizing TRL real training prompts",
    )
    if hasattr(dataset, "remove_columns"):
        if "response" in getattr(dataset, "column_names", []):
            dataset = dataset.remove_columns(["response"])
    else:  # pragma: no cover - used in tests with lightweight dataset stubs
        for record in dataset:
            record.pop("response", None)

    # Initialize RLDK monitor with low thresholds to trigger alerts
    monitor = Monitor(
        output_dir=output_dir,
        kl_threshold=0.05,  # Very low threshold to trigger alerts
        reward_threshold=0.01,
        gradient_threshold=0.5,
        clip_frac_threshold=0.1,
        run_id="trl_real_training"
    )
    
    print("✅ RLDK Monitor initialized")
    
    # PPO configuration - intentionally misconfigured to provoke instability
    ppo_config = PPOConfig(
        learning_rate=1e-3,  # High learning rate to cause instability
        per_device_train_batch_size=2,
        mini_batch_size=1,
        num_ppo_epochs=1,
        max_grad_norm=0.1,  # Low max grad norm to cause clipping
        logging_dir=output_dir,
        save_steps=1000,  # Don't save during short run
        eval_steps=1000,
        num_train_epochs=1,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        max_steps=50,  # Limit to 50 steps for testing
        logging_steps=5,  # Log every 5 steps
        # CPU-only settings
        dataloader_num_workers=0,
        use_cpu=True,
    )
    
    print("⚙️  PPO Config: High LR, Low grad norm (intentionally unstable)")
    
    try:
        trainer = create_ppo_trainer(
            model_name=model_name,
            ppo_config=ppo_config,
            train_dataset=dataset,
            callbacks=[monitor],
            tokenizer=tokenizer,
            model=policy_model,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            event_log_path=event_log_path,
        )
        print("✅ PPO Trainer created with RLDK monitor callback")
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False
    
    # Start training
    print("🎯 Starting REAL PPO training (CPU only)...")
    start_time = time.time()
    
    try:
        # Train for a small number of steps
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save final analysis
        monitor.save_ppo_analysis()
        print("💾 PPO analysis saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎯 Real TRL PPO Training with RLDK Monitoring")
    print("=" * 50)
    
    try:
        success = run_real_trl_training()
        
        if success:
            print("\n🎉 REAL TRL training completed successfully!")
            print("✅ RLDK monitor was active during actual training")
        else:
            print("\n❌ Real TRL training failed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()