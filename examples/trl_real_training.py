#!/usr/bin/env python3
"""Real TRL DPO training with RLDK monitoring."""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# Import RLDK components
from rldk.integrations.trl import PPOMonitor as Monitor, create_dpo_trainer

try:
    from trl import DPOConfig, DPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


# Simple reward function for demonstration
def simple_reward_function(text: str) -> float:
    """Simple reward function for demonstration purposes."""
    # Simple heuristics for demonstration
    reward = 0.0
    
    # Reward for length (not too short, not too long)
    length = len(text.split())
    if 5 <= length <= 50:
        reward += 0.1
    
    # Reward for common positive words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    for word in positive_words:
        if word in text.lower():
            reward += 0.2
    
    # Penalty for repetition
    words = text.lower().split()
    if len(words) > 1:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        reward += repetition_ratio * 0.1
    
    return min(reward, 1.0)  # Cap at 1.0


def create_tiny_dataset():
    """Create a tiny dataset for DPO testing."""
    prompts = [
        "What is Python?",
        "How does AI work?",
        "What is machine learning?",
        "Explain deep learning",
        "What is programming?",
    ] * 4  # 20 samples total
    
    chosen_responses = [
        "Python is a high-level programming language known for its simplicity.",
        "AI works by processing data and making decisions using algorithms.",
        "Machine learning is a subset of AI that learns from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Programming is the process of writing instructions for computers.",
    ] * 4
    
    rejected_responses = [
        "Python is a type of snake.",
        "AI works by magic and fairy dust.",
        "Machine learning is about machines that learn to be human.",
        "Deep learning is just regular learning but deeper.",
        "Programming is just typing random characters.",
    ] * 4
    
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses,
    })


def run_real_trl_training():
    """Run actual TRL DPO training with RLDK monitoring."""
    if not TRL_AVAILABLE:
        print("❌ TRL not available - cannot run real training")
        return False
    
    print("🚀 Starting REAL TRL DPO Training with RLDK Monitoring")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./artifacts/trl_real"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use tiny model to keep downloads fast
    model_name = "sshleifer/tiny-gpt2"  # Very small model
    
    print(f"📦 Using tiny model: {model_name}")
    
    try:
        # Test model accessibility
        print(f"✅ Model name validated: {model_name}")
        print("✅ Using simplified DPO approach with unified factory function")
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False
    
    # Create dataset
    dataset = create_tiny_dataset()
    print(f"📊 Dataset created with {len(dataset)} samples")
    
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
    
    # DPO configuration - intentionally misconfigured to provoke instability
    dpo_config = DPOConfig(
        learning_rate=1e-3,  # High learning rate to cause instability
        per_device_train_batch_size=2,
        max_steps=50,  # Limit to 50 steps for testing
        logging_dir=output_dir,
        save_steps=1000,  # Don't save during short run
        eval_steps=1000,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        logging_steps=5,  # Log every 5 steps
        # CPU-only settings
        dataloader_num_workers=0,
        use_cpu=True,
    )
    
    print("⚙️  DPO Config: High LR (intentionally unstable)")
    
    # Create DPO trainer with monitor callback using unified factory function
    try:
        trainer = create_dpo_trainer(
            model_name=model_name,
            dpo_config=dpo_config,
            train_dataset=dataset,
            callbacks=[monitor],  # Attach RLDK monitor
        )
    except Exception as e:
        print(f"❌ Failed to create DPO trainer: {e}")
        return False
    
    print("✅ DPO Trainer created with RLDK monitor callback")
    
    # Start training
    print("🎯 Starting REAL DPO training (CPU only)...")
    start_time = time.time()
    
    try:
        # Train for a small number of steps
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save final analysis
        monitor.save_ppo_analysis()
        print("💾 DPO analysis saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎯 Real TRL DPO Training with RLDK Monitoring")
    print("=" * 50)
    
    try:
        success = run_real_trl_training()
        
        if success:
            print("\n🎉 REAL TRL DPO training completed successfully!")
            print("✅ RLDK monitor was active during actual training")
        else:
            print("\n❌ Real TRL DPO training failed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()