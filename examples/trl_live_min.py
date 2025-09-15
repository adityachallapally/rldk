#!/usr/bin/env python3
"""Minimal TRL PPO loop with RLDK monitor callback attached."""

import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# Import RLDK monitor
from rldk.integrations.trl.monitors import PPOMonitor as Monitor

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


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


def run_minimal_trl_loop():
    """Run a minimal TRL PPO loop with RLDK monitoring."""
    if not TRL_AVAILABLE:
        print("❌ TRL not available - skipping TRL loop test")
        return False
    
    print("🚀 Starting Minimal TRL Loop with RLDK Monitoring")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./artifacts/trl_live"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use tiny model to keep downloads fast
    model_name = "sshleifer/tiny-gpt2"  # Very small model
    
    print(f"📦 Using tiny model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create reference model (same as policy model for simplicity)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("✅ Models loaded successfully")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("⚠️  Falling back to simulation mode")
        return run_simulation_mode()
    
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
        run_id="trl_live_min"
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
        max_steps=200,  # Limit to 200 steps
        logging_steps=5,  # Log every 5 steps
    )
    
    print("⚙️  PPO Config: High LR, Low grad norm (intentionally unstable)")
    
    # Create PPO trainer with monitor callback
    trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[monitor],  # Attach RLDK monitor
    )
    
    print("✅ PPO Trainer created with RLDK monitor callback")
    
    # Start training
    print("🎯 Starting training (CPU only)...")
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
        print("⚠️  This might be expected due to intentional misconfiguration")
        return False


def run_simulation_mode():
    """Run simulation mode if model loading fails."""
    print("🎭 Running Simulation Mode")
    print("=" * 40)
    
    output_dir = "./artifacts/trl_live"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize monitor
    monitor = Monitor(
        output_dir=output_dir,
        kl_threshold=0.05,
        reward_threshold=0.01,
        gradient_threshold=0.5,
        clip_frac_threshold=0.1,
        run_id="trl_simulation"
    )
    
    # Simulate training logs with instability
    from transformers import TrainerControl, TrainerState, TrainingArguments
    
    args = TrainingArguments(output_dir=output_dir)
    state = TrainerState()
    control = TrainerControl()
    
    print("🔄 Simulating training steps with instability...")
    
    for step in range(200):
        state.global_step = step
        state.epoch = step / 100.0
        
        # Simulate logs with increasing instability
        logs = {
            'ppo/rewards/mean': 0.5 + step * 0.01,
            'ppo/rewards/std': 0.1 + step * 0.005,  # Increasing variance
            'ppo/policy/kl_mean': 0.02 + step * 0.001,  # Increasing KL
            'ppo/policy/entropy': 2.0 - step * 0.01,
            'ppo/policy/clipfrac': 0.05 + step * 0.001,  # Increasing clip fraction
            'ppo/val/value_loss': 0.3 - step * 0.001,
            'learning_rate': 1e-3,
            'grad_norm': 0.3 + step * 0.01,  # Increasing gradient norm
        }
        
        # Call monitor callbacks
        monitor.on_step_end(args, state, control)
        monitor.on_log(args, state, control, logs)
        
        if step % 20 == 0:
            print(f"   Step {step}: KL={logs['ppo/policy/kl_mean']:.4f}, "
                  f"Reward_std={logs['ppo/rewards/std']:.4f}")
    
    # Save analysis
    monitor.save_ppo_analysis()
    print("💾 Simulation analysis saved")
    
    return True


if __name__ == "__main__":
    print("🎯 Minimal TRL Loop with RLDK Monitoring")
    print("=" * 50)
    
    try:
        success = run_minimal_trl_loop()
        
        if success:
            print("\n🎉 TRL loop completed successfully!")
            print("✅ RLDK monitor was active during training")
        else:
            print("\n⚠️  TRL loop had issues but monitor was active")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()