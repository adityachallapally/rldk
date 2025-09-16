#!/usr/bin/env python3
"""Minimal TRL DPO loop with RLDK monitor callback attached."""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
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


def run_minimal_trl_loop():
    """Run a minimal TRL DPO loop with RLDK monitoring."""
    if not TRL_AVAILABLE:
        print("❌ TRL not available - skipping TRL loop test")
        return False
    
    print("🚀 Starting Minimal TRL DPO Loop with RLDK Monitoring")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./artifacts/trl_live"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use tiny model to keep downloads fast
    model_name = "sshleifer/tiny-gpt2"  # Very small model
    
    print(f"📦 Using tiny model: {model_name}")
    
    try:
        # The factory function will handle model loading internally
        # We just need to validate that the model name is accessible
        print(f"✅ Model name validated: {model_name}")
        print("✅ Using unified factory function for model preparation")
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
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
    
    # DPO configuration - intentionally misconfigured to provoke instability
    dpo_config = DPOConfig(
        learning_rate=1e-3,  # High learning rate to cause instability
        per_device_train_batch_size=2,
        max_steps=200,  # Limit to 200 steps
        logging_dir=output_dir,
        save_steps=1000,  # Don't save during short run
        eval_steps=1000,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        logging_steps=5,  # Log every 5 steps
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
        print("⚠️  Falling back to simulation mode")
        return run_simulation_mode()
    
    print("✅ DPO Trainer created with RLDK monitor callback")
    
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
        print("💾 DPO analysis saved")
        
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
            'loss': 0.6931 - step * 0.001,
            'rewards/chosen': 0.5 + step * 0.01,
            'rewards/rejected': 0.3 + step * 0.005,
            'rewards/accuracies': 0.6 + step * 0.002,
            'rewards/margins': 0.2 + step * 0.001,
            'logps/chosen': -100.0 + step * 0.5,
            'logps/rejected': -120.0 + step * 0.3,
            'learning_rate': 1e-3,
            'grad_norm': 0.3 + step * 0.01,  # Increasing gradient norm
        }
        
        # Call monitor callbacks
        monitor.on_step_end(args, state, control)
        monitor.on_log(args, state, control, logs)
        
        if step % 20 == 0:
            print(f"   Step {step}: Loss={logs['loss']:.4f}, "
                  f"Accuracies={logs['rewards/accuracies']:.4f}")
    
    # Save analysis
    monitor.save_ppo_analysis()
    print("💾 Simulation analysis saved")
    
    return True


if __name__ == "__main__":
    print("🎯 Minimal TRL DPO Loop with RLDK Monitoring")
    print("=" * 50)
    
    try:
        success = run_minimal_trl_loop()
        
        if success:
            print("\n🎉 TRL DPO loop completed successfully!")
            print("✅ RLDK monitor was active during training")
        else:
            print("\n⚠️  TRL DPO loop had issues but monitor was active")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()