#!/usr/bin/env python3
"""
03_run_ppo_baseline_full.py - Run TRL PPO training with GPT-2

This script runs a full PPO training loop using TRL with GPT-2 as the policy model
and the trained reward model for scoring. CPU-optimized settings for 200+ steps.
"""

import json
import os
import random
import hashlib
import time
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class RewardModelWrapper:
    """Wrapper for the trained reward model."""
    
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModel
        import torch.nn as nn
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        # Add classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        self.classifier.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu'))
        
    def get_reward(self, prompt: str, response: str) -> float:
        """Get reward for a prompt-response pair."""
        text = f"{prompt} [SEP] {response}"
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
            reward = self.classifier(pooled_output).squeeze().item()
        
        return reward

def load_ppo_prompts(filename: str) -> List[str]:
    """Load PPO prompts from JSONL file."""
    prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    return prompts

def load_probe_prompts(filename: str) -> List[str]:
    """Load probe prompts from JSONL file."""
    prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    return prompts

def create_ppo_config() -> PPOConfig:
    """Create PPO configuration optimized for CPU training."""
    return PPOConfig(
        model_name="gpt2",
        learning_rate=1.41e-5,
        batch_size=2,  # CPU-optimized
        mini_batch_size=1,  # CPU-optimized
        ppo_epochs=4,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.2,
        log_with="tensorboard",
        tracker_project_name="rldk_ppo_baseline",
        tracker_kwargs={"logging_dir": "./rldk_demos/ppo_a/logs"},
        seed=RANDOM_SEED,
        optimize_cuda_cache=False,  # CPU optimization
        dataloader_num_workers=0,  # CPU optimization
        remove_unused_columns=False,
        gradient_accumulation_steps=4,  # Effective batch size: 8
        max_steps=250,  # 200+ steps as required
        save_steps=50,
        eval_steps=25,
        logging_steps=10,
        warmup_steps=20,
        report_to=None,  # Disable wandb for CPU training
        fp16=False,  # Disable mixed precision for CPU
        bf16=False,  # Disable bfloat16 for CPU
    )

def run_ppo_training():
    """Run PPO training with comprehensive logging."""
    print("Starting PPO baseline training...")
    print(f"Using random seed: {RANDOM_SEED}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CPU' if not torch.cuda.is_available() else 'CUDA'}")
    
    # Load data
    print("\nLoading training data...")
    ppo_prompts = load_ppo_prompts("./rldk_demos/ppo_prompts.jsonl")
    probe_prompts = load_probe_prompts("./rldk_demos/probes.jsonl")
    
    print(f"PPO prompts: {len(ppo_prompts)}")
    print(f"Probe prompts: {len(probe_prompts)}")
    
    # Initialize tokenizer and model
    print("\nInitializing tokenizer and model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Initialize reward model
    print("\nLoading reward model...")
    reward_model = RewardModelWrapper("./rldk_demos/rm_a")
    
    # Create PPO config
    ppo_config = create_ppo_config()
    
    # Create PPO trainer
    print("\nInitializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,  # Use model itself as reference
        tokenizer=tokenizer,
        dataset=ppo_prompts,
    )
    
    # Training loop
    print("\nStarting PPO training loop...")
    start_time = time.time()
    
    # Metrics tracking
    metrics_history = []
    probe_outputs = []
    
    # Length sampler for response generation
    response_length_sampler = LengthSampler(20, 50)
    
    for step in range(ppo_config.max_steps):
        step_start_time = time.time()
        
        # Sample prompts for this step
        batch_prompts = [random.choice(ppo_prompts) for _ in range(ppo_config.batch_size)]
        
        # Generate responses
        response_length = response_length_sampler()
        generation_kwargs = {
            "max_new_tokens": response_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        response_tensors = ppo_trainer.generate(
            batch_prompts,
            return_prompt=False,
            **generation_kwargs
        )
        
        # Decode responses
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        # Get rewards from reward model
        rewards = []
        for prompt, response in zip(batch_prompts, responses):
            reward = reward_model.get_reward(prompt, response)
            rewards.append(reward)
        
        # PPO step
        stats = ppo_trainer.step(batch_prompts, responses, rewards)
        
        step_time = time.time() - step_start_time
        
        # Log metrics
        step_metrics = {
            "step": step,
            "kl": stats.get("ppo/mean_kl", 0.0),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "loss": stats.get("ppo/loss/total", 0.0),
            "policy_loss": stats.get("ppo/loss/policy", 0.0),
            "value_loss": stats.get("ppo/loss/value", 0.0),
            "sequence_length": response_length,
            "step_time": step_time
        }
        
        metrics_history.append(step_metrics)
        
        # Log progress
        if step % 10 == 0:
            print(f"Step {step}/{ppo_config.max_steps}: "
                  f"KL={step_metrics['kl']:.4f}, "
                  f"Reward={step_metrics['reward_mean']:.4f}±{step_metrics['reward_std']:.4f}, "
                  f"Loss={step_metrics['loss']:.4f}")
        
        # Generate probe outputs every 25 steps
        if step % 25 == 0:
            print(f"Generating probe outputs at step {step}...")
            for probe_prompt in probe_prompts:
                probe_response = ppo_trainer.generate([probe_prompt], **generation_kwargs)
                probe_text = tokenizer.decode(probe_response[0].squeeze(), skip_special_tokens=True)
                probe_reward = reward_model.get_reward(probe_prompt, probe_text)
                
                probe_outputs.append({
                    "step": step,
                    "prompt": probe_prompt,
                    "response": probe_text,
                    "reward": probe_reward,
                    "generation_kwargs": generation_kwargs
                })
    
    training_time = time.time() - start_time
    print(f"PPO training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save artifacts
    print("\nSaving training artifacts...")
    output_dir = "./rldk_demos/ppo_a"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.jsonl"), 'w') as f:
        for metrics in metrics_history:
            f.write(json.dumps(metrics) + '\n')
    
    # Save probe outputs
    with open(os.path.join(output_dir, "probes_outputs.jsonl"), 'w') as f:
        for probe in probe_outputs:
            f.write(json.dumps(probe, ensure_ascii=False) + '\n')
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Create trace file
    trace_data = {
        "training_config": ppo_config.to_dict(),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "total_steps": len(metrics_history),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "random_seed": RANDOM_SEED,
        "model_name": model_name,
        "tokenizer_config": {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id
        }
    }
    
    with open(os.path.join(output_dir, "trace.json"), 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    # Create metadata
    metadata = {
        "run_id": "ppo_a",
        "model_name": model_name,
        "training_samples": len(ppo_prompts),
        "probe_samples": len(probe_prompts),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "total_steps": len(metrics_history),
        "random_seed": RANDOM_SEED,
        "ppo_config": ppo_config.to_dict(),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "tokenizer_config": trace_data["tokenizer_config"],
        "data_hash": hashlib.sha256(json.dumps(ppo_prompts, sort_keys=True).encode()).hexdigest()
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nPPO baseline training completed!")
    print(f"Artifacts saved to: {output_dir}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Total steps: {len(metrics_history)}")
    if metrics_history:
        final_metrics = metrics_history[-1]
        print(f"Final KL divergence: {final_metrics['kl']:.4f}")
        print(f"Final reward mean: {final_metrics['reward_mean']:.4f}")
        print(f"Final loss: {final_metrics['loss']:.4f}")

if __name__ == "__main__":
    run_ppo_training()