#!/usr/bin/env python3
"""
Minimal PPO training test to verify RLDK functionality.
"""

import json
import os
import random
import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def create_simple_reward_model():
    """Create a simple reward model for testing."""
    class SimpleRewardModel:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def get_reward(self, prompt: str, response: str) -> float:
            """Simple reward based on response length and content."""
            # Simple heuristic: longer, more detailed responses get higher rewards
            length_reward = min(len(response.split()) / 50.0, 1.0)
            
            # Bonus for certain keywords
            content_bonus = 0.0
            if any(word in response.lower() for word in ["helpful", "detailed", "explain", "understand"]):
                content_bonus = 0.2
            
            # Add some randomness to simulate real reward model
            noise = random.gauss(0, 0.1)
            
            return length_reward + content_bonus + noise
    
    return SimpleRewardModel()

def run_minimal_ppo(run_id: str, seed: int = 42, modify_tokenizer: bool = False):
    """Run minimal PPO training."""
    print(f"Running minimal PPO training: {run_id}")
    
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load prompts
    prompts = []
    with open("./rldk_demos/ppo_prompts.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    
    # Initialize model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Modify tokenizer if requested
    if modify_tokenizer:
        if tokenizer.padding_side == 'right':
            tokenizer.padding_side = 'left'
        else:
            tokenizer.padding_side = 'right'
        print(f"Modified tokenizer padding side to: {tokenizer.padding_side}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Create reward model
    reward_model = create_simple_reward_model()
    
    # PPO config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,  # Very small for quick testing
        mini_batch_size=1,
        num_ppo_epochs=2,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.2,
        seed=seed,
        max_steps=10,  # Very few steps for quick testing
        save_steps=5,
        logging_steps=2,
        report_to=None,
        fp16=False,
        bf16=False,
    )
    
    # Create trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=prompts[:5],  # Use only 5 prompts for speed
    )
    
    # Training loop
    metrics_history = []
    
    for step in range(ppo_config.max_steps):
        # Sample prompts
        batch_prompts = [random.choice(prompts[:5]) for _ in range(ppo_config.batch_size)]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            batch_prompts,
            return_prompt=False,
            max_new_tokens=20,  # Short responses for speed
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode responses
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        # Get rewards
        rewards = []
        for prompt, response in zip(batch_prompts, responses):
            reward = reward_model.get_reward(prompt, response)
            rewards.append(reward)
        
        # PPO step
        stats = ppo_trainer.step(batch_prompts, responses, rewards)
        
        # Log metrics
        step_metrics = {
            "step": step,
            "kl": stats.get("ppo/mean_kl", 0.0),
            "reward_mean": sum(rewards) / len(rewards),
            "loss": stats.get("ppo/loss/total", 0.0),
        }
        
        metrics_history.append(step_metrics)
        print(f"Step {step}: KL={step_metrics['kl']:.4f}, Reward={step_metrics['reward_mean']:.4f}")
    
    # Save artifacts
    output_dir = f"./rldk_demos/ppo_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.jsonl"), 'w') as f:
        for metrics in metrics_history:
            f.write(json.dumps(metrics) + '\n')
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "total_steps": len(metrics_history),
        "random_seed": seed,
        "tokenizer_padding_side": tokenizer.padding_side,
        "final_metrics": metrics_history[-1] if metrics_history else {},
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved artifacts to {output_dir}")
    return output_dir

def main():
    """Run minimal PPO tests."""
    print("Running minimal PPO training tests for RLDK validation...")
    
    # Run baseline
    run_minimal_ppo("a", seed=42, modify_tokenizer=False)
    
    # Run variant with tokenizer modification
    run_minimal_ppo("b", seed=42, modify_tokenizer=True)
    
    # Run variant with different seed
    run_minimal_ppo("c", seed=123, modify_tokenizer=False)
    
    print("Minimal PPO training completed!")
    print("Now testing RLDK commands...")

if __name__ == "__main__":
    main()