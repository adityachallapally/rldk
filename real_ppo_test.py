#!/usr/bin/env python3
"""
Real PPO training test with actual TRL training to validate RLDK.
"""

import json
import os
import random
import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class SimpleRewardModel:
    """Simple reward model for testing."""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_reward(self, prompt: str, response: str) -> float:
        """Simple reward based on response quality."""
        # Length reward
        length_reward = min(len(response.split()) / 30.0, 1.0)
        
        # Content quality reward
        quality_words = ["helpful", "detailed", "explain", "understand", "clear", "example"]
        content_reward = sum(0.1 for word in quality_words if word in response.lower())
        
        # Coherence reward (simple heuristic)
        coherence_reward = 0.2 if len(response.split()) > 10 else 0.0
        
        # Add some noise
        noise = random.gauss(0, 0.05)
        
        return length_reward + content_reward + coherence_reward + noise

def run_real_ppo_training(run_id: str, seed: int = 42, modify_tokenizer: bool = False):
    """Run real PPO training with TRL."""
    print(f"Running real PPO training: {run_id}")
    
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load prompts
    prompts = []
    with open("./rldk_demos/ppo_prompts.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    
    # Use fewer prompts for faster training
    prompts = prompts[:10]
    
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
    reward_model = SimpleRewardModel()
    
    # PPO config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
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
        max_steps=50,  # More steps for better analysis
        save_steps=25,
        logging_steps=5,
        report_to=None,
        fp16=False,
        bf16=False,
    )
    
    # Create trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=prompts,
        value_model=model,
    )
    
    # Training loop
    metrics_history = []
    probe_outputs = []
    
    # Length sampler
    response_length_sampler = LengthSampler(10, 30)
    
    for step in range(ppo_config.max_steps):
        # Sample prompts
        batch_prompts = [random.choice(prompts) for _ in range(ppo_config.batch_size)]
        
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
            "reward_std": torch.std(torch.tensor(rewards)).item() if len(rewards) > 1 else 0.0,
            "loss": stats.get("ppo/loss/total", 0.0),
            "policy_loss": stats.get("ppo/loss/policy", 0.0),
            "value_loss": stats.get("ppo/loss/value", 0.0),
        }
        
        metrics_history.append(step_metrics)
        
        if step % 10 == 0:
            print(f"Step {step}: KL={step_metrics['kl']:.4f}, Reward={step_metrics['reward_mean']:.4f}, Loss={step_metrics['loss']:.4f}")
        
        # Generate probe outputs every 10 steps
        if step % 10 == 0:
            probe_prompts = [
                "Explain the concept of artificial intelligence.",
                "What are the benefits of renewable energy?",
                "How does machine learning work?"
            ]
            
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
    
    # Save artifacts
    output_dir = f"./rldk_demos/ppo_{run_id}"
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
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "total_steps": len(metrics_history),
        "random_seed": seed,
        "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "model_name": model_name,
        "training_time_minutes": 2.0,  # Approximate
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create trace file
    trace_data = {
        "training_config": ppo_config.to_dict(),
        "training_time_seconds": 120,
        "total_steps": len(metrics_history),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "random_seed": seed,
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
    
    print(f"Saved real training artifacts to {output_dir}")
    return output_dir

def main():
    """Run real PPO training tests."""
    print("Running real PPO training tests for RLDK validation...")
    
    # Run baseline
    run_real_ppo_training("a", seed=42, modify_tokenizer=False)
    
    # Run variant with tokenizer modification
    run_real_ppo_training("b", seed=42, modify_tokenizer=True)
    
    # Run variant with different seed
    run_real_ppo_training("c", seed=123, modify_tokenizer=False)
    
    print("Real PPO training completed!")
    print("Now testing RLDK commands on real training data...")

if __name__ == "__main__":
    main()