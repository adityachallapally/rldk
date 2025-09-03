#!/usr/bin/env python3
"""
Direct test of RLDK functionality with mock training data.
"""

import json
import os
import random
import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def create_mock_training_run(run_id: str, seed: int = 42, modify_tokenizer: bool = False):
    """Create mock training run data for RLDK testing."""
    print(f"Creating mock training run: {run_id}")
    
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Modify tokenizer if requested
    if modify_tokenizer:
        if tokenizer.padding_side == 'right':
            tokenizer.padding_side = 'left'
        else:
            tokenizer.padding_side = 'right'
        print(f"Modified tokenizer padding side to: {tokenizer.padding_side}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create mock metrics
    metrics_history = []
    for step in range(20):
        # Simulate training metrics with some variation
        kl = 0.1 + random.gauss(0, 0.02)
        reward = 0.5 + random.gauss(0, 0.1)
        loss = 0.3 + random.gauss(0, 0.05)
        
        # Add some systematic differences between runs
        if run_id == "b" and modify_tokenizer:
            # Simulate different behavior due to tokenizer change
            reward += 0.1
            kl += 0.05
        elif run_id == "c" and seed != 42:
            # Simulate different behavior due to different seed
            reward -= 0.05
            loss += 0.02
        
        step_metrics = {
            "step": step,
            "kl": max(0, kl),
            "reward_mean": reward,
            "reward_std": 0.1 + random.gauss(0, 0.02),
            "loss": max(0, loss),
            "policy_loss": loss * 0.7,
            "value_loss": loss * 0.3,
        }
        
        metrics_history.append(step_metrics)
    
    # Create mock probe outputs
    probe_prompts = [
        "Explain the concept of artificial intelligence.",
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
        "What is the future of technology?",
        "Describe the process of photosynthesis."
    ]
    
    probe_outputs = []
    for step in [0, 5, 10, 15, 19]:  # Sample at different steps
        for prompt in probe_prompts:
            # Generate mock response
            response = f"This is a mock response to: {prompt[:30]}... The answer involves several key concepts and principles that are important to understand."
            
            # Mock reward with some variation
            base_reward = 0.6 + random.gauss(0, 0.1)
            if run_id == "b" and modify_tokenizer:
                base_reward += 0.2  # Different due to tokenizer
            elif run_id == "c" and seed != 42:
                base_reward -= 0.1  # Different due to seed
            
            probe_outputs.append({
                "step": step,
                "prompt": prompt,
                "response": response,
                "reward": base_reward,
                "generation_kwargs": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
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
    
    # Save model (just copy the tokenizer config)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "total_steps": len(metrics_history),
        "random_seed": seed,
        "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "model_name": "gpt2",
        "training_time_minutes": 5.0,  # Mock time
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create trace file
    trace_data = {
        "training_config": {
            "learning_rate": 1e-5,
            "batch_size": 2,
            "max_steps": 20,
            "seed": seed
        },
        "training_time_seconds": 300,
        "total_steps": len(metrics_history),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "random_seed": seed,
        "model_name": "gpt2",
        "tokenizer_config": {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id
        }
    }
    
    with open(os.path.join(output_dir, "trace.json"), 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"Saved mock artifacts to {output_dir}")
    return output_dir

def create_mock_reward_model():
    """Create mock reward model for testing."""
    print("Creating mock reward model...")
    
    # Create a simple mock reward model
    model_dir = "./rldk_demos/rm_a"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save mock model files
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.save_pretrained(model_dir)
    
    # Create mock config
    config = {
        "model_type": "distilbert",
        "hidden_size": 768,
        "num_labels": 1,
        "vocab_size": tokenizer.vocab_size
    }
    
    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create mock evaluation pairs
    eval_pairs = []
    for i in range(50):
        eval_pairs.append({
            "prompt": f"Test prompt {i}",
            "chosen": f"This is a good response to prompt {i}",
            "rejected": f"This is a bad response to prompt {i}",
            "metadata": {"generated": True}
        })
    
    with open("./rldk_demos/rm_eval_pairs.jsonl", 'w') as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Create mock metadata
    metadata = {
        "model_name": "distilbert-base-uncased",
        "training_samples": 1000,
        "validation_samples": 250,
        "training_time_minutes": 30.0,
        "random_seed": 42,
        "evaluation_results": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
            "chosen_reward_mean": 0.6,
            "rejected_reward_mean": 0.3,
            "preference_margin_mean": 0.3
        }
    }
    
    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved mock reward model to {model_dir}")
    return model_dir

def main():
    """Create mock training data and test RLDK."""
    print("Creating mock training data for RLDK testing...")
    
    # Create mock reward model
    create_mock_reward_model()
    
    # Create mock PPO runs
    create_mock_training_run("a", seed=42, modify_tokenizer=False)
    create_mock_training_run("b", seed=42, modify_tokenizer=True)  # Tokenizer change
    create_mock_training_run("c", seed=123, modify_tokenizer=False)  # Different seed
    
    print("\nMock training data created!")
    print("Now testing RLDK commands...")

if __name__ == "__main__":
    main()