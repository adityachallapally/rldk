#!/usr/bin/env python3
"""
04_run_variants_with_issues_full.py - Run PPO variants with intentional issues

This script runs three PPO variants with intentional issues to test RLDK detection:
- ppo_b: Tokenizer padding side change (nondeterminism)
- ppo_c: Different seed and shuffle order (reward drift)
- ppo_d: Reward clamping/rescaling (saturation/hacking)
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

class ModifiedRewardModelWrapper(RewardModelWrapper):
    """Modified reward model with intentional issues."""
    
    def __init__(self, model_path: str, modification_type: str):
        super().__init__(model_path)
        self.modification_type = modification_type
        
    def get_reward(self, prompt: str, response: str) -> float:
        """Get modified reward based on modification type."""
        base_reward = super().get_reward(prompt, response)
        
        if self.modification_type == "clamp":
            # Clamp rewards to create saturation
            return np.clip(base_reward, -2.0, 2.0)
        elif self.modification_type == "rescale":
            # Rescale rewards to create hacking signals
            return base_reward * 0.5 + 1.0
        elif self.modification_type == "noise":
            # Add noise to rewards
            return base_reward + np.random.normal(0, 0.1)
        else:
            return base_reward

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

def create_ppo_config(run_id: str, seed: int = 42) -> PPOConfig:
    """Create PPO configuration for variant runs."""
    return PPOConfig(
        model_name="gpt2",
        learning_rate=1.41e-5,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=4,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.2,
        log_with="tensorboard",
        tracker_project_name=f"rldk_ppo_{run_id}",
        tracker_kwargs={"logging_dir": f"./rldk_demos/ppo_{run_id}/logs"},
        seed=seed,
        optimize_cuda_cache=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        max_steps=250,
        save_steps=50,
        eval_steps=25,
        logging_steps=10,
        warmup_steps=20,
        report_to=None,
        fp16=False,
        bf16=False,
    )

def run_ppo_variant(run_id: str, variant_config: Dict[str, Any]):
    """Run a PPO variant with specific modifications."""
    print(f"\n{'='*60}")
    print(f"Running PPO variant: {run_id}")
    print(f"Modifications: {variant_config['description']}")
    print(f"{'='*60}")
    
    # Set random seed for this variant
    seed = variant_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Using random seed: {seed}")
    
    # Load data
    ppo_prompts = load_ppo_prompts("./rldk_demos/ppo_prompts.jsonl")
    probe_prompts = load_probe_prompts("./rldk_demos/probes.jsonl")
    
    # Apply data modifications if specified
    if variant_config.get('shuffle_data', False):
        random.shuffle(ppo_prompts)
        print("Applied data shuffling")
    
    print(f"PPO prompts: {len(ppo_prompts)}")
    print(f"Probe prompts: {len(probe_prompts)}")
    
    # Initialize tokenizer and model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply tokenizer modifications
    if variant_config.get('modify_tokenizer', False):
        if tokenizer.padding_side == 'right':
            tokenizer.padding_side = 'left'
        else:
            tokenizer.padding_side = 'right'
        print(f"Modified tokenizer padding side to: {tokenizer.padding_side}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Initialize reward model with modifications
    reward_model_path = "./rldk_demos/rm_a"
    if variant_config.get('modify_reward_model', False):
        reward_model = ModifiedRewardModelWrapper(
            reward_model_path, 
            variant_config.get('reward_modification_type', 'none')
        )
        print(f"Applied reward model modification: {variant_config.get('reward_modification_type', 'none')}")
    else:
        reward_model = RewardModelWrapper(reward_model_path)
    
    # Create PPO config
    ppo_config = create_ppo_config(run_id, seed)
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=ppo_prompts,
    )
    
    # Training loop
    start_time = time.time()
    metrics_history = []
    probe_outputs = []
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
    print(f"PPO variant {run_id} completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
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
    
    # Create trace file
    trace_data = {
        "training_config": ppo_config.to_dict(),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "total_steps": len(metrics_history),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "random_seed": seed,
        "model_name": model_name,
        "variant_config": variant_config,
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
        "run_id": run_id,
        "model_name": model_name,
        "training_samples": len(ppo_prompts),
        "probe_samples": len(probe_prompts),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "total_steps": len(metrics_history),
        "random_seed": seed,
        "ppo_config": ppo_config.to_dict(),
        "variant_config": variant_config,
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "tokenizer_config": trace_data["tokenizer_config"],
        "data_hash": hashlib.sha256(json.dumps(ppo_prompts, sort_keys=True).encode()).hexdigest()
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"PPO variant {run_id} artifacts saved to: {output_dir}")
    if metrics_history:
        final_metrics = metrics_history[-1]
        print(f"Final KL divergence: {final_metrics['kl']:.4f}")
        print(f"Final reward mean: {final_metrics['reward_mean']:.4f}")
        print(f"Final loss: {final_metrics['loss']:.4f}")

def main():
    """Run all PPO variants with intentional issues."""
    print("Starting PPO variants with intentional issues...")
    
    # Define variant configurations
    variants = {
        "b": {
            "description": "Tokenizer padding side change (nondeterminism)",
            "seed": 42,
            "modify_tokenizer": True,
            "modify_reward_model": False,
            "shuffle_data": False
        },
        "c": {
            "description": "Different seed and shuffle order (reward drift)",
            "seed": 123,  # Different seed
            "modify_tokenizer": False,
            "modify_reward_model": False,
            "shuffle_data": True  # Shuffle data order
        },
        "d": {
            "description": "Reward clamping/rescaling (saturation/hacking)",
            "seed": 42,
            "modify_tokenizer": False,
            "modify_reward_model": True,
            "reward_modification_type": "clamp",  # or "rescale"
            "shuffle_data": False
        }
    }
    
    # Run each variant
    for run_id, config in variants.items():
        try:
            run_ppo_variant(run_id, config)
        except Exception as e:
            print(f"Error running variant {run_id}: {e}")
            continue
    
    print("\n" + "="*60)
    print("All PPO variants completed!")
    print("="*60)
    print("Variants created:")
    print("- ppo_b: Tokenizer padding side change (nondeterminism)")
    print("- ppo_c: Different seed and shuffle order (reward drift)")
    print("- ppo_d: Reward clamping/rescaling (saturation/hacking)")
    print("\nThese variants are designed to trigger RLDK detection mechanisms.")

if __name__ == "__main__":
    main()