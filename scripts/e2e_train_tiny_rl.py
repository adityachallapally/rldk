#!/usr/bin/env python3
"""
End-to-end acceptance test training script for RLDK.

Trains a tiny causal language model with REINFORCE on a toy dataset,
logging metrics with RLDK EventWriter for comprehensive testing.
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rldk.emit import EventWriter


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_toy_dataset(num_prompts: int = 20) -> List[str]:
    """Create a tiny synthetic dataset of prompts."""
    prompts = []
    for i in range(num_prompts):
        prompts.append(f"Write a short sentence about a banana number {i}")
    return prompts


def generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Generate text continuation and return tokens, logits, and log probs."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generate continuation
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Get generated tokens (excluding input)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = prompt + " " + generated_text
    
    # Get logits for generated tokens
    logits = torch.stack(outputs.scores, dim=1)[0]  # Shape: [max_new_tokens, vocab_size]
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    sampled_log_probs = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
    
    return full_text, generated_ids, sampled_log_probs


def compute_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between policy and reference logits."""
    # Ensure logits and tokens have matching dimensions
    min_len = min(policy_logits.shape[0], reference_logits.shape[0], len(tokens))
    policy_logits = policy_logits[:min_len]
    reference_logits = reference_logits[:min_len]
    tokens = tokens[:min_len]
    
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    
    # Get log probs for the actual tokens
    policy_token_log_probs = policy_log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
    reference_token_log_probs = reference_log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
    
    # KL = policy_log_prob - reference_log_prob
    kl_div = policy_token_log_probs - reference_token_log_probs
    return kl_div


def compute_reward(text: str, kl_div: torch.Tensor) -> Tuple[float, float]:
    """Compute shaped reward: presence reward minus KL penalty."""
    # Presence reward: 1 point if "banana" appears (case insensitive)
    presence_reward = 1.0 if "banana" in text.lower() else 0.0
    
    # Additional reward for longer responses (encourages generation)
    length_reward = min(len(text.split()) / 20.0, 0.5)  # Up to 0.5 points for length
    
    # KL penalty: mean KL divergence
    kl_penalty = kl_div.mean().item()
    
    # Total reward (deterministic)
    reward_total = presence_reward + length_reward - 0.01 * kl_penalty
    
    return reward_total, kl_penalty


def update_policy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
    clip_grad_norm: float = 1.0,
) -> float:
    """Update policy using REINFORCE with baseline."""
    # Compute advantages
    advantages = rewards - baseline
    
    # Policy gradient loss
    loss = -(log_probs * advantages).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    
    # Optimizer step
    optimizer.step()
    
    return grad_norm.item()


def train_step(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    baseline: float,
    run_id: str,
    step: int,
    writer: EventWriter,
) -> Tuple[float, float, float, float]:
    """Execute one training step."""
    # Sample batch
    batch_prompts = random.sample(prompts, min(batch_size, len(prompts)))
    
    all_rewards = []
    all_kls = []
    all_log_probs = []
    
    for prompt in batch_prompts:
        # Generate with policy (no gradients during generation)
        with torch.no_grad():
            full_text, generated_ids, _ = generate(
                model, tokenizer, prompt, max_new_tokens
            )
        
        # Now compute log probabilities with gradients enabled
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Get policy logits for the generated tokens
        policy_outputs = model(**inputs)
        policy_logits = policy_outputs.logits[0, -len(generated_ids):]
        
        # Ensure dimensions match
        min_len = min(policy_logits.shape[0], len(generated_ids))
        policy_logits = policy_logits[:min_len]
        generated_ids = generated_ids[:min_len]
        
        # Compute log probabilities for the generated tokens
        log_probs = F.log_softmax(policy_logits, dim=-1)
        sampled_log_probs = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
        
        # Get reference logits for KL computation
        with torch.no_grad():
            ref_outputs = reference_model(**inputs)
            ref_logits = ref_outputs.logits[0, -min_len:]
        
        # Compute KL divergence
        kl_div = compute_kl(policy_logits, ref_logits, generated_ids)
        
        # Compute reward
        reward, kl_penalty = compute_reward(full_text, kl_div)
        
        all_rewards.append(reward)
        all_kls.append(kl_penalty)
        all_log_probs.append(sampled_log_probs.sum())  # Keep gradients
    
    # Convert to tensors
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    log_probs_tensor = torch.stack(all_log_probs)
    
    # Update policy
    grad_norm = update_policy(model, optimizer, log_probs_tensor, rewards_tensor, baseline)
    
    # Compute metrics
    reward_mean = rewards_tensor.mean().item()
    reward_std = rewards_tensor.std().item()
    kl_mean = np.mean(all_kls)
    loss = -(log_probs_tensor * (rewards_tensor - baseline)).mean().item()
    
    # Log metrics
    writer.log(
        step=step,
        name="reward_mean",
        value=reward_mean,
        run_id=run_id,
        tags={"model": "tiny-gpt2", "seed": "1337"},
    )
    writer.log(
        step=step,
        name="reward_std",
        value=reward_std,
        run_id=run_id,
        tags={"model": "tiny-gpt2", "seed": "1337"},
    )
    writer.log(
        step=step,
        name="kl_mean",
        value=kl_mean,
        run_id=run_id,
        tags={"model": "tiny-gpt2", "seed": "1337"},
    )
    writer.log(
        step=step,
        name="loss",
        value=loss,
        run_id=run_id,
        tags={"model": "tiny-gpt2", "seed": "1337"},
    )
    writer.log(
        step=step,
        name="lr",
        value=optimizer.param_groups[0]["lr"],
        run_id=run_id,
        tags={"model": "tiny-gpt2", "seed": "1337"},
    )
    writer.log(
        step=step,
        name="grad_norm",
        value=grad_norm,
        run_id=run_id,
        tags={"model": "tiny-gpt2", "seed": "1337"},
    )
    
    return reward_mean, reward_std, kl_mean, loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="E2E RL training script")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--steps", type=int, default=120, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max new tokens")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--outdir", type=str, default="artifacts/e2e", help="Output directory")
    parser.add_argument(
        "--determinism-probe", 
        action="store_true", 
        help="Run determinism probe (generates det_run_a.jsonl and det_run_b.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Load model and tokenizer
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create policy model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.train()
    
    # Create frozen reference model
    reference_model = AutoModelForCausalLM.from_pretrained(model_name)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create dataset
    prompts = create_toy_dataset(20)
    
    # Initialize baseline
    baseline = 0.0
    baseline_window = []
    
    if args.determinism_probe:
        # Run determinism probe
        for probe_run in ["a", "b"]:
            set_seeds(args.seed)  # Reset seeds for each probe run
            
            # Reset model and optimizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
            output_file = outdir / f"det_run_{probe_run}.jsonl"
            with EventWriter(output_file) as writer:
                for step in range(10):
                    reward_mean, reward_std, kl_mean, loss = train_step(
                        model, reference_model, tokenizer, optimizer,
                        prompts, args.batch_size, args.max_new_tokens,
                        baseline, run_id, step, writer
                    )
                    
                    # Update baseline
                    baseline_window.append(reward_mean)
                    if len(baseline_window) > 10:
                        baseline_window.pop(0)
                    baseline = np.mean(baseline_window)
    else:
        # Run baseline (warmup without updates)
        baseline_file = outdir / "baseline.jsonl"
        with EventWriter(baseline_file) as writer:
            for step in range(5):
                # Sample batch but don't update
                batch_prompts = random.sample(prompts, min(args.batch_size, len(prompts)))
                
                all_rewards = []
                all_kls = []
                
                for prompt in batch_prompts:
                    full_text, generated_ids, log_probs = generate(
                        model, tokenizer, prompt, args.max_new_tokens
                    )
                    
                    with torch.no_grad():
                        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                        ref_outputs = reference_model(**inputs)
                        ref_logits = ref_outputs.logits[0, -len(generated_ids):]
                        
                        policy_outputs = model(**inputs)
                        policy_logits = policy_outputs.logits[0, -len(generated_ids):]
                    
                    kl_div = compute_kl(policy_logits, ref_logits, generated_ids)
                    reward, kl_penalty = compute_reward(full_text, kl_div)
                    
                    all_rewards.append(reward)
                    all_kls.append(kl_penalty)
                
                reward_mean = np.mean(all_rewards)
                reward_std = np.std(all_rewards)
                kl_mean = np.mean(all_kls)
                
                writer.log(step=step, name="reward_mean", value=reward_mean, run_id=run_id)
                writer.log(step=step, name="reward_std", value=reward_std, run_id=run_id)
                writer.log(step=step, name="kl_mean", value=kl_mean, run_id=run_id)
        
        # Main training run
        main_file = outdir / "run.jsonl"
        with EventWriter(main_file) as writer:
            for step in range(args.steps):
                reward_mean, reward_std, kl_mean, loss = train_step(
                    model, reference_model, tokenizer, optimizer,
                    prompts, args.batch_size, args.max_new_tokens,
                    baseline, run_id, step, writer
                )
                
                # Update baseline
                baseline_window.append(reward_mean)
                if len(baseline_window) > 10:
                    baseline_window.pop(0)
                baseline = np.mean(baseline_window)


if __name__ == "__main__":
    main()