#!/usr/bin/env python3
"""
Fullscale RL Training Script for RLDK Acceptance Testing.

This script trains a GPT-2 medium/large model with an on-policy RL loop for several hours
on CPU, using realistic synthetic datasets and multi-component rewards. It logs to JSONL
using RLDK EventWriter for comprehensive monitoring and analysis.

Usage:
    python scripts/fullscale_train_rl.py --seed 42 --max-steps 2000 --outdir artifacts/fullscale
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent))

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        get_linear_schedule_with_warmup,
    )
except ImportError as e:
    print(f"❌ Error: Missing transformers dependency: {e}")
    print("💡 Install with: pip install transformers")
    sys.exit(1)

from rldk.emit import EventWriter
from rldk.utils.seed import set_global_seed


class SyntheticDataset(Dataset):
    """Synthetic dataset with varied domains for realistic RL training."""
    
    def __init__(self, num_samples: int = 5000, max_length: int = 128):
        self.max_length = max_length
        
        math_prompts = [
            "Calculate the derivative of x^2 + 3x:",
            "Solve for x: 2x + 5 = 13:",
            "What is the integral of sin(x)?",
            "Find the area of a circle with radius 5:",
            "Simplify the expression (x+2)(x-3):",
            "What is the square root of 144?",
            "Convert 0.75 to a fraction:",
            "What is 15% of 200?",
            "Solve the quadratic equation x^2 - 4x + 3 = 0:",
            "What is the slope of the line y = 2x + 1?",
        ]
        
        code_prompts = [
            "Write a Python function to reverse a string:",
            "How do you create a list in Python?",
            "Explain what a for loop does:",
            "Write a function to check if a number is prime:",
            "How do you handle exceptions in Python?",
            "What is the difference between a list and a tuple?",
            "Write a function to find the maximum in a list:",
            "How do you read a file in Python?",
            "Explain what recursion is:",
            "Write a function to sort a list of numbers:",
        ]
        
        narrative_prompts = [
            "Once upon a time in a distant kingdom,",
            "The old lighthouse keeper noticed something strange:",
            "In the year 2050, humanity discovered:",
            "The detective examined the crime scene and found:",
            "Deep in the Amazon rainforest, explorers uncovered:",
            "The spaceship landed on the mysterious planet:",
            "The ancient book contained a secret that would:",
            "On a stormy night, the traveler sought shelter:",
            "The scientist's experiment had unexpected results:",
            "The treasure map led to a hidden cave where:",
        ]
        
        science_prompts = [
            "Explain the process of photosynthesis:",
            "What causes the greenhouse effect?",
            "Describe the structure of an atom:",
            "How does DNA replication work?",
            "What is the theory of evolution?",
            "Explain how vaccines work:",
            "What causes earthquakes?",
            "Describe the water cycle:",
            "How do solar panels generate electricity?",
            "What is quantum mechanics?",
        ]
        
        all_prompts = math_prompts + code_prompts + narrative_prompts + science_prompts
        
        self.prompts = []
        self.domains = []
        
        for i in range(num_samples):
            prompt = all_prompts[i % len(all_prompts)]
            self.prompts.append(prompt)
            
            if i % len(all_prompts) < len(math_prompts):
                self.domains.append("math")
            elif i % len(all_prompts) < len(math_prompts) + len(code_prompts):
                self.domains.append("code")
            elif i % len(all_prompts) < len(math_prompts) + len(code_prompts) + len(narrative_prompts):
                self.domains.append("narrative")
            else:
                self.domains.append("science")
        
        combined = list(zip(self.prompts, self.domains))
        random.shuffle(combined)
        self.prompts, self.domains = zip(*combined)
        
        print(f"📊 Created synthetic dataset with {len(self.prompts)} samples")
        print(f"   Domains: {len(set(self.domains))} unique domains")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "domain": self.domains[idx],
        }


class MultiComponentReward:
    """Multi-component reward function with task reward, KL penalty, and auxiliary signals."""
    
    def __init__(self, kl_coef: float = 0.1, complexity_coef: float = 0.05):
        self.kl_coef = kl_coef
        self.complexity_coef = complexity_coef
        
        self.domain_keywords = {
            "math": ["calculate", "solve", "equation", "derivative", "integral", "formula"],
            "code": ["function", "python", "loop", "variable", "algorithm", "programming"],
            "narrative": ["story", "character", "adventure", "mystery", "journey", "tale"],
            "science": ["theory", "experiment", "hypothesis", "research", "discovery", "analysis"],
        }
    
    def compute_reward(
        self,
        response: str,
        domain: str,
        kl_divergence: float,
        response_length: int,
        entropy: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute multi-component reward."""
        
        task_reward = 0.0
        if domain in self.domain_keywords:
            keywords = self.domain_keywords[domain]
            response_lower = response.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in response_lower)
            task_reward = min(keyword_count * 0.2, 1.0)  # Cap at 1.0
        
        kl_penalty = -self.kl_coef * kl_divergence
        
        length_score = 1.0 - abs(response_length - 50) / 100.0  # Optimal around 50 tokens
        length_score = max(0.0, length_score)
        
        entropy_score = min(entropy / 3.0, 1.0)  # Normalize entropy
        
        complexity_reward = self.complexity_coef * (length_score + entropy_score) / 2.0
        
        noise = np.random.normal(0, 0.05)
        
        total_reward = task_reward + kl_penalty + complexity_reward + noise
        
        components = {
            "task_reward": task_reward,
            "kl_penalty": kl_penalty,
            "complexity_reward": complexity_reward,
            "noise": noise,
            "total_reward": total_reward,
        }
        
        return total_reward, components


class RLTrainer:
    """REINFORCE-based RL trainer for language models."""
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: Any,
        reward_fn: MultiComponentReward,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        max_length: int = 128,
        generation_kwargs: Optional[Dict] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.batch_size = batch_size
        self.max_length = max_length
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 50,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "pad_token_id": tokenizer.pad_token_id,
        }
        
        self.baseline = 0.0
        self.baseline_momentum = 0.9
        
        print(f"🎯 RL Trainer initialized")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max length: {max_length}")
    
    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Generate responses and compute log probabilities."""
        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=prompt_tokens["input_ids"],
                attention_mask=prompt_tokens["attention_mask"],
                **self.generation_kwargs,
            )
        
        prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1)
        response_tokens = []
        responses = []
        
        for i, (gen_seq, prompt_len) in enumerate(zip(generated, prompt_lengths)):
            response_seq = gen_seq[prompt_len:]
            response_tokens.append(response_seq)
            response_text = self.tokenizer.decode(response_seq, skip_special_tokens=True)
            responses.append(response_text)
        
        self.model.train()  # Switch to train mode for gradient computation
        policy_logprobs = self._compute_logprobs(generated, prompt_tokens["attention_mask"])
        
        with torch.no_grad():
            ref_logprobs = self._compute_logprobs_ref(generated, prompt_tokens["attention_mask"])
        
        return responses, policy_logprobs, ref_logprobs
    
    def _compute_logprobs(self, sequences: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for generated sequences."""
        outputs = self.model(input_ids=sequences)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sequences[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        response_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for i, prompt_len in enumerate(prompt_mask.sum(dim=1)):
            response_mask[i, prompt_len:] = True
        
        masked_log_probs = selected_log_probs * response_mask.float()
        sequence_log_probs = masked_log_probs.sum(dim=1)
        
        return sequence_log_probs
    
    def _compute_logprobs_ref(self, sequences: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        """Compute reference model log probabilities."""
        outputs = self.ref_model(input_ids=sequences)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sequences[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        response_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for i, prompt_len in enumerate(prompt_mask.sum(dim=1)):
            response_mask[i, prompt_len:] = True
        
        masked_log_probs = selected_log_probs * response_mask.float()
        sequence_log_probs = masked_log_probs.sum(dim=1)
        
        return sequence_log_probs
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one training step."""
        self.model.train()
        
        prompts = batch["prompt"]
        domains = batch["domain"]
        
        responses, policy_logprobs, ref_logprobs = self.generate_responses(prompts)
        
        kl_divergences = policy_logprobs - ref_logprobs
        kl_mean = kl_divergences.mean().item()
        
        rewards = []
        reward_components = []
        
        for response, domain, kl_div in zip(responses, domains, kl_divergences):
            response_tokens = self.tokenizer.encode(response)
            unique_token_ratio = len(set(response_tokens)) / len(response_tokens) if response_tokens else 0.0
            
            reward, components = self.reward_fn.compute_reward(
                response=response,
                domain=domain,
                kl_divergence=kl_div.item(),
                response_length=len(response_tokens),
                entropy=unique_token_ratio,
            )
            
            rewards.append(reward)
            reward_components.append(components)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        reward_mean = rewards.mean().item()
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * reward_mean
        
        advantages = rewards - self.baseline
        
        policy_loss = -(policy_logprobs * advantages).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        metrics = {
            "loss": policy_loss.item(),
            "reward_mean": reward_mean,
            "reward_std": rewards.std().item(),
            "kl_mean": kl_mean,
            "baseline": self.baseline,
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "grad_norm": grad_norm.item(),
            "entropy_mean": np.mean([len(set(self.tokenizer.encode(resp))) / len(self.tokenizer.encode(resp)) if self.tokenizer.encode(resp) else 0.0 for resp in responses]),
        }
        
        for key in ["task_reward", "kl_penalty", "complexity_reward"]:
            metrics[f"{key}_mean"] = np.mean([comp[key] for comp in reward_components])
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Fullscale RL Training for RLDK Acceptance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum training steps")
    parser.add_argument("--max-hours", type=float, default=3.0, help="Maximum training hours")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--outdir", type=str, default="artifacts/fullscale", help="Output directory")
    parser.add_argument("--model-name", type=str, default="gpt2-medium", help="Model name (gpt2-medium or gpt2-large)")
    parser.add_argument("--determinism-check", action="store_true", help="Run determinism check")
    
    args = parser.parse_args()
    
    set_global_seed(args.seed, deterministic=True)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Fullscale RL Training for RLDK Acceptance Testing")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"Max hours: {args.max_hours}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output dir: {args.outdir}")
    print(f"Device: CPU (stress-testing monitoring)")
    print()
    
    print(f"📦 Loading {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        print(f"✅ Models loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return 1
    
    # Create dataset
    print("📊 Creating synthetic dataset...")
    dataset = SyntheticDataset(num_samples=5000, max_length=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    reward_fn = MultiComponentReward(kl_coef=0.1, complexity_coef=0.05)
    
    trainer = RLTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    run_id = f"fullscale_run_{args.seed}_{int(time.time())}"
    log_path = outdir / "run.jsonl"
    
    print(f"📝 Initializing EventWriter: {log_path}")
    writer = EventWriter(str(log_path))
    
    # Training loop
    print("🎯 Starting training...")
    start_time = time.time()
    step = 0
    
    try:
        while step < args.max_steps:
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= args.max_hours:
                print(f"⏰ Time limit reached ({elapsed_hours:.2f} hours)")
                break
            
            for batch in dataloader:
                if step >= args.max_steps:
                    break
                
                metrics = trainer.train_step(batch)
                
                for metric_name, metric_value in metrics.items():
                    writer.log(
                        step=step,
                        name=metric_name,
                        value=float(metric_value),
                        run_id=run_id,
                        tags={"model": args.model_name, "seed": args.seed},
                        meta={"batch_size": args.batch_size, "learning_rate": args.learning_rate},
                    )
                
                if step % 50 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Step {step:4d} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"Reward: {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f} | "
                          f"KL: {metrics['kl_mean']:.4f} | "
                          f"Grad: {metrics['grad_norm']:.4f} | "
                          f"Time: {elapsed_time:.1f}s")
                
                step += 1
        
        print(f"✅ Training completed after {step} steps")
        
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    finally:
        writer.close()
    
    final_metrics = {
        "total_steps": step,
        "total_time_hours": (time.time() - start_time) / 3600,
        "model_name": args.model_name,
        "seed": args.seed,
        "run_id": run_id,
    }
    
    with open(outdir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"💾 Training artifacts saved to: {args.outdir}")
    print(f"   JSONL log: {log_path}")
    print(f"   Final metrics: {outdir / 'final_metrics.json'}")
    
    if args.determinism_check:
        print("🔍 Running determinism check...")
        baseline_log_path = outdir / "baseline.jsonl"
        
        set_global_seed(args.seed, deterministic=True)
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        trainer = RLTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        
        baseline_writer = EventWriter(str(baseline_log_path))
        
        step = 0
        for batch in dataloader:
            if step >= 100:
                break
            
            metrics = trainer.train_step(batch)
            
            for metric_name, metric_value in metrics.items():
                baseline_writer.log(
                    step=step,
                    name=metric_name,
                    value=float(metric_value),
                    run_id=f"{run_id}_baseline",
                    tags={"model": args.model_name, "seed": args.seed},
                    meta={"batch_size": args.batch_size, "learning_rate": args.learning_rate},
                )
            
            step += 1
        
        baseline_writer.close()
        print(f"✅ Baseline run completed: {baseline_log_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
