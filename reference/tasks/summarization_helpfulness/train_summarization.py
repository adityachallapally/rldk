#!/usr/bin/env python3
"""
Summarization Helpfulness Training Script

This script trains a GPT-2 model to generate helpful summaries using RLHF.
It contains intentional bugs that RLDK will detect:

1. KL divergence spike at step 47 (learning rate too high)
2. Non-deterministic training (missing seed in data loader)
3. Reward saturation (aggressive scaling)
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

# Simulate imports that would be available in a real environment
try:
    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using mock implementation")


class MockGPT2Model:
    """Mock GPT-2 model for demonstration when PyTorch unavailable."""

    def __init__(self, model_size: str = "125M"):
        self.model_size = model_size
        self.vocab_size = 50257
        self.hidden_size = 768 if model_size == "125M" else 1024

    def generate(self, input_ids, max_length=50, **kwargs):
        # Mock generation - just return random tokens
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Generate random continuation
        new_tokens = torch.randint(
            0, self.vocab_size, (batch_size, max_length - seq_len)
        )
        return torch.cat([input_ids, new_tokens], dim=1)

    def forward(self, input_ids, labels=None):
        # Mock forward pass
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


class MockTokenizer:
    """Mock tokenizer for demonstration."""

    def __init__(self):
        self.vocab_size = 50257

    def encode(self, text, **kwargs):
        # Mock encoding - just return random token IDs
        words = text.split()
        return [random.randint(0, self.vocab_size - 1) for _ in range(len(words))]

    def decode(self, token_ids, **kwargs):
        # Mock decoding
        return " ".join([f"token_{tid}" for tid in token_ids])


class SummarizationDataset:
    """Dataset for summarization with human preferences."""

    def __init__(self, size: int = 1000, seed: int = None):
        # BUG: Missing seed setting - causes non-deterministic behavior
        # This is intentional for RLDK to catch
        if seed is not None:
            random.seed(seed)  # But we don't always set this!

        self.size = size
        self.data = self._generate_data()

    def _generate_data(self) -> List[Dict]:
        """Generate mock summarization data."""
        articles = [
            "The quick brown fox jumps over the lazy dog. This is a sample article about animals and their behavior.",
            "Machine learning is transforming industries worldwide. Companies are adopting AI at unprecedented rates.",
            "Climate change poses significant challenges. Scientists warn of irreversible damage if action isn't taken soon.",
            "The future of transportation is electric. Electric vehicles are becoming more affordable and practical.",
            "Space exploration continues to advance. Private companies are joining government agencies in space missions.",
        ]

        data = []
        for i in range(self.size):
            article = random.choice(articles)
            summary = f"Summary {i}: {article[:50]}..."
            preference_score = random.uniform(0.1, 0.9)

            data.append(
                {
                    "id": i,
                    "article": article,
                    "summary": summary,
                    "preference_score": preference_score,
                    "tokens": random.randint(50, 200),
                }
            )

        return data

    def __iter__(self):
        # BUG: Non-deterministic iteration order
        # This is intentional for RLDK to catch
        items = list(self.data)
        random.shuffle(items)  # Always shuffle, no seed control
        yield from items


class RewardModel:
    """Mock reward model with intentional scaling issues."""

    def __init__(self, scaling_factor: float = 10.0):
        # BUG: Aggressive scaling causes saturation
        # This is intentional for RLDK to catch
        self.scaling_factor = scaling_factor

    def compute_reward(self, summary: str, preference_score: float) -> float:
        """Compute reward with problematic scaling."""
        # BUG: Aggressive scaling leads to saturation
        base_reward = preference_score
        scaled_reward = base_reward * self.scaling_factor

        # Apply additional non-linear scaling that causes saturation
        saturated_reward = (
            torch.tanh(scaled_reward) if hasattr(torch, "tanh") else scaled_reward
        )

        return float(saturated_reward)


class PPOAgent:
    """PPO agent for RLHF training."""

    def __init__(self, model, tokenizer, reward_model, learning_rate: float = 1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.learning_rate = learning_rate

        # BUG: Learning rate too high - causes KL divergence spike
        # This is intentional for RLDK to catch
        self.optimizer = None  # Would be torch.optim.AdamW in real implementation

    def compute_kl_divergence(self, old_logits, new_logits):
        """Compute KL divergence between old and new policies."""
        # Mock KL computation
        if hasattr(torch, "softmax"):
            old_probs = F.softmax(old_logits, dim=-1)
            new_probs = F.softmax(new_logits, dim=-1)
            kl_div = F.kl_div(new_probs.log(), old_probs, reduction="batchmean")
            return float(kl_div)
        else:
            # Mock KL divergence that spikes at step 47
            return random.uniform(0.01, 0.05)

    def update_policy(self, batch_data):
        """Update policy using PPO."""
        # Mock policy update
        kl_div = self.compute_kl_divergence(
            torch.randn(32, 100, 50257),  # old logits
            torch.randn(32, 100, 50257),  # new logits
        )

        # BUG: KL divergence spike at step 47
        # This is intentional for RLDK to catch
        if hasattr(self, "_step_count"):
            self._step_count += 1
        else:
            self._step_count = 1

        if self._step_count == 47:
            kl_div *= 2.3  # Intentional spike!

        return {
            "kl_divergence": kl_div,
            "policy_loss": random.uniform(0.1, 0.5),
            "value_loss": random.uniform(0.05, 0.2),
            "entropy": random.uniform(0.5, 1.0),
        }


def train_summarization(
    model_size: str = "125M",
    learning_rate: float = 1e-4,
    num_steps: int = 100,
    seed: int = 42,
    output_dir: str = "summarization_outputs",
):
    """Main training function with intentional bugs."""

    print("🚀 Starting summarization helpfulness training")
    print(f"Model: GPT-2 {model_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps: {num_steps}")
    print(f"Seed: {seed}")

    # Set seed for reproducibility (but dataset doesn't use it consistently)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Initialize components
    if TORCH_AVAILABLE:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        model = MockGPT2Model(model_size)
        tokenizer = MockTokenizer()

    reward_model = RewardModel(scaling_factor=10.0)  # BUG: Too aggressive
    agent = PPOAgent(model, tokenizer, reward_model, learning_rate)

    # Create dataset (BUG: Non-deterministic due to missing seed)
    dataset = SummarizationDataset(size=1000, seed=None)  # Intentional bug!

    # Training loop
    training_metrics = []

    for step in range(num_steps):
        start_time = time.time()

        # Get batch (non-deterministic due to dataset bug)
        batch = list(itertools.islice(dataset, 32))

        # Compute rewards (will saturate due to scaling bug)
        rewards = []
        for item in batch:
            reward = reward_model.compute_reward(
                item["summary"], item["preference_score"]
            )
            rewards.append(reward)

        # Update policy (KL divergence spike at step 47)
        update_metrics = agent.update_policy(batch)

        # Record metrics
        step_metrics = {
            "step": step,
            "reward_mean": (
                float(torch.tensor(rewards).mean())
                if TORCH_AVAILABLE
                else sum(rewards) / len(rewards)
            ),
            "reward_std": (
                float(torch.tensor(rewards).std()) if TORCH_AVAILABLE else 0.1
            ),
            "kl_divergence": update_metrics["kl_divergence"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "entropy": update_metrics["entropy"],
            "learning_rate": learning_rate,
            "batch_size": len(batch),
            "wall_time": time.time() - start_time,
            "seed": seed,
            "run_id": f"summarization_{seed}_{model_size}",
        }

        training_metrics.append(step_metrics)

        # Print progress
        if step % 10 == 0:
            print(
                f"Step {step:3d}: Reward={step_metrics['reward_mean']:.3f}, "
                f"KL={step_metrics['kl_divergence']:.4f}, "
                f"Loss={step_metrics['policy_loss']:.3f}"
            )

    # Save training metrics
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    metrics_file = output_path / "training_metrics.jsonl"
    with open(metrics_file, "w") as f:
        for metrics in training_metrics:
            f.write(json.dumps(metrics) + "\n")

    print("\n✅ Training completed!")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Total steps: {len(training_metrics)}")

    # Print summary of bugs that RLDK should catch
    print("\n🐛 Intentional bugs for RLDK to detect:")
    print("  1. KL divergence spike at step 47 (2.3x normal)")
    print("  2. Non-deterministic training (15% variance expected)")
    print("  3. Reward saturation (0.87 saturation score expected)")

    return training_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train summarization helpfulness with RLHF"
    )
    parser.add_argument(
        "--model", default="125M", choices=["125M", "355M", "774M"], help="Model size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", default="summarization_outputs", help="Output directory"
    )

    args = parser.parse_args()

    # Run training
    train_summarization(
        model_size=args.model,
        learning_rate=args.learning_rate,
        num_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\n🎯 Now run RLDK to catch the bugs:")
    print(
        f"  rldk diff --a {args.output_dir} --b {args.output_dir}_run2 --signals kl_divergence,reward_mean"
    )
    print(f"  rldk check-determinism --cmd 'python {__file__}' --compare reward_mean")
    print(f"  rldk reward-health --run {args.output_dir} --output-dir analysis")


if __name__ == "__main__":
    # Add missing import for demonstration
    import itertools

    main()
