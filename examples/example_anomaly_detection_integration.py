#!/usr/bin/env python3
"""
Example integration of Advanced Anomaly Detection System with RLHF training.

This example shows how to integrate the anomaly detection system into a real
training loop to monitor for various types of anomalies during training.
"""

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from profiler.anomaly_detection import AdvancedAnomalyDetector
from profiler.hooks import AnomalyDetectionHook, profiler_registry


class RewardModel(nn.Module):
    """Simple reward model for demonstration."""

    def __init__(self, input_size: int = 512, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layers(x)


class PolicyModel(nn.Module):
    """Simple policy model for demonstration."""

    def __init__(self, input_size: int = 512, vocab_size: int = 10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, nhead=8, batch_first=True),
            num_layers=6
        )
        self.output = nn.Linear(input_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)


class RLHFTrainer:
    """RLHF trainer with integrated anomaly detection."""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)

        # Initialize models
        self.reward_model = RewardModel().to(self.device)
        self.policy_model = PolicyModel().to(self.device)

        # Initialize optimizers
        self.reward_optimizer = optim.AdamW(self.reward_model.parameters(), lr=1e-4)
        self.policy_optimizer = optim.AdamW(self.policy_model.parameters(), lr=1e-4)

        # Initialize anomaly detection system with improved thresholds
        self.anomaly_detector = AdvancedAnomalyDetector(
            output_dir="training_anomaly_detection",
            save_alerts=True,
            gradient={
                'explosion_threshold': 50.0,  # More lenient to reduce false positives
                'vanishing_threshold': 1e-8,  # More lenient to reduce false positives
                'alert_threshold': 2.0,       # More specific variance detection
                'window_size': 100
            },
            learning_rate={
                'change_threshold': 0.8,      # Allow normal scheduler behavior
                'min_lr': 1e-10,             # More lenient minimum
                'max_lr': 10.0,              # More lenient maximum
                'consecutive_threshold': 3    # Require consecutive changes
            },
            batch_size={
                'performance_threshold': 0.1,
                'window_size': 20
            },
            convergence={
                'plateau_threshold': 0.001,
                'plateau_window': 50
            },
            reward_drift={
                'drift_threshold': 0.3,       # More lenient to reduce false positives
                'calibration_threshold': 0.5, # More lenient calibration threshold
                'min_samples': 20             # Require more samples for reliable detection
            }
        )

        # Initialize anomaly detection hook
        self.anomaly_hook = AnomalyDetectionHook(self.anomaly_detector)
        self.anomaly_hook.register_with_profiler()

        # Training statistics
        self.training_stats = {
            'reward_losses': [],
            'policy_losses': [],
            'rewards': [],
            'predictions': [],
            'alerts': []
        }

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _generate_batch(self, batch_size: int, seq_len: int = 128) -> torch.Tensor:
        """Generate a random batch of token sequences."""
        vocab_size = self.policy_model.embedding.num_embeddings
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

    def _generate_rewards(self, batch_size: int) -> np.ndarray:
        """Generate synthetic rewards for demonstration."""
        # Generate rewards with some realistic distribution
        rewards = np.random.beta(2, 5, batch_size)  # Skewed towards lower rewards
        return rewards

    def train_reward_model(self, batch_size: int = 16) -> float:
        """Train the reward model for one step."""
        # Generate synthetic data
        sequences = self._generate_batch(batch_size)
        rewards = self._generate_rewards(batch_size)

        # Forward pass - use embedding and mean
        embeddings = self.policy_model.embedding(sequences)  # [batch_size, seq_len, embed_dim]
        mean_embeddings = embeddings.mean(dim=1)  # [batch_size, embed_dim]
        reward_predictions = self.reward_model(mean_embeddings)
        reward_targets = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Calculate loss
        reward_loss = nn.MSELoss()(reward_predictions.squeeze(), reward_targets)

        # Backward pass
        self.reward_optimizer.zero_grad()
        reward_loss.backward()

        # Analyze for anomalies
        alerts = self.anomaly_detector.analyze_training_step(
            model=self.reward_model,
            optimizer=self.reward_optimizer,
            loss=reward_loss.item(),
            batch_size=batch_size,
            rewards=rewards,
            predictions=reward_predictions.squeeze().detach().cpu().numpy()
        )

        # Store statistics
        self.training_stats['reward_losses'].append(reward_loss.item())
        self.training_stats['rewards'].extend(rewards.tolist())
        self.training_stats['predictions'].extend(reward_predictions.squeeze().detach().cpu().numpy().tolist())
        self.training_stats['alerts'].extend(alerts)

        # Optimizer step
        self.reward_optimizer.step()

        return reward_loss.item()

    def train_policy_model(self, batch_size: int = 16) -> float:
        """Train the policy model for one step."""
        # Generate synthetic data
        sequences = self._generate_batch(batch_size)
        targets = torch.randint(0, self.policy_model.embedding.num_embeddings,
                              (batch_size, sequences.size(1)), device=self.device)

        # Forward pass
        policy_outputs = self.policy_model(sequences)
        policy_loss = nn.CrossEntropyLoss()(policy_outputs.view(-1, policy_outputs.size(-1)),
                                          targets.view(-1))

        # Backward pass
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # Analyze for anomalies
        alerts = self.anomaly_detector.analyze_training_step(
            model=self.policy_model,
            optimizer=self.policy_optimizer,
            loss=policy_loss.item(),
            batch_size=batch_size
        )

        # Store statistics
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['alerts'].extend(alerts)

        # Optimizer step
        self.policy_optimizer.step()

        return policy_loss.item()

    def train_step(self, step: int) -> Dict[str, float]:
        """Perform one training step for both models."""
        # Train reward model
        reward_loss = self.train_reward_model()

        # Train policy model
        policy_loss = self.train_policy_model()

        # Print progress
        if step % 10 == 0:
            print(f"Step {step:4d}: Reward Loss = {reward_loss:.4f}, Policy Loss = {policy_loss:.4f}")

            # Check for recent alerts
            recent_alerts = [alert for alert in self.training_stats['alerts']
                           if alert.step >= step - 10]
            if recent_alerts:
                critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
                high_alerts = [a for a in recent_alerts if a.severity == 'high']

                if critical_alerts:
                    print(f"  ⚠️  {len(critical_alerts)} CRITICAL alerts detected!")
                if high_alerts:
                    print(f"  ⚠️  {len(high_alerts)} HIGH severity alerts detected!")

        return {
            'reward_loss': reward_loss,
            'policy_loss': policy_loss,
            'step': step
        }

    def train(self, num_steps: int = 100):
        """Run training with anomaly detection."""
        print(f"Starting RLHF training with anomaly detection for {num_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Reward model parameters: {sum(p.numel() for p in self.reward_model.parameters()):,}")
        print(f"Policy model parameters: {sum(p.numel() for p in self.policy_model.parameters()):,}")
        print()

        start_time = time.time()

        for step in range(num_steps):
            # Perform training step
            self.train_step(step)

            # Simulate some anomalies for demonstration
            if step == 30:
                # Simulate gradient explosion
                print("  🔥 Simulating gradient explosion...")
                for param in self.reward_model.parameters():
                    if param.grad is not None:
                        param.grad.data *= 100.0

            elif step == 50:
                # Simulate learning rate anomaly
                print("  📈 Simulating learning rate anomaly...")
                for param_group in self.reward_optimizer.param_groups:
                    param_group['lr'] = 10.0

            elif step == 70:
                # Simulate batch size change
                print("  📊 Simulating batch size change...")
                # This will be detected in the next training step

        end_time = time.time()

        # Generate final report
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total training time: {end_time - start_time:.2f} seconds")

        # Get anomaly detection summary
        summary = self.anomaly_detector.get_summary()
        print(f"Total alerts detected: {summary['total_alerts']}")
        print(f"Alerts by category: {summary['by_category']}")
        print(f"Alerts by severity: {summary['by_severity']}")

        # Save final report
        report_file = self.anomaly_detector.save_final_report()
        print(f"Anomaly detection report saved to: {report_file}")

        # Print some example alerts
        if summary['latest_alerts']:
            print("\nLatest alerts:")
            for alert in summary['latest_alerts'][-5:]:  # Show last 5 alerts
                if hasattr(alert, 'severity'):
                    print(f"  [{alert.severity.upper()}] {alert.category}: {alert.message}")
                else:
                    print(f"  [{alert['severity'].upper()}] {alert['category']}: {alert['message']}")

        return summary


def main():
    """Main function to run the RLHF training example."""
    print("RLHF Training with Advanced Anomaly Detection")
    print("=" * 50)

    # Create trainer
    trainer = RLHFTrainer(device="auto")

    # Run training
    trainer.train(num_steps=100)

    print("\n🎉 Training completed successfully!")
    print("Check the 'training_anomaly_detection' directory for detailed reports.")


if __name__ == "__main__":
    main()
