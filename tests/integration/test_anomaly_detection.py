#!/usr/bin/env python3
"""
Comprehensive test script for Advanced Anomaly Detection System.

This script tests all anomaly detection features with a large model to ensure
the system works correctly under realistic conditions.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from tools.profiler module
from tools.profiler.anomaly_detection import AdvancedAnomalyDetector
from tools.profiler.hooks import AnomalyDetectionHook, profiler_registry


class LargeTransformerModel(nn.Module):
    """Large transformer-like model for testing anomaly detection."""

    def __init__(self, vocab_size: int = 50000, d_model: int = 1024, n_heads: int = 16,
                 n_layers: int = 12, seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layers
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Token and position embeddings
        token_emb = self.token_embedding(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)

        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x)

        # Transformer layers
        x = self.transformer(x)

        # Output projection
        x = self.output_projection(x)

        return x


class AnomalyDetectionTester:
    """Test suite for anomaly detection system."""

    def __init__(self, model_size: str = "large", device: str = "auto"):
        self.device = self._get_device(device)
        self.model_size = model_size
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

        # Initialize anomaly detector
        self.anomaly_detector = AdvancedAnomalyDetector(
            output_dir="anomaly_test_results",
            save_alerts=False,  # Disable auto-saving during testing
            gradient={
                'explosion_threshold': 10.0,
                'vanishing_threshold': 1e-6,
                'window_size': 50
            },
            learning_rate={
                'change_threshold': 0.3,
                'min_lr': 1e-8,
                'max_lr': 1.0
            },
            batch_size={
                'performance_threshold': 0.1,
                'window_size': 20
            },
            convergence={
                'plateau_threshold': 0.001,
                'plateau_window': 30
            },
            reward_drift={
                'drift_threshold': 0.1,
                'calibration_threshold': 0.7
            }
        )

        # Initialize anomaly detection hook
        self.anomaly_hook = AnomalyDetectionHook(self.anomaly_detector)
        self.anomaly_hook.register_with_profiler()

        # Test results
        self.test_results = {}
        self.all_alerts = []

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

    def _create_model(self) -> nn.Module:
        """Create a model based on size specification."""
        if self.model_size == "small":
            return LargeTransformerModel(
                vocab_size=10000, d_model=256, n_heads=8, n_layers=4, seq_len=128
            ).to(self.device)
        elif self.model_size == "medium":
            return LargeTransformerModel(
                vocab_size=30000, d_model=512, n_heads=8, n_layers=8, seq_len=256
            ).to(self.device)
        elif self.model_size == "large":
            return LargeTransformerModel(
                vocab_size=50000, d_model=1024, n_heads=16, n_layers=12, seq_len=512
            ).to(self.device)
        elif self.model_size == "xlarge":
            return LargeTransformerModel(
                vocab_size=100000, d_model=2048, n_heads=32, n_layers=24, seq_len=1024
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with learning rate scheduler."""
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)

        # Add learning rate scheduler that will trigger anomalies
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        return optimizer

    def _generate_batch(self, batch_size: int, seq_len: int = None) -> torch.Tensor:
        """Generate a random batch of data."""
        if seq_len is None:
            seq_len = self.model.seq_len

        # Generate random token IDs
        vocab_size = self.model.token_embedding.num_embeddings
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

    def _generate_rewards_and_predictions(self, batch_size: int) -> tuple:
        """Generate synthetic rewards and predictions for testing."""
        # Generate random predictions (logits)
        predictions = torch.randn(batch_size, device=self.device)
        predictions = torch.sigmoid(predictions)  # Convert to probabilities

        # Generate rewards with some correlation to predictions
        noise = torch.randn(batch_size, device=self.device) * 0.1
        rewards = predictions + noise
        rewards = torch.clamp(rewards, 0, 1)

        return rewards.cpu().numpy(), predictions.cpu().numpy()

    def test_gradient_anomalies(self, num_steps: int = 100) -> Dict[str, Any]:
        """Test gradient explosion and vanishing detection."""
        print("Testing gradient anomaly detection...")

        # Temporarily modify model to trigger gradient anomalies
        original_weights = {}
        for name, param in self.model.named_parameters():
            original_weights[name] = param.data.clone()

        alerts = []

        for step in range(num_steps):
            # Generate batch
            batch = self._generate_batch(batch_size=8)
            targets = torch.randint(0, self.model.token_embedding.num_embeddings,
                                  (batch.size(0), batch.size(1)), device=self.device)

            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Trigger gradient explosion every 20 steps
            if step % 20 == 19:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad.data *= 100.0  # Explode gradients

            # Trigger gradient vanishing every 30 steps
            elif step % 30 == 29:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad.data *= 1e-8  # Vanish gradients

            # Analyze with anomaly detector
            step_alerts = self.anomaly_detector.analyze_training_step(
                model=self.model,
                optimizer=self.optimizer,
                loss=loss.item(),
                batch_size=batch.size(0)
            )
            alerts.extend(step_alerts)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

        # Restore original weights
        for name, param in self.model.named_parameters():
            param.data = original_weights[name]

        gradient_alerts = [a for a in alerts if a.category == 'gradient']

        return {
            "total_alerts": len(gradient_alerts),
            "explosion_alerts": len([a for a in gradient_alerts if "explosion" in a.message]),
            "vanishing_alerts": len([a for a in gradient_alerts if "vanishing" in a.message]),
            "variance_alerts": len([a for a in gradient_alerts if "variance" in a.message]),
            "alerts": gradient_alerts
        }

    def test_learning_rate_anomalies(self, num_steps: int = 100) -> Dict[str, Any]:
        """Test learning rate anomaly detection."""
        print("Testing learning rate anomaly detection...")

        alerts = []

        for step in range(num_steps):
            # Generate batch
            batch = self._generate_batch(batch_size=8)
            targets = torch.randint(0, self.model.token_embedding.num_embeddings,
                                  (batch.size(0), batch.size(1)), device=self.device)

            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Trigger learning rate anomalies
            if step % 25 == 24:
                # Set learning rate too high
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 10.0
            elif step % 35 == 34:
                # Set learning rate too low
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-10

            # Analyze with anomaly detector
            step_alerts = self.anomaly_detector.analyze_training_step(
                model=self.model,
                optimizer=self.optimizer,
                loss=loss.item(),
                batch_size=batch.size(0)
            )
            alerts.extend(step_alerts)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

        lr_alerts = [a for a in alerts if a.category == 'learning_rate']

        return {
            "total_alerts": len(lr_alerts),
            "high_lr_alerts": len([a for a in lr_alerts if "too high" in a.message]),
            "low_lr_alerts": len([a for a in lr_alerts if "too low" in a.message]),
            "change_alerts": len([a for a in lr_alerts if "change" in a.message]),
            "alerts": lr_alerts
        }

    def test_batch_size_impact(self, num_steps: int = 100) -> Dict[str, Any]:
        """Test batch size impact analysis."""
        print("Testing batch size impact analysis...")

        alerts = []
        batch_sizes = [4, 8, 16, 32, 8, 4, 16, 8]  # Varying batch sizes

        for step in range(num_steps):
            batch_size = batch_sizes[step % len(batch_sizes)]

            # Generate batch
            batch = self._generate_batch(batch_size=batch_size)
            targets = torch.randint(0, self.model.token_embedding.num_embeddings,
                                  (batch.size(0), batch.size(1)), device=self.device)

            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Analyze with anomaly detector
            step_alerts = self.anomaly_detector.analyze_training_step(
                model=self.model,
                optimizer=self.optimizer,
                loss=loss.item(),
                batch_size=batch_size
            )
            alerts.extend(step_alerts)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

        batch_alerts = [a for a in alerts if a.category == 'batch_size']

        return {
            "total_alerts": len(batch_alerts),
            "impact_alerts": len([a for a in batch_alerts if "impact" in a.message]),
            "alerts": batch_alerts
        }

    def test_convergence_tracking(self, num_steps: int = 200) -> Dict[str, Any]:
        """Test convergence tracking."""
        print("Testing convergence tracking...")

        alerts = []
        base_loss = 5.0

        for step in range(num_steps):
            # Generate batch
            batch = self._generate_batch(batch_size=8)
            targets = torch.randint(0, self.model.token_embedding.num_embeddings,
                                  (batch.size(0), batch.size(1)), device=self.device)

            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Simulate different convergence patterns
            if step < 50:
                # Normal decreasing loss
                simulated_loss = base_loss - step * 0.05 + np.random.normal(0, 0.1)
            elif step < 100:
                # Plateau
                simulated_loss = base_loss - 2.5 + np.random.normal(0, 0.01)
            elif step < 150:
                # Increasing loss (divergence)
                simulated_loss = base_loss - 2.5 + (step - 100) * 0.02 + np.random.normal(0, 0.1)
            else:
                # Recovery
                simulated_loss = base_loss - 2.5 + 1.0 - (step - 150) * 0.01 + np.random.normal(0, 0.05)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Analyze with anomaly detector
            step_alerts = self.anomaly_detector.analyze_training_step(
                model=self.model,
                optimizer=self.optimizer,
                loss=simulated_loss,
                batch_size=batch.size(0)
            )
            alerts.extend(step_alerts)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

        convergence_alerts = [a for a in alerts if a.category == 'convergence']

        return {
            "total_alerts": len(convergence_alerts),
            "plateau_alerts": len([a for a in convergence_alerts if "plateau" in a.message]),
            "divergence_alerts": len([a for a in convergence_alerts if "increasing" in a.message]),
            "alerts": convergence_alerts
        }

    def test_reward_calibration_drift(self, num_steps: int = 100) -> Dict[str, Any]:
        """Test reward calibration drift detection."""
        print("Testing reward calibration drift detection...")

        alerts = []

        for step in range(num_steps):
            # Generate batch
            batch = self._generate_batch(batch_size=16)
            targets = torch.randint(0, self.model.token_embedding.num_embeddings,
                                  (batch.size(0), batch.size(1)), device=self.device)

            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Generate rewards and predictions
            rewards, predictions = self._generate_rewards_and_predictions(batch.size(0))

            # Simulate calibration drift
            if step > 50:
                # Add systematic bias to predictions
                predictions = predictions + 0.2 * np.sin(step * 0.1)
                predictions = np.clip(predictions, 0, 1)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Analyze with anomaly detector
            step_alerts = self.anomaly_detector.analyze_training_step(
                model=self.model,
                optimizer=self.optimizer,
                loss=loss.item(),
                batch_size=batch.size(0),
                rewards=rewards,
                predictions=predictions
            )
            alerts.extend(step_alerts)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

        reward_alerts = [a for a in alerts if a.category == 'reward_drift']

        return {
            "total_alerts": len(reward_alerts),
            "calibration_alerts": len([a for a in reward_alerts if "calibration" in a.message]),
            "drift_alerts": len([a for a in reward_alerts if "drift" in a.message]),
            "alerts": reward_alerts
        }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all anomaly detection tests."""
        print(f"Starting comprehensive anomaly detection test with {self.model_size} model...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        # Run all tests
        self.test_results = {
            "gradient_anomalies": self.test_gradient_anomalies(),
            "learning_rate_anomalies": self.test_learning_rate_anomalies(),
            "batch_size_impact": self.test_batch_size_impact(),
            "convergence_tracking": self.test_convergence_tracking(),
            "reward_calibration_drift": self.test_reward_calibration_drift()
        }

        # Get overall summary
        summary = self.anomaly_detector.get_summary()
        self.test_results["overall_summary"] = summary

        # Save final report
        self.anomaly_detector.save_final_report()

        end_time = time.time()
        self.test_results["execution_time"] = end_time - start_time

        return self.test_results

    def print_results(self):
        """Print test results in a formatted way."""
        print("\n" + "="*80)
        print("ANOMALY DETECTION TEST RESULTS")
        print("="*80)

        for test_name, results in self.test_results.items():
            if test_name == "execution_time":
                print(f"\nTotal execution time: {results:.2f} seconds")
                continue
            elif test_name == "overall_summary":
                print("\nOverall Summary:")
                print(f"  Total alerts: {results['total_alerts']}")
                print(f"  By category: {results['by_category']}")
                print(f"  By severity: {results['by_severity']}")
                continue

            print(f"\n{test_name.replace('_', ' ').title()}:")
            for key, value in results.items():
                if key != "alerts":
                    print(f"  {key}: {value}")

        print("\n" + "="*80)


def main():
    """Main function to run anomaly detection tests."""
    parser = argparse.ArgumentParser(description="Test Advanced Anomaly Detection System")
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xlarge"],
                       default="large", help="Model size for testing")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--output", default="anomaly_test_results", help="Output directory")

    args = parser.parse_args()

    try:
        # Create tester
        tester = AnomalyDetectionTester(model_size=args.model_size, device=args.device)

        # Run comprehensive test
        results = tester.run_comprehensive_test()

        # Print results
        tester.print_results()

        # Save detailed results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {results_file}")

        # Check if any critical alerts were detected
        critical_alerts = 0
        for test_results in results.values():
            if isinstance(test_results, dict) and "alerts" in test_results:
                critical_alerts += len([a for a in test_results["alerts"] if a.severity == "critical"])

        if critical_alerts > 0:
            print(f"\n⚠️  {critical_alerts} critical alerts detected!")
        else:
            print("\n✅ No critical alerts detected.")

        print("\n🎉 Anomaly detection system test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
