#!/usr/bin/env python3
"""
Distributed Training Guide for RLDK

This example demonstrates how to use RLDK for distributed training scenarios,
including multi-GPU training, parameter server architectures, and federated learning.
We'll show how to track distributed experiments and handle synchronization issues.

Learning Objectives:
- How to structure distributed training experiments with RLDK
- How to handle multi-GPU and multi-node training scenarios
- How to track distributed metrics and synchronization
- How to detect and debug distributed training issues
- How to implement federated learning with RLDK
- How to handle parameter server architectures

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of distributed training
- Familiarity with PyTorch distributed training
- Understanding of parameter servers and federated learning
"""

import json

# Set up logging
import logging
import multiprocessing as mp
import os
import pickle
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

# RLDK imports
import rldk
from rldk.diff import first_divergence
from rldk.forensics import ComprehensivePPOForensics
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed, validate_numeric_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 4
    rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 12355
    backend: str = "nccl"  # or "gloo" for CPU
    use_cuda: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    sync_every: int = 10  # Sync parameters every N steps
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = False

class SimpleModel(nn.Module):
    """Simple neural network model for distributed training demo."""

    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class SyntheticDataset(Dataset):
    """Synthetic dataset for distributed training demo."""

    def __init__(self, size=1000, input_dim=10, noise_level=0.1):
        self.size = size
        self.input_dim = input_dim
        self.noise_level = noise_level

        # Generate synthetic data
        self.X = torch.randn(size, input_dim)
        self.y = torch.sum(self.X ** 2, dim=1, keepdim=True) + noise_level * torch.randn(size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ParameterServer:
    """Simple parameter server for distributed training."""

    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.parameters = {name: param.clone() for name, param in model.named_parameters()}
        self.lock = threading.Lock()
        self.client_updates = {}
        self.step_count = 0

    def get_parameters(self):
        """Get current model parameters."""
        with self.lock:
            return {name: param.clone() for name, param in self.parameters.items()}

    def update_parameters(self, client_id, gradients, step_count):
        """Update parameters with client gradients."""
        with self.lock:
            self.client_updates[client_id] = gradients
            self.step_count = step_count

            # Aggregate gradients from all clients
            if len(self.client_updates) >= 2:  # Wait for at least 2 clients
                self._aggregate_and_update()
                self.client_updates.clear()

    def _aggregate_and_update(self):
        """Aggregate gradients and update parameters."""
        # Simple averaging
        avg_gradients = {}
        for client_id, gradients in self.client_updates.items():
            for name, grad in gradients.items():
                if name not in avg_gradients:
                    avg_gradients[name] = grad.clone()
                else:
                    avg_gradients[name] += grad

        # Average the gradients
        for name in avg_gradients:
            avg_gradients[name] /= len(self.client_updates)

        # Update parameters
        for name, param in self.parameters.items():
            if name in avg_gradients:
                param.data -= self.optimizer.param_groups[0]['lr'] * avg_gradients[name]

class DistributedTrainer:
    """Distributed trainer using parameter server architecture."""

    def __init__(self, config: DistributedConfig, model: nn.Module, dataset: Dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.rank = config.rank
        self.world_size = config.world_size

        # Create data loader
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0  # Simplified for demo
        )

        # Initialize optimizer
        self.optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

        # Training metrics
        self.metrics = {
            'loss': [],
            'gradient_norm': [],
            'parameter_norm': [],
            'sync_time': [],
            'communication_overhead': []
        }

    def train_epoch(self, epoch: int, parameter_server: Optional[ParameterServer] = None):
        """Train for one epoch."""

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            time.time()

            # Forward pass
            output = self.model(data)
            loss = nn.MSELoss()(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradients
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()

            # Update metrics
            grad_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradients.values()]))
            param_norm = torch.norm(torch.stack([torch.norm(param) for param in self.model.parameters()]))

            self.metrics['gradient_norm'].append(grad_norm.item())
            self.metrics['parameter_norm'].append(param_norm.item())

            # Parameter server update
            if parameter_server is not None:
                sync_start = time.time()
                parameter_server.update_parameters(
                    client_id=f"client_{self.rank}",
                    gradients=gradients,
                    step_count=epoch * len(self.data_loader) + batch_idx
                )
                sync_time = time.time() - sync_start
                self.metrics['sync_time'].append(sync_time)

                # Get updated parameters
                updated_params = parameter_server.get_parameters()
                for name, param in self.model.named_parameters():
                    if name in updated_params:
                        param.data = updated_params[name].data
            else:
                # Standard SGD update
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Rank {self.rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        self.metrics['loss'].append(avg_loss)

        return avg_loss

class FederatedLearningTrainer:
    """Federated learning trainer with RLDK integration."""

    def __init__(self, config: DistributedConfig, model: nn.Module, datasets: List[Dataset]):
        self.config = config
        self.model = model
        self.datasets = datasets
        self.num_clients = len(datasets)

        # Initialize forensics for each client
        self.client_forensics = [
            ComprehensivePPOForensics(kl_target=0.1)
            for _ in range(self.num_clients)
        ]

        # Global model state
        self.global_model_state = None
        self.round_count = 0

        # Training metrics
        self.metrics = {
            'round_loss': [],
            'client_losses': [[] for _ in range(self.num_clients)],
            'communication_rounds': [],
            'model_divergence': [],
            'client_weights': []
        }

    def federated_round(self, num_local_epochs: int = 3):
        """Perform one round of federated learning."""

        logger.info(f"Starting federated round {self.round_count}")

        # Train on each client
        client_models = []
        client_losses = []
        client_weights = []

        for client_id in range(self.num_clients):
            logger.info(f"Training client {client_id}")

            # Create local model copy
            local_model = SimpleModel()
            local_model.load_state_dict(self.model.state_dict())

            # Train locally
            local_trainer = DistributedTrainer(
                config=self.config,
                model=local_model,
                dataset=self.datasets[client_id]
            )

            # Local training
            for epoch in range(num_local_epochs):
                loss = local_trainer.train_epoch(epoch)
                client_losses.append(loss)

                # Update forensics
                self.client_forensics[client_id].update(
                    step=epoch,
                    kl=0.1,  # Placeholder
                    kl_coef=0.2,
                    entropy=0.5,  # Placeholder
                    reward_mean=loss,
                    reward_std=0.1,
                    policy_grad_norm=local_trainer.metrics['gradient_norm'][-1] if local_trainer.metrics['gradient_norm'] else 0.1,
                    value_grad_norm=0.1,  # Placeholder
                    advantage_mean=0.0,
                    advantage_std=0.1
                )

            client_models.append(local_model.state_dict())
            client_weights.append(len(self.datasets[client_id]))

            logger.info(f"Client {client_id} completed with loss: {loss:.4f}")

        # Aggregate models (Federated Averaging)
        aggregated_state = self._federated_averaging(client_models, client_weights)

        # Update global model
        self.model.load_state_dict(aggregated_state)

        # Compute metrics
        avg_loss = np.mean(client_losses)
        self.metrics['round_loss'].append(avg_loss)
        self.metrics['client_losses'] = client_losses
        self.metrics['communication_rounds'].append(self.round_count)
        self.metrics['client_weights'].append(client_weights)

        # Compute model divergence
        if self.global_model_state is not None:
            divergence = self._compute_model_divergence(aggregated_state, self.global_model_state)
            self.metrics['model_divergence'].append(divergence)

        self.global_model_state = aggregated_state.copy()
        self.round_count += 1

        logger.info(f"Federated round {self.round_count-1} completed with avg loss: {avg_loss:.4f}")

        return avg_loss

    def _federated_averaging(self, client_models: List[Dict], client_weights: List[int]) -> Dict:
        """Perform federated averaging of client models."""

        total_weight = sum(client_weights)
        aggregated_state = {}

        # Initialize aggregated state
        for key in client_models[0].keys():
            aggregated_state[key] = torch.zeros_like(client_models[0][key])

        # Weighted averaging
        for client_model, weight in zip(client_models, client_weights):
            weight_ratio = weight / total_weight
            for key in aggregated_state.keys():
                aggregated_state[key] += weight_ratio * client_model[key]

        return aggregated_state

    def _compute_model_divergence(self, state1: Dict, state2: Dict) -> float:
        """Compute divergence between two model states."""

        total_divergence = 0
        total_params = 0

        for key in state1.keys():
            if key in state2:
                diff = torch.norm(state1[key] - state2[key])
                total_divergence += diff.item()
                total_params += 1

        return total_divergence / total_params if total_params > 0 else 0

class DistributedTrainingTracker:
    """Specialized tracker for distributed training experiments."""

    def __init__(self, config: TrackingConfig, distributed_config: DistributedConfig):
        self.tracker = ExperimentTracker(config)
        self.distributed_config = distributed_config
        self.distributed_metrics = {
            'sync_times': [],
            'communication_overhead': [],
            'gradient_norms': [],
            'parameter_norms': [],
            'model_divergences': [],
            'client_losses': []
        }

    def start_experiment(self):
        """Start distributed training experiment."""

        tracking_data = self.tracker.start_experiment()

        # Add distributed-specific metadata
        self.tracker.add_metadata("world_size", self.distributed_config.world_size)
        self.tracker.add_metadata("backend", self.distributed_config.backend)
        self.tracker.add_metadata("batch_size", self.distributed_config.batch_size)
        self.tracker.add_metadata("learning_rate", self.distributed_config.learning_rate)
        self.tracker.add_metadata("sync_every", self.distributed_config.sync_every)
        self.tracker.add_metadata("mixed_precision", self.distributed_config.mixed_precision)

        return tracking_data

    def track_distributed_metrics(self, metrics: Dict[str, Any], step: int):
        """Track distributed training metrics."""

        # Store metrics
        for key, value in metrics.items():
            if key in self.distributed_metrics:
                self.distributed_metrics[key].append(value)

        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'step': [step],
            'sync_time': metrics.get('sync_time', 0),
            'communication_overhead': metrics.get('communication_overhead', 0),
            'gradient_norm': metrics.get('gradient_norm', 0),
            'parameter_norm': metrics.get('parameter_norm', 0),
            'model_divergence': metrics.get('model_divergence', 0)
        })

        # Track as dataset
        self.tracker.track_dataset(
            metrics_df,
            f"distributed_metrics_step_{step}",
            {"step": step, "world_size": self.distributed_config.world_size}
        )

    def track_federated_round(self, round_num: int, round_loss: float, client_losses: List[float],
                            model_divergence: float, client_weights: List[int]):
        """Track federated learning round."""

        # Create round summary
        round_summary = {
            'round': round_num,
            'round_loss': round_loss,
            'client_losses': client_losses,
            'model_divergence': model_divergence,
            'client_weights': client_weights,
            'num_clients': len(client_losses),
            'timestamp': time.time()
        }

        # Track as dataset
        self.tracker.track_dataset(
            pd.DataFrame([round_summary]),
            f"federated_round_{round_num}",
            {"round": round_num, "num_clients": len(client_losses)}
        )

    def finish_experiment(self):
        """Finish distributed training experiment."""

        # Track final distributed metrics
        final_metrics_df = pd.DataFrame(self.distributed_metrics)
        self.tracker.track_dataset(
            final_metrics_df,
            "final_distributed_metrics",
            {"total_steps": len(final_metrics_df)}
        )

        return self.tracker.finish_experiment()

def run_parameter_server_training():
    """Run distributed training with parameter server architecture."""

    print("🚀 Starting Parameter Server Distributed Training")
    print("=" * 60)

    # Setup
    set_global_seed(42)

    # Configuration
    config = DistributedConfig(
        world_size=4,
        rank=0,
        use_cuda=False,  # Simplified for demo
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=5
    )

    # Create model and dataset
    model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=1)
    dataset = SyntheticDataset(size=1000, input_dim=10)

    # Initialize parameter server
    parameter_server = ParameterServer(model, learning_rate=config.learning_rate)

    # Initialize trainer
    trainer = DistributedTrainer(config, model, dataset)

    # Initialize tracker
    tracking_config = TrackingConfig(
        experiment_name="parameter_server_training",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        output_dir="./runs",
        tags=["demo", "distributed", "parameter-server"],
        notes="Parameter server distributed training example"
    )

    tracker = DistributedTrainingTracker(tracking_config, config)
    tracking_data = tracker.start_experiment()

    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # Training loop
    print("\n📊 Training with parameter server...")

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train epoch
        loss = trainer.train_epoch(epoch, parameter_server)

        # Track metrics
        metrics = {
            'sync_time': np.mean(trainer.metrics['sync_time']) if trainer.metrics['sync_time'] else 0,
            'communication_overhead': len(trainer.metrics['sync_time']) * 0.01,  # Simulated
            'gradient_norm': np.mean(trainer.metrics['gradient_norm']) if trainer.metrics['gradient_norm'] else 0,
            'parameter_norm': np.mean(trainer.metrics['parameter_norm']) if trainer.metrics['parameter_norm'] else 0,
            'model_divergence': 0.1  # Simulated
        }

        tracker.track_distributed_metrics(metrics, epoch)

        print(f"   Loss: {loss:.4f}")
        print(f"   Sync time: {metrics['sync_time']:.4f}s")
        print(f"   Gradient norm: {metrics['gradient_norm']:.4f}")

    # Track final model
    tracker.tracker.track_model(
        model,
        "parameter_server_model",
        {
            "algorithm": "DistributedSGD",
            "architecture": "ParameterServer",
            "world_size": config.world_size,
            "final_loss": loss
        }
    )

    # Finish experiment
    summary = tracker.finish_experiment()

    print("\n✅ Parameter server training completed!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Final loss: {loss:.4f}")

    return trainer, parameter_server, tracker

def run_federated_learning():
    """Run federated learning with RLDK tracking."""

    print("\n🌐 Starting Federated Learning")
    print("=" * 60)

    # Setup
    set_global_seed(42)

    # Configuration
    config = DistributedConfig(
        world_size=4,
        rank=0,
        use_cuda=False,  # Simplified for demo
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=5
    )

    # Create model and datasets for different clients
    model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=1)

    # Create heterogeneous datasets for different clients
    datasets = []
    for i in range(4):
        # Each client has different data distribution
        size = 500 + i * 100  # Different dataset sizes
        noise_level = 0.1 + i * 0.05  # Different noise levels
        dataset = SyntheticDataset(size=size, input_dim=10, noise_level=noise_level)
        datasets.append(dataset)

    # Initialize federated trainer
    federated_trainer = FederatedLearningTrainer(config, model, datasets)

    # Initialize tracker
    tracking_config = TrackingConfig(
        experiment_name="federated_learning",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        output_dir="./runs",
        tags=["demo", "distributed", "federated-learning"],
        notes="Federated learning example with heterogeneous clients"
    )

    tracker = DistributedTrainingTracker(tracking_config, config)
    tracking_data = tracker.start_experiment()

    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # Federated learning rounds
    print("\n📊 Running federated learning rounds...")

    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")

        # Run federated round
        round_loss = federated_trainer.federated_round(num_local_epochs=2)

        # Track round
        tracker.track_federated_round(
            round_num=round_num,
            round_loss=round_loss,
            client_losses=federated_trainer.metrics['client_losses'],
            model_divergence=federated_trainer.metrics['model_divergence'][-1] if federated_trainer.metrics['model_divergence'] else 0,
            client_weights=federated_trainer.metrics['client_weights'][-1] if federated_trainer.metrics['client_weights'] else [1] * 4
        )

        print(f"   Round loss: {round_loss:.4f}")
        print(f"   Client losses: {[f'{l:.4f}' for l in federated_trainer.metrics['client_losses']]}")
        if federated_trainer.metrics['model_divergence']:
            print(f"   Model divergence: {federated_trainer.metrics['model_divergence'][-1]:.4f}")

    # Track final model
    tracker.tracker.track_model(
        federated_trainer.model,
        "federated_model",
        {
            "algorithm": "FederatedAveraging",
            "architecture": "FederatedLearning",
            "num_clients": len(datasets),
            "num_rounds": num_rounds,
            "final_round_loss": round_loss
        }
    )

    # Finish experiment
    summary = tracker.finish_experiment()

    print("\n✅ Federated learning completed!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Final round loss: {round_loss:.4f}")
    print(f"   Total rounds: {num_rounds}")

    return federated_trainer, tracker

def create_distributed_training_visualizations(ps_trainer, ps_tracker, fl_trainer, fl_tracker):
    """Create visualizations for distributed training results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Parameter Server Training Loss
    axes[0, 0].plot(ps_trainer.metrics['loss'], 'b-', label='Parameter Server')
    axes[0, 0].set_title('Parameter Server Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Federated Learning Round Loss
    axes[0, 1].plot(fl_trainer.metrics['round_loss'], 'r-', label='Federated Learning')
    axes[0, 1].set_title('Federated Learning Round Loss')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Gradient Norms Comparison
    if ps_trainer.metrics['gradient_norm']:
        axes[0, 2].plot(ps_trainer.metrics['gradient_norm'], 'b-', label='Parameter Server')
    if fl_trainer.metrics['gradient_norm']:
        axes[0, 2].plot(fl_trainer.metrics['gradient_norm'], 'r-', label='Federated Learning')
    axes[0, 2].set_title('Gradient Norms')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Communication Overhead
    if ps_trainer.metrics['sync_time']:
        axes[1, 0].plot(ps_trainer.metrics['sync_time'], 'b-', label='Parameter Server')
    axes[1, 0].set_title('Communication Overhead')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Sync Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Model Divergence (Federated Learning)
    if fl_trainer.metrics['model_divergence']:
        axes[1, 1].plot(fl_trainer.metrics['model_divergence'], 'r-', label='Federated Learning')
    axes[1, 1].set_title('Model Divergence')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Divergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Client Loss Distribution (Federated Learning)
    if fl_trainer.metrics['client_losses']:
        client_losses = fl_trainer.metrics['client_losses']
        if isinstance(client_losses[0], list):
            # Multiple rounds
            for i, losses in enumerate(client_losses[:3]):  # Show first 3 rounds
                axes[1, 2].plot(losses, label=f'Round {i+1}')
        else:
            # Single round
            axes[1, 2].plot(client_losses, label='Client Losses')
    axes[1, 2].set_title('Client Loss Distribution')
    axes[1, 2].set_xlabel('Client')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./distributed_training_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Distributed training plots saved to ./distributed_training_plots.png")

def main():
    """Main function demonstrating distributed training with RLDK."""

    print("🚀 RLDK Distributed Training Guide")
    print("=" * 60)

    # 1. Setup
    print("\n1. Setting up distributed training environment...")
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")

    # 2. Parameter Server Training
    print("\n2. Running Parameter Server Distributed Training...")

    ps_trainer, ps_parameter_server, ps_tracker = run_parameter_server_training()

    # 3. Federated Learning
    print("\n3. Running Federated Learning...")

    fl_trainer, fl_tracker = run_federated_learning()

    # 4. Create Visualizations
    print("\n4. Creating distributed training visualizations...")

    create_distributed_training_visualizations(ps_trainer, ps_tracker, fl_trainer, fl_tracker)

    # 5. Analysis and Comparison
    print("\n5. Analyzing distributed training results...")

    print("\n📊 Parameter Server Results:")
    print(f"   Final loss: {ps_trainer.metrics['loss'][-1]:.4f}")
    print(f"   Average sync time: {np.mean(ps_trainer.metrics['sync_time']):.4f}s")
    print(f"   Average gradient norm: {np.mean(ps_trainer.metrics['gradient_norm']):.4f}")

    print("\n📊 Federated Learning Results:")
    print(f"   Final round loss: {fl_trainer.metrics['round_loss'][-1]:.4f}")
    print(f"   Total rounds: {len(fl_trainer.metrics['round_loss'])}")
    print(f"   Average model divergence: {np.mean(fl_trainer.metrics['model_divergence']):.4f}")

    # 6. Save Results
    print("\n6. Saving distributed training results...")

    results = {
        "parameter_server": {
            "metrics": ps_trainer.metrics,
            "final_loss": ps_trainer.metrics['loss'][-1],
            "average_sync_time": np.mean(ps_trainer.metrics['sync_time']),
            "average_gradient_norm": np.mean(ps_trainer.metrics['gradient_norm'])
        },
        "federated_learning": {
            "metrics": fl_trainer.metrics,
            "final_round_loss": fl_trainer.metrics['round_loss'][-1],
            "total_rounds": len(fl_trainer.metrics['round_loss']),
            "average_model_divergence": np.mean(fl_trainer.metrics['model_divergence'])
        }
    }

    with open("./distributed_training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("💾 Results saved to ./distributed_training_results.json")

    # 7. Summary
    print("\n7. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Parameter Server: Implemented distributed training with parameter server")
    print("   2. Federated Learning: Implemented federated averaging across clients")
    print("   3. RLDK Integration: Tracked distributed metrics and synchronization")
    print("   4. Heterogeneous Data: Handled different client data distributions")
    print("   5. Communication Analysis: Analyzed sync times and overhead")
    print("   6. Model Divergence: Tracked model convergence across clients")

    print("\n📊 Key Findings:")
    print(f"   - Parameter server final loss: {ps_trainer.metrics['loss'][-1]:.4f}")
    print(f"   - Federated learning final loss: {fl_trainer.metrics['round_loss'][-1]:.4f}")
    print(f"   - Average sync time: {np.mean(ps_trainer.metrics['sync_time']):.4f}s")
    print(f"   - Average model divergence: {np.mean(fl_trainer.metrics['model_divergence']):.4f}")

    print("\n🚀 Next Steps:")
    print("   1. Multi-GPU Training: Implement true multi-GPU distributed training")
    print("   2. Gradient Compression: Reduce communication overhead")
    print("   3. Asynchronous Updates: Implement asynchronous parameter updates")
    print("   4. Differential Privacy: Add privacy-preserving techniques")
    print("   5. Model Compression: Implement model compression for efficiency")
    print("   6. Fault Tolerance: Add fault tolerance and recovery mechanisms")

    print("\n📚 Key Takeaways:")
    print("   - RLDK makes distributed training tracking systematic and reproducible")
    print("   - Parameter servers provide centralized coordination")
    print("   - Federated learning enables privacy-preserving training")
    print("   - Communication overhead is a key bottleneck")
    print("   - Model divergence indicates training stability")
    print("   - Heterogeneous data requires careful aggregation")

    print("\nHappy distributed training! 🎉")

if __name__ == "__main__":
    main()
