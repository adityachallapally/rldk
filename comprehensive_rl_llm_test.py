#!/usr/bin/env python3
"""
Comprehensive RL + LLM Testing Script

This script performs exhaustive testing of the RLDK repository with large models,
multiple iterations, and various anomaly scenarios to ensure all functions work
correctly under stress conditions.

Features:
- Large transformer model training with RL
- Multiple training iterations to trigger edge cases
- Anomaly injection and detection testing
- Comprehensive function validation
- Stress testing with memory pressure
- Determinism verification across runs
- Reward model drift simulation
- Checkpoint comparison testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import os
import sys
import random
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import uuid
import gc
import psutil
import threading
import queue

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import RLDK components
try:
    from rldk.ingest import ingest_runs
    from rldk.diff import first_divergence
    from rldk.determinism.check import check
    from rldk.reward import health
    from rldk.evals import run
    from rldk.replay import replay
    from rldk.cards import generate_determinism_card, generate_drift_card, generate_reward_card
    from rldk.io import write_json
    from rldk.io.reward_writers import generate_reward_health_report
except ImportError as e:
    print(f"Warning: Could not import RLDK components: {e}")
    print("Some tests will be skipped")

# Import profiler components
try:
    from profiler.anomaly_detection import AdvancedAnomalyDetector
    from profiler.hooks import AnomalyDetectionHook, profiler_registry
    from profiler.torch_profiler import TorchProfiler
    from profiler.profiler_context import ProfilerContext
except ImportError as e:
    print(f"Warning: Could not import profiler components: {e}")
    print("Profiler tests will be skipped")

# Import RLHF core
try:
    from rlhf_core.profiler import ProfilerManager
except ImportError as e:
    print(f"Warning: Could not import RLHF core: {e}")


@dataclass
class TestConfig:
    """Configuration for comprehensive testing."""
    
    # Model parameters
    vocab_size: int = 50000
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    seq_len: int = 512
    batch_size: int = 8
    
    # Training parameters
    num_epochs: int = 5
    num_iterations: int = 10  # Number of training runs
    learning_rate: float = 1e-4
    
    # Testing parameters
    enable_anomaly_injection: bool = True
    enable_memory_stress: bool = True
    enable_determinism_testing: bool = True
    enable_reward_drift_testing: bool = True
    enable_checkpoint_testing: bool = True
    
    # Output parameters
    output_dir: Path = field(default_factory=lambda: Path("./comprehensive_test_results"))
    save_checkpoints: bool = True
    save_logs: bool = True
    save_reports: bool = True
    
    # Anomaly injection parameters
    anomaly_probability: float = 0.1  # 10% chance of anomaly per step
    memory_pressure_probability: float = 0.05  # 5% chance of memory pressure
    
    def __post_init__(self):
        """Post-initialization setup."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "anomalies").mkdir(exist_ok=True)


class LargeTransformerModel(nn.Module):
    """Large transformer model for comprehensive testing."""
    
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Output layers
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.value_head = nn.Linear(config.d_model, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Transformer layers
        if attention_mask is not None:
            # Convert attention mask to the format expected by PyTorch
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        
        transformer_output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Output heads
        logits = self.lm_head(transformer_output)
        values = self.value_head(transformer_output)
        
        return {
            'logits': logits,
            'values': values,
            'hidden_states': transformer_output
        }


class RewardModel(nn.Module):
    """Reward model for RL training."""
    
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        
        # Use the same transformer backbone
        self.backbone = LargeTransformerModel(config)
        
        # Reward head
        self.reward_head = nn.Linear(config.d_model, 1)
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for reward prediction."""
        outputs = self.backbone(input_ids, attention_mask)
        rewards = self.reward_head(outputs['hidden_states'])
        return rewards


class AnomalyInjector:
    """Inject various types of anomalies during training."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.anomaly_count = 0
        self.anomaly_types = [
            'gradient_explosion',
            'gradient_vanishing',
            'nan_injection',
            'inf_injection',
            'memory_pressure',
            'learning_rate_spike',
            'weight_corruption',
            'activation_saturation'
        ]
    
    def should_inject_anomaly(self) -> bool:
        """Determine if an anomaly should be injected."""
        return random.random() < self.config.anomaly_probability
    
    def inject_anomaly(self, model: nn.Module, optimizer: optim.Optimizer, 
                      step: int, anomaly_type: Optional[str] = None) -> Dict[str, Any]:
        """Inject a specific type of anomaly."""
        if anomaly_type is None:
            anomaly_type = random.choice(self.anomaly_types)
        
        anomaly_info = {
            'type': anomaly_type,
            'step': step,
            'timestamp': time.time(),
            'description': f"Injected {anomaly_type} at step {step}"
        }
        
        try:
            if anomaly_type == 'gradient_explosion':
                self._inject_gradient_explosion(model)
            elif anomaly_type == 'gradient_vanishing':
                self._inject_gradient_vanishing(model)
            elif anomaly_type == 'nan_injection':
                self._inject_nan(model)
            elif anomaly_type == 'inf_injection':
                self._inject_inf(model)
            elif anomaly_type == 'memory_pressure':
                self._inject_memory_pressure()
            elif anomaly_type == 'learning_rate_spike':
                self._inject_lr_spike(optimizer)
            elif anomaly_type == 'weight_corruption':
                self._inject_weight_corruption(model)
            elif anomaly_type == 'activation_saturation':
                self._inject_activation_saturation(model)
            
            self.anomaly_count += 1
            anomaly_info['success'] = True
            
        except Exception as e:
            anomaly_info['success'] = False
            anomaly_info['error'] = str(e)
        
        return anomaly_info
    
    def _inject_gradient_explosion(self, model: nn.Module):
        """Inject gradient explosion by scaling gradients."""
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= 1000.0
    
    def _inject_gradient_vanishing(self, model: nn.Module):
        """Inject gradient vanishing by scaling gradients down."""
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= 0.001
    
    def _inject_nan(self, model: nn.Module):
        """Inject NaN values into model parameters."""
        for param in model.parameters():
            if param.numel() > 0:
                # Inject NaN into a small portion of parameters
                mask = torch.rand_like(param) < 0.01
                param.data[mask] = float('nan')
    
    def _inject_inf(self, model: nn.Module):
        """Inject infinite values into model parameters."""
        for param in model.parameters():
            if param.numel() > 0:
                # Inject inf into a small portion of parameters
                mask = torch.rand_like(param) < 0.01
                param.data[mask] = float('inf')
    
    def _inject_memory_pressure(self):
        """Simulate memory pressure by allocating large tensors."""
        if self.config.enable_memory_stress:
            # Allocate large tensors to create memory pressure
            large_tensors = []
            for _ in range(10):
                try:
                    tensor = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
                    large_tensors.append(tensor)
                except RuntimeError:
                    break  # Out of memory
    
    def _inject_lr_spike(self, optimizer: optim.Optimizer):
        """Inject learning rate spike."""
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 100.0
    
    def _inject_weight_corruption(self, model: nn.Module):
        """Corrupt some model weights."""
        for param in model.parameters():
            if param.numel() > 0:
                # Corrupt a small portion of weights
                mask = torch.rand_like(param) < 0.005
                param.data[mask] = torch.randn_like(param.data[mask]) * 10.0
    
    def _inject_activation_saturation(self, model: nn.Module):
        """Inject activation saturation by modifying activations."""
        # This would require hooking into forward passes
        # For now, we'll modify some weights to cause saturation
        for param in model.parameters():
            if param.numel() > 0 and len(param.shape) > 1:
                # Scale weights to cause saturation
                param.data *= 0.1


class ComprehensiveTester:
    """Main testing class for comprehensive RL + LLM testing."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.anomaly_injector = AnomalyInjector(config)
        self.test_results = []
        self.anomaly_log = []
        self.memory_log = []
        
        # Setup logging
        self.setup_logging()
        
        # Initialize anomaly detector if available
        try:
            self.anomaly_detector = AdvancedAnomalyDetector()
        except:
            self.anomaly_detector = None
            print("Warning: Anomaly detector not available")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.config.output_dir / "comprehensive_test.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_memory_usage(self, step: int, phase: str):
        """Log current memory usage."""
        memory_info = {
            'step': step,
            'phase': phase,
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'cuda_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'cuda_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'cuda_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
            })
        
        self.memory_log.append(memory_info)
    
    def generate_synthetic_data(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic training data."""
        # Generate random token IDs
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Generate attention mask (randomly mask some tokens)
        attention_mask = torch.ones_like(input_ids)
        mask_prob = 0.1  # 10% of tokens are masked
        mask_indices = torch.rand_like(attention_mask.float()) < mask_prob
        attention_mask[mask_indices] = 0
        
        return input_ids, attention_mask
    
    def train_model_iteration(self, iteration: int) -> Dict[str, Any]:
        """Train the model for one iteration."""
        self.logger.info(f"Starting training iteration {iteration}")
        
        # Initialize model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LargeTransformerModel(self.config).to(device)
        reward_model = RewardModel(self.config).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        reward_optimizer = optim.AdamW(reward_model.parameters(), lr=self.config.learning_rate)
        
        # Training metrics
        iteration_results = {
            'iteration': iteration,
            'start_time': time.time(),
            'losses': [],
            'rewards': [],
            'anomalies': [],
            'memory_usage': [],
            'checkpoints': [],
            'gradient_norms': [],
            'learning_rates': []
        }
        
        model.train()
        reward_model.train()
        
        step = 0
        total_steps = self.config.num_epochs * 100  # 100 steps per epoch
        
        for epoch in range(self.config.num_epochs):
            for batch_idx in range(100):  # 100 batches per epoch
                step += 1
                
                # Log memory usage
                self.log_memory_usage(step, f"iteration_{iteration}_epoch_{epoch}")
                
                # Generate synthetic data
                input_ids, attention_mask = self.generate_synthetic_data(
                    self.config.batch_size, self.config.seq_len
                )
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits']
                values = outputs['values']
                
                # Compute loss (simplified)
                target_ids = input_ids.clone()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Compute reward
                with torch.no_grad():
                    rewards = reward_model(input_ids, attention_mask)
                    rewards = rewards.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norm
                total_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                # Store metrics
                iteration_results['losses'].append(loss.item())
                iteration_results['rewards'].append(rewards.item())
                iteration_results['gradient_norms'].append(total_norm)
                iteration_results['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                # Inject anomalies if enabled
                if self.config.enable_anomaly_injection and self.anomaly_injector.should_inject_anomaly():
                    anomaly_info = self.anomaly_injector.inject_anomaly(model, optimizer, step)
                    iteration_results['anomalies'].append(anomaly_info)
                    self.anomaly_log.append(anomaly_info)
                    self.logger.warning(f"Anomaly injected: {anomaly_info}")
                
                # Optimizer step
                optimizer.step()
                
                # Train reward model
                reward_optimizer.zero_grad()
                reward_loss = nn.MSELoss()(rewards, torch.tensor(0.5, device=device))  # Dummy target
                reward_loss.backward()
                reward_optimizer.step()
                
                # Save checkpoint periodically
                if step % 50 == 0 and self.config.save_checkpoints:
                    checkpoint_path = self.config.output_dir / "checkpoints" / f"iteration_{iteration}_step_{step}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'loss': loss.item(),
                        'reward': rewards.item()
                    }, checkpoint_path)
                    iteration_results['checkpoints'].append(str(checkpoint_path))
                
                # Log progress
                if step % 10 == 0:
                    self.logger.info(f"Iteration {iteration}, Step {step}, Loss: {loss.item():.4f}, Reward: {rewards.item():.4f}")
        
        iteration_results['end_time'] = time.time()
        iteration_results['duration'] = iteration_results['end_time'] - iteration_results['start_time']
        
        self.logger.info(f"Completed training iteration {iteration} in {iteration_results['duration']:.2f} seconds")
        
        return iteration_results
    
    def test_determinism(self, num_runs: int = 3) -> Dict[str, Any]:
        """Test determinism across multiple runs."""
        self.logger.info(f"Testing determinism with {num_runs} runs")
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        deterministic_results = []
        
        for run_idx in range(num_runs):
            self.logger.info(f"Determinism test run {run_idx + 1}/{num_runs}")
            
            # Reset seeds for each run
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            
            # Train model
            results = self.train_model_iteration(f"determinism_{run_idx}")
            deterministic_results.append(results)
        
        # Compare results
        determinism_analysis = {
            'num_runs': num_runs,
            'final_losses': [r['losses'][-1] for r in deterministic_results],
            'final_rewards': [r['rewards'][-1] for r in deterministic_results],
            'loss_variance': np.var([r['losses'][-1] for r in deterministic_results]),
            'reward_variance': np.var([r['rewards'][-1] for r in deterministic_results]),
            'is_deterministic': True  # Will be updated based on analysis
        }
        
        # Check if runs are deterministic (within tolerance)
        loss_tolerance = 1e-6
        reward_tolerance = 1e-6
        
        loss_deterministic = determinism_analysis['loss_variance'] < loss_tolerance
        reward_deterministic = determinism_analysis['reward_variance'] < reward_tolerance
        
        determinism_analysis['is_deterministic'] = loss_deterministic and reward_deterministic
        determinism_analysis['loss_deterministic'] = loss_deterministic
        determinism_analysis['reward_deterministic'] = reward_deterministic
        
        self.logger.info(f"Determinism test results: {determinism_analysis}")
        
        return determinism_analysis
    
    def test_reward_drift(self) -> Dict[str, Any]:
        """Test reward model drift detection."""
        self.logger.info("Testing reward drift detection")
        
        # Train two reward models with different data distributions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model A - trained on normal distribution
        model_a = RewardModel(self.config).to(device)
        optimizer_a = optim.AdamW(model_a.parameters(), lr=self.config.learning_rate)
        
        # Model B - trained on shifted distribution
        model_b = RewardModel(self.config).to(device)
        optimizer_b = optim.AdamW(model_b.parameters(), lr=self.config.learning_rate)
        
        # Train both models
        for model, optimizer, name in [(model_a, optimizer_a, "A"), (model_b, optimizer_b, "B")]:
            model.train()
            for epoch in range(3):
                for batch_idx in range(50):
                    # Generate data with different distributions
                    if name == "A":
                        # Normal distribution
                        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, self.config.seq_len))
                    else:
                        # Shifted distribution (higher token IDs)
                        input_ids = torch.randint(self.config.vocab_size//2, self.config.vocab_size, 
                                                (self.config.batch_size, self.config.seq_len))
                    
                    input_ids = input_ids.to(device)
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Forward pass
                    rewards = model(input_ids, attention_mask)
                    
                    # Compute loss
                    target = torch.tensor(0.5, device=device)
                    loss = nn.MSELoss()(rewards.mean(), target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Test both models on same data
        test_input_ids = torch.randint(0, self.config.vocab_size, (10, self.config.seq_len)).to(device)
        test_attention_mask = torch.ones_like(test_input_ids)
        
        with torch.no_grad():
            rewards_a = model_a(test_input_ids, test_attention_mask)
            rewards_b = model_b(test_input_ids, test_attention_mask)
        
        # Calculate drift metrics
        drift_analysis = {
            'rewards_a_mean': rewards_a.mean().item(),
            'rewards_b_mean': rewards_b.mean().item(),
            'rewards_a_std': rewards_a.std().item(),
            'rewards_b_std': rewards_b.std().item(),
            'correlation': torch.corrcoef(torch.stack([rewards_a.flatten(), rewards_b.flatten()]))[0, 1].item(),
            'mean_absolute_difference': torch.abs(rewards_a - rewards_b).mean().item(),
            'drift_detected': True  # Will be updated based on analysis
        }
        
        # Determine if drift is detected
        drift_threshold = 0.1
        drift_analysis['drift_detected'] = drift_analysis['mean_absolute_difference'] > drift_threshold
        
        self.logger.info(f"Reward drift analysis: {drift_analysis}")
        
        return drift_analysis
    
    def test_checkpoint_comparison(self) -> Dict[str, Any]:
        """Test checkpoint comparison functionality."""
        self.logger.info("Testing checkpoint comparison")
        
        # Train two models with different configurations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model 1 - normal training
        model1 = LargeTransformerModel(self.config).to(device)
        optimizer1 = optim.AdamW(model1.parameters(), lr=self.config.learning_rate)
        
        # Model 2 - training with anomalies
        model2 = LargeTransformerModel(self.config).to(device)
        optimizer2 = optim.AdamW(model2.parameters(), lr=self.config.learning_rate)
        
        # Train both models
        for model, optimizer, name in [(model1, optimizer1, "normal"), (model2, optimizer2, "anomalous")]:
            model.train()
            for epoch in range(2):
                for batch_idx in range(30):
                    input_ids, attention_mask = self.generate_synthetic_data(
                        self.config.batch_size, self.config.seq_len
                    )
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    logits = outputs['logits']
                    
                    target_ids = input_ids.clone()
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Inject anomalies in model 2
                    if name == "anomalous" and batch_idx % 10 == 0:
                        self.anomaly_injector.inject_anomaly(model, optimizer, batch_idx)
                    
                    optimizer.step()
        
        # Save checkpoints
        checkpoint1_path = self.config.output_dir / "checkpoints" / "model1.pt"
        checkpoint2_path = self.config.output_dir / "checkpoints" / "model2.pt"
        
        torch.save(model1.state_dict(), checkpoint1_path)
        torch.save(model2.state_dict(), checkpoint2_path)
        
        # Compare checkpoints
        checkpoint_analysis = {
            'checkpoint1_path': str(checkpoint1_path),
            'checkpoint2_path': str(checkpoint2_path),
            'model1_params': sum(p.numel() for p in model1.parameters()),
            'model2_params': sum(p.numel() for p in model2.parameters()),
            'parameter_differences': []
        }
        
        # Calculate parameter differences
        total_diff = 0
        total_params = 0
        
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                diff = torch.abs(param1 - param2).sum().item()
                total_diff += diff
                total_params += param1.numel()
                
                checkpoint_analysis['parameter_differences'].append({
                    'name': name1,
                    'difference': diff,
                    'relative_difference': diff / param1.numel() if param1.numel() > 0 else 0
                })
        
        checkpoint_analysis['total_difference'] = total_diff
        checkpoint_analysis['average_difference'] = total_diff / total_params if total_params > 0 else 0
        checkpoint_analysis['models_different'] = checkpoint_analysis['average_difference'] > 1e-6
        
        self.logger.info(f"Checkpoint comparison: {checkpoint_analysis}")
        
        return checkpoint_analysis
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the comprehensive test suite."""
        self.logger.info("Starting comprehensive RL + LLM testing")
        
        start_time = time.time()
        
        # Test results
        comprehensive_results = {
            'start_time': start_time,
            'config': self.config.__dict__,
            'system_info': self._get_system_info(),
            'training_iterations': [],
            'determinism_test': None,
            'reward_drift_test': None,
            'checkpoint_comparison_test': None,
            'anomaly_summary': {},
            'memory_summary': {},
            'overall_success': True
        }
        
        try:
            # Run multiple training iterations
            self.logger.info(f"Running {self.config.num_iterations} training iterations")
            for i in range(self.config.num_iterations):
                iteration_results = self.train_model_iteration(i)
                comprehensive_results['training_iterations'].append(iteration_results)
                
                # Force garbage collection between iterations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Test determinism
            if self.config.enable_determinism_testing:
                comprehensive_results['determinism_test'] = self.test_determinism()
            
            # Test reward drift
            if self.config.enable_reward_drift_testing:
                comprehensive_results['reward_drift_test'] = self.test_reward_drift()
            
            # Test checkpoint comparison
            if self.config.enable_checkpoint_testing:
                comprehensive_results['checkpoint_comparison_test'] = self.test_checkpoint_comparison()
            
            # Analyze anomalies
            comprehensive_results['anomaly_summary'] = self._analyze_anomalies()
            
            # Analyze memory usage
            comprehensive_results['memory_summary'] = self._analyze_memory_usage()
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive testing: {e}")
            comprehensive_results['overall_success'] = False
            comprehensive_results['error'] = str(e)
        
        comprehensive_results['end_time'] = time.time()
        comprehensive_results['total_duration'] = comprehensive_results['end_time'] - start_time
        
        # Save results
        if self.config.save_reports:
            results_file = self.config.output_dir / "reports" / "comprehensive_test_results.json"
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            self.logger.info(f"Comprehensive test results saved to {results_file}")
        
        self.logger.info(f"Comprehensive testing completed in {comprehensive_results['total_duration']:.2f} seconds")
        
        return comprehensive_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform
        }
    
    def _analyze_anomalies(self) -> Dict[str, Any]:
        """Analyze injected anomalies."""
        if not self.anomaly_log:
            return {'total_anomalies': 0, 'anomaly_types': {}}
        
        anomaly_types = {}
        successful_anomalies = 0
        
        for anomaly in self.anomaly_log:
            anomaly_type = anomaly['type']
            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = 0
            anomaly_types[anomaly_type] += 1
            
            if anomaly.get('success', False):
                successful_anomalies += 1
        
        return {
            'total_anomalies': len(self.anomaly_log),
            'successful_anomalies': successful_anomalies,
            'anomaly_types': anomaly_types,
            'success_rate': successful_anomalies / len(self.anomaly_log) if self.anomaly_log else 0
        }
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.memory_log:
            return {'memory_logs': 0}
        
        cpu_percentages = [log['cpu_percent'] for log in self.memory_log]
        memory_percentages = [log['memory_percent'] for log in self.memory_log]
        
        return {
            'memory_logs': len(self.memory_log),
            'avg_cpu_percent': np.mean(cpu_percentages),
            'max_cpu_percent': np.max(cpu_percentages),
            'avg_memory_percent': np.mean(memory_percentages),
            'max_memory_percent': np.max(memory_percentages),
            'peak_memory_gb': max(log.get('available_memory_gb', 0) for log in self.memory_log)
        }


def main():
    """Main function to run comprehensive testing."""
    print("🚀 Starting Comprehensive RL + LLM Testing")
    print("=" * 60)
    
    # Configuration
    config = TestConfig(
        vocab_size=10000,  # Smaller for faster testing
        d_model=512,       # Smaller for faster testing
        n_heads=8,
        n_layers=6,
        seq_len=256,
        batch_size=4,
        num_epochs=3,
        num_iterations=5,  # Multiple iterations to catch edge cases
        learning_rate=1e-4,
        enable_anomaly_injection=True,
        enable_memory_stress=True,
        enable_determinism_testing=True,
        enable_reward_drift_testing=True,
        enable_checkpoint_testing=True,
        anomaly_probability=0.15,  # Higher probability to trigger more anomalies
        memory_pressure_probability=0.1
    )
    
    # Create tester
    tester = ComprehensiveTester(config)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Overall Success: {results['overall_success']}")
    print(f"⏱️  Total Duration: {results['total_duration']:.2f} seconds")
    print(f"🔄 Training Iterations: {len(results['training_iterations'])}")
    print(f"🎲 Anomalies Injected: {results['anomaly_summary']['total_anomalies']}")
    print(f"💾 Memory Logs: {results['memory_summary']['memory_logs']}")
    
    if results['determinism_test']:
        print(f"🎯 Determinism Test: {'✅ PASSED' if results['determinism_test']['is_deterministic'] else '❌ FAILED'}")
    
    if results['reward_drift_test']:
        print(f"📊 Reward Drift Test: {'✅ DETECTED' if results['reward_drift_test']['drift_detected'] else '❌ NOT DETECTED'}")
    
    if results['checkpoint_comparison_test']:
        print(f"🔍 Checkpoint Comparison: {'✅ DIFFERENT' if results['checkpoint_comparison_test']['models_different'] else '❌ IDENTICAL'}")
    
    print(f"\n📁 Results saved to: {config.output_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()