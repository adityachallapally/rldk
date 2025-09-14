# OpenRLHF Integration Example

This example demonstrates how to integrate RLDK with OpenRLHF for distributed RLHF training with comprehensive monitoring and analysis.

## Overview

This example shows how to:
- Set up RLDK with OpenRLHF distributed training
- Monitor network performance and distributed metrics
- Track large-scale RLHF experiments
- Analyze distributed training health

## Prerequisites

```bash
pip install rldk[dev,openrlhf]
pip install openrlhf torch transformers datasets
```

## Complete OpenRLHF Integration Example

```python
#!/usr/bin/env python3
"""
OpenRLHF integration example with RLDK.
Demonstrates distributed RLHF training with comprehensive monitoring.
"""

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.integrations.openrlhf import OpenRLHFCallback, NetworkMonitor
from rldk.utils.seed import set_global_seed


def setup_distributed():
    """Setup distributed training environment."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single node setup for demo
        return 0, 1, 0


def create_preference_dataset(tokenizer, size=1000):
    """Create a dummy preference dataset for RLHF."""
    prompts = [
        "Explain the concept of",
        "What is the best way to",
        "How can I improve",
        "Tell me about",
        "The most important thing about",
    ] * (size // 5)
    
    # Create chosen and rejected responses
    chosen_responses = [
        "This is a helpful and detailed response that provides value.",
        "Here's a comprehensive answer that addresses your question.",
        "I'd be happy to help you with this important topic.",
        "This is an informative explanation that should be useful.",
        "Let me provide you with a clear and accurate answer.",
    ] * (size // 5)
    
    rejected_responses = [
        "I don't know.",
        "That's not important.",
        "Figure it out yourself.",
        "This is boring.",
        "I can't help with that.",
    ] * (size // 5)
    
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses
    })
    
    return dataset


class MockOpenRLHFTrainer:
    """Mock OpenRLHF trainer for demonstration."""
    
    def __init__(self, model, tokenizer, dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.callbacks = []
        
    def add_callback(self, callback):
        self.callbacks.append(callback)
        
    def train(self, max_steps=100):
        """Mock training loop."""
        print(f"🚀 Starting OpenRLHF training for {max_steps} steps...")
        
        # Initialize callbacks
        for callback in self.callbacks:
            callback.on_train_begin()
        
        for step in range(max_steps):
            # Simulate training metrics
            metrics = {
                "step": step,
                "reward_mean": 0.5 + 0.3 * torch.randn(1).item(),
                "reward_std": 0.1 + 0.05 * torch.randn(1).item(),
                "kl_divergence": 0.1 + 0.05 * torch.randn(1).item(),
                "policy_loss": 0.5 + 0.2 * torch.randn(1).item(),
                "value_loss": 0.3 + 0.1 * torch.randn(1).item(),
                "entropy": 2.0 + 0.5 * torch.randn(1).item(),
                "learning_rate": self.config.get("learning_rate", 1e-5),
                "grad_norm": 1.0 + 0.5 * torch.randn(1).item(),
                
                # Distributed metrics
                "network_latency": 10 + 5 * torch.randn(1).item(),
                "bandwidth_utilization": 0.7 + 0.2 * torch.randn(1).item(),
                "memory_usage": 0.8 + 0.1 * torch.randn(1).item(),
                "gpu_utilization": 0.9 + 0.05 * torch.randn(1).item(),
            }
            
            # Call callbacks
            for callback in self.callbacks:
                callback.on_step_end(step, metrics)
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step:3d}: "
                      f"Reward={metrics['reward_mean']:.3f}, "
                      f"KL={metrics['kl_divergence']:.4f}, "
                      f"Loss={metrics['policy_loss']:.4f}")
        
        # Finish callbacks
        for callback in self.callbacks:
            callback.on_train_end()
        
        print("✅ Training completed!")


def main():
    """Main function demonstrating OpenRLHF + RLDK integration."""
    
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    # Configuration
    seed = 42
    model_name = "gpt2"  # Use small model for demo
    max_steps = 100
    
    print(f"🚀 Starting OpenRLHF + RLDK Integration Demo")
    print(f"Rank: {rank}/{world_size}, Model: {model_name}, Steps: {max_steps}")
    
    # Set reproducible seeds
    set_global_seed(seed, deterministic=True)
    
    # Set up RLDK experiment tracking (only on rank 0)
    if rank == 0:
        tracking_config = TrackingConfig(
            experiment_name="openrlhf_rlhf_demo",
            enable_dataset_tracking=True,
            enable_model_tracking=True,
            enable_environment_tracking=True,
            enable_seed_tracking=True,
            enable_git_tracking=True,
            tags=["openrlhf", "rlhf", "distributed", "demo"],
            notes="OpenRLHF distributed RLHF training with RLDK integration"
        )
        
        tracker = ExperimentTracker(tracking_config)
        tracker.start_experiment()
        tracker.set_seeds(seed)
        
        # Add distributed training metadata
        tracker.add_metadata("world_size", world_size)
        tracker.add_metadata("distributed_backend", "nccl")
        tracker.add_metadata("model_name", model_name)
    else:
        tracker = None
    
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Track models (only on rank 0)
    if rank == 0:
        tracker.track_model(model, "rlhf_model")
        tracker.track_tokenizer(tokenizer, "tokenizer")
    
    # Create dataset
    print("📊 Creating preference dataset...")
    dataset = create_preference_dataset(tokenizer, size=1000)
    
    if rank == 0:
        tracker.track_dataset(dataset, "preference_dataset")
    
    # Training configuration
    training_config = {
        "learning_rate": 1e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "warmup_steps": 10,
        "seed": seed,
        "distributed": world_size > 1
    }
    
    if rank == 0:
        tracker.add_metadata("training_config", training_config)
    
    # Create OpenRLHF callback with RLDK integration
    openrlhf_callback = OpenRLHFCallback(
        enable_tracking=(rank == 0),  # Only track on rank 0
        enable_forensics=True,
        enable_network_monitoring=True,
        enable_performance_analysis=True,
        
        # Tracking configuration
        tracking_config=tracking_config if rank == 0 else None,
        
        # Forensics configuration
        forensics_config={
            "kl_target": 0.1,
            "enable_kl_schedule_tracking": True,
            "enable_gradient_norms_analysis": True,
            "enable_advantage_statistics": True,
            "distributed_analysis": True
        },
        
        # Network monitoring configuration
        network_config={
            "monitor_bandwidth": True,
            "monitor_latency": True,
            "monitor_memory": True,
            "alert_thresholds": {
                "latency_ms": 100,
                "bandwidth_utilization": 0.9,
                "memory_usage": 0.95
            }
        },
        
        # Performance analysis configuration
        performance_config={
            "profile_frequency": 50,  # Every 50 steps
            "analyze_bottlenecks": True,
            "track_gpu_utilization": True
        }
    )
    
    # Create mock OpenRLHF trainer
    print("🏋️ Setting up OpenRLHF trainer...")
    trainer = MockOpenRLHFTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=training_config
    )
    
    # Add RLDK callback
    trainer.add_callback(openrlhf_callback)
    
    # Start training
    print("🎯 Starting distributed RLHF training...")
    trainer.train(max_steps=max_steps)
    
    # Get reports (only on rank 0)
    if rank == 0:
        print("\n📋 Generating RLDK reports...")
        
        # Get forensics report
        forensics_report = openrlhf_callback.get_forensics_report()
        print(f"🔍 Forensics Summary:")
        print(f"  Total anomalies detected: {len(forensics_report.anomalies)}")
        print(f"  Training health score: {forensics_report.health_score:.2f}")
        
        if forensics_report.anomalies:
            print("  Recent anomalies:")
            for anomaly in forensics_report.anomalies[-3:]:  # Show last 3
                print(f"    - {anomaly.type}: {anomaly.description}")
        
        # Get network monitoring report
        network_report = openrlhf_callback.get_network_report()
        print(f"🌐 Network Monitoring Summary:")
        print(f"  Average latency: {network_report.avg_latency:.2f}ms")
        print(f"  Peak bandwidth usage: {network_report.peak_bandwidth:.1%}")
        print(f"  Network alerts: {len(network_report.alerts)}")
        
        if network_report.alerts:
            print("  Recent network alerts:")
            for alert in network_report.alerts[-3:]:
                print(f"    - {alert.type}: {alert.message}")
        
        # Get performance analysis
        performance_report = openrlhf_callback.get_performance_report()
        print(f"⚡ Performance Analysis:")
        print(f"  Average GPU utilization: {performance_report.avg_gpu_util:.1%}")
        print(f"  Memory efficiency: {performance_report.memory_efficiency:.1%}")
        print(f"  Throughput: {performance_report.tokens_per_second:.1f} tokens/sec")
        
        # Get distributed training insights
        distributed_report = openrlhf_callback.get_distributed_report()
        print(f"🔄 Distributed Training Insights:")
        print(f"  Communication overhead: {distributed_report.comm_overhead:.1%}")
        print(f"  Load balance score: {distributed_report.load_balance:.2f}")
        print(f"  Synchronization efficiency: {distributed_report.sync_efficiency:.1%}")
        
        # Finish experiment tracking
        experiment_path = tracker.finish_experiment()
        print(f"💾 Experiment saved to: {experiment_path}")
        
        print(f"\n🎉 OpenRLHF + RLDK integration demo completed!")
        
        return {
            'experiment_path': experiment_path,
            'forensics_report': forensics_report,
            'network_report': network_report,
            'performance_report': performance_report,
            'distributed_report': distributed_report
        }
    
    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()
    
    return None


def run_network_diagnostics():
    """Demonstrate network diagnostics capabilities."""
    print("\n🔧 Running network diagnostics...")
    
    from rldk.integrations.openrlhf import NetworkDiagnostics
    
    diagnostics = NetworkDiagnostics()
    
    # Run comprehensive network analysis
    report = diagnostics.run_full_analysis()
    
    print(f"Network Diagnostics Report:")
    print(f"  Bandwidth: {report.bandwidth_mbps:.1f} Mbps")
    print(f"  Latency: {report.latency_ms:.2f} ms")
    print(f"  Packet loss: {report.packet_loss:.2%}")
    print(f"  Network stability: {report.stability_score:.2f}")
    
    # Check for distributed training readiness
    readiness = diagnostics.check_distributed_readiness()
    
    if readiness.ready:
        print("✅ Network ready for distributed training")
    else:
        print("⚠️  Network issues detected:")
        for issue in readiness.issues:
            print(f"    - {issue}")


if __name__ == "__main__":
    try:
        # Run main demo
        result = main()
        
        # Run network diagnostics (only on rank 0)
        if result is not None:  # Only rank 0 returns results
            run_network_diagnostics()
            
        print(f"\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Demo encountered an issue: {e}")
        print("This is normal for the demo - it shows integration patterns.")
        print("For production use, ensure proper distributed setup and network configuration.")
```

## Advanced Features

### Network Monitoring

```python
from rldk.integrations.openrlhf import NetworkMonitor

# Advanced network monitoring
monitor = NetworkMonitor(
    monitor_interval=10,  # Check every 10 seconds
    alert_thresholds={
        "latency_ms": 50,
        "bandwidth_utilization": 0.8,
        "packet_loss": 0.01,
        "jitter_ms": 5
    },
    enable_auto_scaling=True,
    enable_fault_tolerance=True
)

# Start monitoring
monitor.start()

# Get real-time metrics
metrics = monitor.get_current_metrics()
print(f"Current latency: {metrics.latency_ms}ms")
print(f"Bandwidth usage: {metrics.bandwidth_utilization:.1%}")

# Check for alerts
alerts = monitor.get_active_alerts()
for alert in alerts:
    print(f"Alert: {alert.message}")
```

### Performance Analysis

```python
from rldk.integrations.openrlhf import PerformanceAnalyzer

# Comprehensive performance analysis
analyzer = PerformanceAnalyzer(
    profile_memory=True,
    profile_compute=True,
    profile_communication=True,
    enable_bottleneck_detection=True
)

# Analyze training step
with analyzer.profile_step():
    # Your training step here
    pass

# Get analysis results
report = analyzer.get_report()
print(f"Bottlenecks detected: {len(report.bottlenecks)}")
print(f"Optimization suggestions: {len(report.suggestions)}")
```

### Distributed Health Monitoring

```python
from rldk.integrations.openrlhf import DistributedHealthMonitor

# Monitor distributed training health
health_monitor = DistributedHealthMonitor(
    check_interval=30,  # Check every 30 seconds
    enable_auto_recovery=True,
    failure_detection_threshold=3
)

# Check cluster health
health = health_monitor.check_cluster_health()
print(f"Cluster health: {health.status}")
print(f"Active nodes: {health.active_nodes}/{health.total_nodes}")

# Get node-specific health
for node_id, node_health in health.node_status.items():
    print(f"Node {node_id}: {node_health.status}")
```

## Production Configuration

### Large-Scale Setup

```yaml
# openrlhf_config.yaml
distributed:
  world_size: 8
  backend: "nccl"
  init_method: "env://"
  
model:
  name: "llama-7b"
  gradient_checkpointing: true
  mixed_precision: "fp16"
  
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  learning_rate: 1e-5
  
rldk:
  tracking:
    save_model_weights: false  # Too large for tracking
    dataset_sample_size: 10000
    enable_git_tracking: true
    
  forensics:
    kl_target: 0.1
    gradient_threshold: 5.0
    enable_distributed_analysis: true
    
  network:
    monitor_bandwidth: true
    monitor_latency: true
    alert_on_degradation: true
    
  performance:
    profile_frequency: 100
    enable_memory_profiling: true
    track_communication_overhead: true
```

### Multi-Node Setup

```python
# multi_node_setup.py
import os
from rldk.integrations.openrlhf import MultiNodeSetup

# Configure multi-node training
setup = MultiNodeSetup(
    master_addr=os.environ.get("MASTER_ADDR", "localhost"),
    master_port=int(os.environ.get("MASTER_PORT", "29500")),
    world_size=int(os.environ.get("WORLD_SIZE", "1")),
    rank=int(os.environ.get("RANK", "0"))
)

# Initialize distributed training with RLDK
setup.initialize_with_rldk(
    tracking_config=tracking_config,
    enable_network_monitoring=True,
    enable_fault_tolerance=True
)
```

## Best Practices

### 1. Resource Management
- Monitor GPU memory usage across nodes
- Track network bandwidth utilization
- Set appropriate alert thresholds
- Enable automatic scaling when possible

### 2. Fault Tolerance
- Enable checkpoint saving
- Monitor node health
- Implement automatic recovery
- Track communication failures

### 3. Performance Optimization
- Profile communication overhead
- Monitor load balancing
- Track synchronization efficiency
- Optimize data pipeline

### 4. Monitoring Strategy
- Use centralized logging (rank 0 only)
- Aggregate metrics across nodes
- Set up real-time alerts
- Regular health checks

## Troubleshooting

### Common Issues

1. **Network Bottlenecks**: Monitor bandwidth and latency
2. **Memory Issues**: Track GPU memory across nodes
3. **Synchronization Problems**: Check communication overhead
4. **Load Imbalance**: Monitor per-node utilization

### Performance Tips

- Use efficient communication backends (NCCL for GPUs)
- Optimize batch sizes for your hardware
- Monitor and tune gradient accumulation
- Use mixed precision training when possible

## Related Examples

- [Basic PPO CartPole](basic-ppo-cartpole.md) - Simple RLDK introduction
- [TRL Integration](trl-integration.md) - Single-node transformer training
- [Network Monitoring Guide](../user-guide/forensics.md#network-monitoring)

For more details, see the [OpenRLHF Integration Guide](../user-guide/tracking.md#openrlhf-integration) and [API Reference](../reference/api.md#openrlhf-integration).
