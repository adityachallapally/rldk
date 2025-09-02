# OpenRLHF Integration Guide

This guide provides comprehensive documentation for the RLDK OpenRLHF integration, which offers real-time monitoring, distributed training support, and advanced analytics for OpenRLHF training workflows.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Real-time Monitoring](#real-time-monitoring)
- [Distributed Training Support](#distributed-training-support)
- [Analytics and Dashboard](#analytics-and-dashboard)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Overview

The RLDK OpenRLHF integration provides:

- **Real-time monitoring callbacks** similar to TRL integration
- **Live training metrics collection** with comprehensive tracking
- **Distributed training monitoring** with multi-GPU/multi-node support
- **OpenRLHF-specific analytics** and health monitoring
- **Interactive dashboard** for real-time visualization
- **Checkpoint analysis** and model health tracking
- **Resource monitoring** and performance optimization

## Installation

### Prerequisites

```bash
# Install OpenRLHF (optional - integration works without it)
pip install openrlhf

# Core dependencies are included in RLDK
# Optional: Install Parquet support for better performance
pip install pyarrow
```

### Install RLDK with OpenRLHF Support

```bash
# Install RLDK (if not already installed)
pip install rldk

# The OpenRLHF integration is included in RLDK
```

## Quick Start

### Basic Usage

```python
from rldk.integrations.openrlhf import OpenRLHFCallback, OpenRLHFMonitor
import openrlhf

# Initialize your OpenRLHF trainer
trainer = openrlhf.PPOTrainer(...)

# Add RLDK monitoring
monitor = OpenRLHFMonitor(
    output_dir="./rldk_logs",
    log_interval=10,
    run_id="my_training_run"
)

# Integrate with your training loop
for step in range(num_steps):
    # Your training step
    trainer.step()
    
    # Collect metrics
    metrics = monitor.collect_metrics(trainer)
    monitor.on_step_end(trainer, step)
```

### Real-time Dashboard

```python
from rldk.integrations.openrlhf import OpenRLHFDashboard

# Start the dashboard
dashboard = OpenRLHFDashboard(
    output_dir="./rldk_logs",
    port=5000,
    host="localhost"
)

# Start monitoring (this will open a web interface)
dashboard.start_dashboard()
```

## Core Components

### 1. OpenRLHFCallback

The main callback class for monitoring OpenRLHF training.

```python
from rldk.integrations.openrlhf import OpenRLHFCallback

callback = OpenRLHFCallback(
    output_dir="./rldk_logs",
    log_interval=10,
    alert_thresholds={
        'loss': 2.0,
        'kl_mean': 1.0,
        'gpu_memory_used': 20.0
    },
    enable_resource_monitoring=True,
    enable_distributed_monitoring=True,
    run_id="experiment_001"
)
```

**Key Features:**
- Real-time metrics collection
- Alert system for threshold breaches
- Resource monitoring (GPU/CPU)
- Distributed training support
- Automatic log saving

### 2. OpenRLHFMetrics

Comprehensive metrics container for OpenRLHF training.

```python
from rldk.integrations.openrlhf import OpenRLHFMetrics

metrics = OpenRLHFMetrics(
    step=100,
    loss=0.5,
    reward_mean=2.3,
    kl_mean=0.1,
    learning_rate=1e-4,
    gpu_memory_used=8.5,
    run_id="experiment_001"
)

# Convert to dictionary or DataFrame
metrics_dict = metrics.to_dict()
df_row = metrics.to_dataframe_row()
```

**Available Metrics:**
- Training metrics: loss, learning rate, gradients
- PPO metrics: rewards, KL divergence, entropy, clip ratio
- Resource metrics: GPU/CPU memory, utilization
- Timing metrics: step time, wall time
- Health indicators: stability score, convergence rate

### 3. Distributed Training Support

#### DistributedTrainingMonitor

```python
from rldk.integrations.openrlhf import DistributedTrainingMonitor

monitor = DistributedTrainingMonitor(
    output_dir="./distributed_logs",
    sync_interval=5,
    network_monitoring=True
)
```

#### MultiGPUMonitor

```python
from rldk.integrations.openrlhf import MultiGPUMonitor

monitor = MultiGPUMonitor(
    output_dir="./multi_gpu_logs",
    enable_resource_monitoring=True
)

# Get metrics for all GPUs
gpu_metrics = monitor.get_gpu_metrics()
```

## Real-time Monitoring

### Live Metrics Collection

```python
from rldk.integrations.openrlhf import OpenRLHFCallback

# Initialize with real-time monitoring
callback = OpenRLHFCallback(
    output_dir="./logs",
    log_interval=5,  # Log every 5 steps
    enable_resource_monitoring=True
)

# In your training loop
for step in range(num_steps):
    # Training step
    trainer.step()
    
    # Collect and log metrics
    callback.on_step_end(trainer, step)
    
    # Get latest metrics
    latest_metrics = callback.get_latest_metrics()
    print(f"Step {step}: Loss={latest_metrics.loss:.4f}, Reward={latest_metrics.reward_mean:.3f}")
```

### Alert System

```python
# Define alert thresholds
alert_thresholds = {
    'loss': 2.0,           # Alert if loss > 2.0
    'kl_mean': 1.0,        # Alert if KL > 1.0
    'gpu_memory_used': 20.0,  # Alert if GPU memory > 20GB
    'training_stability_score': 0.3  # Alert if stability < 0.3
}

callback = OpenRLHFCallback(
    alert_thresholds=alert_thresholds
)

# Add custom alert callback
def custom_alert_handler(alert_data):
    print(f"ALERT: {alert_data['metric']} = {alert_data['current_value']} (threshold: {alert_data['threshold']})")

callback.add_alert_callback(custom_alert_handler)
```

## Distributed Training Support

### Multi-Node Monitoring

```python
from rldk.integrations.openrlhf.distributed import (
    DistributedMetricsCollector,
    MultiNodeMonitor
)

# Initialize distributed monitoring
distributed_collector = DistributedMetricsCollector(
    collect_interval=1.0,
    enable_network_monitoring=True,
    enable_gpu_monitoring=True
)

node_monitor = MultiNodeMonitor(
    master_node="localhost",
    master_port=12355
)

# Start monitoring
distributed_collector.start_collection()
node_monitor.start_monitoring()

# Get aggregated metrics
latest_metrics = distributed_collector.get_latest_metrics()
print(f"World size: {latest_metrics.world_size}")
print(f"Total GPU memory: {latest_metrics.total_gpu_memory_used:.2f} GB")
```

### GPU Memory Monitoring

```python
from rldk.integrations.openrlhf.distributed import GPUMemoryMonitor

gpu_monitor = GPUMemoryMonitor()

# Get current memory usage for all GPUs
memory_usage = gpu_monitor.get_current_memory_usage()
for device_id, usage in memory_usage.items():
    print(f"GPU {device_id}: {usage['used']:.2f} GB used, {usage['free']:.2f} GB free")

# Get memory trends
trends = gpu_monitor.get_memory_trends()
for device_id, trend in trends.items():
    print(f"GPU {device_id} memory trend: {trend['used_trend']:.3f} GB/step")
```

## Analytics and Dashboard

### Training Health Analysis

```python
from rldk.integrations.openrlhf import OpenRLHFTrainingMonitor

monitor = OpenRLHFTrainingMonitor(
    output_dir="./analysis",
    analysis_window=100,
    enable_anomaly_detection=True,
    enable_convergence_analysis=True
)

# Add metrics during training
for step in range(num_steps):
    metrics = collect_metrics(trainer)
    monitor.add_metrics(metrics)

# Get health summary
health_summary = monitor.get_health_summary()
print(f"Overall health: {health_summary['overall_health']:.3f}")
print(f"Stability score: {health_summary['stability_score']:.3f}")
print(f"Convergence rate: {health_summary['convergence_rate']:.3f}")

# Save analysis
monitor.save_analysis("training_health_analysis.json")
```

### Interactive Dashboard

```python
from rldk.integrations.openrlhf import OpenRLHFDashboard

# Initialize dashboard
dashboard = OpenRLHFDashboard(
    output_dir="./rldk_logs",
    port=5000,
    host="0.0.0.0",  # Allow external access
    enable_auto_refresh=True,
    refresh_interval=2.0
)

# Start dashboard (opens web interface)
dashboard.start_dashboard()
```

**Dashboard Features:**
- Real-time metrics visualization
- Training health monitoring
- Resource usage tracking
- Alert notifications
- Data export capabilities
- Responsive design for mobile/desktop

### Checkpoint Analysis

```python
from rldk.integrations.openrlhf import OpenRLHFCheckpointMonitor

checkpoint_monitor = OpenRLHFCheckpointMonitor(
    checkpoint_dir="./checkpoints",
    enable_validation=True,
    enable_size_analysis=True
)

# Analyze a checkpoint
metrics = checkpoint_monitor.analyze_checkpoint(
    checkpoint_path="./checkpoints/checkpoint_1000.pt",
    step=1000
)

print(f"Checkpoint size: {metrics.model_size:.2f} GB")
print(f"Validation score: {metrics.validation_score:.3f}")

# Get checkpoint summary
summary = checkpoint_monitor.get_checkpoint_summary()
print(f"Total checkpoints: {summary['total_checkpoints']}")
print(f"Average size: {summary['avg_checkpoint_size']:.2f} GB")
```

## Advanced Usage

### Custom Analytics

```python
from rldk.integrations.openrlhf import OpenRLHFAnalytics

analytics = OpenRLHFAnalytics(output_dir="./analytics")

# Analyze complete training run
analysis_results = analytics.analyze_training_run(metrics_history)

print("Training Health:", analysis_results['training_health'])
print("Resource Summary:", analysis_results['resource_summary'])
print("Checkpoint Summary:", analysis_results['checkpoint_summary'])
```

### Resource Monitoring

```python
from rldk.integrations.openrlhf import OpenRLHFResourceMonitor

resource_monitor = OpenRLHFResourceMonitor(monitor_interval=1.0)

# Start monitoring
resource_monitor.start_monitoring()

# During training...
time.sleep(60)  # Monitor for 1 minute

# Stop and get summary
resource_monitor.stop_monitoring()
summary = resource_monitor.get_resource_summary()

print(f"Average CPU utilization: {summary['avg_cpu_utilization']:.1f}%")
print(f"Peak GPU memory usage: {summary['peak_memory_usage']:.2f} GB")
print(f"Total monitoring time: {summary['monitoring_duration']:.1f} seconds")
```

### Integration with Existing Training Scripts

```python
# Example integration with existing OpenRLHF training script
import openrlhf
from rldk.integrations.openrlhf import OpenRLHFMonitor

def train_with_monitoring():
    # Initialize trainer
    trainer = openrlhf.PPOTrainer(
        model=model,
        config=config,
        # ... other parameters
    )
    
    # Initialize RLDK monitor
    monitor = OpenRLHFMonitor(
        output_dir="./rldk_logs",
        log_interval=10,
        run_id=f"experiment_{int(time.time())}",
        enable_resource_monitoring=True,
        enable_distributed_monitoring=True
    )
    
    # Training loop
    for step in range(config.max_steps):
        # Training step
        trainer.step()
        
        # Monitor training
        monitor.on_step_end(trainer, step)
        
        # Optional: Print progress
        if step % 100 == 0:
            latest = monitor.get_latest_metrics()
            print(f"Step {step}: Loss={latest.loss:.4f}, Reward={latest.reward_mean:.3f}")
    
    # Save final analysis
    monitor._save_metrics()
    print("Training completed. Check ./rldk_logs for detailed analysis.")

if __name__ == "__main__":
    train_with_monitoring()
```

## API Reference

### OpenRLHFCallback

```python
class OpenRLHFCallback:
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        log_interval: int = 10,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_resource_monitoring: bool = True,
        enable_distributed_monitoring: bool = True,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    )
    
    def on_train_begin(self, trainer, **kwargs)
    def on_train_end(self, trainer, **kwargs)
    def on_step_begin(self, trainer, step: int, **kwargs)
    def on_step_end(self, trainer, step: int, **kwargs)
    def add_alert_callback(self, callback: Callable)
    def get_metrics_dataframe(self) -> pd.DataFrame
    def get_latest_metrics(self) -> OpenRLHFMetrics
```

### OpenRLHFMetrics

```python
@dataclass
class OpenRLHFMetrics:
    # Training metrics
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    
    # PPO metrics
    reward_mean: float = 0.0
    kl_mean: float = 0.0
    entropy_mean: float = 0.0
    clip_frac: float = 0.0
    
    # Resource metrics
    gpu_memory_used: float = 0.0
    cpu_utilization: float = 0.0
    
    # Health indicators
    training_stability_score: float = 1.0
    convergence_indicator: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]
    def to_dataframe_row(self) -> Dict[str, Any]
```

### OpenRLHFDashboard

```python
class OpenRLHFDashboard:
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        port: int = 5000,
        host: str = "localhost",
        enable_auto_refresh: bool = True,
        refresh_interval: float = 1.0,
    )
    
    def start_dashboard(self)
    def stop_dashboard(self)
    def add_metrics(self, metrics: OpenRLHFMetrics)
    def get_dashboard_url(self) -> str
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# If you get import errors for OpenRLHF
pip install openrlhf

# If you get import errors for RLDK
pip install rldk
```

#### 2. Dashboard Not Starting

```python
# Check if port is available
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 5000))
if result == 0:
    print("Port 5000 is in use, try a different port")
sock.close()

# Use different port
dashboard = OpenRLHFDashboard(port=5001)
```

#### 3. GPU Monitoring Issues

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Disable GPU monitoring if not available
monitor = OpenRLHFCallback(enable_resource_monitoring=False)
```

#### 4. Distributed Training Issues

```python
# Check if distributed training is initialized
import torch.distributed as dist
print(f"Distributed initialized: {dist.is_initialized()}")

# Disable distributed monitoring if not needed
monitor = OpenRLHFCallback(enable_distributed_monitoring=False)
```

### Performance Optimization

#### 1. Reduce Monitoring Frequency

```python
# Increase log interval to reduce overhead
monitor = OpenRLHFCallback(log_interval=50)  # Log every 50 steps instead of 10
```

#### 2. Disable Resource Monitoring

```python
# Disable resource monitoring for better performance
monitor = OpenRLHFCallback(enable_resource_monitoring=False)
```

#### 3. Use Efficient Data Formats

```python
# The integration automatically saves in both CSV and Parquet formats
# Parquet is more efficient for large datasets
df = pd.read_parquet("./rldk_logs/metrics_run_123.parquet")
```

### Getting Help

1. **Check the logs**: Look in your output directory for detailed logs
2. **Run tests**: Use the provided test suite to verify installation
3. **Enable debug mode**: Set logging level to DEBUG for more information
4. **Check dependencies**: Ensure all required packages are installed

```bash
# Run the test suite
python test_openrlhf_integration.py
```

## Examples

### Complete Training Script with Monitoring

```python
#!/usr/bin/env python3
"""Complete OpenRLHF training script with RLDK monitoring."""

import time
import openrlhf
from rldk.integrations.openrlhf import OpenRLHFMonitor, OpenRLHFDashboard

def main():
    # Initialize trainer
    trainer = openrlhf.PPOTrainer(
        model=model,
        config=config,
        # ... other parameters
    )
    
    # Initialize monitoring
    monitor = OpenRLHFMonitor(
        output_dir="./training_logs",
        log_interval=10,
        run_id=f"experiment_{int(time.time())}",
        enable_resource_monitoring=True,
        enable_distributed_monitoring=True
    )
    
    # Start dashboard in background
    dashboard = OpenRLHFDashboard(
        output_dir="./training_logs",
        port=5000,
        enable_auto_refresh=True
    )
    
    print("🚀 Starting training with monitoring...")
    print(f"📊 Dashboard available at: {dashboard.get_dashboard_url()}")
    
    try:
        # Training loop
        for step in range(config.max_steps):
            # Training step
            trainer.step()
            
            # Monitor training
            monitor.on_step_end(trainer, step)
            
            # Print progress
            if step % 100 == 0:
                latest = monitor.get_latest_metrics()
                print(f"Step {step}: Loss={latest.loss:.4f}, Reward={latest.reward_mean:.3f}")
        
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
    
    finally:
        # Save final metrics
        monitor._save_metrics()
        print("📁 Logs saved to ./training_logs")

if __name__ == "__main__":
    main()
```

This comprehensive integration provides everything needed for professional OpenRLHF training monitoring, from basic metrics collection to advanced analytics and real-time visualization.