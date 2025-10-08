"""RLDK callbacks for OpenRLHF training loops with real-time monitoring."""

import json
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil
import torch

try:
    import openrlhf
    from openrlhf.models import ActorCritic, RewardModel
    from openrlhf.trainer import CPOTrainer, DDPOTrainer, PPOTrainer
    OPENRLHF_AVAILABLE = True
except ImportError:
    OPENRLHF_AVAILABLE = False
    PPOTrainer = None
    DDPOTrainer = None
    CPOTrainer = None

# Import Event schema for proper JSONL emission
try:
    from ...io.event_schema import Event, create_event_from_row
    EVENT_SCHEMA_AVAILABLE = True
except ImportError:
    EVENT_SCHEMA_AVAILABLE = False
    Event = None
    create_event_from_row = None


@dataclass
class OpenRLHFMetrics:
    """Container for OpenRLHF-specific metrics collected during training."""

    # Training step info
    step: int = 0
    epoch: float = 0.0
    phase: str = "train"  # train, eval, test

    # Learning metrics
    learning_rate: float = 0.0
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    kl_loss: float = 0.0
    entropy_loss: float = 0.0

    # PPO-specific metrics
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    kl_mean: float = 0.0
    kl_std: float = 0.0
    kl_divergence: float = 0.0

    entropy_mean: float = 0.0
    entropy_std: float = 0.0

    clip_frac: float = 0.0
    clip_ratio: float = 0.0

    # Value function metrics
    value_mean: float = 0.0
    value_std: float = 0.0
    value_loss_mean: float = 0.0

    # Gradient metrics
    grad_norm: float = 0.0
    grad_norm_actor: float = 0.0
    grad_norm_critic: float = 0.0

    # Token metrics
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0

    # Resource metrics
    gpu_memory_used: float = 0.0
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    gpu_utilization: float = 0.0

    cpu_memory_used: float = 0.0
    cpu_utilization: float = 0.0

    # Timing metrics
    step_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    wall_time: float = 0.0

    # Training health indicators
    training_stability_score: float = 1.0
    convergence_indicator: float = 0.0
    reward_trend: float = 0.0
    kl_trend: float = 0.0

    # Distributed training metrics
    world_size: int = 1
    local_rank: int = 0
    global_rank: int = 0
    node_id: str = ""

    # Network metrics (for multi-node)
    network_bandwidth: float = 0.0  # Legacy attribute for backward compatibility
    network_latency: float = 0.0    # Legacy attribute for backward compatibility
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    bandwidth_upload_mbps: float = 0.0
    bandwidth_download_mbps: float = 0.0
    total_bandwidth_mbps: float = 0.0
    allreduce_time: float = 0.0
    allreduce_bandwidth: float = 0.0
    broadcast_bandwidth: float = 0.0
    gather_bandwidth: float = 0.0
    scatter_bandwidth: float = 0.0
    packet_loss_percent: float = 0.0
    network_errors: int = 0
    dns_resolution_ms: float = 0.0

    # Metadata
    run_id: str = ""
    git_sha: str = ""
    seed: int = 0
    model_name: str = ""
    dataset_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert to DataFrame row format."""
        return self.to_dict()


class OpenRLHFCallback:
    """Base callback for OpenRLHF training with real-time monitoring."""

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
        enable_jsonl_logging: bool = True,
        jsonl_log_interval: int = 1,
        network_sampling_frequency: int = None,
    ):
        """Initialize OpenRLHF callback.

        Args:
            output_dir: Directory to save monitoring logs
            log_interval: Steps between detailed logging
            alert_thresholds: Thresholds for triggering alerts
            enable_resource_monitoring: Whether to monitor resource usage
            enable_distributed_monitoring: Whether to monitor distributed training
            run_id: Unique identifier for this training run
            model_name: Name of the model being trained
            dataset_name: Name of the dataset being used
            enable_jsonl_logging: Whether to enable JSONL event logging
            jsonl_log_interval: Steps between JSONL event logging
            network_sampling_frequency: How often to sample network metrics (every N steps).
                                      If None, will use RLDK_NETWORK_SAMPLING_FREQUENCY env var or default to 10.
        """
        if not OPENRLHF_AVAILABLE:
            raise ImportError(
                "OpenRLHF is required for OpenRLHFCallback. Install with: pip install openrlhf"
            )

        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate log intervals
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if jsonl_log_interval <= 0:
            raise ValueError("jsonl_log_interval must be positive")

        # Handle network sampling frequency
        if network_sampling_frequency is None:
            import os
            network_sampling_frequency = int(os.environ.get('RLDK_NETWORK_SAMPLING_FREQUENCY', '10'))
        if network_sampling_frequency <= 0:
            raise ValueError("network_sampling_frequency must be positive")

        self.log_interval = log_interval
        self.alert_thresholds = alert_thresholds or {}
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_distributed_monitoring = enable_distributed_monitoring
        self.enable_jsonl_logging = enable_jsonl_logging
        self.jsonl_log_interval = jsonl_log_interval
        self.network_sampling_frequency = network_sampling_frequency

        self.run_id = run_id or f"openrlhf_run_{int(time.time())}"
        self.model_name = model_name or "unknown_model"
        self.dataset_name = dataset_name or "unknown_dataset"

        # Metrics storage
        self.metrics_history: List[OpenRLHFMetrics] = []
        self.current_metrics = OpenRLHFMetrics()

        # JSONL logging setup
        self.jsonl_file = None
        if self.enable_jsonl_logging:
            self._setup_jsonl_logging()

        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()

        # Alert system
        self.alert_callbacks: List[Callable] = []

        # Performance tracking
        self.step_times = []
        self.last_step_time = time.time()

        print(f"🚀 OpenRLHF Callback initialized - Run ID: {self.run_id}")
        print(f"📊 Output directory: {self.output_dir}")
        if self.enable_jsonl_logging:
            print(f"📝 JSONL logging enabled - interval: {self.jsonl_log_interval}")

        # Initialize distributed monitoring if enabled
        if self.enable_distributed_monitoring:
            self._init_distributed_monitoring()

    def _init_distributed_monitoring(self):
        """Initialize distributed training monitoring."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                self.current_metrics.world_size = dist.get_world_size()
                self.current_metrics.local_rank = dist.get_local_rank()
                self.current_metrics.global_rank = dist.get_rank()

                # Safe node ID calculation - handle CPU-only distributed training
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    self.current_metrics.node_id = f"node_{dist.get_rank() // torch.cuda.device_count()}"
                else:
                    # For CPU-only distributed training, use a simple node identifier
                    self.current_metrics.node_id = f"node_{dist.get_rank()}"
        except Exception as e:
            warnings.warn(f"Failed to initialize distributed monitoring: {e}", stacklevel=2)

    def on_train_begin(self, trainer, **kwargs):
        """Called when training begins."""
        self.monitoring_active = True
        self._start_monitoring_thread()

        # Log training start
        self._log_event("training_started", {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "world_size": self.current_metrics.world_size,
        })

    def on_train_end(self, trainer, **kwargs):
        """Called when training ends."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # Close JSONL file
        self._close_jsonl_file()

        # Save final metrics
        self._save_metrics()

        # Log training end
        self._log_event("training_ended", {
            "run_id": self.run_id,
            "total_steps": len(self.metrics_history),
            "final_loss": self.current_metrics.loss,
        })

    def on_step_begin(self, trainer, step: int, **kwargs):
        """Called at the beginning of each training step."""
        self.current_metrics.step = step
        self.current_metrics.phase = "train"
        self.last_step_time = time.time()

        # Collect resource metrics at step start
        if self.enable_resource_monitoring:
            self._collect_resource_metrics()

    def on_step_end(self, trainer, step: int, **kwargs):
        """Called at the end of each training step."""
        # Calculate step time
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.current_metrics.step_time = step_time
        self.current_metrics.wall_time = current_time

        # Store step time for trend analysis
        self.step_times.append(step_time)
        if len(self.step_times) > 100:  # Keep last 100 steps
            self.step_times.pop(0)

        # Collect training metrics from trainer
        self._collect_training_metrics(trainer)

        # Collect resource metrics at step end
        if self.enable_resource_monitoring:
            self._collect_resource_metrics()

        # Store metrics
        self.metrics_history.append(self.current_metrics)

        # Log JSONL event at specified intervals
        if (self.enable_jsonl_logging and
            self.jsonl_log_interval > 0 and
            step % self.jsonl_log_interval == 0):
            self._log_jsonl_event(step, {})

        # Check for alerts
        self._check_alerts()

        # Log detailed metrics at intervals
        if step % self.log_interval == 0:
            self._log_detailed_metrics(step)

        # Update current metrics for next step
        self.current_metrics = OpenRLHFMetrics(
            run_id=self.run_id,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            world_size=self.current_metrics.world_size,
            local_rank=self.current_metrics.local_rank,
            global_rank=self.current_metrics.global_rank,
            node_id=self.current_metrics.node_id,
        )

    def _collect_training_metrics(self, trainer):
        """Collect training metrics from the trainer."""
        try:
            # Get basic training info
            if hasattr(trainer, 'get_train_dataloader'):
                dataloader = trainer.get_train_dataloader()
                if hasattr(dataloader, 'dataset'):
                    self.current_metrics.tokens_in = len(dataloader.dataset)

            # Get loss information
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                log_history = trainer.state.log_history
                if log_history:
                    latest_log = log_history[-1]
                    self.current_metrics.loss = latest_log.get('loss', 0.0)
                    self.current_metrics.learning_rate = latest_log.get('learning_rate', 0.0)
                    self.current_metrics.grad_norm = latest_log.get('grad_norm', 0.0)

            # Get OpenRLHF-specific metrics
            if hasattr(trainer, 'ppo_config'):
                config = trainer.ppo_config
                self.current_metrics.clip_ratio = getattr(config, 'cliprange', 0.2)

            # Get reward and KL metrics from trainer state
            if hasattr(trainer, 'reward_stats'):
                reward_stats = trainer.reward_stats
                if reward_stats:
                    self.current_metrics.reward_mean = reward_stats.get('mean', 0.0)
                    self.current_metrics.reward_std = reward_stats.get('std', 0.0)
                    self.current_metrics.reward_min = reward_stats.get('min', 0.0)
                    self.current_metrics.reward_max = reward_stats.get('max', 0.0)

            if hasattr(trainer, 'kl_stats'):
                kl_stats = trainer.kl_stats
                if kl_stats:
                    self.current_metrics.kl_mean = kl_stats.get('mean', 0.0)
                    self.current_metrics.kl_std = kl_stats.get('std', 0.0)
                    self.current_metrics.kl_divergence = kl_stats.get('divergence', 0.0)

            if hasattr(trainer, 'entropy_stats'):
                entropy_stats = trainer.entropy_stats
                if entropy_stats:
                    self.current_metrics.entropy_mean = entropy_stats.get('mean', 0.0)
                    self.current_metrics.entropy_std = entropy_stats.get('std', 0.0)

            # Calculate training health indicators
            self._calculate_health_indicators()

        except Exception as e:
            warnings.warn(f"Failed to collect training metrics: {e}", stacklevel=2)

    def _collect_resource_metrics(self):
        """Collect system resource metrics."""
        try:
            # GPU metrics
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                self.current_metrics.gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3  # GB
                self.current_metrics.gpu_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
                self.current_metrics.gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB

                # GPU utilization (approximate)
                if hasattr(torch.cuda, 'utilization'):
                    self.current_metrics.gpu_utilization = torch.cuda.utilization(device)

            # CPU metrics
            self.current_metrics.cpu_memory_used = psutil.virtual_memory().used / 1024**3  # GB
            self.current_metrics.cpu_utilization = psutil.cpu_percent()

        except Exception as e:
            warnings.warn(f"Failed to collect resource metrics: {e}", stacklevel=2)

    def _calculate_health_indicators(self):
        """Calculate training health indicators."""
        if len(self.metrics_history) < 10:
            return

        # Calculate trends over last 10 steps
        recent_metrics = self.metrics_history[-10:]

        # Reward trend
        rewards = [m.reward_mean for m in recent_metrics if m.reward_mean != 0]
        if len(rewards) > 1:
            self.current_metrics.reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]

        # KL trend
        kls = [m.kl_mean for m in recent_metrics if m.kl_mean != 0]
        if len(kls) > 1:
            self.current_metrics.kl_trend = np.polyfit(range(len(kls)), kls, 1)[0]

        # Training stability score (based on loss variance)
        losses = [m.loss for m in recent_metrics if m.loss != 0]
        if len(losses) > 1:
            loss_std = np.std(losses)
            loss_mean = np.mean(losses)
            if loss_mean > 0:
                self.current_metrics.training_stability_score = max(0, 1 - (loss_std / loss_mean))

        # Convergence indicator (based on reward improvement)
        if len(rewards) > 5:
            early_rewards = rewards[:5]
            late_rewards = rewards[-5:]
            if np.mean(early_rewards) > 0:
                improvement = (np.mean(late_rewards) - np.mean(early_rewards)) / np.mean(early_rewards)
                self.current_metrics.convergence_indicator = improvement

    def _check_alerts(self):
        """Check for alert conditions."""
        for threshold_name, threshold_value in self.alert_thresholds.items():
            current_value = getattr(self.current_metrics, threshold_name, None)
            if current_value is not None and current_value > threshold_value:
                self._trigger_alert(threshold_name, current_value, threshold_value)

    def _trigger_alert(self, metric_name: str, current_value: float, threshold: float):
        """Trigger an alert for a metric threshold breach."""
        alert_data = {
            "metric": metric_name,
            "current_value": current_value,
            "threshold": threshold,
            "step": self.current_metrics.step,
            "timestamp": time.time(),
        }

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}", stacklevel=2)

        # Log alert
        self._log_event("alert_triggered", alert_data)

    def _start_monitoring_thread(self):
        """Start background monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return

        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Process queued metrics
                while not self.metrics_queue.empty():
                    try:
                        metrics = self.metrics_queue.get_nowait()
                        self._process_queued_metrics(metrics)
                    except queue.Empty:
                        break

                # Collect additional metrics
                if self.enable_distributed_monitoring:
                    self._collect_distributed_metrics()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                warnings.warn(f"Monitoring loop error: {e}", stacklevel=2)
                time.sleep(5.0)

    def _process_queued_metrics(self, metrics: Dict[str, Any]):
        """Process metrics from the queue."""
        # Update current metrics with queued data
        for key, value in metrics.items():
            if hasattr(self.current_metrics, key):
                setattr(self.current_metrics, key, value)

    def _collect_distributed_metrics(self):
        """Collect distributed training metrics."""
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return

            # Collect network metrics (approximate)
            start_time = time.time()
            # This would need actual network monitoring implementation
            self.current_metrics.allreduce_time = time.time() - start_time

        except Exception as e:
            warnings.warn(f"Failed to collect distributed metrics: {e}", stacklevel=2)

    def _log_detailed_metrics(self, step: int):
        """Log detailed metrics at specified intervals."""
        metrics_dict = self.current_metrics.to_dict()
        metrics_dict['timestamp'] = time.time()

        # Save to JSONL file
        log_file = self.output_dir / f"metrics_{self.run_id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event."""
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "run_id": self.run_id,
            **data
        }

        # Save to events log
        events_file = self.output_dir / f"events_{self.run_id}.jsonl"
        with open(events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def _save_metrics(self):
        """Save all collected metrics to files."""
        if not self.metrics_history:
            return

        # Convert to DataFrame
        metrics_data = [m.to_dict() for m in self.metrics_history]
        df = pd.DataFrame(metrics_data)

        # Save as CSV (aggregates only)
        csv_file = self.output_dir / f"metrics_{self.run_id}.csv"
        df.to_csv(csv_file, index=False)

        # Save as Parquet for better performance (with fallback)
        parquet_file = self.output_dir / f"metrics_{self.run_id}.parquet"
        try:
            df.to_parquet(parquet_file, index=False)
        except ImportError as e:
            warnings.warn(f"Parquet export failed (missing engine): {e}. Install pyarrow or fastparquet for Parquet support.", stacklevel=2)
            # Fallback to JSON for better performance than CSV for large datasets
            json_file = self.output_dir / f"metrics_{self.run_id}.json"
            df.to_json(json_file, orient='records', indent=2)

        # Note: JSONL file is handled separately and should not be modified here
        # The JSONL file contains per-step events and is written in real-time

        # Save summary statistics
        summary = {
            "run_id": self.run_id,
            "total_steps": len(self.metrics_history),
            "total_time": self.metrics_history[-1].wall_time - self.metrics_history[0].wall_time if len(self.metrics_history) > 1 else 0,
            "final_loss": self.metrics_history[-1].loss,
            "final_reward": self.metrics_history[-1].reward_mean,
            "final_kl": self.metrics_history[-1].kl_mean,
            "max_gpu_memory": max(m.gpu_memory_used for m in self.metrics_history),
            "avg_step_time": np.mean(self.step_times) if self.step_times else 0,
        }

        summary_file = self.output_dir / f"summary_{self.run_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all collected metrics as a DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()

        metrics_data = [m.to_dict() for m in self.metrics_history]
        return pd.DataFrame(metrics_data)

    def get_latest_metrics(self) -> OpenRLHFMetrics:
        """Get the latest collected metrics."""
        return self.current_metrics

    def _setup_jsonl_logging(self):
        """Setup JSONL logging file."""
        if not EVENT_SCHEMA_AVAILABLE:
            warnings.warn("Event schema not available, JSONL logging disabled", stacklevel=2)
            self.enable_jsonl_logging = False
            return

        # Safety check: ensure run_id is available
        if not hasattr(self, 'run_id') or self.run_id is None:
            warnings.warn("run_id not available, JSONL logging disabled", stacklevel=2)
            self.enable_jsonl_logging = False
            return

        jsonl_path = self.output_dir / f"{self.run_id}_events.jsonl"
        self.jsonl_file = open(jsonl_path, "w")
        print(f"📝 JSONL events will be written to: {jsonl_path}")

    def _log_jsonl_event(self, step: int, logs: Dict[str, float]):
        """Log a standardized JSONL event with Event schema structure."""
        if not self.jsonl_file or not EVENT_SCHEMA_AVAILABLE:
            return

        try:
            # Create event data with required fields for Event schema
            event_data = {
                "step": step,
                "timestamp": time.time(),
                "phase": "train",
                "wall_time": self.current_metrics.wall_time,
                "reward_mean": self.current_metrics.reward_mean,
                "reward_std": self.current_metrics.reward_std,
                "kl_mean": self.current_metrics.kl_mean,
                "kl_std": self.current_metrics.kl_std,
                "entropy_mean": self.current_metrics.entropy_mean,
                "entropy_std": self.current_metrics.entropy_std,
                "clip_frac": self.current_metrics.clip_frac,
                "grad_norm": self.current_metrics.grad_norm,
                "lr": self.current_metrics.learning_rate,
                "loss": self.current_metrics.loss,
                "policy_loss": self.current_metrics.policy_loss,
                "value_loss": self.current_metrics.value_loss,
                "tokens_in": self.current_metrics.tokens_in,
                "tokens_out": self.current_metrics.tokens_out,
                "seed": self.current_metrics.seed,
                "run_id": self.current_metrics.run_id,
                "git_sha": self.current_metrics.git_sha,
                "step_time": self.current_metrics.step_time,
                "gpu_memory_used": self.current_metrics.gpu_memory_used,
                "gpu_memory_allocated": self.current_metrics.gpu_memory_allocated,
                "cpu_memory_used": self.current_metrics.cpu_memory_used,
                "training_stability_score": self.current_metrics.training_stability_score,
                "convergence_indicator": self.current_metrics.convergence_indicator,
                # Network metrics
                "network_bandwidth": self.current_metrics.network_bandwidth,
                "network_latency": self.current_metrics.network_latency,
                "bandwidth_mbps": self.current_metrics.bandwidth_mbps,
                "latency_ms": self.current_metrics.latency_ms,
                "bandwidth_upload_mbps": self.current_metrics.bandwidth_upload_mbps,
                "bandwidth_download_mbps": self.current_metrics.bandwidth_download_mbps,
                "total_bandwidth_mbps": self.current_metrics.total_bandwidth_mbps,
                "allreduce_bandwidth": self.current_metrics.allreduce_bandwidth,
                "broadcast_bandwidth": self.current_metrics.broadcast_bandwidth,
                "gather_bandwidth": self.current_metrics.gather_bandwidth,
                "scatter_bandwidth": self.current_metrics.scatter_bandwidth,
                "packet_loss_percent": self.current_metrics.packet_loss_percent,
                "network_errors": self.current_metrics.network_errors,
                "dns_resolution_ms": self.current_metrics.dns_resolution_ms,
            }

            # Create Event object using the schema
            event = create_event_from_row(event_data, self.run_id, self.current_metrics.git_sha)

            # Write JSONL line with proper formatting
            json_line = event.to_json()
            self.jsonl_file.write(json_line + "\n")
            self.jsonl_file.flush()  # Ensure immediate write

        except Exception as e:
            warnings.warn(f"Failed to log JSONL event: {e}", stacklevel=2)

    def _close_jsonl_file(self):
        """Close the JSONL file."""
        if self.jsonl_file:
            self.jsonl_file.close()
            self.jsonl_file = None


class OpenRLHFMonitor(OpenRLHFCallback):
    """Simplified OpenRLHF monitor for easy integration."""

    def __init__(self, **kwargs):
        """Initialize with sensible defaults."""
        super().__init__(
            log_interval=kwargs.get('log_interval', 10),
            enable_resource_monitoring=kwargs.get('enable_resource_monitoring', True),
            enable_distributed_monitoring=kwargs.get('enable_distributed_monitoring', True),
            **kwargs
        )


class DistributedTrainingMonitor(OpenRLHFCallback):
    """Specialized monitor for distributed OpenRLHF training."""

    def __init__(self, **kwargs):
        """Initialize distributed training monitor."""
        # Remove duplicate parameters from kwargs
        kwargs.pop('enable_distributed_monitoring', None)
        kwargs.pop('enable_resource_monitoring', None)

        super().__init__(
            enable_distributed_monitoring=True,
            enable_resource_monitoring=True,
            **kwargs
        )

        # Additional distributed-specific settings
        self.sync_interval = kwargs.get('sync_interval', 5)  # Steps between sync
        self.network_monitoring = kwargs.get('network_monitoring', True)

    def _collect_distributed_metrics(self):
        """Enhanced distributed metrics collection."""
        super()._collect_distributed_metrics()

        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return

            # Collect additional distributed metrics
            if self.network_monitoring:
                self._collect_network_metrics()

        except Exception as e:
            warnings.warn(f"Failed to collect enhanced distributed metrics: {e}", stacklevel=2)

    def _collect_network_metrics(self):
        """Collect network performance metrics using real measurements."""
        # Only collect network metrics at the specified sampling frequency
        if not hasattr(self, '_last_network_collection_step'):
            self._last_network_collection_step = 0

        if self.current_metrics.step - self._last_network_collection_step < self.network_sampling_frequency:
            return

        self._last_network_collection_step = self.current_metrics.step

        try:
            import threading

            from .network_monitor import RealNetworkMonitor

            # Thread-safe initialization of network monitor with proper lock initialization
            if not hasattr(self, '_network_monitor_lock'):
                # Use a class-level lock to ensure atomic lock initialization
                if not hasattr(self.__class__, '_class_lock'):
                    self.__class__._class_lock = threading.Lock()

                with self.__class__._class_lock:
                    if not hasattr(self, '_network_monitor_lock'):
                        self._network_monitor_lock = threading.Lock()

            with self._network_monitor_lock:
                if not hasattr(self, '_network_monitor'):
                    self._network_monitor = RealNetworkMonitor(
                        enable_distributed_monitoring=True,
                        enable_distributed_measurements=False  # Safer default - doesn't interfere with training
                    )

            # Get comprehensive network metrics
            network_metrics = self._network_monitor.get_comprehensive_metrics()

            # Update current metrics with real measurements
            self.current_metrics.bandwidth_mbps = network_metrics.bandwidth_mbps
            self.current_metrics.latency_ms = network_metrics.latency_ms
            self.current_metrics.bandwidth_upload_mbps = network_metrics.bandwidth_out_mbps
            self.current_metrics.bandwidth_download_mbps = network_metrics.bandwidth_in_mbps
            self.current_metrics.total_bandwidth_mbps = network_metrics.bandwidth_in_mbps + network_metrics.bandwidth_out_mbps

            # Update legacy attributes for backward compatibility
            self.current_metrics.network_bandwidth = network_metrics.bandwidth_mbps
            self.current_metrics.network_latency = network_metrics.latency_ms

            # Add additional network metrics
            self.current_metrics.allreduce_bandwidth = network_metrics.allreduce_bandwidth
            self.current_metrics.broadcast_bandwidth = network_metrics.broadcast_bandwidth
            self.current_metrics.gather_bandwidth = network_metrics.gather_bandwidth
            self.current_metrics.scatter_bandwidth = network_metrics.scatter_bandwidth
            self.current_metrics.packet_loss_percent = network_metrics.packet_loss_percent
            self.current_metrics.network_errors = network_metrics.network_errors
            self.current_metrics.dns_resolution_ms = network_metrics.dns_resolution_ms

        except Exception as e:
            print(f"Error collecting network metrics: {e}")
            # Try to get basic network diagnostics as fallback
            try:
                from .network_monitor import NetworkDiagnostics
                diagnostics = NetworkDiagnostics()

                # Get basic ping and DNS diagnostics
                ping_tests = diagnostics._run_ping_diagnostics()
                dns_tests = diagnostics._run_dns_diagnostics()

                # Calculate average latency from successful ping tests
                successful_pings = [result['latency'] for result in ping_tests.values()
                                 if result.get('success', False) and result['latency'] != float('inf')]
                avg_latency = np.mean(successful_pings) if successful_pings else 0.0

                # Get DNS resolution time
                successful_dns = [result['resolution_time_ms'] for result in dns_tests.values()
                                if result.get('success', False)]
                avg_dns_time = np.mean(successful_dns) if successful_dns else 0.0

                # Set fallback values
                self.current_metrics.bandwidth_mbps = 0.0
                self.current_metrics.latency_ms = avg_latency
                self.current_metrics.bandwidth_upload_mbps = 0.0
                self.current_metrics.bandwidth_download_mbps = 0.0
                self.current_metrics.total_bandwidth_mbps = 0.0
                self.current_metrics.dns_resolution_ms = avg_dns_time

                # Update legacy attributes for backward compatibility
                self.current_metrics.network_bandwidth = 0.0
                self.current_metrics.network_latency = avg_latency

            except Exception as fallback_error:
                print(f"Fallback network diagnostics also failed: {fallback_error}")
                # Final fallback to zero values
                self.current_metrics.bandwidth_mbps = 0.0
                self.current_metrics.latency_ms = 0.0
                self.current_metrics.bandwidth_upload_mbps = 0.0
                self.current_metrics.bandwidth_download_mbps = 0.0
                self.current_metrics.total_bandwidth_mbps = 0.0

                # Update legacy attributes for backward compatibility
                self.current_metrics.network_bandwidth = 0.0
                self.current_metrics.network_latency = 0.0


class MultiGPUMonitor(OpenRLHFCallback):
    """Specialized monitor for multi-GPU OpenRLHF training."""

    def __init__(self, **kwargs):
        """Initialize multi-GPU monitor."""
        super().__init__(
            enable_resource_monitoring=True,
            enable_distributed_monitoring=True,
            **kwargs
        )

        # Safe GPU count calculation
        self.gpu_count = 0
        if torch.cuda.is_available():
            try:
                self.gpu_count = torch.cuda.device_count()
            except Exception:
                self.gpu_count = 0

        self.gpu_metrics = {}

    def _collect_resource_metrics(self):
        """Collect metrics for all GPUs."""
        super()._collect_resource_metrics()

        if not torch.cuda.is_available():
            return

        # Collect metrics for each GPU
        for device_id in range(self.gpu_count):
            try:
                gpu_metrics = {
                    'memory_used': torch.cuda.memory_allocated(device_id) / 1024**3,
                    'memory_allocated': torch.cuda.max_memory_allocated(device_id) / 1024**3,
                    'memory_reserved': torch.cuda.memory_reserved(device_id) / 1024**3,
                }
                self.gpu_metrics[f'gpu_{device_id}'] = gpu_metrics

            except Exception as e:
                warnings.warn(f"Failed to collect metrics for GPU {device_id}: {e}", stacklevel=2)

    def get_gpu_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all GPUs."""
        return self.gpu_metrics.copy()
