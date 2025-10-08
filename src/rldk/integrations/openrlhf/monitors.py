"""Specialized monitors for OpenRLHF training analytics."""

import json
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ...utils.torch_compat import safe_torch_load
from .callbacks import OpenRLHFMetrics
from .distributed import GPUMemoryMonitor


@dataclass
class TrainingHealthMetrics:
    """Metrics for training health analysis."""
    stability_score: float = 1.0
    convergence_rate: float = 0.0
    reward_trend: float = 0.0
    kl_trend: float = 0.0
    loss_variance: float = 0.0
    gradient_norm_trend: float = 0.0
    learning_rate_effectiveness: float = 0.0
    memory_efficiency: float = 1.0
    training_speed: float = 0.0
    anomaly_score: float = 0.0


@dataclass
class CheckpointMetrics:
    """Metrics for checkpoint analysis."""
    checkpoint_path: str = ""
    step: int = 0
    loss: float = 0.0
    reward_mean: float = 0.0
    kl_mean: float = 0.0
    model_size: float = 0.0
    checkpoint_time: float = 0.0
    validation_score: float = 0.0
    training_time: float = 0.0
    memory_usage: float = 0.0


class OpenRLHFTrainingMonitor:
    """Advanced training monitor for OpenRLHF with real-time analytics."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        analysis_window: int = 100,
        enable_anomaly_detection: bool = True,
        enable_convergence_analysis: bool = True,
        enable_performance_analysis: bool = True,
    ):
        """Initialize OpenRLHF training monitor.

        Args:
            output_dir: Directory to save analysis results
            analysis_window: Number of recent steps to analyze
            enable_anomaly_detection: Whether to detect training anomalies
            enable_convergence_analysis: Whether to analyze convergence
            enable_performance_analysis: Whether to analyze performance
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analysis_window = analysis_window
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_convergence_analysis = enable_convergence_analysis
        self.enable_performance_analysis = enable_performance_analysis

        # Metrics storage
        self.metrics_history: List[OpenRLHFMetrics] = []
        self.health_metrics: List[TrainingHealthMetrics] = []

        # Analysis components
        self.anomaly_detector = None
        if self.enable_anomaly_detection:
            self.anomaly_detector = TrainingAnomalyDetector()

        self.convergence_analyzer = None
        if self.enable_convergence_analysis:
            self.convergence_analyzer = ConvergenceAnalyzer()

        self.performance_analyzer = None
        if self.enable_performance_analysis:
            self.performance_analyzer = PerformanceAnalyzer()

        # Real-time monitoring
        self.monitoring_active = False
        self.analysis_thread = None

    def add_metrics(self, metrics: OpenRLHFMetrics):
        """Add new metrics for analysis."""
        self.metrics_history.append(metrics)

        # Keep only recent metrics for analysis
        if len(self.metrics_history) > self.analysis_window * 2:
            self.metrics_history = self.metrics_history[-self.analysis_window:]

        # Perform real-time analysis
        if len(self.metrics_history) >= 10:  # Need minimum data for analysis
            self._analyze_metrics()

    def _analyze_metrics(self):
        """Perform real-time analysis on collected metrics."""
        if len(self.metrics_history) < 10:
            return

        # Get recent metrics for analysis
        recent_metrics = self.metrics_history[-self.analysis_window:]

        # Calculate health metrics
        health_metrics = self._calculate_health_metrics(recent_metrics)
        self.health_metrics.append(health_metrics)

        # Keep only recent health metrics
        if len(self.health_metrics) > self.analysis_window:
            self.health_metrics = self.health_metrics[-self.analysis_window:]

        # Perform specialized analysis
        if self.anomaly_detector:
            anomaly_score = self.anomaly_detector.detect_anomalies(recent_metrics)
            health_metrics.anomaly_score = anomaly_score

        if self.convergence_analyzer:
            convergence_rate = self.convergence_analyzer.analyze_convergence(recent_metrics)
            health_metrics.convergence_rate = convergence_rate

        if self.performance_analyzer:
            performance_metrics = self.performance_analyzer.analyze_performance(recent_metrics)
            health_metrics.training_speed = performance_metrics.get('training_speed', 0.0)
            health_metrics.memory_efficiency = performance_metrics.get('memory_efficiency', 1.0)

    def _calculate_health_metrics(self, metrics: List[OpenRLHFMetrics]) -> TrainingHealthMetrics:
        """Calculate training health metrics."""
        if len(metrics) < 2:
            return TrainingHealthMetrics()

        # Extract metric arrays
        losses = [m.loss for m in metrics if m.loss != 0]
        rewards = [m.reward_mean for m in metrics if m.reward_mean != 0]
        kls = [m.kl_mean for m in metrics if m.kl_mean != 0]
        grad_norms = [m.grad_norm for m in metrics if m.grad_norm != 0]
        step_times = [m.step_time for m in metrics if m.step_time != 0]

        # Calculate stability score (inverse of loss variance)
        loss_variance = np.var(losses) if len(losses) > 1 else 0.0
        loss_mean = np.mean(losses) if losses else 0.0
        stability_score = max(0, 1 - (loss_variance / (loss_mean + 1e-8)))

        # Calculate trends
        reward_trend = 0.0
        if len(rewards) > 1:
            reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]

        kl_trend = 0.0
        if len(kls) > 1:
            kl_trend = np.polyfit(range(len(kls)), kls, 1)[0]

        gradient_norm_trend = 0.0
        if len(grad_norms) > 1:
            gradient_norm_trend = np.polyfit(range(len(grad_norms)), grad_norms, 1)[0]

        # Calculate training speed (steps per second)
        training_speed = 0.0
        if step_times:
            avg_step_time = np.mean(step_times)
            if avg_step_time > 0:
                training_speed = 1.0 / avg_step_time

        # Calculate memory efficiency (based on GPU memory usage)
        memory_usage = []
        for m in metrics:
            if hasattr(m, 'gpu_memory_used') and m.gpu_memory_used > 0:
                memory_usage.append(m.gpu_memory_used)
        memory_efficiency = 1.0
        if memory_usage:
            # Efficiency based on memory usage stability
            memory_variance = np.var(memory_usage)
            memory_mean = np.mean(memory_usage)
            if memory_mean > 0:
                memory_efficiency = max(0, 1 - (memory_variance / memory_mean))

        # Calculate learning rate effectiveness
        learning_rate_effectiveness = 0.0
        if len(losses) > 5:
            # Check if loss is decreasing with current learning rate
            early_losses = losses[:len(losses)//2]
            late_losses = losses[len(losses)//2:]
            if np.mean(early_losses) > 0:
                improvement = (np.mean(early_losses) - np.mean(late_losses)) / np.mean(early_losses)
                learning_rate_effectiveness = max(0, improvement)

        return TrainingHealthMetrics(
            stability_score=stability_score,
            convergence_rate=0.0,  # Will be set by convergence analyzer
            reward_trend=reward_trend,
            kl_trend=kl_trend,
            loss_variance=loss_variance,
            gradient_norm_trend=gradient_norm_trend,
            learning_rate_effectiveness=learning_rate_effectiveness,
            memory_efficiency=memory_efficiency,
            training_speed=training_speed,
            anomaly_score=0.0,  # Will be set by anomaly detector
        )

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of training health."""
        if not self.health_metrics:
            return {}

        latest_health = self.health_metrics[-1]

        return {
            'stability_score': latest_health.stability_score,
            'convergence_rate': latest_health.convergence_rate,
            'reward_trend': latest_health.reward_trend,
            'kl_trend': latest_health.kl_trend,
            'training_speed': latest_health.training_speed,
            'memory_efficiency': latest_health.memory_efficiency,
            'anomaly_score': latest_health.anomaly_score,
            'overall_health': self._calculate_overall_health(latest_health),
        }

    def _calculate_overall_health(self, health: TrainingHealthMetrics) -> float:
        """Calculate overall training health score."""
        # Weighted combination of health indicators
        weights = {
            'stability_score': 0.25,
            'convergence_rate': 0.20,
            'reward_trend': 0.15,
            'kl_trend': 0.10,
            'training_speed': 0.10,
            'memory_efficiency': 0.10,
            'anomaly_score': 0.10,  # Lower is better
        }

        # Normalize anomaly score (lower is better)
        normalized_anomaly = max(0, 1 - health.anomaly_score)

        # Normalize trends (positive reward trend is good, negative KL trend is good)
        normalized_reward_trend = max(0, min(1, (health.reward_trend + 1) / 2))
        normalized_kl_trend = max(0, min(1, (1 - health.kl_trend) / 2))

        overall_health = (
            weights['stability_score'] * health.stability_score +
            weights['convergence_rate'] * max(0, health.convergence_rate) +
            weights['reward_trend'] * normalized_reward_trend +
            weights['kl_trend'] * normalized_kl_trend +
            weights['training_speed'] * min(1, health.training_speed / 10) +  # Normalize speed
            weights['memory_efficiency'] * health.memory_efficiency +
            weights['anomaly_score'] * normalized_anomaly
        )

        return min(1.0, max(0.0, overall_health))

    def save_analysis(self, filename: Optional[str] = None):
        """Save analysis results to file."""
        if not filename:
            filename = f"training_analysis_{int(time.time())}.json"

        analysis_data = {
            'health_summary': self.get_health_summary(),
            'health_metrics': [h.__dict__ for h in self.health_metrics],
            'metrics_summary': self._get_metrics_summary(),
            'analysis_timestamp': time.time(),
        }

        analysis_file = self.output_dir / filename
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}

        latest_metrics = self.metrics_history[-1]

        return {
            'total_steps': len(self.metrics_history),
            'latest_step': latest_metrics.step,
            'latest_loss': latest_metrics.loss,
            'latest_reward': latest_metrics.reward_mean,
            'latest_kl': latest_metrics.kl_mean,
            'latest_gpu_memory': latest_metrics.gpu_memory_used,
            'avg_step_time': np.mean([m.step_time for m in self.metrics_history if m.step_time > 0]),
        }


class OpenRLHFCheckpointMonitor:
    """Monitor for checkpoint analysis and model health."""

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        enable_validation: bool = True,
        enable_size_analysis: bool = True,
    ):
        """Initialize checkpoint monitor.

        Args:
            checkpoint_dir: Directory containing checkpoints
            enable_validation: Whether to validate checkpoints
            enable_size_analysis: Whether to analyze checkpoint sizes
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.enable_validation = enable_validation
        self.enable_size_analysis = enable_size_analysis

        self.checkpoint_metrics: List[CheckpointMetrics] = []
        self.checkpoint_history: List[Dict[str, Any]] = []

    def analyze_checkpoint(self, checkpoint_path: Union[str, Path], step: int) -> CheckpointMetrics:
        """Analyze a checkpoint and return metrics."""
        checkpoint_path = Path(checkpoint_path)

        metrics = CheckpointMetrics(
            checkpoint_path=str(checkpoint_path),
            step=step,
            checkpoint_time=time.time(),
        )

        try:
            # Get checkpoint size
            if self.enable_size_analysis:
                metrics.model_size = self._get_checkpoint_size(checkpoint_path)

            # Load checkpoint and extract metrics
            checkpoint_data = self._load_checkpoint(checkpoint_path)
            if checkpoint_data:
                metrics.loss = checkpoint_data.get('loss', 0.0)
                metrics.reward_mean = checkpoint_data.get('reward_mean', 0.0)
                metrics.kl_mean = checkpoint_data.get('kl_mean', 0.0)
                metrics.training_time = checkpoint_data.get('training_time', 0.0)
                metrics.memory_usage = checkpoint_data.get('memory_usage', 0.0)

            # Validate checkpoint if enabled
            if self.enable_validation:
                metrics.validation_score = self._validate_checkpoint(checkpoint_path)

        except Exception as e:
            warnings.warn(f"Failed to analyze checkpoint {checkpoint_path}: {e}", stacklevel=2)

        # Store metrics
        self.checkpoint_metrics.append(metrics)
        self.checkpoint_history.append(metrics.__dict__)

        return metrics

    def _get_checkpoint_size(self, checkpoint_path: Path) -> float:
        """Get checkpoint size in GB."""
        try:
            total_size = 0
            if checkpoint_path.is_file():
                total_size = checkpoint_path.stat().st_size
            elif checkpoint_path.is_dir():
                for file_path in checkpoint_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

            return total_size / 1024**3  # Convert to GB
        except Exception:
            return 0.0

    def _load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Load checkpoint and extract metadata."""
        try:
            # Try to load as PyTorch checkpoint
            if checkpoint_path.suffix == '.pt' or checkpoint_path.suffix == '.pth':
                checkpoint = safe_torch_load(checkpoint_path, map_location='cpu')
                return checkpoint.get('metadata', {})

            # Try to load as JSON metadata
            metadata_file = checkpoint_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    return json.load(f)

            return None
        except Exception:
            return None

    def _validate_checkpoint(self, checkpoint_path: Path) -> float:
        """Validate checkpoint integrity and return score."""
        try:
            # Basic validation - check if checkpoint can be loaded
            if checkpoint_path.suffix in ['.pt', '.pth']:
                checkpoint = safe_torch_load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    return 1.0  # Valid checkpoint
                else:
                    return 0.5  # Partially valid
            else:
                return 0.0  # Unknown format
        except Exception:
            return 0.0  # Invalid checkpoint

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoint analysis."""
        if not self.checkpoint_metrics:
            return {}

        latest_checkpoint = self.checkpoint_metrics[-1]

        return {
            'total_checkpoints': len(self.checkpoint_metrics),
            'latest_step': latest_checkpoint.step,
            'latest_loss': latest_checkpoint.loss,
            'latest_reward': latest_checkpoint.reward_mean,
            'latest_kl': latest_checkpoint.kl_mean,
            'latest_size': latest_checkpoint.model_size,
            'latest_validation_score': latest_checkpoint.validation_score,
            'avg_checkpoint_size': np.mean([c.model_size for c in self.checkpoint_metrics]),
            'avg_validation_score': np.mean([c.validation_score for c in self.checkpoint_metrics]),
        }


class OpenRLHFResourceMonitor:
    """Monitor for resource usage during OpenRLHF training."""

    def __init__(self, monitor_interval: float = 1.0):
        """Initialize resource monitor.

        Args:
            monitor_interval: Interval between resource measurements (seconds)
        """
        self.monitor_interval = monitor_interval
        self.monitoring_active = False
        self.monitor_thread = None

        # Resource metrics storage
        self.resource_history: List[Dict[str, Any]] = []
        self.gpu_monitor = GPUMemoryMonitor()

        # Resource usage tracking
        self.peak_memory_usage = 0.0
        self.peak_cpu_usage = 0.0
        self.total_energy_consumption = 0.0  # Placeholder

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                resource_metrics = self._collect_resource_metrics()
                self.resource_history.append(resource_metrics)

                # Update peak usage
                self.peak_memory_usage = max(self.peak_memory_usage, resource_metrics['gpu_memory_used'])
                self.peak_cpu_usage = max(self.peak_cpu_usage, resource_metrics['cpu_utilization'])

                time.sleep(self.monitor_interval)

            except Exception as e:
                warnings.warn(f"Error in resource monitoring: {e}", stacklevel=2)
                time.sleep(self.monitor_interval)

    def _collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        import psutil

        metrics = {
            'timestamp': time.time(),
            'cpu_utilization': psutil.cpu_percent(),
            'memory_used': psutil.virtual_memory().used / 1024**3,  # GB
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
        }

        # GPU metrics
        if torch.cuda.is_available():
            gpu_metrics = self.gpu_monitor.get_current_memory_usage()
            if gpu_metrics:
                # Sum across all GPUs
                total_gpu_memory = sum(gpu['used'] for gpu in gpu_metrics.values())
                metrics['gpu_memory_used'] = total_gpu_memory
                metrics['gpu_count'] = len(gpu_metrics)
            else:
                metrics['gpu_memory_used'] = 0.0
                metrics['gpu_count'] = 0
        else:
            metrics['gpu_memory_used'] = 0.0
            metrics['gpu_count'] = 0

        return metrics

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        if not self.resource_history:
            return {}

        # Calculate statistics
        cpu_utilizations = [r['cpu_utilization'] for r in self.resource_history]
        memory_usage = [r['memory_used'] for r in self.resource_history]
        gpu_memory_usage = [r['gpu_memory_used'] for r in self.resource_history]

        return {
            'monitoring_duration': len(self.resource_history) * self.monitor_interval,
            'avg_cpu_utilization': np.mean(cpu_utilizations),
            'max_cpu_utilization': np.max(cpu_utilizations),
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage),
            'avg_gpu_memory_usage': np.mean(gpu_memory_usage),
            'max_gpu_memory_usage': np.max(gpu_memory_usage),
            'peak_memory_usage': self.peak_memory_usage,
            'peak_cpu_usage': self.peak_cpu_usage,
            'total_measurements': len(self.resource_history),
        }


class OpenRLHFAnalytics:
    """Comprehensive analytics for OpenRLHF training."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize OpenRLHF analytics.

        Args:
            output_dir: Directory to save analytics results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_analytics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.training_monitor = OpenRLHFTrainingMonitor(output_dir=self.output_dir)
        self.checkpoint_monitor = OpenRLHFCheckpointMonitor()
        self.resource_monitor = OpenRLHFResourceMonitor()

        # Analytics results
        self.analytics_results: Dict[str, Any] = {}

    def analyze_training_run(self, metrics_history: List[OpenRLHFMetrics]) -> Dict[str, Any]:
        """Perform comprehensive analysis of a training run."""
        # Add metrics to training monitor
        for metrics in metrics_history:
            self.training_monitor.add_metrics(metrics)

        # Perform analysis
        analysis_results = {
            'training_health': self.training_monitor.get_health_summary(),
            'metrics_summary': self.training_monitor._get_metrics_summary(),
            'resource_summary': self.resource_monitor.get_resource_summary(),
            'checkpoint_summary': self.checkpoint_monitor.get_checkpoint_summary(),
            'analytics_timestamp': time.time(),
        }

        # Store results
        self.analytics_results = analysis_results

        # Save analysis
        self._save_analytics()

        return analysis_results

    def _save_analytics(self):
        """Save analytics results to file."""
        if not self.analytics_results:
            return

        timestamp = int(time.time())
        analytics_file = self.output_dir / f"analytics_{timestamp}.json"

        with open(analytics_file, 'w') as f:
            json.dump(self.analytics_results, f, indent=2)

        # Also save training monitor analysis
        self.training_monitor.save_analysis(f"training_analysis_{timestamp}.json")


# Helper classes for specialized analysis

class TrainingAnomalyDetector:
    """Detect anomalies in training metrics."""

    def __init__(self, threshold: float = 2.0):
        """Initialize anomaly detector.

        Args:
            threshold: Standard deviation threshold for anomaly detection
        """
        self.threshold = threshold

    def detect_anomalies(self, metrics: List[OpenRLHFMetrics]) -> float:
        """Detect anomalies in training metrics."""
        if len(metrics) < 10:
            return 0.0

        # Extract key metrics
        losses = [m.loss for m in metrics if m.loss != 0]
        rewards = [m.reward_mean for m in metrics if m.reward_mean != 0]
        kls = [m.kl_mean for m in metrics if m.kl_mean != 0]

        anomaly_scores = []

        # Check for anomalies in loss
        if len(losses) > 5:
            loss_anomaly = self._detect_metric_anomaly(losses)
            anomaly_scores.append(loss_anomaly)

        # Check for anomalies in reward
        if len(rewards) > 5:
            reward_anomaly = self._detect_metric_anomaly(rewards)
            anomaly_scores.append(reward_anomaly)

        # Check for anomalies in KL divergence
        if len(kls) > 5:
            kl_anomaly = self._detect_metric_anomaly(kls)
            anomaly_scores.append(kl_anomaly)

        # Return average anomaly score
        return np.mean(anomaly_scores) if anomaly_scores else 0.0

    def _detect_metric_anomaly(self, values: List[float]) -> float:
        """Detect anomalies in a single metric."""
        if len(values) < 5:
            return 0.0

        # Calculate z-scores
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return 0.0

        z_scores = [abs((val - mean_val) / std_val) for val in values]

        # Count anomalies
        anomalies = sum(1 for z in z_scores if z > self.threshold)

        # Return anomaly ratio
        return anomalies / len(values)


class ConvergenceAnalyzer:
    """Analyze training convergence."""

    def __init__(self, convergence_threshold: float = 0.01):
        """Initialize convergence analyzer.

        Args:
            convergence_threshold: Threshold for considering convergence
        """
        self.convergence_threshold = convergence_threshold

    def analyze_convergence(self, metrics: List[OpenRLHFMetrics]) -> float:
        """Analyze convergence rate."""
        if len(metrics) < 20:
            return 0.0

        # Extract reward values
        rewards = [m.reward_mean for m in metrics if m.reward_mean != 0]

        if len(rewards) < 10:
            return 0.0

        # Calculate convergence rate based on reward improvement
        early_rewards = rewards[:len(rewards)//3]
        late_rewards = rewards[-len(rewards)//3:]

        if not early_rewards or not late_rewards:
            return 0.0

        early_mean = np.mean(early_rewards)
        late_mean = np.mean(late_rewards)

        if early_mean == 0:
            return 0.0

        improvement_rate = (late_mean - early_mean) / abs(early_mean)

        # Normalize to 0-1 range
        return max(0, min(1, improvement_rate))


class PerformanceAnalyzer:
    """Analyze training performance."""

    def __init__(
        self,
        min_steps_for_analysis: int = 5,
        speed_threshold: float = 0.1,
        memory_efficiency_threshold: float = 0.8,
        gpu_utilization_threshold: float = 0.7,
        enable_detailed_analysis: bool = True,
        performance_window: int = 50
    ):
        """Initialize performance analyzer.

        Args:
            min_steps_for_analysis: Minimum number of steps required for analysis
            speed_threshold: Threshold for considering training speed acceptable (steps/sec)
            memory_efficiency_threshold: Threshold for memory efficiency (0-1)
            gpu_utilization_threshold: Threshold for GPU utilization (0-1)
            enable_detailed_analysis: Whether to perform detailed performance analysis
            performance_window: Number of recent steps to analyze for performance trends
        """
        self.min_steps_for_analysis = min_steps_for_analysis
        self.speed_threshold = speed_threshold
        self.memory_efficiency_threshold = memory_efficiency_threshold
        self.gpu_utilization_threshold = gpu_utilization_threshold
        self.enable_detailed_analysis = enable_detailed_analysis
        self.performance_window = performance_window

        # Internal state for tracking performance trends
        self.performance_history: List[Dict[str, float]] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.performance_degradation_count = 0
        self.last_analysis_step = 0

    def analyze_performance(self, metrics: List[OpenRLHFMetrics]) -> Dict[str, float]:
        """Analyze training performance."""
        if len(metrics) < self.min_steps_for_analysis:
            return {
                'training_speed': 0.0,
                'memory_efficiency': 1.0,
                'gpu_utilization': 0.0,
                'performance_score': 0.0,
                'performance_status': 'insufficient_data'
            }

        # Get recent metrics for analysis
        recent_metrics = metrics[-self.performance_window:] if len(metrics) > self.performance_window else metrics

        # Calculate basic performance metrics
        training_speed = self._calculate_training_speed(recent_metrics)
        memory_efficiency = self._calculate_memory_efficiency(recent_metrics)
        gpu_utilization = self._calculate_gpu_utilization(recent_metrics)

        # Calculate detailed performance metrics if enabled
        detailed_metrics = {}
        if self.enable_detailed_analysis:
            detailed_metrics = self._calculate_detailed_metrics(recent_metrics)

        # Update performance history
        current_metrics = {
            'training_speed': training_speed,
            'memory_efficiency': memory_efficiency,
            'gpu_utilization': gpu_utilization,
            **detailed_metrics
        }
        self.performance_history.append(current_metrics)

        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history = self.performance_history[-self.performance_window:]

        # Calculate overall performance score
        performance_score = self._calculate_performance_score(current_metrics)

        # Add performance_score to current_metrics for status determination
        current_metrics['performance_score'] = performance_score

        # Determine performance status
        performance_status = self._determine_performance_status(current_metrics)

        # Update baseline if not set
        if self.baseline_metrics is None and len(self.performance_history) >= 10:
            self.baseline_metrics = self._calculate_baseline_metrics()

        return {
            'training_speed': training_speed,
            'memory_efficiency': memory_efficiency,
            'gpu_utilization': gpu_utilization,
            'performance_score': performance_score,
            'performance_status': performance_status,
            **detailed_metrics
        }

    def _calculate_training_speed(self, metrics: List[OpenRLHFMetrics]) -> float:
        """Calculate training speed in steps per second."""
        step_times = []
        for m in metrics:
            if hasattr(m, 'step_time') and m.step_time > 0:
                step_times.append(m.step_time)
        if not step_times:
            return 0.0
        return 1.0 / np.mean(step_times)

    def _calculate_memory_efficiency(self, metrics: List[OpenRLHFMetrics]) -> float:
        """Calculate memory efficiency based on variance in memory usage."""
        memory_usage = []
        for m in metrics:
            if hasattr(m, 'gpu_memory_used') and m.gpu_memory_used > 0:
                memory_usage.append(m.gpu_memory_used)
        if not memory_usage:
            return 1.0

        memory_variance = np.var(memory_usage)
        memory_mean = np.mean(memory_usage)
        if memory_mean == 0:
            return 1.0

        # Efficiency decreases with higher variance relative to mean
        return max(0, 1 - (memory_variance / memory_mean))

    def _calculate_gpu_utilization(self, metrics: List[OpenRLHFMetrics]) -> float:
        """Calculate average GPU utilization."""
        gpu_utils = []
        for m in metrics:
            if hasattr(m, 'gpu_utilization'):
                # Include all values, not just > 0, since 0 might be valid
                gpu_utils.append(m.gpu_utilization)
        return np.mean(gpu_utils) if gpu_utils else 0.0

    def _calculate_detailed_metrics(self, metrics: List[OpenRLHFMetrics]) -> Dict[str, float]:
        """Calculate detailed performance metrics."""
        detailed = {}

        # Forward/backward time ratio
        forward_times = []
        backward_times = []
        for m in metrics:
            if hasattr(m, 'forward_time'):
                forward_times.append(m.forward_time)
            if hasattr(m, 'backward_time'):
                backward_times.append(m.backward_time)

        if forward_times and backward_times:
            avg_forward = np.mean(forward_times)
            avg_backward = np.mean(backward_times)
            detailed['forward_backward_ratio'] = avg_forward / avg_backward if avg_backward > 0 else 0.0

        # Memory allocation efficiency
        memory_allocated = []
        memory_used = []
        for m in metrics:
            if hasattr(m, 'gpu_memory_allocated'):
                memory_allocated.append(m.gpu_memory_allocated)
            if hasattr(m, 'gpu_memory_used'):
                memory_used.append(m.gpu_memory_used)

        if memory_allocated and memory_used:
            avg_allocated = np.mean(memory_allocated)
            avg_used = np.mean(memory_used)
            detailed['memory_allocation_efficiency'] = avg_used / avg_allocated if avg_allocated > 0 else 0.0

        # CPU utilization
        cpu_utils = []
        for m in metrics:
            if hasattr(m, 'cpu_utilization'):
                cpu_utils.append(m.cpu_utilization)
        detailed['cpu_utilization'] = np.mean(cpu_utils) if cpu_utils else 0.0

        # Network performance (for distributed training)
        allreduce_times = []
        for m in metrics:
            if hasattr(m, 'allreduce_time'):
                allreduce_times.append(m.allreduce_time)

        if allreduce_times:
            detailed['avg_allreduce_time'] = np.mean(allreduce_times)

        return detailed

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-1)."""
        # Weighted combination of key metrics
        speed_score = min(1.0, metrics['training_speed'] / self.speed_threshold) if self.speed_threshold > 0 else 0.0
        memory_score = metrics['memory_efficiency']
        gpu_score = metrics['gpu_utilization']

        # Weighted average (can be adjusted based on priorities)
        weights = {'speed': 0.4, 'memory': 0.3, 'gpu': 0.3}
        performance_score = (
            weights['speed'] * speed_score +
            weights['memory'] * memory_score +
            weights['gpu'] * gpu_score
        )

        return max(0, min(1, performance_score))

    def _determine_performance_status(self, metrics: Dict[str, float]) -> str:
        """Determine performance status based on thresholds."""
        if metrics['training_speed'] < self.speed_threshold:
            return 'slow_training'
        elif metrics['memory_efficiency'] < self.memory_efficiency_threshold:
            return 'memory_inefficient'
        elif metrics['gpu_utilization'] < self.gpu_utilization_threshold:
            return 'low_gpu_utilization'
        elif metrics['performance_score'] >= 0.8:
            return 'excellent'
        elif metrics['performance_score'] >= 0.6:
            return 'good'
        else:
            return 'poor'

    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline metrics from performance history."""
        if not self.performance_history:
            return {}

        baseline = {}
        for key in self.performance_history[0].keys():
            if key != 'performance_status':  # Skip non-numeric fields
                values = [h[key] for h in self.performance_history if key in h]
                if values:
                    baseline[key] = np.mean(values)

        return baseline

    def get_performance_trends(self) -> Dict[str, float]:
        """Get performance trends over time."""
        if len(self.performance_history) < 5:
            return {}

        trends = {}
        for key in self.performance_history[0].keys():
            if key != 'performance_status':  # Skip non-numeric fields
                values = [h[key] for h in self.performance_history if key in h]
                if len(values) >= 5:
                    # Calculate trend (positive = improving, negative = degrading)
                    early_values = values[:len(values)//3]
                    late_values = values[-len(values)//3:]
                    if early_values and late_values:
                        early_mean = np.mean(early_values)
                        late_mean = np.mean(late_values)
                        if early_mean != 0:
                            trends[key] = (late_mean - early_mean) / abs(early_mean)

        return trends
