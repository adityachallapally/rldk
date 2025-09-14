"""Custom callback examples for specific use cases."""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from rldk.integrations.trl import PPOMonitor, RLDKCallback
from rldk.utils.math_utils import safe_divide, safe_rate_calculation


class RewardModelMonitor(TrainerCallback):
    """Monitor for reward model training and reliability."""

    def __init__(
        self,
        output_dir: str,
        reward_model_threshold: float = 0.8,
        calibration_threshold: float = 0.1,
        run_id: Optional[str] = None,
    ):
        """Initialize reward model monitor."""
        self.output_dir = output_dir
        self.reward_model_threshold = reward_model_threshold
        self.calibration_threshold = calibration_threshold
        self.run_id = run_id or f"reward_model_{int(time.time())}"

        # Metrics storage
        self.reward_model_metrics: List[Dict[str, Any]] = []
        self.calibration_history: List[float] = []
        self.reliability_scores: List[float] = []

        print(f"🎯 Reward Model Monitor initialized - Run ID: {self.run_id}")

    def on_log(self, args, state, control, logs: Dict[str, float], **kwargs):
        """Monitor reward model metrics from logs."""
        # Extract reward model specific metrics
        reward_model_metrics = {}

        # Reward model accuracy
        if 'reward_model/accuracy' in logs:
            reward_model_metrics['accuracy'] = logs['reward_model/accuracy']

        # Reward model loss
        if 'reward_model/loss' in logs:
            reward_model_metrics['loss'] = logs['reward_model/loss']

        # Calibration metrics
        if 'reward_model/calibration_error' in logs:
            calibration_error = logs['reward_model/calibration_error']
            self.calibration_history.append(calibration_error)
            reward_model_metrics['calibration_error'] = calibration_error

        # Reliability metrics
        if 'reward_model/reliability' in logs:
            reliability = logs['reward_model/reliability']
            self.reliability_scores.append(reliability)
            reward_model_metrics['reliability'] = reliability

        if reward_model_metrics:
            reward_model_metrics.update({
                'step': state.global_step,
                'timestamp': time.time(),
            })
            self.reward_model_metrics.append(reward_model_metrics)

            # Check for issues
            self._check_reward_model_health(reward_model_metrics)

    def _check_reward_model_health(self, metrics: Dict[str, Any]):
        """Check reward model health indicators."""
        # Low accuracy alert
        if 'accuracy' in metrics and metrics['accuracy'] < self.reward_model_threshold:
            print(f"⚠️  Reward Model Alert: Low accuracy {metrics['accuracy']:.3f}")

        # High calibration error alert
        if 'calibration_error' in metrics and metrics['calibration_error'] > self.calibration_threshold:
            print(f"⚠️  Reward Model Alert: High calibration error {metrics['calibration_error']:.3f}")

        # Low reliability alert
        if 'reliability' in metrics and metrics['reliability'] < 0.7:
            print(f"⚠️  Reward Model Alert: Low reliability {metrics['reliability']:.3f}")

    def get_reward_model_summary(self) -> Dict[str, Any]:
        """Get reward model training summary."""
        if not self.reward_model_metrics:
            return {}

        df = pd.DataFrame(self.reward_model_metrics)

        return {
            "total_steps": len(self.reward_model_metrics),
            "final_accuracy": df['accuracy'].iloc[-1] if 'accuracy' in df.columns else 0,
            "average_calibration_error": np.mean(self.calibration_history) if self.calibration_history else 0,
            "average_reliability": np.mean(self.reliability_scores) if self.reliability_scores else 0,
            "calibration_trend": self._calculate_calibration_trend(),
        }

    def _calculate_calibration_trend(self) -> str:
        """Calculate calibration trend."""
        if len(self.calibration_history) < 10:
            return "insufficient_data"

        recent_calibration = self.calibration_history[-10:]
        trend = np.polyfit(range(len(recent_calibration)), recent_calibration, 1)[0]

        if trend < -0.01:
            return "improving"
        elif trend > 0.01:
            return "worsening"
        else:
            return "stable"


class DataPipelineMonitor(TrainerCallback):
    """Monitor for data pipeline efficiency and quality."""

    def __init__(
        self,
        output_dir: str,
        target_throughput: float = 1000.0,  # samples per second
        quality_threshold: float = 0.95,
        run_id: Optional[str] = None,
    ):
        """Initialize data pipeline monitor."""
        self.output_dir = output_dir
        self.target_throughput = target_throughput
        self.quality_threshold = quality_threshold
        self.run_id = run_id or f"data_pipeline_{int(time.time())}"

        # Metrics storage
        self.pipeline_metrics: List[Dict[str, Any]] = []
        self.throughput_history: List[float] = []
        self.quality_scores: List[float] = []

        print(f"📊 Data Pipeline Monitor initialized - Run ID: {self.run_id}")

    def on_step_begin(self, args, state, control, **kwargs):
        """Monitor data loading at step begin."""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate pipeline metrics at step end."""
        step_time = time.time() - self.step_start_time

        # Extract batch size from args
        batch_size = getattr(args, 'per_device_train_batch_size', 1)
        total_batch_size = batch_size * getattr(args, 'world_size', 1)

        # Calculate throughput
        throughput = safe_rate_calculation(total_batch_size, step_time, 0.0)
        self.throughput_history.append(throughput)

        # Simulate data quality check (in practice, this would be real quality metrics)
        quality_score = self._simulate_quality_check()
        self.quality_scores.append(quality_score)

        # Store metrics
        pipeline_metrics = {
            'step': state.global_step,
            'throughput': throughput,
            'quality_score': quality_score,
            'batch_size': total_batch_size,
            'step_time': step_time,
            'efficiency': safe_divide(throughput, self.target_throughput, 0.0),
            'timestamp': time.time(),
        }

        self.pipeline_metrics.append(pipeline_metrics)

        # Check for issues
        self._check_pipeline_health(pipeline_metrics)

    def _simulate_quality_check(self) -> float:
        """Simulate data quality check (replace with real implementation)."""
        # In practice, this would check for:
        # - Token distribution anomalies
        # - Sequence length outliers
        # - Data format consistency
        # - Label quality
        return np.random.uniform(0.9, 1.0)  # Simulate high quality

    def _check_pipeline_health(self, metrics: Dict[str, Any]):
        """Check data pipeline health."""
        # Low throughput alert
        if metrics['throughput'] < self.target_throughput * 0.5:
            print(f"⚠️  Pipeline Alert: Low throughput {metrics['throughput']:.1f} samples/sec")

        # Low quality alert
        if metrics['quality_score'] < self.quality_threshold:
            print(f"⚠️  Pipeline Alert: Low data quality {metrics['quality_score']:.3f}")

        # Low efficiency alert
        if metrics['efficiency'] < 0.5:
            print(f"⚠️  Pipeline Alert: Low efficiency {metrics['efficiency']:.3f}")

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get data pipeline summary."""
        if not self.pipeline_metrics:
            return {}

        df = pd.DataFrame(self.pipeline_metrics)

        return {
            "total_steps": len(self.pipeline_metrics),
            "average_throughput": df['throughput'].mean(),
            "average_quality": df['quality_score'].mean(),
            "average_efficiency": df['efficiency'].mean(),
            "throughput_trend": self._calculate_throughput_trend(),
            "quality_trend": self._calculate_quality_trend(),
        }

    def _calculate_throughput_trend(self) -> str:
        """Calculate throughput trend."""
        if len(self.throughput_history) < 10:
            return "insufficient_data"

        recent_throughput = self.throughput_history[-10:]
        trend = np.polyfit(range(len(recent_throughput)), recent_throughput, 1)[0]

        if trend > 10:
            return "improving"
        elif trend < -10:
            return "declining"
        else:
            return "stable"

    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend."""
        if len(self.quality_scores) < 10:
            return "insufficient_data"

        recent_quality = self.quality_scores[-10:]
        trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]

        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"


class MemoryOptimizationMonitor(TrainerCallback):
    """Monitor for memory usage and optimization opportunities."""

    def __init__(
        self,
        output_dir: str,
        memory_threshold: float = 0.9,  # 90% of available memory
        run_id: Optional[str] = None,
    ):
        """Initialize memory optimization monitor."""
        self.output_dir = output_dir
        self.memory_threshold = memory_threshold
        self.run_id = run_id or f"memory_monitor_{int(time.time())}"

        # Metrics storage
        self.memory_metrics: List[Dict[str, Any]] = []
        self.optimization_suggestions: List[str] = []

        print(f"💾 Memory Optimization Monitor initialized - Run ID: {self.run_id}")

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor memory usage at step end."""
        # Get memory usage
        memory_info = self._get_memory_info()

        # Calculate memory efficiency
        memory_efficiency = self._calculate_memory_efficiency(memory_info)

        # Store metrics
        memory_metrics = {
            'step': state.global_step,
            'gpu_memory_used': memory_info['gpu_memory_used'],
            'gpu_memory_total': memory_info['gpu_memory_total'],
            'gpu_memory_percent': memory_info['gpu_memory_percent'],
            'cpu_memory_used': memory_info['cpu_memory_used'],
            'cpu_memory_percent': memory_info['cpu_memory_percent'],
            'memory_efficiency': memory_efficiency,
            'timestamp': time.time(),
        }

        self.memory_metrics.append(memory_metrics)

        # Check for optimization opportunities
        self._check_memory_optimization(memory_metrics)

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = {
            'gpu_memory_used': 0.0,
            'gpu_memory_total': 0.0,
            'gpu_memory_percent': 0.0,
            'cpu_memory_used': 0.0,
            'cpu_memory_percent': 0.0,
        }

        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_memory_used'] = safe_divide(torch.cuda.memory_allocated(), 1024**3, 0.0)  # GB
            memory_info['gpu_memory_total'] = safe_divide(torch.cuda.get_device_properties(0).total_memory, 1024**3, 0.0)  # GB
            memory_info['gpu_memory_percent'] = safe_divide(memory_info['gpu_memory_used'], memory_info['gpu_memory_total'], 0.0)

        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            memory_info['cpu_memory_used'] = safe_divide(process.memory_info().rss, 1024**3, 0.0)  # GB
            memory_info['cpu_memory_percent'] = process.memory_percent()
        except ImportError:
            pass

        return memory_info

    def _calculate_memory_efficiency(self, memory_info: Dict[str, float]) -> float:
        """Calculate memory efficiency score."""
        # Efficiency based on memory utilization vs performance
        gpu_efficiency = memory_info['gpu_memory_percent']
        cpu_efficiency = safe_divide(memory_info['cpu_memory_percent'], 100.0, 0.0)

        # Optimal range is 70-90% for GPU, 50-80% for CPU
        gpu_score = 1.0 if 0.7 <= gpu_efficiency <= 0.9 else max(0, 1 - abs(gpu_efficiency - 0.8) * 2)
        cpu_score = 1.0 if 0.5 <= cpu_efficiency <= 0.8 else max(0, 1 - abs(cpu_efficiency - 0.65) * 2)

        return safe_divide(gpu_score + cpu_score, 2, 0.0)

    def _check_memory_optimization(self, metrics: Dict[str, Any]):
        """Check for memory optimization opportunities."""
        # High memory usage alert
        if metrics['gpu_memory_percent'] > self.memory_threshold:
            suggestion = f"High GPU memory usage ({metrics['gpu_memory_percent']:.1%}). Consider reducing batch size or using gradient checkpointing."
            self.optimization_suggestions.append(suggestion)
            print(f"💾 Memory Alert: {suggestion}")

        # Low memory efficiency
        if metrics['memory_efficiency'] < 0.5:
            suggestion = f"Low memory efficiency ({metrics['memory_efficiency']:.3f}). Consider optimizing memory usage."
            self.optimization_suggestions.append(suggestion)
            print(f"💾 Efficiency Alert: {suggestion}")

        # CPU memory high
        if metrics['cpu_memory_percent'] > 80:
            suggestion = f"High CPU memory usage ({metrics['cpu_memory_percent']:.1%}). Consider reducing data loading workers."
            self.optimization_suggestions.append(suggestion)
            print(f"💾 CPU Alert: {suggestion}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_metrics:
            return {}

        df = pd.DataFrame(self.memory_metrics)

        return {
            "total_steps": len(self.memory_metrics),
            "average_gpu_usage": df['gpu_memory_percent'].mean(),
            "max_gpu_usage": df['gpu_memory_percent'].max(),
            "average_cpu_usage": df['cpu_memory_percent'].mean(),
            "average_efficiency": df['memory_efficiency'].mean(),
            "optimization_suggestions": len(self.optimization_suggestions),
            "memory_trend": self._calculate_memory_trend(),
        }

    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        if len(self.memory_metrics) < 10:
            return "insufficient_data"

        recent_usage = [m['gpu_memory_percent'] for m in self.memory_metrics[-10:]]
        trend = np.polyfit(range(len(recent_usage)), recent_usage, 1)[0]

        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"


def test_custom_callbacks():
    """Test custom callback implementations."""
    print("🧪 Testing Custom Callbacks")

    output_dir = "./test_custom_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize custom callbacks
    reward_monitor = RewardModelMonitor(
        output_dir=output_dir,
        reward_model_threshold=0.8,
        run_id="custom_test"
    )

    pipeline_monitor = DataPipelineMonitor(
        output_dir=output_dir,
        target_throughput=1000.0,
        run_id="custom_test"
    )

    memory_monitor = MemoryOptimizationMonitor(
        output_dir=output_dir,
        memory_threshold=0.9,
        run_id="custom_test"
    )

    print("✅ Custom callbacks initialized")

    # Simulate training with various scenarios
    from transformers import TrainerControl, TrainerState, TrainingArguments

    args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=4)

    for step in range(20):
        state = TrainerState()
        state.global_step = step
        control = TrainerControl()

        # Simulate reward model logs
        reward_logs = {
            'reward_model/accuracy': 0.7 + step * 0.01,
            'reward_model/loss': 0.5 - step * 0.02,
            'reward_model/calibration_error': 0.1 - step * 0.005,
            'reward_model/reliability': 0.8 + step * 0.01,
        }

        # Call callbacks
        reward_monitor.on_log(args, state, control, reward_logs)
        pipeline_monitor.on_step_begin(args, state, control)
        pipeline_monitor.on_step_end(args, state, control)
        memory_monitor.on_step_end(args, state, control)

        if step % 5 == 0:
            print(f"✅ Step {step} completed")

    # Generate summaries
    print("\n📊 Custom Callback Summaries")

    reward_summary = reward_monitor.get_reward_model_summary()
    print(f"Reward Model Summary: {reward_summary}")

    pipeline_summary = pipeline_monitor.get_pipeline_summary()
    print(f"Pipeline Summary: {pipeline_summary}")

    memory_summary = memory_monitor.get_memory_summary()
    print(f"Memory Summary: {memory_summary}")

    print("✅ Custom callbacks test completed")
    return True


if __name__ == "__main__":
    import time

    import pandas as pd

    print("🎯 Custom RLDK Callbacks Test Suite")
    print("=" * 50)

    success = test_custom_callbacks()

    if success:
        print("\n🎉 Custom callbacks test passed!")
    else:
        print("\n❌ Custom callbacks test failed.")
