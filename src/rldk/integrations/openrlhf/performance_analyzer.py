"""Performance analyzer for OpenRLHF training with threshold monitoring and rolling window analysis."""

import os
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class PerformanceThresholds:
    """Configuration for performance thresholds."""
    kl_high: float = 0.1
    kl_low: float = 0.01
    entropy_low: float = 0.1
    throughput_low: float = 0.1  # steps per second
    window_size: int = 10
    emit: bool = True


class PerformanceAnalyzer:
    """Analyzes OpenRLHF training performance with rolling window thresholds and alerts."""

    def __init__(
        self,
        kl_high: float = 0.1,
        kl_low: float = 0.01,
        entropy_low: float = 0.1,
        throughput_low: float = 0.1,
        window_size: int = 10,
        emit: bool = True
    ):
        """Initialize PerformanceAnalyzer with thresholds and configuration.

        Args:
            kl_high: High threshold for KL divergence (triggers alert)
            kl_low: Low threshold for KL divergence (triggers warning)
            entropy_low: Low threshold for entropy (triggers warning)
            throughput_low: Low threshold for throughput in steps/sec (triggers warning)
            window_size: Number of recent steps to keep in rolling window
            emit: Whether to emit alerts and warnings
        """
        self.kl_high = kl_high
        self.kl_low = kl_low
        self.entropy_low = entropy_low
        self.throughput_low = throughput_low
        self.window_size = window_size
        self.emit = emit

        # Rolling buffers for metrics
        self.kl_history = deque(maxlen=window_size)
        self.entropy_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)

        # Analysis state
        self.step_count = 0
        self.last_analysis_step = 0

    @classmethod
    def from_env(cls) -> 'PerformanceAnalyzer':
        """Create PerformanceAnalyzer from environment variables.

        Environment variables:
            RLHF_PERF_KL_HIGH: High KL threshold (default: 0.1)
            RLHF_PERF_KL_LOW: Low KL threshold (default: 0.01)
            RLHF_PERF_ENTROPY_LOW: Low entropy threshold (default: 0.1)
            RLHF_PERF_THROUGHPUT_LOW: Low throughput threshold (default: 0.1)
            RLHF_PERF_WINDOW_SIZE: Window size (default: 10)
            RLHF_PERF_EMIT: Whether to emit alerts (default: true)
        """
        return cls(
            kl_high=float(os.getenv('RLHF_PERF_KL_HIGH', '0.1')),
            kl_low=float(os.getenv('RLHF_PERF_KL_LOW', '0.01')),
            entropy_low=float(os.getenv('RLHF_PERF_ENTROPY_LOW', '0.1')),
            throughput_low=float(os.getenv('RLHF_PERF_THROUGHPUT_LOW', '0.1')),
            window_size=int(os.getenv('RLHF_PERF_WINDOW_SIZE', '10')),
            emit=os.getenv('RLHF_PERF_EMIT', 'true').lower() == 'true'
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PerformanceAnalyzer':
        """Create PerformanceAnalyzer from configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters
        """
        return cls(
            kl_high=config.get('kl_high', 0.1),
            kl_low=config.get('kl_low', 0.01),
            entropy_low=config.get('entropy_low', 0.1),
            throughput_low=config.get('throughput_low', 0.1),
            window_size=config.get('window_size', 10),
            emit=config.get('emit', True)
        )

    def analyze(self, step_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze step metrics and return performance report.

        Args:
            step_metrics: Dictionary containing metrics for current step
                Expected keys: kl_mean, entropy_mean, step_time, loss, reward_mean

        Returns:
            Dictionary with keys:
                - status: "ok", "warn", or "alert"
                - signals: Dictionary of computed signals
                - reasons: List of reasons for status
        """
        self.step_count += 1

        # Extract metrics from step_metrics
        kl_mean = step_metrics.get('kl_mean', 0.0)
        entropy_mean = step_metrics.get('entropy_mean', 0.0)
        step_time = step_metrics.get('step_time', 0.0)
        loss = step_metrics.get('loss', 0.0)
        reward_mean = step_metrics.get('reward_mean', 0.0)

        # Update rolling buffers
        self._update_buffers(kl_mean, entropy_mean, step_time, loss, reward_mean)

        # Compute signals
        signals = self._compute_signals()

        # Determine status and reasons
        status, reasons = self._evaluate_status(signals)

        # Emit alerts if enabled
        if self.emit and status in ['warn', 'alert']:
            self._emit_alert(status, reasons, signals)

        return {
            'status': status,
            'signals': signals,
            'reasons': reasons
        }

    def _update_buffers(self, kl_mean: float, entropy_mean: float, step_time: float,
                       loss: float, reward_mean: float):
        """Update rolling buffers with new metrics."""
        self.kl_history.append(kl_mean)
        self.entropy_history.append(entropy_mean)
        self.step_times.append(step_time)
        self.loss_history.append(loss)
        self.reward_history.append(reward_mean)

        # Calculate throughput (steps per second)
        if step_time > 0:
            throughput = 1.0 / step_time
            self.throughput_history.append(throughput)

    def _compute_signals(self) -> Dict[str, Any]:
        """Compute performance signals from rolling buffers."""
        signals = {}

        # KL divergence signals
        if self.kl_history:
            signals['kl_mean'] = np.mean(self.kl_history)
            signals['kl_std'] = np.std(self.kl_history)
            signals['kl_max'] = np.max(self.kl_history)
            signals['kl_trend'] = self._compute_trend(list(self.kl_history))
        else:
            signals['kl_mean'] = 0.0
            signals['kl_std'] = 0.0
            signals['kl_max'] = 0.0
            signals['kl_trend'] = 0.0

        # Entropy signals
        if self.entropy_history:
            signals['entropy_mean'] = np.mean(self.entropy_history)
            signals['entropy_std'] = np.std(self.entropy_history)
            signals['entropy_min'] = np.min(self.entropy_history)
            signals['entropy_trend'] = self._compute_trend(list(self.entropy_history))
        else:
            signals['entropy_mean'] = 0.0
            signals['entropy_std'] = 0.0
            signals['entropy_min'] = 0.0
            signals['entropy_trend'] = 0.0

        # Throughput signals
        if self.throughput_history:
            signals['throughput_mean'] = np.mean(self.throughput_history)
            signals['throughput_std'] = np.std(self.throughput_history)
            signals['throughput_min'] = np.min(self.throughput_history)
            signals['throughput_trend'] = self._compute_trend(list(self.throughput_history))
        else:
            signals['throughput_mean'] = 0.0
            signals['throughput_std'] = 0.0
            signals['throughput_min'] = 0.0
            signals['throughput_trend'] = 0.0

        # Loss signals
        if self.loss_history:
            signals['loss_mean'] = np.mean(self.loss_history)
            signals['loss_std'] = np.std(self.loss_history)
            signals['loss_trend'] = self._compute_trend(list(self.loss_history))
        else:
            signals['loss_mean'] = 0.0
            signals['loss_std'] = 0.0
            signals['loss_trend'] = 0.0

        # Reward signals
        if self.reward_history:
            signals['reward_mean'] = np.mean(self.reward_history)
            signals['reward_std'] = np.std(self.reward_history)
            signals['reward_trend'] = self._compute_trend(list(self.reward_history))
        else:
            signals['reward_mean'] = 0.0
            signals['reward_std'] = 0.0
            signals['reward_trend'] = 0.0

        # Window statistics
        signals['window_size'] = len(self.kl_history)
        signals['step_count'] = self.step_count

        return signals

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend of values (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Handle case where all values are the same
        if np.std(y) == 0:
            return 0.0

        # Compute linear regression slope
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def _evaluate_status(self, signals: Dict[str, Any]) -> tuple[str, List[str]]:
        """Evaluate performance status based on signals and thresholds."""
        status = "ok"
        reasons = []

        # Check KL divergence thresholds
        kl_mean = signals['kl_mean']
        if kl_mean >= self.kl_high:
            status = "alert"
            reasons.append(f"KL divergence {kl_mean:.4f} exceeds high threshold {self.kl_high}")
        elif kl_mean >= self.kl_low:
            if status == "ok":
                status = "warn"
            reasons.append(f"KL divergence {kl_mean:.4f} exceeds low threshold {self.kl_low}")

        # Check entropy threshold
        entropy_mean = signals['entropy_mean']
        if entropy_mean <= self.entropy_low:
            if status == "ok":
                status = "warn"
            reasons.append(f"Entropy {entropy_mean:.4f} below threshold {self.entropy_low}")

        # Check throughput threshold
        throughput_mean = signals['throughput_mean']
        if throughput_mean <= self.throughput_low and throughput_mean > 0:
            if status == "ok":
                status = "warn"
            reasons.append(f"Throughput {throughput_mean:.4f} below threshold {self.throughput_low}")

        # Check for sustained high KL divergence over window
        if len(self.kl_history) >= self.window_size:
            high_kl_count = sum(1 for kl in self.kl_history if kl >= self.kl_high)
            if high_kl_count >= self.window_size * 0.8:  # 80% of window
                status = "alert"
                reasons.append(f"KL divergence high for {high_kl_count}/{len(self.kl_history)} recent steps")

        # Check for sustained low entropy over window
        if len(self.entropy_history) >= self.window_size:
            low_entropy_count = sum(1 for ent in self.entropy_history if ent <= self.entropy_low)
            if low_entropy_count >= self.window_size * 0.8:  # 80% of window
                if status == "ok":
                    status = "warn"
                reasons.append(f"Entropy low for {low_entropy_count}/{len(self.entropy_history)} recent steps")

        # Check for sustained low throughput over window
        if len(self.throughput_history) >= self.window_size:
            low_throughput_count = sum(1 for tp in self.throughput_history if tp <= self.throughput_low)
            if low_throughput_count >= self.window_size * 0.8:  # 80% of window
                if status == "ok":
                    status = "warn"
                reasons.append(f"Throughput low for {low_throughput_count}/{len(self.throughput_history)} recent steps")

        return status, reasons

    def _emit_alert(self, status: str, reasons: List[str], signals: Dict[str, Any]):
        """Emit alert or warning message."""
        if not self.emit:
            return

        message = f"PerformanceAnalyzer {status.upper()}: {', '.join(reasons)}"
        message += f" (Step {self.step_count}, Window: {signals['window_size']})"

        if status == "alert":
            warnings.warn(message, UserWarning, stacklevel=2)
        else:
            print(f"WARNING: {message}")

    def get_current_state(self) -> Dict[str, Any]:
        """Get current analyzer state for debugging."""
        return {
            'thresholds': {
                'kl_high': self.kl_high,
                'kl_low': self.kl_low,
                'entropy_low': self.entropy_low,
                'throughput_low': self.throughput_low,
                'window_size': self.window_size
            },
            'buffers': {
                'kl_history': list(self.kl_history),
                'entropy_history': list(self.entropy_history),
                'throughput_history': list(self.throughput_history),
                'loss_history': list(self.loss_history),
                'reward_history': list(self.reward_history)
            },
            'state': {
                'step_count': self.step_count,
                'last_analysis_step': self.last_analysis_step,
                'emit': self.emit
            }
        }

    def reset(self):
        """Reset analyzer state and clear all buffers."""
        self.kl_history.clear()
        self.entropy_history.clear()
        self.throughput_history.clear()
        self.step_times.clear()
        self.loss_history.clear()
        self.reward_history.clear()

        self.step_count = 0
        self.last_analysis_step = 0
