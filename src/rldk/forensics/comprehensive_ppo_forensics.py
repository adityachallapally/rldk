"""Comprehensive PPO forensics with advanced tracking and analysis."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from .advantage_statistics_tracker import (
    AdvantageStatisticsMetrics,
    AdvantageStatisticsTracker,
)
from .gradient_norms_analyzer import GradientNormsAnalyzer, GradientNormsMetrics
from .kl_schedule_tracker import KLScheduleMetrics, KLScheduleTracker
from .ppo_scan import scan_ppo_events
from ..reward.length_bias import LengthBiasDetector, LengthBiasMetrics


@dataclass
class ComprehensivePPOMetrics:
    """Container for comprehensive PPO metrics."""

    # Basic PPO metrics
    step: int = 0
    kl: float = 0.0
    kl_coef: float = 1.0
    entropy: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0

    # Advanced tracking metrics
    kl_schedule_metrics: Optional[KLScheduleMetrics] = None
    gradient_norms_metrics: Optional[GradientNormsMetrics] = None
    advantage_statistics_metrics: Optional[AdvantageStatisticsMetrics] = None

    # Overall health scores
    overall_health_score: float = 1.0
    training_stability_score: float = 1.0
    convergence_quality_score: float = 1.0

    # Length bias metrics
    length_bias_score: Optional[float] = None
    length_reward_correlation: Optional[float] = None
    length_reward_spearman: Optional[float] = None
    length_reward_correlation_abs: Optional[float] = None
    length_reward_spearman_abs: Optional[float] = None
    mean_response_length: Optional[float] = None
    length_bias_detected: bool = False
    length_bias_threshold: Optional[float] = None
    length_bias_corr_threshold: Optional[float] = None
    length_bias_metrics: Optional[LengthBiasMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "step": self.step,
            "kl": self.kl,
            "kl_coef": self.kl_coef,
            "entropy": self.entropy,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "overall_health_score": self.overall_health_score,
            "training_stability_score": self.training_stability_score,
            "convergence_quality_score": self.convergence_quality_score,
        }

        result.update(
            {
                "length_bias_score": self.length_bias_score,
                "length_reward_correlation": self.length_reward_correlation,
                "length_reward_spearman": self.length_reward_spearman,
                "length_reward_correlation_abs": self.length_reward_correlation_abs,
                "length_reward_spearman_abs": self.length_reward_spearman_abs,
                "mean_response_length": self.mean_response_length,
                "length_bias_detected": self.length_bias_detected,
                "length_bias_threshold": self.length_bias_threshold,
                "length_bias_corr_threshold": self.length_bias_corr_threshold,
            }
        )

        if self.length_bias_metrics:
            result["length_bias_metrics"] = self.length_bias_metrics.to_dict()

        if self.kl_schedule_metrics:
            result.update({f"kl_schedule_{k}": v for k, v in self.kl_schedule_metrics.to_dict().items()})

        if self.gradient_norms_metrics:
            result.update({f"gradient_{k}": v for k, v in self.gradient_norms_metrics.to_dict().items()})

        if self.advantage_statistics_metrics:
            result.update({f"advantage_{k}": v for k, v in self.advantage_statistics_metrics.to_dict().items()})

        return result


class ComprehensivePPOForensics:
    """Comprehensive PPO forensics with advanced tracking and analysis."""

    def __init__(
        self,
        kl_target: float = 0.1,
        kl_target_tolerance: float = 0.05,
        window_size: int = 100,
        enable_kl_schedule_tracking: bool = True,
        enable_gradient_norms_analysis: bool = True,
        enable_advantage_statistics: bool = True,
        enable_kl_drift_tracking: bool = True,
        kl_drift_threshold: float = 0.15,
        kl_drift_window_size: int = 100,
        kl_drift_reference_period: int = 500,
        enable_length_bias_detection: bool = True,
        length_bias_threshold: float = 0.35,
        response_data: Optional[Sequence[Mapping[str, Any]]] = None,
    ):
        """Initialize comprehensive PPO forensics.

        Args:
            kl_target: Target KL divergence value
            kl_target_tolerance: Tolerance around target for "in range" calculation
            window_size: Size of rolling window for analysis
            enable_kl_schedule_tracking: Enable KL schedule tracking
            enable_gradient_norms_analysis: Enable gradient norms analysis
            enable_advantage_statistics: Enable advantage statistics tracking
            enable_kl_drift_tracking: Enable KL drift tracking and analysis
            kl_drift_threshold: KL divergence threshold for drift detection
            kl_drift_window_size: Rolling window size for drift calculations
            kl_drift_reference_period: Number of steps used to build the reference distribution
            enable_length_bias_detection: Toggle length bias detector integration
            length_bias_threshold: Threshold applied to detector severity scores
            response_data: Optional initial response/reward records for the detector
        """
        self.kl_target = kl_target
        self.kl_target_tolerance = kl_target_tolerance
        self.window_size = window_size

        # Initialize trackers
        self.kl_schedule_tracker = None
        self.gradient_norms_analyzer = None
        self.advantage_statistics_tracker = None

        if enable_kl_schedule_tracking:
            self.kl_schedule_tracker = KLScheduleTracker(
                kl_target=kl_target,
                kl_target_tolerance=kl_target_tolerance,
                window_size=window_size,
                drift_threshold=kl_drift_threshold,
                drift_window_size=kl_drift_window_size,
                reference_period=kl_drift_reference_period,
                enable_drift_tracking=enable_kl_drift_tracking,
            )

        if enable_gradient_norms_analysis:
            self.gradient_norms_analyzer = GradientNormsAnalyzer(
                window_size=window_size
            )

        if enable_advantage_statistics:
            self.advantage_statistics_tracker = AdvantageStatisticsTracker(
                window_size=window_size
            )

        # Metrics storage
        self.comprehensive_metrics_history: List[ComprehensivePPOMetrics] = []
        self.current_metrics = ComprehensivePPOMetrics()

        # Drift tracking configuration
        self.enable_kl_drift_tracking = enable_kl_drift_tracking
        self.kl_drift_threshold = kl_drift_threshold

        # Length bias detection configuration
        self.enable_length_bias_detection = enable_length_bias_detection
        self.length_bias_threshold = length_bias_threshold
        self.length_bias_corr_threshold = max(0.2, min(0.6, length_bias_threshold + 0.05))
        self._length_bias_detector: Optional[LengthBiasDetector] = (
            LengthBiasDetector() if enable_length_bias_detection else None
        )
        self._length_bias_metrics_cache: LengthBiasMetrics = LengthBiasMetrics()
        self._length_bias_analysis_cache: Dict[str, Any] = {
            "enabled": enable_length_bias_detection,
            "detected": False,
            "threshold": length_bias_threshold,
            "corr_threshold": self.length_bias_corr_threshold,
            "metrics": self._length_bias_metrics_cache.to_dict(),
            "recommendations": [],
            "sample_count": 0,
        }
        self._response_records: List[Dict[str, Any]] = []
        self._length_bias_dirty: bool = False
        if response_data and enable_length_bias_detection:
            self._ingest_response_batch(response_data)
            self._update_length_bias_metrics()
        elif enable_length_bias_detection:
            self._apply_length_bias_metrics()

        # Analysis results
        self.anomalies: List[Dict[str, Any]] = []
        self.analysis_summary: Dict[str, Any] = {}

        print("🔍 Comprehensive PPO Forensics initialized")
        print(f"   KL Schedule Tracking: {enable_kl_schedule_tracking}")
        print(f"   Gradient Norms Analysis: {enable_gradient_norms_analysis}")
        print(f"   Advantage Statistics: {enable_advantage_statistics}")
        print(f"   Length Bias Detection: {enable_length_bias_detection}")

    def update(
        self,
        step: int,
        kl: float,
        kl_coef: float,
        entropy: float,
        reward_mean: float,
        reward_std: float,
        policy_grad_norm: Optional[float] = None,
        value_grad_norm: Optional[float] = None,
        total_grad_norm: Optional[float] = None,
        advantage_mean: Optional[float] = None,
        advantage_std: Optional[float] = None,
        advantage_min: Optional[float] = None,
        advantage_max: Optional[float] = None,
        advantage_median: Optional[float] = None,
        advantage_samples: Optional[List[float]] = None,
        response_data: Optional[Sequence[Mapping[str, Any]]] = None,
        response_length: Optional[float] = None,
        response_text: Optional[str] = None,
        response_reward: Optional[float] = None,
    ) -> ComprehensivePPOMetrics:
        """Update forensics with new training data."""
        # Update basic metrics
        self.current_metrics.step = step
        self.current_metrics.kl = kl
        self.current_metrics.kl_coef = kl_coef
        self.current_metrics.entropy = entropy
        self.current_metrics.reward_mean = reward_mean
        self.current_metrics.reward_std = reward_std

        # Update KL schedule tracking
        if self.kl_schedule_tracker:
            kl_schedule_metrics = self.kl_schedule_tracker.update(step, kl, kl_coef)
            self.current_metrics.kl_schedule_metrics = kl_schedule_metrics

        # Update gradient norms analysis
        if self.gradient_norms_analyzer and policy_grad_norm is not None and value_grad_norm is not None:
            gradient_metrics = self.gradient_norms_analyzer.update(
                step, policy_grad_norm, value_grad_norm, total_grad_norm
            )
            self.current_metrics.gradient_norms_metrics = gradient_metrics

        # Update advantage statistics
        if self.advantage_statistics_tracker and advantage_mean is not None and advantage_std is not None:
            advantage_metrics = self.advantage_statistics_tracker.update(
                step, advantage_mean, advantage_std, advantage_min, advantage_max,
                advantage_median, advantage_samples
            )
            self.current_metrics.advantage_statistics_metrics = advantage_metrics

        # Update length bias detector
        if self.enable_length_bias_detection and self._length_bias_detector:
            batch: List[Mapping[str, Any]] = []
            if response_data:
                if isinstance(response_data, Mapping):
                    batch.append(response_data)
                else:
                    batch.extend(response_data)
            single_record: Dict[str, Any] = {}
            if response_reward is not None:
                single_record["reward"] = response_reward
            if response_length is not None:
                single_record["length"] = response_length
            if response_text is not None:
                single_record["response"] = response_text
            if single_record.get("reward") is not None and (
                single_record.get("length") is not None or single_record.get("response")
            ):
                batch.append(single_record)
            if batch:
                self._ingest_response_batch(batch)
            if self._length_bias_dirty:
                self._update_length_bias_metrics()
            else:
                self._apply_length_bias_metrics()
        else:
            self._reset_length_bias_metrics()

        # Calculate overall health scores
        self._calculate_overall_health_scores()

        # Store metrics - use deep copy to avoid issues with nested dataclasses
        metrics_copy = copy.deepcopy(self.current_metrics)
        self.comprehensive_metrics_history.append(metrics_copy)

        return metrics_copy

    def get_length_bias_analysis(self) -> Dict[str, Any]:
        """Return cached length bias detector output."""

        if not self.enable_length_bias_detection or not self._length_bias_detector:
            return {}
        if self._length_bias_dirty:
            self._update_length_bias_metrics()
        return copy.deepcopy(self._length_bias_analysis_cache)

    def _calculate_overall_health_scores(self):
        """Calculate overall health scores from all trackers."""
        health_scores = []
        stability_scores = []
        convergence_scores = []

        # KL schedule health
        if self.current_metrics.kl_schedule_metrics:
            health_scores.append(self.current_metrics.kl_schedule_metrics.kl_health_score)
            health_scores.append(self.current_metrics.kl_schedule_metrics.schedule_health_score)
            stability_scores.append(self.current_metrics.kl_schedule_metrics.target_range_stability)
            if self.enable_kl_drift_tracking:
                drift_score = self.current_metrics.kl_schedule_metrics.kl_drift_score
                health_scores.append(max(0.0, 1.0 - drift_score))
                stability_scores.append(max(0.0, 1.0 - drift_score))
                convergence_scores.append(max(0.0, 1.0 - drift_score))

        # Gradient norms health
        if self.current_metrics.gradient_norms_metrics:
            health_scores.append(self.current_metrics.gradient_norms_metrics.gradient_health_score)
            stability_scores.append(self.current_metrics.gradient_norms_metrics.training_stability)

        # Advantage statistics health
        if self.current_metrics.advantage_statistics_metrics:
            health_scores.append(self.current_metrics.advantage_statistics_metrics.advantage_health_score)
            convergence_scores.append(self.current_metrics.advantage_statistics_metrics.advantage_quality_score)

        # Calculate overall scores
        self.current_metrics.overall_health_score = np.mean(health_scores) if health_scores else 1.0
        self.current_metrics.training_stability_score = np.mean(stability_scores) if stability_scores else 1.0
        self.current_metrics.convergence_quality_score = np.mean(convergence_scores) if convergence_scores else 1.0

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get all detected anomalies from all trackers."""
        all_anomalies = []

        # KL schedule anomalies
        if self.kl_schedule_tracker:
            kl_anomalies = self.kl_schedule_tracker.get_anomalies()
            for anomaly in kl_anomalies:
                anomaly["tracker"] = "kl_schedule"
            all_anomalies.extend(kl_anomalies)

        # Gradient norms anomalies
        if self.gradient_norms_analyzer:
            grad_anomalies = self.gradient_norms_analyzer.get_anomalies()
            for anomaly in grad_anomalies:
                anomaly["tracker"] = "gradient_norms"
            all_anomalies.extend(grad_anomalies)

        # Advantage statistics anomalies
        if self.advantage_statistics_tracker:
            adv_anomalies = self.advantage_statistics_tracker.get_anomalies()
            for anomaly in adv_anomalies:
                anomaly["tracker"] = "advantage_statistics"
            all_anomalies.extend(adv_anomalies)

        # Store anomalies
        self.anomalies = all_anomalies

        return all_anomalies

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis results."""
        analysis = {
            "version": "2.0",
            "timestamp": time.time(),
            "total_steps": len(self.comprehensive_metrics_history),
            "overall_health_score": self.current_metrics.overall_health_score,
            "training_stability_score": self.current_metrics.training_stability_score,
            "convergence_quality_score": self.current_metrics.convergence_quality_score,
            "anomalies": self.get_anomalies(),
            "trackers": {}
        }

        # KL schedule analysis
        if self.kl_schedule_tracker:
            kl_summary = self.kl_schedule_tracker.get_summary()
            analysis["trackers"]["kl_schedule"] = kl_summary
            if self.enable_kl_drift_tracking:
                analysis["trackers"]["kl_drift"] = self.get_kl_drift_analysis()

        # Gradient norms analysis
        if self.gradient_norms_analyzer:
            analysis["trackers"]["gradient_norms"] = self.gradient_norms_analyzer.get_summary()

        # Advantage statistics analysis
        if self.advantage_statistics_tracker:
            analysis["trackers"]["advantage_statistics"] = self.advantage_statistics_tracker.get_summary()

        if self.enable_length_bias_detection and self._length_bias_detector:
            analysis["trackers"]["length_bias"] = self.get_length_bias_analysis()

        # Store analysis summary
        self.analysis_summary = analysis

        return analysis

    def get_kl_drift_analysis(self) -> Dict[str, Any]:
        """Summarize KL drift metrics for downstream consumers."""

        if not self.kl_schedule_tracker or not self.enable_kl_drift_tracking:
            return {}

        summary = self.kl_schedule_tracker.get_summary()
        anomalies = [
            anomaly
            for anomaly in summary.get("anomalies", [])
            if anomaly.get("type", "").startswith("kl_drift")
        ]

        return {
            "detected": summary.get("kl_drift_detected", False),
            "score": summary.get("kl_drift_score", 0.0),
            "trend": summary.get("kl_drift_trend", "stable"),
            "divergence": summary.get("kl_drift_kl_divergence", 0.0),
            "reference_mean": summary.get("kl_reference_mean", 0.0),
            "reference_std": summary.get("kl_reference_std", 0.0),
            "current_mean": summary.get("kl_current_mean", 0.0),
            "current_std": summary.get("kl_current_std", 0.0),
            "anomalies": anomalies,
        }

    def scan_ppo_events_comprehensive(self, events: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced PPO scan with comprehensive tracking."""
        # Convert events to list first to avoid iterator exhaustion
        events_list = list(events) if hasattr(events, '__iter__') else []

        if not events_list:
            return {"version": "1", "rules_fired": [], "earliest_step": None, "stats": {}}

        # Run the original PPO scan on the buffered events
        original_scan = scan_ppo_events(iter(events_list))

        # Run comprehensive analysis on the events
        for event in events_list:
            step = event.get("step", event.get("global_step", 0))
            kl = event.get("kl", event.get("kl_div", 0.0))
            kl_coef = event.get("kl_coef", event.get("kl_coefficient", 1.0))
            entropy = event.get("entropy", 0.0)
            reward_mean = event.get("reward_mean", event.get("ppo/rewards/mean", 0.0))
            reward_std = event.get("reward_std", event.get("ppo/rewards/std", 0.0))

            # Extract gradient norms
            policy_grad_norm = event.get("grad_norm_policy", event.get("policy_grad_norm", None))
            value_grad_norm = event.get("grad_norm_value", event.get("value_grad_norm", None))
            total_grad_norm = event.get("grad_norm", None)

            # Extract advantage statistics
            advantage_mean = event.get("advantage_mean", event.get("adv_mean", None))
            advantage_std = event.get("advantage_std", event.get("adv_std", None))
            advantage_min = event.get("advantage_min", event.get("adv_min", None))
            advantage_max = event.get("advantage_max", event.get("adv_max", None))
            advantage_median = event.get("advantage_median", event.get("adv_median", None))

            response_payload = event.get("response_data")
            response_length = event.get("response_length", event.get("response_tokens", None))
            response_text = event.get("response_text", event.get("response", None))
            response_reward = event.get("response_reward", event.get("sample_reward", None))

            # Update comprehensive forensics
            self.update(
                step=step,
                kl=kl,
                kl_coef=kl_coef,
                entropy=entropy,
                reward_mean=reward_mean,
                reward_std=reward_std,
                policy_grad_norm=policy_grad_norm,
                value_grad_norm=value_grad_norm,
                total_grad_norm=total_grad_norm,
                advantage_mean=advantage_mean,
                advantage_std=advantage_std,
                advantage_min=advantage_min,
                advantage_max=advantage_max,
                advantage_median=advantage_median,
                response_data=response_payload,
                response_length=response_length,
                response_text=response_text,
                response_reward=response_reward,
            )

        # Get comprehensive analysis
        comprehensive_analysis = self.get_comprehensive_analysis()

        # Merge with original scan results
        enhanced_scan = {
            **original_scan,
            "comprehensive_analysis": comprehensive_analysis,
            "enhanced_version": "2.0"
        }

        return enhanced_scan

    def _ingest_response_batch(
        self, payloads: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]
    ) -> None:
        """Normalize response payloads for incremental length bias detection."""

        added = False
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            reward = payload.get("reward")
            if reward is None:
                continue
            response = payload.get("response") or payload.get("response_text")
            length = (
                payload.get("length")
                if payload.get("length") is not None
                else payload.get("response_length")
            )
            record: Dict[str, Any] = {"reward": float(reward)}
            if response is not None:
                record["response"] = str(response)
            if length is not None:
                try:
                    record["length"] = float(length)
                except (TypeError, ValueError):
                    record["length"] = None
            if "response" not in record and record.get("length") is None:
                continue
            self._response_records.append(record)
            added = True
        if added:
            self._length_bias_dirty = True

    def _reset_length_bias_metrics(self) -> None:
        """Clear length bias metrics when the detector is disabled."""

        self.current_metrics.length_bias_score = None
        self.current_metrics.length_reward_correlation = None
        self.current_metrics.length_reward_spearman = None
        self.current_metrics.length_reward_correlation_abs = None
        self.current_metrics.length_reward_spearman_abs = None
        self.current_metrics.mean_response_length = None
        self.current_metrics.length_bias_detected = False
        self.current_metrics.length_bias_threshold = None
        self.current_metrics.length_bias_corr_threshold = None
        self.current_metrics.length_bias_metrics = None

    def _apply_length_bias_metrics(self) -> None:
        """Mirror cached detector metrics onto the current metrics."""

        if not self.enable_length_bias_detection:
            self._reset_length_bias_metrics()
            return

        metrics = self._length_bias_metrics_cache
        severity = metrics.bias_severity
        detected = (
            bool(metrics.valid_sample_count)
            and severity is not None
            and severity >= self.length_bias_threshold
        )
        self._length_bias_analysis_cache.update(
            {
                "detected": detected,
                "threshold": self.length_bias_threshold,
                "corr_threshold": self.length_bias_corr_threshold,
                "metrics": metrics.to_dict(),
                "recommendations": list(metrics.recommendations),
                "sample_count": metrics.valid_sample_count,
            }
        )

        self.current_metrics.length_bias_score = severity
        self.current_metrics.length_reward_correlation = metrics.pearson_correlation
        self.current_metrics.length_reward_spearman = metrics.spearman_correlation
        self.current_metrics.length_reward_correlation_abs = (
            abs(metrics.pearson_correlation) if metrics.pearson_correlation is not None else None
        )
        self.current_metrics.length_reward_spearman_abs = (
            abs(metrics.spearman_correlation)
            if metrics.spearman_correlation is not None
            else None
        )
        self.current_metrics.mean_response_length = metrics.mean_length
        self.current_metrics.length_bias_detected = detected
        self.current_metrics.length_bias_threshold = self.length_bias_threshold
        self.current_metrics.length_bias_corr_threshold = self.length_bias_corr_threshold
        self.current_metrics.length_bias_metrics = metrics

    def _update_length_bias_metrics(self) -> None:
        """Recompute cached length bias metrics when new samples arrive."""

        if not self.enable_length_bias_detection or not self._length_bias_detector:
            self._reset_length_bias_metrics()
            self._length_bias_dirty = False
            return

        if not self._response_records:
            self._length_bias_metrics_cache = LengthBiasMetrics()
            self._length_bias_dirty = False
            self._apply_length_bias_metrics()
            return

        responses: List[str] = []
        rewards: List[float] = []
        lengths: List[Optional[float]] = []
        include_lengths = False
        for record in self._response_records:
            responses.append(record.get("response", ""))
            rewards.append(float(record["reward"]))
            length_value = record.get("length")
            if length_value is not None:
                include_lengths = True
                lengths.append(float(length_value))
            else:
                lengths.append(None)

        length_inputs: Optional[List[Optional[float]]] = None
        if include_lengths:
            normalized_lengths: List[Optional[float]] = []
            for value in lengths:
                if value is None:
                    normalized_lengths.append(None)
                else:
                    normalized_lengths.append(float(value))
            length_inputs = normalized_lengths

        metrics = self._length_bias_detector.analyze_length_bias(
            responses,
            rewards,
            lengths=length_inputs,
        )
        self._length_bias_metrics_cache = metrics
        self._length_bias_dirty = False
        self._apply_length_bias_metrics()

    def save_analysis(self, output_path: str):
        """Save comprehensive analysis to file."""
        analysis = self.get_comprehensive_analysis()

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"💾 Comprehensive PPO analysis saved to: {output_path}")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a concise health summary."""
        if not self.comprehensive_metrics_history:
            return {"status": "no_data"}

        current = self.current_metrics
        anomalies = self.get_anomalies()

        # Categorize anomalies by severity
        critical_anomalies = [a for a in anomalies if a.get("severity") == "critical"]
        warning_anomalies = [a for a in anomalies if a.get("severity") == "warning"]

        # Determine overall status
        if critical_anomalies:
            status = "critical"
        elif warning_anomalies:
            status = "warning"
        elif current.overall_health_score < 0.7:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "overall_health_score": current.overall_health_score,
            "training_stability_score": current.training_stability_score,
            "convergence_quality_score": current.convergence_quality_score,
            "total_anomalies": len(anomalies),
            "critical_anomalies": len(critical_anomalies),
            "warning_anomalies": len(warning_anomalies),
            "current_kl": current.kl,
            "current_kl_coef": current.kl_coef,
            "current_reward_mean": current.reward_mean,
            "current_entropy": current.entropy,
            "kl_drift": self.get_kl_drift_analysis() if self.enable_kl_drift_tracking else {},
            "length_bias": self.get_length_bias_analysis()
            if self.enable_length_bias_detection and self._length_bias_detector
            else {},
        }
