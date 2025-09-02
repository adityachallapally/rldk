"""Comprehensive PPO forensics with advanced tracking and analysis."""

import numpy as np
from typing import Dict, Any, Iterator, List, Optional
from dataclasses import dataclass
import json
import time

from .kl_schedule_tracker import KLScheduleTracker, KLScheduleMetrics
from .gradient_norms_analyzer import GradientNormsAnalyzer, GradientNormsMetrics
from .advantage_statistics_tracker import AdvantageStatisticsTracker, AdvantageStatisticsMetrics
from .ppo_scan import scan_ppo_events


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
    ):
        """Initialize comprehensive PPO forensics.
        
        Args:
            kl_target: Target KL divergence value
            kl_target_tolerance: Tolerance around target for "in range" calculation
            window_size: Size of rolling window for analysis
            enable_kl_schedule_tracking: Enable KL schedule tracking
            enable_gradient_norms_analysis: Enable gradient norms analysis
            enable_advantage_statistics: Enable advantage statistics tracking
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
                window_size=window_size
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
        
        # Analysis results
        self.anomalies: List[Dict[str, Any]] = []
        self.analysis_summary: Dict[str, Any] = {}
        
        print(f"🔍 Comprehensive PPO Forensics initialized")
        print(f"   KL Schedule Tracking: {enable_kl_schedule_tracking}")
        print(f"   Gradient Norms Analysis: {enable_gradient_norms_analysis}")
        print(f"   Advantage Statistics: {enable_advantage_statistics}")
    
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
        
        # Calculate overall health scores
        self._calculate_overall_health_scores()
        
        # Store metrics
        metrics_copy = ComprehensivePPOMetrics(**self.current_metrics.to_dict())
        self.comprehensive_metrics_history.append(metrics_copy)
        
        return metrics_copy
    
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
            analysis["trackers"]["kl_schedule"] = self.kl_schedule_tracker.get_summary()
        
        # Gradient norms analysis
        if self.gradient_norms_analyzer:
            analysis["trackers"]["gradient_norms"] = self.gradient_norms_analyzer.get_summary()
        
        # Advantage statistics analysis
        if self.advantage_statistics_tracker:
            analysis["trackers"]["advantage_statistics"] = self.advantage_statistics_tracker.get_summary()
        
        # Store analysis summary
        self.analysis_summary = analysis
        
        return analysis
    
    def scan_ppo_events_comprehensive(self, events: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced PPO scan with comprehensive tracking."""
        # First run the original PPO scan
        original_scan = scan_ppo_events(events)
        
        # Convert events to list for comprehensive analysis
        events_list = list(events) if hasattr(events, '__iter__') else []
        
        if not events_list:
            return original_scan
        
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
        }