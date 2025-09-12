"""Comprehensive GRPO forensics with advanced tracking and analysis."""

import numpy as np
from typing import Dict, Any, Iterator, List, Optional
from dataclasses import dataclass
import json
import time
import copy

from .comprehensive_ppo_forensics import ComprehensivePPOForensics, ComprehensivePPOMetrics
from .kl_schedule_tracker import KLScheduleTracker, KLScheduleMetrics
from .gradient_norms_analyzer import GradientNormsAnalyzer, GradientNormsMetrics
from .advantage_statistics_tracker import AdvantageStatisticsTracker, AdvantageStatisticsMetrics


@dataclass
class ComprehensiveGRPOMetrics:
    """Container for comprehensive GRPO metrics."""
    
    # Basic GRPO metrics (inherits from PPO)
    step: int = 0
    kl: float = 0.0
    kl_coef: float = 1.0
    entropy: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    
    # GRPO-specific metrics
    pass_rate: float = 0.0
    verifier_score: float = 0.0
    verifier_confidence: float = 0.0
    reward_model_score: float = 0.0
    policy_reward: float = 0.0
    reference_reward: float = 0.0
    
    # Advanced tracking metrics (inherits from PPO)
    kl_schedule_metrics: Optional[KLScheduleMetrics] = None
    gradient_norms_metrics: Optional[GradientNormsMetrics] = None
    advantage_statistics_metrics: Optional[AdvantageStatisticsMetrics] = None
    
    # GRPO-specific health scores
    verifier_health_score: float = 1.0
    reward_model_health_score: float = 1.0
    pass_rate_consistency_score: float = 1.0
    gaming_detection_score: float = 1.0
    
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
            "pass_rate": self.pass_rate,
            "verifier_score": self.verifier_score,
            "verifier_confidence": self.verifier_confidence,
            "reward_model_score": self.reward_model_score,
            "policy_reward": self.policy_reward,
            "reference_reward": self.reference_reward,
            "verifier_health_score": self.verifier_health_score,
            "reward_model_health_score": self.reward_model_health_score,
            "pass_rate_consistency_score": self.pass_rate_consistency_score,
            "gaming_detection_score": self.gaming_detection_score,
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


class ComprehensiveGRPOForensics:
    """Comprehensive GRPO forensics with advanced tracking and analysis."""
    
    def __init__(
        self,
        kl_target: float = 0.1,
        kl_target_tolerance: float = 0.05,
        window_size: int = 100,
        enable_kl_schedule_tracking: bool = True,
        enable_gradient_norms_analysis: bool = True,
        enable_advantage_statistics: bool = True,
        enable_verifier_tracking: bool = True,
        enable_reward_model_tracking: bool = True,
        enable_gaming_detection: bool = True,
    ):
        """Initialize comprehensive GRPO forensics.
        
        Args:
            kl_target: Target KL divergence value
            kl_target_tolerance: Tolerance around target for "in range" calculation
            window_size: Size of rolling window for analysis
            enable_kl_schedule_tracking: Enable KL schedule tracking
            enable_gradient_norms_analysis: Enable gradient norms analysis
            enable_advantage_statistics: Enable advantage statistics tracking
            enable_verifier_tracking: Enable verifier health tracking
            enable_reward_model_tracking: Enable reward model health tracking
            enable_gaming_detection: Enable gaming detection
        """
        self.kl_target = kl_target
        self.kl_target_tolerance = kl_target_tolerance
        self.window_size = window_size
        
        # Initialize PPO trackers (reuse existing functionality)
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
        
        # GRPO-specific tracking
        self.enable_verifier_tracking = enable_verifier_tracking
        self.enable_reward_model_tracking = enable_reward_model_tracking
        self.enable_gaming_detection = enable_gaming_detection
        
        # GRPO-specific state
        self.verifier_scores_history: List[float] = []
        self.reward_model_scores_history: List[float] = []
        self.pass_rates_history: List[float] = []
        self.policy_rewards_history: List[float] = []
        self.reference_rewards_history: List[float] = []
        
        # Metrics storage
        self.comprehensive_metrics_history: List[ComprehensiveGRPOMetrics] = []
        self.current_metrics = ComprehensiveGRPOMetrics()
        
        # Analysis results
        self.anomalies: List[Dict[str, Any]] = []
        self.analysis_summary: Dict[str, Any] = {}
        
        print(f"🔍 Comprehensive GRPO Forensics initialized")
        print(f"   KL Schedule Tracking: {enable_kl_schedule_tracking}")
        print(f"   Gradient Norms Analysis: {enable_gradient_norms_analysis}")
        print(f"   Advantage Statistics: {enable_advantage_statistics}")
        print(f"   Verifier Tracking: {enable_verifier_tracking}")
        print(f"   Reward Model Tracking: {enable_reward_model_tracking}")
        print(f"   Gaming Detection: {enable_gaming_detection}")
    
    def update(
        self,
        step: int,
        kl: float,
        kl_coef: float,
        entropy: float,
        reward_mean: float,
        reward_std: float,
        pass_rate: float,
        verifier_score: Optional[float] = None,
        verifier_confidence: Optional[float] = None,
        reward_model_score: Optional[float] = None,
        policy_reward: Optional[float] = None,
        reference_reward: Optional[float] = None,
        policy_grad_norm: Optional[float] = None,
        value_grad_norm: Optional[float] = None,
        total_grad_norm: Optional[float] = None,
        advantage_mean: Optional[float] = None,
        advantage_std: Optional[float] = None,
        advantage_min: Optional[float] = None,
        advantage_max: Optional[float] = None,
        advantage_median: Optional[float] = None,
        advantage_samples: Optional[List[float]] = None,
    ) -> ComprehensiveGRPOMetrics:
        """Update forensics with new GRPO training data."""
        # Update basic metrics
        self.current_metrics.step = step
        self.current_metrics.kl = kl
        self.current_metrics.kl_coef = kl_coef
        self.current_metrics.entropy = entropy
        self.current_metrics.reward_mean = reward_mean
        self.current_metrics.reward_std = reward_std
        
        # Update GRPO-specific metrics
        self.current_metrics.pass_rate = pass_rate
        self.current_metrics.verifier_score = verifier_score or 0.0
        self.current_metrics.verifier_confidence = verifier_confidence or 0.0
        self.current_metrics.reward_model_score = reward_model_score or 0.0
        self.current_metrics.policy_reward = policy_reward or 0.0
        self.current_metrics.reference_reward = reference_reward or 0.0
        
        # Update PPO trackers (reuse existing functionality)
        if self.kl_schedule_tracker:
            kl_schedule_metrics = self.kl_schedule_tracker.update(step, kl, kl_coef)
            self.current_metrics.kl_schedule_metrics = kl_schedule_metrics
        
        if self.gradient_norms_analyzer and policy_grad_norm is not None and value_grad_norm is not None:
            gradient_metrics = self.gradient_norms_analyzer.update(
                step, policy_grad_norm, value_grad_norm, total_grad_norm
            )
            self.current_metrics.gradient_norms_metrics = gradient_metrics
        
        if self.advantage_statistics_tracker and advantage_mean is not None and advantage_std is not None:
            advantage_metrics = self.advantage_statistics_tracker.update(
                step, advantage_mean, advantage_std, advantage_min, advantage_max, 
                advantage_median, advantage_samples
            )
            self.current_metrics.advantage_statistics_metrics = advantage_metrics
        
        # Update GRPO-specific history
        self.verifier_scores_history.append(self.current_metrics.verifier_score)
        self.reward_model_scores_history.append(self.current_metrics.reward_model_score)
        self.pass_rates_history.append(pass_rate)
        self.policy_rewards_history.append(self.current_metrics.policy_reward)
        self.reference_rewards_history.append(self.current_metrics.reference_reward)
        
        # Keep only recent history
        if len(self.verifier_scores_history) > self.window_size:
            self.verifier_scores_history = self.verifier_scores_history[-self.window_size:]
            self.reward_model_scores_history = self.reward_model_scores_history[-self.window_size:]
            self.pass_rates_history = self.pass_rates_history[-self.window_size:]
            self.policy_rewards_history = self.policy_rewards_history[-self.window_size:]
            self.reference_rewards_history = self.reference_rewards_history[-self.window_size:]
        
        # Calculate GRPO-specific health scores
        self._calculate_grpo_health_scores()
        
        # Calculate overall health scores
        self._calculate_overall_health_scores()
        
        # Store metrics - use deep copy to avoid issues with nested dataclasses
        metrics_copy = copy.deepcopy(self.current_metrics)
        self.comprehensive_metrics_history.append(metrics_copy)
        
        return metrics_copy
    
    def _calculate_grpo_health_scores(self):
        """Calculate GRPO-specific health scores."""
        # Verifier health score
        if self.enable_verifier_tracking and len(self.verifier_scores_history) > 1:
            verifier_variance = np.var(self.verifier_scores_history)
            verifier_trend = np.polyfit(range(len(self.verifier_scores_history)), self.verifier_scores_history, 1)[0]
            
            # Healthy verifier: low variance, stable trend
            verifier_health = 1.0 - min(verifier_variance, 1.0) - min(abs(verifier_trend), 0.5)
            self.current_metrics.verifier_health_score = max(0.0, verifier_health)
        else:
            self.current_metrics.verifier_health_score = 1.0
        
        # Reward model health score
        if self.enable_reward_model_tracking and len(self.reward_model_scores_history) > 1:
            reward_model_variance = np.var(self.reward_model_scores_history)
            reward_model_trend = np.polyfit(range(len(self.reward_model_scores_history)), self.reward_model_scores_history, 1)[0]
            
            # Healthy reward model: low variance, stable trend
            reward_model_health = 1.0 - min(reward_model_variance, 1.0) - min(abs(reward_model_trend), 0.5)
            self.current_metrics.reward_model_health_score = max(0.0, reward_model_health)
        else:
            self.current_metrics.reward_model_health_score = 1.0
        
        # Pass rate consistency score
        if len(self.pass_rates_history) > 1:
            pass_rate_variance = np.var(self.pass_rates_history)
            pass_rate_trend = np.polyfit(range(len(self.pass_rates_history)), self.pass_rates_history, 1)[0]
            
            # Consistent pass rate: low variance, stable trend
            pass_rate_consistency = 1.0 - min(pass_rate_variance, 1.0) - min(abs(pass_rate_trend), 0.5)
            self.current_metrics.pass_rate_consistency_score = max(0.0, pass_rate_consistency)
        else:
            self.current_metrics.pass_rate_consistency_score = 1.0
        
        # Gaming detection score
        if self.enable_gaming_detection and len(self.pass_rates_history) > 1:
            gaming_score = self._detect_gaming()
            self.current_metrics.gaming_detection_score = gaming_score
        else:
            self.current_metrics.gaming_detection_score = 1.0
    
    def _detect_gaming(self) -> float:
        """Detect potential gaming behavior."""
        if len(self.pass_rates_history) < 10:
            return 1.0
        
        # Check for rising pass rate while external eval proxy declines
        recent_pass_rates = self.pass_rates_history[-10:]
        recent_verifier_scores = self.verifier_scores_history[-10:] if len(self.verifier_scores_history) >= 10 else []
        recent_reward_model_scores = self.reward_model_scores_history[-10:] if len(self.reward_model_scores_history) >= 10 else []
        
        gaming_indicators = 0.0
        
        # Indicator 1: Rising pass rate while verifier score declines
        if len(recent_verifier_scores) >= 5:
            pass_rate_trend = np.polyfit(range(len(recent_pass_rates)), recent_pass_rates, 1)[0]
            verifier_trend = np.polyfit(range(len(recent_verifier_scores)), recent_verifier_scores, 1)[0]
            
            if pass_rate_trend > 0.01 and verifier_trend < -0.01:
                gaming_indicators += 0.3
        
        # Indicator 2: Rising pass rate while reward model score declines
        if len(recent_reward_model_scores) >= 5:
            pass_rate_trend = np.polyfit(range(len(recent_pass_rates)), recent_pass_rates, 1)[0]
            reward_model_trend = np.polyfit(range(len(recent_reward_model_scores)), recent_reward_model_scores, 1)[0]
            
            if pass_rate_trend > 0.01 and reward_model_trend < -0.01:
                gaming_indicators += 0.3
        
        # Indicator 3: Unstable KL dispersion with entropy collapse
        if len(self.comprehensive_metrics_history) >= 10:
            recent_kls = [m.kl for m in self.comprehensive_metrics_history[-10:]]
            recent_entropies = [m.entropy for m in self.comprehensive_metrics_history[-10:]]
            
            kl_variance = np.var(recent_kls)
            entropy_trend = np.polyfit(range(len(recent_entropies)), recent_entropies, 1)[0]
            
            if kl_variance > 0.1 and entropy_trend < -0.01:
                gaming_indicators += 0.2
        
        # Indicator 4: Verifier correlation breaks
        if len(recent_verifier_scores) >= 5 and len(recent_pass_rates) >= 5:
            correlation = np.corrcoef(recent_verifier_scores, recent_pass_rates)[0, 1]
            if correlation < -0.5:  # Strong negative correlation
                gaming_indicators += 0.2
        
        # Convert to health score (higher is better)
        gaming_detection_score = max(0.0, 1.0 - gaming_indicators)
        
        return gaming_detection_score
    
    def _calculate_overall_health_scores(self):
        """Calculate overall health scores from all trackers."""
        health_scores = []
        stability_scores = []
        convergence_scores = []
        
        # PPO health scores
        if self.current_metrics.kl_schedule_metrics:
            health_scores.append(self.current_metrics.kl_schedule_metrics.kl_health_score)
            health_scores.append(self.current_metrics.kl_schedule_metrics.schedule_health_score)
            stability_scores.append(self.current_metrics.kl_schedule_metrics.target_range_stability)
        
        if self.current_metrics.gradient_norms_metrics:
            health_scores.append(self.current_metrics.gradient_norms_metrics.gradient_health_score)
            stability_scores.append(self.current_metrics.gradient_norms_metrics.training_stability)
        
        if self.current_metrics.advantage_statistics_metrics:
            health_scores.append(self.current_metrics.advantage_statistics_metrics.advantage_health_score)
            convergence_scores.append(self.current_metrics.advantage_statistics_metrics.advantage_quality_score)
        
        # GRPO-specific health scores
        health_scores.append(self.current_metrics.verifier_health_score)
        health_scores.append(self.current_metrics.reward_model_health_score)
        health_scores.append(self.current_metrics.pass_rate_consistency_score)
        health_scores.append(self.current_metrics.gaming_detection_score)
        
        # Calculate overall scores
        self.current_metrics.overall_health_score = np.mean(health_scores) if health_scores else 1.0
        self.current_metrics.training_stability_score = np.mean(stability_scores) if stability_scores else 1.0
        self.current_metrics.convergence_quality_score = np.mean(convergence_scores) if convergence_scores else 1.0
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get all detected anomalies from all trackers."""
        all_anomalies = []
        
        # PPO anomalies
        if self.kl_schedule_tracker:
            kl_anomalies = self.kl_schedule_tracker.get_anomalies()
            for anomaly in kl_anomalies:
                anomaly["tracker"] = "kl_schedule"
            all_anomalies.extend(kl_anomalies)
        
        if self.gradient_norms_analyzer:
            grad_anomalies = self.gradient_norms_analyzer.get_anomalies()
            for anomaly in grad_anomalies:
                anomaly["tracker"] = "gradient_norms"
            all_anomalies.extend(grad_anomalies)
        
        if self.advantage_statistics_tracker:
            adv_anomalies = self.advantage_statistics_tracker.get_anomalies()
            for anomaly in adv_anomalies:
                anomaly["tracker"] = "advantage_statistics"
            all_anomalies.extend(adv_anomalies)
        
        # GRPO-specific anomalies
        grpo_anomalies = self._detect_grpo_anomalies()
        for anomaly in grpo_anomalies:
            anomaly["tracker"] = "grpo_specific"
        all_anomalies.extend(grpo_anomalies)
        
        # Store anomalies
        self.anomalies = all_anomalies
        
        return all_anomalies
    
    def _detect_grpo_anomalies(self) -> List[Dict[str, Any]]:
        """Detect GRPO-specific anomalies."""
        anomalies = []
        
        if len(self.comprehensive_metrics_history) < 5:
            return anomalies
        
        current = self.current_metrics
        
        # Anomaly 1: Rising pass rate while external eval proxy declines
        if self.enable_verifier_tracking and len(self.verifier_scores_history) >= 5:
            recent_pass_rates = self.pass_rates_history[-5:]
            recent_verifier_scores = self.verifier_scores_history[-5:]
            
            pass_rate_trend = np.polyfit(range(len(recent_pass_rates)), recent_pass_rates, 1)[0]
            verifier_trend = np.polyfit(range(len(recent_verifier_scores)), recent_verifier_scores, 1)[0]
            
            if pass_rate_trend > 0.01 and verifier_trend < -0.01:
                anomalies.append({
                    "type": "verifier_gaming",
                    "severity": "warning",
                    "description": "Rising pass rate while verifier score declines",
                    "pass_rate_trend": pass_rate_trend,
                    "verifier_trend": verifier_trend,
                    "step": current.step
                })
        
        # Anomaly 2: Unstable KL dispersion with entropy collapse
        if len(self.comprehensive_metrics_history) >= 5:
            recent_kls = [m.kl for m in self.comprehensive_metrics_history[-5:]]
            recent_entropies = [m.entropy for m in self.comprehensive_metrics_history[-5:]]
            
            kl_variance = np.var(recent_kls)
            entropy_trend = np.polyfit(range(len(recent_entropies)), recent_entropies, 1)[0]
            
            if kl_variance > 0.1 and entropy_trend < -0.01:
                anomalies.append({
                    "type": "kl_entropy_instability",
                    "severity": "warning",
                    "description": "Unstable KL dispersion with entropy collapse",
                    "kl_variance": kl_variance,
                    "entropy_trend": entropy_trend,
                    "step": current.step
                })
        
        # Anomaly 3: Verifier correlation breaks
        if self.enable_verifier_tracking and len(self.verifier_scores_history) >= 5:
            recent_verifier_scores = self.verifier_scores_history[-5:]
            recent_pass_rates = self.pass_rates_history[-5:]
            
            correlation = np.corrcoef(recent_verifier_scores, recent_pass_rates)[0, 1]
            
            if correlation < -0.5:
                anomalies.append({
                    "type": "verifier_correlation_break",
                    "severity": "critical",
                    "description": "Verifier correlation breaks, reward increases with no matching pass rate lift",
                    "correlation": correlation,
                    "step": current.step
                })
        
        # Anomaly 4: Reward model health degradation
        if self.enable_reward_model_tracking and len(self.reward_model_scores_history) >= 5:
            recent_reward_model_scores = self.reward_model_scores_history[-5:]
            reward_model_variance = np.var(recent_reward_model_scores)
            reward_model_trend = np.polyfit(range(len(recent_reward_model_scores)), recent_reward_model_scores, 1)[0]
            
            if reward_model_variance > 0.1 or abs(reward_model_trend) > 0.05:
                anomalies.append({
                    "type": "reward_model_degradation",
                    "severity": "warning",
                    "description": "Reward model health degradation detected",
                    "variance": reward_model_variance,
                    "trend": reward_model_trend,
                    "step": current.step
                })
        
        # Anomaly 5: Pass rate inconsistency
        if len(self.pass_rates_history) >= 5:
            recent_pass_rates = self.pass_rates_history[-5:]
            pass_rate_variance = np.var(recent_pass_rates)
            
            if pass_rate_variance > 0.1:
                anomalies.append({
                    "type": "pass_rate_inconsistency",
                    "severity": "warning",
                    "description": "High pass rate variance detected",
                    "variance": pass_rate_variance,
                    "step": current.step
                })
        
        return anomalies
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive GRPO analysis results."""
        analysis = {
            "version": "2.0",
            "timestamp": time.time(),
            "total_steps": len(self.comprehensive_metrics_history),
            "overall_health_score": self.current_metrics.overall_health_score,
            "training_stability_score": self.current_metrics.training_stability_score,
            "convergence_quality_score": self.current_metrics.convergence_quality_score,
            "verifier_health_score": self.current_metrics.verifier_health_score,
            "reward_model_health_score": self.current_metrics.reward_model_health_score,
            "pass_rate_consistency_score": self.current_metrics.pass_rate_consistency_score,
            "gaming_detection_score": self.current_metrics.gaming_detection_score,
            "anomalies": self.get_anomalies(),
            "trackers": {}
        }
        
        # PPO tracker analysis
        if self.kl_schedule_tracker:
            analysis["trackers"]["kl_schedule"] = self.kl_schedule_tracker.get_summary()
        
        if self.gradient_norms_analyzer:
            analysis["trackers"]["gradient_norms"] = self.gradient_norms_analyzer.get_summary()
        
        if self.advantage_statistics_tracker:
            analysis["trackers"]["advantage_statistics"] = self.advantage_statistics_tracker.get_summary()
        
        # GRPO-specific analysis
        analysis["trackers"]["grpo_specific"] = {
            "verifier_tracking_enabled": self.enable_verifier_tracking,
            "reward_model_tracking_enabled": self.enable_reward_model_tracking,
            "gaming_detection_enabled": self.enable_gaming_detection,
            "verifier_scores_history_length": len(self.verifier_scores_history),
            "reward_model_scores_history_length": len(self.reward_model_scores_history),
            "pass_rates_history_length": len(self.pass_rates_history),
            "current_pass_rate": self.current_metrics.pass_rate,
            "current_verifier_score": self.current_metrics.verifier_score,
            "current_reward_model_score": self.current_metrics.reward_model_score,
        }
        
        # Store analysis summary
        self.analysis_summary = analysis
        
        return analysis
    
    def save_analysis(self, output_path: str):
        """Save comprehensive GRPO analysis to file."""
        analysis = self.get_comprehensive_analysis()
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"💾 Comprehensive GRPO analysis saved to: {output_path}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a concise GRPO health summary."""
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
            "verifier_health_score": current.verifier_health_score,
            "reward_model_health_score": current.reward_model_health_score,
            "pass_rate_consistency_score": current.pass_rate_consistency_score,
            "gaming_detection_score": current.gaming_detection_score,
            "total_anomalies": len(anomalies),
            "critical_anomalies": len(critical_anomalies),
            "warning_anomalies": len(warning_anomalies),
            "current_kl": current.kl,
            "current_kl_coef": current.kl_coef,
            "current_reward_mean": current.reward_mean,
            "current_entropy": current.entropy,
            "current_pass_rate": current.pass_rate,
            "current_verifier_score": current.verifier_score,
            "current_reward_model_score": current.reward_model_score,
        }