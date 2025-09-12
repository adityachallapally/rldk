"""Unit tests for GRPO forensics functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from rldk.forensics import ComprehensiveGRPOForensics, ComprehensiveGRPOMetrics


class TestComprehensiveGRPOMetrics:
    """Test ComprehensiveGRPOMetrics dataclass."""
    
    def test_grpo_metrics_creation(self):
        """Test creating GRPO metrics."""
        metrics = ComprehensiveGRPOMetrics(
            step=100,
            kl=0.05,
            kl_coef=1.0,
            entropy=2.0,
            reward_mean=0.8,
            reward_std=0.1,
            pass_rate=0.7,
            verifier_score=0.85,
            verifier_confidence=0.9,
            reward_model_score=0.75,
            policy_reward=0.8,
            reference_reward=0.6
        )
        
        assert metrics.step == 100
        assert metrics.kl == 0.05
        assert metrics.pass_rate == 0.7
        assert metrics.verifier_score == 0.85
        assert metrics.reward_model_score == 0.75
    
    def test_grpo_metrics_to_dict(self):
        """Test converting GRPO metrics to dictionary."""
        metrics = ComprehensiveGRPOMetrics(
            step=100,
            kl=0.05,
            pass_rate=0.7,
            verifier_score=0.85,
            reward_model_score=0.75
        )
        
        result = metrics.to_dict()
        
        assert "step" in result
        assert "kl" in result
        assert "pass_rate" in result
        assert "verifier_score" in result
        assert "reward_model_score" in result
        assert "verifier_health_score" in result
        assert "reward_model_health_score" in result
        assert "gaming_detection_score" in result


class TestComprehensiveGRPOForensics:
    """Test ComprehensiveGRPOForensics class."""
    
    def test_grpo_forensics_initialization(self):
        """Test GRPO forensics initialization."""
        forensics = ComprehensiveGRPOForensics(
            kl_target=0.1,
            window_size=50,
            enable_verifier_tracking=True,
            enable_reward_model_tracking=True,
            enable_gaming_detection=True
        )
        
        assert forensics.kl_target == 0.1
        assert forensics.window_size == 50
        assert forensics.enable_verifier_tracking is True
        assert forensics.enable_reward_model_tracking is True
        assert forensics.enable_gaming_detection is True
        assert len(forensics.comprehensive_metrics_history) == 0
    
    def test_grpo_forensics_update_basic(self):
        """Test basic GRPO forensics update."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=False,
            enable_reward_model_tracking=False,
            enable_gaming_detection=False
        )
        
        metrics = forensics.update(
            step=1,
            kl=0.05,
            kl_coef=1.0,
            entropy=2.0,
            reward_mean=0.8,
            reward_std=0.1,
            pass_rate=0.7
        )
        
        assert metrics.step == 1
        assert metrics.kl == 0.05
        assert metrics.pass_rate == 0.7
        assert len(forensics.comprehensive_metrics_history) == 1
    
    def test_grpo_forensics_update_with_verifier(self):
        """Test GRPO forensics update with verifier tracking."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=True,
            enable_reward_model_tracking=False,
            enable_gaming_detection=False
        )
        
        metrics = forensics.update(
            step=1,
            kl=0.05,
            kl_coef=1.0,
            entropy=2.0,
            reward_mean=0.8,
            reward_std=0.1,
            pass_rate=0.7,
            verifier_score=0.85,
            verifier_confidence=0.9
        )
        
        assert metrics.verifier_score == 0.85
        assert metrics.verifier_confidence == 0.9
        assert len(forensics.verifier_scores_history) == 1
    
    def test_grpo_forensics_update_with_reward_model(self):
        """Test GRPO forensics update with reward model tracking."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=False,
            enable_reward_model_tracking=True,
            enable_gaming_detection=False
        )
        
        metrics = forensics.update(
            step=1,
            kl=0.05,
            kl_coef=1.0,
            entropy=2.0,
            reward_mean=0.8,
            reward_std=0.1,
            pass_rate=0.7,
            reward_model_score=0.75,
            policy_reward=0.8,
            reference_reward=0.6
        )
        
        assert metrics.reward_model_score == 0.75
        assert metrics.policy_reward == 0.8
        assert metrics.reference_reward == 0.6
        assert len(forensics.reward_model_scores_history) == 1
    
    def test_grpo_forensics_gaming_detection(self):
        """Test GRPO gaming detection."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=True,
            enable_reward_model_tracking=True,
            enable_gaming_detection=True
        )
        
        # Simulate gaming behavior: rising pass rate while verifier score declines
        for i in range(10):
            forensics.update(
                step=i,
                kl=0.05,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.8,
                reward_std=0.1,
                pass_rate=0.5 + i * 0.02,  # Rising pass rate
                verifier_score=0.9 - i * 0.01,  # Declining verifier score
                reward_model_score=0.75,
                policy_reward=0.8,
                reference_reward=0.6
            )
        
        # Check gaming detection score
        current_metrics = forensics.current_metrics
        assert current_metrics.gaming_detection_score < 1.0  # Should detect gaming
    
    def test_grpo_forensics_anomaly_detection(self):
        """Test GRPO-specific anomaly detection."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=True,
            enable_reward_model_tracking=True,
            enable_gaming_detection=True
        )
        
        # Simulate normal training
        for i in range(5):
            forensics.update(
                step=i,
                kl=0.05,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.8,
                reward_std=0.1,
                pass_rate=0.7,
                verifier_score=0.85,
                reward_model_score=0.75
            )
        
        # Simulate gaming behavior
        for i in range(5, 10):
            forensics.update(
                step=i,
                kl=0.05,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.8,
                reward_std=0.1,
                pass_rate=0.7 + (i - 5) * 0.02,  # Rising pass rate
                verifier_score=0.85 - (i - 5) * 0.01,  # Declining verifier score
                reward_model_score=0.75
            )
        
        anomalies = forensics.get_anomalies()
        
        # Should detect gaming anomalies
        gaming_anomalies = [a for a in anomalies if a.get("tracker") == "grpo_specific"]
        assert len(gaming_anomalies) > 0
        
        # Check for specific anomaly types
        anomaly_types = [a.get("type") for a in gaming_anomalies]
        assert "verifier_gaming" in anomaly_types or "verifier_correlation_break" in anomaly_types
    
    def test_grpo_forensics_comprehensive_analysis(self):
        """Test comprehensive GRPO analysis."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=True,
            enable_reward_model_tracking=True,
            enable_gaming_detection=True
        )
        
        # Add some training data
        for i in range(10):
            forensics.update(
                step=i,
                kl=0.05,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.8,
                reward_std=0.1,
                pass_rate=0.7,
                verifier_score=0.85,
                reward_model_score=0.75
            )
        
        analysis = forensics.get_comprehensive_analysis()
        
        assert "version" in analysis
        assert "total_steps" in analysis
        assert "overall_health_score" in analysis
        assert "verifier_health_score" in analysis
        assert "reward_model_health_score" in analysis
        assert "gaming_detection_score" in analysis
        assert "anomalies" in analysis
        assert "trackers" in analysis
        
        # Check GRPO-specific tracker
        assert "grpo_specific" in analysis["trackers"]
        grpo_tracker = analysis["trackers"]["grpo_specific"]
        assert "verifier_tracking_enabled" in grpo_tracker
        assert "reward_model_tracking_enabled" in grpo_tracker
        assert "gaming_detection_enabled" in grpo_tracker
    
    def test_grpo_forensics_health_summary(self):
        """Test GRPO health summary."""
        forensics = ComprehensiveGRPOForensics(
            enable_verifier_tracking=True,
            enable_reward_model_tracking=True,
            enable_gaming_detection=True
        )
        
        # Add some training data
        for i in range(10):
            forensics.update(
                step=i,
                kl=0.05,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.8,
                reward_std=0.1,
                pass_rate=0.7,
                verifier_score=0.85,
                reward_model_score=0.75
            )
        
        summary = forensics.get_health_summary()
        
        assert "status" in summary
        assert "overall_health_score" in summary
        assert "verifier_health_score" in summary
        assert "reward_model_health_score" in summary
        assert "gaming_detection_score" in summary
        assert "current_pass_rate" in summary
        assert "current_verifier_score" in summary
        assert "current_reward_model_score" in summary
    
    def test_grpo_forensics_window_size_limit(self):
        """Test that GRPO forensics respects window size limits."""
        forensics = ComprehensiveGRPOForensics(
            window_size=5,
            enable_verifier_tracking=True,
            enable_reward_model_tracking=True,
            enable_gaming_detection=True
        )
        
        # Add more data than window size
        for i in range(10):
            forensics.update(
                step=i,
                kl=0.05,
                kl_coef=1.0,
                entropy=2.0,
                reward_mean=0.8,
                reward_std=0.1,
                pass_rate=0.7,
                verifier_score=0.85,
                reward_model_score=0.75
            )
        
        # Check that history is limited to window size
        assert len(forensics.verifier_scores_history) <= 5
        assert len(forensics.reward_model_scores_history) <= 5
        assert len(forensics.pass_rates_history) <= 5
        assert len(forensics.policy_rewards_history) <= 5
        assert len(forensics.reference_rewards_history) <= 5
    
    def test_grpo_forensics_empty_data(self):
        """Test GRPO forensics with no data."""
        forensics = ComprehensiveGRPOForensics()
        
        summary = forensics.get_health_summary()
        assert summary["status"] == "no_data"
        
        anomalies = forensics.get_anomalies()
        assert len(anomalies) == 0
        
        analysis = forensics.get_comprehensive_analysis()
        assert analysis["total_steps"] == 0