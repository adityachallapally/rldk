"""Configuration validation utilities."""

import logging
from typing import Dict, List, Optional

from .evaluation_config import EvaluationConfig
from .forensics_config import ForensicsConfig
from .suite_config import SuiteConfig
from .visualization_config import VisualizationConfig

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validator for configuration objects."""

    @staticmethod
    def validate_evaluation_config(config: EvaluationConfig) -> List[str]:
        """Validate evaluation configuration values and return any issues."""
        issues = []

        # KL Divergence validation
        if config.KL_DIVERGENCE_MIN >= config.KL_DIVERGENCE_MAX:
            issues.append("KL_DIVERGENCE_MIN must be less than KL_DIVERGENCE_MAX")

        if not 0 <= config.KL_DIVERGENCE_TARGET <= 1:
            issues.append("KL_DIVERGENCE_TARGET must be between 0 and 1")

        # Memory thresholds validation
        if config.MEMORY_EFFICIENCY_THRESHOLD <= 0:
            issues.append("MEMORY_EFFICIENCY_THRESHOLD must be positive")

        if config.GPU_MEMORY_EFFICIENCY_THRESHOLD <= 0:
            issues.append("GPU_MEMORY_EFFICIENCY_THRESHOLD must be positive")

        # Gradient thresholds validation
        if config.GRADIENT_STABILITY_THRESHOLD <= 0:
            issues.append("GRADIENT_STABILITY_THRESHOLD must be positive")

        if config.GRADIENT_EXPLOSION_THRESHOLD <= config.GRADIENT_STABILITY_THRESHOLD:
            issues.append("GRADIENT_EXPLOSION_THRESHOLD must be greater than GRADIENT_STABILITY_THRESHOLD")

        # Toxicity thresholds validation
        if not 0 <= config.HIGH_TOXICITY_THRESHOLD <= 1:
            issues.append("HIGH_TOXICITY_THRESHOLD must be between 0 and 1")

        if config.CONFIDENCE_CALIBRATION_MIN >= config.CONFIDENCE_CALIBRATION_MAX:
            issues.append("CONFIDENCE_CALIBRATION_MIN must be less than CONFIDENCE_CALIBRATION_MAX")

        if not 0 <= config.CONFIDENCE_CALIBRATION_MIN <= 1:
            issues.append("CONFIDENCE_CALIBRATION_MIN must be between 0 and 1")

        if not 0 <= config.CONFIDENCE_CALIBRATION_MAX <= 1:
            issues.append("CONFIDENCE_CALIBRATION_MAX must be between 0 and 1")

        # Performance thresholds validation
        if config.INFERENCE_TIME_THRESHOLD <= 0:
            issues.append("INFERENCE_TIME_THRESHOLD must be positive")

        if config.LATENCY_THRESHOLD <= 0:
            issues.append("LATENCY_THRESHOLD must be positive")

        if config.STEPS_PER_SECOND_MAX <= 0:
            issues.append("STEPS_PER_SECOND_MAX must be positive")

        # Consistency thresholds validation
        if not 0 <= config.CV_CONSISTENCY_THRESHOLD <= 1:
            issues.append("CV_CONSISTENCY_THRESHOLD must be between 0 and 1")

        if config.OUTLIER_THRESHOLD_MULTIPLIER <= 0:
            issues.append("OUTLIER_THRESHOLD_MULTIPLIER must be positive")

        # Robustness thresholds validation
        if not 0 <= config.TREND_DEGRADATION_THRESHOLD <= 1:
            issues.append("TREND_DEGRADATION_THRESHOLD must be between 0 and 1")

        if not 0 <= config.MAX_EXPECTED_DEGRADATION <= 1:
            issues.append("MAX_EXPECTED_DEGRADATION must be between 0 and 1")

        if not 0 <= config.LOW_ROBUSTNESS_THRESHOLD <= 1:
            issues.append("LOW_ROBUSTNESS_THRESHOLD must be between 0 and 1")

        # Efficiency thresholds validation
        if not 0 <= config.CONVERGENCE_IMPROVEMENT_THRESHOLD <= 1:
            issues.append("CONVERGENCE_IMPROVEMENT_THRESHOLD must be between 0 and 1")

        if not 0 <= config.EARLY_CONVERGENCE_THRESHOLD <= 1:
            issues.append("EARLY_CONVERGENCE_THRESHOLD must be between 0 and 1")

        if config.FLOP_EFFICIENCY_THRESHOLD <= 0:
            issues.append("FLOP_EFFICIENCY_THRESHOLD must be positive")

        # Calibration thresholds validation
        if config.UNCERTAINTY_CALIBRATION_MIN >= config.UNCERTAINTY_CALIBRATION_MAX:
            issues.append("UNCERTAINTY_CALIBRATION_MIN must be less than UNCERTAINTY_CALIBRATION_MAX")

        if config.ENTROPY_CALIBRATION_MIN >= config.ENTROPY_CALIBRATION_MAX:
            issues.append("ENTROPY_CALIBRATION_MIN must be less than ENTROPY_CALIBRATION_MAX")

        if config.TEMPERATURE_CALIBRATION_MIN >= config.TEMPERATURE_CALIBRATION_MAX:
            issues.append("TEMPERATURE_CALIBRATION_MIN must be less than TEMPERATURE_CALIBRATION_MAX")

        # Sample size validation
        if config.MIN_SAMPLES_FOR_ANALYSIS <= 0:
            issues.append("MIN_SAMPLES_FOR_ANALYSIS must be positive")

        if config.MIN_SAMPLES_FOR_CONSISTENCY <= 0:
            issues.append("MIN_SAMPLES_FOR_CONSISTENCY must be positive")

        if config.MIN_SAMPLES_FOR_DISTRIBUTION <= 0:
            issues.append("MIN_SAMPLES_FOR_DISTRIBUTION must be positive")

        if config.MIN_SAMPLES_FOR_TREND <= 0:
            issues.append("MIN_SAMPLES_FOR_TREND must be positive")

        # Percentile validation
        if config.PERCENTILES:
            for p in config.PERCENTILES:
                if not 0 <= p <= 100:
                    issues.append(f"Percentile {p} must be between 0 and 100")

        # Bootstrap confidence level validation
        if not 0 < config.BOOTSTRAP_CONFIDENCE_LEVEL < 1:
            issues.append("BOOTSTRAP_CONFIDENCE_LEVEL must be between 0 and 1")

        return issues

    @staticmethod
    def validate_forensics_config(config: ForensicsConfig) -> List[str]:
        """Validate forensics configuration values and return any issues."""
        issues = []

        # Window size validation
        if config.ADVANTAGE_WINDOW_SIZE <= 0:
            issues.append("ADVANTAGE_WINDOW_SIZE must be positive")

        if config.ADVANTAGE_TREND_WINDOW <= 0:
            issues.append("ADVANTAGE_TREND_WINDOW must be positive")

        if config.ADVANTAGE_TREND_WINDOW > config.ADVANTAGE_WINDOW_SIZE:
            issues.append("ADVANTAGE_TREND_WINDOW must be less than or equal to ADVANTAGE_WINDOW_SIZE")

        # Threshold validation
        if config.ADVANTAGE_BIAS_THRESHOLD <= 0:
            issues.append("ADVANTAGE_BIAS_THRESHOLD must be positive")

        if config.ADVANTAGE_SCALE_THRESHOLD <= 0:
            issues.append("ADVANTAGE_SCALE_THRESHOLD must be positive")

        if config.GRADIENT_EXPLOSION_THRESHOLD <= 0:
            issues.append("GRADIENT_EXPLOSION_THRESHOLD must be positive")

        if config.GRADIENT_VANISHING_THRESHOLD <= 0:
            issues.append("GRADIENT_VANISHING_THRESHOLD must be positive")

        if config.GRADIENT_STABILITY_THRESHOLD <= 0:
            issues.append("GRADIENT_STABILITY_THRESHOLD must be positive")

        # KL divergence validation
        if config.KL_WINDOW_SIZE <= 0:
            issues.append("KL_WINDOW_SIZE must be positive")

        if not 0 <= config.KL_ANOMALY_THRESHOLD <= 1:
            issues.append("KL_ANOMALY_THRESHOLD must be between 0 and 1")

        # PPO scan validation
        if config.PPO_SCAN_WINDOW_SIZE <= 0:
            issues.append("PPO_SCAN_WINDOW_SIZE must be positive")

        if not 0 <= config.PPO_ANOMALY_THRESHOLD <= 1:
            issues.append("PPO_ANOMALY_THRESHOLD must be between 0 and 1")

        # Checkpoint diff validation
        if config.CKPT_DIFF_THRESHOLD <= 0:
            issues.append("CKPT_DIFF_THRESHOLD must be positive")

        if not 0 < config.CKPT_SIGNIFICANCE_LEVEL < 1:
            issues.append("CKPT_SIGNIFICANCE_LEVEL must be between 0 and 1")

        # Environment audit validation
        if config.ENV_AUDIT_SAMPLE_SIZE <= 0:
            issues.append("ENV_AUDIT_SAMPLE_SIZE must be positive")

        if not 0 < config.ENV_AUDIT_CONFIDENCE_LEVEL < 1:
            issues.append("ENV_AUDIT_CONFIDENCE_LEVEL must be between 0 and 1")

        # Sample size validation
        if config.MIN_SAMPLES_FOR_STATS <= 0:
            issues.append("MIN_SAMPLES_FOR_STATS must be positive")

        if config.MIN_SAMPLES_FOR_DISTRIBUTION <= 0:
            issues.append("MIN_SAMPLES_FOR_DISTRIBUTION must be positive")

        if config.MIN_SAMPLES_FOR_CORRELATION <= 0:
            issues.append("MIN_SAMPLES_FOR_CORRELATION must be positive")

        if config.MIN_SAMPLES_FOR_TREND <= 0:
            issues.append("MIN_SAMPLES_FOR_TREND must be positive")

        # Anomaly detection validation
        if not 0 <= config.ANOMALY_SEVERITY_CRITICAL <= 1:
            issues.append("ANOMALY_SEVERITY_CRITICAL must be between 0 and 1")

        if not 0 <= config.ANOMALY_SEVERITY_WARNING <= 1:
            issues.append("ANOMALY_SEVERITY_WARNING must be between 0 and 1")

        if not 0 <= config.ANOMALY_SEVERITY_INFO <= 1:
            issues.append("ANOMALY_SEVERITY_INFO must be between 0 and 1")

        if config.ANOMALY_SEVERITY_INFO >= config.ANOMALY_SEVERITY_WARNING:
            issues.append("ANOMALY_SEVERITY_INFO must be less than ANOMALY_SEVERITY_WARNING")

        if config.ANOMALY_SEVERITY_WARNING >= config.ANOMALY_SEVERITY_CRITICAL:
            issues.append("ANOMALY_SEVERITY_WARNING must be less than ANOMALY_SEVERITY_CRITICAL")

        # Health scoring weights validation
        total_health_weight = (
            config.HEALTH_WEIGHT_NORMALIZATION +
            config.HEALTH_WEIGHT_BIAS +
            config.HEALTH_WEIGHT_SCALE +
            config.HEALTH_WEIGHT_DISTRIBUTION
        )
        if abs(total_health_weight - 1.0) > 1e-6:
            issues.append("Health scoring weights must sum to 1.0")

        # Quality scoring weights validation
        total_quality_weight = (
            config.QUALITY_WEIGHT_SCALE_STABILITY +
            config.QUALITY_WEIGHT_MEAN_TREND +
            config.QUALITY_WEIGHT_VOLATILITY +
            config.QUALITY_WEIGHT_SKEWNESS
        )
        if abs(total_quality_weight - 1.0) > 1e-6:
            issues.append("Quality scoring weights must sum to 1.0")

        # Bootstrap validation
        if config.BOOTSTRAP_SAMPLES <= 0:
            issues.append("BOOTSTRAP_SAMPLES must be positive")

        if not 0 < config.BOOTSTRAP_CONFIDENCE_LEVEL < 1:
            issues.append("BOOTSTRAP_CONFIDENCE_LEVEL must be between 0 and 1")

        return issues

    @staticmethod
    def validate_visualization_config(config: VisualizationConfig) -> List[str]:
        """Validate visualization configuration values and return any issues."""
        issues = []

        # Figure size validation
        if len(config.DEFAULT_FIGSIZE) != 2:
            issues.append("DEFAULT_FIGSIZE must be a tuple of length 2")
        else:
            width, height = config.DEFAULT_FIGSIZE
            if width <= 0 or height <= 0:
                issues.append("DEFAULT_FIGSIZE dimensions must be positive")

        # DPI validation
        if config.DEFAULT_DPI <= 0:
            issues.append("DEFAULT_DPI must be positive")

        if config.SAVE_DPI <= 0:
            issues.append("SAVE_DPI must be positive")

        # Font size validation
        if config.TITLE_FONTSIZE <= 0:
            issues.append("TITLE_FONTSIZE must be positive")

        if config.LABEL_FONTSIZE <= 0:
            issues.append("LABEL_FONTSIZE must be positive")

        if config.TICK_FONTSIZE <= 0:
            issues.append("TICK_FONTSIZE must be positive")

        if config.LEGEND_FONTSIZE <= 0:
            issues.append("LEGEND_FONTSIZE must be positive")

        if config.TEXT_FONTSIZE <= 0:
            issues.append("TEXT_FONTSIZE must be positive")

        # Alpha validation
        if not 0 <= config.GRID_ALPHA <= 1:
            issues.append("GRID_ALPHA must be between 0 and 1")

        if not 0 <= config.LINE_ALPHA <= 1:
            issues.append("LINE_ALPHA must be between 0 and 1")

        if not 0 <= config.SCATTER_ALPHA <= 1:
            issues.append("SCATTER_ALPHA must be between 0 and 1")

        if not 0 <= config.HISTOGRAM_ALPHA <= 1:
            issues.append("HISTOGRAM_ALPHA must be between 0 and 1")

        # Line width validation
        if config.LINE_WIDTH <= 0:
            issues.append("LINE_WIDTH must be positive")

        if config.TREND_LINE_WIDTH <= 0:
            issues.append("TREND_LINE_WIDTH must be positive")

        if config.HISTOGRAM_LINEWIDTH <= 0:
            issues.append("HISTOGRAM_LINEWIDTH must be positive")

        # Histogram bins validation
        if config.HISTOGRAM_BINS <= 0:
            issues.append("HISTOGRAM_BINS must be positive")

        if config.CALIBRATION_BINS <= 0:
            issues.append("CALIBRATION_BINS must be positive")

        # Scatter size validation
        if config.SCATTER_SIZE <= 0:
            issues.append("SCATTER_SIZE must be positive")

        # Sampling validation
        if config.MAX_POINTS_FOR_PLOT <= 0:
            issues.append("MAX_POINTS_FOR_PLOT must be positive")

        if config.MIN_DATA_POINTS_FOR_PLOT <= 0:
            issues.append("MIN_DATA_POINTS_FOR_PLOT must be positive")

        if config.MIN_DATA_POINTS_FOR_CALIBRATION <= 0:
            issues.append("MIN_DATA_POINTS_FOR_CALIBRATION must be positive")

        if config.MIN_DATA_POINTS_FOR_CORRELATION <= 0:
            issues.append("MIN_DATA_POINTS_FOR_CORRELATION must be positive")

        # Trend line validation
        if config.TREND_POLYFIT_DEGREE < 0:
            issues.append("TREND_POLYFIT_DEGREE must be non-negative")

        return issues

    @staticmethod
    def validate_suite_config(config: SuiteConfig) -> List[str]:
        """Validate suite configuration values and return any issues."""
        issues = []

        # Sample size validation
        if config.QUICK_SAMPLE_SIZE <= 0:
            issues.append("QUICK_SAMPLE_SIZE must be positive")

        if config.COMPREHENSIVE_SAMPLE_SIZE <= 0:
            issues.append("COMPREHENSIVE_SAMPLE_SIZE must be positive")

        if config.SAFETY_SAMPLE_SIZE <= 0:
            issues.append("SAFETY_SAMPLE_SIZE must be positive")

        if config.INTEGRITY_SAMPLE_SIZE <= 0:
            issues.append("INTEGRITY_SAMPLE_SIZE must be positive")

        if config.PERFORMANCE_SAMPLE_SIZE <= 0:
            issues.append("PERFORMANCE_SAMPLE_SIZE must be positive")

        if config.TRUST_SAMPLE_SIZE <= 0:
            issues.append("TRUST_SAMPLE_SIZE must be positive")

        # Runtime validation
        if config.QUICK_RUNTIME_MIN <= 0:
            issues.append("QUICK_RUNTIME_MIN must be positive")

        if config.QUICK_RUNTIME_MAX <= config.QUICK_RUNTIME_MIN:
            issues.append("QUICK_RUNTIME_MAX must be greater than QUICK_RUNTIME_MIN")

        if config.COMPREHENSIVE_RUNTIME_MIN <= 0:
            issues.append("COMPREHENSIVE_RUNTIME_MIN must be positive")

        if config.COMPREHENSIVE_RUNTIME_MAX <= config.COMPREHENSIVE_RUNTIME_MIN:
            issues.append("COMPREHENSIVE_RUNTIME_MAX must be greater than COMPREHENSIVE_RUNTIME_MIN")

        # Baseline score validation
        baseline_score_names = [
            (config.QUICK_ALIGNMENT_BASELINE, "QUICK_ALIGNMENT_BASELINE"),
            (config.QUICK_HELPFULNESS_BASELINE, "QUICK_HELPFULNESS_BASELINE"),
            (config.QUICK_HARMLESSNESS_BASELINE, "QUICK_HARMLESSNESS_BASELINE"),
            (config.QUICK_HALLUCINATION_BASELINE, "QUICK_HALLUCINATION_BASELINE"),
            (config.QUICK_REWARD_ALIGNMENT_BASELINE, "QUICK_REWARD_ALIGNMENT_BASELINE"),
            (config.QUICK_KL_DIVERGENCE_BASELINE, "QUICK_KL_DIVERGENCE_BASELINE"),
            (config.QUICK_PROMPT_CONTAMINATION_BASELINE, "QUICK_PROMPT_CONTAMINATION_BASELINE"),
            (config.QUICK_ANSWER_LEAKAGE_BASELINE, "QUICK_ANSWER_LEAKAGE_BASELINE"),
            (config.QUICK_THROUGHPUT_BASELINE, "QUICK_THROUGHPUT_BASELINE"),
            (config.QUICK_TOXICITY_BASELINE, "QUICK_TOXICITY_BASELINE"),
            (config.QUICK_BIAS_BASELINE, "QUICK_BIAS_BASELINE"),
        ]

        for score, name in baseline_score_names:
            if not 0 <= score <= 1:
                issues.append(f"{name} ({score}) must be between 0 and 1")

        # Timeout validation
        if config.EVALUATION_TIMEOUT_SECONDS <= 0:
            issues.append("EVALUATION_TIMEOUT_SECONDS must be positive")

        if config.RETRY_DELAY_SECONDS <= 0:
            issues.append("RETRY_DELAY_SECONDS must be positive")

        if config.MAX_RETRIES < 0:
            issues.append("MAX_RETRIES must be non-negative")

        # Parallel processing validation
        if config.MAX_PARALLEL_EVALUATIONS <= 0:
            issues.append("MAX_PARALLEL_EVALUATIONS must be positive")

        if config.PARALLEL_CHUNK_SIZE <= 0:
            issues.append("PARALLEL_CHUNK_SIZE must be positive")

        # Cache validation
        if config.CACHE_TTL_SECONDS <= 0:
            issues.append("CACHE_TTL_SECONDS must be positive")

        return issues

def validate_all_configs(
    eval_config: Optional[EvaluationConfig] = None,
    forensics_config: Optional[ForensicsConfig] = None,
    visualization_config: Optional[VisualizationConfig] = None,
    suite_config: Optional[SuiteConfig] = None,
) -> Dict[str, List[str]]:
    """Validate all provided configurations and return issues grouped by config type."""
    issues = {}

    if eval_config:
        issues["evaluation"] = ConfigValidator.validate_evaluation_config(eval_config)

    if forensics_config:
        issues["forensics"] = ConfigValidator.validate_forensics_config(forensics_config)

    if visualization_config:
        issues["visualization"] = ConfigValidator.validate_visualization_config(visualization_config)

    if suite_config:
        issues["suite"] = ConfigValidator.validate_suite_config(suite_config)

    return issues

def print_validation_results(issues: Dict[str, List[str]]) -> None:
    """Print validation results in a formatted way."""
    if not any(issues.values()):
        print("✅ All configurations are valid!")
        return

    for config_type, config_issues in issues.items():
        if config_issues:
            print(f"❌ {config_type.title()} configuration issues:")
            for issue in config_issues:
                print(f"  - {issue}")
        else:
            print(f"✅ {config_type.title()} configuration is valid")
