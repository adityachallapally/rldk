"""Tests for RLDK configuration system."""

import os
from unittest.mock import patch

import pytest

from src.rldk.config import (
    ConfigValidator,
    EvaluationConfig,
    ForensicsConfig,
    SuiteConfig,
    VisualizationConfig,
    create_custom_eval_config,
    create_custom_forensics_config,
    create_custom_suite_config,
    create_custom_visualization_config,
    get_eval_config,
    get_forensics_config,
    get_suite_config,
    get_visualization_config,
    print_validation_results,
    validate_all_configs,
)


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = EvaluationConfig()
        assert config.KL_DIVERGENCE_MIN == 0.01
        assert config.KL_DIVERGENCE_MAX == 0.5
        assert config.MEMORY_EFFICIENCY_THRESHOLD == 8.0
        assert config.MIN_SAMPLES_FOR_ANALYSIS == 10

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = EvaluationConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "KL_DIVERGENCE_MIN" in config_dict
        assert "MEMORY_EFFICIENCY_THRESHOLD" in config_dict
        assert config_dict["KL_DIVERGENCE_MIN"] == 0.01

    def test_get_eval_config_default(self):
        """Test getting default evaluation configuration."""
        config = get_eval_config()
        assert isinstance(config, EvaluationConfig)
        assert config.KL_DIVERGENCE_MIN == 0.01

    def test_get_eval_config_strict(self):
        """Test getting strict evaluation configuration."""
        config = get_eval_config("strict")
        assert isinstance(config, EvaluationConfig)
        assert config.KL_DIVERGENCE_MAX == 0.3
        assert config.HIGH_TOXICITY_THRESHOLD == 0.5

    def test_get_eval_config_lenient(self):
        """Test getting lenient evaluation configuration."""
        config = get_eval_config("lenient")
        assert isinstance(config, EvaluationConfig)
        assert config.KL_DIVERGENCE_MAX == 0.8
        assert config.HIGH_TOXICITY_THRESHOLD == 0.9

    def test_get_eval_config_unknown(self):
        """Test getting unknown configuration falls back to default."""
        config = get_eval_config("unknown")
        assert isinstance(config, EvaluationConfig)
        assert config.KL_DIVERGENCE_MIN == 0.01

    def test_create_custom_eval_config(self):
        """Test creating custom evaluation configuration."""
        config = create_custom_eval_config(
            MIN_SAMPLES_FOR_ANALYSIS=5,
            HIGH_TOXICITY_THRESHOLD=0.8
        )
        assert config.MIN_SAMPLES_FOR_ANALYSIS == 5
        assert config.HIGH_TOXICITY_THRESHOLD == 0.8
        assert config.KL_DIVERGENCE_MIN == 0.01  # Default value

    def test_create_custom_eval_config_invalid_param(self):
        """Test creating custom config with invalid parameter."""
        config = create_custom_eval_config(
            INVALID_PARAMETER=123
        )
        # Should not raise error, just ignore invalid parameter
        assert config.KL_DIVERGENCE_MIN == 0.01


class TestForensicsConfig:
    """Test forensics configuration."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = ForensicsConfig()
        assert config.ADVANTAGE_WINDOW_SIZE == 100
        assert config.GRADIENT_EXPLOSION_THRESHOLD == 10.0
        assert config.MIN_SAMPLES_FOR_STATS == 3

    def test_get_forensics_config_default(self):
        """Test getting default forensics configuration."""
        config = get_forensics_config()
        assert isinstance(config, ForensicsConfig)
        assert config.ADVANTAGE_WINDOW_SIZE == 100

    def test_get_forensics_config_strict(self):
        """Test getting strict forensics configuration."""
        config = get_forensics_config("strict")
        assert isinstance(config, ForensicsConfig)
        assert config.ADVANTAGE_BIAS_THRESHOLD == 0.05
        assert config.GRADIENT_EXPLOSION_THRESHOLD == 5.0

    def test_create_custom_forensics_config(self):
        """Test creating custom forensics configuration."""
        config = create_custom_forensics_config(
            ADVANTAGE_WINDOW_SIZE=50,
            GRADIENT_EXPLOSION_THRESHOLD=5.0
        )
        assert config.ADVANTAGE_WINDOW_SIZE == 50
        assert config.GRADIENT_EXPLOSION_THRESHOLD == 5.0


class TestVisualizationConfig:
    """Test visualization configuration."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = VisualizationConfig()
        assert config.DEFAULT_FIGSIZE == (12, 8)
        assert config.DEFAULT_DPI == 300
        assert config.HISTOGRAM_BINS == 30

    def test_get_visualization_config_default(self):
        """Test getting default visualization configuration."""
        config = get_visualization_config()
        assert isinstance(config, VisualizationConfig)
        assert config.DEFAULT_FIGSIZE == (12, 8)

    def test_get_visualization_config_publication(self):
        """Test getting publication visualization configuration."""
        config = get_visualization_config("publication")
        assert isinstance(config, VisualizationConfig)
        assert config.DEFAULT_DPI == 600
        assert config.TITLE_FONTSIZE == 14

    def test_create_custom_visualization_config(self):
        """Test creating custom visualization configuration."""
        config = create_custom_visualization_config(
            DEFAULT_FIGSIZE=(10, 6),
            HISTOGRAM_BINS=50
        )
        assert config.DEFAULT_FIGSIZE == (10, 6)
        assert config.HISTOGRAM_BINS == 50


class TestSuiteConfig:
    """Test suite configuration."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = SuiteConfig()
        assert config.QUICK_SAMPLE_SIZE == 50
        assert config.COMPREHENSIVE_SAMPLE_SIZE == 200
        assert config.GENERATES_PLOTS is True

    def test_get_suite_config_default(self):
        """Test getting default suite configuration."""
        config = get_suite_config()
        assert isinstance(config, SuiteConfig)
        assert config.QUICK_SAMPLE_SIZE == 50

    def test_get_suite_config_fast(self):
        """Test getting fast suite configuration."""
        config = get_suite_config("fast")
        assert isinstance(config, SuiteConfig)
        assert config.QUICK_SAMPLE_SIZE == 25
        assert config.COMPREHENSIVE_SAMPLE_SIZE == 100

    def test_get_suite_baseline_scores(self):
        """Test getting suite baseline scores."""
        config = SuiteConfig()
        quick_scores = config.get_suite_baseline_scores("quick")
        assert isinstance(quick_scores, dict)
        assert "alignment" in quick_scores
        assert "toxicity" in quick_scores
        assert quick_scores["alignment"] == 0.7

    def test_get_suite_sample_size(self):
        """Test getting suite sample size."""
        config = SuiteConfig()
        sample_size = config.get_suite_sample_size("quick")
        assert sample_size == 50

        sample_size = config.get_suite_sample_size("unknown")
        assert sample_size == 50  # Falls back to quick

    def test_get_suite_runtime_range(self):
        """Test getting suite runtime range."""
        config = SuiteConfig()
        runtime_min, runtime_max = config.get_suite_runtime_range("quick")
        assert runtime_min == 2
        assert runtime_max == 5

        runtime_min, runtime_max = config.get_suite_runtime_range("unknown")
        assert runtime_min == 2  # Falls back to quick
        assert runtime_max == 5


class TestConfigValidator:
    """Test configuration validation."""

    def test_validate_evaluation_config_valid(self):
        """Test validation of valid evaluation configuration."""
        config = EvaluationConfig()
        issues = ConfigValidator.validate_evaluation_config(config)
        assert issues == []

    def test_validate_evaluation_config_invalid_kl_divergence(self):
        """Test validation with invalid KL divergence values."""
        config = EvaluationConfig()
        config.KL_DIVERGENCE_MIN = 0.6
        config.KL_DIVERGENCE_MAX = 0.5  # Invalid: min > max
        issues = ConfigValidator.validate_evaluation_config(config)
        assert len(issues) > 0
        assert any("KL_DIVERGENCE_MIN must be less than KL_DIVERGENCE_MAX" in issue for issue in issues)

    def test_validate_evaluation_config_invalid_memory_threshold(self):
        """Test validation with invalid memory threshold."""
        config = EvaluationConfig()
        config.MEMORY_EFFICIENCY_THRESHOLD = -1.0  # Invalid: negative
        issues = ConfigValidator.validate_evaluation_config(config)
        assert len(issues) > 0
        assert any("MEMORY_EFFICIENCY_THRESHOLD must be positive" in issue for issue in issues)

    def test_validate_evaluation_config_invalid_confidence_calibration(self):
        """Test validation with invalid confidence calibration."""
        config = EvaluationConfig()
        config.CONFIDENCE_CALIBRATION_MIN = 0.9
        config.CONFIDENCE_CALIBRATION_MAX = 0.8  # Invalid: min > max
        issues = ConfigValidator.validate_evaluation_config(config)
        assert len(issues) > 0
        assert any("CONFIDENCE_CALIBRATION_MIN must be less than CONFIDENCE_CALIBRATION_MAX" in issue for issue in issues)

    def test_validate_forensics_config_valid(self):
        """Test validation of valid forensics configuration."""
        config = ForensicsConfig()
        issues = ConfigValidator.validate_forensics_config(config)
        assert issues == []

    def test_validate_forensics_config_invalid_window_size(self):
        """Test validation with invalid window size."""
        config = ForensicsConfig()
        config.ADVANTAGE_WINDOW_SIZE = -1  # Invalid: negative
        issues = ConfigValidator.validate_forensics_config(config)
        assert len(issues) > 0
        assert any("ADVANTAGE_WINDOW_SIZE must be positive" in issue for issue in issues)

    def test_validate_forensics_config_invalid_anomaly_severity(self):
        """Test validation with invalid anomaly severity thresholds."""
        config = ForensicsConfig()
        config.ANOMALY_SEVERITY_INFO = 0.8
        config.ANOMALY_SEVERITY_WARNING = 0.5  # Invalid: info > warning
        issues = ConfigValidator.validate_forensics_config(config)
        assert len(issues) > 0
        assert any("ANOMALY_SEVERITY_INFO must be less than ANOMALY_SEVERITY_WARNING" in issue for issue in issues)

    def test_validate_visualization_config_valid(self):
        """Test validation of valid visualization configuration."""
        config = VisualizationConfig()
        issues = ConfigValidator.validate_visualization_config(config)
        assert issues == []

    def test_validate_visualization_config_invalid_figsize(self):
        """Test validation with invalid figure size."""
        config = VisualizationConfig()
        config.DEFAULT_FIGSIZE = (10, 6, 8)  # Invalid: wrong length
        issues = ConfigValidator.validate_visualization_config(config)
        assert len(issues) > 0
        assert any("DEFAULT_FIGSIZE must be a tuple of length 2" in issue for issue in issues)

    def test_validate_visualization_config_invalid_dpi(self):
        """Test validation with invalid DPI."""
        config = VisualizationConfig()
        config.DEFAULT_DPI = -1  # Invalid: negative
        issues = ConfigValidator.validate_visualization_config(config)
        assert len(issues) > 0
        assert any("DEFAULT_DPI must be positive" in issue for issue in issues)

    def test_validate_suite_config_valid(self):
        """Test validation of valid suite configuration."""
        config = SuiteConfig()
        issues = ConfigValidator.validate_suite_config(config)
        assert issues == []

    def test_validate_suite_config_invalid_sample_size(self):
        """Test validation with invalid sample size."""
        config = SuiteConfig()
        config.QUICK_SAMPLE_SIZE = -1  # Invalid: negative
        issues = ConfigValidator.validate_suite_config(config)
        assert len(issues) > 0
        assert any("QUICK_SAMPLE_SIZE must be positive" in issue for issue in issues)

    def test_validate_suite_config_invalid_runtime(self):
        """Test validation with invalid runtime range."""
        config = SuiteConfig()
        config.QUICK_RUNTIME_MAX = 2
        config.QUICK_RUNTIME_MIN = 5  # Invalid: min > max
        issues = ConfigValidator.validate_suite_config(config)
        assert len(issues) > 0
        assert any("QUICK_RUNTIME_MAX must be greater than QUICK_RUNTIME_MIN" in issue for issue in issues)


class TestValidateAllConfigs:
    """Test validation of all configurations."""

    def test_validate_all_configs_valid(self):
        """Test validation of all valid configurations."""
        eval_config = EvaluationConfig()
        forensics_config = ForensicsConfig()
        visualization_config = VisualizationConfig()
        suite_config = SuiteConfig()

        issues = validate_all_configs(
            eval_config=eval_config,
            forensics_config=forensics_config,
            visualization_config=visualization_config,
            suite_config=suite_config
        )

        assert all(len(config_issues) == 0 for config_issues in issues.values())

    def test_validate_all_configs_partial(self):
        """Test validation of partial configurations."""
        eval_config = EvaluationConfig()
        forensics_config = ForensicsConfig()

        issues = validate_all_configs(
            eval_config=eval_config,
            forensics_config=forensics_config
        )

        assert "evaluation" in issues
        assert "forensics" in issues
        assert "visualization" not in issues
        assert "suite" not in issues
        assert len(issues["evaluation"]) == 0
        assert len(issues["forensics"]) == 0


class TestEnvironmentVariableLoading:
    """Test loading configuration from environment variables."""

    def test_load_eval_config_from_env(self):
        """Test loading evaluation configuration from environment variables."""
        with patch.dict(os.environ, {
            'RLDK_MIN_SAMPLES_FOR_ANALYSIS': '20',
            'RLDK_HIGH_TOXICITY_THRESHOLD': '0.8',
            'RLDK_MEMORY_EFFICIENCY_THRESHOLD': '6.0'
        }):
            from src.rldk.config.evaluation_config import load_config_from_env
            config = load_config_from_env()

            assert config.MIN_SAMPLES_FOR_ANALYSIS == 20
            assert config.HIGH_TOXICITY_THRESHOLD == 0.8
            assert config.MEMORY_EFFICIENCY_THRESHOLD == 6.0
            assert config.KL_DIVERGENCE_MIN == 0.01  # Default value

    def test_load_eval_config_from_env_invalid_type(self):
        """Test loading configuration with invalid environment variable type."""
        with patch.dict(os.environ, {
            'RLDK_MIN_SAMPLES_FOR_ANALYSIS': 'invalid_number'
        }):
            from src.rldk.config.evaluation_config import load_config_from_env
            config = load_config_from_env()

            # Should fall back to default value
            assert config.MIN_SAMPLES_FOR_ANALYSIS == 10

    def test_load_eval_config_from_env_list(self):
        """Test loading configuration with list environment variable."""
        with patch.dict(os.environ, {
            'RLDK_PERCENTILES': '[5, 10, 25, 50, 75, 90, 95]'
        }):
            from src.rldk.config.evaluation_config import load_config_from_env
            config = load_config_from_env()

            assert config.PERCENTILES == [5, 10, 25, 50, 75, 90, 95]

    def test_load_eval_config_from_env_comma_separated(self):
        """Test loading configuration with comma-separated environment variable."""
        with patch.dict(os.environ, {
            'RLDK_PERCENTILES': '5,10,25,50,75,90,95'
        }):
            from src.rldk.config.evaluation_config import load_config_from_env
            config = load_config_from_env()

            assert config.PERCENTILES == [5, 10, 25, 50, 75, 90, 95]


class TestPrintValidationResults:
    """Test printing validation results."""

    def test_print_validation_results_no_issues(self, capsys):
        """Test printing validation results with no issues."""
        issues = {
            "evaluation": [],
            "forensics": [],
            "visualization": [],
            "suite": []
        }

        print_validation_results(issues)
        captured = capsys.readouterr()
        assert "✅ All configurations are valid!" in captured.out

    def test_print_validation_results_with_issues(self, capsys):
        """Test printing validation results with issues."""
        issues = {
            "evaluation": ["KL_DIVERGENCE_MIN must be less than KL_DIVERGENCE_MAX"],
            "forensics": [],
            "visualization": ["DEFAULT_DPI must be positive"],
            "suite": []
        }

        print_validation_results(issues)
        captured = capsys.readouterr()

        expected_outputs = {
            "evaluation": ("❌", "configuration issues"),
            "forensics": ("✅", "configuration is valid"),
            "visualization": ("❌", "configuration issues"),
            "suite": ("✅", "configuration is valid"),
        }

        for key, (status, suffix) in expected_outputs.items():
            expected = f"{status} {key.title()} {suffix}:" if "issues" in suffix else f"{status} {key.title()} {suffix}"
            assert expected in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
