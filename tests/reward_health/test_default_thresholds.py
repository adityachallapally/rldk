"""Tests for default health thresholds functionality."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from rldk.reward.health_config.config import (
    _deep_merge,
    get_default_config_path,
    get_detector_thresholds,
    get_legacy_thresholds,
    load_config,
    validate_config,
)


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_get_default_config_path(self):
        """Test that default config path is found."""
        config_path = get_default_config_path()
        assert config_path.exists()
        assert config_path.name == "health_default.yaml"

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()

        # Check that required sections exist
        assert 'detectors' in config
        assert 'threshold_drift' in config  # Legacy threshold

        # Check that detectors are properly configured
        detectors = config['detectors']
        expected_detectors = [
            'reward_length_correlation',
            'saturation_high_tail',
            'ece_pairwise',
            'slice_drift_domain',
            'label_flip_sensitivity'
        ]

        for detector in expected_detectors:
            assert detector in detectors
            assert 'thresholds' in detectors[detector]
            assert 'warn' in detectors[detector]['thresholds']
            assert 'fail' in detectors[detector]['thresholds']
            assert 'enabled' in detectors[detector]

    def test_load_user_config_override(self):
        """Test loading user configuration that overrides defaults."""
        user_config = {
            'detectors': {
                'reward_length_correlation': {
                    'enabled': True,
                    'thresholds': {
                        'warn': 0.15,
                        'fail': 0.30
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(user_config, f)
            user_config_path = f.name

        try:
            config = load_config(user_config_path)

            # Check that user override is applied
            assert config['detectors']['reward_length_correlation']['thresholds']['warn'] == 0.15
            assert config['detectors']['reward_length_correlation']['thresholds']['fail'] == 0.30

            # Check that other detectors still have defaults
            assert 'saturation_high_tail' in config['detectors']
            assert config['detectors']['saturation_high_tail']['thresholds']['warn'] == 0.15  # Default value

        finally:
            Path(user_config_path).unlink()

    def test_load_nonexistent_config(self):
        """Test that loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_deep_merge(self):
        """Test deep merging of configurations."""
        base = {
            'detectors': {
                'detector1': {
                    'enabled': True,
                    'thresholds': {
                        'warn': 0.1,
                        'fail': 0.2
                    }
                },
                'detector2': {
                    'enabled': False
                }
            },
            'other_setting': 'base_value'
        }

        override = {
            'detectors': {
                'detector1': {
                    'thresholds': {
                        'warn': 0.15  # Override warn threshold
                    }
                },
                'detector3': {  # Add new detector
                    'enabled': True
                }
            },
            'new_setting': 'override_value'
        }

        merged = _deep_merge(base, override)

        # Check that override values are applied
        assert merged['detectors']['detector1']['thresholds']['warn'] == 0.15
        assert merged['detectors']['detector1']['thresholds']['fail'] == 0.2  # Preserved from base
        assert merged['detectors']['detector1']['enabled']  # Preserved from base

        # Check that new detector is added
        assert 'detector3' in merged['detectors']
        assert merged['detectors']['detector3']['enabled']

        # Check that other detector is preserved
        assert 'detector2' in merged['detectors']
        assert not merged['detectors']['detector2']['enabled']

        # Check top-level settings
        assert merged['other_setting'] == 'base_value'
        assert merged['new_setting'] == 'override_value'


class TestThresholdExtraction:
    """Test threshold extraction functionality."""

    def test_get_detector_thresholds(self):
        """Test extracting thresholds for a specific detector."""
        config = {
            'detectors': {
                'test_detector': {
                    'enabled': True,
                    'thresholds': {
                        'warn': 0.1,
                        'fail': 0.2
                    }
                }
            }
        }

        thresholds = get_detector_thresholds(config, 'test_detector')

        assert thresholds['warn'] == 0.1
        assert thresholds['fail'] == 0.2
        assert thresholds['enabled']

    def test_get_detector_thresholds_missing(self):
        """Test extracting thresholds for missing detector."""
        config = {'detectors': {}}

        thresholds = get_detector_thresholds(config, 'missing_detector')

        assert thresholds['warn'] == 0.0
        assert thresholds['fail'] == 0.0
        assert thresholds['enabled']  # Default enabled

    def test_get_legacy_thresholds(self):
        """Test extracting legacy thresholds."""
        config = {
            'threshold_drift': 0.1,
            'threshold_saturation': 0.8,
            'threshold_calibration': 0.7,
            'threshold_shortcut': 0.6,
            'threshold_leakage': 0.3
        }

        thresholds = get_legacy_thresholds(config)

        assert thresholds['threshold_drift'] == 0.1
        assert thresholds['threshold_saturation'] == 0.8
        assert thresholds['threshold_calibration'] == 0.7
        assert thresholds['threshold_shortcut'] == 0.6
        assert thresholds['threshold_leakage'] == 0.3

    def test_get_legacy_thresholds_defaults(self):
        """Test extracting legacy thresholds with defaults."""
        config = {}  # Empty config

        thresholds = get_legacy_thresholds(config)

        # Should use default values
        assert thresholds['threshold_drift'] == 0.1
        assert thresholds['threshold_saturation'] == 0.8
        assert thresholds['threshold_calibration'] == 0.7
        assert thresholds['threshold_shortcut'] == 0.6
        assert thresholds['threshold_leakage'] == 0.3


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'detectors': {
                'test_detector': {
                    'enabled': True,
                    'thresholds': {
                        'warn': 0.1,
                        'fail': 0.2
                    }
                }
            }
        }

        # Should not raise any exception
        validate_config(config)

    def test_validate_missing_detectors(self):
        """Test validation fails when detectors section is missing."""
        config = {}

        with pytest.raises(ValueError, match="must contain 'detectors' section"):
            validate_config(config)

    def test_validate_invalid_detector_structure(self):
        """Test validation fails when detector is not a dictionary."""
        config = {
            'detectors': {
                'invalid_detector': 'not_a_dict'
            }
        }

        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_config(config)

    def test_validate_invalid_thresholds_structure(self):
        """Test validation fails when thresholds is not a dictionary."""
        config = {
            'detectors': {
                'test_detector': {
                    'thresholds': 'not_a_dict'
                }
            }
        }

        with pytest.raises(ValueError, match="thresholds must be a dictionary"):
            validate_config(config)

    def test_validate_negative_threshold(self):
        """Test validation fails when threshold is negative."""
        config = {
            'detectors': {
                'test_detector': {
                    'thresholds': {
                        'warn': -0.1,  # Negative threshold
                        'fail': 0.2
                    }
                }
            }
        }

        with pytest.raises(ValueError, match="must be a non-negative number"):
            validate_config(config)

    def test_validate_fail_not_higher_than_warn(self):
        """Test validation fails when fail threshold is not higher than warn."""
        config = {
            'detectors': {
                'test_detector': {
                    'thresholds': {
                        'warn': 0.2,
                        'fail': 0.1  # Fail lower than warn
                    }
                }
            }
        }

        with pytest.raises(ValueError, match="fail threshold must be higher than warn threshold"):
            validate_config(config)


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_default_config_workflow(self):
        """Test the complete workflow with default configuration."""
        # Load default config
        config = load_config()

        # Validate it
        validate_config(config)

        # Extract thresholds for all detectors
        expected_detectors = [
            'reward_length_correlation',
            'saturation_high_tail',
            'ece_pairwise',
            'slice_drift_domain',
            'label_flip_sensitivity'
        ]

        for detector in expected_detectors:
            thresholds = get_detector_thresholds(config, detector)
            assert thresholds['enabled']
            assert thresholds['warn'] > 0
            assert thresholds['fail'] > thresholds['warn']

        # Extract legacy thresholds
        legacy = get_legacy_thresholds(config)
        assert all(threshold > 0 for threshold in legacy.values())

    def test_user_override_workflow(self):
        """Test the complete workflow with user configuration override."""
        # Create user config that overrides one detector
        user_config = {
            'detectors': {
                'reward_length_correlation': {
                    'enabled': False,  # Disable this detector
                    'thresholds': {
                        'warn': 0.10,
                        'fail': 0.20
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(user_config, f)
            user_config_path = f.name

        try:
            # Load merged config
            config = load_config(user_config_path)

            # Validate it
            validate_config(config)

            # Check that override is applied
            thresholds = get_detector_thresholds(config, 'reward_length_correlation')
            assert not thresholds['enabled']
            assert thresholds['warn'] == 0.10
            assert thresholds['fail'] == 0.20

            # Check that other detectors still have defaults
            thresholds = get_detector_thresholds(config, 'saturation_high_tail')
            assert thresholds['enabled']
            assert thresholds['warn'] == 0.15  # Default value

        finally:
            Path(user_config_path).unlink()
