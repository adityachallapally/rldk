"""Configuration loading and merging for reward health analysis."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def get_default_config_path() -> Path:
    """Get the path to the default health configuration file."""
    # Try to find the default config in the package data
    try:
        # Look for the config in the package data
        try:
            # Try to get the package location using importlib.metadata
            dist = metadata.distribution('rldk')
            config_path = dist.locate_file('rldk/reward/health_config/data/health_default.yaml')
        except Exception:
            # Fallback to using files() if available
            try:
                files = metadata.files('rldk')
                config_file = next((f for f in files if f.name == 'health_default.yaml' and 'health_config/data' in str(f)), None)
                if config_file:
                    config_path = str(config_file.locate())
                else:
                    raise FileNotFoundError
            except Exception:
                raise FileNotFoundError
        if Path(config_path).exists():
            return Path(config_path)
    except Exception:
        pass

    # Fallback: look for it relative to the current file
    current_dir = Path(__file__).parent
    config_path = current_dir / 'data' / 'health_default.yaml'
    if config_path.exists():
        return config_path

    # If not found, raise an error
    raise FileNotFoundError(
        "Default health configuration file not found. "
        "Expected at: rldk.reward.health_config.data.health_default.yaml"
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load health configuration, merging user overrides on top of defaults.

    Args:
        config_path: Optional path to user configuration file

    Returns:
        Merged configuration dictionary
    """
    # Load default configuration
    default_path = get_default_config_path()
    with open(default_path) as f:
        default_config = yaml.safe_load(f)

    # If no user config provided, return defaults
    if config_path is None:
        return default_config

    # Load user configuration
    user_path = Path(config_path)
    if not user_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(user_path) as f:
        user_config = yaml.safe_load(f)

    # Merge configurations (user config takes precedence)
    merged_config = _deep_merge(default_config, user_config)

    return merged_config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_detector_thresholds(config: Dict[str, Any], detector_name: str) -> Dict[str, float]:
    """
    Extract thresholds for a specific detector from the configuration.

    Args:
        config: Configuration dictionary
        detector_name: Name of the detector

    Returns:
        Dictionary with 'warn' and 'fail' thresholds
    """
    detectors = config.get('detectors', {})
    detector_config = detectors.get(detector_name, {})

    thresholds = detector_config.get('thresholds', {})

    return {
        'warn': thresholds.get('warn', 0.0),
        'fail': thresholds.get('fail', 0.0),
        'enabled': detector_config.get('enabled', True)
    }


def get_legacy_thresholds(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract legacy thresholds for backward compatibility.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with legacy threshold values
    """
    return {
        'threshold_drift': config.get('threshold_drift', 0.1),
        'threshold_saturation': config.get('threshold_saturation', 0.8),
        'threshold_calibration': config.get('threshold_calibration', 0.7),
        'threshold_shortcut': config.get('threshold_shortcut', 0.6),
        'threshold_leakage': config.get('threshold_leakage', 0.3),
        'threshold_length_bias': config.get('threshold_length_bias', 0.4),
        'enable_length_bias_detection': config.get('enable_length_bias_detection', True),
    }


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration structure and values.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required top-level keys
    if 'detectors' not in config:
        raise ValueError("Configuration must contain 'detectors' section")

    # Validate detector configurations
    detectors = config['detectors']
    for detector_name, detector_config in detectors.items():
        if not isinstance(detector_config, dict):
            raise ValueError(f"Detector '{detector_name}' must be a dictionary")

        # Check for thresholds
        if 'thresholds' in detector_config:
            thresholds = detector_config['thresholds']
            if not isinstance(thresholds, dict):
                raise ValueError(f"Detector '{detector_name}' thresholds must be a dictionary")

            # Validate threshold values
            for threshold_type in ['warn', 'fail']:
                if threshold_type in thresholds:
                    value = thresholds[threshold_type]
                    if not isinstance(value, (int, float)) or value < 0:
                        raise ValueError(
                            f"Detector '{detector_name}' {threshold_type} threshold must be a non-negative number"
                        )

            # Ensure fail threshold is higher than warn threshold
            if 'warn' in thresholds and 'fail' in thresholds:
                if thresholds['fail'] <= thresholds['warn']:
                    raise ValueError(
                        f"Detector '{detector_name}' fail threshold must be higher than warn threshold"
                    )
