"""Core evaluation metrics for RL Debug Kit."""

from .bias import evaluate_bias
from .length_bias import (
    evaluate_length_bias,
    length_bias_score_from_metrics,
    prepare_length_bias_inputs,
    resolve_length_bias_columns,
)
from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .utils import calculate_confidence_intervals, calculate_effect_sizes

# Import additional functions from the parent metrics.py file
try:
    import importlib.util
    import logging
    import os
    import warnings

    # Get the path to the parent directory's metrics.py file
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    metrics_file = os.path.join(parent_dir, 'metrics.py')

    # Check if the file exists
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("evals_metrics", metrics_file)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create module spec for metrics.py")

    evals_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evals_metrics)

    # Check if the required functions exist and import them
    kl_divergence_functions = {}

    if hasattr(evals_metrics, 'calculate_kl_divergence'):
        kl_divergence_functions['calculate_kl_divergence'] = evals_metrics.calculate_kl_divergence
    else:
        logger = logging.getLogger(__name__)
        logger.warning("calculate_kl_divergence function not found in metrics.py")

    if hasattr(evals_metrics, 'calculate_kl_divergence_between_runs'):
        kl_divergence_functions['calculate_kl_divergence_between_runs'] = evals_metrics.calculate_kl_divergence_between_runs
    else:
        logger = logging.getLogger(__name__)
        logger.warning("calculate_kl_divergence_between_runs function not found in metrics.py")

    if hasattr(evals_metrics, 'calculate_kl_divergence_confidence_interval'):
        kl_divergence_functions['calculate_kl_divergence_confidence_interval'] = evals_metrics.calculate_kl_divergence_confidence_interval
    else:
        logger = logging.getLogger(__name__)
        logger.warning("calculate_kl_divergence_confidence_interval function not found in metrics.py")

    # Make the functions available in the module namespace
    for func_name, func in kl_divergence_functions.items():
        globals()[func_name] = func

except (ImportError, FileNotFoundError, AttributeError, OSError, Exception) as e:
    # Fallback: define minimal implementations if import fails
    import logging
    import warnings

    # Log the specific error for debugging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import functions from metrics.py: {e}. Using fallback implementations.")

    def calculate_kl_divergence(p, q, epsilon=1e-10):
        """Calculate KL divergence (fallback implementation)."""
        warnings.warn("calculate_kl_divergence: Using fallback implementation.", UserWarning, stacklevel=2)
        return 0.0

    def calculate_kl_divergence_between_runs(run1_data, run2_data, metric="reward_mean", bins=20, epsilon=1e-10):
        """Calculate KL divergence between runs (fallback implementation)."""
        warnings.warn("calculate_kl_divergence_between_runs: Using fallback implementation.", UserWarning, stacklevel=2)
        return {"kl_divergence": 0.0, "error": "Fallback implementation"}

    def calculate_kl_divergence_confidence_interval(kl_values, confidence_level=0.95):
        """Calculate KL divergence confidence interval (fallback implementation)."""
        warnings.warn("calculate_kl_divergence_confidence_interval: Using fallback implementation.", UserWarning, stacklevel=2)
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "error": "Fallback implementation"}

# Ensure all functions are defined (either imported or fallback)
if 'calculate_kl_divergence' not in globals():
    def calculate_kl_divergence(p, q, epsilon=1e-10):
        """Calculate KL divergence (fallback implementation)."""
        warnings.warn("calculate_kl_divergence: Using fallback implementation.", UserWarning, stacklevel=2)
        return 0.0

if 'calculate_kl_divergence_between_runs' not in globals():
    def calculate_kl_divergence_between_runs(run1_data, run2_data, metric="reward_mean", bins=20, epsilon=1e-10):
        """Calculate KL divergence between runs (fallback implementation)."""
        warnings.warn("calculate_kl_divergence_between_runs: Using fallback implementation.", UserWarning, stacklevel=2)
        return {"kl_divergence": 0.0, "error": "Fallback implementation"}

if 'calculate_kl_divergence_confidence_interval' not in globals():
    def calculate_kl_divergence_confidence_interval(kl_values, confidence_level=0.95):
        """Calculate KL divergence confidence interval (fallback implementation)."""
        warnings.warn("calculate_kl_divergence_confidence_interval: Using fallback implementation.", UserWarning, stacklevel=2)
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "error": "Fallback implementation"}

__all__ = [
    "evaluate_throughput",
    "evaluate_toxicity",
    "evaluate_bias",
    "evaluate_length_bias",
    "length_bias_score_from_metrics",
    "prepare_length_bias_inputs",
    "resolve_length_bias_columns",
    "calculate_confidence_intervals",
    "calculate_effect_sizes",
    "calculate_kl_divergence",
    "calculate_kl_divergence_between_runs",
    "calculate_kl_divergence_confidence_interval",
]
