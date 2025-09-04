"""Core evaluation metrics for RL Debug Kit."""

from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .bias import evaluate_bias
from .utils import calculate_confidence_intervals, calculate_effect_sizes

# Import statistical functions from the metrics.py file in the parent directory
try:
    import importlib.util
    import os
    
    # Get the path to the parent directory's metrics.py file
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    metrics_file = os.path.join(parent_dir, 'metrics.py')
    
    # Check if the file exists
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("evals_metrics", metrics_file)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create module spec")
    
    evals_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evals_metrics)
    
    # Check if the required functions exist
    if not hasattr(evals_metrics, 'calculate_confidence_intervals'):
        raise AttributeError("calculate_confidence_intervals function not found")
    if not hasattr(evals_metrics, 'calculate_effect_sizes'):
        raise AttributeError("calculate_effect_sizes function not found")
    
    # Import the functions
    calculate_confidence_intervals = evals_metrics.calculate_confidence_intervals
    calculate_effect_sizes = evals_metrics.calculate_effect_sizes
    
except (ImportError, FileNotFoundError, AttributeError, OSError, Exception) as e:
    # Fallback: define minimal implementations if import fails
    import warnings
    import logging
    
    # Log the specific error for debugging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import statistical functions from metrics.py: {e}. Using fallback implementations.")
    
    def calculate_confidence_intervals(scores, sample_size, confidence_level=0.95):
        """Calculate confidence intervals for evaluation scores (fallback implementation).
        
        This is a minimal implementation that returns the score as both lower and upper bounds.
        For full statistical functionality, ensure the metrics.py file is available and scipy is installed.
        """
        warnings.warn(
            "calculate_confidence_intervals: Using fallback implementation. "
            "For full statistical functionality, ensure metrics.py is available and scipy is installed.",
            UserWarning
        )
        if not isinstance(scores, dict):
            raise TypeError("scores must be a dictionary")
        return {metric: (float(score), float(score)) for metric, score in scores.items()}
    
    def calculate_effect_sizes(scores, baseline_scores):
        """Calculate effect sizes for evaluation scores (fallback implementation).
        
        This is a minimal implementation that returns 0.0 for all metrics.
        For full statistical functionality, ensure the metrics.py file is available and scipy is installed.
        """
        warnings.warn(
            "calculate_effect_sizes: Using fallback implementation. "
            "For full statistical functionality, ensure metrics.py is available and scipy is installed.",
            UserWarning
        )
        if not isinstance(scores, dict):
            raise TypeError("scores must be a dictionary")
        if not isinstance(baseline_scores, dict):
            raise TypeError("baseline_scores must be a dictionary")
        return {metric: 0.0 for metric in scores.keys()}

__all__ = [
    "evaluate_throughput",
    "evaluate_toxicity", 
    "evaluate_bias",
    "calculate_confidence_intervals",
    "calculate_effect_sizes",
]