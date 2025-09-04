"""Core evaluation metrics for RL Debug Kit."""

from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .bias import evaluate_bias

# Import statistical functions from the metrics.py file in the parent directory
try:
    import importlib.util
    import os
    # Get the path to the parent directory's metrics.py file
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    metrics_file = os.path.join(parent_dir, 'metrics.py')
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("evals_metrics", metrics_file)
    evals_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evals_metrics)
    
    # Import the functions
    calculate_confidence_intervals = evals_metrics.calculate_confidence_intervals
    calculate_effect_sizes = evals_metrics.calculate_effect_sizes
except ImportError:
    # Fallback: define minimal implementations if import fails
    def calculate_confidence_intervals(scores, sample_size, confidence_level=0.95):
        """Calculate confidence intervals for evaluation scores."""
        import warnings
        warnings.warn("calculate_confidence_intervals: Using fallback implementation. Install scipy for full functionality.")
        return {metric: (score, score) for metric, score in scores.items()}
    
    def calculate_effect_sizes(scores, baseline_scores):
        """Calculate effect sizes for evaluation scores."""
        import warnings
        warnings.warn("calculate_effect_sizes: Using fallback implementation. Install scipy for full functionality.")
        return {metric: 0.0 for metric in scores.keys()}

__all__ = [
    "evaluate_throughput",
    "evaluate_toxicity", 
    "evaluate_bias",
    "calculate_confidence_intervals",
    "calculate_effect_sizes",
]