"""Core evaluation metrics for RL Debug Kit."""

from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .bias import evaluate_bias
from .utils import calculate_confidence_intervals, calculate_effect_sizes

# Import KL divergence functions from the parent metrics.py file
try:
    import importlib.util
    import os
    
    # Get the path to the parent directory's metrics.py file
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    metrics_file = os.path.join(parent_dir, 'metrics.py')
    
    # Check if the file exists
    if os.path.exists(metrics_file):
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("evals_metrics", metrics_file)
        if spec is not None and spec.loader is not None:
            evals_metrics = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evals_metrics)
            
            # Import KL divergence functions
            calculate_kl_divergence = evals_metrics.calculate_kl_divergence
            calculate_kl_divergence_between_runs = evals_metrics.calculate_kl_divergence_between_runs
            calculate_kl_divergence_confidence_interval = evals_metrics.calculate_kl_divergence_confidence_interval
        else:
            raise ImportError("Failed to create module spec for metrics.py")
    else:
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
except (ImportError, FileNotFoundError, AttributeError, OSError, Exception) as e:
    # Fallback: define minimal implementations if import fails
    import warnings
    import logging
    
    # Log the specific error for debugging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import KL divergence functions from metrics.py: {e}. Using fallback implementations.")
    
    def calculate_kl_divergence(p, q, epsilon=1e-10):
        """Calculate KL divergence (fallback implementation)."""
        warnings.warn("calculate_kl_divergence: Using fallback implementation.", UserWarning)
        return 0.0
    
    def calculate_kl_divergence_between_runs(run1_data, run2_data, metric="reward_mean", bins=20, epsilon=1e-10):
        """Calculate KL divergence between runs (fallback implementation)."""
        warnings.warn("calculate_kl_divergence_between_runs: Using fallback implementation.", UserWarning)
        return {"kl_divergence": 0.0, "error": "Fallback implementation"}
    
    def calculate_kl_divergence_confidence_interval(kl_values, confidence_level=0.95):
        """Calculate KL divergence confidence interval (fallback implementation)."""
        warnings.warn("calculate_kl_divergence_confidence_interval: Using fallback implementation.", UserWarning)
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "error": "Fallback implementation"}

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

def calculate_confidence_intervals(scores, confidence_level=0.95):
    """Calculate confidence intervals for evaluation scores."""
    import numpy as np
    from scipy import stats
    
    if len(scores) < 2:
        return {
            'mean': np.mean(scores) if len(scores) == 1 else None,
            'lower': None,
            'upper': None,
            'error': 'Insufficient data: need at least 2 samples for confidence intervals'
        }
    
    try:
        mean_score = np.mean(scores)
        std_error = stats.sem(scores)
        
        if std_error == 0:
            # All scores are identical - confidence interval is just the mean
            return {
                'mean': mean_score,
                'lower': mean_score,
                'upper': mean_score,
                'note': 'All scores are identical'
            }
        
        confidence_interval = stats.t.interval(
            confidence_level, 
            len(scores) - 1, 
            loc=mean_score, 
            scale=std_error
        )
        
        return {
            'mean': mean_score,
            'lower': confidence_interval[0],
            'upper': confidence_interval[1]
        }
        
    except (ValueError, ArithmeticError) as e:
        return {
            'mean': None,
            'lower': None,
            'upper': None,
            'error': f'Confidence interval calculation failed: {str(e)}'
        }

def calculate_effect_sizes(scores1, scores2):
    """Calculate effect sizes between two sets of scores."""
    import numpy as np
    from scipy import stats
    
    if len(scores1) < 2 or len(scores2) < 2:
        return {
            'cohens_d': None,
            'mann_whitney_u': None,
            'p_value': None,
            'error': 'Insufficient data: need at least 2 samples in each group'
        }
    
    # Cohen's d calculation with proper error handling
    try:
        pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                             (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                            (len(scores1) + len(scores2) - 2))
        
        if pooled_std == 0:
            # Both groups have identical values - no effect size can be calculated
            cohens_d = None
        else:
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
            
    except (ZeroDivisionError, ValueError, ArithmeticError) as e:
        return {
            'cohens_d': None,
            'mann_whitney_u': None,
            'p_value': None,
            'error': f'Cohen\'s d calculation failed: {str(e)}'
        }
    
    # Mann-Whitney U test with specific error handling
    try:
        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    except ValueError as e:
        # Handle specific ValueError cases (e.g., identical distributions)
        return {
            'cohens_d': cohens_d,
            'mann_whitney_u': None,
            'p_value': None,
            'error': f'Mann-Whitney U test failed: {str(e)}'
        }
    except Exception as e:
        # Catch other unexpected errors
        return {
            'cohens_d': cohens_d,
            'mann_whitney_u': None,
            'p_value': None,
            'error': f'Unexpected error in Mann-Whitney U test: {str(e)}'
        }
    
    return {
        'cohens_d': cohens_d,
        'mann_whitney_u': statistic,
        'p_value': p_value
    }

__all__ = [
    "evaluate_throughput",
    "evaluate_toxicity", 
    "evaluate_bias",
    "calculate_confidence_intervals",
    "calculate_effect_sizes",
    "calculate_kl_divergence",
    "calculate_kl_divergence_between_runs",
    "calculate_kl_divergence_confidence_interval",
]