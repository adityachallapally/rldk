"""Core evaluation metrics for RL Debug Kit."""

from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .bias import evaluate_bias

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
]