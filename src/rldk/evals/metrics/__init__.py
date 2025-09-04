"""Core evaluation metrics for RL Debug Kit."""

from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .bias import evaluate_bias

def calculate_confidence_intervals(scores, confidence_level=0.95):
    """Calculate confidence intervals for evaluation scores."""
    import numpy as np
    from scipy import stats
    
    if len(scores) < 2:
        return {}
    
    mean_score = np.mean(scores)
    std_error = stats.sem(scores)
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

def calculate_effect_sizes(scores1, scores2):
    """Calculate effect sizes between two sets of scores."""
    import numpy as np
    from scipy import stats
    
    if len(scores1) < 2 or len(scores2) < 2:
        return {}
    
    # Cohen's d
    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                         (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                        (len(scores1) + len(scores2) - 2))
    
    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    # Mann-Whitney U test
    try:
        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    except:
        statistic, p_value = 0, 1.0
    
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