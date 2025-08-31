"""Statistical metrics for evaluation results."""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from scipy import stats
from scipy.stats import bootstrap


def calculate_confidence_intervals(scores: Dict[str, float], 
                                 sample_size: int,
                                 confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for evaluation scores.
    
    Args:
        scores: Dictionary of metric names to scores
        sample_size: Number of samples used in evaluation
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Dictionary mapping metric names to (lower, upper) confidence intervals
    """
    
    confidence_intervals = {}
    
    for metric, score in scores.items():
        if np.isnan(score):
            confidence_intervals[metric] = (np.nan, np.nan)
            continue
        
        # For now, use a simple approach based on sample size
        # In practice, you might want to bootstrap or use more sophisticated methods
        
        # Standard error approximation (assuming score is roughly normal)
        # This is a simplified approach - in practice you'd want actual sampling data
        if sample_size > 1:
            # Use a conservative estimate of standard error
            # Assuming scores are roughly in [0, 1] range
            estimated_std = 0.3  # Conservative estimate
            standard_error = estimated_std / np.sqrt(sample_size)
            
            # Calculate confidence interval using normal approximation
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * standard_error
            
            lower_bound = max(0, score - margin_of_error)
            upper_bound = min(1, score + margin_of_error)
            
            confidence_intervals[metric] = (lower_bound, upper_bound)
        else:
            confidence_intervals[metric] = (score, score)
    
    return confidence_intervals


def calculate_effect_sizes(scores: Dict[str, float], 
                          baseline_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate effect sizes comparing current scores to baseline.
    
    Args:
        scores: Current evaluation scores
        baseline_scores: Baseline scores for comparison
    
    Returns:
        Dictionary mapping metric names to effect sizes (Cohen's d)
    """
    
    effect_sizes = {}
    
    for metric, score in scores.items():
        if np.isnan(score):
            effect_sizes[metric] = np.nan
            continue
        
        if metric in baseline_scores:
            baseline = baseline_scores[metric]
            if not np.isnan(baseline):
                # Calculate Cohen's d effect size
                # For now, use a simplified approach since we don't have full distributions
                
                # Assume a reasonable pooled standard deviation
                # In practice, you'd calculate this from actual data
                pooled_std = 0.3  # Conservative estimate for scores in [0, 1]
                
                if pooled_std > 0:
                    effect_size = (score - baseline) / pooled_std
                    effect_sizes[metric] = float(effect_size)
                else:
                    effect_sizes[metric] = 0.0
            else:
                effect_sizes[metric] = np.nan
        else:
            # No baseline available
            effect_sizes[metric] = np.nan
    
    return effect_sizes


def bootstrap_confidence_interval(data: np.ndarray, 
                                statistic_func: callable,
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Array of data points
        statistic_func: Function to calculate the statistic
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    
    try:
        # Use scipy's bootstrap function
        bootstrap_result = bootstrap(
            (data,), 
            statistic_func, 
            n_resamples=n_bootstrap,
            confidence_level=confidence_level
        )
        
        return bootstrap_result.confidence_interval
    except Exception:
        # Fallback to manual bootstrap if scipy bootstrap fails
        return _manual_bootstrap(data, statistic_func, confidence_level, n_bootstrap)


def _manual_bootstrap(data: np.ndarray, 
                     statistic_func: callable,
                     confidence_level: float = 0.95,
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Manual bootstrap implementation as fallback."""
    
    n_samples = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        try:
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        except Exception:
            continue
    
    if not bootstrap_stats:
        return (np.nan, np.nan)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return (lower_bound, upper_bound)


def calculate_statistical_significance(group1_scores: np.ndarray, 
                                     group2_scores: np.ndarray,
                                     test_type: str = 't_test') -> Dict[str, Any]:
    """
    Calculate statistical significance between two groups of scores.
    
    Args:
        group1_scores: Scores from first group
        group2_scores: Scores from second group
        test_type: Type of statistical test ('t_test', 'mann_whitney', 'ks_test')
    
    Returns:
        Dictionary with test results including p-value and statistic
    """
    
    if len(group1_scores) == 0 or len(group2_scores) == 0:
        return {
            'p_value': np.nan,
            'statistic': np.nan,
            'significant': False,
            'test_type': test_type,
            'error': 'Empty data groups'
        }
    
    try:
        if test_type == 't_test':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(group1_scores, group2_scores)
        elif test_type == 'mann_whitney':
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')
        elif test_type == 'ks_test':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(group1_scores, group2_scores)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine significance
        significant = p_value < 0.05
        
        return {
            'p_value': float(p_value),
            'statistic': float(statistic),
            'significant': significant,
            'test_type': test_type,
            'group1_size': len(group1_scores),
            'group2_size': len(group2_scores),
            'group1_mean': float(np.mean(group1_scores)),
            'group2_mean': float(np.mean(group2_scores))
        }
        
    except Exception as e:
        return {
            'p_value': np.nan,
            'statistic': np.nan,
            'significant': False,
            'test_type': test_type,
            'error': str(e)
        }


def calculate_reliability_metrics(scores: np.ndarray, 
                                n_splits: int = 5) -> Dict[str, float]:
    """
    Calculate reliability metrics for evaluation scores.
    
    Args:
        scores: Array of evaluation scores
        n_splits: Number of splits for cross-validation
    
    Returns:
        Dictionary with reliability metrics
    """
    
    if len(scores) < n_splits * 2:
        return {
            'split_half_reliability': np.nan,
            'cronbach_alpha': np.nan,
            'standard_error': np.nan,
            'error': 'Insufficient data for reliability analysis'
        }
    
    try:
        # Split-half reliability
        np.random.shuffle(scores)
        split_point = len(scores) // 2
        half1 = scores[:split_point]
        half2 = scores[split_point:]
        
        if len(half1) > 0 and len(half2) > 0:
            correlation = np.corrcoef(half1, half2)[0, 1]
            # Spearman-Brown correction
            split_half_reliability = (2 * correlation) / (1 + correlation) if not np.isnan(correlation) else np.nan
        else:
            split_half_reliability = np.nan
        
        # Cronbach's alpha (simplified - assumes items are parallel)
        if len(scores) > 1:
            scores_std = np.std(scores)
            if scores_std > 0:
                # Simplified Cronbach's alpha calculation
                n_items = len(scores)
                item_variances = np.var(scores)
                total_variance = np.var(scores)
                
                if total_variance > 0:
                    cronbach_alpha = (n_items / (n_items - 1)) * (1 - item_variances / total_variance)
                else:
                    cronbach_alpha = np.nan
            else:
                cronbach_alpha = np.nan
        else:
            cronbach_alpha = np.nan
        
        # Standard error of measurement
        if not np.isnan(split_half_reliability) and split_half_reliability > 0:
            standard_error = np.std(scores) * np.sqrt(1 - split_half_reliability)
        else:
            standard_error = np.nan
        
        return {
            'split_half_reliability': float(split_half_reliability) if not np.isnan(split_half_reliability) else np.nan,
            'cronbach_alpha': float(cronbach_alpha) if not np.isnan(cronbach_alpha) else np.nan,
            'standard_error': float(standard_error) if not np.isnan(standard_error) else np.nan
        }
        
    except Exception as e:
        return {
            'split_half_reliability': np.nan,
            'cronbach_alpha': np.nan,
            'standard_error': np.nan,
            'error': str(e)
        }


def calculate_effect_size_interpretation(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        effect_size: Cohen's d effect size value
    
    Returns:
        String interpretation of the effect size
    """
    
    effect_size = abs(effect_size)
    
    if effect_size < 0.2:
        return "negligible"
    elif effect_size < 0.5:
        return "small"
    elif effect_size < 0.8:
        return "medium"
    elif effect_size < 1.2:
        return "large"
    elif effect_size < 2.0:
        return "very large"
    else:
        return "huge"


def calculate_power_analysis(effect_size: float, 
                           alpha: float = 0.05,
                           power: float = 0.8) -> Dict[str, Any]:
    """
    Calculate required sample size for statistical power analysis.
    
    Args:
        effect_size: Expected effect size (Cohen's d)
        alpha: Significance level
        power: Desired statistical power
    
    Returns:
        Dictionary with power analysis results
    """
    
    try:
        from statsmodels.stats.power import TTestPower
        
        # Create power analysis object
        power_analysis = TTestPower()
        
        # Calculate required sample size
        required_n = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        
        return {
            'required_sample_size': int(required_n),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test_type': 't-test'
        }
        
    except ImportError:
        # Fallback calculation if statsmodels not available
        # Simplified power analysis using normal approximation
        
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Required sample size (per group for two-sample t-test)
        required_n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return {
            'required_sample_size': int(required_n),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test_type': 't-test (approximate)',
            'note': 'Approximate calculation - statsmodels recommended for accuracy'
        }