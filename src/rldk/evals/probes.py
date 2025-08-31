"""Evaluation probes for different aspects of model performance."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats


def evaluate_alignment(data: pd.DataFrame, seed: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Evaluate how well the model follows human preferences and instructions.
    
    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with alignment score and details
    """
    
    np.random.seed(seed)
    
    # Check if we have alignment-related metrics
    alignment_metrics = []
    
    # Check for reward alignment with human preferences
    if 'human_preference' in data.columns and 'reward_mean' in data.columns:
        # Calculate correlation between rewards and human preferences
        valid_mask = ~(data['human_preference'].isna() | data['reward_mean'].isna())
        if valid_mask.sum() > 10:
            human_prefs = data.loc[valid_mask, 'human_preference']
            rewards = data.loc[valid_mask, 'reward_mean']
            correlation = np.corrcoef(human_prefs, rewards)[0, 1]
            alignment_metrics.append(('human_preference_correlation', correlation))
    
    # Check for instruction following (if we have instruction compliance data)
    if 'instruction_compliance' in data.columns:
        compliance_score = data['instruction_compliance'].mean()
        alignment_metrics.append(('instruction_compliance', compliance_score))
    
    # Check for safety adherence (if we have safety metrics)
    if 'safety_score' in data.columns:
        safety_score = data['safety_score'].mean()
        alignment_metrics.append(('safety_adherence', safety_score))
    
    # Calculate overall alignment score
    if alignment_metrics:
        scores = [score for _, score in alignment_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use reward consistency as proxy for alignment
        if 'reward_mean' in data.columns:
            # Check if rewards are consistent (low variance might indicate good alignment)
            reward_std = data['reward_mean'].std()
            reward_mean = data['reward_mean'].mean()
            if reward_mean != 0:
                cv = reward_std / abs(reward_mean)  # Coefficient of variation
                overall_score = max(0, 1 - cv)  # Lower CV = better alignment
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        'score': float(overall_score),
        'details': f'Alignment evaluation based on {len(alignment_metrics)} metrics',
        'method': 'correlation_and_consistency',
        'metrics': alignment_metrics,
        'sample_size': len(data)
    }


def evaluate_helpfulness(data: pd.DataFrame, seed: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Evaluate the quality and utility of model responses.
    
    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with helpfulness score and details
    """
    
    np.random.seed(seed)
    
    helpfulness_metrics = []
    
    # Check for response quality metrics
    if 'response_quality' in data.columns:
        quality_score = data['response_quality'].mean()
        helpfulness_metrics.append(('response_quality', quality_score))
    
    # Check for relevance scores
    if 'relevance_score' in data.columns:
        relevance_score = data['relevance_score'].mean()
        helpfulness_metrics.append(('relevance', relevance_score))
    
    # Check for completeness scores
    if 'completeness_score' in data.columns:
        completeness_score = data['completeness_score'].mean()
        helpfulness_metrics.append(('completeness', completeness_score))
    
    # Check for user satisfaction (if available)
    if 'user_satisfaction' in data.columns:
        satisfaction_score = data['user_satisfaction'].mean()
        helpfulness_metrics.append(('user_satisfaction', satisfaction_score))
    
    # Calculate overall helpfulness score
    if helpfulness_metrics:
        scores = [score for _, score in helpfulness_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use reward mean as proxy for helpfulness
        if 'reward_mean' in data.columns:
            # Normalize rewards to [0, 1] range for scoring
            rewards = data['reward_mean'].dropna()
            if len(rewards) > 0:
                # Assume rewards are roughly in [-1, 1] range, normalize to [0, 1]
                normalized_rewards = (rewards + 1) / 2
                overall_score = float(normalized_rewards.mean())
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        'score': float(overall_score),
        'details': f'Helpfulness evaluation based on {len(helpfulness_metrics)} metrics',
        'method': 'quality_metrics_and_rewards',
        'metrics': helpfulness_metrics,
        'sample_size': len(data)
    }


def evaluate_harmlessness(data: pd.DataFrame, seed: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model safety and harmlessness.
    
    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with harmlessness score and details
    """
    
    np.random.seed(seed)
    
    safety_metrics = []
    
    # Check for explicit safety scores
    if 'safety_score' in data.columns:
        safety_score = data['safety_score'].mean()
        safety_metrics.append(('safety_score', safety_score))
    
    # Check for toxicity scores (lower is better)
    if 'toxicity_score' in data.columns:
        toxicity_score = data['toxicity_score'].mean()
        # Invert toxicity score (lower toxicity = higher safety)
        inverted_toxicity = 1 - toxicity_score
        safety_metrics.append(('toxicity_inverted', inverted_toxicity))
    
    # Check for bias scores
    if 'bias_score' in data.columns:
        bias_score = data['bias_score'].mean()
        # Invert bias score (lower bias = higher safety)
        inverted_bias = 1 - bias_score
        safety_metrics.append(('bias_inverted', inverted_bias))
    
    # Check for adversarial robustness
    if 'adversarial_score' in data.columns:
        adv_score = data['adversarial_score'].mean()
        safety_metrics.append(('adversarial_robustness', adv_score))
    
    # Calculate overall harmlessness score
    if safety_metrics:
        scores = [score for _, score in safety_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use reward consistency as proxy for safety
        if 'reward_mean' in data.columns:
            # Check if rewards are stable (stable rewards might indicate safer behavior)
            rewards = data['reward_mean'].dropna()
            if len(rewards) > 1:
                reward_std = rewards.std()
                reward_mean = rewards.mean()
                if reward_mean != 0:
                    cv = reward_std / abs(reward_mean)
                    # Lower coefficient of variation = more stable = potentially safer
                    stability_score = max(0, 1 - cv)
                    overall_score = 0.5 + 0.3 * stability_score  # Base 0.5 + stability bonus
                else:
                    overall_score = 0.5
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        'score': float(overall_score),
        'details': f'Harmlessness evaluation based on {len(safety_metrics)} metrics',
        'method': 'safety_metrics_and_stability',
        'metrics': safety_metrics,
        'sample_size': len(data)
    }


def evaluate_hallucination(data: pd.DataFrame, seed: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Evaluate factual accuracy and hallucination detection.
    
    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with hallucination score and details (lower is better)
    """
    
    np.random.seed(seed)
    
    accuracy_metrics = []
    
    # Check for factual accuracy scores
    if 'factual_accuracy' in data.columns:
        accuracy_score = data['factual_accuracy'].mean()
        accuracy_metrics.append(('factual_accuracy', accuracy_score))
    
    # Check for hallucination detection scores
    if 'hallucination_score' in data.columns:
        hallucination_score = data['hallucination_score'].mean()
        # Invert hallucination score (lower hallucination = better)
        inverted_hallucination = 1 - hallucination_score
        accuracy_metrics.append(('hallucination_inverted', inverted_hallucination))
    
    # Check for citation accuracy
    if 'citation_accuracy' in data.columns:
        citation_score = data['citation_accuracy'].mean()
        accuracy_metrics.append(('citation_accuracy', citation_score))
    
    # Check for consistency scores
    if 'consistency_score' in data.columns:
        consistency_score = data['consistency_score'].mean()
        accuracy_metrics.append(('consistency', consistency_score))
    
    # Calculate overall hallucination score (lower is better)
    if accuracy_metrics:
        scores = [score for _, score in accuracy_metrics]
        overall_score = 1 - np.mean(scores)  # Invert so lower = better
    else:
        # Fallback: use reward variance as proxy for hallucination
        if 'reward_mean' in data.columns:
            rewards = data['reward_mean'].dropna()
            if len(rewards) > 1:
                # Higher variance might indicate more hallucination
                reward_std = rewards.std()
                reward_mean = rewards.mean()
                if reward_mean != 0:
                    cv = reward_std / abs(reward_mean)
                    # Higher CV = potentially more hallucination
                    overall_score = min(1.0, cv)
                else:
                    overall_score = 0.5
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        'score': float(overall_score),
        'details': f'Hallucination evaluation based on {len(accuracy_metrics)} metrics',
        'method': 'accuracy_metrics_and_consistency',
        'metrics': accuracy_metrics,
        'sample_size': len(data),
        'note': 'Lower scores indicate better performance (less hallucination)'
    }


def evaluate_kl_divergence(data: pd.DataFrame, 
                          reference_data: Optional[pd.DataFrame] = None,
                          seed: int = 42, 
                          **kwargs) -> Dict[str, Any]:
    """
    Evaluate KL divergence between current run and reference distribution.
    
    Args:
        data: Current training run data
        reference_data: Reference data for comparison (if None, uses baseline)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments including metrics to compare
    
    Returns:
        Dictionary with KL divergence scores and analysis
    """
    
    np.random.seed(seed)
    
    # Import KL divergence functions
    from .metrics import (
        calculate_kl_divergence_between_runs,
        calculate_kl_divergence_confidence_interval
    )
    
    # Determine metrics to evaluate
    metrics_to_evaluate = kwargs.get('metrics', ['reward_mean', 'kl_mean', 'entropy_mean'])
    
    # If no reference data provided, create a synthetic baseline
    if reference_data is None:
        # Create baseline distribution based on expected behavior
        # This could be from a pre-trained model or theoretical expectations
        baseline_data = _create_baseline_distribution(data, metrics_to_evaluate, seed)
        reference_data = baseline_data
        reference_source = 'synthetic_baseline'
    else:
        reference_source = 'provided_reference'
    
    kl_results = {}
    overall_score = 0.0
    valid_metrics = 0
    
    for metric in metrics_to_evaluate:
        if metric not in data.columns:
            continue
            
        try:
            # Calculate KL divergence for this metric
            kl_result = calculate_kl_divergence_between_runs(
                reference_data, data, metric=metric
            )
            
            if 'kl_divergence' in kl_result and not np.isnan(kl_result['kl_divergence']):
                kl_div = kl_result['kl_divergence']
                
                # Convert KL divergence to a score (lower KL = higher score)
                # Use exponential decay: score = exp(-kl_div / scale_factor)
                scale_factor = 0.1  # Adjust based on expected KL divergence range
                metric_score = np.exp(-kl_div / scale_factor)
                
                kl_results[metric] = {
                    'kl_divergence': kl_div,
                    'score': metric_score,
                    'details': kl_result
                }
                
                overall_score += metric_score
                valid_metrics += 1
            else:
                kl_results[metric] = {
                    'kl_divergence': np.nan,
                    'score': np.nan,
                    'error': kl_result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            kl_results[metric] = {
                'kl_divergence': np.nan,
                'score': np.nan,
                'error': str(e)
            }
    
    # Calculate overall score
    if valid_metrics > 0:
        overall_score = overall_score / valid_metrics
    else:
        overall_score = 0.0
    
    # Calculate confidence intervals if we have sufficient data
    confidence_intervals = {}
    if valid_metrics > 0 and len(data) > 20:
        for metric in metrics_to_evaluate:
            if metric in kl_results and 'kl_divergence' in kl_results[metric]:
                try:
                    ci_result = calculate_kl_divergence_confidence_interval(
                        reference_data, data, metric=metric
                    )
                    if 'confidence_interval' in ci_result:
                        confidence_intervals[metric] = ci_result['confidence_interval']
                except Exception:
                    confidence_intervals[metric] = (np.nan, np.nan)
    
    return {
        'score': float(overall_score),
        'kl_divergence_mean': float(np.mean([r.get('kl_divergence', np.nan) for r in kl_results.values() 
                                           if not np.isnan(r.get('kl_divergence', np.nan))])),
        'details': f'KL divergence evaluation across {valid_metrics} metrics',
        'method': 'distribution_comparison',
        'reference_source': reference_source,
        'metrics_evaluated': list(kl_results.keys()),
        'kl_results': kl_results,
        'confidence_intervals': confidence_intervals,
        'sample_size': len(data),
        'reference_size': len(reference_data) if reference_data is not None else 0
    }


def _create_baseline_distribution(data: pd.DataFrame, 
                                metrics: list, 
                                seed: int) -> pd.DataFrame:
    """
    Create a synthetic baseline distribution for KL divergence comparison.
    
    Args:
        data: Current run data to base baseline on
        metrics: List of metrics to create baselines for
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with baseline distributions
    """
    np.random.seed(seed)
    
    baseline_data = pd.DataFrame()
    
    for metric in metrics:
        if metric in data.columns:
            # Get current metric values
            current_values = data[metric].dropna()
            
            if len(current_values) > 0:
                # Create baseline with similar distribution but slight differences
                # This simulates a "good" reference model
                mean_val = np.mean(current_values)
                std_val = np.std(current_values)
                
                # Add small random variation to create baseline
                baseline_values = np.random.normal(
                    mean_val, 
                    std_val * 0.8,  # Slightly more concentrated
                    size=len(current_values)
                )
                
                # Ensure baseline values are in reasonable range
                if metric == 'reward_mean':
                    baseline_values = np.clip(baseline_values, -10, 10)
                elif metric == 'kl_mean':
                    baseline_values = np.clip(baseline_values, 0, 5)
                elif metric == 'entropy_mean':
                    baseline_values = np.clip(baseline_values, 0, 10)
                
                baseline_data[metric] = baseline_values
            else:
                # If no data, create reasonable defaults
                if metric == 'reward_mean':
                    baseline_data[metric] = np.random.normal(0, 1, size=100)
                elif metric == 'kl_mean':
                    baseline_data[metric] = np.random.exponential(0.1, size=100)
                elif metric == 'entropy_mean':
                    baseline_data[metric] = np.random.uniform(0, 5, size=100)
                else:
                    baseline_data[metric] = np.random.normal(0, 1, size=100)
    
    return baseline_data


def evaluate_reward_alignment(data: pd.DataFrame, seed: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Evaluate correlation between reward scores and human judgments.
    
    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with reward alignment score and details
    """
    
    np.random.seed(seed)
    
    alignment_metrics = []
    
    # Check for direct reward-human preference correlation
    if 'human_preference' in data.columns and 'reward_mean' in data.columns:
        valid_mask = ~(data['human_preference'].isna() | data['reward_mean'].isna())
        if valid_mask.sum() > 10:
            human_prefs = data.loc[valid_mask, 'human_preference']
            rewards = data.loc[valid_mask, 'reward_mean']
            
            # Calculate correlation
            correlation = np.corrcoef(human_prefs, rewards)[0, 1]
            alignment_metrics.append(('human_preference_correlation', correlation))
            
            # Calculate rank correlation (Spearman)
            try:
                spearman_corr = stats.spearmanr(human_prefs, rewards)[0]
                alignment_metrics.append(('spearman_correlation', spearman_corr))
            except:
                pass
    
    # Check for reward calibration
    if 'ground_truth' in data.columns and 'reward_mean' in data.columns:
        valid_mask = ~(data['ground_truth'].isna() | data['reward_mean'].isna())
        if valid_mask.sum() > 10:
            ground_truth = data.loc[valid_mask, 'ground_truth']
            rewards = data.loc[valid_mask, 'reward_mean']
            
            # Calculate correlation with ground truth
            gt_correlation = np.corrcoef(ground_truth, rewards)[0, 1]
            alignment_metrics.append(('ground_truth_correlation', gt_correlation))
    
    # Check for reward consistency over time
    if 'step' in data.columns and 'reward_mean' in data.columns:
        # Sort by step and check for trends
        sorted_data = data.sort_values('step')
        rewards = sorted_data['reward_mean'].dropna()
        
        if len(rewards) > 10:
            # Calculate trend (slope of linear fit)
            steps = np.arange(len(rewards))
            if len(steps) > 1:
                slope = np.polyfit(steps, rewards, 1)[0]
                # Convert slope to stability score (lower absolute slope = more stable)
                stability_score = max(0, 1 - abs(slope))
                alignment_metrics.append(('reward_stability', stability_score))
    
    # Calculate overall reward alignment score
    if alignment_metrics:
        scores = [score for _, score in alignment_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use reward distribution properties
        if 'reward_mean' in data.columns:
            rewards = data['reward_mean'].dropna()
            if len(rewards) > 1:
                # Check if rewards are well-distributed (not all the same)
                reward_std = rewards.std()
                reward_mean = rewards.mean()
                if reward_mean != 0:
                    cv = reward_std / abs(reward_mean)
                    # Moderate CV suggests good alignment (not too uniform, not too chaotic)
                    if 0.1 <= cv <= 0.5:
                        overall_score = 0.7
                    elif cv < 0.1:
                        overall_score = 0.5  # Too uniform
                    else:
                        overall_score = 0.3  # Too chaotic
                else:
                    overall_score = 0.5
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        'score': float(overall_score),
        'details': f'Reward alignment evaluation based on {len(alignment_metrics)} metrics',
        'method': 'correlation_and_stability',
        'metrics': alignment_metrics,
        'sample_size': len(data)
    }