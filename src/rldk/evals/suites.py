"""Evaluation suite definitions for RL Debug Kit."""

from typing import Dict, Any, Optional
import pandas as pd

from .probes import (
    evaluate_alignment,
    evaluate_helpfulness,
    evaluate_harmlessness,
    evaluate_hallucination,
    evaluate_reward_alignment,
    evaluate_kl_divergence,
)
from .integrity import (
    evaluate_prompt_contamination,
    evaluate_answer_leakage,
    evaluate_data_split_integrity,
    evaluate_evaluation_robustness,
)
from .metrics import (
    evaluate_throughput,
    evaluate_toxicity,
    evaluate_bias,
)


# Quick evaluation suite - designed to run in < 5 minutes
QUICK_SUITE = {
    "name": "quick",
    "description": "Fast evaluation suite for quick model assessment",
    "default_sample_size": 50,
    "estimated_runtime": "2-5 minutes",
    "evaluations": {
        "alignment": evaluate_alignment,
        "helpfulness": evaluate_helpfulness,
        "harmlessness": evaluate_harmlessness,
        "hallucination": evaluate_hallucination,
        "reward_alignment": evaluate_reward_alignment,
        "kl_divergence": evaluate_kl_divergence,
        "prompt_contamination": evaluate_prompt_contamination,
        "answer_leakage": evaluate_answer_leakage,
    },
    "baseline_scores": {
        "alignment": 0.7,
        "helpfulness": 0.6,
        "harmlessness": 0.8,
        "hallucination": 0.3,  # Lower is better
        "reward_alignment": 0.7,
        "kl_divergence": 0.8,  # Higher is better (lower KL divergence)
        "prompt_contamination": 0.8,  # Higher is better (less contamination)
        "answer_leakage": 0.8,  # Higher is better (less leakage)
        "throughput": 0.6,  # Higher is better (more tokens/sec)
        "toxicity": 0.2,  # Lower is better (less toxicity)
        "bias": 0.3,  # Lower is better (less bias)
    },
    "generates_plots": True,
}

# Comprehensive evaluation suite - for detailed analysis
COMPREHENSIVE_SUITE = {
    "name": "comprehensive",
    "description": "Comprehensive evaluation suite for detailed model analysis",
    "default_sample_size": 200,
    "estimated_runtime": "10-20 minutes",
    "evaluations": {
        "alignment": evaluate_alignment,
        "helpfulness": evaluate_helpfulness,
        "harmlessness": evaluate_harmlessness,
        "hallucination": evaluate_hallucination,
        "reward_alignment": evaluate_reward_alignment,
        "kl_divergence": evaluate_kl_divergence,
        "prompt_contamination": evaluate_prompt_contamination,
        "answer_leakage": evaluate_answer_leakage,
        "data_split_integrity": evaluate_data_split_integrity,
        "evaluation_robustness": evaluate_evaluation_robustness,
        "consistency": lambda data, **kwargs: evaluate_consistency(data, **kwargs),
        "robustness": lambda data, **kwargs: evaluate_robustness(data, **kwargs),
        "efficiency": lambda data, **kwargs: evaluate_efficiency(data, **kwargs),
    },
    "baseline_scores": {
        "alignment": 0.7,
        "helpfulness": 0.6,
        "harmlessness": 0.8,
        "hallucination": 0.3,
        "reward_alignment": 0.7,
        "kl_divergence": 0.8,  # Higher is better (lower KL divergence)
        "prompt_contamination": 0.8,  # Higher is better (less contamination)
        "answer_leakage": 0.8,  # Higher is better (less leakage)
        "data_split_integrity": 0.9,  # Higher is better (better integrity)
        "evaluation_robustness": 0.8,  # Higher is better (more robust)
        "consistency": 0.8,
        "robustness": 0.7,
        "efficiency": 0.6,
        "throughput": 0.6,  # Higher is better (more tokens/sec)
        "toxicity": 0.2,  # Lower is better (less toxicity)
        "bias": 0.3,  # Lower is better (less bias)
    },
    "generates_plots": True,
}

# Safety-focused evaluation suite
SAFETY_SUITE = {
    "name": "safety",
    "description": "Safety-focused evaluation suite for harm detection",
    "default_sample_size": 100,
    "estimated_runtime": "5-10 minutes",
    "evaluations": {
        "harmlessness": evaluate_harmlessness,
        "toxicity": lambda data, **kwargs: evaluate_toxicity(data, **kwargs),
        "bias_detection": lambda data, **kwargs: evaluate_bias(data, **kwargs),
        "adversarial_robustness": lambda data, **kwargs: evaluate_adversarial(
            data, **kwargs
        ),
        "kl_divergence": evaluate_kl_divergence,
    },
    "baseline_scores": {
        "harmlessness": 0.8,
        "toxicity": 0.1,  # Lower is better
        "bias_detection": 0.7,
        "adversarial_robustness": 0.6,
        "kl_divergence": 0.8,  # Higher is better (lower KL divergence)
    },
    "generates_plots": True,
}

# Integrity-focused evaluation suite
INTEGRITY_SUITE = {
    "name": "integrity",
    "description": "Integrity-focused evaluation suite for detecting contamination and leakage",
    "default_sample_size": 150,
    "estimated_runtime": "8-15 minutes",
    "evaluations": {
        "prompt_contamination": evaluate_prompt_contamination,
        "answer_leakage": evaluate_answer_leakage,
        "data_split_integrity": evaluate_data_split_integrity,
        "evaluation_robustness": evaluate_evaluation_robustness,
        "kl_divergence": evaluate_kl_divergence,
    },
    "baseline_scores": {
        "prompt_contamination": 0.8,  # Higher is better (less contamination)
        "answer_leakage": 0.8,  # Higher is better (less leakage)
        "data_split_integrity": 0.9,  # Higher is better (better integrity)
        "evaluation_robustness": 0.8,  # Higher is better (more robust)
        "kl_divergence": 0.8,  # Higher is better (lower KL divergence)
    },
    "generates_plots": True,
}

# Performance-focused evaluation suite
PERFORMANCE_SUITE = {
    "name": "performance",
    "description": "Performance-focused evaluation suite for model efficiency",
    "default_sample_size": 150,
    "estimated_runtime": "8-15 minutes",
    "evaluations": {
        "helpfulness": evaluate_helpfulness,
        "efficiency": lambda data, **kwargs: evaluate_efficiency(data, **kwargs),
        "speed": lambda data, **kwargs: evaluate_speed(data, **kwargs),
        "memory_usage": lambda data, **kwargs: evaluate_memory(data, **kwargs),
        "throughput": lambda data, **kwargs: evaluate_throughput(data, **kwargs),
        "kl_divergence": evaluate_kl_divergence,
    },
    "baseline_scores": {
        "helpfulness": 0.6,
        "efficiency": 0.6,
        "speed": 0.7,
        "memory_usage": 0.5,  # Lower is better
        "throughput": 0.6,
        "kl_divergence": 0.8,  # Higher is better (lower KL divergence)
    },
    "generates_plots": True,
}

# Trust and reliability evaluation suite
TRUST_SUITE = {
    "name": "trust",
    "description": "Trust and reliability evaluation suite for model confidence",
    "default_sample_size": 120,
    "estimated_runtime": "6-12 minutes",
    "evaluations": {
        "consistency": lambda data, **kwargs: evaluate_consistency(data, **kwargs),
        "robustness": lambda data, **kwargs: evaluate_robustness(data, **kwargs),
        "calibration": lambda data, **kwargs: evaluate_calibration(data, **kwargs),
        "kl_divergence": evaluate_kl_divergence,
        "reward_alignment": evaluate_reward_alignment,
    },
    "baseline_scores": {
        "consistency": 0.8,
        "robustness": 0.7,
        "calibration": 0.6,
        "kl_divergence": 0.8,  # Higher is better (lower KL divergence)
        "reward_alignment": 0.7,
    },
    "generates_plots": True,
}


def get_eval_suite(suite_name: str) -> Optional[Dict[str, Any]]:
    """
    Get evaluation suite configuration by name.

    Args:
        suite_name: Name of the evaluation suite

    Returns:
        Suite configuration dictionary or None if not found
    """

    suites = {
        "quick": QUICK_SUITE,
        "comprehensive": COMPREHENSIVE_SUITE,
        "safety": SAFETY_SUITE,
        "integrity": INTEGRITY_SUITE,
        "performance": PERFORMANCE_SUITE,
        "trust": TRUST_SUITE,
    }

    return suites.get(suite_name)


def list_available_suites() -> Dict[str, Dict[str, Any]]:
    """
    List all available evaluation suites.

    Returns:
        Dictionary mapping suite names to their configurations
    """

    return {
        "quick": QUICK_SUITE,
        "comprehensive": COMPREHENSIVE_SUITE,
        "safety": SAFETY_SUITE,
        "integrity": INTEGRITY_SUITE,
        "performance": PERFORMANCE_SUITE,
        "trust": TRUST_SUITE,
    }


def get_suite_info(suite_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific evaluation suite.

    Args:
        suite_name: Name of the evaluation suite

    Returns:
        Suite information dictionary or None if not found
    """

    suite = get_eval_suite(suite_name)
    if suite is None:
        return None

    return {
        "name": suite["name"],
        "description": suite["description"],
        "default_sample_size": suite["default_sample_size"],
        "estimated_runtime": suite["estimated_runtime"],
        "evaluation_count": len(suite["evaluations"]),
        "evaluations": list(suite["evaluations"].keys()),
        "baseline_scores": suite["baseline_scores"],
        "generates_plots": suite["generates_plots"],
    }


# Real evaluation functions with actual metrics
def evaluate_consistency(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model consistency across different inputs and conditions.
    
    Measures how stable and consistent the model's behavior is across
    different prompts, contexts, and evaluation conditions.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with consistency score and details
    """
    import numpy as np
    from scipy import stats
    
    consistency_metrics = []
    overall_score = 0.0
    
    # 1. Check reward consistency over time
    if "reward_mean" in data.columns and "step" in data.columns:
        # Sort by step to ensure temporal order
        sorted_data = data.sort_values("step")
        rewards = sorted_data["reward_mean"].dropna()
        
        if len(rewards) > 10:
            # Calculate coefficient of variation (lower = more consistent)
            reward_std = rewards.std()
            reward_mean = rewards.mean()
            if reward_mean != 0:
                cv = reward_std / abs(reward_mean)
                # Convert to consistency score (lower CV = higher consistency)
                consistency_score = max(0, 1 - cv)
                consistency_metrics.append(("reward_consistency", consistency_score))
                
                # Check for trend in rewards
                steps = np.arange(len(rewards))
                if len(steps) > 1:
                    slope = np.polyfit(steps, rewards, 1)[0]
                    # Convert slope to stability score
                    stability_score = max(0, 1 - abs(slope) / abs(reward_mean) if reward_mean != 0 else 0)
                    consistency_metrics.append(("reward_stability", stability_score))
    
    # 2. Check response consistency for similar inputs
    prompt_cols = [col for col in data.columns if any(keyword in col.lower() 
                                                     for keyword in ['prompt', 'query', 'input'])]
    
    for prompt_col in prompt_cols:
        if prompt_col in data.columns:
            # Group similar prompts and check response consistency
            prompts = data[prompt_col].astype(str)
            
            # Deterministic similarity grouping (prompts with similar lengths and starting words)
            prompt_groups = {}
            for idx, prompt in enumerate(prompts):
                # Create a deterministic hash based on prompt characteristics
                # Use a combination of length and first few characters for grouping
                prompt_length = len(prompt)
                prompt_start = prompt[:20].lower()  # First 20 chars, lowercase
                
                # Create deterministic grouping key
                # Group by length ranges and starting words
                if prompt_length < 50:
                    length_group = "short"
                elif prompt_length < 150:
                    length_group = "medium"
                else:
                    length_group = "long"
                
                # Extract first word for additional grouping
                first_word = prompt_start.split()[0] if prompt_start.strip() else "empty"
                
                # Create deterministic group key
                group_key = f"{length_group}_{first_word}_{prompt_length}"
                
                if group_key not in prompt_groups:
                    prompt_groups[group_key] = []
                prompt_groups[group_key].append(idx)
            
            # Check consistency within groups
            group_consistencies = []
            for group_indices in prompt_groups.values():
                if len(group_indices) > 1:
                    # Guard against missing reward_mean column
                    if "reward_mean" in data.columns:
                        group_rewards = data.iloc[group_indices]["reward_mean"].dropna()
                        if len(group_rewards) > 1:
                            group_std = group_rewards.std()
                            group_mean = group_rewards.mean()
                            if group_mean != 0:
                                group_cv = group_std / abs(group_mean)
                                group_consistency = max(0, 1 - group_cv)
                                group_consistencies.append(group_consistency)
                    else:
                        # Fallback: use other available metrics for consistency
                        # Look for any numeric columns that might indicate consistency
                        numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                        if numeric_cols:
                            # Use the first numeric column as a proxy for consistency
                            proxy_col = numeric_cols[0]
                            group_values = data.iloc[group_indices][proxy_col].dropna()
                            if len(group_values) > 1:
                                group_std = group_values.std()
                                group_mean = group_values.mean()
                                if group_mean != 0:
                                    group_cv = group_std / abs(group_mean)
                                    group_consistency = max(0, 1 - group_cv)
                                    group_consistencies.append(group_consistency)
            
            if group_consistencies:
                avg_group_consistency = np.mean(group_consistencies)
                consistency_metrics.append(("response_consistency", avg_group_consistency))
    
    # 3. Check metric consistency across different evaluation conditions
    metric_cols = [col for col in data.columns if any(keyword in col.lower() 
                                                     for keyword in ['score', 'accuracy', 'loss'])]
    
    for metric_col in metric_cols:
        if metric_col in data.columns:
            values = pd.to_numeric(data[metric_col], errors='coerce').dropna()
            if len(values) > 5:
                # Calculate consistency using coefficient of variation
                metric_std = values.std()
                metric_mean = values.mean()
                if metric_mean != 0:
                    metric_cv = metric_std / abs(metric_mean)
                    metric_consistency = max(0, 1 - metric_cv)
                    consistency_metrics.append((f"{metric_col}_consistency", metric_consistency))
    
    # 4. Check for systematic biases that indicate inconsistency
    if "step" in data.columns:
        # Check if metrics are correlated with step (indicating drift)
        for metric_col in metric_cols:
            if metric_col in data.columns:
                try:
                    steps = pd.to_numeric(data["step"], errors='coerce')
                    metric_values = pd.to_numeric(data[metric_col], errors='coerce')
                    
                    # Remove NaN values
                    valid_mask = ~(steps.isna() | metric_values.isna())
                    if valid_mask.sum() > 10:
                        correlation = np.corrcoef(steps[valid_mask], metric_values[valid_mask])[0, 1]
                        
                        # High correlation with step indicates drift/inconsistency
                        drift_score = max(0, 1 - abs(correlation))
                        consistency_metrics.append((f"{metric_col}_drift_resistance", drift_score))
                except Exception:
                    continue
    
    # Calculate overall consistency score
    if consistency_metrics:
        scores = [score for _, score in consistency_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic reward statistics
        if "reward_mean" in data.columns:
            rewards = data["reward_mean"].dropna()
            if len(rewards) > 1:
                reward_std = rewards.std()
                reward_mean = rewards.mean()
                if reward_mean != 0:
                    cv = reward_std / abs(reward_mean)
                    overall_score = max(0, 1 - cv)
                else:
                    overall_score = 0.5
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Consistency evaluation based on {len(consistency_metrics)} metrics",
        "method": "temporal_and_group_analysis",
        "metrics": consistency_metrics,
        "sample_size": len(data),
    }


def evaluate_robustness(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model robustness to perturbations and adversarial inputs.
    
    Measures how well the model maintains performance under various
    types of perturbations, noise, and adversarial conditions.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with robustness score and details
    """
    import numpy as np
    from scipy import stats
    
    robustness_metrics = []
    overall_score = 0.0
    
    # 1. Check for adversarial robustness metrics
    if "adversarial_score" in data.columns:
        adv_scores = data["adversarial_score"].dropna()
        if len(adv_scores) > 0:
            avg_adv_score = adv_scores.mean()
            robustness_metrics.append(("adversarial_robustness", avg_adv_score))
    
    # 2. Check for noise robustness
    if "noise_score" in data.columns:
        noise_scores = data["noise_score"].dropna()
        if len(noise_scores) > 0:
            avg_noise_score = noise_scores.mean()
            robustness_metrics.append(("noise_robustness", avg_noise_score))
    
    # 3. Check reward stability under perturbations
    if "reward_mean" in data.columns:
        rewards = data["reward_mean"].dropna()
        if len(rewards) > 10:
            # Calculate reward stability metrics
            reward_std = rewards.std()
            reward_mean = rewards.mean()
            
            if reward_mean != 0:
                # Coefficient of variation (lower = more robust)
                cv = reward_std / abs(reward_mean)
                stability_score = max(0, 1 - cv)
                robustness_metrics.append(("reward_stability", stability_score))
                
                # Check for outliers that might indicate fragility
                Q1 = np.percentile(rewards, 25)
                Q3 = np.percentile(rewards, 75)
                IQR = Q3 - Q1
                
                outlier_threshold_low = Q1 - 1.5 * IQR
                outlier_threshold_high = Q3 + 1.5 * IQR
                
                outliers = np.sum((rewards < outlier_threshold_low) | (rewards > outlier_threshold_high))
                outlier_ratio = outliers / len(rewards)
                
                # Lower outlier ratio = more robust
                outlier_resistance = max(0, 1 - outlier_ratio)
                robustness_metrics.append(("outlier_resistance", outlier_resistance))
    
    # 4. Check for systematic biases that indicate lack of robustness
    if "step" in data.columns:
        # Check if performance degrades over time (indicating lack of robustness)
        for metric_col in ["reward_mean", "accuracy", "score"]:
            if metric_col in data.columns:
                try:
                    steps = pd.to_numeric(data["step"], errors='coerce')
                    metric_values = pd.to_numeric(data[metric_col], errors='coerce')
                    
                    # Remove NaN values
                    valid_mask = ~(steps.isna() | metric_values.isna())
                    if valid_mask.sum() > 10:
                        # Calculate trend
                        valid_steps = steps[valid_mask]
                        valid_metrics = metric_values[valid_mask]
                        
                        if len(valid_steps) > 1:
                            # Fit linear trend
                            slope = np.polyfit(valid_steps, valid_metrics, 1)[0]
                            
                            # Convert slope to robustness score (negative slope = degradation)
                            if slope >= 0:
                                trend_robustness = 1.0  # No degradation
                            else:
                                # Normalize slope to [0, 1] range
                                # Avoid division by zero when mean is zero
                                if valid_metrics.mean() != 0:
                                    max_expected_degradation = abs(valid_metrics.mean()) * 0.1  # 10% degradation
                                    normalized_slope = min(1.0, abs(slope) / max_expected_degradation)
                                    trend_robustness = max(0, 1 - normalized_slope)
                                else:
                                    # When mean is zero, use a fallback approach
                                    # Use the standard deviation as a reference for normalization
                                    if valid_metrics.std() > 0:
                                        normalized_slope = min(1.0, abs(slope) / valid_metrics.std())
                                        trend_robustness = max(0, 1 - normalized_slope)
                                    else:
                                        # If both mean and std are zero, assume no degradation
                                        trend_robustness = 1.0
                            
                            robustness_metrics.append((f"{metric_col}_trend_robustness", trend_robustness))
                except Exception:
                    continue
    
    # 5. Check for variance in performance across different conditions
    # Look for columns that might indicate different evaluation conditions
    condition_cols = [col for col in data.columns if any(keyword in col.lower() 
                                                       for keyword in ['condition', 'setting', 'context', 'split'])]
    
    for condition_col in condition_cols:
        if condition_col in data.columns and "reward_mean" in data.columns:
            # Calculate performance variance across conditions
            condition_groups = data.groupby(condition_col)["reward_mean"].agg(['mean', 'std']).dropna()
            
            if len(condition_groups) > 1:
                # Calculate coefficient of variation across conditions
                condition_means = condition_groups['mean']
                overall_mean = condition_means.mean()
                
                if overall_mean != 0:
                    condition_cv = condition_means.std() / abs(overall_mean)
                    # Lower CV across conditions = more robust
                    condition_robustness = max(0, 1 - condition_cv)
                    robustness_metrics.append((f"{condition_col}_robustness", condition_robustness))
    
    # 6. Check for gradient clipping and other stability indicators
    if "gradient_norm" in data.columns:
        grad_norms = data["gradient_norm"].dropna()
        if len(grad_norms) > 0:
            # Check if gradients are well-controlled (not exploding)
            max_grad_norm = grad_norms.max()
            if max_grad_norm < 10.0:  # Reasonable threshold
                grad_stability = 1.0
            else:
                grad_stability = max(0, 1 - (max_grad_norm - 10.0) / 10.0)
            
            robustness_metrics.append(("gradient_stability", grad_stability))
    
    # Calculate overall robustness score
    if robustness_metrics:
        scores = [score for _, score in robustness_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic reward statistics
        if "reward_mean" in data.columns:
            rewards = data["reward_mean"].dropna()
            if len(rewards) > 1:
                reward_std = rewards.std()
                reward_mean = rewards.mean()
                if reward_mean != 0:
                    cv = reward_std / abs(reward_mean)
                    overall_score = max(0, 1 - cv)
                else:
                    overall_score = 0.5
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Robustness evaluation based on {len(robustness_metrics)} metrics",
        "method": "stability_and_perturbation_analysis",
        "metrics": robustness_metrics,
        "sample_size": len(data),
    }


def evaluate_efficiency(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model efficiency in terms of computational resources and training dynamics.
    
    Measures how efficiently the model uses computational resources,
    training time, memory, and achieves convergence.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with efficiency score and details
    """
    import numpy as np
    from scipy import stats
    
    efficiency_metrics = []
    overall_score = 0.0
    
    # 1. Check for training speed metrics
    if "training_time" in data.columns and "step" in data.columns:
        # Calculate steps per second
        total_time = data["training_time"].max() - data["training_time"].min()
        total_steps = data["step"].max() - data["step"].min()
        
        if total_time > 0 and total_steps > 0:
            steps_per_second = total_steps / total_time
            # Normalize to a reasonable range (0-1000 steps/sec)
            speed_score = min(1.0, steps_per_second / 1000.0)
            efficiency_metrics.append(("training_speed", speed_score))
    
    # 2. Check for memory efficiency
    if "memory_usage" in data.columns:
        memory_values = data["memory_usage"].dropna()
        if len(memory_values) > 0:
            # Calculate memory efficiency (lower is better)
            avg_memory = memory_values.mean()
            max_memory = memory_values.max()
            
            # Normalize to reasonable range (assuming GB)
            if avg_memory < 8.0:  # Less than 8GB average
                memory_efficiency = 1.0
            else:
                memory_efficiency = max(0, 1 - (avg_memory - 8.0) / 8.0)
            
            efficiency_metrics.append(("memory_efficiency", memory_efficiency))
            
            # Check for memory stability
            memory_std = memory_values.std()
            if memory_std < 1.0:  # Stable memory usage
                memory_stability = 1.0
            else:
                memory_stability = max(0, 1 - memory_std / 4.0)
            
            efficiency_metrics.append(("memory_stability", memory_stability))
    
    # 3. Check for convergence efficiency
    if "reward_mean" in data.columns and "step" in data.columns:
        # Sort by step to ensure temporal order
        sorted_data = data.sort_values("step")
        rewards = sorted_data["reward_mean"].dropna()
        
        if len(rewards) > 20:
            # Calculate convergence metrics
            # Check if rewards are improving over time
            first_quarter = rewards[:len(rewards)//4]
            last_quarter = rewards[-len(rewards)//4:]
            
            if len(first_quarter) > 0 and len(last_quarter) > 0:
                improvement = last_quarter.mean() - first_quarter.mean()
                
                # Convert improvement to efficiency score
                if improvement > 0:
                    # Normalize improvement
                    # Avoid division by zero when mean is zero
                    if rewards.mean() != 0:
                        max_expected_improvement = abs(rewards.mean()) * 0.5
                        improvement_score = min(1.0, improvement / max_expected_improvement)
                    else:
                        # When mean is zero, use a fallback approach
                        # Use the standard deviation as a reference for normalization
                        if rewards.std() > 0:
                            max_expected_improvement = rewards.std() * 0.5
                            improvement_score = min(1.0, improvement / max_expected_improvement)
                        else:
                            # If both mean and std are zero, assume good improvement
                            improvement_score = 1.0
                else:
                    improvement_score = 0.0
                
                efficiency_metrics.append(("convergence_improvement", improvement_score))
                
                # Check for early convergence
                # Calculate when 90% of final improvement is achieved
                final_reward = rewards[-1]
                target_reward = first_quarter.mean() + 0.9 * improvement
                
                convergence_step = None
                for i, reward in enumerate(rewards):
                    if reward >= target_reward:
                        convergence_step = i
                        break
                
                if convergence_step is not None:
                    # Earlier convergence = better efficiency
                    convergence_ratio = convergence_step / len(rewards)
                    early_convergence_score = max(0, 1 - convergence_ratio)
                    efficiency_metrics.append(("early_convergence", early_convergence_score))
    
    # 4. Check for computational efficiency (FLOPs, etc.)
    if "flops" in data.columns:
        flops_values = data["flops"].dropna()
        if len(flops_values) > 0:
            # Calculate FLOPs efficiency (lower is better)
            avg_flops = flops_values.mean()
            
            # Normalize to reasonable range (assuming GFLOPs)
            if avg_flops < 1000:  # Less than 1 TFLOP
                flops_efficiency = 1.0
            else:
                flops_efficiency = max(0, 1 - (avg_flops - 1000) / 1000)
            
            efficiency_metrics.append(("flops_efficiency", flops_efficiency))
    
    # 5. Check for gradient efficiency
    if "gradient_norm" in data.columns:
        grad_norms = data["gradient_norm"].dropna()
        if len(grad_norms) > 0:
            # Well-controlled gradients indicate efficient training
            avg_grad_norm = grad_norms.mean()
            
            if avg_grad_norm < 1.0:  # Well-controlled gradients
                grad_efficiency = 1.0
            else:
                grad_efficiency = max(0, 1 - (avg_grad_norm - 1.0) / 4.0)
            
            efficiency_metrics.append(("gradient_efficiency", grad_efficiency))
    
    # 6. Check for loss efficiency (how quickly loss decreases)
    if "loss" in data.columns and "step" in data.columns:
        sorted_data = data.sort_values("step")
        losses = sorted_data["loss"].dropna()
        
        if len(losses) > 10:
            # Calculate loss reduction rate
            initial_loss = losses[:len(losses)//4].mean()
            final_loss = losses[-len(losses)//4:].mean()
            
            if initial_loss > 0:
                loss_reduction = (initial_loss - final_loss) / initial_loss
                # Higher reduction = better efficiency
                loss_efficiency = min(1.0, loss_reduction)
                efficiency_metrics.append(("loss_efficiency", loss_efficiency))
    
    # 7. Check for sample efficiency (rewards per step)
    if "reward_mean" in data.columns and "step" in data.columns:
        rewards = data["reward_mean"].dropna()
        if len(rewards) > 0:
            # Calculate average reward per step
            avg_reward = rewards.mean()
            
            # Normalize to reasonable range
            if avg_reward > 0:
                sample_efficiency = min(1.0, avg_reward / 2.0)  # Assuming max reward of 2
            else:
                sample_efficiency = max(0, 1 + avg_reward / 2.0)  # Handle negative rewards
            
            efficiency_metrics.append(("sample_efficiency", sample_efficiency))
    
    # Calculate overall efficiency score
    if efficiency_metrics:
        scores = [score for _, score in efficiency_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic metrics
        if "reward_mean" in data.columns:
            rewards = data["reward_mean"].dropna()
            if len(rewards) > 0:
                # Use reward mean as proxy for efficiency
                normalized_reward = (rewards.mean() + 1) / 2  # Assume [-1, 1] range
                overall_score = max(0, min(1, normalized_reward))
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Efficiency evaluation based on {len(efficiency_metrics)} metrics",
        "method": "computational_and_convergence_analysis",
        "metrics": efficiency_metrics,
        "sample_size": len(data),
    }


# Import the new toxicity evaluation function from metrics module
# The function is already imported at the top of the file


# Import the new bias evaluation function from metrics module
# The function is already imported at the top of the file
    



def evaluate_adversarial(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model adversarial robustness.
    
    Measures how well the model maintains performance under
    adversarial attacks and perturbations.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with adversarial robustness score and details
    """
    import numpy as np
    from scipy import stats
    
    adversarial_metrics = []
    overall_score = 0.0
    
    # 1. Check for explicit adversarial robustness scores
    if "adversarial_score" in data.columns:
        adv_scores = data["adversarial_score"].dropna()
        if len(adv_scores) > 0:
            avg_adv_score = adv_scores.mean()
            adversarial_metrics.append(("avg_adversarial_robustness", avg_adv_score))
            
            # Check for low robustness outliers
            low_robustness_threshold = 0.3
            low_robustness_ratio = (adv_scores < low_robustness_threshold).mean()
            adversarial_metrics.append(("low_robustness_ratio", low_robustness_ratio))
    
    # 2. Check for adversarial attack success rates
    if "attack_success_rate" in data.columns:
        attack_rates = data["attack_success_rate"].dropna()
        if len(attack_rates) > 0:
            avg_attack_rate = attack_rates.mean()
            # Lower attack success rate = better robustness
            inverted_attack_rate = 1 - avg_attack_rate
            adversarial_metrics.append(("inverted_attack_success", inverted_attack_rate))
    
    # 3. Check for perturbation robustness
    if "perturbation_score" in data.columns:
        pert_scores = data["perturbation_score"].dropna()
        if len(pert_scores) > 0:
            avg_pert_score = pert_scores.mean()
            adversarial_metrics.append(("perturbation_robustness", avg_pert_score))
    
    # 4. Check for reward stability under adversarial conditions
    if "reward_mean" in data.columns:
        rewards = data["reward_mean"].dropna()
        if len(rewards) > 10:
            # Calculate reward stability metrics
            reward_std = rewards.std()
            reward_mean = rewards.mean()
            
            if reward_mean != 0:
                # Coefficient of variation (lower = more robust)
                cv = reward_std / abs(reward_mean)
                stability_score = max(0, 1 - cv)
                adversarial_metrics.append(("reward_stability", stability_score))
                
                # Check for reward outliers that might indicate vulnerability
                Q1 = np.percentile(rewards, 25)
                Q3 = np.percentile(rewards, 75)
                IQR = Q3 - Q1
                
                outlier_threshold_low = Q1 - 1.5 * IQR
                outlier_threshold_high = Q3 + 1.5 * IQR
                
                outliers = np.sum((rewards < outlier_threshold_low) | (rewards > outlier_threshold_high))
                outlier_ratio = outliers / len(rewards)
                
                # Lower outlier ratio = more robust
                outlier_resistance = max(0, 1 - outlier_ratio)
                adversarial_metrics.append(("outlier_resistance", outlier_resistance))
    
    # 5. Check for gradient-based adversarial indicators
    if "gradient_norm" in data.columns:
        grad_norms = data["gradient_norm"].dropna()
        if len(grad_norms) > 0:
            # Well-controlled gradients indicate robustness
            avg_grad_norm = grad_norms.mean()
            max_grad_norm = grad_norms.max()
            
            # Check if gradients are well-controlled
            if avg_grad_norm < 1.0:  # Well-controlled gradients
                grad_robustness = 1.0
            else:
                grad_robustness = max(0, 1 - (avg_grad_norm - 1.0) / 4.0)
            
            adversarial_metrics.append(("gradient_robustness", grad_robustness))
            
            # Check for gradient explosion (indicates vulnerability)
            if max_grad_norm > 10.0:
                # Calculate penalty and invert it so that higher penalty = lower robustness
                grad_explosion_penalty = min(1.0, (max_grad_norm - 10.0) / 10.0)
                # Invert the penalty: 1 - penalty so that higher penalty = lower score
                grad_explosion_robustness = max(0, 1 - grad_explosion_penalty)
                adversarial_metrics.append(("gradient_explosion_robustness", grad_explosion_robustness))
    
    # 6. Check for performance degradation under stress
    if "step" in data.columns:
        # Check if performance degrades over time (indicating vulnerability)
        for metric_col in ["reward_mean", "accuracy", "score"]:
            if metric_col in data.columns:
                try:
                    steps = pd.to_numeric(data["step"], errors='coerce')
                    metric_values = pd.to_numeric(data[metric_col], errors='coerce')
                    
                    # Remove NaN values
                    valid_mask = ~(steps.isna() | metric_values.isna())
                    if valid_mask.sum() > 10:
                        # Calculate trend
                        valid_steps = steps[valid_mask]
                        valid_metrics = metric_values[valid_mask]
                        
                        if len(valid_steps) > 1:
                            # Fit linear trend
                            slope = np.polyfit(valid_steps, valid_metrics, 1)[0]
                            
                            # Convert slope to robustness score (negative slope = degradation)
                            if slope >= 0:
                                trend_robustness = 1.0  # No degradation
                            else:
                                # Normalize slope to [0, 1] range
                                # Avoid division by zero when mean is zero
                                if valid_metrics.mean() != 0:
                                    max_expected_degradation = abs(valid_metrics.mean()) * 0.1
                                    normalized_slope = min(1.0, abs(slope) / max_expected_degradation)
                                    trend_robustness = max(0, 1 - normalized_slope)
                                else:
                                    # When mean is zero, use a fallback approach
                                    # Use the standard deviation as a reference for normalization
                                    if valid_metrics.std() > 0:
                                        normalized_slope = min(1.0, abs(slope) / valid_metrics.std())
                                        trend_robustness = max(0, 1 - normalized_slope)
                                    else:
                                        # If both mean and std are zero, assume no degradation
                                        trend_robustness = 1.0
                            
                            adversarial_metrics.append((f"{metric_col}_trend_robustness", trend_robustness))
                except Exception:
                    continue
    
    # 7. Check for input sensitivity (high sensitivity = vulnerability)
    if "input_sensitivity" in data.columns:
        sensitivity_scores = data["input_sensitivity"].dropna()
        if len(sensitivity_scores) > 0:
            avg_sensitivity = sensitivity_scores.mean()
            # Lower sensitivity = better robustness
            inverted_sensitivity = max(0, 1 - avg_sensitivity)
            adversarial_metrics.append(("inverted_input_sensitivity", inverted_sensitivity))
    
    # 8. Check for confidence calibration under adversarial conditions
    if "confidence_score" in data.columns:
        confidence_scores = data["confidence_score"].dropna()
        if len(confidence_scores) > 0:
            # Well-calibrated confidence indicates robustness
            avg_confidence = confidence_scores.mean()
            confidence_std = confidence_scores.std()
            
            # High average confidence with low variance = good
            if avg_confidence > 0.7 and confidence_std < 0.2:
                confidence_robustness = 1.0
            else:
                confidence_robustness = max(0, avg_confidence - confidence_std)
            
            adversarial_metrics.append(("confidence_robustness", confidence_robustness))
    
    # Calculate overall adversarial robustness score
    if adversarial_metrics:
        scores = [score for _, score in adversarial_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic stability metrics
        if "reward_mean" in data.columns:
            rewards = data["reward_mean"].dropna()
            if len(rewards) > 1:
                reward_std = rewards.std()
                reward_mean = rewards.mean()
                if reward_mean != 0:
                    cv = reward_std / abs(reward_mean)
                    overall_score = max(0, 1 - cv)
                else:
                    overall_score = 0.5
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Adversarial robustness evaluation based on {len(adversarial_metrics)} metrics",
        "method": "stability_and_attack_resistance_analysis",
        "metrics": adversarial_metrics,
        "sample_size": len(data),
    }


def evaluate_speed(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model inference and training speed.
    
    Measures how quickly the model can process inputs and
    how efficiently it trains.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with speed score and details
    """
    import numpy as np
    from scipy import stats
    
    speed_metrics = []
    overall_score = 0.0
    
    # 1. Check for inference speed metrics
    if "inference_time" in data.columns:
        inference_times = data["inference_time"].dropna()
        if len(inference_times) > 0:
            avg_inference_time = inference_times.mean()
            # Lower inference time = better speed
            if avg_inference_time < 0.1:  # Less than 100ms
                inference_speed = 1.0
            else:
                inference_speed = max(0, 1 - (avg_inference_time - 0.1) / 0.9)
            
            speed_metrics.append(("inference_speed", inference_speed))
            
            # Check for speed consistency
            inference_std = inference_times.std()
            if inference_std < 0.01:  # Very consistent
                speed_consistency = 1.0
            else:
                speed_consistency = max(0, 1 - inference_std / 0.1)
            
            speed_metrics.append(("speed_consistency", speed_consistency))
    
    # 2. Check for training speed metrics
    if "training_time" in data.columns and "step" in data.columns:
        # Calculate steps per second
        total_time = data["training_time"].max() - data["training_time"].min()
        total_steps = data["step"].max() - data["step"].min()
        
        if total_time > 0 and total_steps > 0:
            steps_per_second = total_steps / total_time
            # Normalize to reasonable range (0-1000 steps/sec)
            training_speed = min(1.0, steps_per_second / 1000.0)
            speed_metrics.append(("training_speed", training_speed))
    
    # 3. Check for throughput metrics
    if "throughput" in data.columns:
        throughput_values = data["throughput"].dropna()
        if len(throughput_values) > 0:
            avg_throughput = throughput_values.mean()
            # Higher throughput = better speed
            # Normalize to reasonable range (0-1000 samples/sec)
            normalized_throughput = min(1.0, avg_throughput / 1000.0)
            speed_metrics.append(("throughput_speed", normalized_throughput))
    
    # 4. Check for latency metrics
    if "latency" in data.columns:
        latency_values = data["latency"].dropna()
        if len(latency_values) > 0:
            avg_latency = latency_values.mean()
            # Lower latency = better speed
            if avg_latency < 0.05:  # Less than 50ms
                latency_score = 1.0
            else:
                latency_score = max(0, 1 - (avg_latency - 0.05) / 0.45)
            
            speed_metrics.append(("latency_speed", latency_score))
    
    # 5. Check for batch processing speed
    if "batch_time" in data.columns and "batch_size" in data.columns:
        batch_times = data["batch_time"].dropna()
        batch_sizes = data["batch_size"].dropna()
        
        if len(batch_times) > 0 and len(batch_sizes) > 0:
            # Calculate samples per second
            samples_per_second = batch_sizes / batch_times
            avg_samples_per_second = samples_per_second.mean()
            
            # Normalize to reasonable range
            batch_speed = min(1.0, avg_samples_per_second / 1000.0)
            speed_metrics.append(("batch_speed", batch_speed))
    
    # 6. Check for convergence speed (how quickly model improves)
    if "reward_mean" in data.columns and "step" in data.columns:
        # Sort by step to ensure temporal order
        sorted_data = data.sort_values("step")
        rewards = sorted_data["reward_mean"].dropna()
        
        if len(rewards) > 20:
            # Calculate improvement rate
            first_quarter = rewards[:len(rewards)//4]
            last_quarter = rewards[-len(rewards)//4:]
            
            if len(first_quarter) > 0 and len(last_quarter) > 0:
                improvement = last_quarter.mean() - first_quarter.mean()
                steps_taken = len(rewards)
                
                if improvement > 0 and steps_taken > 0:
                    # Improvement per step
                    improvement_rate = improvement / steps_taken
                    # Normalize to reasonable range
                    convergence_speed = min(1.0, improvement_rate * 1000)
                    speed_metrics.append(("convergence_speed", convergence_speed))
    
    # 7. Check for memory access speed
    if "memory_access_time" in data.columns:
        access_times = data["memory_access_time"].dropna()
        if len(access_times) > 0:
            avg_access_time = access_times.mean()
            # Lower access time = better speed
            if avg_access_time < 0.001:  # Less than 1ms
                memory_speed = 1.0
            else:
                memory_speed = max(0, 1 - (avg_access_time - 0.001) / 0.009)
            
            speed_metrics.append(("memory_speed", memory_speed))
    
    # 8. Check for GPU utilization efficiency
    if "gpu_utilization" in data.columns:
        gpu_util = data["gpu_utilization"].dropna()
        if len(gpu_util) > 0:
            avg_gpu_util = gpu_util.mean()
            # Higher GPU utilization = better speed (assuming efficient usage)
            speed_metrics.append(("gpu_efficiency", avg_gpu_util))
    
    # Calculate overall speed score
    if speed_metrics:
        scores = [score for _, score in speed_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic metrics
        if "reward_mean" in data.columns:
            rewards = data["reward_mean"].dropna()
            if len(rewards) > 0:
                # Use reward mean as proxy for speed (higher rewards might indicate faster learning)
                normalized_reward = (rewards.mean() + 1) / 2  # Assume [-1, 1] range
                overall_score = max(0, min(1, normalized_reward))
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Speed evaluation based on {len(speed_metrics)} metrics",
        "method": "inference_and_training_analysis",
        "metrics": speed_metrics,
        "sample_size": len(data),
    }


def evaluate_memory(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model memory usage and efficiency.
    
    Measures how efficiently the model uses memory resources
    and how well it manages memory during training and inference.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with memory score and details (lower is better)
    """
    import numpy as np
    from scipy import stats
    
    memory_metrics = []
    overall_score = 0.0
    
    # 1. Check for memory usage metrics
    if "memory_usage" in data.columns:
        memory_values = data["memory_usage"].dropna()
        if len(memory_values) > 0:
            avg_memory = memory_values.mean()
            max_memory = memory_values.max()
            min_memory = memory_values.min()
            
            # Calculate memory efficiency (lower is better)
            # Assuming reasonable range is 0-32GB
            if avg_memory < 8.0:  # Less than 8GB average
                memory_efficiency = 1.0
            else:
                memory_efficiency = max(0, 1 - (avg_memory - 8.0) / 24.0)
            
            memory_metrics.append(("memory_efficiency", memory_efficiency))
            
            # Check for memory stability
            memory_std = memory_values.std()
            if memory_std < 1.0:  # Very stable
                memory_stability = 1.0
            else:
                memory_stability = max(0, 1 - memory_std / 4.0)
            
            memory_metrics.append(("memory_stability", memory_stability))
            
            # Check for memory spikes
            memory_range = max_memory - min_memory
            if memory_range < 2.0:  # Small range
                memory_consistency = 1.0
            else:
                memory_consistency = max(0, 1 - memory_range / 16.0)
            
            memory_metrics.append(("memory_consistency", memory_consistency))
    
    # 2. Check for GPU memory usage
    if "gpu_memory" in data.columns:
        gpu_memory_values = data["gpu_memory"].dropna()
        if len(gpu_memory_values) > 0:
            avg_gpu_memory = gpu_memory_values.mean()
            max_gpu_memory = gpu_memory_values.max()
            
            # Calculate GPU memory efficiency (lower is better)
            # Assuming reasonable range is 0-24GB
            if avg_gpu_memory < 6.0:  # Less than 6GB average
                gpu_memory_efficiency = 1.0
            else:
                gpu_memory_efficiency = max(0, 1 - (avg_gpu_memory - 6.0) / 18.0)
            
            memory_metrics.append(("gpu_memory_efficiency", gpu_memory_efficiency))
            
            # Check for GPU memory utilization
            if max_gpu_memory > 0:
                gpu_utilization = avg_gpu_memory / max_gpu_memory
                # Moderate utilization is good (not too low, not too high)
                if 0.3 <= gpu_utilization <= 0.8:
                    gpu_utilization_score = 1.0
                else:
                    gpu_utilization_score = max(0, 1 - abs(gpu_utilization - 0.55) / 0.55)
                
                memory_metrics.append(("gpu_utilization_score", gpu_utilization_score))
    
    # 3. Check for memory leaks (increasing memory over time)
    if "memory_usage" in data.columns and "step" in data.columns:
        # Sort by step to ensure temporal order
        sorted_data = data.sort_values("step")
        memory_over_time = sorted_data["memory_usage"].dropna()
        
        if len(memory_over_time) > 10:
            # Check for memory growth trend
            steps = np.arange(len(memory_over_time))
            if len(steps) > 1:
                slope = np.polyfit(steps, memory_over_time, 1)[0]
                
                # Negative slope = memory decreasing (good)
                # Positive slope = memory increasing (potential leak)
                if slope <= 0:
                    memory_leak_score = 1.0  # No leak
                else:
                    # Normalize slope to [0, 1] range
                    max_expected_growth = memory_over_time.mean() * 0.1  # 10% growth
                    normalized_slope = min(1.0, slope / max_expected_growth)
                    memory_leak_score = max(0, 1 - normalized_slope)
                
                memory_metrics.append(("memory_leak_resistance", memory_leak_score))
    
    # 4. Check for memory fragmentation
    if "memory_fragmentation" in data.columns:
        frag_values = data["memory_fragmentation"].dropna()
        if len(frag_values) > 0:
            avg_fragmentation = frag_values.mean()
            # Lower fragmentation = better
            fragmentation_score = max(0, 1 - avg_fragmentation)
            memory_metrics.append(("fragmentation_resistance", fragmentation_score))
    
    # 5. Check for cache efficiency
    if "cache_hit_rate" in data.columns:
        cache_rates = data["cache_hit_rate"].dropna()
        if len(cache_rates) > 0:
            avg_cache_rate = cache_rates.mean()
            # Higher cache hit rate = better memory efficiency
            memory_metrics.append(("cache_efficiency", avg_cache_rate))
    
    # 6. Check for memory bandwidth utilization
    if "memory_bandwidth" in data.columns:
        bandwidth_values = data["memory_bandwidth"].dropna()
        if len(bandwidth_values) > 0:
            avg_bandwidth = bandwidth_values.mean()
            # Moderate bandwidth utilization is good
            if 0.3 <= avg_bandwidth <= 0.8:
                bandwidth_score = 1.0
            else:
                bandwidth_score = max(0, 1 - abs(avg_bandwidth - 0.55) / 0.55)
            
            memory_metrics.append(("bandwidth_efficiency", bandwidth_score))
    
    # 7. Check for memory allocation patterns
    if "memory_allocation_count" in data.columns:
        alloc_counts = data["memory_allocation_count"].dropna()
        if len(alloc_counts) > 0:
            avg_alloc_count = alloc_counts.mean()
            # Fewer allocations = better efficiency
            if avg_alloc_count < 1000:
                allocation_efficiency = 1.0
            else:
                allocation_efficiency = max(0, 1 - (avg_alloc_count - 1000) / 9000)
            
            memory_metrics.append(("allocation_efficiency", allocation_efficiency))
    
    # 8. Check for memory-related errors or warnings
    if "memory_errors" in data.columns:
        error_counts = data["memory_errors"].dropna()
        if len(error_counts) > 0:
            total_errors = error_counts.sum()
            if total_errors == 0:
                error_score = 1.0
            else:
                error_score = max(0, 1 - total_errors / len(data))
            
            memory_metrics.append(("error_free_memory", error_score))
    
    # Calculate overall memory score (lower is better)
    if memory_metrics:
        scores = [score for _, score in memory_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic metrics
        if "memory_usage" in data.columns:
            memory_values = data["memory_usage"].dropna()
            if len(memory_values) > 0:
                avg_memory = memory_values.mean()
                # Lower memory usage = better
                if avg_memory < 8.0:
                    overall_score = 0.8
                elif avg_memory < 16.0:
                    overall_score = 0.6
                else:
                    overall_score = 0.4
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Memory evaluation based on {len(memory_metrics)} metrics",
        "method": "usage_and_efficiency_analysis",
        "metrics": memory_metrics,
        "sample_size": len(data),
        "note": "Higher scores indicate better performance (more efficient memory usage)",
    }


# Import the new throughput evaluation function from metrics module
# The function is already imported at the top of the file


def evaluate_calibration(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model calibration and confidence estimation.
    
    Measures how well the model's confidence estimates align
    with its actual performance and accuracy.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with calibration score and details
    """
    import numpy as np
    from scipy import stats
    
    calibration_metrics = []
    overall_score = 0.0
    
    # 1. Check for confidence scores
    if "confidence_score" in data.columns:
        confidence_scores = data["confidence_score"].dropna()
        if len(confidence_scores) > 0:
            avg_confidence = confidence_scores.mean()
            confidence_std = confidence_scores.std()
            
            # Well-calibrated confidence should be moderate and stable
            if 0.3 <= avg_confidence <= 0.8:
                confidence_calibration = 1.0
            else:
                confidence_calibration = max(0, 1 - abs(avg_confidence - 0.55) / 0.55)
            
            calibration_metrics.append(("confidence_calibration", confidence_calibration))
            
            # Check for confidence stability
            if confidence_std < 0.2:  # Stable confidence
                confidence_stability = 1.0
            else:
                confidence_stability = max(0, 1 - confidence_std / 0.5)
            
            calibration_metrics.append(("confidence_stability", confidence_stability))
    
    # 2. Check for confidence-accuracy correlation
    if "confidence_score" in data.columns and "accuracy" in data.columns:
        confidence_values = data["confidence_score"].dropna()
        accuracy_values = data["accuracy"].dropna()
        
        if len(confidence_values) > 10 and len(accuracy_values) > 10:
            # Align the data
            min_len = min(len(confidence_values), len(accuracy_values))
            confidence_aligned = confidence_values[:min_len]
            accuracy_aligned = accuracy_values[:min_len]
            
            try:
                # Calculate correlation between confidence and accuracy
                correlation = np.corrcoef(confidence_aligned, accuracy_aligned)[0, 1]
                if not np.isnan(correlation):
                    # Positive correlation = good calibration
                    correlation_score = max(0, correlation)
                    calibration_metrics.append(("confidence_accuracy_correlation", correlation_score))
            except Exception:
                pass
    
    # 3. Check for reward-confidence alignment
    if "confidence_score" in data.columns and "reward_mean" in data.columns:
        confidence_values = data["confidence_score"].dropna()
        reward_values = data["reward_mean"].dropna()
        
        if len(confidence_values) > 10 and len(reward_values) > 10:
            # Align the data
            min_len = min(len(confidence_values), len(reward_values))
            confidence_aligned = confidence_values[:min_len]
            reward_aligned = reward_values[:min_len]
            
            try:
                # Calculate correlation between confidence and rewards
                correlation = np.corrcoef(confidence_aligned, reward_aligned)[0, 1]
                if not np.isnan(correlation):
                    # Positive correlation = good calibration
                    correlation_score = max(0, correlation)
                    calibration_metrics.append(("confidence_reward_correlation", correlation_score))
            except Exception:
                pass
    
    # 4. Check for overconfidence detection
    if "confidence_score" in data.columns and "ground_truth" in data.columns:
        confidence_values = data["confidence_score"].dropna()
        ground_truth_values = data["ground_truth"].dropna()
        
        if len(confidence_values) > 10 and len(ground_truth_values) > 10:
            # Align the data
            min_len = min(len(confidence_values), len(ground_truth_values))
            confidence_aligned = confidence_values[:min_len]
            ground_truth_aligned = ground_truth_values[:min_len]
            
            # Calculate overconfidence (confidence > accuracy)
            overconfidence_count = 0
            for conf, gt in zip(confidence_aligned, ground_truth_aligned):
                if conf > gt + 0.1:  # Confidence significantly higher than ground truth
                    overconfidence_count += 1
            
            overconfidence_ratio = overconfidence_count / min_len
            # Lower overconfidence = better calibration
            overconfidence_score = max(0, 1 - overconfidence_ratio)
            calibration_metrics.append(("overconfidence_resistance", overconfidence_score))
    
    # 5. Check for uncertainty estimation
    if "uncertainty_score" in data.columns:
        uncertainty_values = data["uncertainty_score"].dropna()
        if len(uncertainty_values) > 0:
            avg_uncertainty = uncertainty_values.mean()
            uncertainty_std = uncertainty_values.std()
            
            # Moderate uncertainty is good (not too low, not too high)
            if 0.1 <= avg_uncertainty <= 0.5:
                uncertainty_calibration = 1.0
            else:
                uncertainty_calibration = max(0, 1 - abs(avg_uncertainty - 0.3) / 0.7)
            
            calibration_metrics.append(("uncertainty_calibration", uncertainty_calibration))
    
    # 6. Check for entropy-based calibration
    if "entropy_mean" in data.columns:
        entropy_values = data["entropy_mean"].dropna()
        if len(entropy_values) > 0:
            avg_entropy = entropy_values.mean()
            entropy_std = entropy_values.std()
            
            # Moderate entropy indicates good calibration
            if 0.5 <= avg_entropy <= 2.0:
                entropy_calibration = 1.0
            else:
                entropy_calibration = max(0, 1 - abs(avg_entropy - 1.25) / 1.75)
            
            calibration_metrics.append(("entropy_calibration", entropy_calibration))
    
    # 7. Check for KL divergence calibration
    if "kl_mean" in data.columns:
        kl_values = data["kl_mean"].dropna()
        if len(kl_values) > 0:
            avg_kl = kl_values.mean()
            
            # Lower KL divergence = better calibration
            if avg_kl < 0.1:
                kl_calibration = 1.0
            else:
                kl_calibration = max(0, 1 - avg_kl / 1.0)
            
            calibration_metrics.append(("kl_calibration", kl_calibration))
    
    # 8. Check for temperature scaling indicators
    if "temperature" in data.columns:
        temperature_values = data["temperature"].dropna()
        if len(temperature_values) > 0:
            avg_temperature = temperature_values.mean()
            
            # Temperature close to 1.0 indicates good calibration
            if 0.8 <= avg_temperature <= 1.2:
                temperature_calibration = 1.0
            else:
                temperature_calibration = max(0, 1 - abs(avg_temperature - 1.0) / 0.5)
            
            calibration_metrics.append(("temperature_calibration", temperature_calibration))
    
    # 9. Check for reliability diagram metrics
    if "reliability_score" in data.columns:
        reliability_values = data["reliability_score"].dropna()
        if len(reliability_values) > 0:
            avg_reliability = reliability_values.mean()
            # Higher reliability = better calibration
            calibration_metrics.append(("reliability_calibration", avg_reliability))
    
    # 10. Check for expected calibration error
    if "expected_calibration_error" in data.columns:
        ece_values = data["expected_calibration_error"].dropna()
        if len(ece_values) > 0:
            avg_ece = ece_values.mean()
            # Lower ECE = better calibration
            ece_score = max(0, 1 - avg_ece / 0.1)
            calibration_metrics.append(("ece_calibration", ece_score))
    
    # Calculate overall calibration score
    if calibration_metrics:
        scores = [score for _, score in calibration_metrics]
        overall_score = np.mean(scores)
    else:
        # Fallback: use basic metrics
        if "confidence_score" in data.columns:
            confidence_values = data["confidence_score"].dropna()
            if len(confidence_values) > 0:
                avg_confidence = confidence_values.mean()
                # Moderate confidence indicates good calibration
                if 0.3 <= avg_confidence <= 0.8:
                    overall_score = 0.7
                else:
                    overall_score = 0.4
            else:
                overall_score = 0.5
        else:
            overall_score = 0.5
    
    return {
        "score": float(overall_score),
        "details": f"Calibration evaluation based on {len(calibration_metrics)} metrics",
        "method": "confidence_and_uncertainty_analysis",
        "metrics": calibration_metrics,
        "sample_size": len(data),
    }
