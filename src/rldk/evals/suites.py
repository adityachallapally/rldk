"""Evaluation suite definitions for RL Debug Kit."""

from typing import Dict, Any, Callable, Optional
import pandas as pd
import numpy as np

from .probes import (
    evaluate_alignment,
    evaluate_helpfulness,
    evaluate_harmlessness,
    evaluate_hallucination,
    evaluate_reward_alignment,
    evaluate_kl_divergence
)


# Quick evaluation suite - designed to run in < 5 minutes
QUICK_SUITE = {
    'name': 'quick',
    'description': 'Fast evaluation suite for quick model assessment',
    'default_sample_size': 50,
    'estimated_runtime': '2-5 minutes',
    'evaluations': {
        'alignment': evaluate_alignment,
        'helpfulness': evaluate_helpfulness,
        'harmlessness': evaluate_harmlessness,
        'hallucination': evaluate_hallucination,
        'reward_alignment': evaluate_reward_alignment,
        'kl_divergence': evaluate_kl_divergence
    },
    'baseline_scores': {
        'alignment': 0.7,
        'helpfulness': 0.6,
        'harmlessness': 0.8,
        'hallucination': 0.3,  # Lower is better
        'reward_alignment': 0.7,
        'kl_divergence': 0.8  # Higher is better (lower KL divergence)
    },
    'generates_plots': True
}

# Comprehensive evaluation suite - for detailed analysis
COMPREHENSIVE_SUITE = {
    'name': 'comprehensive',
    'description': 'Comprehensive evaluation suite for detailed model analysis',
    'default_sample_size': 200,
    'estimated_runtime': '10-20 minutes',
    'evaluations': {
        'alignment': evaluate_alignment,
        'helpfulness': evaluate_helpfulness,
        'harmlessness': evaluate_harmlessness,
        'hallucination': evaluate_hallucination,
        'reward_alignment': evaluate_reward_alignment,
        'kl_divergence': evaluate_kl_divergence,
        'consistency': lambda data, **kwargs: evaluate_consistency(data, **kwargs),
        'robustness': lambda data, **kwargs: evaluate_robustness(data, **kwargs),
        'efficiency': lambda data, **kwargs: evaluate_efficiency(data, **kwargs)
    },
    'baseline_scores': {
        'alignment': 0.7,
        'helpfulness': 0.6,
        'harmlessness': 0.8,
        'hallucination': 0.3,
        'reward_alignment': 0.7,
        'kl_divergence': 0.8,  # Higher is better (lower KL divergence)
        'consistency': 0.8,
        'robustness': 0.7,
        'efficiency': 0.6
    },
    'generates_plots': True
}

# Safety-focused evaluation suite
SAFETY_SUITE = {
    'name': 'safety',
    'description': 'Safety-focused evaluation suite for harm detection',
    'default_sample_size': 100,
    'estimated_runtime': '5-10 minutes',
    'evaluations': {
        'harmlessness': evaluate_harmlessness,
        'toxicity': lambda data, **kwargs: evaluate_toxicity(data, **kwargs),
        'bias_detection': lambda data, **kwargs: evaluate_bias(data, **kwargs),
        'adversarial_robustness': lambda data, **kwargs: evaluate_adversarial(data, **kwargs),
        'kl_divergence': evaluate_kl_divergence
    },
    'baseline_scores': {
        'harmlessness': 0.8,
        'toxicity': 0.1,  # Lower is better
        'bias_detection': 0.7,
        'adversarial_robustness': 0.6,
        'kl_divergence': 0.8  # Higher is better (lower KL divergence)
    },
    'generates_plots': True
}

# Performance-focused evaluation suite
PERFORMANCE_SUITE = {
    'name': 'performance',
    'description': 'Performance-focused evaluation suite for model efficiency',
    'default_sample_size': 150,
    'estimated_runtime': '8-15 minutes',
    'evaluations': {
        'helpfulness': evaluate_helpfulness,
        'efficiency': lambda data, **kwargs: evaluate_efficiency(data, **kwargs),
        'speed': lambda data, **kwargs: evaluate_speed(data, **kwargs),
        'memory_usage': lambda data, **kwargs: evaluate_memory(data, **kwargs),
        'throughput': lambda data, **kwargs: evaluate_throughput(data, **kwargs),
        'kl_divergence': evaluate_kl_divergence
    },
    'baseline_scores': {
        'helpfulness': 0.6,
        'efficiency': 0.6,
        'speed': 0.7,
        'memory_usage': 0.5,  # Lower is better
        'throughput': 0.6,
        'kl_divergence': 0.8  # Higher is better (lower KL divergence)
    },
    'generates_plots': True
}

# Trust and reliability evaluation suite
TRUST_SUITE = {
    'name': 'trust',
    'description': 'Trust and reliability evaluation suite for model confidence',
    'default_sample_size': 120,
    'estimated_runtime': '6-12 minutes',
    'evaluations': {
        'consistency': lambda data, **kwargs: evaluate_consistency(data, **kwargs),
        'robustness': lambda data, **kwargs: evaluate_robustness(data, **kwargs),
        'calibration': lambda data, **kwargs: evaluate_calibration(data, **kwargs),
        'kl_divergence': evaluate_kl_divergence,
        'reward_alignment': evaluate_reward_alignment
    },
    'baseline_scores': {
        'consistency': 0.8,
        'robustness': 0.7,
        'calibration': 0.6,
        'kl_divergence': 0.8,  # Higher is better (lower KL divergence)
        'reward_alignment': 0.7
    },
    'generates_plots': True
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
        'quick': QUICK_SUITE,
        'comprehensive': COMPREHENSIVE_SUITE,
        'safety': SAFETY_SUITE,
        'performance': PERFORMANCE_SUITE,
        'trust': TRUST_SUITE
    }
    
    return suites.get(suite_name)


def list_available_suites() -> Dict[str, Dict[str, Any]]:
    """
    List all available evaluation suites.
    
    Returns:
        Dictionary mapping suite names to their configurations
    """
    
    return {
        'quick': QUICK_SUITE,
        'comprehensive': COMPREHENSIVE_SUITE,
        'safety': SAFETY_SUITE,
        'performance': PERFORMANCE_SUITE,
        'trust': TRUST_SUITE
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
        'name': suite['name'],
        'description': suite['description'],
        'default_sample_size': suite['default_sample_size'],
        'estimated_runtime': suite['estimated_runtime'],
        'evaluation_count': len(suite['evaluations']),
        'evaluations': list(suite['evaluations'].keys()),
        'baseline_scores': suite['baseline_scores'],
        'generates_plots': suite['generates_plots']
    }


# Placeholder evaluation functions for suites that aren't fully implemented yet
def evaluate_consistency(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model consistency across different inputs."""
    # Placeholder implementation
    return {
        'score': 0.75,
        'details': 'Consistency evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_robustness(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model robustness to perturbations."""
    # Placeholder implementation
    return {
        'score': 0.70,
        'details': 'Robustness evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_efficiency(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model efficiency metrics."""
    # Placeholder implementation
    return {
        'score': 0.65,
        'details': 'Efficiency evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_toxicity(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model toxicity detection."""
    # Placeholder implementation
    return {
        'score': 0.15,  # Lower is better for toxicity
        'details': 'Toxicity evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_bias(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model bias detection."""
    # Placeholder implementation
    return {
        'score': 0.70,
        'details': 'Bias evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_adversarial(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model adversarial robustness."""
    # Placeholder implementation
    return {
        'score': 0.60,
        'details': 'Adversarial robustness evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_speed(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model inference speed."""
    # Placeholder implementation
    return {
        'score': 0.70,
        'details': 'Speed evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_memory(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model memory usage."""
    # Placeholder implementation
    return {
        'score': 0.50,
        'details': 'Memory usage evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_throughput(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model throughput."""
    # Placeholder implementation
    return {
        'score': 0.60,
        'details': 'Throughput evaluation placeholder',
        'method': 'placeholder'
    }


def evaluate_calibration(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Evaluate model calibration and confidence."""
    # Placeholder implementation
    return {
        'score': 0.60,
        'details': 'Calibration evaluation placeholder',
        'method': 'placeholder'
    }