"""Main evaluation runner for RL Debug Kit."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path

from .suites import get_eval_suite
from .metrics import calculate_confidence_intervals, calculate_effect_sizes


@dataclass
class EvalResult:
    """Result of an evaluation suite run."""
    
    suite_name: str
    scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    sample_size: int
    seed: int
    metadata: Dict[str, Any]
    raw_results: List[Dict[str, Any]]


def run(run_data: pd.DataFrame,
        suite: str = "quick",
        seed: int = 42,
        sample_size: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None) -> EvalResult:
    """
    Run evaluation suite on training run data.
    
    Args:
        run_data: Training run data to evaluate
        suite: Name of evaluation suite to run
        seed: Random seed for reproducibility
        sample_size: Number of samples to evaluate (None for suite default)
        output_dir: Directory to save results (optional)
    
    Returns:
        EvalResult with comprehensive evaluation metrics
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get evaluation suite
    eval_suite = get_eval_suite(suite)
    if eval_suite is None:
        raise ValueError(f"Unknown evaluation suite: {suite}")
    
    # Determine sample size - use actual data size for empty or small datasets
    if sample_size is None:
        if len(run_data) == 0:
            sample_size = 0
        elif len(run_data) <= eval_suite.get('default_sample_size', 100):
            sample_size = len(run_data)
        else:
            sample_size = eval_suite.get('default_sample_size', 100)
    
    # Sample data if needed
    if len(run_data) > sample_size:
        sampled_data = run_data.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        sampled_data = run_data.copy()
    
    # Run evaluations
    raw_results = []
    scores = {}
    
    # If no data, return empty results
    if sample_size == 0:
        for eval_name in eval_suite['evaluations'].keys():
            raw_results.append({
                'evaluation': eval_name,
                'result': {'score': np.nan, 'details': f'{eval_name} evaluation based on 0 metrics'},
                'timestamp': pd.Timestamp.now().isoformat()
            })
            scores[eval_name] = np.nan
    else:
        for eval_name, eval_func in eval_suite['evaluations'].items():
            try:
                result = eval_func(sampled_data, seed=seed)
                raw_results.append({
                    'evaluation': eval_name,
                    'result': result,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                # Extract score if available
                if isinstance(result, dict) and 'score' in result:
                    scores[eval_name] = result['score']
                elif isinstance(result, (int, float)):
                    scores[eval_name] = float(result)
                else:
                    scores[eval_name] = np.nan
                    
            except Exception as e:
                raw_results.append({
                    'evaluation': eval_name,
                    'error': str(e),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                scores[eval_name] = np.nan
    
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(scores, sample_size)
    
    # Calculate effect sizes (compared to baseline if available)
    effect_sizes = calculate_effect_sizes(scores, eval_suite.get('baseline_scores', {}))
    
    # Create result object
    result = EvalResult(
        suite_name=suite,
        scores=scores,
        confidence_intervals=confidence_intervals,
        effect_sizes=effect_sizes,
        sample_size=sample_size,
        seed=seed,
        metadata={
            'suite_config': eval_suite,
            'run_data_shape': run_data.shape,
            'sampled_data_shape': sampled_data.shape,
            'evaluation_count': len(eval_suite['evaluations'])
        },
        raw_results=raw_results
    )
    
    # Save results if output directory specified
    if output_dir:
        save_eval_results(result, output_dir)
    
    return result


def save_eval_results(result: EvalResult, output_dir: Union[str, Path]) -> None:
    """Save evaluation results to files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSONL
    results_path = output_dir / "eval_results.jsonl"
    with open(results_path, 'w') as f:
        for raw_result in result.raw_results:
            f.write(json.dumps(raw_result) + '\n')
    
    # Save summary as JSON
    summary_path = output_dir / "eval_summary.json"
    summary = {
        'suite_name': result.suite_name,
        'scores': result.scores,
        'confidence_intervals': result.confidence_intervals,
        'effect_sizes': result.effect_sizes,
        'sample_size': result.sample_size,
        'seed': result.seed,
        'metadata': result.metadata
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate evaluation card
    generate_eval_card(result, output_dir)


def generate_eval_card(result: EvalResult, output_dir: Path) -> None:
    """Generate human-readable evaluation card."""
    
    card_path = output_dir / "eval_card.md"
    
    with open(card_path, 'w') as f:
        f.write("# Evaluation Results Card\n\n")
        f.write(f"**Suite:** {result.suite_name}\n")
        f.write(f"**Sample Size:** {result.sample_size}\n")
        f.write(f"**Seed:** {result.seed}\n")
        f.write(f"**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Overall Scores\n\n")
        f.write("| Metric | Score | Confidence Interval | Effect Size |\n")
        f.write("|--------|-------|-------------------|-------------|\n")
        
        for metric, score in result.scores.items():
            if not np.isnan(score):
                ci = result.confidence_intervals.get(metric, (np.nan, np.nan))
                effect_size = result.effect_sizes.get(metric, np.nan)
                
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "N/A"
                effect_str = f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"
                
                f.write(f"| {metric} | {score:.3f} | {ci_str} | {effect_str} |\n")
        
        f.write("\n## 🔍 Detailed Results\n\n")
        
        for raw_result in result.raw_results:
            f.write(f"### {raw_result['evaluation']}\n\n")
            
            if 'error' in raw_result:
                f.write(f"❌ **Error:** {raw_result['error']}\n\n")
            else:
                result_data = raw_result['result']
                if isinstance(result_data, dict):
                    for key, value in result_data.items():
                        if key != 'score':  # Already shown in summary
                            f.write(f"- **{key}:** {value}\n")
                else:
                    f.write(f"**Result:** {result_data}\n")
                f.write("\n")
        
        f.write("## 📁 Files Generated\n\n")
        f.write(f"- **Evaluation Card:** `{card_path.name}`\n")
        f.write(f"- **Detailed Results:** `eval_results.jsonl`\n")
        f.write(f"- **Summary:** `eval_summary.json`\n")
        
        if result.metadata.get('suite_config', {}).get('generates_plots', False):
            f.write(f"- **Plots:** `tradeoff_plots.png`\n")


def compare_evaluations(results: List[EvalResult], 
                       output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Compare multiple evaluation results.
    
    Args:
        results: List of EvalResult objects to compare
        output_dir: Directory to save comparison results
    
    Returns:
        Dictionary with comparison metrics
    """
    
    if len(results) < 2:
        raise ValueError("Need at least 2 evaluation results to compare")
    
    comparison = {
        'runs_compared': len(results),
        'run_names': [f"run_{i}" for i in range(len(results))],
        'comparisons': {}
    }
    
    # Get common metrics
    all_metrics = set()
    for result in results:
        all_metrics.update(result.scores.keys())
    
    # Compare each metric across runs
    for metric in all_metrics:
        metric_scores = []
        metric_cis = []
        
        for result in results:
            if metric in result.scores and not np.isnan(result.scores[metric]):
                metric_scores.append(result.scores[metric])
                if metric in result.confidence_intervals:
                    ci = result.confidence_intervals[metric]
                    if not np.isnan(ci[0]):
                        metric_cis.append(ci)
        
        if len(metric_scores) >= 2:
            # Calculate statistics
            comparison['comparisons'][metric] = {
                'scores': metric_scores,
                'mean': np.mean(metric_scores),
                'std': np.std(metric_scores),
                'min': np.min(metric_scores),
                'max': np.max(metric_scores),
                'range': np.max(metric_scores) - np.min(metric_scores),
                'cv': np.std(metric_scores) / np.mean(metric_scores) if np.mean(metric_scores) != 0 else np.nan
            }
            
            # Add confidence intervals if available
            if len(metric_cis) >= 2:
                comparison['comparisons'][metric]['confidence_intervals'] = metric_cis
    
    # Save comparison if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = output_dir / "eval_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
    
    return comparison