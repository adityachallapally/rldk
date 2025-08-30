"""Determinism checking for training runs."""

import os
import subprocess
import re
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from ..io import read_metrics_jsonl, write_metrics_jsonl


@dataclass
class DeterminismReport:
    """Report of determinism check results."""
    
    passed: bool
    culprit: Optional[str]
    fixes: List[str]
    replica_variance: Dict[str, float]
    rng_map: Dict[str, str]
    mismatches: List[Dict[str, Any]]


def check(
    cmd: str,
    compare: List[str],
    steps: Optional[List[int]] = None,
    replicas: int = 5,
    device: Optional[str] = None
) -> DeterminismReport:
    """
    Check if a training command is deterministic.
    
    Args:
        cmd: Command to run
        compare: List of metric names to compare
        steps: Specific steps to compare, or None for all
        replicas: Number of replicas to run
        device: Device to use (auto-detected if None)
    
    Returns:
        DeterminismReport with analysis results
    """
    # Auto-detect device
    if device is None:
        device = _detect_device()
    
    # Set deterministic environment
    env = _get_deterministic_env(device)
    
    # Run multiple replicas
    print(f"Running {replicas} replicas for determinism check...")
    replica_results = []
    
    for i in range(replicas):
        print(f"Running replica {i+1}/{replicas}...")
        result = _run_deterministic_cmd(cmd, env, replica_id=i)
        replica_results.append(result)
    
    # Compare results
    mismatches = _compare_replicas(replica_results, compare, steps)
    
    # Parse stderr for non-deterministic operations
    culprit, fixes = _parse_nondeterministic_ops([r.stderr for r in replica_results])
    
    # Calculate variance across replicas
    replica_variance = _calculate_replica_variance(replica_results, compare)
    
    # Create RNG map
    rng_map = _create_rng_map(env)
    
    # Determine if passed
    passed = len(mismatches) == 0
    
    return DeterminismReport(
        passed=passed,
        culprit=culprit,
        fixes=fixes,
        replica_variance=replica_variance,
        rng_map=rng_map,
        mismatches=mismatches
    )


def _detect_device() -> str:
    """Auto-detect available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def _get_deterministic_env(device: str) -> Dict[str, str]:
    """Get environment variables for deterministic execution."""
    env = os.environ.copy()
    
    # PyTorch deterministic settings
    env.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_USE_CUDA_DSA': '1',
    })
    
    # Set deterministic flags
    if device == "cuda":
        env.update({
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'CUDA_LAUNCH_BLOCKING': '1',
        })
    
    # Python deterministic settings
    env.update({
        'PYTHONHASHSEED': '42',
        'PYTHONUNBUFFERED': '1',
    })
    
    return env


def _run_deterministic_cmd(
    cmd: str, 
    env: Dict[str, str], 
    replica_id: int
) -> subprocess.CompletedProcess:
    """Run a command with deterministic settings."""
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file = f.name
    
    # Modify command to output to our file
    modified_cmd = f"{cmd} --output {output_file}"
    
    try:
        # Run the command
        result = subprocess.run(
            modified_cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Read output file if it exists
        if Path(output_file).exists():
            try:
                df = read_metrics_jsonl(output_file)
                result.metrics_df = df
            except Exception as e:
                print(f"Warning: Could not read metrics from {output_file}: {e}")
                result.metrics_df = pd.DataFrame()
        else:
            result.metrics_df = pd.DataFrame()
        
        result.output_file = output_file
        
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(
            args=modified_cmd,
            returncode=-1,
            stdout="",
            stderr="Command timed out"
        )
        result.metrics_df = pd.DataFrame()
        result.output_file = output_file
    
    return result


def _compare_replicas(
    replica_results: List[subprocess.CompletedProcess],
    compare: List[str],
    steps: Optional[List[int]]
) -> List[Dict[str, Any]]:
    """Compare metrics across replicas."""
    mismatches = []
    
    # Get the first replica as reference
    if not replica_results or replica_results[0].metrics_df.empty:
        return mismatches
    
    reference_df = replica_results[0].metrics_df
    
    # Determine steps to compare
    if steps is None:
        steps_to_compare = reference_df['step'].tolist()
    else:
        steps_to_compare = [s for s in steps if s in reference_df['step'].values]
    
    # Compare each replica against the reference
    for i, result in enumerate(replica_results[1:], 1):
        if result.metrics_df.empty:
            mismatches.append({
                'replica': i,
                'issue': 'No metrics data available',
                'details': 'Replica failed to produce metrics'
            })
            continue
        
        df = result.metrics_df
        
        for step in steps_to_compare:
            if step not in df['step'].values:
                continue
            
            ref_row = reference_df[reference_df['step'] == step].iloc[0]
            rep_row = df[df['step'] == step].iloc[0]
            
            for metric in compare:
                if metric not in ref_row or metric not in rep_row:
                    continue
                
                ref_val = ref_row[metric]
                rep_val = rep_row[metric]
                
                if pd.isna(ref_val) or pd.isna(rep_val):
                    continue
                
                # Check for significant difference
                if abs(ref_val - rep_val) > 1e-6:
                    mismatches.append({
                        'replica': i,
                        'step': step,
                        'metric': metric,
                        'reference_value': ref_val,
                        'replica_value': rep_val,
                        'difference': abs(ref_val - rep_val),
                        'issue': f'Metric {metric} differs at step {step}'
                    })
    
    return mismatches


def _parse_nondeterministic_ops(stderr_list: List[str]) -> tuple[Optional[str], List[str]]:
    """Parse stderr for non-deterministic operations."""
    culprit = None
    fixes = []
    
    # Common non-deterministic operation patterns
    patterns = {
        'cudnn': {
            'pattern': r'cuDNN.*non-deterministic',
            'fix': 'Set torch.backends.cudnn.deterministic = True'
        },
        'dropout': {
            'pattern': r'dropout.*non-deterministic',
            'fix': 'Use torch.nn.Dropout with deterministic=True'
        },
        'convolution': {
            'pattern': r'convolution.*non-deterministic',
            'fix': 'Set torch.backends.cudnn.benchmark = False'
        },
        'reduction': {
            'pattern': r'reduction.*non-deterministic',
            'fix': 'Use deterministic reduction operations'
        }
    }
    
    for stderr in stderr_list:
        for op_name, info in patterns.items():
            if re.search(info['pattern'], stderr, re.IGNORECASE):
                culprit = op_name
                if info['fix'] not in fixes:
                    fixes.append(info['fix'])
    
    # Add general fixes if no specific culprit found
    if not fixes:
        fixes.extend([
            'Set torch.backends.cudnn.deterministic = True',
            'Set torch.backends.cudnn.benchmark = False',
            'Use torch.manual_seed() consistently',
            'Disable dropout or use deterministic=True',
            'Use deterministic reduction operations'
        ])
    
    return culprit, fixes


def _calculate_replica_variance(
    replica_results: List[subprocess.CompletedProcess],
    compare: List[str]
) -> Dict[str, float]:
    """Calculate variance of metrics across replicas."""
    variance = {}
    
    # Collect all metrics data
    all_dfs = []
    for result in replica_results:
        if not result.metrics_df.empty:
            all_dfs.append(result.metrics_df)
    
    if len(all_dfs) < 2:
        return variance
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Calculate variance for each metric
    for metric in compare:
        if metric in combined_df.columns:
            values = combined_df[metric].dropna()
            if len(values) > 1:
                variance[metric] = float(values.var())
            else:
                variance[metric] = 0.0
    
    return variance


def _create_rng_map(env: Dict[str, str]) -> Dict[str, str]:
    """Create a map of RNG settings."""
    rng_map = {}
    
    # PyTorch settings
    rng_map['torch_seed'] = 'Set via torch.manual_seed()'
    rng_map['cuda_seed'] = 'Set via torch.cuda.manual_seed_all()'
    rng_map['numpy_seed'] = 'Set via np.random.seed()'
    rng_map['python_hash'] = env.get('PYTHONHASHSEED', 'Not set')
    
    # CUDA settings
    if 'CUBLAS_WORKSPACE_CONFIG' in env:
        rng_map['cublas_workspace'] = env['CUBLAS_WORKSPACE_CONFIG']
    
    return rng_map