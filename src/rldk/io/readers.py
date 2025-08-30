"""Readers and writers for training metrics data."""

import json
from pathlib import Path
from typing import Union, List
import pandas as pd

from .schema import TrainingMetrics, MetricsSchema


def read_metrics_jsonl(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read metrics from JSONL file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    
    metrics = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                metrics.append(TrainingMetrics(**data))
            except (json.JSONDecodeError, Exception) as e:
                raise ValueError(f"Error parsing line {line_num}: {e}")
    
    schema = MetricsSchema(metrics=metrics)
    return schema.to_dataframe()


def write_metrics_jsonl(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Write metrics DataFrame to JSONL file."""
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    schema = MetricsSchema.from_dataframe(df)
    
    with open(file_path, 'w') as f:
        for metric in schema.metrics:
            json.dump(metric.model_dump(), f)
            f.write('\n')
