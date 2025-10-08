"""Pydantic schemas for training metrics data."""

from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class TrainingMetrics(BaseModel):
    """Schema for individual training metrics record."""

    step: int = Field(..., description="Training step")
    phase: Optional[str] = Field(
        None, description="Training phase (e.g., 'train', 'eval')"
    )
    reward_mean: Optional[float] = Field(None, description="Mean reward")
    reward_std: Optional[float] = Field(None, description="Reward standard deviation")
    kl_mean: Optional[float] = Field(None, description="Mean KL divergence")
    entropy_mean: Optional[float] = Field(None, description="Mean entropy")
    clip_frac: Optional[float] = Field(None, description="Clipping fraction")
    grad_norm: Optional[float] = Field(None, description="Gradient norm")
    lr: Optional[float] = Field(None, description="Learning rate")
    loss: Optional[float] = Field(None, description="Training loss")
    tokens_in: Optional[int] = Field(None, description="Input tokens")
    tokens_out: Optional[int] = Field(None, description="Output tokens")
    wall_time: Optional[float] = Field(None, description="Wall time in seconds")
    seed: Optional[int] = Field(None, description="Random seed")
    run_id: Optional[str] = Field(None, description="Run identifier")
    git_sha: Optional[str] = Field(None, description="Git commit SHA")


class MetricsSchema(BaseModel):
    """Schema for a collection of training metrics."""

    metrics: List[TrainingMetrics]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([m.model_dump() for m in self.metrics])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MetricsSchema":
        """Create from pandas DataFrame."""
        metrics = []
        for _, row in df.iterrows():
            # Handle NaN values by converting to None
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None
            metrics.append(TrainingMetrics(**row_dict))
        return cls(metrics=metrics)
