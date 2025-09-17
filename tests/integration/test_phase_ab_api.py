"""Phase B API acceptance coverage."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rldk.ingest.stream_normalizer import stream_jsonl_to_dataframe
from rldk.reward.api import HealthAnalysisResult, reward_health


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase_ab"


def _extract_health(result: HealthAnalysisResult) -> float:
    value = float(result.report.calibration_score)
    assert np.isfinite(value)
    return value


def _load_stream_as_dicts(path: Path) -> Sequence[dict]:
    df = stream_jsonl_to_dataframe(path)
    return df.to_dict(orient="records")


def test_api_accepts_dataframe() -> None:
    df = stream_jsonl_to_dataframe(FIXTURES / "stream_small.jsonl")

    result = reward_health(df)

    assert isinstance(result, HealthAnalysisResult)
    _extract_health(result)
    assert not result.metrics.empty


def test_api_accepts_list_of_dicts() -> None:
    records = _load_stream_as_dicts(FIXTURES / "stream_small.jsonl")

    result = reward_health(records)

    assert isinstance(result, HealthAnalysisResult)
    _extract_health(result)


def test_api_accepts_jsonl_path() -> None:
    path = FIXTURES / "stream_small.jsonl"

    result = reward_health(path)

    assert isinstance(result, HealthAnalysisResult)
    _extract_health(result)


def test_api_golden_path_equivalence() -> None:
    path = FIXTURES / "stream_small.jsonl"
    df = stream_jsonl_to_dataframe(path)
    records = df.to_dict(orient="records")

    from_dataframe = reward_health(df)
    from_records = reward_health(records)
    from_path = reward_health(path)

    scores = [_extract_health(item) for item in (from_dataframe, from_records, from_path)]
    assert max(scores) - min(scores) <= 1e-9
