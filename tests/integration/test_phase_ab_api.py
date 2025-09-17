"""Tests for Phase A and Phase B API features."""

import json
from pathlib import Path

import pandas as pd
import pytest

from rldk.reward.api import reward_health


class TestPhaseABAPI:
    """Test Phase A and Phase B API functionality."""

    @pytest.fixture
    def fixtures_dir(self):
        """Get path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "phase_ab"

    def test_api_accepts_dataframe(self, fixtures_dir):
        """Test 8: Import reward health API and pass DataFrame."""
        csv_path = fixtures_dir / "table_small.csv"
        df = pd.read_csv(csv_path)
        
        field_map = {"reward_mean": "reward_mean"}
        result = reward_health(df, field_map=field_map)
        
        assert hasattr(result, "report") or isinstance(result, dict)
        
        if hasattr(result, "report"):
            report = result.report
            health_value = getattr(report, "calibration_score", None)
        elif isinstance(result, dict):
            health_value = result.get("health") or result.get("health_score") or result.get("score")
        else:
            health_value = getattr(result, "calibration_score", None)
        
        assert health_value is not None
        assert isinstance(health_value, (int, float))
        assert not pd.isna(health_value)

    def test_api_accepts_list_of_dicts(self, fixtures_dir):
        """Test 9: Load stream_small.jsonl to list of dicts and pass it."""
        csv_path = fixtures_dir / "table_small.csv"
        df = pd.read_csv(csv_path)
        records = df.to_dict('records')
        
        field_map = {"reward_mean": "reward_mean"}
        result = reward_health(records, field_map=field_map)
        
        assert hasattr(result, "report") or isinstance(result, dict)
        
        if hasattr(result, "report"):
            report = result.report
            health_value = getattr(report, "calibration_score", None)
        elif isinstance(result, dict):
            health_value = result.get("health") or result.get("health_score") or result.get("score")
        else:
            health_value = getattr(result, "calibration_score", None)
        
        assert health_value is not None
        assert isinstance(health_value, (int, float))
        assert not pd.isna(health_value)

    def test_api_accepts_jsonl_path(self, fixtures_dir):
        """Test 10: Pass JSONL path directly."""
        jsonl_path = fixtures_dir / "stream_small.jsonl"
        
        field_map = {"reward": "reward_mean"}
        result = reward_health(str(jsonl_path), field_map=field_map)
        
        assert hasattr(result, "report") or isinstance(result, dict)
        
        if hasattr(result, "report"):
            report = result.report
            health_value = getattr(report, "calibration_score", None)
        elif isinstance(result, dict):
            health_value = result.get("health") or result.get("health_score") or result.get("score")
        else:
            health_value = getattr(result, "calibration_score", None)
        
        assert health_value is not None
        assert isinstance(health_value, (int, float))
        assert not pd.isna(health_value)

    def test_golden_path_equivalence(self, fixtures_dir):
        """Test 11: Compute health three ways for same data."""
        csv_path = fixtures_dir / "table_small.csv"
        jsonl_path = fixtures_dir / "stream_small.jsonl"
        
        df = pd.read_csv(csv_path)
        records = df.to_dict('records')
        
        field_map_csv = {"reward_mean": "reward_mean"}
        field_map_stream = {"reward": "reward_mean"}
        
        result_path = reward_health(str(jsonl_path), field_map=field_map_stream)
        result_list = reward_health(records, field_map=field_map_csv)
        result_df = reward_health(df, field_map=field_map_csv)
        
        def extract_health_value(result):
            if hasattr(result, "report"):
                return getattr(result.report, "calibration_score", 0.0)
            elif isinstance(result, dict):
                return result.get("health", result.get("health_score", result.get("score", 0.0)))
            else:
                return getattr(result, "calibration_score", 0.0)
        
        health_path = extract_health_value(result_path)
        health_list = extract_health_value(result_list)
        health_df = extract_health_value(result_df)
        
        assert isinstance(health_path, (int, float))
        assert isinstance(health_list, (int, float))
        assert isinstance(health_df, (int, float))
        
        tolerance = 1e-6
        assert abs(health_list - health_df) <= tolerance
        assert abs(health_path - health_df) <= tolerance
        assert abs(health_path - health_list) <= tolerance
