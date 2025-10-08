"""Tests for flexible data adapters."""

import csv
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml


_ORIGINAL_NAMED_TEMPFILE = tempfile.NamedTemporaryFile


def _text_mode_named_temporary_file(*args, **kwargs):
    kwargs.setdefault("mode", "w")
    kwargs.setdefault("encoding", "utf-8")
    return _ORIGINAL_NAMED_TEMPFILE(*args, **kwargs)


tempfile.NamedTemporaryFile = _text_mode_named_temporary_file

from rldk.adapters.field_resolver import SchemaError
from rldk.adapters.flexible import FlexibleDataAdapter, FlexibleJSONLAdapter


class TestFlexibleDataAdapter:
    """Test flexible data adapter functionality."""

    def test_init_with_field_map(self):
        """Test initialization with field map."""
        adapter = FlexibleDataAdapter(
            "test.jsonl",
            field_map={"step": "global_step", "reward": "reward_scalar"}
        )
        assert adapter.field_map == {"step": "global_step", "reward": "reward_scalar"}
        assert adapter.required_fields == ["step", "reward"]
        assert adapter.allow_dot_paths is True

    def test_init_with_config_file(self):
        """Test initialization with config file."""
        config_data = {
            "field_map": {
                "step": "global_step",
                "reward": "reward_scalar",
                "kl": "kl_to_ref"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            adapter = FlexibleDataAdapter("test.jsonl", config_file=config_path)
            assert adapter.field_map == config_data["field_map"]
        finally:
            Path(config_path).unlink()

    def test_init_with_custom_required_fields(self):
        """Test initialization with custom required fields."""
        adapter = FlexibleDataAdapter(
            "test.jsonl",
            required_fields=["step", "reward", "kl", "entropy"]
        )
        assert adapter.required_fields == ["step", "reward", "kl", "entropy"]

    def test_can_handle_jsonl_file(self):
        """Test can_handle with JSONL file."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write('{"step": 0, "reward": 0.5}\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            assert adapter.can_handle() is True

        Path(f.name).unlink()

    def test_can_handle_json_file(self):
        """Test can_handle with JSON file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump([{"step": 0, "reward": 0.5}], f)
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            assert adapter.can_handle() is True

        Path(f.name).unlink()

    def test_can_handle_csv_file(self):
        """Test can_handle with CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward"])
            writer.writerow([0, 0.5])
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            assert adapter.can_handle() is True

        Path(f.name).unlink()

    def test_can_handle_parquet_file(self):
        """Test can_handle with Parquet file."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df = pd.DataFrame({"step": [0], "reward": [0.5]})
            df.to_parquet(f.name, index=False)
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            assert adapter.can_handle() is True

        Path(f.name).unlink()

    def test_can_handle_directory(self):
        """Test can_handle with directory containing supported files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a JSONL file
            jsonl_file = temp_path / "data.jsonl"
            with open(jsonl_file, 'w') as f:
                f.write('{"step": 0, "reward": 0.5}\n')

            adapter = FlexibleDataAdapter(temp_path)
            assert adapter.can_handle() is True

    def test_can_handle_unsupported_file(self):
        """Test can_handle with unsupported file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write("some text")
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            assert adapter.can_handle() is False

        Path(f.name).unlink()

    def test_load_jsonl_data(self):
        """Test loading JSONL data."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"step": 0, "reward": 0.5, "kl": 0.1},
                {"step": 1, "reward": 0.6, "kl": 0.12},
                {"step": 2, "reward": 0.7, "kl": 0.14}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 3
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_load_jsonl_with_synonyms(self):
        """Test loading JSONL data with field synonyms."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1},
                {"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.12}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_load_jsonl_with_field_map(self):
        """Test loading JSONL data with explicit field mapping."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"iteration": 0, "score": 0.5, "kl_divergence": 0.1},
                {"iteration": 1, "score": 0.6, "kl_divergence": 0.12}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            field_map = {
                "step": "iteration",
                "reward": "score",
                "kl": "kl_divergence"
            }
            adapter = FlexibleDataAdapter(f.name, field_map=field_map)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_field_map_converts_iteration_to_step(self):
        """Field map entries map custom keys into the TrainingMetrics schema."""

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            data = [
                {"iteration": 0, "score": 0.5, "extra": 3.2},
                {"iteration": 1, "score": 0.6, "extra": 4.5},
            ]
            for record in data:
                f.write(json.dumps(record) + "\n")
            f.flush()

        try:
            adapter = FlexibleDataAdapter(
                f.name,
                field_map={"iteration": "step", "score": "reward"},
            )
            df = adapter.load()

            assert df["step"].tolist() == [0, 1]
            assert df["reward_mean"].tolist() == [0.5, 0.6]
            # Unknown fields should pass through untouched
            assert "iteration" in df.columns
            assert df["iteration"].tolist() == [0, 1]
            assert "extra" in df.columns
            assert df["extra"].tolist() == [3.2, 4.5]
        finally:
            Path(f.name).unlink()

    def test_preset_applies_trl_field_map(self):
        """The flexible adapter uses the shared preset registry."""

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            data = [
                {"global_step": 0, "reward": 0.25},
                {"global_step": 1, "reward": 0.50},
            ]
            for record in data:
                f.write(json.dumps(record) + "\n")
            f.flush()

        try:
            adapter = FlexibleDataAdapter(f.name, preset="trl")
            df = adapter.load()

            assert df["step"].tolist() == [0, 1]
            # Reward column is copied into the canonical TrainingMetrics column
            assert df["reward_mean"].tolist() == [0.25, 0.50]
        finally:
            Path(f.name).unlink()

    def test_load_jsonl_with_nested_fields(self):
        """Test loading JSONL data with nested fields."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {
                    "step": 0,
                    "metrics": {"reward": 0.5, "kl": 0.1},
                    "data": {"entropy": 0.8}
                },
                {
                    "step": 1,
                    "metrics": {"reward": 0.6, "kl": 0.12},
                    "data": {"entropy": 0.82}
                }
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            field_map = {
                "reward": "metrics.reward",
                "kl": "metrics.kl",
                "entropy": "data.entropy"
            }
            adapter = FlexibleDataAdapter(f.name, field_map=field_map)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1
            assert df["entropy"].iloc[0] == 0.8

        Path(f.name).unlink()

    def test_load_json_data(self):
        """Test loading JSON data."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            data = [
                {"step": 0, "reward": 0.5, "kl": 0.1},
                {"step": 1, "reward": 0.6, "kl": 0.12}
            ]
            json.dump(data, f)
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns

        Path(f.name).unlink()

    def test_load_json_single_record(self):
        """Test loading JSON data with single record."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            data = {"step": 0, "reward": 0.5, "kl": 0.1}
            json.dump(data, f)
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 1
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns

        Path(f.name).unlink()

    def test_load_csv_data(self):
        """Test loading CSV data."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "kl"])
            writer.writerow([0, 0.5, 0.1])
            writer.writerow([1, 0.6, 0.12])
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_load_parquet_data(self):
        """Test loading Parquet data."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df_input = pd.DataFrame({
                "step": [0, 1],
                "reward": [0.5, 0.6],
                "kl": [0.1, 0.12]
            })
            df_input.to_parquet(f.name, index=False)
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_load_missing_required_fields(self):
        """Test loading data with missing required fields."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = {"unrelated_field": "value"}
            f.write(json.dumps(data) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)

            with pytest.raises(SchemaError) as exc_info:
                adapter.load()

            assert "step" in str(exc_info.value)
            assert "reward" in str(exc_info.value)

    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            # Empty file
            pass

            adapter = FlexibleDataAdapter(f.name)

            with pytest.raises(Exception):  # Should raise some error
                adapter.load()

        Path(f.name).unlink()

    def test_get_metadata(self):
        """Test getting adapter metadata."""
        adapter = FlexibleDataAdapter(
            "test.jsonl",
            field_map={"step": "global_step"},
            required_fields=["step", "reward", "kl"]
        )

        metadata = adapter.get_metadata()
        assert metadata["source"] == "test.jsonl"
        assert metadata["field_map"] == {"step": "global_step"}
        assert metadata["required_fields"] == ["step", "reward", "kl"]
        assert metadata["allow_dot_paths"] is True
        assert ".jsonl" in metadata["supported_extensions"]


class TestFlexibleJSONLAdapter:
    """Test flexible JSONL adapter functionality."""

    def test_init_with_streaming(self):
        """Test initialization with streaming enabled."""
        adapter = FlexibleJSONLAdapter(
            "test.jsonl",
            stream_large_files=True
        )
        assert adapter.stream_large_files is True

    def test_can_handle_jsonl_only(self):
        """Test that JSONL adapter only handles JSONL files."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write('{"step": 0, "reward": 0.5}\n')
            f.flush()

            adapter = FlexibleJSONLAdapter(f.name)
            assert adapter.can_handle() is True

        Path(f.name).unlink()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump({"step": 0, "reward": 0.5}, f)
            f.flush()

            adapter = FlexibleJSONLAdapter(f.name)
            assert adapter.can_handle() is False

        Path(f.name).unlink()

    def test_load_small_jsonl_file(self):
        """Test loading small JSONL file (no streaming)."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"step": 0, "reward": 0.5, "kl": 0.1},
                {"step": 1, "reward": 0.6, "kl": 0.12}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            adapter = FlexibleJSONLAdapter(f.name, stream_large_files=True)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns

        Path(f.name).unlink()

    def test_load_jsonl_with_invalid_json(self):
        """Test loading JSONL file with some invalid JSON lines."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write('{"step": 0, "reward": 0.5}\n')
            f.write('invalid json line\n')
            f.write('{"step": 1, "reward": 0.6}\n')
            f.flush()

            adapter = FlexibleJSONLAdapter(f.name)
            df = adapter.load()

            # Should skip invalid line and load valid ones
            assert len(df) == 2
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[1] == 1

        Path(f.name).unlink()

    def test_load_jsonl_with_non_dict_records(self):
        """Test loading JSONL file with non-dict records."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write('{"step": 0, "reward": 0.5}\n')
            f.write('"string record"\n')
            f.write('{"step": 1, "reward": 0.6}\n')
            f.flush()

            adapter = FlexibleJSONLAdapter(f.name)
            df = adapter.load()

            # Should skip non-dict records and load valid ones
            assert len(df) == 2
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[1] == 1

        Path(f.name).unlink()

    def test_streaming_path_resolves_fields(self, tmp_path, monkeypatch):
        """Streaming mode should honor field maps and standardize output."""
        file_path = tmp_path / "large.jsonl"
        records = [
            {"iteration": 0, "score": 0.5, "kl_divergence": 0.1},
            {"iteration": 1, "score": 0.6, "kl_divergence": 0.12},
        ]

        with file_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        adapter = FlexibleJSONLAdapter(
            file_path,
            field_map={"step": "iteration", "reward": "score"},
            stream_large_files=True,
        )

        original_stat = Path.stat

        def fake_stat(path_obj: Path, *args, follow_symlinks: bool | None = None, **kwargs):
            result = original_stat(
                path_obj,
                *args,
                follow_symlinks=follow_symlinks if follow_symlinks is not None else kwargs.pop(
                    "follow_symlinks", True
                ),
                **kwargs,
            )
            if path_obj == file_path:
                values = list(result)
                values[6] = 101 * 1024 * 1024
                return type(result)(values)
            return result

        monkeypatch.setattr(Path, "stat", fake_stat)

        df = adapter.load()

        assert df["step"].tolist() == [0, 1]
        assert df["reward_mean"].tolist() == pytest.approx([0.5, 0.6])
        assert df["kl_mean"].tolist() == pytest.approx([0.1, 0.12])


class TestFlexibleAdapterIntegration:
    """Integration tests for flexible adapters."""

    def test_real_world_trl_scenario(self):
        """Test real-world TRL data scenario."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001
                },
                {
                    "step": 1,
                    "phase": "train",
                    "reward_mean": 0.6,
                    "kl_mean": 0.12,
                    "entropy_mean": 0.82,
                    "loss": 0.38,
                    "lr": 0.001
                }
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns
            assert "loss" in df.columns
            assert "lr" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_real_world_custom_jsonl_scenario(self):
        """Test real-world custom JSONL data scenario."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {
                    "global_step": 0,
                    "reward_scalar": 0.5,
                    "kl_to_ref": 0.1,
                    "entropy": 0.8,
                    "loss": 0.4,
                    "learning_rate": 0.001
                },
                {
                    "global_step": 1,
                    "reward_scalar": 0.6,
                    "kl_to_ref": 0.12,
                    "entropy": 0.82,
                    "loss": 0.38,
                    "learning_rate": 0.001
                }
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns
            assert "loss" in df.columns
            assert "lr" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_real_world_nested_data_scenario(self):
        """Test real-world nested data scenario."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {
                    "step": 0,
                    "metrics": {
                        "reward": 0.5,
                        "kl": 0.1,
                        "entropy": 0.8
                    },
                    "training": {
                        "loss": 0.4,
                        "lr": 0.001
                    }
                },
                {
                    "step": 1,
                    "metrics": {
                        "reward": 0.6,
                        "kl": 0.12,
                        "entropy": 0.82
                    },
                    "training": {
                        "loss": 0.38,
                        "lr": 0.001
                    }
                }
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            field_map = {
                "reward": "metrics.reward",
                "kl": "metrics.kl",
                "entropy": "metrics.entropy",
                "loss": "training.loss",
                "lr": "training.lr"
            }
            adapter = FlexibleDataAdapter(f.name, field_map=field_map)
            df = adapter.load()

            assert len(df) == 2
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns
            assert "loss" in df.columns
            assert "lr" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1
            assert df["entropy"].iloc[0] == 0.8
            assert df["loss"].iloc[0] == 0.4
            assert df["lr"].iloc[0] == 0.001

        Path(f.name).unlink()

    def test_error_handling_with_suggestions(self):
        """Test error handling with helpful suggestions."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = {
                "step_count": 0,
                "reward_value": 0.5,
                "kl_divergence": 0.1
            }
            f.write(json.dumps(data) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)

            with pytest.raises(SchemaError) as exc_info:
                adapter.load()

            error_message = str(exc_info.value)
            # Should contain suggestions for similar field names
            assert "step_count" in error_message
            assert "reward_value" in error_message
            # KL should be resolved automatically via synonyms
            assert "kl" not in exc_info.value.missing_fields
            # Should contain field map suggestion
            assert "field_map" in error_message.lower()
