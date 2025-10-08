"""Integration tests for flexible data ingestion."""

import csv
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from rldk.adapters.field_resolver import SchemaError
from rldk.adapters.flexible import FlexibleDataAdapter, FlexibleJSONLAdapter


class TestFlexibleIngestionIntegration:
    """Integration tests for flexible data ingestion scenarios."""

    def test_acceptance_check_sample_a_jsonl(self):
        """Test acceptance check: Sample A JSONL with fields: global_step, reward_scalar, kl_to_ref"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8},
                {"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.12, "entropy": 0.82},
                {"global_step": 2, "reward_scalar": 0.7, "kl_to_ref": 0.14, "entropy": 0.84}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            # Test zero-config loading
            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            # Verify canonical output columns
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns

            # Verify data values
            assert len(df) == 3
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1
            assert df["entropy"].iloc[0] == 0.8

            # Verify all records loaded correctly
            assert df["step"].tolist() == [0, 1, 2]
            assert df["reward"].tolist() == [0.5, 0.6, 0.7]
            assert df["kl"].tolist() == [0.1, 0.12, 0.14]

        Path(f.name).unlink()

    def test_acceptance_check_sample_b_csv(self):
        """Test acceptance check: Sample B CSV with fields: step, reward, kl"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "kl", "entropy"])
            writer.writerow([0, 0.5, 0.1, 0.8])
            writer.writerow([1, 0.6, 0.12, 0.82])
            writer.writerow([2, 0.7, 0.14, 0.84])
            f.flush()

            # Test zero-config loading
            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            # Verify canonical output columns
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns

            # Verify data values
            assert len(df) == 3
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1
            assert df["entropy"].iloc[0] == 0.8

        Path(f.name).unlink()

    def test_acceptance_check_sample_c_parquet(self):
        """Test acceptance check: Sample C Parquet with fields: iteration, score, metrics.kl_ref"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            # Create data with nested structure
            data = [
                {
                    "iteration": 0,
                    "score": 0.5,
                    "metrics": {"kl_ref": 0.1, "entropy": 0.8}
                },
                {
                    "iteration": 1,
                    "score": 0.6,
                    "metrics": {"kl_ref": 0.12, "entropy": 0.82}
                },
                {
                    "iteration": 2,
                    "score": 0.7,
                    "metrics": {"kl_ref": 0.14, "entropy": 0.84}
                }
            ]

            # Convert to DataFrame and save as Parquet
            df_input = pd.DataFrame(data)
            df_input.to_parquet(f.name, index=False)
            f.flush()

            # Test with field mapping for nested fields
            field_map = {
                "step": "iteration",
                "reward": "score",
                "kl": "metrics.kl_ref",
                "entropy": "metrics.entropy"
            }
            adapter = FlexibleDataAdapter(f.name, field_map=field_map)
            df = adapter.load()

            # Verify canonical output columns
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns

            # Verify data values
            assert len(df) == 3
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1
            assert df["entropy"].iloc[0] == 0.8

        Path(f.name).unlink()

    def test_missing_kl_produces_schema_error(self):
        """Test that missing kl produces SchemaError only when required by downstream metric"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"step": 0, "reward": 0.5, "entropy": 0.8},
                {"step": 1, "reward": 0.6, "entropy": 0.82}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            # Test with kl as required field
            adapter = FlexibleDataAdapter(f.name, required_fields=["step", "reward", "kl"])

            with pytest.raises(SchemaError) as exc_info:
                adapter.load()

            error_message = str(exc_info.value)
            # Should include synonym attempts
            assert "kl" in error_message
            # Should include suggestions
            assert "field_map" in error_message.lower()
            # Should include ready-to-paste field_map suggestion
            assert "{" in error_message and "}" in error_message

        Path(f.name).unlink()

    def test_yaml_mapping_file_works(self):
        """Test that YAML mapping file works as field_map"""
        import yaml

        # Create YAML config file
        config_data = {
            "field_map": {
                "step": "iteration",
                "reward": "score",
                "kl": "kl_divergence",
                "entropy": "policy_entropy"
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as config_f:
            yaml.dump(config_data, config_f)
            config_path = config_f.name

        try:
            # Create test data
            with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as data_f:
                data = [
                    {"iteration": 0, "score": 0.5, "kl_divergence": 0.1, "policy_entropy": 0.8},
                    {"iteration": 1, "score": 0.6, "kl_divergence": 0.12, "policy_entropy": 0.82}
                ]
                for record in data:
                    data_f.write(json.dumps(record) + '\n')
                data_f.flush()

                # Test with YAML config
                adapter = FlexibleDataAdapter(data_f.name, config_file=config_path)
                df = adapter.load()

                # Verify canonical output columns
                assert "step" in df.columns
                assert "reward" in df.columns
                assert "kl" in df.columns
                assert "entropy" in df.columns

                # Verify data values
                assert len(df) == 2
                assert df["step"].iloc[0] == 0
                assert df["reward"].iloc[0] == 0.5
                assert df["kl"].iloc[0] == 0.1
                assert df["entropy"].iloc[0] == 0.8

        finally:
            Path(config_path).unlink()
            Path(data_f.name).unlink()

    def test_canonical_output_dataframe_format(self):
        """Test that canonical output is DataFrame with columns step, reward, kl where available"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = [
                {"step": 0, "reward": 0.5, "kl": 0.1, "entropy": 0.8, "loss": 0.4},
                {"step": 1, "reward": 0.6, "kl": 0.12, "entropy": 0.82, "loss": 0.38}
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            # Verify it's a pandas DataFrame
            assert isinstance(df, pd.DataFrame)

            # Verify canonical columns exist
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns
            assert "loss" in df.columns

            # Verify data types are appropriate
            assert df["step"].dtype in ['int64', 'int32', 'float64']
            assert df["reward"].dtype in ['float64', 'float32']
            assert df["kl"].dtype in ['float64', 'float32']

            # Verify values are correct
            assert df["step"].iloc[0] == 0
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_streaming_behavior_large_jsonl(self):
        """Test streaming behavior for large JSONL files"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            # Create a large dataset (simulate large file)
            for i in range(1000):
                record = {
                    "step": i,
                    "reward": 0.5 + i * 0.001,
                    "kl": 0.1 + i * 0.0001,
                    "entropy": 0.8 - i * 0.0001
                }
                f.write(json.dumps(record) + '\n')
            f.flush()

            # Test streaming JSONL adapter
            adapter = FlexibleJSONLAdapter(f.name, stream_large_files=True)
            df = adapter.load()

            # Verify all records loaded
            assert len(df) == 1000
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns

            # Verify data integrity
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[-1] == 999
            assert df["reward"].iloc[0] == 0.5
            assert df["kl"].iloc[0] == 0.1

        Path(f.name).unlink()

    def test_property_style_round_trip(self):
        """Test property-style round trip: write synthetic logs with random synonym choices and verify normalized output"""
        import random

        # Define synonyms for each canonical field
        synonyms = {
            "step": ["global_step", "step", "iteration", "iter", "timestep"],
            "reward": ["reward_scalar", "reward", "score", "return", "r"],
            "kl": ["kl_to_ref", "kl", "kl_divergence", "kl_ref", "kl_value"],
            "entropy": ["entropy", "entropy_mean", "avg_entropy", "mean_entropy"]
        }

        # Generate synthetic data with random synonym choices
        synthetic_data = []
        for i in range(100):
            record = {}
            for canonical, synonym_list in synonyms.items():
                # Randomly choose a synonym
                chosen_synonym = random.choice(synonym_list)
                # Generate synthetic value
                if canonical == "step":
                    record[chosen_synonym] = i
                elif canonical == "reward":
                    record[chosen_synonym] = 0.5 + i * 0.001 + random.uniform(-0.01, 0.01)
                elif canonical == "kl":
                    record[chosen_synonym] = 0.1 + i * 0.0001 + random.uniform(-0.001, 0.001)
                elif canonical == "entropy":
                    record[chosen_synonym] = 0.8 - i * 0.0001 + random.uniform(-0.001, 0.001)
            synthetic_data.append(record)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            for record in synthetic_data:
                f.write(json.dumps(record) + '\n')
            f.flush()

            # Load with flexible adapter
            adapter = FlexibleDataAdapter(f.name)
            df = adapter.load()

            # Verify canonical columns exist
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns
            assert "entropy" in df.columns

            # Verify all records loaded
            assert len(df) == 100

            # Verify data integrity - step should be sequential
            assert df["step"].tolist() == list(range(100))

            # Verify other fields have reasonable values
            assert df["reward"].min() > 0
            assert df["reward"].max() < 1
            assert df["kl"].min() > 0
            assert df["kl"].max() < 1
            assert df["entropy"].min() > 0
            assert df["entropy"].max() < 1

        Path(f.name).unlink()

    def test_multiple_file_formats_in_directory(self):
        """Test loading multiple file formats from a directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create JSONL file
            jsonl_file = temp_path / "data1.jsonl"
            with open(jsonl_file, 'w') as f:
                data = [{"step": 0, "reward": 0.5, "kl": 0.1}]
                for record in data:
                    f.write(json.dumps(record) + '\n')

            # Create CSV file
            csv_file = temp_path / "data2.csv"
            with open(csv_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "reward", "kl"])
                writer.writerow([1, 0.6, 0.12])

            # Create JSON file
            json_file = temp_path / "data3.json"
            with open(json_file, 'w') as f:
                json.dump([{"step": 2, "reward": 0.7, "kl": 0.14}], f)

            # Load from directory
            adapter = FlexibleDataAdapter(temp_path)
            df = adapter.load()

            # Verify all files loaded
            assert len(df) == 3
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "kl" in df.columns

            # Verify data from all files
            steps = sorted(df["step"].tolist())
            assert steps == [0, 1, 2]

    def test_error_handling_with_helpful_suggestions(self):
        """Test error handling provides helpful suggestions"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            data = {
                "step_count": 0,
                "reward_value": 0.5,
                "kl_divergence": 0.1,
                "entropy_measure": 0.8
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
            assert "kl_divergence" in error_message
            assert "entropy_measure" in error_message

            # Should contain field map suggestion
            assert "field_map" in error_message.lower()
            assert "{" in error_message and "}" in error_message

            # Should contain ready-to-paste suggestion
            assert '"step": "step_count"' in error_message
            assert '"reward": "reward_value"' in error_message
            assert '"kl": "kl_divergence"' in error_message
