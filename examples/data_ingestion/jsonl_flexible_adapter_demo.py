#!/usr/bin/env python3
"""
Flexible JSONL Adapter Demo

This example demonstrates how to use the flexible data adapters to ingest
RL training logs with various schemas without requiring manual field mapping.

The flexible adapters automatically resolve field names using synonyms and
provide helpful error messages when fields are missing.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from rldk.adapters.field_resolver import SchemaError
from rldk.adapters.flexible import FlexibleDataAdapter


def create_sample_data():
    """Create sample data files with different schemas."""
    samples = {}

    # Sample A: TRL-style data with standard field names
    samples['trl_style'] = [
        {"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8, "loss": 0.4, "lr": 0.001},
        {"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.12, "entropy_mean": 0.82, "loss": 0.38, "lr": 0.001},
        {"step": 2, "phase": "train", "reward_mean": 0.7, "kl_mean": 0.14, "entropy_mean": 0.84, "loss": 0.36, "lr": 0.001}
    ]

    # Sample B: Custom JSONL-style data with different field names
    samples['custom_style'] = [
        {"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8, "loss": 0.4, "learning_rate": 0.001},
        {"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.12, "entropy": 0.82, "loss": 0.38, "learning_rate": 0.001},
        {"global_step": 2, "reward_scalar": 0.7, "kl_to_ref": 0.14, "entropy": 0.84, "loss": 0.36, "learning_rate": 0.001}
    ]

    # Sample C: Nested data structure
    samples['nested_style'] = [
        {
            "step": 0,
            "metrics": {"reward": 0.5, "kl": 0.1, "entropy": 0.8},
            "training": {"loss": 0.4, "lr": 0.001}
        },
        {
            "step": 1,
            "metrics": {"reward": 0.6, "kl": 0.12, "entropy": 0.82},
            "training": {"loss": 0.38, "lr": 0.001}
        },
        {
            "step": 2,
            "metrics": {"reward": 0.7, "kl": 0.14, "entropy": 0.84},
            "training": {"loss": 0.36, "lr": 0.001}
        }
    ]

    return samples


def demo_zero_config_ingestion():
    """Demonstrate zero-config ingestion with automatic field resolution."""
    print("=== Zero-Config Ingestion Demo ===")

    samples = create_sample_data()

    for style_name, data in samples.items():
        print(f"\n--- {style_name.replace('_', ' ').title()} Data ---")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
            file_path = f.name

        try:
            # Load with flexible adapter (zero config)
            if style_name == 'nested_style':
                # For nested data, we need to provide field mapping
                field_map = {
                    "reward": "metrics.reward",
                    "kl": "metrics.kl",
                    "entropy": "metrics.entropy",
                    "loss": "training.loss",
                    "lr": "training.lr"
                }
                adapter = FlexibleDataAdapter(file_path, field_map=field_map)
            else:
                adapter = FlexibleDataAdapter(file_path)

            df = adapter.load()

            print(f"✅ Successfully loaded {len(df)} records")
            print(f"📊 Columns: {list(df.columns)}")
            print("📈 Sample data:")
            print(df.head(2).to_string(index=False))

        except SchemaError as e:
            print(f"❌ Schema error: {e}")
            print(f"💡 Suggestion: {e.suggestion}")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            Path(file_path).unlink()


def demo_explicit_field_mapping():
    """Demonstrate explicit field mapping for custom schemas."""
    print("\n=== Explicit Field Mapping Demo ===")

    # Create data with completely custom field names
    custom_data = [
        {"iteration": 0, "score": 0.5, "kl_divergence": 0.1, "policy_entropy": 0.8, "total_loss": 0.4, "learning_rate": 0.001},
        {"iteration": 1, "score": 0.6, "kl_divergence": 0.12, "policy_entropy": 0.82, "total_loss": 0.38, "learning_rate": 0.001},
        {"iteration": 2, "score": 0.7, "kl_divergence": 0.14, "policy_entropy": 0.84, "total_loss": 0.36, "learning_rate": 0.001}
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in custom_data:
            f.write(json.dumps(record) + '\n')
        file_path = f.name

    try:
        # Define explicit field mapping
        field_map = {
            "step": "iteration",
            "reward": "score",
            "kl": "kl_divergence",
            "entropy": "policy_entropy",
            "loss": "total_loss",
            "lr": "learning_rate"
        }

        print(f"🔧 Using field mapping: {field_map}")

        # Load with explicit field mapping
        adapter = FlexibleDataAdapter(file_path, field_map=field_map)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
        print("📈 Sample data:")
        print(df.head(2).to_string(index=False))

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        Path(file_path).unlink()


def demo_yaml_config_mapping():
    """Demonstrate YAML config file for field mapping."""
    print("\n=== YAML Config Mapping Demo ===")

    import yaml

    # Create YAML config file
    config_data = {
        "field_map": {
            "step": "iteration",
            "reward": "score",
            "kl": "kl_divergence",
            "entropy": "policy_entropy",
            "loss": "total_loss",
            "lr": "learning_rate"
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    # Create data file
    custom_data = [
        {"iteration": 0, "score": 0.5, "kl_divergence": 0.1, "policy_entropy": 0.8, "total_loss": 0.4, "learning_rate": 0.001},
        {"iteration": 1, "score": 0.6, "kl_divergence": 0.12, "policy_entropy": 0.82, "total_loss": 0.38, "learning_rate": 0.001}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in custom_data:
            f.write(json.dumps(record) + '\n')
        data_path = f.name

    try:
        print(f"📁 Config file: {config_path}")
        print(f"📁 Data file: {data_path}")

        # Load with YAML config
        adapter = FlexibleDataAdapter(data_path, config_file=config_path)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records using YAML config")
        print(f"📊 Columns: {list(df.columns)}")
        print("📈 Sample data:")
        print(df.head(2).to_string(index=False))

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        Path(config_path).unlink()
        Path(data_path).unlink()


def demo_error_handling():
    """Demonstrate error handling with helpful suggestions."""
    print("\n=== Error Handling Demo ===")

    # Create data with missing required fields
    incomplete_data = [
        {"step_count": 0, "reward_value": 0.5, "kl_divergence": 0.1},
        {"step_count": 1, "reward_value": 0.6, "kl_divergence": 0.12}
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in incomplete_data:
            f.write(json.dumps(record) + '\n')
        file_path = f.name

    try:
        # Try to load without field mapping (should fail with helpful error)
        adapter = FlexibleDataAdapter(file_path)
        df = adapter.load()

    except SchemaError as e:
        print("❌ Schema validation failed (expected)")
        print(f"📝 Error message: {e}")
        print(f"💡 Suggestion: {e.suggestion}")

        # Show how to fix with field mapping
        print("\n🔧 Fix with field mapping:")
        field_map = {
            "step": "step_count",
            "reward": "reward_value",
            "kl": "kl_divergence"
        }

        adapter = FlexibleDataAdapter(file_path, field_map=field_map)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records with field mapping")
        print(f"📊 Columns: {list(df.columns)}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        Path(file_path).unlink()


def demo_large_files():
    """Demonstrate loading large JSONL files."""
    print("\n=== Large Files Demo ===")

    # Create a large dataset
    large_data = []
    for i in range(1000):
        record = {
            "step": i,
            "reward": 0.5 + i * 0.001,
            "kl": 0.1 + i * 0.0001,
            "entropy": 0.8 - i * 0.0001,
            "loss": 0.4 - i * 0.0001
        }
        large_data.append(record)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in large_data:
            f.write(json.dumps(record) + '\n')
        file_path = f.name

    try:
        print(f"📁 Created large file with {len(large_data)} records")

        # Load with flexible adapter (automatically handles large files)
        adapter = FlexibleDataAdapter(file_path)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📈 Data range: step {df['step'].min()} to {df['step'].max()}")
        print(f"📈 Reward range: {df['reward'].min():.3f} to {df['reward'].max():.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        Path(file_path).unlink()


def demo_multiple_formats():
    """Demonstrate loading multiple file formats."""
    print("\n=== Multiple Formats Demo ===")

    # Create different format files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # JSONL file
        jsonl_file = temp_path / "data.jsonl"
        with open(jsonl_file, 'w') as f:
            data = [{"step": 0, "reward": 0.5, "kl": 0.1}]
            for record in data:
                f.write(json.dumps(record) + '\n')

        # JSON file
        json_file = temp_path / "data.json"
        with open(json_file, 'w') as f:
            json.dump([{"step": 1, "reward": 0.6, "kl": 0.12}], f)

        # CSV file
        csv_file = temp_path / "data.csv"
        with open(csv_file, 'w') as f:
            f.write("step,reward,kl\n")
            f.write("2,0.7,0.14\n")

        print(f"📁 Created files in: {temp_dir}")
        print(f"   - {jsonl_file.name}")
        print(f"   - {json_file.name}")
        print(f"   - {csv_file.name}")

        # Load from directory
        adapter = FlexibleDataAdapter(temp_path)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records from multiple formats")
        print(f"📊 Columns: {list(df.columns)}")
        print("📈 Data:")
        print(df.to_string(index=False))


def main():
    """Run all demos."""
    print("🚀 Flexible Data Adapter Demo")
    print("=" * 50)

    try:
        demo_zero_config_ingestion()
        demo_explicit_field_mapping()
        demo_yaml_config_mapping()
        demo_error_handling()
        demo_large_files()
        demo_multiple_formats()

        print("\n✅ All demos completed successfully!")
        print("\n📚 Key Takeaways:")
        print("   • Zero-config ingestion works for common field names")
        print("   • Explicit field mapping handles custom schemas")
        print("   • YAML config files provide reusable mappings")
        print("   • Helpful error messages guide you to solutions")
        print("   • Efficiently handles large files")
        print("   • Multiple formats can be loaded from directories")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
