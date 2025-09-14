#!/usr/bin/env python3
"""
CSV and Parquet Adapter Demo

This example demonstrates how to use the flexible data adapters to ingest
RL training logs from CSV and Parquet files with automatic field resolution.

The flexible adapters work with multiple formats and automatically map
column names to canonical field names using synonyms.
"""

import csv
import json
import tempfile
from pathlib import Path

import pandas as pd

from rldk.adapters.field_resolver import SchemaError
from rldk.adapters.flexible import FlexibleDataAdapter


def create_csv_sample_data():
    """Create sample CSV data with different column naming conventions."""
    samples = {}

    # Sample A: Standard field names
    samples['standard'] = [
        {"step": 0, "reward": 0.5, "kl": 0.1, "entropy": 0.8, "loss": 0.4, "lr": 0.001},
        {"step": 1, "reward": 0.6, "kl": 0.12, "entropy": 0.82, "loss": 0.38, "lr": 0.001},
        {"step": 2, "reward": 0.7, "kl": 0.14, "entropy": 0.84, "loss": 0.36, "lr": 0.001}
    ]

    # Sample B: TRL-style field names
    samples['trl_style'] = [
        {"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8, "loss": 0.4, "lr": 0.001},
        {"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.12, "entropy_mean": 0.82, "loss": 0.38, "lr": 0.001},
        {"step": 2, "phase": "train", "reward_mean": 0.7, "kl_mean": 0.14, "entropy_mean": 0.84, "loss": 0.36, "lr": 0.001}
    ]

    # Sample C: Custom field names
    samples['custom'] = [
        {"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8, "total_loss": 0.4, "learning_rate": 0.001},
        {"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.12, "entropy": 0.82, "total_loss": 0.38, "learning_rate": 0.001},
        {"global_step": 2, "reward_scalar": 0.7, "kl_to_ref": 0.14, "entropy": 0.84, "total_loss": 0.36, "learning_rate": 0.001}
    ]

    return samples


def create_parquet_sample_data():
    """Create sample Parquet data with different structures."""
    samples = {}

    # Sample A: Flat structure with standard names
    samples['flat_standard'] = pd.DataFrame({
        "step": [0, 1, 2],
        "reward": [0.5, 0.6, 0.7],
        "kl": [0.1, 0.12, 0.14],
        "entropy": [0.8, 0.82, 0.84],
        "loss": [0.4, 0.38, 0.36],
        "lr": [0.001, 0.001, 0.001]
    })

    # Sample B: Flat structure with custom names
    samples['flat_custom'] = pd.DataFrame({
        "iteration": [0, 1, 2],
        "score": [0.5, 0.6, 0.7],
        "kl_divergence": [0.1, 0.12, 0.14],
        "policy_entropy": [0.8, 0.82, 0.84],
        "total_loss": [0.4, 0.38, 0.36],
        "learning_rate": [0.001, 0.001, 0.001]
    })

    # Sample C: Nested structure (stored as JSON strings in Parquet)
    nested_data = []
    for i in range(3):
        record = {
            "step": i,
            "metrics": {"reward": 0.5 + i * 0.1, "kl": 0.1 + i * 0.02, "entropy": 0.8 + i * 0.02},
            "training": {"loss": 0.4 - i * 0.02, "lr": 0.001}
        }
        nested_data.append(record)

    samples['nested'] = pd.DataFrame({
        "step": [0, 1, 2],
        "metrics": [json.dumps({"reward": 0.5 + i * 0.1, "kl": 0.1 + i * 0.02, "entropy": 0.8 + i * 0.02}) for i in range(3)],
        "training": [json.dumps({"loss": 0.4 - i * 0.02, "lr": 0.001}) for i in range(3)]
    })

    return samples


def demo_csv_ingestion():
    """Demonstrate CSV file ingestion with different schemas."""
    print("=== CSV Ingestion Demo ===")

    samples = create_csv_sample_data()

    for style_name, data in samples.items():
        print(f"\n--- {style_name.replace('_', ' ').title()} CSV Data ---")

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            if data:  # Check if data is not empty
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            file_path = f.name

        try:
            # Load with flexible adapter
            if style_name == 'custom':
                # For custom names, we need field mapping
                field_map = {
                    "step": "global_step",
                    "reward": "reward_scalar",
                    "kl": "kl_to_ref",
                    "loss": "total_loss",
                    "lr": "learning_rate"
                }
                adapter = FlexibleDataAdapter(file_path, field_map=field_map)
            else:
                adapter = FlexibleDataAdapter(file_path)

            df = adapter.load()

            print(f"✅ Successfully loaded {len(df)} records from CSV")
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


def demo_parquet_ingestion():
    """Demonstrate Parquet file ingestion with different structures."""
    print("\n=== Parquet Ingestion Demo ===")

    samples = create_parquet_sample_data()

    for style_name, df_data in samples.items():
        print(f"\n--- {style_name.replace('_', ' ').title()} Parquet Data ---")

        # Create temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df_data.to_parquet(f.name, index=False)
            file_path = f.name

        try:
            # Load with flexible adapter
            if style_name == 'flat_custom':
                # For custom names, we need field mapping
                field_map = {
                    "step": "iteration",
                    "reward": "score",
                    "kl": "kl_divergence",
                    "entropy": "policy_entropy",
                    "loss": "total_loss",
                    "lr": "learning_rate"
                }
                adapter = FlexibleDataAdapter(file_path, field_map=field_map)
            elif style_name == 'nested':
                # For nested data, we need field mapping for nested paths
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

            print(f"✅ Successfully loaded {len(df)} records from Parquet")
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


def demo_mixed_format_directory():
    """Demonstrate loading from a directory with mixed file formats."""
    print("\n=== Mixed Format Directory Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create JSONL file
        jsonl_file = temp_path / "data1.jsonl"
        with open(jsonl_file, 'w') as f:
            data = [{"step": 0, "reward": 0.5, "kl": 0.1, "entropy": 0.8}]
            for record in data:
                f.write(json.dumps(record) + '\n')

        # Create CSV file
        csv_file = temp_path / "data2.csv"
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "kl", "entropy"])
            writer.writerow([1, 0.6, 0.12, 0.82])

        # Create JSON file
        json_file = temp_path / "data3.json"
        with open(json_file, 'w') as f:
            json.dump([{"step": 2, "reward": 0.7, "kl": 0.14, "entropy": 0.84}], f)

        # Create Parquet file
        parquet_file = temp_path / "data4.parquet"
        df_parquet = pd.DataFrame({
            "step": [3, 4],
            "reward": [0.8, 0.9],
            "kl": [0.16, 0.18],
            "entropy": [0.86, 0.88]
        })
        df_parquet.to_parquet(parquet_file, index=False)

        print(f"📁 Created files in: {temp_dir}")
        print(f"   - {jsonl_file.name}")
        print(f"   - {csv_file.name}")
        print(f"   - {json_file.name}")
        print(f"   - {parquet_file.name}")

        # Load from directory
        adapter = FlexibleDataAdapter(temp_path)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records from mixed formats")
        print(f"📊 Columns: {list(df.columns)}")
        print("📈 Data:")
        print(df.to_string(index=False))


def demo_large_parquet_file():
    """Demonstrate loading large Parquet files efficiently."""
    print("\n=== Large Parquet File Demo ===")

    # Create a large dataset
    large_data = []
    for i in range(1000):
        record = {
            "step": i,
            "reward": 0.5 + i * 0.001,
            "kl": 0.1 + i * 0.0001,
            "entropy": 0.8 - i * 0.0001,
            "loss": 0.4 - i * 0.0001,
            "lr": 0.001
        }
        large_data.append(record)

    # Create DataFrame and save as Parquet
    df_large = pd.DataFrame(large_data)

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df_large.to_parquet(f.name, index=False)
        file_path = f.name

    try:
        print(f"📁 Created large Parquet file with {len(large_data)} records")

        # Load with flexible adapter
        adapter = FlexibleDataAdapter(file_path)
        df = adapter.load()

        print(f"✅ Successfully loaded {len(df)} records from large Parquet file")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📈 Data range: step {df['step'].min()} to {df['step'].max()}")
        print(f"📈 Reward range: {df['reward'].min():.3f} to {df['reward'].max():.3f}")
        print(f"📈 KL range: {df['kl'].min():.3f} to {df['kl'].max():.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        Path(file_path).unlink()


def demo_error_handling_csv():
    """Demonstrate error handling with CSV files."""
    print("\n=== CSV Error Handling Demo ===")

    # Create CSV with missing required fields
    incomplete_data = [
        {"step_count": 0, "reward_value": 0.5, "kl_divergence": 0.1},
        {"step_count": 1, "reward_value": 0.6, "kl_divergence": 0.12}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=incomplete_data[0].keys())
        writer.writeheader()
        writer.writerows(incomplete_data)
        file_path = f.name

    try:
        # Try to load without field mapping (should fail with helpful error)
        adapter = FlexibleDataAdapter(file_path)
        df = adapter.load()

        # If we get here, the load succeeded unexpectedly
        print("⚠️  Load succeeded unexpectedly - this should have failed")
        print(f"   Loaded {len(df)} records with columns: {list(df.columns)}")

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


def demo_performance_comparison():
    """Demonstrate performance comparison between formats."""
    print("\n=== Performance Comparison Demo ===")

    import time

    # Create test data
    test_data = []
    for i in range(1000):
        record = {
            "step": i,
            "reward": 0.5 + i * 0.001,
            "kl": 0.1 + i * 0.0001,
            "entropy": 0.8 - i * 0.0001,
            "loss": 0.4 - i * 0.0001
        }
        test_data.append(record)

    formats = {}

    # Create JSONL file
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        for record in test_data:
            f.write(json.dumps(record) + '\n')
        formats['jsonl'] = f.name

    # Create CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=test_data[0].keys())
        writer.writeheader()
        writer.writerows(test_data)
        formats['csv'] = f.name

    # Create Parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df = pd.DataFrame(test_data)
        df.to_parquet(f.name, index=False)
        formats['parquet'] = f.name

    try:
        print(f"📊 Testing with {len(test_data)} records")

        for format_name, file_path in formats.items():
            start_time = time.time()

            adapter = FlexibleDataAdapter(file_path)
            df = adapter.load()

            end_time = time.time()
            load_time = end_time - start_time

            print(f"   {format_name.upper():>8}: {load_time:.3f}s ({len(df)} records)")

        print("\n💡 Performance Tips:")
        print("   • Parquet is fastest for large datasets")
        print("   • JSONL is good for streaming large files")
        print("   • CSV is human-readable but slower")
        print("   • Use Parquet for production workloads")

    finally:
        for file_path in formats.values():
            Path(file_path).unlink()


def main():
    """Run all demos."""
    print("🚀 CSV and Parquet Adapter Demo")
    print("=" * 50)

    try:
        demo_csv_ingestion()
        demo_parquet_ingestion()
        demo_mixed_format_directory()
        demo_large_parquet_file()
        demo_error_handling_csv()
        demo_performance_comparison()

        print("\n✅ All demos completed successfully!")
        print("\n📚 Key Takeaways:")
        print("   • CSV files work with automatic field resolution")
        print("   • Parquet files are efficient for large datasets")
        print("   • Mixed format directories are supported")
        print("   • Field mapping handles custom column names")
        print("   • Error messages provide helpful suggestions")
        print("   • Parquet is fastest for large data")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
