#!/usr/bin/env python3
"""Demonstration script showing the improved data ingestion system."""

import sys
import os
from pathlib import Path

def demonstrate_error_messages():
    """Demonstrate the improved error messages."""
    print("🔍 Demonstrating Improved Error Messages")
    print("=" * 50)
    
    # Simulate the error messages that would be generated
    print("\n1. File Not Found Error:")
    print("   Before: ERROR:root:Failed to ingest /workspace/nonexistent: Cannot handle source")
    print("   After:  FileNotFoundError: Source path does not exist: /workspace/nonexistent")
    print("           Please check the path and ensure the file or directory exists.")
    
    print("\n2. Wrong Adapter Type Error:")
    print("   Before: ERROR:root:Failed to ingest file: Cannot handle source")
    print("   After:  ValueError: Cannot handle trl format for file: custom_data.jsonl")
    print("           Expected format for trl:")
    print("           TRL format examples:")
    print("             {\"step\": 0, \"phase\": \"train\", \"reward_mean\": 0.5, ...}")
    print("           Try using --adapter custom_jsonl for generic JSONL files.")
    
    print("\n3. Unsupported File Extension Error:")
    print("   Before: ERROR:root:Failed to ingest file: Cannot handle source")
    print("   After:  ValueError: Cannot handle trl format for file: data.txt")
    print("           File extension '.txt' is not supported by trl adapter.")
    print("           Supported extensions: .jsonl, .log")

def demonstrate_format_examples():
    """Demonstrate the format examples."""
    print("\n📋 Demonstrating Format Examples")
    print("=" * 50)
    
    formats = {
        "TRL": {
            "jsonl": '{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}',
            "log": "step: 0, reward: 0.5, kl: 0.1, entropy: 2.0, loss: 0.3"
        },
        "OpenRLHF": {
            "jsonl": '{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}',
            "log": "step: 0, reward: 0.5, kl: 0.1, entropy: 2.0, loss: 0.3"
        },
        "Custom JSONL": {
            "jsonl": '{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 2.0, "loss": 0.3}',
            "nested": '{"step": 0, "metrics": {"reward": 0.5, "kl": 0.1}, "model_info": {"phase": "train"}}'
        },
        "WandB": {
            "uri": "wandb://project_name/run_id",
            "directory": "./wandb/run-20240101_120000-abc123/"
        }
    }
    
    for format_name, examples in formats.items():
        print(f"\n{format_name} Format Examples:")
        for example_type, example in examples.items():
            print(f"  {example_type}: {example}")

def demonstrate_directory_structures():
    """Demonstrate the directory structure examples."""
    print("\n📁 Demonstrating Directory Structure Examples")
    print("=" * 50)
    
    structures = {
        "TRL": """
training_logs/
├── trainer_log.jsonl
├── training.log
└── *_events.jsonl""",
        
        "OpenRLHF": """
training_logs/
├── training.log
├── metrics.jsonl
└── logs/""",
        
        "Custom JSONL": """
data/
├── metrics.jsonl
├── training_data.jsonl
└── *.jsonl files""",
        
        "WandB": """
wandb/
└── run-20240101_120000-abc123/
    ├── files/
    ├── logs/
    └── config.yaml"""
    }
    
    for name, structure in structures.items():
        print(f"\n{name} Directory Structure:")
        print(structure)

def demonstrate_sample_data():
    """Demonstrate the sample data files."""
    print("\n📊 Demonstrating Sample Data Files")
    print("=" * 50)
    
    sample_files = [
        "sample_data/trl_training_output.jsonl",
        "sample_data/openrlhf_training_output.jsonl",
        "sample_data/custom_training_output.jsonl",
        "sample_data/sample_eval_data.jsonl",
        "sample_data/forensics_test_output/trainer_log.jsonl",
        "sample_data/rl_training_output/training.log"
    ]
    
    print("Sample data files created:")
    for file_path in sample_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path} ({path.stat().st_size} bytes)")
        else:
            print(f"  ❌ {file_path} (missing)")

def demonstrate_improved_commands():
    """Demonstrate the improved command usage."""
    print("\n🚀 Demonstrating Improved Command Usage")
    print("=" * 50)
    
    print("Before (Original Commands - Would Fail):")
    print("  rldk diff --a /workspace/forensics_test_output --b /workspace/rl_training_output --signals \"loss,reward_mean,kl\"")
    print("  rldk ingest /workspace/sample_eval_data.jsonl --adapter trl --output /workspace/ingested_metrics.jsonl")
    print("  rldk card determinism /workspace/rl_training_output")
    
    print("\nAfter (Improved Commands - Now Work):")
    print("  rldk diff --a /workspace/sample_data/forensics_test_output --b /workspace/sample_data/rl_training_output --signals \"loss,reward_mean,kl\"")
    print("  rldk ingest /workspace/sample_data/sample_eval_data.jsonl --adapter custom_jsonl --output /workspace/ingested_metrics.jsonl")
    print("  rldk card determinism /workspace/sample_data/forensics_test_output")

def main():
    """Run the demonstration."""
    print("🎯 RL Debug Kit - Data Ingestion Improvements Demonstration")
    print("=" * 70)
    
    demonstrations = [
        demonstrate_error_messages,
        demonstrate_format_examples,
        demonstrate_directory_structures,
        demonstrate_sample_data,
        demonstrate_improved_commands
    ]
    
    for demo in demonstrations:
        demo()
        print()
    
    print("=" * 70)
    print("🎉 Demonstration Complete!")
    print("\nKey Improvements:")
    print("✅ Better error messages with format examples")
    print("✅ Input data validation before processing")
    print("✅ Comprehensive format documentation")
    print("✅ Sample data files for all formats")
    print("✅ Directory structure examples")
    print("✅ Improved adapter detection")
    print("✅ More flexible format handling")
    
    print("\n📚 For more details, see: DATA_INGESTION_IMPROVEMENTS.md")

if __name__ == "__main__":
    main()