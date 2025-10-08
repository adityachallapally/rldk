#!/usr/bin/env python3
"""
Test experiment tracking with real models - Fixed version without WandB
"""

import sys
import tempfile
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

import _path_setup  # noqa: F401


def test_experiment_tracking():
    """Test experiment tracking with real models"""
    print("Testing Experiment Tracking with Real Models (No WandB)")
    print("=" * 60)
    
    try:
        from rldk.tracking import ExperimentTracker, TrackingConfig
        
        # Create a temporary directory for the experiment
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Configure tracking WITHOUT WandB
            config = TrackingConfig(
                experiment_name="gpt2_test_experiment",
                output_dir=temp_path,
                enable_dataset_tracking=True,
                enable_model_tracking=True,
                enable_environment_tracking=True,
                enable_seed_tracking=True,
                enable_git_tracking=True,
                save_to_json=True,
                save_to_yaml=True,
                save_to_wandb=False  # Disable WandB
            )
            
            print(f"✓ TrackingConfig created successfully")
            
            # Initialize tracker
            tracker = ExperimentTracker(config)
            print(f"✓ ExperimentTracker initialized successfully")
            
            # Start experiment
            tracker.start_experiment()
            print(f"✓ Experiment started successfully")
            
            # Load real model and tokenizer
            print("Loading GPT-2 model and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            model = AutoModelForCausalLM.from_pretrained('gpt2')
            
            # Track model and tokenizer
            tracker.track_model(model, "gpt2_policy")
            tracker.track_tokenizer(tokenizer, "gpt2_tokenizer")
            print(f"✓ Model and tokenizer tracked successfully")
            
            # Set seeds
            tracker.set_seeds(42)
            print(f"✓ Seeds set successfully")
            
            # Add metadata
            tracker.add_metadata("learning_rate", 1e-5)
            tracker.add_metadata("batch_size", 32)
            tracker.add_metadata("model_size", "117M")
            print(f"✓ Metadata added successfully")
            
            # Finish experiment
            tracker.finish_experiment()
            print(f"✓ Experiment finished successfully")
            
            # Check if files were created
            json_file = temp_path / "experiment.json"
            yaml_file = temp_path / "experiment.yaml"
            
            if json_file.exists():
                print(f"✓ JSON file created: {json_file}")
                with open(json_file) as f:
                    data = json.load(f)
                    print(f"  - Experiment ID: {data.get('experiment_id', 'N/A')}")
                    print(f"  - Model count: {len(data.get('models', {}))}")
                    print(f"  - Metadata count: {len(data.get('metadata', {}))}")
                    
                    # Show some model details
                    models = data.get('models', {})
                    if models:
                        model_name = list(models.keys())[0]
                        model_info = models[model_name]
                        print(f"  - Model '{model_name}' architecture: {model_info.get('architecture', 'N/A')}")
                        print(f"  - Model parameters: {model_info.get('num_parameters', 'N/A')}")
            else:
                print(f"✗ JSON file not created")
                
            if yaml_file.exists():
                print(f"✓ YAML file created: {yaml_file}")
            else:
                print(f"✗ YAML file not created")
                
        return True
        
    except Exception as e:
        print(f"✗ Experiment tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_experiment_tracking()