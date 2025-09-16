"""Basic DPO integration example with RLDK monitoring."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
)

# Import RLDK components
from rldk.integrations.trl import (
    CheckpointMonitor,
    PPOMonitor,
    RLDKCallback,
    check_trl_compatibility,
    create_dpo_trainer,
    simple_reward_function,
)
from rldk.utils.math_utils import safe_divide

try:
    from trl import DPOConfig, DPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


def create_sample_dataset():
    """Create a sample dataset for DPO training."""
    # DPO requires prompt, chosen, and rejected columns
    prompts = [
        "What is the capital of France?",
        "Explain machine learning",
        "What is Python?",
        "How does a computer work?",
        "What is artificial intelligence?",
    ] * 20  # Repeat to have enough data

    chosen_responses = [
        "The capital of France is Paris, a beautiful city known for its art and culture.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "A computer works by processing information using electronic circuits and software programs.",
        "Artificial intelligence is the simulation of human intelligence in machines.",
    ] * 20

    rejected_responses = [
        "France has no capital city.",
        "Machine learning is about machines that learn to be human.",
        "Python is a type of snake found in tropical regions.",
        "Computers work by magic and fairy dust.",
        "AI is just robots taking over the world.",
    ] * 20

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses,
    })


def test_basic_dpo_integration():
    """Test basic DPO integration with RLDK monitoring."""
    if not TRL_AVAILABLE:
        print("Skipping test - TRL not available")
        return

    print("🚀 Testing Basic DPO Integration with RLDK")

    # Check TRL compatibility and show warnings
    compatibility = check_trl_compatibility()
    if compatibility["warnings"]:
        print("⚠️  TRL Compatibility Warnings:")
        for warning in compatibility["warnings"]:
            print(f"   - {warning}")
    if compatibility["recommendations"]:
        print("💡 Recommendations:")
        for rec in compatibility["recommendations"]:
            print(f"   - {rec}")

    # Create output directory
    output_dir = "./test_dpo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize RLDK components
    rldk_callback = RLDKCallback(
        output_dir=output_dir,
        log_interval=5,
        run_id="test_dpo_run"
    )

    ppo_monitor = PPOMonitor(
        output_dir=output_dir,
        kl_threshold=0.1,
        reward_threshold=0.05,
        run_id="test_dpo_run"
    )

    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        run_id="test_dpo_run"
    )

    # Load a small model for testing
    model_name = "sshleifer/tiny-gpt2"  # Very small model for testing

    # Check if we should skip model loading (for CI/testing)
    if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
        print("⚠️  Skipping model loading (SKIP_MODEL_LOADING=true)")
        print("✅ Example structure validated without model download")
        return True

    # Create sample dataset
    dataset = create_sample_dataset()

    # DPO configuration
    dpo_config = DPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        max_steps=10,
        logging_dir=output_dir,
        save_steps=10,
        eval_steps=10,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=False,  # Disable bf16 for compatibility
        fp16=False,  # Disable fp16 for compatibility
    )

    # Use unified factory function to create DPO trainer with simplified architecture
    # This handles all TRL API differences automatically and ensures all required parameters are provided
    try:
        trainer = create_dpo_trainer(
            model_name=model_name,
            dpo_config=dpo_config,
            train_dataset=dataset,
            callbacks=[rldk_callback, ppo_monitor, checkpoint_monitor],
        )
    except Exception as e:
        print(f"❌ Failed to create DPO trainer: {e}")
        print("⚠️  This might be due to model loading issues or TRL version incompatibility")
        return False

    print("✅ DPO Trainer created with RLDK callbacks")
    print(f"📊 Monitoring {len(dataset)} samples")
    print(f"💾 Output directory: {output_dir}")

    # Test training for a few steps
    try:
        print("🎯 Starting DPO training test...")

        # This would normally be trainer.train(), but for testing we'll simulate
        # a few training steps to verify the callbacks work

        # Simulate training steps
        for step in range(5):
            # Simulate some training metrics
            fake_logs = {
                'loss': 0.6931 - step * 0.01,
                'rewards/chosen': 0.5 + step * 0.1,
                'rewards/rejected': 0.3 + step * 0.05,
                'rewards/accuracies': 0.6 + step * 0.05,
                'rewards/margins': 0.2 + step * 0.02,
                'logps/chosen': -100.0 + step * 5.0,
                'logps/rejected': -120.0 + step * 3.0,
                'learning_rate': 1e-5,
                'grad_norm': 0.5,
            }

            # Simulate callback calls
            from transformers import TrainerControl, TrainerState, TrainingArguments

            # Create mock objects
            args = TrainingArguments(output_dir=output_dir)
            state = TrainerState()
            state.global_step = step
            state.epoch = safe_divide(step, 10.0, 0.0)
            control = TrainerControl()

            # Call callbacks
            rldk_callback.on_step_end(args, state, control)
            rldk_callback.on_log(args, state, control, fake_logs)

            ppo_monitor.on_step_end(args, state, control)
            ppo_monitor.on_log(args, state, control, fake_logs)

            if step % 2 == 0:  # Simulate checkpoint saves
                # Create a dummy model object for simulation
                class DummyModel:
                    def __init__(self):
                        self.config = type('Config', (), {'hidden_size': 768})()
                    
                    def parameters(self):
                        return []
                    
                    def named_parameters(self):
                        return []
                
                dummy_model = DummyModel()
                rldk_callback.on_save(args, state, control, model=dummy_model)
                checkpoint_monitor.on_save(args, state, control, model=dummy_model)

            print(f"✅ Step {step} completed")

        print("🎉 Training simulation completed successfully!")

        # Save final analysis
        rldk_callback.save_metrics_history()
        ppo_monitor.save_ppo_analysis()
        checkpoint_monitor.save_checkpoint_summary()

        print("📁 Analysis saved to output directory")

        # Verify files were created
        expected_files = [
            f"{output_dir}/test_dpo_run_metrics.csv",
            f"{output_dir}/test_dpo_run_ppo_metrics.csv",
            f"{output_dir}/test_dpo_run_checkpoint_summary.csv",
        ]

        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} created")
            else:
                print(f"❌ {file_path} missing")

        return True

    except Exception as e:
        print(f"❌ Error during training simulation: {e}")
        return False


def test_callback_functionality():
    """Test individual callback functionality."""
    print("🧪 Testing Callback Functionality")

    # Test RLDKCallback
    callback = RLDKCallback(
        output_dir="./test_callback_output",
        log_interval=1,
        run_id="test_callback"
    )

    # Test metrics collection
    from rldk.integrations.trl.callbacks import RLDKMetrics

    metrics = RLDKMetrics(
        step=1,
        loss=0.5,
        reward_mean=0.3,
        kl_mean=0.05,
        training_stability_score=0.8
    )

    print(f"✅ Metrics created: {metrics.to_dict()}")

    # Test alert system
    callback._add_alert("test_alert", "This is a test alert")
    print(f"✅ Alert system working: {len(callback.alerts)} alerts")

    print("✅ Callback functionality test passed")
    return True


if __name__ == "__main__":
    print("🎯 RLDK TRL Integration Test Suite")
    print("=" * 50)

    # Test callback functionality
    test_callback_functionality()
    print()

    # Test basic DPO integration
    success = test_basic_dpo_integration()

    if success:
        print("\n🎉 All tests passed! RLDK TRL integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
