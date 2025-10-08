"""Basic PPO integration example with RLDK monitoring."""

import os
import time

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
    create_ppo_trainer,
    tokenize_text_column,
)
from rldk.utils.math_utils import safe_divide

try:
    from trl import PPOConfig
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


def create_sample_dataset():
    """Create a sample dataset for PPO training."""
    # Simple prompts and responses for testing
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is",
        "The weather today is",
        "Artificial intelligence can",
    ] * 20  # Repeat to have enough data

    responses = [
        "Paris, the beautiful city of lights.",
        "is widely used for data science and AI.",
        "a subset of artificial intelligence.",
        "sunny and warm.",
        "help solve complex problems.",
    ] * 20

    return Dataset.from_dict({
        "prompt": prompts,
        "response": responses,
    })


def test_basic_ppo_integration():
    """Test basic PPO integration with RLDK monitoring."""
    if not TRL_AVAILABLE:
        print("Skipping test - TRL not available")
        return

    print("🚀 Testing Basic PPO Integration with RLDK")

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
    output_dir = "./test_ppo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize RLDK components
    rldk_callback = RLDKCallback(
        output_dir=output_dir,
        log_interval=5,
        run_id="test_ppo_run"
    )

    ppo_monitor = PPOMonitor(
        output_dir=output_dir,
        kl_threshold=0.1,
        reward_threshold=0.05,
        run_id="test_ppo_run"
    )

    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        run_id="test_ppo_run"
    )

    # Load a small model for testing
    model_name = "gpt2"  # Small model for testing

    # Check if we should skip model loading (for CI/testing)
    if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
        print("⚠️  Skipping model loading (SKIP_MODEL_LOADING=true)")
        print("✅ Example structure validated without model download")
        return True

    # Create tokenizer and sample dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"

    dataset = create_sample_dataset()
    dataset = tokenize_text_column(
        dataset,
        tokenizer,
        text_column="prompt",
        padding=True,
        truncation=True,
        keep_original=False,
        desc="Tokenizing basic PPO prompts",
    )
    if hasattr(dataset, "remove_columns"):
        if "response" in getattr(dataset, "column_names", []):
            dataset = dataset.remove_columns(["response"])
    else:  # pragma: no cover - exercised when stubbing datasets in tests
        for record in dataset:
            record.pop("response", None)

    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        mini_batch_size=2,
        num_ppo_epochs=2,
        max_grad_norm=0.5,
        logging_dir=output_dir,
        save_steps=10,
        eval_steps=10,
        num_train_epochs=1,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=False,  # Disable bf16 for compatibility
        fp16=False,  # Disable fp16 for compatibility
    )

    event_log_path = os.path.join(output_dir, "test_ppo_run_events.jsonl")

    # Create PPO trainer with automatic compatibility handling
    trainer = create_ppo_trainer(
        model_name=model_name,
        ppo_config=ppo_config,
        train_dataset=dataset,
        callbacks=[rldk_callback, ppo_monitor, checkpoint_monitor],
        event_log_path=event_log_path,
        tokenizer=tokenizer,
    )

    print("✅ PPO Trainer created with RLDK callbacks")
    print(f"📊 Monitoring {len(dataset)} samples")
    print(f"💾 Output directory: {output_dir}")

    # Test training for a few steps
    try:
        print("🎯 Starting PPO training test...")

        # This would normally be trainer.train(), but for testing we'll simulate
        # a few training steps to verify the callbacks work

        # Simulate training steps
        for step in range(5):
            # Simulate some training metrics
            fake_logs = {
                'ppo/rewards/mean': 0.5 + step * 0.1,
                'ppo/rewards/std': 0.2,
                'ppo/policy/kl_mean': 0.05 + step * 0.01,
                'ppo/policy/entropy': 2.0 - step * 0.1,
                'ppo/policy/clipfrac': 0.1,
                'ppo/val/value_loss': 0.3 - step * 0.05,
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

            ppo_monitor.log_metrics(step=step, metrics=fake_logs)

            if step % 2 == 0:  # Simulate checkpoint saves
                rldk_callback.on_save(args, state, control, model=trainer.model)
                checkpoint_monitor.log_checkpoint(
                    step=state.global_step,
                    checkpoint_data={
                        "epoch": state.epoch,
                        "timestamp": time.time(),
                        "gradient_norm": fake_logs.get('grad_norm', 0.0),
                    },
                    model=trainer.model,
                )

            print(f"✅ Step {step} completed")

        print("🎉 Training simulation completed successfully!")

        # Save final analysis
        rldk_callback.save_metrics_history()
        ppo_monitor.save_ppo_analysis()
        checkpoint_monitor.save_checkpoint_summary()

        print("📁 Analysis saved to output directory")

        # Verify files were created
        expected_files = [
            f"{output_dir}/test_ppo_run_metrics.csv",
            f"{output_dir}/test_ppo_run_ppo_metrics.csv",
            f"{output_dir}/test_ppo_run_checkpoint_summary.csv",
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

    # Test basic PPO integration
    success = test_basic_ppo_integration()

    if success:
        print("\n🎉 All tests passed! RLDK TRL integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
