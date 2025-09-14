#!/usr/bin/env python3
"""
RLHF Training Script with Hugging Face Models and Integrated Profiler.

This script demonstrates how to integrate the profiler system with a training loop
using real Hugging Face transformer models. It supports the --profiler on/off argument
to enable/disable profiling.

Example usage:
    python examples/train_hf_model.py --model distilbert-base-uncased --profiler on
    python examples/train_hf_model.py --model bert-base-uncased --profiler off
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Check dependencies before importing
try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError as e:
    print("❌ Error: Missing transformers dependency")
    print(f"   {e}")
    print("\n💡 To fix this, install the missing dependencies:")
    print("   pip install transformers")
    print("   # or with --break-system-packages if needed:")
    print("   pip install transformers --break-system-packages")
    sys.exit(1)

from rlhf_core.profiler import ProfilerManager
from tools.profiler.hooks import StepProfiler, profiler_registry
from tools.profiler.profiler_context import ProfilerContext
from tools.profiler.torch_profiler import TorchProfiler


class HuggingFaceModelWrapper(nn.Module):
    """Wrapper for Hugging Face models with classification head."""

    def __init__(self, model_name: str, num_labels: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use the last hidden state and pool it (mean pooling)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler_output (like DistilBERT), use mean pooling
            pooled_output = outputs.last_hidden_state.mean(dim=1)

        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class TrainingConfig:
    """Configuration for training parameters."""

    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.num_labels = 2
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.max_length = 128
        self.warmup_steps = 100
        self.dropout_rate = 0.1
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.profiler_enabled = False
        self.output_dir = "hf_training_output"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_length": self.max_length,
            "warmup_steps": self.warmup_steps,
            "dropout_rate": self.dropout_rate,
            "seed": self.seed,
            "device": self.device,
            "profiler_enabled": self.profiler_enabled,
            "output_dir": self.output_dir,
        }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dummy_dataset(tokenizer, num_samples: int = 1000, max_length: int = 128):
    """Create a dummy dataset for demonstration purposes."""
    texts = [
        "This is a positive example of text classification.",
        "This is a negative example that should be classified differently.",
        "Machine learning models can be very effective for NLP tasks.",
        "Natural language processing is a fascinating field of study.",
        "Transformers have revolutionized the field of deep learning.",
    ] * (num_samples // 5 + 1)

    labels = [0, 1, 0, 1, 0] * (num_samples // 5 + 1)

    # Tokenize texts
    encodings = tokenizer(
        texts[:num_samples],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels[:num_samples], dtype=torch.long)
    )


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, profiler_context=None, torch_profiler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Core training step - single implementation
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Profiler step - record after each batch
        if torch_profiler:
            torch_profiler.step()

        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {correct_predictions/total_predictions:.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Hugging Face model with profiler")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                       help="Hugging Face model name")
    parser.add_argument("--profiler", type=str, choices=["on", "off"], default="off",
                       help="Enable or disable profiler")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="hf_training_output",
                       help="Output directory for artifacts")

    args = parser.parse_args()

    # Initialize configuration
    config = TrainingConfig()
    config.model_name = args.model
    config.profiler_enabled = args.profiler == "on"
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir

    # Set seed for reproducibility
    set_seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    print("Training configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Profiler: {'Enabled' if config.profiler_enabled else 'Disabled'}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    # Initialize tokenizer and model
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = HuggingFaceModelWrapper(
            config.model_name,
            config.num_labels,
            config.dropout_rate
        ).to(config.device)
        print(f"✓ Successfully loaded {config.model_name}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you have internet connection and the model name is correct.")
        return 1

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dummy_dataset(tokenizer, num_samples=800, max_length=config.max_length)
    val_dataset = create_dummy_dataset(tokenizer, num_samples=200, max_length=config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # Initialize profiler if enabled
    profiler_context = None
    torch_profiler = None
    if config.profiler_enabled:
        print("Initializing profiler...")
        try:
            # Initialize stage-level profiler
            profiler_context = ProfilerContext(
                output_dir=config.output_dir,
                model_name=config.model_name,
                config=config.to_dict()
            )

            # Initialize step-level profiler
            torch_profiler = TorchProfiler(
                output_dir=config.output_dir,
                warmup_steps=5,
                active_steps=10,
                repeat=1
            )
            torch_profiler.start()

            print("✓ Profiler initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing profiler: {e}")
            print("Continuing without profiler...")
            config.profiler_enabled = False

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    training_history = []

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            config.device, profiler_context, torch_profiler
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)

        # Record metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_metrics)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

    # Save results
    print(f"\nSaving results to {config.output_dir}...")

    # Save training history
    with open(os.path.join(config.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    # Save model
    model_path = os.path.join(config.output_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'training_history': training_history
    }, model_path)

    # Save configuration
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("✓ Training completed successfully!")
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Training history saved to: {os.path.join(config.output_dir, 'training_history.json')}")

    if config.profiler_enabled and torch_profiler:
        torch_profiler.stop()
        print(f"✓ Profiler artifacts saved to: {config.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
