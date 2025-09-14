#!/usr/bin/env python3
"""
Demonstration of the comprehensive tracking system for enhanced data lineage & reproducibility.

This script shows how to use the tracking system with a real ML pipeline,
including dataset tracking, model fingerprinting, environment capture,
seed tracking, and Git integration.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rldk.tracking import ExperimentTracker, TrackingConfig


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier for demonstration."""

    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_synthetic_dataset(n_samples=10000, n_features=20, n_classes=3, random_state=42):
    """Create a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=2,
        n_informative=n_features-2,
        random_state=random_state
    )

    return X, y


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the model and return training history."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # Record metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(100 * train_correct / train_total)
        val_accuracies.append(100 * val_correct / val_total)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


def main():
    """Main demonstration function."""
    print("="*60)
    print("COMPREHENSIVE TRACKING SYSTEM DEMONSTRATION")
    print("="*60)

    # Configuration
    experiment_name = "ml_classification_demo"
    output_dir = Path("./tracking_demo_output")
    random_seed = 42

    # Create tracking configuration
    config = TrackingConfig(
        experiment_name=experiment_name,
        output_dir=output_dir,
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_model_architecture=True,
        save_model_weights=False,  # Don't save weights for demo
        save_to_json=True,
        save_to_yaml=True,
        save_to_wandb=False,  # Set to True if you have wandb configured
        tags=["classification", "pytorch", "demo"],
        notes="Demonstration of comprehensive tracking system for ML pipeline"
    )

    # Initialize experiment tracker
    tracker = ExperimentTracker(config)

    print(f"\n1. Starting experiment: {experiment_name}")
    print(f"   Output directory: {output_dir}")
    print(f"   Experiment ID: {tracker.experiment_id}")

    # Start experiment (captures environment, git, and initial seeds)
    tracker.start_experiment()

    print("\n2. Creating reproducible environment...")
    # Set seeds for reproducibility
    tracker.create_reproducible_environment(random_seed)

    print("\n3. Creating and tracking datasets...")
    # Create synthetic dataset
    X, y = create_synthetic_dataset(random_state=random_seed)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_seed, stratify=y_train
    )

    # Track datasets
    tracker.track_dataset(
        X_train, "X_train",
        metadata={
            "split": "train",
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "data_type": "features"
        }
    )

    tracker.track_dataset(
        y_train, "y_train",
        metadata={
            "split": "train",
            "n_samples": len(y_train),
            "n_classes": len(np.unique(y_train)),
            "data_type": "labels"
        }
    )

    tracker.track_dataset(
        X_val, "X_val",
        metadata={
            "split": "validation",
            "n_samples": len(X_val),
            "n_features": X_val.shape[1],
            "data_type": "features"
        }
    )

    tracker.track_dataset(
        y_val, "y_val",
        metadata={
            "split": "validation",
            "n_samples": len(y_val),
            "n_classes": len(np.unique(y_val)),
            "data_type": "labels"
        }
    )

    tracker.track_dataset(
        X_test, "X_test",
        metadata={
            "split": "test",
            "n_samples": len(X_test),
            "n_features": X_test.shape[1],
            "data_type": "features"
        }
    )

    tracker.track_dataset(
        y_test, "y_test",
        metadata={
            "split": "test",
            "n_samples": len(y_test),
            "n_classes": len(np.unique(y_test)),
            "data_type": "labels"
        }
    )

    print("\n4. Preprocessing data...")
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Track preprocessed datasets
    tracker.track_dataset(
        X_train_scaled, "X_train_scaled",
        metadata={
            "split": "train",
            "preprocessing": "StandardScaler",
            "n_samples": len(X_train_scaled),
            "n_features": X_train_scaled.shape[1]
        }
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\n5. Creating and tracking model...")
    # Create model
    model = MLPClassifier(
        input_size=X_train.shape[1],
        hidden_sizes=[128, 64, 32],
        num_classes=len(np.unique(y)),
        dropout_rate=0.2
    )

    # Track model
    model_info = tracker.track_model(
        model, "mlp_classifier",
        metadata={
            "task": "classification",
            "architecture": "MLP",
            "input_size": X_train.shape[1],
            "hidden_sizes": [128, 64, 32],
            "num_classes": len(np.unique(y)),
            "dropout_rate": 0.2,
            "total_parameters": sum(p.numel() for p in model.parameters())
        }
    )

    print(f"   Model parameters: {model_info['num_parameters']:,}")
    print(f"   Architecture checksum: {model_info['architecture_checksum'][:16]}...")

    print("\n6. Training model...")
    # Add training metadata
    tracker.add_metadata("optimizer", "Adam")
    tracker.add_metadata("loss_function", "CrossEntropyLoss")
    tracker.add_metadata("learning_rate", 0.001)
    tracker.add_metadata("batch_size", 32)
    tracker.add_metadata("num_epochs", 10)
    tracker.add_metadata("device", str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    # Train model
    training_history = train_model(
        model, train_loader, val_loader,
        num_epochs=10, learning_rate=0.001
    )

    # Track training results
    tracker.add_metadata("final_train_accuracy", training_history['train_accuracies'][-1])
    tracker.add_metadata("final_val_accuracy", training_history['val_accuracies'][-1])
    tracker.add_metadata("final_train_loss", training_history['train_losses'][-1])
    tracker.add_metadata("final_val_loss", training_history['val_losses'][-1])

    print("\n7. Evaluating on test set...")
    # Evaluate on test set
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predictions = torch.max(test_outputs, 1)
        test_accuracy = (test_predictions == y_test_tensor).float().mean().item()

    tracker.add_metadata("test_accuracy", test_accuracy)
    print(f"   Test accuracy: {test_accuracy:.4f}")

    print("\n8. Adding final metadata and tags...")
    # Add more metadata
    tracker.add_metadata("dataset_type", "synthetic_classification")
    tracker.add_metadata("preprocessing", "StandardScaler")
    tracker.add_metadata("cross_validation", False)
    tracker.add_metadata("feature_selection", False)

    # Add tags
    tracker.add_tag("synthetic_data")
    tracker.add_tag("standardized")
    tracker.add_tag("no_cv")

    # Update notes
    tracker.set_notes(
        f"ML classification demo with {len(X_train)} training samples, "
        f"final test accuracy: {test_accuracy:.4f}"
    )

    print("\n9. Finishing experiment...")
    # Finish experiment and get summary
    summary = tracker.finish_experiment()

    print("\n" + "="*60)
    print("EXPERIMENT TRACKING SUMMARY")
    print("="*60)
    print(f"Experiment: {summary['experiment_name']}")
    print(f"ID: {summary['experiment_id']}")
    print(f"Output Directory: {summary['output_dir']}")
    print(f"Datasets Tracked: {summary['datasets_tracked']}")
    print(f"Models Tracked: {summary['models_tracked']}")
    print(f"Environment Captured: {summary['has_environment']}")
    print(f"Git State Captured: {summary['has_git']}")
    print(f"Seeds Tracked: {summary['has_seeds']}")
    print(f"Tags: {', '.join(summary['tags'])}")
    print(f"Metadata Keys: {len(summary['metadata_keys'])}")

    print("\n" + "="*60)
    print("FILES CREATED")
    print("="*60)

    # List created files
    output_files = list(output_dir.glob("*"))
    for file_path in sorted(output_files):
        if file_path.is_file():
            size = file_path.stat().st_size
            print(f"  {file_path.name} ({size:,} bytes)")

    print("\n" + "="*60)
    print("REPRODUCIBILITY INFORMATION")
    print("="*60)

    # Show key reproducibility info
    if summary['has_seeds']:
        print("✓ Random seeds captured and set")
    if summary['has_environment']:
        print("✓ Environment state captured")
    if summary['has_git']:
        print("✓ Git repository state captured")
    if summary['models_tracked'] > 0:
        print("✓ Model architecture fingerprinted")
    if summary['datasets_tracked'] > 0:
        print("✓ Dataset checksums computed")

    print("\nTo reproduce this experiment:")
    print(f"1. Use the same seed: {random_seed}")
    print(f"2. Check environment in: {output_dir}/ml_classification_demo_latest.json")
    print("3. Verify Git state in tracking files")
    print("4. Use the same model architecture (checksum available)")
    print("5. Verify dataset checksums match")

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
