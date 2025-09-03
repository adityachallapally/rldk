#!/usr/bin/env python3
"""
02_train_reward_model_full.py - Train DistilBERT reward model on preference data

This script trains a reward model using DistilBERT to predict which response
is preferred in a pair. The model is trained for 3 epochs with CPU-optimized
settings and comprehensive evaluation.
"""

import json
import os
import random
import hashlib
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize chosen response
        chosen_text = f"{item['prompt']} [SEP] {item['chosen']}"
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize rejected response
        rejected_text = f"{item['prompt']} [SEP] {item['rejected']}"
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
            'label': 1  # 1 means chosen is preferred
        }

class RewardModel(nn.Module):
    """Reward model based on DistilBERT."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.float())
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

class RewardModelTrainer:
    """Custom trainer for reward model training."""
    
    def __init__(self, model, tokenizer, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute preference loss."""
        chosen_input_ids = inputs['chosen_input_ids']
        chosen_attention_mask = inputs['chosen_attention_mask']
        rejected_input_ids = inputs['rejected_input_ids']
        rejected_attention_mask = inputs['rejected_attention_mask']
        
        # Get rewards for chosen and rejected responses
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        chosen_rewards = chosen_outputs['logits'].squeeze()
        rejected_rewards = rejected_outputs['logits'].squeeze()
        
        # Preference loss: chosen should have higher reward than rejected
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        return (loss, chosen_outputs) if return_outputs else loss
    
    def train(self, train_dataset, eval_dataset=None):
        """Train the model."""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        # Convert to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_preds, average='binary')
        
        try:
            auc = roc_auc_score(labels, predictions)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

def load_preference_data(filename: str) -> List[Dict[str, Any]]:
    """Load preference data from JSONL file."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_reward_model(model, tokenizer, eval_data: List[Dict[str, Any]], 
                         batch_size: int = 4) -> Dict[str, Any]:
    """Comprehensive evaluation of the reward model."""
    model.eval()
    
    all_chosen_rewards = []
    all_rejected_rewards = []
    all_predictions = []
    all_labels = []
    
    eval_dataset = PreferenceDataset(eval_data, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_loader:
            chosen_input_ids = batch['chosen_input_ids']
            chosen_attention_mask = batch['chosen_attention_mask']
            rejected_input_ids = batch['rejected_input_ids']
            rejected_attention_mask = batch['rejected_attention_mask']
            
            # Get rewards
            chosen_outputs = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            rejected_outputs = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            )
            
            chosen_rewards = chosen_outputs['logits'].squeeze().cpu().numpy()
            rejected_rewards = rejected_outputs['logits'].squeeze().cpu().numpy()
            
            all_chosen_rewards.extend(chosen_rewards)
            all_rejected_rewards.extend(rejected_rewards)
            
            # Predictions: 1 if chosen > rejected, 0 otherwise
            predictions = (chosen_rewards > rejected_rewards).astype(int)
            all_predictions.extend(predictions)
            all_labels.extend([1] * len(predictions))  # All should be 1 (chosen preferred)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    # Calculate reward statistics
    chosen_mean = np.mean(all_chosen_rewards)
    chosen_std = np.std(all_chosen_rewards)
    rejected_mean = np.mean(all_rejected_rewards)
    rejected_std = np.std(all_rejected_rewards)
    
    # Calculate preference margin
    margins = np.array(all_chosen_rewards) - np.array(all_rejected_rewards)
    margin_mean = np.mean(margins)
    margin_std = np.std(margins)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'chosen_reward_mean': chosen_mean,
        'chosen_reward_std': chosen_std,
        'rejected_reward_mean': rejected_mean,
        'rejected_reward_std': rejected_std,
        'preference_margin_mean': margin_mean,
        'preference_margin_std': margin_std,
        'total_samples': len(all_labels)
    }

def create_calibration_plot(model, tokenizer, eval_data: List[Dict[str, Any]], 
                           output_dir: str):
    """Create calibration plot for the reward model."""
    model.eval()
    
    all_chosen_rewards = []
    all_rejected_rewards = []
    
    eval_dataset = PreferenceDataset(eval_data, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_loader:
            chosen_input_ids = batch['chosen_input_ids']
            chosen_attention_mask = batch['chosen_attention_mask']
            rejected_input_ids = batch['rejected_input_ids']
            rejected_attention_mask = batch['rejected_attention_mask']
            
            chosen_outputs = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            rejected_outputs = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            )
            
            chosen_rewards = chosen_outputs['logits'].squeeze().cpu().numpy()
            rejected_rewards = rejected_outputs['logits'].squeeze().cpu().numpy()
            
            all_chosen_rewards.extend(chosen_rewards)
            all_rejected_rewards.extend(rejected_rewards)
    
    # Calculate preference probabilities
    chosen_rewards = np.array(all_chosen_rewards)
    rejected_rewards = np.array(all_rejected_rewards)
    preference_probs = torch.sigmoid(torch.tensor(chosen_rewards - rejected_rewards)).numpy()
    
    # Create calibration plot
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        [1] * len(preference_probs), preference_probs, n_bins=10
    )
    
    plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
             label=f'Reward Model (ECE: {np.mean(np.abs(fraction_of_positives - mean_predicted_value)):.3f})')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reward Model Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'ece': np.mean(np.abs(fraction_of_positives - mean_predicted_value)),
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist()
    }

def save_model_artifacts(model, tokenizer, output_dir: str, metadata: Dict[str, Any]):
    """Save model and tokenizer artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save tokenizer config
    tokenizer_config = {
        'tokenizer_class': tokenizer.__class__.__name__,
        'model_max_length': tokenizer.model_max_length,
        'padding_side': tokenizer.padding_side,
        'pad_token_id': tokenizer.pad_token_id,
        'vocab_size': tokenizer.vocab_size
    }
    
    with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

def main():
    """Main training function."""
    print("Starting reward model training...")
    print(f"Using random seed: {RANDOM_SEED}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CPU' if not torch.cuda.is_available() else 'CUDA'}")
    
    # Load data
    print("\nLoading preference data...")
    train_data = load_preference_data("./rldk_demos/rm_pairs_train.jsonl")
    val_data = load_preference_data("./rldk_demos/rm_pairs_val.jsonl")
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Initialize tokenizer and model
    print("\nInitializing tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = RewardModel(model_name)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer max length: {tokenizer.model_max_length}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = PreferenceDataset(train_data, tokenizer, max_length=512)
    val_dataset = PreferenceDataset(val_data, tokenizer, max_length=512)
    
    # Training arguments (CPU-optimized)
    training_args = TrainingArguments(
        output_dir="./rldk_demos/rm_a/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # CPU-optimized
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size: 16
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb for CPU training
        dataloader_num_workers=0,  # CPU optimization
        remove_unused_columns=False,
        fp16=False,  # Disable mixed precision for CPU
        bf16=False,  # Disable bfloat16 for CPU
        seed=RANDOM_SEED,
        disable_tqdm=False
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = RewardModelTrainer(model, tokenizer, training_args)
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    trained_trainer = trainer.train(train_dataset, val_dataset)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = evaluate_reward_model(model, tokenizer, val_data)
    
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Create calibration plot
    print("\nCreating calibration plot...")
    calibration_results = create_calibration_plot(model, tokenizer, val_data, "./rldk_demos/rm_a")
    
    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "random_seed": RANDOM_SEED,
        "training_args": {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "warmup_steps": training_args.warmup_steps
        },
        "evaluation_results": eval_results,
        "calibration_results": calibration_results,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_config": {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id
        }
    }
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    save_model_artifacts(model, tokenizer, "./rldk_demos/rm_a", metadata)
    
    # Save evaluation pairs for later use
    print("\nSaving evaluation pairs...")
    with open("./rldk_demos/rm_eval_pairs.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\nReward model training completed!")
    print(f"Model saved to: ./rldk_demos/rm_a")
    print(f"Evaluation pairs saved to: ./rldk_demos/rm_eval_pairs.jsonl")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Final accuracy: {eval_results['accuracy']:.4f}")
    print(f"Final F1 score: {eval_results['f1']:.4f}")
    print(f"Preference margin: {eval_results['preference_margin_mean']:.4f} ± {eval_results['preference_margin_std']:.4f}")

if __name__ == "__main__":
    main()