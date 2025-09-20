#!/usr/bin/env python3
"""Download HuggingFace models of different sizes for RLDK testing."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('Starting HuggingFace model downloads for RL testing...')
    
    models = [
        ("microsoft/DialoGPT-small", "small"),  # ~117M parameters
        ("gpt2", "medium"),  # ~124M parameters  
        ("microsoft/DialoGPT-medium", "large"),  # ~345M parameters
    ]
    
    for model_name, size in models:
        print(f"\n=== Downloading {size} model: {model_name} ===")
        try:
            print(f"Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Downloading model for {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print(f"✓ Successfully downloaded {model_name} ({size})")
            print(f"  Model parameters: {model.num_parameters():,}")
            print(f"  Tokenizer vocab size: {len(tokenizer)}")
            
            model_info = {
                'name': model_name,
                'size': size,
                'parameters': model.num_parameters(),
                'vocab_size': len(tokenizer)
            }
            
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            continue
    
    print("\n=== Model download complete ===")

if __name__ == "__main__":
    main()
