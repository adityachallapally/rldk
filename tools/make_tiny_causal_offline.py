#!/usr/bin/env python3
"""Create a tiny offline causal model and tokenizer for PPO testing."""

import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def create_tiny_tokenizer(output_dir: str):
    """Create a tiny ByteLevel BPE tokenizer."""
    print("🔧 Creating tiny ByteLevel BPE tokenizer...")
    
    # Create a tiny in-memory corpus
    corpus = [
        "Write a one word positive review:",
        "Say something nice:",
        "The weather is good today.",
        "This is a great product.",
        "I love this awesome movie.",
        "The food tastes great.",
        "This is an amazing experience.",
        "I feel good about this.",
        "The service was excellent.",
        "This makes me happy.",
        "I think this is wonderful.",
        "The quality is outstanding.",
        "I'm impressed with this.",
        "This exceeds my expectations.",
        "I would recommend this to anyone.",
        "This is exactly what I needed.",
        "I'm very satisfied with this.",
        "This is perfect for me.",
        "I can't believe how good this is.",
        "This is the best thing ever.",
    ] * 3  # Repeat to have more data
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # Train tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=1000,  # Small vocab size
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
    )
    
    tokenizer.train_from_iterator(corpus, trainer)
    
    # Save tokenizer files
    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))
    
    # Create vocab.json and merges.txt manually for compatibility
    vocab = tokenizer.get_vocab()
    vocab_json = {token: idx for token, idx in vocab.items()}
    
    with open(Path(output_dir) / "vocab.json", "w") as f:
        json.dump(vocab_json, f, indent=2)
    
    # Create merges.txt (simplified version)
    merges = []
    with open(Path(output_dir) / "merges.txt", "w") as f:
        f.write("#version: 0.2\n")
        for merge in merges:
            f.write(f"{merge}\n")
    
    print(f"✅ Tokenizer saved to {output_dir}")
    return tokenizer


def create_tiny_model(output_dir: str, tokenizer):
    """Create a tiny GPT-2 model."""
    print("🔧 Creating tiny GPT-2 model...")
    
    vocab_size = len(tokenizer.get_vocab())
    
    # Import transformers here to avoid heavy imports at module level
    from transformers import GPT2Config, GPT2LMHeadModel
    
    # Create tiny GPT-2 config
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=256,  # n_ctx
        n_ctx=256,
        n_embd=128,
        n_layer=2,
        n_head=2,
        n_inner=512,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        bos_token_id=tokenizer.token_to_id("<|bos|>"),
        eos_token_id=tokenizer.token_to_id("<|eos|>"),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
    )
    
    # Create model
    model = GPT2LMHeadModel(config)
    
    # Save model
    model.save_pretrained(output_dir)
    
    print(f"✅ Model saved to {output_dir}")
    return model


def create_tokenizer_from_files(output_dir: str):
    """Create GPT2TokenizerFast from saved files."""
    print("🔧 Creating GPT2TokenizerFast from files...")
    
    # Import transformers here to avoid heavy imports at module level
    from transformers import GPT2TokenizerFast
    
    # Create tokenizer from files
    tokenizer = GPT2TokenizerFast(
        vocab_file=str(Path(output_dir) / "vocab.json"),
        merges_file=str(Path(output_dir) / "merges.txt"),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ GPT2TokenizerFast saved to {output_dir}")
    return tokenizer


def main():
    """Main function to create tiny model and tokenizer."""
    parser = argparse.ArgumentParser(description="Create tiny offline causal model")
    parser.add_argument("--out", default="assets/tiny_causal", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Creating tiny offline causal model and tokenizer")
    print(f"📁 Output directory: {output_dir}")
    
    # Create tokenizer
    tokenizer = create_tiny_tokenizer(output_dir)
    
    # Create model
    model = create_tiny_model(output_dir, tokenizer)
    
    # Create GPT2TokenizerFast from files
    gpt2_tokenizer = create_tokenizer_from_files(output_dir)
    
    # Verify files exist
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    
    print("\n📋 Verifying created files:")
    for file_name in required_files:
        file_path = output_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - Missing!")
    
    # Test tokenizer
    print("\n🧪 Testing tokenizer:")
    test_text = "Write a one word positive review:"
    tokens = gpt2_tokenizer.encode(test_text)
    decoded = gpt2_tokenizer.decode(tokens)
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Test model
    print("\n🧪 Testing model:")
    test_input = gpt2_tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**test_input)
    print(f"Model output shape: {outputs.logits.shape}")
    
    print(f"\n🎉 Tiny model creation completed!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Vocab size: {len(gpt2_tokenizer)}")


if __name__ == "__main__":
    main()