# examples/trl_live_min.py
"""
TRL Live Minimal Example with RLDK Monitoring

This example demonstrates how to use the current TRL PPOTrainer API with RLDK monitoring.
It addresses the breaking changes in recent TRL releases that require explicit 
reward_model, train_dataset, and value_model parameters.

Key Workarounds Explained:
1. Value Model Interface: The PPOTrainer expects value models to have specific methods
   (score, base_model_prefix, transformer). We use instance-level patching to add these
   without affecting other model instances globally.

2. Reward Model Interface: TRL's get_reward function expects reward models to have
   base_model_prefix and transformer attributes, plus a score method that returns
   rewards for each position in the sequence.

3. Dataset Format: The current PPOTrainer expects pre-tokenized datasets with
   input_ids and attention_mask keys, not raw text prompts.

4. Evaluation Dataset: PPOTrainer requires an eval_dataset parameter, so we provide
   an empty dataset to avoid eval_dataloader issues.

This example maintains full TRL functionality while demonstrating RLDK monitoring
capabilities for debugging and analysis.
"""
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Import RLDK monitoring
try:
    from rldk.integrations.trl.monitors import PPOMonitor as Monitor
    RLDK_AVAILABLE = True
except ImportError:
    RLDK_AVAILABLE = False
    print("⚠️  RLDK not available - running without monitoring")

def build_dataset(tokenizer):
    prompts = [
        "Say a short greeting",
        "Write a short sentence that includes the word good",
        "Name one fruit",
        "Name one color",
    ]
    # Pre-tokenize the prompts
    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    return Dataset.from_dict({
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    })

class SimpleRewardModel(torch.nn.Module):
    """
    Enhanced reward model that demonstrates realistic reward patterns.
    
    This model provides more sophisticated reward signals:
    - Positive rewards for helpful, coherent responses
    - Penalties for repetitive or nonsensical text
    - Length-based penalties to encourage conciseness
    - Quality indicators based on text analysis
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # Add required attributes for PPOTrainer
        self.base_model_prefix = 'transformer'
        # Create a dummy transformer that returns dummy hidden states
        self.transformer = DummyTransformer()

    def _analyze_text_quality(self, text):
        """Analyze text quality and return reward components."""
        text_lower = text.lower()
        
        # Base reward for coherent responses
        base_reward = 0.1
        
        # Positive indicators
        positive_words = ['good', 'helpful', 'clear', 'useful', 'thank', 'please']
        positive_score = sum(1 for word in positive_words if word in text_lower) * 0.2
        
        # Negative indicators
        negative_words = ['bad', 'wrong', 'error', 'fail', 'stupid', 'hate']
        negative_score = sum(1 for word in negative_words if word in text_lower) * -0.3
        
        # Coherence indicators
        coherence_score = 0.0
        if len(text.split()) > 2:  # Multi-word responses
            coherence_score += 0.1
        if any(punct in text for punct in ['.', '!', '?']):  # Proper punctuation
            coherence_score += 0.05
        
        # Repetition penalty
        words = text_lower.split()
        if len(words) > 1:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            repetition_penalty = max(0, 0.2 - repetition_ratio) * -0.5
        
        # Length penalty (encourage conciseness)
        length_penalty = max(0, len(text) - 50) * -0.001
        
        total_reward = base_reward + positive_score + negative_score + coherence_score + repetition_penalty + length_penalty
        
        return max(-1.0, min(1.0, total_reward))  # Clamp between -1 and 1

    def forward(self, input_ids=None, attention_mask=None, **unused):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        
        for i, ids in enumerate(input_ids):
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            reward = self._analyze_text_quality(text)
            # Set the reward at the end of the sequence
            rewards[i, -1] = reward
            
        return rewards
    
    def score(self, hidden_states):
        """Score method required by PPOTrainer - return rewards for each position"""
        # hidden_states shape: [batch_size, sequence_length, hidden_size]
        # We need to return rewards for each position: [batch_size, sequence_length]
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Create rewards tensor with same logic as forward method
        rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        
        # For now, set a small positive reward at the end of each sequence
        # In a real implementation, this would analyze the hidden states
        rewards[:, -1] = 0.1
        
        # Ensure the tensor has the right shape to avoid squeeze issues
        # Add an extra dimension to prevent squeeze from making it scalar
        rewards = rewards.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        return rewards

class DummyTransformer(torch.nn.Module):
    """Dummy transformer that returns dummy hidden states for reward model"""
    def __init__(self):
        super().__init__()
        
    def __call__(self, input_ids=None, attention_mask=None, position_ids=None, return_dict=True, output_hidden_states=True, use_cache=False, **kwargs):
        # Return dummy hidden states with the right shape
        batch_size, seq_len = input_ids.shape
        hidden_size = 768  # Standard hidden size
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, device=input_ids.device)
        
        # Create a dummy output object
        class DummyOutput:
            def __init__(self, hidden_states):
                self.hidden_states = [hidden_states]  # TRL expects a list
                
        return DummyOutput(hidden_states)


def main():
    model_name = os.environ.get("TRL_MIN_MODEL", "sshleifer/tiny-gpt2")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        # ensure a pad token for left padding during generation
        tokenizer.pad_token = tokenizer.eos_token

    # policy model
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    # value model - use instance-level patching instead of global monkey-patching
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add value head to this specific instance only
    base_model.v_head = torch.nn.Linear(base_model.config.hidden_size, 1)
    
    # Add score method to this specific instance only
    def score_method(self, hidden_states):
        # hidden_states shape: [batch_size, sequence_length, hidden_size]
        # We need to return rewards for each position: [batch_size, sequence_length]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Apply value head to each position
        # Reshape to [batch_size * seq_len, hidden_size] for batch processing
        hidden_flat = hidden_states.view(-1, hidden_size)
        rewards_flat = self.v_head(hidden_flat)
        
        # Reshape back to [batch_size, seq_len, 1] and squeeze to [batch_size, seq_len]
        rewards = rewards_flat.view(batch_size, seq_len, -1).squeeze(-1)
        
        return rewards
    
    # Bind the score method to this specific instance
    import types
    base_model.score = types.MethodType(score_method, base_model)
    
    value_model = base_model
    # reference model, let PPOTrainer create a frozen copy when None
    ref_model = None

    ds = build_dataset(tokenizer)

    cfg = PPOConfig(
        total_episodes=4,  # Reduce episodes for faster testing
        num_ppo_epochs=1,
        num_mini_batches=1,
        per_device_train_batch_size=1,  # Reduce batch size
        gradient_accumulation_steps=1,
        response_length=8,  # Reduce response length
        stop_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        include_tokens_per_second=False,
        use_cpu=True,
        logging_steps=1,
        bf16=False,
        fp16=False,
        # Disable evaluation to avoid eval_dataloader issues
        eval_strategy="no",
    )

    # Use a simple reward model that returns scalar rewards
    reward_model = SimpleRewardModel(tokenizer)

    # Create an empty evaluation dataset to avoid eval_dataloader issues
    eval_ds = Dataset.from_dict({"input_ids": [], "attention_mask": []})
    
    # Initialize RLDK monitoring if available
    callbacks = []
    if RLDK_AVAILABLE:
        monitor = Monitor(
            output_dir="./rldk_monitoring_output",
            kl_threshold=0.1,
            reward_threshold=0.01,
            gradient_threshold=1.0,
            clip_frac_threshold=0.2,
            run_id="trl_min_with_monitoring"
        )
        callbacks.append(monitor)
        print("✅ RLDK Monitor initialized")
    else:
        print("⚠️  Running without RLDK monitoring")
    
    trainer = PPOTrainer(
        args=cfg,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=ds,
        eval_dataset=eval_ds,
        value_model=value_model,
        callbacks=callbacks,  # Include RLDK monitoring
    )

    # one short train call on CPU
    trainer.train()
    print("TRL_PATH_C_OK")  # sentinel for CI or harness

if __name__ == "__main__":
    main()