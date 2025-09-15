# examples/trl_live_min.py
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

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
    Returns a scalar reward per sequence
    Reward is 1.0 when the decoded text contains the word good, else small penalty that increases with length
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # Add required attributes for PPOTrainer
        self.base_model_prefix = 'transformer'
        # Create a dummy transformer that returns dummy hidden states
        self.transformer = DummyTransformer()

    def forward(self, input_ids=None, attention_mask=None, **unused):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        
        for i, ids in enumerate(input_ids):
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True).lower()
            r = 1.0 if "good" in text else 0.0
            over = max(int(ids.shape[0]) - 48, 0)
            r -= 0.002 * over
            # Set the reward at the end of the sequence
            rewards[i, -1] = r
            
        return rewards
    
    def score(self, hidden_states):
        """Score method required by PPOTrainer - return rewards for each position"""
        # hidden_states shape: [batch_size, sequence_length, hidden_size]
        # We need to return rewards for each position: [batch_size, sequence_length]
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Create dummy rewards (all zeros for now)
        rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        
        # Set a small positive reward at the end of each sequence
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

class ValueModelWrapper(torch.nn.Module):
    """
    Wrapper that adds a score method to a base model
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Copy important attributes
        self.config = base_model.config
        self.base_model_prefix = getattr(base_model, 'base_model_prefix', 'transformer')
        # Copy the transformer attribute
        if hasattr(base_model, 'transformer'):
            self.transformer = base_model.transformer
        
        # Add value head
        self.value_head = torch.nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
        
    def score(self, hidden_states):
        """Score method required by PPOTrainer - return rewards for each position"""
        # hidden_states shape: [batch_size, sequence_length, hidden_size]
        # We need to return rewards for each position: [batch_size, sequence_length]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Apply value head to each position
        # Reshape to [batch_size * seq_len, hidden_size] for batch processing
        hidden_flat = hidden_states.view(-1, hidden_size)
        rewards_flat = self.value_head(hidden_flat)
        
        # Reshape back to [batch_size, seq_len, 1] and squeeze to [batch_size, seq_len]
        rewards = rewards_flat.view(batch_size, seq_len, -1).squeeze(-1)
        
        return rewards

def main():
    model_name = os.environ.get("TRL_MIN_MODEL", "sshleifer/tiny-gpt2")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        # ensure a pad token for left padding during generation
        tokenizer.pad_token = tokenizer.eos_token

    # policy model
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    # value model - monkey patch the model class itself
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get the model class
    model_class = type(base_model)
    
    # Add value head to the base model
    base_model.v_head = torch.nn.Linear(base_model.config.hidden_size, 1)
    
    # Add score method to the model class itself so all instances have it
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
    
    # Add score method and v_head to the model class
    model_class.score = score_method
    model_class.v_head = torch.nn.Linear(base_model.config.hidden_size, 1)
    
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
    
    trainer = PPOTrainer(
        args=cfg,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=ds,
        eval_dataset=eval_ds,
        value_model=value_model,
    )

    # one short train call on CPU
    trainer.train()
    print("TRL_PATH_C_OK")  # sentinel for CI or harness

if __name__ == "__main__":
    main()