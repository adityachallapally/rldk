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
        """Score method required by PPOTrainer"""
        return self.value_head(hidden_states.mean(dim=1))

def main():
    model_name = os.environ.get("TRL_MIN_MODEL", "sshleifer/tiny-gpt2")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        # ensure a pad token for left padding during generation
        tokenizer.pad_token = tokenizer.eos_token

    # policy model
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    # value model - use base model directly
    value_model = AutoModelForCausalLM.from_pretrained(model_name)
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
    )

    # Use a simple reward model that returns scalar rewards
    reward_model = SimpleRewardModel(tokenizer)

    trainer = PPOTrainer(
        args=cfg,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=ds,
        value_model=policy,  # Use policy as value model
    )

    # one short train call on CPU
    trainer.train()
    print("TRL_PATH_C_OK")  # sentinel for CI or harness

if __name__ == "__main__":
    main()