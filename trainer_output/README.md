---
base_model: sshleifer/tiny-gpt2
library_name: transformers
model_name: trainer_output
tags:
- generated_from_trainer
- ppo
- trl
licence: license
---

# Model Card for trainer_output

This model is a fine-tuned version of [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with PPO, a method introduced in [Fine-Tuning Language Models from Human Preferences](https://huggingface.co/papers/1909.08593).

### Framework versions

- TRL: 0.23.0
- Transformers: 4.56.1
- Pytorch: 2.8.0
- Datasets: 4.1.0
- Tokenizers: 0.22.0

## Citations

Cite PPO as:

```bibtex
@article{mziegler2019fine-tuning,
    title        = {{Fine-Tuning Language Models from Human Preferences}},
    author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
    year         = 2019,
    eprint       = {arXiv:1909.08593}
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```