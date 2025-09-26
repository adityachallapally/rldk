# Tiny run configuration schema

The example runners in `examples/run_ppo_tiny.py` and `examples/run_grpo_tiny.py` load
YAML files from this directory to obtain both trainer arguments and metadata for
logging. Each file shares a common structure:

- `model` *(str)* – model identifier forwarded to the tokenizer and policy/value heads.
- `dataset_seed` *(int)* – seed applied before constructing the toy dataset so the
  sampling order is deterministic.
- `steps` *(int)* – maximum number of optimisation steps; mapped to the trainer
  configuration if it is not explicitly overridden.
- `logging_interval` *(int)* – frequency (in steps) for emitting training metrics.
- `log_path` *(str)* – JSONL path where the runners should write `EventWriter` output;
  relative paths are resolved against the repository root so they mirror the CLI defaults.
- `ppo_kwargs` / `grpo_kwargs` *(mapping)* – keyword arguments forwarded directly to
  `trl.PPOConfig` or `trl.GRPOConfig` after the helper injects the step and logging
  defaults listed above.

This keeps the example scripts declarative while giving tests a single source of truth
for verifying schema changes.
