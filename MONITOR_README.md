# RLDK Monitor Core

Framework-agnostic monitoring for RL training with live, log-first monitoring and gating.

## Quick Start

### 1. Emit Events
```python
from rldk.emit import EventWriter

# In your training loop
with EventWriter("artifacts/run.jsonl") as writer:
    writer.log(step=step, name="kl", value=kl_value)
    writer.log(step=step, name="reward", value=reward_value)
    writer.log(step=step, name="grad_norm", value=grad_norm_value)
```

### 2. Define Rules
Create `rules.yaml`:
```yaml
rules:
  - id: stop_on_high_kl
    where: name == "kl"
    condition: value > 0.35
    window:
      size: 5
      kind: consecutive
    cooldown_steps: 5
    actions:
      - warn:
          msg: "KL {value:.3f} exceeded at step {step}"
```

### 3. Monitor Training
```bash
# Live streaming
rldk monitor --stream artifacts/run.jsonl --rules rules.yaml --alerts artifacts/alerts.jsonl

# Batch analysis
rldk monitor --once artifacts/run.jsonl --rules rules.yaml --report artifacts/report.json
```

## Event Schema

Canonical JSONL format (one JSON per line):
```json
{
  "time": "2025-09-16T18:00:00Z",
  "step": 101,
  "name": "kl",
  "value": 0.41,
  "run_id": "run-123",
  "tags": {"env": "prod"},
  "meta": {}
}
```

**Required fields**: `time`, `step`, `name`, `value`  
**Optional fields**: `run_id`, `tags`, `meta`

## Rules DSL

### Rule Structure
```yaml
rules:
  - id: unique_string
    where: Python-like boolean filter
    condition: expression on event window
    window:
      size: N
      kind: consecutive|rolling
    cooldown_steps: N
    grace_steps: N
    actions:
      - warn:
          msg: "Template message {value:.3f} at step {step}"
```

### Where Clauses
- `name == "kl"` - Match metric name
- `tags.env == "prod"` - Match tag values
- `step > 100` - Match step numbers

### Conditions
- `value > 0.35` - Direct value comparison
- `mean(value) > 0.2` - Rolling average
- `max(value) > 0.9` - Maximum in window
- `any(value > 0.5)` - Any event in window

### Windows
- **Consecutive**: Reset if steps are not sequential
- **Rolling**: Maintains fixed-size sliding window

## CLI Commands

### Monitor
```bash
rldk monitor --stream PATH|- --rules FILE [--alerts PATH] [--field-map JSON]
rldk monitor --once PATH --rules FILE [--report FILE] [--field-map JSON]
```

### Emit
```bash
rldk emit --to PATH --name X --value Y --step S [--run-id ID] [--tags JSON] [--meta JSON]
```

## Field Mapping

Map custom schemas to canonical format:
```bash
--field-map '{"s":"step","metric":"name","v":"value"}'
```

## Examples

### Minimal Training Loop
```python
#!/usr/bin/env python3
import random
import time
from rldk.emit import EventWriter

with EventWriter("artifacts/run.jsonl") as writer:
    for step in range(100):
        kl = random.uniform(0.1, 0.5)
        reward = random.uniform(-1.0, 1.0)
        
        writer.log(step=step, name="kl", value=kl)
        writer.log(step=step, name="reward", value=reward)
        
        time.sleep(1)
```

### TRL Integration (Log-First)
```python
# In your PPO training loop
import json
import time
from datetime import datetime

class JsonlLogger:
    def __init__(self, path, run_id=None):
        self.f = open(path, "a", buffering=1)
        self.run_id = run_id or f"run-{int(time.time())}"
    
    def log(self, step, name, value, **kw):
        evt = {
            "time": datetime.utcnow().isoformat()+"Z",
            "step": int(step),
            "name": str(name),
            "value": float(value),
            "run_id": self.run_id
        }
        evt.update(kw)
        self.f.write(json.dumps(evt) + "\n")
        self.f.flush()

# Usage
logger = JsonlLogger("artifacts/run.jsonl", run_id="trl-ppo-test")
metrics = {"kl": kl_value, "reward": reward_value, "grad_norm": grad_norm}
for k, v in metrics.items():
    logger.log(step=global_step, name=k, value=v)
```

## Output Files

### alerts.jsonl
One line per alert activation:
```json
{
  "ts": "2025-09-16T18:00:00Z",
  "step": 101,
  "rule_id": "stop_on_high_kl",
  "metric": "kl",
  "value": 0.41,
  "window": {"size": 5, "kind": "consecutive"},
  "action": "warn",
  "message": "KL 0.410 exceeded at step 101",
  "run_id": "run-123",
  "tags": {},
  "meta": {}
}
```

### report.json
Deterministic summary per run:
```json
{
  "timestamp": "2025-09-16T18:00:00Z",
  "rules_summary": [...],
  "last_seen_metrics": {...},
  "total_alerts": 5
}
```

## Framework Agnostic

No dependencies on TRL, Hugging Face, or any specific framework. Works with any trainer that can write JSONL logs.

## Robustness

- Handles partial lines and file rotation
- Graceful error handling for malformed JSON
- Supports stdin piping (`--stream -`)
- Atomic, line-buffered writes