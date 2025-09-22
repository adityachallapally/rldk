# Live Monitor Rules Cookbook

Use this cookbook to go from raw JSONL metrics to automated, rule-driven safeguards in minutes. All examples use the
framework-agnostic `rldk monitor` CLI that consumes JSONL events written by any trainer.

## Reproduce the auto-stop demo (10 commands)

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -e .`
4. `python examples/minimal_streaming_loop.py &`
5. `export LOOP_PID=$!`
6. `sleep 1`  
   *(give the loop time to create `artifacts/run.jsonl`)*
7. `rldk monitor --stream artifacts/run.jsonl --rules rules.yaml --pid $LOOP_PID`
8. `wait $LOOP_PID`
9. `cat artifacts/alerts.jsonl`
10. `cat artifacts/report.json`

Prefer a single command? `make monitor-demo` orchestrates the same sequence, captures logs, and tails the resulting alerts.

## JSONL event schema recap

Every line is an independent JSON object. RLDK expects the following canonical keys:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `time` | ISO8601 string | ✅ | When absent, the emitter will set the current UTC time. |
| `step` | integer | ✅ | Training step or update number. |
| `name` | string | ✅ | Metric identifier; windows are tracked per unique `name` + `tags`. |
| `value` | number | ✅ | Metric value for gating or aggregation. |
| `run_id` | string | Optional | Associates events with a specific run. |
| `tags` | object | Optional | Any structured metadata used for filtering, e.g. `{ "env": "prod" }`. |
| `meta` | object | Optional | Free-form metadata recorded alongside alerts. |

Use field-map presets (`--preset trl|accelerate|openrlhf`) or the `--field-map '{"s":"step","metric":"name"}'` flag to
normalize custom schemas on the fly.

## Rule anatomy

Rules are defined in YAML under a top-level `rules:` key. Each rule supports the following attributes:

- `id` – unique identifier used in logs and reports.
- `where` – boolean expression evaluated against each event. Supports dot access for tags/meta (`tags.env == "prod"`).
- `condition` – expression evaluated on the current event window. Use `value`, `mean(value)`, `max(value)`, `min(value)`,
  or aggregators like `sum(value)`, `count(value > 0)`, `any(value > 0.5)`, and `all(value < 0.2)`.
- `window` – `{ size: N, kind: consecutive|rolling }`. Consecutive windows require `N` back-to-back hits, while rolling windows
  consider the last `N` events.
- `grace_steps` – minimum number of events required before the rule activates (default `0`).
- `cooldown_steps` – suppresses re-triggering for `N` steps after an alert fires.
- `actions` – ordered list of actions to execute (`warn`, `stop`, `shell`, `http`). Each action accepts templated messages using
  Python-style formatting with the most recent event in scope, e.g. `"KL {value:.3f} at step {step}"`.

### Starter rule set (`rules.yaml`)

```yaml
rules:
  - id: stop_on_high_kl
    where: name == "kl"
    condition: value > 0.35
    window:
      size: 5
      kind: consecutive
    grace_steps: 5
    cooldown_steps: 5
    actions:
      - warn:
          msg: "KL {value:.3f} exceeded at step {step}"
      - stop:
          msg: "Stopping run due to KL {value:.3f}"
```

This rule matches the demo loop: once KL stays above `0.35` for five consecutive events (with a five-step warm-up), the monitor
emits a warning and terminates the process specified with `--pid`.

### Cookbook patterns

**Catch reward free fall**

```yaml
  - id: reward_freefall
    where: name in ("reward", "reward_mean", "train/reward")
    condition: mean(value) < -0.2
    window:
      size: 12
      kind: rolling
    grace_steps: 10
    cooldown_steps: 10
    actions:
      - warn:
          msg: "Reward trending negative ({value:.3f} rolling mean)"
```

**Detect exploding gradients**

```yaml
  - id: grad_norm_spike
    where: name in ("grad_norm", "policy_grad_norm", "ppo/policy/grad_norm")
    condition: value > 12.0
    window:
      size: 3
      kind: consecutive
    grace_steps: 6
    cooldown_steps: 6
    actions:
      - warn:
          msg: "Gradient norm {value:.2f} at step {step}"
```

**Alert on stuck metrics using `any`/`all`**

```yaml
  - id: reward_plateau
    where: name == "reward"
    condition: max(value) - min(value) < 0.05
    window:
      size: 12
      kind: rolling
    actions:
      - warn:
          msg: "Reward plateau detected at {value:.3f}"
```

Mix and match these patterns with presets via `rldk monitor --rules ppo_safe` or your custom YAML.

### New: length bias preset

The `length_bias` preset watches for reward hacking driven by response length. It consumes the
`length_bias_score`, `length_reward_correlation_abs`, and `length_reward_spearman_abs` metrics emitted
by the comprehensive PPO forensics pipeline. When the configured thresholds (surfaced via the event
metadata) are exceeded, the preset first warns and then halts persistent length-driven optimization.
Enable it alongside PPO presets to stay ahead of length bias regressions during reward modeling or
online PPO runs:

```bash
rldk monitor --rules ppo_safe --rules length_bias --stream artifacts/run.jsonl
```

## Fullscale remediation hints

The fullscale acceptance run reuses the following guardrails from
`rules/fullscale_rules.yaml`. When they fire, apply the same remediations suggested by the
alerts:

- **KL spike guard (`kl_spike_guard`)** – Lower the policy temperature or learning rate to
  tighten updates. The acceptance defaults are `--temperature 0.95` and `--learning-rate 8e-5`.
- **Reward collapse watch (`reward_collapse_watch`)** – Increase the batch size for steadier
  gradients or reduce the sampling temperature. Defaults are `--batch-size 4` and
  `--temperature 0.95`.
- **Gradient norm ceiling (`grad_norm_ceiling`)** – Ensure clipping is enabled with
  `--max-grad-norm 2.5` or lower the learning rate when gradients breach the ceiling.

These are the same hints echoed by the monitor when `scripts/fullscale_acceptance.sh`
detects issues, so you can remediate failures locally before rerunning the acceptance
pipeline.

## Action reference

| Action | Description | Key fields |
| --- | --- | --- |
| `warn` | Appends to `alerts.jsonl` and prints to stderr/stdout. | `msg` (templated string) |
| `stop` | Sends SIGTERM to `--pid` (falls back to SIGKILL after `--kill-timeout-sec`). | `msg`, `pid`, `kill_timeout_sec` |
| `shell` | Executes a shell command once per activation and records exit status/output. | `cmd`, `timeout`, `cwd`, `env` |
| `http` | Issues an HTTP request with retries and timeout controls. | `url`, `payload`, `method`, `headers`, `timeout`, `retries` |

All actions append a structured entry to `alerts.jsonl` that includes `rule_id`, `action`, `message`, and the latest event data.

## Outputs to expect

- `artifacts/alerts.jsonl` – chronological log of alert activations and action outcomes.
- `artifacts/report.json` – deterministic per-run summary with activation counts, first/last trigger times, and final metric values.
- Terminal stderr/stdout – mirrors warning actions so you can see them while tailing.

## Next steps

- Use `--once` for batch verification of completed runs: `rldk monitor --once artifacts/run.jsonl --rules rules.yaml --report artifacts/report.json`.
- Tail remote training runs without code changes: `rldk monitor --from-wandb entity/project/run --rules ppo_safe`.
- Scrape stdout directly using regex presets: `python train.py | rldk monitor --regex trl --rules ppo_safe`.

For a narrated version, follow the [screencast script](monitor_screencast.md) to record the full workflow end-to-end.
