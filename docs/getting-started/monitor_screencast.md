# RLDK Monitor Screencast Script

Record a live demo of the JSONL monitor in action using the script below. It mirrors the acceptance flow: emit metrics, tail them
with `rldk monitor`, and confirm the auto-stop rule fires.

## Preparation

1. Ensure RLDK is installed (`pip install -e .` from the repository root).
2. Open two terminal windows (`Terminal A` for the trainer, `Terminal B` for the monitor).
3. `cd` into the repository root in both terminals.
4. Optional: `rm -f artifacts/run.jsonl artifacts/alerts.jsonl artifacts/report.json` to start fresh.

## Terminal A – start the training loop

```bash
python examples/minimal_streaming_loop.py &
export LOOP_PID=$!
```

Leave Terminal A visible so viewers see the PID printout and the loop logging JSON lines.

## Terminal B – attach the monitor

```bash
rldk monitor \
  --stream artifacts/run.jsonl \
  --rules rules.yaml \
  --pid $LOOP_PID \
  --alerts artifacts/alerts.jsonl \
  --report artifacts/report.json
```

Call out that the monitor is following the JSONL file live, applying the `stop_on_high_kl` rule, and will terminate the loop when
KL exceeds 0.35 for five consecutive steps.

## Show the alerts

Once the monitor exits, highlight the generated artifacts:

```bash
cat artifacts/alerts.jsonl
cat artifacts/report.json
```

Explain the key fields: `rule_id`, `message`, `action`, and `window` size.

## Batch parity clip

Record a quick follow-up showing batch mode producing the same summary:

```bash
rldk monitor --once artifacts/run.jsonl --rules rules.yaml --report artifacts/report.json
```

Point out that alerts are unchanged and the report is deterministic.

## TRL/hosted run teaser (optional)

If you have a TRL run handy:

```bash
python train_trl.py &
export PPO_PID=$!
# Emit JSONL metrics from your TRL loop as shown in the README's two-line logger snippet.
rldk monitor --stream artifacts/run.jsonl --rules ppo_safe --pid $PPO_PID --preset trl
```

Alternatively, mention that `rldk monitor --from-wandb entity/project/run --rules ppo_safe` follows hosted runs without modifying
trainer code.

## Clean up

Back in Terminal A:

```bash
wait $LOOP_PID
```

Remind viewers they can repeat the full flow at any time with `make monitor-demo`.
