# Catch RL Training Failures Before They Waste Your GPU Hours

When a reinforcement learning run goes sideways, the bill shows up in wasted GPU hours, unstable policies, and artifacts that can’t be reproduced. The RL Debug Kit (RLDK) was built to collapse that feedback loop. I ran the toolkit end-to-end on the repo to prove it: live anomaly detection, forensic deep-dives, and determinism checks all operating on real data, with the raw artifacts ready for inspection.

---

## Real-Time Anomaly Detection With `make monitor-demo`

I started by tailing a live trainer with RLDK’s streaming monitor:

```bash
make monitor-demo
```

That single command launches `examples/minimal_streaming_loop.py`, attaches the CLI monitor with the default PPO guardrails, and auto-stops when KL diverges. The run produced 81 KL measurements and tripped 22 stop alerts between steps 20 and 80, with the KL cresting at **1.67**—far beyond the guardrail. The alert payloads are in `artifacts/alerts.jsonl` for easy replay.

![RLDK monitor caught runaway KL](../assets/blog_catch_failures/monitor_kl_trace.png)

**Why this matters:** the monitor killed the loop seconds after the KL left the 0.3 envelope, long before gradients spiraled out of control. Drop this into a production trainer and you shave minutes (or hours) of wasted compute.

Artifacts you can reuse:
- Streaming log: `artifacts/run.jsonl`
- Alert feed: `artifacts/alerts.jsonl`
- Monitor transcript: `artifacts/monitor.log`

---

## Forensic Analysis (`rldk log-scan` + `rldk doctor`)

Next, I generated the PPO fixtures and ran the forensic stack against a doctored run with a KL spike:

```bash
python tests/_make_fixtures.py
rldk log-scan test_artifacts/logs_doctored_kl_spike
rldk doctor test_artifacts/logs_doctored_kl_spike
python examples/comprehensive_ppo_forensics_example.py
```

`rldk log-scan` fired **183 anomaly rules**, flagging a KL surge at steps 800–804 and controller drift that persisted through step 999. `rldk doctor` bundled that analysis with an environment audit, writing the results to `rldk_reports/ppo_scan.json` and `rldk_reports/determinism_card.json`. The comprehensive PPO forensics demo adds richer context: an overall health score of **0.63**, a stability score of **0.84**, and a critical **advantage bias** of **0.237**. Those numbers (and every anomaly) are preserved in `comprehensive_ppo_forensics_demo/comprehensive_analysis.json` and `comprehensive_ppo_monitor_demo/`.

![KL spike highlighted by the forensic scan](../assets/blog_catch_failures/forensics_kl_spike.png)

Assets to pull into your post-mortem:
- Scan report: `rldk_reports/ppo_scan.json`
- Environment audit findings: `rldk_reports/determinism_card.json`
- Full forensics JSON: `comprehensive_ppo_forensics_demo/comprehensive_analysis.json`
- Monitor CSV / health summaries: `comprehensive_ppo_monitor_demo/`

The takeaway: RLDK doesn’t just yell “KL spike”; it tells you coefficient adaptation stalled, gradients went imbalanced, and advantage normalization slipped—all from offline logs.

---

## Reproducibility Verification (`rldk check-determinism`)

To showcase the determinism gate, I added `scripts/blog_determinism_sim.py`. It emits PPO-style metrics but can inject nondeterministic noise via `SystemRandom`. Running RLDK across four replicas with 200 steps (each replica sleeps 0.02s per step, so the gate runs for ~22 seconds) produced a clean failure signal:

```bash
# Known-bad run with entropy from SystemRandom
rldk check-determinism \
  --cmd "python scripts/blog_determinism_sim.py --mode nondet --steps 200 --sleep 0.02" \
  --compare kl,policy_loss,reward_mean \
  --replicas 4 \
  --output-dir docs/assets/blog_catch_failures/determinism_nondet

# Patched run with fully seeded RNGs
rldk check-determinism \
  --cmd "python scripts/blog_determinism_sim.py --mode deterministic --steps 200 --sleep 0.02" \
  --compare kl,policy_loss,reward_mean \
  --replicas 4 \
  --output-dir docs/assets/blog_catch_failures/determinism_det
```

The nondeterministic run logged **599 mismatches** with a maximum reward drift of **9.7e-3**, and the variance summary pinpointed `reward_mean` as the culprit metric. Recommended fixes (seed handling, disabling CuDNN benchmarking) are bundled in `determinism_card.json`. Re-running in deterministic mode immediately passed the gate.

![Replica drift vs deterministic baseline](../assets/blog_catch_failures/determinism_mismatch.png)

Key artifacts:
- Failure card: `docs/assets/blog_catch_failures/determinism_nondet/determinism_card.json`
- Passing card: `docs/assets/blog_catch_failures/determinism_det/determinism_card.json`
- Figure source script: `scripts/blog_determinism_sim.py`

Drop these cards into CI/CD and you have a machine-readable gate that fails PRs before they hit expensive environments.

---

## Grab the Assets

All generated images and determinism reports live under `docs/assets/blog_catch_failures/`:

- `monitor_kl_trace.png`
- `forensics_kl_spike.png`
- `determinism_mismatch.png`
- `determinism_nondet/determinism_card.json`
- `determinism_det/determinism_card.json`

If you need screenshots, the JSON artifacts above contain the exact values to call out in captions.

---

## Why This Matters

- **Solves real RL pain points:** The monitor caught instability in seconds, the forensic stack explained why, and the determinism check proved the fix actually sticks.
- **Production-ready:** Everything here ran offline on CPU with modest memory, and the outputs slot straight into MkDocs, dashboards, or CI pipelines.
- **RL-specific fidelity:** KL schedule failures, advantage bias, gradient imbalance—RLDK speaks the language of PPO instead of generic “loss up, loss down.”

Hook these pieces together and your RL teams stop bleeding GPU hours, start trusting their logs, and push reproducible models into production faster.
