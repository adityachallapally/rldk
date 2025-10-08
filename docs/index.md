# Silent Failures in Modern RL: A Field Guide

## Who This Is For

This guide covers RLHF, RLVR, environment RL, and agentic setups using PPO, GRPO, or hybrids. The failure modes recur across methods because they all share:

- **Reward source**: learned reward model, verifier/critic, environment reward, or mixed evaluators
- **Policy optimizer**: trust-regularized gradients (PPO clipping, KL penalties, or equivalent)
- **Stability constraints**: KL penalties and/or entropy bonuses

## About Preference Optimization Methods

Recent work (DPO, IPO, KTO, RLAIF) reframes RL as preference optimization, avoiding explicit reward models. DPO directly optimizes on preference pairs:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

This eliminates reward model training and some failure modes. But silent failures still occur:

- Distribution shift between preference data and policy outputs
- Length bias and spurious correlations in preference labels
- Entropy collapse from strong preference signals
- Offline overoptimization (analog of proxy exploitation)

**Why this guide matters**: Whether you're doing online RL with a reward model or offline DPO, you're optimizing a proxy signal under distributional constraints. The failures manifest differently but need similar vigilance. This guide focuses on online RL where silent failures are most acute, but the principles apply broadly.

## Metric Conventions

- **KL**: mean per-token forward KL. Healthy targets: 0.05–0.3 (calibrate on your baseline)
- **Entropy (H)**: per-token entropy in nats. Healthy: 2–4 nats. Under 1.2 nats broadly suggests collapse
- **Step**: one optimizer step over a batch of rollouts
- **β**: KL coefficient (reward penalty weight)
- **ε (cliprange)**: PPO ratio clip (typically 0.1–0.2)

## What We Actually Optimize

Keep this mental model:

**Shaped reward per token**:
$$\tilde{r}_t = r^{proxy}_t - \beta(\log \pi_\theta(a_t|s_t) - \log \pi_{ref}(a_t|s_t))$$

where $r^{proxy}$ is your learned RM/verifier or environment reward.

**Advantages** come from either:
- Critic-based (GAE on returns)
- Group-normalized (GRPO): normalize rewards within prompt groups

**Clipped objective** (PPO-style):
$$L^{CLIP} = \mathbb{E}[\min(r_t\hat{A}_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t)]$$

Think of β as a thermostat: it keeps your policy close to the reference by penalizing divergence. But like any thermostat, it can't save you from a fundamentally broken reward signal.

### Why Trust Regions Aren't Enough

The KL constraint assumes your reference policy is a good anchor. In practice:

1. The reward is an approximation (learned RM, verifier, environment)
2. Distribution shift accumulates as the policy moves
3. KL measures distribution distance, not semantic quality

Policies can mode-collapse within the KL budget. This is why you need additional monitoring.

## Failure Mode 1: KL Oscillation or Drift

**What happens**: Trust regularization is a feedback controller. You adapt β to keep KL near a target τ. If β reacts too slowly, the policy outruns the budget. Too fast, learning stalls.

**Why it happens**: The system has natural oscillations because KL is measured after updates (one-step lag), and advantages change as policy and baseline evolve. When policy learning rate × advantage scale × (1-ε) exceeds what β can compensate for, you get spikes or drift.

**How to detect**: Oscillating KL, slow KL climb while β rises, single-step gradient spikes, policy step norms near zero while baseline keeps updating.

**Cost example**: If an incident at step 3,847 goes unseen until 15,000, you wasted 11,153 steps. At 7.5s per step: 23 hours × 32 GPUs × $2/GPU-hour = $1,487.

**How to fix**: 
- Tighten controller (faster β adaptation, narrower bands)
- Reduce cliprange ε or inner epochs
- Clip gradients, clamp minimum advantage std
- If KL drifts as β rises, suspect proxy miscalibration and early stop

### RLDK Commands

```bash
# Live monitoring
rldk monitor \
  --from-wandb <entity/project/run> \
  --preset grpo \
  --rules grpo_safe \
  --alerts artifacts/alerts.jsonl

# Offline forensics
rldk forensics log-scan ./runs/my_run
rldk forensics doctor ./runs/my_run
```

## Failure Mode 2: Proxy Exploitation

**What happens**: Pairwise reward models learn a utility function where higher scores mean better. But they're not calibrated probabilities. Optimizing against them finds surface correlates: length, formatting, polite phrases, echoing the prompt.

**Why it happens**: The reward model has approximation error ε(x,y). The policy finds inputs where ε is systematically positive. As the policy moves away from the RM's training distribution, this error grows. Features that correlated with quality in training data (like length or markdown) become exploitable shortcuts.

**How to detect**: 
- Proxy reward rises while task success drops
- Response length increases
- High correlation between reward and spurious features (>0.6 with length is suspicious)

**How to fix**:
- Add length penalty
- Clip reward/advantages
- Refresh proxy on on-policy data every 5–10k steps
- Mix in sparse gold evaluations as anchors

### RLDK Commands

```bash
# Health check with CI gate
rldk reward-health \
  --run runs/my_run/rollouts.jsonl \
  --reference runs/baseline/rollouts.jsonl \
  --gold eval_results.jsonl \
  --gate

# Compare reward models for drift
rldk reward reward-drift RM_A RM_B --prompts prompts.jsonl
```

## Failure Mode 3: Policy vs. Baseline Imbalance

**What happens**: In critic-based PPO, if the value function lags, advantages bias high. If it chases noise, variance explodes. In GRPO, small or heterogeneous groups make advantages noisy.

**Why it happens**: The advantage function measures how much better an action is than average. If your baseline (critic or group mean) is wrong, your gradient estimates are wrong. This creates a feedback loop between policy and value updates.

**How to detect**: Track the step norm ratio ρ = |Δθ_policy| / |Δφ_baseline|. Compare to a healthy baseline. Sustained ρ much larger than baseline suggests the policy is dominating. Also watch group advantage std in GRPO.

**How to fix**:
- Adjust learning rate ratio (often value LR = 2-5× policy LR)
- Add value loss clipping or extra value epochs
- Lower GAE λ (e.g., 0.95 → 0.90)
- Regularize group statistics

### RLDK Commands

```bash
# Live monitoring with ratio guards
rldk monitor --stream artifacts/grpo_run.jsonl \
  --preset grpo --rules grpo_safe \
  --alerts artifacts/grpo_alerts.jsonl
```

## Failure Mode 4: Distribution Shift & Proxy Miscalibration

**What happens**: Your reward model was trained on earlier distributions (SFT, initial policy, mixtures). As the policy moves, the RM's predictions become less reliable even if KL to reference looks fine.

**Why it happens**: This is the core tension in online RL. The RM's training distribution and the policy's current distribution diverge. The KL constraint bounds distance to initialization, but the RM might have been trained on a different mix. Poorly calibrated models don't express uncertainty well, so they give confident but wrong scores.

**How to detect**:
- Proxy reward climbs (0.7 → 0.9 → 1.1) while gold metrics drop (78% → 74% → 69%)
- Proxy score variance compresses
- Correlation with gold evaluations declines

**How to fix**:
- Refresh proxy with on-policy data every 5–10k steps
- Early stop when divergence exceeds guardrails (e.g., proxy up 10%, gold down 5%)

### RLDK Commands

```bash
# Detect overoptimization and correlation decay
rldk reward-health \
  --run runs/current/rollouts.jsonl \
  --reference runs/baseline/rollouts.jsonl \
  --gold eval_results.jsonl

# Quantify drift between models
rldk reward reward-drift ./rmA ./rmB --prompts prompts.jsonl
```

## Failure Mode 5: Entropy Collapse & Mode Seeking

**What happens**: With strong reward pressure and weak exploration, policies concentrate mass on a few high-scoring modes. Entropy falls across prompt categories, yet proxy reward climbs.

**Why it happens**: Think of entropy as measuring the breadth of your policy's distribution. High entropy means spreading across many responses. Low entropy means concentrating on few.

If β is too small relative to reward scale, the policy exploits: it finds the highest-reward response per prompt and puts all probability there. The KL penalty is too weak to prevent this.

**Why we care**: Low entropy means no exploration (can't discover better modes), brittleness (small prompt changes cause mode switches), and silent exploitation (overfit to proxy shortcuts).

**How to detect**: Per-token entropy sustained at ≤1.2 nats broadly. Always compare to your baseline.

**How to fix**:
- Increase β (stronger pull toward higher-entropy reference)
- Add entropy bonus α·H (α ~ 0.005–0.02)
- Diversify prompts/state space

### RLDK Commands

```bash
# Strict monitoring with tighter entropy guards
rldk monitor --stream artifacts/run.jsonl \
  --rules grpo_strict --preset grpo
```

## Failure Mode 6: Non-Deterministic Training

**Why it matters**: Without determinism, small improvements are indistinguishable from noise. You need 3–5× more runs to tell what works.

**Cost**: If you want to detect a 5% improvement with 95% confidence and your training std is 3%, you need ~1.4 runs. If std is 8% due to non-determinism, you need ~10 runs. At thousands of GPU-hours per run, this kills iteration speed.

### Lockdown Checklist

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Seed dataloader workers and keep identical batch/reduction orders across ranks.

### RLDK Commands

```bash
# Audit determinism
rldk forensics env-audit ./my_repo_or_run

# Verify training stability
rldk check-determinism \
  --cmd "python train.py --config config.yaml" \
  --compare kl_mean,reward_mean,entropy_mean \
  --replicas 3 --steps 1000,5000,10000
```

## When Monitoring Isn't Enough

Metrics can't fully capture semantics:
- Human-sounding fluff at normal lengths (subtle exploitation)
- Category-specific collapse (e.g., code only)
- Safety regressions needing red team evals

RLDK provides observability. You still need task-specific eval suites, red teaming, and manual inspection.

## Offline Methods: How Failures Translate

| Online RL Failure | Offline/DPO Analog | Detection Strategy |
|-------------------|--------------------|--------------------|
| Proxy exploitation | Overfitting to preference biases | Track train vs. gold eval divergence |
| Distribution shift | Preference data ≠ deployment | Monitor OOD detection on new prompts |
| Entropy collapse | Concentrating on safe modes | Measure output diversity vs. baseline |
| KL drift | Implicit regularization weakens | Track β-scaled log-ratio norms |
| Proxy miscalibration | Noisy/inconsistent labels | Validate with held-out human evals |
| Non-determinism | Same as online | Same protocol |

Offline methods change when and how problems manifest, but don't eliminate them.

## Practical Workflow

### Pre-flight

```bash
rldk check-determinism --cmd "python train.py --steps 1000" \
  --compare kl_mean,reward_mean,entropy_mean --replicas 2
rldk forensics env-audit --repo-dir . --output-dir reports/
```

### Live

```bash
rldk monitor --from-wandb org/project/run-id \
  --preset grpo \
  --rules kl_spike,kl_drift,entropy_floor,gradient_spike \
  --alert-webhook https://your-slack-webhook
```

### Post-mortem

```bash
rldk forensics doctor --run-dir runs/YYYY-MM-DD/
rldk reward-health \
  --run runs/YYYY-MM-DD/rollouts.jsonl \
  --gold eval_results.jsonl \
  --output-dir reports/reward_health/
```

### Regression hunting

```bash
rldk bisect \
  --good origin/main~15 --bad HEAD \
  --cmd "python train.py --steps 2000 --output metrics.jsonl" \
  --metric reward_mean --cmp "> 0.87" --window 100
```

## When to Stop vs. Adjust

**Stop now if**:
- KL >3× target for >1k steps
- Broad entropy <1.2 nats
- High severity overoptimization
- Policy:baseline step norm ratio far beyond baseline for thousands of steps

**Adjust/continue if**:
- Mild KL oscillation (retune controller/ε)
- Slow entropy decline (add small entropy bonus, increase β slightly)
- Moderate length bias (add weak length penalty)

**Rule of thumb**: Early in a long run, roll back on first credible anomaly. Late in a run, finish for checkpointing, then restart from last healthy point with corrected settings.

## Getting Started

```bash
pip install git+https://github.com/adityachallapally/rldk.git
```

### Quick Start

```bash
# Pre-flight
rldk check-determinism \
  --cmd "python train.py --steps 1000" \
  --compare reward_mean,kl_mean,entropy_mean \
  --replicas 2

# Live monitor
rldk monitor \
  --from-wandb your-org/your-project/run-id \
  --preset grpo

# Post-mortem
rldk forensics doctor --run-dir runs/YYYY-MM-DD/
rldk reward-health \
  --run runs/YYYY-MM-DD/rollouts.jsonl \
  --output-dir reports/reward_health/
```

## Bottom Line

All large-scale RL optimizes an approximation under constraints. Whether online (PPO/GRPO/RLHF) or offline (DPO/IPO/KTO), you're working with learned reward models, preference fits, or environment signals. Failures rarely crash. They degrade silently by exploiting proxy weaknesses, drifting distributions, or collapsing exploration.

The cure is disciplined observability: KL, entropy, step norms, proxy vs. gold correlation, and determinism. Watch live and audit offline. RLDK won't tune your run for you, but it will surface weak signals early enough that you can.
