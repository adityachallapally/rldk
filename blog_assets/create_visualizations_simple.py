import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
FORENSICS_PATH = BASE_DIR / "comprehensive_ppo_forensics_demo" / "comprehensive_analysis.json"
IMAGES_DIR = BASE_DIR / "images"


def load_data():
    alerts_path = ARTIFACT_DIR / "alerts.jsonl"
    run_path = ARTIFACT_DIR / "run.jsonl"

    alerts_df = pd.read_json(alerts_path, lines=True)
    run_df = pd.read_json(run_path, lines=True)

    with FORENSICS_PATH.open("r", encoding="utf-8") as f:
        forensics = json.load(f)

    return alerts_df, run_df, forensics


def plot_kl_progression(kl_df, alerts_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kl_df["step"], kl_df["value"], label="KL divergence", color="#1f77b4", linewidth=2)
    ax.axhline(0.4, color="#ffbf00", linestyle="--", linewidth=1.5, label="Warning threshold (0.4)")
    ax.axhline(0.8, color="#d62728", linestyle="--", linewidth=1.5, label="Critical threshold (0.8)")

    for _, alert in alerts_df.iterrows():
        ax.scatter(alert["step"], alert["kl_value"], color="#d62728" if alert["level"] == "critical" else "#ff7f0e", s=80)
        ax.text(alert["step"], alert["kl_value"] + 0.015, f"{alert['kl_value']:.3f}", ha="center", fontsize=9)

    ax.set_title("KL divergence spike to 0.937 triggered an automatic shutdown")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL divergence")
    ax.set_xlim(kl_df["step"].min(), kl_df["step"].max())
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "kl_spike_progression.png", dpi=150)
    plt.close(fig)


def plot_alert_timeline(alerts_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    levels = {"warning": 1, "critical": 2}
    colors = {"warning": "#ff7f0e", "critical": "#d62728"}

    ax.scatter(alerts_df["timestamp"], [levels.get(l, 0) for l in alerts_df["level"]],
               c=[colors.get(l, "#1f77b4") for l in alerts_df["level"]], s=90)

    for _, alert in alerts_df.iterrows():
        ax.text(alert["timestamp"], levels.get(alert["level"], 0) + 0.05,
                f"step {int(alert['step'])}: {alert['kl_value']:.3f}", rotation=45, ha="left", fontsize=8)

    ax.set_yticks([1, 2])
    ax.set_yticklabels(["warning", "critical"])
    ax.set_title("Real-time KL alerts escalate from warning to critical in 12 minutes")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Alert level")
    ax.grid(axis="y", alpha=0.3)

    stop_alerts = pd.DataFrame(columns=alerts_df.columns)
    if "auto_terminated" in alerts_df.columns:
        mask = alerts_df["auto_terminated"].fillna(False).astype(bool)
        stop_alerts = alerts_df[mask]
    if stop_alerts.empty:
        stop_alerts = alerts_df[alerts_df["level"] == "critical"].tail(1)

    if not stop_alerts.empty:
        final_alert = stop_alerts.iloc[-1]
        ax.axvline(final_alert["timestamp"], color="#2ca02c", linestyle=":", linewidth=1.5,
                   label="Auto-termination (step {})".format(int(final_alert["step"])))
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "alert_timeline.png", dpi=150)
    plt.close(fig)


def plot_health_scores(forensics):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Overall", "Stability", "Convergence"]
    values = [forensics["overall_health_score"],
              forensics["training_stability_score"],
              forensics["convergence_quality_score"]]

    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0, 1.05)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    ax.set_title("Forensic health scores confirm convergence risk despite stability")
    ax.set_ylabel("Health score")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "health_dashboard.png", dpi=150)
    plt.close(fig)


def plot_reward_and_gradients(reward_df, grad_df, kl_df):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(reward_df["step"], reward_df["value"], color="#2ca02c", label="Reward", linewidth=2)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Reward", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")

    reward_series = reward_df["value"]
    max_abs = reward_series.abs().max()
    if max_abs and max_abs != 0:
        normalized_rewards = reward_series / max_abs
    else:
        normalized_rewards = reward_series.copy()
    ax1.fill_between(reward_df["step"], normalized_rewards * reward_series.max(), color="#2ca02c", alpha=0.08,
                     label="Normalized reward baseline")

    ax2 = ax1.twinx()
    ax2.plot(grad_df["step"], grad_df["value"], color="#9467bd", label="Grad norm", linewidth=2)
    ax2.set_ylabel("Gradient norm", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    ax3.plot(kl_df["step"], kl_df["value"], color="#1f77b4", linestyle="--", label="KL", linewidth=1.5)
    ax3.set_ylabel("KL", color="#1f77b4")
    ax3.tick_params(axis="y", labelcolor="#1f77b4")

    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    final_step = reward_df.iloc[-1]
    ax1.text(final_step["step"], final_step["value"] + 0.05,
             f"Normalized reward={normalized_rewards.iloc[-1]:.2f}", color="#2ca02c", fontsize=9)

    ax1.set_title("Reward and gradient behaviour leading into KL-triggered shutdown")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "reward_gradient_diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    alerts_df, run_df, forensics = load_data()

    kl_df = run_df[run_df["name"] == "kl"].sort_values("step")
    reward_df = run_df[run_df["name"] == "reward"].sort_values("step")
    grad_df = run_df[run_df["name"] == "grad_norm"].sort_values("step")

    plot_kl_progression(kl_df, alerts_df)
    plot_alert_timeline(alerts_df)
    plot_health_scores(forensics)
    plot_reward_and_gradients(reward_df, grad_df, kl_df)

    print("Visualizations saved to", IMAGES_DIR)


if __name__ == "__main__":
    main()
