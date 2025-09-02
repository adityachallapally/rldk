"""Forensics CLI commands for RL Debug Kit."""

import typer

from rldk.artifacts.ckpt_diff import diff_checkpoints
from rldk.artifacts.env_audit import audit_environment
from rldk.artifacts.log_scan import scan_logs
from rldk.io.writers import write_json, write_png, mkdir_reports
from rldk.io.schemas import (
    validate,
    DeterminismCardV1,
    PPOScanReportV1,
    CkptDiffReportV1,
)

app = typer.Typer(name="forensics", help="Forensics commands for RL training analysis")


@app.command(name="compare-runs")
def compare_runs(
    run_a: str = typer.Argument(..., help="Path to first run directory"),
    run_b: str = typer.Argument(..., help="Path to second run directory"),
):
    """Compare two training runs and identify divergences."""
    try:
        typer.echo("Comparing runs:")
        typer.echo(f"  Run A: {run_a}")
        typer.echo(f"  Run B: {run_b}")

        # Scan both runs
        scan_a = scan_logs(run_a)
        scan_b = scan_logs(run_b)

        # Create comparison report
        comparison = {
            "version": "1",
            "run_a": {"path": run_a, "anomalies": scan_a.get("rules_fired", [])},
            "run_b": {"path": run_b, "anomalies": scan_b.get("rules_fired", [])},
            "earliest_divergent_step": None,
        }

        # Find earliest divergent step if both have step data
        if scan_a.get("earliest_step") and scan_b.get("earliest_step"):
            comparison["earliest_divergent_step"] = min(
                scan_a["earliest_step"], scan_b["earliest_step"]
            )

        # Write report
        mkdir_reports()
        write_json(comparison, "rldk_reports/run_comparison.json")

        typer.echo(
            "\nComparison complete. Report saved to rldk_reports/run_comparison.json"
        )

        # Print summary
        anomalies_a = len(scan_a.get("rules_fired", []))
        anomalies_b = len(scan_b.get("rules_fired", []))
        typer.echo(f"Run A anomalies: {anomalies_a}")
        typer.echo(f"Run B anomalies: {anomalies_b}")

        if comparison["earliest_divergent_step"]:
            typer.echo(
                f"Earliest divergent step: {comparison['earliest_divergent_step']}"
            )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="diff-ckpt")
def diff_ckpt(
    ckpt_a: str = typer.Argument(..., help="Path to first checkpoint"),
    ckpt_b: str = typer.Argument(..., help="Path to second checkpoint"),
):
    """Compare two model checkpoints and identify parameter differences."""
    try:
        typer.echo("Comparing checkpoints:")
        typer.echo(f"  Checkpoint A: {ckpt_a}")
        typer.echo(f"  Checkpoint B: {ckpt_b}")

        # Diff checkpoints
        report = diff_checkpoints(ckpt_a, ckpt_b)

        # Validate report
        validate(CkptDiffReportV1, report)

        # Write report and plot
        mkdir_reports()
        write_json(report, "rldk_reports/ckpt_diff.json")

        # Create bar plot of top movers
        if report["top_movers"]:
            import matplotlib.pyplot as plt

            names = [m["name"] for m in report["top_movers"][:10]]
            l2_norms = [m["l2"] for m in report["top_movers"][:10]]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(names)), l2_norms)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("L2 Norm of Parameter Difference")
            ax.set_title("Top Parameter Changes")
            ax.invert_yaxis()

            write_png(fig, "rldk_reports/ckpt_diff.png")
            plt.close()

        typer.echo(
            "\nCheckpoint diff complete. Report saved to rldk_reports/ckpt_diff.json"
        )

        # Print summary
        summary = report["summary"]
        typer.echo(f"Total parameters: {summary['num_params']}")
        typer.echo(f"Common parameters: {summary['num_common_params']}")
        if summary['num_only_in_a'] > 0:
            typer.echo(f"Only in checkpoint A: {summary['num_only_in_a']}")
        if summary['num_only_in_b'] > 0:
            typer.echo(f"Only in checkpoint B: {summary['num_only_in_b']}")
        typer.echo(f"Average cosine similarity: {summary['avg_cosine']:.4f}")
        typer.echo(
            f"L2 norm percentiles - 5%: {summary['l2_p05']:.6f}, 50%: {summary['l2_p50']:.6f}, 95%: {summary['l2_p95']:.6f}"
        )

        if report["top_movers"]:
            typer.echo("\nTop parameter changes:")
            for i, mover in enumerate(report["top_movers"][:5]):
                note = mover.get('note', '')
                if note:
                    typer.echo(
                        f"  {i+1}. {mover['name']}: L2={mover['l2']:.6f}, cosine={mover['cosine']:.4f} ({note})"
                    )
                else:
                    typer.echo(
                        f"  {i+1}. {mover['name']}: L2={mover['l2']:.6f}, cosine={mover['cosine']:.4f}"
                    )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="env-audit")
def env_audit(
    repo_or_run: str = typer.Argument(..., help="Path to repository or run directory"),
):
    """Audit environment for determinism and reproducibility."""
    try:
        typer.echo(f"Auditing environment for: {repo_or_run}")

        # Run audit
        determinism_card, lock_content = audit_environment(repo_or_run)

        # Validate determinism card
        validate(DeterminismCardV1, determinism_card)

        # Write outputs
        mkdir_reports()
        write_json(determinism_card, "rldk_reports/determinism_card.json")

        with open("rldk.lock", "w") as f:
            f.write(lock_content)

        typer.echo("\nEnvironment audit complete.")
        typer.echo("  Determinism card: rldk_reports/determinism_card.json")
        typer.echo("  Lock file: rldk.lock")

        # Print summary
        flags = determinism_card["flags"]
        typer.echo("\nKey findings:")
        typer.echo(f"  Deterministic: {determinism_card['pass']}")
        typer.echo(f"  CUDNN deterministic: {flags['cudnn_deterministic']}")
        typer.echo(f"  Tokenizers parallelism: {flags['tokenizers_parallelism']}")

        if determinism_card["nondeterminism_hints"]:
            typer.echo(
                f"  Nondeterminism hints: {len(determinism_card['nondeterminism_hints'])}"
            )
            for hint in determinism_card["nondeterminism_hints"][:3]:
                typer.echo(f"    - {hint}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="log-scan")
def log_scan(
    run_or_export: str = typer.Argument(..., help="Path to run or export directory"),
):
    """Scan training logs for PPO anomalies and issues."""
    try:
        typer.echo(f"Scanning logs: {run_or_export}")

        # Scan logs
        report = scan_logs(run_or_export)

        # Validate report
        validate(PPOScanReportV1, report)

        # Write report
        mkdir_reports()
        write_json(report, "rldk_reports/ppo_scan.json")

        typer.echo("\nLog scan complete. Report saved to rldk_reports/ppo_scan.json")

        # Print summary
        rules_fired = report.get("rules_fired", [])
        typer.echo(f"Rules fired: {len(rules_fired)}")

        if rules_fired:
            typer.echo("Anomalies detected:")
            for rule in rules_fired:
                typer.echo(f"  - {rule['rule']}: {rule['description']}")
                if rule.get("step_range"):
                    typer.echo(
                        f"    Steps: {rule['step_range'][0]} to {rule['step_range'][1]}"
                    )
        else:
            typer.echo("No anomalies detected.")

        if report.get("earliest_step"):
            typer.echo(f"Earliest step with data: {report['earliest_step']}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="doctor")
def doctor(
    run_or_repo: str = typer.Argument(..., help="Path to run or repository directory"),
):
    """Run comprehensive diagnostics on a training run or repository."""
    try:
        typer.echo(f"Running diagnostics on: {run_or_repo}")

        # Run env audit
        typer.echo("\n1. Environment audit...")
        determinism_card, lock_content = audit_environment(run_or_repo)

        # Run log scan
        typer.echo("2. Log scan...")
        scan_report = scan_logs(run_or_repo)

        # Write outputs
        mkdir_reports()
        write_json(determinism_card, "rldk_reports/determinism_card.json")
        write_json(scan_report, "rldk_reports/ppo_scan.json")

        with open("rldk.lock", "w") as f:
            f.write(lock_content)

        # Print summary
        typer.echo("\nDiagnostics complete!")
        typer.echo("Files generated:")
        typer.echo("  - rldk_reports/determinism_card.json")
        typer.echo("  - rldk_reports/ppo_scan.json")
        typer.echo("  - rldk.lock")

        # Health summary
        env_healthy = determinism_card["pass"]
        log_healthy = len(scan_report.get("rules_fired", [])) == 0

        if env_healthy and log_healthy:
            typer.echo("\n✅ All systems healthy!")
        else:
            typer.echo("\n⚠️  Issues detected:")
            if not env_healthy:
                typer.echo("  - Environment has nondeterminism issues")
            if not log_healthy:
                typer.echo(
                    f"  - Training logs show {len(scan_report.get('rules_fired', []))} anomalies"
                )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
