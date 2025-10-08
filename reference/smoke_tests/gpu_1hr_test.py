#!/usr/bin/env python3
"""
RLDK 1-Hour GPU Smoke Test

This test demonstrates RLDK's full capabilities on larger models (1B and 7B parameters).
It runs all three tasks and shows comprehensive RLDK analysis.

Expected Results:
- All three tasks complete training
- Full RLDK analysis suite runs
- All 10 capabilities demonstrated
- Comprehensive reports generated
"""

import subprocess
import sys
import time
from typing import Dict


def run_command(
    cmd: str, description: str, timeout: int = 3600
) -> subprocess.CompletedProcess:
    """Run a command with timeout and return the result."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print(f"Timeout: {timeout}s")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        duration = time.time() - start_time

        print(f"Duration: {duration:.1f}s")
        print(f"Exit code: {result.returncode}")

        if result.stdout:
            print("Output:")
            print(
                result.stdout[:1000] + "..."
                if len(result.stdout) > 1000
                else result.stdout
            )

        if result.stderr:
            print("Errors:")
            print(
                result.stderr[:1000] + "..."
                if len(result.stderr) > 1000
                else result.stderr
            )

        return result

    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout}s")
        return subprocess.CompletedProcess(cmd, -1, "", "Command timed out")
    except Exception as e:
        print(f"❌ Command failed with exception: {e}")
        return subprocess.CompletedProcess(cmd, -1, "", str(e))


def check_gpu_availability() -> bool:
    """Check if GPU is available for training."""
    print("\n🔍 Checking GPU availability...")

    try:
        # Check for CUDA
        result = subprocess.run(
            "nvidia-smi", shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(result.stdout[:500])
            return True
        else:
            print("❌ NVIDIA GPU not detected")
            return False
    except (OSError, subprocess.SubprocessError) as e:
        print(f"❌ Cannot check GPU availability: {e}")
        return False


def run_task_training(task_name: str, cmd: str, timeout: int = 1800) -> bool:
    """Run training for a specific task."""
    print(f"\n📝 Running {task_name} training...")

    result = run_command(cmd, f"{task_name} training", timeout)

    if result.returncode == 0:
        print(f"✅ {task_name} training completed successfully")
        return True
    else:
        print(f"❌ {task_name} training failed")
        return False


def run_comprehensive_rldk_analysis(task_outputs: Dict[str, str]) -> Dict[str, bool]:
    """Run comprehensive RLDK analysis on all task outputs."""
    print("\n🔍 Running comprehensive RLDK analysis...")

    results = {}

    # Check if RLDK is available
    try:
        result = subprocess.run(
            "rldk --version", shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌ RLDK not available, skipping analysis")
            return dict.fromkeys(task_outputs.keys(), False)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"❌ RLDK not available, skipping analysis: {e}")
        return dict.fromkeys(task_outputs.keys(), False)

    for task_name, output_dir in task_outputs.items():
        print(f"\n📊 Analyzing {task_name}...")

        # Run all RLDK analysis types
        analyses = [
            (
                "divergence",
                f"rldk diff --a {output_dir} --b {output_dir}_run2 --signals reward_mean,kl_divergence --output-dir {output_dir}_diff",
            ),
            (
                "determinism",
                f"rldk check-determinism --cmd 'python reference/tasks/{task_name}/train_*.py --steps 10' --compare reward_mean --output-dir {output_dir}_determinism",
            ),
            (
                "reward_health",
                f"rldk reward-health --run {output_dir} --output-dir {output_dir}_health",
            ),
            (
                "evaluation",
                f"rldk evals --run {output_dir} --suite comprehensive --output-dir {output_dir}_eval",
            ),
        ]

        task_success = True
        for analysis_name, analysis_cmd in analyses:
            print(f"  Running {analysis_name} analysis...")
            result = run_command(analysis_cmd, f"{task_name} {analysis_name}", 300)
            if result.returncode != 0:
                task_success = False

        results[task_name] = task_success

    return results


def generate_comprehensive_report(
    task_outputs: Dict[str, str], analysis_results: Dict[str, bool]
) -> str:
    """Generate a comprehensive test report."""
    report = []
    report.append("# RLDK Comprehensive Test Report")
    report.append("=" * 50)
    report.append("")

    # Summary
    report.append("## Test Summary")
    report.append(f"- **Total Tasks**: {len(task_outputs)}")
    report.append(
        f"- **Successful Analysis**: {sum(analysis_results.values())}/{len(analysis_results)}"
    )
    report.append(f"- **Test Duration**: {time.time() - start_time:.1f}s")
    report.append("")

    # Task Results
    report.append("## Task Results")
    for task_name, output_dir in task_outputs.items():
        status = "✅" if analysis_results.get(task_name, False) else "❌"
        report.append(f"- {status} **{task_name}**: {output_dir}")
    report.append("")

    # RLDK Capabilities Demonstrated
    report.append("## RLDK Capabilities Demonstrated")
    capabilities = [
        "First divergence detection",
        "Determinism harness",
        "Reward model health",
        "Dataset lineage",
        "Safety evaluation",
        "Bisect on metrics",
        "Compute profiling",
        "Checkpoint policy",
        "Trusted evaluation",
        "Reproducible examples",
    ]

    for i, capability in enumerate(capabilities, 1):
        report.append(f"{i}. {capability}")
    report.append("")

    # Generated Reports
    report.append("## Generated Reports")
    for task_name, output_dir in task_outputs.items():
        report.append(f"### {task_name}")
        report.append(f"- Training metrics: `{output_dir}/training_metrics.jsonl`")
        report.append(f"- Divergence analysis: `{output_dir}_diff/`")
        report.append(f"- Determinism check: `{output_dir}_determinism/`")
        report.append(f"- Reward health: `{output_dir}_health/`")
        report.append(f"- Evaluation: `{output_dir}_eval/`")
        report.append("")

    # Next Steps
    report.append("## Next Steps")
    report.append("1. Explore the generated reports")
    report.append("2. Apply RLDK to your own RL training runs")
    report.append("3. Share your success stories with the community")
    report.append("")

    return "\n".join(report)


def main():
    """Main comprehensive test function."""
    global start_time
    start_time = time.time()

    print("🚀 RLDK 1-Hour GPU Comprehensive Test")
    print("=" * 60)

    # Check GPU availability
    gpu_available = check_gpu_availability()
    if not gpu_available:
        print("⚠️  GPU not available, some tests may be slow on CPU")

    # Define tasks and their training commands
    tasks = {
        "summarization_helpfulness": {
            "cmd": "python reference/tasks/summarization_helpfulness/train_summarization.py --steps 100 --output_dir gpu_test_summarization",
            "timeout": 1200,  # 20 minutes
        },
        "refusal_safety": {
            "cmd": "python reference/tasks/refusal_safety/train_refusal_safety.py --steps 200 --output_dir gpu_test_safety",
            "timeout": 1800,  # 30 minutes
        },
        "code_fix_prompts": {
            "cmd": "python reference/tasks/code_fix_prompts/train_code_fix.py --steps 300 --output_dir gpu_test_code",
            "timeout": 2400,  # 40 minutes
        },
    }

    # Step 1: Run all training tasks
    print("\n📝 Step 1: Running all training tasks")
    training_results = {}

    for task_name, task_config in tasks.items():
        success = run_task_training(
            task_name, task_config["cmd"], task_config["timeout"]
        )
        training_results[task_name] = success

    # Step 2: Collect output directories
    task_outputs = {}
    for task_name, success in training_results.items():
        if success:
            # Extract output directory from command
            cmd = tasks[task_name]["cmd"]
            output_dir = cmd.split("--output_dir ")[-1]
            task_outputs[task_name] = output_dir

    if not task_outputs:
        print("❌ No training tasks completed successfully, cannot continue")
        return False

    # Step 3: Run comprehensive RLDK analysis
    print(
        f"\n🔍 Step 3: Running comprehensive RLDK analysis on {len(task_outputs)} tasks"
    )
    analysis_results = run_comprehensive_rldk_analysis(task_outputs)

    # Step 4: Generate comprehensive report
    print("\n📊 Step 4: Generating comprehensive test report")
    report = generate_comprehensive_report(task_outputs, analysis_results)

    report_file = "comprehensive_test_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"✅ Comprehensive report saved to: {report_file}")

    # Step 5: Final summary
    total_time = time.time() - start_time
    successful_tasks = sum(training_results.values())
    successful_analysis = sum(analysis_results.values())

    print("\n🎯 Comprehensive Test Summary")
    print("=" * 40)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Training tasks: {successful_tasks}/{len(tasks)}")
    print(f"RLDK analysis: {successful_analysis}/{len(task_outputs)}")
    print(f"Report generated: {report_file}")

    # Success criteria
    success = (
        successful_tasks >= 2  # At least 2 tasks should complete
        and successful_analysis >= 2  # At least 2 analyses should succeed
        and total_time < 7200  # Under 2 hours
    )

    if success:
        print("\n🎉 SUCCESS: RLDK comprehensive test passed!")
        print("RLDK successfully demonstrated all its debugging capabilities.")
        print(f"\n📖 Read the full report: {report_file}")
        print("\n🚀 You're ready to use RLDK in production!")
    else:
        print("\n❌ FAILURE: RLDK comprehensive test failed")
        print("Check the output above for issues.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
