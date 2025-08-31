"""Report writers for diff and determinism analysis."""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ..diff import DivergenceReport
from ..determinism.check import DeterminismReport


def write_drift_card(report: DivergenceReport, output_dir: Path) -> None:
    """Write drift card to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_card.md"
    
    with open(report_path, 'w') as f:
        f.write("# Drift Detection Card\n\n")
        
        if report.diverged:
            f.write(f"## 🚨 Drift Detected\n\n")
            f.write(f"**First divergence at step:** {report.first_step}\n\n")
            f.write(f"**Tripped signals:** {', '.join(report.tripped_signals)}\n\n")
            
            f.write("## 📊 Analysis\n\n")
            f.write("The runs have diverged significantly. Here are the most likely causes:\n\n")
            
            for cause in report.suspected_causes:
                f.write(f"- {cause}\n")
            f.write("\n")
            
            if not report.details.empty:
                f.write("## 📈 Divergence Details\n\n")
                f.write("| Step | Signal | Z-Score | Run A Value | Run B Value | Consecutive Count |\n")
                f.write("|------|--------|---------|-------------|-------------|-------------------|\n")
                
                for _, row in report.details.head(10).iterrows():  # Show first 10
                    f.write(f"| {row['step']} | {row['signal']} | {row['z_score']:.3f} | {row['run_a_value']:.6f} | {row['run_b_value']:.6f} | {row['consecutive_count']} |\n")
                
                if len(report.details) > 10:
                    f.write(f"\n*... and {len(report.details) - 10} more violations*\n")
        else:
            f.write("## ✅ No Drift Detected\n\n")
            f.write("The runs appear to be consistent within the specified tolerance.\n\n")
        
        f.write("## 📁 Report Location\n\n")
        f.write(f"Full report saved to: `{report_path}`\n")
        
        # Also save details to CSV for further analysis
        if not report.details.empty:
            csv_path = output_dir / "diff_events.csv"
            report.details.to_csv(csv_path, index=False)
            f.write(f"Detailed events saved to: `{csv_path}`\n")
        
        f.write("\n## 🔍 Analysis Parameters\n\n")
        f.write(f"- **Total divergence events:** {len(report.details)}\n")
        if not report.details.empty:
            f.write(f"- **First event step:** {report.details['step'].min()}\n")
            f.write(f"- **Last event step:** {report.details['step'].max()}\n")


def write_determinism_card(report: DeterminismReport, output_dir: Path) -> None:
    """Write determinism card to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "determinism_card.md"
    
    with open(report_path, 'w') as f:
        f.write("# Determinism Check Card\n\n")
        
        if report.passed:
            f.write("## ✅ Determinism Check Passed\n\n")
            f.write("The training run appears to be deterministic.\n\n")
        else:
            f.write("## 🚨 Determinism Issues Found\n\n")
            f.write("The training run shows non-deterministic behavior.\n\n")
        
        f.write("## ⚙️ Environment Settings\n\n")
        f.write("### RNG Settings\n")
        for key, value in report.rng_map.items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")
        
        f.write("### Deterministic Flags Set\n")
        f.write("- **PYTHONHASHSEED:** Set to 42\n")
        f.write("- **OMP_NUM_THREADS:** Set to 1\n")
        f.write("- **MKL_NUM_THREADS:** Set to 1\n")
        f.write("- **NUMEXPR_NUM_THREADS:** Set to 1\n")
        f.write("- **OPENBLAS_NUM_THREADS:** Set to 1\n")
        f.write("- **VECLIB_MAXIMUM_THREADS:** Set to 1\n")
        f.write("- **CUDA_LAUNCH_BLOCKING:** Set to 1\n")
        f.write("- **TORCH_USE_CUDA_DSA:** Set to 1\n")
        if any('cuda' in key.lower() for key in report.rng_map.keys()):
            f.write("- **CUBLAS_WORKSPACE_CONFIG:** Set to :4096:8\n")
        f.write("\n")
        
        f.write("### PyTorch Deterministic Settings Applied\n")
        f.write("- **torch.backends.cudnn.deterministic = True**\n")
        f.write("- **torch.backends.cudnn.benchmark = False**\n")
        f.write("- **torch.use_deterministic_algorithms(True)**\n")
        f.write("- **torch.backends.cuda.matmul.allow_tf32 = False**\n")
        f.write("- **torch.backends.cudnn.allow_tf32 = False**\n")
        f.write("- **torch.manual_seed(42)**\n")
        f.write("- **torch.cuda.manual_seed(42)** (if CUDA available)\n")
        f.write("\n")
        
        f.write("### Random Seeds Set\n")
        f.write("- **Python random.seed(42)**\n")
        f.write("- **NumPy np.random.seed(42)**\n")
        f.write("- **PyTorch torch.manual_seed(42)**\n")
        f.write("\n")
        
        if report.replica_variance:
            f.write("## 📊 Replica Variance\n\n")
            f.write("| Metric | Variance |\n")
            f.write("|--------|----------|\n")
            for metric, var in report.replica_variance.items():
                f.write(f"| {metric} | {var:.6f} |\n")
            f.write("\n")
        
        if report.mismatches:
            f.write("## 🚨 Metric Mismatches\n\n")
            f.write("| Replica | Step | Metric | Reference | Replica | Difference |\n")
            f.write("|---------|------|--------|-----------|---------|------------|\n")
            
            for mismatch in report.mismatches[:10]:  # Show first 10
                if 'step' in mismatch and 'metric' in mismatch:
                    f.write(f"| {mismatch['replica']} | {mismatch['step']} | {mismatch['metric']} | {mismatch['reference_value']:.6f} | {mismatch['replica_value']:.6f} | {mismatch['difference']:.6f} |\n")
                else:
                    f.write(f"| {mismatch['replica']} | - | - | - | - | {mismatch['issue']} |\n")
            
            if len(report.mismatches) > 10:
                f.write(f"\n*... and {len(report.mismatches) - 10} more mismatches*\n")
        
        if report.culprit:
            f.write(f"## 🎯 Identified Culprit\n\n")
            f.write(f"**Operation:** {report.culprit}\n\n")
        
        if report.fixes:
            f.write("## 🔧 Recommended Fixes\n\n")
            for fix in report.fixes:
                f.write(f"- {fix}\n")
            f.write("\n")
        
        f.write("## 📁 Report Location\n\n")
        f.write(f"Full report saved to: `{report_path}`\n")


def write_diff_report(report: DivergenceReport, output_dir: Path) -> None:
    """Write detailed diff report to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diff_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Divergence Analysis Report\n\n")
        
        f.write("## 📊 Summary\n\n")
        f.write(f"- **Diverged:** {'Yes' if report.diverged else 'No'}\n")
        if report.diverged:
            f.write(f"- **First divergence step:** {report.first_step}\n")
            f.write(f"- **Tripped signals:** {', '.join(report.tripped_signals)}\n")
        
        f.write(f"- **Total divergence events:** {len(report.details)}\n")
        f.write(f"- **Signals analyzed:** {', '.join(report.suspected_causes)}\n\n")
        
        if not report.details.empty:
            f.write("## 📈 Detailed Events\n\n")
            f.write("| Step | Signal | Z-Score | Run A Value | Run B Value | Consecutive Count |\n")
            f.write("|------|--------|---------|-------------|-------------|-------------------|\n")
            
            for _, row in report.details.iterrows():
                f.write(f"| {row['step']} | {row['signal']} | {row['z_score']:.3f} | {row['run_a_value']:.6f} | {row['run_b_value']:.6f} | {row['consecutive_count']} |\n")
        
        f.write(f"\n## 📁 Files Generated\n\n")
        f.write(f"- **Diff Report:** `{report_path}`\n")
        f.write(f"- **Drift Card:** `{output_dir}/drift_card.md`\n")
        if not report.details.empty:
            f.write(f"- **Events CSV:** `{output_dir}/diff_events.csv`\n")
