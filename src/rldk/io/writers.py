"""Report writers for diff and determinism analysis."""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ..diff.first_divergence import DivergenceReport
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
        
        f.write("## ⚙️ RNG Settings\n\n")
        for key, value in report.rng_map.items():
            f.write(f"- **{key}:** {value}\n")
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
