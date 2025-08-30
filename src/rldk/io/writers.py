"""Report writers for diff and determinism analysis."""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ..diff import DivergenceReport
from ..determinism import DeterminismReport


def write_diff_report(report: DivergenceReport, output_dir: Path) -> None:
    """Write diff report to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diff_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Divergence Analysis Report\n\n")
        
        if report.diverged:
            f.write(f"## 🚨 Divergence Detected\n\n")
            f.write(f"**First divergence at step:** {report.first_step}\n\n")
            f.write(f"**Tripped signals:** {', '.join(report.tripped_signals)}\n\n")
            
            f.write("## 📊 Analysis\n\n")
            f.write("The runs have diverged significantly. Here are the most likely causes:\n\n")
            f.write("1. **Learning rate changes** - Sudden spikes in learning rate can cause instability\n")
            f.write("2. **Reward scaling issues** - Inconsistent reward normalization between runs\n")
            f.write("3. **Random seed differences** - Different initialization or sampling\n\n")
            
            if report.notes:
                f.write("## 📝 Additional Notes\n\n")
                for note in report.notes:
                    f.write(f"- {note}\n")
                f.write("\n")
            
            # Check if trace.json exists for local traces
            trace_file = output_dir / "trace.json"
            if trace_file.exists():
                f.write("## 🔍 Local Traces\n\n")
                f.write(f"Detailed traces available at: `{trace_file}`\n\n")
        else:
            f.write("## ✅ No Divergence Detected\n\n")
            f.write("The runs appear to be consistent within the specified tolerance.\n\n")
        
        f.write("## 📈 Events CSV\n\n")
        f.write(f"Detailed divergence events saved to: `{report.events_csv_path}`\n")


def write_determinism_report(report: DeterminismReport, output_dir: Path) -> None:
    """Write determinism report to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "determinism_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Determinism Check Report\n\n")
        
        if report.passed:
            f.write("## ✅ Determinism Check Passed\n\n")
            f.write("The training run appears to be deterministic.\n\n")
        else:
            f.write("## 🚨 Determinism Issues Found\n\n")
            f.write("The training run shows non-deterministic behavior.\n\n")
        
        f.write("## ⚙️ Enforced Settings\n\n")
        for key, value in report.enforced_settings.items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")
        
        if report.mismatches:
            f.write("## 📊 Metric Mismatches\n\n")
            for mismatch in report.mismatches:
                f.write(f"- **{mismatch['metric']}:** {mismatch['details']}\n")
            f.write("\n")
        
        if report.culprit:
            f.write(f"## 🎯 Identified Culprit\n\n")
            f.write(f"**Operation:** {report.culprit}\n\n")
        
        if report.fixes:
            f.write("## 🔧 Recommended Fixes\n\n")
            for fix in report.fixes:
                f.write(f"- {fix}\n")
            f.write("\n")
        
        f.write("## 📁 Report Location\n\n")
        f.write(f"Full report saved to: `{report.report_path}`\n")
