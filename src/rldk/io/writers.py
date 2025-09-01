"""Writers for reports and visualizations."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Union


def mkdir_reports() -> Path:
    """Create rldk_reports directory and return path."""
    reports_dir = Path("rldk_reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def write_json(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write report dictionary to JSON file."""
    path = Path(path)
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def write_png(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save matplotlib figure as PNG."""
    path = Path(path)
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path, dpi=150, bbox_inches='tight')
