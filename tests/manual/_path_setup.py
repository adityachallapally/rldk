"""Utilities for manually invoked legacy test scripts."""

from pathlib import Path
import sys
from typing import Optional


def add_repo_paths(src_subdir: Optional[str] = None) -> Path:
    """Ensure the repository's src directory is on sys.path.

    Args:
        src_subdir: Optional relative path inside ``src`` to append as well.

    Returns:
        The resolved repository root path.
    """
    current_file = Path(__file__).resolve()
    project_root = next(
        (parent for parent in current_file.parents if (parent / "pyproject.toml").exists()),
        current_file.parent,
    )

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    if src_subdir:
        extra_path = src_path / src_subdir
        if str(extra_path) not in sys.path:
            sys.path.insert(0, str(extra_path))

    return project_root


PROJECT_ROOT = add_repo_paths()

__all__ = ["add_repo_paths", "PROJECT_ROOT"]
