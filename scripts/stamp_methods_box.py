#!/usr/bin/env python3
"""Write the blog Methods and Config box with the current commit metadata."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "docs" / "blog" / "_methods_box.template.md"
OUTPUT_PATH = REPO_ROOT / "docs" / "blog" / "_methods_box.md"


def resolve_commit() -> str:
    """Return the short commit SHA from env or git."""
    env_commit = os.environ.get("RLDK_COMMIT_SHORT")
    if env_commit:
        return env_commit.strip()
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Missing template at {TEMPLATE_PATH}")

    commit = resolve_commit()

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    content = template.format(commit=commit)
    OUTPUT_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
