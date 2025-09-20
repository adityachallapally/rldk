"""Git bisect wrapper for finding regressions."""

from .bisect import BisectResult, bisect_commits

__all__ = ["bisect_commits", "BisectResult"]
