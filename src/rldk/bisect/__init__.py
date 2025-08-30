"""Git bisect wrapper for finding regressions."""

from .bisect import bisect_commits, BisectResult

__all__ = ["bisect_commits", "BisectResult"]
