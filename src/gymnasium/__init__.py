"""Lightweight gymnasium stub used in environments without the real dependency."""

from typing import Any


def make(*_args: Any, **_kwargs: Any) -> Any:
    """Placeholder factory mirroring gymnasium.make.

    The real gymnasium package is optional for RLDK and isn't required for the unit
    suite. This stub allows tests to patch ``gymnasium.make`` without importing the
    full dependency.
    """

    raise ModuleNotFoundError(
        "gymnasium is not installed; this is a compatibility stub"
    )


__all__ = ["make"]
