"""Experiment 2-6-CR package with a container-safe lazy entry point."""

from __future__ import annotations

from typing import Any


def run_batch(*args: Any, **kwargs: Any) -> None:
    """Load and run the experiment without eager cross-image imports."""

    from .runner import run_batch as _run_batch

    _run_batch(*args, **kwargs)

__all__ = ["run_batch"]
