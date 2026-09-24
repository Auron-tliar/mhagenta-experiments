"""CLI entry points for experiment 2-5-BW.

The runtime imports this package while restoring agent and environment objects,
so the host-only runner must remain lazy.
"""

from typing import Any

__all__ = ["run_batch", "run_matched_batch"]


def __getattr__(name: str) -> Any:
    """Load host-side batch entry points only when explicitly requested."""

    if name not in __all__:
        raise AttributeError(name)
    from . import runner

    return getattr(runner, name)
