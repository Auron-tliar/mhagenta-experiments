from typing import Any


def run_batch(*args: Any, **kwargs: Any) -> None:
    """Run the current runtime AchieveOn treatment through the standard CLI."""
    from ..exp2_6_direct.batch import run_batch as run

    run(*args, **kwargs)

__all__ = ["run_batch"]
