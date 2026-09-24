"""Atomic replacement resilient to short-lived Windows reader locks."""

from pathlib import Path
from time import sleep


def replace_file(temporary: Path, destination: Path) -> None:
    """Retry permission-denied replacements for at most 2.75 seconds.

    The old destination and complete temporary file survive a permanent failure.
    Other I/O errors propagate immediately.
    """
    for attempt in range(11):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 10:
                raise
            sleep(.05 * (attempt + 1))
