"""Interrupt handling for resumable 2-5-CR preparation."""

from __future__ import annotations

import signal
from typing import Any, Self


class StopController:
    """Defer the first interrupt until the active episode is committed."""

    def __init__(self) -> None:
        self.requested = False
        self.previous: Any = None

    def __enter__(self) -> Self:
        """Install the handler while retaining the caller's previous handler."""

        self.previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum: int, frame: Any) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True

    def __exit__(self, *args: object) -> None:
        """Restore signal handling when training leaves the controlled scope."""

        signal.signal(signal.SIGINT, self.previous)
