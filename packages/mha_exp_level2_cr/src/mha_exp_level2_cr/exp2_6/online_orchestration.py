"""Keep image-build latency outside this experiment's startup and execution clocks."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from mhagenta import Orchestrator
from mhagenta.core.orchestrator import AgentEntry, BuildSpec, Entry


class OnlineOrchestrator(Orchestrator):
    """Adapt 1.4.12's build-time timestamps without modifying the shared framework."""

    def _docker_build_runtime(self, spec: BuildSpec, out_dir: Path, params: dict[str, Any],
                              entry: Entry, rebuild_image: bool):
        """Restore environment lifetime and set the agent's start at container launch."""
        params = dict(params)
        if isinstance(entry, AgentEntry):
            spec = replace(spec, launcher_src=Path(__file__).parent / "launchers" / "agent_launcher.py")
        else:
            params["exec_duration"] = entry.kwargs["exec_duration"]
        return super()._docker_build_runtime(spec, out_dir, params, entry, rebuild_image)
