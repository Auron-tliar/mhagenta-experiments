"""Preserve the startup allowance and execution budgets across cold image builds."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from docker.models.images import Image
from mhagenta import Orchestrator
from mhagenta.core.orchestrator import AgentEntry, BuildSpec, Entry


class ContainerStartOrchestrator(Orchestrator):
    """Rebase 1.4.12's build-time timestamps within this experiment only."""

    def _docker_build_runtime(
        self, spec: BuildSpec, out_dir: Path, params: dict[str, Any],
        entry: Entry, rebuild_image: bool,
    ) -> Image:
        """Package a fresh agent start and preserve the full environment lifetime."""

        params = dict(params)
        if isinstance(entry, AgentEntry):
            spec = replace(spec, launcher_src=Path(__file__).parent / "launchers" / "agent_launcher.py")
            params["container_start_delay"] = self._agent_start_delay
        else:
            # add_environment already includes the configured agent startup allowance.
            params["exec_duration"] = entry.kwargs["exec_duration"]
        return super()._docker_build_runtime(spec, out_dir, params, entry, rebuild_image)
