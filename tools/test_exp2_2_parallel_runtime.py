"""Check both production runners' Docker requests without starting containers."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import docker
import pytest
from mhagenta import Orchestrator

from mha_exp_level2_bw.exp2_2 import runner as bw
from mha_exp_level2_cr.exp2_2 import runner as cr


class LaunchInspected(Exception):
    """Stop a runner after inspecting its actual container launch arguments."""


def test_parallel_gpu_placement_and_resource_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Resolve inherited GPU selections through the standard Docker launch path."""
    client = Mock()
    client.containers.get.side_effect = docker.errors.NotFound("no stale container")
    monkeypatch.setattr(docker, "from_env", lambda: client)
    cleanup: list[tuple[str, str]] = []
    for name in ("cleanup_run_containers", "cleanup_run_images"):
        monkeypatch.setattr(bw, name, lambda agent, env, **_: cleanup.append((agent, env)))
    configured: list[Orchestrator] = []

    def inspect_launch(self: Orchestrator, **kwargs: Any) -> None:
        configured.append(self)
        assert kwargs["local_build"].name == "mhagenta"
        assert self._mas_rmq_close_on_exit is False
        self._log_parser = Mock()
        agent = next(iter(self._agents.values()))
        environment = next(iter(self._environments.values()))
        # Supply the paths and image tags normally assigned by the image builds.
        agent.dir = self._save_dir / agent.agent_id
        environment.dir = self._save_dir / environment.env_id
        agent.image = f"mhagent:{agent.agent_id}"
        environment.image = f"mhagent-env:{environment.env_id}"
        asyncio.run(self._run_agent(agent))
        asyncio.run(self._run_env(environment))
        raise LaunchInspected

    monkeypatch.setattr(Orchestrator, "run", inspect_launch)
    # Explicit GPU selection must also win over an inherited shell setting.
    monkeypatch.setenv("MHA_EXP_GPU_DEVICE", "9")
    for domain, runner, gpu in (("bw", bw, 0), ("cr", cr, 1)):
        with pytest.raises(LaunchInspected):
            runner.run_experiment(0, tmp_path / domain, gpu_device=gpu)

    launches = [call.kwargs for call in client.containers.run.call_args_list]
    assert len(launches) == 4
    assert len({launch["name"] for launch in launches}) == 4
    assert len({launch["image"] for launch in launches}) == 4
    mounts = [next(iter(launch["volumes"])) for launch in launches]
    assert len(set(mounts)) == 4
    for index, gpu in ((0, "0"), (2, "1")):
        request, = launches[index]["device_requests"]
        assert request["DeviceIDs"] == [gpu]
        assert request["Capabilities"] == [["gpu"]]
        assert launches[index + 1]["device_requests"] is None
    assert configured[0]._mas_rmq_exchange_name != configured[1]._mas_rmq_exchange_name
    assert all(pair == bw.runtime_resources(0)[:2] for pair in cleanup)
