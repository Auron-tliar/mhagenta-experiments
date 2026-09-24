"""Keep cold image builds from consuming startup and execution time."""

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path

from mhagenta import Orchestrator
from mhagenta.core.orchestrator import AgentEntry, BuildSpec, EnvironmentEntry

from mha_exp_level2_cr.exp2_5 import runner
from mha_exp_level2_cr.exp2_5.orchestration import ContainerStartOrchestrator


def test_cold_build_preserves_declared_budgets(monkeypatch, tmp_path: Path) -> None:
    """Expired build timestamps neither remove startup time nor shorten the environment."""

    captured = []
    monkeypatch.setattr(
        Orchestrator, "_docker_build_runtime",
        lambda self, spec, out, params, entry, rebuild: captured.append((spec, params)),
    )
    orchestrator = object.__new__(ContainerStartOrchestrator)
    orchestrator._agent_start_delay = runner.STARTUP_DELAY
    spec = BuildSpec(
        image_tag="test", display_name="test", launcher_src=tmp_path / "original.py",
        start_script_src=tmp_path / "start.sh", params_filename="params",
        runtime_objects=(), extra_runtime_sources=(),
    )
    environment = EnvironmentEntry(
        kwargs={"exec_duration": runner.DURATION + runner.ENVIRONMENT_OVERRUN + runner.STARTUP_DELAY},
        env_id="environment", address={},
    )
    orchestrator._docker_build_runtime(spec, tmp_path, {"exec_duration": -100}, environment, True)
    assert captured[-1][0] == spec
    assert captured[-1][1]["exec_duration"] == 650.0

    agent = AgentEntry(kwargs={}, agent_id="agent")
    original = {"exec_start_time": 100.0, "exec_duration": runner.DURATION}
    orchestrator._docker_build_runtime(spec, tmp_path, original, agent, True)
    packaged_spec, params = captured[-1]
    assert packaged_spec.launcher_src.is_file()
    assert packaged_spec.launcher_src.name == "agent_launcher.py"
    assert params["container_start_delay"] == 20.0
    assert params["exec_duration"] == 600.0
    assert "container_start_delay" not in original
    assert runner.Orchestrator is ContainerStartOrchestrator


def test_launcher_rebases_expired_start_before_initializing(monkeypatch) -> None:
    """A late container still waits the full allowance and receives all 600 seconds."""

    path = Path(runner.__file__).parent / "launchers" / "agent_launcher.py"
    spec = spec_from_file_location("cr25_startup_launcher", path)
    assert spec is not None and spec.loader is not None
    launcher = module_from_spec(spec)
    spec.loader.exec_module(launcher)
    params = {"agent_id": "old", "exec_start_time": 100.0,
              "exec_duration": 600.0, "container_start_delay": 20.0}
    events = []

    class CaptureRoot:
        """Record constructor inputs and asynchronous lifecycle order."""

        def __init__(self, **kwargs):
            events.append(kwargs)

        async def initialize(self) -> None:
            events.append("initialize")

        async def start(self) -> None:
            events.append("start")

    monkeypatch.setattr(launcher, "open", lambda *args: BytesIO(), raising=False)
    monkeypatch.setattr(launcher.dill, "load", lambda stream: dict(params))
    monkeypatch.setattr(launcher.time, "time", lambda: 1000.0)
    monkeypatch.setattr(launcher, "MHARoot", CaptureRoot)
    monkeypatch.setenv("AGENT_ID", "current")
    asyncio.run(launcher.main())
    assert events == [
        {"agent_id": "current", "exec_start_time": 1020.0, "exec_duration": 600.0},
        "initialize", "start",
    ]
