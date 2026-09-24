"""Check actual framework source packaging and GPU configuration without launching Docker."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from mhagenta import Orchestrator
from mha_exp_level2_cr.exp2_6 import online_runner
from mha_exp_level2_cr.exp2_6.online_learning import LearningConfig
from dataclasses import asdict


def test_build_latency_does_not_consume_runtime_budget(monkeypatch, tmp_path):
    """Even an expired build timestamp preserves the environment's full lifetime."""
    from mhagenta.core.orchestrator import AgentEntry, EnvironmentEntry, BuildSpec
    from mha_exp_level2_cr.exp2_6.online_orchestration import OnlineOrchestrator
    captured = []
    monkeypatch.setattr(Orchestrator, "_docker_build_runtime", lambda self, spec, out, params, entry, rebuild:
                        captured.append((spec, params)))
    orc = object.__new__(OnlineOrchestrator)
    spec = BuildSpec(image_tag="test", display_name="test", launcher_src=tmp_path / "launcher.py",
                     start_script_src=tmp_path / "start.sh", params_filename="params", runtime_objects=(),
                     extra_runtime_sources=())
    env = EnvironmentEntry(kwargs={"exec_duration": 3690}, env_id="env", address={})
    orc._docker_build_runtime(spec, tmp_path, {"exec_duration": -100}, env, True)
    assert captured[-1][1]["exec_duration"] == 3690
    agent = AgentEntry(kwargs={}, agent_id="agent")
    orc._docker_build_runtime(spec, tmp_path, {"exec_duration": 3600}, agent, True)
    launcher, params = captured[-1]
    assert launcher.launcher_src.name == "agent_launcher.py" and launcher.launcher_src.is_file()
    assert params["exec_duration"] == 3600


def test_gpu_topology_and_packaged_imports(tmp_path, monkeypatch):
    """Agent and environment receive resolvable sibling imports and local framework code."""
    class Capture:
        INFO = 20

        def __init__(self, **kwargs):
            self.config = kwargs

        def add_agent(self, **kwargs):
            self.agent = kwargs

        def add_environment(self, **kwargs):
            self.environment = kwargs

    monkeypatch.setattr(online_runner, "Orchestrator", Capture)
    config = {"identity": "a" * 64, "run": 19, "device": "cuda", "gpu_id": 0,
              "duration_seconds": 3600, "learning": asdict(LearningConfig())}
    orc, _, _ = online_runner.build_orchestrator(config, tmp_path)
    assert orc.config["gpu_device_ids"] == [0] and orc.config["exec_duration"] == 3600
    assert orc.environment["gpu_device_ids"] == "none" and orc.environment["exec_duration"] == 3630
    assert "torch==2.14.0" in orc.agent["requirements_path"].read_text()
    copier = object.__new__(Orchestrator)
    for role in ("agent", "environment"):
        entry = getattr(orc, role)
        destination = tmp_path / f"packaged-{role}"
        destination.mkdir()
        modules = [entry["base"]] if role == "environment" else [entry[key] for key in
            ("perceptors", "actuators", "ll_reasoners", "hl_reasoners", "knowledge", "memory", "learners", "goal_graphs")]
        copied = {}
        copier._copy_runtime_python_modules(modules, destination, copied)
        copier._copy_explicit_runtime_sources(entry["extra_runtime_sources"], destination, copied)
        if role == "environment":
            shutil.move(str(destination / "exp2_5"), str(destination / "mha_exp_level2_cr" / "exp2_5"))
        code = ("import sys; from pathlib import Path; "
                "from mha_exp_level2_cr.exp2_6 import online_runtime, online_environment; "
                "assert Path(online_runtime.__file__).is_relative_to(Path(sys.argv[1])); "
                "from mha_exp_level2_cr.exp2_6.online_policy import load_basics; import torch; "
                "assert set(load_basics(torch)) == {'explore', 'navigate_to'}; "
                "from mha_exp_level2_cr.exp2_5.grounding import resolve_active_grounding_bundle; "
                "resolve_active_grounding_bundle()")
        environment = {**os.environ, "PYTHONPATH": str(destination) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        subprocess.run([sys.executable, "-c", code, str(destination)], env=environment, check=True, capture_output=True)
