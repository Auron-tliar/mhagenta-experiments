"""Public twenty-run, one-hour Crafter learning batch with non-overwriting outputs."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

import mhagenta
import mha_exp_common
import mha_env_crafter
from mha_exp_common.batch import normalize_runs
from mha_exp_common.defaults import DEFAULT_TORCH_MHAGENTA_VERSION
from mha_exp_common.utils import Seeder

from ..exp2_5.grounding import resolve_active_grounding_bundle
from ..exp2_5.policy import file_sha256
from .online_environment import RUNS, ContinuingEnvironment, episode_seed, initial_state as environment_state
from .online_learning import LearningConfig
from .online_orchestration import OnlineOrchestrator as Orchestrator
from .online_policy import BASIC_HASHES, SKILLS, load_basics, tensor_hash
from .online_runtime import MODULES, initial_states, require, write_json

DURATION = 3600


def source_identity() -> dict:
    """Hash executable sources and input artifacts, including uncommitted edits."""
    package = Path(__file__).resolve().parent
    framework = Path(mhagenta.__file__).resolve().parent
    roots = {"online": package, "baseline": package.parent / "exp2_5", "framework": framework,
             "environment": Path(mha_env_crafter.__file__).resolve().parent,
             "common": Path(mha_exp_common.__file__).resolve().parent}
    identities = {}
    for label, root in roots.items():
        for file in sorted(root.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix in {".py", ".json", ".npz", ".pt", ".txt", ".sh", ".yaml", ".png"}:
                identities[f"{label}/{file.relative_to(root).as_posix()}"] = file_sha256(file)
    return identities


def build_orchestrator(config: dict, output: Path):
    """Package sibling framework sources and expose GPU only to the agent."""
    framework = Path(mhagenta.__file__).resolve().parent
    require('version = "1.4.12"' in (framework.parent / "pyproject.toml").read_text(), "Local MHAgentA 1.4.12 required")
    workspace = Path(__file__).resolve().parents[5]
    require(framework == workspace.parent / "mhagenta" / "mhagenta", "Framework must be the sibling checkout")
    short = config["identity"][:12]
    agent_id, env_id = f"exp_cr26_agent_{short}", f"exp_cr26_env_{short}"
    exchange = f"cr26-{short}"
    orc = Orchestrator(save_dir=output, step_frequency=0.01, control_frequency=0.01,
                       status_frequency=5, agent_start_delay=60, exec_duration=config["duration_seconds"],
                       save_format="json", log_level=Orchestrator.INFO, save_logs=True, no_stdout_logs=False,
                       mas_rmq_uri="localhost:5672", mas_rmq_exchange_name=exchange,
                       state_autosave_interval=30, stop_on_agents_term=True,
                       gpu_device_ids=[config["gpu_id"]] if config["device"] == "cuda" else "none")
    initial = initial_states()
    seeds = Seeder(config["run"])
    init = {"ll_reasoner": {"seed": seeds.ll_reasoner, "duration_seconds": config["duration_seconds"]},
            "hl_reasoner": {"duration_seconds": config["duration_seconds"]},
            "learner": {"seed": seeds.learner, "device": config["device"], "learning": config["learning"]}}
    modules = {name: cls(module_id=name, initial_state=initial[name], init_kwargs=init.get(name, {}),
                         **({"exchange_name": exchange} if name in {"perceptor", "actuator"} else {}))
               for name, cls in MODULES.items()}
    package = Path(__file__).parent
    common = Path(mha_exp_common.__file__).resolve().parent
    orc.add_agent(agent_id=agent_id, perceptors=modules["perceptor"], actuators=modules["actuator"],
                  ll_reasoners=modules["ll_reasoner"], hl_reasoners=modules["hl_reasoner"],
                  knowledge=modules["knowledge"], memory=modules["memory"], learners=modules["learner"],
                  goal_graphs=modules["goal_graph"], requirements_path=package / "requirements-online.txt",
                  extra_runtime_sources=[common, framework])
    # The agent automatically includes exp2_5 through its reused Perceptor.
    # The environment's inherited implementation needs that namespace explicitly.
    install = output / "environment-init.sh"
    install.write_text('set -eu\nmv /agent/exp2_5 /agent/mha_exp_level2_cr/exp2_5\n', encoding="utf-8", newline="\n")
    orc.add_environment(base=ContinuingEnvironment(environment_state(config["run"], agent_id)), env_id=env_id,
                        exec_duration=config["duration_seconds"] + 30, exchange_name=exchange,
                        requirements_path=package.parent / "exp2_5" / "requirements-env.txt",
                        extra_runtime_sources=[package.parent / "exp2_5", common, framework,
                                               Path(mha_env_crafter.__file__).resolve().parent],
                        init_script=install, gpu_device_ids="none")
    return orc, agent_id, env_id


def read_states(output: Path, agent_id: str, env_id: str) -> tuple[dict, dict]:
    """Read only the nine expected final state files, ignoring progress JSON."""
    states = {name: json.loads((output / agent_id / "out" / f"{agent_id}.{name}.json").read_text()) for name in MODULES}
    environment = json.loads((output / env_id / "out" / f"{env_id}.json").read_text())
    return states, environment


def check_results(states: dict, env: dict, config: dict, agent_output: Path | None = None,
                  env_output: Path | None = None) -> dict:
    """Validate execution and replay with live trainers or final-only policy files."""
    errors = []
    skill_outcomes = {}
    def check(condition, message):
        if not condition:
            errors.append(message)
    try:
        check(set(states) == set(MODULES), "missing-module-states")
        check(all(row["failure"] is None for row in states.values()) and env["failure"] is None, "runtime-contract-failure")
        ll, hl, actuator, learner, memory, knowledge, graph = [states[name] for name in
            ("ll_reasoner", "hl_reasoner", "actuator", "learner", "memory", "knowledge", "goal_graph")]
        check(ll["closed"] and hl["closed"] and env["closed"] and env["close_count"] == 1, "unacknowledged-close")
        check(ll["actions"] == actuator["requests"] == actuator["statuses"] == env["native_action_count"] == env["status_count"], "native-action-counts")
        check(ll["observations"] == ll["actions"] + env["resets"] + 1 == knowledge["observations"]
              == states["perceptor"]["observation_count"] == env["observation_count"], "observation-reset-counts")
        check(ll["episode"] == env["episode"] == env["resets"], "episode-counts")
        check(actuator["controls"] == actuator["control_statuses"] == env["controls"] == env["resets"] + 1, "control-counts")
        check(graph["sent"] == graph["received"] == hl["commands"] == hl["replies"] == ll["replies"], "goal-reconciliation")
        check(ll["segments"] == knowledge["segments"] == memory["received"] == memory["delivered"]
              == learner["segments"] == learner["publications"] == ll["model_installs"], "learning-flow-counts")
        check(actuator["pending"] is None and ll["session"] is None and ll["command"] is None
              and ll["waiting_model"] is None and not ll["awaiting_observation"]
              and learner["pending"] is None and memory["pending"] is None and memory["stored"] == 0
              and graph["active"] is None and hl["pending"] is None, "unfinished-work")
        check(env["illegal_count"] == env["lethal_count"] == 0, "prohibited-actions")
        check(learner["hardware"]["torch"].split("+")[0] == "2.14.0", "wrong-torch")
        check(learner["hardware"]["device"] == config["device"], "wrong-learner-device")
        check(hl["execution_seconds"] >= config["duration_seconds"] - 45, "premature-agent-stop")
        check(sum(row["actions"] for row in env["episodes"]) == ll["actions"], "episode-action-sum")
        check(env["terminal_count"] == sum(row["reason"] != "time_limit" for row in env["episodes"]), "episode-terminal-count")
        for index, episode in enumerate(env["episodes"]):
            check(episode["episode"] == index and episode["seed"] == episode_seed(config["run"], index), "episode-seed-order")
            check(0 <= episode["actions"] <= 900, "episode-action-limit")
            check(episode["reason"] in {"death", "diamond", "action_budget", "time_limit"}, "unexpected-episode-end")
            check(episode["reason"] != "action_budget" or episode["actions"] == 900, "incorrect-budget-reset")
            check(episode["reason"] != "death" or episode["inventory"].get("health", 0) <= 0, "incorrect-death-reset")
            check(episode["reason"] != "diamond" or episode["inventory"].get("diamond", 0) > 0
                  and episode["achievements"].get("collect_diamond", 0) > 0, "incorrect-diamond-reset")
        if agent_output is not None:
            import torch
            files = list((agent_output / "experience").glob("segment-*.pt"))
            check(len(files) == ll["segments"], "missing-experience-files")
            totals = {skill: 0 for skill in SKILLS}
            excluded = 0
            for file in files:
                segment = torch.load(file, map_location="cpu", weights_only=False)
                rows = segment["rows"]
                category = f"{segment['skill']}:{segment['resource']}:{segment['mode']}"
                aggregate = skill_outcomes.setdefault(category, {"attempts": 0, "successes": 0, "actions": 0, "failures": {}})
                aggregate["attempts"] += 1
                aggregate["successes"] += int(segment["success"])
                aggregate["actions"] += len(rows)
                if not segment["success"]:
                    aggregate["failures"][segment["reason"]] = aggregate["failures"].get(segment["reason"], 0) + 1
                check(len(rows) == segment["steps"] <= 32 and rows[-1]["terminal"]
                      and not any(row["terminal"] for row in rows[:-1]), "invalid-skill-boundary")
                check(all(row["mask"][row["action"]] for row in rows), "illegal-replay-action")
                check(all(np_equal(left["next"], right["state"]) for left, right in zip(rows, rows[1:])), "noncontiguous-replay")
                if segment["mode"] == "probe":
                    excluded += 1
                else:
                    totals[segment["skill"]] += len(rows)
            check(excluded == learner["excluded_probes"], "probe-replay-leak")
            check(all(totals[skill] == learner["skills"][skill]["transitions"] for skill in SKILLS), "replay-transition-counts")
            for skill in SKILLS:
                policy = agent_output / f"{skill}-policy.pt"
                if policy.is_file():
                    saved = torch.load(policy, map_location="cpu", weights_only=True)
                    check(set(saved) == {"model", "summary"}, "invalid-final-policy-fields")
                else:
                    saved = torch.load(agent_output / f"{skill}-trainer.pt", map_location="cpu", weights_only=False)
                check(saved["summary"] == learner["skills"][skill], "final-trainer-state-mismatch")
                check(tensor_hash(saved["model"]) == saved["summary"]["sha256"], "final-trainer-weights-mismatch")
            if env_output is not None:
                with (agent_output / "actions.jsonl").open() as left, (env_output / "actions.jsonl").open() as right:
                    from itertools import zip_longest
                    count = 0
                    for ll_line, env_line in zip_longest(left, right):
                        check(ll_line is not None and env_line is not None, "trace-length-mismatch")
                        if ll_line is None or env_line is None:
                            break
                        row, status = json.loads(ll_line), json.loads(env_line)
                        count += 1
                        check(row["status"] == status and status["environment_atomic_id"] == count, "trace-action-join")
                        if row["q_values"] is not None:
                            check(row["action"] in row["legal_actions"], "policy-mask-violation")
                            if not row["exploratory"]:
                                check(row["action"] == max(row["legal_actions"], key=lambda a: row["q_values"][a]), "non-greedy-policy-action")
                    check(count == ll["actions"], "missing-action-evidence")
    except (KeyError, IndexError, ValueError, TypeError, OSError) as error:
        errors.append(f"incomplete-evidence:{type(error).__name__}:{error}")
    return {"execution_valid": not errors, "errors": errors, "episodes": env.get("episodes", []),
            "experiment_id": "2-6-CR", "skill_outcomes": skill_outcomes,
            "episode_outcomes": {reason: sum(row["reason"] == reason for row in env.get("episodes", []))
                                 for reason in ("action_budget", "death", "diamond", "time_limit")},
            "learning": states.get("learner", {}).get("skills", {}),
            "adoption": states.get("hl_reasoner", {}).get("skills", {}),
            "native_actions": env.get("native_action_count", 0),
            "elapsed_seconds": states.get("hl_reasoner", {}).get("execution_seconds", 0)}


def np_equal(left, right) -> bool:
    """Compare archived image bytes without expanding them into JSON."""
    import numpy as np
    return bool(np.array_equal(left, right))


def run_experiment(run: int, output: Path, *, duration_seconds=DURATION, device="cuda", gpu_id=0,
                   learning: dict | None = None, source: dict | None = None) -> dict:
    """Run one fresh independent learner; retain operationally invalid attempts."""
    import torch
    require(0 <= run < RUNS, "Run index must be 0–19")
    require(torch.__version__.split("+")[0] == "2.14.0", "Host requires Torch 2.14.0")
    load_basics(torch)
    grounding, _ = resolve_active_grounding_bundle()
    require(not output.exists(), "Refusing to overwrite existing run")
    config = {"run": run, "duration_seconds": duration_seconds, "device": device, "gpu_id": gpu_id,
              "seeds": {"environment": Seeder(run).environment, "ll_reasoner": Seeder(run).ll_reasoner,
                        "learner": Seeder(run).learner, "episode_environment_offset": Seeder.ENV_MULTIPLIER},
              "execution_kind": "main" if duration_seconds == DURATION else "preflight",
              "learning": learning or asdict(LearningConfig()), "initial_policies": BASIC_HASHES,
              "grounding_manifest": file_sha256(grounding / "manifest.json"),
              "source": source or source_identity(), "treatment": "continuing-native-skills-v1"}
    config["identity"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    orc, agent_id, env_id = build_orchestrator(config, output)
    config.update(agent_id=agent_id, env_id=env_id)
    write_json(output / "config.json", config)
    image = f"aurontliar/mhagenta:{DEFAULT_TORCH_MHAGENTA_VERSION}"
    subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True)
    probe = ("import torch; from torch import nn; assert torch.__version__.split('+')[0]=='2.14.0'; "
             "m=nn.Sequential(nn.Conv2d(3,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),"
             "nn.Conv2d(64,64,3),nn.Flatten(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,6))"
             f".to('{device}'); x=torch.ones(2,3,64,64,device='{device}'); "
             "m(x).sum().backward(); print(torch.__version__)")
    gpu = ["--gpus", f"device={gpu_id}"] if device == "cuda" else []
    subprocess.run(["docker", "run", "--rm", *gpu, "--entrypoint", "python", image, "-c", probe], check=True)
    # Unique source/run identities and fresh directories make replacement unnecessary.
    existing = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], check=True, capture_output=True, text=True).stdout.splitlines()
    require(agent_id not in existing and env_id not in existing, "Matching experiment containers already exist")
    try:
        orc.run(mhagenta_version=DEFAULT_TORCH_MHAGENTA_VERSION, local_build=Path(mhagenta.__file__).parents[1], force_run=False)
        require(source_identity() == config["source"], "Sources changed during execution")
        return process_run(output)
    except BaseException:
        # These IDs were proved absent before this launch. Preserve their
        # containers/images and output for diagnosis, stopping only this run.
        for identifier in (agent_id, env_id):
            subprocess.run(["docker", "stop", "--time", "30", identifier], capture_output=True, check=False)
        raise


def process_run(output: Path) -> dict:
    """Reconstruct validity from retained state, logs and experience without running worlds."""
    config = json.loads((output / "config.json").read_text())
    identity = {key: value for key, value in config.items() if key not in {"identity", "agent_id", "env_id"}}
    require(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == config["identity"], "Retained configuration identity changed")
    states, environment = read_states(output, config["agent_id"], config["env_id"])
    result = check_results(states, environment, config, output / config["agent_id"] / "out", output / config["env_id"] / "out")
    for identifier in (config["agent_id"], config["env_id"]):
        log = output / f"{identifier}.log"
        if not log.is_file() or any(marker in log.read_text(errors="replace").lower() for marker in
                                   ("traceback", "caught exception", "failed to save state")):
            result["errors"].append(f"runtime-log:{identifier}")
    result["execution_valid"] = not result["errors"]
    write_json(output / "result.json", result)
    return result


def run_batch(runs: int | Sequence[int] | None = None, exp_path: str | Path = ".",
              mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION, process_only=False, *,
              duration_seconds: int = DURATION, device: str = "cuda", gpu_id: int = 0) -> None:
    """Execute requested global IDs, stopping on operational faults, never poor science."""
    root = Path(exp_path).resolve()
    if process_only:
        manifest = json.loads((root / "batch.json").read_text())
        ids = manifest["run_ids"] if runs is None else list(normalize_runs(runs)[0])
        results = []
        for run in ids:
            try:
                results.append(process_run(root / f"run-{run:05d}"))
            except (OSError, ValueError, KeyError) as error:
                results.append({"execution_valid": False, "errors": [f"run-{run}: {error}"]})
        write_json(root / "summary.json", {"run_ids": ids, "results": results})
        require(all(result["execution_valid"] for result in results), "Invalid retained execution")
        return
    ids = list(normalize_runs(RUNS if runs is None else runs)[0])
    require(ids and len(set(ids)) == len(ids) and all(0 <= run < RUNS for run in ids), "Select unique run IDs in 0–19")
    require(duration_seconds == DURATION, "The main treatment has a fixed 60-minute agent timeout")
    require(device == "cuda" and type(gpu_id) is int and gpu_id >= 0, "Main execution requires an assigned CUDA device")
    require(mha_version == DEFAULT_TORCH_MHAGENTA_VERSION, "Require the preinstalled Torch container")
    require(not root.exists(), "Refusing to overwrite an existing batch")
    source = source_identity()
    root.mkdir(parents=True)
    manifest = {"run_ids": ids, "status": "running", "current_run": None, "runs": [],
                "source": source, "started_at": time.time(), "duration_seconds": DURATION}
    write_json(root / "batch.json", manifest)
    for run in ids:
        try:
            require(source_identity() == source, "Sources changed during batch")
            manifest["current_run"] = run
            write_json(root / "batch.json", manifest)
            result = run_experiment(run, root / f"run-{run:05d}", device=device, gpu_id=gpu_id, source=source)
            manifest["runs"].append({"run": run, "result": result})
            require(result["execution_valid"], "Execution failed validation; remaining runs stopped")
        except BaseException as error:
            manifest.update(status="blocked", error=f"{type(error).__name__}: {error}")
            write_json(root / "batch.json", manifest)
            raise
        write_json(root / "batch.json", manifest)
    manifest.update(status="completed", current_run=None, finished_at=time.time())
    write_json(root / "batch.json", manifest)
