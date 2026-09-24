"""Prepare and execute the direct AchieveOn treatment on a dedicated EC2 host."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import mha_env_blocksworld
import mha_exp_common
import mhagenta
from mhagenta import Orchestrator
from mhagenta.containers import REPO

from mha_exp_common.batch import cleanup_run_containers, cleanup_run_images
from mha_exp_common.defaults import DEFAULT_TORCH_MHAGENTA_VERSION
from mha_exp_level2_bw.achieve_on import learning, policy
from mha_exp_level2_bw.achieve_on.demonstrations import demonstration_start
from mha_exp_level2_bw.exp2_5.contracts import GoalSpec
from mha_exp_level2_bw.exp2_5.grounding import ground_observation
from mha_exp_level2_bw.exp2_5.planning import PlanningService
from mha_exp_level2_bw.exp2_5.policy import artifact_paths, validate_manifest
from .environment import Environment, initial_state
from . import runtime

TREATMENT = "direct-achieve-on-dqfd-static-recovery-v2"


@dataclass(frozen=True)
class Profile:
    """Declared learning counts and provisional outcome-independent safety cap."""
    demonstrations: int
    training_episodes: int
    warm_updates: int
    updates_per_episode: int
    probe_cases: int
    probe_interval: int
    duration_seconds: int
    action_cap: int | None = 64


PROFILES = {
    "smoke": Profile(2, 2, 4, 2, 2, 1, 900),
    "pilot": Profile(60, 100, 600, 32, 20, 20, 7200),
    "main": Profile(600, 1000, 6000, 32, 100, 100, 3600),
}


def sha(path: Path) -> str:
    """Return a file identity for retained artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(candidate: Path, assessment: Path) -> tuple[Path, dict]:
    """Require completed pretraining and its matching frozen assessment."""
    report = json.loads((candidate / "report.json").read_text())
    measured = json.loads((assessment / "assessment.json").read_text())
    runtime.require(report["status"] == "completed-pilot-unqualified", "Pretraining has not completed.")
    runtime.require(measured["status"] == "completed-assessment", "Frozen pretraining assessment is missing.")
    name = report["selected_checkpoint"]
    runtime.require(Path(name).name == name, "Invalid selected checkpoint path.")
    checkpoint = candidate / name
    checksum = sha(checkpoint)
    selected = next(item for item in report["selection"] if item["checkpoint"] == name)
    runtime.require(checksum == selected["checkpoint_sha256"] == measured["checkpoint_sha256"], "Pretraining/assessment checkpoint identity differs.")
    manifest = validate_manifest(*artifact_paths())
    runtime.require(report["provenance"]["transfer_sha256"] == measured["transfer_sha256"] == manifest["checkpoint_sha256"], "Transfer lineage differs.")
    payload = learning.torch.load(checkpoint, map_location="cpu", weights_only=True)
    runtime.require(payload["architecture"] == policy.ARCHITECTURE and payload["provenance"] == report["provenance"], "Pretrained payload lineage differs.")
    model = policy.build_network(learning.torch)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    runtime.require(all(learning.torch.isfinite(parameter).all() for parameter in model.parameters()), "Nonfinite pretrained weights.")
    return checkpoint, {"checkpoint_sha256": checksum, "report_sha256": sha(candidate / "report.json"),
                        "assessment_sha256": sha(assessment / "assessment.json"),
                        "transfer_sha256": manifest["checkpoint_sha256"], "assessment": measured["paired"]}


def build_schedule(run: int, profile: Profile, service: Any | None = None,
                   production: bool = False, frozen_policy: str | None = None) -> list[dict]:
    """Prepare paired seeds/goals; do not admit cases based on policy success."""
    runtime.require(0 <= run < 50, "Run index must be 0..49.")
    service = service or PlanningService(
        domain_path=artifact_paths()[0].parents[2] / "blocksworld-transfer-domain.pddl",
        blocks=[f"b{i}" for i in range(8)], locations=[f"t{i}" for i in range(5)])
    env = mha_env_blocksworld.BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    env.expose_snapshot = True
    schedule = []

    def task(mode: str, index: int, probe: int | str | None = None, ordinary: bool = False) -> dict:
        base = 710_000_000 if ordinary else {"demo": 610_000_000, "train": 710_000_000}.get(mode, 810_000_000)
        seed = base + run * 10_000 + index
        observation, _ = env.reset(seed=seed)
        facts = ground_observation(observation).facts
        goals = [GoalSpec(f"b{a}", f"b{b}") for a in range(8) for b in range(8)
                 if a != b and f"on(b{a},b{b})" not in facts]
        goal = goals[index % len(goals)]
        metadata = {"reset_actions": [], "difficulty": 0}
        if mode == "demo":
            observation, goal, metadata = demonstration_start(env, observation, seed, index)
            facts = ground_observation(observation).facts
        result = {"id": f"{mode}-{probe if probe is not None else 'once'}-{index}", "mode": mode,
                  "seed": seed, "goal": goal.as_dict(), "case_index": index,
                  "probe": probe, "initial_facts": sorted(facts), "plan": [], **metadata}
        if mode in {"demo", "transfer"} and not production:
            plan = service.solve(set(facts), goal, f"direct-{mode}-{index}")
            result.update(plan=plan.actions if plan.accepted else [], planner=plan.engine,
                          planner_accepted=plan.accepted, planner_seconds=plan.elapsed_seconds)
        if mode == "train":
            result["train_index"] = index
        return result

    try:
        if frozen_policy:
            runtime.require(frozen_policy in {"transfer", "achieve_on"}, "Unknown frozen policy.")
            mode = "transfer" if frozen_policy == "transfer" else "pretrained"
            return [task(mode, index, ordinary=True) for index in range(profile.training_episodes)]
        for mode in ("transfer", "pretrained", "probe"):
            schedule.extend(task(mode, index, 0) for index in range(profile.probe_cases))
        schedule.extend(task("demo", index) for index in range(profile.demonstrations))
        schedule.extend(task("probe", index, "warm") for index in range(profile.probe_cases))
        for index in range(profile.training_episodes):
            schedule.append(task("train", index))
            if (index + 1) % profile.probe_interval == 0 or index + 1 == profile.training_episodes:
                schedule.extend(task("probe", case, index + 1) for case in range(profile.probe_cases))
    finally:
        env.close()
    return schedule


def source_identity() -> dict:
    """Bind runtime, framework, shared code, and dependency specifications."""
    sources = {}
    for name, directory in [("direct", Path(__file__).parent), ("achieve_on", Path(policy.__file__).parent),
                            ("transfer", artifact_paths()[0].parents[2]),
                            ("environment", Path(mha_env_blocksworld.__file__).parent),
                            ("common", Path(mha_exp_common.__file__).parent),
                            ("framework", Path(mhagenta.__file__).parent)]:
        sources.update({f"{name}/{path.relative_to(directory).as_posix()}": sha(path)
                        for path in sorted(directory.rglob("*.py"))})
    sources.update({f"direct/{path.name}": sha(path) for path in Path(__file__).parent.glob("requirements*.txt")})
    transfer = artifact_paths()[0].parents[2]
    sources.update({f"transfer/{path.relative_to(transfer).as_posix()}": sha(path)
                    for path in sorted(transfer.rglob("*")) if path.suffix in {".json", ".pt", ".pddl"}})
    return sources


def prepare(candidate: Path, assessment: Path, output: Path, run: int, profile_name: str,
            device: str, gpu_id: int, production: bool = False, *,
            profile_override: Profile | None = None, frozen_policy: str | None = None,
            matched_2_4: bool = False, policy_family: str | None = None) -> dict:
    """Prepare immutable run inputs without launching Docker or touching CUDA."""
    runtime.require(not output.exists(), "Run output already exists.")
    runtime.require(device in {"cpu", "cuda"} and gpu_id >= 0, "Invalid device selection.")
    runtime.require(policy_family is None or matched_2_4, "Policy families require matched frozen Transfer.")
    profile = profile_override or PROFILES[profile_name]
    started = time.monotonic()
    matching = None
    if matched_2_4:
        from mha_exp_level2_bw.exp2_5.matched import matched_task
        from mha_exp_level2_bw.exp2_5.resized import frozen_artifact
        runtime.require(production and frozen_policy == "transfer" and profile.training_episodes == 1
                        and profile.action_cap is None, "Matched execution requires one frozen Transfer goal without a goal action cap.")
        task, matching = matched_task(run)
        dimensions = {key: matching["source_treatment"][key] for key in ("table_len", "num_blocks")}
        if policy_family is None:
            checkpoint, artifact = frozen_artifact(**dimensions)
        else:
            from mha_exp_level2_bw.exp2_5.family import resolve
            checkpoint, artifact = resolve(policy_family, **dimensions)
        baseline = {"checkpoint_sha256": artifact["checkpoint_sha256"],
                    "transfer_sha256": artifact["checkpoint_sha256"], "transfer_artifact": artifact}
        schedule = [task]
    else:
        checkpoint, baseline = reference(candidate, assessment)
        schedule = build_schedule(run, profile, production=production, frozen_policy=frozen_policy)
    config = {"treatment": TREATMENT, "run": run, "profile_name": profile_name, "profile": asdict(profile),
              "production": production,
              "frozen_policy": frozen_policy,
              "schedule": schedule, "reference": baseline, "device": device, "gpu_id": gpu_id,
              "sources": source_identity(), "preparation_seconds": time.monotonic() - started,
              "learning": None if frozen_policy else asdict(learning.Config(seed=12605 + run, demonstrations=profile.demonstrations,
                  warm_updates=profile.warm_updates, online_steps=profile.training_episodes * profile.action_cap,
                  action_cap=profile.action_cap, selection_cases=profile.probe_cases))}
    if production:
        config["treatment"] = "planner-transfer-with-runtime-achieve-on-dqfd-v3"
        config["adoption_gate"] = {"minimum_episodes": 100, "all_probes_succeed": True,
                                   "joint_actions_strictly_fewer": True, "freeze_admitted_model": True,
                                   "revoke_after_production_failure": True}
    if profile_override:
        config["treatment"] = "bw-one-hour-runtime-achieve-on-dqfd-v1"
        if profile.duration_seconds > 3600:
            config["treatment"] = "bw-long-runtime-achieve-on-dqfd-v1"
    if frozen_policy:
        config["treatment"] = f"bw-one-hour-frozen-{frozen_policy}-v1"
        config.pop("adoption_gate", None)
    if matched_2_4:
        from mha_exp_level2_bw.exp2_5.matched import PROTOCOL
        config.update(treatment=PROTOCOL, matched_2_4=matching, dimensions=dimensions,
                      stopping={"goal_completion_limit": 1, "goal_action_cap": None,
                                "transfer_action_cap": 32, "shutdown_margin_seconds": 2})
    if policy_family is not None:
        config["policy_family"] = policy_family
    config["identity"] = runtime.digest({key: value for key, value in config.items() if key != "preparation_seconds"})
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "bw_atomic_input"
    inputs.mkdir()
    (inputs / "__init__.py").write_text('"""Immutable comparator for one direct AchieveOn run."""\n')
    shutil.copyfile(checkpoint, inputs / "pretrained.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def build_orchestrator(config: dict, output: Path) -> tuple[Any, str, str]:
    """Expose the requested GPU only to the agent; LL explicitly stays on CPU."""
    root = Path(__file__).resolve().parents[5]
    framework = root.parent / "mhagenta"
    runtime.require('version = "1.4.12"' in (framework / "pyproject.toml").read_text(), "Local MHAgentA must be 1.4.12.")
    runtime.require(Path(mhagenta.__file__).resolve().is_relative_to(framework), "MHAgentA import is not the sibling source.")
    profile = config["profile"]
    short = config["identity"][:12]
    agent_id, env_id = f"exp_direct_bw_{short}", f"exp_direct_env_{short}"
    exchange = f"direct-bw-{short}"
    orchestrator = Orchestrator(save_dir=output, step_frequency=0.01, control_frequency=0.01,
        status_frequency=5.0, agent_start_delay=60.0, exec_duration=profile["duration_seconds"],
        save_format="json", log_level=Orchestrator.INFO, save_logs=True, no_stdout_logs=False,
        mas_rmq_uri="localhost:5672", mas_rmq_exchange_name=exchange, state_autosave_interval=30,
        stop_on_agents_term=True, gpu_device_ids=[config["gpu_id"]] if config["device"] == "cuda" else "none")
    states = runtime.initial_states()
    classes = {"perceptor": runtime.Perceptor, "actuator": runtime.Actuator, "llreasoner": runtime.LLReasoner,
               "hlreasoner": runtime.HLReasoner, "knowledge": runtime.Knowledge, "memory": runtime.Memory,
               "learner": runtime.Learner, "goalgraph": runtime.GoalGraph}
    init = {"llreasoner": {"seed": 22605 + config["run"], "action_cap": profile["action_cap"],
                          "training_episodes": profile["training_episodes"]},
            "hlreasoner": {"schedule": config["schedule"], "production": config.get("production", False),
                           "duration_seconds": profile["duration_seconds"]},
            "learner": {"learning": config["learning"], "device": config["device"], "identity": config["identity"],
                        "updates_per_episode": profile["updates_per_episode"]}}
    if config.get("matched_2_4"):
        for name in ("llreasoner", "hlreasoner"):
            init[name]["dimensions"] = config["dimensions"]
        init["llreasoner"]["transfer_sha256"] = config["reference"]["transfer_sha256"]
        if config.get("policy_family") is not None:
            init["llreasoner"]["policy_family"] = config["policy_family"]
        init["hlreasoner"]["matched_single_goal"] = True
    if config.get("frozen_policy"):
        for name in ("memory", "learner"):
            classes.pop(name)
        for name in ("llreasoner", "hlreasoner"):
            init[name]["frozen_policy"] = config["frozen_policy"]
    modules = {name: cls(module_id=name, initial_state=states[name], init_kwargs=init.get(name, {}),
                         **({"exchange_name": exchange} if name in {"perceptor", "actuator"} else {}))
               for name, cls in classes.items()}
    common = Path(mha_exp_common.__file__).parent
    direct = Path(__file__).parent
    sources = [common, Path(policy.__file__).parent, artifact_paths()[0].parents[2], output / "bw_atomic_input"]
    # Explicit framework sources are copied flat; restore their Python namespace.
    install = output / "runtime-init.sh"
    install.write_text('set -eu\nroot="${1:-/agent}"\n'
                       'for package in achieve_on exp2_5; do\n'
                       '  test ! -e "$root/mha_exp_level2_bw/$package"\n'
                       '  mv "$root/$package" "$root/mha_exp_level2_bw/$package"\n'
                       'done\n', encoding="utf-8", newline="\n")
    if config.get("production"):
        with install.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write('command -v java >/dev/null || { apt-get update && apt-get install -y --no-install-recommends default-jre-headless; }\n')
    orchestrator.add_agent(agent_id=agent_id, perceptors=modules["perceptor"], actuators=modules["actuator"],
        ll_reasoners=modules["llreasoner"], hl_reasoners=modules["hlreasoner"], knowledge=modules["knowledge"],
        memory=modules.get("memory", []), learners=modules.get("learner", []), goal_graphs=modules["goalgraph"],
        requirements_path=direct / ("requirements-production.txt" if config.get("production") else "requirements.txt"),
        extra_runtime_sources=sources, init_script=install)
    orchestrator.add_environment(base=Environment(initial_state(), **config.get("dimensions", {})), env_id=env_id,
        exec_duration=profile["duration_seconds"] + 30, exchange_name=exchange, gpu_device_ids="none",
        requirements_path=direct / "requirements-env.txt", extra_runtime_sources=[Path(mha_env_blocksworld.__file__).parent])
    return orchestrator, agent_id, env_id


def read_states(output: Path, agent_id: str, env_id: str) -> tuple[dict, dict]:
    """Read only named framework states, excluding inputs and episode evidence."""
    states = {}
    for name in runtime.initial_states():
        path = output / agent_id / Orchestrator.SAVE_SUBDIR / f"{agent_id}.{name}.json"
        if path.is_file():
            states[name] = json.loads(path.read_text())
    path = output / env_id / Orchestrator.SAVE_SUBDIR / f"{env_id}.json"
    return states, json.loads(path.read_text()) if path.is_file() else {}


def artifact_receipt(states: dict, env: dict, config: dict) -> dict:
    """Bind archived artifact validation to all retained execution evidence.

    Exporters may persist this receipt only after the full checker passes with
    the original checkpoint and episode files present. It is an audit record,
    not a replacement for a fresh binary-artifact audit of the source archive.
    """
    return {"format": "bw-verified-artifacts-v1",
            "evidence_sha256": runtime.digest({"config": config, "states": states, "environment": env}),
            "checkpoints_verified": len(states.get("learner", {}).get("checkpoints", [])),
            "episodes_verified": len(states.get("memory", {}).get("episodes", []))}


def check_results(states: dict, env: dict, config: dict, checkpoint_dir: Path,
                  *, artifact_validation: dict | None = None) -> dict:
    """Replay saved execution; compact exports use prior artifact verification.

    An artifact receipt binds the complete configuration and final states to a
    successful full check before checkpoints and episode files were archived.
    Behavioral checks are always repeated, including action replay and adoption.
    """
    failures = []
    frozen = bool(config.get("frozen_policy"))
    expected_modules = set(runtime.initial_states()) - ({"memory", "learner"} if frozen else set())
    if set(states) != expected_modules or not env:
        return {"operationally_valid": False, "failures": ["missing-module-states"]}
    hl, ll = states["hlreasoner"], states["llreasoner"]
    # Absent learning modules have no events; they are never fabricated in saved states.
    learner = states.get("learner", runtime.initial_states()["learner"])
    memory = states.get("memory", runtime.initial_states()["memory"])
    results = hl["results"]
    profile = config["profile"]
    limited = bool(hl.get("time_limited"))
    completed_train = sum(row["mode"] == "train" for row in results)
    completed_demo = sum(row["mode"] == "demo" for row in results)
    warmed = completed_demo == profile["demonstrations"]
    expected_train = completed_train if limited else profile["training_episodes"]
    expected_demo = completed_demo if limited else profile["demonstrations"]
    expected_warm = int(warmed) if limited else 1
    if frozen:
        expected_train = expected_demo = expected_warm = 0
    checks = {
        "time-limit": not limited or hl["execution_seconds"] >= max(0, profile["duration_seconds"] - 45),
        "schedule-completed": hl["phase"] == "completed" and hl["index"] == len(results) and (len(results) == len(config["schedule"]) or limited),
        "learning-completed": learner["trained_episodes"] == expected_train and learner["optimizer_steps"] == expected_warm * profile["warm_updates"] + expected_train * profile["updates_per_episode"],
        "device": frozen or learner["device"] == config["device"],
        "models": ll["model_installs"] == len(learner["checkpoints"]) == (0 if frozen else expected_train + expected_warm + 1),
        "memory": len(memory["episodes"]) == states["knowledge"]["learning_episodes"] == expected_demo + expected_train,
        "memory-boundary": memory["requests"] == memory["responses"] == expected_train + expected_warm,
        "knowledge": states["knowledge"]["revisions"] == ll["completed"] == len(results),
        "closed": env["closed"] and env["illegal_actions"] == 0,
        "setup-actions": env.get("setup_actions", 0) == sum(len(task.get("reset_actions", [])) for task in config["schedule"][:len(results)]),
        "actions": env["actions"] == sum(len(result["actions"]) for result in results) == ll["teacher_inferences"] + ll["direct_inferences"],
        "observations": env["observations"] == ll["observations"] == ll["observation_requests"] == env["actions"] + env["resets"],
        "resets": env["resets"] == len(results),
        "action-boundary": states["actuator"]["requests"] == states["actuator"]["responses"] == ll["requests"] == ll["statuses"] == env["actions"] + env["resets"] + 1,
        "perception-boundary": states["perceptor"]["requests"] == states["perceptor"]["responses"] == env["observations"],
        "goals": states["goalgraph"]["requests"] == states["goalgraph"]["responses"] == len(results) + 1,
        "quiescent": all(state.get("pending") is None for state in states.values()) and ll["active"] is None and hl["learning_id"] is None and hl["goal_result"] is None and hl["belief_result"] is None,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    checkpoints = {item["revision"]: item for item in learner["checkpoints"]}
    dimensions = config.get("dimensions", {"table_len": 5, "num_blocks": 8})
    matched = config.get("matched_2_4")
    if matched:
        from mha_exp_level2_bw.exp2_5.matched import matched_task
        from mha_exp_level2_bw.exp2_5.resized import frozen_artifact
        task, expected_matching = matched_task(config["run"])
        if config.get("policy_family") is None:
            _, artifact = frozen_artifact(**dimensions)
        else:
            from mha_exp_level2_bw.exp2_5.family import resolve
            _, artifact = resolve(config["policy_family"], **dimensions)
        if (matched != expected_matching or config["schedule"] != [task]
                or dimensions != {key: matched["source_treatment"][key] for key in ("table_len", "num_blocks")}
                or config["reference"]["transfer_sha256"] != artifact["checkpoint_sha256"]
                or (config.get("policy_family") is not None and config["reference"]["transfer_artifact"] != artifact)
                or len(results) != 1 or profile["action_cap"] is not None):
            failures.append("matched-task-or-artifact-identity")
        if (len(hl["planning_records"]) != 1 or hl["planning_records"][0]["id"] != task["id"]
                or hl["planning_records"][0]["mode"] != "transfer"):
            failures.append("matched-planner-accounting")
        from mha_exp_level2_bw.exp2_5.resized import load_frozen
        if config.get("policy_family") is None:
            frozen_model, _ = load_frozen(learning.torch, **dimensions)
        else:
            from mha_exp_level2_bw.exp2_5.family import load
            frozen_model, _ = load(learning.torch, config["policy_family"], **dimensions)
    replay_env = mha_env_blocksworld.BlocksWorldEnv(**dimensions, symbolic=False)
    replay_env.expose_snapshot = True
    for result, task in zip(results, config["schedule"]):
        if any(result.get(key) != task.get(key) for key in ("id", "mode", "seed", "goal", "probe", "case_index")):
            failures.append("task-identity")
        if result.get("reset_actions", []) != task.get("reset_actions", []):
            failures.append("setup-identity")
        if task.get("reset_actions") and task["mode"] != "demo":
            failures.append("perturbed-nondemonstration")
        goal = GoalSpec(**result["goal"])
        if result["success"] != (goal.fact in result["facts"] and "hand-empty()" in result["facts"]):
            failures.append("ungrounded-success")
        expected = (config["reference"]["checkpoint_sha256"] if task["mode"] == "pretrained" else
                    config["reference"]["transfer_sha256"] if task["mode"] in {"demo", "transfer"} or result.get("executor") == "transfer" else
                    (result.get("adoption") or {}).get("sha256") if result.get("executor") == "adopted" else result["model_sha256"])
        if result["policy_sha256"] != expected:
            failures.append("policy-identity")
        record = checkpoints.get(result["revision"])
        if frozen:
            if result["revision"] != 0 or result["model_sha256"] != expected:
                failures.append("frozen-model-identity")
        elif record is None or record["model_sha256"] != result["model_sha256"]:
            failures.append("model-revision-identity")
        observation, _ = replay_env.reset(seed=result["seed"])
        for action in task.get("reset_actions", []):
            observation, _, _, _, info = replay_env.step(action)
            if not info["snapshot"].legal:
                failures.append("replayed-illegal-setup")
        if task.get("initial_facts") is not None and set(ground_observation(observation, **dimensions).facts) != set(task["initial_facts"]):
            failures.append("initial-facts-do-not-replay")
        transfer_index = transfer_steps = 0
        for action in result["actions"]:
            if matched:
                from mha_exp_level2_bw.exp2_5.contracts import TransferSpec
                from mha_exp_level2_bw.exp2_5.grounding import transfer_succeeded
                from mha_exp_level2_bw.exp2_5.policy import greedy_inference
                facts = ground_observation(observation, **dimensions).facts
                plan = result["execution_plan"]
                if goal.fact in facts and "hand-empty()" in facts:
                    failures.append("actions-after-goal")
                if transfer_index < len(plan) and transfer_succeeded(facts, TransferSpec.from_mapping(plan[transfer_index])):
                    transfer_index += 1
                    transfer_steps = 0
                if transfer_index >= len(plan) or transfer_steps >= 32:
                    failures.append("action-outside-transfer-plan")
                else:
                    spec = TransferSpec.from_mapping(plan[transfer_index])
                    expected_action, _, _ = greedy_inference(learning.torch, frozen_model, observation, spec, **dimensions)
                    if action != expected_action:
                        failures.append("action-does-not-match-frozen-policy")
                    transfer_steps += 1
            observation, _, _, _, info = replay_env.step(action)
            if not info["snapshot"].legal:
                failures.append("replayed-illegal-action")
        if set(ground_observation(observation, **dimensions).facts) != set(result["facts"]):
            failures.append("terminal-facts-do-not-replay")
        if profile["action_cap"] is not None and len(result["actions"]) > profile["action_cap"]:
            failures.append("action-cap-exceeded")
        if matched and result["reason"] not in {"goal", "time-limit", "planner-failed", "teacher-plan-incomplete", "teacher-transfer-cap"}:
            failures.append("unexpected-single-goal-stop")
    replay_env.close()
    if artifact_validation is not None:
        if artifact_validation != artifact_receipt(states, env, config):
            failures.append("archived-artifact-evidence-changed")
    else:
        for record in learner["checkpoints"]:
            path = checkpoint_dir / record["file"]
            if not path.is_file() or sha(path) != record["file_sha256"]:
                failures.append("checkpoint-missing-or-changed")
            else:
                payload = learning.torch.load(path, map_location="cpu", weights_only=True)
                weights = {key: value.tolist() for key, value in payload["model_state_dict"].items()}
                if (payload.get("identity") != config["identity"] or payload.get("revision") != record["revision"]
                        or payload.get("optimizer_steps") != record["optimizer_steps"]
                        or runtime.digest(weights) != record["model_sha256"]
                        or payload["provenance"]["transfer_sha256"] != config["reference"]["transfer_sha256"]):
                    failures.append("checkpoint-provenance")
        for record in memory["episodes"]:
            path = checkpoint_dir / f"episode-{record['id']}.json"
            if not path.is_file() or runtime.digest(json.loads(path.read_text())) != record["sha256"]:
                failures.append("episode-missing-or-changed")
    if learner["online_transitions"] != sum(item["rows"] for item in memory["episodes"] if item["mode"] == "train"):
        failures.append("online-transition-count")
    if config.get("production"):
        admitted, revoked, bypasses = None, False, 0
        events, seen, planned_ids = [], [], []
        plans = {row["id"]: row for row in hl["planning_records"] if row["mode"] == "train"}
        teaching_plans = {row["id"]: row for row in hl["planning_records"] if row["mode"] != "train"}
        for index, result in enumerate(results):
            seen.append(result)
            if result["id"] in teaching_plans and result.get("execution_plan") != teaching_plans[result["id"]]["actions"]:
                failures.append("runtime-teacher-planning")
            if result["mode"] == "train":
                if admitted is None:
                    planned_ids.append(result["id"])
                    record = plans.get(result["id"])
                    if (result.get("executor") != "transfer" or record is None
                            or result.get("execution_plan") != record["actions"]):
                        failures.append("production-planning")
                else:
                    bypasses += 1
                    if result.get("executor") != "adopted" or result.get("adoption") != admitted:
                        failures.append("unqualified-production-adoption")
                    if not result["success"]:
                        events.append({"revoked_after": result["id"], "sha256": admitted["sha256"]})
                        admitted, revoked = None, True
            milestone = result.get("probe")
            next_result = config["schedule"][index + 1] if index + 1 < len(config["schedule"]) else None
            if (result["mode"] == "probe" and isinstance(milestone, int) and milestone >= 100
                    and (next_result is None or next_result["mode"] != "probe")):
                evidence = runtime.adoption_evidence(seen, milestone, profile["probe_cases"])
                events.append(evidence)
                if evidence["passed"] and admitted is None and not revoked:
                    admitted = evidence
        if (events != hl["adoption_records"] or admitted != hl["adopted"]
                or revoked != hl["adoption_revoked"] or bypasses != hl["planner_bypasses"]
                or planned_ids != [row["id"] for row in hl["planning_records"] if row["mode"] == "train"]):
            failures.append("production-adoption-accounting")
    elif any(result.get("executor") for result in results):
        failures.append("unexpected-production-executor")
    comparisons = []
    for probe in dict.fromkeys(item["probe"] for item in results if item["mode"] == "probe"):
        learned = {item["case_index"]: item for item in results if item["mode"] == "probe" and item["probe"] == probe}
        row = {"milestone": probe, "successes": sum(item["success"] for item in learned.values()), "cases": len(learned)}
        for mode in ("transfer", "pretrained"):
            baseline = {item["case_index"]: item for item in results if item["mode"] == mode}
            pairs = [(item, baseline[index]) for index, item in learned.items() if index in baseline]
            row[mode] = {"paired_cases": len(pairs), "net_success_gain": sum(int(a["success"]) - int(b["success"]) for a, b in pairs),
                         "baseline_successes": sum(b["success"] for _, b in pairs),
                         "joint_successes": sum(a["success"] and b["success"] for a, b in pairs),
                         "actions_saved_on_joint_successes": sum(len(b["actions"]) - len(a["actions"]) for a, b in pairs if a["success"] and b["success"])}
        comparisons.append(row)
    return {"operationally_valid": not failures, "failures": sorted(set(failures)), "comparisons": comparisons,
            "outcome": "incomplete" if failures else "time-limit" if limited else "completed",
            "completed_schedule": hl["phase"] == "completed" and hl["index"] == len(results) == len(config["schedule"]),
            "counts_by_mode": dict(Counter(item["mode"] for item in results)), "identity": config["identity"],
            "production": {"enabled": config.get("production", False),
                           "goals": len(results) if frozen else sum(r["mode"] == "train" for r in results) if config.get("production") else 0,
                           "accomplished": sum(r["success"] for r in results) if frozen else sum(r["mode"] == "train" and r["success"] for r in results) if config.get("production") else 0,
                           "planner_calls": sum(row["mode"] == "train" for row in hl["planning_records"]),
                           "all_planner_calls": len(hl["planning_records"]), "planner_bypasses": hl["planner_bypasses"],
                           "adoption_records": hl["adoption_records"]}}


def execute(config: dict, output: Path) -> dict:
    """Run exactly one prepared treatment, retaining evidence and exact cleanup."""
    runtime.require(config["sources"] == source_identity(), "Sources changed after run preparation.")
    runtime.require(config["identity"] == runtime.digest({key: value for key, value in config.items()
                    if key not in {"identity", "preparation_seconds"}}), "Prepared configuration identity differs.")
    runtime.require(sha(output / "bw_atomic_input" / "pretrained.pt") == config["reference"]["checkpoint_sha256"], "Comparator changed after preparation.")
    base = f"{REPO}:{DEFAULT_TORCH_MHAGENTA_VERSION}"
    subprocess.run(["docker", "image", "inspect", base], check=True, capture_output=True)
    # A suffix does not prove the actual image contains working CUDA kernels.
    gpu = ["--gpus", f"device={config['gpu_id']}"] if config["device"] == "cuda" else []
    probe = "import torch; assert str(torch.__version__) == '2.14.0+cu130', torch.__version__; torch.set_num_threads(1); x=torch.ones((16,16),device='" + config["device"] + "',requires_grad=True); (x@x).sum().backward(); print(torch.__version__,x.device)"
    subprocess.run(["docker", "run", "--rm", *gpu, "--entrypoint", "python", base, "-c", probe], check=True)
    orchestrator, agent_id, env_id = build_orchestrator(config, output)
    started = time.monotonic()
    cleanup_started = False
    result = {"operationally_valid": False, "identity": config["identity"], "failures": []}
    try:
        # Existing exact names indicate another attempt, which must not be killed.
        present = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], check=True,
                                 capture_output=True, text=True).stdout.splitlines()
        runtime.require(not ({agent_id, env_id} & set(present)), "Run containers already exist; refusing replacement.")
        cleanup_started = True
        orchestrator.run(mhagenta_version=DEFAULT_TORCH_MHAGENTA_VERSION,
                         local_build=Path(__file__).resolve().parents[5].parent / "mhagenta", force_run=True)
        states, environment = read_states(output, agent_id, env_id)
        result = check_results(states, environment, config, output / agent_id / Orchestrator.SAVE_SUBDIR)
        for identifier in (agent_id, env_id):
            path = output / f"{identifier}.log"
            if not path.is_file() or any(marker in path.read_text(errors="replace").lower() for marker in ("traceback", "[error]", "caught exception")):
                result["failures"].append(f"log-{identifier}")
        result["operationally_valid"] = not result["failures"]
    except BaseException as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        try:
            if cleanup_started:
                cleanup_run_containers(agent_id, env_id, phase="after")
                cleanup_run_images(agent_id, env_id, phase="after")
        except BaseException as error:
            result["operationally_valid"] = False
            result["failures"].append(f"cleanup:{type(error).__name__}:{error}")
            raise
        finally:
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Prepare first; execute only when explicitly requested on the target host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-run", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--production", action="store_true",
                        help="Use runtime HLR Planner+Transfer on ordinary goals until AchieveOn passes its gate.")
    args = parser.parse_args()
    config = prepare(args.pretrained_run.resolve(), args.assessment.resolve(), args.output.resolve(),
                     args.run, args.profile, args.device, args.gpu_id, args.production)
    print(json.dumps({"identity": config["identity"], "tasks": len(config["schedule"]), "profile": args.profile}))
    if args.execute:
        result = execute(config, args.output.resolve())
        print(json.dumps(result, indent=2))
        if not result["operationally_valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
