"""CPU integration checks for the complete eight-module direct-learning flow."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.achieve_on.learning import Config
from mha_exp_level2_bw.achieve_on.policy import checkpoint_payload, warm_start
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_5.policy import artifact_paths, legal_action_indices
from mha_exp_level2_bw.exp2_6_direct import runner, runtime
from mha_exp_level2_bw.exp2_6_direct.environment import Environment, initial_state


class State(dict):
    """Small state adapter exposing the same attribute contract as MHAgentA."""
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def fixture_schedule() -> list[dict]:
    """Use a real solvable transfer case while exercising every schedule phase."""
    env = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    observation, _ = env.reset(seed=1000)
    initial_facts = sorted(ground_observation(observation).facts)
    move = 2 if 2 in legal_action_indices(observation) else 3
    prefix = [move, 5 - move]
    spec = next(item for item in enumerate_transfer_targets(ground_observation(observation).facts)
                if item.destination_support.startswith("b"))
    env.close()
    rows = []
    stages = [("transfer", 0), ("pretrained", 0), ("probe", 0), ("demo", None),
              ("demo", None), ("probe", "warm"), ("train", None), ("probe", 1),
              ("train", None), ("probe", 2)]
    trained = 0
    for index, (mode, probe) in enumerate(stages):
        row = {"id": f"task-{index}", "mode": mode, "seed": 1000,
               "goal": {"top": spec.block, "bottom": spec.destination_support},
               "plan": [spec.as_dict()] if mode in {"demo", "transfer"} else [],
               "case_index": 0, "probe": probe, "initial_facts": initial_facts}
        if mode == "demo":
            row["reset_actions"] = prefix
            row["difficulty"] = 0
        if mode == "train":
            row["train_index"] = trained
            trained += 1
        rows.append(row)
    return rows


def test_environment_rejects_perturbations_outside_demonstrations() -> None:
    """Evaluation and online training must retain the declared reset distribution."""
    environment = Environment(initial_state())
    try:
        with pytest.raises(ValueError, match="Only demonstrations"):
            environment.on_action(initial_state(), "test", action="reset", seed=1000,
                                  reset_actions=[2, 3], mode="probe")
    finally:
        environment._environment().close()


def exercise(tmp_path: Path, reverse_knowledge: bool, production: bool = False,
             timed: bool = False, frozen_policy: str | None = None,
             config_override: dict | None = None) -> tuple:
    """Execute actual module callbacks with queued, registered message edges."""
    torch.set_num_threads(1)
    model, provenance = warm_start(torch)
    pretrained = tmp_path / "pretrained.pt"
    torch.save(checkpoint_payload(model, provenance), pretrained)
    schedule = fixture_schedule()
    profile = runner.Profile(2, 2, 4, 2, 1, 1, 0 if timed else 900)
    if frozen_policy:
        schedule = [row for row in schedule if row["mode"] == ("transfer" if frozen_policy == "transfer" else "pretrained")]
        profile = runner.Profile(0, len(schedule), 0, 0, 0, 1, 0 if timed else 900)
    config = {"profile": asdict(profile), "schedule": schedule, "identity": "cpu-test",
              "device": "cpu", "production": production, "frozen_policy": frozen_policy, "reference": {"checkpoint_sha256": runner.sha(pretrained),
                                              "transfer_sha256": runner.sha(artifact_paths()[1])}}
    if config_override is not None:
        config = config_override
        schedule = config["schedule"]
        profile = runner.Profile(**config["profile"])
    values = runtime.initial_states()
    classes = {"perceptor": runtime.Perceptor, "actuator": runtime.Actuator, "goalgraph": runtime.GoalGraph,
               "knowledge": runtime.Knowledge, "memory": runtime.Memory, "learner": runtime.Learner,
               "llreasoner": runtime.LLReasoner, "hlreasoner": runtime.HLReasoner}
    if frozen_policy:
        for name in ("memory", "learner"):
            classes.pop(name)
            values.pop(name)
    directory = SimpleNamespace(internal=SimpleNamespace(**{
        key: [SimpleNamespace(module_id=value)] for key, value in {
            "perception": "perceptor", "actuation": "actuator", "goals": "goalgraph",
            "knowledge": "knowledge", "memory": "memory", "learning": "learner",
            "ll_reasoning": "llreasoner", "hl_reasoning": "hlreasoner"}.items()}),
        external=SimpleNamespace(environment=SimpleNamespace(address={"env_id": "env"})))
    states = {name: State(value) for name, value in values.items()}
    modules = {name: cls(module_id=name, initial_state=values[name]) for name, cls in classes.items()}
    queue = deque()
    stopped = []

    class Outbox:
        def __init__(self, sender):
            self.sender = sender

        def __getattr__(self, method):
            def send(*args, **kwargs):
                if method == "terminate_agent":
                    stopped.append(args[0])
                    return
                receiver, *arguments = args
                if method == "send_memories":
                    assert len(arguments[0]) == 1
                callbacks = {"request_observation": "on_request", "request_action": "on_request",
                    "send_observation": "on_observation", "send_status": "on_action_status",
                    "send_goals": "on_goal_update", "send_goal_update": "on_goal_update",
                    "send_observations": "on_observation_update", "request_memories": "on_memory_request",
                    "send_memories": "on_memories", "send_model": "on_model", "send_learner_task": "on_task"}
                callback = ("on_observed_beliefs" if self.sender == "llreasoner" else "on_belief_update") if method == "send_beliefs" else callbacks[method]
                event = (receiver, callback, self.sender, arguments, kwargs)
                if reverse_knowledge and self.sender == "knowledge":
                    queue.appendleft(event)
                else:
                    queue.append(event)
            return send

    for name, state in states.items():
        state.directory, state.outbox = directory, Outbox(name)
    if not frozen_policy:
        modules["memory"].on_init(output_dir=str(tmp_path))
        modules["learner"].on_init(learning=asdict(Config(seed=1, warm_updates=4, online_steps=128, batch_size=2)),
            device="cpu", output_dir=str(tmp_path), identity="cpu-test", updates_per_episode=2)
    dimensions = config.get("dimensions", {})
    modules["llreasoner"].on_init(pretrained_path=str(pretrained), seed=1, action_cap=profile.action_cap,
                                  training_episodes=2, frozen_policy=frozen_policy,
                                  **({"policy_family": config["policy_family"]} if config.get("policy_family") else {}),
                                  **({"dimensions": dimensions, "transfer_sha256": config["reference"]["transfer_sha256"]} if dimensions else {}))
    modules["hlreasoner"].on_init(schedule=schedule, production=production,
                                duration_seconds=profile.duration_seconds, frozen_policy=frozen_policy,
                                dimensions=dimensions, matched_single_goal=bool(config.get("matched_2_4")))
    environment = Environment(initial_state(), **dimensions)

    def observe(env_id, **kwargs):
        _, response = environment.on_observe(environment.state, "agent", **kwargs)
        queue.append(("perceptor", "on_observation", env_id, [], response))

    def act(env_id, **kwargs):
        _, response = environment.on_action(environment.state, "agent", **kwargs)
        queue.append(("actuator", "on_status", env_id, [], response))

    modules["perceptor"].observe = observe
    modules["actuator"].act = act
    modules["hlreasoner"].on_first(states["hlreasoner"])
    for _ in range(20_000):
        if queue:
            receiver, callback, sender, args, kwargs = queue.popleft()
            getattr(modules[receiver], callback)(states[receiver], sender, *args, **kwargs)
        else:
            if not frozen_policy:
                modules["learner"].step(states["learner"])
            modules["hlreasoner"].step(states["hlreasoner"])
        if stopped and not queue:
            break
    assert stopped == ["direct-achieve-on-time-limit" if timed else "direct-achieve-on-schedule-completed"]
    modules["hlreasoner"].on_last(states["hlreasoner"])
    plain = {name: {key: value for key, value in state.items() if key not in {"directory", "outbox"}}
             for name, state in states.items()}
    json.dumps(plain)
    return plain, environment.state, config


@pytest.mark.parametrize("policy", ["transfer", "achieve_on"])
def test_frozen_hourly_treatment_has_six_modules_and_no_learning(tmp_path, policy) -> None:
    """Both frozen executors use the same goal flow without Memory or Learner."""
    states, env, config = exercise(tmp_path, False, frozen_policy=policy)
    assert len(states) == 6 and "learner" not in states and "memory" not in states
    result = runner.check_results(states, env, config, tmp_path)
    assert result["operationally_valid"], result
    assert result["production"]["goals"] == 1
    assert states["llreasoner"]["model_installs"] == 0


def test_compact_receipt_binds_verified_artifacts_to_final_states(tmp_path) -> None:
    """Archived artifacts need a matching receipt; changed final states fail."""
    states, env, config = exercise(tmp_path, False)
    assert runner.check_results(states, env, config, tmp_path)["operationally_valid"]
    receipt = runner.artifact_receipt(states, env, config)
    empty = tmp_path / "without-artifacts"
    empty.mkdir()
    assert not runner.check_results(states, env, config, empty)["operationally_valid"]
    assert runner.check_results(states, env, config, empty,
                                artifact_validation=receipt)["operationally_valid"]
    changed = deepcopy(states)
    changed["learner"]["checkpoints"][0]["file_sha256"] = "changed"
    result = runner.check_results(changed, env, config, empty, artifact_validation=receipt)
    assert "archived-artifact-evidence-changed" in result["failures"]


def test_export_and_process_only_revalidate_compact_results(tmp_path, monkeypatch) -> None:
    """The CLI-facing path validates relocated evidence without executing agents."""
    from mha_exp_level2_bw.exp2_6_direct import batch, results
    from mha_exp_level2_bw.exp2_5 import runner as frozen_runner

    states, env, config = exercise(tmp_path, False, frozen_policy="transfer")
    config.update(run=0, sources={})
    config["identity"] = runtime.digest({key: value for key, value in config.items() if key != "identity"})
    source = tmp_path / "source"
    directory = source / "run-00000"
    short = config["identity"][:12]
    agent, environment = f"exp_direct_bw_{short}", f"exp_direct_env_{short}"
    for name, value in states.items():
        path = directory / agent / "out" / f"{agent}.{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    path = directory / environment / "out" / f"{environment}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(env))
    for name in (agent, environment):
        (directory / f"{name}.log").write_text("completed")
    (directory / "config.json").write_text(json.dumps(config))
    result = runner.check_results(states, env, config, tmp_path)
    (directory / "result.json").write_text(json.dumps(result))
    status = {"status": "completed", "run_ids": [0], "frozen_policy": "transfer",
              "duration_seconds": config["profile"]["duration_seconds"], "source_identity": {},
              "runs": [{"run": 0, "result": result, "directory": "/remote/old-path"}]}
    (source / "batch.json").write_text(json.dumps(status))
    destination = tmp_path / "final"
    results.export_batch(source, destination)
    monkeypatch.setattr(runner, "execute", lambda *args: pytest.fail("must not execute"))
    batch.run_batch(exp_path=destination, process_only=True)
    frozen_runner.run_batch(exp_path=destination, process_only=True)
    with pytest.raises(FileExistsError):
        results.export_batch(source, destination)
    with pytest.raises(ValueError, match="selection"):
        results.process_batch(destination, [1])
    retained = destination / "run-00000" / agent / "out" / f"{agent}.hlreasoner.json"
    changed = json.loads(retained.read_text())
    changed["results"][0]["success"] = not changed["results"][0]["success"]
    retained.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="validation failed"):
        results.process_batch(destination, [0])
    retained.unlink()
    with pytest.raises(ValueError, match="validation failed"):
        results.process_batch(destination, [0])


def test_hourly_treatments_share_ordinary_goals_and_disjoint_run_streams() -> None:
    """Frozen and runtime agents differ in learning, not ordinary reset cases."""
    profile = runner.Profile(3, 4, 4, 2, 2, 2, 3600)
    runtime_rows = runner.build_schedule(0, profile, production=True)
    ordinary = [row for row in runtime_rows if row["mode"] == "train"]
    frozen = runner.build_schedule(0, profile, production=True, frozen_policy="transfer")
    assert [(r["seed"], r["goal"], r["initial_facts"]) for r in ordinary] == [(r["seed"], r["goal"], r["initial_facts"]) for r in frozen]
    other = runner.build_schedule(1, profile, production=True)
    assert not {r["seed"] for r in runtime_rows} & {r["seed"] for r in other}


def test_hourly_batch_retains_failed_runs_and_continues(tmp_path, monkeypatch) -> None:
    """A failed scientific run does not discard evidence or block the next run."""
    from mha_exp_level2_bw.exp2_6_direct import batch
    calls = []
    monkeypatch.setattr(runner, "source_identity", lambda: {"fixture": "fixed"})
    def prepare(*args, **kwargs):
        args[2].mkdir()
        return {"run": args[3]}
    def execute(config, directory):
        calls.append(config["run"])
        if config["run"] == 0:
            raise RuntimeError("fixture failure")
        return {"operationally_valid": True}
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "execute", execute)
    batch.run_batch(2, tmp_path / "batch", duration_seconds=3600)
    status = json.loads((tmp_path / "batch" / "batch.json").read_text())
    assert calls == [0, 1] and status["status"] == "completed"
    assert status["operationally_valid_runs"] == 1
    assert (tmp_path / "batch" / "run-00000" / "batch-error.json").exists()
    with pytest.raises(FileExistsError):
        batch.run_batch(2, tmp_path / "batch", duration_seconds=3600)


def test_frozen_preparation_does_not_construct_a_learning_configuration(tmp_path, monkeypatch) -> None:
    """The six-module baseline has zero demos and must not validate learner budgets."""
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(runner, "reference", lambda *args: (checkpoint, {"checkpoint_sha256": runner.sha(checkpoint)}))
    monkeypatch.setattr(runner, "source_identity", lambda: {})
    profile = runner.Profile(0, 2, 0, 0, 0, 100, 3600)
    config = runner.prepare(tmp_path, tmp_path, tmp_path / "frozen", 0, "hourly", "cpu", 0,
                            True, profile_override=profile, frozen_policy="transfer")
    assert config["learning"] is None and len(config["schedule"]) == 2


def test_runtime_production_plans_and_learns_from_goal_accomplishments(tmp_path, monkeypatch) -> None:
    """Ordinary goals use HLR planning and Transfer while updating the separate candidate."""
    from concurrent.futures import Future
    class InlinePlannerPool:
        def __init__(self, **kwargs): pass
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future
        def shutdown(self, **kwargs): pass
    plan = fixture_schedule()[0]["plan"]
    monkeypatch.setattr(runtime, "ThreadPoolExecutor", InlinePlannerPool)
    monkeypatch.setattr(runtime, "production_planner", lambda: SimpleNamespace(
        solve=lambda *args: SimpleNamespace(actions=plan, accepted=True, elapsed_seconds=0.01, engine="fixture")))
    states, env, config = exercise(tmp_path, False, production=True)
    report = runner.check_results(states, env, config, tmp_path)
    assert report["operationally_valid"], report
    assert report["production"]["accomplished"] == 2
    assert report["production"]["planner_calls"] == 2
    assert report["production"]["planner_bypasses"] == 0
    assert states["learner"]["trained_episodes"] == 2
    changed = deepcopy(states)
    next(r for r in changed["hlreasoner"]["results"] if r["mode"] == "train")["executor"] = "adopted"
    assert not runner.check_results(changed, env, config, tmp_path)["operationally_valid"]


def test_deadline_closes_cleanly_and_reports_partial_schedule(tmp_path) -> None:
    """A requested deadline is an explicit partial outcome, not automatic retry."""
    states, env, config = exercise(tmp_path, False, timed=True)
    report = runner.check_results(states, env, config, tmp_path)
    assert report["operationally_valid"], report
    assert report["outcome"] == "time-limit" and not report["completed_schedule"]
    assert states["hlreasoner"]["index"] == 0 and env["closed"]
    assert runner.PROFILES["main"].duration_seconds == 3600


def test_interrupted_learning_is_not_reported_as_completed(tmp_path) -> None:
    """A missing warm-up response must not become a successful schedule result."""
    states, env, config = exercise(tmp_path, False)
    states["hlreasoner"].update(phase="learning", pending=None, learning_id="warm")
    states["learner"].update(optimizer_steps=0, trained_episodes=0)
    report = runner.check_results(states, env, config, tmp_path)
    assert not report["operationally_valid"]
    assert report["outcome"] == "incomplete"
    assert not report["completed_schedule"]


def test_memory_delivers_main_batch_as_ordered_individual_episodes(tmp_path) -> None:
    """All 600 requested episodes retain identity without one oversized message."""
    ids = [f"demo-{index}" for index in range(600)]
    module = runtime.Memory(module_id="memory", initial_state=runtime.initial_states()["memory"])
    module.on_init(output_dir=str(tmp_path))
    sent = []
    state = State(pending={"ids": ids, "learning_id": "warm"}, responses=0)
    state.directory = SimpleNamespace(internal=SimpleNamespace(learning=[SimpleNamespace(module_id="learner")]))
    state.outbox = SimpleNamespace(send_memories=lambda *args, **kwargs: sent.append((args, kwargs)))
    for identifier in ids:
        path = tmp_path / f"{identifier}.json"
        path.write_text(json.dumps({"result": {"id": identifier}, "rows": []}))
        module._episodes[identifier] = path
    module._reply(state)
    assert state.pending is None and state.responses == 1
    assert len(sent) == 600
    for index, (args, metadata) in enumerate(sent):
        assert args[0] == "learner" and len(args[1]) == 1
        assert args[1][0].content["result"]["id"] == ids[index]
        assert metadata == {"learning_id": "warm", "memory_index": index}


def test_adoption_requires_perfect_probes_fewer_actions_and_same_checkpoint() -> None:
    """Weak, slower, early, or mixed-checkpoint candidates cannot replace Transfer."""
    rows = []
    for index in range(2):
        base = {"mode": "transfer", "case_index": index, "success": True, "actions": [0, 1, 2, 3],
                "seed": index, "goal": {"top": "b0", "bottom": "b1"}}
        rows.extend([base, {**base, "mode": "probe", "probe": 100, "actions": [0, 1],
                            "revision": 101, "model_sha256": "same"}])
    assert runtime.adoption_evidence(rows, 100, 2)["passed"]
    for key, value in (("success", False), ("actions", [0] * 8), ("model_sha256", "different"), ("seed", 42)):
        changed = deepcopy(rows)
        changed[-1][key] = value
        assert not runtime.adoption_evidence(changed, 100, 2)["passed"]
    assert not runtime.adoption_evidence(rows, 0, 2)["passed"]


def test_adopted_executor_stays_frozen_when_candidate_keeps_learning() -> None:
    """Production must never silently switch to an untested learner revision."""
    values = runtime.initial_states()["llreasoner"]
    state = State(deepcopy(values))
    state.directory = SimpleNamespace(internal=SimpleNamespace(goals=[SimpleNamespace(module_id="goalgraph")]))
    state.revision, state.model_sha256 = 5, "evaluated"
    module = runtime.LLReasoner(module_id="ll", initial_state=values)
    module.model = torch.nn.Linear(2, 2)
    module.frozen_policy = None
    module._adopted = None
    module._act = lambda *args, **kwargs: None
    task = {"id": "ordinary", "mode": "train", "seed": 1000, "executor": "adopted",
            "adoption": {"revision": 5, "sha256": "evaluated"}}
    module.on_goal_update(state, "goalgraph", [runtime.Goal([], task=task)])
    admitted = module._adopted.weight.detach().clone()
    with torch.no_grad():
        module.model.weight.add_(1)
    state.active, state.pending, state.revision, state.model_sha256 = None, None, 6, "untested"
    module.on_goal_update(state, "goalgraph", [runtime.Goal([], task=task)])
    assert torch.equal(module._adopted.weight, admitted)
    assert not torch.equal(module.model.weight, admitted)


def test_production_schedule_does_not_plan_outside_agent_timeout() -> None:
    def forbidden(*args):
        raise AssertionError("Planning must happen in the live HLR.")
    schedule = runner.build_schedule(0, runner.PROFILES["smoke"],
                                     SimpleNamespace(solve=forbidden), production=True)
    assert len(schedule) == 16
    assert all(not task["plan"] for task in schedule)


@pytest.mark.parametrize("reverse_knowledge", [False, True])
def test_all_eight_modules_complete_with_both_arrival_orders(tmp_path, reverse_knowledge) -> None:
    states, env, config = exercise(tmp_path, reverse_knowledge)
    report = runner.check_results(states, env, config, tmp_path)
    assert report["operationally_valid"], report
    assert states["learner"]["optimizer_steps"] == 8
    assert states["learner"]["revision"] == 3
    assert states["llreasoner"]["model_installs"] == 4
    assert {item["mode"] for item in states["memory"]["episodes"]} == {"demo", "train"}
    assert len(report["comparisons"]) == 4
    changed = deepcopy(states)
    changed["llreasoner"]["model_installs"] -= 1
    assert not runner.check_results(changed, env, config, tmp_path)["operationally_valid"]


def test_pretrained_assessment_gate_fails_before_preparation(tmp_path) -> None:
    candidate, assessment = tmp_path / "candidate", tmp_path / "assessment"
    candidate.mkdir()
    assessment.mkdir()
    (candidate / "report.json").write_text(json.dumps({"status": "running"}))
    (assessment / "assessment.json").write_text(json.dumps({"status": "started"}))
    with pytest.raises(ValueError, match="Pretraining has not completed"):
        runner.prepare(candidate, assessment, tmp_path / "run", 0, "smoke", "cuda", 0)
    assert not (tmp_path / "run").exists()


def test_container_gpu_exposure_and_cpu_environment(tmp_path, monkeypatch) -> None:
    captured = {}
    class RecordingOrchestrator:
        INFO = 20
        def __init__(self, **kwargs): captured["orchestrator"] = kwargs
        def add_agent(self, **kwargs): captured["agent"] = kwargs
        def add_environment(self, **kwargs): captured["environment"] = kwargs

    monkeypatch.setattr(runner, "Orchestrator", RecordingOrchestrator)
    config = {"identity": "a" * 64, "profile": asdict(runner.PROFILES["smoke"]), "run": 0,
              "device": "cuda", "gpu_id": 0, "learning": asdict(Config()), "schedule": fixture_schedule()}
    runner.build_orchestrator(config, tmp_path)
    assert captured["orchestrator"]["gpu_device_ids"] == [0]
    assert captured["environment"]["gpu_device_ids"] == "none"
    assert captured["agent"]["learners"].init_kwargs["device"] == "cuda"
    assert captured["orchestrator"]["stop_on_agents_term"] is True


def test_state_reader_ignores_inputs_and_episode_evidence(tmp_path) -> None:
    """Only framework state names belong in the eight-module result checker."""
    (tmp_path / "bw_atomic_input").mkdir()
    output = tmp_path / "agent" / runtime.Orchestrator.SAVE_SUBDIR
    output.mkdir(parents=True)
    expected = runtime.initial_states()
    for name, value in expected.items():
        (output / f"agent.{name}.json").write_text(json.dumps(value))
    (output / "episode-train-0.json").write_text("not a module state")
    environment = tmp_path / "env" / runtime.Orchestrator.SAVE_SUBDIR
    environment.mkdir(parents=True)
    (environment / "env.json").write_text('{"closed": true}')
    assert runner.read_states(tmp_path, "agent", "env") == (expected, {"closed": True})


@pytest.mark.skipif(sys.platform != "linux", reason="Container namespace installation uses Linux sh.")
def test_actual_framework_packaging_imports_in_isolation(tmp_path, monkeypatch) -> None:
    """Exercise the framework copier and installation script without editable imports."""
    captured = {}
    class RecordingOrchestrator:
        INFO = 20
        def __init__(self, **kwargs): pass
        def add_agent(self, **kwargs): captured.update(kwargs)
        def add_environment(self, **kwargs): pass

    monkeypatch.setattr(runner, "Orchestrator", RecordingOrchestrator)
    inputs = tmp_path / "bw_atomic_input"
    inputs.mkdir()
    (inputs / "__init__.py").write_text('"""Test comparator package."""')
    config = {"identity": "b" * 64, "profile": asdict(runner.PROFILES["smoke"]), "run": 0,
              "device": "cuda", "gpu_id": 0, "learning": asdict(Config()), "schedule": fixture_schedule()}
    runner.build_orchestrator(config, tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    packager = runtime.Orchestrator.__new__(runtime.Orchestrator)
    copied = {}
    packager._copy_runtime_python_modules([runtime.LLReasoner], staged, copied)
    packager._copy_explicit_runtime_sources(captured["extra_runtime_sources"], staged, copied)
    subprocess.run(["sh", str(captured["init_script"]), str(staged)], check=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(staged) + os.pathsep + str(Path(runner.mhagenta.__file__).parents[1])
    subprocess.run([sys.executable, "-c", "from pathlib import Path; "
        "from mha_exp_level2_bw.exp2_6_direct import runtime; "
        "from mha_exp_level2_bw.achieve_on import policy; "
        "assert Path(policy.__file__).resolve().is_relative_to(Path.cwd()); "
        "assert policy.artifact_paths()[1].is_file()"], cwd=staged, env=environment, check=True)
