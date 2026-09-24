"""Behavioral checks for the current continuous native-action learning treatment."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from mha_exp_level2_cr.exp2_6.online_environment import episode_seed
from mha_exp_level2_cr.exp2_6.online_learning import LearningConfig, Trainer, transitions
from mha_exp_level2_cr.exp2_6.online_planning import epsilon
from mha_exp_level2_cr.exp2_6.online_policy import load_basics, candidate, weights_hash
from mha_exp_level2_cr.exp2_6.online_runner import check_results, run_batch
from exp2_6_online_harness import RuntimeHarness


@pytest.mark.parametrize("reason", ["action_budget", "death", "diamond"])
def test_all_environment_endings_reset_without_ending_agent(tmp_path, reason):
    from mha_exp_level2_cr.exp2_6.online_environment import ContinuingEnvironment, initial_state
    env = ContinuingEnvironment(initial_state(19, "agent", str(tmp_path)))
    if reason == "action_budget":
        env.state["episode_actions"] = 899
    elif reason == "death":
        env._crafter._player.inventory.update(health=0, food=0, drink=0, energy=0)
    else:
        env._crafter._player.inventory["diamond"] = 1
        env._crafter._player.achievements["collect_diamond"] = 1
    _, status = env.on_action(env.state, "agent", action=0, environment_atomic_id=1,
                              owner_action_id="a", requester_kind="hl_primitive")
    assert status["done"] and status["episode_reason"] == reason
    _, first = env.on_action(env.state, "agent", action="reset", control_id="r1")
    assert first["contract_error"] is None and not first["closed"]
    assert env.state["episode_actions"] == 0 and env.state["native_action_count"] == 1
    assert env.state["achievement_counts"] == {} and env.state["environment_seed"] == 1119
    _, duplicate = env.on_action(env.state, "agent", action="reset", control_id="r1")
    assert duplicate == first and env.state["resets"] == 1
    _, wrong = env.on_action(env.state, "agent", action="close", control_id="r1")
    assert wrong["contract_error"] == "reused-control-id"


def test_eat_target_has_safe_trial_opportunities_before_urgent_food_recovery():
    from mha_exp_level2_cr.exp2_6.online_runtime import HLReasoner, initial_states
    from mha_exp_level2_cr.exp2_5.beliefs import initial_belief_state
    hl = HLReasoner(module_id="hl_reasoner")
    state = initial_states()["hl_reasoner"]
    state["snapshot"] = {"actions": 0, "episode": 0}
    state["skills"]["eat_target"].update(teacher_successes=4, opportunities=3)
    belief = initial_belief_state()
    belief.update(revision=1, facing="right", inventory={"health": 9, "food": 6, "drink": 9, "energy": 9},
                  safe_cells=["0,0", "0,1"], known_cells=["0,0", "0,1", "1,0"],
                  visible_cells=["0,0", "0,1", "1,0"],
                  terrain={"0,0": "grass", "0,1": "grass", "1,0": "grass"},
                  occupants={"1,0": {"kind": "cow", "last_seen_revision": 1}})
    command = hl._choose(state, belief)
    assert command["kind"] == "learned" and command["session"]["mode"] == "probe"
    assert command["session"]["skill"] == "eat_target"


def test_admission_requires_one_frozen_revision_and_failure_revokes_it():
    from mha_exp_level2_cr.exp2_6.online_runtime import HLReasoner, initial_states
    hl = HLReasoner(module_id="hl_reasoner")
    state = initial_states()["hl_reasoner"]
    state["snapshot"] = {"actions": 50}
    result = {"skill": "get_resource", "success": True, "mode": "probe", "model_sha256": "a" * 64,
              "steps": 5, "reason": "success", "initial_target": [1, 0], "purpose": "technology:table"}
    for _ in range(10):
        hl._account(state, result)
    assert state["skills"]["get_resource"]["adopted"] == 0
    hl._account(state, {**result, "mode": "adopted", "success": False, "reason": "action_bound"})
    assert state["skills"]["get_resource"]["adopted"] is None
    hl._account(state, result)
    for _ in range(8):
        hl._account(state, result)
    with pytest.raises(ValueError, match="mixed model"):
        hl._account(state, {**result, "model_sha256": "b" * 64})


def test_cli_entry_resolves_current_treatment(monkeypatch):
    from mha_exp_level2_cr import exp2_6
    from mha_exp_level2_cr.exp2_6 import online_runner, runner
    received = []
    assert runner.run_batch is online_runner.run_batch
    monkeypatch.setattr(runner, "run_batch", lambda **kwargs: received.append(kwargs))
    exp2_6.run_batch(runs=[10, 11], process_only=True)
    assert received == [{"runs": [10, 11], "process_only": True}]


def test_warm_start_is_exact_and_only_allowed_basics_are_loaded():
    torch.set_num_threads(1)
    basics = load_basics(torch)
    assert set(basics) == {"explore", "navigate_to"}
    new = candidate(torch, basics["navigate_to"])
    assert weights_hash(new) == weights_hash(basics["navigate_to"])
    frames = torch.randint(0, 256, (3, 3, 64, 64), dtype=torch.uint8)
    contexts = torch.tensor([[0.0, 0.0], [0.02, -0.03], [0.05, 0.02]])
    assert torch.equal(new(frames, contexts), basics["navigate_to"](frames, contexts))


def test_episode_seed_streams_are_disjoint_and_exploration_is_need_gated():
    seeds = [episode_seed(run, episode) for run in range(20) for episode in range(1000)]
    assert len(seeds) == len(set(seeds))
    assert [episode_seed(run, 0) for run in range(20)] == list(range(1000, 1020))
    assert [episode_seed(run, 1) for run in range(20)] == list(range(1100, 1120))
    assert epsilon(0, teacher=True, safe=True) == 0.05
    assert epsilon(1, teacher=False, safe=True) == pytest.approx(0.02)
    assert epsilon(0, teacher=False, safe=False) == 0
    assert epsilon(0, teacher=False, safe=True, evaluation=True) == 0


def test_main_cli_rejects_transition_like_or_shortened_budgets(tmp_path):
    with pytest.raises(ValueError, match="60-minute"):
        run_batch(exp_path=tmp_path / "batch", duration_seconds=120)
    with pytest.raises(ValueError, match="0–19"):
        run_batch(runs=[20], exp_path=tmp_path / "batch")


def test_real_eight_module_loop_trains_and_closes(tmp_path):
    """Validate live and final-only evidence, rejecting corrupted final weights."""
    assert torch.__version__.split("+")[0] == "2.14.0", "Run with the isolated Torch 2.14 validation environment"
    harness = RuntimeHarness(tmp_path)
    harness.run(actions=80)
    assert harness.terminated == "execution-time-complete", harness.terminated
    states = harness.clean_states()
    assert states["learner"]["skills"]["get_resource"]["updates"] > 0
    assert states["ll_reasoner"]["model_installs"] > 0
    result = check_results(states, harness.env.state, {"run": 0, "duration_seconds": 3600, "device": "cpu"},
                           tmp_path / "agent", tmp_path / "environment")
    assert result["execution_valid"], result["errors"]
    for skill in ("get_resource", "eat_target"):
        trainer_path = tmp_path / "agent" / f"{skill}-trainer.pt"
        saved = torch.load(trainer_path, weights_only=False)
        torch.save({"model": saved["model"], "summary": saved["summary"]},
                   tmp_path / "agent" / f"{skill}-policy.pt")
        trainer_path.unlink()
    final = check_results(states, harness.env.state, {"run": 0, "duration_seconds": 3600, "device": "cpu"},
                          tmp_path / "agent", tmp_path / "environment")
    assert final == result
    policy_path = tmp_path / "agent" / "get_resource-policy.pt"
    policy = torch.load(policy_path, weights_only=True)
    next(iter(policy["model"].values())).add_(1)
    torch.save(policy, policy_path)
    corrupted = check_results(states, harness.env.state, {"run": 0, "duration_seconds": 3600, "device": "cpu"},
                              tmp_path / "agent", tmp_path / "environment")
    assert not corrupted["execution_valid"]
    assert "final-trainer-weights-mismatch" in corrupted["errors"]


def test_replay_excludes_probes_and_terminal_suffixes_stop_bootstrapping():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    row = {"state": frame, "next": frame, "context": np.zeros(2), "next_context": np.zeros(2),
           "mask": np.array([False, True, False, False, False, False]),
           "next_mask": np.array([False] * 6), "terminal": True, "action": 1, "reward": -1.0}
    segment = {"rows": [row], "mode": "trial", "success": False}
    converted = transitions(segment)
    assert converted[0]["n_reward"] == -1 and converted[0]["n_terminal"]
    assert not transitions({**segment, "mode": "teacher", "success": True,
                            "rows": [{**row, "exploratory": True}]})[0]["demo"]
    trainer = Trainer(torch, load_basics(torch)["navigate_to"], 2, "cpu", LearningConfig(batch_size=2))
    trainer.ingest(segment)
    trainer.optimize(0.5)
    assert trainer.seen == 1 and trainer.updates == 1 and np.isfinite(trainer.last_loss)
    # Probe exclusion is enforced at the module boundary before Trainer.ingest.
    with pytest.raises(ValueError, match="Evaluation"):
        trainer.ingest({**segment, "mode": "probe"})


@pytest.mark.parametrize("memory_first", [True, False])
def test_memory_request_and_segment_can_arrive_in_either_order(tmp_path, memory_first):
    """Real typed Memory outboxes deliver one archived segment without duplication."""
    from collections import deque
    from types import SimpleNamespace
    from mhagenta import State, Observation
    from mha_exp_level2_cr.exp2_6.online_runtime import Memory, initial_states
    from exp2_6_online_harness import outbox
    harness = SimpleNamespace(queue=deque(), terminated=None)
    directory = SimpleNamespace(internal=SimpleNamespace(
        knowledge=[SimpleNamespace(module_id="knowledge")], learning=[SimpleNamespace(module_id="learner")]))
    memory = Memory(module_id="memory")
    memory.on_init(output=str(tmp_path))
    state = State(agent_id="agent", module_id="memory", time_func=lambda: 0,
                  directory=directory, outbox=outbox(harness, "memory"), **initial_states()["memory"])
    segment = {"id": "segment-1", "rows": [{"terminal": True}]}
    supply = lambda: memory.on_observation_update(state, "knowledge", [Observation(None)], segment=segment)
    request = lambda: memory.on_memory_request(state, "learner", segment_id=segment["id"])
    for action in (supply, request) if memory_first else (request, supply):
        action()
    assert state["failure"] is None and state["delivered"] == 1 and state["stored"] == 0 and state["pending"] is None
    receiver, callback, payload = harness.queue.popleft()
    assert receiver == "learner" and callback == "on_memories"
    assert payload["memories"][0].content == segment


def test_monitor_flags_reported_faults_and_stale_autosaves(tmp_path):
    """Monitoring is a bounded read-only snapshot, not an automatic kill decision."""
    import json
    import os
    import time
    from tools.monitor_exp2_6_cr import snapshot
    (tmp_path / "batch.json").write_text(json.dumps({"run_ids": [0], "runs": [], "status": "running", "current_run": 0}))
    run = tmp_path / "run-00000"
    output = run / "agent" / "out"
    output.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"agent_id": "agent", "env_id": "env"}))
    path = output / "agent.ll_reasoner.json"
    path.write_text(json.dumps({"actions": 200, "episode": 0, "exploratory_actions": 5,
                               "model_installs": 8, "waiting_model": None, "failure": None}))
    assert not snapshot(tmp_path)["needs_inspection"]
    os.utime(path, (time.time() - 200, time.time() - 200))
    assert snapshot(tmp_path)["needs_inspection"]
    (output / "agent.learner.json").write_text(json.dumps({"failure": "nonfinite loss"}))
    assert snapshot(tmp_path)["faults"] == [{"module": "learner", "error": "nonfinite loss"}]


def test_heartbeat_stops_only_assigned_service_and_containers(tmp_path, monkeypatch):
    """A reported execution fault freezes its controller and exact owned containers."""
    import json
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "tools"))
    import heartbeat_exp2_6_cr as heartbeat
    calls = []
    monkeypatch.setattr(heartbeat, "snapshot", lambda root: {
        "status": "running", "run": 19, "faults": ["learner-failure"],
        "containers": ["exp_cr26_agent_abc", "exp_cr26_env_abc", "unrelated"]})
    monkeypatch.setattr(heartbeat, "command", lambda args: calls.append(args) or "active")
    output = tmp_path / "heartbeat.json"
    monkeypatch.setattr(sys, "argv", ["heartbeat", str(tmp_path), "--unit", "cr26-main-4", "--output", str(output)])
    heartbeat.main()
    assert calls == [["systemctl", "is-active", "cr26-main-4"], ["systemctl", "stop", "cr26-main-4"],
                     ["docker", "stop", "--time", "30", "exp_cr26_agent_abc"],
                     ["docker", "stop", "--time", "30", "exp_cr26_env_abc"]]
    assert json.loads(output.read_text())["heartbeat_action"] == "stopped-assigned-job"
