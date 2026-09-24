"""Exact-target resource semantics, randomized cases, and bounded recovery regressions."""

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/exp2_5_cr_preparation"))
from exp2_5_harness import RuntimeHarness
from mha_exp_common.names import HLREASONER, LLREASONER
from mha_exp_level2_cr.exp2_5 import runtime
from mha_exp_level2_cr.exp2_5.contracts import resource_collected, resource_stagnated
from resource_cases import (
    SPLIT_SEEDS,
    ResourceCases,
    ResourceEnvironment,
    case_request,
    navigation_request,
)
from test_exp2_5_runtime import public_state


def test_exact_collection_uses_event_not_inventory():
    assert resource_collected(5, (0, 0), (0, -1), (0, -1), "water", ["collect_drink"], False)
    assert not resource_collected(5, (0, 0), (1, 0), (0, -1), "water", ["collect_drink"], False)
    assert not resource_collected(5, (0, 0), (0, -1), (0, -1), "water", [], False)
    assert not resource_collected(5, (0, 0), (0, -1), (0, -1), "water", ["collect_drink"], True)


def test_eight_step_loop_and_progressing_detour():
    history = [{"source": (i % 2, 0), "destination": ((i + 1) % 2, 0),
                "discovered": False, "collected": False} for i in range(8)]
    assert resource_stagnated(history)
    assert not resource_stagnated(history[:7])
    for flag in ("discovered", "collected"):
        changed = deepcopy(history)
        changed[-1][flag] = True
        assert not resource_stagnated(changed)
    history[-1]["destination"] = (2, 0)
    assert not resource_stagnated(history)


def test_split_ranges_and_balanced_requests():
    bases = [base for policy in SPLIT_SEEDS.values() for base in policy.values()]
    ranges = [{base + i for i in range(256)} for base in bases]
    assert all(not first & second for i, first in enumerate(ranges) for second in ranges[i + 1:])
    requests = [case_request(index) for index in range(240)]
    assert {kind: sum(row[0] == kind for row in requests) for kind in {row[0] for row in requests}} == {
        "tree": 40, "water": 40, "stone": 40, "coal": 40, "iron": 40, "diamond": 40}
    assert [sum(row[1] == source for row in requests) for source in ("natural", "table", "furnace")] == [120, 60, 60]
    for count in (240, 600):
        requests = [case_request(index) for index in range(count)]
        assert {sum(row[2] == band for row in requests) for band in {row[2] for row in requests}} == {count // 4}
    navigation = [navigation_request(index) for index in range(240)]
    assert [sum(row[0] == source for row in navigation) for source in ("natural", "table", "furnace")] == [120, 60, 60]
    assert {sum(row[1] == band for row in navigation) for band in {row[1] for row in navigation}} == {80}
    assert {sum(row[2] == direction for row in navigation) for direction in range(1, 5)} == {60}
    assert [sum(row[3] == variant for row in navigation)
            for variant in ("clear", "obstructed")] == [160, 80]


def test_natural_case_reconstruction_preserves_pixels_and_goal(tmp_path):
    cases = ResourceCases(tmp_path, "pilot")
    first, goal, record = cases.get(0)
    second, replayed, repeated = cases.get(0)
    np.testing.assert_array_equal(first.render(), second.render())
    assert goal.target_cell == replayed.target_cell and record == repeated
    assert first._player.inventory["wood"] < 9


def test_resource_replays_are_independent_of_object_set_order(monkeypatch):
    """Native creature balancing must replay identical states despite different object allocations."""

    from cases import make_env

    first, second = [make_env(300003200, environment_type=ResourceEnvironment) for _ in range(2)]
    first.reset()
    second.reset()
    original = second._balance_chunk
    monkeypatch.setattr(second, "_balance_chunk", lambda chunk, objs: original(chunk, list(reversed(list(objs)))))
    prefix = [1, 1, 1, 3, 3, 5, 1, 5, 3, 5, 8, 11, 1, 3, 3, 5, 2, 5, 12, 1, 5, 2,
              5, 1, 1, 5, 3, 5, 1, 3, 5, 1, 4, 5, 2, 5, 2, 4, 4, 5, 1, 5, 1, 1]
    for action in prefix:
        first.step(action)
        second.step(action)
        np.testing.assert_array_equal(first._world._mat_map, second._world._mat_map)
        np.testing.assert_array_equal(first._world._obj_map, second._world._obj_map)
    np.testing.assert_array_equal(first.render(), second.render())


def test_moving_cow_blocks_expert_without_resampling_case(tmp_path, monkeypatch):
    """A route blocked after the first move retains a failed demonstration, not a harness crash."""

    import torch
    import cases as case_helpers
    from resource_training import resource_episode

    original = case_helpers.expert_action
    actions = []

    def block_after_first_action(env, case, first_direction=None):
        if actions:
            raise ValueError("No safe route to interaction target.")
        action = original(env, case, first_direction)
        actions.append(action)
        return action

    monkeypatch.setattr(case_helpers, "expert_action", block_after_first_action)
    cases = ResourceCases(tmp_path, "train")
    rows, success, stats = resource_episode(torch, None, cases, 8, "cpu", expert=True)
    assert not success and stats["expert_route_blocked"] == 1
    assert len(rows) == 1 and rows[-1].done and rows[-1].n_done
    assert rows[0].expert and rows[0].imitation_action == rows[0].action == actions[0]
    assert stats["world_seed"] == 30000128 and stats["generation_rejections"] == 0


def test_failed_recovery_rotates_and_is_not_immediately_preempted(monkeypatch):
    harness = RuntimeHarness()
    state = harness.states[HLREASONER]
    belief = public_state()
    belief["inventory"].update(drink=0, food=1, energy=2)
    belief["native_action_count"] = 8
    state.update(latest_beliefs=belief, recovery="drink", recovery_failed=True, recovery_start=0)
    monkeypatch.setattr(runtime, "choose_step", lambda belief, goal, recovery, *args: (
        runtime.ActivitySpec(goal, runtime.ActivityId.EXPLORE, f"recovery:{recovery}", "", None,
                             belief["revision"], len(belief["known_cells"]), len(belief["known_cells"]) + 10), None))
    harness.modules[HLREASONER]._dispatch(state)
    assert state["recovery"] == "food" and state["deferred_needs"] == ["drink"]
    assert state["active_goal"]["extras"]["max_actions"] == 16
    ll = harness.states[LLREASONER]
    ll.update(current_goal=state["active_goal"], belief_state=belief)
    assert harness.modules[LLREASONER]._outcome(ll, {"status": {"illegal_action": False}}) is None


@pytest.mark.parametrize("movement,retained", [(0, True), (2, True), (3, False)])
def test_target_deferral_requires_real_position_change(monkeypatch, movement, retained):
    harness = RuntimeHarness()
    state = harness.states[HLREASONER]
    belief = public_state()
    belief["terrain"]["2,0"] = "tree"
    belief["player"] = [movement, 0]
    state.update(latest_beliefs=belief, target_deferrals=[{
        "key": "resource:tree:(2, 0)", "target": [2, 0], "source": [0, 0], "material": "tree"}])
    monkeypatch.setattr(runtime, "choose_step", lambda *args: ({"action": 0, "purpose": "test"}, None))
    harness.modules[HLREASONER]._dispatch(state)
    assert bool(state["target_deferrals"]) is retained


@pytest.mark.parametrize("validation_passes,test_passes,phase", [
    (True, True, "selected"), (False, True, "failed_validation"), (True, False, "failed_test")])
def test_randomized_training_resume_and_single_test(monkeypatch, tmp_path, validation_passes, test_passes, phase):
    """Persist transition/optimizer counters and never test rejected validation weights."""

    import resource_training as training
    import torch
    from test_exp2_5_policy_preparation import _completed_rows

    policy = training.PolicyId.NAVIGATE_TO
    configuration = {**training.RANDOMIZED_CONFIGS[policy], "demo_steps": 4, "online_steps": 8,
                     "pretraining_steps": 2, "interval": 8, "minimum_steps": 8,
                     "validation_cases": 1, "test_cases": 1}
    monkeypatch.setitem(training.RANDOMIZED_CONFIGS, policy, configuration)
    monkeypatch.setattr(training, "ResourceCases", lambda root, split, policy: SimpleNamespace(
        get=lambda index: None, split=split, policy=policy))
    monkeypatch.setattr(training, "optimize", lambda *args: {"loss": 0.0})
    stop = SimpleNamespace(requested=False)
    seen = []
    evaluations = []

    def episode(torch, model, cases, index, device, *, expert=False, epsilon=0.0,
                execution_index=None):
        seen.append(index)
        if index == 1:
            stop.requested = True
        stats = {"source": "natural", "kind": "safe_cell", "world_seed": index, "distance": 1,
                 "distance_band": "1-2", "direction": 1, "variant": "clear",
                 "illegal_actions": 0, "lethal_actions": 0, "stagnation": 0,
                 "wrong_target_collections": 0, "generation_rejections": 0,
                 "case_seconds": 0.0, "execution_seconds": 0.0}
        return _completed_rows(policy, expert=expert), True, stats

    def evaluation(torch, model, cases, device, count):
        evaluations.append(cases.split)
        return {"succeeded": 1, "attempted": 1, "mean_cost": 4.0,
                "passed": validation_passes if cases.split == "validation" else test_passes}

    monkeypatch.setattr(training, "randomized_episode", episode)
    monkeypatch.setattr(training, "evaluate_randomized", evaluation)
    state = training.train_randomized(torch, tmp_path, policy, "cpu", False, stop, None)
    assert state["online_steps"] == 4 and state["case_index"] == 2
    assert state["pretraining_steps"] == 2 and state["optimizer_steps"] == 3
    stop.requested = False
    state = training.train_randomized(torch, tmp_path, policy, "cpu", True, stop, None)
    assert state["phase"] == phase and seen == list(range(3))
    assert evaluations == ["validation"] + (["test"] if validation_passes else [])
    assert (tmp_path / "candidates/navigate-to-policy.pt").exists() is (phase == "selected")
    training.train_randomized(torch, tmp_path, policy, "cpu", True, stop, None)
    assert evaluations.count("test") == int(validation_passes)


def test_recovery_turn_expires_across_goal_boundaries(monkeypatch):
    """A new goal does not reset the native-action budget of an active recovery turn."""

    harness = RuntimeHarness()
    state = harness.states[HLREASONER]
    belief = public_state()
    belief["inventory"].update(drink=0, food=1, energy=2)
    belief["native_action_count"] = 20
    state.update(latest_beliefs=belief, recovery="drink", recovery_start=4)
    monkeypatch.setattr(runtime, "choose_step", lambda *args: ({"action": 0, "purpose": "test"}, None))
    harness.modules[HLREASONER]._dispatch(state)
    assert state["recovery"] == "food" and state["recovery_start"] == 20
    assert state["deferred_needs"] == ["drink"]
