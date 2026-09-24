"""Five-policy runtime, passive synchronization, and progression tests."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from exp2_5_harness import HarnessState, RuntimeHarness
from mha_exp_common.names import ACTUATOR, HLREASONER, LLREASONER
from mha_exp_level2_cr.exp2_5 import runtime
from mha_exp_level2_cr.exp2_5.beliefs import (
    CrafterPercept,
    Direction,
    ObservedTile,
    abstract_beliefs,
    available_movement_actions,
    initial_belief_state,
    movement_evidence,
    novel_movement_actions,
    parse_abstract_beliefs,
    revise_belief_state,
)
from mha_exp_level2_cr.exp2_5.contracts import (
    ActivityId,
    ActivitySpec,
    activity_from_goal,
    goal_to_dict,
    make_activity_goal,
)
from mha_exp_level2_cr.exp2_5.deliberation import choose_step, failure_key, urgent_need
from mhagenta import ActionStatus, Observation


def public_state():
    state = initial_belief_state()
    state.update(revision=1, facing="right", inventory={
        "health": 9, "food": 9, "drink": 9, "energy": 9,
    })
    state["terrain"] = {f"{x},{y}": "grass" for x in range(-4, 5) for y in range(-3, 4)}
    for key in ("safe_cells", "known_cells", "visible_cells"):
        state[key] = sorted(state["terrain"])
    return state


@pytest.mark.parametrize("target,previous,current,events,expected_target,expected_status", [
    ((1, 0), [(1, 0), (4, 0)], [(4, 0)], [], None, "failed"),
    ((2, 0), [(1, 0), (2, 0)], [(2, 0), (3, 0)], [], (3, 0), None),
    ((1, 0), [(1, 0), (2, 1)], [(1, 1), (2, 0)], [], None, "failed"),
    ((0, 1), [(1, 0), (0, 1)], [(0, 1)], ["eat_cow"], (0, 1), None),
    ((1, 0), [(1, 0), (4, 0)], [(4, 0)], ["eat_cow"], None, "succeeded"),
    ((1, 0), [(1, 0)], [(1, 0), (2, 0)], [], None, "failed"),
])
def test_eat_target_observation_tracks_and_attributes_completion(
    monkeypatch, target, previous, current, events, expected_target, expected_status,
):
    """The real observation callback tracks jointly and rejects unrelated consumption."""
    belief = public_state()
    belief["occupants"] = {f"{x},{y}": {"kind": "cow", "last_seen_revision": 1} for x, y in previous}
    # An old achievement must never satisfy the new activity.
    belief["achievement_counts"]["eat_cow"] = 3
    state = HarnessState(runtime.initial_states()[LLREASONER])
    state.outbox = SimpleNamespace(send_beliefs=lambda *args, **kwargs: None, send_goal_update=lambda *args, **kwargs: None)
    spec = ActivitySpec("track", ActivityId.EAT_TARGET, "recovery:food", "cow", target, 1, 3, 4)
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)),
                 cow_target=list(target), observation_count=1, awaiting_observation=True,
                 current_goal_action_count=1)
    state["pending_action"] = {
        **movement_evidence(5, belief), "policy_id": "eat_target", "target_cell": list(target),
        "facing": [1, 0], "status": {"illegal_action": False, "contract_error": None,
                                   "new_achievements": events, "done": False, "dead": False},
    }
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    module._perceptor_id, module._knowledge_id, module._goal_graph_id = "perceptor", "knowledge", "goals"
    module._templates = None
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_dispatch", lambda state: None)
    tiles = {(x, y): ObservedTile("grass", "cow" if (x, y) in current else "none")
             for x in range(-4, 5) for y in range(-3, 4)}
    percept = CrafterPercept(False, Direction.RIGHT, belief["inventory"], tiles)
    monkeypatch.setattr(runtime, "ground_observation", lambda *args, **kwargs: (percept, None))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    module.on_observation(state, "perceptor", Observation(frame), observation_id=2,
                          observation_digest=runtime.rgb_sha256(frame))
    assert state["failure"] is None
    assert state["cow_target"] == (list(expected_target) if expected_target is not None else None)
    if expected_status is None:
        assert not state["activities"] and state["current_goal"] is not None
    else:
        assert state["activities"][-1]["status"] == expected_status
        if expected_status == "failed":
            assert state["activities"][-1]["reason"] == "target_lost"


def test_eat_target_dispatch_cannot_reacquire_another_cow(monkeypatch):
    """Missing designated positions fail before another inference is possible."""
    state = runtime.initial_states()[LLREASONER]
    belief = public_state()
    belief["occupants"]["4,0"] = {"kind": "cow", "last_seen_revision": 1}
    spec = ActivitySpec("track", ActivityId.EAT_TARGET, "recovery:food", "cow", (1, 0), 1, 0, 1)
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)), cow_target=[1, 0])
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    outcomes = []
    monkeypatch.setattr(module, "_terminal", lambda state, *result: outcomes.append(result))
    module._dispatch(state)
    assert outcomes == [("failed", "target_lost")]
    assert state["inference_count"] == 0 and state["cow_target"] is None


@pytest.mark.parametrize("activity", list(ActivityId))
def test_five_activity_contracts(activity):
    target = (2, 0) if activity in {ActivityId.NAVIGATE_TO, ActivityId.GET_RESOURCE, ActivityId.EAT_TARGET} else None
    spec = ActivitySpec("one", activity, "test", "tree", target, 1, 0, 1)
    assert activity_from_goal(make_activity_goal(spec)) == spec


def test_current_beliefs_round_trip_and_illegal_primitive():
    state = public_state()
    percept = CrafterPercept(False, Direction.RIGHT, state["inventory"], {(1, 0): ObservedTile("grass", "none")})
    status = {"illegal_action": True, "contract_error": None, "new_achievements": [], "done": False, "dead": False}
    pending = {**movement_evidence(6, state), "status": status}
    revised = revise_belief_state(state, percept, revision=2, pending_action=pending)
    assert revised["player"] == [0, 0]
    assert revised["last_action_status"] == status
    assert parse_abstract_beliefs(abstract_beliefs(revised)) == revised


def test_movement_mask_allows_only_a_needed_turn_toward_an_unsafe_target() -> None:
    state = public_state()
    state["safe_cells"].remove("1,0")
    assert 2 not in available_movement_actions(state)
    state["facing"] = "up"
    assert 2 in available_movement_actions(state, (1, 0))
    state["facing"] = "right"
    assert 2 not in available_movement_actions(state, (1, 0))


def test_movement_mask_rejects_safe_terrain_occupied_by_a_cow() -> None:
    state = public_state()
    state["safe_cells"].remove("1,0")
    state["occupants"]["1,0"] = {"kind": "cow", "last_seen_revision": 1}
    assert 2 not in available_movement_actions(state)


def test_movement_mask_prefers_novel_cells_but_allows_required_backtracking() -> None:
    assert novel_movement_actions((1, 2, 3, 4), (0, 0), [(1, 0)]) == (1, 3, 4)
    assert novel_movement_actions((2,), (0, 0), [(1, 0)]) == (2,)


@pytest.mark.parametrize("status_first", [False, True])
@pytest.mark.parametrize("illegal", [False, True])
def test_passive_cycle_accepts_both_orders_and_keeps_outcome(monkeypatch, status_first, illegal):
    monkeypatch.setattr(runtime, "MAX_TOTAL_ACTIONS", 1)
    monkeypatch.setattr(runtime, "choose_step", lambda *args, **kwargs: ({"action": 6, "purpose": "recovery:energy"}, None))
    harness = RuntimeHarness()
    harness.env._crafter._player.inventory["energy"] = 9 if illegal else 8
    for _ in range(100):
        if harness.terminated:
            break
        if status_first:
            priorities = [i for i, (name, method, _) in enumerate(harness.queue)
                          if name == ACTUATOR or method == "on_action_status"]
            if priorities:
                index = priorities[0]
                harness.queue.rotate(-index)
                message = harness.queue.popleft()
                harness.queue.rotate(index)
                harness.queue.appendleft(message)
        harness.deliver()
    assert harness.terminated == "action_budget"
    ll = harness.states[LLREASONER]
    hl = harness.states[HLREASONER]
    assert ll["inference_count"] == 0
    assert len(ll["trace"]) == 1 and ll["observation_count"] == 2
    assert ll["activities"][0]["status"] == "succeeded"
    assert ll["trace"][0]["status"]["illegal_action"] is illegal
    assert hl["activities"][0]["primitive_status"]["illegal_action"] is illegal
    assert all(state["failure"] is None for state in harness.states.values())
    assert ll["belief_state"]["sleeping"] is not illegal


def test_complete_sleep_wake_cycle(monkeypatch):
    monkeypatch.setattr(runtime, "MAX_TOTAL_ACTIONS", 12)
    monkeypatch.setattr(runtime, "choose_step", lambda state, *args, **kwargs: (
        {"action": 6 if state["inventory"]["energy"] < 9 else 0, "purpose": "recovery:energy"}, None))
    harness = RuntimeHarness()
    harness.env._crafter._player.inventory["energy"] = 8
    harness.run()
    state = harness.states[LLREASONER]
    assert state["inference_count"] == 0
    assert state["belief_state"]["inventory"]["energy"] == 9
    assert not state["belief_state"]["sleeping"]
    assert state["belief_state"]["achievement_counts"]["wake_up"] == 1
    assert all(not row["status"]["illegal_action"] for row in state["trace"])


def test_terminal_primitive_still_completes_passive_synchronization(monkeypatch):
    monkeypatch.setattr(runtime, "choose_step", lambda *args, **kwargs: ({"action": 0, "purpose": "recovery:health"}, None))
    harness = RuntimeHarness()
    harness.env._crafter._length = 1
    harness.run()
    ll = harness.states[LLREASONER]
    assert harness.terminated == "environment_terminal"
    assert ll["trace"][0]["status"]["done"]
    assert ll["activities"][0]["status"] == "succeeded"
    assert len(ll["trace"]) == 1 and ll["observation_count"] == 2
    assert all(state["failure"] is None for state in harness.states.values())


@pytest.mark.parametrize("purpose,expected", [
    ("technology:table", ("interrupted", "survival_interrupt")),
    ("recovery:drink", None),
])
def test_urgent_need_interrupts_other_work_but_not_its_recovery(purpose, expected):
    state = runtime.initial_states()[LLREASONER]
    belief = public_state()
    belief["inventory"]["drink"] = 2
    state["belief_state"] = belief
    spec = ActivitySpec("a", ActivityId.EXPLORE, purpose, "", None, 1, 63, 73)
    state["current_goal"] = goal_to_dict(make_activity_goal(spec))
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    assert module._outcome(state, {"status": {"illegal_action": False}}) == expected


def test_invalid_uncorrelated_status_stops_without_action():
    harness = RuntimeHarness()
    module = harness.modules[LLREASONER]
    module.on_action_status(harness.states[LLREASONER], ACTUATOR, ActionStatus({}))
    assert harness.terminated == "invalid-action-status"
    assert harness.env.state["native_action_count"] == 0


def test_invalid_owner_request_returns_contract_status():
    harness = RuntimeHarness()
    harness.modules[ACTUATOR].on_request(
        harness.states[ACTUATOR], HLREASONER, owner_action_id="hl-bad", requester_kind="hl_primitive", action=5,
    )
    assert harness.terminated == "invalid-action-request"
    assert any(method == "on_action_status" and args["action_status"].status["owner_action_id"] == "hl-bad"
               for _, method, args in harness.queue)
    assert harness.env.state["native_action_count"] == 0


def test_duplicate_passive_status_stops_without_another_observation(monkeypatch):
    monkeypatch.setattr(runtime, "choose_step", lambda *args, **kwargs: ({"action": 0, "purpose": "recovery:health"}, None))
    harness = RuntimeHarness()
    while not any(method == "on_action_status" for _, method, _ in harness.queue):
        harness.deliver()
    index = next(i for i, (_, method, _) in enumerate(harness.queue) if method == "on_action_status")
    name, _, args = harness.queue[index]
    duplicate = deepcopy(args)
    while harness.states[LLREASONER]["status_count"] == 0:
        harness.deliver()
    before = harness.states[LLREASONER]["observation_request_count"]
    harness.modules[name].on_action_status(harness.states[name], **duplicate)
    assert harness.terminated is not None
    assert harness.states[LLREASONER]["observation_request_count"] == before


def test_survival_priorities_and_crafting_steps():
    state = public_state()
    assert urgent_need({"food": 2, "drink": 2, "energy": 2, "health": 9}) == "drink"
    state["terrain"]["2,0"] = "tree"
    state["safe_cells"].remove("2,0")
    step, _ = choose_step(state, "a", None, [], None)
    assert step.activity is ActivityId.GET_RESOURCE and step.target_kind == "tree"
    state["terrain"]["1,1"] = "table"
    state["safe_cells"].remove("1,1")
    state["inventory"].update(wood=1, stone=1, wood_pickaxe=1)
    step, _ = choose_step(state, "b", None, [], None)
    assert step == {"action": 12, "purpose": "technology:stone_pickaxe"}
    state["inventory"]["drink"] = 1
    state["terrain"]["-1,0"] = "water"
    state["safe_cells"].remove("-1,0")
    step, _ = choose_step(state, "c", "drink", [], None)
    assert step.activity is ActivityId.GET_RESOURCE and step.target_kind == "water"


def test_placement_setup_is_grounded_and_bounded():
    state = public_state()
    state["inventory"]["wood"] = 2
    step, placement = choose_step(state, "a", None, [], None)
    assert placement is not None and placement["moves"] <= 2
    assert step["action"] in range(1, 5)
    movement = movement_evidence(step["action"], state)
    state.update(player=movement["destination_cell"], facing=Direction.from_action(step["action"]).name.lower())
    step, placement = choose_step(state, "b", None, [], placement)
    assert step["action"] == 8 and placement is None


def test_excluded_resource_is_not_immediately_retried():
    state = public_state()
    state["terrain"]["2,0"] = "tree"
    state["safe_cells"].remove("2,0")
    excluded = [failure_key("technology:table", (2, 0))]
    step, _ = choose_step(state, "a", None, excluded, None)
    assert step.activity is ActivityId.EXPLORE


def test_eat_cow_runtime_uses_approved_budget() -> None:
    """HLR emits 96, wire parsing accepts it, and LLR stops exactly at 96."""
    from dataclasses import replace
    from mha_exp_level2_cr.exp2_5.contracts import activity_action_bound
    belief = public_state()
    belief['inventory']['food'] = 2
    spec, _ = choose_step(belief, 'cow-budget', 'food', [], None)
    assert spec.activity is ActivityId.EAT_COW and spec.max_actions == 96
    assert activity_from_goal(make_activity_goal(spec)) == spec
    with pytest.raises(ValueError, match='Invalid activity bound'):
        activity_from_goal(make_activity_goal(replace(spec, max_actions=97)))
    other = replace(spec, activity=ActivityId.EXPLORE, max_actions=33)
    with pytest.raises(ValueError, match='Invalid activity bound'):
        activity_from_goal(make_activity_goal(other))
    assert activity_action_bound(ActivityId.EAT_TARGET) == 32
    state = runtime.initial_states()[LLREASONER]
    state['belief_state'] = belief
    state['current_goal'] = goal_to_dict(make_activity_goal(spec))
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    for count in (32, 64, 95):
        state['current_goal_action_count'] = count
        assert module._outcome(state, {'status': {'illegal_action': False}}) is None
    state['current_goal_action_count'] = 96
    assert module._outcome(state, {'status': {'illegal_action': False}}) == ('failed', 'action_bound')


@pytest.mark.parametrize("used", [0, 7, 15])
def test_hlr_dispatch_preserves_eat_cow_budget(used):
    """The actual HLR wire request must retain 96 after earlier recovery work."""
    belief = public_state()
    belief["inventory"]["food"] = 2
    belief["native_action_count"] = used
    sent = []
    state = HarnessState(**runtime.initial_states()[HLREASONER], outbox=SimpleNamespace(
        send_goals=lambda receiver, goals: sent.extend(goals),
    ))
    state.update(latest_beliefs=belief, recovery="food", recovery_start=0)
    module = runtime.CrafterActivityHLReasoner(module_id=HLREASONER, initial_state=state)
    module._goal_graph_id = "goals"
    module.log = lambda *args, **kwargs: None
    module._dispatch(state)
    assert len(sent) == 1
    spec = activity_from_goal(sent[0])
    assert spec.activity is ActivityId.EAT_COW and spec.max_actions == 96


def test_hlr_yields_failed_full_eat_cow_to_another_urgent_need():
    """After 64 food-search actions, another urgent need gets the next recovery turn."""
    belief = public_state()
    belief["inventory"].update(food=1, drink=2)
    belief["native_action_count"] = 64
    sent = []
    state = HarnessState(**runtime.initial_states()[HLREASONER], outbox=SimpleNamespace(
        send_goals=lambda receiver, goals: sent.extend(goals),
    ))
    state.update(latest_beliefs=belief, recovery="food", recovery_failed=True, recovery_start=0)
    module = runtime.CrafterActivityHLReasoner(module_id=HLREASONER, initial_state=state)
    module._goal_graph_id = "goals"
    module.log = lambda *args, **kwargs: None
    module._dispatch(state)
    assert len(sent) == 1 and sent[0].extras["purpose"] == "recovery:drink"
    assert state["deferred_needs"] == ["food"] and state["recovery_start"] == 64


def test_hlr_eat_target_still_respects_remaining_recovery_window():
    """Extending EatCow does not extend the other recovery skills."""
    belief = public_state()
    belief["inventory"]["food"] = 2
    belief["occupants"]["1,0"] = {"kind": "cow"}
    belief["native_action_count"] = 7
    sent = []
    state = HarnessState(**runtime.initial_states()[HLREASONER], outbox=SimpleNamespace(
        send_goals=lambda receiver, goals: sent.extend(goals),
    ))
    state.update(latest_beliefs=belief, recovery="food", recovery_start=0)
    module = runtime.CrafterActivityHLReasoner(module_id=HLREASONER, initial_state=state)
    module._goal_graph_id = "goals"
    module.log = lambda *args, **kwargs: None
    module._dispatch(state)
    spec = activity_from_goal(sent[0])
    assert spec.activity is ActivityId.EAT_TARGET and spec.max_actions == 9


@pytest.mark.parametrize("terminal,expected", [
    (False, ("interrupted", "total_action_bound")),
    (True, ("failed", "environment_terminal")),
])
def test_eat_cow_long_budget_does_not_override_episode_stop(terminal, expected):
    """Global action and terminal-environment stops still interrupt a long search."""
    belief = public_state()
    belief["inventory"]["food"] = 2
    spec, _ = choose_step(belief, "cow-global-stop", "food", [], None)
    belief.update(native_action_count=runtime.MAX_TOTAL_ACTIONS, terminal=terminal)
    state = runtime.initial_states()[LLREASONER]
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)),
                 current_goal_action_count=1)
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    assert module._outcome(state, {"status": {"illegal_action": False}}) == expected


@pytest.mark.parametrize("vitals,should_yield", [
    ({"food": 1, "drink": 0}, True),
    ({"food": 1, "energy": 0}, True),
    ({"food": 0, "drink": 0}, True),
    ({"food": 3, "health": 2}, True),
    ({"food": 0, "drink": 1}, False),
    ({"food": 1, "energy": 1}, False),
    ({"food": 2}, False),
])
def test_eat_cow_yields_only_to_a_higher_priority_recovery(vitals, should_yield):
    """Long searches yield to more urgent needs without restoring a fixed short cap."""
    belief = public_state()
    belief["inventory"]["food"] = 2
    spec, _ = choose_step(belief, "cow-urgent", "food", [], None)
    belief["inventory"].update(vitals)
    state = runtime.initial_states()[LLREASONER]
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)),
                 current_goal_action_count=17)
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    expected = ("interrupted", "survival_interrupt") if should_yield else None
    assert module._outcome(state, {"status": {"illegal_action": False}}) == expected


def test_hlr_reconciles_urgent_cow_interruption_into_water_recovery():
    """The real terminal handshake must dispatch water instead of restarting food search."""
    from mha_exp_level2_cr.exp2_5.contracts import terminal_goal

    belief = public_state()
    belief["inventory"]["food"] = 2
    spec, _ = choose_step(belief, "cow-urgent", "food", [], None)
    requested = make_activity_goal(spec)
    belief.update(revision=9, native_action_count=8)
    belief["inventory"].update(food=1, drink=0)
    terminal = terminal_goal(requested, 9, "interrupted", "survival_interrupt")
    sent = []
    state = HarnessState(**runtime.initial_states()[HLREASONER], outbox=SimpleNamespace(
        send_goals=lambda receiver, goals: sent.extend(goals),
    ))
    state.update(latest_beliefs=belief, recovery="food", recovery_start=0,
                 active_goal=goal_to_dict(requested), pending_terminal=goal_to_dict(terminal))
    module = runtime.CrafterActivityHLReasoner(module_id=HLREASONER, initial_state=state)
    module._goal_graph_id = "goals"
    module.log = lambda *args, **kwargs: None
    module._reconcile(state)
    assert len(sent) == 1 and sent[0].extras["purpose"] == "recovery:drink"
    assert state["deferred_needs"] == ["food"] and state["recovery_start"] == 8
    assert state["failures_by_target"] == {}


@pytest.mark.parametrize("later_vitals,expected", [
    ({"food": 1, "drink": 0}, None),
    ({"food": 0, "drink": 0}, None),
    ({"food": 1, "drink": 1, "energy": 0}, ("interrupted", "survival_interrupt")),
])
def test_eat_cow_respects_hlr_recovery_handoff(monkeypatch, later_vitals, expected):
    """A known deferred priority must not reduce a deliberate food turn to one action."""
    belief = public_state()
    belief["inventory"].update(food=1, drink=0)
    spec, _ = choose_step(belief, "cow-fair-turn", "food", [], None)
    state = runtime.initial_states()[LLREASONER]
    state["belief_state"] = belief
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    module._goal_graph_id = "goals"
    monkeypatch.setattr(module, "_dispatch", lambda state: None)
    module.on_goal_update(state, "goals", [make_activity_goal(spec)])
    assert state["failure"] is None
    state["current_goal_action_count"] = 1
    belief["inventory"].update(later_vitals)
    assert module._outcome(state, {"status": {"illegal_action": False}}) == expected


@pytest.mark.parametrize("activity,actions,budget,valid", [
    ("eat_cow", 96, 96, True),
    ("eat_cow", 97, 96, False),
    ("eat_cow", 96, 97, False),
    ("eat_cow", 33, 32, False),
    ("eat_target", 32, 32, True),
    ("eat_target", 33, 33, False),
    ("explore", 33, 33, False),
])
def test_run_result_checker_enforces_approved_activity_budgets(activity, actions, budget, valid):
    """Saved EatCow episodes may use 96 actions; each request still bounds execution."""
    from mha_exp_common.utils import module_name
    from mha_exp_level2_cr.exp2_5 import runner
    from mha_exp_level2_cr.exp2_5.environment import initial_state

    states = runtime.initial_states()
    states[LLREASONER]["activities"] = [{
        "kind": "activity", "activity": activity, "actions": actions, "max_actions": budget,
    }]
    _, errors = runner.check_results(
        {module_name(name, 0): state for name, state in states.items()},
        initial_state(), require_objective=False,
    )
    assert ("policy activity bound exceeded" not in errors) is valid
    assert not any("incomplete or invalid result evidence" in error for error in errors)
