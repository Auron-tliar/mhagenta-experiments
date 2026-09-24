"""Regression coverage for diamond stopping and matched EatCow availability."""
from types import SimpleNamespace

import pytest
from exp2_5_harness import HarnessState
from mha_exp_common.names import HLREASONER, LLREASONER
from mha_exp_common.utils import module_name
from mha_exp_level2_cr.exp2_5 import runtime, runner
from mha_exp_level2_cr.exp2_5.contracts import ActivityId, activity_from_goal, goal_to_dict, make_activity_goal
from mha_exp_level2_cr.exp2_5.deliberation import choose_step
from mha_exp_level2_cr.exp2_5.environment import initial_state as environment_state
from test_exp2_5_runtime import public_state


@pytest.mark.parametrize("enabled,visible,expected", [
    (True, False, ActivityId.EAT_COW), (False, False, ActivityId.EXPLORE),
    (True, True, ActivityId.EAT_TARGET), (False, True, ActivityId.EAT_TARGET),
])
def test_hlr_food_options_follow_availability(enabled, visible, expected) -> None:
    """Actual HLR dispatch removes EatCow while retaining visible-cow pursuit."""
    belief = public_state()
    belief["inventory"]["food"] = 1
    if visible:
        belief["occupants"] = {"1,0": {"kind": "cow"}}
    sent = []
    state = HarnessState(runtime.initial_states(enable_eat_cow=enabled)[HLREASONER])
    state.update(latest_beliefs=belief, outbox=SimpleNamespace(send_goals=lambda receiver, goals: sent.extend(goals)))
    module = runtime.CrafterActivityHLReasoner(module_id=HLREASONER, initial_state=state)
    module._goal_graph_id = "goals"
    module.log = lambda *args: None
    module._dispatch(state)
    assert len(sent) == 1
    spec = activity_from_goal(sent[0])
    assert spec.activity is expected
    assert spec.max_actions == (96 if expected is ActivityId.EAT_COW else 8 if expected is ActivityId.EXPLORE else 16)


@pytest.mark.parametrize("diamond,dead,count,reason", [
    (1, False, 200, "diamond_acquired"), (1, False, 900, "diamond_acquired"),
    (1, True, 200, "environment_terminal"), (0, False, 900, "action_budget"),
])
def test_diamond_stops_hlr_without_another_dispatch(diamond, dead, count, reason) -> None:
    """Diamond ends the run after reconciliation, with death taking priority."""
    belief = public_state()
    belief["inventory"]["diamond"] = diamond
    belief.update(dead=dead, native_action_count=count)
    stopped = []
    state = HarnessState(runtime.initial_states()[HLREASONER])
    state.update(latest_beliefs=belief, outbox=SimpleNamespace(terminate_agent=stopped.append))
    module = runtime.CrafterActivityHLReasoner(module_id=HLREASONER, initial_state=state)
    module._dispatch(state)
    assert stopped == [reason]
    assert state["terminal_reason"] == reason
    assert state["dispatch_count"] == 0
    assert state["survived"] is (not dead)


def test_diamond_interrupts_an_unfinished_neural_activity() -> None:
    """A new diamond cannot leave an otherwise unfinished policy running."""
    belief = public_state()
    spec, _ = choose_step(belief, "test", None, [], None)
    state = runtime.initial_states()[LLREASONER]
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)))
    belief["inventory"]["diamond"] = 1
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    assert module._outcome(state, {"status": {"illegal_action": False}}) == ("interrupted", "diamond_acquired")


def test_disabled_eat_cow_never_reaches_inference() -> None:
    """Even an erroneous EatCow request is rejected before selecting an action."""
    belief = public_state()
    spec, _ = choose_step(belief, "test", "food", [], None)
    state = runtime.initial_states(enable_eat_cow=False)[LLREASONER]
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)))
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    finished = []
    module._terminal = lambda state, status, reason: finished.append((status, reason))
    module._dispatch(state)
    assert finished == [("failed", "policy_disabled")]
    assert state["inference_count"] == 0


@pytest.mark.parametrize("reason,diamond,event,expected", [
    ("diamond_acquired", 1, 1, True), ("diamond_acquired", 0, 0, False),
    ("diamond_acquired", 1, 0, False), ("action_budget", 1, 1, False),
])
def test_result_checker_requires_evidence_for_early_diamond(reason, diamond, event, expected) -> None:
    """Early completion requires both inventory and achievement evidence."""
    states = runtime.initial_states()
    ll, hl = states[LLREASONER], states[HLREASONER]
    hl.update(terminal_reason=reason, survived=True)
    env = environment_state()
    env.update(inventory={"health": 9, "diamond": diamond}, closed=True,
               achievement_counts={"make_stone_pickaxe": 1, "collect_diamond": event})
    ll["belief_state"]["achievement_counts"] = {k: v for k, v in env["achievement_counts"].items() if v}
    _, errors = runner.check_results({module_name(k, 0): v for k, v in states.items()}, env)
    objective_errors = [error for error in errors if "action budget incomplete" in error or "survival objective" in error]
    assert (not objective_errors) is expected


def test_batch_rejects_non_boolean_availability(tmp_path) -> None:
    """A string false in CLI JSON must not silently enable the policy."""
    with pytest.raises(ValueError, match="boolean"):
        runner.run_batch(runs=[0], exp_path=tmp_path, enable_eat_cow="false")
