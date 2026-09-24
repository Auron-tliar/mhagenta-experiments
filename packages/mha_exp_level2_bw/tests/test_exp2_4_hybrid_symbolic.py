"""Scientific-contract tests for experiment 2-4-BW."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mhagenta import ActionStatus, Goal, Observation
from mha_env_blocksworld import BlocksWorldEnv

from mha_exp_level2_bw.exp2_4.planning import (
    GoalSpec, PlanningOutcome, PlanningService, TransferSpec, beliefs_to_facts,
    missing_transfer_preconditions, observed_transfer_targets,
    parse_symbolic_observation, project_abstract_facts, transfer_goal,
)
from mha_exp_level2_bw.exp2_4.runner import (
    A_PICK_UP, A_PUT_DOWN, K_ACTION, K_LEGAL, HybridBDIReasoner,
    TransferGoalGraph, TransferLLReasoner, check_results, transfer_initial_states,
)

DOMAIN = Path(__file__).parents[1] / "src/mha_exp_level2_bw/exp2_4/blocksworld-transfer-domain.pddl"
START = [
    "HandEmpty", "Above(t0)", "On(b0,t0)", "AtLoc(b0,t0)", "Clear(b0)",
    "On(b1,t1)", "AtLoc(b1,t1)", "Clear(b1)", "AtLoc(t0,t0)",
    "AtLoc(t1,t1)", "LeftOf(t0,t1)",
]
FINISH = [
    "HandEmpty", "Above(t1)", "On(b0,b1)", "AtLoc(b0,t1)", "Clear(b0)",
    "On(b1,t1)", "AtLoc(b1,t1)", "AtLoc(t0,t0)", "AtLoc(t1,t1)",
    "LeftOf(t0,t1)",
]


class Outbox:
    def __init__(self) -> None:
        self.observation_requests, self.actions = [], []
        self.beliefs, self.goal_updates, self.routed_goals = [], [], []

    def request_observation(self, receiver: str, **kwargs: Any) -> None:
        self.observation_requests.append((receiver, kwargs))

    def request_action(self, receiver: str, **kwargs: Any) -> None:
        self.actions.append((receiver, kwargs))

    def send_beliefs(self, receiver: str, *args: Any, **kwargs: Any) -> None:
        self.beliefs.append((receiver, args, kwargs))

    def send_goal_update(self, receiver: str, goals: list[Goal], **kwargs: Any) -> None:
        self.goal_updates.append((receiver, goals, kwargs))

    def send_goals(self, receiver: str, goals: list[Goal], **kwargs: Any) -> None:
        self.routed_goals.append((receiver, goals, kwargs))

    def clear(self) -> None:
        pass


def _directory() -> Any:
    item = lambda name: SimpleNamespace(module_id=name)
    return SimpleNamespace(internal=SimpleNamespace(
        perception=[item("perceptor_0")], actuation=[item("actuator_0")],
        knowledge=[item("knowledge_0")], hl_reasoning=[item("hlreasoner_0")],
        ll_reasoning=[item("llreasoner_0")], goals=[item("goalgraph_0")],
    ))


class State(dict[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.outbox, self.directory = Outbox(), _directory()


def _facts(observation: list[str]) -> set[str]:
    return beliefs_to_facts(parse_symbolic_observation(observation))


def _spec() -> TransferSpec:
    return TransferSpec("b0", "t0", "b1", "t0", "t1")


def _goal(status: str = "requested") -> Goal:
    return transfer_goal(
        _spec(), status=status, hierarchy_id="hierarchy-1", goal_id="compound-1",
        plan_id="plan-1", step_index=0, based_on_observation_seq=1,
        completion_observation_seq=2 if status == "succeeded" else None,
        observed_target_facts=_spec().target_facts if status == "succeeded" else (),
        atomic_rows=[
            {K_ACTION: A_PICK_UP, "phase": "pick-up", "legal": True, "observation_seq": 1},
            {K_ACTION: A_PUT_DOWN, "phase": "put-down", "legal": True, "observation_seq": 2},
        ] if status == "succeeded" else (),
    )


def test_fact_transfer_and_planning_contract() -> None:
    facts = _facts(START)
    assert project_abstract_facts(facts) == {
        "on(b0,t0)", "at-location(b0,t0)", "clear(b0)", "on(b1,t1)",
        "at-location(b1,t1)", "clear(b1)", "at-location(t0,t0)", "at-location(t1,t1)",
    }
    assert missing_transfer_preconditions(facts, _spec()) == ()
    assert observed_transfer_targets(_facts(FINISH), _spec()) == sorted(_spec().target_facts)
    service = PlanningService(domain_path=DOMAIN, blocks=("b0", "b1"), locations=("t0", "t1"))
    outcome = service.solve(facts, GoalSpec("b0", "b1"), "compact")
    assert outcome.accepted and outcome.validation_status == "VALID"
    assert outcome.actions == [{"name": "transfer", **_spec().as_dict()}]
    assert set(outcome.__dict__) == {
        "accepted", "engine", "status", "elapsed_seconds", "actions",
        "validation_status", "sanity_bound", "failure",
    }


def _ll() -> tuple[TransferLLReasoner, State]:
    state = State(deepcopy(transfer_initial_states()["llreasoner"]))
    behavior = TransferLLReasoner(module_id="llreasoner_0", initial_state={})
    behavior.on_first(state)
    return behavior, state


def test_ll_fsm_completes_one_real_transfer() -> None:
    env, initial = BlocksWorldEnv(table_len=2, num_blocks=2, symbolic=True), None
    env.expose_snapshot = True
    for seed in range(1000, 1100):
        candidate, _ = env.reset(seed=seed)
        if not missing_transfer_preconditions(_facts(list(candidate)), _spec()):
            initial = list(candidate)
            break
    if initial is None:
        pytest.skip("No matching deterministic two-block seed.")
    behavior, state = _ll()
    behavior.on_observation(state, "perceptor_0", Observation(initial))
    behavior.on_goal_update(state, "goalgraph_0", [_goal()])
    cursor = 0
    for _ in range(12):
        if state.outbox.goal_updates:
            break
        _, request = state.outbox.actions[cursor]
        cursor += 1
        observation, reward, _, _, info = env.step(request[K_ACTION])
        behavior.on_action_status(state, "actuator_0", ActionStatus({
            **request, K_LEGAL: bool(info["snapshot"].legal), "reward": float(reward)}))
        behavior.on_observation(state, "perceptor_0", Observation(list(observation)), **request)
    terminal = state.outbox.goal_updates[-1][1][0]
    assert terminal.extras["status"] == "succeeded"
    assert terminal.extras["observed_target_facts"] == sorted(_spec().target_facts)
    assert state["active"] is None and state["failure"] is None


def test_ll_illegal_action_is_fail_fast() -> None:
    behavior, state = _ll()
    behavior.on_observation(state, "perceptor_0", Observation(START))
    behavior.on_goal_update(state, "goalgraph_0", [_goal()])
    _, request = state.outbox.actions[-1]
    behavior.on_action_status(state, "actuator_0", ActionStatus({**request, K_LEGAL: False}))
    assert state["failure"] == "illegal-action" and state["active"] is None
    assert state.outbox.goal_updates[-1][1][0].extras["status"] == "failed"


def test_goal_graph_is_one_correlated_relay() -> None:
    state = State(deepcopy(transfer_initial_states()["goalgraph"]))
    graph = TransferGoalGraph(module_id="goalgraph_0", initial_state={})
    graph.on_first(state)
    graph.on_goal_update(state, "hlreasoner_0", [_goal()])
    graph.on_goal_update(state, "llreasoner_0", [_goal("succeeded")])
    assert [row[0] for row in state.outbox.routed_goals] == ["llreasoner_0", "hlreasoner_0"]
    assert state["active"] is None and state["dispatched"] == state["terminal"] == 1


class ScriptedPlanning:
    def solve(self, *args: Any) -> PlanningOutcome:
        return PlanningOutcome(True, "lpg", "SOLVED_SATISFICING", 0.0,
                               [{"name": "transfer", **_spec().as_dict()}], "VALID", 8, None)


def _hl() -> tuple[HybridBDIReasoner, State]:
    state = State(deepcopy(transfer_initial_states()["hlreasoner"]))
    behavior = HybridBDIReasoner(module_id="hlreasoner_0", initial_state={})
    behavior.on_init(seed=5, num_blocks=2, table_len=2, planner_timeout=1, goal_completion_limit=1)
    behavior.on_first(state)
    behavior._planning = ScriptedPlanning()
    behavior._select_intention = lambda state, facts: GoalSpec("b0", "b1")  # type: ignore[method-assign]
    behavior.on_belief_update(state, "knowledge_0", parse_symbolic_observation(START), observation_seq=1)
    return behavior, state


@pytest.mark.parametrize("terminal_first", [True, False])
def test_hl_reconciles_both_message_orders(terminal_first: bool) -> None:
    behavior, state = _hl()
    context = {"hierarchy_id": "hierarchy-1", "goal_id": "compound-1",
               "plan_id": "plan-1", "step_index": 0, "observation_seq": 2}
    terminal = lambda: behavior.on_goal_update(state, "goalgraph_0", [_goal("succeeded")])
    beliefs = lambda: behavior.on_belief_update(
        state, "knowledge_0", parse_symbolic_observation(FINISH), **context)
    (terminal(), beliefs()) if terminal_first else (beliefs(), terminal())
    assert state["phase"] == "completed" and state["completed_transfers"] == 1
    assert state["current_hierarchy"]["status"] == "completed"


def _passing_states() -> dict[str, dict[str, Any]]:
    states = {f"{name}_0": deepcopy(value) for name, value in transfer_initial_states().items()}
    rows = [{K_ACTION: A_PICK_UP, "phase": "pick-up", "legal": True, "observation_seq": 2},
            {K_ACTION: A_PUT_DOWN, "phase": "put-down", "legal": True, "observation_seq": 3}]
    hierarchy = {
        "hierarchy_id": "h1", "intention": {"top": "b0", "bottom": "b1"},
        "plan": {"plan_id": "p1", "engine": "lpg", "validation": "VALID", "bound": 8,
                 "transfers": [_spec().as_dict()]},
        "execution": [{"step_index": 0, "goal_id": "g1", "dispatch_observation_seq": 1,
                       "atomic_rows": rows, "completion_observation_seq": 3,
                       "observed_target_facts": sorted(_spec().target_facts),
                       "completion_belief_observation_seq": 3, "status": "completed"}],
        "final_goal_observation_seq": 3, "status": "completed",
    }
    states["perceptor_0"].update(requests=3, observations=3)
    states["actuator_0"].update(requests=2, statuses=2)
    states["llreasoner_0"].update(observation_requests=3, observations=3, action_requests=2,
                                  action_statuses=2, transfer_activations=1, terminal_updates=1)
    states["knowledge_0"].update(revisions=3, forwards=3)
    states["goalgraph_0"].update(dispatched=1, terminal=1)
    states["hlreasoner_0"].update(current_hierarchy=hierarchy, transfer_dispatches=1,
                                  terminal_results=1, reconciled_outcomes=1, goal_completions=1)
    return states


def test_json_state_checker_and_active_tail() -> None:
    states = _passing_states()
    json.dumps(states)
    assert check_results(states, [])
    states["hlreasoner_0"]["retained_hierarchy"] = states["hlreasoner_0"]["current_hierarchy"]
    states["hlreasoner_0"]["current_hierarchy"] = {"hierarchy_id": "h2", "status": "active"}
    states["hlreasoner_0"]["current_hierarchy"]["execution"] = [{"status": "dispatched"}]
    states["hlreasoner_0"]["transfer_dispatches"] = 2
    assert check_results(states, [])
    states["hlreasoner_0"]["retained_hierarchy"]["execution"][0]["observed_target_facts"] = []
    assert not check_results(states, [])


@pytest.mark.parametrize("terminal_first", [True, False])
def test_timeout_reconciles_both_message_orders(terminal_first: bool) -> None:
    """A timed-out transfer retains its prefix and closes both message paths."""
    behavior, state = _hl()
    context = {"hierarchy_id": "hierarchy-1", "goal_id": "compound-1",
               "plan_id": "plan-1", "step_index": 0, "observation_seq": 2}
    failed = transfer_goal(_spec(), status="failed", **{k: v for k, v in context.items() if k != "observation_seq"},
                           based_on_observation_seq=1, completion_observation_seq=2,
                           failure_reason="time_budget_exhausted")
    terminal = lambda: behavior.on_goal_update(state, "goalgraph_0", [failed])
    beliefs = lambda: behavior.on_belief_update(state, "knowledge_0", parse_symbolic_observation(START), **context)
    (terminal(), beliefs()) if terminal_first else (beliefs(), terminal())
    assert state["phase"] == "unsolved" and state["failure"] is None
    assert state["terminal_results"] == state["reconciled_outcomes"] == 1
    assert state["current_hierarchy"]["execution"][0]["status"] == "failed"


def test_ll_timeout_dispatches_no_extra_action() -> None:
    """The execution cutoff is checked before the next atomic action."""
    behavior, state = _ll()
    behavior.on_observation(state, "perceptor_0", Observation(START))
    state.time = 1795.0
    behavior.on_goal_update(state, "goalgraph_0", [_goal()])
    assert state.outbox.actions == []
    assert state.outbox.goal_updates[-1][1][0].extras["failure_reason"] == "time_budget_exhausted"
    assert state["failure"] is None


def test_controlled_planner_failure_is_a_reportable_result() -> None:
    """An unsolved goal is accepted only with observed, quiescent state."""
    states = {f"{name}_0": deepcopy(value) for name, value in transfer_initial_states().items()}
    for name in ("perceptor_0", "llreasoner_0"):
        states[name]["observations"] = 1
    states["perceptor_0"]["requests"] = 1
    states["llreasoner_0"]["observation_requests"] = 1
    states["knowledge_0"].update(revisions=1, forwards=1)
    states["hlreasoner_0"].update(phase="unsolved", terminal_reason="planner_did_not_solve",
                                current_hierarchy={"status": "unsolved", "plan": None, "execution": []})
    assert check_results(states, [])
    states["llreasoner_0"]["failure"] = "broken transport"
    assert not check_results(states, [])


def test_fifty_template_worlds_reproduce_unsatisfied_goals() -> None:
    """All 2-3-BW templates reproduce their distinct seeded worlds and goals."""
    from collections import Counter
    from hashlib import sha256
    from mha_exp_level2_bw.exp2_4.treatment import treatment_for_run
    from mha_exp_level2_bw.exp2_3.treatment import TASKS as source_tasks
    tasks = [treatment_for_run(run) for run in range(50)]
    assert [t["seed"] for t in tasks] == [t.seed for t in source_tasks]
    assert len({t["seed"] for t in tasks}) == 50
    assert Counter((t["table_len"], t["num_blocks"]) for t in tasks[:30]) == {(4, 6): 10, (5, 8): 10, (7, 12): 10}
    original_digest = "3edfdd182cd04a9d69dd40f6c9a893a5bc763bca4c433155545cd83bba58aa7d"
    assert all(t["manifest_digest"] == original_digest for t in tasks[:30])
    for task in tasks:
        env = BlocksWorldEnv(table_len=task["table_len"], num_blocks=task["num_blocks"], symbolic=True)
        observation, _ = env.reset(seed=task["seed"])
        facts = _facts(observation)
        assert sha256(json.dumps(sorted(facts), separators=(",", ":")).encode()).hexdigest() == task["initial_state_digest"]
        assert GoalSpec(**task["goal"]).fact not in facts
        env.close()
    with pytest.raises(ValueError):
        treatment_for_run(50)


def test_environment_close_requests_graceful_save_once(monkeypatch) -> None:
    """Closing schedules framework shutdown after the final state update."""
    from types import SimpleNamespace
    from mha_exp_level2_bw.exp2_4 import agent
    calls = []

    class FakeTimer:
        def __init__(self, interval, function, args):
            calls.append((interval, function, args))
            self.daemon = False

        def start(self):
            calls.append("started")

    monkeypatch.setattr(agent, "Timer", FakeTimer)
    environment = agent.TestEnvironment.__new__(agent.TestEnvironment)
    environment._env = SimpleNamespace(close=lambda: None)
    environment._stop_timer = None
    state = {"close_requests": 0, "closed": False}
    assert environment.on_action(state, "agent", action=agent.A_CLOSE) == (state, None)
    assert state == {"close_requests": 1, "closed": True}
    assert calls == [(0.05, agent.os.kill, (agent.os.getpid(), agent.signal.SIGTERM)), "started"]
    assert environment._stop_timer.daemon
    environment.on_action(state, "agent", action=agent.A_CLOSE)
    assert len(calls) == 2
    assert environment.__getstate__()["_stop_timer"] is None


def test_actuator_shutdown_closes_environment_once() -> None:
    """Repeated terminal callbacks must not send duplicate close requests."""
    from mha_exp_level2_bw.exp2_4.agent import HybridBlocksWorldActuator
    behavior = HybridBlocksWorldActuator(module_id="actuator_0", initial_state={})
    behavior._env_id = "environment"
    calls = []
    behavior.act = lambda *args, **kwargs: calls.append((args, kwargs))
    state = State(transfer_initial_states()["actuator"])
    behavior.on_last(state)
    behavior.on_last(state)
    assert calls == [(("environment",), {"action": "close"})]
