"""Focused compact-runtime and acceptance tests for experiment 2-5-BW."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mhagenta import Belief
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase

from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import module_name
from mha_exp_level2_bw.exp2_5.contracts import TransferSpec, transfer_goal
from mha_exp_level2_bw.exp2_5.policy import POLICY_ARCHITECTURE
from mha_exp_level2_bw.exp2_5.runner import check_results
from mha_exp_level2_bw.exp2_5.runtime import (
    BlocksWorldActuator,
    BlocksWorldPerceptor,
    ClosedWorldKnowledge,
    NeuralTransferLLReasoner,
    RepeatedGoalHLReasoner,
    TransferGoalGraph,
    initial_states,
)


class RecordingOutbox:
    def __init__(self) -> None:
        self.goals: list[Any] = []

    def send_goals(self, receiver: str, goals: list[Any], **kwargs: Any) -> None:
        self.goals.append((receiver, goals, kwargs))


class FakeHLState(dict[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.outbox = RecordingOutbox()


def _active_hl_state() -> tuple[RepeatedGoalHLReasoner, FakeHLState, TransferSpec]:
    reasoner = RepeatedGoalHLReasoner(
        module_id="hlreasoner_0",
        initial_state=initial_states()[HLREASONER],
    )
    reasoner._goal_graph_id = "goalgraph_0"
    reasoner._completion_limit = 1
    reasoner._log_func = lambda level, message: None
    spec = TransferSpec("b0", "t0", "b1", "t0", "t1")
    state = FakeHLState(deepcopy(initial_states()[HLREASONER]))
    state.update(
        intention={"top": "b0", "bottom": "b1"},
        current_plan_id="plan-1",
        pending_goal={"goal_id": "compound-1", "plan_id": "plan-1", "step_index": 0},
        phase="awaiting-transfer-evidence",
        goal_runs=[
            {
                "goal": {"top": "b0", "bottom": "b1"},
                "plan": {
                    "plan_id": "plan-1",
                    "engine": "lpg",
                    "fallback_used": False,
                    "validation_status": "VALID",
                    "transfers": [
                        {
                            "spec": spec.as_dict(),
                            "goal_id": "compound-1",
                            "target_facts": sorted(spec.target_facts),
                            "terminal_observation_id": None,
                            "revision_observation_id": None,
                            "atomic": [],
                        }
                    ],
                },
                "final_goal_fact": None,
                "final_observation_id": None,
                "status": "active",
            }
        ],
    )
    return reasoner, state, spec


def _terminal(spec: TransferSpec, *, action: int = 3):
    return transfer_goal(
        spec,
        status="succeeded",
        goal_id="compound-1",
        plan_id="plan-1",
        step_index=0,
        based_on_observation_seq=1,
        completion_observation_seq=2,
        observed_facts=spec.target_facts,
        atomic_rows=[
            {
                "atomic_action_id": "atomic-1",
                "observation_id": 1,
                "input_sha256": "a" * 64,
                "selected_action": action,
                "q_values": [0.0, 0.0, 0.0, 1.0],
                "legal": True,
            }
        ],
    )


def _beliefs() -> list[Belief]:
    return [
        Belief(predicate="On", arguments=("b0", "b1")),
        Belief(predicate="AtLoc", arguments=("b0", "t1")),
    ]


def _revision(reasoner: RepeatedGoalHLReasoner, state: FakeHLState) -> None:
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        _beliefs(),
        goal_id="compound-1",
        plan_id="plan-1",
        step_index=0,
        observation_id=2,
    )


def test_reconciliation_is_order_independent_and_closes_from_beliefs() -> None:
    for terminal_first in (True, False):
        reasoner, state, spec = _active_hl_state()
        if terminal_first:
            reasoner.on_goal_update(state, "goalgraph_0", [_terminal(spec)])
            assert state["phase"] == "awaiting-transfer-evidence"
            _revision(reasoner, state)
        else:
            _revision(reasoner, state)
            assert state["phase"] == "awaiting-transfer-evidence"
            reasoner.on_goal_update(state, "goalgraph_0", [_terminal(spec)])
        transfer = state["goal_runs"][0]["plan"]["transfers"][0]
        assert state["phase"] == "completed"
        assert state["goal_completion_count"] == 1
        assert state["goal_runs"][0]["final_goal_fact"] == "on(b0,b1)"
        assert transfer["revision_observation_id"] >= transfer["terminal_observation_id"]


def test_stale_revision_waits_and_conflicting_terminal_fails() -> None:
    reasoner, state, spec = _active_hl_state()
    reasoner.on_goal_update(state, "goalgraph_0", [_terminal(spec)])
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        _beliefs(),
        goal_id="compound-1",
        plan_id="plan-1",
        step_index=0,
        observation_id=1,
    )
    assert state["pending_goal"] is not None
    reasoner.on_goal_update(state, "goalgraph_0", [_terminal(spec, action=2)])
    assert state["failure"]["code"] == "conflicting-terminal-duplicate"


def _passing_states() -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    states = {module_name(name, 0): deepcopy(value) for name, value in initial_states().items()}
    goals = []
    for index in range(5):
        top = f"b{index}"
        bottom = "b7"
        spec = TransferSpec(top, f"t{index}", bottom, f"t{index}", "t4")
        goals.append(
            {
                "goal": {"top": top, "bottom": bottom},
                "plan": {
                    "plan_id": f"plan-{index + 1}",
                    "engine": "lpg",
                    "fallback_used": False,
                    "validation_status": "VALID",
                    "transfers": [
                        {
                            "spec": spec.as_dict(),
                            "goal_id": f"compound-{index + 1}",
                            "target_facts": sorted(spec.target_facts),
                            "terminal_observation_id": index + 2,
                            "revision_observation_id": index + 2,
                            "atomic": [
                                {
                                    "atomic_action_id": f"atomic-{index + 1}",
                                    "observation_id": index + 1,
                                    "input_sha256": f"{index:x}" * 64,
                                    "selected_action": 3,
                                    "q_values": [0.0, 0.0, 0.0, 1.0],
                                    "legal": True,
                                }
                            ],
                        }
                    ],
                },
                "final_goal_fact": f"on({top},{bottom})",
                "final_observation_id": index + 2,
                "status": "succeeded",
            }
        )
    states["perceptor_0"].update(request_count=6, observation_count=6, last_observation_id=6)
    states["actuator_0"].update(request_count=5, status_count=5, successful_status_count=5)
    states["llreasoner_0"].update(
        policy_loaded=True,
        policy_id=POLICY_ARCHITECTURE,
        checkpoint_sha256="checkpoint",
        observation_request_count=6,
        observation_count=6,
        observation_id=6,
        belief_count=60,
        activated_transfer_count=5,
        completed_transfer_count=5,
        inference_count=5,
        atomic_request_count=5,
        status_count=5,
        atomic_action_counter=5,
    )
    states["knowledge_0"].update(revision_count=6, forwarded_count=6)
    states["goalgraph_0"].update(dispatch_count=5, terminal_count=5)
    states["hlreasoner_0"].update(
        phase="completed",
        terminal_reason="goal-completion-limit",
        goal_runs=goals,
        goal_completion_count=5,
        plan_count=5,
        transfer_dispatch_count=5,
        completed_transfer_count=5,
    )
    environment = {"failure": None, "observation_count": 6, "action_count": 5}
    manifest = {
        "architecture": POLICY_ARCHITECTURE,
        "checkpoint_sha256": "checkpoint",
        "training": {"elapsed_seconds": 1.0},
    }
    return states, environment, manifest


def test_compact_checker_accepts_science_and_rejects_boundary_mismatch() -> None:
    states, environment, manifest = _passing_states()
    assert check_results(states, environment, [], manifest)
    broken = deepcopy(states)
    broken["actuator_0"]["successful_status_count"] -= 1
    assert not check_results(broken, environment, [], manifest)


def test_declared_active_reconciling_failure_and_final_states_are_json_safe() -> None:
    initial = initial_states()
    json.dumps(initial)
    _, active, spec = _active_hl_state()
    json.dumps(active)
    active["pending_terminal"] = _terminal(spec).extras
    json.dumps(active)
    active["failure"] = {"code": "example", "observation_id": 2}
    json.dumps(active)
    states, environment, _ = _passing_states()
    json.dumps(states)
    json.dumps(environment)


def test_six_behaviors_use_public_mhagenta_bases() -> None:
    assert issubclass(BlocksWorldPerceptor, RMQPerceptorBase)
    assert issubclass(BlocksWorldActuator, RMQActuatorBase)
    assert issubclass(NeuralTransferLLReasoner, LLReasonerBase)
    assert issubclass(ClosedWorldKnowledge, KnowledgeBase)
    assert issubclass(TransferGoalGraph, GoalGraphBase)
    assert issubclass(RepeatedGoalHLReasoner, HLReasonerBase)


def test_exp2_5_has_no_exp2_4_or_offline_tool_dependency() -> None:
    package = Path(__file__).parents[1] / "src" / "mha_exp_level2_bw" / "exp2_5"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package.glob("*.py")
    )
    assert "exp2_4" not in source
    assert "exp2_5_policy_preparation" not in source
