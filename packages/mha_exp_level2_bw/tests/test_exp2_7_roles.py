from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mhagenta import ActionStatus, Directory, Observation, State
from mhagenta.bases import (
    GoalGraphBase,
    HLReasonerBase,
    KnowledgeBase,
    LearnerBase,
    LLReasonerBase,
    MemoryBase,
)
from mhagenta.outboxes import LearnerOutbox, LLOutbox
from pydantic import ValidationError

from mha_exp_level2_bw.exp2_7 import llm
from mha_exp_level2_bw.exp2_7.llm import (
    LLReasonerResponse,
    LearnerResponse,
    MODEL_POLICY,
)
from mha_exp_level2_bw.exp2_7.roles import (
    LLMGoalGraph,
    LLMHighLevelReasoner,
    LLMKnowledge,
    LLMLearner,
    LLMLowLevelReasoner,
    LLMMemory,
)
from mha_exp_level2_bw.exp2_7.runner import (
    STANDARD_BEHAVIOR_CUTOFF,
    LifecycleConfig,
    _build_agent_modules,
)


class FakeResponses:
    def __init__(self, outputs: list[Any], hook: Any = None) -> None:
        self.outputs = iter(outputs)
        self.hook = hook
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.hook is not None:
            self.hook()
            self.hook = None
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        if isinstance(output, dict):
            output = kwargs["text_format"].model_validate(output)
        return SimpleNamespace(
            id=f"response_{len(self.calls)}",
            model=kwargs["model"],
            output_parsed=output,
        )


class FakeClient:
    def __init__(self, outputs: list[Any], hook: Any = None) -> None:
        self.responses = FakeResponses(outputs, hook)
        self.closed = False

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "pricing_policy_version": "test",
            "budget_source": "estimated",
            "max_budget_usd": "4",
            "forwarded_request_count": len(self.responses.calls),
            "input_tokens": 1,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 1,
            "reasoning_tokens": 0,
            "total_tokens": 2,
            "estimated_cost_usd": "0.01",
            "budget_exhausted": False,
        }

    def close(self) -> None:
        self.closed = True


def _built() -> dict[str, Any]:
    return _build_agent_modules(
        primary_goals=[
            {
                "goal_id": "primary_0",
                "predicate": "On",
                "arguments": ["B0", "B1"],
                "status": "pending",
                "primary": True,
                "order": 0,
            }
        ],
        value_condition="balanced",
        exchange_name="test",
        lifecycle=LifecycleConfig(),
        max_budget_usd="4.00",
    )[0]


def test_knowledge_keeps_latest_observation_and_request() -> None:
    """Coalescing retains explicit work and records every received boundary."""
    module = _built()["knowledge"]
    state = _state(module, TerminationOutbox())
    module.on_belief_request(state, "hlreasoner_0", boundary_id="request:0")
    for index in range(4):
        module.on_observed_beliefs(state, "llreasoner_0", Observation(str(index)), [], boundary_id=f"obs:{index}")
    assert len(state["pending"]) == 2
    assert state["pending"][-1]["payload"]["observation"]["content"] == "3"
    assert state["coalesced_observations"] == 3
    assert len(state["processed_boundaries"]) == 5


def test_goal_suppression_preserves_explicit_replies_and_changed_progress() -> None:
    """Only identical unsolicited goal payloads are suppressed."""
    from mhagenta import Goal, Belief
    from mha_exp_level2_bw.exp2_7.roles import _changed_goal
    state = _state(_built()["goal_graph"], TerminationOutbox())
    goal = Goal([Belief("On", ("B0", "B1"))], goal_id="primary_0", status="active")
    assert _changed_goal(state, "llreasoner_0", goal)
    assert not _changed_goal(state, "llreasoner_0", goal)
    assert _changed_goal(state, "llreasoner_0", goal, requested=True)
    goal.extras["status"] = "achieved"
    assert _changed_goal(state, "llreasoner_0", goal)


def _goalgraph_with_responses(tmp_path: Path, responses: list[Any]) -> tuple[Any, Any, FakeClient]:
    """Build the native role with two model-authored intentions and no LL progress."""
    from mhagenta import Belief, Goal
    from mhagenta.outboxes import GoalGraphOutbox
    module = _built()["goal_graph"]
    client = FakeClient(responses)
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    state = _state(module, GoalGraphOutbox())
    module.on_goal_update(state, "hlreasoner_0", [
        Goal([Belief("On", ("B0", "B1"))], goal_id="primary", status="active"),
        Goal([Belief("Holding", ("B0",))], goal_id="supporting", status="pending"),
    ])
    return module, state, client


def test_goal_feedback_prevents_partial_dispatch_and_accepts_model_correction(tmp_path: Path) -> None:
    """Feedback retries the complete choice; it never silently prunes a bad relay."""
    module, state, client = _goalgraph_with_responses(tmp_path, [
        {"state": "No progress received", "deliver_goal_ids": ["primary"], "relay_progress_ids": ["supporting"]},
        {"state": "Considered provenance", "deliver_goal_ids": ["primary"], "relay_progress_ids": []},
    ])
    previous_text = state["text_state"]
    module.step(state)
    assert not state.outbox
    assert not state["sent_boundaries"]
    assert state["failure"] is None
    assert state["text_state"] == previous_text
    assert len(state["pending"]) == 1
    module.step(state)
    messages = list(state.outbox)
    assert len(messages) == 1 and messages[0][0] == "llreasoner_0"
    assert "No low-level progress update" in client.responses.calls[1]["input"]
    assert state["relay_repair_attempts"] == 0
    assert len(state["relay_feedback_history"]) == 1
    assert "goal_relay_feedback" not in state["context"]


def test_goal_feedback_allows_empty_response_without_inventing_a_replacement(tmp_path: Path) -> None:
    """The model can decide not to send anything; the runtime does not fill it in."""
    module, state, _ = _goalgraph_with_responses(tmp_path, [
        {"state": "Wrong identity", "deliver_goal_ids": ["unseen"]},
        {"state": "Waiting for useful information", "deliver_goal_ids": [], "relay_progress_ids": []},
    ])
    module.step(state)
    module.step(state)
    assert not state.outbox
    assert state["failure"] is None
    assert state["pending"] == []


def test_goal_feedback_is_bounded_and_preserves_unresolved_failure(tmp_path: Path) -> None:
    """An unresolved native payload error cannot trigger unbounded paid retries."""
    bad = {"state": "Still selecting unavailable goal", "deliver_goal_ids": ["unseen"]}
    module, state, client = _goalgraph_with_responses(tmp_path, [bad, bad, bad])
    for _ in range(4):
        module.step(state)
    assert len(client.responses.calls) == 3
    assert len(state["relay_feedback_history"]) == 3
    assert state["failure"]["stage"] == "dispatch"
    assert not state.outbox


def test_empty_goal_selection_answers_request_and_clears_wait(tmp_path: Path) -> None:
    """An empty model choice reaches LL as evidence, without inventing a goal."""
    from mha_exp_level2_bw.exp2_7.evidence import send_message

    module, state, _ = _goalgraph_with_responses(tmp_path, [{"state": "No goals selected"}])
    low = _built()["ll_reasoner"]
    low_state = _state(low, LLOutbox())
    send_message(low_state, recipient="goalgraph_0", kind="request_goals",
                 dispatch=lambda **meta: low_state.outbox.request_goals("goalgraph_0", **meta))
    request = list(low_state.outbox)[0][3]
    module.on_goal_request(state, "llreasoner_0", **request)
    module.step(state)
    messages = list(state.outbox)
    assert len(messages) == 1 and messages[0][0] == "llreasoner_0"
    assert messages[0][3]["goals"] == []
    assert state["goals"]["primary"]["extras"]["status"] == "active"
    low.on_goal_update(low_state, "goalgraph_0", **messages[0][3])
    assert low_state["waiting_requests"] == {}
    assert low_state["pending"][-1]["payload"] == {"goals": []}
    assert low_state["processed_boundaries"][-1]["boundary_id"] == state["sent_boundaries"][-1]["boundary_id"]
    assert low_state["failure"] is None


def test_empty_goal_reply_waits_for_valid_model_response(tmp_path: Path) -> None:
    """A rejected selection sends nothing until the model corrects its response."""
    module, state, _ = _goalgraph_with_responses(tmp_path, [
        {"state": "Unavailable progress", "relay_progress_ids": ["unseen"]},
        {"state": "No goals selected"},
    ])
    module.on_goal_request(state, "llreasoner_0")
    module.step(state)
    assert not state.outbox
    module.step(state)
    assert list(state.outbox)[0][3]["goals"] == []
    assert state["failure"] is None


def test_empty_goal_selection_does_not_answer_a_later_request(tmp_path: Path) -> None:
    """An arrival during inference must be considered in its own model input."""
    module, state, client = _goalgraph_with_responses(tmp_path, [
        {"state": "No goals selected"}, {"state": "Answering new request"},
    ])
    client.responses.hook = lambda: module.on_goal_request(state, "llreasoner_0")
    module.step(state)
    assert not state.outbox
    module.step(state)
    assert len(list(state.outbox)) == 1
    assert list(state.outbox)[0][3]["goals"] == []


def test_empty_goal_reply_respects_terminal_marker(tmp_path: Path) -> None:
    """The reply repair cannot dispatch after the agent has stopped."""
    from mha_exp_level2_bw.exp2_7.terminal import publish_terminal

    module, state, client = _goalgraph_with_responses(tmp_path, [{"state": "No goals selected"}])
    state["terminal_file"] = str(tmp_path / "terminal.json")
    module.on_goal_request(state, "llreasoner_0")
    client.responses.hook = lambda: publish_terminal(state, "time_limit")
    module.step(state)
    assert not state.outbox
    assert not state["sent_boundaries"]


def test_goal_feedback_preserves_model_selected_order_and_subgoals(tmp_path: Path) -> None:
    """No validator chooses a primary-first ordering or disallows a subgoal."""
    module, state, client = _goalgraph_with_responses(tmp_path, [
        {"state": "Chosen order", "deliver_goal_ids": ["supporting", "primary"]},
    ])
    module.step(state)
    messages = list(state.outbox)
    assert [message[3]["goals"][0].extras["goal_id"] for message in messages] == ["supporting", "primary"]
    assert state["relay_feedback_history"] == []
    schema = client.responses.calls[0]["text_format"].model_json_schema()
    assert schema["properties"]["deliver_goal_ids"]["items"] == {"type": "string"}


def test_goal_feedback_retry_respects_budget_termination(tmp_path: Path) -> None:
    """Feedback does not grant a module extra budget or dispatch a crossing response."""
    module, state, client = _goalgraph_with_responses(tmp_path, [
        {"state": "Incorrect", "relay_progress_ids": ["supporting"]},
        {"state": "Corrected", "deliver_goal_ids": ["primary"]},
    ])
    snapshot = client.usage_snapshot
    client.usage_snapshot = lambda: {**snapshot(), "budget_exhausted": len(client.responses.calls) >= 2}
    module.step(state)
    module.step(state)
    assert state["termination_reason"] == "budget_exhausted"
    assert state["failure"] is None
    assert not state["sent_boundaries"]
    assert state.outbox.pop_term_request()[0]


def test_goal_feedback_preserves_newer_inputs_and_uses_them_on_retry(tmp_path: Path) -> None:
    """Repair reuses original messages without deleting arrivals during the call."""
    module, state, client = _goalgraph_with_responses(tmp_path, [
        {"state": "Incorrect", "deliver_goal_ids": ["unseen"]},
        {"state": "No justified relay"},
    ])
    client.responses.hook = lambda: module.on_goal_request(state, "llreasoner_0")
    module.step(state)
    assert [item["kind"] for item in state["pending"]] == ["goal_update", "goal_request"]
    module.step(state)
    assert '"kind": "goal_request"' in client.responses.calls[1]["input"]
    assert state["pending"] == []


def _state(module: Any, outbox: Any, *, time: float = 0.0) -> State[Any]:
    return State(
        agent_id="agent",
        module_id=module.module_id,
        time_func=lambda: time,
        directory=Directory(),
        outbox=outbox,
        **module.initial_state,
    )


class TerminationOutbox:
    """Record focused termination requests from a role callback."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    def terminate_agent(self, reason: str) -> None:
        self.reasons.append(reason)


def test_scientific_completion_is_recorded_before_termination() -> None:
    module = _built()["ll_reasoner"]
    outbox = TerminationOutbox()
    state = _state(module, outbox, time=42.0)

    module.on_action_status(
        state,
        "actuator_0",
        ActionStatus({
            "scientific_complete": True,
            "eligible_post_goal_discretionary_states": 1,
        }),
        cycle_id="cycle:3",
        action_id="action:9",
    )

    assert state["scientific_complete"] is True
    assert state["scientific_completion_evidence"] == {
        "cycle_id": "cycle:3",
        "action_id": "action:9",
        "eligible_post_goal_discretionary_states": 1,
        "module_time": 42.0,
    }
    assert state["admission_open"] is False
    assert outbox.reasons == ["Experiment 2-7-BW scientific completion reached"]


def test_seven_roles_directly_subclass_native_bases_without_mixin() -> None:
    assert issubclass(LLMLowLevelReasoner, LLReasonerBase)
    assert issubclass(LLMKnowledge, KnowledgeBase)
    assert issubclass(LLMHighLevelReasoner, HLReasonerBase)
    assert issubclass(LLMGoalGraph, GoalGraphBase)
    assert issubclass(LLMMemory, MemoryBase)
    assert issubclass(LLMLearner, LearnerBase)
    assert not hasattr(llm, "LLMModuleMixin")
    assert not hasattr(llm, "FAST_MODEL_CANDIDATES")


def test_role_runtime_paths_remain_container_portable_before_initialization() -> None:
    modules = _built()
    cognitive = [
        modules["ll_reasoner"],
        modules["knowledge"],
        modules["hl_reasoner"],
        modules["goal_graph"],
        modules["memory"],
        *modules["learners"],
    ]

    assert all(isinstance(module._runtime._response_root, str) for module in cognitive)


def test_role_schemas_reject_foreign_routes_and_action_aliases() -> None:
    with pytest.raises(ValidationError):
        LLReasonerResponse.model_validate(
            {"state": "x", "recipient": "knowledge_0"}
        )
    with pytest.raises(ValidationError):
        LLReasonerResponse.model_validate({"state": "x", "action": "left"})


def test_only_hl_owns_primary_desires_and_only_knowledge_gets_value_text() -> None:
    modules = _built()
    cognitive = [
        modules["ll_reasoner"],
        modules["knowledge"],
        modules["hl_reasoner"],
        modules["goal_graph"],
        modules["memory"],
        *modules["learners"],
    ]
    assert "primary_desires" in modules["hl_reasoner"].initial_state
    assert all(
        "primary_desires" not in module.initial_state
        for module in cognitive
        if module is not modules["hl_reasoner"]
    )
    assert modules["goal_graph"].initial_state["goals"] == {}
    assert modules["knowledge"].init_kwargs["value_system"]
    assert all(
        not module.init_kwargs["value_system"]
        for module in cognitive
        if module is not modules["knowledge"]
    )


def test_startup_observes_before_model_and_idle_response_does_not_rebootstrap(tmp_path: Path) -> None:
    """An empty initial inbox cannot strand the agent before its first sensor reading."""
    module = _built()["ll_reasoner"]
    client = FakeClient([LLReasonerResponse(state="waiting for goals")])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    state = _state(module, LLOutbox())
    module.on_first(state)
    messages = list(state.outbox)
    assert len(messages) == 1
    assert messages[0][0] == "perceptor_0"
    assert messages[0][3]["cycle_id"] == "cycle:0"
    module.step(state)
    assert client.responses.calls == []
    observation = Observation("[T0; holding=empty]\nT0:B0\nT1:B1", observation_type="blocks-world-text")
    module.on_observation(state, "perceptor_0", observation, cycle_id="cycle:0")
    module.step(state)
    module.step(state)
    assert len(client.responses.calls) == 1
    assert state["current_observation"]["content"] == observation.content
    assert state["cycle_sequence"] == 1
    assert state["failure"] is None


def test_low_level_initial_call_dispatches_native_observation_request(tmp_path: Path) -> None:
    module = _built()["ll_reasoner"]
    client = FakeClient(
        [LLReasonerResponse(state="waiting", request_observation=True)]
    )
    module._runtime.set_client_for_testing(  # noqa: SLF001 - explicit test seam
        client,
        profile=MODEL_POLICY.fast,
        response_root=tmp_path,
    )
    state = _state(module, LLOutbox())
    module.step(state)
    messages = list(state.outbox)
    assert len(messages) == 1
    recipient, _, _, body = messages[0]
    assert recipient == "perceptor_0"
    assert body["cycle_id"] == "cycle:0"
    assert body["boundary_id"] == "llreasoner_0:boundary:0"
    assert state["failure"] is None
    assert state["processed_input_ids"] == ["llreasoner_0:input:0"]
    record = json.loads((tmp_path / "llreasoner_0.jsonl").read_text(encoding="utf-8"))
    assert record["requested_model"] == MODEL_POLICY.fast.model
    assert record["reasoning_effort"] == "low"
    assert record["schema_retry"] is False


def test_schema_retry_retains_inputs_and_runs_on_a_later_step(tmp_path: Path) -> None:
    module = _built()["ll_reasoner"]
    client = FakeClient(
        [
            {"state": "bad", "action": "left"},
            LLReasonerResponse(state="valid", request_observation=True),
        ]
    )
    module._runtime.set_client_for_testing(  # noqa: SLF001
        client,
        profile=MODEL_POLICY.fast,
        response_root=tmp_path,
    )
    state = _state(module, LLOutbox())
    module.step(state)
    assert len(client.responses.calls) == 1
    assert state["failure"] is None
    state._time_func = lambda: 1.0
    module.step(state)
    assert len(client.responses.calls) == 2
    assert state["repair_count"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "llreasoner_0.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["schema_retry"] for record in records] == [False, True]
    assert "raw_response" in records[0]


def test_callback_added_during_call_is_not_removed_with_snapshot(tmp_path: Path) -> None:
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    state["pending"].append(
        {
            "message_id": "before",
            "sender": "source",
            "kind": "test",
            "payload": {},
            "metadata": {},
            "received_at": 0.0,
        }
    )

    def append_late() -> None:
        state["pending"].append(
            {
                "message_id": "after",
                "sender": "source",
                "kind": "test",
                "payload": {},
                "metadata": {},
                "received_at": 0.1,
            }
        )

    client = FakeClient([LLReasonerResponse(state="done")], hook=append_late)
    module._runtime.set_client_for_testing(  # noqa: SLF001
        client,
        profile=MODEL_POLICY.fast,
        response_root=tmp_path,
    )
    module.step(state)
    assert [item["message_id"] for item in state["pending"]] == ["after"]


def test_learners_are_idle_until_requested_then_can_report_no_change(tmp_path: Path) -> None:
    learner = _built()["learners"][0]
    client = FakeClient([LearnerResponse(state="reviewed", no_change=True)])
    learner._runtime.set_client_for_testing(  # noqa: SLF001
        client,
        profile=MODEL_POLICY.deliberative,
        response_root=tmp_path,
    )
    state = _state(learner, LearnerOutbox())
    learner.step(state)
    assert client.responses.calls == []
    learner.on_task(state, "llreasoner_0", "Review recent failures", boundary_id="b0")
    learner.step(state)
    assert len(client.responses.calls) == 1
    assert state["involvement_requested_by_reasoner"] is True
    assert state["involvement_status"] == "requested_no_change"
    assert list(state.outbox)[0][0] == "llreasoner_0"


@pytest.mark.parametrize("decision", [
    {"no_change": True}, {"revision": "Use the supplied goal direction"},
    {"request_memories": True},
])
def test_learner_empty_reply_gets_feedback_without_choosing_answer(tmp_path: Path, decision: dict) -> None:
    """Reproduce a learner claiming to wait for observations without sending anything."""
    learner = _built()["learners"][1]
    state = _state(learner, LearnerOutbox())
    client = FakeClient([{"state": "Awaiting a fresh observation"}, {"state": "Reconsidered", **decision}])
    learner._runtime.set_client_for_testing(client, response_root=tmp_path)
    learner.on_task(state, "hlreasoner_0", "Check the supplied relation", boundary_id="task:1")
    learner.step(state)
    assert not state["sent_boundaries"] and state["pending"]
    assert state["initial_call_pending"] and state["failure"] is None
    learner.step(state)
    assert len(client.responses.calls) == 2
    assert "No message was sent" in client.responses.calls[1]["input"]
    assert "task:1" in client.responses.calls[1]["input"]
    assert len(state["sent_boundaries"]) == 1
    expected = "request_memories" if decision.get("request_memories") else "send_model"
    assert state["sent_boundaries"][0]["kind"] == expected
    assert "learner_reply_feedback" not in state["context"]


def test_learner_feedback_exhaustion_stops_after_two_corrections(tmp_path: Path) -> None:
    """No fabricated reply is substituted when the learner repeatedly chooses silence."""
    learner = _built()["learners"][1]
    state = _state(learner, LearnerOutbox())
    client = FakeClient([{"state": "Waiting"}] * 3)
    learner._runtime.set_client_for_testing(client, response_root=tmp_path)
    learner.on_task(state, "hlreasoner_0", "Review evidence")
    for _ in range(4):
        learner.step(state)
    assert len(client.responses.calls) == 3
    assert len(state["context"]["learner_reply_feedback_history"]) == 3
    assert state["failure"]["stage"] == "dispatch"
    assert not state["sent_boundaries"]


def test_learner_can_wait_for_requested_memories(tmp_path: Path) -> None:
    """A real outstanding memory request permits an otherwise empty response."""
    learner = _built()["learners"][1]
    state = _state(learner, LearnerOutbox())
    client = FakeClient([{"state": "Need evidence", "request_memories": True}, {"state": "Still awaiting requested memories"}])
    learner._runtime.set_client_for_testing(client, response_root=tmp_path)
    learner.on_task(state, "hlreasoner_0", "Review evidence")
    learner.step(state)
    learner.on_model_request(state, "hlreasoner_0")
    learner.step(state)
    assert len(client.responses.calls) == 2
    assert state["waiting_requests"] and not state["pending"]
    assert "learner_reply_feedback" not in state["context"]
    assert state["failure"] is None


def test_learner_feedback_respects_episode_cutoff(tmp_path: Path) -> None:
    """Feedback cannot extend the scientific episode deadline."""
    learner = _built()["learners"][1]
    state = _state(learner, LearnerOutbox())
    client = FakeClient([{"state": "Waiting"}])
    learner._runtime.set_client_for_testing(client, response_root=tmp_path)
    learner.on_task(state, "hlreasoner_0", "Review evidence")
    learner.step(state)
    state._time_func = lambda: 600.0
    learner.step(state)
    assert len(client.responses.calls) == 1
    assert state["termination_reason"] == "time_limit"
    assert not state["sent_boundaries"]


def test_behavior_cutoff_suppresses_new_root_cycle_without_second_scheduler(
    tmp_path: Path,
) -> None:
    module = _built()["ll_reasoner"]
    client = FakeClient(
        [LLReasonerResponse(state="late", request_observation=True)]
    )
    module._runtime.set_client_for_testing(  # noqa: SLF001
        client,
        profile=MODEL_POLICY.fast,
        response_root=tmp_path,
    )
    state = _state(module, LLOutbox(), time=STANDARD_BEHAVIOR_CUTOFF + 1.0)
    module.step(state)
    assert not state.outbox
    assert state["admission_open"] is False
    assert state["suppressed_dispatches"] == []
    assert client.responses.calls == []
    assert state["termination_reason"] == "time_limit"
    assert state.outbox.pop_term_request()[0] is True

@pytest.mark.parametrize("crosses_cap", [False, True])
def test_any_role_budget_stops_agent_without_software_failure(tmp_path: Path, crosses_cap: bool) -> None:
    """Both pre-call refusal and a response crossing the cap are scientific stops."""
    from mha_exp_common.openai_budget import BudgetExceededError
    module = _built()["ll_reasoner"]
    client = FakeClient([LLReasonerResponse(state="done")] if crosses_cap else [BudgetExceededError("cap", request_forwarded=False)])
    snapshot = client.usage_snapshot
    client.usage_snapshot = lambda: {**snapshot(), "budget_exhausted": True}
    module._runtime.set_client_for_testing(client, profile=MODEL_POLICY.fast, response_root=tmp_path)
    state = _state(module, LLOutbox())
    module.step(state)
    assert state["failure"] is None
    assert state["termination_reason"] == "budget_exhausted"
    assert state.outbox.pop_term_request()[0] is True
    assert state["usage"]["budget_exhausted"] is True


def test_every_cognitive_role_retries_failed_requests_and_preserves_arrivals(tmp_path: Path) -> None:
    """API failures retain the work and release the module until a bounded retry."""
    built = _built()
    modules = [built[k] for k in ("ll_reasoner", "knowledge", "hl_reasoner", "goal_graph", "memory")] + built["learners"]
    for module in modules:
        state = _state(module, TerminationOutbox())
        state["initial_call_pending"] = True
        state["pending"] = [{"message_id": "old", "kind": "test", "payload": {}}]
        def arrive() -> None:
            state["pending"].append({"message_id": "new", "kind": "test", "payload": {}})
        client = FakeClient([TimeoutError("sensitive error text"), {"state": "recovered"}], hook=arrive)
        runtime = module._runtime
        runtime.set_client_for_testing(client, response_root=tmp_path / module.module_id)
        assert runtime.call(state, module.module_id) is None
        assert [m["message_id"] for m in state["pending"]] == ["old", "new"]
        assert state["failure"] is None
        assert runtime.call(state, module.module_id) is None
        assert len(client.responses.calls) == 1
        state._time_func = lambda: 1.0
        assert runtime.call(state, module.module_id) is not None
        assert state["pending"] == [] and state["api_retry_count"] == 0
        assert state["api_recovery_count"] == 1 and state["repair_count"] == 0
        assert 'sensitive error text' not in str(state["request_failure_history"])
        assert "request_feedback" in client.responses.calls[1]["input"]
        assert "request_feedback" not in state["context"]


@pytest.mark.parametrize("outputs", [
    [TimeoutError(), TimeoutError(), TimeoutError()],
    [{"state": "invalid", "action": "bad"}] * 3,
    [SimpleNamespace(state="not the response schema")] * 3,
])
def test_failed_api_or_schema_stops_after_three_attempts(tmp_path: Path, outputs: list[Any]) -> None:
    """Repeated failures terminate promptly and leave every failed attempt inspectable."""
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    client = FakeClient(outputs)
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    for now in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
        state._time_func = lambda now=now: now
        module.step(state)
    assert len(client.responses.calls) == 3
    assert len(state["request_failure_history"]) == 3
    assert state["failure"]["stage"] == "api_call"
    assert state.outbox.pop_term_request()[0]
    assert not state.outbox


def test_api_retry_does_not_cross_behavior_deadline(tmp_path: Path) -> None:
    """A retry may not turn a scientific time stop into a software failure."""
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox(), time=599.0)
    client = FakeClient([TimeoutError(), {"state": "never called"}])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.step(state)
    state._time_func = lambda: 600.0
    module.step(state)
    assert len(client.responses.calls) == 1
    assert state["termination_reason"] == "time_limit" and state["failure"] is None


@pytest.mark.parametrize("request_timeout", [30.0, 60.0])
def test_api_timeout_client_has_no_hidden_sdk_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_timeout: float) -> None:
    """The socket timeout and visible runtime attempts own request recovery."""
    configured = {}
    monkeypatch.setattr(llm, "load_runtime_credentials", lambda *a, **k: (llm.encode_api_key("unit-test-placeholder"), {}))
    def client_factory(**kwargs: Any) -> FakeClient:
        configured.update(kwargs)
        return FakeClient([])
    monkeypatch.setattr(llm, "BudgetedOpenAIClient", client_factory)
    runtime = llm.RoleRuntime("ll_reasoner", LLReasonerResponse)
    runtime.initialize(credential_path="unused", environment_prompt="test", value_system="",
                       profile=MODEL_POLICY.fast.as_dict(), max_budget_usd="2",
                       response_log_root=str(tmp_path), request_timeout=request_timeout)
    assert configured["timeout"] == request_timeout and configured["max_retries"] == 0
    runtime.close()


def test_progress_for_previously_received_goal_keeps_native_payload(tmp_path: Path) -> None:
    """Reproduce the paid run: new selection does not invalidate an earlier progress ID."""
    from mhagenta import Belief, Goal
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    previous = Goal([Belief("Holding", ("B4",))], goal_id="sub_0_hold_B4", status="active")
    current = Goal([Belief("On", ("B4", "B7"))], goal_id="sub_1", status="active")
    module.on_goal_update(state, "goalgraph_0", [previous])
    module.on_goal_update(state, "goalgraph_0", [current])
    client = FakeClient([{"state": "Earlier goal achieved", "goal_progress": [{"goal_id": "sub_0_hold_B4", "status": "achieved"}]}])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.step(state)
    assert state["failure"] is None and not state["progress_feedback_history"]
    assert state["active_goals"][0]["extras"]["goal_id"] == "sub_1"
    assert set(state["goal_ledger"]) == {"sub_0_hold_B4", "sub_1"}
    message = list(state.outbox)[0]
    goal = message[3]["goals"][0]
    assert goal.extras["goal_id"] == "sub_0_hold_B4" and goal.extras["status"] == "achieved"
    assert goal.state[0].predicate == "Holding"


def test_unknown_progress_gets_model_feedback_before_any_dispatch(tmp_path: Path) -> None:
    """An invalid model reference cannot cause partial sends or an invented replacement."""
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    client = FakeClient([
        {"state": "mistake", "request_observation": True, "goal_progress": [{"goal_id": "unseen", "status": "achieved"}]},
        {"state": "I choose to send nothing"},
    ])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.step(state)
    assert not state.outbox and not state["sent_boundaries"]
    assert state["failure"] is None
    module.step(state)
    assert not state.outbox and state["failure"] is None
    assert "never been received" in client.responses.calls[1]["input"]
    assert len(state["progress_feedback_history"]) == 1


def test_missing_observation_wakes_model_and_matching_reply_clears_reminder(tmp_path: Path) -> None:
    """A lost sensor response causes a model-selected retry, not indefinite idle."""
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    client = FakeClient([{"state": "Retry sensor", "request_observation": True}, {"state": "Evidence received"}])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.on_first(state)
    for now in (1.0, 29.0):
        state._time_func = lambda now=now: now
        module.step(state)
    assert not client.responses.calls
    state._time_func = lambda: 30.0
    module.step(state)
    assert len(client.responses.calls) == 1
    assert state["cycle_sequence"] == 2
    assert "unanswered_requests" in client.responses.calls[0]["input"]
    # A late response to the old request cannot clear the retry's wait.
    module.on_observation(state, "perceptor_0", Observation("old"), cycle_id="cycle:0")
    assert state["waiting_requests"]
    module.on_observation(state, "perceptor_0", Observation("new"), cycle_id="cycle:1")
    assert state["waiting_requests"] == {}
    assert "unanswered_requests" not in state["context"]


def test_unanswered_request_stops_after_two_reminders(tmp_path: Path) -> None:
    """Even valid but unhelpful empty model answers cannot leave a missing reply hanging."""
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    client = FakeClient([{"state": "Still waiting"}, {"state": "Still no reply"}])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.on_first(state)
    for now in (30.0, 60.0, 90.0):
        state._time_func = lambda now=now: now
        module.step(state)
    assert len(client.responses.calls) == 2
    assert state["failure"]["kind"] == "ResponseTimeout"
    assert state.outbox.pop_term_request()[0]


def test_action_freshness_tracks_matching_status_and_post_result_observation(tmp_path: Path) -> None:
    """Goal-only wakeups get factual in-flight context; delayed old observations stay stale."""
    from mhagenta import Belief, Goal
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    module.on_goal_update(state, "goalgraph_0", [Goal([Belief("Holding", ("B4",))], goal_id="g", status="active")])
    client = FakeClient([{"state": "Move", "action": "Move-Right"}, {"state": "Waiting for action status"}, {"state": "Action outcome uncertain", "request_observation": True}])
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.step(state)
    module.on_goal_update(state, "goalgraph_0", [Goal([Belief("On", ("B4", "B7"))], goal_id="next", status="active")])
    module.step(state)
    assert '"status_received_at": null' in client.responses.calls[1]["input"]
    assert '"observation_after_action": false' in client.responses.calls[1]["input"]
    state._time_func = lambda: 30.0
    module.step(state)
    assert state["action_sequence"] == 1  # Missing acknowledgement never automatically replays an action.
    state._time_func = lambda: 31.0
    module.on_action_status(state, "actuator_0", ActionStatus({"success": True}), action_id="other")
    assert state["action_observation_state"]["last_action"]["status_received_at"] is None
    module.on_action_status(state, "actuator_0", ActionStatus({"success": True}), action_id="action:0")
    module.on_observation(state, "perceptor_0", Observation("requested before acknowledgement"), cycle_id="cycle:0")
    assert not state["action_observation_state"]["observation_after_action"]
    module._request_observation(state)
    module.on_observation(state, "perceptor_0", Observation("fresh"), cycle_id="cycle:1")
    assert state["action_observation_state"]["observation_after_action"]
    assert not state["waiting_requests"]


def test_terminal_stop_cancels_inflight_output_and_blocks_new_requests(tmp_path: Path) -> None:
    """A sibling's stop during a call cannot cause a late action or another API call."""
    from mha_exp_level2_bw.exp2_7.terminal import publish_terminal, finalize_module
    from mha_exp_level2_bw.exp2_7.evidence import send_message
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    stopper = _state(_built()["knowledge"], TerminationOutbox(), time=1.0)
    for item in (state, stopper):
        item["terminal_file"] = str(tmp_path / "terminal.json")
    client = FakeClient([LLReasonerResponse(state="late", request_observation=True)],
                        hook=lambda: publish_terminal(stopper, "time_limit"))
    module._runtime.set_client_for_testing(client, response_root=tmp_path / "responses")
    module.step(state)
    assert not state.outbox
    assert state["admission_open"] is False
    module.step(state)
    assert len(client.responses.calls) == 1
    dispatched = []
    send_message(state, recipient="perceptor_0", kind="request_observation",
                 dispatch=lambda **meta: dispatched.append(meta))
    assert not dispatched and not state["sent_boundaries"]
    assert len(state["suppressed_terminal_dispatches"]) == 1
    state["pending"].append({"message_id": "terminal-status", "kind": "action_status"})
    state["waiting_requests"]["observation"] = {"kind": "request_observation"}
    finalize_module(state)
    assert not state["pending"] and not state["waiting_requests"]
    assert state["cancelled_inputs"][0]["message_id"] == "terminal-status"
    assert "observation" in state["cancelled_waits"]
    assert state["shutdown_record"]["terminal"] == stopper["terminal_observed"]
    record = json.loads((tmp_path / "responses/llreasoner_0.jsonl").read_text())
    assert record["cancelled_at_termination"] is True


def test_terminal_first_stop_is_shared_and_never_overwritten(tmp_path: Path) -> None:
    """All modules observe one atomic stop record, including late shutdown hooks."""
    from mha_exp_level2_bw.exp2_7.terminal import publish_terminal, observe_terminal
    first = _state(_built()["ll_reasoner"], TerminationOutbox(), time=10.0)
    second = _state(_built()["knowledge"], TerminationOutbox(), time=12.0)
    for state in (first, second):
        state["terminal_file"] = str(tmp_path / "terminal.json")
    record = publish_terminal(first, "task_completed")
    assert publish_terminal(second, "time_limit") == record
    assert observe_terminal(second) == record
    assert first["admission_open"] is False and second["admission_open"] is False


@pytest.mark.parametrize("profile, expected", [(MODEL_POLICY.fast, 4096), (MODEL_POLICY.deliberative, 8192)])
def test_output_allowance_matches_reasoning_profile_and_is_recorded(tmp_path: Path, profile: Any, expected: int) -> None:
    """High reasoning has room for complete JSON; low reasoning keeps its existing cap."""
    module = _built()["ll_reasoner"]
    state = _state(module, LLOutbox())
    client = FakeClient([LLReasonerResponse(state="done")])
    module._runtime.set_client_for_testing(client, profile=profile, response_root=tmp_path)
    module.step(state)
    assert client.responses.calls[0]["max_output_tokens"] == expected
    assert client.responses.calls[0]["reasoning"]["effort"] == profile.reasoning_effort
    record = json.loads((tmp_path / "llreasoner_0.jsonl").read_text())
    assert record["max_output_tokens"] == expected

def _memory_with_responses(tmp_path: Path, responses: list[Any]) -> tuple[Any, Any, FakeClient]:
    """Receive a native observation under a memory identity distinct from its source."""
    from mhagenta.outboxes import MemoryOutbox
    module = _built()['memory']
    client = FakeClient(responses)
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    state = _state(module, MemoryOutbox())
    module.on_observation_update(state, 'knowledge_0', [Observation('source observation:abc')])
    module.on_memory_request(state, 'learner_0')
    return module, state, client


def test_memory_feedback_retries_unknown_source_id_without_partial_retention(tmp_path: Path) -> None:
    """The paid failure gets feedback before any valid prefix is applied."""
    module, state, client = _memory_with_responses(tmp_path, [
        {'state': 'Mistake', 'retained_memory_ids': ['memory:0', 'observation:abc'],
         'deliveries': [{'recipient': 'learner_0', 'memory_ids': ['memory:0']}]},
        {'state': 'Corrected selection', 'retained_memory_ids': ['memory:0'],
         'deliveries': [{'recipient': 'learner_0', 'memory_ids': ['memory:0']}]},
    ])
    previous_text = state['text_state']
    module.step(state)
    assert state['memories'] == [] and not state.outbox and not state['sent_boundaries']
    assert state['failure'] is None and state['text_state'] == previous_text
    module.step(state)
    assert [item['memory_id'] for item in state['memories']] == ['memory:0']
    assert len(list(state.outbox)) == 1
    assert 'Unknown retained memory_id: observation:abc' in client.responses.calls[1]['input']
    assert state['memory_repair_attempts'] == 0 and 'memory_feedback' not in state['context']


@pytest.mark.parametrize('delivery', [
    {'recipient': 'learner_0', 'memory_ids': ['missing']},
    {'recipient': 'learner_1', 'memory_ids': ['memory:0']},
])
def test_memory_invalid_delivery_can_be_replaced_by_model_empty_choice(tmp_path: Path, delivery: dict) -> None:
    """Validate the entire delivery list before retaining or sending its valid prefix."""
    module, state, client = _memory_with_responses(tmp_path, [
        {'state': 'Mistake', 'retained_memory_ids': ['memory:0'],
         'deliveries': [{'recipient': 'learner_0', 'memory_ids': ['memory:0']}, delivery]},
        {'state': 'I choose nothing', 'retained_memory_ids': [], 'deliveries': []},
    ])
    module.step(state)
    assert not state.outbox and not state['memories']
    module.step(state)
    assert not state.outbox and not state['memories'] and state['failure'] is None
    assert len(client.responses.calls) == 2


def test_memory_reference_corrections_are_bounded(tmp_path: Path) -> None:
    """Three invalid responses retain evidence and halt rather than retry forever."""
    bad = {'state': 'Unknown', 'retained_memory_ids': ['observation:abc'], 'deliveries': []}
    module, state, client = _memory_with_responses(tmp_path, [bad, bad, bad])
    for _ in range(4):
        module.step(state)
    assert len(client.responses.calls) == 3 and len(state['memory_feedback_history']) == 3
    assert state['failure']['stage'] == 'dispatch'
    assert not state['memories'] and not state['sent_boundaries']

def test_long_request_crossing_behavior_cutoff_cannot_dispatch(tmp_path: Path) -> None:
    """A response allowed 60 seconds still cannot act beyond the 600-second episode."""
    module = _built()['ll_reasoner']
    state = _state(module, LLOutbox(), time=570.0)
    client = FakeClient([{'state': 'Late response', 'request_observation': True}],
                        hook=lambda: setattr(state, '_time_func', lambda: 615.0))
    module._runtime.set_client_for_testing(client, response_root=tmp_path)
    module.step(state)
    assert len(client.responses.calls) == 1
    assert state['termination_reason'] == 'time_limit'
    assert not state['sent_boundaries'] and not state.outbox
    module.step(state)
    assert len(client.responses.calls) == 1
