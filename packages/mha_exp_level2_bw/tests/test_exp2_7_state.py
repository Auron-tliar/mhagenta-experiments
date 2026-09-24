from __future__ import annotations

import json
from pathlib import Path

import pytest
from mhagenta import Directory, State
from mhagenta.outboxes import (
    ActuatorOutbox,
    GoalGraphOutbox,
    HLOutbox,
    KnowledgeOutbox,
    LearnerOutbox,
    LLOutbox,
    MemoryOutbox,
    PerceptorOutbox,
)

from mha_exp_level2_bw.exp2_7.llm import (
    STATE_SCHEMA_VERSION,
    json_safe,
    normalized_failure,
)
from mha_exp_level2_bw.exp2_7.runner import LifecycleConfig, _build_agent_modules


def _states() -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    modules, schemas = _build_agent_modules(
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
    )
    states = {
        "perceptor_0": modules["perceptor"].initial_state,
        "actuator_0": modules["actuator"].initial_state,
        "llreasoner_0": modules["ll_reasoner"].initial_state,
        "knowledge_0": modules["knowledge"].initial_state,
        "hlreasoner_0": modules["hl_reasoner"].initial_state,
        "goalgraph_0": modules["goal_graph"].initial_state,
        "memory_0": modules["memory"].initial_state,
        "learner_0": modules["learners"][0].initial_state,
        "learner_1": modules["learners"][1].initial_state,
    }
    return states, schemas


OUTBOXES = {
    "perceptor_0": PerceptorOutbox,
    "actuator_0": ActuatorOutbox,
    "llreasoner_0": LLOutbox,
    "knowledge_0": KnowledgeOutbox,
    "hlreasoner_0": HLOutbox,
    "goalgraph_0": GoalGraphOutbox,
    "memory_0": MemoryOutbox,
    "learner_0": LearnerOutbox,
    "learner_1": LearnerOutbox,
}


@pytest.mark.parametrize("module_id", sorted(OUTBOXES))
def test_all_nine_real_states_match_schema_and_are_strict_json(module_id: str) -> None:
    states, schemas = _states()
    state = State(
        agent_id="agent",
        module_id=module_id,
        time_func=lambda: 1.0,
        directory=Directory(),
        outbox=OUTBOXES[module_id](),
        **states[module_id],
    )
    assert set(state.dump()) == set(schemas[module_id])
    assert state["state_schema_version"] == STATE_SCHEMA_VERSION
    json.dumps(state.dump(), allow_nan=False)


def test_final_state_json_round_trip_preserves_evaluator_fields(tmp_path: Path) -> None:
    states, schemas = _states()
    for module_id, initial in states.items():
        state = State(
            agent_id="agent",
            module_id=module_id,
            time_func=lambda: 2.0,
            directory=Directory(),
            outbox=OUTBOXES[module_id](),
            **initial,
        )
        state["pending"].append(
            {
                "message_id": "m0",
                "sender": "source",
                "kind": "test",
                "payload": {},
                "metadata": {},
                "received_at": 1.0,
            }
        )
        state["failure"] = normalized_failure(
            kind="TestFailure", stage="test", module_time=2.0
        )
        path = tmp_path / f"{module_id}.json"
        path.write_text(json.dumps(state.dump(), allow_nan=False), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert set(loaded) == set(schemas[module_id])
        assert loaded["pending"][0]["message_id"] == "m0"
        assert loaded["failure"]["kind"] == "TestFailure"


def test_predeclared_fields_survive_an_autosave_boundary() -> None:
    states, schemas = _states()
    initial = states["llreasoner_0"]
    state = State(
        agent_id="agent",
        module_id="llreasoner_0",
        time_func=lambda: 3.0,
        directory=Directory(),
        outbox=LLOutbox(),
        **initial,
    )
    first = json.loads(json.dumps(state.dump(), allow_nan=False))
    state["last_response_id"] = "response_after_autosave"
    state["sent_boundaries"].append({"boundary_id": "b0"})
    second = json.loads(json.dumps(state.dump(), allow_nan=False))
    assert set(first) == set(second) == set(schemas["llreasoner_0"])
    assert second["last_response_id"] == "response_after_autosave"
    assert second["sent_boundaries"] == [{"boundary_id": "b0"}]


def test_undeclared_state_assignment_is_not_mistaken_for_persistence() -> None:
    states, _ = _states()
    state = State(
        agent_id="agent",
        module_id="llreasoner_0",
        time_func=lambda: 0.0,
        directory=Directory(),
        outbox=LLOutbox(),
        **states["llreasoner_0"],
    )
    state["undeclared"] = "lost"
    assert "undeclared" not in state.dump()


@pytest.mark.parametrize(
    "value",
    [{"bad"}, Path("not-persistent"), RuntimeError("not-persistent")],
)
def test_non_json_runtime_objects_are_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        json_safe(value)


def test_failure_shape_contains_no_exception_or_traceback() -> None:
    failure = normalized_failure(
        kind="ValueError", stage="dispatch", module_time=1.25, response_id="r1"
    )
    assert failure == {
        "kind": "ValueError",
        "stage": "dispatch",
        "message": "operation failed",
        "module_time": 1.25,
        "response_id": "r1",
    }
    json.dumps(failure, allow_nan=False)
