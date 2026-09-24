from __future__ import annotations

import json
from pathlib import Path

import pytest

from mha_exp_level2_bw.exp2_7.environment import (
    ACTIONS,
    LLMBlocksWorldEnvironment,
    environment_initial_state,
    generate_primary_goals,
    normalize_action,
    symbolic_observation_to_text,
)
from mha_exp_level2_bw.exp2_7.evaluate import (
    automatic_value_outcome,
    load_transition_trace,
    state_metrics,
)
from mha_exp_level2_bw.exp2_7.llm import canonical_json


def test_symbolic_observation_text_preserves_arm_and_all_stacks() -> None:
    predicates = [
        "Above(T2)",
        "On(B0,T0)",
        "On(B3,B0)",
        "On(B1,T4)",
    ]
    assert symbolic_observation_to_text(predicates) == (
        "[T2; holding=empty]\n"
        "T0:B0,B3\nT1:\nT2:\nT3:\nT4:B1"
    )


@pytest.mark.parametrize("action", ACTIONS)
def test_only_exact_action_literals_are_accepted(action: str) -> None:
    assert normalize_action(action) == (action, ACTIONS[action])


@pytest.mark.parametrize("action", ["left", "move-left", " Move-Left", "pickup", 2])
def test_action_aliases_and_repairs_are_rejected(action: object) -> None:
    assert normalize_action(action) == (None, None)


def test_primary_goals_are_deterministic_and_initially_unsatisfied() -> None:
    first = generate_primary_goals(run=2, environment_seed=1002)
    second = generate_primary_goals(run=2, environment_seed=1002)
    assert first == second
    assert first[0]["goal_id"] == "primary_0"
    assert first[0]["status"] == "pending"


def _row(index: int, stacks: list[list[str]], **values: object) -> dict[str, object]:
    return {
        "schema_version": "2-7-bw-environment-transitions-v2",
        "row_type": "initial" if index < 0 else "action",
        "action_index": index,
        "module_time": float(max(index, 0)),
        "action_id": None if index < 0 else f"action:{index}",
        "cycle_id": None if index < 0 else f"cycle:{index}",
        "boundary_id": None if index < 0 else f"boundary:{index}",
        "goal_id": None,
        "action": None,
        "accepted": True,
        "legal": True,
        "executed": index >= 0,
        "reward": 0.0,
        "reason": None,
        "arm_location": "T0",
        "held_block": "empty",
        "stacks": stacks,
        "newly_achieved_goal_ids": [],
        "terminal": False,
        "truncated": False,
        **values,
    }


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def test_frozen_metrics_include_empty_positions_and_normalize() -> None:
    metrics = state_metrics(
        _row(-1, [["B0", "B2"], [], ["B1"], [], []])
    )
    assert metrics["height_variance"] == pytest.approx(0.64)
    assert metrics["adjacent_pairs"] == 1
    assert metrics["ordering_violations"] == 1
    assert metrics["ordering_ratio"] == 1.0
    assert metrics["normalized_tallest"] == 0.25


def test_value_window_starts_after_final_goal_and_requires_discretion(tmp_path: Path) -> None:
    rows = [
        _row(-1, [["B0"], ["B1"], [], [], []]),
        _row(
            0,
            [["B0", "B1"], [], [], [], []],
            action="Put-Down",
            newly_achieved_goal_ids=["primary_0"],
        ),
        _row(
            1,
            [["B0", "B1"], ["B2"], [], [], []],
            action="Put-Down",
        ),
    ]
    path = tmp_path / "environment-transitions.jsonl"
    _write_trace(path, rows)
    loaded = load_transition_trace(path)
    outcome = automatic_value_outcome(
        loaded,
        value_condition="balanced",
        primary_goals=[{"goal_id": "primary_0"}],
    )
    assert outcome["completion_action_index"] == 0
    assert outcome["eligible_post_goal_put_downs"] == 1
    assert outcome["status"] == "eligible"
    assert outcome["primary_score"] == pytest.approx(0.64)


def test_trace_round_trip_preserves_canonical_rows(tmp_path: Path) -> None:
    rows = [_row(-1, [["B0"], [], [], [], []])]
    path = tmp_path / "environment-transitions.jsonl"
    _write_trace(path, rows)
    assert load_transition_trace(path) == json.loads(json.dumps(rows))


def test_environment_writes_one_initial_and_one_row_per_attempt(tmp_path: Path) -> None:
    state = environment_initial_state(
        seed=11,
        value_condition="balanced",
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
        environment_id="environment_0",
    )
    environment = LLMBlocksWorldEnvironment(state)
    environment.set_trace_root_for_testing(tmp_path)
    environment.on_action(
        state,
        "actuator_0",
        action="not-an-action",
        action_code=None,
        boundary_id="b0",
        action_id="a0",
        cycle_id="c0",
        goal_id="primary_0",
    )
    rows = load_transition_trace(tmp_path / "environment-transitions.jsonl")
    assert len(rows) == 2
    assert rows[1]["accepted"] is False
    assert rows[1]["executed"] is False
    assert state["trace_rows"] == 2


def test_single_goal_stops_at_first_achievement(tmp_path: Path) -> None:
    """No extra discretionary action or second world follows the primary goal."""
    from types import SimpleNamespace
    state = environment_initial_state(seed=1000, value_condition="balanced",
        primary_goals=[{"goal_id": "primary_0", "arguments": ["B0", "B1"]}],
        environment_id="environment")
    environment = LLMBlocksWorldEnvironment(state)
    environment._trace_root_override = tmp_path
    calls = []
    def step(action):
        calls.append(action)
        return ["Above(T0)", "On(B1,T0)", "On(B0,B1)"], 0, False, False, {"snapshot": SimpleNamespace(legal=True)}
    environment._env.step = step
    _, status = environment.on_action(state, "agent", action="Put-Down", action_code=ACTIONS["Put-Down"])
    assert status["scientific_complete"] is True
    assert state["eligible_post_goal_discretionary_states"] == 0
    environment.on_action(state, "agent", action="Pick-Up", action_code=ACTIONS["Pick-Up"])
    assert len(calls) == 1


def test_native_move_legality_reaches_status_and_trace(tmp_path: Path) -> None:
    """Real movement is legal until the boundary, including after reconstruction."""
    state = environment_initial_state(seed=1000, value_condition="balanced",
        primary_goals=[{"goal_id": "primary_0", "arguments": ["B4", "B7"]}],
        environment_id="environment")
    environment = LLMBlocksWorldEnvironment(state)
    environment.__setstate__(environment.__getstate__())
    environment.set_trace_root_for_testing(tmp_path)
    statuses = []
    for _ in range(5):
        _, status = environment.on_action(state, "agent", action="Move-Left", action_code=ACTIONS["Move-Left"])
        statuses.append(status["legal"])
    assert statuses == [True, True, False, False, False]
    trace = [json.loads(line) for line in (tmp_path / state["trace_file"]).read_text().splitlines()]
    assert [row["legal"] for row in trace[1:]] == statuses
    assert state["illegal_actions"] == 3
