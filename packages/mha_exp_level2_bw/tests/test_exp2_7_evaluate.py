from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from mha_exp_level2_bw.exp2_7.evaluate import (
    architecture_checks,
    evaluate_run,
    learner_outcomes,
    matched_summary,
)
from mha_exp_level2_bw.exp2_7.llm import MODEL_POLICY, STATE_SCHEMA_VERSION, canonical_json


MODULES = (
    "perceptor_0",
    "actuator_0",
    "llreasoner_0",
    "knowledge_0",
    "hlreasoner_0",
    "goalgraph_0",
    "memory_0",
    "learner_0",
    "learner_1",
)
LLM_MODULES = {
    "llreasoner_0": MODEL_POLICY.fast.model,
    "knowledge_0": MODEL_POLICY.deliberative.model,
    "hlreasoner_0": MODEL_POLICY.deliberative.model,
    "goalgraph_0": MODEL_POLICY.deliberative.model,
    "memory_0": MODEL_POLICY.deliberative.model,
}


def _state(module_id: str) -> dict[str, Any]:
    selected = LLM_MODULES.get(
        module_id,
        MODEL_POLICY.deliberative.model if module_id.startswith("learner_") else None,
    )
    reasoning_effort = (
        MODEL_POLICY.deliberative.reasoning_effort
        if module_id in {"knowledge_0", "hlreasoner_0", "memory_0"}
        else MODEL_POLICY.fast.reasoning_effort
        if selected is not None
        else None
    )
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "sent_boundaries": [],
        "processed_boundaries": [],
        "failure": None,
        "processed_input_ids": ["input:0"] if module_id in LLM_MODULES else [],
        "response_records": 1 if module_id in LLM_MODULES else 0,
        "selected_model": selected,
        "reasoning_effort": reasoning_effort,
        "actual_models": [selected] if module_id in LLM_MODULES else [],
        "pending": [],
        "in_flight": None,
        "incomplete_reason": None,
        "repair_count": 0,
        "used_revision_ids": [],
        "involvement_requested_by_reasoner": False,
        "involvement_status": "not_requested",
        "operational_involvement_decision": "not_requested",
        "revision_id": None,
    }


def _valid_fixture() -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    states = {module_id: _state(module_id) for module_id in MODULES}
    environment = {
        "environment_id": "environment_0",
        "processed_boundaries": [],
        "trace_rows": 2,
        "initial_state_sha256": "initial",
    }
    sequence = 0

    def edge(source: str, recipient: str, kind: str, **metadata: Any) -> None:
        nonlocal sequence
        boundary_id = f"b{sequence}"
        sequence += 1
        record = {
            "boundary_id": boundary_id,
            "source": source,
            "recipient": recipient,
            "kind": kind,
            "module_time": float(sequence),
            **metadata,
        }
        states[source]["sent_boundaries"].append(record)
        processed = dict(record)
        if recipient == "environment_0":
            environment["processed_boundaries"].append(processed)
        else:
            states[recipient]["processed_boundaries"].append(processed)

    obs = {
        "evidence_id": "obs0",
        "observation_payload_sha256": "obs-hash",
        "cycle_id": "cycle0",
    }
    edge("perceptor_0", "llreasoner_0", "send_observation", **obs)
    edge("llreasoner_0", "knowledge_0", "send_beliefs", **obs)
    evaluation = {
        "evidence_id": "evaluation0",
        "source_evidence_id": "obs0",
        "cycle_id": "cycle0",
    }
    edge("knowledge_0", "memory_0", "send_observations", **evaluation)
    edge("knowledge_0", "hlreasoner_0", "send_beliefs", **evaluation)
    goal = {
        "goal_id": "primary_0",
        "goal_payload_sha256": "goal-hash",
        "input_evaluation_id": "evaluation0",
    }
    edge("hlreasoner_0", "goalgraph_0", "send_goals", **goal)
    edge("goalgraph_0", "llreasoner_0", "send_goals", **goal)
    action = {
        "action_id": "action0",
        "cycle_id": "cycle0",
        "goal_id": "primary_0",
        "goal_payload_sha256": "goal-hash",
    }
    edge("llreasoner_0", "actuator_0", "request_action", **action)
    edge("actuator_0", "environment_0", "environment_action", **action)
    edge("actuator_0", "llreasoner_0", "send_status", **action)
    edge(
        "llreasoner_0",
        "goalgraph_0",
        "send_goal_update",
        goal_id="primary_0",
        goal_payload_sha256="goal-hash",
        cycle_id="cycle0",
    )
    edge(
        "goalgraph_0",
        "hlreasoner_0",
        "send_goals",
        goal_id="primary_0",
        goal_payload_sha256="goal-hash",
    )
    trace = [
        {
            "schema_version": "2-7-bw-environment-transitions-v1",
            "row_type": "initial",
            "action_index": -1,
            "module_time": 0.0,
            "action_id": None,
            "cycle_id": None,
            "boundary_id": None,
            "goal_id": None,
            "action": None,
            "accepted": True,
            "legal": True,
            "executed": False,
            "reward": 0.0,
            "reason": None,
            "arm_location": "T0",
            "held_block": "empty",
            "stacks": [["B0"], ["B1"], [], [], []],
            "newly_achieved_goal_ids": [],
            "terminal": False,
            "truncated": False,
        },
        {
            "schema_version": "2-7-bw-environment-transitions-v1",
            "row_type": "action",
            "action_index": 0,
            "module_time": 1.0,
            "action_id": "action0",
            "cycle_id": "cycle0",
            "boundary_id": "b7",
            "goal_id": "primary_0",
            "action": "Put-Down",
            "accepted": True,
            "legal": True,
            "executed": True,
            "reward": 0.0,
            "reason": None,
            "arm_location": "T0",
            "held_block": "empty",
            "stacks": [["B1", "B0"], [], [], [], []],
            "newly_achieved_goal_ids": ["primary_0"],
            "terminal": False,
            "truncated": False,
        },
    ]
    return states, environment, trace


def _write_evidence(
    tmp_path: Path,
    states: dict[str, dict[str, Any]],
    trace: list[dict[str, Any]],
) -> tuple[Path, Path]:
    trace_path = tmp_path / "environment-transitions.jsonl"
    trace_path.write_text(
        "".join(canonical_json(row) + "\n" for row in trace), encoding="utf-8"
    )
    response_dir = tmp_path / "llm_responses"
    response_dir.mkdir()
    for module_id, model in LLM_MODULES.items():
        profile = (
            MODEL_POLICY.deliberative
            if module_id in {"knowledge_0", "hlreasoner_0", "memory_0"}
            else MODEL_POLICY.fast
        )
        response_dir.joinpath(f"{module_id}.jsonl").write_text(
            canonical_json(
                {
                    "requested_model": model,
                    "actual_model": model,
                    "reasoning_effort": profile.reasoning_effort,
                    "success": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return trace_path, response_dir


def test_complete_connected_fixture_is_architecture_valid() -> None:
    states, environment, trace = _valid_fixture()
    checks = architecture_checks(states, environment, trace)
    assert checks["architecture_valid"] is True


@pytest.mark.parametrize(
    ("source", "recipient", "kind", "field"),
    [
        ("llreasoner_0", "knowledge_0", "send_beliefs", "evidence_id"),
        ("knowledge_0", "hlreasoner_0", "send_beliefs", "evidence_id"),
        ("goalgraph_0", "llreasoner_0", "send_goals", "goal_id"),
        ("actuator_0", "llreasoner_0", "send_status", "action_id"),
    ],
)
def test_crossed_identities_are_excluded_even_when_every_edge_exists(
    source: str, recipient: str, kind: str, field: str
) -> None:
    states, environment, trace = _valid_fixture()
    record = next(
        item
        for item in states[source]["sent_boundaries"]
        if item["recipient"] == recipient and item["kind"] == kind
    )
    record[field] = "crossed"
    checks = architecture_checks(states, environment, trace)
    assert checks["all_sent_boundaries_reconciled"] is True
    assert checks["architecture_valid"] is False


def test_successful_short_path_is_scientific_outcome_but_architecture_invalid() -> None:
    states, environment, trace = _valid_fixture()
    states["llreasoner_0"]["sent_boundaries"] = [
        item
        for item in states["llreasoner_0"]["sent_boundaries"]
        if item["recipient"] != "knowledge_0"
    ]
    states["knowledge_0"]["processed_boundaries"] = []
    checks = architecture_checks(states, environment, trace)
    assert checks["connected_observation_evaluation_chain"] is False
    assert checks["architecture_valid"] is False


def test_evaluate_separates_architecture_success_and_comparability(tmp_path: Path) -> None:
    states, environment, trace = _valid_fixture()
    trace_path, response_dir = _write_evidence(tmp_path, states, trace)
    schemas = {module_id: sorted(state) for module_id, state in states.items()}
    result = evaluate_run(
        agent_states=states,
        environment_state=environment,
        trace_path=trace_path,
        response_dir=response_dir,
        primary_goals=[{"goal_id": "primary_0"}],
        value_condition="balanced",
        state_schemas=schemas,
        treatment_digest="treatment",
    )
    assert result["architecture_valid"] is True
    assert result["strict_bw_success"] is True
    assert result["cohort_comparable"] is True
    assert result["matched_inclusion"] is True
    assert result["scientific_outcomes"]["value"]["status"] == (
        "insufficient_discretionary_evidence"
    )


def test_goal_failure_remains_eligible_when_architecture_is_valid(tmp_path: Path) -> None:
    states, environment, trace = _valid_fixture()
    trace[1]["newly_achieved_goal_ids"] = []
    trace_path, response_dir = _write_evidence(tmp_path, states, trace)
    result = evaluate_run(
        agent_states=states,
        environment_state=environment,
        trace_path=trace_path,
        response_dir=response_dir,
        primary_goals=[{"goal_id": "primary_0"}],
        value_condition="balanced",
        state_schemas={module_id: sorted(state) for module_id, state in states.items()},
        treatment_digest="treatment",
    )
    assert result["architecture_valid"] is True
    assert result["strict_bw_success"] is False
    assert result["matched_inclusion"] is True


def test_timeout_after_initial_observation_is_valid_without_actions(tmp_path: Path) -> None:
    """A stalled cognitive loop is a scientific failure when initial evidence exists."""
    states, environment, trace = _valid_fixture()
    for state in states.values():
        state["sent_boundaries"] = []
        state["processed_boundaries"] = []
    states["llreasoner_0"]["termination_reason"] = "time_limit"
    environment.update(trace_rows=1, attempted_actions=0, achieved_goal_ids=[],
                       scientific_complete=False)
    trace_path, response_dir = _write_evidence(tmp_path, states, trace[:1])
    result = evaluate_run(
        agent_states=states, environment_state=environment, trace_path=trace_path,
        response_dir=response_dir, primary_goals=[{"goal_id": "primary_0"}],
        value_condition="balanced",
        state_schemas={module_id: sorted(state) for module_id, state in states.items()},
        treatment_digest="treatment",
    )
    assert result["execution_valid"] is True
    assert result["task_success"] is False
    assert result["architecture_valid"] is False
    assert result["termination_reason"] == "time_limit"


def test_missing_trace_is_reported_without_crashing(tmp_path: Path) -> None:
    states, environment, trace = _valid_fixture()
    trace_path, response_dir = _write_evidence(tmp_path, states, trace)
    trace_path.unlink()
    environment["trace_rows"] = 0
    result = evaluate_run(
        agent_states=states,
        environment_state=environment,
        trace_path=trace_path,
        response_dir=response_dir,
        primary_goals=[{"goal_id": "primary_0"}],
        value_condition="balanced",
        state_schemas={module_id: sorted(state) for module_id, state in states.items()},
        treatment_digest="treatment",
    )

    assert result["schema_version"] == "2-7-bw-result-v2"
    assert result["matched_inclusion"] is False
    assert result["comparability_checks"]["trace_present"] is False
    assert result["scientific_outcomes"]["attempted_actions"] == 0
    assert result["scientific_outcomes"]["value"] == {
        "automatic_value_effect_implemented": True,
        "qualitative_value_effect_review": "human_required",
        "completion_action_index": None,
        "eligible_post_goal_put_downs": 0,
        "status": "unavailable",
        "unavailable_reason": "environment_trace_missing",
        "primary_score": None,
        "final_metrics": None,
        "whole_run_max_normalized_tallest": None,
        "evidence_link": {
            "condition_evaluation_ids": ["evaluation0"],
            "model_authored_link_present": False,
            "linked_decisions": [],
        },
    }
    assert result["evidence"]["environment_trace"]["present"] is False
    assert result["evidence"]["environment_trace"]["sha256"] is None


def test_learner_reporting_never_claims_counterfactual_necessity() -> None:
    states, _, _ = _valid_fixture()
    states["learner_0"].update(
        involvement_requested_by_reasoner=True,
        involvement_status="revision_delivered",
        operational_involvement_decision="requested_by_reasoner",
        revision_id="learner_0:revision:0",
    )
    states["llreasoner_0"]["used_revision_ids"] = ["learner_0:revision:0"]
    outcomes = learner_outcomes(states)
    assert outcomes["learner_0"]["involvement_status"] == "revision_used"
    assert outcomes["learner_0"]["revision_used_by_reasoner"] is True
    assert outcomes["learner_0"]["counterfactual_necessity"] == "not_established"
    assert outcomes["learner_1"]["involvement_status"] == "not_requested"


def test_matched_summary_uses_oriented_delta_and_unavailable_policy() -> None:
    def summary(name: str, score: float | None, included: bool = True) -> dict[str, Any]:
        return {
            "runs": [
                {
                    "run": 0,
                    "matched_inclusion": included,
                    "primary_value_score": score,
                    "treatment_digest": "treatment",
                    "initial_state_sha256": "world",
                    "primary_goals_sha256": "goals",
                }
            ]
        }

    result = matched_summary(
        {
            "balanced": summary("balanced", 0.1),
            "number_order": summary("number_order", 0.3),
            "tallest_stack": summary("tallest_stack", 0.8),
        }
    )
    balanced = next(
        item for item in result["comparisons"] if item["target_condition"] == "balanced"
    )
    assert balanced["oriented_delta"] == pytest.approx(0.45)
    assert balanced["classification"] == "favorable"

    unavailable = matched_summary(
        {
            "balanced": summary("balanced", None),
            "number_order": summary("number_order", 0.3),
            "tallest_stack": summary("tallest_stack", 0.8),
        }
    )
    target = next(
        item
        for item in unavailable["comparisons"]
        if item["target_condition"] == "balanced"
    )
    assert target["classification"] == "unavailable"
    assert target["oriented_delta"] is None

    mismatched = {
        "balanced": summary("balanced", 0.1),
        "number_order": summary("number_order", 0.3),
        "tallest_stack": summary("tallest_stack", 0.8),
    }
    mismatched["balanced"]["runs"][0]["initial_state_sha256"] = "other-world"
    target = next(
        item
        for item in matched_summary(mismatched)["comparisons"]
        if item["target_condition"] == "balanced"
    )
    assert target["matched_identity"] is False
    assert target["classification"] == "unavailable"


@pytest.mark.parametrize("exhausted", [False, True])
def test_recovery_evidence_distinguishes_exhausted_error_from_valid_run(tmp_path: Path, exhausted: bool) -> None:
    """Recovered schema errors remain eligible; exhausted retries are execution errors."""
    states, environment, trace = _valid_fixture()
    states["llreasoner_0"]["repair_count"] = 1
    if exhausted:
        states["llreasoner_0"]["failure"] = {"kind": "ResponseTimeout", "stage": "message_reply", "module_time": 90.0}
    trace_path, response_dir = _write_evidence(tmp_path, states, trace)
    result = evaluate_run(agent_states=states, environment_state=environment,
        trace_path=trace_path, response_dir=response_dir, primary_goals=[{"goal_id": "primary_0"}],
        value_condition="balanced", state_schemas={key: sorted(value) for key, value in states.items()},
        treatment_digest="recovery-v7")
    assert result["execution_valid"] is (not exhausted)
    assert result["comparability_checks"]["no_schema_repairs"] is False
    assert (result["termination_reason"] == "execution_error") is exhausted


def _terminal_fixture() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """A verified success interrupted one recent non-action message."""
    states, environment, rows = _valid_fixture()
    terminal = {"reason": "task_completed", "module_id": "llreasoner_0", "module_time": 50.0}
    for state in states.values():
        state.update(terminal_observed=terminal, admission_open=False,
                     lifecycle={"request_timeout": 30.0, "behavior_cutoff": 600.0},
                     shutdown_record={"module_time": 51.0, "terminal": terminal})
    states["llreasoner_0"].update(scientific_complete=True, termination_reason="task_completed",
                                 scientific_completion_evidence={"action_id": "action0"})
    environment["scientific_complete"] = True
    states["knowledge_0"]["sent_boundaries"].append({
        "boundary_id": "cancelled:0", "source": "knowledge_0", "recipient": "memory_0",
        "kind": "send_beliefs", "module_time": 49.99})
    return states, environment, rows


def test_terminal_accounting_keeps_delivery_diagnostic_and_accepts_verified_cancellation() -> None:
    """Cancellation is distinct from receipt and does not break valid earlier chains."""
    from mha_exp_level2_bw.exp2_7.terminal import terminal_accounting
    states, environment, rows = _terminal_fixture()
    accounting = terminal_accounting(states, environment, rows)
    assert accounting["terminal_verified"]
    assert len(accounting["cancelled_boundaries"]) == 1
    checks = architecture_checks(states, environment, rows)
    assert checks["all_sent_boundaries_reconciled"] is False
    assert checks["all_sent_boundaries_accounted"] is True
    assert checks["architecture_valid"] is True


@pytest.mark.parametrize("mutation", ["old", "after", "action", "duplicate", "unverified", "unfinished"])
def test_terminal_accounting_does_not_hide_real_integrity_errors(mutation: str) -> None:
    """Only a narrow, verified terminal cancellation may replace missing receipt."""
    from mha_exp_level2_bw.exp2_7.terminal import terminal_accounting
    states, environment, rows = _terminal_fixture()
    message = states["knowledge_0"]["sent_boundaries"][-1]
    if mutation == "old": message["module_time"] = 10.0
    elif mutation == "after": message["module_time"] = 50.1
    elif mutation == "action": message["kind"] = "request_action"
    elif mutation == "duplicate": states["knowledge_0"]["sent_boundaries"].append(dict(message))
    elif mutation == "unverified": environment["scientific_complete"] = False
    elif mutation == "unfinished": states["memory_0"]["shutdown_record"] = None
    assert terminal_accounting(states, environment, rows)["all_boundaries_accounted"] is False


@pytest.mark.parametrize("reason", ["time_limit", "budget_exhausted"])
def test_terminal_accounting_preserves_scientific_limit_failures(reason: str) -> None:
    """Verified time/budget stops can cancel recent non-action work without success."""
    from mha_exp_level2_bw.exp2_7.terminal import terminal_accounting
    states, environment, rows = _terminal_fixture()
    terminal = {"reason": reason, "module_id": "llreasoner_0", "module_time": 600.0}
    for state in states.values():
        state["terminal_observed"] = terminal
        state["shutdown_record"] = {"module_time": 601.0, "terminal": terminal}
    states["llreasoner_0"].update(termination_reason=reason, scientific_complete=False,
                                 usage={"budget_exhausted": reason == "budget_exhausted"})
    environment["scientific_complete"] = False
    states["knowledge_0"]["sent_boundaries"][-1]["module_time"] = 599.99
    report = terminal_accounting(states, environment, rows)
    assert report["terminal_verified"] and report["all_boundaries_accounted"]
