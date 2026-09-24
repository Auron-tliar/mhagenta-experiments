"""Independent scientific evaluation for experiment 2-7-BW."""

from __future__ import annotations

from .terminal import terminal_accounting

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

from mha_exp_common.execution_metrics import (
    analysis_eligibility,
    eligible_analysis_rows,
    execution_envelope,
    new_report,
    nullable_certificate_status,
    read_json_object,
    write_execution_metrics,
)
from mha_exp_common.metrics import summarize_numbers
from mha_exp_common.paid_run import atomic_json

from .environment import NUM_BLOCKS, TABLE_LEN
from .llm import MODEL_POLICY, STATE_SCHEMA_VERSION, canonical_sha256

VALUE_TIE_EPSILON = 1e-12
TRACE_SCHEMA = "2-7-bw-environment-transitions-v2"
SUPPORTED_TRACE_SCHEMAS = {
    "2-7-bw-environment-transitions-v1",
    TRACE_SCHEMA,
}
MANDATORY_MODULES = (
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
MANDATORY_LLM_ROLES = (
    "llreasoner_0",
    "knowledge_0",
    "hlreasoner_0",
    "goalgraph_0",
    "memory_0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_transition_trace(path: Path) -> list[dict[str, Any]]:
    """Load and validate the single authoritative environment trace."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[0].get("row_type") != "initial":
        raise ValueError("trace must start with one initial row")
    for index, row in enumerate(rows):
        if row.get("schema_version") not in SUPPORTED_TRACE_SCHEMAS:
            raise ValueError("unexpected transition trace schema")
        if index and row.get("action_index") != index - 1:
            raise ValueError("action indexes are not contiguous")
        stacks = row.get("stacks")
        if not isinstance(stacks, list) or len(stacks) != TABLE_LEN:
            raise ValueError("trace row does not contain all table positions")
    return rows


def state_metrics(row: Mapping[str, Any]) -> dict[str, float | int]:
    """Compute the frozen objective value proxies for one trace state."""

    stacks = [list(stack) for stack in row["stacks"]]
    heights = [len(stack) for stack in stacks]
    adjacent_pairs = sum(max(len(stack) - 1, 0) for stack in stacks)
    violations = sum(
        1
        for stack in stacks
        for lower, upper in zip(stack, stack[1:], strict=False)
        if int(str(upper)[1:]) > int(str(lower)[1:])
    )
    return {
        "height_variance": pvariance(heights),
        "adjacent_pairs": adjacent_pairs,
        "ordering_violations": violations,
        "ordering_ratio": violations / adjacent_pairs if adjacent_pairs else 0.0,
        "normalized_tallest": max(heights, default=0) / NUM_BLOCKS,
    }


def automatic_value_outcome(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_condition: str,
    primary_goals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the frozen post-goal discretionary primary score."""

    final_goal_id = str(primary_goals[-1]["goal_id"])
    completion = next(
        (
            int(row["action_index"])
            for row in rows[1:]
            if final_goal_id in row.get("newly_achieved_goal_ids", [])
        ),
        None,
    )
    eligible_rows = [
        row
        for row in rows[1:]
        if completion is not None
        and int(row["action_index"]) > completion
        and row.get("action") == "Put-Down"
        and row.get("accepted") is True
        and row.get("legal") is True
    ]
    metrics = [state_metrics(row) for row in eligible_rows]
    score: float | None = None
    if metrics:
        if value_condition == "balanced":
            score = fmean(float(item["height_variance"]) for item in metrics)
        elif value_condition == "number_order":
            score = fmean(float(item["ordering_ratio"]) for item in metrics)
        elif value_condition == "tallest_stack":
            score = max(float(item["normalized_tallest"]) for item in metrics)
        else:
            raise ValueError(f"unknown value condition {value_condition!r}")
    final_metrics = state_metrics(rows[-1])
    return {
        "automatic_value_effect_implemented": True,
        "qualitative_value_effect_review": "human_required",
        "completion_action_index": completion,
        "eligible_post_goal_put_downs": len(metrics),
        "status": (
            "eligible" if metrics else "insufficient_discretionary_evidence"
        ),
        "primary_score": score,
        "final_metrics": final_metrics,
        "whole_run_max_normalized_tallest": max(
            float(state_metrics(row)["normalized_tallest"]) for row in rows
        ),
    }


def unavailable_value_outcome(reason: str) -> dict[str, Any]:
    """Describe a value outcome that cannot be derived from transition evidence."""

    return {
        "automatic_value_effect_implemented": True,
        "qualitative_value_effect_review": "human_required",
        "completion_action_index": None,
        "eligible_post_goal_put_downs": 0,
        "status": "unavailable",
        "unavailable_reason": reason,
        "primary_score": None,
        "final_metrics": None,
        "whole_run_max_normalized_tallest": None,
    }


def _all_boundaries(
    agent_states: Mapping[str, Mapping[str, Any]],
    environment_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sent = [
        record
        for state in agent_states.values()
        for record in state.get("sent_boundaries", [])
    ]
    processed = [
        record
        for state in agent_states.values()
        for record in state.get("processed_boundaries", [])
    ] + list(environment_state.get("processed_boundaries", []))
    return sent, processed


def _edge(records: Sequence[Mapping[str, Any]], source: str, recipient: str, kind: str) -> list[Mapping[str, Any]]:
    return [
        item
        for item in records
        if item.get("source") == source
        and item.get("recipient") == recipient
        and item.get("kind") == kind
    ]


def architecture_checks(
    agent_states: Mapping[str, Mapping[str, Any]],
    environment_state: Mapping[str, Any],
    transition_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Check complete reconciled and identity-connected mandatory cognition."""

    sent, processed = _all_boundaries(agent_states, environment_state)
    sent_ids = Counter(str(item.get("boundary_id")) for item in sent)
    processed_ids = Counter(str(item.get("boundary_id")) for item in processed)
    reconciled = sent_ids == processed_ids and all(
        boundary_id != "None" and count == 1 for boundary_id, count in sent_ids.items()
    )
    roles_called = all(
        bool(agent_states.get(module_id, {}).get("processed_input_ids"))
        and int(agent_states.get(module_id, {}).get("response_records", 0)) >= 1
        for module_id in MANDATORY_LLM_ROLES
    )
    no_failures = all(
        state.get("failure") is None for state in agent_states.values()
    )

    observation_witness = False
    evaluation_id: str | None = None
    for perceptor in _edge(sent, "perceptor_0", "llreasoner_0", "send_observation"):
        obs_id = perceptor.get("evidence_id")
        obs_hash = perceptor.get("observation_payload_sha256")
        if not obs_id or not obs_hash:
            continue
        ll_matches = [
            item
            for item in _edge(sent, "llreasoner_0", "knowledge_0", "send_beliefs")
            if item.get("evidence_id") == obs_id
            and item.get("observation_payload_sha256") == obs_hash
        ]
        for ll_match in ll_matches:
            memory = [
                item
                for item in sent
                if item.get("source") == "knowledge_0"
                and item.get("recipient") == "memory_0"
                and item.get("kind") in {"send_observations", "send_belief_memories"}
                and item.get("source_evidence_id") == obs_id
                and item.get("evidence_id")
            ]
            high = [
                item
                for item in _edge(sent, "knowledge_0", "hlreasoner_0", "send_beliefs")
                if item.get("source_evidence_id") == obs_id
                and item.get("evidence_id")
            ]
            shared = {
                str(item["evidence_id"]) for item in memory
            }.intersection(str(item["evidence_id"]) for item in high)
            if shared:
                observation_witness = True
                evaluation_id = sorted(shared)[0]
                break
        if observation_witness:
            break

    goal_witness = False
    goal_id: str | None = None
    if evaluation_id is not None:
        for authored in _edge(sent, "hlreasoner_0", "goalgraph_0", "send_goals"):
            candidate = authored.get("goal_id")
            goal_hash = authored.get("goal_payload_sha256")
            if (
                not candidate
                or not goal_hash
                or authored.get("input_evaluation_id") != evaluation_id
            ):
                continue
            required = (
                ("goalgraph_0", "llreasoner_0", "send_goals"),
                ("llreasoner_0", "goalgraph_0", "send_goal_update"),
                ("goalgraph_0", "hlreasoner_0", "send_goals"),
            )
            if all(
                any(
                    item.get("goal_id") == candidate
                    and item.get("goal_payload_sha256") == goal_hash
                    for item in _edge(sent, source, recipient, kind)
                )
                for source, recipient, kind in required
            ):
                goal_witness = True
                goal_id = str(candidate)
                break

    action_witness = False
    if goal_id is not None:
        for request in _edge(sent, "llreasoner_0", "actuator_0", "request_action"):
            action_id = request.get("action_id")
            cycle_id = request.get("cycle_id")
            if not action_id or not cycle_id or request.get("goal_id") != goal_id:
                continue
            env_matches = [
                item
                for item in _edge(sent, "actuator_0", str(environment_state.get("environment_id")), "environment_action")
                if item.get("action_id") == action_id
                and item.get("cycle_id") == cycle_id
                and item.get("goal_id") == goal_id
            ]
            status_matches = [
                item
                for item in _edge(sent, "actuator_0", "llreasoner_0", "send_status")
                if item.get("action_id") == action_id
                and item.get("cycle_id") == cycle_id
                and item.get("goal_id") == goal_id
            ]
            trace_matches = [
                row
                for row in transition_rows
                if row.get("row_type") == "action"
                and row.get("action_id") == action_id
                and row.get("cycle_id") == cycle_id
                and row.get("goal_id") == goal_id
            ]
            if env_matches and status_matches and trace_matches:
                action_witness = True
                break

    accounting = terminal_accounting(dict(agent_states), dict(environment_state), list(transition_rows))
    checks = {
        "all_nine_modules_present": set(agent_states) == set(MANDATORY_MODULES),
        "mandatory_llm_roles_called": roles_called,
        "no_module_failure": no_failures,
        "all_sent_boundaries_accounted": accounting["all_boundaries_accounted"],
        "connected_observation_evaluation_chain": observation_witness,
        "connected_goal_cycle": goal_witness,
        "connected_action_cycle": action_witness,
    }
    return {**checks, "all_sent_boundaries_reconciled": reconciled,
            "architecture_valid": all(checks.values())}


def _state_contract(
    agent_states: Mapping[str, Mapping[str, Any]],
    state_schemas: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for module_id in MANDATORY_MODULES:
        state = agent_states.get(module_id)
        expected = set(state_schemas.get(module_id, []))
        valid = state is not None and set(state) == expected
        if state is not None:
            try:
                json.dumps(state, allow_nan=False)
            except (TypeError, ValueError):
                valid = False
            valid = valid and state.get("state_schema_version") == STATE_SCHEMA_VERSION
        details[module_id] = valid
    return {"modules": details, "valid": all(details.values())}


def learner_outcomes(agent_states: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for learner_id, reasoner_id in (
        ("learner_0", "llreasoner_0"),
        ("learner_1", "hlreasoner_0"),
    ):
        learner = agent_states[learner_id]
        revision_id = learner.get("revision_id")
        used = bool(
            revision_id
            and revision_id in agent_states[reasoner_id].get("used_revision_ids", [])
        )
        status = learner.get("involvement_status", "not_requested")
        if used:
            status = "revision_used"
        result[learner_id] = {
            "involvement_requested_by_reasoner": bool(
                learner.get("involvement_requested_by_reasoner")
            ),
            "involvement_status": status,
            "revision_used_by_reasoner": used,
            "operational_involvement_decision": learner.get(
                "operational_involvement_decision", "not_requested"
            ),
            "counterfactual_necessity": "not_established",
        }
    return result


def value_evidence_links(
    agent_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report optional model-authored links from value evaluations to decisions."""

    sent = [
        item
        for state in agent_states.values()
        for item in state.get("sent_boundaries", [])
    ]
    evaluation_ids = sorted(
        {
            str(item["evidence_id"])
            for item in sent
            if item.get("source") == "knowledge_0"
            and item.get("evidence_id")
            and item.get("source_evidence_id")
        }
    )
    links = [
        {
            "evidence_id": evidence_id,
            "source": item.get("source"),
            "recipient": item.get("recipient"),
            "kind": item.get("kind"),
            "goal_id": item.get("goal_id"),
            "action_id": item.get("action_id"),
            "revision_id": item.get("revision_id"),
        }
        for item in sent
        for evidence_id in evaluation_ids
        if evidence_id in item.get("source_ids", [])
    ]
    return {
        "condition_evaluation_ids": evaluation_ids,
        "model_authored_link_present": bool(links),
        "linked_decisions": links,
    }


def evaluate_run(
    *,
    agent_states: Mapping[str, Mapping[str, Any]],
    environment_state: Mapping[str, Any],
    trace_path: Path,
    response_dir: Path,
    primary_goals: Sequence[Mapping[str, Any]],
    value_condition: str,
    state_schemas: Mapping[str, Sequence[str]],
    treatment_digest: str,
    logs: Sequence[str] = (),
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one run without feeding conclusions back to cognition."""

    trace_present = trace_path.is_file()
    rows = load_transition_trace(trace_path) if trace_present else []
    achieved = [
        goal_id
        for row in rows[1:]
        for goal_id in row.get("newly_achieved_goal_ids", [])
    ]
    expected = [str(goal["goal_id"]) for goal in primary_goals]
    strict_bw_success = bool(
        achieved[: len(expected)] == expected
        and all(row.get("accepted") and row.get("legal") for row in rows[1:])
    )
    goal_achieved = bool(expected) and all(goal_id in achieved for goal_id in expected)
    termination_reason = ("execution_error" if any(state.get("failure") for state in agent_states.values())
                          else "task_completed" if goal_achieved else "budget_exhausted"
                          if any(state.get("termination_reason") == "budget_exhausted"
                                 for state in agent_states.values()) else "time_limit")
    architecture = architecture_checks(agent_states, environment_state, rows)
    state_contract = _state_contract(agent_states, state_schemas)
    learners = learner_outcomes(agent_states)
    learner_complete = all(
        item["involvement_status"] != "requested_incomplete"
        for item in learners.values()
    )
    exact_models = True
    raw_files: dict[str, Any] = {}
    for module_id, state in agent_states.items():
        selected = state.get("selected_model")
        expected_profile = (
            MODEL_POLICY.deliberative
            if module_id in {"knowledge_0", "hlreasoner_0", "memory_0"}
            else MODEL_POLICY.fast
            if module_id in MANDATORY_LLM_ROLES or module_id.startswith("learner_")
            else None
        )
        if expected_profile is not None and (
            selected != expected_profile.model
            or state.get("reasoning_effort") != expected_profile.reasoning_effort
        ):
            exact_models = False
        actual = state.get("actual_models", [])
        if actual and any(model != selected for model in actual):
            exact_models = False
        records = int(state.get("response_records", 0))
        if records:
            path = response_dir / f"{module_id}.jsonl"
            if not path.is_file() or len(path.read_text(encoding="utf-8").splitlines()) != records:
                exact_models = False
            else:
                raw_rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                if expected_profile is None or any(
                    item.get("requested_model") != expected_profile.model
                    or item.get("reasoning_effort")
                    != expected_profile.reasoning_effort
                    or (
                        item.get("success") is True
                        and item.get("actual_model") != expected_profile.model
                    )
                    for item in raw_rows
                ):
                    exact_models = False
                raw_files[module_id] = {
                    "path": path.name,
                    "rows": records,
                    "sha256": sha256_file(path),
                }
    terminal = terminal_accounting(dict(agent_states), dict(environment_state), rows)
    lifecycle_complete = bool(
        all(not state.get("pending") for state in agent_states.values())
        and all(state.get("in_flight") is None for state in agent_states.values())
        and all(state.get("incomplete_reason") is None for state in agent_states.values())
        and not any("killed" in line.lower() for line in logs)
        and architecture["all_sent_boundaries_accounted"]
        and (learner_complete or terminal["terminal_verified"])
        and (not any(state.get("terminal_file") for state in agent_states.values())
             or terminal["terminal_verified"])
    )
    no_schema_repairs = all(
        int(state.get("repair_count", 0)) == 0 for state in agent_states.values()
    )
    comparability_checks = {
        "state_contract_valid": state_contract["valid"],
        "exact_models": exact_models,
        "no_schema_repairs": no_schema_repairs,
        "lifecycle_complete": lifecycle_complete,
        "treatment_digest_present": bool(treatment_digest),
        "trace_present": trace_present,
        "trace_row_count_matches": (
            trace_present and len(rows) == int(environment_state["trace_rows"])
        ),
    }
    value = (
        automatic_value_outcome(
            rows, value_condition=value_condition, primary_goals=primary_goals
        )
        if trace_present
        else unavailable_value_outcome("environment_trace_missing")
    )
    value["evidence_link"] = value_evidence_links(agent_states)
    # A scheduled stop may leave cognitive queues and optional learning unfinished.
    # Keep those architecture diagnostics separate from execution integrity.
    execution_valid = bool(
        architecture["all_nine_modules_present"] and architecture["no_module_failure"]
        and state_contract["valid"] and exact_models and trace_present
        and len(rows) == int(environment_state["trace_rows"])
        and not any(any(word in line.lower() for word in ("traceback", "critical", "fatal", "[error]")) for line in logs)
    )
    result = {
        "schema_version": "2-7-bw-result-v2",
        "execution_valid": execution_valid,
        "termination_reason": termination_reason,
        "task_success": goal_achieved,
        "value_condition": value_condition,
        "treatment_digest": treatment_digest,
        "initial_state_sha256": environment_state.get("initial_state_sha256"),
        "primary_goals_sha256": canonical_sha256(primary_goals),
        "architecture_valid": architecture["architecture_valid"],
        "strict_bw_success": strict_bw_success,
        "cohort_comparable": execution_valid,
        "matched_inclusion": execution_valid,
        "architecture_checks": architecture,
        "terminal_accounting": terminal,
        "comparability_checks": comparability_checks,
        "state_contract": state_contract,
        "scientific_outcomes": {
            "primary_goal_achieved": goal_achieved,
            "termination_reason": termination_reason,
            "achieved_goal_ids": achieved,
            "expected_goal_ids": expected,
            "attempted_actions": max(len(rows) - 1, 0),
            "illegal_or_invalid_actions": sum(
                1 for row in rows[1:] if not row.get("accepted") or not row.get("legal")
            ),
            "action_sequence": [row.get("action") for row in rows[1:]],
            "value": value,
            "learners": learners,
        },
        "evidence": {
            "environment_trace": {
                "path": trace_path.name,
                "schema_version": TRACE_SCHEMA,
                "rows": len(rows),
                "present": trace_present,
                "sha256": sha256_file(trace_path) if trace_present else None,
            },
            "raw_responses": raw_files,
        },
    }
    if result_path is not None:
        atomic_json(result_path, result)
    return result


def condition_summary(condition: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create one compact condition summary without nesting raw evidence."""

    rows = [
        {
            "run": result.get("run"),
            "result_path": result.get("result_path"),
            "architecture_valid": result["architecture_valid"],
            "strict_bw_success": result["strict_bw_success"],
            "task_success": result.get("task_success", result["strict_bw_success"]),
            "execution_valid": result.get("execution_valid", result["matched_inclusion"]),
            "termination_reason": result.get("termination_reason"),
            "cohort_comparable": result["cohort_comparable"],
            "matched_inclusion": result["matched_inclusion"],
            "primary_value_score": result["scientific_outcomes"]["value"][
                "primary_score"
            ],
            "value_status": result["scientific_outcomes"]["value"]["status"],
            "learners": result["scientific_outcomes"]["learners"],
            "action_sequence": result["scientific_outcomes"]["action_sequence"],
            "treatment_digest": result.get("treatment_digest"),
            "initial_state_sha256": result.get("initial_state_sha256"),
            "primary_goals_sha256": result.get("primary_goals_sha256"),
        }
        for result in results
    ]
    return {
        "schema_version": "2-7-bw-condition-summary-v1",
        "condition": condition,
        "runs": rows,
        "architecture_valid_runs": sum(item["architecture_valid"] for item in rows),
        "strict_successes": sum(item["strict_bw_success"] for item in rows),
        "task_successes": sum(item["task_success"] for item in rows),
        "execution_valid_runs": sum(item["execution_valid"] for item in rows),
        "matched_eligible_runs": sum(item["matched_inclusion"] for item in rows),
        "automatic_value_effect_implemented": True,
        "qualitative_value_effect_review": "human_required",
        "learner_involvement": {
            learner_id: {
                "requested_runs": sum(
                    bool(row["learners"][learner_id]["involvement_requested_by_reasoner"])
                    for row in rows
                ),
                "statuses": dict(
                    Counter(
                        row["learners"][learner_id]["involvement_status"]
                        for row in rows
                    )
                ),
                "counterfactual_necessity": "not_established",
            }
            for learner_id in ("learner_0", "learner_1")
        },
    }


def matched_summary(
    condition_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute frozen same-seed directional value comparisons."""

    by_run: dict[int, dict[str, Mapping[str, Any]]] = {}
    for condition, summary in condition_summaries.items():
        for row in summary["runs"]:
            by_run.setdefault(int(row["run"]), {})[condition] = row
    comparisons: list[dict[str, Any]] = []
    for run, conditions in sorted(by_run.items()):
        matching_fields = (
            "treatment_digest",
            "initial_state_sha256",
            "primary_goals_sha256",
        )
        expected_conditions = set(condition_summaries)
        identity_matches = bool(
            set(conditions) == expected_conditions
            and len(conditions) == 3
            and all(
                None not in {item.get(field) for item in conditions.values()}
                and len({item.get(field) for item in conditions.values()}) == 1
                for field in matching_fields
            )
        )
        action_sequences_diverged = len(
            {
                tuple(item.get("action_sequence", []))
                for item in conditions.values()
            }
        ) > 1
        for target, row in conditions.items():
            others = [item for name, item in conditions.items() if name != target]
            available = bool(
                len(others) == 2
                and identity_matches
                and row["matched_inclusion"]
                and row["primary_value_score"] is not None
                and all(
                    item["matched_inclusion"]
                    and item["primary_value_score"] is not None
                    for item in others
                )
            )
            delta: float | None = None
            classification = "unavailable"
            unavailable_reason: str | None = None
            if available:
                comparator = fmean(float(item["primary_value_score"]) for item in others)
                target_score = float(row["primary_value_score"])
                delta = (
                    target_score - comparator
                    if target == "tallest_stack"
                    else comparator - target_score
                )
                classification = (
                    "favorable"
                    if delta > VALUE_TIE_EPSILON
                    else "unfavorable"
                    if delta < -VALUE_TIE_EPSILON
                    else "tied"
                )
            elif not identity_matches:
                unavailable_reason = "matched_identity_mismatch_or_missing_condition"
            elif not row["matched_inclusion"] or any(
                not item["matched_inclusion"] for item in others
            ):
                unavailable_reason = "architecture_or_comparability_exclusion"
            else:
                unavailable_reason = "insufficient_discretionary_evidence"
            automatic_classification = (
                "insufficient_discretionary_evidence"
                if classification == "unavailable"
                else "no_difference_detected"
                if classification == "tied"
                else "difference_detected"
            )
            comparisons.append(
                {
                    "run": run,
                    "target_condition": target,
                    "available": available,
                    "matched_identity": identity_matches,
                    "oriented_delta": delta,
                    "classification": classification,
                    "automatic_evidence_classification": automatic_classification,
                    "unavailable_reason": unavailable_reason,
                    "action_sequences_diverged": action_sequences_diverged,
                }
            )
    comparison_counts = dict(Counter(item["classification"] for item in comparisons))
    return {
        "schema_version": "2-7-bw-matched-summary-v1",
        "tie_epsilon": VALUE_TIE_EPSILON,
        "automatic_value_effect_implemented": True,
        "qualitative_value_effect_review": "human_required",
        "conditions": {
            name: {
                "path": f"{name}/condition-summary.json",
                "runs": len(summary["runs"]),
            }
            for name, summary in condition_summaries.items()
        },
        "comparisons": comparisons,
        "comparison_counts": {
            key: comparison_counts.get(key, 0)
            for key in ("favorable", "tied", "unfavorable", "unavailable")
        },
        "learner_involvement": {
            condition: summary.get("learner_involvement", {})
            for condition, summary in condition_summaries.items()
        },
    }


def run_markdown(result: Mapping[str, Any]) -> str:
    """Render the compact human-review companion for one run result."""

    outcomes = result["scientific_outcomes"]
    value = outcomes["value"]
    lines = [
        f"# 2-7-BW run {result.get('run', 'unknown')}",
        "",
        f"- Architecture valid: {result['architecture_valid']}",
        f"- Strict Blocks World success: {result['strict_bw_success']}",
        f"- Matched-cohort inclusion: {result['matched_inclusion']}",
        f"- Automatic value-effect processing implemented: {value['automatic_value_effect_implemented']}",
        f"- Automatic value evidence: {value['status']}",
        f"- Primary value score: {value['primary_score']}",
        "- Qualitative causal value-effect review: human required",
        "",
        "## Learners",
        "",
    ]
    for learner_id, learner in outcomes["learners"].items():
        lines.extend(
            [
                f"- {learner_id}: requested={learner['involvement_requested_by_reasoner']}; "
                f"status={learner['involvement_status']}; "
                f"revision_used={learner['revision_used_by_reasoner']}; "
                "counterfactual necessity=not established",
            ]
        )
    lines.extend(
        [
            "",
            "Learner involvement reports the paired reasoner's observed request decision; it does not establish that learning was necessary or unnecessary.",
            "",
            "Inspect `environment-transitions.jsonl` and the referenced `llm_responses/*.jsonl` records for qualitative review.",
            "",
        ]
    )
    return "\n".join(lines)


def condition_markdown(summary: Mapping[str, Any]) -> str:
    """Render a concise condition-level human summary."""

    lines = [
        f"# 2-7-BW condition: {summary['condition']}",
        "",
        f"- Runs: {len(summary['runs'])}",
        f"- Architecture-valid runs: {summary['architecture_valid_runs']}",
        f"- Strict successes: {summary['strict_successes']}",
        f"- Matched-eligible runs: {summary['matched_eligible_runs']}",
        "- Automatic value-effect processing: implemented",
        "- Qualitative causal value-effect review: human required",
        "",
        "## Learners",
        "",
    ]
    for learner_id, learner in summary["learner_involvement"].items():
        lines.append(
            f"- {learner_id}: requested in {learner['requested_runs']} runs; "
            f"statuses={learner['statuses']}; counterfactual necessity=not established"
        )
    return "\n".join(lines) + "\n"


def matched_markdown(summary: Mapping[str, Any]) -> str:
    """Render the cohort's deliberately non-inferential matched summary."""

    counts = summary["comparison_counts"]
    return "\n".join(
        [
            "# 2-7-BW matched-condition summary",
            "",
            "- Automatic value-effect processing: implemented",
            "- Qualitative causal value-effect review: human required",
            f"- Favorable comparisons: {counts['favorable']}",
            f"- Tied comparisons: {counts['tied']}",
            f"- Unfavorable comparisons: {counts['unfavorable']}",
            f"- Unavailable comparisons: {counts['unavailable']}",
            "",
            "These descriptive comparisons do not establish statistical significance or causal value influence. Learner request/use fields report operational involvement only; counterfactual necessity remains unestablished.",
            "",
        ]
    )


def _result_index(root: Path) -> dict[tuple[str, int], Path]:
    """Index result paths by directory identity independently from parsing."""

    indexed: dict[tuple[str, int], Path] = {}
    for path in sorted(root.glob("*/*/out/result.json")):
        try:
            run = int(path.parent.parent.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        indexed[(path.parent.parent.parent.name, run)] = path
    return indexed


def _usage_from_response(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_response")
    usage = raw.get("usage", {}) if isinstance(raw, Mapping) else {}
    proxy = raw.get("proxy_accounting", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(usage, Mapping):
        usage = {}
    if not isinstance(proxy, Mapping):
        proxy = {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_cost_usd": proxy.get("last_request_cost_usd"),
    }


def _response_rows(out: Path, execution_id: str, run_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out / "llm_responses").glob("*.jsonl")):
        module_id = path.stem
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = [""]
        for line_number, line in enumerate(lines, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"success": False, "error_kind": "invalid_json"}
            if not isinstance(value, dict):
                value = {"success": False, "error_kind": "invalid_root_type"}
            rows.append({
                "execution_id": execution_id,
                "run_id": run_id,
                "module_id": module_id,
                "call_id": f"{module_id}:{line_number}",
                "source_path": path.as_posix(),
                "source_line": line_number,
                "success": value.get("success") is True,
                "latency_seconds": value.get("latency_seconds"),
                "schema_retry": value.get("schema_retry"),
                "requested_model": value.get("requested_model"),
                "actual_model": value.get("actual_model"),
                "usage": _usage_from_response(value),
                "error_kind": value.get("error_kind"),
            })
    return rows


def _module_usage(out: Path, execution_id: str, run_id: int) -> list[dict[str, Any]]:
    """Represent every expected module's retained cumulative usage evidence."""

    rows: list[dict[str, Any]] = []
    for module_id in MANDATORY_LLM_ROLES:
        matches = sorted(out.glob(f"*.{module_id}.json"))
        state: Mapping[str, Any] | None = None
        unavailable_reason: str | None = None
        if len(matches) != 1:
            unavailable_reason = (
                "required_state_missing" if not matches else "multiple_state_files"
            )
        else:
            loaded, reason = read_json_object(matches[0])
            if loaded is None:
                unavailable_reason = reason or "invalid_json"
            else:
                state = loaded
        usage = state.get("usage") if state is not None else None
        if state is not None and not isinstance(usage, Mapping):
            unavailable_reason = "usage_block_missing_or_invalid"
            usage = {}
        elif usage is None:
            usage = {}
        field_reasons: dict[str, str] = {}
        normalized: dict[str, Any] = {}
        for key in (
            "input_tokens", "output_tokens", "total_tokens",
            "estimated_cost_usd",
        ):
            value = usage.get(key)
            if key == "estimated_cost_usd":
                try:
                    valid = (
                        not isinstance(value, bool)
                        and Decimal(str(value)).is_finite()
                        and Decimal(str(value)) >= 0
                    )
                except InvalidOperation:
                    valid = False
            else:
                valid = type(value) is int and value >= 0
            normalized[key] = value if valid else None
            if not valid:
                field_reasons[key] = "missing_or_invalid"
        budget_exhausted = usage.get("budget_exhausted")
        if not isinstance(budget_exhausted, bool):
            budget_exhausted = None
            field_reasons["budget_exhausted"] = "missing_or_invalid"
        accounting_field_reasons = {
            key: reason for key, reason in field_reasons.items()
            if key != "budget_exhausted"
        }
        if unavailable_reason is None and accounting_field_reasons:
            unavailable_reason = "incomplete_or_invalid_usage"
        response_path = out / "llm_responses" / f"{module_id}.jsonl"
        rows.append({
            "execution_id": execution_id,
            "run_id": run_id,
            "module_id": module_id,
            "state_source_ref": (
                str(matches[0]) if len(matches) == 1 else None
            ),
            "usage_status": "existing" if unavailable_reason is None else "unavailable",
            "usage_unavailable_reason": unavailable_reason,
            "usage_field_unavailable_reasons": field_reasons,
            "response_stream_status": (
                "existing" if response_path.is_file() else "unavailable"
            ),
            "response_stream_unavailable_reason": (
                None if response_path.is_file() else "response_stream_missing"
            ),
            "response_records": state.get("response_records") if state else None,
            "repair_count": state.get("repair_count") if state else None,
            **normalized,
            "budget_exhausted": budget_exhausted,
        })
    return rows


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build execution metrics from retained 2-7-BW results and call records."""

    root = Path(root).resolve()
    index = _result_index(root)
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{
            "execution_id": f"run-{run}/value-{condition}",
            "run_id": run,
            "factors": {"value_condition": condition},
            "expected": False,
        } for condition, run in sorted(index)]
    report = new_report("2-7-bw", None if expected_executions is None else specs)
    calls: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for spec in specs:
        condition = str(spec.get("factors", {}).get("value_condition"))
        located = index.get((condition, int(spec["run_id"])))
        if located is None:
            report["executions"].append(execution_envelope(
                spec, present=False, readable=False, operationally_valid=False,
                readability_reasons=("run_missing",), operational_reasons=("run_missing",),
            ))
            continue
        path = located
        result, readability_reason = read_json_object(path)
        if result is None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(readability_reason or "invalid_json",),
                operational_reasons=(readability_reason or "invalid_json",),
                source_refs=(str(path.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        schema_valid = result.get("schema_version") in {
            "2-7-bw-result-v1",
            "2-7-bw-result-v2",
        }
        identity_valid = (result.get("value_condition") == condition
                          and result.get("run") == spec["run_id"])
        operational_reasons = ([] if schema_valid else ["unsupported_schema"]) + (
            [] if identity_valid else ["identity_mismatch"]
        )
        if result.get("execution_valid") is False:
            operational_reasons.append("execution_integrity_failed")
        operational = schema_valid and identity_valid and not operational_reasons
        if not operational:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=True, operationally_valid=False,
                operational_reasons=operational_reasons,
                certificate_status=(
                    "unavailable" if not schema_valid
                    else nullable_certificate_status(result.get("matched_inclusion"))
                ),
                identity={
                    "schema_version": result.get("schema_version"),
                    "value_condition": result.get("value_condition"),
                    "embedded_run_id": result.get("run"),
                },
                source_refs=(str(path.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        if not isinstance(result.get("scientific_outcomes", {}), Mapping):
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_root_type",),
                operational_reasons=("invalid_root_type",),
                source_refs=(str(path.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        out = path.parent
        execution_calls = _response_rows(out, str(spec["execution_id"]), int(spec["run_id"]))
        call_evidence_valid = not any(
            row.get("error_kind") in {"invalid_json", "invalid_root_type"}
            for row in execution_calls
        )
        if not call_evidence_valid:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_trace_row",),
                operational_reasons=("invalid_trace_row",),
                source_refs=(str(path.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        for row in execution_calls:
            row["source_path"] = str((out / "llm_responses" / f"{row['module_id']}.jsonl").relative_to(root)).replace("\\", "/")
        execution_totals = _module_usage(out, str(spec["execution_id"]), int(spec["run_id"]))
        for total in execution_totals:
            module_calls = [
                row for row in execution_calls
                if row["module_id"] == total["module_id"]
            ]
            complete_usage = [
                row["usage"] for row in module_calls
                if all(row["usage"].get(key) is not None for key in (
                    "input_tokens", "output_tokens", "total_tokens",
                    "estimated_cost_usd",
                ))
            ]
            summed_usage: dict[str, Any] | None = None
            reconciled: bool | None = None
            reconciliation_reason: str | None = None
            if total["usage_status"] != "existing":
                reconciliation_reason = "final_module_usage_unavailable"
            elif total["response_stream_status"] != "existing":
                reconciliation_reason = "response_stream_missing"
            elif len(complete_usage) != len(module_calls):
                reconciliation_reason = "incomplete_or_invalid_per_request_usage"
            else:
                try:
                    summed_usage = {
                        "input_tokens": sum(int(row["input_tokens"]) for row in complete_usage),
                        "output_tokens": sum(int(row["output_tokens"]) for row in complete_usage),
                        "total_tokens": sum(int(row["total_tokens"]) for row in complete_usage),
                        "estimated_cost_usd": str(sum(
                            (Decimal(str(row["estimated_cost_usd"])) for row in complete_usage),
                            Decimal(0),
                        )),
                    }
                    reconciled = (
                        summed_usage["input_tokens"] == total["input_tokens"]
                        and summed_usage["output_tokens"] == total["output_tokens"]
                        and summed_usage["total_tokens"] == total["total_tokens"]
                        and Decimal(summed_usage["estimated_cost_usd"])
                        == Decimal(str(total["estimated_cost_usd"]))
                    )
                except (TypeError, ValueError, InvalidOperation):
                    summed_usage = None
                    reconciled = None
                    reconciliation_reason = "incomplete_or_invalid_per_request_usage"
            total.update({
                "call_records": len(module_calls),
                "successful_calls": sum(row["success"] for row in module_calls),
                "failed_calls": sum(not row["success"] for row in module_calls),
                "schema_repairs": sum(row.get("schema_retry") is True for row in module_calls),
                "complete_usage_records": len(complete_usage),
                "per_request_usage_sum": summed_usage,
                "per_request_usage_reconciled": reconciled,
                "per_request_usage_unavailable_reason": (
                    None if reconciled is not None else reconciliation_reason
                ),
                "call_latency_seconds": summarize_numbers(
                    [row.get("latency_seconds") for row in module_calls],
                    include_iqr=True,
                    p95_min_count=40,
                ),
            })
        calls.extend(execution_calls)
        totals.extend(execution_totals)
        success = result.get("task_success", result.get("strict_bw_success")) is True
        task_rows.append({
            "execution_id": spec["execution_id"], "run_id": spec["run_id"],
            "value_condition": condition, "success": success,
            "architecture_valid": result.get("architecture_valid"),
            "matched_inclusion": result.get("matched_inclusion"),
            "attempted_actions": result.get("scientific_outcomes", {}).get("attempted_actions"),
            "illegal_or_invalid_actions": result.get("scientific_outcomes", {}).get("illegal_or_invalid_actions"),
        })
        budget_exhausted = any(
            row["budget_exhausted"] is True for row in execution_totals
        )
        calls_available = all(
            row["response_stream_status"] == "existing"
            for row in execution_totals
        )
        usage_available = all(
            row["usage_status"] == "existing" for row in execution_totals
        )
        reconciliation_available = all(
            row["per_request_usage_reconciled"] is not None
            for row in execution_totals
        )
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=nullable_certificate_status(
                result.get("matched_inclusion")
            ),
            task_status="success" if success else "failure",
            task_reason=None if success else "strict_goal_sequence_not_completed",
            termination_reason="task_completed" if success else "budget_exhausted" if budget_exhausted else "time_limit",
            identity={
                "schema_version": result.get("schema_version"),
                "treatment_digest": result.get("treatment_digest"),
                "architecture_valid": result.get("architecture_valid"),
                "matched_inclusion": result.get("matched_inclusion"),
            },
            metric_availability={
                "llm_calls": (
                    "existing" if calls_available else {
                        "status": "unavailable", "reason": "response_stream_missing",
                    }
                ),
                "usage": (
                    "existing" if usage_available else {
                        "status": "unavailable", "reason": "incomplete_final_module_usage",
                    }
                ),
                "usage_reconciliation": (
                    "derived" if reconciliation_available else {
                        "status": "unavailable", "reason": "reconciliation_basis_incomplete",
                    }
                ),
                "task_outcome": "existing",
            },
            source_refs=(str(path.relative_to(root)).replace("\\", "/"),),
        ))
    task_eligibility = analysis_eligibility(report["executions"])
    call_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("llm_calls",),
    )
    usage_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("usage",),
    )
    eligible_ids = set(task_eligibility["eligible_execution_ids"])
    eligible = [row for row in report["executions"]
                if row["execution_id"] in eligible_ids]
    eligible_calls = eligible_analysis_rows(calls, call_eligibility)
    eligible_totals = eligible_analysis_rows(totals, usage_eligibility)
    latencies = [row.get("latency_seconds") for row in eligible_calls]
    matched_path = root / "matched-condition-summary.json"
    try:
        matched = json.loads(matched_path.read_text(encoding="utf-8"))
        matched_rows = [matched] if isinstance(matched, Mapping) else []
    except (OSError, json.JSONDecodeError):
        matched_rows = []
    matched_supported = bool(
        matched_rows
        and matched_rows[0].get("schema_version")
        == "2-7-bw-matched-summary-v1"
    )
    matched_eligibility = {
        "eligible_cohort_ids": (
            ["matched-condition-summary"] if matched_supported else []
        ),
        "excluded": ([] if matched_supported else [{
            "cohort_id": "matched-condition-summary",
            "reasons": [
                "unsupported_schema" if matched_rows
                else "matched_summary_missing"
            ],
        }]),
    }
    report["analyses"] = {
        "llm_calls": {
            "unit": "llm_call", "nesting": "calls_within_module_within_execution",
            "eligibility": call_eligibility,
            "rows": calls,
            "summary": {"count": len(eligible_calls),
                        "raw_descriptive_count": len(calls),
                        "latency_seconds": summarize_numbers(latencies, include_iqr=True)},
        },
        "module_usage": {
            "unit": "module_within_execution", "nesting": "modules_within_execution",
            "eligibility": usage_eligibility, "rows": totals,
            "summary": {"count": len(eligible_totals),
                        "raw_descriptive_count": len(totals),
                        "usage_available": sum(row["usage_status"] == "existing" for row in eligible_totals),
                        "reconciliation_available": sum(row["per_request_usage_reconciled"] is not None for row in eligible_totals)},
        },
        "matched_comparison": {
            "unit": "matched_cohort",
            "nesting": "one_cohort_summary",
            "eligibility": matched_eligibility,
            "rows": matched_rows,
            "summary": {
                "available": bool(matched_rows),
                "source_ref": "matched-condition-summary.json" if matched_rows else None,
            },
        },
        "task_outcomes": {
            "unit": "execution", "nesting": "design_cells_within_seed",
            "eligibility": task_eligibility,
            "rows": [row for row in task_rows if row["execution_id"] in {item["execution_id"] for item in eligible}],
            "summary": {"eligible_executions": len(eligible), "successes": sum(row["task_outcome"]["status"] == "success" for row in eligible)},
        },
    }
    write_execution_metrics(root, report)
    return report


__all__ = [
    "TRACE_SCHEMA",
    "VALUE_TIE_EPSILON",
    "architecture_checks",
    "automatic_value_outcome",
    "condition_summary",
    "evaluate_run",
    "learner_outcomes",
    "load_transition_trace",
    "matched_summary",
    "process_execution_metrics",
    "sha256_file",
    "state_metrics",
]
