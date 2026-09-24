"""Compact architecture and scientific evaluation for 2-7-CR."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from mha_exp_common.execution_metrics import (
    analysis_eligibility,
    eligible_analysis_rows,
    execution_envelope,
    new_report,
    read_json_object,
    write_execution_metrics,
)
from mha_exp_common.metrics import summarize_numbers

from .llm import PRIMARY_COMPLETION_ACHIEVEMENT

MODULE_IDS = {
    "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
    "hlreasoner_0", "goalgraph_0", "memory_0", "learner_0", "learner_1",
}
LLM_MODULE_IDS = MODULE_IDS - {"perceptor_0", "actuator_0"}
LOG_CHECK_VERSION = "2-7-cr-runtime-log-check-v1"
_RUNTIME_ERROR = re.compile(
    r"^\s*(?:\[[^\]\r\n]*\])*\[(?:ERROR|CRITICAL|FATAL)\]::"
    r"|^\s*Traceback \(most recent call last\):\s*$"
    r"|^\s*\+?\s*Exception Group Traceback \(most recent call last\):\s*$"
)


def runtime_error_lines(logs: Sequence[str]) -> list[str]:
    """Recognize framework error records and Python traceback headers, not quoted prose."""
    return [line.strip() for line in logs if _RUNTIME_ERROR.match(line)]


def read_events(root: Path) -> list[dict[str, Any]]:
    """Read compact event shards and retain their stable source positions."""
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("event row is not an object")
                    value["module_id"] = path.stem
                    value["source_path"] = path.name
                    value["source_line"] = line_number
                    records.append(value)
    return records


def evaluate_run(agent_states: Mapping[str, Mapping[str, Any]],
                 environment_state: Mapping[str, Any], event_root: Path,
                 logs: Sequence[str] = ()) -> dict[str, Any]:
    """Separate architecture validity from environment-owned outcomes."""
    missing = sorted(MODULE_IDS - set(agent_states))
    events = read_events(event_root) if event_root.is_dir() else []
    called = {event["module_id"] for event in events if event.get("kind") == "llm_call"
              and event.get("outcome") == "success"}
    sends = {(event["module_id"], event.get("recipient"), event.get("type"))
             for event in events if event.get("kind") == "send"}
    required_edges = {
        ("perceptor_0", "llreasoner_0", "send_observation"),
        ("llreasoner_0", "actuator_0", "request_action"),
        ("actuator_0", "llreasoner_0", "send_status"),
        ("llreasoner_0", "learner_0", "send_learner_task"),
        ("llreasoner_0", "learner_0", "request_model"),
        ("hlreasoner_0", "knowledge_0", "request_beliefs"),
        ("hlreasoner_0", "learner_1", "send_learner_task"),
        ("hlreasoner_0", "learner_1", "request_model"),
        ("learner_0", "memory_0", "request_memories"),
        ("learner_1", "memory_0", "request_memories"),
    }
    runtime_errors = runtime_error_lines(logs)
    architecture_reasons: list[str] = []
    if missing:
        architecture_reasons.append("missing_modules")
    if LLM_MODULE_IDS - called:
        architecture_reasons.append("uncalled_llm_roles")
    if required_edges - sends:
        architecture_reasons.append("incomplete_reactive_loop")
    if runtime_errors:
        architecture_reasons.append("runtime_errors")
    achievements = dict(environment_state.get("final_achievements", {}))
    scientific = {
        "alive": not bool(environment_state.get("dead", False)),
        "primary_goal_achieved": int(achievements.get(PRIMARY_COMPLETION_ACHIEVEMENT, 0)) > 0,
        "highest_primary_achievement": environment_state.get("highest_primary_achievement"),
        "achievements": achievements,
        "native_return": float(environment_state.get("native_return", 0.0)),
        "steps": int(environment_state.get("step_count", 0)),
        "illegal_actions": int(environment_state.get("illegal_actions", 0)),
        "final_inventory": dict(environment_state.get("final_inventory", {})),
    }
    module_usage = {
        module_id: {"calls": int(state.get("calls", 0)),
                    "estimated_cost_usd": str(state.get("estimated_cost_usd", "0")),
                    "budget_exhausted": bool(state.get("budget_exhausted", False))}
        for module_id, state in agent_states.items() if module_id in LLM_MODULE_IDS
    }
    termination_reason = ("task_completed" if scientific["primary_goal_achieved"] else
                          "death" if not scientific["alive"] else
                          "budget_exhausted" if any(row["budget_exhausted"] for row in module_usage.values()) else
                          "environment_limit" if environment_state.get("terminal") else "time_limit")
    scientific["termination_reason"] = termination_reason
    execution_valid = bool(not missing and environment_state and not runtime_errors
                           and not any(state.get("halted") for state in agent_states.values()))
    architecture_valid = not architecture_reasons
    return {
        "schema_version": "2-7-cr-evaluation-v10",
        "log_check_version": LOG_CHECK_VERSION,
        "execution_valid": execution_valid,
        "architecture_valid": architecture_valid,
        "architecture_reasons": architecture_reasons,
        "cohort_comparable": execution_valid,
        "scientific_outcomes": scientific,
        "module_usage": module_usage,
        "event_count": len(events),
    }


def _module_states(run_root: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for path in run_root.glob("*/out/*.json"):
        name = path.stem.rsplit(".", 1)[-1]
        if name not in LLM_MODULE_IDS:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            states[name] = value
    return states


def _usage_rows(
    events: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    execution_id: str,
    run_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    for module_id in sorted(LLM_MODULE_IDS):
        module_events = [row for row in events
                         if row.get("module_id") == module_id and row.get("kind") == "llm_call"]
        previous = {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": Decimal(0),
        }
        ambiguous = False
        last_snapshot: dict[str, Any] | None = None
        for call_index, row in enumerate(module_events):
            success = row.get("outcome") == "success"
            snapshot: dict[str, Any] | None = None
            delta: dict[str, Any] | None = None
            if success:
                try:
                    snapshot = {
                        "input_tokens": int(row["input_tokens"]),
                        "output_tokens": int(row["output_tokens"]),
                        "estimated_cost_usd": Decimal(str(row["estimated_cost_usd"])),
                    }
                    monotone = (
                        snapshot["input_tokens"] >= previous["input_tokens"]
                        and snapshot["output_tokens"] >= previous["output_tokens"]
                        and snapshot["estimated_cost_usd"] >= previous["estimated_cost_usd"]
                    )
                except (KeyError, TypeError, ValueError, InvalidOperation):
                    snapshot, monotone = None, False
                if snapshot is not None and monotone and not ambiguous:
                    delta = {
                        "input_tokens": snapshot["input_tokens"] - previous["input_tokens"],
                        "output_tokens": snapshot["output_tokens"] - previous["output_tokens"],
                        "estimated_cost_usd": str(
                            snapshot["estimated_cost_usd"] - previous["estimated_cost_usd"]
                        ),
                    }
                if snapshot is not None and monotone:
                    previous = snapshot
                    last_snapshot = snapshot
                else:
                    ambiguous = True
                ambiguous = False if snapshot is not None and monotone else ambiguous
            else:
                ambiguous = True
            calls.append({
                "execution_id": execution_id, "run_id": run_id,
                "module_id": module_id, "call_id": f"{module_id}-{call_index}",
                "source_path": row.get("source_path"), "source_line": row.get("source_line"),
                "outcome": row.get("outcome"), "latency_seconds": row.get("latency_seconds"),
                "cumulative_input_tokens": None if snapshot is None else snapshot["input_tokens"],
                "cumulative_output_tokens": None if snapshot is None else snapshot["output_tokens"],
                "cumulative_estimated_cost_usd": None if snapshot is None else str(snapshot["estimated_cost_usd"]),
                "per_call_usage": delta,
                "per_call_usage_unavailable_reason": None if delta is not None else "ambiguous_or_absent_snapshot",
            })
        state = states.get(module_id)
        state_available = isinstance(state, Mapping)
        state_calls = state.get("calls") if state_available else None
        state_failures = state.get("failures") if state_available else None
        state_budget = state.get("budget_exhausted") if state_available else None
        raw_state_cost = state.get("estimated_cost_usd") if state_available else None
        calls_valid = type(state_calls) is int and state_calls >= 0
        failures_valid = type(state_failures) is int and state_failures >= 0
        budget_valid = type(state_budget) is bool
        try:
            parsed_state_cost = Decimal(str(raw_state_cost))
            cost_valid = parsed_state_cost.is_finite() and parsed_state_cost >= 0
        except (InvalidOperation, TypeError, ValueError):
            parsed_state_cost, cost_valid = None, False
        state_cost = str(raw_state_cost) if cost_valid else None
        event_cost = None if last_snapshot is None else str(last_snapshot["estimated_cost_usd"])
        cost_reconciled = (
            parsed_state_cost == last_snapshot["estimated_cost_usd"]
            if cost_valid and last_snapshot is not None
            else None
        )
        unavailable_reason = None
        if not state_available:
            unavailable_reason = "final_module_state_missing"
        elif not all((calls_valid, failures_valid, budget_valid, cost_valid)):
            unavailable_reason = "invalid_final_module_state"
        elif last_snapshot is None:
            unavailable_reason = "successful_event_snapshot_missing"
        totals.append({
            "execution_id": execution_id, "run_id": run_id, "module_id": module_id,
            "final_state_available": state_available,
            "final_state_valid": (
                state_available
                and all((calls_valid, failures_valid, budget_valid, cost_valid))
            ),
            "event_records": len(module_events),
            "successful_event_snapshot_available": last_snapshot is not None,
            "calls": state_calls if calls_valid else None,
            "failures": state_failures if failures_valid else None,
            "budget_exhausted": state_budget if budget_valid else None,
            "estimated_cost_usd": state_cost,
            "cumulative_input_tokens": None if last_snapshot is None else last_snapshot["input_tokens"],
            "cumulative_output_tokens": None if last_snapshot is None else last_snapshot["output_tokens"],
            "final_snapshot_cost_usd": event_cost,
            "cost_reconciled": cost_reconciled,
            "reconciliation_unavailable_reason": (
                unavailable_reason if cost_reconciled is None else None
            ),
        })
    return calls, totals


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build 2-7-CR execution-cell metrics without summing usage snapshots."""

    root = Path(root).resolve()
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        for path in sorted(root.glob("*/run-*/evaluation.json")):
            try:
                run_id = int(path.parent.name.removeprefix("run-"))
                profile_name, observation, condition = path.parent.parent.name.rsplit("-", 2)
            except ValueError:
                continue
            specs.append({
                "execution_id": (
                    f"profile-{profile_name}/run-{run_id}/"
                    f"observation-{observation}/value-{condition}"
                ),
                "run_id": run_id,
                "factors": {"profile": profile_name, "observation_format": observation,
                            "value_condition": condition},
                "expected": False,
            })
    report = new_report("2-7-cr", None if expected_executions is None else specs)
    call_rows: list[dict[str, Any]] = []
    usage_totals: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    matched_context_rows: list[dict[str, Any]] = []
    matched_contexts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("matched-context-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        profile = value.get("profile")
        profile_name = profile.get("name") if isinstance(profile, Mapping) else None
        if not isinstance(profile_name, str):
            continue
        row = {
            "profile": profile_name,
            "schema_version": value.get("schema_version"),
            "supported": value.get("schema_version") in {
                "2-7-cr-matched-context-v9",
                "2-7-cr-matched-context-v10",
            },
            "protocol_version": value.get("protocol_version"),
            "model_policy_version": value.get("model_policy_version"),
            "goal_chain_policy_version": value.get("goal_chain_policy_version"),
            "knowledge_admission_policy_version": value.get(
                "knowledge_admission_policy_version"
            ),
            "control_treatment": value.get("control_treatment"),
            "designs": value.get("designs"),
            "run_context_count": (
                len(value["runs"]) if isinstance(value.get("runs"), list) else None
            ),
            "source_ref": str(path.relative_to(root)).replace("\\", "/"),
        }
        matched_context_rows.append(row)
        matched_contexts[profile_name] = row
    for spec in specs:
        factors = spec.get("factors", {})
        subset = root / f"{factors['profile']}-{factors['observation_format']}-{factors['value_condition']}"
        run_root = subset / f"run-{int(spec['run_id']):03d}"
        result_path = run_root / "evaluation.json"
        if not result_path.is_file():
            report["executions"].append(execution_envelope(
                spec, present=False, readable=False, operationally_valid=False,
                readability_reasons=("run_missing",), operational_reasons=("run_missing",),
            ))
            continue
        result, readability_reason = read_json_object(result_path)
        if result is None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(readability_reason or "invalid_json",),
                operational_reasons=(readability_reason or "invalid_json",),
                source_refs=(str(result_path.relative_to(root)),),
            ))
            continue
        try:
            event_dirs = list(run_root.glob("*/out/events"))
            call_evidence_available = len(event_dirs) == 1
            events = read_events(event_dirs[0]) if call_evidence_available else []
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_trace_row",),
                operational_reasons=("invalid_trace_row",),
                source_refs=(str(result_path.relative_to(root)),),
            ))
            continue
        schema_valid = result.get("schema_version") == "2-7-cr-evaluation-v10"
        result_profile = result.get("profile")
        identity_valid = (
            result.get("run") == spec["run_id"]
            and result.get("observation_format") == factors["observation_format"]
            and result.get("value_condition") == factors["value_condition"]
            and isinstance(result_profile, Mapping)
            and result_profile.get("name") == factors["profile"]
            and (
                factors.get("protocol_version") is None
                or result.get("protocol_version") == factors.get("protocol_version")
            )
        )
        shape_valid = (
            isinstance(result.get("scientific_outcomes", {}), Mapping)
            and isinstance(result.get("architecture_reasons", []), list)
        )
        if not shape_valid:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_root_type",),
                operational_reasons=("invalid_root_type",),
                source_refs=(str(result_path.relative_to(root)),),
            ))
            continue
        if not schema_valid or not identity_valid:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=True, operationally_valid=False,
                operational_reasons=([] if schema_valid else ["unsupported_schema"])
                + ([] if identity_valid else ["identity_mismatch"]),
                certificate_status="not_evaluated",
                identity={"schema_version": result.get("schema_version"), **dict(factors)},
                source_refs=(str(result_path.relative_to(root)),),
            ))
            continue
        operational_reasons = [reason for reason in result.get("architecture_reasons", [])
                               if reason in {"missing_modules", "runtime_errors"}]
        if result.get("execution_valid") is False:
            operational_reasons.append("execution_integrity_failed")
        operational = schema_valid and identity_valid and not operational_reasons
        scientific = result.get("scientific_outcomes", {})
        primary = scientific.get("primary_goal_achieved") is True
        module_calls, module_totals = _usage_rows(
            events, _module_states(run_root), str(spec["execution_id"]), int(spec["run_id"]),
        )
        call_rows.extend(module_calls)
        for total in module_totals:
            total_calls = [
                row for row in module_calls
                if row["module_id"] == total["module_id"]
            ]
            total["call_latency_seconds"] = summarize_numbers(
                [row.get("latency_seconds") for row in total_calls],
                include_iqr=True,
                p95_min_count=40,
            )
        usage_totals.extend(module_totals)
        usage_available = all(
            row["final_state_valid"] is True for row in module_totals
        )
        reconciliation_available = all(
            row["cost_reconciled"] is not None for row in module_totals
        )
        transitions.extend({
            "execution_id": spec["execution_id"], "run_id": spec["run_id"],
            "transition_id": f"{row.get('module_id')}:{row.get('source_line')}",
            "module_id": row.get("module_id"), "source_path": row.get("source_path"),
            "source_line": row.get("source_line"), "type": row.get("type"),
            "recipient": row.get("recipient"),
        } for row in events if row.get("kind") == "send" and row.get("type") in {
            "send_goals", "send_goal_update", "request_action",
        })
        transitions.append({
            "execution_id": spec["execution_id"],
            "run_id": spec["run_id"],
            "transition_id": "environment-outcome",
            "module_id": "environment",
            "source_path": str(result_path.relative_to(root)).replace("\\", "/"),
            "source_line": None,
            "type": "environment_outcome",
            "recipient": None,
            "primary_goal_achieved": primary,
            "alive": scientific.get("alive"),
            "highest_achievement": scientific.get("highest_achievement"),
        })
        budget_exhausted = any(
            row["budget_exhausted"] is True for row in module_totals
        )
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=([] if schema_valid else ["unsupported_schema"])
            + ([] if identity_valid else ["identity_mismatch"])
            + operational_reasons,
            certificate_status="not_evaluated",
            task_status="success" if primary else "failure",
            task_reason=None if primary else "primary_goal_not_achieved",
            termination_reason=scientific.get("termination_reason") or ("task_completed" if primary else "budget_exhausted"
                                if budget_exhausted else "environment_terminal"
                                if scientific.get("alive") is False else "time_limit"),
            identity={"schema_version": result.get("schema_version"), **dict(factors)},
            metric_availability={
                "llm_calls": (
                    "existing" if call_evidence_available else {
                        "status": "unavailable",
                        "reason": "event_stream_missing",
                    }
                ),
                "usage": (
                    "existing" if usage_available else {
                        "status": "unavailable",
                        "reason": "incomplete_final_module_usage",
                    }
                ),
                "usage_reconciliation": (
                    "derived" if reconciliation_available else {
                        "status": "unavailable",
                        "reason": "successful_event_snapshot_missing_or_invalid",
                    }
                ),
                "goal_transitions": "derived",
                "matched_context": (
                    "existing"
                    if matched_contexts.get(str(factors["profile"]), {}).get(
                        "supported"
                    )
                    and (
                        factors.get("protocol_version") is None
                        or matched_contexts[str(factors["profile"])].get(
                            "protocol_version"
                        )
                        == factors.get("protocol_version")
                    )
                    else "unavailable"
                ),
            },
            source_refs=(str(result_path.relative_to(root)),),
        ))
    task_eligibility = analysis_eligibility(report["executions"])
    call_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("llm_calls",),
    )
    usage_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("usage",),
    )
    transition_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("goal_transitions",),
    )
    eligible_ids = set(task_eligibility["eligible_execution_ids"])
    eligible = [row for row in report["executions"]
                if row["execution_id"] in eligible_ids]
    eligible_calls = eligible_analysis_rows(call_rows, call_eligibility)
    eligible_totals = eligible_analysis_rows(usage_totals, usage_eligibility)
    eligible_transitions = eligible_analysis_rows(
        transitions, transition_eligibility
    )
    expected_profiles = (
        sorted(matched_contexts)
        if expected_executions is None
        else sorted({
            profile
            for spec in specs
            if isinstance(
                profile := spec.get("factors", {}).get("profile"), str
            )
        })
    )
    matched_context_eligibility = {
        "eligible_profile_ids": [
            profile for profile in expected_profiles
            if matched_contexts.get(profile, {}).get("supported") is True
        ],
        "excluded": [
            {
                "profile_id": profile,
                "reasons": [
                    "unsupported_schema" if profile in matched_contexts
                    else "matched_context_missing"
                ],
            }
            for profile in expected_profiles
            if matched_contexts.get(profile, {}).get("supported") is not True
        ],
    }
    report["analyses"] = {
        "module_calls": {
            "unit": "llm_call", "nesting": "calls_within_module_within_execution",
            "eligibility": call_eligibility,
            "rows": call_rows,
            "summary": {"count": len(eligible_calls),
                        "raw_descriptive_count": len(call_rows),
                        "latency_seconds": summarize_numbers(
                [row.get("latency_seconds") for row in eligible_calls], include_iqr=True,
            )},
        },
        "module_usage": {"unit": "module_within_execution",
                         "nesting": "modules_within_execution",
                         "eligibility": usage_eligibility, "rows": usage_totals,
                         "summary": {"count": len(eligible_totals),
                                     "raw_descriptive_count": len(usage_totals)}},
        "goal_behavior_transitions": {"unit": "send_event", "nesting": "events_within_execution",
                                      "eligibility": transition_eligibility, "rows": transitions,
                                      "summary": {"count": len(eligible_transitions),
                                                  "raw_descriptive_count": len(transitions)}},
        "matched_contexts": {
            "unit": "profile_context",
            "nesting": "one_context_per_profile",
            "eligibility": matched_context_eligibility,
            "rows": matched_context_rows,
            "summary": {
                "raw_descriptive_count": len(matched_context_rows),
                "supported_raw_descriptive_count": sum(
                    row["supported"] for row in matched_context_rows
                ),
            },
        },
        "task_outcomes": {
            "unit": "execution", "nesting": "design_cells_within_seed",
            "eligibility": task_eligibility,
            "rows": [{"execution_id": row["execution_id"], "run_id": row["run_id"],
                      **row["task_outcome"]} for row in eligible],
            "summary": {"eligible_executions": len(eligible),
                        "successes": sum(row["task_outcome"]["status"] == "success" for row in eligible)},
        },
    }
    write_execution_metrics(root, report)
    return report


__all__ = [
    "LLM_MODULE_IDS", "MODULE_IDS", "evaluate_run", "process_execution_metrics",
    "read_events",
]
