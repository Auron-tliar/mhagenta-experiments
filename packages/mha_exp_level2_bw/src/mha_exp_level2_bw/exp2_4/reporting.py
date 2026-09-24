"""Execution reporting for hierarchical 2-4-BW runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mha_exp_common.execution_metrics import (
    analysis_eligibility,
    eligible_analysis_rows,
    execution_envelope,
    new_report,
    treatment_identity_reasons,
    write_execution_metrics,
)
from mha_exp_common.metrics import summarize_numbers


def _states(out: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in out.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values[path.stem.rsplit(".", 1)[-1]] = value
    return values


def _hierarchy_shape_error(value: Any) -> str | None:
    """Validate containers required to normalize retained hierarchy rows."""

    if not isinstance(value, Mapping):
        return "invalid_root_type"
    plan = value.get("plan")
    execution = value.get("execution")
    if plan is None and value.get("status") == "unsolved" and execution == []:
        return None
    if not isinstance(plan, Mapping) or not isinstance(execution, list):
        return "invalid_root_type"
    transfers = plan.get("transfers")
    if not isinstance(transfers, list) or not all(
        isinstance(row, Mapping) for row in transfers
    ):
        return "invalid_trace_row"
    if not all(isinstance(row, Mapping) for row in execution):
        return "invalid_trace_row"
    if any(
        not isinstance(row.get("atomic_rows"), list)
        or not all(isinstance(item, Mapping) for item in row["atomic_rows"])
        for row in execution
    ):
        return "invalid_trace_row"
    return None


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build retained hierarchy and transfer-efficiency metrics."""

    root = Path(root).resolve()
    discovered = sorted(int(path.name.rsplit("_", 1)[-1]) for path in root.glob("exp_agent2_4_*")
                        if path.name.rsplit("_", 1)[-1].isdigit())
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in discovered]
    report = new_report("2-4-bw", None if expected_executions is None else specs)
    transfer_rows: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_id = int(spec["run_id"])
        out = root / f"exp_agent2_4_{run_id}" / "out"
        states = _states(out) if out.is_dir() else {}
        high = states.get("hlreasoner_0")
        if not isinstance(high, Mapping):
            reason = "required_state_missing" if out.is_dir() else "run_missing"
            report["executions"].append(execution_envelope(
                spec, present=out.is_dir(), readable=False, operationally_valid=False,
                readability_reasons=(reason,), operational_reasons=(reason,),
            ))
            continue
        hierarchy = high.get("retained_hierarchy") or high.get("current_hierarchy")
        shape_error = _hierarchy_shape_error(hierarchy)
        if shape_error is not None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(shape_error,), operational_reasons=(shape_error,),
            ))
            continue
        retained = hierarchy
        plan = retained["plan"] or {"transfers": []}
        planned_transfers = plan["transfers"]
        engine = plan.get("engine")
        current_rows: list[dict[str, Any]] = []
        for index, transfer in enumerate(retained["execution"]):
            atomic = transfer["atomic_rows"]
            status = str(transfer.get("status", "open"))
            step_index = transfer.get("step_index")
            planned = (
                planned_transfers[step_index]
                if type(step_index) is int
                and 0 <= step_index < len(planned_transfers)
                and isinstance(planned_transfers[step_index], Mapping)
                else {}
            )
            row = {
                "execution_id": spec["execution_id"], "run_id": run_id,
                "hierarchy_id": retained.get("hierarchy_id"),
                "transfer_id": transfer.get("goal_id", f"transfer-{index}"),
                "step_index": step_index, "status": status,
                "transfer_specification": dict(planned),
                "transfer_direction": {
                    "source": planned.get("source"),
                    "destination": planned.get("destination"),
                    "source_support": planned.get("source_support"),
                    "destination_support": planned.get("destination_support"),
                },
                "planner_engine": engine,
                "plan_source": (
                    "planner" if engine == "lpg" else "fallback" if engine else None
                ),
                "atomic_action_count": len(atomic),
                "legal_atomic_actions": sum(item.get("legal") is True for item in atomic),
                "source_observation_start": transfer.get("dispatch_observation_seq"),
                "source_observation_end": transfer.get("completion_observation_seq"),
                "source_execution_row_range": {
                    "start": 0 if atomic else None,
                    "end": len(atomic) - 1 if atomic else None,
                },
                "reconciled": transfer.get("completion_belief_observation_seq") is not None,
            }
            current_rows.append(row)
            transfer_rows.append(row)
            atomic_rows.extend({
                "execution_id": spec["execution_id"], "run_id": run_id,
                "hierarchy_id": retained.get("hierarchy_id"),
                "transfer_id": row["transfer_id"],
                "atomic_action_id": f"{row['transfer_id']}:atomic-{atomic_index}",
                "source_index": atomic_index,
                **dict(atomic_row),
            } for atomic_index, atomic_row in enumerate(atomic))
        completed = [row for row in current_rows if row["status"] == "completed"]
        failed = [row for row in current_rows if row["status"] == "failed"]
        open_rows = [row for row in current_rows if row["status"] not in {"completed", "failed"}]
        numerator = sum(row["atomic_action_count"] for row in completed)
        run_rows.append({
            "execution_id": spec["execution_id"], "run_id": run_id,
            "completed_goals": high.get("goal_completions"),
            "completed_transfers": high.get("completed_transfers"),
            "failed_transfers": high.get("compound_failures"),
            "retained_open_transfers": len(open_rows),
            "retained_atomic_actions_completed": numerator,
            "retained_atomic_actions_failed_or_open": sum(row["atomic_action_count"] for row in failed + open_rows),
            "atomic_actions_per_completed_transfer": None if not completed else {
                "numerator": numerator, "denominator": len(completed), "value": numerator / len(completed)},
            "elapsed_transfer_time": {"status": "unavailable", "reason": "no_retained_time_boundaries"},
            "hierarchy_status": retained.get("status"),
            "intention": retained.get("intention"),
            "planner_engine": engine,
            "plan_source": (
                "planner" if engine == "lpg" else "fallback" if engine else None
            ),
            "task_id": high.get("treatment", {}).get("task_id"),
            "paired_plan_length": high.get("treatment", {}).get(
                "atomic_plan_length"
            ),
            "paired_transfer_count": high.get("treatment", {}).get(
                "transfer_count"
            ),
            "completion_outcome": retained.get("status"),
        })
        log_path = root / f"exp_agent2_4_{run_id}.log"
        logs = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []
        from .checking import check_results
        required_checker_states = {
            "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
            "goalgraph_0", "hlreasoner_0",
        }
        checker_available = (
            required_checker_states <= states.keys()
            and log_path.is_file()
            and bool(logs)
        )
        environment_states = _states(root / f"exp_env2_4_{run_id}" / "out")
        environment = environment_states.get(f"exp_env2_4_{run_id}")
        environment_log = root / f"exp_env2_4_{run_id}.log"
        environment_logs = environment_log.read_text(encoding="utf-8").splitlines() if environment_log.is_file() else []
        certificate = (
            check_results(dict(states), logs, verbose=False,
                          environment=environment, environment_logs=environment_logs,
                          expected_treatment=dict(spec["factors"]) if spec.get("factors") else None)
            if checker_available else None
        )
        task_success = int(high.get("goal_completions", 0)) > 0
        operational_reasons = []
        operational_reasons.extend(treatment_identity_reasons(
            spec,
            high.get("treatment"),
            keys=("protocol_version", "treatment_id", "task_id"),
        ))
        if not required_checker_states <= states.keys():
            operational_reasons.append("required_state_missing")
        if not log_path.is_file() or not logs:
            operational_reasons.append("required_log_missing")
        if high.get("failure") is not None or any(
            "traceback" in line.lower() for line in logs
        ):
            operational_reasons.append("fatal_runtime_error")
        if environment is None or not environment_logs:
            operational_reasons.append("required_environment_evidence_missing")
        if certificate is False:
            operational_reasons.append("operational_certificate_failed")
        operational = not operational_reasons
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=(
                "unavailable" if certificate is None
                else "passed" if certificate else "failed"
            ),
            task_status="success" if task_success else "failure",
            task_reason=None if task_success else "no_completed_hierarchy_goal",
            termination_reason="task_completed" if task_success else high.get("terminal_reason") or ("operational_failure" if high.get("failure") else "unknown"),
            identity={"treatment": high.get("treatment")},
            metric_availability={"transfer_rows": "existing", "atomic_action_ratio": "derived", "timing": "unavailable"},
            source_refs=(str(next(out.glob("*.hlreasoner_0.json")).relative_to(root)).replace("\\", "/"),),
        ))
    eligibility = analysis_eligibility(report["executions"])
    eligible_transfers = eligible_analysis_rows(transfer_rows, eligibility)
    eligible_atomic = eligible_analysis_rows(atomic_rows, eligibility)
    eligible_runs = eligible_analysis_rows(run_rows, eligibility)
    report["analyses"] = {
        "transfers": {"unit": "transfer", "nesting": "transfers_within_hierarchy_within_execution",
                      "eligibility": eligibility, "rows": transfer_rows,
                      "summary": {"count": len(eligible_transfers),
                                  "raw_descriptive_count": len(transfer_rows)}},
        "atomic_actions": {
            "unit": "atomic_action",
            "nesting": "atomic_actions_within_transfer_within_hierarchy_within_execution",
            "eligibility": eligibility,
            "rows": atomic_rows,
            "summary": {"count": len(eligible_atomic),
                        "raw_descriptive_count": len(atomic_rows)},
        },
        "hierarchy_runs": {"unit": "execution", "nesting": "none",
                           "eligibility": eligibility, "rows": run_rows,
                           "summary": {"completed_transfers": summarize_numbers(
                               [row["completed_transfers"] for row in eligible_runs], include_iqr=True)}},
    }
    write_execution_metrics(root, report)
    return report
