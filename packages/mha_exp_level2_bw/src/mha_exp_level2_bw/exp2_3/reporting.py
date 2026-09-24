"""Execution reporting for finite 2-3-BW planning runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mha_exp_common.execution_metrics import (
    analysis_eligibility,
    execution_envelope,
    new_report,
    treatment_identity_reasons,
    read_json_object,
    write_execution_metrics,
)
from mha_exp_common.metrics import summarize_numbers


def _load_run(
    root: Path,
    run_id: int,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any], Path] | str:
    """Load one retained high-level run or return its readability reason."""

    out = root / f"exp_agent2_3_{run_id}" / "out"
    if not out.is_dir():
        return "run_missing"
    states: dict[str, dict[str, Any]] = {}
    for path in out.glob("*.json"):
        value, _ = read_json_object(path)
        if value is not None:
            states[path.stem.rsplit(".", 1)[-1]] = value
    high_paths = list(out.glob("*.hlreasoner_0.json"))
    if len(high_paths) != 1:
        return "required_state_missing"
    high, reason = read_json_object(high_paths[0])
    if high is None:
        return reason or "invalid_json"
    value = high.get("run") if isinstance(high, dict) else None
    if not isinstance(value, dict):
        return "invalid_root_type"
    return out, states, value, high_paths[0]


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build task and planning rows from retained high-level run state."""

    root = Path(root).resolve()
    discovered = sorted(int(path.name.rsplit("_", 1)[-1]) for path in root.glob("exp_agent2_3_*")
                        if path.name.rsplit("_", 1)[-1].isdigit())
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in discovered]
    report = new_report("2-3-bw", None if expected_executions is None else specs)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        loaded = _load_run(root, int(spec["run_id"]))
        if isinstance(loaded, str):
            present = loaded != "run_missing"
            report["executions"].append(execution_envelope(
                spec, present=present, readable=False, operationally_valid=False,
                readability_reasons=(loaded,), operational_reasons=(loaded,),
            ))
            continue
        _out, states, run, high_path = loaded
        planner_value = run.get("planner")
        executions_value = run.get("executions")
        if ((planner_value is not None and not isinstance(planner_value, Mapping))
                or not isinstance(executions_value, list)
                or not all(isinstance(row, Mapping) for row in executions_value)):
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_root_type",),
                operational_reasons=("invalid_root_type",),
                source_refs=(str(high_path.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        planner = planner_value or {}
        executions = executions_value
        log_path = root / f"exp_agent2_3_{spec['run_id']}.log"
        logs = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []
        from .runner import _result_errors
        required_checker_states = {
            "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
            "hlreasoner_0",
        }
        checker_available = (
            required_checker_states <= states.keys()
            and log_path.is_file()
            and bool(logs)
        )
        environment: dict[str, Any] = {}
        env_out = root / f"exp_env2_3_{spec['run_id']}" / "out"
        for env_path in sorted(env_out.glob("*.json")):
            try:
                env_value = json.loads(env_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(env_value, dict):
                environment = env_value
                break
        env_log_path = root / f"exp_env2_3_{spec['run_id']}.log"
        env_logs = env_log_path.read_text(encoding="utf-8", errors="replace").splitlines() if env_log_path.is_file() else []
        treatment = states.get("hlreasoner_0", {}).get("treatment")
        checker_errors = _result_errors(states, logs, environment or None, env_logs, dict(spec.get("factors", {})) or None) if checker_available else []
        legal = sum(isinstance(row, Mapping) and row.get("legal") is True for row in executions)
        rejected = sum(isinstance(row, Mapping) and row.get("legal") is False for row in executions)
        task_success = run.get("phase") == "complete" and run.get("goal_observed_at") is not None
        failure = run.get("failure") if isinstance(run.get("failure"), Mapping) else {}
        failure_code = failure.get("code")
        analysis_eligible = failure_code not in {"no-intention", "belief-error"}
        rows.append({
            "execution_id": spec["execution_id"], "run_id": spec["run_id"],
            "task_dimensions": {
                "num_blocks": (treatment or {}).get("num_blocks"),
                "table_len": (treatment or {}).get("table_len"),
            },
            "goal_fact": run.get("goal_fact"), "planner_status": planner.get("status"),
            "planning_elapsed_seconds": planner.get("elapsed_seconds"),
            "plan_length": planner.get("length"), "executed_actions": len(executions),
            "legal_actions": legal, "rejected_actions": rejected,
            "goal_observed_at": run.get("goal_observed_at"), "failure": run.get("failure"),
            "analysis_eligible": analysis_eligible,
            "optimality": {"status": "unavailable", "reason": "no_optimal_reference_plan"},
        })
        operational_reasons = list(checker_errors)
        operational_reasons.extend(treatment_identity_reasons(
            spec, treatment,
            keys=("protocol_version", "treatment_id", "task_id"),
        ))
        if not required_checker_states <= states.keys():
            operational_reasons.append("required_state_missing")
        if not log_path.is_file() or not logs:
            operational_reasons.append("required_log_missing")
        if not environment or len(list(env_out.glob("*.json"))) != 1:
            operational_reasons.append("required_environment_state_missing_or_duplicated")
        if not env_logs:
            operational_reasons.append("required_environment_log_missing")
        if any("traceback" in line.lower() for line in (*logs, *env_logs)):
            operational_reasons.append("fatal_runtime_error")
        operational = not operational_reasons
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=(
                "unavailable" if not checker_available
                else "passed" if not checker_errors else "failed"
            ),
            task_status="success" if task_success else "failure",
            task_reason=None if task_success else str((run.get("failure") or {}).get("code", "goal_not_observed")),
            termination_reason="task_completed" if task_success else "operational_failure" if run.get("phase") == "failed" else "time_limit",
            identity={"treatment": treatment},
            metric_availability={
                "planning": "existing", "execution": "existing",
                "optimality": "unavailable",
                "certificate": (
                    "existing" if checker_available
                    else {"status": "unavailable", "reason": "checker_input_missing"}
                ),
            },
            source_refs=(str(high_path.relative_to(root)).replace("\\", "/"),),
        ))
    eligibility = analysis_eligibility(report["executions"])
    operational_ids = set(eligibility["eligible_execution_ids"])
    eligible_ids = [
        row["execution_id"] for row in rows
        if row["execution_id"] in operational_ids and row["analysis_eligible"]
    ]
    invalid_task_ids = {
        row["execution_id"] for row in rows if not row["analysis_eligible"]
    }
    eligibility["eligible_execution_ids"] = eligible_ids
    for exclusion in eligibility["excluded"]:
        if exclusion["execution_id"] in invalid_task_ids:
            exclusion["reasons"].append("invalid_task_construction")
    eligibility["excluded"].extend({
        "execution_id": execution_id,
        "reasons": ["invalid_task_construction"],
    } for execution_id in sorted(invalid_task_ids - {
        row["execution_id"] for row in eligibility["excluded"]
    }))
    eligible_rows = [row for row in rows if row["execution_id"] in eligible_ids]
    report["analyses"] = {"planning_tasks": {
        "unit": "execution", "nesting": "none",
        "eligibility": eligibility,
        "rows": rows,
        "summary": {"plan_length": summarize_numbers([row["plan_length"] for row in eligible_rows], include_iqr=True),
                    "executed_actions": summarize_numbers([row["executed_actions"] for row in eligible_rows], include_iqr=True)},
    }}
    write_execution_metrics(root, report)
    return report
