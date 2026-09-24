"""Execution reporting for pretrained 2-5-BW policy runs."""

from __future__ import annotations

import json
from collections import Counter
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
from mha_exp_common.metrics import empirical_action_diversity, summarize_numbers


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


def _goal_runs_shape_error(value: Any) -> str | None:
    """Validate nested goal, transfer, and atomic decision collections."""

    if not isinstance(value, list):
        return "invalid_root_type"
    if not all(isinstance(goal, Mapping) for goal in value):
        return "invalid_trace_row"
    for goal in value:
        plan = goal.get("plan")
        if not isinstance(plan, Mapping):
            return "invalid_root_type"
        transfers = plan.get("transfers")
        if not isinstance(transfers, list) or not all(
            isinstance(row, Mapping) for row in transfers
        ):
            return "invalid_trace_row"
        if any(
            not isinstance(transfer.get("atomic"), list)
            or not all(
                isinstance(row, Mapping) for row in transfer["atomic"]
            )
            for transfer in transfers
        ):
            return "invalid_trace_row"
    return None


def _artifact_identity(
    state: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str | dict[str, str]]:
    """Return retained policy identity and its current artifact join status."""

    identity = {
        "policy_id": state.get("policy_id") if state is not None else None,
        "checkpoint_sha256": (
            state.get("checkpoint_sha256") if state is not None else None
        ),
    }
    if not all(
        isinstance(identity[field], str) and identity[field]
        for field in identity
    ):
        return identity, {
            "status": "unavailable",
            "reason": "retained_artifact_identity_missing",
        }
    if not manifest:
        return identity, {
            "status": "unavailable",
            "reason": "preparation_artifact_unavailable",
        }
    expected = {
        "policy_id": manifest.get("architecture"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
    }
    if identity != expected:
        return identity, {
            "status": "unavailable",
            "reason": "artifact_identity_mismatch",
        }
    return identity, "existing"


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build goal, transfer, decision, and preparation-artifact metrics."""

    root = Path(root).resolve()
    discovered = sorted(int(path.name.rsplit("_", 1)[-1]) for path in root.glob("exp_agent2_5_*")
                        if path.name.rsplit("_", 1)[-1].isdigit())
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in discovered]
    report = new_report("2-5-bw", None if expected_executions is None else specs)
    goal_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    from .policy import artifact_paths, validate_manifest
    manifest_path, checkpoint_path = artifact_paths()
    try:
        manifest = validate_manifest(manifest_path, checkpoint_path)
        artifact_id = str(manifest.get("checkpoint_sha256", manifest.get("architecture", "frozen-policy")))
        report["preparation_artifacts"] = [{
            "artifact_id": artifact_id, "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path), "architecture": manifest.get("architecture"),
            "training": manifest.get("training"), "certification": manifest.get("certification"),
        }]
    except (OSError, ValueError, KeyError):
        manifest, artifact_id = {}, None
    for spec in specs:
        run_id = int(spec["run_id"])
        out = root / f"exp_agent2_5_{run_id}" / "out"
        states = _states(out) if out.is_dir() else {}
        high = states.get("hlreasoner_0")
        if not isinstance(high, Mapping):
            readability_reason = (
                "required_state_missing" if out.is_dir() else "run_missing"
            )
            report["executions"].append(execution_envelope(
                spec, present=out.is_dir(), readable=False, operationally_valid=False,
                readability_reasons=(readability_reason,),
                operational_reasons=(readability_reason,),
            ))
            continue
        shape_error = _goal_runs_shape_error(high.get("goal_runs"))
        if shape_error is not None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(shape_error,), operational_reasons=(shape_error,),
            ))
            continue
        retained_artifact, artifact_availability = _artifact_identity(
            states.get("llreasoner_0"), manifest
        )
        retained_artifact_id = retained_artifact["checkpoint_sha256"]
        histogram: Counter[str] = Counter()
        for goal_index, goal_run in enumerate(high["goal_runs"]):
            goal_rows.append({
                "execution_id": spec["execution_id"], "run_id": run_id,
                "goal_id": goal_run.get("goal_id", f"goal-{goal_index}"),
                "goal": goal_run.get("goal"), "status": goal_run.get("status"),
                "final_goal_fact": goal_run.get("final_goal_fact"),
                "final_observation_id": goal_run.get("final_observation_id"),
                "elapsed_time": {"status": "unavailable", "reason": "no_goal_time_boundaries"},
            })
            plan = goal_run["plan"]
            for transfer_index, transfer in enumerate(plan["transfers"]):
                transfer_id = transfer.get("goal_id", f"goal-{goal_index}-transfer-{transfer_index}")
                atomic = transfer["atomic"]
                transfer_rows.append({
                    "execution_id": spec["execution_id"], "run_id": run_id,
                    "goal_id": goal_rows[-1]["goal_id"], "transfer_id": transfer_id,
                    "spec": transfer.get("spec"), "status": transfer.get("status", "completed"),
                    "atomic_action_count": len(atomic),
                    "terminal_observation_id": transfer.get("terminal_observation_id"),
                    "revision_observation_id": transfer.get("revision_observation_id"),
                })
                for decision_index, decision in enumerate(atomic):
                    selected_action = decision.get("selected_action")
                    if selected_action is not None:
                        histogram[str(selected_action)] += 1
                    decision_rows.append({
                        "execution_id": spec["execution_id"], "run_id": run_id,
                        "goal_id": goal_rows[-1]["goal_id"], "transfer_id": transfer_id,
                        "decision_id": decision.get("atomic_action_id", f"{transfer_id}:{decision_index}"),
                        "selected_action": selected_action,
                        "legal": decision.get("legal"), "observation_id": decision.get("observation_id"),
                        "input_sha256": decision.get("input_sha256"),
                        "policy_artifact_id": retained_artifact_id,
                        "action_mode": "greedy", "q_values": decision.get("q_values"),
                        "legal_action_q_margin": {"status": "unavailable", "reason": "legal_alternative_set_not_retained"},
                    })
        total = sum(histogram.values())
        run_rows.append({
            "execution_id": spec["execution_id"], "run_id": run_id,
            "policy_artifact_id": retained_artifact_id,
            "action_histogram": dict(sorted(histogram.items())),
            "total_actions": total, "distinct_actions": len(histogram),
            "empirical_action_diversity_bits": empirical_action_diversity(histogram),
            "completed_goals": high.get("goal_completion_count"),
            "completed_transfers": high.get("completed_transfer_count"),
            "task_id": high.get("treatment", {}).get("task_id"),
            "paired_plan_length": high.get("treatment", {}).get(
                "atomic_plan_length"
            ),
            "paired_transfer_count": high.get("treatment", {}).get(
                "transfer_count"
            ),
            "completion_outcome": high.get("terminal_reason"),
        })
        env_states = _states(root / f"exp_env2_5_{run_id}" / "out")
        environment = next(iter(env_states.values()), {})
        log_paths = (
            root / f"exp_agent2_5_{run_id}.log",
            root / f"exp_env2_5_{run_id}.log",
        )
        log_rows = [
            path.read_text(encoding="utf-8", errors="replace").splitlines()
            if path.is_file() else []
            for path in log_paths
        ]
        logs = [line for rows in log_rows for line in rows]
        from .runner import check_results
        required_checker_states = {
            "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
            "goalgraph_0", "hlreasoner_0",
        }
        checker_available = (
            bool(manifest)
            and required_checker_states <= states.keys()
            and bool(env_states)
            and all(log_rows)
        )
        certificate = (
            check_results(dict(states), environment, logs, manifest, verbose=False)
            if checker_available else None
        )
        task_success = int(high.get("goal_completion_count", 0)) > 0
        operational_reasons = []
        operational_reasons.extend(treatment_identity_reasons(
            spec,
            high.get("treatment"),
            keys=("protocol_version", "treatment_id", "task_id"),
        ))
        if not required_checker_states <= states.keys() or not env_states:
            operational_reasons.append("required_state_missing")
        if not all(log_rows):
            operational_reasons.append("required_log_missing")
        if high.get("failure") is not None or any(
            "traceback" in line.lower() for line in logs
        ):
            operational_reasons.append("fatal_runtime_error")
        operational = not operational_reasons
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=(
                "unavailable" if certificate is None
                else "passed" if certificate else "failed"
            ),
            task_status="success" if task_success else "failure",
            task_reason=None if task_success else "no_completed_goal",
            termination_reason="task_completed" if high.get("terminal_reason") == "goal-completion-limit" else "operational_failure" if high.get("failure") else "unknown",
            identity={"policy_artifact": retained_artifact,
                      "treatment": high.get("treatment")},
            metric_availability={"goal_runs": "existing", "atomic_decisions": "existing",
                                 "action_diversity": "derived",
                                 "artifact_identity": artifact_availability,
                                 "legal_action_q_margin": "unavailable"},
            source_refs=(str(next(out.glob("*.hlreasoner_0.json")).relative_to(root)).replace("\\", "/"),),
        ))
    eligibility = analysis_eligibility(
        report["executions"], required_metrics=("artifact_identity",)
    )
    eligible_goals = eligible_analysis_rows(goal_rows, eligibility)
    eligible_transfers = eligible_analysis_rows(transfer_rows, eligibility)
    eligible_decisions = eligible_analysis_rows(decision_rows, eligibility)
    eligible_runs = eligible_analysis_rows(run_rows, eligibility)
    report["analyses"] = {
        "goals": {"unit": "goal", "nesting": "goals_within_execution", "rows": goal_rows,
                  "eligibility": eligibility,
                  "summary": {"count": len(eligible_goals),
                              "raw_descriptive_count": len(goal_rows)}},
        "transfers": {"unit": "transfer", "nesting": "transfers_within_goal_within_execution",
                      "eligibility": eligibility, "rows": transfer_rows,
                      "summary": {"count": len(eligible_transfers),
                                  "raw_descriptive_count": len(transfer_rows)}},
        "atomic_decisions": {"unit": "atomic_decision", "nesting": "decisions_within_transfer",
                             "eligibility": eligibility, "rows": decision_rows,
                             "summary": {"count": len(eligible_decisions),
                                         "raw_descriptive_count": len(decision_rows)}},
        "policy_runs": {"unit": "execution", "nesting": "none",
                        "eligibility": eligibility, "rows": run_rows,
                        "summary": {"total_actions": summarize_numbers(
                            [row["total_actions"] for row in eligible_runs], include_iqr=True)}},
    }
    write_execution_metrics(root, report)
    return report
