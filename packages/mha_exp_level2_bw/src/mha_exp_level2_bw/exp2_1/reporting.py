"""Execution reporting for the 2-1-BW finite-state controller."""

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
from mha_exp_common.metrics import summarize_numbers, wilson_interval


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


def _reasoner_shape_error(reasoner: Mapping[str, Any]) -> str | None:
    """Validate retained episode and decision containers before normalization."""

    for key in ("episode_results", "decision_trace"):
        rows = reasoner.get(key)
        if not isinstance(rows, list):
            return "invalid_root_type"
        if not all(isinstance(row, Mapping) for row in rows):
            return "invalid_trace_row"
    open_episode = reasoner.get("open_episode")
    if open_episode is not None and not isinstance(open_episode, Mapping):
        return "invalid_root_type"
    illegal_actions = reasoner.get("illegal_actions")
    if illegal_actions is not None and (
        type(illegal_actions) is not int or illegal_actions < 0
    ):
        return "invalid_counter"
    return None


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build episode, legality, and observed controller-path metrics."""

    root = Path(root).resolve()
    discovered = sorted(int(path.name.rsplit("_", 1)[-1]) for path in root.glob("exp_agent2_1_*")
                        if path.name.rsplit("_", 1)[-1].isdigit())
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in discovered]
    report = new_report("2-1-bw", None if expected_executions is None else specs)
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_id = int(spec["run_id"])
        out = root / f"exp_agent2_1_{run_id}" / "out"
        states = _states(out) if out.is_dir() else {}
        reasoner = states.get("llreasoner_0")
        if not isinstance(reasoner, Mapping):
            reason = "required_state_missing" if out.is_dir() else "run_missing"
            report["executions"].append(execution_envelope(
                spec, present=out.is_dir(), readable=False, operationally_valid=False,
                readability_reasons=(reason,), operational_reasons=(reason,),
            ))
            continue
        shape_error = _reasoner_shape_error(reasoner)
        if shape_error is not None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(shape_error,),
                operational_reasons=(shape_error,),
            ))
            continue
        current_episodes = [dict(row) for row in reasoner["episode_results"]]
        for row in current_episodes:
            episodes.append({"execution_id": spec["execution_id"], "run_id": run_id,
                             "censoring": {"status": "observed", "reason": None}, **row})
        open_episode = reasoner.get("open_episode")
        if isinstance(open_episode, Mapping):
            episodes.append({"execution_id": spec["execution_id"], "run_id": run_id,
                             "outcome": "unobserved", "failure_reason": None,
                             "censoring": {
                                 "status": "right_censored",
                                 "reason": "external_execution_end",
                             }, **dict(open_episode)})
        histogram: Counter[str] = Counter()
        for index, row in enumerate(reasoner["decision_trace"]):
            histogram[str(row.get("reason"))] += 1
            decisions.append({"execution_id": spec["execution_id"], "run_id": run_id,
                              "decision_id": index, **dict(row)})
        observed = [row for row in current_episodes if row.get("outcome") in {"success", "failure"}]
        successes = sum(row.get("outcome") == "success" for row in observed)
        run_rows.append({
            "execution_id": spec["execution_id"], "run_id": run_id,
            "successes": successes, "completed_episodes": len(observed),
            "success_interval": wilson_interval(successes, len(observed)),
            "observed_controller_path_counts": dict(sorted(histogram.items())),
            "illegal_actions": reasoner.get("illegal_actions"),
        })
        log_paths = (
            root / f"exp_agent2_1_{run_id}.log",
            root / f"exp_env2_1_{run_id}.log",
        )
        log_rows = [
            path.read_text(encoding="utf-8", errors="replace").splitlines()
            if path.is_file() else []
            for path in log_paths
        ]
        logs = [line for rows in log_rows for line in rows]
        env_states = _states(root / f"exp_env2_1_{run_id}" / "out")
        environment = next(iter(env_states.values()), None)
        from .runner import check_results
        required_checker_states = {"perceptor_0", "actuator_0", "llreasoner_0"}
        checker_available = (
            required_checker_states <= states.keys()
            and isinstance(environment, dict)
            and all(log_rows)
        )
        certificate = (
            check_results(
                dict(states), environment, logs, verbose=False,
                expected_treatment=dict(spec.get("factors", {})),
            )
            if checker_available else None
        )
        operational_reasons = []
        operational_reasons.extend(treatment_identity_reasons(
            spec, reasoner,
            keys=(
                "protocol_version", "treatment_id", "manifest_digest", "task_count",
                "run_id", "task_id", "seed", "reasoner_seed", "initial_state_digest",
                "table_len", "num_blocks", "top", "bottom",
            ),
        ))
        if not required_checker_states <= states.keys() or not isinstance(
            environment, dict
        ):
            operational_reasons.append("required_state_missing")
        if not all(log_rows):
            operational_reasons.append("required_log_missing")
        if any("traceback" in line.lower() for line in logs):
            operational_reasons.append("fatal_runtime_error")
        operational = not operational_reasons
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=(
                "unavailable" if certificate is None
                else "passed" if certificate else "failed"
            ),
            task_status="success" if successes else "failure" if observed else "unobserved",
            task_reason=None if successes else "no_successful_episode" if observed else "no_completed_episode",
            termination_reason=(
                "external_stop" if isinstance(open_episode, Mapping)
                else str(reasoner.get("terminal_reason") or "unknown")
            ),
            identity={"treatment": {key: reasoner.get(key) for key in (
                "protocol_version", "treatment_id", "manifest_digest", "run_id",
                "task_id", "seed", "reasoner_seed", "initial_state_digest", "top", "bottom",
            )}},
            metric_availability={
                "episodes": "instrumented", "decision_paths": "existing",
                "legality": "existing",
                "certificate": (
                    "existing" if checker_available
                    else {"status": "unavailable", "reason": "checker_input_missing"}
                ),
            },
            source_refs=(str(next(out.glob("*.llreasoner_0.json")).relative_to(root)).replace("\\", "/"),),
        ))
    eligibility = analysis_eligibility(report["executions"])
    eligible_episodes = eligible_analysis_rows(episodes, eligibility)
    eligible_decisions = eligible_analysis_rows(decisions, eligibility)
    eligible_runs = eligible_analysis_rows(run_rows, eligibility)
    report["analyses"] = {
        "episodes": {"unit": "episode", "nesting": "episodes_within_execution",
                     "eligibility": eligibility, "rows": episodes,
                     "summary": {"count": len(eligible_episodes),
                                 "raw_descriptive_count": len(episodes),
                                 "elapsed_seconds": summarize_numbers(
                         [row.get("elapsed_seconds") for row in eligible_episodes], include_iqr=True)}},
        "decisions": {"unit": "controller_decision", "nesting": "decisions_within_episode_within_execution",
                      "eligibility": eligibility, "rows": decisions,
                      "summary": {"count": len(eligible_decisions),
                                  "raw_descriptive_count": len(decisions)}},
        "controller_runs": {"unit": "execution", "nesting": "none",
                            "eligibility": eligibility, "rows": run_rows,
                            "summary": {"successes": summarize_numbers(
                                [row["successes"] for row in eligible_runs], include_iqr=True)}},
    }
    write_execution_metrics(root, report)
    return report
