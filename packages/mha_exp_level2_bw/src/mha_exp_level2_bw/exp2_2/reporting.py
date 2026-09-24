"""Pure per-run and batch reporting for experiment 2-2-BW."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
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
from mha_exp_common.metrics import summarize_numbers, wilson_interval

from .protocol import DQNProtocol, PROTOCOL_VERSION


RUN_SUMMARY_FILENAME = "run-summary.json"
BATCH_SUMMARY_FILENAME = "training-summary.json"


def protocol_provenance(
    *,
    synchronized_training: bool,
    warmup_transitions: int,
    total_training_transitions: int,
    training_updates: int,
    target_sync_steps: int,
    evaluation_seeds: Sequence[int],
    behavior_window_transitions: int,
    optimization_window_updates: int,
) -> dict[str, Any]:
    """Build the complete cohort-defining protocol record."""

    return {
        "scheduling_mode": "synchronous" if synchronized_training else "asynchronous",
        "warmup_transitions": warmup_transitions,
        "total_training_transitions": total_training_transitions,
        "training_updates": training_updates,
        "target_sync_steps": target_sync_steps,
        "evaluation_seeds": list(evaluation_seeds),
        "behavior_window_transitions": behavior_window_transitions,
        "optimization_window_updates": optimization_window_updates,
    }


def build_run_summary(
    *,
    run: int,
    provenance: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    module_ids: Mapping[str, str],
    environment_state: Mapping[str, Any] | None,
    checker_passed: bool,
    checker_reasons: Sequence[str],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one JSON-safe summary from compact authoritative states."""

    ll = dict(states.get(module_ids["ll_reasoner"], {}))
    memory = dict(states.get(module_ids["memory"], {}))
    learner = dict(states.get(module_ids["learner"], {}))
    goal_graph = dict(states.get(module_ids["goal_graph"], {}))
    environment = dict(environment_state or {})
    behavior = list(ll.get("behavior_windows", []))
    optimization = list(learner.get("optimization_windows", []))
    evaluation = list(ll.get("evaluation_cases", []))

    ordinary_episodes = int(ll.get("training_successes", 0)) + int(
        ll.get("training_truncations", 0)
    )
    evaluation_successes = sum(bool(row.get("success")) for row in evaluation)
    successful_lengths = [
        float(row["steps"])
        for row in evaluation
        if row.get("success") and isinstance(row.get("steps"), (int, float))
    ]
    training_seconds = (
        float(behavior[-1]["elapsed_seconds"]) if behavior else None
    )
    optimization_seconds = (
        float(optimization[-1]["elapsed_seconds"]) if optimization else None
    )
    transition_count = int(memory.get("training_transitions", 0))
    update_count = int(learner.get("training_updates", 0))
    if provenance.get("protocol_version") == PROTOCOL_VERSION:
        protocol_complete = checker_passed and ll.get("phase") == "complete"
        timestamps = ll.get("phase_timestamps", {})
        start = timestamps.get("training_started")
        end = timestamps.get("training_finished")
        training_seconds = end - start if start is not None and end is not None else None
        learned_start = learner.get("training_started_at")
        learned_end = learner.get("last_update_finished")
        optimization_seconds = learned_end - learned_start if learned_start is not None and learned_end is not None else None
    else:
        protocol_complete = (
            ll.get("phase") == "complete"
            and transition_count == int(provenance["total_training_transitions"])
            and update_count == int(provenance["training_updates"])
            and int(memory.get("credits_drained", 0))
            == int(provenance["training_updates"])
            and memory.get("final_training_transition_terminal") is True
            and [row.get("requested_seed") for row in evaluation]
            == list(provenance["evaluation_seeds"])
            and [row.get("applied_seed") for row in evaluation]
            == list(provenance["evaluation_seeds"])
        )

    return {
        "run": run,
        "protocol": dict(provenance),
        "treatment": ll.get("treatment"),
        "phase_timestamps": ll.get("phase_timestamps"),
        "checker_passed": bool(checker_passed),
        "protocol_complete": protocol_complete,
        "incomplete_reasons": list(checker_reasons),
        "replay": {key: memory.get(key) for key in (
            "replay_entries", "buffer_size", "evictions", "stale_priority_feedback", "pending_tail",
            "collection_closed", "sampling_closed", "closure_acknowledged",
            "her_entries", "her_success_entries", "her_episodes", "her_source_transitions",
            "her_candidate_goals", "pending_her_episode")},
        "counts": {
            "unused_batches": learner.get("unused_batches"),
            "updates_per_transition": update_count / transition_count if transition_count else None,
            "configured_warmup_transitions": provenance["warmup_transitions"],
            "training_transitions": transition_count,
            "learner_updates": update_count,
            "target_syncs": int(learner.get("target_syncs", 0)),
            "training_seconds": training_seconds,
            "optimization_seconds": optimization_seconds,
            "transition_throughput": (
                None
                if not training_seconds
                else transition_count / training_seconds
            ),
            "update_throughput": (
                None if not optimization_seconds else update_count / optimization_seconds
            ),
        },
        "models": {
            "published": learner.get("models_published"),
            "first_publication_update": learner.get("first_model_publication_update"),
            "final_publication_update": learner.get("final_model_publication_update"),
            "installed": ll.get("models_installed"),
            "first_install_update": ll.get("first_model_install_update"),
            "final_install_update": ll.get("final_model_install_update"),
            "learner_fingerprint": learner.get("final_model_fingerprint"),
            "reasoner_fingerprint": ll.get("installed_final_model_fingerprint"),
            "checkpoint_fingerprint": checkpoint.get("model_fingerprint"),
        },
        "training": {
            "successes": int(ll.get("training_successes", 0)),
            "truncations": int(ll.get("training_truncations", 0)),
            "budget_cutoffs": int(ll.get("training_budget_cutoffs", 0)),
            "cutoff_episode_length": ll.get("cutoff_episode_length"),
            "ordinary_episodes": ordinary_episodes,
            "success_rate": (
                None
                if ordinary_episodes == 0
                else int(ll.get("training_successes", 0)) / ordinary_episodes
            ),
            "training_resets_at_cutoff": ll.get("training_resets_at_cutoff"),
            "training_resets_final": ll.get("training_resets_final"),
            "behavior_windows": behavior,
        },
        "optimization": {
            "device": learner.get("device"),
            "torch_version": learner.get("torch_version"),
            "cuda_runtime": learner.get("cuda_runtime"),
            "cuda_available": learner.get("cuda_available"),
            "cuda_device_name": learner.get("cuda_device_name"),
            "windows": optimization,
            "freeze_update": learner.get("freeze_update"),
            "evaluation_start_update": learner.get("evaluation_start_update"),
            "final_save_update": learner.get("final_save_update"),
            "post_freeze_update_attempts": learner.get("post_freeze_update_attempts"),
        },
        "evaluation": {
            "cases": evaluation,
            "successes": evaluation_successes,
            "success_rate": (
                None if not evaluation else evaluation_successes / len(evaluation)
            ),
            "mean_successful_episode_length": (
                None if not successful_lengths else fmean(successful_lengths)
            ),
            "requested_seeds": [row.get("requested_seed") for row in evaluation],
            "applied_seeds": [row.get("applied_seed") for row in evaluation],
            "action_modes": [row.get("action_mode") for row in evaluation],
        },
        "isolation": {
            "evaluation_replay_insertion_attempts": memory.get(
                "evaluation_replay_insertion_attempts"
            ),
            "rejected_post_budget_transitions": memory.get(
                "rejected_post_budget_transitions"
            ),
            "final_training_transition_terminal": memory.get(
                "final_training_transition_terminal"
            ),
        },
        "goal_graph": {
            key: value
            for key, value in goal_graph.items()
            if key != "active_goal"
        },
        "environment": environment,
        "checkpoint": dict(checkpoint),
    }


def write_run_summary(path: Path, summary: Mapping[str, Any]) -> Path:
    """Write one run summary outside the agent state directory."""

    path.mkdir(parents=True, exist_ok=True)
    target = path / RUN_SUMMARY_FILENAME
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return target


def print_run_summary(summary: Mapping[str, Any]) -> None:
    """Print the most useful descriptive outcome without performance gating."""

    counts = summary["counts"]
    training = summary["training"]
    evaluation = summary["evaluation"]
    print(
        "2-2-BW run "
        f"{summary['run']}: checker={'PASS' if summary['checker_passed'] else 'FAIL'}, "
        f"transitions={counts['training_transitions']}, "
        f"updates={counts['learner_updates']}, "
        f"training successes={training['successes']}, "
        f"evaluation successes={evaluation['successes']}/{len(evaluation['cases'])}."
    )


def _numeric_summary(values: Sequence[float | int | None]) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None]
    return {
        "mean": None if not defined else fmean(defined),
        "std": None if not defined else pstdev(defined),
        "min": None if not defined else min(defined),
        "max": None if not defined else max(defined),
        "contributing_runs": len(defined),
    }


def _protocol_error(summary: Mapping[str, Any]) -> str | None:
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        return "missing protocol provenance"
    current = protocol.get("protocol_version") == PROTOCOL_VERSION
    if current:
        try:
            DQNProtocol.from_record(protocol)
        except (KeyError, TypeError, ValueError):
            return "invalid Rainbow protocol provenance"
        transition_count = int(summary.get("counts", {}).get("training_transitions", 0))
        update_count = int(summary.get("counts", {}).get("learner_updates", 0))
    else:
        required = {
            "scheduling_mode",
            "warmup_transitions",
            "total_training_transitions",
            "training_updates",
            "target_sync_steps",
            "evaluation_seeds",
            "behavior_window_transitions",
            "optimization_window_updates",
        }
        if set(protocol) != required:
            return "incomplete protocol provenance"
        integer_fields = required - {"scheduling_mode", "evaluation_seeds"}
        if any(type(protocol.get(field)) is not int or protocol[field] < 0
               for field in integer_fields):
            return "invalid protocol provenance"
        if (protocol["behavior_window_transitions"] == 0
                or protocol["optimization_window_updates"] == 0
                or not isinstance(protocol.get("evaluation_seeds"), list)):
            return "invalid protocol provenance"
        transition_count = int(protocol["total_training_transitions"])
        update_count = int(protocol["training_updates"])
    behavior = summary.get("training", {}).get("behavior_windows", [])
    for index, row in enumerate(behavior):
        expected_start = index * int(protocol["behavior_window_transitions"]) + 1
        expected_end = min(
            expected_start + int(protocol["behavior_window_transitions"]) - 1,
            transition_count,
        )
        if row.get("transition_start") != expected_start or row.get("transition_end") != expected_end:
            return "invalid behavioral window boundary"
    optimization = summary.get("optimization", {}).get("windows", [])
    for index, row in enumerate(optimization):
        expected_start = index * int(protocol["optimization_window_updates"]) + 1
        expected_end = min(
            expected_start + int(protocol["optimization_window_updates"]) - 1,
            update_count,
        )
        if row.get("update_start") != expected_start or row.get("update_end") != expected_end:
            return "invalid optimization window boundary"
    return None


def _summary_shape_error(summary: Mapping[str, Any]) -> str | None:
    """Validate required container types before normalizing summary rows."""

    for key in (
        "counts", "training", "optimization", "evaluation", "models",
        "isolation", "checkpoint", "goal_graph", "environment",
    ):
        if not isinstance(summary.get(key), Mapping):
            return "invalid_root_type"
    for parent, key in (
        ("training", "behavior_windows"),
        ("optimization", "windows"),
        ("evaluation", "cases"),
    ):
        rows = summary[parent].get(key)
        if not isinstance(rows, list) or not all(
            isinstance(row, Mapping) for row in rows
        ):
            return "invalid_root_type"
    for parent, fields in (
        ("training", ("successes", "truncations", "budget_cutoffs", "ordinary_episodes")),
        ("evaluation", ("successes",)),
    ):
        if any(type(summary[parent].get(field)) is not int
               or summary[parent][field] < 0 for field in fields):
            return "invalid_root_type"
    return None


def _aligned_behavior(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maximum = max((len(run["training"]["behavior_windows"]) for run in runs), default=0)
    for index in range(maximum):
        current = [
            run["training"]["behavior_windows"][index]
            for run in runs
            if len(run["training"]["behavior_windows"]) > index
        ]
        completed = sum(int(row["completed_episodes"]) for row in current)
        successes = sum(int(row["successes"]) for row in current)
        native_actions = sum(int(row["native_actions"]) for row in current)
        illegal_actions = sum(int(row["illegal_actions"]) for row in current)
        defined_rates = [row["success_rate"] for row in current if row["success_rate"] is not None]
        defined_lengths = [
            row["mean_episode_length"]
            for row in current
            if row["mean_episode_length"] is not None
        ]
        rows.append(
            {
                "window_index": index,
                "transition_start": current[0]["transition_start"],
                "transition_end": max(row["transition_end"] for row in current),
                "minimum_transition_end": min(row["transition_end"] for row in current),
                "contributing_runs": len(current),
                "pooled_successes": successes,
                "pooled_completed_episodes": completed,
                "pooled_success_rate": None if completed == 0 else successes / completed,
                "mean_defined_success_rate": None if not defined_rates else fmean(defined_rates),
                "success_rate_contributing_runs": len(defined_rates),
                "mean_defined_episode_length": None if not defined_lengths else fmean(defined_lengths),
                "episode_length_contributing_runs": len(defined_lengths),
                "pooled_illegal_action_rate": (
                    None if native_actions == 0 else illegal_actions / native_actions
                ),
            }
        )
    return rows


def _aligned_optimization(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maximum = max((len(run["optimization"]["windows"]) for run in runs), default=0)
    for index in range(maximum):
        current = [
            run["optimization"]["windows"][index]
            for run in runs
            if len(run["optimization"]["windows"]) > index
        ]
        rows.append(
            {
                "window_index": index,
                "update_start": current[0]["update_start"],
                "update_end": max(row["update_end"] for row in current),
                "minimum_update_end": min(row["update_end"] for row in current),
                "contributing_runs": len(current),
                "mean_loss": _numeric_summary([row.get("mean_loss") for row in current]),
                **{field: _numeric_summary([row.get(field) for row in current]) for field in (
                    "mean_abs_td_error", "beta", "mean_replay_age", "mean_policy_lag")},
                "final_loss": _numeric_summary([row.get("final_loss") for row in current]),
                "target_syncs": sum(int(row.get("target_syncs", 0)) for row in current),
            }
        )
    return rows


def _paired_delta(
    runs: Sequence[Mapping[str, Any]],
    rows_path: tuple[str, str],
    field: str,
) -> dict[str, Any]:
    deltas: list[float] = []
    for run in runs:
        rows = run[rows_path[0]][rows_path[1]]
        if not rows:
            continue
        first = rows[0].get(field)
        last = rows[-1].get(field)
        if first is not None and last is not None:
            deltas.append(float(last) - float(first))
    return {
        "mean": None if not deltas else fmean(deltas),
        "contributing_runs": len(deltas),
    }


def aggregate_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    attempted_runs: int | None = None,
    malformed: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Partition compatible protocols and aggregate all readable runs."""

    compatible: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    incompatible: list[dict[str, Any]] = [dict(item) for item in malformed]
    for summary in summaries:
        error = _protocol_error(summary)
        if error:
            incompatible.append({"run": summary.get("run"), "reason": error})
            continue
        key = json.dumps(summary["protocol"], sort_keys=True, separators=(",", ":"))
        compatible[key].append(summary)

    groups: list[dict[str, Any]] = []
    for key, runs in sorted(compatible.items()):
        training_successes = sum(int(run["training"]["successes"]) for run in runs)
        ordinary_episodes = sum(int(run["training"]["ordinary_episodes"]) for run in runs)
        evaluation_successes = sum(int(run["evaluation"]["successes"]) for run in runs)
        evaluation_cases = sum(len(run["evaluation"]["cases"]) for run in runs)
        successful_lengths = [
            float(case["steps"])
            for run in runs
            for case in run["evaluation"]["cases"]
            if case.get("success") and case.get("steps") is not None
        ]
        groups.append(
            {
                "protocol": json.loads(key),
                "runs": [run.get("run") for run in runs],
                "readable_runs": len(runs),
                "protocol_complete_runs": sum(bool(run.get("protocol_complete")) for run in runs),
                "checker_passing_runs": sum(bool(run.get("checker_passed")) for run in runs),
                "counts": {
                    field: _numeric_summary([run["counts"].get(field) for run in runs])
                    for field in (
                        "training_transitions",
                        "learner_updates",
                        "target_syncs",
                        "training_seconds",
                        "transition_throughput",
                        "update_throughput",
                        "updates_per_transition",
                        "unused_batches",
                    )
                },
                "replay": {
                    field: _numeric_summary([run.get("replay", {}).get(field) for run in runs])
                    for field in ("replay_entries", "buffer_size", "evictions", "stale_priority_feedback",
                                  "her_entries", "her_success_entries", "her_episodes", "her_source_transitions",
                                  "her_candidate_goals")
                },
                "training": {
                    "pooled_successes": training_successes,
                    "pooled_ordinary_episodes": ordinary_episodes,
                    "pooled_success_rate": (
                        None if ordinary_episodes == 0 else training_successes / ordinary_episodes
                    ),
                    "mean_per_run_success_rate": _numeric_summary(
                        [run["training"].get("success_rate") for run in runs]
                    ),
                    "behavior_windows": _aligned_behavior(runs),
                },
                "optimization": {
                    "windows": _aligned_optimization(runs),
                },
                "evaluation": {
                    "pooled_successes": evaluation_successes,
                    "pooled_cases": evaluation_cases,
                    "pooled_success_rate": (
                        None if evaluation_cases == 0 else evaluation_successes / evaluation_cases
                    ),
                    "mean_per_run_success_rate": _numeric_summary(
                        [run["evaluation"].get("success_rate") for run in runs]
                    ),
                    "mean_successful_episode_length": (
                        None if not successful_lengths else fmean(successful_lengths)
                    ),
                    "successful_length_contributing_cases": len(successful_lengths),
                },
                "initial_to_final": {
                    "success_rate": _paired_delta(runs, ("training", "behavior_windows"), "success_rate"),
                    "episode_length": _paired_delta(runs, ("training", "behavior_windows"), "mean_episode_length"),
                    "illegal_action_rate": _behavior_illegal_delta(runs),
                    "loss": _paired_delta(runs, ("optimization", "windows"), "mean_loss"),
                },
            }
        )
    return {
        "attempted_or_discovered_runs": attempted_runs if attempted_runs is not None else len(summaries) + len(malformed),
        "readable_runs": len(summaries),
        "protocol_complete_runs": sum(bool(run.get("protocol_complete")) for run in summaries),
        "checker_passing_runs": sum(bool(run.get("checker_passed")) for run in summaries),
        "protocol_groups": groups,
        "protocol_incompatible": incompatible,
    }


def _behavior_illegal_delta(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas: list[float] = []
    for run in runs:
        rows = run["training"]["behavior_windows"]
        if not rows:
            continue
        rates: list[float | None] = []
        for row in (rows[0], rows[-1]):
            native = int(row.get("native_actions", 0))
            rates.append(None if native == 0 else int(row.get("illegal_actions", 0)) / native)
        if rates[0] is not None and rates[1] is not None:
            deltas.append(float(rates[1]) - float(rates[0]))
    return {
        "mean": None if not deltas else fmean(deltas),
        "contributing_runs": len(deltas),
    }


def aggregate_run_directory(
    batch_root: Path,
    *,
    attempted_runs: int | None = None,
) -> dict[str, Any]:
    """Read all run summaries, write the batch report, and return it."""

    summaries: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for path in sorted(batch_root.glob(f"*/{RUN_SUMMARY_FILENAME}")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("summary root is not an object")
            summaries.append(value)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            malformed.append({"path": str(path), "reason": str(exc)})
    report = aggregate_summaries(
        summaries,
        attempted_runs=attempted_runs,
        malformed=malformed,
    )
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / BATCH_SUMMARY_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if len(report["protocol_groups"]) > 1:
        print("WARNING: multiple incompatible 2-2-BW protocol groups were reported separately.")
    for index, group in enumerate(report["protocol_groups"], start=1):
        print(
            f"2-2-BW group {index}: runs={group['readable_runs']}, "
            f"checker passes={group['checker_passing_runs']}, "
            f"training pooled success={group['training']['pooled_success_rate']}, "
            f"evaluation pooled success={group['evaluation']['pooled_success_rate']}."
        )
    return report


def process_execution_metrics(
    batch_root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build execution-level learning metrics from canonical run summaries."""

    batch_root = Path(batch_root).resolve()
    indexed: dict[int, Path] = {}
    for path in sorted(batch_root.glob(f"*/{RUN_SUMMARY_FILENAME}")):
        try:
            run_id = int(path.parent.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        indexed[run_id] = path
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in sorted(indexed)]
    report = new_report("2-2-bw", None if expected_executions is None else specs)
    behavior_rows: list[dict[str, Any]] = []
    training_episode_rows: list[dict[str, Any]] = []
    optimization_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for spec in specs:
        located = indexed.get(int(spec["run_id"]))
        if located is None:
            report["executions"].append(execution_envelope(
                spec, present=False, readable=False, operationally_valid=False,
                readability_reasons=("run_missing",), operational_reasons=("run_missing",),
            ))
            continue
        path = located
        summary, readability_reason = read_json_object(path)
        if summary is None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(readability_reason or "invalid_json",),
                operational_reasons=(readability_reason or "invalid_json",),
                source_refs=(str(path.relative_to(batch_root)).replace("\\", "/"),),
            ))
            continue
        shape_error = _summary_shape_error(summary)
        if shape_error is not None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(shape_error,), operational_reasons=(shape_error,),
                source_refs=(str(path.relative_to(batch_root)).replace("\\", "/"),),
            ))
            continue
        protocol_error = _protocol_error(summary)
        identity_valid = summary.get("run") == spec["run_id"]
        requested_protocol = spec.get("factors", {}).get("protocol")
        if requested_protocol is not None:
            identity_valid = identity_valid and summary.get("protocol") == requested_protocol
        operational = (protocol_error is None and identity_valid
                       and summary.get("protocol_complete") is True)
        execution_id = str(spec["execution_id"])
        if protocol_error is not None or not identity_valid:
            reasons = ([] if protocol_error is None else ["unsupported_schema"])
            if not identity_valid:
                reasons.append("identity_mismatch")
            report["executions"].append(execution_envelope(
                spec, present=True, readable=True, operationally_valid=False,
                operational_reasons=reasons,
                certificate_status=(
                    "unavailable" if protocol_error is not None
                    else nullable_certificate_status(summary.get("checker_passed"))
                ),
                identity={"protocol": summary.get("protocol")},
                source_refs=(str(path.relative_to(batch_root)).replace("\\", "/"),),
            ))
            continue
        for index, row in enumerate(summary.get("training", {}).get("behavior_windows", [])):
            behavior_rows.append({"execution_id": execution_id, "run_id": spec["run_id"],
                                  "window_id": index, **dict(row)})
        for index, row in enumerate(summary.get("optimization", {}).get("windows", [])):
            optimization_rows.append({"execution_id": execution_id, "run_id": spec["run_id"],
                                      "window_id": index, **dict(row)})
        for index, row in enumerate(summary.get("evaluation", {}).get("cases", [])):
            evaluation_rows.append({"execution_id": execution_id, "run_id": spec["run_id"],
                                    "evaluation_case_id": index,
                                    "analysis_eligible": True,
                                    "eligibility_reasons": [],
                                    **dict(row)})
        eval_data = summary.get("evaluation", {})
        eval_cases = list(eval_data.get("cases", []))
        successes = int(eval_data.get("successes", 0))
        training = summary["training"]
        training_successes = int(training.get("successes", 0))
        ordinary_episodes = int(training.get("ordinary_episodes", 0))
        training_episode_rows.append({
            "execution_id": execution_id,
            "run_id": spec["run_id"],
            "successes": training_successes,
            "truncations": int(training.get("truncations", 0)),
            "budget_cutoffs": int(training.get("budget_cutoffs", 0)),
            "ordinary_episodes": ordinary_episodes,
            "success_rate": training.get("success_rate"),
            "success_wilson_95": wilson_interval(
                training_successes, ordinary_episodes
            ),
        })
        counts = summary["counts"]
        run_rows.append({
            "execution_id": execution_id, "run_id": spec["run_id"],
            "protocol_complete": summary.get("protocol_complete"),
            "training_transitions": counts.get("training_transitions"),
            "learner_updates": counts.get("learner_updates"),
            "target_syncs": counts.get("target_syncs"),
            "training_seconds": counts.get("training_seconds"),
            "optimization_seconds": counts.get("optimization_seconds"),
            "transition_throughput": counts.get("transition_throughput"),
            "update_throughput": counts.get("update_throughput"),
            "training_success_rate": training.get("success_rate"),
            "evaluation_successes": successes, "evaluation_cases": len(eval_cases),
            "evaluation_success_rate": eval_data.get("success_rate"),
            "evaluation_success_interval": wilson_interval(successes, len(eval_cases)),
            "successful_evaluations_per_1000_training_transitions": (
                None if not counts.get("training_transitions")
                else successes / (counts["training_transitions"] / 1_000)
            ),
            "mean_successful_episode_length": eval_data.get(
                "mean_successful_episode_length"
            ),
            "models": dict(summary["models"]),
            "isolation": dict(summary["isolation"]),
            "checkpoint": dict(summary["checkpoint"]),
        })
        task_success = successes > 0
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=[] if operational else ["incomplete_protocol"],
            certificate_status=nullable_certificate_status(
                summary.get("checker_passed")
            ),
            task_status="success" if task_success else "failure",
            task_reason=None if task_success else "no_evaluation_success",
            termination_reason="task_completed" if operational else "operational_failure",
            identity={"protocol": summary.get("protocol"),
                      "treatment": summary.get("treatment")},
            metric_availability={"training_windows": "existing", "optimization_windows": "existing",
                                 "evaluation_cases": "existing"},
            source_refs=(str(path.relative_to(batch_root)).replace("\\", "/"),),
        ))
    eligibility = analysis_eligibility(report["executions"])
    eligible_ids = set(eligibility["eligible_execution_ids"])
    eligible_case_ids = [
        f"{row['execution_id']}/case-{row['evaluation_case_id']}"
        for row in evaluation_rows
        if row["execution_id"] in eligible_ids and row["analysis_eligible"]
    ]
    evaluation_eligibility = {
        **eligibility,
        "eligible_case_ids": eligible_case_ids,
        "excluded_cases": [
            {
                "case_id": (
                    f"{row['execution_id']}/case-"
                    f"{row['evaluation_case_id']}"
                ),
                "reasons": row.get("eligibility_reasons")
                or ["case_analysis_ineligible"],
            }
            for row in evaluation_rows
            if row["execution_id"] in eligible_ids
            and not row["analysis_eligible"]
        ],
    }
    eligible_training = [row for row in training_episode_rows
                         if row["execution_id"] in eligible_ids]
    eligible_runs = [row for row in run_rows if row["execution_id"] in eligible_ids]
    pooled_training_successes = sum(row["successes"] for row in eligible_training)
    pooled_training_episodes = sum(row["ordinary_episodes"] for row in eligible_training)
    report["analyses"] = {
        "training_episodes": {
            "unit": "training_episode_count",
            "nesting": "episode_counts_within_execution",
            "eligibility": eligibility,
            "rows": training_episode_rows,
            "summary": {
                "eligible_executions": len(eligible_training),
                "per_execution_success_rate": summarize_numbers(
                    [row["success_rate"] for row in eligible_training],
                    include_iqr=True,
                ),
                "pooled_descriptive": {
                    "successes": pooled_training_successes,
                    "ordinary_episodes": pooled_training_episodes,
                    "success_rate": (
                        None if pooled_training_episodes == 0
                        else pooled_training_successes / pooled_training_episodes
                    ),
                },
            },
        },
        "behavior_windows": {"unit": "training_window", "nesting": "windows_within_execution",
                             "eligibility": eligibility, "rows": behavior_rows,
                             "summary": {"count": len(eligible_analysis_rows(behavior_rows, eligibility)),
                                         "raw_descriptive_count": len(behavior_rows)}},
        "optimization_windows": {
            "unit": "optimization_window",
            "nesting": "windows_within_execution",
            "eligibility": eligibility,
            "rows": optimization_rows,
            "summary": {
                "count": len(eligible_analysis_rows(optimization_rows, eligibility)),
                "raw_descriptive_count": len(optimization_rows),
                "mean_loss": summarize_numbers(
                    [row.get("mean_loss") for row in optimization_rows
                     if row["execution_id"] in eligible_ids],
                    include_iqr=True,
                ),
                "final_loss": summarize_numbers(
                    [row.get("final_loss") for row in optimization_rows
                     if row["execution_id"] in eligible_ids],
                    include_iqr=True,
                ),
            },
        },
        "evaluation_cases": {
            "unit": "evaluation_case",
            "nesting": "cases_within_execution",
            "eligibility": evaluation_eligibility,
            "rows": evaluation_rows,
            "summary": {
                "count": len(eligible_case_ids),
                "raw_descriptive_count": len(evaluation_rows),
                "eligible_cases": sum(
                    row["execution_id"] in eligible_ids
                    and row["analysis_eligible"] for row in evaluation_rows
                ),
                "successes": sum(
                    row["execution_id"] in eligible_ids
                    and row["analysis_eligible"]
                    and row.get("success") is True for row in evaluation_rows
                ),
            },
        },
        "run_learning": {
            "unit": "execution", "nesting": "none",
            "eligibility": eligibility,
            "rows": run_rows,
            "summary": {
                "evaluation_success_rate": summarize_numbers(
                    [row["evaluation_success_rate"] for row in eligible_runs],
                    include_iqr=True,
                ),
                "training_seconds": summarize_numbers(
                    [row["training_seconds"] for row in eligible_runs], include_iqr=True,
                ),
                "optimization_seconds": summarize_numbers(
                    [row["optimization_seconds"] for row in eligible_runs], include_iqr=True,
                ),
                "transition_throughput": summarize_numbers(
                    [row["transition_throughput"] for row in eligible_runs], include_iqr=True,
                ),
                "update_throughput": summarize_numbers(
                    [row["update_throughput"] for row in eligible_runs], include_iqr=True,
                ),
            },
        },
    }
    write_execution_metrics(batch_root, report)
    return report
