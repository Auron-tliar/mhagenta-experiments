"""Execution reporting for full-domain hierarchical 2-4-CR runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
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


PROVENANCE_FILENAME = "execution-provenance.json"
PROVENANCE_PROTOCOL_VERSION = "2-4-cr-execution-provenance-v1"
PROVENANCE_FIELDS = {
    "protocol_version",
    "workspace_git_commit",
    "workspace_git_dirty",
    "mhagenta_git_commit",
    "mhagenta_git_dirty",
    "mhagenta_version",
    "experiment_source_sha256",
    "crafter_source_sha256",
    "mhagenta_source_sha256",
    "uv_lock_sha256",
    "provenance_id",
}
TERMINAL_ACTIVITY_STATUSES = {"succeeded", "failed", "interrupted"}
ATOMIC_FIELDS = {
    "action_id",
    "action",
    "movement_kind",
    "source_cell",
    "destination_cell",
    "dispatch_revision",
    "legal",
    "confirmation_revision",
}
TREATMENT_IDENTITY_KEYS = (
    "protocol_version",
    "treatment_id",
    "manifest_digest",
    "task_id",
    "seed",
    "stratum",
    "primary_intention",
    "no_mobs",
    "daylight_effects",
    "duration",
    "total_action_cap",
)
MILESTONE_ACTIVITIES = {
    "table": "place_table",
    "wood_pickaxe": "make_wood_pickaxe",
    "stone_pickaxe": "make_stone_pickaxe",
    "furnace": "place_furnace",
    "iron_pickaxe": "make_iron_pickaxe",
    "diamond": "get_diamond",
}


def _is_hex(value: Any, length: int) -> bool:
    """Return whether a value is a fixed-width hexadecimal digest."""

    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


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


def _read_provenance(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read and validate the exact cohort execution provenance manifest."""

    path = root / PROVENANCE_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None, "required_provenance_missing"
    except (UnicodeError, json.JSONDecodeError):
        return None, "required_provenance_malformed"
    if not isinstance(value, dict) or set(value) != PROVENANCE_FIELDS:
        return None, "required_provenance_malformed"
    digest_fields = (
        "experiment_source_sha256",
        "crafter_source_sha256",
        "mhagenta_source_sha256",
        "uv_lock_sha256",
    )
    if (
        value["protocol_version"] != PROVENANCE_PROTOCOL_VERSION
        or value["mhagenta_version"] != "1.4.12"
        or type(value["workspace_git_dirty"]) is not bool
        or type(value["mhagenta_git_dirty"]) is not bool
        or not all(_is_hex(value[key], 40) for key in ("workspace_git_commit", "mhagenta_git_commit"))
        or not all(_is_hex(value[key], 64) for key in (*digest_fields, "provenance_id"))
    ):
        return None, "required_provenance_malformed"
    payload = {key: item for key, item in value.items() if key != "provenance_id"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != value["provenance_id"]:
        return None, "required_provenance_malformed"
    return value, None


def _hierarchy_shape_error(value: Any) -> str | None:
    """Validate the retained intention, activity, atomic, and map-boundary data."""

    if not isinstance(value, Mapping) or not isinstance(value.get("intention"), Mapping):
        return "invalid_root_type"
    activities = value.get("activities")
    if not isinstance(activities, list) or not all(isinstance(row, Mapping) for row in activities):
        return "invalid_trace_row"
    for activity in activities:
        atomic = activity.get("atomic")
        status = activity.get("status", "open")
        start = activity.get("known_cell_count_start")
        end = activity.get("known_cell_count_end")
        if (
            not isinstance(atomic, list)
            or not all(isinstance(row, Mapping) for row in atomic)
            or not isinstance(status, str)
            or status not in {*TERMINAL_ACTIVITY_STATUSES, "active"}
            or type(start) is not int
            or start < 0
            or (
                status in TERMINAL_ACTIVITY_STATUSES
                and (type(end) is not int or end < start)
            )
            or (status not in TERMINAL_ACTIVITY_STATUSES and end is not None)
        ):
            return "invalid_trace_row"
        for row in atomic:
            source = row.get("source_cell")
            destination = row.get("destination_cell")
            movement = row.get("movement_kind")
            if (
                set(row) != ATOMIC_FIELDS
                or not isinstance(row.get("action_id"), str)
                or not row["action_id"]
                or type(row.get("action")) is not int
                or not isinstance(movement, str)
                or movement not in {"walk", "turn", "none"}
                or not _valid_coordinate(source)
                or not _valid_coordinate(destination)
                or type(row.get("dispatch_revision")) is not int
                or type(row.get("legal")) is not bool
                or type(row.get("confirmation_revision")) is not int
            ):
                return "invalid_trace_row"
    return None


def _valid_coordinate(value: Any) -> bool:
    """Return whether a saved atomic coordinate has the current JSON shape."""

    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(coordinate) is int for coordinate in value)
    )


def _metric_state_shape_error(
    high: Mapping[str, Any], environment: Mapping[str, Any]
) -> str | None:
    """Validate saved scalar values used by report derivations."""

    final = high.get("abstract_state")
    if not isinstance(final, Mapping):
        return "invalid_metric_value"
    inventory = final.get("inventory")
    if (
        not isinstance(inventory, Mapping)
        or any(
            type(inventory.get(name)) is not int or inventory[name] < 0
            for name in ("diamond", "health", "food", "drink", "energy")
        )
        or type(final.get("known_cell_count")) is not int
        or final["known_cell_count"] < 0
        or type(high.get("need_interruptions")) is not int
        or high["need_interruptions"] < 0
        or type(high.get("recoveries")) is not int
        or high["recoveries"] < 0
    ):
        return "invalid_metric_value"
    if not environment:
        return None
    achievements = environment.get("achievement_counts")
    if (
        type(environment.get("native_actions")) is not int
        or environment["native_actions"] < 0
        or not isinstance(achievements, Mapping)
        or any(
            not isinstance(name, str) or type(count) is not int or count < 0
            for name, count in achievements.items()
        )
    ):
        return "invalid_metric_value"
    return None


def _treatment_identity_errors(
    spec: Mapping[str, Any], actual: Any
) -> list[str]:
    """Compare every field in the frozen expected treatment identity."""

    return treatment_identity_reasons(
        spec, actual, keys=TREATMENT_IDENTITY_KEYS
    )


def _logs(root: Path, run_id: int) -> dict[str, list[str]]:
    result = {}
    for name, path in {
        "agent": root / f"exp_agent2_4_{run_id}.log",
        "environment": root / f"exp_env2_4_{run_id}.log",
    }.items():
        result[name] = (
            path.read_text(encoding="utf-8", errors="replace").splitlines()
            if path.is_file()
            else []
        )
    return result


def _purpose(row: Mapping[str, Any]) -> str | None:
    desired = row.get("desired")
    if not isinstance(desired, Mapping):
        return None
    arguments = desired.get("arguments")
    if not isinstance(arguments, list) or not arguments:
        return None
    prefix = "target" if desired.get("predicate") == "reachable_target_kind" else "placement"
    return f"{prefix}:{arguments[0]}"


def _rate(
    successes: int, total: int, *, count_name: str = "attained"
) -> dict[str, Any]:
    interval = wilson_interval(successes, total)
    return {
        count_name: successes,
        "denominator": total,
        "fraction": None if total == 0 else successes / total,
        "wilson_95_interval": None if interval is None else list(interval),
    }


def _numeric_summary(values: Sequence[float | int | None]) -> dict[str, Any]:
    """Return the requested compact distribution summary, including its range."""

    summary = summarize_numbers(values, include_iqr=True)
    summary["range"] = (
        None if summary["count"] == 0 else [summary["min"], summary["max"]]
    )
    return summary


def _count_fractions(values: Sequence[Any]) -> dict[str, dict[str, int | float]]:
    counts = Counter(str(value) for value in values)
    total = len(values)
    if not total:
        return {}
    return {
        key: {"count": count, "fraction": count / total}
        for key, count in sorted(counts.items())
    }


def _descriptive_groups(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {
        name: {
            "activity_attempts": len(group),
            "status_counts": dict(sorted(Counter(str(row["status"]) for row in group).items())),
            "atomic_actions": _numeric_summary(
                [row["atomic_action_count"] for row in group]
            ),
        }
        for name, group in sorted(grouped.items())
    }


def _milestone_rows(
    spec: Mapping[str, Any],
    activities: Sequence[Mapping[str, Any]],
    native_actions: int,
    terminal_reason: Any,
) -> list[dict[str, Any]]:
    cumulative = 0
    first: dict[str, tuple[Mapping[str, Any], int]] = {}
    for row in activities:
        cumulative += int(row["atomic_action_count"])
        if row["status"] != "succeeded":
            continue
        for milestone, activity in MILESTONE_ACTIVITIES.items():
            if row["activity"] == activity and milestone not in first:
                first[milestone] = (row, cumulative)
    result = []
    for milestone in MILESTONE_ACTIVITIES:
        evidence = first.get(milestone)
        result.append({
            "execution_id": spec["execution_id"],
            "run_id": int(spec["run_id"]),
            "milestone": milestone,
            "attained": evidence is not None,
            "first_activity_id": None if evidence is None else evidence[0]["activity_id"],
            "completion_revision": None if evidence is None else evidence[0]["belief_revision_end"],
            "native_action_ordinal": None if evidence is None else evidence[1],
            "censoring": {
                "status": "right_censored" if evidence is None else "observed",
                "action": native_actions if evidence is None else None,
                "reason": str(terminal_reason or "execution_ended") if evidence is None else None,
            },
        })
    return result


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build execution, milestone, activity, and atomic cohort evidence."""

    root = Path(root).resolve()
    discovered = sorted(
        int(path.name.rsplit("_", 1)[-1])
        for path in root.glob("exp_agent2_4_*")
        if path.name.rsplit("_", 1)[-1].isdigit()
    )
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [
            {"execution_id": f"run-{run}", "run_id": run, "factors": {}, "expected": False}
            for run in discovered
        ]
    report = new_report("2-4-cr", None if expected_executions is None else specs)
    provenance, provenance_error = _read_provenance(root)
    report["execution_provenance"] = {
        "status": "valid" if provenance is not None else "invalid",
        "reason": provenance_error,
        "source_ref": PROVENANCE_FILENAME,
        "manifest": provenance,
    }
    activity_rows: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    milestone_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    required_states = {
        "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
        "goalgraph_0", "hlreasoner_0",
    }

    for spec in specs:
        run_id = int(spec["run_id"])
        out = root / f"exp_agent2_4_{run_id}" / "out"
        states = _states(out) if out.is_dir() else {}
        high = states.get("hlreasoner_0")
        if not isinstance(high, Mapping):
            reason = "required_state_missing" if out.is_dir() else "run_missing"
            operational = [reason, *([provenance_error] if provenance_error else [])]
            report["executions"].append(execution_envelope(
                spec, present=out.is_dir(), readable=False, operationally_valid=False,
                readability_reasons=(reason,), operational_reasons=operational,
            ))
            continue
        hierarchy = high.get("hierarchy")
        shape_error = _hierarchy_shape_error(hierarchy)
        if shape_error is not None:
            operational = [shape_error, *([provenance_error] if provenance_error else [])]
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(shape_error,), operational_reasons=operational,
            ))
            continue

        env_states = _states(root / f"exp_env2_4_{run_id}" / "out")
        environment = next(iter(env_states.values()), {})
        metric_shape_error = _metric_state_shape_error(high, environment)
        if metric_shape_error is not None:
            operational = [
                metric_shape_error,
                *([provenance_error] if provenance_error else []),
            ]
            report["executions"].append(execution_envelope(
                spec,
                present=True,
                readable=False,
                operationally_valid=False,
                readability_reasons=(metric_shape_error,),
                operational_reasons=operational,
            ))
            continue

        current: list[dict[str, Any]] = []
        native_ordinal = 0
        for index, activity in enumerate(hierarchy["activities"], 1):
            atomic = activity["atomic"]
            status = activity.get("status", "open")
            start = activity["known_cell_count_start"]
            end = activity["known_cell_count_end"]
            row = {
                "execution_id": spec["execution_id"],
                "run_id": run_id,
                "activity_id": activity.get("goal_id", f"activity-{index}"),
                "stage": activity.get("stage"),
                "plan_revision": activity.get("plan_revision"),
                "activity": activity.get("activity"),
                "desired": activity.get("desired"),
                "final_value": activity.get("final_value"),
                "status": status,
                "failure_reason": activity.get("failure_reason"),
                "interruption": activity.get("interruption"),
                "censoring": {
                    "status": "observed" if status in TERMINAL_ACTIVITY_STATUSES else "right_censored",
                    "reason": None if status in TERMINAL_ACTIVITY_STATUSES else "execution_ended_with_open_activity",
                },
                "belief_revision_start": activity.get("based_on_revision"),
                "belief_revision_end": activity.get("completion_revision"),
                "known_cell_count_start": start,
                "known_cell_count_end": end,
                "known_cells_added": None if end is None else end - start,
                "atomic_action_count": len(atomic),
                "legal_atomic_actions": sum(item.get("legal") is True for item in atomic),
                "atomic_confirmation_revisions": [item.get("confirmation_revision") for item in atomic],
            }
            current.append(row)
            activity_rows.append(row)
            for atomic_index, atomic_row in enumerate(atomic, 1):
                native_ordinal += 1
                atomic_rows.append({
                    "execution_id": spec["execution_id"],
                    "run_id": run_id,
                    "activity_id": row["activity_id"],
                    "stage": row["stage"],
                    "activity": row["activity"],
                    "atomic_action_id": f"{row['activity_id']}:atomic-{atomic_index}",
                    "source_index": atomic_index - 1,
                    "native_action_ordinal": native_ordinal,
                    **dict(atomic_row),
                })

        final = high.get("abstract_state") if isinstance(high.get("abstract_state"), Mapping) else {}
        inventory = final.get("inventory") if isinstance(final.get("inventory"), Mapping) else {}
        by_status = {
            status: [row for row in current if row["status"] == status]
            for status in TERMINAL_ACTIVITY_STATUSES
        }
        open_rows = [row for row in current if row["status"] not in TERMINAL_ACTIVITY_STATUSES]
        completed = by_status["succeeded"]
        status_actions = {
            "succeeded": sum(row["atomic_action_count"] for row in completed),
            "interrupted": sum(row["atomic_action_count"] for row in by_status["interrupted"]),
            "failed": sum(row["atomic_action_count"] for row in by_status["failed"]),
            "open": sum(row["atomic_action_count"] for row in open_rows),
        }
        attributed_actions = sum(status_actions.values())
        activity_counts: Counter[str] = Counter()
        activity_actions: Counter[str] = Counter()
        stage_counts: Counter[str] = Counter()
        stage_actions: Counter[str] = Counter()
        activity_stage: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "actions": 0})
        for row in current:
            activity, stage = str(row["activity"]), str(row["stage"])
            actions = int(row["atomic_action_count"])
            activity_counts[activity] += 1
            activity_actions[activity] += actions
            stage_counts[stage] += 1
            stage_actions[stage] += actions
            activity_stage[f"{stage}/{activity}"]["attempts"] += 1
            activity_stage[f"{stage}/{activity}"]["actions"] += actions
        exploration_by_purpose: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {
                "activities": 0,
                "actions": 0,
                "known_cells_added": 0,
                "known_cells_added_missing": 0,
            }
        )
        exploration = [row for row in current if row["activity"] == "explore"]
        for row in exploration:
            purpose = _purpose(row)
            if purpose is None:
                continue
            exploration_by_purpose[purpose]["activities"] += 1
            exploration_by_purpose[purpose]["actions"] += int(row["atomic_action_count"])
            if row["known_cells_added"] is None:
                exploration_by_purpose[purpose]["known_cells_added_missing"] += 1
            else:
                exploration_by_purpose[purpose]["known_cells_added"] += int(row["known_cells_added"])
        interruptions_by_need = Counter(
            str(row["interruption"]["need"])
            for row in by_status["interrupted"]
            if isinstance(row.get("interruption"), Mapping) and "need" in row["interruption"]
        )
        run_atomic = [row for row in atomic_rows if row["execution_id"] == spec["execution_id"]]
        movement_counts = Counter(str(row.get("movement_kind", "none")) for row in run_atomic)
        visited_cells: set[tuple[int, int]] = set()
        for row in run_atomic:
            source = row.get("source_cell")
            destination = row.get("destination_cell")
            if isinstance(source, list) and len(source) == 2:
                visited_cells.add((int(source[0]), int(source[1])))
            if row.get("movement_kind") == "walk" and isinstance(destination, list) and len(destination) == 2:
                visited_cells.add((int(destination[0]), int(destination[1])))
        failure_reasons = Counter(
            str(row["failure_reason"]) for row in by_status["failed"] if row.get("failure_reason")
        )
        native_actions = int(environment.get("native_actions", 0))
        exploration_actions = sum(row["atomic_action_count"] for row in exploration)
        exploration_gain = sum(
            int(row["known_cells_added"])
            for row in exploration
            if row["known_cells_added"] is not None
        )
        exploration_gain_missing = sum(
            row["known_cells_added"] is None for row in exploration
        )
        achievement_value = environment.get("achievement_counts")
        achievement_counts = (
            {str(name): int(count) for name, count in sorted(achievement_value.items())}
            if isinstance(achievement_value, Mapping) else {}
        )
        module_active_seconds = {
            name: state.get("active_seconds")
            for name, state in sorted(states.items()) if name in required_states
        }
        terminal_reason = hierarchy.get("terminal_reason")
        run_rows.append({
            "execution_id": spec["execution_id"],
            "run_id": run_id,
            "outcome": hierarchy.get("status"),
            "terminal_reason": terminal_reason,
            "diamond": int(inventory.get("diamond", 0)),
            "highest_milestone": hierarchy.get("highest_milestone"),
            "final_stage": hierarchy.get("current_stage"),
            "completed_activities": len(completed),
            "interrupted_activities": len(by_status["interrupted"]),
            "failed_activities": len(by_status["failed"]),
            "open_activities": len(open_rows),
            "native_actions": native_actions,
            "attributed_atomic_actions": attributed_actions,
            "succeeded_activity_actions": status_actions["succeeded"],
            "interrupted_activity_actions": status_actions["interrupted"],
            "failed_activity_actions": status_actions["failed"],
            "open_activity_actions": status_actions["open"],
            "atomic_actions_per_completed_activity": {
                "numerator": status_actions["succeeded"],
                "denominator": len(completed),
                "value": None if not completed else status_actions["succeeded"] / len(completed),
            },
            "activity_counts": dict(sorted(activity_counts.items())),
            "activity_actions": dict(sorted(activity_actions.items())),
            "stage_counts": dict(sorted(stage_counts.items())),
            "stage_actions": dict(sorted(stage_actions.items())),
            "activity_stage": dict(sorted(activity_stage.items())),
            "interruptions_by_need": dict(sorted(interruptions_by_need.items())),
            "exploration_by_purpose": dict(sorted(exploration_by_purpose.items())),
            "exploration_activities": len(exploration),
            "exploration_actions": exploration_actions,
            "exploration_known_cells_added": exploration_gain,
            "exploration_known_cells_added_missing": exploration_gain_missing,
            "known_cells_added_per_explore_action": (
                None
                if exploration_actions == 0 or exploration_gain_missing
                else exploration_gain / exploration_actions
            ),
            "final_known_cell_count": final.get("known_cell_count"),
            "unique_visited_cells": len(visited_cells),
            "walk_actions": movement_counts["walk"],
            "turn_actions": movement_counts["turn"],
            "nonmovement_actions": movement_counts["none"],
            "need_interruptions": high.get("need_interruptions"),
            "recoveries": high.get("recoveries"),
            "activity_failure_reasons": dict(sorted(failure_reasons.items())),
            "final_needs": {
                need: inventory.get(need) for need in ("health", "food", "drink", "energy")
            },
            "environment_terminal": environment.get("terminal"),
            "environment_dead": environment.get("dead"),
            "achievement_counts": achievement_counts,
            "module_compute_active_seconds": module_active_seconds,
        })
        milestone_rows.extend(_milestone_rows(spec, current, native_actions, terminal_reason))

        logs = _logs(root, run_id)
        from .runner import RECORD, check_results

        checker_available = set(states) == required_states and bool(env_states) and all(logs.values())
        certificate = (
            check_results(
                dict(states), environment, logs,
                expected_recording=RECORD == "all" or RECORD == "first" and run_id == 0,
                verbose=False,
            )
            if checker_available else None
        )
        task_success = int(inventory.get("diamond", 0)) >= 1 and hierarchy.get("status") == "succeeded"
        operational_reasons = _treatment_identity_errors(
            spec, high.get("treatment")
        )
        if provenance_error:
            operational_reasons.append(provenance_error)
        if set(states) != required_states or not env_states:
            operational_reasons.append("required_state_missing")
        if not all(logs.values()):
            operational_reasons.append("required_log_missing")
        if high.get("failure") is not None or any(
            "traceback" in line.lower() for values in logs.values() for line in values
        ):
            operational_reasons.append("fatal_runtime_error")
        if attributed_actions != native_actions:
            operational_reasons.append("atomic_action_accounting_mismatch")
        if certificate is False:
            operational_reasons.append("certificate_failed")
        source_ref = next(out.glob("*.hlreasoner_0.json"))
        report["executions"].append(execution_envelope(
            spec,
            present=True,
            readable=True,
            operationally_valid=not operational_reasons,
            operational_reasons=operational_reasons,
            certificate_status="unavailable" if certificate is None else "passed" if certificate else "failed",
            task_status="success" if task_success else "failure",
            task_reason=None if task_success else terminal_reason or "diamond_not_obtained",
            termination_reason=terminal_reason,
            identity={
                "treatment": high.get("treatment"),
                "execution_provenance_id": None if provenance is None else provenance["provenance_id"],
            },
            metric_availability={
                "activities": "existing",
                "atomic_actions": "existing",
                "hierarchy_outcome": "existing",
                "known_cell_boundaries": "instrumented",
                "module_compute_active_seconds": "existing",
                "wall_clock_timing": {"status": "unavailable", "reason": "not_instrumented"},
            },
            source_refs=(
                str(source_ref.relative_to(root)).replace("\\", "/"),
                PROVENANCE_FILENAME,
            ),
        ))

    eligibility = analysis_eligibility(report["executions"])
    eligible_activities = eligible_analysis_rows(activity_rows, eligibility)
    eligible_atomic = eligible_analysis_rows(atomic_rows, eligibility)
    eligible_milestones = eligible_analysis_rows(milestone_rows, eligibility)
    eligible_runs = eligible_analysis_rows(run_rows, eligibility)
    denominator = len(eligible_runs)
    diamond_successes = sum(
        row["diamond"] >= 1 and row["outcome"] == "succeeded" for row in eligible_runs
    )
    achievement_names = sorted({
        name for row in eligible_runs for name in row["achievement_counts"]
    })
    milestone_summary = {}
    for milestone in MILESTONE_ACTIVITIES:
        rows = [row for row in eligible_milestones if row["milestone"] == milestone]
        attained = [row for row in rows if row["attained"]]
        milestone_summary[milestone] = {
            **_rate(len(attained), denominator),
            "native_action_ordinal_among_attained": _numeric_summary(
                [row["native_action_ordinal"] for row in attained]
            ),
        }
    numeric_fields = (
        "native_actions", "completed_activities", "interrupted_activities",
        "failed_activities", "open_activities", "exploration_actions",
        "exploration_known_cells_added", "final_known_cell_count",
        "need_interruptions", "recoveries",
    )
    run_summary = {
        "diamond_success": _rate(
            diamond_successes, denominator, count_name="successes"
        ),
        "terminal_outcomes": _count_fractions([row["outcome"] for row in eligible_runs]),
        "terminal_reasons": _count_fractions([row["terminal_reason"] for row in eligible_runs]),
        "milestones": milestone_summary,
        "achievements": {
            name: _rate(
                sum(int(row["achievement_counts"].get(name, 0)) > 0 for row in eligible_runs),
                denominator,
            )
            for name in achievement_names
        },
        "numeric": {
            field: _numeric_summary([row.get(field) for row in eligible_runs])
            for field in numeric_fields
        },
        "atomic_actions_per_completed_activity": _numeric_summary(
            [row["atomic_actions_per_completed_activity"]["value"] for row in eligible_runs]
        ),
        "activities_by_activity": _descriptive_groups(eligible_activities, "activity"),
        "activities_by_stage": _descriptive_groups(eligible_activities, "stage"),
        "activities_by_activity_and_stage": _descriptive_groups(
            [
                {**row, "activity_and_stage": f"{row['stage']}/{row['activity']}"}
                for row in eligible_activities
            ],
            "activity_and_stage",
        ),
        "interruption_totals_by_need": dict(sorted(sum(
            (Counter(row["interruptions_by_need"]) for row in eligible_runs), Counter()
        ).items())),
    }
    report["analyses"] = {
        "activities": {
            "unit": "activity",
            "nesting": "activities_within_execution",
            "eligibility": eligibility,
            "rows": activity_rows,
            "summary": {
                "eligible_descriptive_count": len(eligible_activities),
                "raw_descriptive_count": len(activity_rows),
            },
        },
        "atomic_actions": {
            "unit": "atomic_action",
            "nesting": "atomic_actions_within_activity_within_execution",
            "eligibility": eligibility,
            "rows": atomic_rows,
            "summary": {
                "eligible_descriptive_count": len(eligible_atomic),
                "raw_descriptive_count": len(atomic_rows),
            },
        },
        "milestones": {
            "unit": "milestone_within_execution",
            "nesting": "milestones_within_execution",
            "eligibility": eligibility,
            "rows": milestone_rows,
            "summary": milestone_summary,
        },
        "hierarchy_runs": {
            "unit": "execution",
            "nesting": "none",
            "eligibility": eligibility,
            "rows": run_rows,
            "summary": run_summary,
        },
    }
    write_execution_metrics(root, report)
    return report
