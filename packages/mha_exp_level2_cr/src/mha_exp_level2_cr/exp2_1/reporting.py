"""Execution reporting for the 2-1-CR reactive controller."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from statistics import mean
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

STALL_ACTION_THRESHOLD = 5


def _achievement_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize run-level attainment without counting repeated events twice."""
    from .runner import DIAMOND_PATH_ACHIEVEMENTS

    count = len(rows)
    attained = Counter(name for row in rows for name in set(row["obtained_achievements"]))
    events = Counter()
    for row in rows:
        events.update(row["achievement_event_histogram"])
    names = sorted(set(DIAMOND_PATH_ACHIEVEMENTS) | attained.keys())
    highest = Counter(row["highest_milestone"] or "none" for row in rows)
    distinct_counts = [len(row["obtained_achievements"]) for row in rows]
    successes = sum(row["target_achieved"] for row in rows)
    return {
        "runs": count,
        "successes": successes,
        "failures": count - successes,
        "success_rate": successes / count if count else None,
        "termination_counts": dict(Counter(row["termination_reason"] for row in rows)),
        "distinct_achievements_per_run": {
            **summarize_numbers(distinct_counts, include_iqr=True),
            "mean": mean(distinct_counts) if count else None,
        },
        "achievement_events_per_run": summarize_numbers(
            [sum(row["achievement_event_histogram"].values()) for row in rows],
            include_iqr=True,
        ),
        "execution_seconds": summarize_numbers(
            [row.get("execution_seconds") for row in rows], include_iqr=True,
        ),
        "native_actions_per_run": summarize_numbers(
            [sum(row["action_histogram"].values()) for row in rows], include_iqr=True,
        ),
        "achievement_attainment": {
            name: {"runs": attained[name], "rate": attained[name] / count if count else None,
                   "total_events": events[name]}
            for name in names
        },
        "highest_diamond_path_achievement": {
            name: {"runs": highest[name], "rate": highest[name] / count if count else None}
            for name in ("none", *DIAMOND_PATH_ACHIEVEMENTS)
        },
    }


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


def _episode_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive episode-local movement, need, milestone, and stall evidence."""

    native = [row for row in rows if row.get("event_kind") == "native_action"]
    cells = {tuple(row.get("player_pos", ())) for row in native if len(row.get("player_pos", ())) == 2}
    position_changes = sum(
        first.get("player_pos") != second.get("player_pos")
        for first, second in pairwise(native)
    )
    needs = {key: [row.get("inventory", {}).get(key) for row in native]
             for key in ("health", "food", "drink", "energy")}
    stall_windows: list[dict[str, Any]] = []
    active_stall: dict[str, Any] | None = None
    streak = 0
    previous: dict[str, Any] | None = None
    for action_index, row in enumerate(native):
        unchanged = previous is not None and (
            row.get("player_pos") == previous.get("player_pos")
            and row.get("action") == previous.get("action")
            and row.get("reason") == previous.get("reason")
            and row.get("inventory") == previous.get("inventory")
            and not row.get("newly_achieved")
        )
        streak = streak + 1 if unchanged else 1
        if streak == STALL_ACTION_THRESHOLD:
            start = native[action_index - STALL_ACTION_THRESHOLD + 1]
            active_stall = {
                "start_action_index": action_index - STALL_ACTION_THRESHOLD + 1,
                "end_action_index": action_index,
                "start_event_id": start.get("event_id"),
                "end_event_id": row.get("event_id"),
                "start_elapsed_seconds": start.get("elapsed_seconds"),
                "end_elapsed_seconds": row.get("elapsed_seconds"),
                "actions_observed": STALL_ACTION_THRESHOLD,
            }
        elif streak > STALL_ACTION_THRESHOLD and active_stall is not None:
            active_stall.update({
                "end_action_index": action_index,
                "end_event_id": row.get("event_id"),
                "end_elapsed_seconds": row.get("elapsed_seconds"),
                "actions_observed": streak,
            })
        elif not unchanged and active_stall is not None:
            stall_windows.append(active_stall)
            active_stall = None
        previous = row
    if active_stall is not None:
        active_stall["open_at_episode_end"] = True
        stall_windows.append(active_stall)
    milestones = {}
    for row in native:
        for name in row.get("newly_achieved", []):
            milestones.setdefault(name, {
                "first_action_event_id": row.get("event_id"),
                "first_elapsed_seconds": row.get("elapsed_seconds"),
            })
    return {
        "native_actions": len(native), "unique_visited_cells": len(cells),
        "position_changes": position_changes,
        "minimum_needs": {key: min((value for value in values if isinstance(value, (int, float))), default=None)
                          for key, values in needs.items()},
        "need_trajectories": needs, "milestones": milestones,
        "stall_definition": {"minimum_consecutive_actions": STALL_ACTION_THRESHOLD,
                             "same_position_action_reason": True,
                             "no_inventory_or_achievement_change": True},
        "stall_count": len(stall_windows),
    }, stall_windows


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join correlated reasoner/environment traces and derive episode metrics."""

    root = Path(root).resolve()
    discovered = sorted(int(path.name.rsplit("_", 1)[-1]) for path in root.glob("exp_agent2_1_*")
                        if path.name.rsplit("_", 1)[-1].isdigit())
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in discovered]
    report = new_report("2-1-cr", None if expected_executions is None else specs)
    action_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    milestone_rows: list[dict[str, Any]] = []
    stall_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_id = int(spec["run_id"])
        agent_out = root / f"exp_agent2_1_{run_id}" / "out"
        env_out = root / f"exp_env2_1_{run_id}" / "out"
        states = _states(agent_out) if agent_out.is_dir() else {}
        reasoner = states.get("llreasoner_0")
        env_states = _states(env_out) if env_out.is_dir() else {}
        environment = next(iter(env_states.values()), {})
        trace = environment.get("execution_trace") if isinstance(environment, Mapping) else None
        decisions = reasoner.get("decision_trace") if isinstance(reasoner, Mapping) else None
        if not isinstance(trace, list) or not isinstance(decisions, list):
            present = agent_out.is_dir() or env_out.is_dir()
            report["executions"].append(execution_envelope(
                spec, present=present, readable=False, operationally_valid=False,
                readability_reasons=(
                    "required_state_missing" if present else "run_missing",
                ),
                operational_reasons=(
                    "unsupported_schema" if present else "run_missing",
                ),
            ))
            continue
        if not all(
            isinstance(row, Mapping)
            and type(row.get("request_id")) is int
            and type(row.get("episode_id")) is int
            for row in decisions
        ):
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_trace_row",),
                operational_reasons=("invalid_trace_row",),
            ))
            continue
        decision_by_request = {row.get("request_id"): row for row in decisions
                               if type(row.get("request_id")) is int}
        joined: list[dict[str, Any]] = []
        correlation_valid = True
        trace_shape_valid = True
        for value in trace:
            if not isinstance(value, Mapping):
                trace_shape_valid = False
                continue
            row = dict(value)
            if row.get("event_kind") == "native_action" and (
                not isinstance(row.get("player_pos"), list)
                or len(row["player_pos"]) != 2
                or not isinstance(row.get("inventory"), Mapping)
                or not isinstance(row.get("newly_achieved"), list)
            ):
                trace_shape_valid = False
            request_id = row.get("request_id")
            decision = decision_by_request.get(request_id) if type(request_id) is int else None
            if row.get("event_kind") in {"native_action", "reset"}:
                if not isinstance(decision, Mapping) or decision.get("episode_id") != row.get("episode_id"):
                    correlation_valid = False
                else:
                    row["reason"] = decision.get("reason")
            row.update({"execution_id": spec["execution_id"], "run_id": run_id})
            joined.append(row)
        env_source = next(env_out.glob("*.json"))
        if not trace_shape_valid:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=("invalid_trace_row",),
                operational_reasons=("invalid_trace_row",),
                source_refs=(str(env_source.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        if not correlation_valid:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=True, operationally_valid=False,
                operational_reasons=("identity_mismatch",),
                source_refs=(str(env_source.relative_to(root)).replace("\\", "/"),),
            ))
            continue
        action_rows.extend(
            row for row in joined if row.get("event_kind") == "native_action"
        )
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in joined:
            if type(row.get("episode_id")) is int:
                grouped[int(row["episode_id"])].append(row)
        for episode_id, rows in sorted(grouped.items()):
            metrics, stalls = _episode_metrics(rows)
            reset = next((row for row in rows if row.get("event_kind") == "reset"), None)
            terminal = any(row.get("environment_done") is True for row in rows)
            death = any(row.get("environment_done") is True
                        and row.get("inventory", {}).get("health") == 0 for row in rows)
            treatment_ended = (
                reasoner.get("phase") == "complete"
                and reasoner.get("episode_id") == episode_id
                and reasoner.get("terminal_reason") in {
                    "target_achieved", "action_budget_exhausted", "death", "environment_terminal",
                }
            )
            closed = reset is not None or terminal or treatment_ended
            episode_rows.append({
                "execution_id": spec["execution_id"], "run_id": run_id,
                "episode_id": episode_id, **metrics,
                "reset_cause": None if reset is None else reset.get("reset_cause"),
                "environment_terminal": terminal, "death": death,
                "censoring": {"status": "observed" if closed else "right_censored",
                              "reason": None if closed else "external_execution_end"},
            })
            native = [row for row in rows if row.get("event_kind") == "native_action"]
            milestones = metrics["milestones"]
            from .runner import DIAMOND_PATH_ACHIEVEMENTS
            for milestone in DIAMOND_PATH_ACHIEVEMENTS:
                attained = milestones.get(milestone)
                action_count = None
                if attained is not None:
                    action_count = next(
                        (index for index, row in enumerate(native, 1)
                         if milestone in row.get("newly_achieved", [])),
                        None,
                    )
                milestone_rows.append({
                    "execution_id": spec["execution_id"], "run_id": run_id,
                    "episode_id": episode_id, "milestone": milestone,
                    "attained": attained is not None,
                    "first_attainment_action": action_count,
                    "first_action_event_id": (
                        None if attained is None else attained["first_action_event_id"]
                    ),
                    "first_elapsed_seconds": (
                        None if attained is None else attained["first_elapsed_seconds"]
                    ),
                    "censor_at_action_count": len(native),
                    "censor_at_elapsed_seconds": (
                        native[-1].get("elapsed_seconds") if native else None
                    ),
                    "censoring": {
                        "status": "observed" if attained is not None else "right_censored",
                        "reason": (
                            None if attained is not None
                            else "episode_ended_before_attainment" if closed
                            else "external_execution_end"
                        ),
                    },
                })
            for stall_index, stall in enumerate(stalls):
                open_at_end = stall.pop("open_at_episode_end", False)
                stall_rows.append({
                    "execution_id": spec["execution_id"], "run_id": run_id,
                    "episode_id": episode_id,
                    "stall_id": f"episode-{episode_id}-stall-{stall_index}",
                    **stall,
                    "censoring": {
                        "status": (
                            "right_censored" if open_at_end and not closed else "observed"
                        ),
                        "reason": (
                            "external_execution_end" if open_at_end and not closed else None
                        ),
                    },
                })
        histogram = Counter(str(row.get("action")) for row in action_rows
                            if row["execution_id"] == spec["execution_id"])
        reason_histogram = Counter(str(row.get("reason")) for row in action_rows
                                   if row["execution_id"] == spec["execution_id"])
        obtained = sorted({name for row in joined if row.get("event_kind") == "native_action"
                           for name in row.get("newly_achieved", [])})
        achievement_events = Counter(
            name for row in joined if row.get("event_kind") == "native_action"
            for name in row.get("newly_achieved", [])
        )
        termination_reason = reasoner.get("terminal_reason") or (
            "environment_terminal" if any(row.get("environment_done") for row in joined)
            else "external_stop"
        )
        run_rows.append({
            "execution_id": spec["execution_id"], "run_id": run_id,
            "highest_milestone": environment.get("highest_diamond_path_achievement"),
            "target_achieved": reasoner.get("target_achieved") is True,
            "obtained_achievements": obtained,
            "achievement_event_histogram": dict(sorted(achievement_events.items())),
            "termination_reason": termination_reason,
            "execution_seconds": reasoner.get("execution_seconds"),
            "action_histogram": dict(sorted(histogram.items())),
            "controller_reason_histogram": dict(sorted(reason_histogram.items())),
            "empirical_action_diversity_bits": empirical_action_diversity(histogram),
            "stall_count": sum(row["stall_count"] for row in episode_rows
                               if row["execution_id"] == spec["execution_id"]),
            "stall_denominator_actions": sum(row["native_actions"] for row in episode_rows
                                             if row["execution_id"] == spec["execution_id"]),
        })
        agent_log = root / f"exp_agent2_1_{run_id}.log"
        env_log = root / f"exp_env2_1_{run_id}.log"
        agent_logs = agent_log.read_text(encoding="utf-8", errors="replace").splitlines() if agent_log.is_file() else None
        environment_logs = env_log.read_text(encoding="utf-8", errors="replace").splitlines() if env_log.is_file() else None
        from .runner import check_results
        required_checker_states = {"perceptor_0", "actuator_0", "llreasoner_0"}
        checker_available = (
            required_checker_states <= states.keys()
            and bool(env_states)
            and bool(agent_logs)
            and bool(environment_logs)
        )
        certificate = (
            check_results(states, environment, agent_logs, environment_logs, verbose=False)
            if checker_available else None
        )
        task_success = reasoner.get("target_achieved") is True
        operational_reasons = []
        operational_reasons.extend(treatment_identity_reasons(
            spec, reasoner,
            keys=("protocol_version", "treatment_id", "treatment_digest",
                  "target_achievement", "action_budget"),
        ))
        if not required_checker_states <= states.keys() or not env_states:
            operational_reasons.append("required_state_missing")
        if not correlation_valid:
            operational_reasons.append("identity_mismatch")
        if not agent_logs or not environment_logs:
            operational_reasons.append("required_log_missing")
        if any("traceback" in line.lower()
               for line in (agent_logs or []) + (environment_logs or [])):
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
            task_reason=None if task_success else str(
                reasoner.get("terminal_reason") or "target_not_achieved"
            ),
            termination_reason=termination_reason,
            identity={"treatment": {key: reasoner.get(key) for key in (
                "protocol_version", "treatment_id", "treatment_digest", "target_achievement", "action_budget"
            )}},
            metric_availability={"execution_trace": "instrumented", "milestones": "derived",
                                 "movement": "derived", "needs": "derived", "stalls": "derived",
                                 "certificate": (
                                     "existing" if checker_available
                                     else {"status": "unavailable", "reason": "checker_input_missing"}
                                 )},
            source_refs=(str(env_source.relative_to(root)).replace("\\", "/"),),
        ))
    eligibility = analysis_eligibility(report["executions"])
    eligible_actions = eligible_analysis_rows(action_rows, eligibility)
    eligible_episodes = eligible_analysis_rows(episode_rows, eligibility)
    eligible_milestones = eligible_analysis_rows(milestone_rows, eligibility)
    eligible_stalls = eligible_analysis_rows(stall_rows, eligibility)
    eligible_runs = eligible_analysis_rows(run_rows, eligibility)
    report["analyses"] = {
        "native_actions": {"unit": "native_action", "nesting": "actions_within_episode_within_execution",
                           "eligibility": eligibility, "rows": action_rows,
                           "summary": {"count": len(eligible_actions),
                                       "raw_descriptive_count": len(action_rows)}},
        "episodes": {"unit": "episode", "nesting": "episodes_within_execution",
                     "eligibility": eligibility, "rows": episode_rows,
                     "summary": {"count": len(eligible_episodes),
                                 "raw_descriptive_count": len(episode_rows),
                                 "unique_visited_cells": summarize_numbers(
                         [row["unique_visited_cells"] for row in eligible_episodes], include_iqr=True)}},
        "milestones": {
            "unit": "milestone_time_to_event",
            "nesting": "milestones_within_episode_within_execution",
            "eligibility": eligibility,
            "rows": milestone_rows,
            "summary": {
                "count": len(eligible_milestones),
                "raw_descriptive_count": len(milestone_rows),
                "attained": sum(row["attained"] for row in eligible_milestones),
                "right_censored": sum(
                    row["censoring"]["status"] == "right_censored"
                    for row in eligible_milestones
                ),
            },
        },
        "stall_windows": {
            "unit": "stall_window",
            "nesting": "stall_windows_within_episode_within_execution",
            "eligibility": eligibility,
            "rows": stall_rows,
            "summary": {
                "count": len(eligible_stalls),
                "raw_descriptive_count": len(stall_rows),
                "minimum_consecutive_actions": STALL_ACTION_THRESHOLD,
            },
        },
        "controller_runs": {"unit": "execution", "nesting": "none",
                            "eligibility": eligibility, "rows": run_rows,
                            "summary": {**_achievement_summary(eligible_runs),
                                "stall_count": summarize_numbers(
                                [row["stall_count"] for row in eligible_runs], include_iqr=True)}},
    }
    write_execution_metrics(root, report)
    return report
