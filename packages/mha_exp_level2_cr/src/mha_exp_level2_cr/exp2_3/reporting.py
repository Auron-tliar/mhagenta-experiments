"""Execution reporting for the full-domain 2-3-CR experiment."""

from __future__ import annotations

import json
import math
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
from mha_exp_common.metrics import summarize_numbers

from .beliefs import DIAMOND_MILESTONES
from .planning import LOW_HEALTH_THRESHOLD, recovery_entry_threshold


RECOVERY_NEEDS = ("food", "drink", "energy")
TRACKED_NEEDS = ("health", *RECOVERY_NEEDS)
ACTION_DECISIONS = frozenset(
    {"planned-action", "recovery-action", "explore-action"}
)
TECHNOLOGY_STAGE_ORDER = (
    "collect-table-wood",
    "place-table",
    "collect-wood-pickaxe-wood",
    "make-wood-pickaxe",
    "collect-stone-pickaxe-wood",
    "collect-stone-pickaxe-stone",
    "make-stone-pickaxe",
    "collect-furnace-stone",
    "place-furnace",
    "collect-iron-pickaxe-wood",
    "collect-iron-pickaxe-coal",
    "collect-iron-pickaxe-iron",
    "make-iron-pickaxe",
    "collect-diamond",
)
TREATMENT_IDENTITY_KEYS = (
    "protocol_version",
    "treatment_id",
    "manifest_digest",
    "task_id",
    "seed",
    "initial_support",
    "action_budget",
    "episode_length",
)
MAP_COVERAGE_FIELDS = (
    "known_terrain",
    "reachable_cells",
    "unknown_frontier",
)


def _json_objects(out: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in out.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values[path.stem.rsplit(".", 1)[-1]] = value
    return values


def _number(value: object) -> int | float | None:
    """Return one finite JSON number, excluding booleans."""

    if type(value) is int:
        try:
            return value if math.isfinite(float(value)) else None
        except OverflowError:
            return None
    if type(value) is float and math.isfinite(value):
        return value
    return None


def _trace_shape_error(value: Any) -> str | None:
    """Validate required trace values used by the reporting derivations."""

    if not isinstance(value, list):
        return "invalid_root_type"
    for row in value:
        if not isinstance(row, Mapping):
            return "invalid_trace_row"
        if _number(row.get("elapsed_seconds")) is None:
            return "invalid_trace_row"
        belief = row.get("belief")
        if not isinstance(belief, Mapping):
            return "invalid_trace_row"
        achievements = belief.get("achievements")
        if not isinstance(achievements, list) or not all(
            isinstance(item, str) for item in achievements
        ):
            return "invalid_trace_row"
        needs = belief.get("needs")
        if not isinstance(needs, Mapping) or any(
            type(needs.get(need)) is not int
            or _number(needs.get(need)) is None
            for need in TRACKED_NEEDS
        ):
            return "invalid_trace_row"
        if any(
            type(belief.get(field)) is not int
            or _number(belief.get(field)) is None
            for field in MAP_COVERAGE_FIELDS
        ):
            return "invalid_trace_row"
        if not isinstance(row.get("intentions"), list):
            return "invalid_trace_row"
        if not isinstance(row.get("monitoring"), Mapping):
            return "invalid_trace_row"
        if not isinstance(row.get("stage"), Mapping):
            return "invalid_trace_row"
        if row.get("planning") is not None and not isinstance(
            row["planning"], Mapping
        ):
            return "invalid_trace_row"
        if isinstance(row.get("planning"), Mapping) and not isinstance(
            row["planning"].get("intention_id"), str
        ):
            return "invalid_planning_intention_id"
        if not isinstance(row.get("decision"), Mapping):
            return "invalid_trace_row"
    return None


def _run_shape_error(run: Mapping[str, Any]) -> str | None:
    """Validate run-level scalars before deriving any execution metrics."""

    trace_error = _trace_shape_error(run.get("trace"))
    if trace_error is not None:
        return trace_error
    if (
        type(run.get("agent_steps")) is not int
        or _number(run.get("agent_steps")) is None
    ):
        return "invalid_run_state"
    if _number(run.get("elapsed_seconds")) is None:
        return "invalid_run_state"
    return None


def _source_ref(relative_state: str, source_index: int) -> dict[str, object]:
    return {"state": relative_state, "trace_index": source_index}


def _treatment_identity_errors(
    spec: Mapping[str, Any],
    hlr_treatment: object,
    environment: Mapping[str, Any] | None,
) -> list[str]:
    """Return mismatches against the frozen treatment and environment copy."""

    reasons = treatment_identity_reasons(
        spec,
        hlr_treatment,
        keys=TREATMENT_IDENTITY_KEYS,
    )
    if (
        environment is not None
        and environment.get("treatment") != hlr_treatment
    ):
        reasons.append("environment_treatment_mismatch")
    return reasons


def _unavailable_outcome(
    spec: Mapping[str, Any],
    initial_support: object,
) -> dict[str, Any]:
    """Return an explicit non-eligible row when run evidence is unreadable."""

    return {
        "execution_id": spec["execution_id"],
        "run_id": spec["run_id"],
        "operationally_valid": False,
        "scientifically_eligible": False,
        "task_outcome": "unobserved",
        "terminal_reason": "unknown",
        "steps": None,
        "wall_seconds": None,
        "highest_milestone": None,
        "diamond_obtained": None,
        "final_needs": None,
        "minimum_needs": None,
        "actions_at_or_below_intervention": None,
        "cumulative_reward": None,
        "achievement_counts": None,
        "environment_counters": None,
        "map_coverage": None,
        "initial_support": initial_support,
        "source_ref": None,
    }


def _belief_number(row: Mapping[str, Any], group: str, name: str) -> int:
    belief = row.get("belief", {})
    values = belief.get(group, {}) if isinstance(belief, Mapping) else {}
    value = values.get(name, 0) if isinstance(values, Mapping) else 0
    return int(value) if type(value) is int else 0


def _action_count(trace: Sequence[Mapping[str, Any]], through: int) -> int:
    return sum(
        isinstance(row.get("monitoring", {}).get("action_result"), Mapping)
        for row in trace[: through + 1]
    )


def _execution_statistics(
    trace: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Derive survival, Crafter, and environment-integrity statistics."""

    minimum_needs: dict[str, int | None] = {}
    for need in TRACKED_NEEDS:
        values: list[int] = []
        for row in trace:
            belief = row.get("belief")
            needs = belief.get("needs") if isinstance(belief, Mapping) else None
            value = needs.get(need) if isinstance(needs, Mapping) else None
            if type(value) is int:
                values.append(value)
        minimum_needs[need] = (
            min(values) if values and len(values) == len(trace) else None
        )

    action_rows = [
        row
        for row in trace
        if isinstance(row.get("decision"), Mapping)
        and row["decision"].get("kind") in ACTION_DECISIONS
    ]
    below: dict[str, int | None] = {}
    for need in TRACKED_NEEDS:
        count = 0
        complete = True
        for row in action_rows:
            needs = row.get("belief", {}).get("needs", {})
            if not isinstance(needs, Mapping):
                complete = False
                break
            value = needs.get(need)
            health = needs.get("health")
            if type(value) is not int or type(health) is not int:
                complete = False
                break
            threshold = (
                LOW_HEALTH_THRESHOLD
                if need == "health"
                else recovery_entry_threshold(need, health)
            )
            count += value <= threshold
        below[need] = count if complete else None

    rewards: list[float] = []
    reward_complete = True
    for row in trace:
        result = row.get("monitoring", {}).get("action_result")
        if not isinstance(result, Mapping):
            continue
        reward = _number(result.get("reward"))
        if reward is None:
            reward_complete = False
        else:
            rewards.append(float(reward))

    counter_names = (
        "observation_requests",
        "native_actions",
        "illegal_actions",
        "post_terminal_observations",
        "close_requests",
    )
    counters: dict[str, int | bool | None] = {}
    for name in counter_names:
        value = environment.get(name) if environment is not None else None
        counters[name] = value if type(value) is int else None
    closed = environment.get("closed") if environment is not None else None
    counters["closed"] = closed if type(closed) is bool else None

    achievement_value = (
        environment.get("achievement_counts")
        if environment is not None
        else None
    )
    achievement_counts = None
    if isinstance(achievement_value, Mapping) and all(
        isinstance(name, str) and type(value) is int
        for name, value in achievement_value.items()
    ):
        achievement_counts = dict(sorted(achievement_value.items()))

    return {
        "minimum_needs": minimum_needs,
        "actions_at_or_below_intervention": below,
        "cumulative_reward": sum(rewards) if reward_complete else None,
        "achievement_counts": achievement_counts,
        "environment_counters": counters,
    }


def _milestone_rows(
    spec: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    state_ref: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_actions = _action_count(trace, len(trace) - 1)
    final_elapsed = float(trace[-1].get("elapsed_seconds", 0.0)) if trace else 0.0
    for milestone in DIAMOND_MILESTONES:
        source_index = next(
            (
                index
                for index, row in enumerate(trace)
                if milestone in row.get("belief", {}).get("achievements", ())
            ),
            None,
        )
        attained = source_index is not None
        source_row = trace[source_index] if source_index is not None else None
        rows.append(
            {
                "execution_id": spec["execution_id"],
                "run_id": spec["run_id"],
                "milestone": milestone,
                "attained": attained,
                "first_revision": (
                    source_row.get("revision") if source_row is not None else None
                ),
                "first_action_count": (
                    _action_count(trace, source_index)
                    if source_index is not None
                    else total_actions
                ),
                "first_elapsed_seconds": (
                    float(source_row.get("elapsed_seconds", 0.0))
                    if source_row is not None
                    else final_elapsed
                ),
                "source_ref": (
                    _source_ref(state_ref, source_index)
                    if source_index is not None
                    else _source_ref(state_ref, len(trace) - 1)
                    if trace
                    else None
                ),
                "censoring": {
                    "status": "observed" if attained else "right_censored",
                    "reason": None if attained else "execution_ended_before_milestone",
                },
            }
        )
    return rows


def _planning_rows(
    spec: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    state_ref: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    attempts_by_intention: Counter[str] = Counter()
    for source_index, row in enumerate(trace):
        planning = row.get("planning")
        if not isinstance(planning, Mapping):
            continue
        intention_id = str(planning["intention_id"])
        attempts_by_intention[intention_id] += 1
        attempt_for_intention = attempts_by_intention[intention_id]
        base = {
            "execution_id": spec["execution_id"],
            "run_id": spec["run_id"],
            "source_index": source_index,
            "revision": row.get("revision"),
            "source_ref": _source_ref(state_ref, source_index),
        }
        attempts.append(
            {
                **base,
                "intention_id": intention_id,
                "attempt_for_intention": attempt_for_intention,
                "is_replan": attempt_for_intention > 1,
                "stage": planning.get("stage"),
                "classification": planning.get("classification"),
                "elapsed_seconds": planning.get("elapsed_seconds"),
                "plan_length": planning.get("plan_length"),
                "validation_status": planning.get("validation_status"),
                "accepted": planning.get("accepted") is True,
                "rejection": planning.get("rejection"),
                "plan_id": planning.get("plan_id"),
            }
        )
        plan_id = planning.get("plan_id")
        actions = planning.get("actions")
        if planning.get("accepted") is not True or not isinstance(
            plan_id, str
        ) or not isinstance(actions, list):
            continue
        dispatched = sum(
            later.get("decision", {}).get("kind") == "planned-action"
            and later.get("decision", {}).get("plan_id") == plan_id
            for later in trace[source_index:]
        )
        outcome = "right-censored"
        reason: str | None = "execution-ended-with-open-chunk"
        terminal_index: int | None = None
        for index, later in enumerate(trace[source_index:], start=source_index):
            monitoring = later.get("monitoring", {})
            invalidated = monitoring.get("plan_invalidated")
            chunk = monitoring.get("plan_chunk")
            goal = monitoring.get("plan_goal")
            if isinstance(invalidated, Mapping) and invalidated.get(
                "plan_id"
            ) == plan_id:
                outcome = "invalidated"
                reason = str(invalidated.get("reason"))
            elif isinstance(chunk, Mapping) and chunk.get("plan_id") == plan_id:
                outcome = "chunk-complete"
                reason = None
            elif isinstance(goal, Mapping) and goal.get("plan_id") == plan_id:
                outcome = str(goal.get("status"))
                reason = None
            else:
                continue
            terminal_index = index
            break
        chunks.append(
            {
                **base,
                "plan_id": plan_id,
                "stage": planning.get("stage"),
                "planned_length": len(actions),
                "dispatched_length": dispatched,
                "outcome": outcome,
                "reason": reason,
                "terminal_source_index": terminal_index,
                "censoring": {
                    "status": (
                        "right_censored"
                        if outcome == "right-censored"
                        else "observed"
                    ),
                    "reason": reason if outcome == "right-censored" else None,
                },
            }
        )
    return attempts, chunks


def _stage_rows(
    spec: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    state_ref: str,
) -> list[dict[str, Any]]:
    """Aggregate action and timing evidence for each observed technology stage."""

    activities: dict[str, dict[str, Any]] = {}
    for source_index, row in enumerate(trace):
        stage_value = row.get("stage", {})
        stage = (
            stage_value.get("derived")
            if isinstance(stage_value, Mapping)
            else None
        )
        if not isinstance(stage, str):
            continue
        activity = activities.setdefault(
            stage,
            {
                "execution_id": spec["execution_id"],
                "run_id": spec["run_id"],
                "stage": stage,
                "revisions": 0,
                "actions": 0,
                "planned_actions": 0,
                "recovery_actions": 0,
                "exploration_actions": 0,
                "planning_attempts": 0,
                "accepted_plans": 0,
                "planner_elapsed_seconds": 0.0,
                "planner_elapsed_available": True,
                "inter_revision_wall_seconds": 0.0,
                "inter_revision_wall_available": True,
                "source_ref": _source_ref(state_ref, source_index),
                "terminal_source_index": source_index,
            },
        )
        activity["revisions"] += 1
        activity["terminal_source_index"] = source_index

        planning = row.get("planning")
        if isinstance(planning, Mapping):
            activity["planning_attempts"] += 1
            activity["accepted_plans"] += planning.get("accepted") is True
            planner_elapsed = _number(planning.get("elapsed_seconds"))
            if (
                planner_elapsed is None
                and planning.get("classification") != "missing-knowledge"
            ):
                activity["planner_elapsed_available"] = False
            elif planner_elapsed is not None:
                activity["planner_elapsed_seconds"] += float(planner_elapsed)

        decision = row.get("decision")
        kind = decision.get("kind") if isinstance(decision, Mapping) else None
        if kind not in ACTION_DECISIONS:
            continue
        activity["actions"] += 1
        if kind == "planned-action":
            activity["planned_actions"] += 1
        elif kind == "recovery-action":
            activity["recovery_actions"] += 1
        else:
            activity["exploration_actions"] += 1
        if source_index + 1 < len(trace):
            start = _number(row.get("elapsed_seconds"))
            end = _number(trace[source_index + 1].get("elapsed_seconds"))
            if start is not None and end is not None:
                activity["inter_revision_wall_seconds"] += max(
                    0.0, float(end) - float(start)
                )
            else:
                activity["inter_revision_wall_available"] = False
        else:
            activity["inter_revision_wall_available"] = False

    order = {stage: index for index, stage in enumerate(TECHNOLOGY_STAGE_ORDER)}
    for activity in activities.values():
        if not activity.pop("planner_elapsed_available"):
            activity["planner_elapsed_seconds"] = None
        if not activity.pop("inter_revision_wall_available"):
            activity["inter_revision_wall_seconds"] = None
    return sorted(
        activities.values(),
        key=lambda row: (
            order.get(str(row["stage"]), len(order)),
            str(row["stage"]),
        ),
    )


def _recovery_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize recovery outcomes separately for each controlled need."""

    summary: dict[str, dict[str, Any]] = {}
    for need in RECOVERY_NEEDS:
        selected = [row for row in rows if row.get("need") == need]
        completed = sum(row.get("completed") is True for row in selected)
        preempted = sum(
            row.get("censoring", {}).get("reason") == "recovery_preempted"
            for row in selected
        )
        right_censored = sum(
            row.get("censoring", {}).get("status") == "right_censored"
            for row in selected
        )
        summary[need] = {
            "episodes": len(selected),
            "completed": completed,
            "preempted": preempted,
            "right_censored": right_censored,
            "actions": sum(int(row.get("actions", 0)) for row in selected),
            "completion_rate": completed / len(selected) if selected else None,
        }
    return summary


def _recovery_rows(
    spec: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    state_ref: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}
    for source_index, row in enumerate(trace):
        for transition in row.get("intentions", ()):
            if not isinstance(transition, Mapping):
                continue
            intention_id = transition.get("id")
            name = str(transition.get("name", ""))
            kind = transition.get("kind")
            if kind == "selected" and name.startswith("restore-") and isinstance(
                intention_id, str
            ):
                need = name.removeprefix("restore-")
                active[intention_id] = {
                    "execution_id": spec["execution_id"],
                    "run_id": spec["run_id"],
                    "intention_id": intention_id,
                    "need": need,
                    "entry_value": _belief_number(row, "needs", need),
                    "entry_health": _belief_number(row, "needs", "health"),
                    "exit_value": None,
                    "actions": 0,
                    "completed": False,
                    "preempted_technology_stage": row.get("stage", {}).get(
                        "derived"
                    ),
                    "source_ref": _source_ref(state_ref, source_index),
                    "terminal_source_index": None,
                    "censoring": {
                        "status": "right_censored",
                        "reason": "execution_ended_during_recovery",
                    },
                }
            if (
                kind in {"completed", "preempted"}
                and isinstance(intention_id, str)
                and intention_id in active
            ):
                episode = active.pop(intention_id)
                episode["exit_value"] = _belief_number(
                    row, "needs", str(episode["need"])
                )
                episode["completed"] = kind == "completed"
                episode["terminal_source_index"] = source_index
                episode["censoring"] = {
                    "status": "observed",
                    "reason": None if kind == "completed" else "recovery_preempted",
                }
                rows.append(episode)
        decision = row.get("decision", {})
        intention_id = decision.get("intention_id") if isinstance(
            decision, Mapping
        ) else None
        if intention_id in active and decision.get("kind") in {
            "planned-action",
            "recovery-action",
            "explore-action",
        }:
            active[intention_id]["actions"] += 1
    if trace:
        last = trace[-1]
        for episode in active.values():
            episode["exit_value"] = _belief_number(
                last, "needs", str(episode["need"])
            )
            rows.append(episode)
    return rows


def _exploration_rows(
    spec: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    state_ref: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(trace):
        decision = trace[index].get("decision", {})
        if not isinstance(decision, Mapping) or decision.get(
            "kind"
        ) != "explore-action":
            index += 1
            continue
        start = index
        reason = decision.get("reason")
        intention_id = decision.get("intention_id")
        while index + 1 < len(trace):
            following = trace[index + 1].get("decision", {})
            if (
                not isinstance(following, Mapping)
                or following.get("kind") != "explore-action"
                or following.get("reason") != reason
                or following.get("intention_id") != intention_id
            ):
                break
            index += 1
        end = index
        result_index = min(end + 1, len(trace) - 1)
        start_known = int(trace[start].get("belief", {}).get("known_terrain", 0))
        end_known = int(trace[result_index].get("belief", {}).get("known_terrain", 0))
        start_support = trace[start].get("stage", {}).get("support", {})
        end_support = trace[result_index].get("stage", {}).get("support", {})
        result_decision = trace[result_index].get("decision", {})
        same_owner = (
            isinstance(result_decision, Mapping)
            and result_decision.get("intention_id") == intention_id
        )
        rows.append(
            {
                "execution_id": spec["execution_id"],
                "run_id": spec["run_id"],
                "intention_id": intention_id,
                "reason": reason,
                "actions": end - start + 1,
                "newly_known_cells": max(0, end_known - start_known),
                "support_revealed": (
                    reason == "missing-knowledge"
                    and same_owner
                    and (
                        (
                            isinstance(start_support, Mapping)
                            and isinstance(end_support, Mapping)
                            and start_support.get("supported") is False
                            and end_support.get("supported") is True
                        )
                        or result_decision.get("kind")
                        in {"planned-action", "recovery-action"}
                    )
                ),
                "stalled": (
                    result_index == len(trace) - 1
                    and trace[result_index].get("decision", {}).get("reason")
                    == "exploration_stalled"
                ),
                "source_ref": _source_ref(state_ref, start),
                "terminal_source_index": result_index,
            }
        )
        index += 1
    return rows


def _stratum_rows(
    outcomes: Sequence[Mapping[str, Any]],
    milestones: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
    explorations: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strata = sorted({str(row.get("initial_support")) for row in outcomes})
    for stratum in strata:
        selected = [row for row in outcomes if row.get("initial_support") == stratum]
        execution_ids = {str(row["execution_id"]) for row in selected}
        selected_milestones = [
            row for row in milestones if row["execution_id"] in execution_ids
        ]
        selected_attempts = [
            row for row in attempts if row["execution_id"] in execution_ids
        ]
        selected_recoveries = [
            row for row in recoveries if row["execution_id"] in execution_ids
        ]
        selected_explorations = [
            row for row in explorations if row["execution_id"] in execution_ids
        ]
        selected_stages = [
            row for row in stages if row["execution_id"] in execution_ids
        ]
        classifications = Counter(
            str(row.get("classification")) for row in selected_attempts
        )
        terminations = Counter(str(row.get("terminal_reason")) for row in selected)
        milestone_summary: dict[str, Any] = {}
        for milestone in DIAMOND_MILESTONES:
            observed = [
                row
                for row in selected_milestones
                if row["milestone"] == milestone and row["attained"]
            ]
            milestone_summary[milestone] = {
                "attained": len(observed),
                "right_censored": len(selected) - len(observed),
                "actions_among_attained": summarize_numbers(
                    row.get("first_action_count") for row in observed
                ),
                "elapsed_seconds_among_attained": summarize_numbers(
                    row.get("first_elapsed_seconds") for row in observed
                ),
            }
        minimum_needs = {
            need: summarize_numbers(
                row.get("minimum_needs", {}).get(need)
                if isinstance(row.get("minimum_needs"), Mapping)
                else None
                for row in selected
            )
            for need in TRACKED_NEEDS
        }
        low_need_actions = {
            need: summarize_numbers(
                row.get("actions_at_or_below_intervention", {}).get(need)
                if isinstance(
                    row.get("actions_at_or_below_intervention"), Mapping
                )
                else None
                for row in selected
            )
            for need in TRACKED_NEEDS
        }
        counter_totals: dict[str, int | None] = {}
        for name in (
            "observation_requests",
            "native_actions",
            "illegal_actions",
            "post_terminal_observations",
            "close_requests",
        ):
            values = [
                row.get("environment_counters", {}).get(name)
                if isinstance(row.get("environment_counters"), Mapping)
                else None
                for row in selected
            ]
            counter_totals[name] = (
                sum(int(value) for value in values)
                if all(type(value) is int for value in values)
                else None
            )
        closed_values = [
            row.get("environment_counters", {}).get("closed")
            if isinstance(row.get("environment_counters"), Mapping)
            else None
            for row in selected
        ]
        counter_totals["closed_executions"] = (
            sum(value is True for value in closed_values)
            if all(type(value) is bool for value in closed_values)
            else None
        )

        achievement_maps: list[Mapping[str, Any]] = []
        for row in selected:
            counts = row.get("achievement_counts")
            if isinstance(counts, Mapping):
                achievement_maps.append(counts)
        achievement_successes: dict[str, int] | None = None
        achievement_rates: dict[str, float] | None = None
        if len(achievement_maps) == len(selected):
            names = sorted(
                {str(name) for values in achievement_maps for name in values}
            )
            achievement_successes = {
                name: sum(
                    int(values.get(name, 0)) > 0
                    for values in achievement_maps
                )
                for name in names
            }
            achievement_rates = {
                name: achievement_successes[name] / len(selected)
                for name in names
            }

        stage_summary: dict[str, dict[str, Any]] = {}
        for stage in TECHNOLOGY_STAGE_ORDER:
            stage_values = [
                row for row in selected_stages if row.get("stage") == stage
            ]
            if not stage_values:
                continue
            stage_summary[stage] = {
                "executions_reaching": len(stage_values),
                "actions": summarize_numbers(
                    row.get("actions") for row in stage_values
                ),
                "inter_revision_wall_seconds": summarize_numbers(
                    row.get("inter_revision_wall_seconds")
                    for row in stage_values
                ),
                "planner_elapsed_seconds": summarize_numbers(
                    row.get("planner_elapsed_seconds") for row in stage_values
                ),
                "planning_attempts": sum(
                    int(row["planning_attempts"]) for row in stage_values
                ),
                "recovery_actions": sum(
                    int(row["recovery_actions"]) for row in stage_values
                ),
                "exploration_actions": sum(
                    int(row["exploration_actions"]) for row in stage_values
                ),
            }
        successes = sum(row.get("diamond_obtained") is True for row in selected)
        rows.append(
            {
                "initial_support": stratum,
                "executions": len(selected),
                "diamond_successes": successes,
                "diamond_rate": successes / len(selected) if selected else None,
                "survival_outcomes": dict(sorted(terminations.items())),
                "milestones": milestone_summary,
                "planning_classifications": dict(sorted(classifications.items())),
                "planning_accepted": sum(
                    row.get("accepted") is True for row in selected_attempts
                ),
                "planning_rejected": sum(
                    row.get("accepted") is not True for row in selected_attempts
                ),
                "replanning_attempts": sum(
                    row.get("is_replan") is True for row in selected_attempts
                ),
                "exploration_episodes": len(selected_explorations),
                "exploration_actions": sum(
                    int(row["actions"]) for row in selected_explorations
                ),
                "exploration_newly_known_cells": sum(
                    int(row["newly_known_cells"])
                    for row in selected_explorations
                ),
                "recovery_episodes": len(selected_recoveries),
                "completed_recoveries": sum(
                    row.get("completed") is True for row in selected_recoveries
                ),
                "recovery_by_need": _recovery_summary(selected_recoveries),
                "minimum_needs": minimum_needs,
                "actions_at_or_below_intervention": low_need_actions,
                "environment_counter_totals": counter_totals,
                "cumulative_reward": summarize_numbers(
                    row.get("cumulative_reward") for row in selected
                ),
                "achievement_successes": achievement_successes,
                "achievement_rates": achievement_rates,
                "stage_activity": stage_summary,
            }
        )
    return rows


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive all full-domain analyses from retained public run evidence."""

    from .runner import _result_errors

    root = Path(root).resolve()
    discovered = sorted(
        int(path.name.rsplit("_", 1)[-1])
        for path in root.glob("exp_agent2_3_*")
        if path.name.rsplit("_", 1)[-1].isdigit()
    )
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [
            {
                "execution_id": f"run-{run_id}",
                "run_id": run_id,
                "factors": {},
                "expected": False,
            }
            for run_id in discovered
        ]
    report = new_report("2-3-cr", None if expected_executions is None else specs)
    outcomes: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    explorations: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    required_states = {
        "perceptor_0",
        "actuator_0",
        "llreasoner_0",
        "knowledge_0",
        "hlreasoner_0",
    }

    for spec in specs:
        run_id = int(spec["run_id"])
        factors = spec.get("factors", {})
        expected_support = (
            factors.get("initial_support")
            if isinstance(factors, Mapping)
            else None
        )
        agent_out = root / f"exp_agent2_3_{run_id}" / "out"
        environment_out = root / f"exp_env2_3_{run_id}" / "out"
        states = _json_objects(agent_out) if agent_out.is_dir() else {}
        environment_states = (
            _json_objects(environment_out) if environment_out.is_dir() else {}
        )
        high = states.get("hlreasoner_0", {})
        run = high.get("run") if isinstance(high, Mapping) else None
        present = agent_out.is_dir()
        if not isinstance(run, Mapping):
            reason = "required_state_missing" if present else "run_missing"
            report["executions"].append(
                execution_envelope(
                    spec,
                    present=present,
                    readable=False,
                    operationally_valid=False,
                    readability_reasons=(reason,),
                    operational_reasons=(reason,),
                )
            )
            outcomes.append(_unavailable_outcome(spec, expected_support))
            continue
        trace_value = run.get("trace")
        shape_error = _run_shape_error(run)
        if shape_error is not None:
            report["executions"].append(
                execution_envelope(
                    spec,
                    present=True,
                    readable=False,
                    operationally_valid=False,
                    readability_reasons=(shape_error,),
                    operational_reasons=(shape_error,),
                )
            )
            treatment = run.get("treatment", {})
            actual_support = (
                treatment.get("initial_support")
                if isinstance(treatment, Mapping)
                else expected_support
            )
            outcomes.append(_unavailable_outcome(spec, actual_support))
            continue
        assert isinstance(trace_value, list)
        trace = [dict(row) for row in trace_value]
        state_path = next(agent_out.glob("*.hlreasoner_0.json"), None)
        state_ref = (
            str(state_path.relative_to(root)).replace("\\", "/")
            if state_path is not None
            else f"exp_agent2_3_{run_id}/out/hlreasoner_0.json"
        )
        agent_log = root / f"exp_agent2_3_{run_id}.log"
        environment_log = root / f"exp_env2_3_{run_id}.log"
        agent_lines = (
            agent_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if agent_log.is_file()
            else []
        )
        environment_lines = (
            environment_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if environment_log.is_file()
            else []
        )
        environment = next(iter(environment_states.values()), None)
        operational_reasons = _treatment_identity_errors(
            spec,
            run.get("treatment"),
            environment if isinstance(environment, Mapping) else None,
        )
        checker_available = required_states == set(states) and isinstance(
            environment, Mapping
        ) and bool(agent_lines)
        checker_errors = (
            _result_errors(states, environment, agent_lines, environment_lines)
            if checker_available
            else []
        )
        if set(states) != required_states or not isinstance(environment, Mapping):
            operational_reasons.append("required_state_missing")
        if not agent_lines:
            operational_reasons.append("required_log_missing")
        if checker_errors:
            operational_reasons.append("operational_contract_failed")
        operational_reasons = list(dict.fromkeys(operational_reasons))
        operational = not operational_reasons
        reason = str(run.get("terminal_reason") or "unknown")
        phase = run.get("phase")
        diamond = reason == "diamond_obtained"
        task_status = (
            "success"
            if diamond
            else "failure"
            if phase == "scientific-terminal"
            else "error"
            if phase == "experiment-error"
            else "unobserved"
        )
        envelope = execution_envelope(
            spec,
            present=True,
            readable=True,
            operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=(
                "unavailable"
                if not checker_available
                else "passed"
                if not checker_errors
                else "failed"
            ),
            task_status=task_status,
            task_reason=None if diamond else reason,
            termination_reason=reason,
            identity={"treatment": run.get("treatment")},
            metric_availability={
                "execution_outcomes": "derived",
                "milestones": "derived",
                "planning_attempts": "existing",
                "plan_chunks": "derived",
                "recovery_episodes": "derived",
                "exploration_episodes": "derived",
                "stage_activity": "derived",
            },
            source_refs=(state_ref,),
        )
        report["executions"].append(envelope)
        final_belief = trace[-1].get("belief", {}) if trace else {}
        treatment = run.get("treatment", {})
        initial_support = (
            treatment.get("initial_support")
            if isinstance(treatment, Mapping)
            else spec.get("factors", {}).get("initial_support")
        )
        statistics = _execution_statistics(trace, environment)
        outcomes.append(
            {
                "execution_id": spec["execution_id"],
                "run_id": run_id,
                "operationally_valid": operational,
                "scientifically_eligible": False,
                "task_outcome": task_status,
                "terminal_reason": reason,
                "steps": (
                    run.get("agent_steps")
                    if type(run.get("agent_steps")) is int
                    else None
                ),
                "wall_seconds": float(run.get("elapsed_seconds", 0.0)),
                "highest_milestone": run.get("highest_milestone"),
                "diamond_obtained": diamond,
                "final_needs": dict(final_belief.get("needs", {})),
                **statistics,
                "map_coverage": {
                    "known_terrain": int(final_belief.get("known_terrain", 0)),
                    "reachable_cells": int(final_belief.get("reachable_cells", 0)),
                    "unknown_frontier": int(final_belief.get("unknown_frontier", 0)),
                },
                "initial_support": initial_support,
                "source_ref": (
                    _source_ref(state_ref, len(trace) - 1) if trace else None
                ),
            }
        )
        milestones.extend(_milestone_rows(spec, trace, state_ref))
        run_attempts, run_chunks = _planning_rows(spec, trace, state_ref)
        attempts.extend(run_attempts)
        chunks.extend(run_chunks)
        recoveries.extend(_recovery_rows(spec, trace, state_ref))
        explorations.extend(_exploration_rows(spec, trace, state_ref))
        stages.extend(_stage_rows(spec, trace, state_ref))

    eligibility = analysis_eligibility(report["executions"])
    eligible_ids = set(eligibility["eligible_execution_ids"])
    for outcome in outcomes:
        outcome["scientifically_eligible"] = outcome["execution_id"] in eligible_ids
    eligible_outcomes = eligible_analysis_rows(outcomes, eligibility)
    eligible_milestones = eligible_analysis_rows(milestones, eligibility)
    eligible_attempts = eligible_analysis_rows(attempts, eligibility)
    eligible_chunks = eligible_analysis_rows(chunks, eligibility)
    eligible_recoveries = eligible_analysis_rows(recoveries, eligibility)
    eligible_explorations = eligible_analysis_rows(explorations, eligibility)
    eligible_stages = eligible_analysis_rows(stages, eligibility)
    aggregates = _stratum_rows(
        eligible_outcomes,
        eligible_milestones,
        eligible_attempts,
        eligible_recoveries,
        eligible_explorations,
        eligible_stages,
    )
    report["analyses"] = {
        "execution_outcomes": {
            "unit": "execution",
            "nesting": "independent_execution",
            "eligibility": eligibility,
            "rows": outcomes,
            "summary": {
                "count": len(eligible_outcomes),
                "raw_descriptive_count": len(outcomes),
                "diamond_successes": sum(
                    row["diamond_obtained"] for row in eligible_outcomes
                ),
                "cumulative_reward": summarize_numbers(
                    row.get("cumulative_reward") for row in eligible_outcomes
                ),
            },
        },
        "milestone_outcomes": {
            "unit": "fixed_milestone",
            "nesting": "milestones_within_execution",
            "eligibility": eligibility,
            "rows": milestones,
            "summary": {
                "count": len(eligible_milestones),
                "raw_descriptive_count": len(milestones),
                "attained": sum(row["attained"] for row in eligible_milestones),
            },
        },
        "planning_attempts": {
            "unit": "planning_attempt",
            "nesting": "attempts_within_execution",
            "eligibility": eligibility,
            "rows": attempts,
            "summary": {
                "count": len(eligible_attempts),
                "raw_descriptive_count": len(attempts),
                "accepted": sum(row["accepted"] for row in eligible_attempts),
                "replanning_attempts": sum(
                    row["is_replan"] for row in eligible_attempts
                ),
                "elapsed_seconds": summarize_numbers(
                    row.get("elapsed_seconds") for row in eligible_attempts
                ),
            },
        },
        "plan_chunks": {
            "unit": "accepted_plan_chunk",
            "nesting": "chunks_within_execution",
            "eligibility": eligibility,
            "rows": chunks,
            "summary": {
                "count": len(eligible_chunks),
                "raw_descriptive_count": len(chunks),
            },
        },
        "recovery_episodes": {
            "unit": "recovery_episode",
            "nesting": "episodes_within_execution",
            "eligibility": eligibility,
            "rows": recoveries,
            "summary": {
                "count": len(eligible_recoveries),
                "raw_descriptive_count": len(recoveries),
                "completed": sum(
                    row["completed"] for row in eligible_recoveries
                ),
                "by_need": _recovery_summary(eligible_recoveries),
            },
        },
        "exploration_episodes": {
            "unit": "exploration_episode",
            "nesting": "episodes_within_execution",
            "eligibility": eligibility,
            "rows": explorations,
            "summary": {
                "count": len(eligible_explorations),
                "raw_descriptive_count": len(explorations),
                "actions": sum(row["actions"] for row in eligible_explorations),
                "newly_known_cells": sum(
                    row["newly_known_cells"] for row in eligible_explorations
                ),
            },
        },
        "stage_activity": {
            "unit": "technology_stage_within_execution",
            "nesting": "stages_within_execution",
            "eligibility": eligibility,
            "rows": stages,
            "summary": {
                "count": len(eligible_stages),
                "raw_descriptive_count": len(stages),
                "actions": sum(row["actions"] for row in eligible_stages),
                "planner_elapsed_seconds": summarize_numbers(
                    row.get("planner_elapsed_seconds")
                    for row in eligible_stages
                ),
                "inter_revision_wall_seconds": summarize_numbers(
                    row.get("inter_revision_wall_seconds")
                    for row in eligible_stages
                ),
            },
        },
        "initial_support_aggregates": {
            "unit": "initial_support_stratum",
            "nesting": "executions_within_stratum",
            "eligibility": eligibility,
            "rows": aggregates,
            "summary": {"count": len(aggregates)},
        },
    }
    write_execution_metrics(root, report)
    return report
