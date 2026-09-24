"""Operational and scientific certificate checks for Experiment 2-4-CR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)
from mha_exp_common.utils import module_name

from .activities import MILESTONES, NEED_THRESHOLDS, ActivityName
from .beliefs import CrafterAction


_TERMINAL_STATUSES = {
    "succeeded",
    "blocked",
    "budget_exhausted",
    "environment_terminal",
}
_ROW_FIELDS = {
    "goal_id",
    "stage",
    "plan_revision",
    "activity",
    "desired",
    "based_on_revision",
    "known_cell_count_start",
    "known_cell_count_end",
    "atomic",
    "completion_revision",
    "final_value",
    "status",
    "failure_reason",
    "interruption",
}
_ATOMIC_FIELDS = {
    "action_id",
    "action",
    "movement_kind",
    "source_cell",
    "destination_cell",
    "dispatch_revision",
    "legal",
    "confirmation_revision",
}


def check_results(
    agent_states: Mapping[str, Any],
    environment: Mapping[str, Any],
    logs: Mapping[str, Sequence[str]],
    *,
    expected_recording: bool,
    verbose: bool = True,
) -> bool:
    """Certify a clean run independently of whether it obtained a diamond."""

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    roles = (PERCEPTOR, ACTUATOR, LLREASONER, KNOWLEDGE, GOALGRAPH, HLREASONER)
    ids = {role: module_name(role, 0) for role in roles}
    if not isinstance(agent_states, Mapping) or set(agent_states) != set(ids.values()):
        failures.append("saved module set is incomplete")
    if not isinstance(environment, Mapping):
        failures.append("saved environment state is not a mapping")
    if failures:
        return _report(failures, verbose)

    try:
        json.dumps({"agent": agent_states, "environment": environment})
        states = {role: agent_states[identifier] for role, identifier in ids.items()}
        for role, state in states.items():
            check(isinstance(state, Mapping), f"{role} state is not a mapping")
            check(state.get("failure") is None, f"{role} recorded a failure")
            active = state.get("active_seconds")
            check(
                isinstance(active, (int, float))
                and not isinstance(active, bool)
                and active > 0,
                f"{role} did not record positive active time",
            )

        p, a, ll = states[PERCEPTOR], states[ACTUATOR], states[LLREASONER]
        knowledge, graph, hl = states[KNOWLEDGE], states[GOALGRAPH], states[HLREASONER]
        observation_values = (
            ll["observation_requests"],
            p["requests"],
            environment["observation_requests"],
            p["observations"],
            ll["observations"],
            ll["belief_sends"],
            knowledge["observed"],
            knowledge["forwarded"],
            hl["belief_updates"],
        )
        action_values = (
            ll["actions"],
            a["requests"],
            environment["action_requests"],
            environment["native_actions"],
            a["statuses"],
            ll["action_statuses"],
        )
        goal_values = (
            hl["dispatches"],
            graph["dispatches"],
            ll["goal_activations"],
            ll["terminal_updates"],
            graph["terminals"],
            hl["terminals"],
        )
        check(len(set(observation_values)) == 1 and observation_values[0] > 0, "observation chain is not quiescent")
        check(len(set(action_values)) == 1 and action_values[0] >= 0, "action chain is not quiescent")
        check(len(set(goal_values)) == 1 and goal_values[0] > 0, "goal chain is not quiescent")
        check(knowledge["last_revision"] == hl["latest_revision"] == observation_values[0], "belief revisions are inconsistent")

        treatment = hl.get("treatment")
        expected_treatment_fields = {
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
        }
        treatment_shape_valid = (
            isinstance(treatment, Mapping)
            and set(treatment) == expected_treatment_fields
        )
        check(treatment_shape_valid, "treatment shape is invalid")
        if treatment_shape_valid:
            check(
                treatment == environment.get("treatment"),
                "agent/environment treatments differ",
            )
            check(
                treatment.get("stratum") == "ordinary_from_scratch"
                and treatment.get("primary_intention") == "obtain_diamond",
                "treatment is not the from-scratch diamond stratum",
            )
            check(
                not (
                    {
                        "inventory_fixture",
                        "required_target",
                        "activity",
                        "requested_activity",
                    }
                    & set(treatment)
                ),
                "treatment contains an obsolete fixture",
            )

        hierarchy = hl["hierarchy"]
        check(
            isinstance(hierarchy, Mapping)
            and set(hierarchy)
            == {
                "intention",
                "current_stage",
                "highest_milestone",
                "override",
                "current_plan",
                "plan_revision",
                "activities",
                "final_revision",
                "terminal_reason",
                "status",
            },
            "diamond hierarchy shape is invalid",
        )
        intention = hierarchy["intention"]
        check(
            intention == {
                "name": "obtain_diamond",
                "target_value": 1,
                "status": intention.get("status"),
            }
            and intention.get("status") in {"succeeded", "unachieved"},
            "primary intention is invalid",
        )
        check(hierarchy["status"] in _TERMINAL_STATUSES, "hierarchy lacks a controlled terminal status")
        check(hl["phase"] == hierarchy["status"], "HL phase and hierarchy status differ")
        check(hl["terminal_reason"] == hierarchy["terminal_reason"], "terminal reasons differ")
        check(hierarchy["current_stage"] in MILESTONES[1:], "current technology stage is invalid")
        check(hierarchy["highest_milestone"] in MILESTONES, "highest milestone is invalid")
        check(hierarchy["final_revision"] == hl["latest_revision"], "final revision is stale")

        activities = hierarchy["activities"]
        check(isinstance(activities, list) and bool(activities), "hierarchy has no activity evidence")
        confirmations: list[int] = []
        for index, row in enumerate(activities, 1):
            check(isinstance(row, Mapping) and set(row) == _ROW_FIELDS, "activity row shape is invalid")
            check(row["goal_id"] == f"activity-{index}", "activity goal IDs are not ordered")
            check(row["plan_revision"] == index, "activity plan revisions are not ordered")
            check(row["stage"] in MILESTONES[1:], "activity stage is invalid")
            check(row["activity"] in {activity.value for activity in ActivityName}, "activity name is invalid")
            check(row["status"] in {"succeeded", "failed", "interrupted"}, "activity is not terminal")
            desired = row["desired"]
            check(
                isinstance(desired, Mapping)
                and set(desired) == {"predicate", "arguments", "extras"}
                and desired["extras"] is None
                and isinstance(desired["arguments"], list),
                "activity desired belief shape is invalid",
            )
            check(
                type(row["based_on_revision"]) is int
                and type(row["completion_revision"]) is int
                and row["completion_revision"] >= row["based_on_revision"],
                "activity revision span is invalid",
            )
            check(
                type(row["known_cell_count_start"]) is int
                and row["known_cell_count_start"] >= 0
                and type(row["known_cell_count_end"]) is int
                and row["known_cell_count_end"] >= row["known_cell_count_start"],
                "activity known-cell boundary is invalid",
            )
            atomic = row["atomic"]
            check(isinstance(atomic, list), "activity atomic trace is invalid")
            for atomic_row in atomic:
                check(isinstance(atomic_row, Mapping) and set(atomic_row) == _ATOMIC_FIELDS, "atomic row shape is invalid")
                check(
                    type(atomic_row["action"]) is int
                    and atomic_row["action"] in range(len(CrafterAction))
                    and atomic_row["movement_kind"] in {"walk", "turn", "none"}
                    and atomic_row["legal"] is True
                    and type(atomic_row["dispatch_revision"]) is int
                    and type(atomic_row["confirmation_revision"]) is int
                    and atomic_row["confirmation_revision"] > atomic_row["dispatch_revision"],
                    "atomic action lacks fresh legal confirmation",
                )
                confirmations.append(atomic_row["confirmation_revision"])
            if row["status"] == "succeeded":
                check(row["failure_reason"] == "" and row["interruption"] is None, "successful row has failure data")
                check(row["completion_revision"] > row["based_on_revision"], "successful row is not fresh")
                if desired["predicate"] == "inventory_at_least":
                    check(type(row["final_value"]) is int and row["final_value"] >= desired["arguments"][1], "successful inventory row lacks proof")
                else:
                    check(row["final_value"] is True, "successful condition row lacks proof")
            elif row["status"] == "interrupted":
                interruption = row["interruption"]
                check(row["failure_reason"] == "need_interruption", "interrupted row has the wrong reason")
                check(_valid_interruption(interruption, row), "interruption evidence is inconsistent")
            else:
                check(bool(row["failure_reason"]) and row["interruption"] is None, "failed row lacks ordinary failure evidence")
        check(confirmations == sorted(confirmations) and len(confirmations) == len(set(confirmations)), "atomic confirmations are not increasing")
        check(sum(row["status"] == "succeeded" for row in activities) == hl["completed_activities"], "completed activity counter mismatch")
        check(sum(row["status"] == "interrupted" for row in activities) == hl["interrupted_activities"], "interrupted activity counter mismatch")
        check(sum(row["status"] == "failed" for row in activities) == hl["failed_activities"], "failed activity counter mismatch")
        check(ll["need_interruptions"] == hl["need_interruptions"] == hl["interrupted_activities"], "need interruption counter mismatch")
        check(type(hl["recoveries"]) is int and 0 <= hl["recoveries"] <= hl["completed_activities"], "recovery counter is invalid")

        override = hierarchy["override"]
        check(override is None or _valid_override(override, hl["latest_revision"]), "recovery override is invalid")
        episode = hl["exploration_episode"]
        check(episode is None or _valid_episode(episode), "exploration episode is invalid")

        final = hl["abstract_state"]
        diamond = int(final["inventory"].get("diamond", 0))
        success = hierarchy["status"] == "succeeded"
        check(
            success
            == (
                diamond >= 1
                and "collect_diamond" in environment.get("achievements", ())
                and hierarchy["highest_milestone"] == "diamond"
            ),
            "diamond success evidence is inconsistent",
        )
        if success:
            check(hierarchy["terminal_reason"] == "diamond_obtained", "success reason is invalid")
            check(intention["status"] == "succeeded", "successful hierarchy has an unachieved intention")
        else:
            check(intention["status"] == "unachieved", "unsuccessful hierarchy has a successful intention")
        if hierarchy["status"] == "budget_exhausted":
            check(
                (hierarchy["terminal_reason"] == "total_action_bound" and ll["total_action_bound_reached"] is True)
                or (hierarchy["terminal_reason"] == "time_budget_exhausted" and ll["time_bound_reached"] is True),
                "budget outcome is inconsistent",
            )
        if hierarchy["status"] == "environment_terminal":
            check(environment["terminal"] or environment["dead"], "environment-terminal outcome lacks environment evidence")
            check(hierarchy["terminal_reason"] == ("death" if environment["dead"] else "episode_limit"),
                  "environment-terminal reason disagrees with the final health state")
        if hierarchy["status"] == "blocked":
            check(bool(hierarchy["terminal_reason"]), "blocked outcome lacks a reason")

        check(environment.get("failure") is None, "environment recorded a failure")
        check(environment["illegal_actions"] == 0, "environment accepted an illegal action")
        if treatment_shape_valid and treatment["total_action_cap"] is not None:
            check(
                environment["native_actions"] <= treatment["total_action_cap"],
                "native action cap was exceeded",
            )
        pending = (
            p["pending"] is False,
            a["pending_action_id"] is None,
            ll["active_activity"] is None,
            ll["pending_atomic"] is None,
            ll["awaiting_observation"] is False,
            graph["active_goal_id"] is None,
            hl["pending_outcome"] is None,
        )
        check(all(pending), "single-flight work remains pending")
        check(environment["close_requests"] == 1 and environment["closed"] is True, "environment was not closed exactly once")
        video = environment.get("video_path")
        check(
            (isinstance(video, str) and bool(video)) if expected_recording else video is None,
            "video path contradicts the recording policy",
        )
    except (IndexError, KeyError, StopIteration, TypeError, ValueError) as exc:
        failures.append(f"malformed scientific state: {exc}")

    markers = (
        "[error]",
        "[critical]",
        "traceback",
        "exceptiongroup",
        "caught exception",
        "failed to save state",
        "could not send message",
    )
    if not isinstance(logs, Mapping) or set(logs) != {"agent", "environment"}:
        failures.append("separate agent and environment logs are required")
    else:
        for name, lines in logs.items():
            if isinstance(lines, (str, bytes)) or not isinstance(lines, Sequence) or not lines:
                failures.append(f"{name} log is missing or empty")
            elif any(
                not isinstance(line, str) or any(token in line.lower() for token in markers)
                for line in lines
            ):
                failures.append(f"{name} log contains a runtime failure")
    return _report(failures, verbose)


def _valid_interruption(value: Any, row: Mapping[str, Any]) -> bool:
    fields = {
        "need",
        "observed_value",
        "interrupted_activity",
        "interrupted_goal_id",
        "belief_revision",
    }
    return (
        isinstance(value, Mapping)
        and set(value) == fields
        and value["need"] in NEED_THRESHOLDS
        and type(value["observed_value"]) is int
        and value["observed_value"] >= 0
        and value["interrupted_activity"] == row["activity"]
        and value["interrupted_goal_id"] == row["goal_id"]
        and value["belief_revision"] == row["completion_revision"]
    )


def _valid_override(value: Any, revision: int) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"kind", "need", "source_goal_id", "started_revision"}
        and value["kind"] == "need_recovery"
        and value["need"] in NEED_THRESHOLDS
        and (value["source_goal_id"] is None or isinstance(value["source_goal_id"], str))
        and type(value["started_revision"]) is int
        and 1 <= value["started_revision"] <= revision
    )


def _valid_episode(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"owner", "purpose", "consecutive_bounds"}
        and isinstance(value["owner"], str)
        and value["owner"].startswith(("technology:", "recovery:"))
        and isinstance(value["purpose"], str)
        and value["purpose"].startswith(("target:", "placement:"))
        and type(value["consecutive_bounds"]) is int
        and value["consecutive_bounds"] >= 0
    )


def _report(failures: Sequence[str], verbose: bool) -> bool:
    if verbose:
        for failure in failures:
            print(f"Check failed: {failure}")
    return not failures
