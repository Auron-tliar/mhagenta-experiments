"""Small experiment-local wire and activity contracts for 2-5-CR."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from mhagenta import Belief, Goal

K_ACTION = "action"
K_ATOMIC_ID = "environment_atomic_id"
K_OWNER_ID = "owner_action_id"
K_REQUESTER = "requester_kind"
MAX_TOTAL_ACTIONS = 900
K_OBSERVATION = "observation"
K_OBSERVATION_ID = "observation_id"
K_OBSERVATION_DIGEST = "observation_digest"
K_REWARD = "reward"
K_DONE = "done"
K_DEAD = "dead"
K_ILLEGAL_ACTION = "illegal_action"
K_LETHAL_MOVEMENT = "lethal_movement"
K_NEW_ACHIEVEMENTS = "new_achievements"
K_CONTRACT_ERROR = "contract_error"
A_CLOSE = "close"
MAX_ACTIVITY_ACTIONS = 32
EXPLORE_KNOWLEDGE_DELTA = 10
GET_RESOURCE_INTERACTION_RESERVE = 2
EAT_TARGET_INTERACTION_RESERVE = 4
RESOURCE_ITEMS: Mapping[str, str] = {
    "tree": "wood",
    "water": "drink",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
}
RESOURCE_TOOLS: Mapping[str, str] = {
    "stone": "wood_pickaxe",
    "coal": "wood_pickaxe",
    "iron": "stone_pickaxe",
    "diamond": "iron_pickaxe",
}
FAILURE_REASONS = frozenset({
    "action_bound", "target_lost", "precondition_invalid", "environment_terminal",
    "survival_interrupt", "total_action_bound", "illegal_action", "stagnation",
})


class ActivityId(str, Enum):
    """The five neural activities; PassiveObservation is not a policy."""

    EXPLORE = "explore"
    NAVIGATE_TO = "navigate_to"
    GET_RESOURCE = "get_resource"
    EAT_TARGET = "eat_target"
    EAT_COW = "eat_cow"


def activity_action_bound(activity: ActivityId | str) -> int:
    """Return the approved execution budget shared by runtime and preparation."""

    return 96 if ActivityId(activity) is ActivityId.EAT_COW else MAX_ACTIVITY_ACTIONS


@dataclass(frozen=True)
class ActivitySpec:
    """One bounded neural skill with an optional persistent public target."""

    goal_id: str
    activity: ActivityId
    purpose: str
    target_kind: str
    target_cell: tuple[int, int] | None
    baseline_revision: int
    baseline_value: int
    target_value: int
    max_actions: int = MAX_ACTIVITY_ACTIONS

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-native execution request."""

        return {
            "goal_id": self.goal_id, "activity": self.activity.value, "purpose": self.purpose,
            "target_kind": self.target_kind,
            "target_cell": list(self.target_cell) if self.target_cell is not None else None,
            "baseline_revision": self.baseline_revision, "baseline_value": self.baseline_value,
            "target_value": self.target_value, "max_actions": self.max_actions,
        }


def make_activity_goal(spec: ActivitySpec) -> Goal:
    """Create a typed neural goal whose metric is checked after every action."""

    return Goal(state=[Belief("activity_target", (spec.activity.value, spec.target_value))],
                extras={"kind": "activity", "status": "requested", **spec.as_dict()})


def activity_from_goal(goal: Goal) -> ActivitySpec:
    """Validate and parse the current neural request."""

    data = goal.extras
    if data.get("kind") != "activity" or data.get("status") != "requested":
        raise ValueError("Expected a requested neural activity.")
    spec = ActivitySpec(
        data["goal_id"], ActivityId(data["activity"]), data["purpose"], data["target_kind"],
        tuple(data["target_cell"]) if data["target_cell"] is not None else None,
        data["baseline_revision"], data["baseline_value"], data["target_value"], data["max_actions"],
    )
    if (not isinstance(spec.goal_id, str) or not spec.goal_id
            or type(spec.max_actions) is not int or not 1 <= spec.max_actions <= activity_action_bound(spec.activity)
            or type(spec.baseline_revision) is not int or spec.baseline_revision < 1
            or type(spec.baseline_value) is not int or type(spec.target_value) is not int
            or spec.target_value <= spec.baseline_value):
        raise ValueError("Invalid activity bound or metric.")
    if (spec.activity in {ActivityId.NAVIGATE_TO, ActivityId.GET_RESOURCE, ActivityId.EAT_TARGET}
            and (spec.target_cell is None or len(spec.target_cell) != 2 or any(type(x) is not int for x in spec.target_cell))):
        raise ValueError("Activity requires a public target cell.")
    if goal.state != make_activity_goal(spec).state:
        raise ValueError("Goal predicate contradicts request metadata.")
    return spec


def passive_goal(goal_id: str, revision: int, action: int, purpose: str) -> Goal:
    """Request one post-primitive observation, correlated by its owner action ID."""

    return Goal(state=[Belief("observation_after", (goal_id,))], extras={
        "kind": "passive", "status": "requested", "goal_id": goal_id,
        K_OWNER_ID: goal_id, "baseline_revision": revision, "action": action, "purpose": purpose,
    })


def resource_collected(action: int, source: Sequence[int], facing: Sequence[int],
                       target: Sequence[int], kind: str, events: Sequence[str], illegal: bool) -> bool:
    """Attribute a public collection event to the exact cell faced before interaction."""

    return (action == 5 and not illegal
            and tuple(source[i] + facing[i] for i in range(2)) == tuple(target)
            and f"collect_{RESOURCE_ITEMS[kind]}" in events)


def resource_stagnated(history: Sequence[Mapping[str, Any]]) -> bool:
    """Detect eight transitions confined to two cells without discovery or collection."""

    recent = history[-8:]
    return (len(recent) == 8
            and len({tuple(row[key]) for row in recent for key in ("source", "destination")}) <= 2
            and not any(row["discovered"] or row["collected"] for row in recent))


def validate_requested_goal(goal: Goal) -> None:
    """Validate the one supported neural or passive goal shape."""

    if goal.extras.get("kind") == "activity":
        activity_from_goal(goal)
        return
    data = goal.extras
    if (data.get("kind") != "passive" or data.get("status") != "requested"
            or not isinstance(data.get("goal_id"), str)
            or data.get(K_OWNER_ID) != data["goal_id"]
            or type(data.get("action")) is not int or data["action"] not in {*range(5), *range(6, 17)}
            or type(data.get("baseline_revision")) is not int or data["baseline_revision"] < 1):
        raise ValueError("Invalid passive request.")


def terminal_goal(requested: Goal, revision: int, status: str, reason: str = "") -> Goal:
    """Return a correlated terminal acknowledgment after a fresh belief revision."""

    if status not in {"succeeded", "failed", "interrupted"}:
        raise ValueError("Invalid terminal status.")
    return Goal(state=list(requested.state), extras={
        "kind": requested.extras["kind"], "goal_id": requested.extras["goal_id"],
        "status": status, "completion_revision": revision, "reason": reason,
    })


def goal_to_dict(goal: Goal) -> dict[str, Any]:
    """Serialize a Goal for compact JSON state."""

    return {
        "state": [{
            "predicate": belief.predicate,
            "arguments": list(belief.arguments or ()),
        } for belief in goal.state],
        "extras": dict(goal.extras),
    }


def goal_from_dict(data: Mapping[str, Any]) -> Goal:
    """Reconstruct a persisted activity Goal."""

    return Goal(
        state=[Belief(item["predicate"], tuple(item.get("arguments", ()))) for item in data["state"]],
        extras=dict(data["extras"]),
    )


def as_rgb_frame(observation: Any) -> np.ndarray:
    """Return an owned contiguous 64x64 uint8 RGB frame."""

    frame = np.asarray(observation)
    if frame.shape != (64, 64, 3) or not np.issubdtype(frame.dtype, np.integer):
        raise ValueError("Expected a 64x64 integer RGB frame.")
    if np.any(frame < 0) or np.any(frame > 255):
        raise ValueError("RGB values must be in [0, 255].")
    return np.ascontiguousarray(frame, dtype=np.uint8)


def rgb_sha256(observation: Any) -> str:
    """Return the validated frame's byte-level SHA-256 digest."""

    return hashlib.sha256(as_rgb_frame(observation).tobytes()).hexdigest()
