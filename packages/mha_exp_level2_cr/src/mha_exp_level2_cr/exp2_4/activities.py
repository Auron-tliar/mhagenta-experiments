"""Full-domain activity contracts and deterministic control for 2-4-CR."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import heapq
from typing import Any

from mhagenta import Belief, Goal

from .beliefs import (
    AbstractState,
    CrafterAction,
    Direction,
    Position,
    STATIC_TARGETS,
    WALKABLE_MATERIALS,
    cell_key,
    known_targets,
    position_from_key,
)


class ActivityName(str, Enum):
    """Supported high-level compound activities."""

    EXPLORE = "explore"
    EAT = "eat"
    DRINK = "drink"
    SLEEP = "sleep"
    GET_WOOD = "get_wood"
    GET_STONE = "get_stone"
    GET_COAL = "get_coal"
    GET_IRON = "get_iron"
    GET_DIAMOND = "get_diamond"
    PLACE_TABLE = "place_table"
    PLACE_FURNACE = "place_furnace"
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"


@dataclass(frozen=True)
class ActivityRule:
    """Frozen targets, requirements, and native action for one activity."""

    metric: str | None
    targets: tuple[str, ...] = ()
    inventory_required: tuple[tuple[str, int], ...] = ()
    station_set: str | None = None
    action: CrafterAction | None = None


ACTIVITY_RULES: Mapping[ActivityName, ActivityRule] = {
    ActivityName.EAT: ActivityRule("food", ("ripe-plant", "cow"), action=CrafterAction.DO),
    ActivityName.DRINK: ActivityRule("drink", ("water",), action=CrafterAction.DO),
    ActivityName.SLEEP: ActivityRule(None, action=CrafterAction.SLEEP),
    ActivityName.GET_WOOD: ActivityRule("wood", ("tree",), action=CrafterAction.DO),
    ActivityName.GET_STONE: ActivityRule(
        "stone", ("stone",), (("wood_pickaxe", 1),), action=CrafterAction.DO
    ),
    ActivityName.GET_COAL: ActivityRule(
        "coal", ("coal",), (("wood_pickaxe", 1),), action=CrafterAction.DO
    ),
    ActivityName.GET_IRON: ActivityRule(
        "iron", ("iron",), (("stone_pickaxe", 1),), action=CrafterAction.DO
    ),
    ActivityName.GET_DIAMOND: ActivityRule(
        "diamond", ("diamond",), (("iron_pickaxe", 1),), action=CrafterAction.DO
    ),
    ActivityName.PLACE_TABLE: ActivityRule(
        None, inventory_required=(("wood", 2),), action=CrafterAction.PLACE_TABLE
    ),
    ActivityName.PLACE_FURNACE: ActivityRule(
        None, inventory_required=(("stone", 4),), action=CrafterAction.PLACE_FURNACE
    ),
    ActivityName.MAKE_WOOD_PICKAXE: ActivityRule(
        "wood_pickaxe", inventory_required=(("wood", 1),),
        station_set="table", action=CrafterAction.MAKE_WOOD_PICKAXE,
    ),
    ActivityName.MAKE_STONE_PICKAXE: ActivityRule(
        "stone_pickaxe", inventory_required=(("wood", 1), ("stone", 1)),
        station_set="table", action=CrafterAction.MAKE_STONE_PICKAXE,
    ),
    ActivityName.MAKE_IRON_PICKAXE: ActivityRule(
        "iron_pickaxe",
        inventory_required=(("wood", 1), ("coal", 1), ("iron", 1)),
        station_set="table_furnace",
        action=CrafterAction.MAKE_IRON_PICKAXE,
    ),
}

RESOURCE_ACTIVITIES = frozenset(
    {
        ActivityName.EAT,
        ActivityName.DRINK,
        ActivityName.GET_WOOD,
        ActivityName.GET_STONE,
        ActivityName.GET_COAL,
        ActivityName.GET_IRON,
        ActivityName.GET_DIAMOND,
    }
)
PLACE_ACTIVITIES = frozenset({ActivityName.PLACE_TABLE, ActivityName.PLACE_FURNACE})
MAKE_ACTIVITIES = frozenset(
    {
        ActivityName.MAKE_WOOD_PICKAXE,
        ActivityName.MAKE_STONE_PICKAXE,
        ActivityName.MAKE_IRON_PICKAXE,
    }
)
TECHNOLOGY_STAGES = (
    "table",
    "wood_pickaxe",
    "stone_pickaxe",
    "furnace",
    "iron_pickaxe",
    "diamond",
)
MILESTONES = ("start", *TECHNOLOGY_STAGES)
NEED_THRESHOLDS: Mapping[str, tuple[int, int]] = {
    "food": (2, 5),
    "drink": (2, 5),
    "energy": (2, 5),
    "health": (2, 4),
}
REMOVABLE_REQUIREMENTS: Mapping[str, str | None] = {
    "tree": None,
    "stone": "wood_pickaxe",
    "coal": "wood_pickaxe",
    "iron": "stone_pickaxe",
    "diamond": "iron_pickaxe",
}
REMOVABLE_ORDER = {kind: index for index, kind in enumerate(REMOVABLE_REQUIREMENTS)}


class ActivityContractError(ValueError):
    """Raised when an activity violates its typed or domain contract."""


class ActivityUnavailable(ValueError):
    """Raised when current beliefs cannot continue a valid activity."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ActivitySpec:
    """Execution contract reconstructed solely from one requested Goal."""

    goal_id: str
    activity: ActivityName
    based_on_revision: int
    desired_predicate: str
    desired_arguments: tuple[Any, ...]
    target_value: int | None


@dataclass(frozen=True)
class ActivityOutcome:
    """Terminal activity evidence returned to the high-level reasoner."""

    goal_id: str
    status: str
    completion_revision: int
    atomic: tuple[dict[str, Any], ...]
    failure_reason: str
    interruption: Mapping[str, Any] | None


@dataclass(frozen=True)
class ActionDecision:
    """One native action and its LL-owned localization evidence."""

    action: CrafterAction
    movement_kind: str
    source_cell: Position
    destination_cell: Position
    target_cell: Position | None
    target_kind: str | None
    reason: str


@dataclass(frozen=True)
class PlanStep:
    """One coordinate-free activity request in a short stage plan."""

    activity: ActivityName
    desired_predicate: str
    desired_arguments: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON-native plan evidence."""

        return {
            "activity": self.activity.value,
            "desired": {
                "predicate": self.desired_predicate,
                "arguments": list(self.desired_arguments),
            },
        }


def _belief_arguments(belief: Belief) -> tuple[Any, ...]:
    if belief.arguments is None:
        return ()
    if isinstance(belief.arguments, (tuple, list)):
        return tuple(belief.arguments)
    return (belief.arguments,)


def make_activity_goal(
    *,
    goal_id: str,
    activity: ActivityName,
    based_on_revision: int,
    desired_predicate: str,
    desired_arguments: Sequence[Any],
) -> Goal:
    """Create and validate one coordinate-free requested activity Goal."""

    goal = Goal(
        state=[Belief(desired_predicate, tuple(desired_arguments))],
        extras={
            "goal_id": goal_id,
            "activity": activity.value,
            "status": "requested",
            "based_on_revision": based_on_revision,
        },
    )
    activity_from_goal(goal)
    return goal


def activity_from_goal(goal: Goal) -> ActivitySpec:
    """Strictly validate and reconstruct a requested activity."""

    extras = goal.extras
    fields = {"goal_id", "activity", "status", "based_on_revision"}
    if not isinstance(extras, Mapping) or set(extras) != fields or len(goal.state) != 1:
        raise ActivityContractError("Requested activity has unexpected fields")
    if extras.get("status") != "requested":
        raise ActivityContractError("Activity is not a request")
    try:
        activity = ActivityName(str(extras["activity"]))
    except (KeyError, ValueError) as exc:
        raise ActivityContractError("Unknown activity") from exc
    goal_id = extras.get("goal_id")
    revision = extras.get("based_on_revision")
    if not isinstance(goal_id, str) or not goal_id or type(revision) is not int or revision < 1:
        raise ActivityContractError("Invalid activity identity or revision")
    desired = goal.state[0]
    predicate = desired.predicate
    arguments = _belief_arguments(desired)
    target_value: int | None = None
    if predicate == "inventory_at_least":
        if len(arguments) != 2 or not isinstance(arguments[0], str):
            raise ActivityContractError("Invalid inventory desired belief")
        target_value = arguments[1]
        if type(target_value) is not int or target_value < 1:
            raise ActivityContractError("Activity target must be a positive integer")
        valid_items = {
            ActivityName.EAT: {"food"},
            ActivityName.DRINK: {"drink"},
            ActivityName.SLEEP: {"energy", "health"},
            ActivityName.GET_WOOD: {"wood"},
            ActivityName.GET_STONE: {"stone"},
            ActivityName.GET_COAL: {"coal"},
            ActivityName.GET_IRON: {"iron"},
            ActivityName.GET_DIAMOND: {"diamond"},
            ActivityName.MAKE_WOOD_PICKAXE: {"wood_pickaxe"},
            ActivityName.MAKE_STONE_PICKAXE: {"stone_pickaxe"},
            ActivityName.MAKE_IRON_PICKAXE: {"iron_pickaxe"},
        }
        if arguments[0] not in valid_items.get(activity, set()):
            raise ActivityContractError("Inventory desired belief contradicts activity")
    elif predicate == "usable_station_set":
        expected = {
            ActivityName.PLACE_TABLE: "table",
            ActivityName.PLACE_FURNACE: "table_furnace",
        }.get(activity)
        if arguments != (expected,) or expected is None:
            raise ActivityContractError("Station desired belief contradicts activity")
    elif predicate == "reachable_target_kind":
        if activity is not ActivityName.EXPLORE or len(arguments) != 1 or arguments[0] not in {
            "food", "water", "tree", "stone", "coal", "iron", "diamond"
        }:
            raise ActivityContractError("Invalid Explore target desired belief")
    elif predicate == "placement_opportunity":
        if activity is not ActivityName.EXPLORE or arguments not in {("table",), ("furnace",)}:
            raise ActivityContractError("Invalid Explore placement desired belief")
    else:
        raise ActivityContractError("Unsupported desired belief")
    return ActivitySpec(
        goal_id,
        activity,
        revision,
        predicate,
        arguments,
        target_value,
    )


def terminal_activity_goal(
    requested: Goal,
    *,
    status: str,
    completion_revision: int,
    atomic: Sequence[Mapping[str, Any]],
    failure_reason: str = "",
    interruption: Mapping[str, Any] | None = None,
) -> Goal:
    """Create a strict terminal activity update, including interruption data."""

    spec = activity_from_goal(requested)
    if status not in {"succeeded", "failed"}:
        raise ActivityContractError("Terminal status must be succeeded or failed")
    if type(completion_revision) is not int or completion_revision < 1:
        raise ActivityContractError("Invalid completion revision")
    if status == "succeeded" and (failure_reason or interruption is not None):
        raise ActivityContractError("Successful activity cannot carry failure data")
    if status == "failed" and not failure_reason:
        raise ActivityContractError("Failed activity needs a reason")
    if interruption is not None:
        _validate_interruption(interruption, spec, completion_revision)
        if failure_reason != "need_interruption":
            raise ActivityContractError("Interruption requires need_interruption reason")
    elif failure_reason == "need_interruption":
        raise ActivityContractError("Need interruption requires structured evidence")
    goal = Goal(
        state=list(requested.state),
        extras={
            "goal_id": spec.goal_id,
            "status": status,
            "completion_revision": completion_revision,
            "atomic": [dict(row) for row in atomic],
            "failure_reason": failure_reason,
            "interruption": dict(interruption) if interruption is not None else None,
        },
    )
    outcome_from_goal(goal)
    return goal


def _validate_interruption(
    value: Mapping[str, Any], spec: ActivitySpec | None, completion_revision: int
) -> None:
    fields = {
        "need",
        "observed_value",
        "interrupted_activity",
        "interrupted_goal_id",
        "belief_revision",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ActivityContractError("Interruption has unexpected fields")
    if value.get("need") not in NEED_THRESHOLDS:
        raise ActivityContractError("Interruption has an invalid need")
    if type(value.get("observed_value")) is not int or value["observed_value"] < 0:
        raise ActivityContractError("Interruption has an invalid observed value")
    if type(value.get("belief_revision")) is not int or value["belief_revision"] != completion_revision:
        raise ActivityContractError("Interruption revision mismatch")
    if not isinstance(value.get("interrupted_goal_id"), str):
        raise ActivityContractError("Interruption goal ID is invalid")
    try:
        interrupted_activity = ActivityName(str(value.get("interrupted_activity")))
    except ValueError as exc:
        raise ActivityContractError("Interruption activity is invalid") from exc
    if spec is not None and (
        value["interrupted_goal_id"] != spec.goal_id
        or interrupted_activity is not spec.activity
    ):
        raise ActivityContractError("Interruption does not identify its request")


def outcome_from_goal(goal: Goal) -> ActivityOutcome:
    """Strictly parse one terminal activity update."""

    extras = goal.extras
    fields = {
        "goal_id",
        "status",
        "completion_revision",
        "atomic",
        "failure_reason",
        "interruption",
    }
    if not isinstance(extras, Mapping) or set(extras) != fields or len(goal.state) != 1:
        raise ActivityContractError("Terminal activity has unexpected fields")
    goal_id, status = extras.get("goal_id"), extras.get("status")
    revision, failure = extras.get("completion_revision"), extras.get("failure_reason")
    if not isinstance(goal_id, str) or not goal_id or status not in {"succeeded", "failed"}:
        raise ActivityContractError("Invalid terminal activity identity or status")
    if type(revision) is not int or revision < 1 or not isinstance(failure, str):
        raise ActivityContractError("Invalid terminal activity evidence")
    interruption = extras.get("interruption")
    if status == "succeeded" and (failure or interruption is not None):
        raise ActivityContractError("Successful terminal activity has failure data")
    if status == "failed" and not failure:
        raise ActivityContractError("Failed terminal activity lacks a reason")
    if interruption is not None:
        _validate_interruption(interruption, None, revision)
        if failure != "need_interruption" or interruption["interrupted_goal_id"] != goal_id:
            raise ActivityContractError("Terminal interruption is inconsistent")
    elif failure == "need_interruption":
        raise ActivityContractError("Terminal interruption evidence is missing")
    atomic = extras.get("atomic")
    if not isinstance(atomic, list):
        raise ActivityContractError("Terminal atomic trace must be a list")
    rows = tuple(_atomic_row(row) for row in atomic)
    return ActivityOutcome(goal_id, status, revision, rows, failure, interruption)


def _position(value: Any) -> Position:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(part) is not int for part in value)
    ):
        raise ActivityContractError("Invalid cell")
    return int(value[0]), int(value[1])


def _atomic_row(value: Any) -> dict[str, Any]:
    fields = {
        "action_id",
        "action",
        "movement_kind",
        "source_cell",
        "destination_cell",
        "dispatch_revision",
        "legal",
        "confirmation_revision",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ActivityContractError("Atomic row has unexpected fields")
    row = dict(value)
    if not isinstance(row["action_id"], str) or not row["action_id"]:
        raise ActivityContractError("Atomic action ID is invalid")
    if type(row["action"]) is not int or row["action"] not in range(len(CrafterAction)):
        raise ActivityContractError("Atomic action is invalid")
    if row["movement_kind"] not in {"walk", "turn", "none"}:
        raise ActivityContractError("Atomic movement kind is invalid")
    _position(row["source_cell"])
    _position(row["destination_cell"])
    if type(row["dispatch_revision"]) is not int or type(row["confirmation_revision"]) is not int:
        raise ActivityContractError("Atomic revisions must be integers")
    if row["confirmation_revision"] <= row["dispatch_revision"] or type(row["legal"]) is not bool:
        raise ActivityContractError("Atomic confirmation is invalid")
    return row


def goal_to_dict(goal: Goal) -> dict[str, Any]:
    """Serialize a typed Goal into stable JSON-compatible evidence."""

    return {
        "state": [
            {
                "predicate": belief.predicate,
                "arguments": list(_belief_arguments(belief)),
                "extras": belief.extras,
            }
            for belief in goal.state
        ],
        "extras": dict(goal.extras),
    }


def goal_from_dict(data: Mapping[str, Any]) -> Goal:
    """Reconstruct a typed Goal from its JSON representation."""

    return Goal(
        state=[
            Belief(item["predicate"], tuple(item.get("arguments", ())), extras=item.get("extras"))
            for item in data["state"]
        ],
        extras=dict(data["extras"]),
    )


def _state_positions(values: Sequence[str]) -> set[Position]:
    return {position_from_key(value) for value in values}


def reachable_cells(
    state: Mapping[str, Any], *, blocked: set[Position] | None = None
) -> tuple[dict[Position, int], dict[Position, Position]]:
    """Return BFS distances and predecessors over the known-safe component."""

    player = _position(state["player"])
    safe = _state_positions(state.get("safe_cells", ())) | {player}
    safe -= blocked or set()
    if player not in safe:
        return {}, {}
    distances = {player: 0}
    predecessors: dict[Position, Position] = {}
    frontier = deque([player])
    while frontier:
        current = frontier.popleft()
        for direction in Direction:
            neighbor = current[0] + direction.delta[0], current[1] + direction.delta[1]
            if neighbor not in safe or neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            predecessors[neighbor] = current
            frontier.append(neighbor)
    return distances, predecessors


def find_path(
    start: Position, goals: set[Position], safe_cells: set[Position]
) -> tuple[Position, ...] | None:
    """Return deterministic safe A* steps after start, or None."""

    if start in goals:
        return ()
    if not goals:
        return None
    frontier: list[tuple[int, int, int, Position]] = []
    counter = 0

    def heuristic(position: Position) -> int:
        return min(abs(position[0] - goal[0]) + abs(position[1] - goal[1]) for goal in goals)

    heapq.heappush(frontier, (heuristic(start), 0, counter, start))
    came_from: dict[Position, Position | None] = {start: None}
    costs = {start: 0}
    reached: Position | None = None
    while frontier:
        _, cost, _, current = heapq.heappop(frontier)
        if cost != costs[current]:
            continue
        if current in goals:
            reached = current
            break
        for direction in Direction:
            neighbor = current[0] + direction.delta[0], current[1] + direction.delta[1]
            if neighbor not in safe_cells:
                continue
            new_cost = cost + 1
            if new_cost >= costs.get(neighbor, 1 << 30):
                continue
            costs[neighbor] = new_cost
            came_from[neighbor] = current
            counter += 1
            heapq.heappush(frontier, (new_cost + heuristic(neighbor), new_cost, counter, neighbor))
    if reached is None:
        return None
    path: list[Position] = []
    cursor = reached
    while cursor != start:
        path.append(cursor)
        parent = came_from[cursor]
        assert parent is not None
        cursor = parent
    return tuple(reversed(path))


def _target_positions(state: Mapping[str, Any]) -> dict[str, tuple[Position, ...]]:
    targets = known_targets(state)
    food = tuple(sorted((*targets.get("ripe-plant", ()), *targets.get("cow", ()))))
    result = {kind: values for kind, values in targets.items() if kind in STATIC_TARGETS}
    if food:
        result["food"] = food
    return dict(sorted(result.items()))


def _target_candidates(
    state: Mapping[str, Any], target_kinds: Sequence[str]
) -> list[tuple[int, str, Position, Position]]:
    distances, _ = reachable_cells(state)
    safe = set(distances)
    raw = known_targets(state)
    candidates: list[tuple[int, str, Position, Position]] = []
    for kind in target_kinds:
        source_kinds = ("ripe-plant", "cow") if kind == "food" else (kind,)
        for source_kind in source_kinds:
            for target in raw.get(source_kind, ()):
                approaches = sorted(
                    (target[0] + direction.delta[0], target[1] + direction.delta[1])
                    for direction in Direction
                    if (target[0] + direction.delta[0], target[1] + direction.delta[1]) in safe
                )
                if approaches:
                    approach = min(approaches, key=lambda cell: (distances[cell], cell))
                    candidates.append((distances[approach], source_kind, target, approach))
    return sorted(candidates)


def reachable_targets(
    state: Mapping[str, Any], target_kinds: Sequence[str]
) -> dict[str, tuple[Position, ...]]:
    """Return known targets having a reachable cardinal safe approach."""

    result: dict[str, set[Position]] = {}
    for _, source_kind, target, _ in _target_candidates(state, target_kinds):
        kind = "food" if source_kind in {"ripe-plant", "cow"} else source_kind
        result.setdefault(kind, set()).add(target)
    return {kind: tuple(sorted(values)) for kind, values in sorted(result.items())}


def _chebyshev(first: Position, second: Position) -> int:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def _stations(state: Mapping[str, Any], kind: str) -> tuple[Position, ...]:
    return tuple(
        sorted(
            position_from_key(key)
            for key, material in state.get("terrain", {}).items()
            if material == kind
        )
    )


def _craft_anchors(state: Mapping[str, Any], station_set: str) -> tuple[Position, ...]:
    distances, _ = reachable_cells(state)
    tables = _stations(state, "table")
    furnaces = _stations(state, "furnace")
    anchors = []
    for cell in distances:
        has_table = any(_chebyshev(cell, table) <= 1 for table in tables)
        has_furnace = any(_chebyshev(cell, furnace) <= 1 for furnace in furnaces)
        if has_table and (station_set == "table" or has_furnace):
            anchors.append(cell)
    return tuple(sorted(anchors, key=lambda cell: (distances[cell], cell)))


def usable_station_sets(state: Mapping[str, Any]) -> frozenset[str]:
    """Return station combinations usable from a reachable safe anchor."""

    values: set[str] = set()
    if _craft_anchors(state, "table"):
        values.add("table")
    if _craft_anchors(state, "table_furnace"):
        values.add("table_furnace")
    return frozenset(values)


def _raw_placement_options(
    state: Mapping[str, Any], *, blocked: set[Position] | None = None
) -> tuple[tuple[Position, Position, Position], ...]:
    distances, _ = reachable_cells(state, blocked=blocked)
    safe = set(distances)
    visible = _state_positions(state.get("visible_cells", ()))
    occupants = state.get("occupants", {})
    terrain = state.get("terrain", {})
    options: list[tuple[int, Position, Position, Position]] = []
    for placement in visible:
        for direction in Direction:
            anchor = placement[0] - direction.delta[0], placement[1] - direction.delta[1]
            orientation = anchor[0] - direction.delta[0], anchor[1] - direction.delta[1]
            if orientation not in safe or anchor not in safe:
                continue
            if terrain.get(cell_key(placement)) not in WALKABLE_MATERIALS:
                continue
            if cell_key(placement) in occupants:
                continue
            options.append((distances[orientation], orientation, anchor, placement))
    return tuple((orientation, anchor, placement) for _, orientation, anchor, placement in sorted(options))


def placement_options(
    state: Mapping[str, Any], station: str, *, first_only: bool = False
) -> tuple[tuple[Position, Position, Position], ...]:
    """Return deterministic orientation-feasible station placement triples.

    ``first_only`` avoids enumerating equivalent later choices when callers need
    only the coordinate-free availability fact.
    """

    if station not in {"table", "furnace"}:
        raise ValueError(f"Unknown station: {station!r}")
    tables = _stations(state, "table")
    furnaces = _stations(state, "furnace")
    result: list[tuple[Position, Position, Position]] = []
    for option in _raw_placement_options(state):
        _, anchor, placement = option
        if station == "furnace":
            if placement not in tables and any(_chebyshev(anchor, table) <= 1 for table in tables):
                result.append(option)
                if first_only:
                    break
            continue
        if any(_chebyshev(anchor, furnace) <= 1 for furnace in furnaces):
            result.append(option)
            if first_only:
                break
            continue
        projected = dict(state)
        projected_terrain = dict(state.get("terrain", {}))
        projected_terrain[cell_key(placement)] = "table"
        projected["terrain"] = projected_terrain
        projected["safe_cells"] = [
            key for key in state.get("safe_cells", ()) if key != cell_key(placement)
        ]
        future = _raw_placement_options(projected, blocked={placement})
        if any(
            future_placement != placement and _chebyshev(future_anchor, placement) <= 1
            for _, future_anchor, future_placement in future
        ):
            result.append(option)
            if first_only:
                break
    return tuple(result)


def planning_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build the sole JSON-native LL-to-HL planning projection."""

    targets = _target_positions(state)
    reachable = reachable_targets(state, tuple(targets))
    station_sets = usable_station_sets(state)
    placements = [
        station
        for station in ("table", "furnace")
        if placement_options(state, station, first_only=True)
    ]
    return {
        "known_target_kinds": sorted(targets),
        "reachable_target_kinds": sorted(reachable),
        "station_counts": {
            "table": len(_stations(state, "table")),
            "furnace": len(_stations(state, "furnace")),
        },
        "usable_station_sets": sorted(station_sets),
        "placement_opportunities": placements,
    }


def desired_value(
    spec: ActivitySpec, state: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> int | bool:
    """Evaluate an activity's desired belief against detailed LL state."""

    if spec.desired_predicate == "inventory_at_least":
        return int(state.get("inventory", {}).get(spec.desired_arguments[0], 0))
    planning = planning_summary(state) if summary is None else summary
    key = {
        "usable_station_set": "usable_station_sets",
        "reachable_target_kind": "reachable_target_kinds",
        "placement_opportunity": "placement_opportunities",
    }[spec.desired_predicate]
    return spec.desired_arguments[0] in planning[key]


def activity_complete(spec: ActivitySpec, state: Mapping[str, Any]) -> bool:
    """Return whether a fresh detailed belief state proves Goal satisfaction."""

    value = desired_value(spec, state)
    satisfied = value is True if spec.target_value is None else int(value) >= spec.target_value
    if spec.activity is ActivityName.SLEEP and state.get("sleeping"):
        return False
    return satisfied


def prerequisite_available(spec: ActivitySpec, state: Mapping[str, Any]) -> bool:
    """Return whether current inventory and stations allow the activity."""

    rule = ACTIVITY_RULES.get(spec.activity)
    if rule is None:
        return True
    inventory = state.get("inventory", {})
    if any(int(inventory.get(item, 0)) < amount for item, amount in rule.inventory_required):
        return False
    if rule.station_set is not None and rule.station_set not in usable_station_sets(state):
        return False
    return True


def urgent_need(inventory: Mapping[str, int]) -> str | None:
    """Select one urgent survival need using the frozen priority rule."""

    tie_order = {"drink": 0, "food": 1, "energy": 2}
    ordinary = [
        (int(inventory.get(need, 0)), tie_order[need], need)
        for need in ("drink", "food", "energy")
        if int(inventory.get(need, 0)) <= NEED_THRESHOLDS[need][0]
    ]
    if ordinary:
        return min(ordinary)[2]
    return "health" if int(inventory.get("health", 0)) <= NEED_THRESHOLDS["health"][0] else None


def protected_needs(spec: ActivitySpec) -> frozenset[str]:
    """Return needs protected from self-interruption by this activity."""

    if spec.activity is ActivityName.EAT:
        return frozenset({"food"})
    if spec.activity is ActivityName.DRINK:
        return frozenset({"drink"})
    if spec.activity is ActivityName.SLEEP:
        return frozenset({"energy", "health"})
    if spec.activity is ActivityName.EXPLORE and spec.desired_predicate == "reachable_target_kind":
        if spec.desired_arguments == ("food",):
            return frozenset({"food"})
        if spec.desired_arguments == ("water",):
            return frozenset({"drink"})
    return frozenset()


def technology_stage(snapshot: AbstractState) -> str:
    """Derive the first unmet prerequisite stage from an abstract snapshot."""

    inventory = snapshot.inventory
    if int(inventory.get("diamond", 0)) >= 1:
        return "diamond"
    if int(inventory.get("iron_pickaxe", 0)) >= 1:
        return "diamond"
    if int(inventory.get("stone_pickaxe", 0)) >= 1:
        if "table" not in snapshot.usable_station_sets:
            return "table"
        if "table_furnace" not in snapshot.usable_station_sets:
            return "furnace"
        return "iron_pickaxe"
    if int(inventory.get("wood_pickaxe", 0)) >= 1:
        if "table" not in snapshot.usable_station_sets:
            return "table"
        return "stone_pickaxe"
    if "table" not in snapshot.usable_station_sets:
        return "table"
    return "wood_pickaxe"


def highest_milestone(snapshot: AbstractState) -> str:
    """Return the highest currently evidenced technology milestone."""

    inventory = snapshot.inventory
    if int(inventory.get("diamond", 0)) >= 1:
        return "diamond"
    if int(inventory.get("iron_pickaxe", 0)) >= 1:
        return "iron_pickaxe"
    if "table_furnace" in snapshot.usable_station_sets:
        return "furnace"
    if int(inventory.get("stone_pickaxe", 0)) >= 1:
        return "stone_pickaxe"
    if int(inventory.get("wood_pickaxe", 0)) >= 1:
        return "wood_pickaxe"
    if "table" in snapshot.usable_station_sets:
        return "table"
    return "start"


def _inventory_step(activity: ActivityName, item: str, value: int) -> PlanStep:
    return PlanStep(activity, "inventory_at_least", (item, value))


def _gather_or_explore(
    snapshot: AbstractState, activity: ActivityName, target_kind: str, item: str, value: int
) -> PlanStep:
    if target_kind in snapshot.reachable_target_kinds:
        return _inventory_step(activity, item, value)
    return PlanStep(ActivityName.EXPLORE, "reachable_target_kind", (target_kind,))


def derive_stage_plan(snapshot: AbstractState) -> tuple[PlanStep, ...]:
    """Derive a deterministic short plan for the current technology stage."""

    inventory = snapshot.inventory
    stage = technology_stage(snapshot)
    if stage == "diamond" and int(inventory.get("diamond", 0)) >= 1:
        return ()
    if stage == "table":
        steps = []
        if int(inventory.get("wood", 0)) < 2:
            step = _gather_or_explore(snapshot, ActivityName.GET_WOOD, "tree", "wood", 2)
            if step.activity is ActivityName.EXPLORE:
                return (step,)
            steps.append(step)
        if "table" in snapshot.placement_opportunities:
            steps.append(PlanStep(ActivityName.PLACE_TABLE, "usable_station_set", ("table",)))
        else:
            steps.append(PlanStep(ActivityName.EXPLORE, "placement_opportunity", ("table",)))
        return tuple(steps)
    if stage == "wood_pickaxe":
        steps = []
        if int(inventory.get("wood", 0)) < 1:
            steps.append(_gather_or_explore(snapshot, ActivityName.GET_WOOD, "tree", "wood", 1))
        steps.append(_inventory_step(ActivityName.MAKE_WOOD_PICKAXE, "wood_pickaxe", 1))
        return tuple(steps[:1] if steps[0].activity is ActivityName.EXPLORE else steps)
    if stage == "stone_pickaxe":
        steps = []
        if int(inventory.get("stone", 0)) < 1:
            steps.append(_gather_or_explore(snapshot, ActivityName.GET_STONE, "stone", "stone", 1))
        if int(inventory.get("wood", 0)) < 1:
            steps.append(_gather_or_explore(snapshot, ActivityName.GET_WOOD, "tree", "wood", 1))
        steps.append(_inventory_step(ActivityName.MAKE_STONE_PICKAXE, "stone_pickaxe", 1))
        first_explore = next((index for index, step in enumerate(steps) if step.activity is ActivityName.EXPLORE), None)
        return tuple(steps[: first_explore + 1] if first_explore is not None else steps)
    if stage == "furnace":
        steps = []
        if int(inventory.get("stone", 0)) < 4:
            step = _gather_or_explore(snapshot, ActivityName.GET_STONE, "stone", "stone", 4)
            if step.activity is ActivityName.EXPLORE:
                return (step,)
            steps.append(step)
        if "furnace" in snapshot.placement_opportunities:
            steps.append(PlanStep(ActivityName.PLACE_FURNACE, "usable_station_set", ("table_furnace",)))
        else:
            steps.append(PlanStep(ActivityName.EXPLORE, "placement_opportunity", ("furnace",)))
        return tuple(steps)
    if stage == "iron_pickaxe":
        needs = (
            ("coal", ActivityName.GET_COAL, "coal"),
            ("iron", ActivityName.GET_IRON, "iron"),
            ("wood", ActivityName.GET_WOOD, "tree"),
        )
        steps = []
        for item, activity, target in needs:
            if int(inventory.get(item, 0)) < 1:
                step = _gather_or_explore(snapshot, activity, target, item, 1)
                steps.append(step)
                if step.activity is ActivityName.EXPLORE:
                    return tuple(steps)
        steps.append(_inventory_step(ActivityName.MAKE_IRON_PICKAXE, "iron_pickaxe", 1))
        return tuple(steps)
    return (_gather_or_explore(snapshot, ActivityName.GET_DIAMOND, "diamond", "diamond", 1),)


def _walk_decision(
    source: Position,
    destination: Position,
    spec: ActivitySpec,
    reason: str,
    target: Position | None = None,
    kind: str | None = None,
) -> ActionDecision:
    direction = Direction.from_delta((destination[0] - source[0], destination[1] - source[1]))
    return ActionDecision(direction.action, "walk", source, destination, target, kind, reason)


def _route_first_step(
    player: Position, destination: Position, predecessors: Mapping[Position, Position]
) -> Position:
    step = destination
    while predecessors[step] != player:
        step = predecessors[step]
    return step


def _face_or_act(
    spec: ActivitySpec,
    state: Mapping[str, Any],
    target: Position,
    kind: str,
    action: CrafterAction,
    reason: str,
) -> ActionDecision:
    player = _position(state["player"])
    direction = Direction.from_delta((target[0] - player[0], target[1] - player[1]))
    facing = Direction.from_name(str(state["facing"]))
    if facing is not direction:
        return ActionDecision(direction.action, "turn", player, target, target, kind, f"face_{reason}")
    return ActionDecision(action, "none", player, target, target, kind, reason)


def _gather_decision(spec: ActivitySpec, state: Mapping[str, Any]) -> ActionDecision:
    rule = ACTIVITY_RULES[spec.activity]
    abstract_kinds = ("food",) if spec.activity is ActivityName.EAT else rule.targets
    candidates = _target_candidates(state, abstract_kinds)
    if not candidates:
        raise ActivityUnavailable("no_reachable_target")
    _, kind, target, approach = candidates[0]
    player = _position(state["player"])
    distances, predecessors = reachable_cells(state)
    if approach != player:
        return _walk_decision(
            player,
            _route_first_step(player, approach, predecessors),
            spec,
            "approach_target",
            target,
            kind,
        )
    return _face_or_act(spec, state, target, kind, CrafterAction.DO, "interact_target")


def _placement_decision(spec: ActivitySpec, state: Mapping[str, Any]) -> ActionDecision:
    station = "table" if spec.activity is ActivityName.PLACE_TABLE else "furnace"
    options = placement_options(state, station)
    if not options:
        raise ActivityUnavailable("no_placement_opportunity")
    player = _position(state["player"])
    facing = Direction.from_name(str(state["facing"]))
    direct = [
        option
        for option in options
        if option[1] == player
        and Direction.from_delta(
            (option[2][0] - player[0], option[2][1] - player[1])
        ) is facing
    ]
    if direct:
        _, _, placement = direct[0]
        return ActionDecision(
            ACTIVITY_RULES[spec.activity].action,
            "none",
            player,
            placement,
            placement,
            station,
            f"place_{station}",
        )
    orientation, anchor, placement = options[0]
    distances, predecessors = reachable_cells(state)
    if player != orientation:
        return _walk_decision(
            player,
            _route_first_step(player, orientation, predecessors),
            spec,
            "approach_placement_orientation",
            placement,
            station,
        )
    return _walk_decision(
        player,
        anchor,
        spec,
        "enter_placement_anchor",
        placement,
        station,
    )


def _craft_decision(spec: ActivitySpec, state: Mapping[str, Any]) -> ActionDecision:
    rule = ACTIVITY_RULES[spec.activity]
    assert rule.station_set is not None and rule.action is not None
    anchors = _craft_anchors(state, rule.station_set)
    if not anchors:
        raise ActivityUnavailable("no_usable_station")
    player = _position(state["player"])
    if player == anchors[0]:
        return ActionDecision(rule.action, "none", player, player, None, None, "craft")
    _, predecessors = reachable_cells(state)
    return _walk_decision(
        player,
        _route_first_step(player, anchors[0], predecessors),
        spec,
        "approach_craft_anchor",
    )


def _sleep_decision(spec: ActivitySpec, state: Mapping[str, Any]) -> ActionDecision:
    player = _position(state["player"])
    energy = int(state.get("inventory", {}).get("energy", 0))
    sleeping = bool(state.get("sleeping", False))
    action = CrafterAction.SLEEP if energy < 9 else CrafterAction.NOOP
    reason = "sleep_tick" if sleeping else "enter_sleep" if action is CrafterAction.SLEEP else "health_recovery_noop"
    return ActionDecision(action, "none", player, player, None, None, reason)


def _explore_decision(spec: ActivitySpec, state: Mapping[str, Any]) -> ActionDecision:
    player = _position(state["player"])
    safe = _state_positions(state.get("safe_cells", ())) | {player}
    known = _state_positions(state.get("known_cells", ()))
    offsets = {
        (int(value[0]), int(value[1]))
        for value in state.get("view_offsets", ())
        if isinstance(value, (tuple, list)) and len(value) == 2
    }
    distances, predecessors = reachable_cells(state)
    frontiers = []
    for candidate in sorted(set(distances) - {player}):
        unknown = sum(
            (candidate[0] + offset[0], candidate[1] + offset[1]) not in known
            for offset in offsets
        )
        if unknown:
            frontiers.append((-unknown, distances[candidate], candidate))
    if frontiers:
        _, _, winner = min(frontiers)
        return _walk_decision(
            player,
            _route_first_step(player, winner, predecessors),
            spec,
            "explore_frontier",
        )

    inventory = state.get("inventory", {})
    terrain = state.get("terrain", {})
    boundaries: list[tuple[int, int, Position, Position]] = []
    for key, material in terrain.items():
        if material not in REMOVABLE_REQUIREMENTS:
            continue
        required = REMOVABLE_REQUIREMENTS[material]
        if required is not None and int(inventory.get(required, 0)) < 1:
            continue
        target = position_from_key(key)
        if not any(
            (target[0] + direction.delta[0], target[1] + direction.delta[1]) not in known
            for direction in Direction
        ):
            continue
        approaches = [
            (target[0] + direction.delta[0], target[1] + direction.delta[1])
            for direction in Direction
            if (target[0] + direction.delta[0], target[1] + direction.delta[1]) in distances
        ]
        if approaches:
            approach = min(approaches, key=lambda cell: (distances[cell], cell))
            boundaries.append((REMOVABLE_ORDER[material], distances[approach], target, approach))
    if not boundaries:
        raise ActivityUnavailable("no_frontier")
    _, _, target, approach = min(boundaries)
    kind = str(terrain[cell_key(target)])
    if player != approach:
        return _walk_decision(
            player,
            _route_first_step(player, approach, predecessors),
            spec,
            "approach_boundary",
            target,
            kind,
        )
    if cell_key(target) not in set(state.get("visible_cells", ())):
        raise ActivityUnavailable("no_frontier")
    return _face_or_act(spec, state, target, kind, CrafterAction.DO, "open_boundary")


def next_activity_action(spec: ActivitySpec, state: Mapping[str, Any]) -> ActionDecision:
    """Select exactly one safe native action from the latest LL beliefs."""

    if not prerequisite_available(spec, state):
        raise ActivityContractError("missing_prerequisite")
    if spec.activity in RESOURCE_ACTIVITIES:
        return _gather_decision(spec, state)
    if spec.activity in PLACE_ACTIVITIES:
        return _placement_decision(spec, state)
    if spec.activity in MAKE_ACTIVITIES:
        return _craft_decision(spec, state)
    if spec.activity is ActivityName.SLEEP:
        return _sleep_decision(spec, state)
    return _explore_decision(spec, state)
