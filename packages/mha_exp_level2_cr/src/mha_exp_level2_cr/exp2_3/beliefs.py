"""Symbolic Crafter observation parsing and belief revision for 2-3-CR."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence

from mhagenta import Belief


Position = tuple[int, int]
WALKABLE_MATERIALS = frozenset({"grass", "path", "sand"})
OBSERVATION_OFFSETS = tuple(
    (x, y)
    for x in range(-4, 5)
    for y in range(-3, 4)
)
DIAMOND_MILESTONES = (
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
    "collect_coal",
    "collect_iron",
    "place_furnace",
    "make_iron_pickaxe",
    "collect_diamond",
)


class CrafterAction(IntEnum):
    """Native Crafter action identifiers."""

    NOOP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16


class Direction(Enum):
    """Cardinal direction with map delta, native action, and Crafter symbol."""

    LEFT = (-1, 0, CrafterAction.MOVE_LEFT, "L1")
    RIGHT = (1, 0, CrafterAction.MOVE_RIGHT, "R1")
    UP = (0, -1, CrafterAction.MOVE_UP, "U1")
    DOWN = (0, 1, CrafterAction.MOVE_DOWN, "D1")

    @property
    def delta(self) -> Position:
        return self.value[0], self.value[1]

    @property
    def action(self) -> CrafterAction:
        return self.value[2]

    @property
    def symbol(self) -> str:
        return self.value[3]

    @classmethod
    def from_symbol(cls, symbol: str) -> Direction:
        for direction in cls:
            if direction.symbol == symbol:
                return direction
        raise ObservationError(f"Unknown facing direction: {symbol!r}")

    @classmethod
    def from_action(cls, action: int) -> Direction | None:
        for direction in cls:
            if int(direction.action) == action:
                return direction
        return None


class ObservationError(ValueError):
    """Raised when a symbolic Crafter percept violates its contract."""


class BeliefRevisionError(ValueError):
    """Raised when an action outcome cannot revise beliefs consistently."""


@dataclass(frozen=True)
class ObservedTile:
    """Terrain and occupant observed at one relative position."""

    material: str
    occupant: str


@dataclass(frozen=True)
class CrafterPercept:
    """Validated symbolic percept in agent-relative coordinates."""

    sleeping: bool
    facing: Direction
    inventory: Mapping[str, int]
    tiles: Mapping[Position, ObservedTile]


@dataclass(frozen=True)
class RevisionResult:
    """Revised JSON belief state and monitoring signals."""

    belief_state: dict[str, Any]
    cow_target_contradicted: bool
    consumed_action: bool


_PREDICATE = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)\s*=\s*(?P<value>.+)$"
)
_LOCATION = re.compile(r"^(?:(L|R)([1-9]\d*))?(?:_?(U|D)([1-9]\d*))?$")
_CELL_KEY = re.compile(r"^(-?\d+),(-?\d+)$")
_PDDL_CELL = re.compile(r"^c_([pn])(\d+)_([pn])(\d+)$")


def _parse_fluent(fluent: str) -> tuple[str, tuple[str, ...], str]:
    if not isinstance(fluent, str):
        raise ObservationError("Every symbolic fluent must be a string")
    match = _PREDICATE.fullmatch(fluent.strip())
    if match is None:
        raise ObservationError(f"Malformed symbolic fluent: {fluent!r}")
    raw_args = match.group("args").strip()
    args = tuple(part.strip() for part in raw_args.split(",")) if raw_args else ()
    if any(not argument for argument in args):
        raise ObservationError(f"Malformed argument list: {fluent!r}")
    return match.group("name"), args, match.group("value").strip()


def _parse_location(label: str) -> Position:
    match = _LOCATION.fullmatch(label)
    if match is None or not any(match.groups()):
        raise ObservationError(f"Invalid relative location: {label!r}")
    horizontal, x_value, vertical, y_value = match.groups()
    x = int(x_value or 0) * (-1 if horizontal == "L" else 1)
    y = int(y_value or 0) * (-1 if vertical == "U" else 1)
    if x == 0 and y == 0:
        raise ObservationError("The current player cell must not be observed as a tile")
    return x, y


def parse_symbolic_observation(content: Sequence[str]) -> CrafterPercept:
    """Parse and strictly validate a Crafter symbolic observation."""

    if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
        raise ObservationError("Symbolic observation must be a sequence of strings")
    sleeping: bool | None = None
    facing: Direction | None = None
    inventory: dict[str, int] = {}
    materials: dict[Position, str] = {}
    occupants: dict[Position, str] = {}

    for fluent in content:
        name, arguments, value = _parse_fluent(fluent)
        if name == "Sleeping":
            if sleeping is not None:
                raise ObservationError("Duplicate Sleeping fact")
            if arguments or value not in {"true", "false"}:
                raise ObservationError("Sleeping must have arity zero and Boolean value")
            sleeping = value == "true"
        elif name == "Facing":
            if facing is not None:
                raise ObservationError("Duplicate Facing fact")
            if len(arguments) != 1 or value != "true":
                raise ObservationError("Facing must have one argument and value true")
            facing = Direction.from_symbol(arguments[0])
        elif name == "Have":
            if len(arguments) != 1 or arguments[0] in inventory:
                raise ObservationError("Have must have one unique item argument")
            try:
                count = int(value)
            except ValueError as exc:
                raise ObservationError(f"Invalid inventory count: {value!r}") from exc
            if count < 0:
                raise ObservationError("Inventory count cannot be negative")
            inventory[arguments[0]] = count
        elif name in {"MadeOf", "OccupiedBy"}:
            if len(arguments) != 2 or value != "true":
                raise ObservationError(f"{name} must have two arguments and value true")
            position = _parse_location(arguments[0])
            target = materials if name == "MadeOf" else occupants
            if position in target:
                raise ObservationError(f"Duplicate {name} fact for {arguments[0]}")
            target[position] = arguments[1]
        else:
            raise ObservationError(f"Unknown symbolic predicate: {name}")

    if sleeping is None or facing is None:
        raise ObservationError("Observation must contain Sleeping and Facing")
    if set(materials) != set(occupants):
        raise ObservationError("Every visible location needs matching terrain and occupant")
    tiles = {
        position: ObservedTile(materials[position], occupants[position])
        for position in materials
    }
    return CrafterPercept(sleeping, facing, inventory, tiles)


def percept_to_beliefs(percept: CrafterPercept) -> list[Belief]:
    """Normalize a percept into typed MHAgentA beliefs."""

    beliefs = [
        Belief(predicate="sleeping", arguments=(), extras={"value": percept.sleeping}),
        Belief(predicate="facing", arguments=(percept.facing.name.lower(),)),
    ]
    beliefs.extend(
        Belief(predicate="inventory", arguments=(item,), extras={"value": count})
        for item, count in sorted(percept.inventory.items())
    )
    for (x, y), tile in sorted(percept.tiles.items()):
        beliefs.append(Belief(predicate="terrain", arguments=(x, y, tile.material)))
        beliefs.append(Belief(predicate="occupant", arguments=(x, y, tile.occupant)))
    return beliefs


def _arguments(belief: Belief) -> tuple[Any, ...]:
    if belief.arguments is None:
        return ()
    if isinstance(belief.arguments, (tuple, list)):
        return tuple(belief.arguments)
    return (belief.arguments,)


def beliefs_to_percept(beliefs: Sequence[Belief]) -> CrafterPercept:
    """Reconstruct and validate a percept from normalized beliefs."""

    sleeping: bool | None = None
    facing: Direction | None = None
    inventory: dict[str, int] = {}
    materials: dict[Position, str] = {}
    occupants: dict[Position, str] = {}
    for belief in beliefs:
        arguments = _arguments(belief)
        extras = belief.extras or {}
        if belief.predicate == "sleeping":
            if sleeping is not None or arguments or type(extras.get("value")) is not bool:
                raise ObservationError("Invalid sleeping belief")
            sleeping = extras["value"]
        elif belief.predicate == "facing":
            if facing is not None or len(arguments) != 1 or not isinstance(arguments[0], str):
                raise ObservationError("Invalid facing belief")
            try:
                facing = Direction[arguments[0].upper()]
            except KeyError as exc:
                raise ObservationError(f"Unknown direction: {arguments[0]!r}") from exc
        elif belief.predicate == "inventory":
            if len(arguments) != 1 or not isinstance(arguments[0], str):
                raise ObservationError("Invalid inventory belief")
            value = extras.get("value")
            if type(value) is not int or value < 0 or arguments[0] in inventory:
                raise ObservationError("Invalid inventory value")
            inventory[arguments[0]] = value
        elif belief.predicate in {"terrain", "occupant"}:
            if (
                len(arguments) != 3
                or type(arguments[0]) is not int
                or type(arguments[1]) is not int
                or not isinstance(arguments[2], str)
            ):
                raise ObservationError(f"Invalid {belief.predicate} belief")
            position = arguments[0], arguments[1]
            if position == (0, 0):
                raise ObservationError("Current cell cannot appear in a tile belief")
            target = materials if belief.predicate == "terrain" else occupants
            if position in target:
                raise ObservationError(f"Duplicate {belief.predicate} belief")
            target[position] = arguments[2]
        else:
            raise ObservationError(f"Unknown normalized belief: {belief.predicate!r}")
    if sleeping is None or facing is None or set(materials) != set(occupants):
        raise ObservationError("Incomplete normalized belief set")
    return CrafterPercept(
        sleeping,
        facing,
        inventory,
        {position: ObservedTile(materials[position], occupants[position]) for position in materials},
    )


def cell_key(position: Position) -> str:
    """Encode an absolute cell for JSON state."""

    return f"{position[0]},{position[1]}"


def position_from_key(key: str) -> Position:
    """Decode a strict JSON cell key."""

    match = _CELL_KEY.fullmatch(key) if isinstance(key, str) else None
    if match is None:
        raise ValueError(f"Invalid cell key: {key!r}")
    return int(match.group(1)), int(match.group(2))


def pddl_cell_name(position: Position) -> str:
    """Encode an absolute cell as a PDDL-safe object name."""

    def part(value: int) -> str:
        return f"n{abs(value)}" if value < 0 else f"p{value}"

    return f"c_{part(position[0])}_{part(position[1])}"


def position_from_pddl_cell(name: str) -> Position:
    """Decode a PDDL cell object name."""

    match = _PDDL_CELL.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise ValueError(f"Invalid PDDL cell name: {name!r}")
    x = int(match.group(2)) * (-1 if match.group(1) == "n" else 1)
    y = int(match.group(4)) * (-1 if match.group(3) == "n" else 1)
    return x, y


def initial_belief_state() -> dict[str, Any]:
    """Return the complete JSON-safe initial belief state."""

    return {
        "revision": 0,
        "player": [0, 0],
        "facing": None,
        "sleeping": False,
        "inventory": {},
        "achievements": [],
        "terrain": {},
        "occupants": {},
        "visible_cells": [],
        "known": [],
        "clear": [],
        "safe_to_enter": [],
        "reachable": ["0,0"],
        "movement_blocked": [],
        "visit_counts": {},
        "terminal": False,
        "dead": False,
        "cow_target": None,
    }


def _validate_correlation(
    action_result: Mapping[str, Any] | None,
    pending_action: Mapping[str, Any] | None,
) -> bool:
    if action_result is None and pending_action is None:
        return False
    if action_result is None or pending_action is None:
        raise BeliefRevisionError("Missing pending action or action status")
    if action_result.get("action_id") != pending_action.get("action_id"):
        raise BeliefRevisionError("Action correlation mismatch: action_id")
    if action_result.get("illegal_action"):
        raise BeliefRevisionError("Action status reported illegal_action")
    return True


def revise_belief_state(
    previous: Mapping[str, Any],
    percept: CrafterPercept,
    *,
    action_result: Mapping[str, Any] | None,
    pending_action: Mapping[str, Any] | None,
) -> RevisionResult:
    """Revise the absolute partial map from one correlated percept."""

    consumed = _validate_correlation(action_result, pending_action)
    state = dict(previous)
    state["revision"] = int(previous.get("revision", 0)) + 1
    previous_player = previous.get("player", (0, 0))
    player = int(previous_player[0]), int(previous_player[1])

    if consumed:
        assert action_result is not None and pending_action is not None
        movement_kind = pending_action.get("movement_kind")
        if movement_kind == "walk":
            walk_source = position_from_key(
                str(pending_action.get("source_cell"))
            )
            destination = position_from_key(str(pending_action.get("destination_cell")))
            if walk_source != player or sum(
                abs(a - b) for a, b in zip(walk_source, destination)
            ) != 1:
                raise BeliefRevisionError("Invalid contextual walk localization")
            player = destination
        elif movement_kind not in {"turn", "none"}:
            raise BeliefRevisionError(f"Unknown movement kind: {movement_kind!r}")
        direction = Direction.from_action(int(pending_action["action"]))
        if movement_kind in {"walk", "turn"} and direction is not percept.facing:
            raise BeliefRevisionError("Observed facing contradicts contextual movement")

    state["player"] = [player[0], player[1]]
    state["facing"] = percept.facing.name.lower()
    state["sleeping"] = percept.sleeping
    state["inventory"] = dict(sorted(percept.inventory.items()))
    terrain = dict(previous.get("terrain", {}))
    previous_target = previous.get("cow_target")
    occupants: dict[str, dict[str, Any]] = {}
    visible: list[str] = []
    for relative, tile in percept.tiles.items():
        absolute = player[0] + relative[0], player[1] + relative[1]
        key = cell_key(absolute)
        visible.append(key)
        terrain[key] = tile.material
        if tile.occupant == "none":
            occupants.pop(key, None)
        else:
            occupants[key] = {
                "kind": tile.occupant,
                "last_seen_revision": state["revision"],
            }
    achievements = set(previous.get("achievements", ()))
    new_achievements: list[str] = []
    if action_result is not None:
        new_achievements = action_result.get("new_achievements", [])
        if not isinstance(new_achievements, list) or not all(
            isinstance(item, str) for item in new_achievements
        ):
            raise BeliefRevisionError("Invalid new-achievements status")
        achievements.update(new_achievements)
        state["terminal"] = bool(action_result.get("done", False))
        state["dead"] = bool(action_result.get("dead", False))
    state["achievements"] = sorted(achievements)

    target = dict(previous_target) if isinstance(previous_target, Mapping) else None
    contradicted = False
    if target is not None:
        target_cell = target["cell"]
        if "eat_cow" in new_achievements:
            occupants.pop(target_cell, None)
            target = None
        elif target_cell in visible:
            observed_kind = occupants.get(target_cell, {}).get("kind", "none")
            if observed_kind != "cow":
                target = None
                contradicted = True
            elif consumed and pending_action is not None:
                operator = pending_action.get("operator", {})
                cow_actions = {"hit-cow-unhurt", "hit-cow-once", "finish-cow"}
                if (
                    pending_action.get("destination_cell") == target_cell
                    and isinstance(operator, Mapping)
                    and operator.get("name") in cow_actions
                ):
                    target["hits"] = min(2, int(target.get("hits", 0)) + 1)
        else:
            target = None
    state["cow_target"] = target

    state["terrain"] = dict(sorted(terrain.items()))
    state["occupants"] = dict(sorted(occupants.items()))
    state["visible_cells"] = sorted(visible)
    player_key = cell_key(player)
    known = set(previous.get("known", ())) | set(visible) | {player_key}
    clear: set[str] = set()
    safe: set[str] = set()
    blocked: set[str] = set()
    for key, material in terrain.items():
        occupant = occupants.get(key, {}).get("kind", "none")
        if occupant == "none":
            clear.add(key)
        if material in WALKABLE_MATERIALS and occupant == "none":
            safe.add(key)
        elif material not in {"lava", "unknown"}:
            blocked.add(key)
    state["known"] = sorted(known)
    state["clear"] = sorted(clear)
    state["safe_to_enter"] = sorted(safe)
    state["movement_blocked"] = sorted(blocked)
    reachable = {player_key}
    frontier = deque([player_key])
    while frontier:
        source_key = frontier.popleft()
        x, y = position_from_key(source_key)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = cell_key((x + dx, y + dy))
            if neighbor in safe and neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    state["reachable"] = sorted(reachable)
    visit_counts = {
        str(key): int(value)
        for key, value in previous.get("visit_counts", {}).items()
    }
    visit_counts[player_key] = visit_counts.get(player_key, 0) + 1
    state["visit_counts"] = dict(sorted(visit_counts.items()))
    return RevisionResult(state, contradicted, consumed)


def summarize_belief_state(
    belief_state: Mapping[str, Any],
    *,
    cow_target_contradicted: bool = False,
) -> dict[str, Any]:
    """Return compact open-world evidence for one BDI revision."""

    player_value = belief_state.get("player", (0, 0))
    player = int(player_value[0]), int(player_value[1])
    terrain = belief_state.get("terrain", {})
    known = set(belief_state.get("known", ()))
    anchors = set(belief_state.get("reachable", ())) or {cell_key(player)}
    unknown_frontier: set[str] = set()
    for anchor in anchors:
        x, y = position_from_key(anchor)
        for dx, dy in OBSERVATION_OFFSETS:
            neighbor = cell_key((x + dx, y + dy))
            if neighbor not in known:
                unknown_frontier.add(neighbor)
    inventory = belief_state.get("inventory", {})
    achievements = list(belief_state.get("achievements", ()))
    attained = set(achievements)
    highest = next(
        (name for name in reversed(DIAMOND_MILESTONES) if name in attained),
        None,
    )
    remembered = {
        material: sum(value == material for value in terrain.values())
        for material in (
            "water",
            "tree",
            "stone",
            "coal",
            "iron",
            "diamond",
            "table",
            "furnace",
        )
    }
    return {
        "player": list(player),
        "inventory": {
            name: int(inventory.get(name, 0))
            for name in (
                "wood",
                "stone",
                "coal",
                "iron",
                "diamond",
                "wood_pickaxe",
                "stone_pickaxe",
                "iron_pickaxe",
            )
        },
        "sleeping": bool(belief_state.get("sleeping")),
        "known_terrain": len(known),
        "reachable_cells": len(anchors),
        "unknown_frontier": len(unknown_frontier),
        "remembered_locations": remembered,
        "achievements": achievements,
        "highest_milestone": highest,
        "needs": {
            name: int(inventory.get(name, 0))
            for name in ("health", "food", "drink", "energy")
        },
        "cow_target_contradicted": cow_target_contradicted,
    }


def support_signature(belief_state: Mapping[str, Any], intention_name: str) -> str:
    """Return a compact stable signature of current planning support."""

    inventory = belief_state.get("inventory", {})
    relevant_materials = {
        "water",
        "tree",
        "stone",
        "coal",
        "iron",
        "diamond",
        "table",
        "furnace",
    }
    occupants = {
        key: value.get("kind")
        for key, value in sorted(belief_state.get("occupants", {}).items())
        if value.get("kind") in {"ripe-plant", "cow"}
    }
    target = belief_state.get("cow_target")
    cow_target = (
        {"cell": target.get("cell"), "hits": int(target.get("hits", 0))}
        if isinstance(target, Mapping)
        else None
    )
    payload = {
        "intention": intention_name,
        "terrain": {
            key: value
            for key, value in sorted(belief_state.get("terrain", {}).items())
            if value in relevant_materials
        },
        "occupants": occupants,
        "inventory": {
            key: int(inventory.get(key, 0))
            for key in (
                "health",
                "food",
                "drink",
                "energy",
                "wood",
                "stone",
                "coal",
                "iron",
                "diamond",
                "wood_pickaxe",
                "stone_pickaxe",
                "iron_pickaxe",
            )
        },
        "reachable": sorted(belief_state.get("reachable", ())),
        "sleeping": bool(belief_state.get("sleeping")),
        "cow_target": cow_target,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


__all__ = [
    "BeliefRevisionError",
    "CrafterAction",
    "CrafterPercept",
    "Direction",
    "DIAMOND_MILESTONES",
    "OBSERVATION_OFFSETS",
    "ObservationError",
    "ObservedTile",
    "Position",
    "RevisionResult",
    "WALKABLE_MATERIALS",
    "beliefs_to_percept",
    "cell_key",
    "initial_belief_state",
    "parse_symbolic_observation",
    "pddl_cell_name",
    "percept_to_beliefs",
    "position_from_key",
    "position_from_pddl_cell",
    "revise_belief_state",
    "summarize_belief_state",
    "support_signature",
]
