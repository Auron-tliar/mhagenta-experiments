"""Experiment-local symbolic beliefs for the visual Crafter runtime."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from mhagenta import Belief

Position = tuple[int, int]
WALKABLE_MATERIALS = frozenset({"grass", "path", "sand"})
STATIC_TARGETS = frozenset({"water", "tree", "stone", "coal", "iron", "diamond"})
_DELTAS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class CrafterAction(IntEnum):
    """Crafter actions used by policies and HLR-owned primitives."""

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
    """Cardinal map direction and its native Crafter action."""

    LEFT = (-1, 0, CrafterAction.MOVE_LEFT)
    RIGHT = (1, 0, CrafterAction.MOVE_RIGHT)
    UP = (0, -1, CrafterAction.MOVE_UP)
    DOWN = (0, 1, CrafterAction.MOVE_DOWN)

    @property
    def delta(self) -> Position:
        return self.value[0], self.value[1]

    @classmethod
    def from_name(cls, name: str) -> Direction:
        try:
            return cls[name.upper()]
        except KeyError as exc:
            raise BeliefRevisionError(f"Unknown direction {name!r}.") from exc

    @classmethod
    def from_action(cls, action: int) -> Direction | None:
        return next((item for item in cls if int(item.value[2]) == action), None)


class BeliefRevisionError(ValueError):
    """Raised when correlated action evidence contradicts a percept."""


@dataclass(frozen=True)
class ObservedTile:
    """Terrain and occupant at one agent-relative position."""

    material: str
    occupant: str


@dataclass(frozen=True)
class CrafterPercept:
    """Complete deterministic interpretation of one RGB observation."""

    sleeping: bool
    facing: Direction
    inventory: Mapping[str, int]
    tiles: Mapping[Position, ObservedTile]


_CELL_KEY = re.compile(r"^(-?\d+),(-?\d+)$")


def cell_key(position: Position) -> str:
    """Encode an absolute cell for JSON state."""

    return f"{position[0]},{position[1]}"


def position_from_key(key: str) -> Position:
    """Decode a strict JSON cell key."""

    match = _CELL_KEY.fullmatch(key) if isinstance(key, str) else None
    if match is None:
        raise ValueError(f"Invalid cell key {key!r}.")
    return int(match.group(1)), int(match.group(2))


def initial_belief_state() -> dict[str, Any]:
    """Return the one current JSON-safe visual belief state."""

    return {
        "revision": 0, "player": [0, 0], "facing": None, "sleeping": False,
        "inventory": {}, "terrain": {}, "occupants": {}, "visible_cells": [],
        "known_cells": [], "safe_cells": [], "blocked_cells": [],
        "terminal": False, "dead": False, "achievement_counts": {},
        "native_action_count": 0, "completed_goal_id": None, "last_action_status": None,
    }


def _position(value: Any, label: str) -> Position:
    if isinstance(value, str):
        try:
            return position_from_key(value)
        except ValueError as exc:
            raise BeliefRevisionError(f"Invalid {label}.") from exc
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(type(part) is int for part in value):
        return int(value[0]), int(value[1])
    raise BeliefRevisionError(f"Invalid {label}.")


def movement_evidence(action: int, state: Mapping[str, Any]) -> dict[str, Any]:
    """Predict turn versus walk from the current public adjacent tile."""

    source = tuple(state["player"])
    direction = Direction.from_action(action)
    destination = source
    kind = "none"
    if direction is not None:
        candidate = source[0] + direction.delta[0], source[1] + direction.delta[1]
        if cell_key(candidate) in state["safe_cells"]:
            destination, kind = candidate, "walk"
        else:
            kind = "turn"
    return {"action": action, "source_cell": list(source),
            "destination_cell": list(destination), "movement_kind": kind}


def available_movement_actions(
    state: Mapping[str, Any],
    target_cell: Position | None = None,
) -> tuple[int, ...]:
    """Return safe moves plus a needed turn toward an interaction target."""

    source = tuple(state["player"])
    facing = Direction.from_name(state["facing"]).delta
    return tuple(int(direction.value[2]) for direction in Direction if (
        movement_evidence(int(direction.value[2]), state)["movement_kind"] == "walk"
        or (
            target_cell == (
                source[0] + direction.delta[0],
                source[1] + direction.delta[1],
            )
            and facing != direction.delta
        )
    ))


def novel_movement_actions(
    actions: Sequence[int],
    player: Position,
    recent_cells: Iterable[Position],
) -> tuple[int, ...]:
    """Prefer moves outside recent cells, retaining backtracking when it is required."""

    available = tuple(actions)
    recent = set(recent_cells)
    novel = tuple(
        action for action in available
        if (direction := Direction.from_action(action)) is not None
        and (player[0] + direction.delta[0], player[1] + direction.delta[1]) not in recent
    )
    return novel or available


def revise_belief_state(
    previous: Mapping[str, Any],
    percept: CrafterPercept,
    *,
    revision: int,
    pending_action: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Join a fresh RGB percept with its public status and update the partial map."""

    if revision != previous["revision"] + 1 or (revision == 1) != (pending_action is None):
        raise BeliefRevisionError("Observation/action sequence is not contiguous.")
    state = initial_belief_state()
    player = tuple(previous["player"])
    achievements = dict(previous["achievement_counts"])
    if pending_action is not None:
        status = pending_action["status"]
        if status.get("contract_error") is not None:
            raise BeliefRevisionError("Action has a contract error.")
        if tuple(pending_action["source_cell"]) != player:
            raise BeliefRevisionError("Action source contradicts inferred position.")
        if not status["illegal_action"]:
            if pending_action["movement_kind"] == "walk":
                player = tuple(pending_action["destination_cell"])
            direction = Direction.from_action(pending_action["action"])
            if direction is not None and percept.facing is not direction:
                raise BeliefRevisionError("Action and observed facing disagree.")
        for name in status["new_achievements"]:
            achievements[name] = achievements.get(name, 0) + 1
        state["terminal"], state["dead"] = status["done"], status["dead"]
        state["last_action_status"] = dict(status)
        state["native_action_count"] = previous["native_action_count"] + 1
    terrain = dict(previous["terrain"])
    occupants = {}
    visible = {cell_key(player)}
    for relative, tile in percept.tiles.items():
        key = cell_key((player[0] + relative[0], player[1] + relative[1]))
        visible.add(key)
        terrain[key] = tile.material
        if tile.occupant != "none":
            occupants[key] = {"kind": tile.occupant, "last_seen_revision": revision}
    safe = {cell_key(player)}
    blocked = set()
    for key, material in terrain.items():
        (safe if material in WALKABLE_MATERIALS and key not in occupants else blocked).add(key)
    state.update(
        revision=revision, player=list(player), facing=percept.facing.name.lower(),
        sleeping=percept.sleeping, inventory=dict(percept.inventory), terrain=terrain,
        occupants=occupants, visible_cells=sorted(visible),
        known_cells=sorted(set(terrain) | {cell_key(player)}),
        safe_cells=sorted(safe), blocked_cells=sorted(blocked), achievement_counts=achievements,
    )
    return state


def known_targets(state: Mapping[str, Any]) -> dict[str, tuple[Position, ...]]:
    """Return remembered resources and currently grounded cows."""

    targets: dict[str, list[Position]] = {}
    for key, material in state["terrain"].items():
        if material in STATIC_TARGETS or material in {"table", "furnace"}:
            targets.setdefault(material, []).append(position_from_key(key))
    for key, occupant in state["occupants"].items():
        if occupant["kind"] == "cow":
            targets.setdefault("cow", []).append(position_from_key(key))
    return {kind: tuple(sorted(values)) for kind, values in targets.items()}


def _positions(values: Iterable[Position]) -> set[Position]:
    return {_position(value, "cell") for value in values}


def frontier_cells(known_cells: Iterable[Position], safe_cells: Iterable[Position]) -> tuple[Position, ...]:
    """Return safe cells adjacent to at least one unknown cardinal cell."""

    known, safe = _positions(known_cells), _positions(safe_cells)
    return tuple(sorted(cell for cell in safe if any((cell[0] + dx, cell[1] + dy) not in known for dx, dy in _DELTAS)))


def _distances(player: Position, safe_cells: Iterable[Position]) -> dict[Position, int]:
    safe = _positions(safe_cells) | {player}
    distances = {player: 0}
    queue = deque([player])
    while queue:
        cell = queue.popleft()
        for dx, dy in _DELTAS:
            neighbor = cell[0] + dx, cell[1] + dy
            if neighbor in safe and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


def select_frontier(player: Position, safe_cells: Iterable[Position], frontiers: Iterable[Position]) -> Position | None:
    """Select the farthest reachable frontier in a stable visible-map sector."""
    values = _positions(frontiers)
    distances = _distances(player, safe_cells)
    eligible = [cell for cell in values if cell in distances and distances[cell] <= 31]
    if not eligible:
        return None
    direction = _DELTAS[sum(x * 31 + y * 17 for x, y in values) % len(_DELTAS)]
    sector = [cell for cell in eligible if sum((cell[index] - player[index]) * direction[index] for index in range(2)) > 0]
    return min(sector or eligible, key=lambda cell: (-distances[cell], cell))

def select_approach_cell(player: Position, target: Position, safe_cells: Iterable[Position], *, max_distance: int | None = None) -> Position | None:
    """Return the nearest safe target neighbor within an optional route bound."""

    if max_distance is not None and (type(max_distance) is not int or max_distance < 0):
        raise ValueError("max_distance must be a nonnegative integer or None.")
    distances = _distances(player, safe_cells)
    neighbors = ((target[0] + dx, target[1] + dy) for dx, dy in _DELTAS)
    eligible = [cell for cell in neighbors if cell in distances and
                (max_distance is None or distances[cell] <= max_distance)]
    return min(eligible, key=lambda cell: (distances[cell], cell)) if eligible else None

def reachable_cell(player: Position, target: Position, safe_cells: Iterable[Position]) -> bool:
    """Return whether a target is connected to the player through safe cells."""

    return target in _distances(player, safe_cells)

def nearest_target(player: Position, targets: Iterable[Position]) -> Position | None:
    """Return the nearest target with a stable coordinate tie-break."""

    values = _positions(targets)
    return min(
        values,
        key=lambda cell: (abs(cell[0] - player[0]) + abs(cell[1] - player[1]), cell),
    ) if values else None


def abstract_beliefs(state: Mapping[str, Any], completed_goal_id: str | None = None) -> list[Belief]:
    """Publish one complete typed map projection; raw frames remain local to LLR."""

    values = dict(state)
    values["completed_goal_id"] = completed_goal_id
    return [Belief(name, (value,)) for name, value in values.items()]


def parse_abstract_beliefs(beliefs: Sequence[Belief]) -> dict[str, Any]:
    """Recover exactly one complete current projection without legacy branches."""

    values = {}
    for belief in beliefs:
        if belief.predicate in values or not isinstance(belief.arguments, (list, tuple)) or len(belief.arguments) != 1:
            raise ValueError("Malformed or duplicate belief.")
        values[belief.predicate] = belief.arguments[0]
    if set(values) != set(initial_belief_state()) or type(values["revision"]) is not int or values["revision"] < 1:
        raise ValueError("Incomplete belief revision.")
    _position(values["player"], "player")
    Direction.from_name(values["facing"])
    for key in ("sleeping", "terminal", "dead"):
        if type(values[key]) is not bool:
            raise ValueError(f"Invalid {key} belief.")
    return values
