"""Symbolic Crafter belief parsing and inferred-map revision for 2-4-CR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
import re
from typing import Any

from mhagenta import Belief


Position = tuple[int, int]
WALKABLE_MATERIALS = frozenset({"grass", "path", "sand"})
STATIC_TARGETS = frozenset({"water", "tree", "stone", "coal", "iron", "diamond"})
DYNAMIC_TARGETS = frozenset({"ripe-plant", "cow"})
ABSTRACT_TARGET_KINDS = frozenset({"food", *STATIC_TARGETS})
STATION_KINDS = frozenset({"table", "furnace"})
STATION_SETS = frozenset({"table", "table_furnace"})


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
        """Return the direction's map-frame delta."""

        return self.value[0], self.value[1]

    @property
    def action(self) -> CrafterAction:
        """Return the corresponding native Crafter action."""

        return self.value[2]

    @property
    def symbol(self) -> str:
        """Return Crafter's facing symbol."""

        return self.value[3]

    @classmethod
    def from_symbol(cls, symbol: str) -> Direction:
        """Resolve a direction from a symbolic-observation value."""

        for direction in cls:
            if direction.symbol == symbol:
                return direction
        raise ObservationError(f"Unknown facing direction: {symbol!r}")

    @classmethod
    def from_name(cls, name: str) -> Direction:
        """Resolve a direction from its persisted lower-case name."""

        try:
            return cls[name.upper()]
        except (AttributeError, KeyError) as exc:
            raise BeliefRevisionError(f"Unknown direction: {name!r}") from exc

    @classmethod
    def from_action(cls, action: int) -> Direction | None:
        """Resolve a directional action, or return None for other actions."""

        for direction in cls:
            if int(direction.action) == action:
                return direction
        return None

    @classmethod
    def from_delta(cls, delta: Position) -> Direction:
        """Resolve a cardinal direction from a map delta."""

        for direction in cls:
            if direction.delta == delta:
                return direction
        raise BeliefRevisionError(f"Non-cardinal direction delta: {delta!r}")


class ObservationError(ValueError):
    """Raised when a symbolic Crafter percept violates its contract."""


class BeliefRevisionError(ValueError):
    """Raised when correlated action evidence cannot revise beliefs."""


@dataclass(frozen=True)
class ObservedTile:
    """Terrain and occupant observed at one relative position."""

    material: str
    occupant: str


@dataclass(frozen=True)
class CrafterPercept:
    """Validated public symbolic percept in agent-relative coordinates."""

    sleeping: bool
    facing: Direction
    inventory: Mapping[str, int]
    tiles: Mapping[Position, ObservedTile]


@dataclass(frozen=True)
class RevisionResult:
    """Return one revised JSON belief state and action-consumption evidence."""

    belief_state: dict[str, Any]
    consumed_action: bool


@dataclass(frozen=True)
class AbstractState:
    """Coordinate-free high-level projection of one complete LL revision."""

    revision: int
    sleeping: bool
    inventory: Mapping[str, int]
    known_target_kinds: frozenset[str]
    reachable_target_kinds: frozenset[str]
    station_counts: Mapping[str, int]
    usable_station_sets: frozenset[str]
    placement_opportunities: frozenset[str]
    known_cell_count: int
    terminal: bool
    dead: bool
    experiment_error: str | None = None


_PREDICATE = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)\s*=\s*(?P<value>.+)$"
)
_LOCATION = re.compile(r"^(?:(L|R)([1-9]\d*))?(?:_?(U|D)([1-9]\d*))?$")
_CELL_KEY = re.compile(r"^(-?\d+),(-?\d+)$")


def _parse_fluent(fluent: str) -> tuple[str, tuple[str, ...], str]:
    if not isinstance(fluent, str):
        raise ObservationError("Every symbolic fluent must be a string")
    match = _PREDICATE.fullmatch(fluent.strip())
    if match is None:
        raise ObservationError(f"Malformed symbolic fluent: {fluent!r}")
    raw_args = match.group("args").strip()
    arguments = tuple(part.strip() for part in raw_args.split(",")) if raw_args else ()
    if any(not argument for argument in arguments):
        raise ObservationError(f"Malformed argument list: {fluent!r}")
    return match.group("name"), arguments, match.group("value").strip()


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
    """Parse and strictly validate a Crafter fluent-list observation."""

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
            if sleeping is not None or arguments or value not in {"true", "false"}:
                raise ObservationError("Sleeping must be unique, nullary, and Boolean")
            sleeping = value == "true"
        elif name == "Facing":
            if facing is not None or len(arguments) != 1 or value != "true":
                raise ObservationError("Facing must be unique with one true argument")
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
    return CrafterPercept(
        sleeping,
        facing,
        dict(sorted(inventory.items())),
        {
            position: ObservedTile(materials[position], occupants[position])
            for position in sorted(materials)
        },
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


def initial_belief_state() -> dict[str, Any]:
    """Return the complete JSON-safe initial LL belief state."""

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
        "known_cells": [],
        "safe_cells": [],
        "blocked_cells": [],
        "view_offsets": [],
        "terminal": False,
        "dead": False,
    }


def _position(value: Any, label: str) -> Position:
    if isinstance(value, str):
        try:
            return position_from_key(value)
        except ValueError as exc:
            raise BeliefRevisionError(f"Invalid {label}: {value!r}") from exc
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(type(part) is int for part in value)
    ):
        return int(value[0]), int(value[1])
    raise BeliefRevisionError(f"Invalid {label}: {value!r}")


def revise_belief_state(
    previous: Mapping[str, Any],
    percept: CrafterPercept,
    *,
    revision: int,
    pending_action: Mapping[str, Any] | None,
) -> RevisionResult:
    """Revise the partial map using LL-owned pending action evidence."""

    expected = int(previous.get("revision", 0)) + 1
    if type(revision) is not int or revision != expected:
        raise BeliefRevisionError(
            f"Expected observation revision {expected}, got {revision!r}"
        )
    consumed = pending_action is not None
    if revision == 1 and consumed:
        raise BeliefRevisionError("Initial observation cannot consume an action")
    if revision > 1 and not consumed:
        raise BeliefRevisionError("Post-initial observation must consume an action")

    state = dict(previous)
    player = _position(previous.get("player", [0, 0]), "inferred player")
    if consumed:
        assert pending_action is not None
        status = pending_action.get("status")
        if not isinstance(status, Mapping):
            raise BeliefRevisionError("Pending action lacks public status")
        if status.get("illegal_action") is not False:
            raise BeliefRevisionError("Pending action was illegal")
        source = _position(pending_action.get("source_cell"), "source cell")
        destination = _position(pending_action.get("destination_cell"), "destination cell")
        if source != player:
            raise BeliefRevisionError("Action source contradicts inferred player")
        movement_kind = pending_action.get("movement_kind")
        direction = Direction.from_action(int(pending_action["action"]))
        if movement_kind in {"walk", "turn"}:
            if direction is None or sum(abs(a - b) for a, b in zip(source, destination)) != 1:
                raise BeliefRevisionError("Directional action needs a cardinal destination")
            if percept.facing is not direction:
                raise BeliefRevisionError("Observed facing contradicts directional action")
        if movement_kind == "walk":
            if cell_key(destination) not in set(previous.get("safe_cells", ())):
                raise BeliefRevisionError("Walk destination was not known safe")
            player = destination
        elif movement_kind == "turn":
            if cell_key(destination) not in set(previous.get("blocked_cells", ())):
                raise BeliefRevisionError("Turn destination was not a supported blocked cell")
            if pending_action.get("facing_before") == direction.name.lower():
                raise BeliefRevisionError("Same-facing blocked direction is not a turn")
        elif movement_kind != "none":
            raise BeliefRevisionError(f"Unknown movement kind: {movement_kind!r}")

    terrain = dict(previous.get("terrain", {}))
    occupants: dict[str, dict[str, Any]] = {}
    visible: list[str] = []
    for relative, tile in percept.tiles.items():
        absolute = player[0] + relative[0], player[1] + relative[1]
        key = cell_key(absolute)
        visible.append(key)
        terrain[key] = tile.material
        if tile.occupant != "none":
            occupants[key] = {"kind": tile.occupant, "last_seen_revision": revision}

    known = set(terrain) | {cell_key(player)}
    safe = {cell_key(player)}
    blocked: set[str] = set()
    for key, material in terrain.items():
        occupant = occupants.get(key, {}).get("kind", "none")
        if material in WALKABLE_MATERIALS and occupant == "none":
            safe.add(key)
        else:
            blocked.add(key)

    achievements = set(previous.get("achievements", ()))
    terminal = bool(previous.get("terminal", False))
    dead = bool(previous.get("dead", False))
    if pending_action is not None:
        status = pending_action["status"]
        new_achievements = status.get("new_achievements", [])
        if not isinstance(new_achievements, list) or not all(
            isinstance(item, str) for item in new_achievements
        ):
            raise BeliefRevisionError("Invalid new-achievements status")
        achievements.update(new_achievements)
        if type(status.get("done")) is not bool or type(status.get("dead")) is not bool:
            raise BeliefRevisionError("Action status lacks strict terminal flags")
        terminal = status["done"]
        dead = status["dead"]

    state.update(
        {
            "revision": revision,
            "player": [player[0], player[1]],
            "facing": percept.facing.name.lower(),
            "sleeping": percept.sleeping,
            "inventory": dict(sorted(percept.inventory.items())),
            "achievements": sorted(achievements),
            "terrain": dict(sorted(terrain.items())),
            "occupants": dict(sorted(occupants.items())),
            "visible_cells": sorted(visible),
            "known_cells": sorted(known),
            "safe_cells": sorted(safe),
            "blocked_cells": sorted(blocked),
            "view_offsets": [[x, y] for x, y in sorted(percept.tiles)],
            "terminal": terminal,
            "dead": dead,
        }
    )
    return RevisionResult(state, consumed)


def known_targets(state: Mapping[str, Any]) -> dict[str, tuple[Position, ...]]:
    """Return deterministic resource and current food targets from LL state."""

    targets: dict[str, list[Position]] = {}
    for key, material in state.get("terrain", {}).items():
        if material in STATIC_TARGETS:
            targets.setdefault(str(material), []).append(position_from_key(key))
    revision = int(state.get("revision", 0))
    for key, record in state.get("occupants", {}).items():
        kind = record.get("kind") if isinstance(record, Mapping) else None
        if kind in DYNAMIC_TARGETS and int(record.get("last_seen_revision", -1)) == revision:
            targets.setdefault(str(kind), []).append(position_from_key(key))
    return {kind: tuple(sorted(positions)) for kind, positions in sorted(targets.items())}


def abstract_beliefs(
    state: Mapping[str, Any], planning: Mapping[str, Any]
) -> list[Belief]:
    """Project LL state and geometry into deterministic coordinate-free beliefs."""

    required = {
        "known_target_kinds",
        "reachable_target_kinds",
        "station_counts",
        "usable_station_sets",
        "placement_opportunities",
    }
    if set(planning) != required:
        raise BeliefRevisionError("Planning summary has unexpected fields")
    station_counts = planning["station_counts"]
    if not isinstance(station_counts, Mapping) or set(station_counts) != STATION_KINDS:
        raise BeliefRevisionError("Planning summary has invalid station counts")
    beliefs = [Belief("observation_revision", (int(state["revision"]),))]
    beliefs.append(Belief("sleeping", (bool(state["sleeping"]),)))
    beliefs.extend(
        Belief("inventory", (item, int(count)))
        for item, count in sorted(state.get("inventory", {}).items())
    )
    beliefs.extend(
        Belief("known_target_kind", (kind,))
        for kind in planning["known_target_kinds"]
    )
    beliefs.extend(
        Belief("reachable_target_kind", (kind,))
        for kind in planning["reachable_target_kinds"]
    )
    beliefs.extend(
        Belief("station_count", (kind, int(station_counts[kind])))
        for kind in sorted(STATION_KINDS)
    )
    beliefs.extend(
        Belief("usable_station_set", (name,))
        for name in planning["usable_station_sets"]
    )
    beliefs.extend(
        Belief("placement_opportunity", (kind,))
        for kind in planning["placement_opportunities"]
    )
    beliefs.append(Belief("known_cell_count", (len(state.get("known_cells", ())),)))
    beliefs.append(Belief("terminal", (bool(state.get("terminal", False)),)))
    beliefs.append(Belief("dead", (bool(state.get("dead", False)),)))
    return beliefs


def error_beliefs(revision: int, message: str) -> list[Belief]:
    """Create a final revision-bearing experiment-error belief collection."""

    return [Belief("observation_revision", (revision,)), Belief("experiment_error", (message,))]


def _belief_arguments(belief: Belief) -> tuple[Any, ...]:
    if belief.arguments is None:
        return ()
    if isinstance(belief.arguments, (tuple, list)):
        return tuple(belief.arguments)
    return (belief.arguments,)


def parse_abstract_beliefs(beliefs: Sequence[Belief]) -> AbstractState:
    """Strictly parse one complete coordinate-free high-level snapshot."""

    revision: int | None = None
    sleeping: bool | None = None
    inventory: dict[str, int] = {}
    known: set[str] = set()
    reachable: set[str] = set()
    station_counts: dict[str, int] = {}
    station_sets: set[str] = set()
    placements: set[str] = set()
    known_count: int | None = None
    terminal: bool | None = None
    dead: bool | None = None
    experiment_error: str | None = None
    for belief in beliefs:
        arguments = _belief_arguments(belief)
        predicate = belief.predicate
        if predicate == "observation_revision":
            if revision is not None or len(arguments) != 1 or type(arguments[0]) is not int:
                raise ObservationError("Invalid observation_revision belief")
            revision = arguments[0]
        elif predicate == "experiment_error":
            if experiment_error is not None or len(arguments) != 1 or not isinstance(arguments[0], str):
                raise ObservationError("Invalid experiment_error belief")
            experiment_error = arguments[0]
        elif predicate in {"sleeping", "terminal", "dead"}:
            if len(arguments) != 1 or type(arguments[0]) is not bool:
                raise ObservationError(f"Invalid {predicate} belief")
            if predicate == "sleeping":
                if sleeping is not None:
                    raise ObservationError("Duplicate sleeping belief")
                sleeping = arguments[0]
            elif predicate == "terminal":
                if terminal is not None:
                    raise ObservationError("Duplicate terminal belief")
                terminal = arguments[0]
            else:
                if dead is not None:
                    raise ObservationError("Duplicate dead belief")
                dead = arguments[0]
        elif predicate == "inventory":
            if (
                len(arguments) != 2
                or not isinstance(arguments[0], str)
                or type(arguments[1]) is not int
                or arguments[1] < 0
                or arguments[0] in inventory
            ):
                raise ObservationError("Invalid inventory belief")
            inventory[arguments[0]] = arguments[1]
        elif predicate in {"known_target_kind", "reachable_target_kind"}:
            if len(arguments) != 1 or arguments[0] not in ABSTRACT_TARGET_KINDS:
                raise ObservationError(f"Invalid {predicate} belief")
            target = known if predicate == "known_target_kind" else reachable
            if arguments[0] in target:
                raise ObservationError(f"Duplicate {predicate} belief")
            target.add(arguments[0])
        elif predicate == "station_count":
            if (
                len(arguments) != 2
                or arguments[0] not in STATION_KINDS
                or type(arguments[1]) is not int
                or arguments[1] < 0
                or arguments[0] in station_counts
            ):
                raise ObservationError("Invalid station_count belief")
            station_counts[arguments[0]] = arguments[1]
        elif predicate == "usable_station_set":
            if len(arguments) != 1 or arguments[0] not in STATION_SETS or arguments[0] in station_sets:
                raise ObservationError("Invalid usable_station_set belief")
            station_sets.add(arguments[0])
        elif predicate == "placement_opportunity":
            if len(arguments) != 1 or arguments[0] not in STATION_KINDS or arguments[0] in placements:
                raise ObservationError("Invalid placement_opportunity belief")
            placements.add(arguments[0])
        elif predicate == "known_cell_count":
            if known_count is not None or len(arguments) != 1 or type(arguments[0]) is not int or arguments[0] < 0:
                raise ObservationError("Invalid known_cell_count belief")
            known_count = arguments[0]
        else:
            raise ObservationError(f"Unknown abstract belief predicate: {predicate!r}")
    if revision is None or revision < 1:
        raise ObservationError("Missing or invalid observation revision")
    if experiment_error is not None:
        if len(beliefs) != 2:
            raise ObservationError("Experiment-error snapshot contains extra beliefs")
        return AbstractState(revision, False, {}, frozenset(), frozenset(), {}, frozenset(), frozenset(), 0, False, False, experiment_error)
    if None in {sleeping, known_count, terminal, dead} or set(station_counts) != STATION_KINDS:
        raise ObservationError("Incomplete abstract belief snapshot")
    assert sleeping is not None and known_count is not None
    assert terminal is not None and dead is not None
    if not reachable <= known:
        raise ObservationError("Reachable targets must also be known")
    return AbstractState(
        revision,
        sleeping,
        dict(sorted(inventory.items())),
        frozenset(known),
        frozenset(reachable),
        dict(sorted(station_counts.items())),
        frozenset(station_sets),
        frozenset(placements),
        known_count,
        terminal,
        dead,
    )
