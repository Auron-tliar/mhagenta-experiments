from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import random
import re
from typing import Iterable, Mapping, Sequence


Position = tuple[int, int]
RESET_ACTION = "reset"
ACTIONS = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do",
    "sleep", "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)

MIN_DRINK, MIN_FOOD, MIN_ENERGY = 2, 3, 1
WALKABLE_MATERIALS = frozenset({"grass", "path", "sand"})


class Direction(Enum):
    LEFT = (-1, 0, "move_left", "L1")
    RIGHT = (1, 0, "move_right", "R1")
    UP = (0, -1, "move_up", "U1")
    DOWN = (0, 1, "move_down", "D1")

    @property
    def delta(self) -> Position:
        return self.value[0], self.value[1]

    @property
    def action(self) -> str:
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


@dataclass(frozen=True)
class Tile:
    material: str
    occupant: str


@dataclass(frozen=True)
class ParsedObservation:
    sleeping: bool
    facing: Direction
    inventory: Mapping[str, int]
    tiles: Mapping[Position, Tile]


@dataclass(frozen=True)
class Route:
    steps: tuple[Position, ...]
    final_facing: Direction | None = None
    target: Position | None = None
    move_cost: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str | None = None


class ObservationError(ValueError):
    pass


class ReactiveCrafterPolicy:
    """World-stateless symbolic Crafter policy; only the seeded RNG persists."""

    _PREDICATE = re.compile(
        r"^(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)\s*=\s*(?P<value>.+)$"
    )

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, predicates: Sequence[str]) -> PolicyDecision:
        obs = self._parse_observation(predicates)
        if obs.inventory["diamond"] >= 1:
            return PolicyDecision(RESET_ACTION, "diamond collected")
        if obs.sleeping:
            return PolicyDecision("noop", reason="sleeping")

        if obs.inventory["iron_pickaxe"] >= 1:
            diamond_action = self._diamond_action(obs)
            if diamond_action is not None:
                return diamond_action

        low_needs = (
            obs.inventory["drink"] < MIN_DRINK
            or obs.inventory["food"] < MIN_FOOD
            or obs.inventory["energy"] < MIN_ENERGY
        )
        if low_needs:
            action = self._self_care_action(obs)
            return action if action is not None else self._fallback_action(obs)

        facing_tile = obs.tiles.get(self._facing_position(obs))
        if (
            obs.inventory["drink"] < 8
            and facing_tile is not None
            and facing_tile.material == "water"
        ):
            return PolicyDecision("do", reason="top up drink")

        if obs.inventory["wood_pickaxe"] < 1:
            action = self._wood_pickaxe_action(obs)
        elif obs.inventory["stone_pickaxe"] < 1:
            action = self._stone_pickaxe_action(obs)
        elif obs.inventory["iron_pickaxe"] < 1:
            action = self._iron_pickaxe_action(obs)
        else:
            action = None
        return action if action is not None else self._fallback_action(obs)

    def _parse_observation(self, predicates: Sequence[str]) -> ParsedObservation:
        if isinstance(predicates, (str, bytes)) or not isinstance(predicates, Sequence):
            raise ObservationError("Symbolic observation must be a sequence of strings")
        sleeping: bool | None = None
        facing: Direction | None = None
        inventory: dict[str, int] = {}
        materials: dict[Position, str] = {}
        occupants: dict[Position, str] = {}

        for fluent in predicates:
            name, args, value = self._parse_fluent(fluent)
            if name == "Sleeping":
                if args or value not in {"true", "false"}:
                    raise ObservationError("Sleeping must have arity zero and Boolean value")
                sleeping = value == "true"
            elif name == "Facing":
                if len(args) != 1 or value != "true":
                    raise ObservationError("Facing must have one argument and value true")
                facing = Direction.from_symbol(args[0])
            elif name == "Have":
                if len(args) != 1:
                    raise ObservationError("Have must have one argument")
                try:
                    count = int(value)
                except ValueError as exc:
                    raise ObservationError(f"Invalid inventory count: {value!r}") from exc
                if count < 0:
                    raise ObservationError("Inventory count cannot be negative")
                inventory[args[0]] = count
            elif name in {"MadeOf", "OccupiedBy"}:
                if len(args) != 2 or value != "true":
                    raise ObservationError(f"{name} must have two arguments and value true")
                position = self._parse_location(args[0])
                target = materials if name == "MadeOf" else occupants
                if position in target:
                    raise ObservationError(f"Duplicate {name} fact for {args[0]}")
                target[position] = args[1]
            else:
                raise ObservationError(f"Unknown symbolic predicate: {name}")

        return self._validate_observation(
            sleeping=sleeping,
            facing=facing,
            inventory=inventory,
            materials=materials,
            occupants=occupants,
        )

    @classmethod
    def _parse_fluent(cls, fluent: str) -> tuple[str, tuple[str, ...], str]:
        if not isinstance(fluent, str):
            raise ObservationError("Every symbolic fluent must be a string")
        match = cls._PREDICATE.fullmatch(fluent.strip())
        if match is None:
            raise ObservationError(f"Malformed symbolic fluent: {fluent!r}")
        raw_args = match.group("args").strip()
        args = tuple(part.strip() for part in raw_args.split(",")) if raw_args else ()
        if any(not arg for arg in args):
            raise ObservationError(f"Malformed argument list: {fluent!r}")
        return match.group("name"), args, match.group("value").strip()

    @staticmethod
    def _parse_location(label: str) -> Position:
        if not label:
            raise ObservationError("Tile location cannot be empty")
        x = y = 0
        seen_x = seen_y = False
        for component in label.split("_"):
            match = re.fullmatch(r"([LRUD])([1-9]\d*)", component)
            if match is None:
                raise ObservationError(f"Invalid tile location: {label!r}")
            axis, distance_text = match.groups()
            distance = int(distance_text)
            if axis in {"L", "R"}:
                if seen_x:
                    raise ObservationError(f"Duplicate horizontal component: {label!r}")
                seen_x = True
                x = -distance if axis == "L" else distance
            else:
                if seen_y:
                    raise ObservationError(f"Duplicate vertical component: {label!r}")
                seen_y = True
                y = -distance if axis == "U" else distance
        if (x, y) == (0, 0):
            raise ObservationError("The player origin must not be encoded as a tile fact")
        return x, y

    @staticmethod
    def _validate_observation(
        *,
        sleeping: bool | None,
        facing: Direction | None,
        inventory: dict[str, int],
        materials: dict[Position, str],
        occupants: dict[Position, str],
    ) -> ParsedObservation:
        if sleeping is None:
            raise ObservationError("Observation is missing Sleeping")
        if facing is None:
            raise ObservationError("Observation is missing Facing")
        required_inventory = {
            "health", "food", "drink", "energy", "wood", "stone", "coal",
            "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
        }
        missing_items = required_inventory - inventory.keys()
        if missing_items:
            raise ObservationError(f"Observation is missing inventory items: {sorted(missing_items)}")
        if materials.keys() != occupants.keys():
            missing_material = sorted(occupants.keys() - materials.keys())
            missing_occupant = sorted(materials.keys() - occupants.keys())
            raise ObservationError(
                "Tile material/occupant facts do not match: "
                f"missing_material={missing_material}, missing_occupant={missing_occupant}"
            )
        if not materials:
            raise ObservationError("Observation contains no visible tiles")
        tiles = {
            position: Tile(material, occupants[position])
            for position, material in materials.items()
        }
        tiles[(0, 0)] = Tile("player-ground", "player")
        return ParsedObservation(sleeping, facing, dict(inventory), tiles)

    @staticmethod
    def _add(position: Position, delta: Position) -> Position:
        return position[0] + delta[0], position[1] + delta[1]

    @classmethod
    def _neighbors4(cls, position: Position) -> tuple[Position, ...]:
        return tuple(cls._add(position, direction.delta) for direction in Direction)

    @staticmethod
    def _neighbors8(position: Position) -> tuple[Position, ...]:
        x, y = position
        return tuple(
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        )

    @staticmethod
    def _direction_between(source: Position, target: Position) -> Direction:
        delta = target[0] - source[0], target[1] - source[1]
        for direction in Direction:
            if direction.delta == delta:
                return direction
        raise ValueError(f"Positions are not cardinal neighbors: {source}, {target}")

    @classmethod
    def _facing_position(cls, obs: ParsedObservation) -> Position:
        return cls._add((0, 0), obs.facing.delta)

    @staticmethod
    def _is_empty(obs: ParsedObservation, position: Position) -> bool:
        if position == (0, 0):
            return True
        tile = obs.tiles.get(position)
        return bool(
            tile is not None
            and tile.material in WALKABLE_MATERIALS
            and tile.occupant in {"none", "player"}
        )

    @staticmethod
    def _is_emptiable(obs: ParsedObservation, position: Position) -> bool:
        tile = obs.tiles.get(position)
        if tile is None or tile.occupant != "none":
            return False
        if tile.material == "tree":
            return True
        if tile.material in {"stone", "coal"}:
            return obs.inventory["wood_pickaxe"] >= 1
        if tile.material == "iron":
            return obs.inventory["stone_pickaxe"] >= 1
        if tile.material == "diamond":
            return obs.inventory["iron_pickaxe"] >= 1
        return False

    @classmethod
    def _is_traversable(cls, obs: ParsedObservation, position: Position) -> bool:
        return cls._is_empty(obs, position) or cls._is_emptiable(obs, position)

    @classmethod
    def _is_non_blocking(cls, obs: ParsedObservation, candidate: Position) -> bool:
        neighbors = [
            position for position in cls._neighbors4(candidate)
            if position in obs.tiles and cls._is_traversable(obs, position)
        ]
        if len(neighbors) <= 1:
            return True
        visited = {neighbors[0]}
        queue = deque([neighbors[0]])
        while queue:
            current = queue.popleft()
            for neighbor in cls._neighbors4(current):
                if neighbor == candidate or neighbor in visited or neighbor not in obs.tiles:
                    continue
                if cls._is_traversable(obs, neighbor):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return all(neighbor in visited for neighbor in neighbors)

    @classmethod
    def _reachable_positions(cls, obs: ParsedObservation) -> set[Position]:
        visited = {(0, 0)}
        queue = deque([(0, 0)])
        while queue:
            current = queue.popleft()
            for neighbor in cls._neighbors4(current):
                if neighbor in visited or neighbor not in obs.tiles:
                    continue
                if cls._is_traversable(obs, neighbor):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _shortest_route_to_cells(
        self, obs: ParsedObservation, targets: Iterable[Position]
    ) -> Route | None:
        goals = {
            target for target in targets
            if target in obs.tiles and self._is_traversable(obs, target)
        }
        if not goals:
            return None
        if (0, 0) in goals:
            return Route((), target=(0, 0), move_cost=0)
        parents: dict[Position, Position | None] = {(0, 0): None}
        distances: dict[Position, int] = {(0, 0): 0}
        queue = deque([(0, 0)])
        found_distance: int | None = None
        found: list[Position] = []
        while queue:
            current = queue.popleft()
            distance = distances[current]
            if found_distance is not None and distance >= found_distance:
                continue
            for neighbor in self._neighbors4(current):
                if neighbor in parents or neighbor not in obs.tiles:
                    continue
                if not self._is_traversable(obs, neighbor):
                    continue
                parents[neighbor] = current
                distances[neighbor] = distance + 1
                if neighbor in goals:
                    found_distance = distance + 1
                    found.append(neighbor)
                else:
                    queue.append(neighbor)
        if not found:
            return None
        endpoint = self._rng.choice(found)
        reversed_steps: list[Position] = []
        current = endpoint
        while current != (0, 0):
            reversed_steps.append(current)
            parent = parents[current]
            assert parent is not None
            current = parent
        steps = tuple(reversed(reversed_steps))
        return Route(steps, target=endpoint, move_cost=len(steps))

    def _route_to_face_targets(
        self, obs: ParsedObservation, targets: Iterable[Position]
    ) -> Route | None:
        candidates: list[Route] = []
        for target in targets:
            if target not in obs.tiles:
                continue
            for approach in self._neighbors4(target):
                route = self._shortest_route_to_cells(obs, [approach])
                if route is None:
                    continue
                desired = self._direction_between(approach, target)
                resulting = (
                    self._direction_between(
                        route.steps[-2] if len(route.steps) > 1 else (0, 0),
                        route.steps[-1],
                    )
                    if route.steps else obs.facing
                )
                candidates.append(
                    Route(
                        route.steps,
                        final_facing=desired,
                        target=target,
                        move_cost=len(route.steps) + int(resulting is not desired),
                    )
                )
        if not candidates:
            return None
        minimum = min(route.move_cost for route in candidates)
        return self._rng.choice([route for route in candidates if route.move_cost == minimum])

    def _route_near_stations(
        self, obs: ParsedObservation, stations: Iterable[Position]
    ) -> Route | None:
        candidates = [
            route for station in stations
            if (route := self._shortest_route_to_cells(obs, self._neighbors8(station))) is not None
        ]
        if not candidates:
            return None
        minimum = min(route.move_cost for route in candidates)
        return self._rng.choice([route for route in candidates if route.move_cost == minimum])

    @classmethod
    def _first_route_action(cls, obs: ParsedObservation, route: Route) -> str | None:
        if route.steps:
            next_position = route.steps[0]
            direction = cls._direction_between((0, 0), next_position)
            if cls._is_empty(obs, next_position):
                return direction.action
            if cls._is_emptiable(obs, next_position):
                return "do" if obs.facing is direction else direction.action
            return None
        if route.final_facing is not None and obs.facing is not route.final_facing:
            return route.final_facing.action
        return None

    @staticmethod
    def _station_positions(obs: ParsedObservation, material: str) -> tuple[Position, ...]:
        return tuple(p for p, tile in obs.tiles.items() if tile.material == material)

    @classmethod
    def _shared_adjacent_cells(cls, first: Position, second: Position) -> set[Position]:
        return set(cls._neighbors8(first)) & set(cls._neighbors8(second))

    def _reachable_coupling_tiles(
        self, obs: ParsedObservation, table: Position, furnace: Position
    ) -> tuple[Position, ...]:
        reachable = self._reachable_positions(obs)
        return tuple(
            p for p in self._shared_adjacent_cells(table, furnace)
            if p in reachable
        )

    def _is_valid_station_site(
        self,
        obs: ParsedObservation,
        position: Position,
        reachable: set[Position] | None = None,
    ) -> bool:
        reachable = self._reachable_positions(obs) if reachable is None else reachable
        return bool(
            position != (0, 0) and position in obs.tiles
            and self._is_traversable(obs, position)
            and self._is_non_blocking(obs, position)
            and any(neighbor in reachable for neighbor in self._neighbors4(position))
        )

    @staticmethod
    def _with_station(
        obs: ParsedObservation, position: Position, station: str
    ) -> ParsedObservation:
        tiles = dict(obs.tiles)
        tiles[position] = Tile(station, "none")
        return replace(obs, tiles=tiles)

    def _all_station_sites(self, obs: ParsedObservation) -> tuple[Position, ...]:
        reachable = self._reachable_positions(obs)
        return tuple(
            position
            for position in obs.tiles
            if position != (0, 0)
            and self._is_traversable(obs, position)
            and self._is_non_blocking(obs, position)
            and any(neighbor in reachable for neighbor in self._neighbors4(position))
        )

    def _coupled_placement_candidates(
        self, obs: ParsedObservation, new_station: str, anchor: Position
    ) -> tuple[Position, ...]:
        candidates: list[Position] = []
        reachable = self._reachable_positions(obs)
        nearby_sites = (
            position
            for position in obs.tiles
            if position != anchor
            and abs(position[0] - anchor[0]) <= 2
            and abs(position[1] - anchor[1]) <= 2
        )
        for candidate in nearby_sites:
            if not self._is_valid_station_site(obs, candidate, reachable):
                continue
            hypothetical = self._with_station(obs, candidate, new_station)
            table, furnace = (
                (candidate, anchor) if new_station == "table" else (anchor, candidate)
            )
            if self._reachable_coupling_tiles(hypothetical, table, furnace):
                candidates.append(candidate)
        return tuple(candidates)

    def _future_table_furnace_candidates(self, obs: ParsedObservation) -> tuple[Position, ...]:
        candidates: list[Position] = []
        for furnace in self._all_station_sites(obs):
            hypothetical = self._with_station(obs, furnace, "furnace")
            if self._coupled_placement_candidates(hypothetical, "table", furnace):
                candidates.append(furnace)
        return tuple(candidates)

    def _approach_and_do(
        self, obs: ParsedObservation, targets: Iterable[Position], reason: str
    ) -> PolicyDecision | None:
        route = self._route_to_face_targets(obs, targets)
        if route is None:
            return None
        return PolicyDecision(self._first_route_action(obs, route) or "do", reason)

    def _approach_station_and_craft(
        self,
        obs: ParsedObservation,
        stations: Iterable[Position],
        craft_action: str,
        reason: str,
    ) -> PolicyDecision | None:
        route = self._route_near_stations(obs, stations)
        if route is None:
            return None
        return PolicyDecision(self._first_route_action(obs, route) or craft_action, reason)

    def _prepare_and_place(
        self,
        obs: ParsedObservation,
        candidates: Iterable[Position],
        place_action: str,
        reason: str,
    ) -> PolicyDecision | None:
        route = self._route_to_face_targets(obs, candidates)
        if route is None or route.target is None:
            return None
        action = self._first_route_action(obs, route)
        if action is not None:
            return PolicyDecision(action, reason)
        if self._is_empty(obs, route.target):
            return PolicyDecision(place_action, reason)
        if self._is_emptiable(obs, route.target):
            return PolicyDecision("do", f"clear site for {reason}")
        return None

    def _self_care_action(self, obs: ParsedObservation) -> PolicyDecision | None:
        if obs.inventory["drink"] < MIN_DRINK:
            water = [p for p, tile in obs.tiles.items() if tile.material == "water"]
            if (action := self._approach_and_do(obs, water, "seek water")) is not None:
                return action
        if obs.inventory["food"] < MIN_FOOD:
            cows = [p for p, tile in obs.tiles.items() if tile.occupant == "cow"]
            if (action := self._approach_and_do(obs, cows, "seek cow")) is not None:
                return action
        if obs.inventory["energy"] < MIN_ENERGY:
            return PolicyDecision("sleep", "restore energy")
        return None

    def _wood_pickaxe_action(self, obs: ParsedObservation) -> PolicyDecision | None:
        if obs.inventory["wood"] >= 1:
            action = self._approach_station_and_craft(
                obs, self._station_positions(obs, "table"),
                "make_wood_pickaxe", "make wood pickaxe",
            )
            if action is not None:
                return action
        if obs.inventory["wood"] >= 3:
            action = self._prepare_and_place(
                obs, self._all_station_sites(obs), "place_table", "place table"
            )
            if action is not None:
                return action
        trees = [p for p, tile in obs.tiles.items() if tile.material == "tree"]
        return self._approach_and_do(obs, trees, "collect wood")

    def _stone_pickaxe_action(self, obs: ParsedObservation) -> PolicyDecision | None:
        if obs.inventory["wood"] >= 1 and obs.inventory["stone"] >= 1:
            action = self._approach_station_and_craft(
                obs, self._station_positions(obs, "table"),
                "make_stone_pickaxe", "make stone pickaxe",
            )
            if action is not None:
                return action
        if obs.inventory["wood"] >= 3 and obs.inventory["stone"] >= 1:
            action = self._prepare_and_place(
                obs, self._all_station_sites(obs), "place_table", "place table"
            )
            if action is not None:
                return action
        if obs.inventory["stone"] < 1:
            stone = [p for p, tile in obs.tiles.items() if tile.material == "stone"]
            if (action := self._approach_and_do(obs, stone, "collect stone")) is not None:
                return action
        if obs.inventory["wood"] < 3:
            trees = [p for p, tile in obs.tiles.items() if tile.material == "tree"]
            return self._approach_and_do(obs, trees, "collect wood")
        return None

    def _existing_coupling_tiles(self, obs: ParsedObservation) -> tuple[Position, ...]:
        coupling: set[Position] = set()
        for table in self._station_positions(obs, "table"):
            for furnace in self._station_positions(obs, "furnace"):
                coupling.update(self._reachable_coupling_tiles(obs, table, furnace))
        return tuple(coupling)

    def _candidate_sites_for_existing_station(
        self, obs: ParsedObservation, *, anchor_material: str, new_station: str
    ) -> tuple[Position, ...]:
        candidates: set[Position] = set()
        for anchor in self._station_positions(obs, anchor_material):
            if self._route_near_stations(obs, [anchor]) is not None:
                candidates.update(self._coupled_placement_candidates(obs, new_station, anchor))
        return tuple(candidates)

    def _iron_pickaxe_action(self, obs: ParsedObservation) -> PolicyDecision | None:
        have_items = (
            obs.inventory["wood"] >= 1 and obs.inventory["coal"] >= 1
            and obs.inventory["iron"] >= 1
        )
        coupling = self._existing_coupling_tiles(obs)
        if have_items and coupling:
            route = self._shortest_route_to_cells(obs, coupling)
            if route is not None:
                return PolicyDecision(
                    self._first_route_action(obs, route) or "make_iron_pickaxe",
                    "make iron pickaxe",
                )

        table_sites = self._candidate_sites_for_existing_station(
            obs, anchor_material="furnace", new_station="table"
        )
        if (
            obs.inventory["wood"] >= 3 and obs.inventory["coal"] >= 1
            and obs.inventory["iron"] >= 1 and table_sites
        ):
            return self._prepare_and_place(
                obs, table_sites, "place_table", "place coupled table"
            )

        furnace_sites = self._candidate_sites_for_existing_station(
            obs, anchor_material="table", new_station="furnace"
        )
        if (
            obs.inventory["wood"] >= 1 and obs.inventory["stone"] >= 4
            and obs.inventory["coal"] >= 1 and obs.inventory["iron"] >= 1
            and furnace_sites
        ):
            return self._prepare_and_place(
                obs, furnace_sites, "place_furnace", "place coupled furnace"
            )

        if (
            obs.inventory["wood"] >= 3 and obs.inventory["stone"] >= 4
            and obs.inventory["coal"] >= 1 and obs.inventory["iron"] >= 1
        ):
            action = self._prepare_and_place(
                obs, self._future_table_furnace_candidates(obs),
                "place_furnace", "place future-coupled furnace",
            )
            if action is not None:
                return action

        if obs.inventory["iron"] < 1:
            iron = [p for p, tile in obs.tiles.items() if tile.material == "iron"]
            if (action := self._approach_and_do(obs, iron, "collect iron")) is not None:
                return action
        if obs.inventory["coal"] < 1:
            coal = [p for p, tile in obs.tiles.items() if tile.material == "coal"]
            if (action := self._approach_and_do(obs, coal, "collect coal")) is not None:
                return action
        if obs.inventory["wood"] < 1:
            trees = [p for p, tile in obs.tiles.items() if tile.material == "tree"]
            if (action := self._approach_and_do(obs, trees, "collect wood")) is not None:
                return action
        if not table_sites and obs.inventory["stone"] < 4:
            stone = [p for p, tile in obs.tiles.items() if tile.material == "stone"]
            if (action := self._approach_and_do(obs, stone, "collect furnace stone")) is not None:
                return action
        if obs.inventory["wood"] < 3:
            trees = [p for p, tile in obs.tiles.items() if tile.material == "tree"]
            return self._approach_and_do(obs, trees, "collect table wood")
        return None

    def _diamond_action(self, obs: ParsedObservation) -> PolicyDecision | None:
        diamond = [p for p, tile in obs.tiles.items() if tile.material == "diamond"]
        return self._approach_and_do(obs, diamond, "collect diamond")

    def _fallback_action(self, obs: ParsedObservation) -> PolicyDecision:
        possible = [
            direction for direction in Direction
            if self._is_traversable(obs, self._add((0, 0), direction.delta))
        ]
        if not possible:
            cows = [
                direction for direction in Direction
                if (tile := obs.tiles.get(self._add((0, 0), direction.delta))) is not None
                and tile.occupant == "cow"
            ]
            if cows:
                direction = self._rng.choice(cows)
                action = "do" if obs.facing is direction else direction.action
                return PolicyDecision(action, "clear adjacent cow")
            return PolicyDecision(RESET_ACTION, "no traversable adjacent tile")

        if obs.facing in possible:
            alternatives = [d for d in possible if d is not obs.facing]
            if (
                not alternatives
                or self._rng.random() < 0.8
            ):
                direction = obs.facing
            else:
                direction = self._rng.choice(alternatives)
        else:
            direction = self._rng.choice(possible)
        target = self._add((0, 0), direction.delta)
        if self._is_empty(obs, target):
            return PolicyDecision(direction.action, "explore")
        action = "do" if obs.facing is direction else direction.action
        return PolicyDecision(action, "clear exploration tile")


__all__ = ["ACTIONS", "Direction", "ObservationError", "ParsedObservation",
           "PolicyDecision", "ReactiveCrafterPolicy", "RESET_ACTION", "Route", "Tile"]
