"""Full-domain intentions and bounded PDDL planning for experiment 2-3-CR."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from time import monotonic
from typing import Any, Literal

from unified_planning.io import PDDLReader
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import OneshotPlanner, PlanValidator

from .beliefs import (
    CrafterAction,
    DIAMOND_MILESTONES,
    Direction,
    OBSERVATION_OFFSETS,
    cell_key,
    pddl_cell_name,
    position_from_key,
    position_from_pddl_cell,
)


NORMAL_NEED_THRESHOLD = 2
LOW_HEALTH_NEED_THRESHOLD = 4
NORMAL_ENERGY_THRESHOLD = 1
LOW_HEALTH_ENERGY_THRESHOLD = 3
LOW_HEALTH_THRESHOLD = 3
RECOVERED_NEED_THRESHOLD = 5
RECOVERED_ENERGY = 9

SOLVED_STATUSES = {"SOLVED_SATISFICING", "SOLVED_OPTIMALLY"}
PLACEABLE_MATERIALS = frozenset({"grass", "path", "sand"})
RESOURCE_TOOLS = {
    "tree": None,
    "stone": "wood_pickaxe",
    "coal": "wood_pickaxe",
    "iron": "stone_pickaxe",
    "diamond": "iron_pickaxe",
}
ITEM_FLUENTS = {
    "wood": "wood",
    "stone": "stone-count",
    "coal": "coal-count",
    "iron": "iron-count",
    "diamond": "diamond-count",
    "wood_pickaxe": "wood-pickaxe",
    "stone_pickaxe": "stone-pickaxe",
    "iron_pickaxe": "iron-pickaxe",
}
ACHIEVEMENT_GOALS = {
    "collect_wood": "collect-wood",
    "collect_drink": "collect-drink",
    "eat_plant": "eat-plant",
    "eat_cow": "eat-cow",
    "place_table": "place-table",
    "make_wood_pickaxe": "make-wood-pickaxe",
    "collect_stone": "collect-stone",
    "make_stone_pickaxe": "make-stone-pickaxe",
    "collect_coal": "collect-coal",
    "collect_iron": "collect-iron",
    "place_furnace": "place-furnace",
    "make_iron_pickaxe": "make-iron-pickaxe",
    "collect_diamond": "collect-diamond",
}
COLLECT_STAGE_ITEMS = {
    "collect-table-wood": "wood",
    "collect-wood-pickaxe-wood": "wood",
    "collect-stone-pickaxe-wood": "wood",
    "collect-stone-pickaxe-stone": "stone",
    "collect-furnace-stone": "stone",
    "collect-iron-pickaxe-wood": "wood",
    "collect-iron-pickaxe-coal": "coal",
    "collect-iron-pickaxe-iron": "iron",
    "collect-diamond": "diamond",
}
CRAFT_STAGE_ITEMS = {
    "make-wood-pickaxe": "wood_pickaxe",
    "make-stone-pickaxe": "stone_pickaxe",
    "make-iron-pickaxe": "iron_pickaxe",
}


@dataclass(frozen=True)
class TechnologyStage:
    """One explicit unmet prerequisite of the diamond dependency chain."""

    name: str
    kind: Literal["collect", "place", "craft"]
    goal_predicate: str
    target_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-native stage record."""

        return asdict(self)


@dataclass(frozen=True)
class GroundedAction:
    """A validated PDDL action mapped to one native Crafter action."""

    name: str
    arguments: tuple[str, ...]
    native_action: int
    movement_kind: Literal["walk", "turn", "none"]
    source_cell: str | None
    destination_cell: str | None
    recovery_attempt: bool
    cow_cell: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-native grounded action."""

        value = asdict(self)
        value["arguments"] = list(self.arguments)
        return value


@dataclass
class PlanningOutcome:
    """JSON-safe result of one bounded LPG planning attempt."""

    actions: list[GroundedAction]
    attempt: dict[str, Any]
    problem_name: str
    selected_cow: str | None
    expected_goal: dict[str, Any] | None

    @property
    def accepted(self) -> bool:
        """Return whether this attempt yielded an executable complete plan."""

        return bool(self.actions) and bool(self.attempt.get("accepted"))


def _inventory(state: Mapping[str, Any], item: str) -> int:
    return int(state.get("inventory", {}).get(item, 0))


def _cardinal_neighbors(key: str) -> tuple[str, ...]:
    x, y = position_from_key(key)
    return tuple(
        cell_key((x + dx, y + dy))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


def _nearby(first: str, second: str) -> bool:
    a = position_from_key(first)
    b = position_from_key(second)
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _approach_cells(target: str, reachable: set[str]) -> list[str]:
    return sorted(set(_cardinal_neighbors(target)) & reachable)


def _material_cells(state: Mapping[str, Any], material: str) -> list[str]:
    return sorted(
        key
        for key, value in state.get("terrain", {}).items()
        if value == material
    )


def _fresh_occupant_cells(state: Mapping[str, Any], kind: str) -> list[str]:
    revision = int(state.get("revision", 0))
    return sorted(
        key
        for key, value in state.get("occupants", {}).items()
        if value.get("kind") == kind
        and int(value.get("last_seen_revision", -1)) == revision
    )


def _usable_tables(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    reachable = set(state.get("reachable", ()))
    return sorted(
        (craft, table)
        for craft in reachable
        for table in _material_cells(state, "table")
        if _nearby(craft, table)
    )


def _station_couplings(state: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    reachable = set(state.get("reachable", ()))
    tables = _material_cells(state, "table")
    furnaces = _material_cells(state, "furnace")
    return sorted(
        (craft, table, furnace)
        for craft in reachable
        for table in tables
        for furnace in furnaces
        if _nearby(craft, table) and _nearby(craft, furnace)
    )


def _table_layouts(state: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    terrain = state.get("terrain", {})
    clear = set(state.get("clear", ()))
    reachable = set(state.get("reachable", ()))
    result: list[tuple[str, str, str]] = []
    for craft in sorted(reachable):
        sites = [
            key
            for key in _cardinal_neighbors(craft)
            if key in reachable
            and key in clear
            and terrain.get(key) in PLACEABLE_MATERIALS
        ]
        result.extend(
            (craft, table, furnace)
            for table in sites
            for furnace in sites
            if table != furnace
        )
    return sorted(result)


def _furnace_layouts(state: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    terrain = state.get("terrain", {})
    clear = set(state.get("clear", ()))
    reachable = set(state.get("reachable", ()))
    result: list[tuple[str, str, str]] = []
    for craft in sorted(reachable):
        neighbors = set(_cardinal_neighbors(craft))
        tables = sorted(
            key for key in neighbors if terrain.get(key) == "table"
        )
        sites = sorted(
            key
            for key in neighbors
            if key in reachable
            and key in clear
            and terrain.get(key) in PLACEABLE_MATERIALS
        )
        result.extend(
            (craft, furnace, table)
            for furnace in sites
            for table in tables
            if furnace != table
        )
    return sorted(result)


def _resource_stage(name: str, item: str, target: int, state: Mapping[str, Any]) -> TechnologyStage:
    count = min(_inventory(state, item) + 1, target)
    return TechnologyStage(
        name=name,
        kind="collect",
        goal_predicate=f"(>= ({ITEM_FLUENTS[item]}) {count})",
        target_count=target,
    )


def derive_technology_stage(
    belief_state: Mapping[str, Any],
) -> TechnologyStage | None:
    """Derive the first unmet prerequisite from current public beliefs."""

    if _inventory(belief_state, "diamond") >= 1:
        return None
    if not _usable_tables(belief_state):
        if _inventory(belief_state, "wood") < 2:
            return _resource_stage(
                "collect-table-wood", "wood", 2, belief_state
            )
        return TechnologyStage("place-table", "place", "(table-established)")
    if _inventory(belief_state, "wood_pickaxe") < 1:
        if _inventory(belief_state, "wood") < 1:
            return _resource_stage(
                "collect-wood-pickaxe-wood", "wood", 1, belief_state
            )
        return TechnologyStage(
            "make-wood-pickaxe", "craft", "(>= (wood-pickaxe) 1)", 1
        )
    if _inventory(belief_state, "stone_pickaxe") < 1:
        if _inventory(belief_state, "wood") < 1:
            return _resource_stage(
                "collect-stone-pickaxe-wood", "wood", 1, belief_state
            )
        if _inventory(belief_state, "stone") < 1:
            return _resource_stage(
                "collect-stone-pickaxe-stone", "stone", 1, belief_state
            )
        return TechnologyStage(
            "make-stone-pickaxe", "craft", "(>= (stone-pickaxe) 1)", 1
        )
    if not _station_couplings(belief_state):
        if _inventory(belief_state, "stone") < 4:
            return _resource_stage(
                "collect-furnace-stone", "stone", 4, belief_state
            )
        return TechnologyStage(
            "place-furnace", "place", "(furnace-established)"
        )
    if _inventory(belief_state, "iron_pickaxe") < 1:
        if _inventory(belief_state, "wood") < 1:
            return _resource_stage(
                "collect-iron-pickaxe-wood", "wood", 1, belief_state
            )
        if _inventory(belief_state, "coal") < 1:
            return _resource_stage(
                "collect-iron-pickaxe-coal", "coal", 1, belief_state
            )
        if _inventory(belief_state, "iron") < 1:
            return _resource_stage(
                "collect-iron-pickaxe-iron", "iron", 1, belief_state
            )
        return TechnologyStage(
            "make-iron-pickaxe", "craft", "(>= (iron-pickaxe) 1)", 1
        )
    return TechnologyStage(
        "collect-diamond", "collect", "(>= (diamond-count) 1)", 1
    )


def recovery_entry_threshold(need: str, health: int) -> int:
    """Return the health-dependent intervention threshold for one need."""

    if need == "energy":
        return (
            LOW_HEALTH_ENERGY_THRESHOLD
            if health <= LOW_HEALTH_THRESHOLD
            else NORMAL_ENERGY_THRESHOLD
        )
    return (
        LOW_HEALTH_NEED_THRESHOLD
        if health <= LOW_HEALTH_THRESHOLD
        else NORMAL_NEED_THRESHOLD
    )


def recovery_complete(
    intention: Mapping[str, Any],
    belief_state: Mapping[str, Any],
) -> bool:
    """Return whether an active recovery intention reached its exit threshold."""

    need = str(intention["need"])
    if need == "energy":
        return (
            not bool(belief_state.get("sleeping"))
            and _inventory(belief_state, "energy") >= RECOVERED_ENERGY
        )
    return _inventory(belief_state, need) >= RECOVERED_NEED_THRESHOLD


def select_recovery_intention(
    belief_state: Mapping[str, Any],
    current_intention: Mapping[str, Any] | None,
) -> dict[str, object] | None:
    """Select or retain one urgent recovery using deterministic hysteresis."""

    health = _inventory(belief_state, "health")
    urgent: list[tuple[float, int, str, int, int]] = []
    order = {"drink": 0, "food": 1, "energy": 2}
    for need in ("drink", "food", "energy"):
        threshold = recovery_entry_threshold(need, health)
        value = _inventory(belief_state, need)
        if value <= threshold:
            completion = RECOVERED_ENERGY if need == "energy" else RECOVERED_NEED_THRESHOLD
            urgent.append((value / threshold, order[need], need, threshold, completion))

    current = (
        dict(current_intention)
        if current_intention is not None
        and current_intention.get("kind") == "recovery"
        and not recovery_complete(current_intention, belief_state)
        else None
    )
    if current is not None:
        current_need = str(current["need"])
        threshold = int(current["entry_threshold"])
        ratio = _inventory(belief_state, current_need) / threshold
        contender = min(urgent, default=None)
        if contender is None or contender[0] >= ratio or contender[2] == current_need:
            return current

    if not urgent:
        return None
    _, _, need, threshold, completion = min(urgent)
    return {
        "kind": "recovery",
        "name": f"restore-{need}",
        "need": need,
        "goal_predicate": (
            "(food-restored)"
            if need == "food"
            else "(drink-restored)"
            if need == "drink"
            else None
        ),
        "entry_threshold": threshold,
        "completion_threshold": completion,
        "selected_revision": int(belief_state.get("revision", 0)),
        "entry_value": _inventory(belief_state, need),
    }


def stage_supported(
    stage: TechnologyStage,
    belief_state: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Return whether current public beliefs support planning for a stage."""

    reachable = set(belief_state.get("reachable", ()))
    if stage.name in COLLECT_STAGE_ITEMS:
        material = "tree" if COLLECT_STAGE_ITEMS[stage.name] == "wood" else COLLECT_STAGE_ITEMS[stage.name]
        targets = _material_cells(belief_state, material)
        if not targets:
            return False, "missing-target"
        if not any(_approach_cells(target, reachable) for target in targets):
            return False, "unreachable-target"
        return True, None
    if stage.name == "place-table":
        return (True, None) if _table_layouts(belief_state) else (False, "missing-site")
    if stage.name == "place-furnace":
        if not _material_cells(belief_state, "table"):
            return False, "missing-station"
        return (True, None) if _furnace_layouts(belief_state) else (False, "missing-site")
    if stage.name in {"make-wood-pickaxe", "make-stone-pickaxe"}:
        return (True, None) if _usable_tables(belief_state) else (False, "missing-station")
    if stage.name == "make-iron-pickaxe":
        return (True, None) if _station_couplings(belief_state) else (False, "missing-station")
    raise ValueError(f"Unknown technology stage: {stage.name}")


def recovery_supported(
    intention: Mapping[str, Any],
    belief_state: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Return whether a recovery action or PDDL target is currently available."""

    need = intention.get("need")
    if need == "energy":
        return True, None
    reachable = set(belief_state.get("reachable", ()))
    if need == "drink":
        targets = _material_cells(belief_state, "water")
    else:
        ripe = _fresh_occupant_cells(belief_state, "ripe-plant")
        cows = _fresh_occupant_cells(belief_state, "cow")
        targets = sorted((*ripe, *cows))
    if not targets:
        return False, "missing-target"
    if not any(_approach_cells(target, reachable) for target in targets):
        return False, "unreachable-target"
    return True, None


def highest_milestone(achievements: Sequence[str]) -> str | None:
    """Return the deepest fixed diamond-path achievement attained."""

    attained = set(achievements)
    return next(
        (name for name in reversed(DIAMOND_MILESTONES) if name in attained),
        None,
    )


def _direction_between(source: tuple[int, int], destination: tuple[int, int]) -> Direction:
    delta = destination[0] - source[0], destination[1] - source[1]
    for direction in Direction:
        if direction.delta == delta:
            return direction
    raise ValueError(f"Cells are not cardinally adjacent: {source}, {destination}")


def _select_cow(belief_state: Mapping[str, Any]) -> str | None:
    target = belief_state.get("cow_target")
    reachable = set(belief_state.get("reachable", ()))
    fresh = {
        key
        for key in _fresh_occupant_cells(belief_state, "cow")
        if _approach_cells(key, reachable)
    }
    if isinstance(target, Mapping) and target.get("cell") in fresh:
        return str(target["cell"])
    player = tuple(belief_state.get("player", (0, 0)))
    if not fresh:
        return None
    return min(
        fresh,
        key=lambda key: (
            sum(abs(a - b) for a, b in zip(position_from_key(key), player)),
            key,
        ),
    )


def _problem_cells(
    belief_state: Mapping[str, Any],
    intention_name: str,
) -> list[str]:
    terrain = belief_state.get("terrain", {})
    reachable = set(belief_state.get("reachable", ()))
    player_key = cell_key(tuple(belief_state.get("player", (0, 0))))
    cells = reachable | {player_key}
    for key, material in terrain.items():
        if material in {*RESOURCE_TOOLS, "water"} and _approach_cells(
            key, reachable
        ):
            cells.add(key)
        elif material in {"table", "furnace"} and any(
            _nearby(key, craft) for craft in reachable
        ):
            cells.add(key)
    for key in (*_fresh_occupant_cells(belief_state, "ripe-plant"), *_fresh_occupant_cells(belief_state, "cow")):
        if _approach_cells(key, reachable):
            cells.add(key)
    if intention_name == "place-table":
        for craft, table, furnace in _table_layouts(belief_state):
            cells.update((craft, table, furnace))
    elif intention_name == "place-furnace":
        for craft, furnace, table in _furnace_layouts(belief_state):
            cells.update((craft, furnace, table))
    return sorted(cells)


def _best_route(
    player: str,
    reachable: set[str],
    destinations: Sequence[str],
) -> tuple[list[str], str] | None:
    candidates: list[tuple[int, str, list[str]]] = []
    for destination in sorted(set(destinations)):
        route = _route(player, destination, reachable)
        if destination == player or route:
            candidates.append((len(route), destination, route))
    if not candidates:
        return None
    _, destination, route = min(candidates)
    return route, destination


def _interaction_focus(
    player: str,
    reachable: set[str],
    targets: Sequence[str],
) -> tuple[list[str], str | None]:
    candidates: list[tuple[int, str, str, list[str]]] = []
    for target in sorted(set(targets)):
        for approach in _approach_cells(target, reachable):
            route = _route(player, approach, reachable)
            if approach == player or route:
                candidates.append((len(route), target, approach, route))
    if not candidates:
        return [player], None
    _, target, _, route = min(candidates)
    return [player, *route], target


def _placement_route(
    belief_state: Mapping[str, Any],
    craft: str,
    target: str,
) -> list[str] | None:
    player = cell_key(tuple(belief_state.get("player", (0, 0))))
    reachable = set(belief_state.get("reachable", ())) | {player}
    direction = _direction_between(
        position_from_key(craft), position_from_key(target)
    )
    if player == craft and belief_state.get("facing") == direction.name.lower():
        return []
    craft_x, craft_y = position_from_key(craft)
    predecessor = cell_key(
        (craft_x - direction.delta[0], craft_y - direction.delta[1])
    )
    if predecessor in reachable:
        route = _route(player, predecessor, reachable)
        if predecessor == player or route:
            return [*route, craft]
    return None


def _planning_scope(
    belief_state: Mapping[str, Any],
    intention_name: str,
) -> tuple[list[str], tuple[str, str, str] | None, str | None]:
    """Select a shortest intention-relevant navigation subgraph."""

    player = cell_key(tuple(belief_state.get("player", (0, 0))))
    reachable = set(belief_state.get("reachable", ())) | {player}
    terrain = belief_state.get("terrain", {})
    if intention_name in COLLECT_STAGE_ITEMS:
        item = COLLECT_STAGE_ITEMS[intention_name]
        material = "tree" if item == "wood" else item
        navigation, target = _interaction_focus(
            player, reachable, _material_cells(belief_state, material)
        )
        return navigation, None, target
    if intention_name in {"restore-drink", "restore-food"}:
        if intention_name == "restore-drink":
            targets = _material_cells(belief_state, "water")
        else:
            selected = _select_cow(belief_state)
            fresh_ripe = _fresh_occupant_cells(belief_state, "ripe-plant")
            targets = [selected] if selected is not None else fresh_ripe
            if not targets:
                targets = _fresh_occupant_cells(belief_state, "cow")
        navigation, target = _interaction_focus(
            player, reachable, targets
        )
        return navigation, None, target
    if intention_name == "place-table":
        layouts = _table_layouts(belief_state)
    elif intention_name == "place-furnace":
        layouts = _furnace_layouts(belief_state)
    else:
        layouts = []
    if layouts:
        candidates = []
        for layout in layouts:
            route = _placement_route(belief_state, layout[0], layout[1])
            if route is not None:
                candidates.append((len(route), layout, route))
        if candidates:
            _, layout, route = min(candidates)
            navigation = [player, *route, layout[1]]
            return navigation, layout, None
    craft_cells: Sequence[str]
    if intention_name == "make-iron-pickaxe":
        craft_cells = [value[0] for value in _station_couplings(belief_state)]
    elif intention_name in CRAFT_STAGE_ITEMS:
        craft_cells = [value[0] for value in _usable_tables(belief_state)]
    else:
        craft_cells = []
    best = _best_route(player, reachable, craft_cells)
    if best is not None:
        route, _ = best
        return [player, *route], None, None
    return [player], None, None


def build_problem_pddl(
    *,
    problem_name: str,
    belief_state: Mapping[str, Any],
    intention: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Build one intention-scoped public-belief PDDL problem."""

    name = str(intention["name"])
    terrain = belief_state.get("terrain", {})
    player = tuple(belief_state.get("player", (0, 0)))
    player_key = cell_key(player)
    cell_keys = _problem_cells(belief_state, name)
    positions = {key: position_from_key(key) for key in cell_keys}
    navigation_path, selected_layout, focus_target = _planning_scope(
        belief_state, name
    )
    navigation_keys = set(navigation_path)
    site_keys: set[str] = set()
    if selected_layout is not None:
        site_keys.add(selected_layout[1])
        if name == "place-table":
            site_keys.add(selected_layout[2])
    station_keys = {
        key for key in cell_keys if terrain.get(key) in {"table", "furnace"}
    }
    target_keys = {
        key
        for key in cell_keys
        if terrain.get(key) in {*RESOURCE_TOOLS, "water"}
        or key in belief_state.get("occupants", {})
    }
    ordinary_keys = navigation_keys - site_keys - station_keys - target_keys
    inactive_keys = (
        set(cell_keys)
        - ordinary_keys
        - site_keys
        - station_keys
        - target_keys
    )
    object_groups = (
        (ordinary_keys, "cell"),
        (site_keys, "station-site"),
        (inactive_keys, "known-cell"),
        (target_keys, "target"),
        (station_keys, "station"),
    )
    objects = " ".join(
        f"{' '.join(pddl_cell_name(positions[key]) for key in sorted(keys))} - {kind}"
        for keys, kind in object_groups
        if keys
    )
    facts: list[str] = ["(alive)", f"(at {pddl_cell_name(player)})"]
    facing = belief_state.get("facing")
    if facing in {direction.name.lower() for direction in Direction}:
        facts.append(f"(facing {facing})")
    for key in sorted(set(belief_state.get("known", ())) & set(cell_keys)):
        facts.append(f"(known {pddl_cell_name(positions[key])})")
    for key in cell_keys:
        material = terrain.get(key)
        if material is not None:
            facts.append(f"(terrain {pddl_cell_name(positions[key])} {material})")
    for predicate, values in (
        ("clear", belief_state.get("clear", ())),
        ("safe-to-enter", belief_state.get("safe_to_enter", ())),
        ("movement-blocked", belief_state.get("movement_blocked", ())),
    ):
        for key in sorted(set(values) & set(cell_keys)):
            facts.append(f"({predicate} {pddl_cell_name(positions[key])})")
    by_position = {
        position: pddl_cell_name(position) for position in positions.values()
    }
    navigation_rank = {key: index for index, key in enumerate(navigation_path)}
    position_keys = {position: key for key, position in positions.items()}
    for source_key in navigation_path:
        source = positions[source_key]
        source_name = by_position[source]
        for direction in Direction:
            target_position = (
                source[0] + direction.delta[0],
                source[1] + direction.delta[1],
            )
            if target_position in by_position:
                target_key = position_keys[target_position]
                target_is_cell = target_key in ordinary_keys | site_keys
                if target_is_cell and navigation_rank.get(target_key) != (
                    navigation_rank[source_key] + 1
                ):
                    continue
                facts.append(
                    f"(adjacent {source_name} {by_position[target_position]} "
                    f"{direction.name.lower()})"
                )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nearby_position = source[0] + dx, source[1] + dy
                if nearby_position in by_position:
                    facts.append(
                        f"(nearby {source_name} {by_position[nearby_position]})"
                    )

    revision = int(belief_state.get("revision", 0))
    fresh_ripe = [
        key
        for key, occupant in belief_state.get("occupants", {}).items()
        if key in positions
        and occupant.get("kind") == "ripe-plant"
        and int(occupant.get("last_seen_revision", -1)) == revision
    ]
    for key in sorted(fresh_ripe):
        facts.append(f"(ripe-plant {pddl_cell_name(positions[key])})")

    selected_cow = (
        focus_target
        if name == "restore-food"
        and belief_state.get("occupants", {}).get(focus_target, {}).get("kind")
        == "cow"
        else None
    )
    if selected_cow is not None:
        if selected_cow not in positions or terrain.get(selected_cow) not in PLACEABLE_MATERIALS:
            raise ValueError("Selected cow must be a fresh approachable target")
        cow_name = pddl_cell_name(positions[selected_cow])
        facts.append(f"(cow-at {cow_name})")
        cow_target = belief_state.get("cow_target")
        hits = (
            int(cow_target.get("hits", 0))
            if isinstance(cow_target, Mapping)
            and cow_target.get("cell") == selected_cow
            else 0
        )
        predicate = "cow-unhurt" if hits == 0 else "cow-hit-once" if hits == 1 else "cow-hit-twice"
        facts.append(f"({predicate} {cow_name})")

    if name == "place-table":
        for craft, table, furnace in (
            (selected_layout,) if selected_layout is not None else ()
        ):
            facts.append(
                f"(future-station-layout {pddl_cell_name(positions[craft])} "
                f"{pddl_cell_name(positions[table])} {pddl_cell_name(positions[furnace])})"
            )
    elif name == "place-furnace":
        for craft, furnace, table in (
            (selected_layout,) if selected_layout is not None else ()
        ):
            facts.append(
                f"(coupled-furnace-site {pddl_cell_name(positions[craft])} "
                f"{pddl_cell_name(positions[furnace])} {pddl_cell_name(positions[table])})"
            )

    achievements = set(belief_state.get("achievements", ()))
    for achievement, constant in sorted(ACHIEVEMENT_GOALS.items()):
        if achievement in achievements:
            facts.append(f"(achieved {constant})")
    facts.extend(
        f"(= ({fluent}) {_inventory(belief_state, item)})"
        for item, fluent in ITEM_FLUENTS.items()
    )
    facts.append("(= (total-cost) 0)")
    problem = f"""(define (problem {problem_name})
  (:domain crafter-bdi)
  (:objects {objects})
  (:init
    {' '.join(sorted(facts))}
  )
  (:goal {intention['goal_predicate']})
  (:metric minimize (total-cost))
)"""
    return problem, selected_cow


def serialize_action(action_instance: Any) -> GroundedAction:
    """Validate and serialize one Unified Planning action instance."""

    raw_name = str(action_instance.action.name).lower()
    name = {
        "place-table-op": "place-table",
        "place-furnace-op": "place-furnace",
        "make-wood-pickaxe-op": "make-wood-pickaxe",
        "make-stone-pickaxe-op": "make-stone-pickaxe",
        "make-iron-pickaxe-op": "make-iron-pickaxe",
    }.get(raw_name, raw_name)
    arguments = tuple(
        str(parameter).lower() for parameter in action_instance.actual_parameters
    )
    directions = {
        "left": CrafterAction.MOVE_LEFT,
        "right": CrafterAction.MOVE_RIGHT,
        "up": CrafterAction.MOVE_UP,
        "down": CrafterAction.MOVE_DOWN,
    }
    arities = {
        **{f"walk-{direction}": 2 for direction in directions},
        **{f"turn-{direction}": 2 for direction in directions},
        "do-tree": 3,
        "do-water": 3,
        "do-ripe-plant": 3,
        "hit-cow-unhurt": 3,
        "hit-cow-once": 3,
        "finish-cow": 3,
        "place-table": 4,
        "make-wood-pickaxe": 2,
        "do-stone": 3,
        "make-stone-pickaxe": 2,
        "do-coal": 3,
        "do-iron": 3,
        "place-furnace": 4,
        "make-iron-pickaxe": 3,
        "do-diamond": 3,
    }
    if name not in arities or len(arguments) != arities[name]:
        raise ValueError(f"Unknown or malformed grounded action: {name}{arguments}")
    if name.startswith(("walk-", "turn-")):
        direction = name.rsplit("-", 1)[1]
        native = directions[direction]
        movement: Literal["walk", "turn", "none"] = (
            "walk" if name.startswith("walk-") else "turn"
        )
    elif name == "place-table":
        native, movement = CrafterAction.PLACE_TABLE, "none"
    elif name == "place-furnace":
        native, movement = CrafterAction.PLACE_FURNACE, "none"
    elif name == "make-wood-pickaxe":
        native, movement = CrafterAction.MAKE_WOOD_PICKAXE, "none"
    elif name == "make-stone-pickaxe":
        native, movement = CrafterAction.MAKE_STONE_PICKAXE, "none"
    elif name == "make-iron-pickaxe":
        native, movement = CrafterAction.MAKE_IRON_PICKAXE, "none"
    else:
        native, movement = CrafterAction.DO, "none"
    source = cell_key(position_from_pddl_cell(arguments[0]))
    destination = (
        cell_key(position_from_pddl_cell(arguments[1]))
        if name not in CRAFT_STAGE_ITEMS
        else None
    )
    cow_cell = destination if name in {"hit-cow-unhurt", "hit-cow-once", "finish-cow"} else None
    return GroundedAction(
        name=name,
        arguments=arguments,
        native_action=int(native),
        movement_kind=movement,
        source_cell=source,
        destination_cell=destination,
        recovery_attempt=name in {"do-water", "do-ripe-plant", "finish-cow"},
        cow_cell=cow_cell,
    )


def _direction_for_action(action: GroundedAction) -> str | None:
    if action.name.startswith(("walk-", "turn-")):
        return action.name.rsplit("-", 1)[1]
    if len(action.arguments) >= 3 and action.name not in {"make-iron-pickaxe"}:
        return (
            action.arguments[-1]
            if action.name in {"place-table", "place-furnace"}
            else action.arguments[2]
        )
    return None


def _visible_clear_target(
    destination: str | None,
    belief_state: Mapping[str, Any],
) -> bool:
    return (
        destination is not None
        and destination in set(belief_state.get("visible_cells", ()))
        and destination in set(belief_state.get("clear", ()))
    )


def action_sound(
    action: GroundedAction,
    belief_state: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Check one grounded action against the latest public belief revision."""

    player = cell_key(tuple(belief_state.get("player", (0, 0))))
    if action.source_cell != player:
        return False, "source-cell-mismatch"
    if action.name == "recovery-noop":
        return (True, None) if belief_state.get("sleeping") else (False, "not-sleeping")
    if action.name == "recovery-sleep":
        return (
            (True, None)
            if not belief_state.get("sleeping") and _inventory(belief_state, "energy") < RECOVERED_ENERGY
            else (False, "sleep-not-needed")
        )

    destination = action.destination_cell
    direction = _direction_for_action(action)
    if destination is not None:
        try:
            expected = _direction_between(
                position_from_key(player), position_from_key(destination)
            ).name.lower()
        except ValueError:
            return False, "non-adjacent-target"
        if direction != expected:
            return False, "direction-mismatch"
    visible = set(belief_state.get("visible_cells", ()))
    occupants = belief_state.get("occupants", {})
    if action.movement_kind == "walk":
        if destination not in visible:
            return False, "walk-destination-not-visible"
        if destination not in set(belief_state.get("safe_to_enter", ())):
            return False, "unsafe-walk-destination"
        if destination in occupants:
            return False, "occupied-walk-destination"
        return True, None
    if action.movement_kind == "turn":
        if destination not in visible:
            return False, "turn-target-not-visible"
        if destination not in set(belief_state.get("movement_blocked", ())):
            return False, "unsupported-turn-obstacle"
        if belief_state.get("facing") == direction:
            return False, "already-facing-target"
        return True, None

    terrain = belief_state.get("terrain", {})
    if direction is not None and belief_state.get("facing") != direction:
        return False, "facing-mismatch"
    resources = {
        "do-tree": ("tree", None),
        "do-stone": ("stone", "wood_pickaxe"),
        "do-coal": ("coal", "wood_pickaxe"),
        "do-iron": ("iron", "stone_pickaxe"),
        "do-diamond": ("diamond", "iron_pickaxe"),
        "do-water": ("water", None),
    }
    if action.name in resources:
        material, tool = resources[action.name]
        if not _visible_clear_target(destination, belief_state) or terrain.get(destination) != material:
            return False, f"{material}-target-changed"
        if tool is not None and _inventory(belief_state, tool) < 1:
            return False, f"missing-{tool.replace('_', '-')}"
    if action.name == "do-ripe-plant":
        occupant = occupants.get(destination, {})
        if (
            destination not in visible
            or occupant.get("kind") != "ripe-plant"
            or int(occupant.get("last_seen_revision", -1))
            != int(belief_state.get("revision", 0))
        ):
            return False, "ripe-plant-target-changed"
    if action.cow_cell is not None:
        target = belief_state.get("cow_target")
        occupant = occupants.get(action.cow_cell, {})
        if (
            action.cow_cell not in visible
            or occupant.get("kind") != "cow"
            or not isinstance(target, Mapping)
            or target.get("cell") != action.cow_cell
        ):
            return False, "cow-target-moved"
        expected_hits = {
            "hit-cow-unhurt": 0,
            "hit-cow-once": 1,
            "finish-cow": 2,
        }[action.name]
        if int(target.get("hits", -1)) != expected_hits:
            return False, "cow-stage-changed"
    if action.name == "place-table":
        future = cell_key(position_from_pddl_cell(action.arguments[2]))
        if _inventory(belief_state, "wood") < 2:
            return False, "insufficient-wood"
        if not _visible_clear_target(destination, belief_state):
            return False, "table-site-changed"
        if terrain.get(destination) not in PLACEABLE_MATERIALS:
            return False, "table-material-changed"
        if (
            future == destination
            or future == player
            or future not in visible
            or future not in set(belief_state.get("safe_to_enter", ()))
            or terrain.get(future) not in PLACEABLE_MATERIALS
            or future not in _cardinal_neighbors(player)
        ):
            return False, "future-furnace-site-changed"
    if action.name == "place-furnace":
        table = cell_key(position_from_pddl_cell(action.arguments[2]))
        if _inventory(belief_state, "stone") < 4:
            return False, "insufficient-stone"
        if not _visible_clear_target(destination, belief_state):
            return False, "furnace-site-changed"
        if terrain.get(destination) not in PLACEABLE_MATERIALS:
            return False, "furnace-material-changed"
        if (
            table == destination
            or table == player
            or table not in visible
            or terrain.get(table) != "table"
            or table not in _cardinal_neighbors(player)
        ):
            return False, "coupled-table-changed"
    craft_requirements = {
        "make-wood-pickaxe": {"wood": 1},
        "make-stone-pickaxe": {"wood": 1, "stone": 1},
        "make-iron-pickaxe": {"wood": 1, "coal": 1, "iron": 1},
    }
    if action.name in craft_requirements:
        if any(_inventory(belief_state, item) < count for item, count in craft_requirements[action.name].items()):
            return False, "insufficient-crafting-resources"
        station_keys = [
            cell_key(position_from_pddl_cell(value))
            for value in action.arguments[1:]
        ]
        expected_materials = (
            ("table",)
            if action.name != "make-iron-pickaxe"
            else ("table", "furnace")
        )
        if any(
            key not in visible
            or terrain.get(key) != material
            or not _nearby(player, key)
            for key, material in zip(station_keys, expected_materials, strict=True)
        ):
            return False, "required-station-not-visible"
    return True, None


def expected_goal_for(
    intention: Mapping[str, Any],
    belief_state: Mapping[str, Any],
    actions: Sequence[GroundedAction],
) -> dict[str, Any]:
    """Create the minimal public evidence record for one accepted plan."""

    name = str(intention["name"])
    if intention.get("kind") == "recovery":
        need = str(intention["need"])
        return {
            "kind": "recovery-interaction",
            "need": need,
            "baseline": _inventory(belief_state, need),
        }
    if name in COLLECT_STAGE_ITEMS:
        item = COLLECT_STAGE_ITEMS[name]
        target = int(intention.get("target_count") or 1)
        return {
            "kind": "inventory-at-least",
            "item": item,
            "count": min(_inventory(belief_state, item) + 1, target),
        }
    if name in CRAFT_STAGE_ITEMS:
        return {
            "kind": "inventory-at-least",
            "item": CRAFT_STAGE_ITEMS[name],
            "count": 1,
        }
    operator = "place-table" if name == "place-table" else "place-furnace"
    action = next(value for value in actions if value.name == operator)
    marker = "table-established" if name == "place-table" else "furnace-established"
    return {
        "kind": "station-established",
        "marker": marker,
        "cell": action.destination_cell,
    }


def plan_goal_observed(
    expected_goal: Mapping[str, Any],
    belief_state: Mapping[str, Any],
    action_result: Mapping[str, Any],
) -> bool:
    """Evaluate one accepted plan's minimal goal using public evidence."""

    kind = expected_goal.get("kind")
    if kind == "inventory-at-least":
        return _inventory(belief_state, str(expected_goal["item"])) >= int(expected_goal["count"])
    if kind == "station-established":
        material = "table" if expected_goal.get("marker") == "table-established" else "furnace"
        return belief_state.get("terrain", {}).get(expected_goal.get("cell")) == material
    if kind == "recovery-interaction":
        return (
            action_result.get("illegal_action") is False
            and _inventory(belief_state, str(expected_goal["need"]))
            > int(expected_goal["baseline"])
        )
    raise ValueError(f"Unknown expected goal kind: {kind!r}")


def _route(source: str, destination: str, allowed: set[str]) -> list[str]:
    if source == destination:
        return []
    parents: dict[str, str | None] = {source: None}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in _cardinal_neighbors(current):
            if neighbor not in allowed or neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == destination:
                path = [destination]
                while parents[path[-1]] != source:
                    path.append(str(parents[path[-1]]))
                return list(reversed(path))
            queue.append(neighbor)
    return []


def _walk_action(source: str, destination: str) -> GroundedAction:
    direction = _direction_between(position_from_key(source), position_from_key(destination))
    return GroundedAction(
        name=f"walk-{direction.name.lower()}",
        arguments=(
            pddl_cell_name(position_from_key(source)),
            pddl_cell_name(position_from_key(destination)),
        ),
        native_action=int(direction.action),
        movement_kind="walk",
        source_cell=source,
        destination_cell=destination,
        recovery_attempt=False,
        cow_cell=None,
    )


def _unknown_count(anchor: str, known: set[str]) -> int:
    x, y = position_from_key(anchor)
    return sum(
        cell_key((x + dx, y + dy)) not in known
        for dx, dy in OBSERVATION_OFFSETS
    )


def choose_exploration_action(
    belief_state: Mapping[str, Any],
    rng: random.Random,
) -> GroundedAction:
    """Choose one frontier-directed or boundary-clearing exploration action."""

    del rng
    player = cell_key(tuple(belief_state.get("player", (0, 0))))
    reachable = set(belief_state.get("reachable", ())) | {player}
    known = set(belief_state.get("known", ()))
    visits = belief_state.get("visit_counts", {})
    frontiers: list[tuple[int, int, int, str, list[str]]] = []
    for candidate in sorted(reachable):
        reveal = _unknown_count(candidate, known)
        if reveal <= 0:
            continue
        route = _route(player, candidate, reachable)
        if candidate != player and not route:
            continue
        frontiers.append(
            (len(route), -reveal, int(visits.get(candidate, 0)), candidate, route)
        )
    if frontiers:
        _, _, _, candidate, route = min(frontiers)
        if candidate != player:
            return _walk_action(player, route[0])

    terrain = belief_state.get("terrain", {})
    visible = set(belief_state.get("visible_cells", ()))
    mineable = {
        material
        for material, tool in RESOURCE_TOOLS.items()
        if material != "diamond" and (tool is None or _inventory(belief_state, tool) >= 1)
    }
    boundaries: list[tuple[int, int, str, str, list[str]]] = []
    safe = set(belief_state.get("safe_to_enter", ()))
    for target in sorted(visible):
        material = terrain.get(target)
        if material not in mineable:
            continue
        approaches = _approach_cells(target, reachable)
        if not approaches:
            continue
        routes = [(_route(player, approach, reachable), approach) for approach in approaches]
        route, approach = min(
            routes,
            key=lambda value: (
                len(value[0]) if value[1] != player else 0,
                value[1],
            ),
        )
        expanded = set(reachable)
        expanded.add(target)
        queue = deque([target])
        while queue:
            current = queue.popleft()
            for neighbor in _cardinal_neighbors(current):
                if neighbor in safe and neighbor not in expanded:
                    expanded.add(neighbor)
                    queue.append(neighbor)
        boundaries.append(
            (
                -_unknown_count(target, known),
                -(len(expanded) - len(reachable)),
                target,
                approach,
                route,
            )
        )
    if boundaries:
        _, _, target, approach, route = min(boundaries)
        if approach != player:
            return _walk_action(player, route[0])
        direction = _direction_between(position_from_key(player), position_from_key(target))
        arguments = (
            pddl_cell_name(position_from_key(player)),
            pddl_cell_name(position_from_key(target)),
            direction.name.lower(),
        )
        if belief_state.get("facing") != direction.name.lower():
            return GroundedAction(
                f"turn-{direction.name.lower()}",
                arguments[:2],
                int(direction.action),
                "turn",
                player,
                target,
                False,
                None,
            )
        return GroundedAction(
            f"do-{terrain[target]}",
            arguments,
            int(CrafterAction.DO),
            "none",
            player,
            target,
            False,
            None,
        )

    adjacent: list[tuple[int, int, str]] = []
    for destination in set(_cardinal_neighbors(player)) & safe & visible:
        adjacent.append(
            (-_unknown_count(destination, known), int(visits.get(destination, 0)), destination)
        )
    if adjacent:
        return _walk_action(player, min(adjacent)[2])
    return GroundedAction(
        "explore-noop",
        (),
        int(CrafterAction.NOOP),
        "none",
        player,
        None,
        False,
        None,
    )


class PlanningService:
    """Build, solve, validate, and serialize one bounded LPG plan."""

    def __init__(
        self,
        domain_path: Path,
        timeout: float,
        planner_factory: Callable[..., Any] = OneshotPlanner,
    ) -> None:
        self.domain_text = domain_path.read_text(encoding="utf-8")
        self.timeout = timeout
        self._planner_factory = planner_factory
        self._reader = PDDLReader()

    def solve(
        self,
        belief_state: Mapping[str, Any],
        intention: Mapping[str, Any],
        problem_name: str,
        remaining_steps: int,
    ) -> PlanningOutcome:
        """Run LPG once and return only a validated complete sequential plan."""

        started = monotonic()
        attempt: dict[str, Any] = {
            "stage": intention.get("name"),
            "engine": "lpg",
            "status": "NOT_RUN",
            "elapsed_seconds": 0.0,
            "plan_length": None,
            "validation_status": None,
            "accepted": False,
            "classification": None,
            "rejection": None,
            "error": None,
        }
        selected_cow: str | None = None
        try:
            problem_pddl, selected_cow = build_problem_pddl(
                problem_name=problem_name,
                belief_state=belief_state,
                intention=intention,
            )
            problem = self._reader.parse_problem_string(
                self.domain_text, problem_pddl
            )
        except Exception as exc:
            attempt.update(
                status="MODEL_ERROR",
                classification="model-error",
                rejection="model-error",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=monotonic() - started,
            )
            return PlanningOutcome([], attempt, problem_name, selected_cow, None)

        try:
            with self._planner_factory(name="lpg") as planner:
                result = planner.solve(problem, timeout=self.timeout)
            attempt["status"] = result.status.name
            plan = result.plan
            if result.status.name not in SOLVED_STATUSES or plan is None:
                attempt["classification"] = "timeout-or-unsolved"
                attempt["rejection"] = "planner-did-not-solve"
            elif not isinstance(plan, SequentialPlan):
                attempt["classification"] = "validator-rejected"
                attempt["rejection"] = "non-sequential-plan"
            elif not plan.actions:
                attempt["classification"] = "timeout-or-unsolved"
                attempt["rejection"] = "empty-plan"
            elif len(plan.actions) > remaining_steps:
                attempt["plan_length"] = len(plan.actions)
                attempt["classification"] = "budget-rejected"
                attempt["rejection"] = "plan-exceeds-remaining-steps"
            else:
                attempt["plan_length"] = len(plan.actions)
                validation_status, rejection = self._validate_plan(problem, plan)
                attempt["validation_status"] = validation_status
                attempt["rejection"] = rejection
                if validation_status != "VALID":
                    attempt["classification"] = "validator-rejected"
                else:
                    try:
                        actions = [
                            serialize_action(action) for action in plan.actions
                        ]
                        expected_goal = expected_goal_for(
                            intention, belief_state, actions
                        )
                    except (TypeError, ValueError, KeyError, StopIteration) as exc:
                        attempt["classification"] = "malformed-action"
                        attempt["rejection"] = "malformed-action"
                        attempt["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        attempt["accepted"] = True
                        attempt["classification"] = "accepted"
                        attempt["elapsed_seconds"] = monotonic() - started
                        return PlanningOutcome(
                            actions,
                            attempt,
                            problem_name,
                            selected_cow,
                            expected_goal,
                        )
        except Exception as exc:
            attempt.update(
                status="EXCEPTION",
                classification="planner-exception",
                rejection="planner-exception",
                error=f"{type(exc).__name__}: {exc}",
            )
        attempt["elapsed_seconds"] = monotonic() - started
        return PlanningOutcome([], attempt, problem_name, selected_cow, None)

    @staticmethod
    def _validate_plan(
        problem: Any,
        plan: SequentialPlan,
    ) -> tuple[str, str | None]:
        """Validate the modeled goal once before any prefix is executable."""

        with PlanValidator(
            problem_kind=problem.kind, plan_kind=plan.kind
        ) as validator:
            result = validator.validate(problem, plan)
        status = result.status.name
        return status, None if status == "VALID" else f"validation-{status.lower()}"


__all__ = [
    "ACHIEVEMENT_GOALS",
    "GroundedAction",
    "PlanningOutcome",
    "PlanningService",
    "LOW_HEALTH_THRESHOLD",
    "RECOVERED_ENERGY",
    "RECOVERED_NEED_THRESHOLD",
    "TechnologyStage",
    "action_sound",
    "build_problem_pddl",
    "choose_exploration_action",
    "derive_technology_stage",
    "expected_goal_for",
    "highest_milestone",
    "plan_goal_observed",
    "recovery_entry_threshold",
    "recovery_complete",
    "recovery_supported",
    "select_recovery_intention",
    "serialize_action",
    "stage_supported",
]
