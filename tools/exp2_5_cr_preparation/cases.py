"""Real-Crafter v3 cases, experts, rewards, and identity scoring."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from mha_exp_level2_cr.exp2_5.beliefs import (
    WALKABLE_MATERIALS,
    frontier_cells,
    novel_movement_actions,
    select_frontier,
)
from mha_exp_level2_cr.exp2_5.contracts import (
    RESOURCE_ITEMS,
    RESOURCE_TOOLS,
    activity_action_bound,
    resource_collected,
    resource_stagnated,
)
from mha_exp_level2_cr.exp2_5.cow_tracking import associate_cow
from mha_exp_level2_cr.exp2_5.cow_search import discovery_goal
from mha_exp_level2_cr.exp2_5.policy import (
    REWARD_CONTRACT,
    PolicyId,
    encode_context,
    legal_actions,
    select_action,
    selection_actions,
)
from replay import Transition, episode_n_step

ACTION_BOUND = 32
DIRECTIONS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


@dataclass(frozen=True)
class CaseRequest:
    """Immutable structural request for one preparation case."""

    policy: PolicyId
    target_kind: str
    distance_band: str = "none"
    minimum_distance: int = 0
    maximum_distance: int = ACTION_BOUND
    first_direction: int | None = None
    variant: str = "ordinary"

    def __post_init__(self) -> None:
        expected = {
            PolicyId.EXPLORE: {"frontier"},
            PolicyId.NAVIGATE_TO: {"safe_cell"},
            PolicyId.GET_RESOURCE: set(RESOURCE_ITEMS),
            PolicyId.EAT_TARGET: {"cow"},
            PolicyId.EAT_COW: {"cow"},
        }
        if self.target_kind not in expected[self.policy]:
            raise ValueError("Case target kind does not match its policy.")
        if (type(self.minimum_distance) is not int or type(self.maximum_distance) is not int or not 0 <= self.minimum_distance <= self.maximum_distance <= ACTION_BOUND):
            raise ValueError("Case distance bounds are invalid.")
        if self.first_direction is not None and self.first_direction not in DIRECTIONS:
            raise ValueError("Case first direction is invalid.")
        if self.distance_band not in {"none", "short", "mid", "long"}:
            raise ValueError("Case distance band is invalid.")
        allowed_variants = {
            PolicyId.EXPLORE: {"ordinary"},
            PolicyId.NAVIGATE_TO: {"clear", "obstructed"},
            PolicyId.GET_RESOURCE: {
                "clear_aligned", "clear_misaligned", "obstructed_aligned",
                "obstructed_misaligned",
            },
            PolicyId.EAT_TARGET: {"single", "distractor"},
            PolicyId.EAT_COW: {"ordinary", "reacquisition"},
        }
        if self.variant not in allowed_variants[self.policy]:
            raise ValueError("Case variant does not match its policy.")

    @property
    def stratum(self) -> str:
        """Return the stable preparation evidence key for this request."""

        values = (
            self.policy.value,
            self.target_kind,
            self.variant,
            self.distance_band if self.distance_band != "none" else "-",
            str(self.first_direction) if self.first_direction is not None else "-",
        )
        return ":".join(values)


@dataclass
class Case:
    """Preparation-only mutable state for one bounded activity case."""

    request: CaseRequest
    target_cell: tuple[int, int]
    baseline: int
    known: set[tuple[int, int]]
    target_object: Any = None
    safe_path_length: int = 0
    manhattan_distance: int = 0
    shortest_path_first_direction: int = 0
    approach_cell: tuple[int, int] | None = None
    initial_facing: tuple[int, int] = (0, 1)
    distractor_objects: tuple[Any, ...] = ()
    acquired_target: bool = False
    forced_reacquisition: bool = False
    collected_target: bool = False
    native_dynamics: bool = False
    search_goal: tuple[int, int] | None = None
    observed_target: tuple[int, int] | None = None
    observed_cows: tuple[tuple[int, int], ...] | None = None

    @property
    def policy(self) -> PolicyId:
        """Return the policy declared by the immutable request."""

        return self.request.policy

    @property
    def target_kind(self) -> str:
        """Return the target kind declared by the immutable request."""

        return self.request.target_kind


def make_env(seed: int, *, environment_type: type | None = None) -> Any:
    """Create the fixed Crafter configuration, optionally using a preparation-only subclass."""

    from mha_env_crafter import CrafterEnv

    return (environment_type or CrafterEnv)(
        area=(64, 64),
        view=(9, 9),
        size=(64, 64),
        length=1000,
        seed=seed,
        no_mobs=True,
        symbolic=False,
        daylight_effects=False,
        sleep_effects=False,
    )


def position(env: Any) -> tuple[int, int]:
    """Return the current private player cell for offline case construction."""

    return tuple(int(value) for value in env._player.pos)


def walkable(env: Any, cell: tuple[int, int]) -> bool:
    """Return whether a cell accepts one Crafter movement action."""

    material, obj = env._world[cell]
    return material in WALKABLE_MATERIALS and (obj is None or obj is env._player)


def shortest_path(
    env: Any,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    maximum: int = ACTION_BOUND,
    first_direction: int | None = None,
) -> list[tuple[int, int]] | None:
    """Return a deterministic shortest safe path including both endpoints."""

    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        cell, path = queue.popleft()
        if cell in goals:
            return path
        if len(path) - 1 == maximum:
            continue
        deltas = list(DIRECTIONS.values())
        if cell == start and first_direction is not None:
            preferred = DIRECTIONS[first_direction]
            deltas.remove(preferred)
            deltas.insert(0, preferred)
        for delta in deltas:
            nxt = cell[0] + delta[0], cell[1] + delta[1]
            if nxt not in visited and 0 <= nxt[0] < 64 and 0 <= nxt[1] < 64 and walkable(env, nxt):
                visited.add(nxt)
                queue.append((nxt, [*path, nxt]))
    return None


def visible_cells(cell: tuple[int, int]) -> set[tuple[int, int]]:
    """Return the exact clipped 9-by-7 Crafter visible-cell footprint."""

    return {(x, y) for x in range(max(0, cell[0] - 4), min(64, cell[0] + 5)) for y in range(max(0, cell[1] - 3), min(64, cell[1] + 4))}


def neighbors(cell: tuple[int, int]) -> set[tuple[int, int]]:
    """Return four cardinal neighbor cells."""

    return {(cell[0] + dx, cell[1] + dy) for dx, dy in DIRECTIONS.values()}


def initial_explore_beliefs(env: Any) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return player, known cells, and safe cells from the initial footprint."""

    player = position(env)
    known = visible_cells(player)
    safe = {cell for cell in known if env._world[cell][0] in WALKABLE_MATERIALS and (env._world[cell][1] is None or env._world[cell][1] is env._player)}
    safe.add(player)
    return player, known, safe


def _direction(source: tuple[int, int], target: tuple[int, int]) -> int:
    delta = target[0] - source[0], target[1] - source[1]
    return next(action for action, value in DIRECTIONS.items() if value == delta)


def _path_metadata(
    player: tuple[int, int],
    target: tuple[int, int],
    path: list[tuple[int, int]],
) -> tuple[int, int, int, bool]:
    length = len(path) - 1
    manhattan = abs(target[0] - player[0]) + abs(target[1] - player[1])
    first = _direction(path[0], path[1]) if length else 0
    return length, manhattan, first, length > manhattan


def _move_player(env: Any, cell: tuple[int, int], facing_action: int) -> None:
    env._world.move(env._player, cell)
    env._player.facing = DIRECTIONS[facing_action]


def _cow_sector(source: tuple[int, int], target: tuple[int, int]) -> int:
    """Return the dominant cardinal sector containing a private cow fixture."""

    dx, dy = target[0] - source[0], target[1] - source[1]
    if abs(dx) >= abs(dy):
        return 1 if dx < 0 else 2
    return 3 if dy < 0 else 4


def _construct_cow_search_fixture(
    env: Any,
    direction: int,
    discovery_distance: int,
    protected_object: Any = None,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Build a one-cell corridor whose cow becomes visible at an exact step."""

    origin = position(env)
    delta = DIRECTIONS[direction]
    view_radius = 4 if delta[0] else 3

    def cell(step: int) -> tuple[int, int]:
        return origin[0] + step * delta[0], origin[1] + step * delta[1]

    target = cell(view_radius + discovery_distance)
    trace = [cell(step) for step in range(discovery_distance + 1)]
    fixture = set().union(*(visible_cells(item) for item in trace))
    if target not in fixture or any(not 0 <= value < 64 for item in fixture for value in item):
        raise ValueError("Cow search fixture is outside the world.")
    for item in fixture:
        _, obj = env._world[item]
        if obj is not None and obj not in {env._player, protected_object}:
            env._world.remove(obj)
        env._world[item] = "stone"
    for step in range(view_radius + discovery_distance + 1):
        env._world[cell(step)] = "grass"
    return target, trace


def _construct_resource_fixture(env: Any, request: CaseRequest) -> None:
    """Place a visible resource with the requested deterministic geometry."""

    if request.first_direction is None:
        return
    player = position(env)
    forward = DIRECTIONS[request.first_direction]
    side = -forward[1], forward[0]

    def offset(forward_steps: int, side_steps: int) -> tuple[int, int]:
        return (
            player[0] + forward_steps * forward[0] + side_steps * side[0],
            player[1] + forward_steps * forward[1] + side_steps * side[1],
        )

    if request.variant.startswith("obstructed"):
        corridor = (
            {offset(step, 0) for step in range(4)}
            | {offset(3, step) for step in range(1, 3)}
            | {offset(step, 2) for step in range(3)}
        )
        target = offset(0, 3)
        barriers = {offset(step, 1) for step in range(-3, 3)} | {
            offset(-1, 3), offset(1, 3), offset(0, 4),
        }
    elif request.distance_band == "short":
        corridor = {offset(step, 0) for step in range(3)}
        target = offset(2, 1)
        barriers = set()
    elif request.distance_band == "mid":
        corridor = {offset(step, 0) for step in range(4)} | {offset(3, 1)}
        target = offset(3, 2)
        barriers = set()
    else:
        return
    fixture = corridor | barriers | {target}
    if any(not 0 <= cell[0] < 64 or not 0 <= cell[1] < 64 for cell in fixture):
        raise ValueError("Obstructed resource fixture is outside the world.")
    for cell in fixture:
        _, obj = env._world[cell]
        if obj is not None and obj is not env._player:
            env._world.remove(obj)
    for cell in corridor:
        env._world[cell] = "grass"
    for cell in barriers:
        env._world[cell] = "water"
    env._world[target] = request.target_kind


def _construct_navigation_fixture(env: Any, request: CaseRequest) -> None:
    """Build an exact visible corridor for one NavigateTo stratum."""

    if request.first_direction is None:
        return
    player = position(env)
    forward = DIRECTIONS[request.first_direction]
    side = -forward[1], forward[0]

    def cell(forward_steps: int, side_steps: int) -> tuple[int, int]:
        return (
            player[0] + forward_steps * forward[0] + side_steps * side[0],
            player[1] + forward_steps * forward[1] + side_steps * side[1],
        )

    if request.variant == "clear":
        forward_steps, side_steps = {
            "short": (2, 0),
            "mid": (3, 1),
            "long": (4, 3) if forward[0] else (3, 4),
        }[request.distance_band]
        offsets = (
            [(step, 0) for step in range(forward_steps + 1)]
            + [(forward_steps, step) for step in range(1, side_steps + 1)]
        )
    else:
        depth, lateral = {
            "short": (1, 2),
            "mid": (1, 3),
            "long": (2, 3),
        }[request.distance_band]
        offsets = (
            [(step, 0) for step in range(depth + 1)]
            + [(depth, step) for step in range(1, lateral + 1)]
            + [(step, lateral) for step in range(depth - 1, -1, -1)]
        )
    corridor = {cell(*offset) for offset in offsets}
    footprint = visible_cells(player)
    if not corridor <= footprint:
        raise ValueError("NavigateTo fixture is outside the visible footprint.")
    for item in footprint:
        _, obj = env._world[item]
        if obj is not None and obj is not env._player:
            env._world.remove(obj)
        env._world[item] = "grass" if item in corridor else "water"


def _resource_case(env: Any, request: CaseRequest) -> Case:
    _construct_resource_fixture(env, request)
    player = position(env)
    choices: list[tuple[Any, ...]] = []
    for x in range(64):
        for y in range(64):
            target = x, y
            if env._world[target][0] != request.target_kind or target not in visible_cells(player):
                continue
            goals = {cell for cell in neighbors(target) if walkable(env, cell)}
            path = shortest_path(
                env,
                player,
                goals,
                request.maximum_distance,
                request.first_direction,
            )
            if path is None:
                continue
            length, _, first, obstructed = _path_metadata(player, path[-1], path)
            expected_obstruction = request.variant.startswith("obstructed")
            if (
                not request.minimum_distance <= length <= request.maximum_distance
                or request.first_direction is not None and first != request.first_direction
                or obstructed != expected_obstruction
            ):
                continue
            choices.append((-length, target, path[-1], path))
    if not choices:
        raise ValueError(f"No eligible {request.target_kind} target.")
    _, target, approach, path = min(choices)
    for tool in set(RESOURCE_TOOLS.values()):
        env._player.inventory[tool] = 0
    required = RESOURCE_TOOLS.get(request.target_kind)
    if required is not None:
        env._player.inventory[required] = 1
    if request.target_kind == "water":
        env._player.inventory["drink"] = min(env._player.inventory["drink"], 8)
    item = RESOURCE_ITEMS[request.target_kind]
    length, manhattan, first, _ = _path_metadata(player, approach, path)
    facing_action = (
        first
        if "misaligned" not in request.variant
        else {1: 2, 2: 1, 3: 4, 4: 3}[first]
    )
    env._player.facing = DIRECTIONS[facing_action]
    return Case(
        request,
        target,
        int(env._player.inventory[item]),
        visible_cells(position(env)),
        safe_path_length=length,
        manhattan_distance=manhattan,
        shortest_path_first_direction=first,
        approach_cell=approach,
        initial_facing=tuple(env._player.facing),
    )


def _cow_case(env: Any, request: CaseRequest) -> Case:
    """Construct an identity-scored target or target-free cow episode."""

    from mha_env_crafter.crafter.objects import Cow

    for obj in list(env._world.objects):
        if isinstance(obj, Cow):
            env._world.remove(obj)
    # Fixed cow identities are required for target-specific reward attribution.
    env._balance_chunk = lambda *_args, **_kwargs: None
    player = position(env)
    footprint = visible_cells(player)
    if request.policy is PolicyId.EAT_COW:
        if request.first_direction is None:
            raise ValueError("EatCow fixtures require a private cow sector.")
        target, path = _construct_cow_search_fixture(
            env,
            request.first_direction,
            request.minimum_distance,
        )
        approach = None
    else:
        choices: list[tuple[Any, ...]] = []
        for target in ((x, y) for x in range(64) for y in range(64)):
            if target == player or not walkable(env, target) or target not in footprint:
                continue
            for approach in sorted(neighbors(target)):
                path = shortest_path(
                    env,
                    player,
                    {approach},
                    request.maximum_distance,
                    request.first_direction,
                )
                if path is None or target in path:
                    continue
                length, _, first, _ = _path_metadata(player, approach, path)
                if (
                    request.minimum_distance <= length <= request.maximum_distance
                    and (request.first_direction is None or first == request.first_direction)
                ):
                    choices.append((-length, target, approach, path))
                    break
        if not choices:
            raise ValueError("No eligible cow target.")
        _, target, approach, path = min(choices)
    cow = Cow(env._world, target)
    env._world.add(cow)
    distractors: list[Any] = []
    if request.policy is PolicyId.EAT_TARGET and request.variant == "distractor":
        for cell in sorted(visible_cells(player)):
            if cell not in {target, player} and walkable(env, cell) and cell not in path:
                distractor = Cow(env._world, cell)
                env._world.add(distractor)
                distractors.append(distractor)
                break
        if not distractors:
            raise ValueError("No eligible distractor cow cell.")
    for sword in ("wood_sword", "stone_sword", "iron_sword"):
        env._player.inventory[sword] = 0
    length = len(path) - 1
    manhattan = length if approach is None else abs(approach[0] - player[0]) + abs(approach[1] - player[1])
    first = request.first_direction if approach is None else _direction(path[0], path[1])
    baseline = int(env._player.achievements["eat_cow"])
    return Case(
        request,
        target,
        baseline,
        visible_cells(position(env)),
        cow,
        length,
        manhattan,
        first,
        approach,
        tuple(env._player.facing),
        tuple(distractors),
    )


def setup_case(env: Any, request: CaseRequest) -> Case:
    """Construct one deterministic case satisfying an explicit request."""

    player = position(env)
    if request.policy is PolicyId.EXPLORE:
        player, known, safe = initial_explore_beliefs(env)
        target = select_frontier(player, safe, frontier_cells(known, safe))
        path = (shortest_path(
            env,
            player,
            {target},
            first_direction=request.first_direction,
        ) if target is not None else None)
        if path is None:
            raise ValueError("No eligible Explore frontier.")
        length, manhattan, first, _ = _path_metadata(player, target, path)
        revealed = set().union(*(visible_cells(cell) for cell in path)) - known
        if (
            not request.minimum_distance <= length <= request.maximum_distance
            or request.first_direction is not None and first != request.first_direction
            or len(revealed) < 10
        ):
            raise ValueError("No eligible Explore frontier.")
        return Case(
            request,
            target,
            len(known),
            known,
            safe_path_length=length,
            manhattan_distance=manhattan,
            shortest_path_first_direction=first,
            initial_facing=tuple(env._player.facing),
        )
    if request.policy is PolicyId.NAVIGATE_TO:
        _construct_navigation_fixture(env, request)
        choices: list[tuple[Any, ...]] = []
        for target in sorted(visible_cells(player)):
            path = (shortest_path(
                env,
                player,
                {target},
                request.maximum_distance,
                request.first_direction,
            ) if walkable(env, target) else None)
            if path is None:
                continue
            length, _, first, obstructed = _path_metadata(player, target, path)
            if (
                not request.minimum_distance <= length <= request.maximum_distance
                or request.first_direction is not None and first != request.first_direction
                or obstructed != (request.variant == "obstructed")
            ):
                continue
            choices.append((-length, target, path))
        if not choices:
            raise ValueError("No eligible NavigateTo target.")
        _, target, path = min(choices)
        length, manhattan, first, _ = _path_metadata(player, target, path)
        return Case(
            request,
            target,
            0,
            set(path),
            safe_path_length=length,
            manhattan_distance=manhattan,
            shortest_path_first_direction=first,
            initial_facing=tuple(env._player.facing),
        )
    if request.policy is PolicyId.GET_RESOURCE:
        return _resource_case(env, request)
    return _cow_case(env, request)

def current_target(case: Case) -> tuple[int, int] | None:
    """Associate native EatTarget observations or acquire a cow for EatCow."""

    if case.native_dynamics and case.policy in {PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
        # Private identity is scoring-only in native cases.
        objects = case.target_object.world.objects
        player = next(obj for obj in objects if type(obj).__name__ == "Player")
        visible = visible_cells(tuple(player.pos))
        cows = [obj for obj in objects if type(obj).__name__ == "Cow" and tuple(obj.pos) in visible]
        if case.policy is PolicyId.EAT_TARGET:
            positions = tuple(sorted(tuple(int(value) for value in obj.pos) for obj in cows))
            if case.observed_cows is None:
                target = case.observed_target if case.observed_target in positions else None
            else:
                target = associate_cow(case.observed_target, case.observed_cows, positions)
            case.observed_target, case.observed_cows = target, positions
            return target
        reference = case.observed_target or tuple(player.pos)
        chosen = min(cows, key=lambda obj: (sum(abs(a - b) for a, b in zip(obj.pos, reference)), tuple(obj.pos))) if cows else None
        case.observed_target = tuple(int(value) for value in chosen.pos) if chosen is not None else None
        if chosen is not None and case.policy is PolicyId.EAT_COW:
            case.target_object = chosen
        return case.observed_target
    if case.target_object is not None:
        objects = case.target_object.world.objects
        if case.policy is PolicyId.EAT_COW:
            players = [obj for obj in objects if type(obj).__name__ == "Player"]
            if not players:
                return None
            player = tuple(int(value) for value in players[0].pos)
            visible = visible_cells(player)
            cows = [obj for obj in objects if type(obj).__name__ == "Cow" and tuple(obj.pos) in visible]
            if not cows:
                return None
            case.target_object = min(
                cows,
                key=lambda obj: (
                    abs(int(obj.pos[0]) - player[0]) + abs(int(obj.pos[1]) - player[1]),
                    tuple(int(value) for value in obj.pos),
                ),
            )
        elif case.target_object not in objects:
            return None
        case.target_cell = tuple(int(value) for value in case.target_object.pos)
        if case.policy is PolicyId.EAT_COW and case.target_cell not in visible:
            return None
    return case.target_cell


def metric(env: Any, case: Case) -> int:
    """Return the case's fresh success metric."""

    if case.policy is PolicyId.EXPLORE:
        return len(case.known)
    if case.policy is PolicyId.NAVIGATE_TO:
        return int(position(env) == case.target_cell)
    if case.policy is PolicyId.GET_RESOURCE:
        return int(env._player.inventory[RESOURCE_ITEMS[case.target_kind]])
    return int(env._player.achievements["eat_cow"])


def succeeded(env: Any, case: Case) -> bool:
    """Return whether the case has fresh terminal success evidence."""

    if case.policy is PolicyId.EXPLORE:
        return metric(env, case) >= case.baseline + 10
    if case.policy is PolicyId.NAVIGATE_TO:
        return position(env) == case.target_cell
    if case.policy is PolicyId.GET_RESOURCE:
        return case.collected_target
    if case.policy is PolicyId.EAT_TARGET:
        return (
            case.target_object not in env._world.objects
            and case.target_object.health <= 0
        )
    if case.policy is PolicyId.EAT_COW:
        return metric(env, case) > case.baseline
    return metric(env, case) > case.baseline


def _visible_safe(env: Any) -> set[tuple[int, int]]:
    player = position(env)
    return {cell for cell in visible_cells(player) if env._world[cell][0] in WALKABLE_MATERIALS and (env._world[cell][1] is None or env._world[cell][1] is env._player)} | {player}


def _visible_path(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    safe: set[tuple[int, int]],
    first_direction: int | None = None,
) -> list[tuple[int, int]] | None:
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        cell, path = queue.popleft()
        if cell in goals:
            return path
        deltas = list(DIRECTIONS.values())
        if cell == start and first_direction is not None:
            preferred = DIRECTIONS[first_direction]
            deltas.remove(preferred)
            deltas.insert(0, preferred)
        for delta in deltas:
            nxt = cell[0] + delta[0], cell[1] + delta[1]
            if nxt in safe and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, [*path, nxt]))
    return None


def visible_cow_pursuit_path(
    player: tuple[int, int], target: tuple[int, int], safe: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Break equally short observed routes by next-step cow visibility margin."""
    goals = neighbors(target) & safe
    possible = {target} | neighbors(target)
    paths = []
    for direction in DIRECTIONS:
        path = _visible_path(player, goals, safe, direction)
        if path is not None and len(path) > 1:
            risk = len(possible - visible_cells(path[1]))
            # The argument above orders BFS ties; the returned first step wins.
            paths.append((len(path), risk, _direction(path[0], path[1]), path))
    return min(paths)[3] if paths else None


def observable_cow_search(
    env: Any, case: Case, movement_actions: tuple[int, ...] = (1, 2, 3, 4),
) -> int:
    """Keep an observable discovery goal reachable under the execution mask."""

    player = position(env)
    visible = visible_cells(player)
    safe = {cell for cell in case.known if env._world[cell][0] in WALKABLE_MATERIALS
            and (cell not in visible or walkable(env, cell))} | {player}
    paths = {player: [player]}
    queue = deque([player])
    while queue:
        cell = queue.popleft()
        if len(paths[cell]) >= 32:
            continue
        for action, (dx, dy) in DIRECTIONS.items():
            if cell == player and action not in movement_actions:
                continue
            other = cell[0] + dx, cell[1] + dy
            if other in safe and other not in paths:
                paths[other] = [*paths[cell], other]
                queue.append(other)
    goal = case.search_goal
    if goal not in paths or goal == player:
        choices = []
        for frontier in frontier_cells(case.known, safe):
            path = paths.get(frontier)
            if path is None or len(path) < 2:
                continue
            gain = len(visible_cells(frontier) - case.known)
            if gain:
                choices.append((-gain / (len(path) - 1), len(path), frontier))
        if not choices:
            # Moving inside known terrain can reveal a new camera strip even
            # when every walkable tile on the known boundary is disconnected.
            for cell, path in paths.items():
                gain = len(visible_cells(cell) - case.known)
                if len(path) > 1 and gain:
                    choices.append((-gain / (len(path) - 1), len(path), cell))
        if not choices:
            raise ValueError("No observable frontier route to an ungrounded cow.")
        goal = min(choices)[2]
        case.search_goal = goal
    return _direction(paths[goal][0], paths[goal][1])


def expert_action(
    env: Any, case: Case, first_direction: int | None = None,
    *, movement_actions: tuple[int, ...] = (1, 2, 3, 4),
) -> int:
    """Return one deterministic preparation-only expert action."""

    target = current_target(case)
    if case.native_dynamics and case.policy is PolicyId.EAT_COW:
        if target is None:
            return observable_cow_search(env, case, movement_actions)
        case.search_goal = None
    if target is None and case.policy is PolicyId.EAT_COW:
        safe = {cell for cell in case.known if walkable(env, cell)}
        frontier = select_frontier(position(env), safe, frontier_cells(case.known, safe))
        path = _visible_path(position(env), {frontier}, safe) if frontier is not None else None
        if path is not None and len(path) > 1:
            return _direction(path[0], path[1])
        raise ValueError("No observable frontier route to an ungrounded cow.")
    if target is None:
        raise ValueError("Active target was lost.")
    player = position(env)
    if case.policy in {PolicyId.GET_RESOURCE, PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
        if sum(abs(a - b) for a, b in zip(player, target, strict=True)) == 1:
            facing = target[0] - player[0], target[1] - player[1]
            if tuple(env._player.facing) == facing:
                return 5
            return next(action for action, delta in DIRECTIONS.items() if delta == facing)
        goals = {cell for cell in neighbors(target) if walkable(env, cell)}
        if case.native_dynamics and case.policy in {PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
            path = visible_cow_pursuit_path(player, target, _visible_safe(env))
        else:
            path = (_visible_path(player, goals, _visible_safe(env), first_direction)
                    if case.native_dynamics else shortest_path(env, player, goals, first_direction=first_direction))
        if path is not None and len(path) > 1:
            return _direction(path[0], path[1])
        if case.native_dynamics and case.policy is PolicyId.EAT_COW:
            return observable_cow_search(env, case, movement_actions)
        raise ValueError("No safe route to interaction target.")

    if case.policy is PolicyId.NAVIGATE_TO:
        path = shortest_path(env, player, {target}, first_direction=first_direction)
        if path is not None and len(path) > 1:
            return _direction(path[0], path[1])
        waiting_cells = {
            cell for cell in neighbors(target)
            if cell != player and 0 <= cell[0] < 64 and 0 <= cell[1] < 64 and walkable(env, cell)
        }
        path = shortest_path(env, player, waiting_cells, first_direction=first_direction)
        if path is not None and len(path) > 1:
            return _direction(path[0], path[1])
        immediate = []
        for action, delta in DIRECTIONS.items():
            cell = player[0] + delta[0], player[1] + delta[1]
            if 0 <= cell[0] < 64 and 0 <= cell[1] < 64 and walkable(env, cell):
                immediate.append((sum(abs(a - b) for a, b in zip(cell, target, strict=True)), action))
        if immediate:
            return min(immediate)[1]
        raise ValueError("No safe route to navigation target.")

    safe = _visible_safe(env)
    if target in safe:
        path = _visible_path(player, {target}, safe, first_direction)
        if path is not None and len(path) > 1:
            return _direction(path[0], path[1])
    footprint = visible_cells(player)
    boundary = {cell for cell in safe if any((cell[0] + dx, cell[1] + dy) not in footprint for dx, dy in DIRECTIONS.values())}
    reachable: list[tuple[Any, ...]] = []
    for cell in boundary:
        path = _visible_path(player, {cell}, safe, first_direction)
        if path is not None and len(path) > 1:
            distance = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
            reachable.append((distance, -(len(path) - 1), cell, path))
    if reachable:
        path = min(reachable)[3]
        return _direction(path[0], path[1])
    immediate = []
    for action, delta in DIRECTIONS.items():
        cell = player[0] + delta[0], player[1] + delta[1]
        if cell in safe:
            distance = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
            immediate.append((distance, action))
    if immediate:
        return min(immediate)[1]
    raise ValueError("No observable route to target.")


def network_action(
    torch: Any,
    model: Any,
    frame: np.ndarray,
    context: np.ndarray,
    device: str,
    policy: PolicyId,
    facing: tuple[int, int],
    movement_actions: tuple[int, ...],
    *,
    interaction_available: bool | None = None,
) -> tuple[int, list[float]]:
    """Return one finite state-masked greedy action and raw Q vector."""

    image = torch.as_tensor(
        np.transpose(frame, (2, 0, 1))[None],
        dtype=torch.uint8,
        device=device,
    )
    target = torch.as_tensor(context[None], dtype=torch.float32, device=device)
    with torch.inference_mode():
        values = model(image, target).detach().cpu().numpy()[0]
    if not np.all(np.isfinite(values)):
        raise ValueError("Policy produced non-finite Q values.")
    actions = selection_actions(
        policy, context, facing, movement_actions,
        interaction_available=interaction_available,
    )
    return select_action(values, actions), values.tolist()



def exploratory_action(
    rng: random.Random,
    policy: PolicyId,
    context: np.ndarray,
    facing: tuple[int, int],
    movement_actions: tuple[int, ...] | None = None,
) -> int:
    """Sample an epsilon action from the state-valid observable action set."""

    available = tuple(DIRECTIONS) if movement_actions is None else movement_actions
    if policy in {PolicyId.EXPLORE, PolicyId.NAVIGATE_TO}:
        return rng.choice(available)
    if policy is PolicyId.GET_RESOURCE:
        dx, dy = np.rint(context * 63).astype(int)
        actions = list(available)
        if abs(dx) + abs(dy) == 1 and facing == (int(dx), int(dy)):
            actions.append(5)
        return rng.choice(actions)
    if policy not in {PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
        return rng.choice(legal_actions(policy))
    if policy is PolicyId.EAT_COW and not np.any(context):
        return rng.choice(available)
    dx, dy = np.rint(context * 63).astype(int)
    if abs(dx) + abs(dy) == 1:
        delta = int(dx), int(dy)
        if facing == delta:
            return 5
        turn = next(action for action, value in DIRECTIONS.items() if value == delta)
        if turn in available:
            return turn
    directions = [
        action
        for action, delta in DIRECTIONS.items()
        if action in available and (dx * delta[0] > 0 or dy * delta[1] > 0)
    ]
    return rng.choice(directions or available)


def _available_movement_actions(
    env: Any,
    interaction_target: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    """Return safe moves plus a needed turn toward an interaction target."""

    player = position(env)
    return tuple(
        action
        for action, delta in DIRECTIONS.items()
        if walkable(env, (player[0] + delta[0], player[1] + delta[1]))
        or (
            interaction_target == (player[0] + delta[0], player[1] + delta[1])
            and tuple(env._player.facing) != delta
        )
    )


def _episode_stats(case: Case) -> dict[str, Any]:
    return {
        "stratum": case.request.stratum,
        "distance_band": case.request.distance_band,
        "first_direction": 0,
        "obstructed": int(case.safe_path_length > case.manhattan_distance),
        "interaction_hits": 0,
        "illegal_actions": 0,
        "first_illegal_action": None,
        "lethal_actions": 0,
        "environment_terminal_failures": 0,
        "network_actions": 0,
        "random_actions": 0,
        "expert_labels": 0,
        "target_lost": 0,
        "target_moves": 0,
        "target_cow_consumed": 0,
        "wrong_cows_consumed": 0,
        "search_actions": 0,
        "acquisition_events": 0,
        "pursuit_actions": 0,
        "reacquisition_events": 0,
        "do_actions": 0,
        "movement_actions": 0,
        "movement_blocked": 0,
        "safe_path_length": case.safe_path_length,
        "shortest_path_first_direction": case.shortest_path_first_direction,
    }


def run_case(
    torch: Any,
    model: Any,
    env: Any,
    case: Case,
    *,
    expert: bool,
    epsilon: float,
    device: str,
    seed: int,
    discount: float,
    horizon: int,
) -> tuple[list[Transition], bool, dict[str, Any]]:
    """Execute a bounded case, retaining expert prefixes blocked by world dynamics."""

    rng = random.Random(seed + 700_000)
    rows: list[Transition] = []
    stats = _episode_stats(case)
    stats.update(stagnation=0, wrong_target_collections=0, expert_route_blocked=0)
    if case.native_dynamics:
        stats["action_trace"] = []
    resource_history: list[dict[str, Any]] = []
    position_history = [position(env)]
    previous_target = current_target(case)
    waypoint_policy = (case.policy is PolicyId.EAT_COW
                       and getattr(model, "uses_discovery_context", False))
    for _ in range(activity_action_bound(case.policy.value)):
        target = current_target(case)
        visible_target = target
        if target is None and case.policy is PolicyId.EAT_COW:
            stats["search_actions"] += 1
            previous_target = None
        elif target is None:
            stats["target_lost"] = 1
            break
        else:
            if case.policy is PolicyId.EAT_COW:
                if not case.acquired_target:
                    stats["acquisition_events"] += 1
                    case.acquired_target = True
                elif previous_target is None:
                    stats["reacquisition_events"] += 1
                stats["pursuit_actions"] += 1
            stats["target_moves"] += int(previous_target is not None and target != previous_target)
            previous_target = target
        frame = np.asarray(env.render(), dtype=np.uint8)
        context_target = None if case.policy is PolicyId.EXPLORE else visible_target
        context = encode_context(case.policy, position(env), context_target)
        interaction_target = target if case.policy in {
            PolicyId.GET_RESOURCE, PolicyId.EAT_TARGET, PolicyId.EAT_COW,
        } else None
        movement_actions = _available_movement_actions(env, interaction_target)
        interaction_available = None
        if case.policy is PolicyId.EAT_TARGET and not case.native_dynamics:
            # Only the visible faced object's type matters, never its target identity.
            ahead = tuple(a + b for a, b in zip(position(env), env._player.facing, strict=True))
            interaction_available = type(env._world[ahead][1]).__name__ == "Cow"
        if case.policy in {PolicyId.NAVIGATE_TO, PolicyId.GET_RESOURCE, PolicyId.EXPLORE} or (
            case.policy is PolicyId.EAT_COW and target is None
        ):
            recent = (position_history[-5:-1] if case.policy in {PolicyId.EXPLORE, PolicyId.EAT_COW}
                      else position_history[-4:])
            movement_actions = novel_movement_actions(
                movement_actions, position(env), recent,
            )
        if not selection_actions(
            case.policy, context, tuple(env._player.facing), movement_actions,
            interaction_available=interaction_available,
        ):
            stats["movement_blocked"] = 1
            break
        mask_policy = case.policy
        if waypoint_policy:
            if target is None:
                player = position(env)
                visible = visible_cells(player)
                safe = {cell for cell in case.known if env._world[cell][0] in WALKABLE_MATERIALS
                        and (cell not in visible or walkable(env, cell))} | {player}
                case.search_goal = discovery_goal(player, case.known, safe, case.search_goal, movement_actions)
                context = encode_context(case.policy, player, case.search_goal)
                mask_policy = PolicyId.EXPLORE  # A discovery waypoint never enables DO.
            else:
                case.search_goal = None
            if rows:
                rows[-1] = replace(rows[-1], next_context=context.copy())
        imitation_action = None
        if expert:
            first_direction = (
                case.request.first_direction
                if not rows and case.policy is not PolicyId.EAT_COW
                else None
            )
            try:
                if case.native_dynamics and case.policy is PolicyId.EAT_COW:
                    action = expert_action(env, case, first_direction,
                                          movement_actions=movement_actions)
                else:
                    action = expert_action(env, case, first_direction)
            except ValueError as error:
                if case.native_dynamics and str(error) in {
                    "No observable frontier route to an ungrounded cow.",
                    "No safe route to interaction target.", "No observable route to target.",
                }:
                    stats["expert_route_blocked"] = 1
                    break
                blocked = {
                    PolicyId.NAVIGATE_TO: "No safe route to navigation target.",
                    PolicyId.GET_RESOURCE: "No safe route to interaction target.",
                }
                if not rows or str(error) != blocked.get(case.policy):
                    raise
                stats["expert_route_blocked"] = 1
                break
            imitation_action = action
            stats["expert_labels"] += 1
        elif rng.random() < epsilon:
            action = exploratory_action(
                rng,
                mask_policy,
                context,
                tuple(env._player.facing),
                movement_actions,
            )
            stats["random_actions"] += 1
        else:
            action = network_action(
                torch,
                model,
                frame,
                context,
                device,
                mask_policy,
                tuple(env._player.facing),
                movement_actions,
                interaction_available=interaction_available,
            )[0]
            stats["network_actions"] += 1
        if action in DIRECTIONS:
            stats["movement_actions"] += 1
            if stats["first_direction"] == 0:
                stats["first_direction"] = action
        if action == 5:
            stats["do_actions"] += 1
        cows_before = {
            obj for obj in env._world.objects if type(obj).__name__ == "Cow"
        }
        source, facing = position(env), tuple(env._player.facing)
        known_before = len(case.known)
        achievements_before = dict(env._player.achievements)
        _, _, done, info = env.step(action)
        position_history.append(position(env))
        case.known.update(visible_cells(position(env)))
        if case.native_dynamics:
            stats["action_trace"].append({
                "action": action, "source": list(source), "destination": list(position(env)),
                "visible_target": list(target) if target is not None else None,
                "known_gain": len(case.known) - known_before,
                "available_movements": list(movement_actions),
            })
        if case.policy is PolicyId.GET_RESOURCE:
            events = [key for key, count in env._player.achievements.items()
                      if count > achievements_before.get(key, 0)]
            case.collected_target = resource_collected(
                action, source, facing, case.target_cell, case.target_kind, events,
                bool(info.get("illegal_action", False)),
            )
            stats["wrong_target_collections"] += int(
                any(event.startswith("collect_") for event in events) and not case.collected_target)
            resource_history.append({"source": source, "destination": position(env),
                                     "discovered": len(case.known) > known_before,
                                     "collected": case.collected_target})
            stats["stagnation"] = int(resource_stagnated(resource_history))
        if (
            case.policy is PolicyId.EAT_COW
            and case.request.variant == "reacquisition"
            and not case.native_dynamics
            and case.acquired_target
            and not case.forced_reacquisition
            and case.target_object in env._world.objects
        ):
            hidden, _ = _construct_cow_search_fixture(
                env,
                case.request.first_direction,
                2,
                case.target_object,
            )
            env._world.move(case.target_object, hidden)
            case.forced_reacquisition = True
        next_frame = np.asarray(env.render(), dtype=np.uint8)
        cows_after = {
            obj for obj in env._world.objects if type(obj).__name__ == "Cow"
        }
        consumed = {
            cow for cow in cows_before - cows_after if cow.health <= 0
        }
        target_consumed = case.target_object in consumed
        wrong_consumed = len(consumed - {case.target_object})
        stats["target_cow_consumed"] += int(target_consumed)
        stats["wrong_cows_consumed"] += wrong_consumed
        success = succeeded(env, case)
        illegal = bool(info.get("illegal_action", False))
        lethal = int(info.get("inventory", {}).get("health", 1)) <= 0
        if illegal and stats["first_illegal_action"] is None:
            stats["first_illegal_action"] = {
                "action": action,
                "source": list(source),
                "destination": list(position(env)),
                "facing": list(facing),
                "target": list(target) if target is not None else None,
                "target_kind": case.target_kind,
                "movement_actions": list(movement_actions),
                "context": context.tolist(),
                "inventory": dict(info.get("inventory", {})),
            }
        refreshed = current_target(case)
        stats["interaction_hits"] += int(action == 5 and (target_consumed or wrong_consumed))
        stats["illegal_actions"] += int(illegal)
        stats["lethal_actions"] += int(lethal)
        stats["environment_terminal_failures"] += int(done and not success)
        failed = illegal or lethal or done and not success or bool(stats["stagnation"])
        if refreshed is None and case.policy is PolicyId.EAT_TARGET and not success:
            stats["target_lost"] = 1
            failed = True
        next_context = context
        if not success and not failed:
            next_context = encode_context(
                case.policy,
                position(env),
                None if case.policy is PolicyId.EXPLORE else refreshed,
            )
        reward = (
            REWARD_CONTRACT["goal"]
            if success
            else REWARD_CONTRACT["eat_target_wrong_cow"]
            if case.policy is PolicyId.EAT_TARGET and wrong_consumed
            else REWARD_CONTRACT["nonterminal_step"]
        )
        rows.append(Transition(
            frame,
            context,
            action,
            reward,
            next_frame,
            next_context,
            success or failed,
            expert,
            imitation_action=imitation_action,
        ))
        if rows[-1].done:
            return (
                episode_n_step(rows, horizon=horizon, discount=discount),
                success,
                stats,
            )
    if rows and not rows[-1].done:
        last = rows[-1]
        rows[-1] = Transition(
            last.frame,
            last.context,
            last.action,
            last.reward,
            last.next_frame,
            last.next_context,
            True,
            last.expert,
            imitation_action=last.imitation_action,
        )
    return episode_n_step(rows, horizon=horizon, discount=discount), False, stats


def collect_episode(
    torch: Any,
    model: Any,
    request: CaseRequest,
    seed: int,
    *,
    expert: bool,
    epsilon: float,
    device: str,
    discount: float,
    horizon: int,
) -> tuple[list[Transition], bool, dict[str, Any]]:
    """Construct, execute, and close one real-Crafter case."""

    env = make_env(seed)
    env.reset()
    try:
        case = setup_case(env, request)
        return run_case(
            torch,
            model,
            env,
            case,
            expert=expert,
            epsilon=epsilon,
            device=device,
            seed=seed,
            discount=discount,
            horizon=horizon,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
