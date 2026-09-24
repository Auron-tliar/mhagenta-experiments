"""Seed-separated natural and later-game GetResource cases, without geometric fixtures."""

from __future__ import annotations

import json
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

from cases import (
    DIRECTIONS,
    Case,
    CaseRequest,
    make_env,
    neighbors,
    position,
    visible_cells,
    walkable,
)
from mha_env_crafter import CrafterEnv
from mha_exp_level2_cr.exp2_1.policy import ACTIONS, RESET_ACTION, ReactiveCrafterPolicy
from mha_exp_level2_cr.exp2_5.contracts import RESOURCE_ITEMS, RESOURCE_TOOLS
from mha_exp_level2_cr.exp2_5.policy import PolicyId

SPLIT_SEEDS = {
    PolicyId.NAVIGATE_TO: {
        "train": 20_000_000,
        "validation": 321_000_000,
        "test": 622_000_000,
        "pilot": 921_000_000,
    },
    PolicyId.GET_RESOURCE: {
        "train": 30_000_000,
        "validation": 330_000_000,
        "test": 630_000_000,
        "pilot": 930_000_000,
    },
}
DISTANCES = ((0, 1), (2, 3), (4, 6), (7, 10))
NAVIGATION_BANDS = {"short": (1, 2), "mid": (3, 5), "long": (6, 10)}
SOURCES = ("natural", "natural", "table", "furnace")
CASE_FORMAT_VERSION = 5
SOURCE_STATE_COUNT = 64


class ResourceEnvironment(CrafterEnv):
    """Keep native mechanics but make seeded creature balancing independent of object addresses."""

    def _balance_chunk(self, chunk: tuple[int, ...], objs: Any) -> None:
        """Order the upstream object set before its random indexed despawn selection."""

        ordered = sorted(objs, key=lambda obj: (type(obj).__name__, tuple(obj.pos)))
        super()._balance_chunk(chunk, ordered)


def case_request(index: int) -> tuple[str, str, tuple[int, int]]:
    """Balance resource, state source, and distance independently of geometric layout."""

    band = (index // 6 + index // 24) % 4
    return tuple(RESOURCE_ITEMS)[index % 6], SOURCES[index // 6 % 4], DISTANCES[band]


def navigation_request(index: int) -> tuple[str, tuple[int, int], int, str]:
    """Balance source, distance, direction, and realizable route geometry."""

    names = tuple(NAVIGATION_BANDS)
    direction = index % 4 + 1
    source = SOURCES[index // 4 % 4]
    band = names[index % len(names)]
    variant = "clear" if band == "short" else ("clear", "obstructed")[index // 3 % 2]
    return source, NAVIGATION_BANDS[band], direction, variant


class ResourceCases:
    """Persist randomized policy cases; assign at most eight cases to one world."""

    def __init__(self, root: Path, split: str, policy: PolicyId = PolicyId.GET_RESOURCE) -> None:
        if policy not in SPLIT_SEEDS:
            raise ValueError("Randomized cases support only NavigateTo and GetResource.")
        self.root = root / policy.value / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.policy = policy
        self._cache: dict[int, tuple[Any, Case, dict[str, Any]]] = {}

    def _trajectory(self, seed: int, source: str) -> dict[str, Any]:
        """Generate enough native 2-1-CR states for one requested source."""

        path = self.root / f"world-{seed}-{source}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("format_version") != CASE_FORMAT_VERSION:
                raise ValueError("Resource trajectory predates deterministic replay; use a fresh case directory.")
            return record
        print(json.dumps({"event": "source_rollout_started", "split": self.split, "seed": seed}), flush=True)
        env = make_env(seed, environment_type=ResourceEnvironment)
        env.reset()
        policy = ReactiveCrafterPolicy(seed)
        record: dict[str, Any] = {"format_version": CASE_FORMAT_VERSION, "seed": seed,
                                  "source": source, "actions": [], "states": []}
        for _ in range(900):
            decision = policy.choose_action(list(env.symbolic_observation()))
            if decision.action == RESET_ACTION:
                break
            action = ACTIONS.index(decision.action)
            _, _, done, _ = env.step(action)
            record["actions"].append(action)
            if done:
                break
            if not env._player.sleeping and env._player.health > 0:
                table = env._player.achievements["place_table"]
                furnace = env._player.achievements["place_furnace"]
                eligible = bool(furnace) if source == "furnace" else bool(table and not furnace)
                if eligible:
                    record["states"].append(len(record["actions"]))
                    if len(record["states"]) >= SOURCE_STATE_COUNT:
                        break
                elif source == "table" and furnace and record["states"]:
                    break
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(path)
        print(json.dumps({"event": "source_rollout_completed", "split": self.split, "seed": seed,
                          "source": source, "actions": len(record["actions"]),
                          "source_states": len(record["states"])}), flush=True)
        return record

    def _navigation_sample(
        self,
        index: int,
        seed: int,
        source: str,
        bounds: tuple[int, int],
        direction: int,
        variant: str,
        actions: list[int],
        env: Any,
        rng: random.Random,
    ) -> dict[str, Any] | None:
        """Find one navigation case with a single bounded search from each candidate start."""

        stations = [(x, y) for x in range(64) for y in range(64)
                    if env._world[x, y][0] in {"table", "furnace"}]
        players = [(x, y) for x in range(64) for y in range(64) if walkable(env, (x, y))]
        rng.shuffle(players)
        if stations:
            players.sort(key=lambda cell: not any(station in visible_cells(cell) for station in stations))
        for player in players:
            paths = {player: [player]}
            queue = deque([player])
            candidates = []
            visible = visible_cells(player)
            while queue:
                cell = queue.popleft()
                path = paths[cell]
                length = len(path) - 1
                if bounds[0] <= length <= bounds[1] and cell in visible:
                    first = next(action for action, delta in DIRECTIONS.items()
                                 if delta == (path[1][0] - player[0], path[1][1] - player[1]))
                    obstructed = length > sum(abs(a - b) for a, b in zip(player, cell, strict=True))
                    if first == direction and obstructed == (variant == "obstructed"):
                        candidates.append((cell, path))
                if length >= bounds[1]:
                    continue
                for delta in DIRECTIONS.values():
                    other = cell[0] + delta[0], cell[1] + delta[1]
                    if (other not in paths and 0 <= other[0] < 64 and 0 <= other[1] < 64
                            and walkable(env, other)):
                        paths[other] = [*path, other]
                        queue.append(other)
            if not candidates:
                continue
            target, route = rng.choice(candidates)
            inventory = dict(env._player.inventory)
            for name in ("food", "drink", "energy"):
                inventory[name] = rng.randint(1, 8)
            inventory["health"] = rng.randint(3, 9)
            for name in ("wood", "stone", "coal", "iron", "diamond", "sapling"):
                inventory[name] = rng.randint(0, 8)
            for tool in RESOURCE_TOOLS.values():
                inventory[tool] = rng.randint(0, 1)
            return {
                "format_version": CASE_FORMAT_VERSION, "index": index,
                "split": self.split, "seed": seed, "prefix": actions,
                "source": source, "kind": "safe_cell", "distance": len(route) - 1,
                "bounds": list(bounds), "direction": direction, "variant": variant,
                "player": list(player), "target": list(target),
                "facing": list(rng.choice(list(DIRECTIONS.values()))), "inventory": inventory,
                "manhattan_distance": sum(abs(a - b) for a, b in zip(player, target, strict=True)),
            }
        return None

    def _sample(self, index: int, seed: int) -> dict[str, Any] | None:
        """Sample pose and target on intact terrain, rejecting only invalid preconditions."""

        if self.policy is PolicyId.GET_RESOURCE:
            kind, source, bounds = case_request(index)
            direction = None
            variant = None
        else:
            source, bounds, direction, variant = navigation_request(index)
            kind = "safe_cell"
        rng = random.Random(seed * 1009 + index)
        actions: list[int] = []
        if source != "natural":
            trajectory = self._trajectory(seed, source)
            if not trajectory["states"]:
                return None
            actions = trajectory["actions"][:rng.choice(trajectory["states"])]
        env = make_env(seed, environment_type=ResourceEnvironment)
        env.reset()
        for action in actions:
            env.step(action)
        if self.policy is PolicyId.NAVIGATE_TO:
            return self._navigation_sample(
                index, seed, source, bounds, direction, variant, actions, env, rng,
            )
        targets = [(x, y) for x in range(64) for y in range(64)
                   if env._world[x, y][0] == kind]
        rng.shuffle(targets)
        stations = [(x, y) for x in range(64) for y in range(64)
                    if env._world[x, y][0] in {"table", "furnace"}]
        if stations:
            targets.sort(key=lambda cell: min(max(abs(cell[0] - s[0]), abs(cell[1] - s[1])) for s in stations) > 7)
        for target in targets:
            starts = [cell for cell in neighbors(target) if walkable(env, cell)]
            distance = dict.fromkeys(starts, 0)
            queue = deque(starts)
            while queue:
                cell = queue.popleft()
                if distance[cell] >= bounds[1]:
                    continue
                for other in sorted(neighbors(cell)):
                    if other not in distance and walkable(env, other):
                        distance[other] = distance[cell] + 1
                        queue.append(other)
            eligible = []
            for cell, length in distance.items():
                if not bounds[0] <= length <= bounds[1] or target not in visible_cells(cell):
                    continue
                eligible.append(cell)
            if not eligible:
                continue
            with_station = [cell for cell in eligible if any(station in visible_cells(cell) for station in stations)]
            player = rng.choice(sorted(with_station or eligible))
            inventory = dict(env._player.inventory)
            for name in ("food", "drink", "energy"):
                inventory[name] = rng.randint(1, 8)
            inventory["health"] = rng.randint(3, 9)
            for name in ("wood", "stone", "coal", "iron", "diamond", "sapling"):
                inventory[name] = rng.randint(0, 8)
            for tool in RESOURCE_TOOLS.values():
                inventory[tool] = rng.randint(0, 1)
            if kind in RESOURCE_TOOLS:
                inventory[RESOURCE_TOOLS[kind]] = 1
            return {"format_version": CASE_FORMAT_VERSION, "index": index,
                    "split": self.split, "seed": seed, "prefix": actions,
                    "source": source, "kind": kind, "distance": distance[player], "bounds": list(bounds),
                    "direction": direction, "variant": variant,
                    "player": list(player), "target": list(target), "facing": list(rng.choice(list(DIRECTIONS.values()))),
                    "inventory": inventory,
                    "manhattan_distance": max(0, sum(abs(a - b) for a, b in zip(
                        player, target, strict=True)) - 1)}
        return None

    def get(self, index: int) -> tuple[Any, Case, dict[str, Any]]:
        """Construct or replay one immutable descriptor without consulting a neural outcome."""

        if not 0 <= index < 1_000_000:
            raise ValueError("Resource case index exceeds its reserved split.")
        if index in self._cache:
            return deepcopy(self._cache[index])
        path = self.root / f"case-{index}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("format_version") != CASE_FORMAT_VERSION:
                raise ValueError("Resource case predates deterministic replay; use a fresh case directory.")
        else:
            for attempt in range(128):
                seed = SPLIT_SEEDS[self.policy][self.split] + (index // 8) * 128 + attempt
                record = self._sample(index, seed)
                if record is not None:
                    record["generation_rejections"] = attempt
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                    temporary.replace(path)
                    break
                if attempt % 10 == 9:
                    print(json.dumps({"event": "resource_case_search", "split": self.split,
                                      "index": index, "rejections": attempt + 1}), flush=True)
            else:
                raise RuntimeError(f"No structurally valid resource case: {self.split}/{index}.")
        env = make_env(record["seed"], environment_type=ResourceEnvironment)
        env.reset()
        for action in record["prefix"]:
            env.step(action)
        if position(env) != tuple(record["player"]):
            env._world.move(env._player, tuple(record["player"]))
        env._player.facing = tuple(record["facing"])
        env._player.inventory.update(record["inventory"])
        if self.policy is PolicyId.NAVIGATE_TO:
            band = next(name for name, bounds in NAVIGATION_BANDS.items() if list(bounds) == record["bounds"])
            request = CaseRequest(self.policy, "safe_cell", distance_band=band,
                                  minimum_distance=record["bounds"][0], maximum_distance=record["bounds"][1],
                                  first_direction=record["direction"], variant=record["variant"])
        else:
            request = CaseRequest(self.policy, record["kind"], variant="clear_aligned")
        case = Case(request, tuple(record["target"]), 0, visible_cells(position(env)),
                    safe_path_length=record["distance"], initial_facing=tuple(record["facing"]),
                    manhattan_distance=record["manhattan_distance"],
                    shortest_path_first_direction=record.get("direction") or 0)
        record = {**record, "visible_stations": sorted({env._world[cell][0]
                  for cell in visible_cells(position(env)) if env._world[cell][0] in {"table", "furnace"}})}
        result = env, case, record
        self._cache[index] = result
        return deepcopy(result)
