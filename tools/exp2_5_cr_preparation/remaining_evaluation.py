"""World-held-out audits for remaining policies on intact, native Crafter worlds.

Descriptors are fixed before inference. No expert or model outcome participates
in sampling. Test access requires a passing validation report for the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

from cases import (
    DIRECTIONS, Case, CaseRequest, _cow_sector, _direction, current_target,
    frontier_cells, make_env, neighbors, position, run_case, select_frontier,
    visible_cells, walkable,
)
from mha_exp_level2_cr.exp2_5.contracts import activity_action_bound
from mha_exp_level2_cr.exp2_5.policy import PolicyId, file_sha256, load_policy_checkpoint
from resource_cases import ResourceEnvironment
from stop_control import StopController
from atomic_io import replace_file

POLICIES = (PolicyId.EXPLORE, PolicyId.EAT_TARGET, PolicyId.EAT_COW)
VERSION = 1
COUNTS = {"pilot": 48, "validation": 240, "test": 600}
# Each policy/split owns 10 million seeds; one world per case and 64 retries.
SPLIT_SEEDS = {
    policy: {split: 1_100_000_000 + pi * 100_000_000 + si * 10_000_000
             for si, split in enumerate(("train", "pilot", "validation", "test"))}
    for pi, policy in enumerate(POLICIES)
}
BANDS = {
    PolicyId.EXPLORE: ((2, 4), (5, 7), (8, 16)),
    PolicyId.EAT_TARGET: ((0, 1), (2, 3), (4, 8)),
    PolicyId.EAT_COW: ((1, 2), (3, 4), (5, 8)),
}
SAFETY_FIELDS = ("illegal_actions", "lethal_actions", "environment_terminal_failures")


def atomic_json(value: Any, path: Path) -> None:
    """Commit JSON after a complete case, allowing safe audit continuation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replace_file(temporary, path)


def request(index: int, policy: PolicyId) -> dict[str, Any]:
    """Balance direction, distance and observed-history source independently."""
    return {"direction": index % 4 + 1, "band": (index // 4) % 3,
            "source": ("fresh", "history")[index // 12 % 2],
            "variant": ("single", "distractor")[index // 24 % 2]
            if policy is PolicyId.EAT_TARGET else "ordinary"}


def paths_from(env: Any, start: tuple[int, int], maximum: int = 20) -> dict:
    """Find bounded native walkable routes once per candidate pose."""
    paths = {start: [start]}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if len(paths[cell]) > maximum:
            continue
        for dx, dy in DIRECTIONS.values():
            other = cell[0] + dx, cell[1] + dy
            if other not in paths and 0 <= other[0] < 64 and 0 <= other[1] < 64 and walkable(env, other):
                paths[other] = [*paths[cell], other]
                queue.append(other)
    return paths


class RemainingCases:
    """Persist intact-world cases with a bounded cache and disjoint world seeds."""

    def __init__(self, root: Path, split: str, policy: PolicyId) -> None:
        if policy not in POLICIES or split not in SPLIT_SEEDS[policy]:
            raise ValueError("Unsupported remaining-policy split.")
        self.root = root / policy.value / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy, self.split = policy, split
        self._cache: dict[int, tuple] = {}

    def _restore(self, record: dict) -> tuple[Any, set]:
        """Replay native history without altering terrain or creature placement."""
        env = make_env(record["seed"], environment_type=ResourceEnvironment)
        env.reset()
        if position(env) != tuple(record["start"]):
            env._world.move(env._player, tuple(record["start"]))
        env._player.facing = tuple(record["facing"])
        env._player.inventory.update(record["inventory"])
        known = visible_cells(position(env))
        for action in record["prefix"]:
            env.step(action)
            known.update(visible_cells(position(env)))
        return env, known

    def _sample(self, index: int, seed: int) -> tuple[dict, Any, set] | None:
        """Reject only structural preconditions, never a policy's success/failure."""
        spec = request(index, self.policy)
        bounds = BANDS[self.policy][spec["band"]]
        rng = random.Random(seed)
        original = make_env(seed, environment_type=ResourceEnvironment)
        original.reset()
        starts = [(x, y) for x in range(3, 61) for y in range(3, 61) if walkable(original, (x, y))]
        rng.shuffle(starts)
        for start in starts[:128]:
            env = deepcopy(original)
            if position(env) != start:
                env._world.move(env._player, start)
            facing = rng.choice(list(DIRECTIONS.values()))
            env._player.facing = facing
            inventory = dict(env._player.inventory)
            for name in ("health", "food", "drink", "energy"):
                inventory[name] = rng.randint(4, 9)
            env._player.inventory.update(inventory)
            known = visible_cells(start)
            prefix = []
            if spec["source"] == "history":
                for _ in range(8):
                    moves = [action for action, delta in DIRECTIONS.items()
                             if walkable(env, tuple(a + b for a, b in zip(position(env), delta)))]
                    if not moves:
                        break
                    action = rng.choice(moves)
                    _, _, done, _ = env.step(action)
                    prefix.append(action)
                    known.update(visible_cells(position(env)))
                    if done:
                        break
            player = position(env)
            if env._player.health <= 0:
                continue
            paths = paths_from(env, player)
            visible = visible_cells(player)
            cows = sorted((obj for obj in env._world.objects if type(obj).__name__ == "Cow"), key=lambda obj: tuple(obj.pos))
            visible_cows = [cow for cow in cows if tuple(cow.pos) in visible]
            if self.policy is PolicyId.EXPLORE:
                safe = {cell for cell in known if walkable(env, cell)}
                target = select_frontier(player, safe, frontier_cells(known, safe))
                path = paths.get(target)
                if path is None or len(path) < 2 or not set(path) <= known:
                    continue
                distance = len(path) - 1
                direction = _direction(path[0], path[1])
                if len(set().union(*(visible_cells(cell) for cell in path)) - known) < 10:
                    continue
            else:
                if self.policy is PolicyId.EAT_TARGET:
                    if not visible_cows or (len(visible_cows) > 1) != (spec["variant"] == "distractor"):
                        continue
                    candidates = []
                    for cow in visible_cows:
                        routes = [paths[cell] for cell in neighbors(tuple(cow.pos)) if cell in paths]
                        if routes:
                            path = min(routes, key=len)
                            candidates.append((cow, path))
                else:
                    if visible_cows:
                        continue
                    candidates = []
                    for cow in cows:
                        routes = [path for cell, path in paths.items()
                                  if abs(int(cow.pos[0])-cell[0]) <= 4 and abs(int(cow.pos[1])-cell[1]) <= 3]
                        if routes:
                            candidates.append((cow, min(routes, key=len)))
                    # Do not label a world as distant if a closer cow is available.
                    if candidates:
                        closest = min(len(path) for _, path in candidates)
                        candidates = [(cow, path) for cow, path in candidates if len(path) == closest]
                rng.shuffle(candidates)
                eligible = [(cow, path) for cow, path in candidates
                            if bounds[0] <= len(path) - 1 <= bounds[1]
                            and _cow_sector(player, tuple(cow.pos)) == spec["direction"]]
                if not eligible:
                    continue
                cow, path = eligible[0]
                target = tuple(cow.pos)
                distance = len(path) - 1
                direction = _cow_sector(player, target)
            if not bounds[0] <= distance <= bounds[1] or direction != spec["direction"]:
                continue
            record = {"version": VERSION, "index": index, "seed": seed, "policy": self.policy.value,
                    "split": self.split, **spec, "bounds": list(bounds), "start": list(start),
                    "prefix": prefix, "facing": list(facing), "inventory": inventory,
                    "player": list(player), "target": [int(value) for value in target], "distance": distance,
                    "known": [list(cell) for cell in sorted(known)],
                    "visible_cows": len(visible_cows), "terrain_modified": False,
                    "cow_placement_modified": False}
            return record, env, known
        return None

    def get(self, index: int) -> tuple[Any, Case, dict]:
        """Return an independent copy; never resample a revealed failed case."""
        if not 0 <= index < 10000:
            raise ValueError("Case index exceeds reserved world block.")
        if index in self._cache:
            return deepcopy(self._cache[index])
        path = self.root / f"case-{index}.json"
        prepared = None
        if path.exists():
            record = json.loads(path.read_text())
            if record["version"] != VERSION or record["policy"] != self.policy.value or record["split"] != self.split:
                raise ValueError("Case protocol changed; use a fresh case root.")
        else:
            for attempt in range(64):
                seed = SPLIT_SEEDS[self.policy][self.split] + index * 64 + attempt
                sampled = self._sample(index, seed)
                if sampled is not None:
                    record, env, known = sampled
                    prepared = env, known
                    record["generation_rejections"] = attempt
                    atomic_json(record, path)
                    break
            else:
                raise RuntimeError(f"No valid natural case: {self.policy.value}/{self.split}/{index}")
        descriptor_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        snapshot = path.with_suffix(".state.pkl")
        metadata = path.with_suffix(".state.json")
        if snapshot.exists() and metadata.exists():
            info = json.loads(metadata.read_text())
            blob = snapshot.read_bytes()
            if info != {"descriptor_sha256": descriptor_hash, "snapshot_sha256": hashlib.sha256(blob).hexdigest()}:
                raise ValueError("Native snapshot checksum or descriptor mismatch.")
            # Only locally generated, checksum-verified preparation snapshots.
            env, case = pickle.loads(blob)
            self._cache[index] = env, case, record
            if len(self._cache) > 8:
                del self._cache[next(iter(self._cache))]
            return deepcopy((env, case, record))
        env, known = prepared if prepared is not None else self._restore(record)
        if list(position(env)) != record["player"] or known != {tuple(cell) for cell in record["known"]}:
            raise ValueError("Native case replay differs from descriptor.")
        target = tuple(record["target"])
        cow = None if self.policy is PolicyId.EXPLORE else env._world[target][1]
        if self.policy is not PolicyId.EXPLORE and type(cow).__name__ != "Cow":
            raise ValueError("Native cow replay differs from descriptor.")
        req = CaseRequest(self.policy, "frontier" if cow is None else "cow",
                          distance_band=("short", "mid", "long")[record["band"]],
                          minimum_distance=record["bounds"][0], maximum_distance=record["bounds"][1],
                          variant=record["variant"])
        case = Case(req, target, len(known) if cow is None else env._player.achievements["eat_cow"], known,
                    target_object=cow, safe_path_length=record["distance"],
                    manhattan_distance=sum(abs(a-b) for a,b in zip(position(env),target)),
                    initial_facing=tuple(env._player.facing), native_dynamics=True,
                    observed_target=target if self.policy is PolicyId.EAT_TARGET else None)
        blob = pickle.dumps((env, case), protocol=5)
        temporary = snapshot.with_suffix(".tmp")
        temporary.write_bytes(blob)
        replace_file(temporary, snapshot)
        atomic_json({"descriptor_sha256": descriptor_hash, "snapshot_sha256": hashlib.sha256(blob).hexdigest()}, metadata)
        self._cache[index] = env, case, record
        if len(self._cache) > 8:
            del self._cache[next(iter(self._cache))]
        return deepcopy((env, case, record))


def summarize(episodes: list[dict], count: int) -> dict:
    """Apply the newer 95% overall / 90% subgroup criteria to full cohorts."""
    groups = {key: {} for key in ("source", "band", "direction", "variant")}
    for row in episodes:
        for key, group in groups.items():
            item = group.setdefault(str(row[key]), {"attempted": 0, "succeeded": 0})
            item["attempted"] += 1
            item["succeeded"] += int(row["success"])
    safety = {key: sum(row.get(key, 0) for row in episodes) for key in SAFETY_FIELDS}
    successes = sum(row["success"] for row in episodes)
    complete = len(episodes) == count
    structure = all(not row["success"] or row.get("policy") != "eat_cow" or all(
        row.get(key, 0) > 0 for key in ("search_actions", "acquisition_events", "pursuit_actions", "do_actions")
    ) for row in episodes)
    return {"attempted": len(episodes), "required": count, "succeeded": successes,
            "groups": groups, **safety, "harness_errors": 0, "complete": complete,
            "successful_cases_structurally_complete": structure,
            "passed": complete and structure and successes / count >= .95 and not any(safety.values())
            and all(item["succeeded"] / item["attempted"] >= .90 for group in groups.values() for item in group.values()),
            "episodes": episodes}


def evaluate(checkpoint: Path, policy: PolicyId, root: Path, split: str,
             device: str, *, expert: bool = False, limit: int | None = None,
             validation: Path | None = None) -> dict:
    """Audit fixed checkpoint bytes with resumable per-case evidence and timings."""
    import torch
    from train import CONFIG, _set_determinism
    if limit is not None and split != "pilot":
        raise ValueError("Only pilot cohorts may be shortened.")
    count = COUNTS[split] if limit is None else limit
    if not 1 <= count <= COUNTS[split] or expert and split == "test":
        raise ValueError("Invalid audit size or test expert access.")
    digest = file_sha256(checkpoint)
    sources = {str(path): file_sha256(path) for path in (
        Path(__file__), Path(__file__).with_name("cases.py"),
        Path(__file__).with_name("atomic_io.py"),
        Path(load_policy_checkpoint.__code__.co_filename),
        Path(load_policy_checkpoint.__code__.co_filename).with_name("runtime.py"),
        Path(load_policy_checkpoint.__code__.co_filename).with_name("cow_tracking.py"),
        Path(load_policy_checkpoint.__code__.co_filename).with_name("cow_search.py"),
        Path(load_policy_checkpoint.__code__.co_filename).with_name("beliefs.py"),
        Path(load_policy_checkpoint.__code__.co_filename).with_name("contracts.py"),
        Path(load_policy_checkpoint.__code__.co_filename).with_name("deliberation.py"),
    )}
    if split == "test":
        prior = json.loads(validation.read_text()) if validation else {}
        if not prior.get("passed") or prior.get("expert") or prior.get("checkpoint_sha256") != digest or prior.get("protocol_version") != VERSION or prior.get("split") != "validation" or prior.get("required") != COUNTS["validation"] or prior.get("source_sha256") != sources:
            raise ValueError("Test requires a passing current validation report for these exact bytes.")
    _set_determinism(torch, CONFIG["master_seed"], device)
    model, _ = load_policy_checkpoint(torch, checkpoint, expected_policy_id=policy)
    if getattr(model, "uses_discovery_context", False) and device == "cuda":
        # Match the CPU runtime's full-precision action ranking.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model.to(device).eval()
    cases = RemainingCases(root / "cases", split, policy)
    path = root / f'{policy.value}-{split}{"-expert" if expert else ""}.json'
    episodes = []
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["checkpoint_sha256"] != digest or previous["protocol_version"] != VERSION or previous["required"] != count or previous.get("source_sha256") != sources:
            raise ValueError("Audit identity changed; use a fresh report directory.")
        if previous.get("harness_errors"):
            raise ValueError("Diagnose the failed harness before starting a fresh audit.")
        episodes = previous["episodes"]
    identity = {"protocol_version": VERSION, "source_sha256": sources,
                "activity_action_bound": activity_action_bound(policy.value),
                "checkpoint_sha256": digest, "checkpoint": str(checkpoint),
                "policy": policy.value, "split": split, "expert": expert}
    with StopController() as stop:
        for index in range(len(episodes), count):
            started = perf_counter()
            try:
                env, case, record = cases.get(index)
                constructed = perf_counter()
                rows, success, stats = run_case(torch, model, env, case, expert=expert, epsilon=0.,
                                                device=device, seed=record["seed"],
                                                discount=CONFIG["discount"], horizon=CONFIG["n_step_horizon"])
            except Exception as error:
                atomic_json({**summarize(episodes, count), **identity, "passed": False,
                             "harness_errors": 1, "failed_index": index, "error": repr(error)}, path)
                raise
            episodes.append({**record, **stats, "success": success, "actions": len(rows),
                             "case_seconds": constructed-started,
                             "execution_seconds": perf_counter()-constructed})
            result = {**summarize(episodes, count), **identity, "interrupted": stop.requested}
            atomic_json(result, path)
            print(json.dumps({"event": "natural_audit_progress", "policy": policy.value,
                              "split": split, "completed": len(episodes), "total": count,
                              "successes": result["succeeded"], "latest_actions": len(rows),
                              "latest_seconds": perf_counter()-started, **{key: stats[key] for key in SAFETY_FIELDS}}), flush=True)
            if stop.requested:
                raise KeyboardInterrupt
            if any(stats[key] for key in SAFETY_FIELDS):
                break
    return json.loads(path.read_text())


def main() -> int:
    """Run an existing checkpoint through pilot, validation, then guarded test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--policy", choices=[p.value for p in POLICIES], required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--split", choices=COUNTS, default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, PolicyId(args.policy), args.directory, args.split,
                      args.device, expert=args.expert, limit=args.limit, validation=args.validation)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
