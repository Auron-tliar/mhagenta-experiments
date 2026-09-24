"""Crafter environment and the two deterministic MHAgentA bridge modules."""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mhagenta import ActionStatus, Observation, Orchestrator
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, PerceptorState
from PIL import Image

from .llm import (
    CRAFTER_ACTIONS, DERIVED_SYMBOLIC_STATUS_HEADER, EXACT_FACED_CELL_HEADER,
    PARSED_SYMBOLIC_GRID_HEADER, RAW_SYMBOLIC_PREDICATES_HEADER, compact_event,
    normalize_action,
)

K_ACTION = "action"
K_OBSERVATION = "observation"
K_AGENT_POSITION = "agent_position"
K_SURVIVAL_NEEDS = "survival_needs"
A_CLOSE = "close"
REPLAY_FRAME_SIZE = (512, 512)
DAYLIGHT_EFFECTS = False
_PREDICATE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)\s*=\s*(?P<value>.+)$")
_DIRECTIONS = {"L1": "left", "R1": "right", "U1": "up", "D1": "down"}
_OFFSETS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
_ITEMS = ("health", "food", "drink", "energy", "sapling", "wood", "stone", "coal",
          "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
          "wood_sword", "stone_sword", "iron_sword")


def _parse(predicate: str) -> tuple[str, list[str], str]:
    match = _PREDICATE.match(predicate.strip())
    if not match:
        return "", [], ""
    arguments = match.group("args").strip()
    return match.group("name"), ([part.strip() for part in arguments.split(",")] if arguments else []), match.group("value").strip()


def _cell(dx: int, dy: int) -> str:
    horizontal = f"L{-dx}" if dx < 0 else f"R{dx}"
    vertical = f"U{-dy}" if dy < 0 else f"D{dy}"
    return vertical if dx == 0 else horizontal if dy == 0 else f"{horizontal}_{vertical}"


def describe_faced_do_effect(material: str, occupant: str,
                             inventory: Mapping[str, Any], *, sleeping: bool = False) -> str:
    """Describe only the public Crafter interaction at the faced cell."""
    if sleeping:
        return "do has no useful interaction while sleeping"
    target = occupant if occupant not in {"none", "unknown"} else material
    requirements = {"stone": "wood_pickaxe", "coal": "wood_pickaxe",
                    "iron": "stone_pickaxe", "diamond": "iron_pickaxe"}
    if target == "tree":
        return "collect wood from the faced tree"
    if target == "water":
        return "drink from the faced water tile"
    if target in requirements:
        tool = requirements[target]
        available = int(inventory.get(tool, 0) or 0) > 0
        return f"collect {target}" if available else f"cannot collect {target} without {tool}"
    if target in {"cow", "zombie", "skeleton", "plant", "ripe-plant"}:
        return f"interact with faced {target}"
    return "no known useful do effect at this faced cell"


def symbolic_observation_to_text(predicates: Sequence[str], *, agent_position: Sequence[int] = (0, 0)) -> str:
    """Render the authoritative predicates plus a checked, redundant local grid."""
    if len(agent_position) != 2 or any(type(value) is not int for value in agent_position):
        raise ValueError("agent_position must contain exactly two integers")
    snapshot = tuple(predicates)
    if not all(isinstance(item, str) for item in snapshot):
        raise ValueError("symbolic predicates must all be strings")
    facing = "unknown"
    sleeping = False
    inventory: dict[str, str] = {}
    materials: dict[str, str] = {}
    occupants: dict[str, str] = {}
    for predicate in snapshot:
        name, args, value = _parse(predicate)
        if name == "Facing" and len(args) == 1 and value.lower() == "true":
            facing = _DIRECTIONS.get(args[0], "unknown")
        elif name == "Sleeping" and not args:
            sleeping = value.lower() == "true"
        elif name == "Have" and len(args) == 1:
            inventory[args[0]] = value
        elif name == "MadeOf" and len(args) == 2 and value.lower() == "true":
            if args[0] in materials and materials[args[0]] != args[1]:
                raise ValueError(f"contradictory MadeOf predicates for {args[0]}")
            materials[args[0]] = args[1]
        elif name == "OccupiedBy" and len(args) == 2 and value.lower() == "true":
            if args[0] in occupants and occupants[args[0]] != args[1]:
                raise ValueError(f"contradictory OccupiedBy predicates for {args[0]}")
            occupants[args[0]] = args[1]
    if facing not in _OFFSETS:
        raise ValueError("symbolic observation must contain one valid Facing fluent")
    dx, dy = _OFFSETS[facing]
    faced = _cell(dx, dy)
    rows = []
    for row_y in range(-3, 4):
        cells = []
        for col_x in range(-4, 5):
            name = _cell(col_x, row_y)
            if col_x == 0 and row_y == 0:
                cells.append(f"{materials.get(name, 'unknown')},PLAYER:{facing}")
            else:
                cells.append(f"{materials.get(name, 'unknown')},{occupants.get(name, 'unknown')}")
        rows.append(f"dy={row_y:+d} | " + " | ".join(cells))
    return "\n".join([
        RAW_SYMBOLIC_PREDICATES_HEADER,
        "The JSON array below is the complete authoritative Crafter symbolic observation.",
        json.dumps(list(snapshot), ensure_ascii=False),
        DERIVED_SYMBOLIC_STATUS_HEADER,
        f"Facing: {facing}", f"Sleeping: {str(sleeping).lower()}",
        f"Agent-relative position: ({agent_position[0]}, {agent_position[1]})",
        "Inventory and needs: " + ", ".join(f"{item}={inventory.get(item, 'unknown')}" for item in _ITEMS),
        EXACT_FACED_CELL_HEADER,
        f"Facing location: {faced} (dx={dx:+d}, dy={dy:+d})",
        f"Faced material: {materials.get(faced, 'unknown')}",
        f"Faced object: {occupants.get(faced, 'unknown')}",
        "Factual do consequence at this exact cell: " + describe_faced_do_effect(
            materials.get(faced, "unknown"), occupants.get(faced, "unknown"), inventory, sleeping=sleeping),
        PARSED_SYMBOLIC_GRID_HEADER,
        "Redundant rendering of MadeOf and OccupiedBy; axes are agent-relative.",
        "Columns: " + " | ".join(f"dx={value:+d}" for value in range(-4, 5)), *rows,
    ])


def png_bytes(value: Any) -> bytes:
    """Encode an HxWx3 uint8 array losslessly."""
    import numpy as np
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"invalid Crafter image shape={array.shape}, dtype={array.dtype}")
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, format="PNG", compress_level=9)
    return output.getvalue()


def frame_reference(value: Any, artifact_root: str | Path,
                    agent_position: Sequence[int]) -> dict[str, Any]:
    """Store one content-addressed frame and return its compact descriptor."""
    payload = png_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    root = Path(artifact_root)
    path = root / "observations" / f"{digest}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_bytes(payload)
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
    return {"kind": "image_reference", "relative_path": f"observations/{digest}.png",
            "sha256": digest, "mime_type": "image/png", "width": width, "height": height,
            "agent_position": list(agent_position)}


def _inventory(env: Any) -> dict[str, int]:
    return {key: int(value) for key, value in env._player.inventory.items()}


def _achievements(env: Any) -> dict[str, int]:
    return {key: int(value) for key, value in env._player.achievements.items()}


class LLMCrafterEnvironment(MHAEnvBase):
    """Environment-owned scientific outcomes and compact transition evidence."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self.seed = int(init_state.pop("seed"))
        self.observation_format = str(init_state.pop("observation_format"))
        self.artifact_root = str(
            init_state.pop("artifact_root", f"/{Orchestrator.SAVE_SUBDIR}")
        )
        self.env: Any = None
        self.recorder: Any = None
        self.origin = (0, 0)
        self.started = time.monotonic()
        super().__init__(init_state)
        self._build()

    def _build(self) -> None:
        from mha_env_crafter import CrafterEnv
        self.env = CrafterEnv(seed=self.seed, length=10_000, no_mobs=True,
                              symbolic=False, daylight_effects=DAYLIGHT_EFFECTS)
        self.env.reset()
        self.origin = tuple(int(value) for value in self.env._player.pos)
        self.started = time.monotonic()
        self._snapshot()

    def __getstate__(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["env"] = None
        value["recorder"] = None
        return value

    def __setstate__(self, value: dict[str, Any]) -> None:
        self.__dict__.update(value)
        self._build()

    def _ensure_recorder(self) -> None:
        """Start recording lazily in the container, without resetting the world."""
        if self.recorder is None:
            from mha_env_crafter.crafter.recorder import VideoRecorder
            self.recorder = VideoRecorder(self.env, Path(self.artifact_root) / "videos", size=(512, 512), fps=10)
            # The single world has already been reset by _build.
            self.recorder._frames = [self.env.render((512, 512))]

    def _close_recording(self) -> None:
        """Finalize the full or partial episode before environment state saving."""
        self._ensure_recorder()
        path = self.recorder.close()
        self.state["video_path"] = str(Path(path).relative_to(self.artifact_root))
        self.state["video_frames"] = len(self.recorder._frames)

    def _relative_position(self) -> list[int]:
        current = [int(value) for value in self.env._player.pos]
        return [current[index] - self.origin[index] for index in range(2)]

    def _snapshot(self) -> None:
        inventory = _inventory(self.env)
        achievements = _achievements(self.env)
        self.state["final_inventory"] = inventory
        self.state["final_achievements"] = achievements
        self.state["final_position"] = self._relative_position()
        self.state["dead"] = inventory["health"] <= 0
        self.state["highest_primary_achievement"] = next(
            (name for name in reversed(self.state["primary_achievement_order"])
             if achievements.get(name, 0) > 0), None)
        self.state["primary_goal_achieved"] = achievements.get("collect_diamond", 0) > 0

    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_recorder()
        state["observation_requests"] += 1
        observation = self.env.render() if self.observation_format == "image" else list(self.env.symbolic_observation())
        inventory = _inventory(self.env)
        return state, {K_OBSERVATION: observation, K_AGENT_POSITION: self._relative_position(),
                       K_SURVIVAL_NEEDS: {name: inventory[name] for name in ("health", "food", "drink", "energy")}}

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            self._close_recording()
            return state, {"success": True, "closed": True}
        if state["terminal"] or type(action) is not int or action not in range(len(CRAFTER_ACTIONS)):
            return state, {"success": False, "illegal": True, "terminal": state["terminal"],
                           "dead": state["dead"], "reason": "invalid or terminal action"}
        self._ensure_recorder()
        previous = _achievements(self.env)
        _, reward, done, info = self.env.step(action)
        self.recorder._frames.append(self.env.render((512, 512)))
        current = {key: int(value) for key, value in info["achievements"].items()}
        unlocked = sorted(key for key, value in current.items() if value > previous.get(key, 0))
        state["step_count"] += 1
        state["native_return"] += float(reward)
        state["illegal_actions"] += int(bool(info.get("illegal_action", False)))
        state["terminal"] = bool(done)
        self._snapshot()
        state["terminal"] = bool(done) or state["primary_goal_achieved"]
        if state["terminal"]:
            self._close_recording()
        compact_event(Path(self.artifact_root) / "transitions.jsonl", {
            "step": state["step_count"], "action": CRAFTER_ACTIONS[action],
            "reward": float(reward), "illegal": bool(info.get("illegal_action", False)),
            "terminal": state["terminal"], "new_achievements": unlocked,
            "inventory": state["final_inventory"], "position": state["final_position"]})
        return state, {"success": not bool(info.get("illegal_action", False)),
                       "illegal": bool(info.get("illegal_action", False)), "reward": float(reward),
                       "terminal": state["terminal"], "dead": state["dead"],
                       "new_achievements": unlocked,
                       "primary_goal_achieved": state["primary_goal_achieved"]}


class CrafterPerceptor(RMQPerceptorBase):
    """Convert an environment observation into one native MHAgentA Observation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.observation_format = str(kwargs.pop("observation_format"))
        self.artifact_root = str(
            kwargs.pop("artifact_root", f"/{Orchestrator.SAVE_SUBDIR}")
        )
        self.env_id = ""
        super().__init__(*args, **kwargs)

    def on_first(self, state: PerceptorState) -> PerceptorState:
        self.env_id = state.directory.external.environment.address["env_id"]
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        state["requests"] += 1
        self.observe(self.env_id)
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        position = kwargs[K_AGENT_POSITION]
        raw = kwargs[K_OBSERVATION]
        if self.observation_format == "image":
            content: Any = frame_reference(raw, self.artifact_root, position)
            kind = "crafter-image-reference"
        else:
            content = symbolic_observation_to_text(raw, agent_position=position)
            kind = "crafter-symbolic-text"
        state["observations"] += 1
        state["last_reference"] = content if isinstance(content, dict) else hashlib.sha256(content.encode()).hexdigest()
        state.outbox.send_observation("llreasoner_0", Observation(content, observation_type=kind),
                                      survival_needs=kwargs[K_SURVIVAL_NEEDS], agent_position=position)
        compact_event(Path(self.artifact_root) / "events" / f"{self.module_id}.jsonl",
                      {"kind": "send", "type": "send_observation", "recipient": "llreasoner_0",
                       "observation_format": self.observation_format})
        return state


class StringCrafterActuator(RMQActuatorBase):
    """Apply one canonical string action; it never chooses an action."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.env_id = ""
        self.pending: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def on_first(self, state: ActuatorState) -> ActuatorState:
        self.env_id = state.directory.external.environment.address["env_id"]
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        requested = kwargs.get(K_ACTION)
        canonical, source = normalize_action(requested)
        state["requests"] += 1
        if canonical is None or self.pending is not None:
            state["rejected"] += 1
            state.outbox.send_status("llreasoner_0", ActionStatus({"success": False,
                "requested_action": requested, "reason": "invalid or overlapping action"}))
            return state
        self.pending = {"requested_action": requested, "canonical_action": canonical,
                        "normalization": source}
        self.act(self.env_id, action=CRAFTER_ACTIONS.index(canonical))
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        status = {**(self.pending or {}), **kwargs}
        self.pending = None
        state["statuses"] += 1
        state["terminal"] = bool(status.get("terminal", False))
        state.outbox.send_status("llreasoner_0", ActionStatus(status))
        compact_event(Path(str(state["artifact_root"])) / "events" / f"{self.module_id}.jsonl",
                      {"kind": "send", "type": "send_status", "recipient": "llreasoner_0",
                       "success": bool(status.get("success")),
                       "canonical_action": status.get("canonical_action")})
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        if self.env_id:
            self.act(self.env_id, action=A_CLOSE)
        return state


__all__ = ["CrafterPerceptor", "DAYLIGHT_EFFECTS", "LLMCrafterEnvironment",
           "StringCrafterActuator", "describe_faced_do_effect", "frame_reference",
           "png_bytes", "symbolic_observation_to_text"]
