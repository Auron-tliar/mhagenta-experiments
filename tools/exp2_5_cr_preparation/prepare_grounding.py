"""Generate and validate the one selected 2-5-CR grounding bundle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from mha_exp_level2_cr.exp2_5.grounding import (
    ENVIRONMENT_CONTRACT,
    GROUNDING_TEMPLATES_FILENAME,
    ground_observation,
    load_grounding_templates,
)
from mha_exp_level2_cr.exp2_5.policy import file_sha256


def _alpha_composite(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    result = np.array(background[..., :3], copy=True)
    if overlay.shape[-1] == 4:
        alpha = overlay[..., 3:].astype(np.float32) / 255
        result = (alpha * overlay[..., :3] + (1 - alpha) * result).astype(np.uint8)
    else:
        result[...] = overlay[..., :3]
    return result


def renderer_templates() -> dict[str, np.ndarray]:
    """Build the finite template arrays from the fixed Crafter renderer."""

    from mha_env_crafter.crafter import constants, engine

    textures = engine.Textures(constants.root / "assets")
    unit = np.array((7, 7))
    cell_templates: list[np.ndarray] = []
    materials: list[str] = []
    occupants: list[str] = []
    for material in (*constants.materials, "impassable"):
        background = (
            np.full((7, 7, 3), 127, dtype=np.uint8)
            if material == "impassable"
            else np.array(textures.get(material, unit), copy=True)[..., :3]
        )
        cell_templates.append(background.transpose((1, 0, 2)))
        materials.append(material)
        occupants.append("none")
    cow = np.array(textures.get("cow", unit), copy=True)
    for material in ("grass", "path", "sand"):
        background = np.array(textures.get(material, unit), copy=True)[..., :3]
        cell_templates.append(_alpha_composite(background, cow).transpose((1, 0, 2)))
        materials.append(material)
        occupants.append("cow")

    player_templates: list[np.ndarray] = []
    player_facings: list[str] = []
    terrains = ("grass", "path", "sand", "lava")
    for facing in ("left", "right", "up", "down"):
        overlay = np.array(textures.get(f"player-{facing}", unit), copy=True)
        for material in terrains:
            background = np.array(textures.get(material, unit), copy=True)[..., :3]
            player_templates.append(_alpha_composite(background, overlay).transpose((1, 0, 2)))
            player_facings.append(facing)
    sleeping = np.array(textures.get("player-sleep", unit), copy=True)
    sleeping_templates = [
        _alpha_composite(np.array(textures.get(material, unit), copy=True)[..., :3], sleeping).transpose((1, 0, 2))
        for material in terrains
    ]

    items = tuple(constants.items)
    item_view = engine.ItemView(textures, [9, 2])
    inventory_templates = np.empty((len(items), 10, 7, 7, 3), np.uint8)
    for item_index, item in enumerate(items):
        for count in range(10):
            inventory = {name: 0 for name in items}
            inventory[item] = count
            rendered = item_view(inventory, unit).transpose((1, 0, 2))
            row, column = (item_index // 9) * 7, (item_index % 9) * 7
            inventory_templates[item_index, count] = rendered[row : row + 7, column : column + 7]
    crops: list[tuple[int, int, int, int]] = []
    offsets: list[tuple[int, int]] = []
    for x in range(-4, 5):
        for y in range(-3, 4):
            if (x, y) != (0, 0):
                crops.append(((y + 3) * 7, (x + 4) * 7, 7, 7))
                offsets.append((x, y))
    return {
        "cell_templates": np.stack(cell_templates).astype(np.uint8),
        "cell_materials": np.asarray(materials, dtype=f"<U{max(map(len, materials))}"),
        "cell_occupants": np.asarray(occupants, dtype="<U4"),
        "player_templates": np.stack(player_templates).astype(np.uint8),
        "player_facings": np.asarray(player_facings, dtype="<U5"),
        "sleeping_templates": np.stack(sleeping_templates).astype(np.uint8),
        "inventory_templates": inventory_templates,
        "inventory_items": np.asarray(items, dtype=f"<U{max(map(len, items))}"),
        "inventory_counts": np.arange(10, dtype=np.int16),
        "local_crops": np.asarray(crops, dtype=np.int16),
        "relative_offsets": np.asarray(offsets, dtype=np.int8),
    }


def _expected_public_state(env: Any) -> tuple[str, dict[str, int]]:
    player = env._player
    name = {
        (-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"
    }[tuple(player.facing)]
    return name, {key: int(value) for key, value in player.inventory.items()}


def prepare(output: Path) -> dict[str, Any]:
    """Create current templates and validate movement plus a full sleep/wake cycle."""

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    arrays = renderer_templates()
    template_path = output / GROUNDING_TEMPLATES_FILENAME
    np.savez_compressed(template_path, **arrays)
    environment = {**ENVIRONMENT_CONTRACT, "version": importlib.metadata.version("mha-env-crafter")}
    manifest = {
        "artifact_id": "current", "format_version": 3,
        "variant_id": "deterministic-renderer-template-v2", "environment": environment,
        "input": {"observation_dtype": "uint8", "observation_shape": [64, 64, 3]},
        "geometry": {
            "cell_size": [7, 7], "inventory_grid": [9, 2], "inventory_origin": [49, 0],
            "local_grid": [9, 7], "local_origin": [0, 0], "player_crop": [21, 28, 7, 7],
        },
        "templates": {"filename": GROUNDING_TEMPLATES_FILENAME, "sha256": file_sha256(template_path)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    templates = load_grounding_templates(output)
    from mha_env_crafter import CrafterEnv

    failures: list[dict[str, Any]] = []
    frames = 0
    actions = (0, 1, 2, 3, 4, 5, 0, 1)
    for seed in range(810_000, 810_100):
        env = CrafterEnv(
            area=(64, 64), view=(9, 9), size=(64, 64), length=1000, seed=seed,
            no_mobs=True, symbolic=False, daylight_effects=False, sleep_effects=False,
        )
        env.reset()
        try:
            for step, action in enumerate((None, *actions)):
                if action is not None:
                    env.step(action)
                percept, _ = ground_observation(env.render(), templates)
                expected_facing, expected_inventory = _expected_public_state(env)
                if percept.facing.name.lower() != expected_facing or dict(percept.inventory) != expected_inventory:
                    failures.append({"seed": seed, "step": step, "reason": "public-state mismatch"})
                frames += 1
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    env = CrafterEnv(
        area=(64, 64), view=(9, 9), size=(64, 64), length=1000, seed=810_100,
        no_mobs=True, symbolic=False, daylight_effects=False, sleep_effects=False,
    )
    env.reset()
    try:
        env._player.inventory["energy"] = 8
        previous, _ = ground_observation(env.render(), templates)
        for action in (*([6] * 11), 0):
            env.step(action)
            percept, _ = ground_observation(env.render(), templates, previous_facing=previous.facing)
            expected_facing, expected_inventory = _expected_public_state(env)
            if (percept.facing.name.lower() != expected_facing or dict(percept.inventory) != expected_inventory
                    or percept.sleeping != env._player.sleeping):
                failures.append({"reason": "sleep/wake mismatch"})
            previous = percept
            frames += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if failures or frames != 912:
        raise RuntimeError(f"Grounding validation failed: frames={frames}, failures={failures[:3]}")
    report = {"validated_frames": frames, "failures": failures, "template_sha256": file_sha256(template_path)}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    prepare(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
