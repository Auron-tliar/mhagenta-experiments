"""Deterministic RGB grounding selected for experiment 2-5-CR."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .beliefs import CrafterPercept, Direction, ObservedTile
from .contracts import as_rgb_frame
from .policy import OBSERVATION_SHAPE, file_sha256

GROUNDING_FORMAT_VERSION = 3
GROUNDING_ARTIFACT_DIRNAME = "current"
GROUNDING_MANIFEST_FILENAME = "manifest.json"
GROUNDING_TEMPLATES_FILENAME = "templates.npz"
ENVIRONMENT_CONTRACT = {
    "package": "mha-env-crafter", "version": "0.1.0", "area": [64, 64],
    "view": [9, 9], "render_size": [64, 64], "episode_length": 1000,
    "no_mobs": True, "symbolic": False, "daylight_effects": False,
    "sleep_effects": False,
}
_ARRAYS = {
    "cell_templates", "cell_materials", "cell_occupants", "player_templates",
    "player_facings", "sleeping_templates", "inventory_templates",
    "inventory_items", "inventory_counts", "local_crops", "relative_offsets",
}


class GroundingError(ValueError):
    """Raised when a frame or frozen grounding artifact is incompatible."""


@dataclass(frozen=True)
class GroundingTemplates:
    """Validated immutable arrays used by the runtime recognizer."""

    artifact_id: str
    manifest_sha256: str
    cell_templates: np.ndarray
    cell_materials: tuple[str, ...]
    cell_occupants: tuple[str, ...]
    player_templates: np.ndarray
    player_facings: tuple[str, ...]
    sleeping_templates: np.ndarray
    inventory_templates: np.ndarray
    inventory_items: tuple[str, ...]
    inventory_counts: tuple[int, ...]
    local_crops: np.ndarray
    relative_offsets: np.ndarray
    player_crop: tuple[int, int, int, int]


def artifact_paths(root: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    """Return the direct immutable grounding bundle and manifest paths."""

    base = Path(root) if root is not None else Path(__file__).with_name("perception_artifacts")
    bundle = base.resolve() / GROUNDING_ARTIFACT_DIRNAME
    return bundle, bundle / GROUNDING_MANIFEST_FILENAME


def validate_grounding_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the compact grounding identity, geometry, and template hash."""

    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format_version") != GROUNDING_FORMAT_VERSION:
        raise GroundingError("Unsupported grounding manifest.")
    if value.get("artifact_id") != GROUNDING_ARTIFACT_DIRNAME or value.get("variant_id") != "deterministic-renderer-template-v2":
        raise GroundingError("Unexpected grounding artifact identity.")
    if value.get("environment") != ENVIRONMENT_CONTRACT:
        raise GroundingError("Grounding environment contract is incompatible.")
    if value.get("input") != {"observation_dtype": "uint8", "observation_shape": list(OBSERVATION_SHAPE)}:
        raise GroundingError("Grounding input contract is incompatible.")
    geometry = value.get("geometry")
    if not isinstance(geometry, dict) or geometry != {
        "cell_size": [7, 7], "inventory_grid": [9, 2],
        "inventory_origin": [49, 0], "local_grid": [9, 7],
        "local_origin": [0, 0], "player_crop": [21, 28, 7, 7],
    }:
        raise GroundingError("Grounding geometry is incompatible.")
    templates = value.get("templates")
    if not isinstance(templates, dict) or templates.get("filename") != GROUNDING_TEMPLATES_FILENAME:
        raise GroundingError("Grounding template record is invalid.")
    if file_sha256(manifest_path.with_name(GROUNDING_TEMPLATES_FILENAME)) != templates.get("sha256"):
        raise GroundingError("Grounding template hash mismatch.")
    return value


def resolve_active_grounding_bundle(
    root: str | os.PathLike[str] | None = None,
    **_: Any,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the direct selected bundle; no mutable pointer is consulted."""

    bundle, manifest_path = artifact_paths(root)
    return bundle, validate_grounding_manifest(manifest_path)


def _strings(array: np.ndarray, label: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind != "U":
        raise GroundingError(f"{label} must be a one-dimensional Unicode array.")
    return tuple(str(value) for value in array.tolist())


def load_grounding_templates(bundle_dir: str | os.PathLike[str]) -> GroundingTemplates:
    """Load the compact NPZ and derive all label cardinalities from it."""

    bundle = Path(bundle_dir).resolve()
    manifest = validate_grounding_manifest(bundle / GROUNDING_MANIFEST_FILENAME)
    try:
        with np.load(bundle / GROUNDING_TEMPLATES_FILENAME, allow_pickle=False) as source:
            if set(source.files) != _ARRAYS:
                raise GroundingError("Grounding NPZ has an incompatible schema.")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as exc:
        raise GroundingError(f"Could not load grounding templates: {exc}") from exc
    materials = _strings(arrays["cell_materials"], "cell_materials")
    occupants = _strings(arrays["cell_occupants"], "cell_occupants")
    facings = _strings(arrays["player_facings"], "player_facings")
    items = _strings(arrays["inventory_items"], "inventory_items")
    counts = tuple(int(value) for value in arrays["inventory_counts"].tolist())
    expected = {
        "cell_templates": (len(materials), 7, 7, 3),
        "player_templates": (len(facings), 7, 7, 3),
        "inventory_templates": (len(items), len(counts), 7, 7, 3),
        "local_crops": (62, 4), "relative_offsets": (62, 2),
    }
    if (
        len(materials) != len(occupants)
        or tuple(dict.fromkeys(facings)) != ("left", "right", "up", "down")
        or counts != tuple(range(10))
    ):
        raise GroundingError("Grounding labels are incompatible.")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise GroundingError(f"{name} has incompatible shape {arrays[name].shape}.")
    for name in ("cell_templates", "player_templates", "sleeping_templates", "inventory_templates"):
        if arrays[name].dtype != np.uint8:
            raise GroundingError(f"{name} must use uint8 storage.")
    if any(tuple(offset) == (0, 0) for offset in arrays["relative_offsets"].tolist()):
        raise GroundingError("Relative offsets cannot contain the player cell.")
    return GroundingTemplates(
        manifest["artifact_id"], file_sha256(bundle / GROUNDING_MANIFEST_FILENAME),
        arrays["cell_templates"], materials, occupants, arrays["player_templates"],
        facings, arrays["sleeping_templates"], arrays["inventory_templates"],
        items, counts, arrays["local_crops"], arrays["relative_offsets"],
        tuple(manifest["geometry"]["player_crop"]),
    )


def _crop(frame: np.ndarray, bounds: Any) -> np.ndarray:
    row, column, height, width = (int(value) for value in bounds)
    crop = frame[row : row + height, column : column + width]
    if row < 0 or column < 0 or crop.shape != (height, width, 3):
        raise GroundingError("Grounding crop escapes the frame.")
    return crop


def _one_match(candidates: np.ndarray, crop: np.ndarray, label: str) -> int:
    matches = np.flatnonzero(np.all(candidates == crop, axis=(1, 2, 3)))
    if len(matches) != 1:
        raise GroundingError(f"Expected one {label} template match, found {len(matches)}.")
    return int(matches[0])


def ground_observation(
    observation: Any,
    templates: GroundingTemplates,
    *,
    previous_facing: Direction | None = None,
) -> tuple[CrafterPercept, dict[str, Any]]:
    """Decode RGB, preserving previously observed facing under the sleeping sprite."""

    frame = as_rgb_frame(observation)
    player_crop = _crop(frame, templates.player_crop)
    sleeping = bool(np.any(np.all(templates.sleeping_templates == player_crop, axis=(1, 2, 3))))
    if sleeping:
        if previous_facing is None:
            raise GroundingError("Sleeping sprite requires previously grounded facing.")
        player_index, facing = None, previous_facing
    else:
        player_index = _one_match(templates.player_templates, player_crop, "player")
        facing = Direction.from_name(templates.player_facings[player_index])
    tiles: dict[tuple[int, int], ObservedTile] = {}
    cell_indices: list[int] = []
    for bounds, offset in zip(templates.local_crops, templates.relative_offsets, strict=True):
        index = _one_match(templates.cell_templates, _crop(frame, bounds), "cell")
        position = int(offset[0]), int(offset[1])
        tiles[position] = ObservedTile(templates.cell_materials[index], templates.cell_occupants[index])
        cell_indices.append(index)
    inventory: dict[str, int] = {}
    inventory_indices: list[int] = []
    for index, item in enumerate(templates.inventory_items):
        bounds = (49 + (index // 9) * 7, (index % 9) * 7, 7, 7)
        matched = _one_match(templates.inventory_templates[index], _crop(frame, bounds), item)
        inventory[item] = templates.inventory_counts[matched]
        inventory_indices.append(matched)
    percept = CrafterPercept(sleeping, facing, dict(sorted(inventory.items())), dict(sorted(tiles.items())))
    return percept, {
        "player_template_index": player_index,
        "cell_template_indices": cell_indices,
        "inventory_template_indices": inventory_indices,
    }


def percept_to_dict(percept: CrafterPercept) -> dict[str, Any]:
    """Serialize a percept for focused tests, not runtime state."""

    return {
        "sleeping": percept.sleeping, "facing": percept.facing.name.lower(),
        "inventory": dict(percept.inventory),
        "tiles": [
            {"relative_offset": list(position), "material": tile.material, "occupant": tile.occupant}
            for position, tile in sorted(percept.tiles.items())
        ],
    }


__all__ = [
    "ENVIRONMENT_CONTRACT", "GROUNDING_ARTIFACT_DIRNAME",
    "GROUNDING_MANIFEST_FILENAME", "GROUNDING_TEMPLATES_FILENAME",
    "GroundingError", "GroundingTemplates", "artifact_paths", "ground_observation",
    "load_grounding_templates", "percept_to_dict", "resolve_active_grounding_bundle",
    "validate_grounding_manifest",
]
