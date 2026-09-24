"""Focused deterministic-grounding tests for experiment 2-5-CR."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import runpy

from mha_env_crafter import CrafterEnv
from mha_exp_level2_cr.exp2_5.grounding import (
    GroundingError, artifact_paths, ground_observation, load_grounding_templates,
    validate_grounding_manifest,
)


def _environment(seed: int = 820_001) -> CrafterEnv:
    env = CrafterEnv(
        area=(64, 64), view=(9, 9), size=(64, 64), length=1000, seed=seed,
        no_mobs=True, symbolic=False, daylight_effects=False, sleep_effects=False,
    )
    env.reset()
    return env


def test_compact_grounding_manifest_and_npz_are_self_consistent() -> None:
    bundle, manifest_path = artifact_paths()
    manifest = validate_grounding_manifest(manifest_path)
    templates = load_grounding_templates(bundle)
    assert manifest["artifact_id"] == templates.artifact_id == "current"
    assert templates.cell_templates.shape == (16, 7, 7, 3)
    assert templates.local_crops.shape == (62, 4)
    assert templates.inventory_counts == tuple(range(10))


def test_grounding_recovers_public_player_and_inventory_state() -> None:
    bundle, _ = artifact_paths()
    templates = load_grounding_templates(bundle)
    env = _environment()
    try:
        percept, matches = ground_observation(env.render(), templates)
        facing = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}[
            tuple(env._player.facing)
        ]
        assert percept.facing.name.lower() == facing
        assert dict(percept.inventory) == {key: int(value) for key, value in env._player.inventory.items()}
        assert len(percept.tiles) == 62
        assert len(matches["cell_template_indices"]) == 62
        assert len(matches["inventory_template_indices"]) == 16
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def test_grounding_rejects_a_non_renderer_frame() -> None:
    bundle, _ = artifact_paths()
    templates = load_grounding_templates(bundle)
    with pytest.raises(GroundingError):
        ground_observation(np.zeros((64, 64, 3), dtype=np.uint8), templates)


def test_grounding_evidence_is_indices_not_duplicated_semantics() -> None:
    bundle, _ = artifact_paths()
    templates = load_grounding_templates(bundle)
    env = _environment(820_002)
    try:
        _, matches = ground_observation(env.render(), templates)
        assert set(matches) == {
            "player_template_index", "cell_template_indices", "inventory_template_indices"
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def test_offline_generator_recreates_the_frozen_template_arrays() -> None:
    tool_path = Path(__file__).parents[3] / "tools" / "exp2_5_cr_preparation" / "prepare_grounding.py"
    generated = runpy.run_path(str(tool_path))["renderer_templates"]()
    bundle, _ = artifact_paths()
    with np.load(bundle / "templates.npz", allow_pickle=False) as frozen:
        assert set(generated) == set(frozen.files)
        assert all(np.array_equal(generated[name], frozen[name]) for name in frozen.files)
