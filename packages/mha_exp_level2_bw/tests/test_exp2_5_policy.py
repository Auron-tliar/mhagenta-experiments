"""Focused grounding, inference, and artifact tests for experiment 2-5-BW."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import TransferSpec
from mha_exp_level2_bw.exp2_5.evaluation import evaluate_transfer
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_5.policy import (
    MODEL_INPUT_SHAPE,
    N_ACTIONS,
    QUALIFICATION_PROTOCOL_ID,
    QUALIFICATION_PROTOCOL_SHA256,
    artifact_paths,
    condition_observation,
    greedy_inference,
    legal_action_indices,
    load_policy_checkpoint,
    validate_manifest,
)


def _observation(seed: int = 1000) -> np.ndarray:
    environment = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    try:
        observation, _ = environment.reset(seed=seed)
        return np.asarray(observation, dtype=np.uint8)
    finally:
        environment.close()


def test_numeric_grounding_and_transfer_conditioning_are_stable() -> None:
    observation = _observation()
    grounded = ground_observation(observation)
    transfers = enumerate_transfer_targets(grounded.facts)
    assert transfers
    conditioned = condition_observation(observation, transfers[0])
    assert conditioned.shape == MODEL_INPUT_SHAPE
    assert set(np.unique(conditioned)) <= {0, 1}
    assert legal_action_indices(observation)


def test_frozen_checkpoint_runs_legal_masked_no_grad_inference() -> None:
    torch = pytest.importorskip("torch")
    manifest_path, checkpoint_path = artifact_paths()
    manifest = validate_manifest(manifest_path, checkpoint_path)
    model, checkpoint = load_policy_checkpoint(torch, checkpoint_path)
    observation = _observation()
    spec = enumerate_transfer_targets(ground_observation(observation).facts)[0]
    action, q_values, conditioned = greedy_inference(torch, model, observation, spec)
    assert action in legal_action_indices(observation)
    assert len(q_values) == N_ACTIONS
    assert conditioned.shape == MODEL_INPUT_SHAPE
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert checkpoint["weight_optimizer_steps"] == manifest["training"]["selected_optimizer_steps"]


def test_shared_greedy_rollout_preserves_the_selected_2_5_path() -> None:
    torch = pytest.importorskip("torch")
    environment = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    environment.expose_snapshot = True
    try:
        observation, _ = environment.reset(seed=1000)
        grounded = ground_observation(observation)
        spec = enumerate_transfer_targets(grounded.facts)[0]
        model, _ = load_policy_checkpoint(torch, artifact_paths()[1])
        result, _ = evaluate_transfer(torch, model, environment, grounded.observation, spec, 1000)
    finally:
        environment.close()
    assert result.success and result.outcome == "succeeded"
    assert result.actions == [3, 0, 3, 1]


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_compact_manifest_binds_exact_protocol_and_all_success(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = artifact_paths()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_manifest(manifest_path, checkpoint_path)
    assert validated["qualification"]["protocol"] == {
        "id": QUALIFICATION_PROTOCOL_ID,
        "sha256": QUALIFICATION_PROTOCOL_SHA256,
    }
    assert validated["training"]["elapsed_seconds"] > 0

    for cohort in ("held_out", "integration", "sequential", "regressions"):
        failed = deepcopy(manifest)
        failed["qualification"][cohort]["successes"] -= 1
        with pytest.raises(ValueError, match="did not fully succeed"):
            validate_manifest(_write_manifest(tmp_path, failed), checkpoint_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["qualification"]["protocol"].update(id="changed"), "protocol"),
        (lambda data: data["qualification"]["protocol"].update(sha256="0" * 64), "protocol"),
        (lambda data: data["qualification"]["held_out"].update(cases=99), "wrong size"),
        (lambda data: data["qualification"].update(transfers_per_sequence=9), "length"),
        (lambda data: data["training"].update(elapsed_seconds=0), "elapsed_seconds"),
    ],
)
def test_compact_manifest_rejects_contract_mutations(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    manifest_path, checkpoint_path = artifact_paths()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    with pytest.raises(ValueError, match=message):
        validate_manifest(_write_manifest(tmp_path, manifest), checkpoint_path)


def test_transfer_contract_remains_structurally_compatible() -> None:
    spec = TransferSpec("b0", "t0", "b1", "t0", "t1")
    assert spec.as_dict() == {
        "block": "b0",
        "source_support": "t0",
        "destination_support": "b1",
        "source": "t0",
        "destination": "t1",
    }
