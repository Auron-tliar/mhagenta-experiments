"""Frozen inference and compact artifact contract for experiment 2-5-BW."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import TransferSpec, location_index
from .grounding import NUM_BLOCKS, OBSERVATION_SHAPE, TABLE_LEN, as_numeric_observation


MODEL_INPUT_SHAPE = (NUM_BLOCKS + 4, TABLE_LEN, NUM_BLOCKS)
N_ACTIONS = 4
MAX_POLICY_STEPS = 32
CHECKPOINT_FORMAT_VERSION = 3
MANIFEST_FORMAT_VERSION = 4
POLICY_ARCHITECTURE = "transfer_conditioned_structured_mlp_v2"
POLICY_FILENAME = "transfer-policy.pt"
MANIFEST_FILENAME = "manifest.json"
ARTIFACT_SET_DIRNAME = "structured-dqfd-lite-v2"
QUALIFICATION_PROTOCOL_ID = "exp2-5-bw-certification-v1"
QUALIFICATION_PROTOCOL_SHA256 = "4e5ada2d56477fa55367b747610cdc738719b67561c7e56d9f7acadd65354bdd"
QUALIFICATION_COHORTS = {
    "held_out": 100,
    "integration": 1,
    "sequential": 20,
    "regressions": 1,
}
SEQUENTIAL_TRANSFERS = 10


def artifact_paths() -> tuple[Path, Path]:
    """Return the packaged compact manifest and frozen checkpoint paths."""

    artifact_dir = Path(__file__).resolve().with_name("artifacts") / ARTIFACT_SET_DIRNAME
    return artifact_dir / MANIFEST_FILENAME, artifact_dir / POLICY_FILENAME


def _block_index(block: str, num_blocks: int = NUM_BLOCKS) -> int:
    normalized = str(block).lower()
    if not normalized.startswith("b") or not normalized[1:].isdigit():
        raise ValueError(f"Invalid block name {block!r}.")
    index = int(normalized[1:])
    if index not in range(num_blocks):
        raise ValueError(f"Block index {index} is outside the configured environment.")
    return index


def condition_observation(
    observation: Any, spec: TransferSpec, *,
    table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> np.ndarray:
    """Add transfer-block and destination channels to a numeric observation."""

    destination = location_index(spec.destination)
    if destination not in range(table_len):
        raise ValueError("Destination is outside the configured environment.")
    conditioned = np.zeros((num_blocks + 4, table_len, num_blocks), dtype=np.uint8)
    conditioned[: num_blocks + 2] = as_numeric_observation(
        observation, table_len=table_len, num_blocks=num_blocks,
    )
    conditioned[num_blocks + 2, :, _block_index(spec.block, num_blocks)] = 1
    conditioned[num_blocks + 3, destination, :] = 1
    return conditioned


def build_q_network(
    torch_module: Any, *, table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> Any:
    """Construct the versioned block-entity Q network used by the checkpoint."""

    nn = torch_module.nn

    class StructuredQNetwork(nn.Module):
        """Encode blocks with shared weights and pool across block identities."""

        def __init__(self) -> None:
            super().__init__()
            entity_features = num_blocks * table_len + table_len + 1 + 2 * table_len
            self.entity_encoder = nn.Sequential(
                nn.Linear(entity_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
            )
            self.value_head = nn.Sequential(
                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, N_ACTIONS)
            )

        def forward(self, inputs: Any) -> Any:
            batch_size = inputs.shape[0]
            stacks = inputs[:, 2 : num_blocks + 2].permute(0, 3, 1, 2)
            stacks = stacks.reshape(batch_size, num_blocks, -1)
            holding = inputs[:, 1].permute(0, 2, 1)
            target = inputs[:, num_blocks + 2].permute(0, 2, 1).mean(2, keepdim=True)
            arm = inputs[:, 0].mean(2)
            destination = inputs[:, num_blocks + 3].mean(2)
            global_features = torch_module.cat((arm, destination), dim=1)
            global_features = global_features.unsqueeze(1).expand(-1, num_blocks, -1)
            entities = torch_module.cat((stacks, holding, target, global_features), dim=2)
            encoded = self.entity_encoder(entities)
            pooled = torch_module.cat((encoded.mean(1), encoded.max(1).values), dim=1)
            return self.value_head(pooled)

    return StructuredQNetwork()


def legal_action_indices(
    observation: Any, *, table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> list[int]:
    """Return environment actions that are legal in a numeric observation."""

    numeric = as_numeric_observation(observation, table_len=table_len, num_blocks=num_blocks)
    arm_columns = np.flatnonzero(np.any(numeric[0] == 1, axis=1))
    if arm_columns.size != 1:
        raise ValueError("Numeric observation must encode exactly one arm column.")
    arm_column = int(arm_columns[0])
    holding = bool(np.any(numeric[1]))
    actions: list[int] = []
    if holding:
        actions.append(1)
    elif np.any(numeric[2:, arm_column]):
        actions.append(0)
    if arm_column > 0:
        actions.append(2)
    if arm_column < table_len - 1:
        actions.append(3)
    if not actions:
        raise RuntimeError("Numeric observation has no legal environment action.")
    return actions


def greedy_inference(
    torch_module: Any,
    model: Any,
    observation: Any,
    spec: TransferSpec,
    *, table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> tuple[int, list[float], np.ndarray]:
    """Run one legal-masked greedy inference under ``no_grad``."""

    model_input = condition_observation(observation, spec, table_len=table_len, num_blocks=num_blocks)
    device = next(model.parameters()).device
    tensor = torch_module.as_tensor(
        model_input, dtype=torch_module.float32, device=device
    ).unsqueeze(0)
    with torch_module.no_grad():
        values = model(tensor).squeeze(0).detach().cpu()
    q_values = [float(value) for value in values.tolist()]
    if not all(math.isfinite(value) for value in q_values):
        raise FloatingPointError("Non-finite Transfer Q values.")
    legal_actions = legal_action_indices(observation, table_len=table_len, num_blocks=num_blocks)
    action = max(legal_actions, key=lambda index: q_values[index])
    return action, q_values, model_input


def _validate_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Policy checkpoint must contain a dictionary.")
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": POLICY_ARCHITECTURE,
        "observation_shape": OBSERVATION_SHAPE,
        "model_input_shape": MODEL_INPUT_SHAPE,
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        "n_actions": N_ACTIONS,
    }
    for field, expected_value in expected.items():
        value = checkpoint.get(field)
        if field.endswith("_shape") and value is not None:
            value = tuple(value)
        if value != expected_value:
            raise ValueError(f"Incompatible checkpoint field {field!r}.")
    if not isinstance(checkpoint.get("weight_optimizer_steps"), int):
        raise ValueError("Checkpoint weight_optimizer_steps must be an integer.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint does not contain a model_state_dict.")
    return checkpoint


def load_policy_checkpoint(
    torch_module: Any,
    path: str | os.PathLike[str],
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load and freeze the existing weights-only transfer-policy checkpoint."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch_module.load(checkpoint_path, map_location=device, weights_only=True)
    validated = _validate_checkpoint(checkpoint)
    model = build_q_network(torch_module)
    model.load_state_dict(validated["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, validated


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Return a stable digest for one conditioned numeric model input."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _require_mapping(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {name!r} must be an object.")
    return value


def validate_manifest(
    manifest_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str] | None = None,
    *,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Validate the compact schema, fixed protocol, all-success rule, and checksum."""

    resolved_manifest = Path(manifest_path).resolve()
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"Policy manifest not found: {resolved_manifest}")
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Policy manifest must contain a JSON object.")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("Unsupported policy manifest version.")
    if manifest.get("architecture") != POLICY_ARCHITECTURE or manifest.get("frozen") is not True:
        raise ValueError("Manifest architecture or frozen status is incompatible.")

    checkpoint = _require_mapping(manifest, "checkpoint")
    expected_checkpoint = {
        "filename": POLICY_FILENAME,
        "observation_shape": list(OBSERVATION_SHAPE),
        "input_shape": list(MODEL_INPUT_SHAPE),
        "actions": N_ACTIONS,
    }
    if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
        raise ValueError("Manifest checkpoint contract is incompatible.")
    checksum = checkpoint.get("sha256")
    resolved_checkpoint = (
        Path(checkpoint_path).resolve()
        if checkpoint_path is not None
        else resolved_manifest.with_name(POLICY_FILENAME)
    )
    if not isinstance(checksum, str) or file_sha256(resolved_checkpoint) != checksum:
        raise ValueError("Policy checkpoint checksum does not match the manifest.")

    training = _require_mapping(manifest, "training")
    integer_fields = (
        "seed",
        "search_environment_steps",
        "search_optimizer_steps",
        "selected_environment_step",
        "selected_optimizer_steps",
    )
    if training.get("method") != "structured-ddqn-per-n3-dqfd-lite":
        raise ValueError("Manifest training method is incompatible.")
    if any(isinstance(training.get(field), bool) or not isinstance(training.get(field), int) for field in integer_fields):
        raise ValueError("Manifest training step fields must be integers.")
    elapsed = training.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Manifest training elapsed_seconds must be positive and finite.")
    if training["selected_environment_step"] > training["search_environment_steps"] or training["selected_optimizer_steps"] > training["search_optimizer_steps"]:
        raise ValueError("Selected checkpoint exceeds the completed search budget.")
    if not isinstance(training.get("hyperparameters"), dict) or not training["hyperparameters"]:
        raise ValueError("Manifest training hyperparameters must be a non-empty object.")

    qualification = _require_mapping(manifest, "qualification")
    protocol = _require_mapping(qualification, "protocol")
    if protocol != {"id": QUALIFICATION_PROTOCOL_ID, "sha256": QUALIFICATION_PROTOCOL_SHA256}:
        raise ValueError("Manifest qualification protocol is incompatible.")
    if qualification.get("transfers_per_sequence") != SEQUENTIAL_TRANSFERS:
        raise ValueError("Manifest sequential qualification length is incompatible.")
    for cohort, cases in QUALIFICATION_COHORTS.items():
        result = qualification.get(cohort)
        if not isinstance(result, dict) or result.get("cases") != cases:
            raise ValueError(f"Manifest qualification cohort {cohort!r} has the wrong size.")
        successes = result.get("successes")
        if not isinstance(successes, int) or isinstance(successes, bool):
            raise ValueError(f"Manifest qualification cohort {cohort!r} has an invalid result.")
        if require_ready and successes != cases:
            raise ValueError(f"Manifest qualification cohort {cohort!r} did not fully succeed.")

    normalized = dict(manifest)
    normalized["checkpoint_sha256"] = checksum
    return normalized
