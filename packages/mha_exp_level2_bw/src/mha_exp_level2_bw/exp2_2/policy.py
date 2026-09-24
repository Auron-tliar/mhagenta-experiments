from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np
from numpy import random


TABLE_LEN = 10
NUM_BLOCKS = 30
OBS_SHAPE = (NUM_BLOCKS + 2, TABLE_LEN, NUM_BLOCKS)
MODEL_INPUT_SHAPE = (NUM_BLOCKS + 4, TABLE_LEN, NUM_BLOCKS)
N_ACTIONS = 4


def legal_action_mask(observation: np.ndarray) -> np.ndarray:
    """Derive pick-up, put-down, left and right applicability from observed state."""
    observation = as_numeric_observation(observation)
    columns = np.flatnonzero(observation[0].any(axis=1))
    if len(columns) != 1:
        raise ValueError('Expected exactly one observed arm position')
    column = int(columns[0])
    holding = bool(observation[1].any())
    occupied = bool(observation[2:, column, :].any())
    return np.array([not holding and occupied, holding, column > 0, column < TABLE_LEN - 1])

CHECKPOINT_FORMAT_VERSION = 1
POLICY_ARCHITECTURE = "goal_conditioned_cnn_v1"
POLICY_ARCHITECTURES = (POLICY_ARCHITECTURE, "goal_spatial_cnn_v1", "goal_volume_cnn3d_v1")
POLICY_FILENAME = "dqn_policy.pt"
MODEL_FINGERPRINT_VERSION = b"mha-exp-2-2-bw-model-state-v1\0"


def as_numeric_observation(content: Any) -> np.ndarray:
    observation = np.asarray(content)
    if observation.shape != OBS_SHAPE:
        raise ValueError(f"Expected observation shape {OBS_SHAPE}, received {observation.shape}.")
    return observation.astype(np.uint8, copy=True)


def block_position(observation: np.ndarray, block: int) -> tuple[int, int] | None:
    locations = np.argwhere(observation[2:, :, block] != 0)
    if len(locations) != 1:
        return None
    row, column = locations[0]
    return int(row), int(column)


def goal_achieved(observation: np.ndarray, goal: tuple[int, int]) -> bool:
    top_position = block_position(observation, goal[0])
    bottom_position = block_position(observation, goal[1])
    if top_position is None or bottom_position is None:
        return False
    top_row, top_column = top_position
    bottom_row, bottom_column = bottom_position
    return top_column == bottom_column and top_row + 1 == bottom_row


def goal_conditioned_observation(
    observation: np.ndarray,
    goal: tuple[int, int],
) -> np.ndarray:
    conditioned: np.ndarray = np.zeros(MODEL_INPUT_SHAPE, dtype=np.uint8)
    conditioned[: OBS_SHAPE[0]] = as_numeric_observation(observation)
    conditioned[OBS_SHAPE[0], :, goal[0]] = 1
    conditioned[OBS_SHAPE[0] + 1, :, goal[1]] = 1
    return conditioned


def sample_goal(
    rng: random.Generator,
    observation: np.ndarray,
) -> tuple[int, int]:
    while True:
        top, bottom = rng.choice(NUM_BLOCKS, size=2, replace=False).tolist()
        goal = int(top), int(bottom)
        if not goal_achieved(observation, goal):
            return goal


def build_q_network(torch_module: Any, architecture: str = POLICY_ARCHITECTURE) -> Any:
    """Build a named model accepting the common goal-conditioned replay input."""
    if architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"Unknown BW policy architecture: {architecture}")
    if architecture != POLICY_ARCHITECTURE:
        from .networks import spatial_network, volume_network
        model = spatial_network() if architecture == "goal_spatial_cnn_v1" else volume_network()
        model.policy_architecture = architecture
        return model
    nn = torch_module.nn
    return nn.Sequential(
        nn.Conv2d(MODEL_INPUT_SHAPE[0], 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 3 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, N_ACTIONS),
    )


def model_state_fingerprint(model_or_state_dict: Any) -> str:
    """Return a canonical SHA-256 identity for a model's tensor state.

    The digest is independent of checkpoint-container serialization and binds
    parameter names, dtypes, shapes, and exact contiguous CPU bytes.
    """

    state_dict = (
        model_or_state_dict.state_dict()
        if hasattr(model_or_state_dict, "state_dict")
        else model_or_state_dict
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Expected a model or state-dict mapping.")

    digest = hashlib.sha256(MODEL_FINGERPRINT_VERSION)

    def add_bytes(value: bytes) -> None:
        digest.update(struct.pack("!Q", len(value)))
        digest.update(value)

    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not hasattr(tensor, "detach"):
            raise TypeError(f"State entry {name!r} is not a tensor.")
        canonical = tensor.detach().cpu().contiguous()
        add_bytes(str(name).encode("utf-8"))
        add_bytes(str(canonical.dtype).encode("utf-8"))
        digest.update(struct.pack("!Q", len(canonical.shape)))
        for dimension in canonical.shape:
            digest.update(struct.pack("!q", int(dimension)))
        add_bytes(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def policy_checkpoint(
    model: Any, training_steps: int, *, training_protocol: dict[str, Any] | None = None,
    frozen: bool = False,
) -> dict[str, Any]:
    """Package weights with optional treatment provenance, preserving legacy playback."""
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": getattr(model, "policy_architecture", POLICY_ARCHITECTURE),
        "observation_shape": OBS_SHAPE,
        "model_input_shape": MODEL_INPUT_SHAPE,
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        "n_actions": N_ACTIONS,
        "training_steps": int(training_steps),
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
    }
    if training_protocol is not None:
        checkpoint.update(training_protocol=training_protocol, frozen_for_evaluation=frozen)
    return checkpoint


def save_policy_checkpoint(
    torch_module: Any,
    model: Any,
    path: str | os.PathLike[str],
    training_steps: int,
    *,
    training_protocol: dict[str, Any] | None = None,
    frozen: bool = False,
) -> Path:
    """Atomically save a policy and, when supplied, its training treatment."""
    checkpoint_path = Path(path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    torch_module.save(
        policy_checkpoint(model, training_steps, training_protocol=training_protocol, frozen=frozen),
        temporary_path,
    )
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def _validate_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Policy checkpoint must contain a dictionary.")
    if checkpoint.get("architecture") not in POLICY_ARCHITECTURES:
        raise ValueError("Incompatible checkpoint architecture")
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_shape": OBS_SHAPE,
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
            raise ValueError(
                f"Incompatible checkpoint field {field!r}: "
                f"expected {expected_value!r}, received {value!r}."
            )
    if not isinstance(checkpoint.get("training_steps"), int):
        raise ValueError("Checkpoint training_steps must be an integer.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint does not contain a model_state_dict.")
    return checkpoint


def load_policy_checkpoint(
    torch_module: Any,
    path: str | os.PathLike[str],
) -> tuple[Any, dict[str, Any]]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch_module.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validated = _validate_checkpoint(checkpoint)
    model = build_q_network(torch_module, validated["architecture"])
    model.load_state_dict(validated["model_state_dict"], strict=True)
    model.cpu()
    model.eval()
    return model, validated
