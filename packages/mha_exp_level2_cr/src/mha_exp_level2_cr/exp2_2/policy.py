"""Four-frame DQN policy and checkpoint contract for experiment 2-2-CR."""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .treatment import (
    DEFAULT_WORKLOAD, DQNWorkload, ILLEGAL_ACTION_PENALTY, PROTOCOL_VERSION,
    STEP_REWARD, TARGET_ACHIEVEMENT_REWARD, algorithm_metadata, reward_metadata,
)


FRAME_SHAPE = (64, 64, 3)
FRAME_STACK_SIZE = 4
STACK_SHAPE = (FRAME_STACK_SIZE, *FRAME_SHAPE)
MODEL_IMAGE_SHAPE = (FRAME_STACK_SIZE * FRAME_SHAPE[2], *FRAME_SHAPE[:2])
N_ACTIONS = 17


class Achievement(str, Enum):
    """Crafter achievements used to validate the environment payload."""
    COLLECT_COAL = "collect_coal"
    COLLECT_DIAMOND = "collect_diamond"
    COLLECT_DRINK = "collect_drink"
    COLLECT_IRON = "collect_iron"
    COLLECT_SAPLING = "collect_sapling"
    COLLECT_STONE = "collect_stone"
    COLLECT_WOOD = "collect_wood"
    DEFEAT_SKELETON = "defeat_skeleton"
    DEFEAT_ZOMBIE = "defeat_zombie"
    EAT_COW = "eat_cow"
    EAT_PLANT = "eat_plant"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    MAKE_IRON_SWORD = "make_iron_sword"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    MAKE_STONE_SWORD = "make_stone_sword"
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    MAKE_WOOD_SWORD = "make_wood_sword"
    PLACE_FURNACE = "place_furnace"
    PLACE_PLANT = "place_plant"
    PLACE_STONE = "place_stone"
    PLACE_TABLE = "place_table"
    WAKE_UP = "wake_up"


ACHIEVEMENTS = tuple(item.value for item in Achievement)
TARGET_ACHIEVEMENT = Achievement.COLLECT_DIAMOND

CHECKPOINT_FORMAT_VERSION = 4
POLICY_ARCHITECTURE = "crafter_four_frame_dueling_dqn_v3"
POLICY_FILENAME = "dqn_policy.pt"


def as_rgb_frame(content: Any) -> np.ndarray:
    """Validate and copy one 64x64 RGB frame as uint8."""
    frame = np.asarray(content)
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"Expected frame shape {FRAME_SHAPE}, received {frame.shape}.")
    return frame.astype(np.uint8, copy=True)


def initial_frame_stack(frame: Any) -> np.ndarray:
    """Initialize temporal context by repeating the first episode frame."""
    return np.repeat(as_rgb_frame(frame)[None, ...], FRAME_STACK_SIZE, axis=0)


def shift_frame_stack(stack: Any, next_frame: Any) -> np.ndarray:
    """Drop the oldest frame and append one validated next frame."""
    validated = np.asarray(stack)
    if validated.shape != STACK_SHAPE:
        raise ValueError(f"Expected stack shape {STACK_SHAPE}, received {validated.shape}.")
    shifted = np.empty(STACK_SHAPE, dtype=np.uint8)
    shifted[:-1] = validated[1:].astype(np.uint8, copy=False)
    shifted[-1] = as_rgb_frame(next_frame)
    return shifted


def stack_batch(stacks: Sequence[Any]) -> np.ndarray:
    """Convert canonical frame stacks to a channel-first model batch."""
    validated: list[np.ndarray] = []
    for stack in stacks:
        value = np.asarray(stack)
        if value.shape != STACK_SHAPE:
            raise ValueError(f"Expected stack shape {STACK_SHAPE}, received {value.shape}.")
        validated.append(value.astype(np.uint8, copy=True))
    batch = np.stack(validated)
    return np.transpose(batch, (0, 1, 4, 2, 3)).reshape(len(validated), *MODEL_IMAGE_SHAPE)


def goal_achieved(achievements: Any) -> bool:
    """Return whether the experiment's fixed achievement is present."""
    if not isinstance(achievements, dict):
        raise TypeError("Crafter achievements must be a dictionary.")
    value = achievements.get(TARGET_ACHIEVEMENT.value)
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"Missing integer count for {TARGET_ACHIEVEMENT.value!r}.")
    return int(value) > 0


def build_q_network(torch_module: Any) -> Any:
    """Build the four-frame CNN with mean-centered dueling value heads."""
    nn = torch_module.nn

    class DuelingQNetwork(nn.Module):
        """Lazy Torch definition, reconstructed from a state dict for playback."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(MODEL_IMAGE_SHAPE[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(), nn.Flatten(),
            )
            self.value = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
            self.advantage = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, N_ACTIONS))

        def forward(self, images: Any) -> Any:
            """Return one Q-value per native action."""
            features = self.encoder(images)
            advantage = self.advantage(features)
            return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)

    return DuelingQNetwork()


def policy_checkpoint(model: Any, training_steps: int, workload: DQNWorkload = DEFAULT_WORKLOAD) -> dict[str, Any]:
    """Build the durable fixed-treatment checkpoint payload."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": POLICY_ARCHITECTURE,
        "frame_shape": FRAME_SHAPE,
        "frame_stack_size": FRAME_STACK_SIZE,
        "stack_shape": STACK_SHAPE,
        "model_image_shape": MODEL_IMAGE_SHAPE,
        "n_actions": N_ACTIONS,
        "target_achievement": TARGET_ACHIEVEMENT.value,
        "protocol_version": PROTOCOL_VERSION,
        "reward": reward_metadata(),
        "environment": {"no_mobs": True},
        "algorithm": algorithm_metadata(workload.synchronized_training),
        "workload": workload.dump(),
        "training_steps": int(training_steps),
        "model_state_dict": {name: tensor.detach().cpu()
                             for name, tensor in model.state_dict().items()},
    }


def save_policy_checkpoint(
    torch_module: Any, model: Any, path: str | os.PathLike[str], training_steps: int,
    workload: DQNWorkload = DEFAULT_WORKLOAD,
) -> Path:
    """Atomically save a policy checkpoint."""
    checkpoint_path = Path(path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    torch_module.save(policy_checkpoint(model, training_steps, workload), temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def _validate_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Policy checkpoint must contain a dictionary.")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Incompatible legacy checkpoint; 2-2-CR requires mob-free Rainbow diamond format 4.")
    try:
        workload = DQNWorkload(**checkpoint["workload"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint workload is invalid.") from exc
    expected = {
        "architecture": POLICY_ARCHITECTURE,
        "frame_shape": FRAME_SHAPE,
        "frame_stack_size": FRAME_STACK_SIZE,
        "stack_shape": STACK_SHAPE,
        "model_image_shape": MODEL_IMAGE_SHAPE,
        "n_actions": N_ACTIONS,
        "target_achievement": TARGET_ACHIEVEMENT.value,
        "protocol_version": PROTOCOL_VERSION,
        "reward": reward_metadata(),
        "environment": {"no_mobs": True},
        "algorithm": algorithm_metadata(workload.synchronized_training),
    }
    for field, expected_value in expected.items():
        value = checkpoint.get(field)
        if field.endswith("_shape") and value is not None:
            value = tuple(value)
        if value != expected_value:
            raise ValueError(f"Incompatible {field}: expected {expected_value!r}, got {value!r}.")
    if (type(checkpoint.get("training_steps")) is not int or checkpoint["training_steps"] < 0
            or (workload.synchronized_training and checkpoint["training_steps"] > workload.training_updates)):
        raise ValueError("Checkpoint training_steps is outside the workload.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint does not contain a model_state_dict.")
    return checkpoint


def load_policy_checkpoint(
    torch_module: Any, path: str | os.PathLike[str],
) -> tuple[Any, dict[str, Any]]:
    """Load and validate a four-frame policy checkpoint."""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=True)
    validated = _validate_checkpoint(checkpoint)
    model = build_q_network(torch_module)
    model.load_state_dict(validated["model_state_dict"], strict=True)
    model.cpu()
    model.eval()
    return model, validated
