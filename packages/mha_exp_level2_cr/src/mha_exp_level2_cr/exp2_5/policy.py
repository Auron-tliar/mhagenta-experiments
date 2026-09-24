"""Frozen neural-policy contracts for experiment 2-5-CR."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

OBSERVATION_SHAPE = (64, 64, 3)
MODEL_IMAGE_SHAPE = (3, 64, 64)
CONTEXT_SIZE = 2
MAX_RELATIVE_TARGET_DELTA = 63
POLICY_ACTIONS = (0, 1, 2, 3, 4, 5)
DIRECTION_ACTIONS = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
N_ACTIONS = len(POLICY_ACTIONS)
CHECKPOINT_FORMAT_VERSION = 4
MANIFEST_FORMAT_VERSION = 5
POLICY_ARCHITECTURE = "crafter_activity_cnn_v1"
CONTEXT_ENCODING = "relative_xy_divided_by_63_persistent_goal_v3"
EAT_COW_WAYPOINT_ENCODING = "relative_xy_divided_by_63_visible_cow_or_discovery_v1"
METHOD_NAME = "DQfD-lite"
ARTIFACT_DIRNAME = "current"
VARIANT_ID = "goal-conditioned-five-policy-v3"
MANIFEST_FILENAME = "manifest.json"

EXPLORE_POLICY_FILENAME = "explore-policy.pt"
NAVIGATE_TO_POLICY_FILENAME = "navigate-to-policy.pt"
GET_RESOURCE_POLICY_FILENAME = "get-resource-policy.pt"
EAT_TARGET_POLICY_FILENAME = "eat-target-policy.pt"
EAT_COW_POLICY_FILENAME = "eat-cow-policy.pt"

ENVIRONMENT_CONTRACT = {
    "area": [64, 64],
    "daylight_effects": False,
    "episode_length": 1000,
    "no_mobs": True,
    "package": "mha-env-crafter",
    "render_size": [64, 64],
    "sleep_effects": False,
    "symbolic": False,
    "version": "0.1.0",
    "view": [9, 9],
}


class PolicyId(str, Enum):
    """The five independently trained v3 policies."""

    EXPLORE = "explore"
    NAVIGATE_TO = "navigate_to"
    GET_RESOURCE = "get_resource"
    EAT_TARGET = "eat_target"
    EAT_COW = "eat_cow"


POLICY_FILENAMES: Mapping[PolicyId, str] = {
    PolicyId.EXPLORE: EXPLORE_POLICY_FILENAME,
    PolicyId.NAVIGATE_TO: NAVIGATE_TO_POLICY_FILENAME,
    PolicyId.GET_RESOURCE: GET_RESOURCE_POLICY_FILENAME,
    PolicyId.EAT_TARGET: EAT_TARGET_POLICY_FILENAME,
    PolicyId.EAT_COW: EAT_COW_POLICY_FILENAME,
}
POLICY_MASKS: Mapping[PolicyId, tuple[int, ...]] = {
    PolicyId.EXPLORE: (1, 2, 3, 4),
    PolicyId.NAVIGATE_TO: (1, 2, 3, 4),
    PolicyId.GET_RESOURCE: (1, 2, 3, 4, 5),
    PolicyId.EAT_TARGET: (1, 2, 3, 4, 5),
    PolicyId.EAT_COW: (1, 2, 3, 4, 5),
}
HUD_MASKED_POLICIES = frozenset({PolicyId.NAVIGATE_TO, PolicyId.GET_RESOURCE})
INPUT_CONTRACT = {
    "context_encoding": CONTEXT_ENCODING,
    "context_size": CONTEXT_SIZE,
    "image_preprocessing": "uint8_nhwc_to_float32_nchw_divided_by_255_v1",
    "max_relative_target_delta": MAX_RELATIVE_TARGET_DELTA,
    "model_image_shape": list(MODEL_IMAGE_SHAPE),
    "observation_dtype": "uint8",
    "observation_shape": list(OBSERVATION_SHAPE),
    "policy_actions": list(POLICY_ACTIONS),
    "policy_masks": {policy.value: list(actions) for policy, actions in POLICY_MASKS.items()},
}
REWARD_CONTRACT = {
    "goal": 1.0,
    "nonterminal_step": -0.01,
    "eat_target_wrong_cow": 0.3,
    "wrong_cow_terminal": False,
    "native_reward_used": False,
}


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-native value using the preparation canonical form."""

    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


REWARD_CONTRACT_SHA256 = canonical_json_sha256(REWARD_CONTRACT)
TRAINING_CONFIG: dict[str, Any] = {
    "master_seed": 2508,
    "nominal_demonstration_steps": 2_000,
    "maximum_demo_episode_overshoot": 31,
    "nominal_maximum_online_steps": 20_000,
    "maximum_committed_online_steps": 20_031,
    "selection_boundaries": [5_000, 10_000, 15_000, 20_000],
    "replay_capacity": 50_000,
    "batch_size": 128,
    "discount": 0.99,
    "n_step_horizon": 3,
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "gradient_clip": 10.0,
    "target_sync_optimizer_steps": 500,
    "large_margin": 0.8,
    "priority_alpha": 0.6,
    "priority_beta_start": 0.4,
    "priority_beta_end": 1.0,
    "priority_beta_steps": 20_000,
    "demonstration_priority_bonus": 0.1,
    "demonstration_batch_fraction": 0.25,
    "one_step_loss_weight": 1.0,
    "n_step_loss_weight": 1.0,
    "margin_loss_weight": 1.0,
    "online_expert_labels": False,
    "imitation_rows": "demonstrations_only_v3",
    "activity_action_bound": 32,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_steps": 16_000,
    "reward_contract_sha256": REWARD_CONTRACT_SHA256,
}
TRAINING_CONFIG_SHA256 = canonical_json_sha256(TRAINING_CONFIG)


def as_rgb_observation(content: Any) -> np.ndarray:
    """Return an owned uint8 frame satisfying the policy input contract."""

    observation = np.asarray(content)
    if observation.shape != OBSERVATION_SHAPE:
        raise ValueError(f"Expected RGB shape {OBSERVATION_SHAPE}, got {observation.shape}.")
    if not np.issubdtype(observation.dtype, np.integer):
        raise TypeError("RGB observations must use an integer dtype.")
    if np.any(observation < 0) or np.any(observation > 255):
        raise ValueError("RGB values must be in [0, 255].")
    return observation.astype(np.uint8, copy=True)


def image_batch(observations: Sequence[Any]) -> np.ndarray:
    """Convert a nonempty NHWC observation sequence to uint8 NCHW."""

    if not observations:
        raise ValueError("At least one observation is required.")
    return np.transpose(
        np.stack([as_rgb_observation(value) for value in observations]),
        (0, 3, 1, 2),
    )


def legal_actions(policy_id: PolicyId) -> tuple[int, ...]:
    """Return the exact native-action mask for one v3 policy."""

    if type(policy_id) is not PolicyId:
        raise TypeError("policy_id must use the exact v3 policy enum.")
    return POLICY_MASKS[policy_id]


def encode_context(
    policy_id: PolicyId,
    player_cell: Sequence[int],
    target_cell: Sequence[int] | None,
) -> np.ndarray:
    """Encode public persistent-goal displacement for one v3 policy."""

    if type(policy_id) is not PolicyId:
        raise TypeError("policy_id must use the exact v3 policy enum.")

    def cell(value: Sequence[int], label: str) -> tuple[int, int]:
        if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
            type(part) is not int for part in value
        ):
            raise TypeError(f"{label} must contain exactly two integers.")
        return int(value[0]), int(value[1])

    player = cell(player_cell, "player_cell")
    if policy_id is PolicyId.EXPLORE or policy_id is PolicyId.EAT_COW and target_cell is None:
        return np.zeros(CONTEXT_SIZE, dtype=np.float32)
    if target_cell is None:
        raise ValueError(f"{policy_id.value} requires a target cell.")
    target = cell(target_cell, "target_cell")
    delta = target[0] - player[0], target[1] - player[1]
    if any(abs(value) > MAX_RELATIVE_TARGET_DELTA for value in delta):
        raise ValueError(f"Relative target delta {delta} is outside the world.")
    return np.asarray(delta, dtype=np.float32) / MAX_RELATIVE_TARGET_DELTA


def build_q_network(
    torch_module: Any, *, mask_hud: bool = False, scale_context: bool = False,
) -> Any:
    """Construct the compact CNN with optional HUD masking and cell-scale goal context."""

    nn = torch_module.nn

    class CrafterActivityQNetwork(nn.Module):  # type: ignore[name-defined,misc]
        """Encode RGB and relative goal context into six native-action Q values."""

        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(64 * 4 * 4 + CONTEXT_SIZE, 512),
                nn.ReLU(),
                nn.Linear(512, N_ACTIONS),
            )

        def forward(self, images: Any, contexts: Any) -> Any:
            """Normalize uint8 NCHW images and append the two float context values."""

            pixels = images.float() / 255.0
            if mask_hud:
                pixels = pixels.clone()
                pixels[:, :, 49:, :] = 0
            features = self.visual(pixels)
            goals = contexts.float() * (MAX_RELATIVE_TARGET_DELTA if scale_context else 1.0)
            return self.head(torch_module.cat((features, goals), dim=1))

    return CrafterActivityQNetwork()


def masked_argmax(values: Any, policy_id: PolicyId) -> Any:
    """Select the greatest legal Q value along the final dimension."""

    actions = legal_actions(policy_id)
    if isinstance(values, np.ndarray):
        if values.shape[-1] != N_ACTIONS or not np.all(np.isfinite(values)):
            raise ValueError("Policy returned an invalid Q vector.")
        selected = np.asarray(actions, dtype=np.int64)
        return selected[np.argmax(values[..., selected], axis=-1)]
    indices = values.new_tensor(actions).long()
    local = values.index_select(-1, indices).argmax(-1)
    return indices[local]


def selection_actions(
    policy_id: PolicyId,
    context: Sequence[float],
    facing: tuple[int, int] | None = None,
    movement_actions: Sequence[int] = (1, 2, 3, 4),
    *,
    interaction_available: bool | None = None,
) -> tuple[int, ...]:
    """Mask observable actions; EatTarget may interact with any faced cow."""

    actions = legal_actions(policy_id)
    if policy_id in {PolicyId.EXPLORE, PolicyId.NAVIGATE_TO}:
        return tuple(action for action in actions if action in movement_actions)
    if policy_id is PolicyId.GET_RESOURCE:
        if facing is None:
            raise ValueError("GetResource selection requires grounded facing.")
        dx, dy = np.rint(np.asarray(context) * 63).astype(int)
        target_delta = int(dx), int(dy)
        if abs(dx) + abs(dy) == 1:
            if facing == target_delta:
                return (5,)
            target_action = DIRECTION_ACTIONS[target_delta]
            return (target_action,) if target_action in movement_actions else ()
        return tuple(action for action in actions if (
            action in movement_actions if action < 5 else False
        ))
    if policy_id not in {PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
        return actions
    if facing is None:
        raise ValueError("Cow-policy selection requires grounded facing.")
    dx, dy = np.rint(np.asarray(context) * 63).astype(int)
    do_legal = abs(dx) + abs(dy) == 1 and facing == (int(dx), int(dy))
    if policy_id is PolicyId.EAT_TARGET and interaction_available is not None:
        do_legal = interaction_available
    return tuple(action for action in actions if (
        action in movement_actions if action < 5 else do_legal
    ))


def select_action(values: Any, actions: Sequence[int]) -> int:
    """Select a finite raw Q vector without modifying the evidence."""

    values = np.asarray(values)
    if values.shape != (N_ACTIONS,) or not np.all(np.isfinite(values)) or not actions:
        raise ValueError("Invalid Q vector or empty execution mask.")
    return int(actions[int(np.argmax(values[list(actions)]))])


def inference_evidence(
    torch_module: Any,
    model: Any,
    observation: Any,
    policy_id: PolicyId,
    player_cell: Sequence[int],
    target_cell: Sequence[int] | None = None,
    *,
    facing: tuple[int, int] | None = None,
    movement_actions: Sequence[int] = (1, 2, 3, 4),
    search_goal_cell: Sequence[int] | None = None,
) -> tuple[int, list[float]]:
    """Return a CPU-greedy action and six unmodified Q values."""

    mask_context = encode_context(policy_id, player_cell, target_cell)
    context = (encode_context(policy_id, player_cell, search_goal_cell)
               if policy_id is PolicyId.EAT_COW and target_cell is None
               and getattr(model, "uses_discovery_context", False) else mask_context)
    images = torch_module.as_tensor(image_batch([observation]), dtype=torch_module.uint8)
    contexts = torch_module.as_tensor(context[None], dtype=torch_module.float32)
    model.cpu().eval()
    with torch_module.inference_mode():
        values = model(images, contexts).detach().cpu().numpy()[0]
    actions = selection_actions(policy_id, mask_context, facing, movement_actions)
    return select_action(values, actions), values.tolist()


def greedy_inference(*args: Any, **kwargs: Any) -> int:
    """Return the selected native action."""

    return inference_evidence(*args, **kwargs)[0]


def checkpoint_data(
    policy_id: PolicyId, model_state: Mapping[str, Any], *, discovery_context: bool = False,
) -> dict[str, Any]:
    """Build the current weights-only checkpoint without training-state dependencies."""

    if discovery_context and policy_id is not PolicyId.EAT_COW:
        raise ValueError("Discovery context belongs only to EatCow.")
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "policy_id": policy_id.value,
        "architecture": POLICY_ARCHITECTURE,
        "input": {**INPUT_CONTRACT, "context_encoding": (
            EAT_COW_WAYPOINT_ENCODING if discovery_context else CONTEXT_ENCODING
        ), "image_preprocessing": (
            "uint8_nhwc_to_float32_nchw_divided_by_255_hud49_zero_v1"
            if policy_id in HUD_MASKED_POLICIES else INPUT_CONTRACT["image_preprocessing"]
        )},
        "model_state_dict": dict(model_state),
    }


def validate_checkpoint(checkpoint: Any, expected_policy_id: PolicyId) -> dict[str, Any]:
    """Validate only the current inference contract; historical schemas are unsupported."""

    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "format_version", "policy_id", "architecture", "input", "model_state_dict",
    }:
        raise ValueError("Policy checkpoint schema is incompatible.")
    discovery_context = (expected_policy_id is PolicyId.EAT_COW
                         and isinstance(checkpoint.get("input"), dict)
                         and checkpoint["input"].get("context_encoding") == EAT_COW_WAYPOINT_ENCODING)
    expected = checkpoint_data(expected_policy_id, {}, discovery_context=discovery_context)
    for field in expected.keys() - {"model_state_dict"}:
        if checkpoint[field] != expected[field]:
            raise ValueError(f"Incompatible checkpoint field {field!r}.")
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise TypeError("Checkpoint has no model weights.")
    return checkpoint


def load_policy_checkpoint(
    torch_module: Any,
    path: str | os.PathLike[str],
    *,
    expected_policy_id: PolicyId,
) -> tuple[Any, dict[str, Any]]:
    """Load a current checkpoint once, validate its weights, and freeze the model."""

    checkpoint = torch_module.load(Path(path), map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, expected_policy_id)
    model = build_q_network(
        torch_module,
        mask_hud=expected_policy_id in HUD_MASKED_POLICIES,
        scale_context=expected_policy_id in HUD_MASKED_POLICIES,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.uses_discovery_context = checkpoint["input"]["context_encoding"] == EAT_COW_WAYPOINT_ENCODING
    if any(not torch_module.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model weights.")
    model.cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(root: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    """Return the single active bundle and manifest paths."""

    base = Path(root) if root is not None else Path(__file__).with_name("artifacts")
    bundle = base.resolve() / ARTIFACT_DIRNAME
    return bundle, bundle / MANIFEST_FILENAME


def validate_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the active models' integrity and inference environment."""

    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "format_version", "artifact_id", "environment", "input", "policies",
    }:
        raise ValueError("Invalid current policy manifest.")
    if (manifest["format_version"] != MANIFEST_FORMAT_VERSION
            or manifest["artifact_id"] != ARTIFACT_DIRNAME
            or manifest["environment"] != ENVIRONMENT_CONTRACT
            or manifest["input"] != INPUT_CONTRACT):
        raise ValueError("Incompatible runtime policy contract.")
    if set(manifest["policies"]) != {policy.value for policy in PolicyId}:
        raise ValueError("Exactly five policies are required.")
    for policy, filename in POLICY_FILENAMES.items():
        record = manifest["policies"][policy.value]
        expected = {"filename": filename, "sha256": file_sha256(path.with_name(filename))}
        if policy is PolicyId.EAT_COW and record.get("context_encoding") == EAT_COW_WAYPOINT_ENCODING:
            expected["context_encoding"] = EAT_COW_WAYPOINT_ENCODING
        if record != expected:
            raise ValueError(f"Invalid checkpoint record: {policy.value}.")
    return manifest


def resolve_policy_bundle(root: str | os.PathLike[str] | None = None) -> tuple[Path, dict[str, Any]]:
    """Resolve the current bundle without inspecting historical artifacts."""

    bundle, path = artifact_paths(root)
    return bundle, validate_manifest(path)


def load_policy_bundle(
    torch_module: Any,
    root: str | os.PathLike[str] | None = None,
) -> tuple[Path, dict[str, Any], dict[PolicyId, Any]]:
    """Load each current frozen policy exactly once."""

    bundle, manifest = resolve_policy_bundle(root)
    models = {
        policy: load_policy_checkpoint(torch_module, bundle / filename, expected_policy_id=policy)[0]
        for policy, filename in POLICY_FILENAMES.items()
    }
    for policy, model in models.items():
        declared = manifest["policies"][policy.value].get("context_encoding") == EAT_COW_WAYPOINT_ENCODING
        if model.uses_discovery_context != declared:
            raise ValueError(f"Manifest and checkpoint context differ: {policy.value}.")
    return bundle, manifest, models
