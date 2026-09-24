"""Direct AchieveOn candidate initialized from the frozen Transfer network.

This model emits the four atomic environment actions. It is
independent of the frozen Transfer artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mha_exp_level2_bw.exp2_5.contracts import GoalSpec
from mha_exp_level2_bw.exp2_5.grounding import (
    NUM_BLOCKS, OBSERVATION_SHAPE, TABLE_LEN, ground_observation,
)
from mha_exp_level2_bw.exp2_5.policy import (
    artifact_paths, legal_action_indices,
    load_policy_checkpoint, validate_manifest,
)

ARCHITECTURE = "achieve_on_transfer_initialized_structured_mlp_v1"
CONV3D_ARCHITECTURE = "achieve_on_transfer_initialized_conv3d_residual_v1"
MODEL_INPUT_SHAPE = (NUM_BLOCKS + 5, TABLE_LEN, NUM_BLOCKS)
FORMAT_VERSION = 1


def condition_goal(observation: Any, goal: GoalSpec) -> np.ndarray:
    """Encode top identity, bottom location, and explicit bottom identity.

    Bottom location follows the held block if the candidate picks it up. No
    planner or intermediate transfer target is supplied to the candidate.
    """
    names = tuple(f"b{index}" for index in range(NUM_BLOCKS))
    if goal.top not in names or goal.bottom not in names or goal.top == goal.bottom:
        raise ValueError("AchieveOn requires two distinct canonical block names.")
    grounded = ground_observation(observation)
    numeric = grounded.observation
    bottom = names.index(goal.bottom)
    columns = np.flatnonzero(np.any(numeric[1:, :, bottom], axis=0))
    if len(columns) != 1:
        raise ValueError("Goal bottom must occupy exactly one table column.")
    result = np.zeros(MODEL_INPUT_SHAPE, dtype=np.uint8)
    result[:OBSERVATION_SHAPE[0]] = numeric
    result[NUM_BLOCKS + 2, :, names.index(goal.top)] = 1
    result[NUM_BLOCKS + 3, int(columns[0]), :] = 1
    result[NUM_BLOCKS + 4, :, bottom] = 1
    return result


def goal_succeeded(observation: Any, goal: GoalSpec) -> bool:
    """Require both the requested support relation and an empty hand."""
    condition_goal(observation, goal)
    facts = ground_observation(observation).facts
    return goal.fact in facts and "hand-empty()" in facts


def conv3d_volume(torch: Any, inputs: Any) -> Any:
    """Map to N,C,height,column,identity; context channels broadcast over height.

    Channel order is occupancy, arm, holding, goal top, goal bottom and bottom
    column. Block identity is categorical, so its local kernel neighborhoods
    are an experimental ordering assumption, not physical spatial depth.
    """
    context = [inputs[:, index].unsqueeze(1).expand(-1, NUM_BLOCKS, -1, -1)
               for index in (0, 1, NUM_BLOCKS + 2, NUM_BLOCKS + 4, NUM_BLOCKS + 3)]
    return torch.stack([inputs[:, 2:NUM_BLOCKS + 2], *context], dim=1)


def build_network(torch: Any, architecture: str = ARCHITECTURE) -> Any:
    """Build the original MLP or its explicitly named Conv3d residual variant."""
    if architecture not in {ARCHITECTURE, CONV3D_ARCHITECTURE}:
        raise ValueError(f"Unknown AchieveOn architecture: {architecture}")
    nn = torch.nn

    class AchieveOnNetwork(nn.Module):
        """Shared entity encoder and four-action head, matching Transfer layout."""

        def __init__(self) -> None:
            super().__init__()
            self.architecture = ARCHITECTURE
            features = NUM_BLOCKS * TABLE_LEN + TABLE_LEN + 1 + 2 * TABLE_LEN + 1
            self.entity_encoder = nn.Sequential(
                nn.Linear(features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            )
            self.value_head = nn.Sequential(
                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4),
            )

        def forward(self, inputs: Any) -> Any:
            """Score atomic actions from numeric state and the final On goal."""
            batch = inputs.shape[0]
            stacks = inputs[:, 2:NUM_BLOCKS + 2].permute(0, 3, 1, 2)
            stacks = stacks.reshape(batch, NUM_BLOCKS, -1)
            holding = inputs[:, 1].permute(0, 2, 1)
            top = inputs[:, NUM_BLOCKS + 2].permute(0, 2, 1).mean(2, keepdim=True)
            bottom = inputs[:, NUM_BLOCKS + 4].permute(0, 2, 1).mean(2, keepdim=True)
            arm = inputs[:, 0].mean(2)
            destination = inputs[:, NUM_BLOCKS + 3].mean(2)
            context = torch.cat((arm, destination), dim=1)
            context = context.unsqueeze(1).expand(-1, NUM_BLOCKS, -1)
            entities = torch.cat((stacks, holding, top, context, bottom), dim=2)
            encoded = self.entity_encoder(entities)
            pooled = torch.cat((encoded.mean(1), encoded.max(1).values), dim=1)
            return self.value_head(pooled)

    class Conv3dResidualNetwork(nn.Module):
        """Retain the transferred MLP and learn a volumetric action correction."""

        def __init__(self) -> None:
            super().__init__()
            self.architecture = CONV3D_ARCHITECTURE
            self.base = AchieveOnNetwork()
            self.convolutions = nn.Sequential(
                nn.Conv3d(6, 8, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            )
            self.correction = nn.Sequential(
                nn.Flatten(), nn.Linear(8 * NUM_BLOCKS * TABLE_LEN * NUM_BLOCKS, 32),
                nn.ReLU(), nn.Linear(32, 4),
            )
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

        def forward(self, inputs: Any) -> Any:
            """Start with exact transferred values and add a learned correction."""
            return self.base(inputs) + self.correction(self.convolutions(conv3d_volume(torch, inputs)))

    return AchieveOnNetwork() if architecture == ARCHITECTURE else Conv3dResidualNetwork()


def warm_start(torch: Any, architecture: str = ARCHITECTURE) -> tuple[Any, dict[str, Any]]:
    """Copy Transfer exactly, with zero new role weights and residual output.

    The returned model is trainable. The packaged teacher remains frozen and
    unchanged. An independent optimizer must be created for the candidate.
    """
    manifest_path, checkpoint_path = artifact_paths()
    manifest = validate_manifest(manifest_path, checkpoint_path)
    teacher, checkpoint = load_policy_checkpoint(torch, checkpoint_path, device="cpu")
    if checkpoint["weight_optimizer_steps"] != manifest["training"]["selected_optimizer_steps"]:
        raise ValueError("Transfer checkpoint and manifest optimizer steps differ.")
    model = build_network(torch, architecture)
    base = model if architecture == ARCHITECTURE else model.base
    original = teacher.state_dict()
    weights = base.state_dict()
    for name, source in original.items():
        if name == "entity_encoder.0.weight":
            weights[name].zero_()
            weights[name][:, :-1].copy_(source)
        else:
            weights[name].copy_(source)
    base.load_state_dict(weights, strict=True)
    return model, {
        "transfer_sha256": manifest["checkpoint_sha256"],
        "transfer_architecture": manifest["architecture"],
        "copied_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
        "added_parameters": sum(parameter.numel() for parameter in model.parameters()) - sum(parameter.numel() for parameter in teacher.parameters()),
        "new_feature_initialization": "zero",
        **({"residual_initialization": "random-convolutions-zero-output-head",
            "initial_values_equal_transfer_mlp": True} if architecture == CONV3D_ARCHITECTURE else {}),
        "optimizer_inherited": False,
        "qualification_inherited": False,
    }


def infer(torch: Any, model: Any, observation: Any, goal: GoalSpec) -> tuple[int, list[float]]:
    """Choose a legal atomic action without consulting a teacher or planner."""
    inputs = torch.as_tensor(
        condition_goal(observation, goal), dtype=torch.float32,
        device=next(model.parameters()).device,
    ).unsqueeze(0)
    with torch.no_grad():
        values = model(inputs).squeeze(0).cpu().tolist()
    legal = legal_action_indices(observation)
    return max(legal, key=lambda action: values[action]), values


def checkpoint_payload(model: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    """Describe initialization honestly as an untrained, unqualified candidate."""
    return {
        "format_version": FORMAT_VERSION,
        "architecture": model.architecture,
        "model_input_shape": MODEL_INPUT_SHAPE,
        "observation_shape": OBSERVATION_SHAPE,
        "action_names": ["pick-up", "put-down", "move-left", "move-right"],
        "goal": "on(top,bottom) and hand-empty",
        "status": "initialized-only",
        "optimizer_steps": 0,
        "provenance": provenance,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
    }
