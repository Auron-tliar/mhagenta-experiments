"""Allowed initial policies and native-action candidates for continuing learning."""

from copy import deepcopy
from typing import Any

from ..exp2_5.policy import (
    PolicyId, artifact_paths, build_q_network, file_sha256, load_policy_checkpoint,
)

BASIC_HASHES = {
    "explore": "703c8bc7b8cde753cc803f54ff9ba193406957871f99aef9128f10212170a683",
    "navigate_to": "f44fd2d91437975d2979c25ac3df186061f93da1b108186dbf4c07dd68f2e00d",
}
SKILLS = ("get_resource", "eat_target")


def load_basics(torch: Any) -> dict[str, Any]:
    """Load exactly two accepted checkpoints, never a pretrained acquisition skill."""
    root, _ = artifact_paths()
    models = {}
    for name, digest in BASIC_HASHES.items():
        path = root / (name.replace("_", "-") + "-policy.pt")
        if file_sha256(path) != digest:
            raise ValueError(f"Initial {name} checkpoint identity changed")
        models[name], _ = load_policy_checkpoint(torch, path, expected_policy_id=PolicyId(name))
    return models


def candidate(torch: Any, basic: Any) -> Any:
    """Warm-start the complete NavigateTo architecture, including its preprocessing.

    Both new skills retain HUD masking and cell-scale context, so initialization
    is prediction-equivalent to NavigateTo. Six output logits already exist.
    """
    model = build_q_network(torch, mask_hud=True, scale_context=True)
    model.load_state_dict(deepcopy(basic.state_dict()), strict=True)
    return model


def weights_hash(model: Any) -> str:
    """Identify the exact tensor bytes independent of checkpoint serialization."""
    return tensor_hash(model.state_dict())


def tensor_hash(weights: dict) -> str:
    """Authenticate a checkpoint without constructing a second neural network."""
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
