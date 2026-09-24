"""Explicit size-specific Transfer warm starts; the original artifact stays frozen."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .policy import artifact_paths, build_q_network, file_sha256, load_policy_checkpoint, validate_manifest


RESIZED_CHECKSUMS = {
    (4, 6): "e564efce25c6b27e91c277774dec654a0e1a29525e5ad0e95f493a604b13bc47",
    (7, 12): "087c2e0d0ad5a4f02ad674ee8cb9e265c5a8878109737ee76ff3af2348c85f0b",
}


def frozen_artifact(table_len: int = 5, num_blocks: int = 8) -> tuple[Path, dict[str, Any]]:
    """Select the certified original or one exact assessed size-specific artifact."""
    if (table_len, num_blocks) == (5, 8):
        manifest_path, checkpoint = artifact_paths()
        manifest = validate_manifest(manifest_path, checkpoint)
        return checkpoint, {"table_len": table_len, "num_blocks": num_blocks,
                            "checkpoint_sha256": manifest["checkpoint_sha256"],
                            "manifest_sha256": file_sha256(manifest_path), "kind": "original"}
    expected = RESIZED_CHECKSUMS.get((table_len, num_blocks))
    if expected is None:
        raise ValueError("No qualified Transfer artifact for this world size.")
    directory = Path(__file__).with_name("artifacts") / "structured-dqfd-resized-v1" / f"{table_len}x{num_blocks}"
    checkpoint, manifest_path = directory / "transfer-policy.pt", directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = manifest["result"]
    measured = result["assessment"]
    if (manifest["format"] != "resized-transfer-preparation-v1" or not result["accepted"]
            or (result["table_len"], result["num_blocks"]) != (table_len, num_blocks)
            or file_sha256(checkpoint) != expected or manifest["checkpoint_sha256"] != expected
            or measured["single_successes"] < 990 or measured["sequence_successes"] < 95
            or measured["successful_transfers"] < 1980 or measured["requested_transfers"] != 2000):
        raise ValueError("Resized Transfer qualification or identity mismatch.")
    return checkpoint, {"table_len": table_len, "num_blocks": num_blocks,
                        "checkpoint_sha256": expected, "manifest_sha256": file_sha256(manifest_path),
                        "kind": "resized", "assessment": measured}


def load_frozen(torch_module: Any, *, table_len: int = 5, num_blocks: int = 8) -> tuple[Any, dict[str, Any]]:
    """Load the exact dimension-compatible policy as frozen CPU inference weights."""
    path, reference = frozen_artifact(table_len, num_blocks)
    if reference["kind"] == "original":
        model, _ = load_policy_checkpoint(torch_module, path)
    else:
        payload = torch_module.load(path, map_location="cpu", weights_only=True)
        if (payload["format_version"] != "resized-transfer-v1"
                or (payload["table_len"], payload["num_blocks"]) != (table_len, num_blocks)):
            raise ValueError("Resized checkpoint dimensions differ from the requested world.")
        model = build_q_network(torch_module, table_len=table_len, num_blocks=num_blocks)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            if not bool(torch_module.isfinite(parameter).all()):
                raise ValueError("Non-finite resized policy weights.")
            parameter.requires_grad_(False)
    return model, reference


def feature_keys(table_len: int, num_blocks: int) -> list[tuple[str, int, int]]:
    """Name encoder inputs, measuring stack height upward from the table."""
    return (
        [("stack", num_blocks - 1 - row, column)
         for row in range(num_blocks) for column in range(table_len)]
        + [("holding", 0, column) for column in range(table_len)]
        + [("target", 0, 0)]
        + [(kind, 0, column) for kind in ("arm", "destination") for column in range(table_len)]
    )


def warm_start(
    torch_module: Any, source: Path, *, table_len: int, num_blocks: int, device: str,
) -> tuple[Any, dict[str, Any]]:
    """Copy the original policy, remapping input features and zeroing new connections."""
    original, checkpoint = load_policy_checkpoint(torch_module, source, device="cpu")
    model = build_q_network(torch_module, table_len=table_len, num_blocks=num_blocks)
    source_state = original.state_dict()
    state = model.state_dict()
    first = "entity_encoder.0.weight"
    for name in state:
        if name != first:
            state[name].copy_(source_state[name])
    source_keys = {key: index for index, key in enumerate(feature_keys(5, 8))}
    mapping = [(index, source_keys[key])
               for index, key in enumerate(feature_keys(table_len, num_blocks)) if key in source_keys]
    state[first].zero_()
    for destination_index, source_index in mapping:
        state[first][:, destination_index].copy_(source_state[first][:, source_index])
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    return model, {
        "source_sha256": file_sha256(source), "source_table_len": 5, "source_num_blocks": 8,
        "source_optimizer_steps": checkpoint["weight_optimizer_steps"],
        "input_mapping": mapping, "new_connections": "zero",
        "stack_alignment": "bottom", "column_alignment": "left",
        "copied_input_features": len(mapping),
        "new_input_features": len(feature_keys(table_len, num_blocks)) - len(mapping),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_parameters": 0,
    }
