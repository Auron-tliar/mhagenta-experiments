"""Prepare a non-overwriting, CPU-only AchieveOn warm-start artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import torch
from mha_exp_level2_bw.achieve_on import policy as shared_policy

from tools.bw_achieve_on_preparation.policy import (
    MODEL_INPUT_SHAPE, checkpoint_payload, warm_start,
)
from mha_exp_level2_bw.exp2_5.policy import (
    MODEL_INPUT_SHAPE as TRANSFER_INPUT_SHAPE,
    artifact_paths, file_sha256, load_policy_checkpoint,
)


def prepare(output: Path) -> dict:
    """Save verified initialization and lineage without training or CUDA use."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    torch.set_num_threads(1)
    model, provenance = warm_start(torch)
    teacher, _ = load_policy_checkpoint(torch, artifact_paths()[1])
    # Zero-extension must preserve Transfer for any feature input, not just
    # one successful trajectory. A local RNG avoids changing training seeds.
    rng = np.random.default_rng(260_905)
    original = rng.integers(0, 2, size=(64, *TRANSFER_INPUT_SHAPE), dtype=np.uint8)
    extended = np.zeros((64, *MODEL_INPUT_SHAPE), dtype=np.uint8)
    extended[:, :TRANSFER_INPUT_SHAPE[0]] = original
    extended[:, -1] = rng.integers(0, 2, size=extended[:, -1].shape, dtype=np.uint8)
    with torch.no_grad():
        expected = teacher(torch.as_tensor(original, dtype=torch.float32))
        actual = model(torch.as_tensor(extended, dtype=torch.float32))
        error = float((expected - actual).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "achieve-on-initial.pt"
    torch.save(checkpoint_payload(model, provenance), checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded["model_state_dict"][key], rtol=0, atol=0)
    report = {
        "status": "initialized-only",
        "ready_for_runtime": False,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "provenance": provenance,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "warm_start_parity_max_absolute_error": error,
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_sha256": {f"{label}/{path.name}": file_sha256(path)
                          for label, parent in [("tools", Path(__file__).parent),
                                                ("shared", Path(shared_policy.__file__).parent)]
                          for path in sorted(parent.glob("*.py"))},
    }
    (output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Create an initial candidate in an explicitly selected fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
