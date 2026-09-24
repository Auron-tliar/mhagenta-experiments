"""Compare an assessed Conv3d candidate with the frozen MLP on the same cases."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mha_exp_level2_bw.achieve_on.policy import ARCHITECTURE, build_network, infer
from mha_exp_level2_bw.exp2_5.policy import file_sha256
from tools.bw_achieve_on_preparation.train import Config, case, evaluate, make_environment


def selected_model(directory: Path) -> tuple[torch.nn.Module, dict, Path]:
    """Load a finished selection with matching checkpoint bytes and provenance."""
    report = json.loads((directory / "report.json").read_text())
    if report["status"] not in {"completed-pilot-unqualified", "time-limit-unqualified"}:
        raise ValueError("Candidate training is not finished.")
    name = report["selected_checkpoint"]
    if Path(name).name != name:
        raise ValueError("Checkpoint must be inside the training directory.")
    path = directory / name
    selection = next(item for item in report["selection"] if item["checkpoint"] == name)
    if file_sha256(path) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["provenance"] != report["provenance"]
            or payload["architecture"] != report.get("architecture", ARCHITECTURE)):
        raise ValueError("Selected checkpoint provenance differs.")
    model = build_network(torch, payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite model.")
    model.eval().requires_grad_(False)
    return model, report, path


def latency(model: torch.nn.Module, observations: list[tuple], device: str) -> dict:
    """Measure complete single-action inference, including encoding and transfers."""
    model.to(device)
    for observation, goal in observations[:10]:
        infer(torch, model, observation, goal)
    samples = []
    for observation, goal in observations:
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        infer(torch, model, observation, goal)
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - started))
    model.cpu()
    return {"device": device, "samples": len(samples), "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95))}


def compare(candidate: Path, reference: Path, assessment: Path, output: Path) -> dict:
    """Reuse Transfer's paired assessment; evaluate MLP and ID-order sensitivity."""
    if output.exists():
        raise FileExistsError(output)
    measured = json.loads(assessment.read_text())
    if measured["status"] != "completed-assessment":
        raise ValueError("The paired Transfer assessment must finish first.")
    torch.set_num_threads(1)
    model, report, path = selected_model(candidate)
    old, old_report, old_path = selected_model(reference)
    if (file_sha256(path) != measured["checkpoint_sha256"]
            or report["provenance"]["transfer_sha256"] != old_report["provenance"]["transfer_sha256"]):
        raise ValueError("Assessment or Transfer initialization identity differs.")
    config = Config(**report["configuration"])
    if config.action_cap != old_report["configuration"]["action_cap"]:
        raise ValueError("Comparator action caps differ.")
    count, start = measured["cases"], measured["start_index"]
    old_result = evaluate(old, config, "assessment", count, start)
    pairs = list(zip(measured["achieve_on"]["results"], old_result["results"], strict=True))
    if any(a["seed"] != b["seed"] or a["goal"] != b["goal"] for a, b in pairs):
        raise ValueError("Comparator cases are not paired.")
    result = {"candidate_sha256": file_sha256(path), "mlp_sha256": file_sha256(old_path),
              "assessment_sha256": file_sha256(assessment), "cases": count, "start_index": start,
              "mlp": old_result, "candidate_successes": measured["achieve_on"]["successes"],
              "mlp_successes": old_result["successes"],
              "joint_successes": sum(a["success"] and b["success"] for a, b in pairs),
              "actions_saved_on_joint_successes": sum(len(b["actions"]) - len(a["actions"])
                                                       for a, b in pairs if a["success"] and b["success"]),
              "latency": {}, "relabeling": {}, "parameters": {}, "automatic_promotion": False}
    environment = make_environment()
    try:
        observations = [case(environment, "assessment", index)[:2] for index in range(start, start + min(count, 100))]
    finally:
        environment.close()

    class Relabeled(torch.nn.Module):
        """Apply a fixed consistent categorical relabeling to all conditioned planes."""
        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.base(inputs[..., [7, 1, 4, 0, 3, 6, 2, 5]])

    for name, network, original in (("conv3d", model, measured["achieve_on"]), ("mlp", old, old_result)):
        result["parameters"][name] = sum(p.numel() for p in network.parameters())
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        result["latency"][name] = [latency(network, observations, device) for device in devices]
        permuted = evaluate(Relabeled(network), config, "assessment", count, start)
        result["relabeling"][name] = {"permutation": [7, 1, 4, 0, 3, 6, 2, 5],
            "successes": permuted["successes"], "results": permuted["results"],
            "changed_action_traces": sum(a["actions"] != b["actions"] for a, b in zip(original["results"], permuted["results"], strict=True))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Write one comparison artifact without repeating Transfer planning."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "reference", "assessment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate, args.reference, args.assessment, args.output)
    print(json.dumps({key: result[key] for key in ("candidate_successes", "mlp_successes", "latency")}))


if __name__ == "__main__":
    main()
