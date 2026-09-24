"""Assess one frozen pilot selection once, without updating its weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mha_exp_level2_bw.exp2_5.policy import artifact_paths, file_sha256, load_policy_checkpoint
from mha_exp_level2_bw.achieve_on.policy import ARCHITECTURE, CONV3D_ARCHITECTURE, build_network
from tools.bw_achieve_on_preparation.train import (
    Config, case, evaluate, make_environment, planner, teacher_episode,
)


def assess(candidate: Path, output: Path, count: int = 200, start_index: int = 0) -> dict:
    """Compare direct AchieveOn and planner-plus-Transfer on identical new cases.

    This is a held-out measurement, not automatic artifact qualification or
    adoption. Once inspected, this fixed assessment set is no longer untouched.
    """
    if output.exists():
        raise FileExistsError(output)
    if count <= 0 or start_index < 0 or start_index + count > 1_000_000:
        raise ValueError("Assessment case count is outside the split window.")
    report = json.loads((candidate / "report.json").read_text(encoding="utf-8"))
    if report["status"] not in {"completed-pilot-unqualified", "time-limit-unqualified"} or not report.get("selected_checkpoint"):
        raise ValueError("Assessment requires a completed selection from a finished or time-limited pilot.")
    filename = report["selected_checkpoint"]
    if Path(filename).name != filename:
        raise ValueError("Selected checkpoint must be inside the candidate directory.")
    checkpoint = candidate / filename
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["architecture"] not in {ARCHITECTURE, CONV3D_ARCHITECTURE}
            or payload["architecture"] != report.get("architecture", ARCHITECTURE)
            or payload["provenance"] != report["provenance"]):
        raise ValueError("Candidate checkpoint identity differs from the pilot.")
    expected_digest = next(item["checkpoint_sha256"] for item in report["selection"]
                           if item["checkpoint"] == filename)
    if file_sha256(checkpoint) != expected_digest:
        raise ValueError("Selected checkpoint bytes differ from the selection record.")
    if file_sha256(artifact_paths()[1]) != payload["provenance"]["transfer_sha256"]:
        raise ValueError("The current Transfer baseline differs from initialization.")
    config = Config(**report["configuration"])
    torch.set_num_threads(1)
    model = build_network(torch, payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": expected_digest, "cases": count, "start_index": start_index,
              "transfer_sha256": payload["provenance"]["transfer_sha256"],
              "automatic_promotion": False, "architecture": payload["architecture"],
              "training_status": report["status"]}
    target = output / "assessment.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    environment = make_environment()
    try:
        result["achieve_on"] = evaluate(model, config, "assessment", count, start_index)
        teacher, _ = load_policy_checkpoint(torch, artifact_paths()[1])
        service = planner()
        baseline = []
        for index in range(start_index, start_index + count):
            observation, goal, seed = case(environment, "assessment", index)
            _, evidence = teacher_episode(environment, observation, goal, teacher, service, config)
            evidence["seed"] = seed
            baseline.append(evidence)
        result["baseline"] = baseline
        pairs = list(zip(result["achieve_on"]["results"], baseline, strict=True))
        both = [(learner, base) for learner, base in pairs if learner["success"] and base["success"]]
        result["paired"] = {
            "achieve_on_successes": sum(learner["success"] for learner, _ in pairs),
            "baseline_successes": sum(base["success"] for _, base in pairs),
            "achieve_on_only_successes": sum(learner["success"] and not base["success"] for learner, base in pairs),
            "baseline_only_successes": sum(base["success"] and not learner["success"] for learner, base in pairs),
            "joint_successes": len(both),
            "actions_saved_on_joint_successes": sum(len(base["actions"]) - len(learner["actions"]) for learner, base in both),
        }
        result["status"] = "completed-assessment"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        environment.close()
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Assess one selected candidate into a fresh evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    result = assess(args.candidate, args.output, args.cases, args.start_index)
    print(json.dumps(result["paired"], indent=2))


if __name__ == "__main__":
    main()
