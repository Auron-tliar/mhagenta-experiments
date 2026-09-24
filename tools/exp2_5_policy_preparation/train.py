"""Train and certify the single selected 2-5-BW DQfD-lite policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from evaluation import (
    BATCH_SIZE,
    DISCOUNT,
    GRADIENT_CLIP,
    HELD_OUT_CASES,
    HELD_OUT_SEED_START,
    INTEGRATION_SEED,
    LEARNING_RATE,
    REPLAY_CAPACITY,
    SEQUENTIAL_EVALUATION_SEEDS,
    SEQUENTIAL_TRANSFERS,
    TARGET_SYNC_STEPS,
    evaluate_case,
    evaluate_sequential_case,
    evaluate_transfer,
    make_environment,
    require_torch,
)
from regression import OSCILLATING_B7_TRANSFER, reconstruct_regression
from selected_dqfd import (
    DEMONSTRATION_PRIORITY_BONUS,
    DEMONSTRATION_STEPS,
    LARGE_MARGIN,
    LARGE_MARGIN_WEIGHT,
    N_STEP_RETURN,
    PRIORITY_ALPHA,
    PRIORITY_BETA_START,
    VARIANT_DQFD_LITE,
    train_selected,
)

from mha_exp_level2_bw.exp2_5.grounding import NUM_BLOCKS, OBSERVATION_SHAPE, TABLE_LEN
from mha_exp_level2_bw.exp2_5.policy import (
    ARTIFACT_SET_DIRNAME,
    CHECKPOINT_FORMAT_VERSION,
    MANIFEST_FILENAME,
    MODEL_INPUT_SHAPE,
    N_ACTIONS,
    POLICY_ARCHITECTURE,
    POLICY_FILENAME,
    QUALIFICATION_PROTOCOL_ID,
    QUALIFICATION_PROTOCOL_SHA256,
    build_q_network,
    file_sha256,
    validate_manifest,
)


FINAL_TRAINING_SEED = 2505
FULL_TRAINING_STEPS = 250_000
DEFAULT_CANDIDATE_DIR = Path("results/2-5-bw-dqfd-v2-full/candidate")
DEFAULT_RESULTS_DIR = Path("results/2-5-bw-dqfd-v2-full/certification")
DEFAULT_ARTIFACT_DIR = Path("results/2-5-bw-dqfd-v2-full/artifact") / ARTIFACT_SET_DIRNAME


QUALIFICATION_PROTOCOL = {
    "id": QUALIFICATION_PROTOCOL_ID,
    "environment": {
        "table_locations": 5,
        "blocks": 8,
        "observation": "numeric",
        "grounding": "exp2-5-bw-v1",
        "actions": ["pick-up", "put-down", "move-left", "move-right"],
        "inference": "greedy-legal-masked",
        "max_actions_per_transfer": 32,
    },
    "held_out": {
        "seed_start": 370000,
        "seed_end": 370099,
        "case_index": "seed",
        "selection": "case_index % len(sorted_legal_candidates)",
    },
    "integration": {
        "seed": 1000,
        "case_index": 0,
        "transfer": {
            "block": "b3",
            "source_support": "t3",
            "destination_support": "b4",
            "source": "t3",
            "destination": "t4",
        },
    },
    "sequential": {
        "seed_start": 371000,
        "seed_end": 371019,
        "transfers_per_sequence": 10,
        "selection": "(seed + transfer_index) % len(sorted_legal_candidates)",
    },
    "regression": {
        "id": OSCILLATING_B7_TRANSFER.name,
        "environment_seed": OSCILLATING_B7_TRANSFER.environment_seed,
        "action_prefix": list(OSCILLATING_B7_TRANSFER.action_prefix),
        "transfer": OSCILLATING_B7_TRANSFER.spec.as_dict(),
        "conditioned_input_sha256": OSCILLATING_B7_TRANSFER.input_sha256,
    },
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if _canonical_sha256(QUALIFICATION_PROTOCOL) != QUALIFICATION_PROTOCOL_SHA256:
    raise RuntimeError("The offline certification protocol differs from the runtime contract.")


def train_candidate(*, output_dir: Path, device: str | None = None) -> dict[str, Any]:
    """Train only the selected structured DQfD-lite method at its frozen budget."""

    if output_dir.exists():
        raise FileExistsError(f"Candidate output already exists at {output_dir}.")
    return train_selected(
        variant=VARIANT_DQFD_LITE,
        seed=FINAL_TRAINING_SEED,
        training_steps=FULL_TRAINING_STEPS,
        output_dir=output_dir,
        device=device,
    )


def _load_candidate(
    torch_module: Any,
    candidate_dir: Path,
    device: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    report = json.loads((candidate_dir / "report.json").read_text(encoding="utf-8"))
    checkpoint = torch_module.load(
        candidate_dir / "checkpoint.pt",
        map_location=device,
        weights_only=True,
    )
    expected = {
        "variant": VARIANT_DQFD_LITE,
        "seed": FINAL_TRAINING_SEED,
        "search_environment_steps": FULL_TRAINING_STEPS,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Candidate checkpoint does not match the fixed preparation run.")
    if report.get("training_environment_steps") != FULL_TRAINING_STEPS:
        raise ValueError("Candidate did not complete the fixed environment-step budget.")
    model = build_q_network(torch_module).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, report


def _save_checkpoint(
    torch_module: Any,
    model: Any,
    path: Path,
    optimizer_steps: int,
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": POLICY_ARCHITECTURE,
        "observation_shape": OBSERVATION_SHAPE,
        "model_input_shape": MODEL_INPUT_SHAPE,
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        "n_actions": N_ACTIONS,
        "weight_optimizer_steps": optimizer_steps,
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(".tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


def certify_candidate(
    *,
    candidate_dir: Path,
    artifact_dir: Path,
    results_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the exact untouched protocol once and emit a compact new artifact."""

    if artifact_dir.exists() or results_dir.exists():
        raise FileExistsError("Certification refuses to overwrite artifact or result directories.")
    torch_module = require_torch()
    selected_device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    model, checkpoint, candidate_report = _load_candidate(
        torch_module, candidate_dir, selected_device
    )
    environment = make_environment()
    try:
        held_out = [
            evaluate_case(
                torch_module,
                model,
                environment,
                HELD_OUT_SEED_START + offset,
                HELD_OUT_SEED_START + offset,
            )
            for offset in range(HELD_OUT_CASES)
        ]
        integration = evaluate_case(
            torch_module, model, environment, INTEGRATION_SEED, 0
        )
        sequential = [
            evaluate_sequential_case(
                torch_module, model, environment, seed, SEQUENTIAL_TRANSFERS
            )
            for seed in SEQUENTIAL_EVALUATION_SEEDS
        ]
        case = OSCILLATING_B7_TRANSFER
        observation = reconstruct_regression(environment, case)
        regression, _ = evaluate_transfer(
            torch_module, model, environment, observation, case.spec, case.environment_seed
        )
    finally:
        environment.close()

    cohorts = {
        "held_out": {"cases": len(held_out), "successes": sum(item.success for item in held_out)},
        "integration": {"cases": 1, "successes": int(integration.success)},
        "sequential": {"cases": len(sequential), "successes": sum(item.success for item in sequential)},
        "regressions": {"cases": 1, "successes": int(regression.success)},
    }
    if integration.goal != QUALIFICATION_PROTOCOL["integration"]["transfer"]:
        raise RuntimeError("The fixed integration case identity changed.")
    if any(result["successes"] != result["cases"] for result in cohorts.values()):
        raise RuntimeError(f"Candidate failed the all-cases-successful rule: {cohorts}")

    artifact_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    optimizer_steps = int(checkpoint["selected_optimizer_step"])
    checkpoint_path = artifact_dir / POLICY_FILENAME
    _save_checkpoint(torch_module, model, checkpoint_path, optimizer_steps)
    manifest = {
        "format_version": 4,
        "architecture": POLICY_ARCHITECTURE,
        "frozen": True,
        "checkpoint": {
            "filename": POLICY_FILENAME,
            "sha256": file_sha256(checkpoint_path),
            "observation_shape": list(OBSERVATION_SHAPE),
            "input_shape": list(MODEL_INPUT_SHAPE),
            "actions": N_ACTIONS,
        },
        "training": {
            "method": VARIANT_DQFD_LITE,
            "seed": FINAL_TRAINING_SEED,
            "elapsed_seconds": candidate_report["elapsed_seconds"],
            "search_environment_steps": FULL_TRAINING_STEPS,
            "search_optimizer_steps": int(candidate_report["optimizer_steps"]),
            "selected_environment_step": int(checkpoint["selected_environment_step"]),
            "selected_optimizer_steps": optimizer_steps,
            "hyperparameters": {
                "online_environment_steps": FULL_TRAINING_STEPS,
                "demonstration_environment_steps": DEMONSTRATION_STEPS,
                "replay_capacity": REPLAY_CAPACITY + DEMONSTRATION_STEPS,
                "protected_demonstrations": DEMONSTRATION_STEPS,
                "batch_size": BATCH_SIZE,
                "discount": DISCOUNT,
                "learning_rate": LEARNING_RATE,
                "gradient_clip": GRADIENT_CLIP,
                "target_sync_steps": TARGET_SYNC_STEPS,
                "double_dqn": True,
                "prioritized_replay": True,
                "priority_alpha": PRIORITY_ALPHA,
                "priority_beta_start": PRIORITY_BETA_START,
                "n_step_return": N_STEP_RETURN,
                "large_margin": LARGE_MARGIN,
                "large_margin_weight": LARGE_MARGIN_WEIGHT,
                "demonstration_priority_bonus": DEMONSTRATION_PRIORITY_BONUS,
                "behavior_cloning_pretraining": False,
            },
        },
        "qualification": {
            "protocol": {
                "id": QUALIFICATION_PROTOCOL_ID,
                "sha256": QUALIFICATION_PROTOCOL_SHA256,
            },
            "transfers_per_sequence": SEQUENTIAL_TRANSFERS,
            **cohorts,
        },
    }
    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "protocol": QUALIFICATION_PROTOCOL,
        "cohorts": cohorts,
        "held_out": [asdict(item) for item in held_out],
        "integration": asdict(integration),
        "sequential": [asdict(item) for item in sequential],
        "regression": asdict(regression),
    }
    (results_dir / "certification-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    validate_manifest(manifest_path, checkpoint_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the selected training and one-time certification CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    train.add_argument("--device", choices=("cpu", "cuda"))
    certify = commands.add_parser("certify")
    certify.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    certify.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    certify.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    certify.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested non-overwriting offline preparation phase."""

    args = build_parser().parse_args(argv)
    if args.command == "train":
        train_candidate(output_dir=args.output_dir.resolve(), device=args.device)
    else:
        certify_candidate(
            candidate_dir=args.candidate_dir.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            results_dir=args.results_dir.resolve(),
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
