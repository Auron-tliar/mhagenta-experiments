"""Fine-tune a native-size Transfer with DQfD and frequent durable progress reports."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from evaluation import transition_reward
from selected_dqfd import (
    CONFIGS, VARIANT_DQFD_LITE, RawTransition, SelectedReplayBuffer, _optimize, emit_n_step,
)
from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.evaluation import evaluate_transfer
from mha_exp_level2_bw.exp2_5.grounding import (
    enumerate_transfer_targets, ground_observation, transfer_phase,
)
from mha_exp_level2_bw.exp2_5.policy import (
    MAX_POLICY_STEPS, POLICY_ARCHITECTURE, artifact_paths, condition_observation,
    file_sha256, greedy_inference, legal_action_indices,
)
from mha_exp_level2_bw.exp2_5.resized import warm_start


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON so monitoring never observes partial files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(torch: Any, path: Path, model: Any, metadata: dict[str, Any]) -> None:
    """Save a complete, size-labelled weights-only checkpoint atomically."""
    temporary = path.with_suffix(".tmp")
    torch.save({**metadata, "model_state_dict": {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }}, temporary)
    os.replace(temporary, path)


def expert_action(grounded: Any, spec: Any) -> int:
    """Execute the direct Transfer controller solely to generate offline demonstrations."""
    arm = int(grounded.arm_location[1:])
    destination = int((spec.source if grounded.held_block is None else spec.destination)[1:])
    if arm != destination:
        return 3 if arm < destination else 2
    return 0 if grounded.held_block is None else 1


def evaluate(
    torch: Any, model: Any, dimensions: dict[str, int], *, seed_start: int,
    singles: int, sequences: int, sequence_length: int = 10,
) -> dict[str, Any]:
    """Evaluate independent single transfers and evolving ten-transfer sequences."""
    environment = BlocksWorldEnv(**dimensions, symbolic=False)
    environment.expose_snapshot = True
    model.eval()
    records = []
    try:
        for index in range(singles + sequences):
            seed = seed_start + index
            observation, _ = environment.reset(seed=seed)
            length = 1 if index < singles else sequence_length
            results = []
            for transfer_index in range(length):
                grounded = ground_observation(observation, **dimensions)
                candidates = enumerate_transfer_targets(grounded.facts)
                if not candidates:
                    raise RuntimeError("Evaluation state has no legal Transfer.")
                spec = candidates[(seed + transfer_index) % len(candidates)]
                result, observation = evaluate_transfer(
                    torch, model, environment, observation, spec, seed, **dimensions,
                )
                if result.illegal_action:
                    raise RuntimeError("Legal-masked evaluation executed an illegal action.")
                results.append(asdict(result))
                if not result.success:
                    break
            records.append({"seed": seed, "requested": length, "results": results,
                            "success": len(results) == length and all(r["success"] for r in results)})
    finally:
        environment.close()
        model.train()
    successful = sum(result["success"] for row in records for result in row["results"])
    requested = singles + sequences * sequence_length
    return {
        "single_successes": sum(row["success"] for row in records[:singles]), "singles": singles,
        "sequence_successes": sum(row["success"] for row in records[singles:]), "sequences": sequences,
        "successful_transfers": successful, "requested_transfers": requested,
        "executed_transfers": sum(len(row["results"]) for row in records),
        "actions": sum(len(result["actions"]) for row in records for result in row["results"]),
        "records": records,
    }


def collect_demonstrations(
    replay: SelectedReplayBuffer, dimensions: dict[str, int], rng: Any, minimum_steps: int,
) -> dict[str, int]:
    """Collect complete expert episodes, resetting after ten evolving transfers."""
    environment = BlocksWorldEnv(**dimensions, symbolic=False)
    environment.expose_snapshot = True
    pending: deque[RawTransition] = deque()
    steps = episodes = resets = 0
    grounded = None
    try:
        while steps < minimum_steps:
            if grounded is None or episodes % 10 == 0:
                observation, _ = environment.reset(seed=8_000_000 + resets)
                resets += 1
                grounded = ground_observation(observation, **dimensions)
            candidates = enumerate_transfer_targets(grounded.facts)
            spec = candidates[int(rng.integers(len(candidates)))]
            phase = transfer_phase(grounded, spec)
            pending.clear()
            for _ in range(MAX_POLICY_STEPS):
                state = condition_observation(grounded.observation, spec, **dimensions)
                action = expert_action(grounded, spec)
                if action not in legal_action_indices(grounded.observation, **dimensions):
                    raise RuntimeError("Expert proposed an illegal action.")
                observation, _, _, _, info = environment.step(action)
                if not info["snapshot"].legal:
                    raise RuntimeError("Demonstration executed an illegal action.")
                grounded = ground_observation(observation, **dimensions)
                reward, outcome, phase = transition_reward(grounded, spec, phase, False)
                terminal = outcome is not None
                pending.append(RawTransition(
                    state, action, reward, condition_observation(observation, spec, **dimensions),
                    terminal, demonstration=True,
                ))
                for transition in emit_n_step(pending, 3, flush=terminal):
                    replay.append(transition)
                steps += 1
                if terminal:
                    episodes += 1
                    break
            else:
                raise RuntimeError("Direct Transfer expert failed within 32 actions.")
    finally:
        environment.close()
    return {"steps": steps, "episodes": episodes, "resets": resets}


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run bounded GPU fine-tuning, retain the best model, then assess it once."""
    import torch

    if str(torch.__version__) != "2.14.0+cu130" or not torch.cuda.is_available():
        raise RuntimeError("Training requires EC2 CUDA and exact Torch 2.14.0+cu130.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    dimensions = {"table_len": args.columns, "num_blocks": args.blocks}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress: dict[str, Any] = {"status": "initializing", "environment_steps": 0, "optimizer_steps": 0}

    def publish(status: str, **values: Any) -> None:
        progress.update(values, status=status, elapsed_seconds=time.monotonic() - started, updated_unix=time.time())
        write_json(args.output / "progress.json", progress)
        print(json.dumps(progress, allow_nan=False), flush=True)

    try:
        model, provenance = warm_start(torch, args.source, **dimensions, device="cuda")
        metadata = {"format_version": "resized-transfer-v1", "architecture": POLICY_ARCHITECTURE,
                    **dimensions, "n_actions": 4, "model_input_shape": [args.blocks + 4, args.columns, args.blocks],
                    "observation_shape": [args.blocks + 2, args.columns, args.blocks], "provenance": provenance}
        protocol = {**metadata, "seed": args.seed, "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(0), "algorithm": VARIANT_DQFD_LITE,
                    "max_environment_steps": args.steps, "demonstration_steps_minimum": args.demonstrations,
                    "warm_updates": args.warm_updates, "learning_rate": args.learning_rate,
                    "epsilon_start": 0.1, "epsilon_end": 0.02, "epsilon_decay_steps": 20_000,
                    "evaluation_interval": args.eval_interval, "target_sync_updates": 500,
                    "max_actions_per_transfer": MAX_POLICY_STEPS, "legal_masked_training": True,
                    "selection": {"seed_start": 9_000_000, "singles": 100, "sequences": 20, "length": 10},
                    "assessment": {"seed_start": 10_000_000, "singles": 1000, "sequences": 100, "length": 10},
                    "early_stop": "Two consecutive perfect selection suites after at least 5000 environment steps",
                    "acceptance": "At least 99% singles, 99% requested transfers and 95% complete sequences",
                    "training_seed_start": 7_000_000, "demonstration_seed_start": 8_000_000,
                    "evaluation_tasks_excluded": "All 2-4-BW seeds 23000 through 23049",
                    "resume": "Weights-only restart into a new directory; no exact replay/optimizer resume"}
        write_json(args.output / "protocol.json", protocol)
        publish("baseline")
        best_score = None
        best_update = 0
        perfect_streak = 0
        history = []

        def assess_selection() -> bool:
            nonlocal best_score, best_update, perfect_streak
            evaluation = evaluate(torch, model, dimensions, seed_start=9_000_000, singles=100, sequences=20)
            summary = {key: value for key, value in evaluation.items() if key != "records"}
            updates = progress["optimizer_steps"]
            write_json(args.output / f"selection-{updates:07d}.json", evaluation)
            score = (summary["successful_transfers"], summary["sequence_successes"], -summary["actions"])
            if best_score is None or score > best_score:
                best_score, best_update = score, updates
                save_checkpoint(torch, args.output / "best.pt", model, {
                    **metadata, "optimizer_steps": updates, "environment_steps": progress["environment_steps"],
                    "selection": summary,
                })
            save_checkpoint(torch, args.output / "latest.pt", model, {
                **metadata, "optimizer_steps": updates, "environment_steps": progress["environment_steps"],
            })
            perfect_streak = perfect_streak + 1 if summary["successful_transfers"] == summary["requested_transfers"] else 0
            history.append({"optimizer_steps": updates, "environment_steps": progress["environment_steps"], **summary})
            write_json(args.output / "selection-history.json", history)
            publish("training", selection=summary, best_optimizer_steps=best_update)
            return perfect_streak >= 2 and progress["environment_steps"] >= 5000

        assess_selection()
        replay = SelectedReplayBuffer(50_000 + args.demonstrations + MAX_POLICY_STEPS,
                                      prioritized=True, protected_capacity=args.demonstrations + MAX_POLICY_STEPS,
                                      input_shape=(args.blocks + 4, args.columns, args.blocks))
        publish("demonstrations")
        demos = collect_demonstrations(replay, dimensions, rng, args.demonstrations)
        write_json(args.output / "demonstrations.json", demos)
        # Close the small unused protection tail; sampling indices must remain contiguous.
        replay.protected_capacity = len(replay)
        replay.next_index = len(replay)
        target = deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        def update() -> None:
            beta = 0.4 + 0.6 * min(1.0, progress["environment_steps"] / args.steps)
            loss, margin = _optimize(torch, model, target, optimizer, replay, rng, "cuda",
                                     CONFIGS[VARIANT_DQFD_LITE], beta, mask_legal_actions=True)
            progress["optimizer_steps"] += 1
            count = progress["optimizer_steps"]
            if count % 500 == 0:
                target.load_state_dict(model.state_dict())
            if count % 100 == 0:
                publish("training", loss=loss, margin_loss=margin)

        publish("warm-up", demonstrations=demos)
        for _ in range(args.warm_updates):
            update()
            if progress["optimizer_steps"] in (100, 500):
                assess_selection()
        if args.warm_updates not in (0, 100, 500):
            assess_selection()
        environment = BlocksWorldEnv(**dimensions, symbolic=False)
        environment.expose_snapshot = True
        pending: deque[RawTransition] = deque()
        grounded = None
        resets = episodes = successes = 0
        next_evaluation = min(1000, args.eval_interval)
        stop = False
        try:
            while progress["environment_steps"] < args.steps and not stop:
                if grounded is None or episodes % 10 == 0:
                    observation, _ = environment.reset(seed=7_000_000 + resets)
                    resets += 1
                    grounded = ground_observation(observation, **dimensions)
                candidates = enumerate_transfer_targets(grounded.facts)
                spec = candidates[int(rng.integers(len(candidates)))]
                phase = transfer_phase(grounded, spec)
                pending.clear()
                for episode_step in range(1, MAX_POLICY_STEPS + 1):
                    state = condition_observation(grounded.observation, spec, **dimensions)
                    epsilon = 0.1 - 0.08 * min(1.0, progress["environment_steps"] / 20_000)
                    if rng.random() < epsilon:
                        action = int(rng.choice(legal_action_indices(grounded.observation, **dimensions)))
                    else:
                        action, _, _ = greedy_inference(torch, model, grounded.observation, spec, **dimensions)
                    observation, _, _, _, info = environment.step(action)
                    if not info["snapshot"].legal:
                        raise RuntimeError("Legal-masked training executed an illegal action.")
                    grounded = ground_observation(observation, **dimensions)
                    reward, outcome, phase = transition_reward(grounded, spec, phase, False)
                    progress["environment_steps"] += 1
                    terminal = outcome is not None or episode_step == MAX_POLICY_STEPS or progress["environment_steps"] == args.steps
                    pending.append(RawTransition(state, action, reward,
                                                  condition_observation(observation, spec, **dimensions), terminal))
                    for transition in emit_n_step(pending, 3, flush=terminal):
                        replay.append(transition)
                        update()
                    if terminal:
                        episodes += 1
                        successes += outcome == "succeeded"
                        progress.update(training_episodes=episodes, training_successes=successes, training_resets=resets)
                        if outcome != "succeeded":
                            grounded = None
                        break
                if progress["environment_steps"] >= next_evaluation or progress["environment_steps"] == args.steps:
                    stop = assess_selection()
                    next_evaluation = (progress["environment_steps"] // args.eval_interval + 1) * args.eval_interval
        finally:
            environment.close()
        publish("assessment")
        selected = torch.load(args.output / "best.pt", map_location="cuda", weights_only=True)
        model.load_state_dict(selected["model_state_dict"], strict=True)
        assessment = evaluate(torch, model, dimensions, seed_start=10_000_000, singles=1000, sequences=100)
        write_json(args.output / "assessment.json", assessment)
        accepted = (assessment["single_successes"] >= 990 and assessment["sequence_successes"] >= 95
                    and assessment["successful_transfers"] >= 1980)
        save_checkpoint(torch, args.output / "transfer-policy.pt", model, {**selected, "assessment_accepted": accepted})
        report = {"status": "completed", "accepted": accepted, **dimensions,
                  "environment_steps": progress["environment_steps"], "optimizer_steps": progress["optimizer_steps"],
                  "selected_optimizer_steps": best_update, "early_stopped": stop,
                  "assessment": {key: value for key, value in assessment.items() if key != "records"},
                  "checkpoint_sha256": file_sha256(args.output / "transfer-policy.pt"),
                  "elapsed_seconds": time.monotonic() - started}
        write_json(args.output / "report.json", report)
        publish("completed", report=report)
        return report
    except BaseException as exc:
        write_json(args.output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), **progress})
        publish("failed", error=repr(exc))
        raise


def main() -> None:
    """Parse an explicit size and new output directory; never overwrite previous attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=artifact_paths()[1])
    parser.add_argument("--seed", type=int, default=2505)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--demonstrations", type=int, default=8000)
    parser.add_argument("--warm-updates", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    if (args.columns, args.blocks) not in ((4, 6), (7, 12)):
        parser.error("This preparation supports only the requested 4x6 and 7x12 models.")
    if args.steps <= 0 or args.demonstrations < 128 or args.warm_updates < 0 or args.eval_interval <= 0 or args.learning_rate <= 0:
        parser.error("Training budgets and learning rate must be positive; at least 128 demonstration steps are required.")
    train(args)


if __name__ == "__main__":
    main()
