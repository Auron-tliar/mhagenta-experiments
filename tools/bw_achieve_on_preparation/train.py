"""Offline pilot training for a Transfer-initialized, atomic AchieveOn policy.

Run as a module from the workspace root. This intentionally bypasses Docker
and MHAgentA messaging for policy preparation, not for the evaluated 2-6 agent.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import torch
from mha_exp_level2_bw.achieve_on import policy as shared_policy
from mha_exp_level2_bw.achieve_on.demonstrations import demonstration_start

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import GoalSpec, TransferSpec
from mha_exp_level2_bw.exp2_5.grounding import ground_observation, transfer_succeeded
from mha_exp_level2_bw.exp2_5.planning import PlanningService
from mha_exp_level2_bw.exp2_5.policy import (
    artifact_paths, file_sha256, greedy_inference, legal_action_indices,
    load_policy_checkpoint,
)
from tools.bw_achieve_on_preparation.policy import (
    checkpoint_payload, condition_goal, goal_succeeded,
    infer, warm_start,
)

SEED_BASES = {"demonstration": 510_000_000, "training": 520_000_000,
              "selection": 530_000_000, "assessment": 540_000_000}


from mha_exp_level2_bw.achieve_on.learning import Config, Replay, Transition, curriculum_limit, emit, legal_mask, optimize


def make_environment() -> BlocksWorldEnv:
    """Create a local numeric environment with actual action legality exposed."""
    environment = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    environment.expose_snapshot = True
    return environment


def case(environment: Any, split: str, index: int) -> tuple[np.ndarray, GoalSpec, int]:
    """Choose a nontrivial goal without filtering on any policy's success."""
    if split not in SEED_BASES or not 0 <= index < 1_000_000:
        raise ValueError("Case index is outside the declared split window.")
    seed = SEED_BASES[split] + index
    observation, _ = environment.reset(seed=seed)
    facts = ground_observation(observation).facts
    goals = [GoalSpec(f"b{top}", f"b{bottom}") for top in range(8) for bottom in range(8)
             if top != bottom and f"on(b{top},b{bottom})" not in facts]
    return observation, goals[index % len(goals)], seed


def planner() -> PlanningService:
    """Use 2-5's validated symbolic planner only for teacher/baseline runs."""
    domain = artifact_paths()[0].parents[2] / "blocksworld-transfer-domain.pddl"
    return PlanningService(domain_path=domain, blocks=[f"b{i}" for i in range(8)],
                           locations=[f"t{i}" for i in range(5)])


def teacher_episode(environment: Any, observation: np.ndarray, goal: GoalSpec,
                    model: Any, service: PlanningService, config: Config) -> tuple[list[Transition], dict]:
    """Collect final-goal-conditioned atomic labels from planner plus Transfer."""
    outcome = service.solve(set(ground_observation(observation).facts), goal, "achieve-on-teacher")
    evidence = {"goal": goal.as_dict(), "planner": outcome.engine, "plan_accepted": outcome.accepted,
                "success": False, "actions": [], "transfers": outcome.actions}
    if not outcome.accepted:
        return [], evidence
    rows = []
    for mapping in outcome.actions:
        spec = TransferSpec.from_mapping(mapping)
        for _ in range(32):
            action, _, _ = greedy_inference(torch, model, observation, spec)
            successor, _, terminated, truncated, info = environment.step(action)
            if not info["snapshot"].legal:
                raise RuntimeError("Frozen teacher selected an illegal action.")
            success = goal_succeeded(successor, goal)
            terminal = success or len(rows) + 1 >= config.action_cap or terminated or truncated
            rows.append(Transition(condition_goal(observation, goal), action,
                                   1.0 if success else (-1.0 if terminal else -0.01),
                                   condition_goal(successor, goal), terminal, legal_mask(successor)))
            evidence["actions"].append(action)
            observation = successor
            if terminal:
                evidence["success"] = success
                return rows, evidence
            if transfer_succeeded(ground_observation(observation).facts, spec):
                break
        else:
            return [], evidence
    return rows, evidence


def replay_teacher(environment: Any, observation: np.ndarray, goal: GoalSpec,
                   evidence: dict, config: Config) -> tuple[list[Transition], dict]:
    """Reconstruct fixed teacher labels from a recorded trace, checking each action."""
    if evidence["goal"] != goal.as_dict() or len(evidence["actions"]) > config.action_cap:
        raise ValueError("Cached teacher goal or action cap differs.")
    rows = []
    for action in evidence["actions"]:
        successor, _, terminated, truncated, info = environment.step(action)
        if not info["snapshot"].legal:
            raise ValueError("Cached teacher action is illegal.")
        success = goal_succeeded(successor, goal)
        terminal = success or len(rows) + 1 >= config.action_cap or terminated or truncated
        rows.append(Transition(condition_goal(observation, goal), action,
                               1.0 if success else (-1.0 if terminal else -0.01),
                               condition_goal(successor, goal), terminal, legal_mask(successor)))
        observation = successor
        if terminal and len(rows) != len(evidence["actions"]):
            raise ValueError("Cached teacher continued after a terminal action.")
    if bool(evidence["success"]) != goal_succeeded(observation, goal):
        raise ValueError("Cached teacher outcome does not replay.")
    return rows, deepcopy(evidence)


def evaluate(model: Any, config: Config, split: str = "selection", count: int | None = None,
             start_index: int = 0) -> dict:
    """Evaluate direct atomic execution with no teacher or planner fallback."""
    environment = make_environment()
    results = []
    try:
        for index in range(start_index, start_index + (count if count is not None else config.selection_cases)):
            observation, goal, seed = case(environment, split, index)
            actions = []
            success = False
            seen = {observation.tobytes(): 0}
            first_repeat = None
            for _ in range(config.action_cap):
                action, _ = infer(torch, model, observation, goal)
                observation, _, terminated, truncated, info = environment.step(action)
                actions.append(action)
                key = observation.tobytes()
                if key in seen and first_repeat is None:
                    first_repeat = {"entry_action": seen[key], "cycle_length": len(actions) - seen[key]}
                seen.setdefault(key, len(actions))
                if not info["snapshot"].legal:
                    raise RuntimeError("Candidate selected an illegal action.")
                success = goal_succeeded(observation, goal)
                if success or terminated or truncated:
                    break
            results.append({"seed": seed, "goal": goal.as_dict(), "success": success, "actions": actions,
                            "first_repeat": first_repeat})
    finally:
        environment.close()
    return {"split": split, "cases": len(results), "successes": sum(row["success"] for row in results),
            "results": results}


def save_model(path: Path, model: Any, provenance: dict, steps: int, updates: int) -> None:
    """Write distinct checkpoints; never overwrite the frozen source artifact."""
    if path.exists():
        raise FileExistsError(path)
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError("Cannot save nonfinite policy weights.")
    payload = checkpoint_payload(model, provenance)
    payload.update(status="candidate-unqualified", environment_steps=steps, optimizer_steps=updates)
    torch.save(payload, path)


class TrainingDeadline(Exception):
    """The declared wall budget ended; retain evidence without retrying."""


def train(output: Path, config: Config, device: str,
          architecture: str = shared_policy.ARCHITECTURE, max_wall_seconds: float | None = None,
          teacher_cache: Path | None = None) -> None:
    """Run the standalone pilot, retaining initialization, baseline, and selection."""
    if output.exists():
        raise FileExistsError(output)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda.")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device == "cuda" and str(torch.__version__) != "2.14.0+cu130":
        raise RuntimeError("CUDA preparation requires Torch 2.14.0+cu130.")
    rng = np.random.default_rng(config.seed)
    if max_wall_seconds is not None and max_wall_seconds <= 0:
        raise ValueError("Wall budget must be positive.")
    model, provenance = warm_start(torch, architecture)
    cached = json.loads(teacher_cache.read_text()) if teacher_cache is not None else None
    if cached is not None:
        if (cached["status"] != "completed-pilot-unqualified"
                or cached["method"] != "dqfd-balanced-static-recovery-v2"
                or cached["provenance"]["transfer_sha256"] != provenance["transfer_sha256"]
                or cached["seed_windows"] != SEED_BASES
                or any(cached["configuration"][key] != getattr(config, key)
                       for key in ("demonstrations", "selection_cases", "action_cap"))
                or len(cached["demonstrations"]) != config.demonstrations
                or len(cached["baseline"]) != config.selection_cases):
            raise ValueError("Teacher cache does not match the declared fixed data recipe.")
    model.to(device)
    target = deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=False)
    source = {str(path.relative_to(Path.cwd())): file_sha256(path)
              for folder in [Path(__file__).parent, Path(shared_policy.__file__).parent, artifact_paths()[0].parents[2]]
              for path in sorted(folder.glob("*.py"))}
    report = {"configuration": asdict(config), "seed_windows": SEED_BASES,
              "provenance": provenance, "source_sha256": source, "device": device,
              "torch_version": str(torch.__version__), "status": "running",
              "demonstrations": [], "selection": [], "baseline": [], "learning_metrics": [],
              "method": "dqfd-balanced-static-recovery-v2", "architecture": architecture,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "max_wall_seconds": max_wall_seconds}
    if teacher_cache is not None:
        report["teacher_cache"] = {"path": str(teacher_cache.resolve()), "sha256": file_sha256(teacher_cache),
                                   "replayed_actions": True, "candidate_dependent_labels": False}
    started = time.monotonic()
    report["hardware"] = {"cuda_build": torch.version.cuda,
                          "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}

    def progress(stage: str, **values: Any) -> None:
        """Publish small, atomic progress records for the external heartbeat."""
        record = {"stage": stage, "updated_at": time.time(),
                  "elapsed_seconds": time.monotonic() - started, **values}
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "progress.json")
        print(json.dumps(record), flush=True)

    def persist() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "report.json")

    persist()
    progress("initializing")
    save_model(output / "initial.pt", model, provenance, 0, 0)
    service = planner() if cached is None else None
    teacher, _ = load_policy_checkpoint(torch, artifact_paths()[1])
    environment = make_environment()
    best = (-1, -float("inf"))
    updates = 0
    step = 0

    def check_deadline() -> None:
        """Check between bounded operations, keeping the last selected model."""
        if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
            raise TrainingDeadline("Declared training wall budget reached.")

    def select(steps: int, label: str) -> None:
        nonlocal best
        check_deadline()
        progress("selection", online_steps=steps, optimizer_steps=updates, checkpoint=label)
        result = evaluate(model, config)
        result.update(environment_steps=steps, optimizer_steps=updates, checkpoint=label,
                      checkpoint_sha256=file_sha256(output / label))
        solved = {row["seed"] for row in result["results"] if row["success"]}
        previous_best = next((item for item in report["selection"]
                              if item["checkpoint"] == report.get("selected_checkpoint")), None)
        previous_solved = {row["seed"] for row in previous_best["results"] if row["success"]} if previous_best else set()
        result.update(loop_failures=sum(not row["success"] and row["first_repeat"] is not None
                                        for row in result["results"]),
                      lost_from_previous_best=len(previous_solved - solved),
                      gained_over_previous_best=len(solved - previous_solved))
        pairs = list(zip(result["results"], report["baseline"], strict=True))
        result["paired_baseline"] = {
            "baseline_successes": sum(base["success"] for _, base in pairs),
            "net_success_gain": sum(int(learner["success"]) - int(base["success"]) for learner, base in pairs),
            "joint_successes": sum(learner["success"] and base["success"] for learner, base in pairs),
            "actions_saved_on_joint_successes": sum(
                len(base["actions"]) - len(learner["actions"])
                for learner, base in pairs if learner["success"] and base["success"]),
        }
        report["selection"].append(result)
        score = (result["successes"], -sum(len(row["actions"]) for row in result["results"] if row["success"]))
        if score > best:
            best = score
            report["selected_checkpoint"] = label
        persist()
        print(json.dumps({"stage": "selection", "step": steps, "successes": result["successes"], "cases": result["cases"]}), flush=True)

    try:
        # Establish the compound baseline on the identical selection cases.
        for index in range(config.selection_cases):
            check_deadline()
            observation, goal, seed = case(environment, "selection", index)
            if cached is None:
                _, evidence = teacher_episode(environment, observation, goal, teacher, service, config)
            else:
                if cached["baseline"][index]["seed"] != seed:
                    raise ValueError("Cached baseline seed differs.")
                _, evidence = replay_teacher(environment, observation, goal, cached["baseline"][index], config)
            evidence["seed"] = seed
            report["baseline"].append(evidence)
            progress("baseline", completed=index + 1, total=config.selection_cases)
        select(0, "initial.pt")
        demonstrations = []
        difficulties = []
        for index in range(config.demonstrations):
            check_deadline()
            observation, goal, seed = case(environment, "demonstration", index)
            observation, goal, metadata = demonstration_start(environment, observation, seed, index)
            if cached is None:
                raw, evidence = teacher_episode(environment, observation, goal, teacher, service, config)
            else:
                item = cached["demonstrations"][index]
                if item["seed"] != seed or any(item[key] != value for key, value in metadata.items()):
                    raise ValueError("Cached demonstration seed or setup differs.")
                raw, evidence = replay_teacher(environment, observation, goal, item, config)
            evidence.update(seed=seed, **metadata)
            report["demonstrations"].append(evidence)
            progress("demonstrations", completed=index + 1, total=config.demonstrations)
            if evidence["success"]:
                demonstrations.extend(emit(deque(raw), config, True))
                difficulties.extend([metadata["difficulty"]] * len(raw))
            if (index + 1) % 20 == 0:
                persist()
                print(json.dumps({"stage": "demonstrations", "attempted": index + 1}), flush=True)
        replay = Replay(config.capacity, demonstrations, difficulties)
        report["demonstration_transitions"] = replay.protected
        metrics = {}
        for warm_step in range(config.warm_updates):
            check_deadline()
            loss = optimize(model, target, optimizer, replay, rng, config, device, 0.4,
                            metrics, curriculum_limit(warm_step, config.warm_updates))
            updates += 1
            if updates % 100 == 0:
                report["learning_metrics"].append({"optimizer_steps": updates, "online_steps": 0, **metrics})
                progress("warm-up", optimizer_steps=updates, total=config.warm_updates, loss=loss, **metrics)
            if updates % config.target_sync == 0:
                target.load_state_dict(model.state_dict())
        save_model(output / "warm.pt", model, provenance, 0, updates)
        select(0, "warm.pt")
        episode = 0
        observation, goal, seed = case(environment, "training", episode)
        pending = deque()
        episode_actions = 0
        report["online_episodes"] = []
        for next_step in range(1, config.online_steps + 1):
            check_deadline()
            step = next_step
            epsilon = 0.3 + (0.05 - 0.3) * min(step / (config.online_steps * 0.8), 1.0)
            action, _ = infer(torch, model, observation, goal)
            if rng.random() < epsilon:
                action = int(rng.choice(legal_action_indices(observation)))
            successor, _, terminated, truncated, info = environment.step(action)
            if not info["snapshot"].legal:
                raise RuntimeError("Online candidate selected an illegal action.")
            episode_actions += 1
            success = goal_succeeded(successor, goal)
            terminal = success or episode_actions >= config.action_cap or terminated or truncated
            budget_boundary = step == config.online_steps
            pending.append(Transition(condition_goal(observation, goal), action,
                                      1.0 if success else (-1.0 if terminal else -0.01),
                                      condition_goal(successor, goal), terminal, legal_mask(successor)))
            for row in emit(pending, config, terminal or budget_boundary):
                replay.append(row)
            if len(replay.rows) > replay.protected:
                loss = optimize(model, target, optimizer, replay, rng, config, device,
                                0.4 + 0.6 * step / config.online_steps, metrics)
                updates += 1
                if updates % config.target_sync == 0:
                    target.load_state_dict(model.state_dict())
            observation = successor
            if terminal or budget_boundary:
                report["online_episodes"].append({"seed": seed, "goal": goal.as_dict(), "success": success,
                                                  "actions": episode_actions, "budget_truncated": budget_boundary and not terminal})
                episode += 1
                if not budget_boundary:
                    observation, goal, seed = case(environment, "training", episode)
                    episode_actions = 0
            if step % 500 == 0:
                report["learning_metrics"].append({"optimizer_steps": updates, "online_steps": step, **metrics})
                progress("online", online_steps=step, total=config.online_steps,
                         optimizer_steps=updates, episodes=episode, loss=loss, **metrics)
            if step % config.selection_interval == 0 or budget_boundary:
                label = f"step-{step:07d}.pt"
                save_model(output / label, model, provenance, step, updates)
                select(step, label)
        report.update(status="completed-pilot-unqualified", optimizer_steps=updates,
                      online_steps=config.online_steps, assessment_used=False)
        persist()
        progress("completed", online_steps=config.online_steps, optimizer_steps=updates)
    except TrainingDeadline as error:
        save_model(output / "time-limit.pt", model, provenance, step, updates)
        report.update(status="time-limit-unqualified", error=str(error), optimizer_steps=updates,
                      online_steps=step, assessment_used=False, exact_resume_available=False)
        persist()
        progress("time-limit", online_steps=step, optimizer_steps=updates)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        progress("failed", error=report["error"])
        raise
    finally:
        environment.close()


def main() -> None:
    """Launch only an explicitly requested offline training budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, default=2605)
    parser.add_argument("--online-steps", type=int, default=50_000)
    parser.add_argument("--demonstrations", type=int, default=600)
    parser.add_argument("--warm-updates", type=int, default=6000)
    parser.add_argument("--selection-cases", type=int, default=100)
    parser.add_argument("--selection-interval", type=int, default=5_000)
    parser.add_argument("--architecture", choices=("mlp", "conv3d"), default="mlp")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--teacher-cache", type=Path)
    args = parser.parse_args()
    config = Config(seed=args.seed, online_steps=args.online_steps, demonstrations=args.demonstrations,
                    warm_updates=args.warm_updates, selection_cases=args.selection_cases,
                    selection_interval=args.selection_interval)
    def stop(signum: int, frame: Any) -> None:
        """Preserve failure evidence when a monitored container is stopped."""
        raise KeyboardInterrupt(f"Received signal {signum}; no optimizer/replay resume is available.")

    signal.signal(signal.SIGTERM, stop)
    architecture = shared_policy.CONV3D_ARCHITECTURE if args.architecture == "conv3d" else shared_policy.ARCHITECTURE
    train(args.output, config, args.device, architecture, args.max_wall_seconds, args.teacher_cache)


if __name__ == "__main__":
    main()
