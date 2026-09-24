"""Randomized policy training with held-out world splits and native-case gates."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from cases import run_case
from mha_exp_level2_cr.exp2_5.contracts import activity_action_bound
from mha_exp_level2_cr.exp2_5.policy import (
    POLICY_FILENAMES,
    PolicyId,
    checkpoint_data,
    file_sha256,
    load_policy_checkpoint,
)
from protocol import POLICY_SPEC_BY_ID
from replay import optimize
from resource_cases import ResourceCases
from train import (
    CONFIG,
    _atomic_json,
    _atomic_torch_save,
    _emit_event,
    _load_state,
    _new_state,
    _rng_state,
)

RANDOMIZED_CONFIGS = {
    PolicyId.NAVIGATE_TO: {
        "version": 11, "demo_steps": 4_000, "pretraining_steps": 3_000,
        "online_steps": 40_000, "training_cases": 960,
        "validation_cases": 240, "test_cases": 600, "interval": 5_000,
        "minimum_steps": 10_000, "patience": 3, "epsilon_steps": 32_000,
        "beta_steps": 40_000, "optimization_interval": 4,
    },
    PolicyId.GET_RESOURCE: {
        "version": 12, "demo_steps": 6_000, "pretraining_steps": 2_000,
        "online_steps": 60_000, "training_cases": 960,
        "validation_cases": 600, "test_cases": 600, "interval": 5_000,
        "minimum_steps": 10_000, "patience": 3, "epsilon_steps": 40_000,
        "beta_steps": 60_000, "optimization_interval": 4,
    },
}
for _policy in (PolicyId.EXPLORE, PolicyId.EAT_TARGET, PolicyId.EAT_COW):
    RANDOMIZED_CONFIGS[_policy] = {
        "version": 30, "demo_steps": 4_000, "pretraining_steps": 1_000,
        "online_steps": 20_000, "training_cases": 960,
        "validation_cases": 240, "test_cases": 600, "interval": 5_000,
        "minimum_steps": 5_000, "patience": 2, "epsilon_steps": 16_000,
        "beta_steps": 20_000, "optimization_interval": 4,
    }
RANDOMIZED_CONFIGS[PolicyId.EAT_TARGET].update(
    version=33, target_tracking="observable_joint_association_v2",
    teacher="observable_visibility_tie_v1",
)
RANDOMIZED_CONFIGS[PolicyId.EAT_COW].update(
    version=34, activity_action_bound=96, teacher="masked_discovery_visibility_pursuit_v3",
)
RESOURCE_CONFIG = RANDOMIZED_CONFIGS[PolicyId.GET_RESOURCE]


def randomized_cases(root: Path, split: str, policy: PolicyId) -> Any:
    """Keep accepted resource/navigation protocols and use native remaining cases."""
    if policy in {PolicyId.NAVIGATE_TO, PolicyId.GET_RESOURCE}:
        return ResourceCases(root, split, policy)
    from remaining_evaluation import RemainingCases
    return RemainingCases(root, split, policy)


def randomized_episode(
    torch: Any,
    model: Any,
    cases: ResourceCases,
    index: int,
    device: str,
    *,
    expert: bool = False,
    epsilon: float = 0.0,
    execution_index: int | None = None,
) -> tuple[Any, bool, dict[str, Any]]:
    """Run one immutable randomized case and report construction and execution time."""

    started = perf_counter()
    env, case, descriptor = cases.get(index)
    constructed = perf_counter()
    rows, success, stats = run_case(
        torch, model, env, case, expert=expert, epsilon=epsilon,
        device=device, seed=descriptor["seed"] + (
            index if execution_index is None else execution_index
        ),
        discount=CONFIG["discount"], horizon=CONFIG["n_step_horizon"],
    )
    stats.update(
        source=descriptor["source"], kind=descriptor.get("kind", case.target_kind),
        world_seed=descriptor["seed"], distance=descriptor["distance"],
        generation_rejections=descriptor["generation_rejections"],
        visible_stations=descriptor.get("visible_stations", []),
        distance_band="-".join(str(value) for value in descriptor["bounds"]),
        direction=descriptor.get("direction"), variant=descriptor.get("variant"),
        case_seconds=constructed - started, execution_seconds=perf_counter() - constructed,
    )
    stats["stratum"] = ":".join((stats["source"], stats["kind"],
                                  stats["distance_band"], str(stats["direction"]),
                                  str(stats["variant"])))
    if stats.get("expert_route_blocked"):
        _emit_event(
            "expert_route_blocked", index=index, world_seed=descriptor["seed"],
            retained_transitions=len(rows), source=descriptor["source"], kind=descriptor.get("kind", case.target_kind),
        )
    return rows, success, stats


def _group(result: dict[str, Any], name: str, key: str, success: bool) -> None:
    """Record one attempted outcome in an evaluation subgroup."""

    record = result[name].setdefault(key, {"attempted": 0, "succeeded": 0})
    record["attempted"] += 1
    record["succeeded"] += int(success)


def _group_passes(group: dict[str, dict[str, int]]) -> bool:
    """Require at least ninety percent success in every populated subgroup."""

    return bool(group) and all(row["succeeded"] / row["attempted"] >= .90 for row in group.values())


def evaluate_randomized(
    torch: Any,
    model: Any,
    cases: ResourceCases,
    device: str,
    count: int,
) -> dict[str, Any]:
    """Evaluate fixed cases without replacement and apply policy-specific gates."""

    policy = cases.policy
    result: dict[str, Any] = {
        "split": cases.split, "attempted": count, "succeeded": 0,
        "by_kind": {}, "by_source": {}, "by_distance": {}, "by_direction": {},
        "by_variant": {}, "episodes": [], "harness_errors": 0,
        "illegal_actions": 0, "lethal_actions": 0, "movement_blocked": 0,
        "environment_terminal_failures": 0, "nontrivial_attempted": 0,
        "nontrivial_succeeded": 0,
    }
    for index in range(count):
        rows, success, stats = randomized_episode(torch, model, cases, index, device)
        result["succeeded"] += int(success)
        result["nontrivial_attempted"] += int(stats["distance"] > 0)
        result["nontrivial_succeeded"] += int(success and stats["distance"] > 0)
        for field in (
            "illegal_actions", "lethal_actions", "environment_terminal_failures",
            "movement_blocked",
        ):
            result[field] += stats[field]
        for group, key in (
            ("by_kind", "kind"), ("by_source", "source"),
            ("by_distance", "distance_band"), ("by_direction", "direction"),
            ("by_variant", "variant"),
        ):
            if stats[key] is not None:
                _group(result, group, str(stats[key]), success)
        result["episodes"].append({"index": index, "success": success, "actions": len(rows), **stats})
        if (index + 1) % 24 == 0:
            _emit_event(
                "randomized_evaluation_progress", policy_id=policy.value, split=cases.split,
                completed=index + 1, total=count, succeeded=result["succeeded"],
            )
    result["mean_cost"] = float(np.mean([
        item["actions"] if item["success"] else activity_action_bound(policy.value) for item in result["episodes"]
    ]))
    clean = not any(result[field] for field in (
        "harness_errors", "illegal_actions", "lethal_actions", "environment_terminal_failures",
    ))
    common = (
        result["succeeded"] / count >= .95 and clean
        and _group_passes(result["by_source"])
        and _group_passes(result["by_distance"])
    )
    if policy in {PolicyId.EXPLORE, PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
        structure = all(not row["success"] or policy is not PolicyId.EAT_COW or all(
            row[key] > 0 for key in ("search_actions", "acquisition_events", "pursuit_actions", "do_actions")
        ) for row in result["episodes"])
        result["successful_cases_structurally_complete"] = structure
        result["passed"] = common and structure and _group_passes(result["by_direction"]) and _group_passes(result["by_variant"])
    elif policy is PolicyId.NAVIGATE_TO:
        result["passed"] = common and _group_passes(result["by_direction"]) and _group_passes(result["by_variant"])
    else:
        nontrivial_rate = result["nontrivial_succeeded"] / result["nontrivial_attempted"]
        result["passed"] = common and nontrivial_rate >= .90 and _group_passes(result["by_kind"])
    return result


def evaluate_resource(
    torch: Any, model: Any, root: Path, split: str, device: str, count: int,
) -> dict[str, Any]:
    """Evaluate GetResource through the randomized fixed-case protocol."""

    return evaluate_randomized(
        torch, model, ResourceCases(root, split, PolicyId.GET_RESOURCE), device, count,
    )


def resource_episode(
    torch: Any, model: Any, cases: ResourceCases, index: int, device: str,
    *, expert: bool = False, epsilon: float = 0.0,
) -> tuple[Any, bool, dict[str, Any]]:
    """Run one randomized GetResource episode."""

    if cases.policy is not PolicyId.GET_RESOURCE:
        raise ValueError("Resource episodes require GetResource cases.")
    return randomized_episode(
        torch, model, cases, index, device, expert=expert, epsilon=epsilon,
    )


def train_randomized(
    torch: Any,
    output: Path,
    policy: PolicyId,
    device: str,
    resume: bool,
    stop: Any,
    init_checkpoint: Path | None,
    *,
    diagnostic: bool = False,
) -> dict[str, Any]:
    """Train one randomized policy, select on validation, and use its test split once."""

    if policy not in RANDOMIZED_CONFIGS:
        raise ValueError("Unsupported randomized policy.")
    if not resume and policy is PolicyId.NAVIGATE_TO and init_checkpoint is not None:
        raise ValueError("NavigateTo uses fresh initialization.")
    if not resume and policy is PolicyId.GET_RESOURCE and init_checkpoint is None:
        raise ValueError("GetResource requires a selected NavigateTo initialization.")
    if init_checkpoint is not None:
        value = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        parent = PolicyId(value["policy_id"])
        expected = {
            PolicyId.GET_RESOURCE: {PolicyId.NAVIGATE_TO},
            PolicyId.EXPLORE: {PolicyId.EXPLORE},
            PolicyId.EAT_TARGET: {PolicyId.NAVIGATE_TO, PolicyId.EAT_TARGET},
            PolicyId.EAT_COW: {PolicyId.EXPLORE, PolicyId.EAT_COW},
        }
        if parent not in expected.get(policy, set()):
            raise ValueError("Initialization has the wrong policy lineage.")
    elif not resume and policy in {PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
        raise ValueError("Cow policies require their specified parent or their own provisional checkpoint.")

    config = RANDOMIZED_CONFIGS[policy]
    spec = POLICY_SPEC_BY_ID[policy]
    work = output / "work" / f"{policy.value}.pt"
    state, online, target, optimizer, replay = (
        _load_state(torch, work, spec, device) if resume and work.exists()
        else _new_state(torch, spec, device, init_checkpoint)
    )
    if "randomized_configuration" not in state:
        state.update(
            randomized_configuration=config, optimization_interval=config["optimization_interval"],
            pretraining_steps=0, case_index=0, best=None, stale_checks=0, worlds=[], outcomes={},
            source_coverage={}, kind_coverage={}, distance_coverage={},
            validation_prepared=False,
            timings={"case_seconds": 0.0, "execution_seconds": 0.0,
                     "optimization_seconds": 0.0, "validation_seconds": 0.0},
        )
    if state["randomized_configuration"] != config:
        raise ValueError("Randomized training configuration changed; use a fresh run directory.")

    train_cases = randomized_cases(output / "cases", "pilot" if diagnostic else "train", policy)
    validation_cases = randomized_cases(output / "cases", "pilot" if diagnostic else "validation", policy)
    recent: Counter[str] = Counter()
    started = perf_counter()

    def save(event: str) -> None:
        nonlocal started
        state["elapsed_seconds"] += perf_counter() - started
        _atomic_torch_save(torch, {
            "state": state, "online": online.state_dict(), "target": target.state_dict(),
            "optimizer": optimizer.state_dict(), "replay": replay.state_dict(), "rng": _rng_state(torch),
        }, work)
        _atomic_json(state, output / f"{policy.value}-progress.json")
        epsilon = (0.0 if state["phase"] in {"demonstrations", "pretraining"} else
                   max(.05, 1.0 - .95 * state["online_steps"] / config["epsilon_steps"]))
        minutes = max(state["elapsed_seconds"] / 60, 1e-9)
        _emit_event(
            event, policy_id=policy.value, phase=state["phase"], episodes=state["episodes"],
            online_steps=state["online_steps"], optimizer_steps=state["optimizer_steps"],
            demo_steps=state["demonstration_steps"], epsilon=epsilon,
            transitions_per_minute=state["online_steps"] / minutes,
            losses=state["losses"], recent=dict(recent), unique_worlds=len(state["worlds"]),
            sources=state["source_coverage"], kinds=state["kind_coverage"],
            distances=state["distance_coverage"], outcomes=state["outcomes"],
            timings=state["timings"], elapsed_seconds=state["elapsed_seconds"], best=state["best"],
        )
        recent.clear()
        started = perf_counter()

    if resume:
        _emit_event(
            "randomized_training_resumed", policy_id=policy.value, phase=state["phase"],
            case_index=state["case_index"], demo_steps=state["demonstration_steps"],
            online_steps=state["online_steps"], validation_prepared=state["validation_prepared"],
        )
    if not diagnostic and not state["validation_prepared"]:
        for index in range(config["validation_cases"]):
            validation_cases.get(index)
            if (index + 1) % 24 == 0:
                _emit_event(
                    "validation_preparation", policy_id=policy.value,
                    completed=index + 1, total=config["validation_cases"],
                )
            if stop.requested:
                save("policy_interrupted")
                return state
        state["validation_prepared"] = True
        save("validation_prepared")

    native_policy = policy in {PolicyId.EXPLORE, PolicyId.EAT_TARGET, PolicyId.EAT_COW}
    diagnostic_cases = 48 if native_policy else 24
    demo_limit = (512 if native_policy else 256) if diagnostic else config["demo_steps"]
    online_limit = 512 if diagnostic else config["online_steps"]
    interval = 512 if diagnostic else config["interval"]
    pretraining_limit = (512 if native_policy else 128) if diagnostic else config["pretraining_steps"]
    while state["phase"] in {"demonstrations", "pretraining", "online"}:
        if state["phase"] == "pretraining":
            optimized = perf_counter()
            state["losses"] = optimize(
                torch, online, target, optimizer, replay, device, .4, CONFIG,
            )
            state["timings"]["optimization_seconds"] += perf_counter() - optimized
            if not all(np.isfinite(value) for value in state["losses"].values()):
                raise FloatingPointError("Non-finite pretraining loss; restart from fresh weights.")
            state["pretraining_steps"] += 1
            state["optimizer_steps"] += 1
            if state["optimizer_steps"] % CONFIG["target_sync_optimizer_steps"] == 0:
                target.load_state_dict(online.state_dict())
            if state["pretraining_steps"] % 250 == 0 or state["pretraining_steps"] == pretraining_limit:
                if state["pretraining_steps"] == pretraining_limit:
                    state["phase"] = "online"
                save("pretraining_completed" if state["phase"] == "online" else "pretraining_progress")
            if stop.requested:
                save("policy_interrupted")
                return state
            continue
        expert = state["phase"] == "demonstrations"
        if expert and state["demonstration_steps"] >= demo_limit:
            replay.seal_demonstrations()
            state["phase"] = "pretraining"
            save("demonstrations_completed")
            continue
        epsilon = 0.0 if expert else max(
            .05, 1.0 - .95 * state["online_steps"] / config["epsilon_steps"],
        )
        case_count = diagnostic_cases if diagnostic else config["training_cases"]
        rows, success, stats = randomized_episode(
            torch, online, train_cases, state["case_index"] % case_count, device,
            expert=expert, epsilon=epsilon, execution_index=state["case_index"],
        )
        state["timings"]["case_seconds"] += stats["case_seconds"]
        state["timings"]["execution_seconds"] += stats["execution_seconds"]
        for row in rows:
            replay.append(row)
            if not expert:
                state["online_steps"] += 1
                if state["online_steps"] % config["optimization_interval"] == 0:
                    optimized = perf_counter()
                    beta = .4 + .6 * min(1.0, state["online_steps"] / config["beta_steps"])
                    state["losses"] = optimize(
                        torch, online, target, optimizer, replay, device, beta, CONFIG,
                    )
                    state["timings"]["optimization_seconds"] += perf_counter() - optimized
                    if not all(np.isfinite(value) for value in state["losses"].values()):
                        raise FloatingPointError("Non-finite loss; resume from the last durable episode.")
                    state["optimizer_steps"] += 1
                    if state["optimizer_steps"] % CONFIG["target_sync_optimizer_steps"] == 0:
                        target.load_state_dict(online.state_dict())
        if expert:
            state["demonstration_steps"] += len(rows)
        state["episodes"] += 1
        state["case_index"] += 1
        if stats["world_seed"] not in state["worlds"]:
            state["worlds"].append(stats["world_seed"])
        for field, value in (
            ("source_coverage", stats["source"]), ("kind_coverage", stats["kind"]),
            ("distance_coverage", stats["distance_band"]),
        ):
            state[field][str(value)] = state[field].get(str(value), 0) + 1
        recent.update(episodes=1, successes=int(success), actions=len(rows))
        recent.update({key: stats.get(key, 0) for key in (
            "illegal_actions", "lethal_actions", "stagnation",
            "wrong_target_collections", "generation_rejections", "movement_blocked",
        )})
        recent.update(expert_route_blocked=stats.get("expert_route_blocked", 0))
        if policy in {PolicyId.EXPLORE, PolicyId.EAT_TARGET, PolicyId.EAT_COW} and any(
            stats.get(key, 0) for key in ("illegal_actions", "lethal_actions", "environment_terminal_failures")
        ):
            state["phase"] = "failed_safety"
            state["failure_episode"] = stats
            save("policy_stopped_for_safety")
            return state
        if any(stats.get(key, 0) for key in (
            "illegal_actions", "lethal_actions", "stagnation",
            "wrong_target_collections", "movement_blocked",
        )):
            _emit_event(
                "episode_anomaly", policy_id=policy.value, split=train_cases.split,
                case_index=state["case_index"], world_seed=stats["world_seed"],
                source=stats["source"], kind=stats["kind"],
                distance=stats["distance"], success=success,
                illegal_actions=stats.get("illegal_actions", 0),
                first_illegal_action=stats.get("first_illegal_action"),
                lethal_actions=stats.get("lethal_actions", 0),
                movement_blocked=stats.get("movement_blocked", 0),
                stagnation=stats.get("stagnation", 0),
                wrong_target_collections=stats.get("wrong_target_collections", 0),
            )
        outcome = f"{state['phase']}:{'success' if success else 'failure'}"
        state["outcomes"][outcome] = state["outcomes"].get(outcome, 0) + 1

        if not expert and state["online_steps"] >= interval * (len(state["selections"]) + 1):
            evaluated = perf_counter()
            result = evaluate_randomized(
                torch, online, validation_cases, device,
                diagnostic_cases if diagnostic else config["validation_cases"],
            )
            state["timings"]["validation_seconds"] += perf_counter() - evaluated
            step = state["online_steps"]
            score = [result["succeeded"] / result["attempted"], -result["mean_cost"], -step]
            path = output / "boundaries" / f"{policy.value}-{step}.pt"
            _atomic_torch_save(torch, checkpoint_data(policy, {
                key: value.detach().cpu() for key, value in online.state_dict().items()
            }), path)
            _atomic_json(result, path.with_suffix(".json"))
            state["selections"].append({"step": step, "score": score, "passed": result["passed"]})
            if state["best"] is None or score > state["best"]["score"]:
                state["best"] = {"step": step, "score": score, "path": str(path), "passed": result["passed"]}
                state["stale_checks"] = 0
            else:
                state["stale_checks"] += 1
            if result["passed"] and policy in {PolicyId.EXPLORE, PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
                # Preserve the first passing candidate even when a failed earlier
                # boundary had a higher aggregate score but collapsed a subgroup.
                state["best"] = {"step": step, "score": score, "path": str(path), "passed": True}
            if diagnostic:
                state["phase"] = "diagnostic_complete"
            elif result["passed"]:
                state["phase"] = "validation_complete"
            elif step >= online_limit or (
                step >= config["minimum_steps"] and state["stale_checks"] >= config["patience"]
            ):
                state["phase"] = "validation_complete"
            save("selection_evaluated")
        if state["episodes"] % 25 == 0 or stop.requested:
            save("policy_interrupted" if stop.requested else "training_progress")
        if stop.requested:
            return state

    if state["phase"] == "validation_complete":
        best = state["best"]
        if not best["passed"]:
            state["phase"] = "failed_validation"
        else:
            candidate = Path(best["path"])
            test_path = output / f"{policy.value}-test.json"
            if test_path.exists():
                result = json.loads(test_path.read_text(encoding="utf-8"))
                if result["checkpoint_sha256"] != file_sha256(candidate):
                    raise ValueError("This test split has already been used for another candidate.")
            else:
                model, _ = load_policy_checkpoint(torch, candidate, expected_policy_id=policy)
                test_cases = randomized_cases(output / "cases", "test", policy)
                result = evaluate_randomized(torch, model.to(device), test_cases, device, config["test_cases"])
                result["checkpoint_sha256"] = file_sha256(candidate)
                _atomic_json(result, test_path)
            state["phase"] = "selected" if result["passed"] else "failed_test"
            if result["passed"]:
                destination = output / "candidates" / POLICY_FILENAMES[policy]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                shutil.copy2(candidate, temporary)
                temporary.replace(destination)
    save("policy_finished")
    return state


def train_resource(
    torch: Any, output: Path, device: str, resume: bool, stop: Any,
    init_checkpoint: Path | None, *, diagnostic: bool = False,
) -> dict[str, Any]:
    """Train GetResource from a selected NavigateTo initialization."""

    return train_randomized(
        torch, output, PolicyId.GET_RESOURCE, device, resume, stop,
        init_checkpoint, diagnostic=diagnostic,
    )
