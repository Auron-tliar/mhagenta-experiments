"""Train, resume, evaluate, and export the five independent Crafter policies."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from cases import CaseRequest, collect_episode
from mha_exp_level2_cr.exp2_5.contracts import RESOURCE_ITEMS, activity_action_bound
from mha_exp_level2_cr.exp2_5.policy import (
    ARTIFACT_DIRNAME,
    ENVIRONMENT_CONTRACT,
    INPUT_CONTRACT,
    MANIFEST_FORMAT_VERSION,
    POLICY_FILENAMES,
    TRAINING_CONFIG,
    TRAINING_CONFIG_SHA256,
    PolicyId,
    HUD_MASKED_POLICIES,
    build_q_network,
    checkpoint_data,
    file_sha256,
    load_policy_checkpoint,
)
from protocol import (
    NAVIGATION_BANDS,
    POLICY_SPEC_BY_ID,
    POLICY_SPECS,
    PolicySpec,
    SeedConsumptionError,
    demonstration_step_target,
    epsilon_at,
    pilot_requests,
    priority_beta,
    seed_windows,
    selection_requests,
    training_requests,
)
from replay import CompressedReplay, optimize
from stop_control import StopController
from atomic_io import replace_file

CONFIG = TRAINING_CONFIG
CONFIG_SHA256 = TRAINING_CONFIG_SHA256
RESOURCE_KINDS = tuple(RESOURCE_ITEMS)
POLICY_ORDER = tuple(PolicyId)
WORKING_FORMAT_VERSION = 5
PROGRESS_EPISODE_INTERVAL = 25


def _emit_event(event: str, **fields: Any) -> None:
    """Write one immediately visible structured operator event to stdout."""

    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _training_specs(policy: PolicyId | None) -> tuple[PolicySpec, ...]:
    """Return all policies or exactly one explicitly requested policy."""

    if policy is None:
        return POLICY_SPECS
    if type(policy) is not PolicyId:
        raise TypeError("policy must use the exact v3 policy enum.")
    return (POLICY_SPEC_BY_ID[policy],)


def _set_determinism(torch: Any, seed: int, device: str) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError("An explicit --device cpu or cuda is required.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch.cuda.get_device_name(0) if device == "cuda" else "cpu"


def _new_replay(index: int, policy_id: PolicyId) -> CompressedReplay:
    """Construct an empty policy-bound replay with a fresh RNG."""

    return CompressedReplay(
        CONFIG["replay_capacity"],
        CONFIG["master_seed"] + index,
        priority_alpha=CONFIG["priority_alpha"],
        demonstration_bonus=CONFIG["demonstration_priority_bonus"],
        demonstration_fraction=CONFIG["demonstration_batch_fraction"],
        policy_id=policy_id,
    )


def _warm_model(torch: Any, policy_id: PolicyId, device: str,
                init_checkpoint: Path | None = None) -> Any:
    """Create trainable weights, optionally transferring navigation into resource collection."""

    model = build_q_network(
        torch,
        mask_hud=policy_id in HUD_MASKED_POLICIES,
        scale_context=policy_id in HUD_MASKED_POLICIES,
    )
    if init_checkpoint is not None:
        value = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        source_policy = PolicyId(value["policy_id"])
        source_model, value = load_policy_checkpoint(torch, init_checkpoint, expected_policy_id=source_policy)
        if getattr(source_model, "uses_discovery_context", False):
            raise ValueError("The randomized DQfD trainer uses zero search context; use the explicit discovery-context training workflow.")
        if policy_id is PolicyId.GET_RESOURCE and source_policy is PolicyId.NAVIGATE_TO:
            state = model.state_dict()
            source = value["model_state_dict"]
            for key in state:
                if key not in {"head.2.weight", "head.2.bias"}:
                    state[key] = source[key]
            state["head.2.weight"][1:5] = source["head.2.weight"][1:5]
            state["head.2.bias"][1:5] = source["head.2.bias"][1:5]
            model.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(value["model_state_dict"], strict=True)
    return model.to(device).train()


def _collect_next(
    torch: Any,
    model: Any,
    request: CaseRequest,
    seed: int,
    *,
    expert: bool,
    epsilon: float,
    device: str,
    max_seed: int | None = None,
) -> tuple[int, list[Any], bool, dict[str, Any]]:
    """Consume construction skips until one requested case is executed."""

    first_seed = seed

    def report_skips() -> None:
        skipped = seed - first_seed
        if skipped == 10 or skipped % 100 == 0:
            _emit_event(
                "case_search",
                policy_id=request.policy.value,
                stratum=request.stratum,
                skipped_seeds=skipped,
                next_seed=seed,
            )

    limit = seed + 999 if max_seed is None else max_seed
    while seed <= limit:
        try:
            rows, success, stats = collect_episode(
                torch,
                model,
                request,
                seed,
                expert=expert,
                epsilon=epsilon,
                device=device,
                discount=CONFIG["discount"],
                horizon=CONFIG["n_step_horizon"],
            )
            if expert and (not rows or len(rows) > CONFIG["activity_action_bound"] or not success or any(stats.get(field) != 0 for field in (
                    "illegal_actions",
                    "lethal_actions",
                    "environment_terminal_failures",
                    "target_lost",
            ))):
                seed += 1
                report_skips()
                continue
            if not expert and not rows:
                raise RuntimeError("Structurally valid online case produced no rows.")
            return seed + 1, rows, success, stats
        except ValueError as error:
            if str(error).startswith("No eligible") or (expert and str(error) in {
                    "Active target was lost.",
                    "No observable frontier route to an ungrounded cow.",
                    "No observable route to target.",
                    "No safe route to interaction target.",
                    "No safe route to navigation target.",
            }):
                seed += 1
                report_skips()
                continue
            raise SeedConsumptionError(seed + 1, error) from error
        except Exception as error:
            raise SeedConsumptionError(seed + 1, error) from error
    raise SeedConsumptionError(
        limit + 1,
        RuntimeError(f"No eligible {request.policy.value} case in seed window."),
    )


def evaluate(
    torch: Any,
    models: Mapping[PolicyId, Any],
    requests: Sequence[CaseRequest],
    start_seed: int,
    *,
    device: str,
) -> dict[str, Any]:
    """Run one exact masked-network schedule and return structural evidence."""

    strata = {
        key: {"attempted": 0, "succeeded": 0}
        for key in dict.fromkeys(request.stratum for request in requests)
    }
    result: dict[str, Any] = {
        "attempted": len(requests),
        "succeeded": 0,
        "action_count": 0,
        "real_network_actions": 0,
        "expert_policy_actions": 0,
        "illegal_actions": 0,
        "lethal_actions": 0,
        "environment_terminal_failures": 0,
        "harness_errors": 0,
        "by_kind": {},
        "first_directions": {},
        "distance_bands": {},
        "variants": {},
        "strata": strata,
        "structural_case_count": 0,
        "successful_structural_cases": 0,
        "moving_target_cases": 0,
        "target_cow_consumed": 0,
        "wrong_cows_consumed": 0,
        "movement_actions": 0,
        "do_actions": 0,
        "search_actions": 0,
        "acquisition_events": 0,
        "pursuit_actions": 0,
        "reacquisition_events": 0,
    }
    seed = start_seed
    for request in requests:
        result["strata"][request.stratum]["attempted"] += 1
        record = result["by_kind"].setdefault(request.target_kind, {"attempted": 0, "succeeded": 0})
        record["attempted"] += 1
        try:
            seed, rows, success, stats = _collect_next(
                torch,
                models[request.policy],
                request,
                seed,
                expert=False,
                epsilon=0.0,
                device=device,
                max_seed=start_seed + 1_999,
            )
            actions = len(rows)
            result["succeeded"] += int(success)
            record["succeeded"] += int(success)
            result["strata"][request.stratum]["succeeded"] += int(success)
            result["action_count"] += actions
            result["real_network_actions"] += stats["network_actions"]
            for field in (
                    "illegal_actions",
                    "lethal_actions",
                    "environment_terminal_failures",
            ):
                result[field] += stats[field]
            first_direction = request.first_direction or stats["first_direction"]
            direction = str(first_direction)
            if first_direction in range(1, 5):
                direction_record = result["first_directions"].setdefault(
                    direction,
                    {
                        "attempted": 0,
                        "succeeded": 0
                    },
                )
                direction_record["attempted"] += 1
                direction_record["succeeded"] += int(success)
            if request.distance_band != "none":
                band = result["distance_bands"].setdefault(
                    request.distance_band,
                    {
                        "attempted": 0,
                        "succeeded": 0
                    },
                )
                band["attempted"] += 1
                band["succeeded"] += int(success)
            variant = result["variants"].setdefault(
                request.variant,
                {"attempted": 0, "succeeded": 0},
            )
            variant["attempted"] += 1
            variant["succeeded"] += int(success)
            result["moving_target_cases"] += int(stats["target_moves"] > 0)
            for field in (
                "target_cow_consumed", "wrong_cows_consumed", "movement_actions",
                "do_actions", "search_actions", "acquisition_events",
                "pursuit_actions", "reacquisition_events",
            ):
                result[field] += stats[field]
            structural = stats["movement_actions"] > 0
            if request.policy in {PolicyId.GET_RESOURCE, PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
                structural = structural and stats["do_actions"] > 0
            if request.policy is PolicyId.EAT_COW:
                structural = structural and stats["search_actions"] > 0 and stats["acquisition_events"] > 0 and stats["pursuit_actions"] > 0
            result["structural_case_count"] += int(structural)
            result["successful_structural_cases"] += int(success and structural)
        except SeedConsumptionError as error:
            result["harness_errors"] += 1
            seed = error.next_seed
    result["next_seed"] = seed
    return result


def selection_passes(spec: PolicySpec, result: Mapping[str, Any]) -> bool:
    """Apply the frozen policy-specific candidate-selection gate."""

    if result["succeeded"] < spec.selection_successes or any(result[field] for field in (
            "harness_errors",
            "illegal_actions",
            "lethal_actions",
            "environment_terminal_failures",
    )):
        return False
    directions, bands, variants = (
        result["first_directions"], result["distance_bands"], result["variants"]
    )
    if spec.policy_id is PolicyId.EXPLORE:
        return all(directions.get(str(value), {}).get("succeeded", 0) >= 2 for value in range(1, 5))
    if spec.policy_id is PolicyId.NAVIGATE_TO:
        return (
            all(directions.get(str(value), {}).get("succeeded", 0) >= 5 for value in range(1, 5))
            and all(bands.get(name, {}).get("succeeded", 0) >= 7 for name in NAVIGATION_BANDS)
            and all(variants.get(name, {}).get("succeeded", 0) >= 10 for name in ("clear", "obstructed"))
        )
    if spec.policy_id is PolicyId.GET_RESOURCE:
        aligned = sum(
            record["succeeded"]
            for name, record in variants.items()
            if "misaligned" not in name
        )
        misaligned = sum(record["succeeded"] for name, record in variants.items() if name.endswith("misaligned"))
        clear = sum(record["succeeded"] for name, record in variants.items() if name.startswith("clear"))
        obstructed = sum(record["succeeded"] for name, record in variants.items() if name.startswith("obstructed"))
        return (
            all(result["by_kind"].get(kind, {}).get("succeeded", 0) >= 3 for kind in RESOURCE_KINDS)
            and all(directions.get(str(value), {}).get("succeeded", 0) >= 5 for value in range(1, 5))
            and bands.get("short", {}).get("succeeded", 0) >= 5
            and bands.get("mid", {}).get("succeeded", 0) >= 5
            and bands.get("long", {}).get("succeeded", 0) >= 10
            and min(aligned, misaligned) >= 11 and min(clear, obstructed) >= 10
        )
    if spec.policy_id is PolicyId.EAT_TARGET:
        return (
            all(directions.get(str(value), {}).get("succeeded", 0) >= 4 for value in range(1, 5))
            and all(bands.get(name, {}).get("succeeded", 0) >= 6 for name in NAVIGATION_BANDS)
            and min(variants.get(name, {}).get("succeeded", 0) for name in ("single", "distractor")) >= 9
            and variants.get("distractor", {}).get("attempted", 0) == 12
            and result["moving_target_cases"] >= 12
            and result["movement_actions"] > 0 and result["do_actions"] > 0
        )
    return (
        all(directions.get(str(value), {}).get("succeeded", 0) >= 4 for value in range(1, 5))
        and all(bands.get(name, {}).get("succeeded", 0) >= 5 for name in NAVIGATION_BANDS)
        and variants.get("ordinary", {}).get("succeeded", 0) >= 9
        and variants.get("reacquisition", {}).get("succeeded", 0) >= 6
        and result["successful_structural_cases"] == result["succeeded"]
    )


def pilot_passes(spec: PolicySpec, result: Mapping[str, Any]) -> bool:
    """Check structural feasibility for one exact pilot schedule."""

    if (result["harness_errors"] or result["real_network_actions"] <= 0 or set(result["strata"]) != {request.stratum
                                                                                                  for request in pilot_requests(spec.policy_id)} or any(record["attempted"] != 1 for record in result["strata"].values())):
        return False
    if spec.policy_id is PolicyId.EXPLORE:
        return result["structural_case_count"] == 4 and result["succeeded"] == 4
    if spec.policy_id is PolicyId.NAVIGATE_TO:
        return result["structural_case_count"] >= 4 and all(result["distance_bands"].get(name, {}).get("attempted", 0) == 4 for name in NAVIGATION_BANDS)
    if spec.policy_id is PolicyId.GET_RESOURCE:
        return result["structural_case_count"] >= 6 and all(result["by_kind"].get(kind, {}).get("attempted", 0) >= 1 for kind in RESOURCE_KINDS)
    if spec.policy_id is PolicyId.EAT_TARGET:
        return result["structural_case_count"] >= 4 and result["moving_target_cases"] >= 1 and result["movement_actions"] > 0 and result["do_actions"] > 0
    return result["structural_case_count"] >= 1 and result["search_actions"] > 0 and result["acquisition_events"] > 0 and result["pursuit_actions"] > 0 and result["do_actions"] > 0


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    replace_file(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    """Atomically write canonical, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replace_file(temporary, path)


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def _restore_rng(torch: Any, value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    if value["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in value["torch_cuda"]])


def _new_state(torch: Any, spec: PolicySpec, device: str, init_checkpoint: Path | None = None) -> tuple[Any, ...]:
    """Initialize fresh optimizer, replay, target model, counters, and RNG."""

    index = POLICY_ORDER.index(spec.policy_id)
    _set_determinism(torch, CONFIG["master_seed"] + index, device)
    online = _warm_model(torch, spec.policy_id, device, init_checkpoint)
    target = build_q_network(
        torch,
        mask_hud=spec.policy_id in HUD_MASKED_POLICIES,
        scale_context=spec.policy_id in HUD_MASKED_POLICIES,
    ).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=CONFIG["learning_rate"])
    state = {
        "format_version": WORKING_FORMAT_VERSION, "configuration_sha256": CONFIG_SHA256,
        "policy_id": spec.policy_id.value, "device": device, "torch_version": str(torch.__version__),
        "initial_checkpoint_sha256": file_sha256(init_checkpoint) if init_checkpoint else None,
        "phase": "demonstrations", "demonstration_steps": 0, "online_steps": 0,
        "optimizer_steps": 0, "episodes": 0, "demo_index": 0, "online_index": 0,
        "coverage": {request.stratum: 0 for request in training_requests(spec.policy_id)},
        "selections": [], "elapsed_seconds": 0.0, "losses": {},
        "next_seeds": {phase: seed_windows()[f"{phase}_{spec.policy_id.value}"][0]
                       for phase in ("demo", "online", "selection")},
    }
    return state, online, target, optimizer, _new_replay(index, spec.policy_id)


def _load_state(torch: Any, path: Path, spec: PolicySpec, device: str) -> tuple[Any, ...]:
    """Resume a trusted local working file only under the same current training setup."""

    value = torch.load(path, map_location=device, weights_only=False)
    state = value["state"]
    if (state["format_version"] != WORKING_FORMAT_VERSION
            or state["configuration_sha256"] != CONFIG_SHA256
            or state["policy_id"] != spec.policy_id.value or state["device"] != device
            or state["torch_version"] != str(torch.__version__)
            or state["optimizer_steps"] != state.get("pretraining_steps", 0)
            + state["online_steps"] // state.get("optimization_interval", 1)):
        raise ValueError("Working state is incompatible with this training invocation.")
    _, online, target, optimizer, _ = _new_state(torch, spec, device)
    online.load_state_dict(value["online"], strict=True)
    target.load_state_dict(value["target"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    replay = CompressedReplay.from_state_dict(value["replay"])
    if replay.policy_id is not spec.policy_id:
        raise ValueError("Replay policy identity differs.")
    _restore_rng(torch, value["rng"])
    return state, online, target, optimizer, replay


def _train_policy(
    torch: Any, output: Path, spec: PolicySpec, device: str, resume: bool,
    stop: StopController, init_checkpoint: Path | None = None, *, diagnostic: bool = False,
) -> dict[str, Any]:
    """Commit complete episodes, checkpoint regularly, and select at fixed boundaries."""

    if spec.policy_id in set(PolicyId):
        from resource_training import train_randomized
        return train_randomized(torch, output, spec.policy_id, device, resume, stop,
                                init_checkpoint, diagnostic=diagnostic)

    work = output / "work" / f"{spec.policy_id.value}.pt"
    state, online, target, optimizer, replay = (
        _load_state(torch, work, spec, device) if resume and work.exists()
        else _new_state(torch, spec, device, init_checkpoint)
    )
    started = perf_counter()
    schedule = training_requests(spec.policy_id)
    recent: Counter[str] = Counter()
    demo_limit = 256 if diagnostic else demonstration_step_target(spec.policy_id)
    boundaries = [512] if diagnostic else CONFIG["selection_boundaries"]

    def save() -> None:
        nonlocal started
        state["elapsed_seconds"] += perf_counter() - started
        _atomic_torch_save(torch, {
            "state": state, "online": online.state_dict(), "target": target.state_dict(),
            "optimizer": optimizer.state_dict(), "replay": replay.state_dict(), "rng": _rng_state(torch),
        }, work)
        _atomic_json(state, output / f"{spec.policy_id.value}-progress.json")
        started = perf_counter()

    def progress(event: str) -> None:
        _emit_event(event, **state, recent_outcomes=dict(recent))
        recent.clear()

    progress("policy_started")
    while state["phase"] in {"demonstrations", "online"}:
        demonstration = state["phase"] == "demonstrations"
        phase = "demo" if demonstration else "online"
        if demonstration and state["demonstration_steps"] >= demo_limit and (
            diagnostic or all(count >= 2 for count in state["coverage"].values())
        ):
            replay.seal_demonstrations()
            state["phase"] = "online"
            save()
            progress("demonstrations_completed")
            continue
        request = schedule[state[f"{phase}_index"] % len(schedule)]
        seed, rows, success, stats = _collect_next(
            torch, online, request, state["next_seeds"][phase], expert=demonstration,
            epsilon=0.0 if demonstration else epsilon_at(state["online_steps"]), device=device,
            max_seed=seed_windows()[f"{phase}_{spec.policy_id.value}"][1],
        )
        for row in rows:
            replay.append(row)
            if not demonstration:
                state["losses"] = optimize(torch, online, target, optimizer, replay, device,
                                           priority_beta(state["online_steps"]), CONFIG)
                if not all(np.isfinite(value) for value in state["losses"].values()):
                    progress("nonfinite_loss")
                    raise FloatingPointError("Non-finite loss; resume from the last durable episode.")
                state["online_steps"] += 1
                state["optimizer_steps"] += 1
                if state["optimizer_steps"] % CONFIG["target_sync_optimizer_steps"] == 0:
                    target.load_state_dict(online.state_dict())
        if demonstration:
            state["demonstration_steps"] += len(rows)
            state["coverage"][request.stratum] += 1
        state["next_seeds"][phase] = seed
        state[f"{phase}_index"] += 1
        state["episodes"] += 1
        recent.update(episodes=1, successes=int(success), actions=len(rows))
        recent.update({key: stats[key] for key in (
            "illegal_actions", "lethal_actions", "environment_terminal_failures",
            "target_lost", "target_cow_consumed", "wrong_cows_consumed",
        )})
        completed = len(state["selections"])
        if not demonstration and completed < len(boundaries) and state["online_steps"] >= boundaries[completed]:
            evaluation = evaluate(torch, {spec.policy_id: online},
                                  pilot_requests(spec.policy_id) if diagnostic else selection_requests(spec.policy_id),
                                  state["next_seeds"]["selection"], device=device)
            record = {
                "boundary": boundaries[completed], "step": state["online_steps"],
                "passed": pilot_passes(spec, evaluation) if diagnostic else selection_passes(spec, evaluation),
                "evaluation": evaluation,
            }
            state["selections"].append(record)
            state["next_seeds"]["selection"] = evaluation["next_seed"]
            if diagnostic:
                state["phase"] = "diagnostic_complete"
            elif record["passed"]:
                state["phase"] = "selected"
            elif completed + 1 == len(boundaries):
                state["phase"] = "failed"
            save()
            checkpoint = checkpoint_data(spec.policy_id, {key: value.detach().cpu() for key, value in online.state_dict().items()})
            _atomic_torch_save(torch, checkpoint, output / "boundaries" / f"{spec.policy_id.value}-{record['boundary']}.pt")
            if state["phase"] == "selected":
                _atomic_torch_save(torch, checkpoint, output / "candidates" / spec.filename)
            progress("selection_evaluated")
        if state["episodes"] % PROGRESS_EPISODE_INTERVAL == 0 or stop.requested:
            save()
            progress("policy_interrupted" if stop.requested else "training_progress")
        if stop.requested:
            return {**state, "status": "interrupted"}
    save()
    if state["phase"] == "selected":
        # Recreate a candidate if interruption occurred after committing selection.
        _atomic_torch_save(torch, checkpoint_data(spec.policy_id, {
            key: value.detach().cpu() for key, value in online.state_dict().items()
        }), output / "candidates" / spec.filename)
    progress("policy_finished")
    if state["phase"] == "failed":
        raise RuntimeError(f"{spec.policy_id.value} failed final selection.")
    return state


def train(
    output: Path, device: str, policy: PolicyId | None = None, *,
    resume: bool = False, init_checkpoint: Path | None = None, diagnostic: bool = False,
) -> dict[str, Any]:
    """Train independent policies; an explicit warm start is limited to a single fresh call."""

    if init_checkpoint is not None and (policy is None or resume):
        raise ValueError("--init-checkpoint requires one policy and cannot be combined with --resume.")
    if output.exists() and not resume and any(path.name != "cases" for path in output.iterdir()):
        raise FileExistsError(output)
    if resume and not output.is_dir():
        raise FileNotFoundError(output)
    import torch

    output.mkdir(parents=True, exist_ok=True)
    device_name = _set_determinism(torch, CONFIG["master_seed"], device)
    config = {"configuration": CONFIG, "device": device, "device_name": device_name,
              "policies": [spec.policy_id.value for spec in _training_specs(policy)], "diagnostic": diagnostic}
    if policy is not None:
        from resource_training import RANDOMIZED_CONFIGS
        config["randomized_configuration"] = RANDOMIZED_CONFIGS[policy]
        if policy is PolicyId.EAT_COW:
            bound = activity_action_bound(policy.value)
            config["configuration"] = {**CONFIG, "activity_action_bound": bound,
                                       "maximum_demo_episode_overshoot": bound - 1,
                                       "maximum_committed_online_steps": 20_000 + bound - 1}
    config_path = output / "config.json"
    if resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration changed.")
    else:
        _atomic_json(config, config_path)
    results = {}
    with StopController() as stop:
        for spec in _training_specs(policy):
            results[spec.policy_id.value] = _train_policy(
                torch, output, spec, device, resume, stop, init_checkpoint, diagnostic=diagnostic,
            )
            _atomic_json(results, output / ("pilot.json" if diagnostic else "training.json"))
            if stop.requested:
                break
    return results


def export_models(checkpoints: Mapping[PolicyId, Path], destination: Path) -> dict[str, Any]:
    """Export exactly five current checkpoints without certification or ancestry gates."""

    if set(checkpoints) != set(PolicyId):
        raise ValueError("Export requires one checkpoint for each of the five policies.")
    if destination.exists():
        raise FileExistsError(destination)
    import torch

    for policy, path in checkpoints.items():
        load_policy_checkpoint(torch, path, expected_policy_id=policy)
    destination.mkdir(parents=True)
    records = {}
    for policy, source in checkpoints.items():
        target = destination / POLICY_FILENAMES[policy]
        shutil.copy2(source, target)
        records[policy.value] = {"filename": target.name, "sha256": file_sha256(target)}
    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION, "artifact_id": ARTIFACT_DIRNAME,
        "environment": ENVIRONMENT_CONTRACT, "input": INPUT_CONTRACT, "policies": records,
    }
    _atomic_json(manifest, destination / "manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the small preparation CLI without hidden prerequisite jobs."""

    from cli import parse_args
    args = parse_args(argv)
    if args.command in {"train", "pilot"}:
        results = train(args.directory.resolve(), args.device, PolicyId(args.policy) if args.policy else None,
                        resume=getattr(args, "resume", False), init_checkpoint=args.init_checkpoint,
                        diagnostic=args.command == "pilot")
        if any(result["phase"] in {"failed_validation", "failed_test", "failed_safety"} for result in results.values()):
            return 1
    elif args.command == "evaluate":
        import torch
        policy = PolicyId(args.policy)
        _set_determinism(torch, CONFIG["master_seed"], args.device)
        model, _ = load_policy_checkpoint(torch, args.checkpoint, expected_policy_id=policy)
        model.to(args.device)
        if policy in {PolicyId.NAVIGATE_TO, PolicyId.GET_RESOURCE}:
            from resource_cases import ResourceCases
            from resource_training import RANDOMIZED_CONFIGS, evaluate_randomized
            if args.seed != 5_240_000:
                raise ValueError("Randomized policies use fixed world-held-out validation seeds, not --seed.")
            report = evaluate_randomized(
                torch, model, ResourceCases(args.report.parent / "cases", "validation", policy),
                args.device, RANDOMIZED_CONFIGS[policy]["validation_cases"],
            )
        else:
            raise ValueError(
                "Legacy 12/24-case evaluation is retired for remaining policies. "
                "Use remaining_evaluation.py with its world-held-out validation/test splits."
            )
        _atomic_json(report, args.report)
        _emit_event("evaluation_completed", policy_id=policy.value, **report)
    else:
        checkpoints = {}
        for assignment in args.checkpoint:
            name, path = assignment.split("=", 1)
            policy = PolicyId(name)
            if policy in checkpoints:
                raise ValueError(f"Repeated export policy: {name}.")
            checkpoints[policy] = Path(path)
        export_models(checkpoints, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
