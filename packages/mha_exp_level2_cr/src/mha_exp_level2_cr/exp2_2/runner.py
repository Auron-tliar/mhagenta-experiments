"""Orchestration and architecture-feasibility checking for 2-2-CR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from importlib import import_module
from importlib.util import find_spec
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, cast

from mhagenta import Orchestrator

import mha_exp_common
from mha_exp_common.batch import normalize_runs, run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_TORCH_MHAGENTA_VERSION
from mha_exp_common.names import ACTUATOR, KNOWLEDGE, LEARNER, LLREASONER, MEMORY, PERCEPTOR
from mha_exp_common.utils import Seeder, gather_states, module_name

from .modules import (
    DURATION, ENVIRONMENT_DURATION, EVALUATION_SEEDS, REPLAY_BATCH_SIZE,
    TOTAL_TRAINING_TRANSITIONS, TRAINING_START_THRESHOLD, TRAINING_UPDATES,
    TRAINING_SHUTDOWN_MARGIN, SHUTDOWN_INELIGIBLE, UPDATE_ELIGIBLE, WARMUP,
    TestActuator, TestEnvironment, TestKnowledge, TestLearner, TestLLReasoner,
    TestMemory, TestPerceptor,
)
from .policy import (
    FRAME_SHAPE, FRAME_STACK_SIZE, ILLEGAL_ACTION_PENALTY, MODEL_IMAGE_SHAPE,
    N_ACTIONS, POLICY_FILENAME, STEP_REWARD, TARGET_ACHIEVEMENT,
    TARGET_ACHIEVEMENT_REWARD, load_policy_checkpoint,
)
from .reporting import process_execution_metrics
from .rewards import reward_state
from .treatment import (
    ACHIEVEMENT_REWARDS, DEFAULT_WORKLOAD, DEATH_PENALTY, DQNWorkload,
    PROTOCOL_VERSION, SURVIVAL_REWARDS, algorithm_metadata, reward_metadata,
    runtime_identity,
)


DEFAULT_MHAGENTA_VERSION = DEFAULT_TORCH_MHAGENTA_VERSION
TREATMENT_ID = PROTOCOL_VERSION
SAVE_SUBDIR = Orchestrator.SAVE_SUBDIR
_FAILURE_MARKERS = (
    "traceback",
    "exceptiongroup",
    "caught exception",
    "could not send message",
    "failed to save state",
)
_LEVEL_PATTERN = re.compile(r"\[(debug|info|warning|error|critical)\]", re.I)


def _zeros(*fields: str) -> dict[str, int]:
    return dict.fromkeys(fields, 0)


def initial_states(workload: DQNWorkload = DEFAULT_WORKLOAD) -> dict[str, dict[str, Any]]:
    """Return compact JSON-safe initial state for all six modules."""

    classes = _zeros(*(f"{name}_transitions" for name in (
        WARMUP, UPDATE_ELIGIBLE, SHUTDOWN_INELIGIBLE
    )))
    return {
        "perceptor": _zeros("requests", "observations_forwarded", "contract_errors"),
        "actuator": _zeros("requests", "statuses_forwarded", "contract_errors"),
        "ll_reasoner": {
            "treatment": {
                "protocol_version": PROTOCOL_VERSION,
                "treatment_id": TREATMENT_ID,
                "training_transitions": workload.training_transitions,
                "warmup_transitions": TRAINING_START_THRESHOLD,
                "training_updates": workload.training_updates,
                "evaluation_seeds": list(workload.evaluation_seeds),
                "evaluation_action_limit": workload.episode_action_limit,
                "workload": workload.dump(),
                "environment": {"no_mobs": True},
                "algorithm": algorithm_metadata(workload.synchronized_training), "reward": reward_metadata(),
                "target_achievement": TARGET_ACHIEVEMENT.value,
            },
            "phase": "warmup",
            "phase_timestamps": {
                "warmup_started": None, "training_started": None,
                "drain_started": None, "frozen_evaluation_started": None,
                "complete": None,
            },
            **_zeros(
                "observations", "actions", "statuses", "transitions_emitted",
                "update_eligible_transitions", "shutdown_ineligible_transitions",
                "post_closure_environment_request_attempts", "episodes_started",
                "target_successes", "deaths", "truncations", "completed_episodes",
                "total_episode_length", "stack_initializations", "stack_shifts",
                "policy_inferences", "stack_contract_errors", "models_installed",
                "cycles_started", "cycles_completed", "training_completions_received",
                "cycle_errors", "invalid_inputs",
            ),
            "training_closed_at_transition": None,
            "training_closed_at_elapsed_seconds": None,
            "action_histogram": [0] * N_ACTIONS,
            "evaluation_actions": 0,
            "evaluation_observations": 0,
            "evaluation_replay_insertion_attempts": 0,
            "evaluation_cases": [],
            "training_episodes": [],
            "evaluation_index": 0,
            "evaluation_return": 0.0,
            "evaluation_progression": [],
            "frozen_checkpoint_digest": None,
            "collection_closed": False, "training_started_at": None,
            "training_deadline": None, "last_training_action_started": None,
            "installed_update": 0,
            "cutoff_reset_observations": 0,
        },
        "knowledge": {
            **_zeros("evaluated_transitions", "contract_errors"),
            **classes,
            "cumulative_intrinsic_reward": 0.0,
            "reward_tracker": reward_state(),
            "reward_components": {}, "episode_rewards": [],
            "collection_closed": False,
        },
        "memory": {
            **_zeros(
                "transitions_admitted", "buffer_size", "batches_sent",
                "ordinal_errors", "classification_errors", "buffer_errors",
                "token_errors", "contract_errors",
                "experiences_finalized", "pending_transitions", "priority_updates",
            ),
            "evaluation_replay_insertion_attempts": 0,
            **classes,
            "next_cycle_id": 1,
            "horizon_counts": [0, 0, 0],
            "collection_closed": False, "sampling_closed": False,
            "closure_acknowledged": False, "pending_batch": False,
            "evictions": 0, "stale_priority_feedback": 0, "cancelled_batches": 0,
            "oldest_replay_id": None, "newest_replay_id": None,
        },
        "learner": {
            **_zeros(
                "training_updates", "target_syncs", "models_published",
                "training_completions_sent", "contract_errors", "saved_training_steps",
                "priority_updates_sent", "priority_acknowledgements",
            ),
            "model_saved": False,
            "model_artifact": "",
            "frozen": False,
            "post_freeze_update_attempts": 0,
            "checkpoint_digest": None,
            "optimization_windows": [],
            "optimization_current": None,
            "device": None,
            "cuda_available": None,
            "cuda_device_name": None,
            "torch_version": None, "cuda_runtime": None,
            "training_started_at": None, "training_deadline": None,
            "last_update_started": None, "freeze_elapsed": None,
            "unused_batches": 0, "evaluation_snapshots": [],
            "next_evaluation_snapshot": None,
        },
    }


def environment_initial_state(seed: int, *, action_masking: bool = False) -> dict[str, Any]:
    """Return compact JSON-safe state for the Crafter adapter."""
    return {"initial_seed": int(seed), "resets": 1, "no_mobs": True,
            "action_masking": action_masking,
            "evaluation_resets": 0, "applied_evaluation_seeds": [],
            **_zeros("observation_requests", "native_actions", "statuses", "contract_errors")}


def _module_ids() -> dict[str, str]:
    return {
        "perceptor": module_name(PERCEPTOR, 0), "actuator": module_name(ACTUATOR, 0),
        "ll_reasoner": module_name(LLREASONER, 0), "knowledge": module_name(KNOWLEDGE, 0),
        "memory": module_name(MEMORY, 0), "learner": module_name(LEARNER, 0),
    }


def _state(states: Mapping[str, Mapping[str, Any]], role: str,
           reasons: list[str]) -> Mapping[str, Any]:
    module_id = _module_ids()[role]
    value = states.get(module_id)
    if not isinstance(value, Mapping):
        reasons.append(f"missing saved module state: {module_id}")
        return {}
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        reasons.append(f"module state is not JSON-safe ({module_id}): {exc}")
    return value


def _runtime_failure_lines(logs: Mapping[str, Sequence[str]]) -> list[str]:
    failures: list[str] = []
    for lines in logs.values():
        for line in lines:
            lowered = line.lower()
            level = _LEVEL_PATTERN.search(line)
            if (level and level.group(1).lower() in {"error", "critical"}) or any(
                marker in lowered for marker in _FAILURE_MARKERS
            ):
                failures.append(line.rstrip())
    return failures


def checkpoint_evidence(model_path: str | os.PathLike[str] | None) -> tuple[dict[str, Any], list[str]]:
    """Load the final artifact and return validated treatment metadata."""
    if model_path is None:
        return {}, ["final policy checkpoint path was not supplied"]
    path = Path(model_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        return {"path": str(path)}, [f"final policy checkpoint is missing or empty: {path}"]
    if find_spec("torch") is None:
        return {"path": str(path)}, ["torch is unavailable for checkpoint validation"]
    try:
        _, metadata = load_policy_checkpoint(import_module("torch"), path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"path": str(path)}, [f"could not validate final policy checkpoint: {exc}"]
    metadata = {key: value for key, value in metadata.items() if key != "model_state_dict"}
    return {
        "path": str(path),
        "file_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **metadata,
    }, []


def _reward_trace_valid(ll: Mapping[str, Any], knowledge: Mapping[str, Any], workload: DQNWorkload) -> bool:
    """Reconcile completed episodes, bounded bonuses, and additive reward totals."""
    episodes, rewards = ll.get("training_episodes"), knowledge.get("episode_rewards")
    if not isinstance(episodes, list) or not isinstance(rewards, list) or len(episodes) != len(rewards):
        return False
    if len(episodes) != ll.get("completed_episodes"):
        return False
    bonuses = {**ACHIEVEMENT_REWARDS, **SURVIVAL_REWARDS}
    totals: dict[str, float] = {}
    length_sum = 0
    try:
        for index, (episode, reward) in enumerate(zip(episodes, rewards, strict=True), 1):
            if episode["episode_id"] != index or reward["episode_id"] != index:
                return False
            length = episode["length"]
            if type(length) is not int or not 0 < length <= workload.episode_action_limit:
                return False
            outcomes = [episode[key] for key in ("success", "death", "truncation")]
            if any(type(value) is not bool for value in outcomes) or sum(outcomes) != 1:
                return False
            components = reward["components"]
            if not isinstance(components, Mapping) or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in components.values()
            ):
                return False
            if not math.isclose(components.get("step", 0), STEP_REWARD * length, abs_tol=1e-8):
                return False
            if components.get("death", 0) != DEATH_PENALTY * episode["death"]:
                return False
            illegal = components.get("illegal_action", 0) / ILLEGAL_ACTION_PENALTY
            if not 0 <= illegal <= length + 1e-8 or not math.isclose(illegal, round(illegal), abs_tol=1e-8):
                return False
            if episode["success"] != (components.get("collect_diamond", 0) == TARGET_ACHIEVEMENT_REWARD):
                return False
            for name, amount in components.items():
                if name not in {"step", "illegal_action", "death"} and amount != bonuses.get(name):
                    return False
                totals[name] = totals.get(name, 0) + amount
            if not math.isclose(sum(components.values()), reward["return"], abs_tol=1e-8):
                return False
            length_sum += length
        global_components = knowledge["reward_components"]
        return (length_sum == ll["transitions_emitted"]
                and set(totals) == set(global_components)
                and all(math.isclose(value, global_components[name], abs_tol=1e-8) for name, value in totals.items())
                and math.isclose(sum(totals.values()), knowledge["cumulative_intrinsic_reward"], abs_tol=1e-8)
                and all(sum(episode[key] for episode in episodes) == ll[field]
                        for key, field in (("success", "target_successes"), ("death", "deaths"), ("truncation", "truncations"))))
    except (KeyError, TypeError, ValueError):
        return False


def check_results_detailed(
    states: Mapping[str, Mapping[str, Any]],
    logs: Mapping[str, Sequence[str]],
    *,
    model_path: str | os.PathLike[str] | None,
    environment_state: Mapping[str, Any] | None,
    required_log_ids: Sequence[str] = (),
    workload: DQNWorkload | None = None,
    expected_device: str | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Check the bounded run as a DQN architecture-feasibility experiment."""
    reasons: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            reasons.append(message)

    def equal(message: str, *values: Any) -> None:
        require(bool(values) and all(value == values[0] for value in values[1:]), message)

    for runtime_id in required_log_ids:
        require(runtime_id in logs, f"missing runtime log: {runtime_id}")
        require(bool(logs.get(runtime_id)), f"runtime log is empty: {runtime_id}")
    if not required_log_ids:
        require(len(logs) >= 2, "agent and environment logs were not both supplied")
    failures = _runtime_failure_lines(logs)
    require(not failures, f"runtime failure log records found: {len(failures)}")

    perceptor = _state(states, "perceptor", reasons)
    actuator = _state(states, "actuator", reasons)
    ll = _state(states, "ll_reasoner", reasons)
    knowledge = _state(states, "knowledge", reasons)
    memory = _state(states, "memory", reasons)
    learner = _state(states, "learner", reasons)
    if expected_device is not None:
        require(
            learner.get("device") == expected_device,
            f"learner device is {learner.get('device')!r}, expected {expected_device!r}",
        )
        if expected_device.startswith("cuda"):
            require(learner.get("cuda_available") is True, "learner did not confirm CUDA availability")
            require(
                isinstance(learner.get("cuda_device_name"), str)
                and bool(learner["cuda_device_name"].strip()),
                "learner did not record its CUDA device name",
            )
    try:
        persisted_workload = DQNWorkload(**ll.get("treatment", {}).get("workload", {}))
        workload = workload or persisted_workload
        require(persisted_workload == workload, "persisted workload differs from requested workload")
        require(ll.get("treatment") == initial_states(workload)["ll_reasoner"]["treatment"],
                "persisted Rainbow treatment differs")
    except (TypeError, ValueError):
        require(False, "invalid workload metadata")
        workload = DEFAULT_WORKLOAD
    environment = environment_state if isinstance(environment_state, Mapping) else {}
    require(environment.get('no_mobs') is True, 'environment must use no_mobs=True')
    require(environment.get('action_masking') is workload.action_masking, 'environment masking differs')
    require(bool(environment), "missing Crafter environment state")
    if environment:
        try:
            json.dumps(environment)
        except (TypeError, ValueError) as exc:
            reasons.append(f"environment state is not JSON-safe: {exc}")

    require(int(perceptor.get("requests", 0)) > 0, "perceptor handled no requests")
    equal(
        "observation path counts differ", perceptor.get("requests"),
        perceptor.get("observations_forwarded"), environment.get("observation_requests"),
        ll.get("observations"),
    )
    require(int(actuator.get("requests", 0)) > 0, "actuator handled no requests")
    equal(
        "action-status path counts differ", actuator.get("requests"),
        actuator.get("statuses_forwarded"), environment.get("statuses"), ll.get("statuses"),
    )
    equal(
        "reasoner/environment native-action counts differ",
        int(ll.get("actions", 0)) + int(ll.get("evaluation_actions", 0)),
        environment.get("native_actions"),
    )
    require(
        environment.get("statuses")
        == int(environment.get("native_actions", 0))
        + int(environment.get("resets", 0))
        - 1,
        "environment reset/action/status counts differ",
    )
    for label, state in {
        "environment": environment, "perceptor": perceptor, "actuator": actuator,
        "knowledge": knowledge, "memory": memory, "learner": learner,
    }.items():
        require(state.get("contract_errors") == 0, f"{label} recorded contract errors")
    for field in ("stack_contract_errors", "cycle_errors", "invalid_inputs"):
        require(ll.get(field) == 0, f"reasoner recorded {field.replace('_', ' ')}")

    transitions = int(ll.get("transitions_emitted", 0))
    actions = int(ll.get("actions", 0))
    observations = int(ll.get("observations", 0))
    initializations = int(ll.get("stack_initializations", 0))
    histogram = ll.get("action_histogram", [])
    require(
        isinstance(histogram, list)
        and len(histogram) == N_ACTIONS
        and all(isinstance(value, int) and value >= 0 for value in histogram),
        "reasoner action histogram is invalid",
    )
    if isinstance(histogram, list):
        require(sum(histogram) == actions, "action histogram does not sum to actions")
    equal("episode/stack initialization counts differ", ll.get("episodes_started"), initializations)
    equal("stack shifts do not match transitions", ll.get("stack_shifts"), transitions)
    equal("policy decisions do not match actions", ll.get("policy_inferences"), actions)
    require(
        observations
        == transitions + initializations + int(ll.get("evaluation_observations", 0))
        + int(ll.get("cutoff_reset_observations", 0)),
        "observation/stack trace is incoherent",
    )
    require(ll.get('cutoff_reset_observations', 0) in ((0,) if workload.synchronized_training else (0, 1)),
            'invalid count of reset observations received at cutoff')
    require(transitions <= actions <= transitions + 1, "action/transition trace is incoherent")
    completed = sum(int(ll.get(field, 0)) for field in (
        "target_successes", "deaths", "truncations"
    ))
    equal("episode outcomes do not partition completions", ll.get("completed_episodes"), completed)
    require(
        int(ll.get("episodes_started", 0)) in {completed, completed + 1},
        "started/completed episode counts are incoherent",
    )

    equal(
        "LL/Knowledge/Memory transition counts differ", transitions,
        knowledge.get("evaluated_transitions"), memory.get("transitions_admitted"),
    )
    warmup = min(transitions, TRAINING_START_THRESHOLD)
    updates = int(ll.get("update_eligible_transitions", 0))
    shutdown = int(ll.get("shutdown_ineligible_transitions", 0))
    require(transitions == warmup + updates + shutdown, "transition eligibility partition is incomplete")
    for classification, expected in (
        (WARMUP, warmup),
        (UPDATE_ELIGIBLE, updates),
        (SHUTDOWN_INELIGIBLE, shutdown),
    ):
        field = f"{classification}_transitions"
        equal(
            f"{classification} admission counts differ", expected,
            knowledge.get(field), memory.get(field),
        )
    if not workload.synchronized_training:
        updates = int(learner.get("training_updates", 0))
        equal("asynchronous update/priority counts differ", updates,
              memory.get("priority_updates"), learner.get("priority_updates_sent"),
              learner.get("priority_acknowledgements"))
        equal("asynchronous batch accounting differs", memory.get("batches_sent"),
              updates + int(learner.get("unused_batches", 0)))
        equal("cancelled batch counts differ", memory.get("cancelled_batches"), learner.get("unused_batches"))
        require(all(s.get("collection_closed") is True for s in (ll, knowledge, memory)),
                "asynchronous collection did not close")
        require(memory.get("sampling_closed") is True and memory.get("closure_acknowledged") is True,
                "replay did not acknowledge closure")
        require(memory.get("pending_batch") is False, "replay batch still pending")
        from .treatment import REPLAY_BUFFER_SIZE
        equal("circular replay retained incorrect size", memory.get("buffer_size"), min(transitions, REPLAY_BUFFER_SIZE))
        equal("circular replay eviction count differs", memory.get("evictions"), max(0, transitions - REPLAY_BUFFER_SIZE))
        equal("replay oldest ID differs", memory.get("oldest_replay_id"), max(1, transitions - REPLAY_BUFFER_SIZE + 1))
        equal("replay newest ID differs", memory.get("newest_replay_id"), transitions)
        equal("published model count differs", ll.get("models_installed"), learner.get("models_published"))
        equal("final actor update differs", ll.get("installed_update"), updates)
        for s, field in ((ll, "last_training_action_started"), (learner, "last_update_started")):
            require(isinstance(s.get(field), (int, float)) and s[field] < s["training_deadline"],
                    f"{field} reached or exceeded training deadline")
    else:
        equal(
            "synchronized update-cycle counts differ", updates,
            ll.get("cycles_started"), ll.get("cycles_completed"), memory.get("batches_sent"),
            learner.get("training_updates"), learner.get("training_completions_sent"),
            ll.get("training_completions_received"),
            memory.get("priority_updates"), learner.get("priority_updates_sent"),
            learner.get("priority_acknowledgements"),
        )
    equal("memory cycle sequence is incomplete", memory.get("next_cycle_id"), int(memory.get("batches_sent", 0)) + 1)
    equal("finalized replay count differs", memory.get("experiences_finalized"), transitions)
    require(memory.get("pending_transitions") == 0, "unfinished three-step replay tail")
    require(sum(memory.get("horizon_counts", [])) == transitions, "replay horizon counts differ")
    require(memory.get("buffer_size", 0) >= TRAINING_START_THRESHOLD, "replay did not reach warm-up threshold")
    require(updates > 0, "learner performed no DQN update")
    require(int(learner.get("models_published", 0)) > 0, "learner published no model")
    require(int(ll.get("models_installed", 0)) > 0, "reasoner installed no model")

    closed_ordinal = ll.get("training_closed_at_transition")
    closed_elapsed = ll.get("training_closed_at_elapsed_seconds")
    require(shutdown == 0, "fixed-workload run emitted a shutdown transition")
    equal("closure transition was not final", closed_ordinal, transitions)
    if workload.synchronized_training:
        equal("training transition budget is not exact", transitions, workload.training_transitions)
        equal("training update budget is not exact", updates, workload.training_updates)
    else:
        require(isinstance(closed_elapsed, (int, float)) and closed_elapsed >= ll['training_deadline'],
                "collection stopped before its time budget")
    valid_elapsed = (
        isinstance(closed_elapsed, (int, float))
        and math.isfinite(float(closed_elapsed)) and float(closed_elapsed) >= 0
    )
    require(valid_elapsed, "closure elapsed time is invalid")
    require(actions == transitions, "training action count differs from transitions")
    require(
        ll.get("post_closure_environment_request_attempts") == 0,
        "an environment request was attempted after closure",
    )
    require(ll.get("phase") == "complete", "reasoner did not complete frozen evaluation")
    require(_reward_trace_valid(ll, knowledge, workload), "episode reward trace is inconsistent")
    evaluation = ll.get("evaluation_cases", [])
    require(isinstance(evaluation, list), "evaluation cases are malformed")
    if isinstance(evaluation, list):
        require(len(evaluation) == len(workload.evaluation_seeds), "frozen evaluation cohort is incomplete")
        require([row.get("seed") for row in evaluation if isinstance(row, Mapping)]
                == list(workload.evaluation_seeds), "evaluation seeds differ from the fixed cohort")
        require(all(isinstance(row, Mapping) and row.get("checkpoint_digest")
                    == learner.get("checkpoint_digest") for row in evaluation),
                "evaluation checkpoint identity is inconsistent")
        require(all(isinstance(row, Mapping)
                    and type(row.get("length")) is int and 0 < row["length"] <= workload.episode_action_limit
                    and all(type(row.get(key)) is bool for key in ("success", "death", "truncation"))
                    and sum(row[key] for key in ("success", "death", "truncation")) == 1
                    for row in evaluation), "invalid evaluation outcomes")
        equal("evaluation action totals differ", ll.get("evaluation_actions"),
              sum(row.get("length", 0) for row in evaluation if isinstance(row, Mapping)))
    equal("environment evaluation seeds differ", environment.get("applied_evaluation_seeds"),
          list(workload.evaluation_seeds))
    equal("environment evaluation resets differ", environment.get("evaluation_resets"), len(workload.evaluation_seeds))
    require(memory.get("evaluation_replay_insertion_attempts") == 0,
            "evaluation attempted replay insertion")
    require(learner.get("frozen") is True, "learner was not frozen for evaluation")
    require(learner.get("post_freeze_update_attempts") == 0,
            "optimization was attempted after freezing")

    checkpoint, checkpoint_reasons = checkpoint_evidence(model_path)
    reasons.extend(checkpoint_reasons)
    require(learner.get("model_saved") is True, "learner did not save the final policy")
    equal("learner recorded the wrong artifact", learner.get("model_artifact"), POLICY_FILENAME)
    equal(
        "saved/checkpoint training steps differ", learner.get("saved_training_steps"),
        learner.get("training_updates"), checkpoint.get("training_steps"),
    )
    equal(
        "saved/evaluation checkpoint digests differ",
        learner.get("checkpoint_digest"), ll.get("frozen_checkpoint_digest"),
        checkpoint.get("sha256"),
    )
    expected_metadata = {
        "frame_shape": FRAME_SHAPE, "frame_stack_size": FRAME_STACK_SIZE,
        "model_image_shape": MODEL_IMAGE_SHAPE,
        "target_achievement": TARGET_ACHIEVEMENT.value,
        "reward": reward_metadata(), "algorithm": algorithm_metadata(workload.synchronized_training),
        "workload": workload.dump(), "protocol_version": PROTOCOL_VERSION,
        "environment": {"no_mobs": True},
    }
    for field, expected in expected_metadata.items():
        value = checkpoint.get(field)
        if field.endswith("shape"):
            value = tuple(value or ())
        equal(f"checkpoint {field.replace('_', ' ')} differs", value, expected)
    return not reasons, reasons, checkpoint


def check_results(
    states: Mapping[str, Mapping[str, Any]],
    logs: Mapping[str, Sequence[str]],
    verbose: bool = False,
    model_path: str | os.PathLike[str] | None = None,
    environment_state: Mapping[str, Any] | None = None,
    required_log_ids: Sequence[str] = (),
    workload: DQNWorkload | None = None,
    expected_device: str | None = None,
) -> bool:
    """Return the architecture-feasibility result and print concise evidence."""
    passed, reasons, _ = check_results_detailed(states, logs, model_path=model_path,
        environment_state=environment_state, required_log_ids=required_log_ids,
        workload=workload, expected_device=expected_device)
    if verbose or not passed:
        ll = states.get(_module_ids()["ll_reasoner"], {})
        learner = states.get(_module_ids()["learner"], {})
        print(f"Target achievement: {TARGET_ACHIEVEMENT.value}")
        print("Intrinsic reward: " f"step={STEP_REWARD}, target={TARGET_ACHIEVEMENT_REWARD}, "
              f"illegal={ILLEGAL_ACTION_PENALTY}")
        print(f"Transitions: {ll.get('transitions_emitted', 0)}")
        print(f"Training updates: {learner.get('training_updates', 0)}")
        print(f"Target successes (non-gate): {ll.get('target_successes', 0)}")
        for reason in reasons:
            print(f"- {reason}")
    return passed


def _workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
            return candidate
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root.")


def _local_mhagenta_root() -> Path:
    root = (_workspace_root().parent / "mhagenta").resolve()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError(f"Local MHAgentA checkout not found at {root}.")
    if 'version = "1.4.12"' not in pyproject.read_text(encoding="utf-8"):
        raise RuntimeError(f"Local MHAgentA checkout at {root} is not version 1.4.12.")
    return root


def _gpu_device(value: str | int | None) -> str | None:
    """Resolve one optional numeric host-GPU assignment."""

    if value is None:
        value = os.environ.get("MHA_EXP_GPU_DEVICE")
    if value is None:
        return None
    device = str(value).strip()
    if not device.isdigit():
        raise ValueError("gpu_device must identify one GPU by a non-negative index")
    return device


def _read_logs(root: Path, runtime_ids: Sequence[str]) -> dict[str, list[str]]:
    logs: dict[str, list[str]] = {}
    for runtime_id in runtime_ids:
        path = (root / runtime_id).with_suffix(".log")
        if path.is_file():
            logs[runtime_id] = path.read_text(encoding="utf-8").splitlines()
    return logs


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    *, workload: DQNWorkload = DEFAULT_WORKLOAD,
    gpu_device: str | int | None = None,
) -> bool:
    """Run one isolated four-frame architecture-feasibility instance."""
    selected_gpu = _gpu_device(gpu_device)
    root = Path(exp_path).resolve()
    crafter_spec = find_spec("mha_env_crafter")
    if crafter_spec is None or crafter_spec.origin is None:
        raise ImportError("Could not locate mha_env_crafter runtime sources.")
    crafter_source = Path(crafter_spec.origin).resolve().parent
    runtime_version = DEFAULT_MHAGENTA_VERSION if mha_version in {"", "latest", "1.4.12"} else mha_version
    if runtime_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f"Experiment 2-2-CR requires runtime {DEFAULT_MHAGENTA_VERSION}.")

    seeder = Seeder(run)
    agent_id, environment_id, exchange = runtime_identity(run)
    states = initial_states(workload)
    ids = _module_ids()
    reasoner_type, knowledge_type, memory_type, learner_type = TestLLReasoner, TestKnowledge, TestMemory, TestLearner
    if not workload.synchronized_training:
        from .async_modules import AsyncReasoner, AsyncKnowledge, AsyncMemory, AsyncLearner
        reasoner_type, knowledge_type, memory_type, learner_type = AsyncReasoner, AsyncKnowledge, AsyncMemory, AsyncLearner
    orchestrator = Orchestrator(
        save_dir=root, step_frequency=0.0, control_frequency=0.0,
        status_frequency=5.0, agent_start_delay=20, exec_duration=workload.duration_seconds,
        save_format="json", log_level=Orchestrator.INFO, save_logs=True,
        no_stdout_logs=False, mas_rmq_uri="localhost:5672",
        mas_rmq_close_on_exit=False,
        mas_rmq_exchange_name=exchange,
        stop_on_agents_term=True,
        gpu_device_ids=selected_gpu if selected_gpu is not None else "none",
    )
    orchestrator.add_agent(
        agent_id=agent_id,
        perceptors=TestPerceptor(module_id=ids["perceptor"],
            initial_state=states["perceptor"], exchange_name=exchange),
        actuators=TestActuator(module_id=ids["actuator"],
            initial_state=states["actuator"], exchange_name=exchange),
        ll_reasoners=reasoner_type(ids["ll_reasoner"], states["ll_reasoner"],
            init_kwargs={"seed": seeder.ll_reasoner, "workload": workload.dump()}),
        knowledge=knowledge_type(ids["knowledge"], states["knowledge"]),
        memory=memory_type(ids["memory"], states["memory"],
            init_kwargs={"seed": seeder.memory, "workload": workload.dump()}),
        learners=learner_type(ids["learner"], states["learner"],
            init_kwargs={
                "seed": seeder.learner,
                "workload": workload.dump(),
                "device": "cuda:0" if selected_gpu is not None else "cpu",
            }),
        requirements_path=Path(__file__).resolve().with_name("requirements.txt"),
        extra_runtime_sources=Path(cast(str, mha_exp_common.__file__)).resolve().parent,
    )
    env_state = environment_initial_state(seeder.environment, action_masking=workload.action_masking)
    orchestrator.add_environment(
        base=TestEnvironment(init_state={"seed": seeder.environment, **env_state}),
        env_id=environment_id,
        exec_duration=workload.duration_seconds + 30.0,
        requirements_path=Path(__file__).resolve().with_name("requirements-env.txt"),
        exchange_name=exchange,
        gpu_device_ids="none",
        extra_runtime_sources=[Path(cast(str, mha_exp_common.__file__)).resolve().parent,
                               crafter_source],
    )
    orchestrator.run(mhagenta_version=runtime_version, force_run=True,
                     local_build=_local_mhagenta_root())

    gathered = {}
    for runtime_id in (agent_id, environment_id):
        gathered.update(gather_states(root / runtime_id, True, no_warnings=True))
    agent_states = gathered.get(agent_id, {})
    saved_environment = gathered.get(environment_id, {}).get(environment_id, {})
    logs = _read_logs(root, (agent_id, environment_id))
    model_path = root / agent_id / SAVE_SUBDIR / POLICY_FILENAME
    result = check_results(
        agent_states, logs, verbose=True, model_path=model_path,
        environment_state=saved_environment, required_log_ids=(agent_id, environment_id),
        workload=workload,
        expected_device="cuda:0" if selected_gpu is not None else "cpu",
    )
    print(f"2-2-CR DQN architecture feasibility: {result}")
    return result


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 3,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    process_only: bool = False,
    *, workload: DQNWorkload | Mapping[str, Any] = DEFAULT_WORKLOAD,
    gpu_device: str | int | None = None,
) -> None:
    """Run or process results, accepting a JSON workload from the main CLI."""

    if isinstance(workload, Mapping):
        workload = DQNWorkload(**dict(workload))
    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [{"execution_id": f"run-{run}", "run_id": run, "factors": {
                    "protocol_version": PROTOCOL_VERSION,
                    "treatment_id": TREATMENT_ID,
                    "training_seed": run,
                    "training_transitions": workload.training_transitions,
                    "training_updates": workload.training_updates,
                }}
                for run in run_ids]
    primary_error: BaseException | None = None
    try:
        run_experiment_batch(
            experiment_id="2-2-CR", title="2-2-CR DQN ARCHITECTURE FEASIBILITY",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=partial(run_experiment, workload=workload, gpu_device=gpu_device),
            process_only=process_only,
            cleanup_before_run=False, stop_on_error=True)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            process_execution_metrics(root, expected_executions=expected)
        except Exception as reporting_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Execution-metrics processing also failed: {type(reporting_error).__name__}"
            )
