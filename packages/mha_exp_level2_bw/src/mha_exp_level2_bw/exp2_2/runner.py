"""Run and validate the bounded 2-2-BW DQN experiment, synchronous by default."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
import hashlib
from importlib import import_module
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal, cast

from mhagenta import Orchestrator

import mha_exp_common
from mha_exp_common.batch import cleanup_run_containers, cleanup_run_images, normalize_runs
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_TORCH_MHAGENTA_VERSION
from mha_exp_common.names import ACTUATOR, GOALGRAPH, KNOWLEDGE, LEARNER, LLREASONER, MEMORY, PERCEPTOR
from mha_exp_common.utils import Seeder, gather_states, module_name

from .modules import (
    BEHAVIOR_WINDOW_TRANSITIONS, EVALUATION_SEEDS, EXPECTED_TARGET_SYNCS,
    MAX_EP_LENGTH, OPTIMIZATION_WINDOW_UPDATES, REPLAY_BATCH_SIZE,
    REPLAY_BUFFER_SIZE, SYNCHRONIZED_TRAINING, TARGET_SYNC_STEPS,
    TOTAL_TRAINING_TRANSITIONS, TRAINING_UPDATES, WARMUP_TRANSITIONS,
    TestActuator, TestEnvironment, TestGoalGraph, TestKnowledge, TestLearner,
    TestLLReasoner, PacedReasoner, TestMemory, TestPerceptor,
    goal_from_record, goal_to_record, intrinsic_reward, make_replay_transition,
)
from .policy import N_ACTIONS, POLICY_FILENAME, load_policy_checkpoint, model_state_fingerprint
from .protocol import DQNProtocol, PROTOCOL_VERSION
from .reporting import (
    aggregate_run_directory, build_run_summary, print_run_summary,
    process_execution_metrics, protocol_provenance, write_run_summary,
)


DURATION = DQNProtocol().duration
ENVIRONMENT_DURATION = DURATION + 30.0
TREATMENT_ID = "2-2-bw-rainbow-her"
SAVE_SUBDIR = Orchestrator.SAVE_SUBDIR
RECORD: Literal["all", "first", "none"] = "all"
DEFAULT_MHAGENTA_VERSION = DEFAULT_TORCH_MHAGENTA_VERSION

# Small compatibility surface used by playback/checkpoint callers.
K_TRAINING_STEPS = "training_updates"
K_TARGET_SYNCS = "target_syncs"
K_MODELS_SENT = "models_published"
K_TRAINING_COMPLETIONS_SENT = "training_completions_sent"
K_MODEL_SAVED = "model_saved"
K_MODEL_ARTIFACT = "model_artifact"
K_SAVED_TRAINING_STEPS = "saved_training_updates"

LOG_PATTERN = re.compile(
    r"^\[(?P<time>[^\]]+)\]\[(?P<level>[^\]]+)\]::"
    r"\[(?P<sender>[^\]]+)\]::(?P<message>.*)$",
    re.IGNORECASE,
)


def initial_states(protocol: DQNProtocol | None = None) -> dict[str, dict[str, Any]]:
    """Return fresh JSON-safe persistent state for all seven modules."""

    protocol = protocol or DQNProtocol()
    goal_counts = {
        f"{phase}_{field}": 0
        for phase in ("training", "evaluation")
        for field in ("requests", "issued", "succeeded", "truncated", "budget_cutoff")
    }
    return {
        "perceptor": {"requests": 0, "observations": 0},
        "actuator": {"requests": 0, "statuses": 0},
        "ll_reasoner": {
            "treatment": {"protocol_version": PROTOCOL_VERSION, "treatment_id": TREATMENT_ID,
                          "protocol": protocol.record()},
            "protocol": protocol.record(), "collection_closed": False,
            "training_deadline": protocol.training_seconds, "evaluation_deadline": None,
            "last_training_action_started": None, "last_training_reset_started": None,
            "actor_update": 0, "failure_reason": None,
            "collection_pauses": 0,
            "phase": "training", "training_transitions": 0,
            "phase_timestamps": {
                "training_started": None,
                "training_finished": None,
                "drain_started": None,
                "evaluation_started": None,
                "complete": None,
            },
            "observations": 0, "actions": 0, "statuses": 0,
            "action_histogram": [0] * N_ACTIONS,
            "training_successes": 0, "training_truncations": 0,
            "training_budget_cutoffs": 0, "cutoff_episode_length": None,
            "active_goal": None, "current_episode_length": 0,
            "training_resets": 1, "training_resets_at_cutoff": None,
            "training_resets_final": None, "reset_seed_mismatches": 0,
            "behavior_windows": [], "behavior_current": None,
            "evaluation_cases": [], "evaluation_index": 0,
            "evaluation_illegal_actions": 0, "models_installed": 0,
            "first_model_install_update": None, "final_model_install_update": None,
            "installed_final_model_fingerprint": None,
            "installed_final_model_update": None,
            "training_completions_received": 0, "invalid_observations": 0,
        },
        "goal_graph": {"goal_sequence": 0, "active_goal": None, **goal_counts},
        "knowledge": {"evaluated_observations": 0, "intrinsic_reward": 0.0},
        "memory": {
            "training_transitions": 0, "buffer_size": 0,
            "protocol": protocol.record(), "collection_closed": False, "collection_closed_at": None,
            "sampling_closed": False, "closure_acknowledged": False,
            "replay_entries": 0, "evictions": 0, "stale_priority_feedback": 0, "pending_tail": 0,
            "her_entries": 0, "her_success_entries": 0,
            "her_episodes": 0, "her_source_transitions": 0,
            "her_candidate_goals": 0, "pending_her_episode": 0,
            "credits_earned": 0, "credits_drained": 0, "update_credits": 0,
            "batches_sent": 0, "pending_request": False,
            "final_training_transition_terminal": None,
            "evaluation_replay_insertion_attempts": 0,
            "rejected_post_budget_transitions": 0,
        },
        "learner": {
            "training_started": False, "training_updates": 0,
            "protocol": protocol.record(), "training_started_at": None,
            "training_deadline": protocol.training_seconds, "stopping": False, "replay_closed": False,
            "last_update_started": None, "last_update_finished": None,
            "failure_reason": None, "batches_received": 0, "unused_batches": 0,
            "target_syncs": 0, "models_published": 0,
            "first_model_publication_update": None,
            "final_model_publication_update": None,
            "training_completions_sent": 0, "frozen": False,
            "freeze_update": None, "evaluation_start_update": None,
            "final_save_update": None, "post_freeze_update_attempts": 0,
            "optimization_windows": [], "optimization_current": None,
            "next_evaluation_snapshot": protocol.evaluation_interval_seconds,
            "evaluation_snapshots": [],
            "final_model_fingerprint": None, "final_model_update": None,
            "model_saved": False, "model_artifact": "",
            "saved_training_updates": 0,
            "device": None, "cuda_available": None, "cuda_device_name": None,
            "torch_version": None, "cuda_runtime": None,
        },
    }


def environment_initial_state(seed: int) -> dict[str, Any]:
    """Return compact JSON-safe state for the environment adapter."""

    return {
        "initial_seed": seed, "total_resets": 1, "training_resets": 1,
        "evaluation_resets": 0, "applied_evaluation_seeds": [],
        "reset_seed_mismatches": 0, "unclassified_resets": 0,
        "native_actions": 0, "observation_requests": 0,
    }


def _module_ids() -> dict[str, str]:
    return {
        "perceptor": module_name(PERCEPTOR, 0),
        "actuator": module_name(ACTUATOR, 0),
        "ll_reasoner": module_name(LLREASONER, 0),
        "goal_graph": module_name(GOALGRAPH, 0),
        "knowledge": module_name(KNOWLEDGE, 0),
        "memory": module_name(MEMORY, 0),
        "learner": module_name(LEARNER, 0),
    }


def runtime_resources(run: int) -> tuple[str, str, str]:
    """Return BW-specific Docker identities and a per-run RabbitMQ exchange."""

    if type(run) is not int or run < 0:
        raise ValueError("run must be a non-negative integer")
    return (
        f"mha_bw_agent_2_2_{run}",
        f"mha_bw_env_2_2_{run}",
        f"mhagenta.2-2-bw.run-{run}",
    )


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


def _protocol_provenance(protocol: DQNProtocol | None = None) -> dict[str, Any]:
    """Describe the selected algorithm and mode without inferring them from results."""
    return (protocol or DQNProtocol()).record()


def _failure_logs(logs: Sequence[str]) -> list[str]:
    failures: list[str] = []
    markers = (
        "traceback", "exceptiongroup", "caught exception", "failed to send",
        "failed send", "failed to save state",
    )
    for line in logs:
        match = LOG_PATTERN.match(line.strip())
        if match:
            level = match.group("level").lower()
            message = match.group("message").lower()
            if level in {"error", "critical"} or any(marker in message for marker in markers):
                failures.append(line.rstrip())
        elif any(marker in line.lower() for marker in markers):
            failures.append(line.rstrip())
    return failures


def checkpoint_evidence(
    model_path: str | os.PathLike[str] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the final artifact and independently identify its model state."""

    if model_path is None:
        return {}, ["final policy checkpoint path was not supplied"]
    path = Path(model_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        return {"path": str(path)}, [f"final policy checkpoint is missing or empty: {path}"]
    record: dict[str, Any] = {
        "filename": path.name, "file_size": path.stat().st_size,
        "file_checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if find_spec("torch") is None:
        return record, ["torch is unavailable for host-side checkpoint validation"]
    try:
        model, metadata = load_policy_checkpoint(import_module("torch"), path)
        record.update({
            "training_updates": metadata["training_steps"],
            "model_fingerprint": model_state_fingerprint(model),
            "frozen_for_evaluation": metadata.get("frozen_for_evaluation", False),
            "training_protocol": metadata.get("training_protocol"),
        })
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return record, [f"could not validate final policy checkpoint: {exc}"]
    return record, []


def _state(
    states: Mapping[str, Mapping[str, Any]], key: str, reasons: list[str],
) -> Mapping[str, Any]:
    module_id = _module_ids()[key]
    value = states.get(module_id)
    if not isinstance(value, Mapping):
        reasons.append(f"missing saved module state: {module_id}")
        return {}
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        reasons.append(f"module state is not JSON-safe ({module_id}): {exc}")
    return value


def check_results_detailed(
    states: Mapping[str, Mapping[str, Any]],
    logs: Sequence[str],
    *,
    model_path: str | os.PathLike[str] | None,
    environment_state: Mapping[str, Any] | None,
    protocol: DQNProtocol | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Check topology, exact protocol, isolation, artifact identity, and health."""

    protocol = protocol or DQNProtocol()
    reasons: list[str] = []
    failures = _failure_logs(logs)
    if failures:
        reasons.append(f"runtime failure log records found: {len(failures)}")
    perceptor = _state(states, "perceptor", reasons)
    actuator = _state(states, "actuator", reasons)
    ll = _state(states, "ll_reasoner", reasons)
    goal_graph = _state(states, "goal_graph", reasons)
    knowledge = _state(states, "knowledge", reasons)
    memory = _state(states, "memory", reasons)
    learner = _state(states, "learner", reasons)
    environment = environment_state if isinstance(environment_state, Mapping) else {}
    if not environment:
        reasons.append("missing Blocks World environment state")
    else:
        try:
            json.dumps(environment)
        except (TypeError, ValueError) as exc:
            reasons.append(f"environment state is not JSON-safe: {exc}")

    def require(condition: bool, message: str) -> None:
        if not condition:
            reasons.append(message)

    require(int(perceptor.get("requests", 0)) > 0, "perceptor handled no requests")
    require(perceptor.get("requests") == perceptor.get("observations"), "perceptor request/observation counts differ")
    require(int(actuator.get("requests", 0)) > 0, "actuator handled no requests")
    require(actuator.get("requests") == actuator.get("statuses"), "actuator request/status counts differ")
    transitions = int(memory.get("training_transitions", 0))
    updates = int(learner.get("training_updates", 0))
    require(transitions >= protocol.warmup_transitions and updates > 0, "training did not reach a valid update")
    if protocol.min_updates_per_transition:
        require(updates >= protocol.min_updates_per_transition * max(0, transitions - protocol.warmup_transitions),
                'collection exceeded the configured learning ratio')
    for owner in (ll, memory, learner):
        require(owner.get("protocol") == protocol.record(), "module treatment differs from requested protocol")
    require(not ll.get("failure_reason") and not learner.get("failure_reason"), "module reported training failure")
    require(knowledge.get("evaluated_observations") == transitions, "knowledge transition count differs")
    require(ll.get("training_transitions") == transitions, "reasoner/memory transition counts differ")
    if protocol.synchronized_training:
        require(transitions == protocol.total_training_transitions, "training-transition count is not exact")
        require(updates == protocol.training_updates, "learner update count is not exact")
        require(memory.get("credits_earned") == updates == memory.get("credits_drained"), "synchronous credits are not exact")
        require(learner.get("unused_batches") == 0, "synchronous batch was not optimized")
    else:
        require(memory.get("credits_earned") == memory.get("credits_drained") == 0, "asynchronous mode accumulated update debt")
        deadline = ll.get("training_deadline")
        require(deadline == learner.get("training_deadline"), "training deadlines differ")
        started = ll.get("phase_timestamps", {}).get("training_started")
        finished = ll.get("phase_timestamps", {}).get("training_finished")
        require(isinstance(started, (int, float)) and deadline == started + protocol.training_seconds, "training deadline is wrong")
        require(isinstance(finished, (int, float)) and isinstance(deadline, (int, float)) and finished >= deadline,
                "asynchronous collection closed before deadline")
        for timestamp in (ll.get("last_training_action_started"), ll.get("last_training_reset_started"), learner.get("last_update_started")):
            require(timestamp is None or (isinstance(deadline, (int, float)) and timestamp < deadline), "training operation started after deadline")
    require(memory.get("update_credits") == 0, "memory retained undrained credits")
    require(memory.get("batches_sent") == learner.get("batches_received") == updates + int(learner.get("unused_batches", 0)), "replay batch counts do not reconcile")
    require(memory.get("pending_request") is False, "memory retained a learner request")
    for field in ("collection_closed", "sampling_closed", "closure_acknowledged"):
        require(memory.get(field) is True, f"memory {field} is not confirmed")
    require(ll.get("collection_closed") is True and learner.get("replay_closed") is True, "collection/replay closure is incomplete")
    require(memory.get("pending_tail") == 0, "n-step tail was not flushed")
    require(memory.get("pending_her_episode") == 0, "HER retained an unfinished episode")
    her_entries = int(memory.get("her_entries", 0))
    replay_entries = int(memory.get("replay_entries", 0))
    require(memory.get("her_episodes", 0) > 0, "HER processed no episodes")
    her_sources = int(memory.get("her_source_transitions", 0))
    require(0 < her_sources <= transitions, "HER contributing-transition count is invalid")
    require(memory.get("her_episodes") == goal_graph.get("training_issued"),
            "HER episode count differs from issued training goals")
    require(memory.get("her_candidate_goals", 0) >= her_entries, "HER candidate accounting is invalid")
    require(her_entries > 0, "HER created no relabeled entries")
    require(her_entries <= her_sources * protocol.her_future_goals,
            "HER created more entries than its per-source limit")
    require(0 < memory.get("her_success_entries", 0) <= her_entries,
            "HER created no successful relabeled targets")
    require(replay_entries == transitions + her_entries, "real and HER replay entries do not reconcile")
    require(memory.get("buffer_size") == min(replay_entries, protocol.replay_capacity), "resident replay size is wrong")
    require(memory.get("evictions") == max(0, replay_entries - protocol.replay_capacity), "FIFO eviction count is wrong")
    require(memory.get("evaluation_replay_insertion_attempts") == 0, "evaluation attempted replay insertion")
    require(memory.get("rejected_post_budget_transitions") == 0, "post-budget transition was attempted")
    require(learner.get("target_syncs") == updates // protocol.target_sync_steps, "target-sync count is wrong")
    publications = protocol.publication_count(updates)
    require(learner.get("models_published") == publications == ll.get("models_installed"), "model publication/install counts differ from cadence")
    require(learner.get("first_model_publication_update") == ll.get("first_model_install_update") == 1, "first publication/install is wrong")
    require(learner.get("final_model_publication_update") == ll.get("final_model_install_update") == updates, "final publication/install is wrong")
    require(learner.get("frozen") is True, "learner did not freeze")
    for field in ("freeze_update", "evaluation_start_update", "final_save_update"):
        require(learner.get(field) == updates, f"learner {field} boundary is wrong")
    require(learner.get("post_freeze_update_attempts") == 0, "post-freeze update attempted")
    completions = updates if protocol.synchronized_training else 0
    require(learner.get("training_completions_sent") == ll.get("training_completions_received") == completions,
            "training completion counts are wrong")

    require(ll.get("phase") == "complete", "reasoner final phase is not complete")
    require(ll.get("active_goal") is None, "reasoner retained an active goal")
    require(goal_graph.get("active_goal") is None, "goal graph retained an active goal")
    require(
        int(goal_graph.get("training_issued", 0))
        == sum(int(goal_graph.get(f"training_{field}", 0)) for field in ("succeeded", "truncated", "budget_cutoff")),
        "training goals were not all closed exactly once",
    )
    require(ll.get("training_resets_at_cutoff") == ll.get("training_resets_final"), "an unseeded reset occurred after training cutoff")
    require(ll.get("training_resets_final") == environment.get("training_resets"), "reasoner/environment training reset counts differ")
    require(environment.get("unclassified_resets") == 0, "environment recorded an unclassified reset")
    require(environment.get("reset_seed_mismatches") == 0, "environment recorded a reset-seed mismatch")
    require(ll.get("reset_seed_mismatches") == 0, "reasoner recorded a reset-seed mismatch")
    require(environment.get("applied_evaluation_seeds") == list(protocol.evaluation_seeds), "environment applied evaluation seeds differ from the cohort")

    behavior = ll.get("behavior_windows", [])
    expected = (transitions + protocol.behavior_window_transitions - 1) // protocol.behavior_window_transitions
    require(len(behavior) == expected, "behavioral dynamics windows are incomplete")
    for index, row in enumerate(behavior):
        start = index * protocol.behavior_window_transitions + 1
        end = min(start + protocol.behavior_window_transitions - 1, transitions)
        require(row.get("transition_start") == start and row.get("transition_end") == end, f"behavior window {index} has wrong boundaries")
    optimization = learner.get("optimization_windows", [])
    expected = (updates + protocol.optimization_window_updates - 1) // protocol.optimization_window_updates
    require(len(optimization) == expected, "optimization dynamics windows are incomplete")
    for index, row in enumerate(optimization):
        start = index * protocol.optimization_window_updates + 1
        end = min(start + protocol.optimization_window_updates - 1, updates)
        require(row.get("update_start") == start and row.get("update_end") == end, f"optimization window {index} has wrong boundaries")

    evaluation = ll.get("evaluation_cases", [])
    if protocol.action_masking:
        require(all(row.get('illegal_actions') == 0 for row in [*behavior, *evaluation]),
                'masked run contains illegal actions')
    if protocol.min_updates_per_transition:
        for row in optimization:
            require(all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ('unweighted_loss', 'gradient_norm', 'weight_min', 'weight_mean', 'weight_max')),
                    'missing or invalid learning-strength diagnostics')
    require(len(evaluation) == len(protocol.evaluation_seeds), "fixed evaluation cohort is incomplete")
    require([row.get("requested_seed") for row in evaluation] == list(protocol.evaluation_seeds), "requested evaluation seeds differ from the cohort")
    require([row.get("applied_seed") for row in evaluation] == list(protocol.evaluation_seeds), "applied evaluation seeds differ from the cohort")
    require(all(row.get("action_mode") == "greedy" for row in evaluation), "evaluation was not wholly greedy")
    require(all(row.get("exploratory_actions") == 0 for row in evaluation), "evaluation recorded exploratory actions")
    require(goal_graph.get("evaluation_issued") == len(protocol.evaluation_seeds), "goal graph did not issue one goal per evaluation seed")
    require(
        int(goal_graph.get("evaluation_succeeded", 0)) + int(goal_graph.get("evaluation_truncated", 0)) == len(protocol.evaluation_seeds),
        "goal graph did not close every evaluation goal",
    )

    checkpoint, artifact_reasons = checkpoint_evidence(model_path)
    reasons.extend(artifact_reasons)
    require(checkpoint.get("training_protocol") == protocol.record(), "checkpoint treatment identity differs")
    require(checkpoint.get("frozen_for_evaluation") is True, "checkpoint is not frozen for evaluation")
    require(learner.get("model_saved") is True, "learner did not report a saved model")
    require(learner.get("model_artifact") == POLICY_FILENAME, "learner reported the wrong model filename")
    require(learner.get("saved_training_updates") == updates, "saved model update count is wrong")
    require(checkpoint.get("filename") == POLICY_FILENAME, "checkpoint filename is wrong")
    require(checkpoint.get("training_updates") == updates, "checkpoint update metadata is wrong")
    fingerprints = {
        learner.get("final_model_fingerprint"),
        ll.get("installed_final_model_fingerprint"),
        checkpoint.get("model_fingerprint"),
    }
    require(None not in fingerprints and len(fingerprints) == 1, "learner, reasoner, and checkpoint model fingerprints differ")
    require(learner.get("final_model_update") == updates, "learner fingerprint update is wrong")
    require(ll.get("installed_final_model_update") == updates, "reasoner fingerprint update is wrong")
    return not reasons, reasons, checkpoint


def check_results(
    states: Mapping[str, Mapping[str, Any]],
    logs: Sequence[str],
    verbose: bool = False,
    model_path: str | os.PathLike[str] | None = None,
    environment_state: Mapping[str, Any] | None = None,
) -> bool:
    """Compatibility wrapper returning only the experiment acceptance result."""

    passed, reasons, _ = check_results_detailed(
        states, logs, model_path=model_path, environment_state=environment_state,
    )
    if verbose or not passed:
        for reason in reasons:
            print(reason)
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


def _read_logs(root: Path, runtime_ids: Sequence[str]) -> list[str]:
    logs: list[str] = []
    for runtime_id in runtime_ids:
        path = (root / runtime_id).with_suffix(".log")
        if path.is_file():
            logs.extend(path.read_text(encoding="utf-8").splitlines())
    return logs


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    *, protocol: DQNProtocol | None = None, record: bool | None = None,
    gpu_device: str | int | None = None,
) -> bool:
    """Run one isolated protocol instance and write its descriptive report."""

    protocol = protocol or DQNProtocol()
    selected_gpu = _gpu_device(gpu_device)
    if selected_gpu is None and not protocol.smoke:
        raise RuntimeError(
            "Production 2-2-BW requires gpu_device or MHA_EXP_GPU_DEVICE"
        )
    root = Path(exp_path).resolve()
    bw_spec = find_spec("mha_env_blocksworld")
    if bw_spec is None or bw_spec.origin is None:
        raise ImportError("Could not locate mha_env_blocksworld for runtime sources.")
    bw_runtime_source = Path(bw_spec.origin).resolve().parent
    seeder = Seeder(run)
    experiment_agent, experiment_environment, exchange_name = runtime_resources(run)
    states = initial_states(protocol)
    ids = _module_ids()

    cleanup_run_containers(
        experiment_agent, experiment_environment, phase="pre_run"
    )
    cleanup_run_images(
        experiment_agent, experiment_environment, phase="pre_run"
    )
    orchestrator = Orchestrator(
        save_dir=root, step_frequency=0.0, control_frequency=0.0,
        status_frequency=5.0, agent_start_delay=20, exec_duration=protocol.duration,
        save_format="json", log_level=Orchestrator.INFO, save_logs=True,
        no_stdout_logs=False, mas_rmq_uri="localhost:5672",
        mas_rmq_close_on_exit=False,
        mas_rmq_exchange_name=exchange_name, stop_on_agents_term=True,
        gpu_device_ids=[selected_gpu] if selected_gpu is not None else "none",
    )
    orchestrator.add_agent(
        agent_id=experiment_agent,
        perceptors=TestPerceptor(
            module_id=ids["perceptor"],
            initial_state=states["perceptor"],
            exchange_name=exchange_name,
        ),
        actuators=TestActuator(
            module_id=ids["actuator"],
            initial_state=states["actuator"],
            exchange_name=exchange_name,
        ),
        ll_reasoners=(PacedReasoner if protocol.min_updates_per_transition else TestLLReasoner)(
            ids["ll_reasoner"], states["ll_reasoner"],
            init_kwargs={"seed": seeder.ll_reasoner, "protocol": protocol},
        ),
        goal_graphs=TestGoalGraph(
            ids["goal_graph"], states["goal_graph"], init_kwargs={"seed": seeder.goal_graph},
        ),
        knowledge=TestKnowledge(ids["knowledge"], states["knowledge"]),
        memory=TestMemory(ids["memory"], states["memory"], init_kwargs={"seed": seeder.memory, "protocol": protocol}),
        learners=TestLearner(
            ids["learner"], states["learner"],
            init_kwargs={
                "seed": seeder.learner,
                "protocol": protocol,
                "device": "cuda:0" if selected_gpu is not None else "cpu",
            },
        ),
        requirements_path=Path(__file__).resolve().with_name("requirements.txt"),
        extra_runtime_sources=Path(cast(str, mha_exp_common.__file__)).resolve().parent,
    )

    if record is None:
        record = RECORD == "all" or (RECORD == "first" and run == 0)
    saved_env_initial = environment_initial_state(seeder.environment)
    orchestrator.add_environment(
        base=TestEnvironment(init_state={
            "seed": seeder.environment, "record": record, **saved_env_initial,
        }),
        env_id=experiment_environment, exec_duration=protocol.duration + 30.0,
        requirements_path=Path(__file__).resolve().with_name("requirements-env.txt"),
        exchange_name=exchange_name,
        gpu_device_ids="none",
        extra_runtime_sources=[
            Path(cast(str, mha_exp_common.__file__)).resolve().parent,
            bw_runtime_source,
        ],
    )
    runtime_version = DEFAULT_MHAGENTA_VERSION if mha_version in {"", "latest", "1.4.12"} else mha_version
    if runtime_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f"Experiment 2-2-BW requires runtime {DEFAULT_MHAGENTA_VERSION}.")
    try:
        orchestrator.run(
            mhagenta_version=runtime_version, force_run=True,
            local_build=_local_mhagenta_root(),
        )

        gathered = {}
        for identity in (experiment_agent, experiment_environment):
            gathered.update(gather_states(root / identity, single_agent=True, no_warnings=True))
        agent_states = gathered.get(experiment_agent, {})
        environment_state = gathered.get(experiment_environment, {}).get(experiment_environment, {})
        logs = _read_logs(root, (experiment_agent, experiment_environment))
        model_path = root / experiment_agent / SAVE_SUBDIR / POLICY_FILENAME
        passed, reasons, checkpoint = check_results_detailed(
            agent_states, logs, model_path=model_path, environment_state=environment_state,
            protocol=protocol,
        )
        learner_state = agent_states.get(ids["learner"], {})
        expected_device = "cuda:0" if selected_gpu is not None else "cpu"
        if learner_state.get("device") != expected_device:
            passed = False
            reasons.append(
                f"learner device is {learner_state.get('device')!r}, expected {expected_device!r}"
            )
        summary = build_run_summary(
            run=run, provenance=_protocol_provenance(protocol), states=agent_states,
            module_ids=ids, environment_state=environment_state,
            checker_passed=passed, checker_reasons=reasons, checkpoint=checkpoint,
        )
        write_run_summary(root / experiment_agent, summary)
        print_run_summary(summary)
        if not passed:
            for reason in reasons:
                print(f"- {reason}")
        return passed
    finally:
        cleanup_run_containers(
            experiment_agent, experiment_environment, phase="post_run"
        )
        cleanup_run_images(
            experiment_agent, experiment_environment, phase="post_run"
        )


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 3,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    process_only: bool = False,
    *, protocol: DQNProtocol | Mapping[str, Any] | None = None,
    gpu_device: str | int | None = None,
) -> None:
    """Run or process a batch, accepting complete protocol records from the CLI."""

    if isinstance(protocol, Mapping):
        protocol = DQNProtocol.from_record(protocol)
    protocol = protocol or DQNProtocol()
    normalized, _ = normalize_runs(runs)
    attempted = len(normalized)
    root = Path(exp_path).resolve()
    expected = [{"execution_id": f"run-{run}", "run_id": run, "factors": {
                    "protocol_version": PROTOCOL_VERSION,
                    "treatment_id": TREATMENT_ID,
                    "training_seed": run,
                    "protocol": protocol.record(),
                }}
                for run in normalized]
    primary_error: BaseException | None = None
    try:
        run_experiment_batch(
            experiment_id="2-2", title="ARCHITECTURE 2-2-BW DQN FEASIBILITY TEST",
            runs=normalized, exp_path=exp_path, mha_version=mha_version,
            runner=partial(
                run_experiment, protocol=protocol, gpu_device=gpu_device
            ),
            process_only=process_only,
            cleanup_before_run=False,
            stop_on_error=True,
        )
        aggregate_run_directory(root, attempted_runs=attempted)
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
                "Execution-metrics processing also failed: "
                f"{type(reporting_error).__name__}"
            )
