from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mhagenta import Goal, Observation, Orchestrator
from mhagenta.utils import State

from mha_exp_level2_bw.exp2_2.modules import (
    BEHAVIOR_WINDOW_TRANSITIONS,
    EVALUATION_SEEDS,
    EXPECTED_TARGET_SYNCS,
    K_ACTION,
    K_NEXT_STATE,
    K_REWARD,
    K_STATE,
    K_TERMINAL,
    OPTIMIZATION_WINDOW_UPDATES,
    REPLAY_BATCH_SIZE,
    SYNCHRONIZED_TRAINING,
    TARGET_SYNC_STEPS,
    TOTAL_TRAINING_TRANSITIONS,
    TRAINING_UPDATES,
    WARMUP_TRANSITIONS,
    TestGoalGraph as GoalGraphBehavior,
    TestLearner as LearnerBehavior,
    TestLLReasoner as ReasonerBehavior,
    TestMemory as MemoryBehavior,
    goal_from_record,
    goal_to_record,
    intrinsic_reward,
    make_replay_transition,
)
from mha_exp_level2_bw.exp2_2.protocol import DQNProtocol
from mha_exp_level2_bw.exp2_2.policy import (
    MODEL_INPUT_SHAPE,
    N_ACTIONS,
    OBS_SHAPE,
    build_q_network,
    goal_achieved,
    goal_conditioned_observation,
    model_state_fingerprint,
    sample_goal,
    save_policy_checkpoint,
)
from mha_exp_level2_bw.exp2_2.reporting import aggregate_summaries, build_run_summary
from mha_exp_level2_bw.exp2_2.runner import (
    POLICY_FILENAME,
    _module_ids,
    _protocol_provenance,
    check_results_detailed,
    environment_initial_state,
    initial_states,
    runtime_resources,
)


class RecordingOutbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return record


def test_batch_disables_repository_wide_docker_cleanup(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    from mha_exp_level2_bw.exp2_2 import runner

    captured: dict[str, Any] = {}
    monkeypatch.setattr(runner, "run_experiment_batch", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runner, "aggregate_run_directory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "process_execution_metrics", lambda *_args, **_kwargs: None)

    runner.run_batch(runs=[], exp_path=tmp_path, gpu_device=0)

    assert captured["cleanup_before_run"] is False
    assert captured["stop_on_error"] is True
    assert captured["runner"].keywords["gpu_device"] == 0


def test_runtime_resources_are_domain_and_run_isolated() -> None:
    first = runtime_resources(0)
    second = runtime_resources(1)

    assert first != second
    assert all("bw" in value for value in first)
    assert not first[0].startswith("exp_")
    assert not first[1].startswith("exp_")
    with pytest.raises(ValueError, match="non-negative"):
        runtime_resources(-1)


def test_production_run_rejects_missing_gpu_assignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mha_exp_level2_bw.exp2_2 import runner

    monkeypatch.delenv("MHA_EXP_GPU_DEVICE", raising=False)
    with pytest.raises(RuntimeError, match="requires gpu_device"):
        runner.run_experiment(0, tmp_path, protocol=DQNProtocol(), record=False)


def test_gpu_assignment_can_come_from_cli_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mha_exp_level2_bw.exp2_2 import runner

    monkeypatch.setenv("MHA_EXP_GPU_DEVICE", "1")
    assert runner._gpu_device(None) == "1"
    monkeypatch.setenv("MHA_EXP_GPU_DEVICE", "0,1")
    with pytest.raises(ValueError, match="one GPU"):
        runner._gpu_device(None)


def test_standard_orchestrator_pins_one_host_gpu() -> None:
    request = Orchestrator._resolve_gpu_ids(["1"])[0]
    assert request["DeviceIDs"] == ["1"]
    assert request["Capabilities"] == [["gpu"]]


class FakeState(dict[str, Any]):
    def __init__(self, values: dict[str, Any], *, time: float = 1.0) -> None:
        super().__init__(values)
        self.outbox = RecordingOutbox()
        self.time = time


def empty_observation() -> np.ndarray:
    return np.zeros(OBS_SHAPE, dtype=np.uint8)


def observation_with_goal(goal: tuple[int, int]) -> np.ndarray:
    observation = empty_observation()
    observation[3, 4, goal[0]] = 1
    observation[4, 4, goal[1]] = 1
    return observation


def active_goal_record() -> dict[str, Any]:
    return {
        "goal_id": "training-000001",
        "phase": "training",
        "top_block": 5,
        "bottom_block": 9,
        "status": "active",
    }


def test_scientific_protocol_and_sync_default_are_frozen() -> None:
    assert SYNCHRONIZED_TRAINING
    assert WARMUP_TRANSITIONS == 1_280
    assert TOTAL_TRAINING_TRANSITIONS == 10_000
    assert TRAINING_UPDATES == 8_721
    assert TARGET_SYNC_STEPS == 100
    assert EXPECTED_TARGET_SYNCS == 87
    assert EVALUATION_SEEDS == tuple(range(2_200, 2_210))
    assert BEHAVIOR_WINDOW_TRANSITIONS == 200
    assert OPTIMIZATION_WINDOW_UPDATES == 20


def test_goal_detection_conditioning_and_sampling_are_unchanged() -> None:
    goal = (5, 9)
    achieved = observation_with_goal(goal)
    assert goal_achieved(achieved, goal)
    conditioned = goal_conditioned_observation(achieved, goal)
    assert conditioned.shape == MODEL_INPUT_SHAPE
    assert np.all(conditioned[OBS_SHAPE[0], :, goal[0]] == 1)
    assert np.all(conditioned[OBS_SHAPE[0] + 1, :, goal[1]] == 1)
    sampled = sample_goal(np.random.default_rng(7), achieved)
    assert sampled[0] != sampled[1]
    assert not goal_achieved(achieved, sampled)


@pytest.mark.parametrize(
    ("illegal", "achieved", "expected"),
    [(False, False, -0.01), (True, False, -0.11), (False, True, 1.0), (True, True, 0.9)],
)
def test_intrinsic_reward_is_unchanged(illegal: bool, achieved: bool, expected: float) -> None:
    assert intrinsic_reward(illegal, achieved) == pytest.approx(expected)


def test_goal_graph_owns_goal_generation_and_persistent_record_is_json_safe() -> None:
    behavior = GoalGraphBehavior(module_id="goalgraph", initial_state={})
    behavior.on_init(seed=31)
    state = FakeState(initial_states()["goal_graph"])
    behavior.on_goal_request(
        state,  # type: ignore[arg-type]
        "llreasoner",
        phase="training",
        observation=empty_observation(),
    )
    assert state["training_issued"] == 1
    assert state["active_goal"] is not None
    name, args, _ = state.outbox.calls[-1]
    assert name == "send_goals"
    typed_goal = args[1][0]
    assert isinstance(typed_goal, Goal)
    assert goal_to_record(typed_goal) == state["active_goal"]
    assert goal_to_record(goal_from_record(state["active_goal"])) == state["active_goal"]

    persistent = State(
        "agent", "reasoner", lambda: 0.0, None, RecordingOutbox(),
        active_goal=state["active_goal"],
    )
    json.dumps(persistent.dump())


def test_memory_records_terminal_from_the_transition_it_admits() -> None:
    memory = MemoryBehavior(module_id="memory", initial_state={})
    memory.on_init(seed=3)
    memory._learner_id = "learner"
    state_values = initial_states()["memory"]
    state_values["training_transitions"] = TOTAL_TRAINING_TRANSITIONS - 1
    state = FakeState(state_values)
    previous = empty_observation()
    evaluated = Observation(empty_observation(), value=-0.01)
    memory.on_observation_update(
        state,  # type: ignore[arg-type]
        "knowledge",
        [evaluated],
        phase="training",
        previous_observation=previous,
        action=1,
        goal=(5, 9),
        terminal=True,
        transition_index=TOTAL_TRAINING_TRANSITIONS,
        illegal_action=False,
    )
    assert state["training_transitions"] == TOTAL_TRAINING_TRANSITIONS
    assert state["final_training_transition_terminal"] is True
    assert state["credits_earned"] == 1


def test_replay_transition_preserves_terminal_and_goal_conditioning() -> None:
    transition = make_replay_transition(
        Observation(empty_observation(), value=0.9),
        empty_observation(),
        2,
        (5, 9),
        True,
    )
    assert transition is not None
    content = transition.content
    assert content[K_ACTION] == 2
    assert content[K_REWARD] == pytest.approx(0.9)
    assert content[K_TERMINAL] is True
    assert content[K_STATE].shape == MODEL_INPUT_SHAPE
    assert content[K_NEXT_STATE].shape == MODEL_INPUT_SHAPE


@pytest.mark.parametrize("synchronized_training", [None, False])
def test_training_waits_after_warmup_only_in_synchronous_mode(
    synchronized_training: bool | None,
) -> None:
    """Default execution waits for learning; explicit asynchronous execution continues."""
    reasoner = ReasonerBehavior(module_id="reasoner", initial_state={})
    if synchronized_training is None:
        reasoner.on_init(seed=1)
    else:
        reasoner.on_init(seed=1, synchronized_training=synchronized_training)
    reasoner._knowledge_id = "knowledge"
    reasoner._actuator_id = "actuator"
    reasoner._previous_observation = empty_observation()
    reasoner._previous_action = 0
    reasoner._previous_goal = (5, 9)
    state_values = initial_states()["ll_reasoner"]
    state_values["active_goal"] = active_goal_record()
    state_values["training_transitions"] = WARMUP_TRANSITIONS - 2
    state_values["current_episode_length"] = 1
    state = FakeState(state_values)
    reasoner._handle_training_observation(state, empty_observation())  # type: ignore[arg-type]
    assert not reasoner._waiting_for_training
    assert state.outbox.calls[-1][0] == "request_action"

    reasoner._previous_observation = empty_observation()
    reasoner._previous_action = 0
    reasoner._previous_goal = (5, 9)
    reasoner._handle_training_observation(state, empty_observation())  # type: ignore[arg-type]
    assert reasoner._waiting_for_training is (synchronized_training is None)
    if synchronized_training is None:
        reasoner.on_model(state, "learner", None, update=1)  # type: ignore[arg-type]
        assert not reasoner._waiting_for_training
    assert state.outbox.calls[-1][0] == "request_action"


def test_model_fingerprint_binds_tensor_content_not_checkpoint_metadata() -> None:
    torch = pytest.importorskip("torch")
    first = build_q_network(torch)
    second = deepcopy(first)
    assert model_state_fingerprint(first) == model_state_fingerprint(second)
    with torch.no_grad():
        next(second.parameters()).view(-1)[0] += 1
    assert model_state_fingerprint(first) != model_state_fingerprint(second)


def test_one_update_changes_online_but_not_target_and_publishes_update_one() -> None:
    torch = pytest.importorskip("torch")
    learner = LearnerBehavior(module_id="learner", initial_state={})
    learner.on_init(seed=5, synchronized_training=False)
    learner._memory_id = "memory"
    learner._reasoner_id = "reasoner"
    state = FakeState(initial_states()["learner"])
    learner.on_task(state, "reasoner", {"kind": "start"})
    target_before = [parameter.detach().clone() for parameter in learner._target.parameters()]
    transition = Observation(
        {
            K_STATE: np.zeros(MODEL_INPUT_SHAPE, dtype=np.uint8),
            K_ACTION: 0,
            K_REWARD: -0.01,
            K_NEXT_STATE: np.zeros(MODEL_INPUT_SHAPE, dtype=np.uint8),
            K_TERMINAL: False, "bootstrap_discount": 0.99 ** 3, "horizon": 3, "actor_update": 0,
        }
    )
    learner.on_memories(state, "memory", [transition] * REPLAY_BATCH_SIZE,
                        entry_ids=list(range(REPLAY_BATCH_SIZE)), weights=[1.0] * REPLAY_BATCH_SIZE,
                        beta=0.4, mean_replay_age=0.0)  # type: ignore[arg-type]
    assert state["training_updates"] == 1
    assert state["models_published"] == 1
    assert state["first_model_publication_update"] == 1
    assert any(
        not torch.equal(online, target)
        for online, target in zip(learner._online.parameters(), learner._target.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, learner._target.parameters())
    )


def complete_fixture(tmp_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path]:
    torch = pytest.importorskip("torch")
    raw = initial_states()
    ll = raw["ll_reasoner"]
    ll.update({
        "phase": "complete", "training_transitions": TOTAL_TRAINING_TRANSITIONS,
        "observations": 1_500, "actions": 1_500, "statuses": 1_500,
        "action_histogram": [375] * 4, "training_truncations": 6,
        "training_budget_cutoffs": 1, "cutoff_episode_length": 179,
        "training_resets": 7, "training_resets_at_cutoff": 7,
        "training_resets_final": 7, "models_installed": DQNProtocol().publication_count(TRAINING_UPDATES),
        "collection_closed": True,
        "first_model_install_update": 1,
        "final_model_install_update": TRAINING_UPDATES,
        "installed_final_model_update": TRAINING_UPDATES,
        "training_completions_received": TRAINING_UPDATES if SYNCHRONIZED_TRAINING else 0,
        "evaluation_index": len(EVALUATION_SEEDS) - 1,
    })
    ll["behavior_windows"] = [
        {
            "window_index": index,
            "transition_start": index * 200 + 1,
            "transition_end": min((index + 1) * 200, TOTAL_TRAINING_TRANSITIONS),
            "elapsed_seconds": float(index + 1),
            "native_actions": min(200, TOTAL_TRAINING_TRANSITIONS - index * 200),
            "illegal_actions": 0, "successes": 0,
            "truncations": 1 if index < 6 else 0,
            "budget_cutoffs": 1 if index == 6 else 0,
            "cutoff_episode_length": 179 if index == 6 else None,
            "epsilon_start": 0.9, "completed_episodes": 1 if index < 6 else 0,
            "success_rate": 0.0 if index < 6 else None,
            "mean_episode_length": 200.0 if index < 6 else None,
            "epsilon_end": 0.8,
        }
        for index in range(
            (TOTAL_TRAINING_TRANSITIONS + BEHAVIOR_WINDOW_TRANSITIONS - 1)
            // BEHAVIOR_WINDOW_TRANSITIONS
        )
    ]
    ll["evaluation_cases"] = [
        {
            "requested_seed": seed, "applied_seed": seed, "goal": [5, 9],
            "success": False, "steps": 200, "illegal_actions": 0,
            "action_mode": "greedy", "exploratory_actions": 0,
            "elapsed_seconds": 1.0,
        }
        for seed in EVALUATION_SEEDS
    ]
    raw["perceptor"].update({"requests": 1_510, "observations": 1_510})
    raw["actuator"].update({"requests": 1_517, "statuses": 1_517})
    raw["knowledge"].update({"evaluated_observations": TOTAL_TRAINING_TRANSITIONS})
    raw["goal_graph"].update({
        "training_requests": 7, "training_issued": 7,
        "training_truncated": 6, "training_budget_cutoff": 1,
        "evaluation_requests": 10, "evaluation_issued": 10,
        "evaluation_truncated": 10,
    })
    raw["memory"].update({
        "training_transitions": TOTAL_TRAINING_TRANSITIONS,
        "buffer_size": TOTAL_TRAINING_TRANSITIONS + 20_000,
        "credits_earned": TRAINING_UPDATES, "credits_drained": TRAINING_UPDATES,
        "batches_sent": TRAINING_UPDATES,
        "replay_entries": TOTAL_TRAINING_TRANSITIONS + 20_000,
        "her_entries": 20_000, "her_success_entries": 5_000, "her_episodes": 7,
        "her_source_transitions": 5_000, "her_candidate_goals": 30_000,
        "collection_closed": True, "sampling_closed": True, "closure_acknowledged": True,
        "final_training_transition_terminal": True,
    })
    learner = raw["learner"]
    learner.update({
        "training_started": True, "training_updates": TRAINING_UPDATES,
        "target_syncs": EXPECTED_TARGET_SYNCS, "models_published": DQNProtocol().publication_count(TRAINING_UPDATES),
        "batches_received": TRAINING_UPDATES, "replay_closed": True,
        "first_model_publication_update": 1,
        "final_model_publication_update": TRAINING_UPDATES,
        "frozen": True, "freeze_update": TRAINING_UPDATES,
        "evaluation_start_update": TRAINING_UPDATES,
        "final_save_update": TRAINING_UPDATES, "final_model_update": TRAINING_UPDATES,
        "model_saved": True, "model_artifact": POLICY_FILENAME,
        "saved_training_updates": TRAINING_UPDATES,
        "training_completions_sent": TRAINING_UPDATES if SYNCHRONIZED_TRAINING else 0,
    })
    learner["optimization_windows"] = [
        {
            "window_index": index,
            "update_start": index * OPTIMIZATION_WINDOW_UPDATES + 1,
            "update_end": min(
                (index + 1) * OPTIMIZATION_WINDOW_UPDATES,
                TRAINING_UPDATES,
            ),
            "elapsed_seconds": float(index + 1),
            "mean_loss": 0.1, "min_loss": 0.05, "max_loss": 0.2,
            "final_loss": 0.08, "target_syncs": 0,
        }
        for index in range(
            (TRAINING_UPDATES + OPTIMIZATION_WINDOW_UPDATES - 1)
            // OPTIMIZATION_WINDOW_UPDATES
        )
    ]
    model = build_q_network(torch)
    fingerprint = model_state_fingerprint(model)
    learner["final_model_fingerprint"] = fingerprint
    ll["installed_final_model_fingerprint"] = fingerprint
    checkpoint_path = save_policy_checkpoint(
        torch, model, tmp_path / POLICY_FILENAME, TRAINING_UPDATES,
        training_protocol=DQNProtocol().record(), frozen=True,
    )
    states = {_module_ids()[name]: value for name, value in raw.items()}
    environment = environment_initial_state(1_000)
    environment.update({
        "total_resets": 17, "training_resets": 7, "evaluation_resets": 10,
        "applied_evaluation_seeds": list(EVALUATION_SEEDS),
    })
    return states, environment, checkpoint_path


def test_checker_accepts_complete_zero_success_protocol(tmp_path: Path) -> None:
    states, environment, checkpoint = complete_fixture(tmp_path)
    passed, reasons, evidence = check_results_detailed(
        states, [], model_path=checkpoint, environment_state=environment,
    )
    assert passed, reasons
    assert evidence["training_updates"] == TRAINING_UPDATES


@pytest.mark.parametrize(
    "mutation",
    [
        lambda states: states[_module_ids()["memory"]].update(pending_tail=1),
        lambda states: states[_module_ids()["learner"]].update(final_model_fingerprint="0" * 64),
        lambda states: states[_module_ids()["memory"]].update(update_credits=1),
    ],
)
def test_checker_rejects_authoritative_protocol_failures(tmp_path: Path, mutation: Any) -> None:
    states, environment, checkpoint = complete_fixture(tmp_path)
    mutation(states)
    passed, reasons, _ = check_results_detailed(
        states, [], model_path=checkpoint, environment_state=environment,
    )
    assert not passed
    assert reasons


def test_reporting_partitions_protocols_and_preserves_undefined_rates(tmp_path: Path) -> None:
    states, environment, checkpoint_path = complete_fixture(tmp_path)
    passed, reasons, checkpoint = check_results_detailed(
        states, [], model_path=checkpoint_path, environment_state=environment,
    )
    summary = build_run_summary(
        run=0, provenance=_protocol_provenance(), states=states,
        module_ids=_module_ids(), environment_state=environment,
        checker_passed=passed, checker_reasons=reasons, checkpoint=checkpoint,
    )
    other = deepcopy(summary)
    other["run"] = 1
    other["protocol"] = DQNProtocol(synchronized_training=not SYNCHRONIZED_TRAINING).record()
    report = aggregate_summaries([summary, other])
    assert len(report["protocol_groups"]) == 2
    asynchronous = next(
        group for group in report["protocol_groups"]
        if group["protocol"]["scheduling_mode"] == "asynchronous"
    )
    final_window = asynchronous["training"]["behavior_windows"][-1]
    assert final_window["pooled_success_rate"] is None
    assert final_window["success_rate_contributing_runs"] == 0
