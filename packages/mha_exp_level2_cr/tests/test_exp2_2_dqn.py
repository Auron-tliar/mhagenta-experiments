from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mhagenta import ActionStatus, Observation

from mha_exp_level2_cr.exp2_2.modules import (
    DURATION, EVALUATION_SEEDS, K_ACTION, K_ACHIEVEMENTS, K_CLASSIFICATION, K_CYCLE_ID, K_DONE,
    K_ILLEGAL_ACTION, K_NEXT_FRAME, K_REWARD, K_STATE_STACK, K_TERMINAL,
    K_TRANSITION_ORDINAL, REPLAY_BATCH_SIZE, SHUTDOWN_INELIGIBLE,
    TOTAL_TRAINING_TRANSITIONS, TRAINING_SHUTDOWN_MARGIN,
    TRAINING_START_THRESHOLD, TRAINING_UPDATES, UPDATE_ELIGIBLE, WARMUP,
    TestEnvironment as EnvironmentBehavior,
    TestLearner as LearnerBehavior,
    TestLLReasoner as ReasonerBehavior,
    TestMemory as MemoryBehavior,
    make_replay_transition,
    parse_action_status,
)
from mha_exp_level2_cr.exp2_2.policy import (
    ACHIEVEMENTS, CHECKPOINT_FORMAT_VERSION, FRAME_SHAPE, FRAME_STACK_SIZE,
    ILLEGAL_ACTION_PENALTY, MODEL_IMAGE_SHAPE, N_ACTIONS, POLICY_FILENAME,
    STACK_SHAPE, STEP_REWARD, TARGET_ACHIEVEMENT, TARGET_ACHIEVEMENT_REWARD,
    as_rgb_frame, build_q_network, initial_frame_stack, load_policy_checkpoint,
    save_policy_checkpoint, shift_frame_stack, stack_batch,
)
from mha_exp_level2_cr.exp2_2.rewards import evaluate_reward, reward_state
from mha_exp_level2_cr.exp2_2.replay import PrioritizedReplay
from mha_exp_level2_cr.exp2_2.runner import (
    _module_ids, check_results_detailed, environment_initial_state, initial_states,
    runtime_identity,
)


class RecordingOutbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return record


class FakeState(dict[str, Any]):
    def __init__(self, values: dict[str, Any], *, time: float = 1.0) -> None:
        super().__init__(values)
        self.outbox = RecordingOutbox()
        self.time = time


def frame(value: int = 0) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def achievement_counts(**updates: int) -> dict[str, int]:
    counts = {name: 0 for name in ACHIEVEMENTS}
    counts.update(updates)
    return counts


def test_runtime_identity_isolated_from_parallel_bw() -> None:
    agent_id, environment_id, exchange = runtime_identity(7)

    assert agent_id == "exp_agent2_2_cr_7"
    assert environment_id == "exp_env2_2_cr_7"
    assert exchange == "mhagenta-2_2_cr-7"
    assert {agent_id, environment_id}.isdisjoint({"exp_agent2_2_7", "exp_env2_2_7"})


def test_batch_disables_repository_wide_docker_cleanup(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    from mha_exp_level2_cr.exp2_2 import runner

    captured: dict[str, Any] = {}
    monkeypatch.setattr(runner, "run_experiment_batch", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runner, "process_execution_metrics", lambda *_args, **_kwargs: None)

    runner.run_batch(runs=[], exp_path=tmp_path)

    assert captured["cleanup_before_run"] is False


def test_reporting_prefers_isolated_runtime_ids_with_legacy_fallback(tmp_path: Path) -> None:
    from mha_exp_level2_cr.exp2_2.reporting import _runtime_ids

    (tmp_path / "exp_agent2_2_7").mkdir()
    assert _runtime_ids(tmp_path, 7) == ("exp_agent2_2_7", "exp_env2_2_7")

    (tmp_path / "exp_agent2_2_cr_7").mkdir()
    assert _runtime_ids(tmp_path, 7) == ("exp_agent2_2_cr_7", "exp_env2_2_cr_7")


def evidence(*, done: bool = False, achieved: bool = False) -> dict[str, Any]:
    before = {"needs": dict.fromkeys(("health", "food", "drink", "energy"), 9),
              "achievements": achievement_counts(), "sleeping": False}
    after = deepcopy(before)
    if done:
        after["needs"]["health"] = 0
    if achieved:
        after["achievements"][TARGET_ACHIEVEMENT.value] = 1
    return {"before": before, "after": after}


def valid_status(*, illegal: bool = False, done: bool = False) -> ActionStatus:
    return ActionStatus(
        {
            K_ILLEGAL_ACTION: illegal,
            K_DONE: done,
            K_ACHIEVEMENTS: achievement_counts(),
            "reward_evidence": evidence(done=done),
        }
    )


def prepared_reasoner(*, time: float = 1.0) -> tuple[ReasonerBehavior, FakeState]:
    reasoner = ReasonerBehavior(module_id="ll", initial_state={})
    reasoner._actuator_id = "actuator"
    reasoner._perceptor_id = "perceptor"
    reasoner._knowledge_id = "knowledge"
    reasoner._duration = DURATION
    reasoner._frame_stack = initial_frame_stack(frame(1))
    reasoner._previous_action = 2
    reasoner._episode_length = 1
    state = FakeState(initial_states()["ll_reasoner"], time=time)
    state["actions"] = 1
    state["action_histogram"][2] = 1
    state["policy_inferences"] = 1
    state["observations"] = 1
    state["episodes_started"] = 1
    state["stack_initializations"] = 1
    reasoner.log = lambda *_: None  # type: ignore[method-assign]
    return reasoner, state


def test_fixed_scientific_treatment() -> None:
    assert TARGET_ACHIEVEMENT.value == "collect_diamond"
    assert FRAME_STACK_SIZE == 4
    assert MODEL_IMAGE_SHAPE == (12, 64, 64)
    assert TRAINING_START_THRESHOLD == 512
    assert TOTAL_TRAINING_TRANSITIONS == 10_000
    assert TRAINING_UPDATES == 9_488
    assert EVALUATION_SEEDS == tuple(range(920_000, 920_010))
    assert REPLAY_BATCH_SIZE == 128
    assert TRAINING_SHUTDOWN_MARGIN == 10.0
    assert (STEP_REWARD, TARGET_ACHIEVEMENT_REWARD, ILLEGAL_ACTION_PENALTY) == (
        -0.001,
        10.0,
        -0.1,
    )


@pytest.mark.parametrize(
    ("illegal", "achieved", "expected"),
    [
        (False, False, -0.001),
        (True, False, -0.101),
        (False, True, 9.999),
        (True, True, 9.899),
    ],
)
def test_intrinsic_reward_is_exact(
    illegal: bool, achieved: bool, expected: float
) -> None:
    assert sum(evaluate_reward(reward_state(), 1, illegal, evidence(achieved=achieved)).values()) == pytest.approx(expected)


def test_frame_stack_initialization_shift_and_batch_order() -> None:
    stack = initial_frame_stack(frame(1))
    assert stack.shape == STACK_SHAPE
    assert np.all(stack == 1)
    for value in (2, 3, 4, 5):
        stack = shift_frame_stack(stack, frame(value))
    assert [int(item[0, 0, 0]) for item in stack] == [2, 3, 4, 5]
    batch = stack_batch([stack])
    assert batch.shape == (1, *MODEL_IMAGE_SHAPE)
    assert [int(batch[0, index * 3, 0, 0]) for index in range(4)] == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="frame shape"):
        as_rgb_frame(np.zeros((32, 32, 3), dtype=np.uint8))


def test_compact_transition_reconstructs_next_stack() -> None:
    stack = initial_frame_stack(frame(7))
    transition = make_replay_transition(
        Observation(frame(8), value=0.9), stack, 4, True
    ).content
    assert transition[K_STATE_STACK].shape == STACK_SHAPE
    assert transition[K_NEXT_FRAME].shape == FRAME_SHAPE
    assert K_TERMINAL in transition and transition[K_TERMINAL] is True
    reconstructed = shift_frame_stack(
        transition[K_STATE_STACK], transition[K_NEXT_FRAME]
    )
    assert np.all(reconstructed[:-1] == 7)
    assert np.all(reconstructed[-1] == 8)


def test_action_status_contract_is_strict() -> None:
    status = valid_status(illegal=True).status
    assert parse_action_status(status)[:2] == (True, False)
    malformed = dict(status)
    malformed[K_ILLEGAL_ACTION] = 1
    with pytest.raises(TypeError, match="illegal_action"):
        parse_action_status(malformed)


def test_environment_forwards_native_reward_for_frozen_evaluation() -> None:
    class FakeEnvironment:
        def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
            return frame(3), 5.5, False, {
                K_ILLEGAL_ACTION: True,
                K_ACHIEVEMENTS: achievement_counts(),
            }

    adapter = SimpleNamespace(_env=FakeEnvironment(), _action_masking=False,
                              _current_frame=frame(), _snapshot=lambda: evidence()["before"])
    state = {
        "native_actions": 0,
        "statuses": 0,
        "resets": 1,
        "contract_errors": 0,
    }
    _, response = EnvironmentBehavior.on_action(
        adapter, state, "actuator", action=5
    )
    assert response is not None
    assert response[K_ILLEGAL_ACTION] is True
    assert response[K_REWARD] == pytest.approx(5.5)
    assert state["native_actions"] == state["statuses"] == 1


def test_reasoner_initializes_once_then_shifts_every_transition() -> None:
    reasoner = ReasonerBehavior(module_id="ll", initial_state={})
    reasoner._actuator_id = "actuator"
    reasoner._perceptor_id = "perceptor"
    reasoner._knowledge_id = "knowledge"
    reasoner._duration = DURATION
    state = FakeState(initial_states()["ll_reasoner"])
    reasoner.on_observation(state, "perceptor", Observation(frame(1)))  # type: ignore[arg-type]
    assert state["stack_initializations"] == 1
    assert state["stack_shifts"] == 0
    assert state.outbox.calls[-1][0] == "request_action"
    reasoner.on_action_status(state, "actuator", valid_status())  # type: ignore[arg-type]
    reasoner.on_observation(state, "perceptor", Observation(frame(2)))  # type: ignore[arg-type]
    assert state["stack_initializations"] == 1
    assert state["stack_shifts"] == state["transitions_emitted"] == 1
    belief_call = next(call for call in state.outbox.calls if call[0] == "send_beliefs")
    assert belief_call[2][K_CLASSIFICATION] == WARMUP


def test_threshold_transition_starts_one_cycle_and_waits() -> None:
    reasoner, state = prepared_reasoner()
    state["transitions_emitted"] = TRAINING_START_THRESHOLD
    reasoner.on_observation(state, "perceptor", Observation(frame(2)))  # type: ignore[arg-type]
    belief = state.outbox.calls[-1]
    assert belief[0] == "send_beliefs"
    assert belief[2][K_CLASSIFICATION] == UPDATE_ELIGIBLE
    assert belief[2][K_CYCLE_ID] == 1
    assert state["cycles_started"] == state["update_eligible_transitions"] == 1
    assert reasoner._pending_cycle_id == 1
    reasoner.on_model(state, "learner", None, cycle_id=1)  # type: ignore[arg-type]
    assert state["cycles_completed"] == 1
    assert state.outbox.calls[-1][0] == "request_action"


@pytest.mark.parametrize("terminal", [False, True])
def test_final_training_transition_enters_drain(terminal: bool) -> None:
    reasoner, state = prepared_reasoner(
        time=DURATION - TRAINING_SHUTDOWN_MARGIN + 0.01
    )
    state["transitions_emitted"] = TOTAL_TRAINING_TRANSITIONS - 1
    state["cycles_started"] = TRAINING_UPDATES - 1
    reasoner._last_done = terminal
    reasoner.on_observation(state, "perceptor", Observation(frame(2)))  # type: ignore[arg-type]
    assert state["shutdown_ineligible_transitions"] == 0
    assert state["training_closed_at_transition"] == state["transitions_emitted"]
    assert state["phase"] == "drain"
    assert state.outbox.calls[-1][0] == "send_beliefs"
    assert state.outbox.calls[-1][2][K_CLASSIFICATION] == UPDATE_ELIGIBLE


def test_wall_clock_does_not_change_fixed_workload_eligibility() -> None:
    reasoner, state = prepared_reasoner(
        time=DURATION - TRAINING_SHUTDOWN_MARGIN
    )
    state["transitions_emitted"] = TRAINING_START_THRESHOLD
    reasoner.on_observation(state, "perceptor", Observation(frame(2)))  # type: ignore[arg-type]
    assert state["update_eligible_transitions"] == 1
    assert state["shutdown_ineligible_transitions"] == 0


def compact_memory() -> Observation:
    return make_replay_transition(
        Observation(frame(2), value=-0.01), initial_frame_stack(frame(1)), 0, False
    )


def test_memory_honors_update_token_without_second_time_veto() -> None:
    memory = MemoryBehavior(module_id="memory", initial_state={})
    memory.on_init(seed=3)
    memory._learner_id = "learner"
    for ordinal in range(1, TRAINING_START_THRESHOLD + 1):
        item = compact_memory().content
        item["replay_id"] = ordinal
        memory._replay.append(item, boundary=False)
    values = initial_states()["memory"]
    values.update(
        {
            "transitions_admitted": TRAINING_START_THRESHOLD,
            "warmup_transitions": TRAINING_START_THRESHOLD,
            "buffer_size": TRAINING_START_THRESHOLD,
        }
    )
    state = FakeState(values, time=DURATION + 100)
    memory.on_observation_update(
        state,  # type: ignore[arg-type]
        "knowledge",
        [Observation(frame(3), value=-0.01)],
        state_stack=initial_frame_stack(frame(2)),
        action=1,
        terminal=False, boundary=False,
        transition_ordinal=TRAINING_START_THRESHOLD + 1,
        classification=UPDATE_ELIGIBLE,
        cycle_id=1,
    )
    assert state["batches_sent"] == 1
    assert state.outbox.calls[-1][0] == "send_memories"
    assert state.outbox.calls[-1][2][K_CYCLE_ID] == 1


def test_memory_admits_the_last_warmup_transition_without_update() -> None:
    memory = MemoryBehavior(module_id="memory", initial_state={})
    memory.on_init(seed=3)
    values = initial_states()["memory"]
    values.update(
        {
            "transitions_admitted": TRAINING_START_THRESHOLD - 1,
            "warmup_transitions": TRAINING_START_THRESHOLD - 1,
        }
    )
    state = FakeState(values, time=DURATION + 100)
    memory.on_observation_update(
        state,  # type: ignore[arg-type]
        "knowledge",
        [Observation(frame(3), value=-0.01)],
        state_stack=initial_frame_stack(frame(2)),
        action=1,
        terminal=True, boundary=True,
        transition_ordinal=TRAINING_START_THRESHOLD,
        classification=WARMUP,
        cycle_id=None,
    )
    assert state["warmup_transitions"] == TRAINING_START_THRESHOLD
    assert state["transitions_admitted"] == TRAINING_START_THRESHOLD
    assert not state.outbox.calls


def test_one_learner_update_returns_matching_completion() -> None:
    torch = pytest.importorskip("torch")
    learner = LearnerBehavior(module_id="learner", initial_state={})
    learner.on_init(seed=4)
    learner._reasoner_id = "ll"
    state = FakeState(initial_states()["learner"])
    transition = compact_memory().content
    transition.update(replay_id=1, next_frames=[transition[K_NEXT_FRAME]], horizon=1, bootstrap_discount=0.99)
    transition = Observation(transition)
    learner.on_memories(
        state, "memory", [transition] * REPLAY_BATCH_SIZE, cycle_id=1, kind="batch", replay_ids=[1] * REPLAY_BATCH_SIZE, weights=[1.0] * REPLAY_BATCH_SIZE  # type: ignore[arg-type]
    )
    assert state["training_updates"] == 1
    assert state["training_completions_sent"] == 0
    assert state.outbox.calls[-1][0] == "request_memories"
    learner.on_memories(state, "memory", [], kind="priority_ack", cycle_id=1)
    assert state["models_published"] == 1
    assert state["training_completions_sent"] == 1
    assert state.outbox.calls[-1][0] == "send_model"
    assert state.outbox.calls[-1][2][K_CYCLE_ID] == 1
    assert torch is not None


def test_network_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = build_q_network(torch)
    images = torch.rand((2, *MODEL_IMAGE_SHAPE))
    assert model(images).shape == (2, N_ACTIONS)
    path = save_policy_checkpoint(torch, model, tmp_path / POLICY_FILENAME, 17)
    loaded, metadata = load_policy_checkpoint(torch, path)
    with torch.no_grad():
        torch.testing.assert_close(model(images), loaded(images))
    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata["frame_stack_size"] == 4
    assert metadata["target_achievement"] == TARGET_ACHIEVEMENT.value
    assert metadata["reward"]["illegal_action"] == ILLEGAL_ACTION_PENALTY


def test_legacy_single_frame_checkpoint_fails_clearly(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "legacy.pt"
    torch.save({"format_version": 1}, path)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_policy_checkpoint(torch, path)


def complete_fixture(tmp_path: Path) -> tuple[
    dict[str, dict[str, Any]], dict[str, Any], Path, dict[str, list[str]]
]:
    torch = pytest.importorskip("torch")
    raw = initial_states()
    transitions = TOTAL_TRAINING_TRANSITIONS
    evaluation_actions = len(EVALUATION_SEEDS)
    evaluation_observations = evaluation_actions + len(EVALUATION_SEEDS)
    ll = raw["ll_reasoner"]
    ll.update(
        {
            "phase": "complete",
            "observations": transitions + 10 + evaluation_observations,
            "actions": transitions,
            "evaluation_actions": evaluation_actions,
            "evaluation_observations": evaluation_observations,
            "statuses": transitions + evaluation_actions + len(EVALUATION_SEEDS),
            "transitions_emitted": transitions,
            "update_eligible_transitions": TRAINING_UPDATES,
            "shutdown_ineligible_transitions": 0,
            "training_closed_at_transition": transitions,
            "training_closed_at_elapsed_seconds": DURATION - 1,
            "action_histogram": [transitions] + [0] * (N_ACTIONS - 1),
            "episodes_started": 10, "completed_episodes": 10,
            "truncations": 10, "total_episode_length": transitions,
            "stack_initializations": 10,
            "stack_shifts": transitions, "policy_inferences": transitions,
            "models_installed": 96,
            "cycles_started": TRAINING_UPDATES,
            "cycles_completed": TRAINING_UPDATES,
            "training_completions_received": TRAINING_UPDATES,
            "evaluation_index": len(EVALUATION_SEEDS),
        }
    )
    ll["evaluation_cases"] = [{
        "seed": seed, "success": False, "death": False,
        "truncation": True, "return": -10.0, "length": 1,
        "achievement_progression": [], "checkpoint_digest": None,
    } for seed in EVALUATION_SEEDS]
    ll["training_episodes"] = [{"episode_id": index, "length": 1000, "success": False,
                               "death": False, "truncation": True, "achievement_progression": []}
                              for index in range(1, 11)]
    raw["perceptor"].update({
        "requests": ll["observations"],
        "observations_forwarded": ll["observations"],
    })
    raw["actuator"].update({
        "requests": ll["statuses"], "statuses_forwarded": ll["statuses"],
    })
    class_counts = {
        "warmup_transitions": TRAINING_START_THRESHOLD,
        "update_eligible_transitions": TRAINING_UPDATES,
        "shutdown_ineligible_transitions": 0,
    }
    raw["knowledge"].update({
        "evaluated_transitions": transitions, **class_counts,
        "reward_components": {"step": -10.0}, "cumulative_intrinsic_reward": -10.0,
        "episode_rewards": [{"episode_id": index, "return": -1.0, "components": {"step": -1.0}}
                            for index in range(1, 11)],
    })
    raw["memory"].update(
        {
            "transitions_admitted": transitions, **class_counts,
            "buffer_size": transitions, "batches_sent": TRAINING_UPDATES,
            "next_cycle_id": TRAINING_UPDATES + 1,
            "experiences_finalized": transitions, "pending_transitions": 0,
            "priority_updates": TRAINING_UPDATES, "horizon_counts": [10, 10, transitions - 20],
        }
    )
    raw["learner"].update(
        {
            "training_updates": TRAINING_UPDATES,
            "target_syncs": TRAINING_UPDATES // 100,
            "models_published": 96,
            "training_completions_sent": TRAINING_UPDATES,
            "priority_updates_sent": TRAINING_UPDATES,
            "priority_acknowledgements": TRAINING_UPDATES,
            "model_saved": True, "model_artifact": POLICY_FILENAME,
            "saved_training_steps": TRAINING_UPDATES,
            "frozen": True,
        }
    )
    states = {_module_ids()[role]: value for role, value in raw.items()}
    environment = environment_initial_state(1_000)
    environment.update({
        "observation_requests": ll["observations"],
        "native_actions": transitions + evaluation_actions,
        "statuses": ll["statuses"],
        "resets": 1 + len(EVALUATION_SEEDS),
        "evaluation_resets": len(EVALUATION_SEEDS),
        "applied_evaluation_seeds": list(EVALUATION_SEEDS),
    })
    checkpoint = save_policy_checkpoint(
        torch, build_q_network(torch), tmp_path / POLICY_FILENAME,
        TRAINING_UPDATES,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    raw["learner"]["checkpoint_digest"] = digest
    ll["frozen_checkpoint_digest"] = digest
    for row in ll["evaluation_cases"]:
        row["checkpoint_digest"] = digest
    logs = {
        "agent": ["[00][INFO]::[agent][ll]::healthy"],
        "environment": ["[00][INFO]::[environment]::healthy"],
    }
    return states, environment, checkpoint, logs


def test_checker_accepts_coherent_zero_success_architecture(tmp_path: Path) -> None:
    states, environment, checkpoint, logs = complete_fixture(tmp_path)
    passed, reasons, _ = check_results_detailed(
        states,
        logs,
        model_path=checkpoint,
        environment_state=environment,
        required_log_ids=("agent", "environment"),
    )
    assert passed, reasons
    assert states[_module_ids()["ll_reasoner"]]["target_successes"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda states: states[_module_ids()["ll_reasoner"]].update(
            training_closed_at_transition=TOTAL_TRAINING_TRANSITIONS - 1
        ),
        lambda states: states[_module_ids()["memory"]].update(batches_sent=0),
        lambda states: states[_module_ids()["memory"]].update(pending_transitions=1),
        lambda states: states[_module_ids()["learner"]].update(priority_acknowledgements=0),
        lambda states: states[_module_ids()["knowledge"]].update(cumulative_intrinsic_reward=1.0),
        lambda states: states[_module_ids()["knowledge"]].update(
            update_eligible_transitions=0
        ),
    ],
)
def test_checker_rejects_partition_and_closure_failures(
    tmp_path: Path, mutation: Any
) -> None:
    states, environment, checkpoint, logs = complete_fixture(tmp_path)
    mutation(states)
    passed, reasons, _ = check_results_detailed(
        states,
        logs,
        model_path=checkpoint,
        environment_state=environment,
        required_log_ids=("agent", "environment"),
    )
    assert not passed
    assert reasons


@pytest.mark.parametrize(
    "marker",
    ["Caught exception", "Could not send message", "Failed to save state"],
)
def test_checker_rejects_exact_multitag_framework_markers(
    tmp_path: Path, marker: str
) -> None:
    states, environment, checkpoint, logs = complete_fixture(tmp_path)
    logs["agent"].append(f"[00][WARNING]::[agent][module][worker]::{marker}: x")
    passed, reasons, _ = check_results_detailed(
        states,
        logs,
        model_path=checkpoint,
        environment_state=environment,
        required_log_ids=("agent", "environment"),
    )
    assert not passed
    assert any("runtime failure" in reason for reason in reasons)


def test_checker_requires_environment_state_and_both_logs(tmp_path: Path) -> None:
    states, _, checkpoint, logs = complete_fixture(tmp_path)
    del logs["environment"]
    passed, reasons, _ = check_results_detailed(
        states,
        logs,
        model_path=checkpoint,
        environment_state=None,
        required_log_ids=("agent", "environment"),
    )
    assert not passed
    assert any("environment state" in reason for reason in reasons)
    assert any("missing runtime log" in reason for reason in reasons)


def test_all_persistent_states_are_json_safe_and_compact() -> None:
    payload = {"modules": initial_states(), "environment": environment_initial_state(7)}
    json.dumps(payload)
    serialized = json.dumps(payload)
    for forbidden in (
        "model_state_dict", '"replay":', '"optimizer_state":', "frame_stack",
    ):
        assert forbidden not in serialized
