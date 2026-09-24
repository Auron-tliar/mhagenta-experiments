"""Numerical and message-boundary coverage for both Rainbow execution modes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mha_exp_level2_bw.exp2_2.modules import TestLearner as Learner, TestMemory as Memory
from mha_exp_level2_bw.exp2_2.modules import TestLLReasoner as Reasoner
from mha_exp_level2_bw.exp2_2.policy import (
    MODEL_INPUT_SHAPE,
    model_state_fingerprint,
    load_policy_checkpoint,
)
from mha_exp_level2_bw.exp2_2.protocol import DQNProtocol
from mha_exp_level2_bw.exp2_2.replay import (
    HindsightRelabeler,
    NStepReturns,
    PrioritizedReplay,
    achieved_on_relations,
)
from mha_exp_level2_bw.exp2_2.runner import initial_states
from mha_exp_level2_bw.exp2_2.reporting import (
    _aligned_optimization, aggregate_summaries, build_run_summary, protocol_provenance,
)
from mha_exp_level2_bw.exp2_2.runner import _module_ids, checkpoint_evidence
from mhagenta import Observation

from test_exp2_2_dqn import FakeState, empty_observation, active_goal_record, complete_fixture


def small_protocol(**kwargs) -> DQNProtocol:
    """Keep arithmetic tests small without altering the production treatment."""
    return replace(DQNProtocol(batch_size=2, warmup_transitions=4, total_training_transitions=6,
                               replay_capacity=5), **kwargs)


def transition(reward=1.0, terminal=False) -> dict:
    """Return a scalar-state transition for numerical tests."""
    return {"state": np.array([0.0]), "next_state": np.array([1.0]), "action": 0,
            "reward": reward, "terminal": terminal, "actor_update": 0}


def blocks_observation(positions: dict[int, tuple[int, int]]) -> np.ndarray:
    observation = empty_observation()
    for block, (row, column) in positions.items():
        observation[row + 2, column, block] = 1
    return observation


def raw_transition(index, state, next_state, *, illegal=False, terminal=False) -> dict:
    return {
        "state_observation": state,
        "next_observation": next_state,
        "action": 2,
        "illegal_action": illegal,
        "terminal": terminal,
        "actor_update": 7,
        "goal": (2, 3),
        "transition_index": index,
    }


def test_future_her_recomputes_goal_rewards_and_nstep_terminal() -> None:
    protocol = small_protocol(n_steps=3, her_future_goals=4, discount=0.5)
    state0 = blocks_observation({0: (0, 0), 1: (4, 0)})
    state1 = blocks_observation({0: (2, 0), 1: (4, 0)})
    state2 = blocks_observation({0: (3, 0), 1: (4, 0)})
    episode = [
        raw_transition(1, state0, state1, illegal=True),
        raw_transition(2, state1, state2),
    ]

    relabeler = HindsightRelabeler(protocol, np.random.default_rng(3),
                                   lambda illegal, achieved: (1.0 if achieved else -0.01)
                                   - (0.1 if illegal else 0.0))
    entries, sources, candidates = relabeler.relabel(episode)

    assert achieved_on_relations(state2) == ((0, 1),)
    assert sources == candidates == len(entries) == 2
    first, second = entries
    assert first["goal"] == second["goal"] == (0, 1)
    assert first["reward"] == pytest.approx(-0.11 + 0.5)
    assert first["horizon"] == 2 and first["terminal"]
    assert first["bootstrap_discount"] == 0
    assert second["reward"] == pytest.approx(1.0)
    assert second["horizon"] == 1 and second["terminal"]
    assert first["her_goal_achieved"] and second["her_goal_achieved"]
    assert first["state"].shape == MODEL_INPUT_SHAPE
    assert first["source_transition_index"] == 1 and first["her"] is True


def test_her_excludes_relations_already_true_at_source() -> None:
    protocol = small_protocol()
    achieved = blocks_observation({0: (3, 0), 1: (4, 0)})
    episode = [raw_transition(1, achieved, achieved)]
    relabeler = HindsightRelabeler(
        protocol, np.random.default_rng(4), lambda _illegal, success: float(success)
    )

    assert relabeler.relabel(episode) == ([], 0, 0)


def test_memory_admits_real_and_her_entries_at_episode_boundary() -> None:
    protocol = small_protocol(replay_capacity=20)
    memory = Memory(module_id="memory", initial_state={})
    memory.on_init(protocol=protocol, seed=8)
    memory._learner_id = "learner"
    state = FakeState(initial_states(protocol)["memory"])
    before = blocks_observation({0: (0, 0), 1: (4, 0)})
    after = blocks_observation({0: (3, 0), 1: (4, 0)})

    memory.on_observation_update(
        state,
        "knowledge",
        [Observation(after, value=-0.01)],
        phase="training",
        transition_index=1,
        previous_observation=before,
        action=2,
        goal=(2, 3),
        terminal=True,
        actor_update=0,
        illegal_action=False,
    )

    assert state["training_transitions"] == 1
    assert state["replay_entries"] == 2
    assert state["her_entries"] == 1
    assert state["her_success_entries"] == 1
    assert state["her_episodes"] == 1
    assert state["pending_her_episode"] == 0


def test_nstep_returns_and_shortened_terminal_tails() -> None:
    returns = NStepReturns(3, 0.5)
    assert returns.append(transition(1), (1, 2)) == []
    assert returns.append(transition(2), (1, 2)) == []
    entries = returns.append(transition(4, True), (1, 2))
    assert [entry["reward"] for entry in entries] == [3, 4, 4]
    assert [entry["horizon"] for entry in entries] == [3, 2, 1]
    assert all(entry["bootstrap_discount"] == 0 for entry in entries)
    assert not returns.pending


def test_collection_cutoff_bootstraps_and_cannot_cross_goals() -> None:
    returns = NStepReturns(3, 0.5)
    returns.append(transition(), (1, 2))
    with pytest.raises(ValueError, match="Goal changed"):
        returns.append(transition(), (2, 3))
    returns.append(transition(), (1, 2))
    entries = returns.flush()
    assert [e["bootstrap_discount"] for e in entries] == [0.25, 0.5]
    assert [e["reward"] for e in entries] == [1.5, 1]
    assert returns.append(transition(), (2, 3)) == []


def test_priorities_weights_duplicate_feedback_and_fifo_ids() -> None:
    protocol = small_protocol(replay_capacity=2, priority_alpha=1.0)
    replay = PrioritizedReplay(protocol, np.random.default_rng(12))
    for _ in range(2):
        replay.append(transition())
    replay.feedback([0, 0, 1], [1.0, 3.0, 1.0])
    assert replay.priorities.tolist() == pytest.approx([3.000001, 1.000001])
    hits = 0
    for _ in range(1000):
        _, metadata = replay.sample(1.0)
        for entry_id, weight in zip(metadata["entry_ids"], metadata["weights"]):
            hits += entry_id == 0
            assert weight == pytest.approx(1.000001 / 3.000001 if entry_id == 0 else 1.0)
    assert 1400 < hits < 1600
    replay.append(transition(5))
    before = replay.priorities[0]
    replay.feedback([0], [100.0])
    assert replay.priorities[0] == before
    assert replay.stale_feedback == 1 and replay.evictions == 1
    assert replay.ids.tolist() == [2, 1]


def admit(memory, state, index, terminal=False) -> None:
    """Send a real-shape raw observation through the memory callback."""
    memory.on_observation_update(state, "knowledge", [Observation(empty_observation(), value=-0.01)],
                                 phase="training", transition_index=index, previous_observation=empty_observation(),
                                 action=0, goal=(5, 9), terminal=terminal, actor_update=0,
                                 illegal_action=False)


@pytest.mark.parametrize("synchronous", [True, False])
def test_warmup_uses_mature_entries_without_future_action_deadlock(synchronous) -> None:
    protocol = small_protocol(synchronized_training=synchronous)
    memory = Memory(module_id="memory", initial_state={})
    memory.on_init(protocol=protocol, seed=2)
    memory._learner_id = "learner"
    state = FakeState(initial_states(protocol)["memory"])
    memory.on_memory_request(state, "learner", beta=0.4)
    for index in range(1, 5):
        admit(memory, state, index)
    assert state["batches_sent"] == 1
    assert state["replay_entries"] == 2
    assert state["credits_earned"] == int(synchronous)


@pytest.mark.parametrize("stop_first", [False, True])
def test_async_closure_flushes_and_acknowledges_in_both_orders(stop_first) -> None:
    protocol = small_protocol(synchronized_training=False)
    memory = Memory(module_id="memory", initial_state={})
    memory.on_init(protocol=protocol)
    memory._learner_id = "learner"
    state = FakeState(initial_states(protocol)["memory"])
    for index in range(1, 12):
        admit(memory, state, index)
    assert state["training_transitions"] > protocol.total_training_transitions
    assert state["credits_earned"] == 0
    if stop_first:
        memory.on_memory_request(state, "learner", stop=True)
    memory.on_observation_update(state, "knowledge", [], phase="training", collection_closed=True, transition_index=11)
    if not stop_first:
        memory.on_memory_request(state, "learner", stop=True)
    assert state["closure_acknowledged"] and state["pending_tail"] == 0
    assert state["replay_entries"] == 11 and state["evictions"] == 6
    assert state.outbox.calls[-1][2]["replay_closed"]


def test_double_selection_weighted_huber_and_priority_feedback() -> None:
    torch = pytest.importorskip("torch")
    protocol = small_protocol()
    learner = Learner(module_id="learner", initial_state={})
    learner.on_init(protocol=protocol)
    learner._memory_id = "memory"
    learner._online = torch.nn.Linear(1, 4)
    learner._target = torch.nn.Linear(1, 4)
    with torch.no_grad():
        learner._online.weight.zero_()
        learner._target.weight.zero_()
        learner._online.bias.copy_(torch.tensor([1., 3., 2., 0.]))
        learner._target.bias.copy_(torch.tensor([10., 20., 30., 40.]))
    learner._optimizer = torch.optim.AdamW(learner._online.parameters(), lr=1e-4)
    state = FakeState(initial_states(protocol)["learner"])
    learner.on_task(state, "reasoner", {"kind": "start"})
    entries = [dict(transition(r), bootstrap_discount=0.5, horizon=1) for r in (0., 1.)]
    learner.on_memories(state, "memory", [Observation(e) for e in entries], entry_ids=[0, 1],
                        weights=[0.25, 1.0], beta=0.4, mean_replay_age=1.0)
    assert state["failure_reason"] is None
    assert state["optimization_current"]["losses"] == pytest.approx([5.8125])
    assert state.outbox.calls[-1][2]["errors"] == pytest.approx([9, 10])
    assert state["target_syncs"] == 0


@pytest.mark.parametrize("updates", [1, 10, 11])
def test_final_publication_and_saved_model_identity(tmp_path: Path, updates: int) -> None:
    torch = pytest.importorskip("torch")
    learner = Learner(module_id="learner", initial_state={})
    path = tmp_path / "policy.pt"
    learner.on_init(policy_path=str(path))
    state = FakeState(initial_states()["learner"])
    state["training_updates"] = updates
    if updates in (1, 10):
        learner._publish_model(state, final=False)
    state["stopping"] = True
    learner.on_memories(state, "memory", [], replay_closed=True)
    assert state["frozen"] and state["models_published"] == 1
    assert state.outbox.calls[-1][2]["final"]
    model, metadata = load_policy_checkpoint(torch, path)
    assert metadata["frozen_for_evaluation"]
    assert model_state_fingerprint(model) == state["final_model_fingerprint"]


def test_deadline_discards_delivered_batch_and_cancels_pending_request() -> None:
    learner = Learner(module_id="learner", initial_state={})
    protocol = small_protocol(synchronized_training=False, async_training_seconds=2)
    learner.on_init(protocol=protocol)
    state = FakeState(initial_states(protocol)["learner"], time=0)
    learner.on_task(state, "reasoner", {"kind": "start", "deadline": 2})
    state.time = 2
    learner.on_memories(state, "memory", [Observation(None)])
    assert state["training_updates"] == 0 and state["unused_batches"] == 1
    assert state["stopping"] and state.outbox.calls[-1][2]["stop"]
    learner.on_memories(state, "memory", [], replay_closed=True)
    assert state["failure_reason"] == "training_deadline_before_first_update"


def test_reasoner_deadline_closes_active_goal_without_another_action() -> None:
    protocol = small_protocol(synchronized_training=False, async_training_seconds=2)
    reasoner = Reasoner(module_id="reasoner", initial_state={})
    reasoner.on_init(protocol=protocol)
    state = FakeState(initial_states(protocol)["ll_reasoner"], time=2)
    state["active_goal"] = active_goal_record()
    reasoner._request_policy_action(state, empty_observation())
    assert state["collection_closed"] and state["phase"] == "training_draining"
    assert state["active_goal"] is None
    assert not any(name == "request_action" for name, _, _ in state.outbox.calls)
    assert any(kwargs.get("collection_closed") for _, _, kwargs in state.outbox.calls)


def test_async_memory_accepts_transition_after_ten_thousand() -> None:
    protocol = small_protocol(synchronized_training=False)
    memory = Memory(module_id="memory", initial_state={})
    memory.on_init(protocol=protocol)
    state = FakeState(initial_states(protocol)["memory"])
    state["training_transitions"] = 10_000
    admit(memory, state, 10_001)
    assert state["training_transitions"] == 10_001
    assert state["credits_earned"] == 0


@pytest.mark.parametrize("invalid", ["missing_metadata", "nonfinite"])
def test_invalid_update_fails_without_consuming_an_optimizer_step(invalid) -> None:
    torch = pytest.importorskip("torch")
    protocol = small_protocol()
    learner = Learner(module_id="learner", initial_state={})
    learner.on_init(protocol=protocol)
    state = FakeState(initial_states(protocol)["learner"])
    learner.on_task(state, "reasoner", {"kind": "start"})
    learner._online = torch.nn.Linear(1, 4)
    learner._target = torch.nn.Linear(1, 4)
    learner._optimizer = torch.optim.AdamW(learner._online.parameters())
    entries = [Observation(dict(transition(), bootstrap_discount=0.9, horizon=1))] * 2
    metadata = {}
    if invalid == "nonfinite":
        with torch.no_grad():
            learner._online.weight.fill_(float("nan"))
        metadata = dict(entry_ids=[0, 1], weights=[1., 1.], beta=0.4, mean_replay_age=1.)
    learner.on_memories(state, "memory", entries, **metadata)
    assert state["training_updates"] == 0
    assert state["failure_reason"].startswith("invalid_training_update")
    assert state.outbox.calls[-1][0] == "terminate_agent"


def test_beta_and_epsilon_follow_the_selected_budget() -> None:
    for synchronous in (True, False):
        protocol = small_protocol(synchronized_training=synchronous, async_training_seconds=10)
        reasoner = Reasoner(module_id="reasoner", initial_state={})
        reasoner.on_init(protocol=protocol)
        reasoner._decision_time = 10
        assert reasoner._epsilon(protocol.total_training_transitions) == pytest.approx(0.05)
        learner = Learner(module_id="learner", initial_state={})
        learner.on_init(protocol=protocol)
        state = FakeState(initial_states(protocol)["learner"], time=10)
        state["training_started_at"] = 0
        state["training_updates"] = protocol.training_updates - 1
        assert learner._beta(state) == 1.0


@pytest.mark.parametrize("kwargs", [
    {"warmup_transitions": 2}, {"replay_capacity": 1}, {"async_training_seconds": float("nan")},
    {"evaluation_seeds": ()}, {"synchronized_training": "false"},
    {"her_future_goals": 0},
])
def test_protocol_rejects_unusable_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        small_protocol(**kwargs)


def test_variable_final_windows_report_their_actual_extent() -> None:
    rows = _aligned_optimization([
        {"optimization": {"windows": [{"update_start": 1, "update_end": end, "mean_loss": 0.1,
                                      "final_loss": 0.1, "target_syncs": 0}]}}
        for end in (13, 20)
    ])
    assert rows[0]["update_end"] == 20
    assert rows[0]["minimum_update_end"] == 13


def test_historical_summary_is_retained_in_a_separate_group(tmp_path) -> None:
    states, environment, path = complete_fixture(tmp_path)
    checkpoint, _ = checkpoint_evidence(path)
    protocol = DQNProtocol()
    common = dict(states=states, environment_state=environment, module_ids=_module_ids(),
                  checker_passed=True, checker_reasons=[], checkpoint=checkpoint)
    current = build_run_summary(run=0, provenance=protocol.record(), **common)
    historical = build_run_summary(run=1, provenance=protocol_provenance(
        synchronized_training=True, warmup_transitions=protocol.warmup_transitions,
        total_training_transitions=protocol.total_training_transitions,
        training_updates=protocol.training_updates, target_sync_steps=protocol.target_sync_steps,
        evaluation_seeds=protocol.evaluation_seeds,
        behavior_window_transitions=protocol.behavior_window_transitions,
        optimization_window_updates=protocol.optimization_window_updates,
    ), **common)
    report = aggregate_summaries([current, historical])
    assert len(report["protocol_groups"]) == 2
    assert not report["protocol_incompatible"]
