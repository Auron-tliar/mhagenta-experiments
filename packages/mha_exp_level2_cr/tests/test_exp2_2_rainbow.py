"""Behavioral coverage for diamond shaping and synchronized Rainbow replay."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mhagenta import ActionStatus, Observation
from mha_exp_level2_cr.exp2_2 import modules
from mha_exp_level2_cr.exp2_2.policy import (
    ACHIEVEMENTS, FRAME_SHAPE, build_q_network, initial_frame_stack, load_policy_checkpoint,
)
from mha_exp_level2_cr.exp2_2.replay import PrioritizedReplay, endpoint_stack
from mha_exp_level2_cr.exp2_2.rewards import evaluate_reward, reward_state
from mha_exp_level2_cr.exp2_2.runner import initial_states
from mha_exp_level2_cr.exp2_2.treatment import (
    ACHIEVEMENT_REWARDS, DQNWorkload, DISCOUNT, PRIORITY_EPSILON, REPLAY_BATCH_SIZE,
    SMOKE_WORKLOAD, SURVIVAL_REWARDS,
)
from test_exp2_2_dqn import FakeState, evidence, frame, prepared_reasoner


def event(name: str, *, need: str | None = None, level: int = 4) -> dict:
    """Build one real achievement increment with optional replenishment."""
    result = evidence()
    result["after"]["achievements"][name] = 1
    if need:
        result["before"]["needs"][need] = level
        result["after"]["needs"][need] = min(9, level + 1)
    return result


@pytest.mark.parametrize("achievement,amount", ACHIEVEMENT_REWARDS.items())
def test_milestone_bonus_once_per_episode(achievement: str, amount: float) -> None:
    tracker = reward_state()
    sample = event(achievement)
    assert sum(evaluate_reward(tracker, 1, False, sample).values()) == pytest.approx(amount - 0.001)
    sample["before"]["achievements"][achievement] = 1
    sample["after"]["achievements"][achievement] = 2
    assert sum(evaluate_reward(tracker, 1, False, sample).values()) == pytest.approx(-0.001)
    assert achievement in evaluate_reward(tracker, 2, False, event(achievement))


@pytest.mark.parametrize("achievement,need", [("collect_drink", "drink"), ("eat_cow", "food"), ("eat_plant", "food")])
def test_survival_eligibility_and_later_qualified_event(achievement: str, need: str) -> None:
    tracker = reward_state()
    assert achievement not in evaluate_reward(tracker, 1, False, event(achievement, need=need, level=5))
    sample = event(achievement, need=need)
    sample["before"]["achievements"][achievement] = 1
    sample["after"]["achievements"][achievement] = 2
    assert evaluate_reward(tracker, 1, False, sample)[achievement] == 0.1
    assert achievement not in evaluate_reward(tracker, 1, False, sample)
    stagnant = event(achievement, need=need)
    stagnant["after"]["needs"][need] = 4
    assert achievement not in evaluate_reward(reward_state(), 1, False, stagnant)


@pytest.mark.parametrize("level,interrupted,expected", [(4, False, True), (5, False, False), (4, True, False)])
def test_sleep_uses_entry_energy_and_requires_natural_completion(level: int, interrupted: bool, expected: bool) -> None:
    tracker = reward_state()
    entry = evidence()
    entry["before"]["needs"]["energy"] = level
    entry["after"]["sleeping"] = True
    evaluate_reward(tracker, 1, False, entry)
    if interrupted:
        hurt = evidence()
        hurt["before"]["sleeping"] = True
        evaluate_reward(tracker, 1, False, hurt)
    wake = event("wake_up")
    wake["before"]["sleeping"] = True
    assert ("wake_up" in evaluate_reward(tracker, 1, False, wake)) == expected


def test_death_suppresses_all_bonuses_and_is_worse_than_legal_survival() -> None:
    sample = event("collect_diamond")
    sample["after"]["needs"]["health"] = 0
    components = evaluate_reward(reward_state(), 1, True, sample)
    assert components == {"step": -0.001, "illegal_action": -0.1, "death": -5.0}
    alive = sum(DISCOUNT ** index * -0.001 for index in range(1000))
    assert alive > sum(evaluate_reward(reward_state(), 1, False, sample).values())
    assert sum(ACHIEVEMENT_REWARDS.values()) - 10 + sum(SURVIVAL_REWARDS.values()) == pytest.approx(4.65)


def test_simultaneous_milestones_add_and_unlisted_achievements_do_not() -> None:
    sample = event("collect_wood")
    sample["after"]["achievements"].update(place_table=1, defeat_zombie=1)
    assert sum(evaluate_reward(reward_state(), 1, False, sample).values()) == pytest.approx(0.499)


def raw_transition(ordinal: int, reward: float, terminal: bool = False) -> dict:
    """Return a frame-numbered raw transition for exact horizon assertions."""
    return {"replay_id": ordinal, "state_stack": initial_frame_stack(frame(ordinal - 1)),
            "next_frame": frame(ordinal), "action": 0, "reward": reward, "terminal": terminal}


@pytest.mark.parametrize("terminal", [False, True])
def test_three_step_return_flush_and_bootstrap(terminal: bool) -> None:
    replay = PrioritizedReplay(np.random.default_rng(1))
    replay.append(raw_transition(1, 1), boundary=False)
    replay.append(raw_transition(2, 2), boundary=False)
    assert not replay.items
    replay.append(raw_transition(3, 3, terminal), boundary=True)
    assert [row["horizon"] for row in replay.items] == [3, 2, 1]
    assert [row["reward"] for row in replay.items] == pytest.approx([1 + .99 * 2 + .99**2 * 3, 2 + .99 * 3, 3])
    assert [row["bootstrap_discount"] for row in replay.items] == pytest.approx([0, 0, 0] if terminal else [.99**3, .99**2, .99])
    assert [int(image[0, 0, 0]) for image in endpoint_stack(replay.items[0])] == [0, 1, 2, 3]
    replay.append(raw_transition(4, 99), boundary=True)
    assert replay.items[-1]["reward"] == 99
    assert not replay.pending


def test_priorities_weights_duplicates_and_ring_ids() -> None:
    replay = PrioritizedReplay(np.random.default_rng(0), capacity=128)
    for index in range(1, 129):
        replay.append(raw_transition(index, 0), boundary=True)
    replay.update_priorities([1, 1] + [2] * 126, [1, 9] + [4] * 126)
    assert replay.priorities[:2] == pytest.approx([9 + PRIORITY_EPSILON, 4 + PRIORITY_EPSILON])
    class FixedRng:
        def choice(self, population: int, *, size: int, replace: bool, p: np.ndarray) -> np.ndarray:
            assert population == size == 128 and replace
            np.testing.assert_allclose(p, replay.priorities**.5 / sum(replay.priorities**.5))
            return np.array([0, 1] * 64)
    replay.rng = FixedRng()
    _, ids, start_weights = replay.sample(1, 128)
    _, _, end_weights = replay.sample(128, 128)
    assert ids == [1, 2] * 64
    ratio = ((9 + PRIORITY_EPSILON) / (4 + PRIORITY_EPSILON))**.5
    assert start_weights[:2] == pytest.approx([ratio**-.4, 1])
    assert end_weights[:2] == pytest.approx([ratio**-1, 1])
    replay.append(raw_transition(129, 0), boundary=True)
    with pytest.raises(ValueError, match="Stale"):
        replay.update_priorities([1] * 128, [1.] * 128)
    with pytest.raises(ValueError, match="finite"):
        replay.update_priorities([129] * 128, [float("nan")] * 128)


def test_dueling_aggregation_and_double_selection() -> None:
    torch = pytest.importorskip("torch")
    network = build_q_network(torch)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.value[-1].bias.fill_(3)
        network.advantage[-1].bias.copy_(torch.arange(17))
        values = network(torch.zeros(2, 12, 64, 64))
    torch.testing.assert_close(values[0], torch.arange(17, dtype=torch.float32) - 8 + 3)
    online = lambda _: torch.tensor([[1., 4.]])
    target = lambda _: torch.tensor([[99., 2.]])
    result = modules.double_dqn_targets(torch, online, target, None, torch.tensor([1.]), torch.tensor([.5]))
    assert result.item() == 2.0


@pytest.mark.parametrize("achievement,dead,boundary,bootstrap_terminal", [
    ("make_stone_pickaxe", False, False, False), ("collect_diamond", False, True, True),
    ("collect_diamond", True, True, True), ("collect_wood", True, True, True),
])
def test_reasoner_goal_and_death_boundaries(achievement: str, dead: bool, boundary: bool, bootstrap_terminal: bool) -> None:
    reasoner, state = prepared_reasoner()
    sample = event(achievement)
    if dead:
        sample["after"]["needs"]["health"] = 0
    reasoner.on_action_status(state, "actuator", ActionStatus({"illegal_action": False, "done": dead,
        "achievements": sample["after"]["achievements"], "reward_evidence": sample}))
    reasoner.on_observation(state, "perceptor", Observation(frame(2)))
    transition = next(call for call in state.outbox.calls if call[0] == "send_beliefs")[2]
    assert transition["boundary"] is boundary
    assert transition["terminal"] is bootstrap_terminal
    assert state["target_successes"] == int(achievement == "collect_diamond" and not dead)


def test_truncation_flushes_without_death_or_terminal_bootstrap() -> None:
    reasoner, state = prepared_reasoner()
    reasoner._episode_length = reasoner._workload.episode_action_limit
    reasoner.on_observation(state, "perceptor", Observation(frame(2)))
    transition = next(call for call in state.outbox.calls if call[0] == "send_beliefs")[2]
    assert transition["boundary"] is True and transition["terminal"] is False
    assert state["truncations"] == 1 and state["deaths"] == 0


def test_invalid_priority_messages_terminate_visibly() -> None:
    memory = modules.TestMemory("memory", {})
    memory.log = lambda *_: None
    state = FakeState(initial_states()["memory"])
    memory.on_memory_request(state, "learner", kind="priority_update", cycle_id=1, replay_ids=[])
    assert state["contract_errors"] == 1
    assert state.outbox.calls[-1][0] == "terminate_agent"


@pytest.mark.parametrize("kwargs", [{"training_transitions": 512}, {"episode_action_limit": 0},
                                    {"evaluation_seeds": ()}, {"duration_seconds": float("inf")}])
def test_workload_bounds(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DQNWorkload(**kwargs)


def test_real_state_callback_lifecycle_freezes_after_priority_ack(tmp_path: Path) -> None:
    """Run the four cognitive modules through actual typed outboxes and two updates."""
    from collections import deque
    from types import SimpleNamespace
    from mhagenta.utils import State
    from mhagenta.outboxes import KnowledgeOutbox, LearnerOutbox, LLOutbox, MemoryOutbox

    torch = pytest.importorskip("torch")
    workload = DQNWorkload(514, 3, (920000, 920001), 60)
    directory = SimpleNamespace(internal=SimpleNamespace(**{
        key: [SimpleNamespace(module_id=value)] for key, value in {
            "actuation": "actuator", "perception": "perceptor", "knowledge": "knowledge",
            "memory": "memory", "learning": "learner", "ll_reasoning": "ll_reasoner",
        }.items()
    }))
    behaviors = {
        "ll_reasoner": modules.TestLLReasoner("ll_reasoner", {}),
        "knowledge": modules.TestKnowledge("knowledge", {}),
        "memory": modules.TestMemory("memory", {}),
        "learner": modules.TestLearner("learner", {}),
    }
    outboxes = dict(ll_reasoner=LLOutbox, knowledge=KnowledgeOutbox,
                   memory=MemoryOutbox, learner=LearnerOutbox)
    states = {name: State(agent_id="agent", module_id=name, time_func=lambda: 1.,
                         directory=directory, outbox=outboxes[name](), **initial_states(workload)[name])
              for name in behaviors}
    queue = deque()
    terminated = []

    def flush(name: str) -> None:
        outbox = states[name].outbox
        if outbox:
            queue.extend((name, receiver, dict(content)) for receiver, _, _, content in outbox)
        outbox.clear()
        term, reason = outbox.pop_term_request()
        if term:
            terminated.append(reason)

    for name, behavior in behaviors.items():
        behavior.log = lambda *_: None
        behavior.on_init(seed=4, workload=workload.dump(), policy_path=str(tmp_path / "policy.pt"))
        behavior.on_first(states[name])
        flush(name)
    snapshot = evidence()["before"]
    episode_action = 0
    callbacks = 0
    while queue:
        callbacks += 1
        assert callbacks < 5000, "Callback cycle failed to reach termination"
        sender, receiver, body = queue.popleft()
        if receiver == "perceptor":
            behaviors["ll_reasoner"].on_observation(states["ll_reasoner"], receiver, Observation(frame(episode_action)))
            flush("ll_reasoner")
            continue
        if receiver == "actuator":
            if body["action"] == "reset":
                episode_action = 0
                snapshot = evidence()["before"]
                status = {"applied_seed": body.get("requested_seed")}
            else:
                before = deepcopy(snapshot)
                episode_action += 1
                # Intermediate achievements must not finish the episode.
                if episode_action == 1:
                    snapshot["achievements"]["make_stone_pickaxe"] = 1
                if episode_action == 3:
                    snapshot["achievements"]["collect_diamond"] = 1
                status = {"illegal_action": False, "done": False, "reward": 1.,
                          "achievements": dict(snapshot["achievements"]),
                          "reward_evidence": {"before": before, "after": deepcopy(snapshot)}}
            behaviors["ll_reasoner"].on_action_status(states["ll_reasoner"], receiver, ActionStatus(status))
            flush("ll_reasoner")
            continue
        behavior, state = behaviors[receiver], states[receiver]
        if receiver == "knowledge":
            behavior.on_observed_beliefs(state, sender, **body)
        elif receiver == "memory" and sender == "knowledge":
            behavior.on_observation_update(state, sender, **body)
        elif receiver == "memory":
            assert not states["learner"]["frozen"]
            behavior.on_memory_request(state, sender, **body)
        elif receiver == "learner":
            behavior.on_memories(state, sender, **body)
        else:
            behavior.on_model(state, sender, **body)
        flush(receiver)
    assert len(terminated) == 1
    ll, memory, learner = (states[name] for name in ("ll_reasoner", "memory", "learner"))
    assert ll["phase"] == "complete"
    assert memory["transitions_admitted"] == memory["experiences_finalized"] == 514
    assert memory["pending_transitions"] == 0
    assert memory["priority_updates"] == learner["priority_acknowledgements"] == ll["cycles_completed"] == 2
    assert learner["frozen"] and learner["post_freeze_update_attempts"] == 0
    assert [case["length"] for case in ll["evaluation_cases"]] == [3, 3]
    assert all(case["success"] for case in ll["evaluation_cases"])
    assert ll["training_episodes"][-1]["truncation"] is True
    assert ll["training_episodes"][-1]["length"] == 1
    _, metadata = load_policy_checkpoint(torch, tmp_path / "policy.pt")
    assert metadata["workload"] == workload.dump() and metadata["training_steps"] == 2
    import json
    for state in states.values():
        json.dumps(state.dump())
