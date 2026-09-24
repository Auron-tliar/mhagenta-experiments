"""Small CPU checks of replay, reward boundaries, and preparation execution."""

from collections import deque
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mha_exp_level2_bw.exp2_5.contracts import GoalSpec
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from tools.bw_achieve_on_preparation import train as training
from tools.bw_achieve_on_preparation.assess import assess
from tools.bw_achieve_on_preparation.policy import MODEL_INPUT_SHAPE
from tools.bw_achieve_on_preparation.prepare import prepare
from tools.bw_achieve_on_preparation.train import Config, Replay, Transition, emit


def transition(reward: float, terminal: bool) -> Transition:
    """Build a synthetic transition for replay boundary checks."""
    state = np.zeros(MODEL_INPUT_SHAPE, dtype=np.uint8)
    return Transition(state, 0, reward, state, terminal, np.ones(4, dtype=np.bool_))


def test_n_step_stops_at_terminal_and_flushes_short_episodes() -> None:
    pending = deque([transition(0.5, False), transition(1.0, True)])
    rows = emit(pending, Config(), True)
    assert len(rows) == 2 and not pending
    assert rows[0].reward == pytest.approx(0.5 + 0.99)
    assert rows[0].discount == pytest.approx(0.99 ** 2)
    assert rows[0].terminal
    assert rows[1].reward == 1.0


def test_replay_preserves_expert_prefix_when_ordinary_capacity_wraps() -> None:
    expert = transition(1, True)
    replay = Replay(4, [expert])
    for index in range(10):
        replay.append(transition(float(index), True))
    assert len(replay.rows) == 4
    assert replay.rows[0] is expert
    indices, _, weights = replay.sample(np.random.default_rng(0), 100, 0.4)
    assert set(indices) == {0, 1, 2, 3}
    assert np.all(weights > 0) and np.all(weights <= 1)


@pytest.mark.parametrize("terminal", [False, True])
def test_double_dqn_masks_illegal_targets_and_terminal_bootstrap(terminal) -> None:
    """A huge illegal Q value must not enter a target; terminal targets stop."""
    class ConstantQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, inputs):
            return self.values.unsqueeze(0).expand(len(inputs), -1)

    row = transition(0, terminal)
    row.action = 2
    row.legal_next = np.array([False, True, False, False])
    replay = Replay(2, [row])
    online = ConstantQ([100, 2, 0, 0])
    target = ConstantQ([1000, 3, 0, 0])
    optimizer = torch.optim.SGD(online.parameters(), lr=0)
    training.optimize(online, target, optimizer, replay, np.random.default_rng(0),
                      Config(batch_size=1), "cpu", 0.4)
    expected = (0 if terminal else 0.99 * 3) + 1e-5 + 0.1
    assert replay.priorities[0] == pytest.approx(expected)


def test_preparation_is_cpu_only_and_refuses_overwrite(tmp_path) -> None:
    report = prepare(tmp_path / "initial")
    assert report["device"] == "cpu"
    assert report["parameters"] == 24900
    assert report["ready_for_runtime"] is False
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "initial")


def test_tiny_training_and_assessment_pipeline(tmp_path, monkeypatch) -> None:
    """Exercise actual tensors and environments with a fixture planner and tiny budget."""
    torch.set_num_threads(1)
    env = training.make_environment()
    observation, _ = env.reset(seed=1000)
    spec = next(item for item in enumerate_transfer_targets(ground_observation(observation).facts)
                if item.destination_support.startswith("b"))
    env.close()
    goal = GoalSpec(spec.block, spec.destination_support)

    def fixed_case(environment, split, index):
        observation, _ = environment.reset(seed=1000)
        return observation, goal, 1000

    service = SimpleNamespace(solve=lambda *args: SimpleNamespace(
        accepted=True, engine="test-fixture", actions=[spec.as_dict()]))
    monkeypatch.setattr(training, "case", fixed_case)
    monkeypatch.setattr(training, "demonstration_start", lambda environment, observation, seed, index:
                        (observation, goal, {"difficulty": 0, "reset_actions": []}))
    monkeypatch.setattr(training, "planner", lambda: service)
    import tools.bw_achieve_on_preparation.assess as assessment
    monkeypatch.setattr(assessment, "case", fixed_case)
    monkeypatch.setattr(assessment, "planner", lambda: service)
    config = Config(demonstrations=1, warm_updates=1, online_steps=4,
                    selection_interval=4, selection_cases=1, batch_size=2)
    training.train(tmp_path / "pilot", config, "cpu")
    result = assess(tmp_path / "pilot", tmp_path / "assessment", 1)
    assert result["status"] == "completed-assessment"
    assert result["paired"]["baseline_successes"] == 1
    assert result["paired"]["achieve_on_successes"] == 1
    assert (tmp_path / "pilot" / "step-0000004.pt").is_file()
    with pytest.raises(FileExistsError):
        assess(tmp_path / "pilot", tmp_path / "assessment", 1)
    # A different architecture can reuse exactly these candidate-independent
    # traces, rebuilding rewards/states without paying for planning again.
    monkeypatch.setattr(training, "planner", lambda: pytest.fail("Cached teachers must not replan"))
    from mha_exp_level2_bw.achieve_on.policy import CONV3D_ARCHITECTURE
    training.train(tmp_path / "cached-conv3d", config, "cpu", CONV3D_ARCHITECTURE,
                   teacher_cache=tmp_path / "pilot" / "report.json")
    cached = json.loads((tmp_path / "cached-conv3d" / "report.json").read_text())
    original = json.loads((tmp_path / "pilot" / "report.json").read_text())
    assert cached["demonstrations"] == original["demonstrations"]
    assert cached["baseline"] == original["baseline"]
    assert cached["demonstration_transitions"] == original["demonstration_transitions"]
    result = assess(tmp_path / "cached-conv3d", tmp_path / "conv3d-assessment", 1)
    assert result["architecture"] == CONV3D_ARCHITECTURE
    from tools.bw_achieve_on_preparation import compare as comparison
    monkeypatch.setattr(comparison, "case", fixed_case)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    compared = comparison.compare(tmp_path / "cached-conv3d", tmp_path / "pilot",
                                  tmp_path / "conv3d-assessment" / "assessment.json",
                                  tmp_path / "comparison.json")
    assert compared["cases"] == 1 and compared["mlp_successes"] == 1
    assert compared["parameters"]["conv3d"] > compared["parameters"]["mlp"]
    assert compared["latency"]["conv3d"][0]["samples"] == 1


def test_training_deadline_keeps_evidence_without_selecting_an_unfinished_evaluation(tmp_path, monkeypatch) -> None:
    """A wall cutoff retains weights but cannot fabricate a selection result."""
    monkeypatch.setattr(training, "planner", lambda: None)
    training.train(tmp_path / "limited", Config(), "cpu", max_wall_seconds=1e-9)
    report = json.loads((tmp_path / "limited" / "report.json").read_text())
    assert report["status"] == "time-limit-unqualified"
    assert report["optimizer_steps"] == report["online_steps"] == 0
    assert "selected_checkpoint" not in report
    assert (tmp_path / "limited" / "time-limit.pt").exists()
    with pytest.raises(ValueError, match="completed selection"):
        assess(tmp_path / "limited", tmp_path / "unselected-assessment", 1)


def test_stratified_replay_keeps_half_demonstrations_despite_large_online_priorities() -> None:
    """Ordinary TD errors cannot starve demonstrations; warm-up keeps easy examples."""
    replay = Replay(10, [transition(1, True) for _ in range(3)], [0, 1, 2])
    for _ in range(7):
        replay.append(transition(-1, True))
    replay.priorities[3:] = 1e6
    for limit in (0, 1, 2):
        indices, _, weights = replay.sample(np.random.default_rng(7), 128, 0.4, 0.5, limit)
        assert sum(indices < 3) == 64
        assert set(indices[indices < 3]) == set(range(limit + 1))
        assert np.isfinite(weights).all() and (weights > 0).all() and (weights <= 1).all()


def test_demonstration_starts_are_reproducible_legal_and_cover_difficulty() -> None:
    """Fixed perturbations replay exactly and never supply already achieved goals."""
    from mha_exp_level2_bw.achieve_on.demonstrations import demonstration_start
    env, replay = training.make_environment(), training.make_environment()
    levels, perturbed = set(), 0
    try:
        for index in range(48):
            seed = 510_000_000 + index
            observation, _ = env.reset(seed=seed)
            observation, goal, metadata = demonstration_start(env, observation, seed, index)
            repeated, _ = env.reset(seed=seed)
            repeated, repeated_goal, repeated_metadata = demonstration_start(env, repeated, seed, index)
            assert np.array_equal(observation, repeated)
            assert (goal, metadata) == (repeated_goal, repeated_metadata)
            rebuilt, _ = replay.reset(seed=seed)
            for action in metadata["reset_actions"]:
                rebuilt, _, _, _, info = replay.step(action)
                assert info["snapshot"].legal
            assert np.array_equal(observation, rebuilt)
            facts = ground_observation(observation).facts
            assert "hand-empty()" in facts and goal.fact not in facts
            levels.add(metadata["difficulty"])
            perturbed += bool(metadata["reset_actions"])
        assert levels == {0, 1, 2} and perturbed == 12
    finally:
        env.close()
        replay.close()


def test_assessment_window_rejects_overlap_with_split_boundary(tmp_path) -> None:
    with pytest.raises(ValueError, match="split window"):
        assess(tmp_path, tmp_path / "out", 200, 999_900)
