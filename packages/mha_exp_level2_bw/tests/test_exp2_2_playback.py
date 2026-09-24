from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mha_exp_level2_bw.exp2_2.play_policy import (
    build_parser,
    evaluate_episode,
    greedy_action,
    run_playback,
)
from mha_exp_level2_bw.exp2_2.policy import (
    MODEL_INPUT_SHAPE,
    OBS_SHAPE,
    build_q_network,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from mha_exp_level2_bw.exp2_2.runner import (
    K_MODEL_ARTIFACT,
    K_MODEL_SAVED,
    K_SAVED_TRAINING_STEPS,
    K_TRAINING_STEPS,
    POLICY_FILENAME,
    TestLearner as LearnerBehavior,
    initial_states,
)


class FakeTensor:
    def unsqueeze(self, dimension: int) -> "FakeTensor":
        return self


class FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class FakeOutput:
    def __init__(self, action: int) -> None:
        self.action = action

    def argmax(self, dim: int) -> FakeScalar:
        return FakeScalar(self.action)


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


class FakeTorch:
    float32 = object()

    @staticmethod
    def as_tensor(value: Any, dtype: Any) -> FakeTensor:
        return FakeTensor()

    @staticmethod
    def no_grad() -> FakeNoGrad:
        return FakeNoGrad()


class FixedPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def __call__(self, value: Any) -> FakeOutput:
        return FakeOutput(self.action)


class FakeEnvironment:
    def __init__(self, next_observation: np.ndarray) -> None:
        self.next_observation = next_observation
        self.window = object()
        self.actions: list[int] = []

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return self.next_observation, 0.0, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


def empty_observation() -> np.ndarray:
    return np.zeros(OBS_SHAPE, dtype=np.uint8)


def achieved_observation(goal: tuple[int, int]) -> np.ndarray:
    observation = empty_observation()
    observation[4, 2, goal[0]] = 1
    observation[5, 2, goal[1]] = 1
    return observation


def test_greedy_action_always_uses_model_argmax() -> None:
    action = greedy_action(
        FakeTorch,
        FixedPolicy(3),
        empty_observation(),
        (5, 9),
    )
    assert action == 3


def test_episode_stops_on_goal_and_uses_requested_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mha_exp_level2_bw.exp2_2.play_policy.time.sleep", lambda _: None)
    goal = (5, 9)
    environment = FakeEnvironment(achieved_observation(goal))
    result = evaluate_episode(
        FakeTorch,
        FixedPolicy(2),
        environment,  # type: ignore[arg-type]
        empty_observation(),
        goal,
        max_steps=10,
        mode="human",
        fps=2.0,
    )
    assert result.success
    assert result.steps == 1
    assert environment.actions == [2]


def test_episode_reports_step_limit() -> None:
    environment = FakeEnvironment(empty_observation())
    result = evaluate_episode(
        FakeTorch,
        FixedPolicy(1),
        environment,  # type: ignore[arg-type]
        empty_observation(),
        (5, 9),
        max_steps=3,
        mode="gif",
        fps=2.0,
    )
    assert not result.success
    assert result.steps == 3
    assert environment.actions == [1, 1, 1]


def test_cli_defaults_to_two_fps_gif() -> None:
    args = build_parser().parse_args(["model.pt", "--episodes", "2"])
    assert args.mode == "gif"
    assert args.fps == 2.0
    assert args.episodes == 2


def test_checkpoint_round_trip_and_learner_on_last(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = build_q_network(torch)
    checkpoint_path = tmp_path / POLICY_FILENAME

    saved = save_policy_checkpoint(torch, model, checkpoint_path, training_steps=17)
    loaded, metadata = load_policy_checkpoint(torch, saved)
    inputs = torch.rand((2, *MODEL_INPUT_SHAPE))
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), loaded(inputs))
    assert metadata["training_steps"] == 17

    learner = LearnerBehavior(module_id="learner", initial_state={})
    learner._torch = torch
    learner._online = model
    learner._policy_path = checkpoint_path
    state = initial_states()["learner"]
    state.update(
        {
            K_TRAINING_STEPS: 23,
            K_MODEL_SAVED: False,
            K_MODEL_ARTIFACT: "",
            K_SAVED_TRAINING_STEPS: 0,
        }
    )
    learner.on_last(state)  # type: ignore[arg-type]
    _, final_metadata = load_policy_checkpoint(torch, checkpoint_path)
    assert state[K_MODEL_SAVED]
    assert state[K_MODEL_ARTIFACT] == POLICY_FILENAME
    assert state[K_SAVED_TRAINING_STEPS] == 23
    assert final_metadata["training_steps"] == 23


def test_gif_playback_writes_episode_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    imageio = pytest.importorskip("imageio.v2")
    model = build_q_network(torch)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model[-1].bias[3] = 1.0
    checkpoint_path = save_policy_checkpoint(
        torch,
        model,
        tmp_path / POLICY_FILENAME,
        training_steps=1,
    )

    generated = run_playback(
        checkpoint_path,
        episodes=1,
        fps=2.0,
        max_steps=2,
        seed=0,
        output_dir=tmp_path / "gifs",
    )
    assert len(generated) == 1
    assert generated[0].is_file()
    assert generated[0].name.startswith("episode_001_")
    frames = imageio.mimread(generated[0])
    assert 2 <= len(frames) <= 3
    assert frames[0].shape == (480, 640, 3)
    reader = imageio.get_reader(generated[0])
    try:
        assert reader.get_meta_data()["duration"] == 500
    finally:
        reader.close()
    output = capsys.readouterr().out
    assert "place block B" in output
    assert "GIF: " + str(generated[0].resolve()) in output


def test_human_playback_smoke_test_with_dummy_video_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(
        "mha_exp_level2_bw.exp2_2.play_policy.time.sleep",
        lambda _: None,
    )
    checkpoint_path = save_policy_checkpoint(
        torch,
        build_q_network(torch),
        tmp_path / POLICY_FILENAME,
        training_steps=1,
    )

    generated = run_playback(
        checkpoint_path,
        episodes=1,
        mode="human",
        fps=2.0,
        max_steps=1,
        seed=0,
    )
    assert generated == []
