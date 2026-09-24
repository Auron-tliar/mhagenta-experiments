from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mha_exp_level2_cr.exp2_2.play_policy import (
    build_parser, evaluate_episode, greedy_action, run_playback,
)
from mha_exp_level2_cr.exp2_2.policy import (
    ACHIEVEMENTS, FRAME_SHAPE, MODEL_IMAGE_SHAPE, POLICY_FILENAME,
    TARGET_ACHIEVEMENT, build_q_network, initial_frame_stack,
    save_policy_checkpoint,
)


class FixedPolicy:
    def __init__(self, torch: Any, action: int = 0) -> None:
        self.torch = torch
        self.action = action
        self.seen_shapes: list[tuple[int, ...]] = []

    def __call__(self, images: Any) -> Any:
        self.seen_shapes.append(tuple(images.shape))
        output = self.torch.zeros((1, 17))
        output[0, self.action] = 1
        return output


class FakeCrafter:
    def __init__(self, *, success: bool = False, done: bool = False) -> None:
        self.success = success
        self.done = done
        self.actions: list[int] = []
        self.step_value = 0

    def render(self, size: tuple[int, int]) -> np.ndarray:
        return np.zeros((*size, 3), dtype=np.uint8)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.actions.append(action)
        self.step_value += 1
        achievements = {name: 0 for name in ACHIEVEMENTS}
        if self.success:
            achievements[TARGET_ACHIEVEMENT.value] = 1
        return (
            np.full(FRAME_SHAPE, self.step_value, dtype=np.uint8),
            0.0,
            self.done,
            {"achievements": achievements, "inventory": {"health": 0 if self.done else 9}},
        )


def test_cli_defaults() -> None:
    args = build_parser().parse_args(["model.pt", "--episodes", "2"])
    assert args.mode == "gif"
    assert args.fps == 2.0
    assert args.max_steps == 1_000


def test_greedy_action_uses_four_frame_argmax() -> None:
    torch = pytest.importorskip("torch")
    policy = FixedPolicy(torch, action=2)
    action = greedy_action(
        torch,
        policy,
        initial_frame_stack(np.zeros(FRAME_SHAPE, dtype=np.uint8)),
    )
    assert action == 2
    assert policy.seen_shapes == [(1, *MODEL_IMAGE_SHAPE)]


@pytest.mark.parametrize(
    ("success", "done", "max_steps", "expected"),
    [
        (True, False, 10, (True, False, 1)),
        (False, True, 10, (False, True, 1)),
        (False, False, 2, (False, False, 2)),
    ],
)
def test_episode_termination_modes(
    success: bool,
    done: bool,
    max_steps: int,
    expected: tuple[bool, bool, int],
) -> None:
    torch = pytest.importorskip("torch")
    environment = FakeCrafter(success=success, done=done)
    policy = FixedPolicy(torch, action=4)
    result = evaluate_episode(
        torch,
        policy,
        environment,
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        max_steps=max_steps,
        mode="gif",
        fps=2,
    )
    assert (result.success, result.died, result.steps) == expected
    assert environment.actions == [4] * expected[2]
    assert all(shape == (1, *MODEL_IMAGE_SHAPE) for shape in policy.seen_shapes)


def _zero_policy_checkpoint(torch: Any, path: Path) -> Path:
    model = build_q_network(torch)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return save_policy_checkpoint(torch, model, path, training_steps=1)


def test_gif_playback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    torch = pytest.importorskip("torch")
    imageio = pytest.importorskip("imageio.v2")
    checkpoint = _zero_policy_checkpoint(torch, tmp_path / POLICY_FILENAME)
    generated = run_playback(
        checkpoint,
        episodes=1,
        fps=2,
        max_steps=2,
        seed=0,
        output_dir=tmp_path / "gifs",
    )
    assert len(generated) == 1
    assert generated[0].name == "episode_001_collect_diamond.gif"
    frames = imageio.mimread(generated[0])
    assert 2 <= len(frames) <= 3
    reader = imageio.get_reader(generated[0])
    try:
        assert reader.get_data(0).shape == (512, 512, 3)
    finally:
        reader.close()
    output = capsys.readouterr().out
    assert "Goal: collect diamond." in output
    assert f"GIF: {generated[0].resolve()}" in output


def test_human_playback_with_dummy_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(
        "mha_exp_level2_cr.exp2_2.play_policy.time.sleep", lambda _: None
    )
    checkpoint = _zero_policy_checkpoint(torch, tmp_path / POLICY_FILENAME)
    assert run_playback(
        checkpoint, episodes=1, mode="human", max_steps=1, seed=0
    ) == []
    result_path = checkpoint.parent / "policy_evaluation" / "playback-results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == "2-2-cr-playback-v1"
    assert result["protocol"]["checkpoint_sha256"]
    assert result["protocol"]["target_achievement"] == TARGET_ACHIEVEMENT.value
    assert result["protocol"]["requested_episodes"] == 1
    assert result["protocol"]["mode"] == "human"
    assert result["protocol"]["crafter_length"] == 10_000
    assert result["episodes"] == [{
        "death": False,
        "episode": 1,
        "gif": None,
        "steps": 1,
        "success": False,
        "window_closed": False,
    }]
