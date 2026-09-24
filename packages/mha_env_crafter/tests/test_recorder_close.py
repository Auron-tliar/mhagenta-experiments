"""Focused tests for public partial-video finalization."""

import imageio
import numpy as np
import pytest

from mha_env_crafter.crafter.recorder import VideoRecorder


class _Env:
    def __init__(self, terminal: bool) -> None:
        self.terminal = terminal

    def reset(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def render(self, size):
        return np.zeros((*size, 3), dtype=np.uint8)

    def step(self, action):
        info = {"achievements": {"wood": int(self.terminal)}}
        return self.reset(), 0.0, self.terminal, info


@pytest.mark.parametrize("terminal", [False, True])
def test_video_close_saves_once_and_is_readable(tmp_path, terminal: bool) -> None:
    recorder = VideoRecorder(_Env(terminal), tmp_path, size=(8, 8), fps=2)
    recorder.reset()
    recorder.step(0)
    recorder.close()
    recorder.close()

    videos = list(tmp_path.glob("*.mp4"))
    assert len(videos) == 1
    reader = imageio.get_reader(videos[0])
    try:
        assert reader.get_data(0).size > 0
    finally:
        reader.close()
