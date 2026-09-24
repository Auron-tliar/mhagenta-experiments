from pathlib import Path
from typing import Any
import numpy as np
import gymnasium as gym

from .blocksworld import BlocksWorldEnv, BWSnapshot


class BWRecorder:
    def __init__(
            self,
            env: BlocksWorldEnv,
            path: Path | str,
            single_trace: bool = False
    ) -> None:
        env.expose_snapshot = True
        self._env = env
        self._path = Path(path).resolve()
        self._trace: list[BWSnapshot] = list()
        self._single_trace = single_trace
        self._idx = 0

    def save_and_increment(self) -> None:
        if not self._trace:
            return

        file = self._path / f'{self._idx:04d}.npz'
        np.savez_compressed(
            file,
            flat_state=np.stack([s.flat_state for s in self._trace]),
            arm_pos=np.array([s.arm_pos for s in self._trace]),
            arm_holds=np.array([s.arm_holds for s in self._trace]),
            last_reward=np.array([s.last_reward for s in self._trace]),
            total_reward=np.array([s.total_reward for s in self._trace]),
            action=np.array([s.action for s in self._trace]),
            legal=np.array([s.legal for s in self._trace]),
        )
        self._idx += 1
        self._trace = list()

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        state, info = self._env.reset(seed=seed, options=options)
        if not self._single_trace and self._trace:
            self.save_and_increment()
        self._trace.append(info['snapshot'])
        return state, info

    def step(
            self,
            action: gym.core.ActType
    ) -> tuple[gym.core.ObsType, gym.core.SupportsFloat, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self._env.step(action)
        self._trace.append(info['snapshot'])
        return state, reward, terminated, truncated, info

    def close(self) -> None:
        self.save_and_increment()
        self._env.close()
