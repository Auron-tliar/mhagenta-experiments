"""Correlated numeric Blocks World reset/action service for direct learning."""

from typing import Any

import numpy as np
from mhagenta.environment import MHAEnvBase


class Environment(MHAEnvBase):
    """Expose one environment; reset seeds are supplied by the fixed schedule."""

    def __init__(self, initial_state: dict[str, Any], *, table_len: int = 5, num_blocks: int = 8) -> None:
        self._env = None
        self._dimensions = {"table_len": table_len, "num_blocks": num_blocks}
        super().__init__(initial_state)

    def _environment(self):
        if self._env is None:
            from mha_env_blocksworld import BlocksWorldEnv
            self._env = BlocksWorldEnv(**self._dimensions, symbolic=False)
            self._env.expose_snapshot = True
        return self._env

    def on_observe(self, state: dict, sender_id: str, **kwargs: Any) -> tuple[dict, dict]:
        """Return current state while preserving the request identity."""
        if state["observation"] is None or state["closed"]:
            raise ValueError("Observation outside an open episode.")
        state["observations"] += 1
        return state, {**kwargs, "observation": state["observation"]}

    def on_action(self, state: dict, sender_id: str, **kwargs: Any) -> tuple[dict, dict]:
        """Apply an atomic action, scheduled reset, or acknowledged close."""
        action = kwargs["action"]
        if action == "close":
            if self._env is not None:
                self._env.close()
            state["closed"] = True
            return state, {**kwargs, "legal": True}
        if state["closed"]:
            raise ValueError("Action after environment closure.")
        env = self._environment()
        if action == "reset":
            observation, _ = env.reset(seed=kwargs["seed"])
            prefix = kwargs.get("reset_actions", [])
            if prefix and kwargs.get("mode") != "demo":
                raise ValueError("Only demonstrations may have perturbed starts.")
            for setup_action in prefix:
                if isinstance(setup_action, bool) or setup_action not in range(4):
                    raise ValueError("Invalid demonstration setup action.")
                observation, _, done, cutoff, info = env.step(setup_action)
                if not info["snapshot"].legal or done or cutoff:
                    raise ValueError("Invalid demonstration setup trajectory.")
                state["setup_actions"] += 1
            state["resets"] += 1
            legal, terminated, truncated = True, False, False
        else:
            if isinstance(action, bool) or action not in range(4):
                raise ValueError("Invalid atomic action.")
            observation, _, terminated, truncated, info = env.step(action)
            legal = bool(info["snapshot"].legal)
            state["actions"] += 1
            state["illegal_actions"] += int(not legal)
        state["observation"] = np.asarray(observation, dtype=np.uint8).tolist()
        return state, {**kwargs, "legal": legal, "terminated": terminated, "truncated": truncated}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"] = None
        return state


def initial_state() -> dict:
    """Return all persisted environment fields."""
    return {"observation": None, "observations": 0, "actions": 0, "setup_actions": 0,
            "resets": 0, "illegal_actions": 0, "closed": False}
