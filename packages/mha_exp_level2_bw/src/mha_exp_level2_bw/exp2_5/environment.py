"""Numeric Blocks World environment adapter for experiment 2-5-BW."""

from __future__ import annotations

from typing import Any

import numpy as np
from mhagenta.environment import MHAEnvBase
from mhagenta import Orchestrator

from .grounding import NUM_BLOCKS, TABLE_LEN


K_ACTION = "action"
K_ACTION_CONTEXT = "action_context"
K_LEGAL = "legal"
K_OBSERVATION = "observation"
K_REWARD = "reward"
K_STATE = "state"

A_PICK_UP = 0
A_PUT_DOWN = 1
A_MOVE_LEFT = 2
A_MOVE_RIGHT = 3
A_CLOSE = "close"
ATOMIC_ACTIONS = {A_PICK_UP, A_PUT_DOWN, A_MOVE_LEFT, A_MOVE_RIGHT}


class NumericBlocksWorldEnvironment(MHAEnvBase):
    """Expose deterministic numeric Blocks World observations and compact counts."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = init_state.pop("seed", None)
        self._record = bool(init_state.pop("record", False))
        self._table_len = int(init_state.pop("table_len", TABLE_LEN))
        self._num_blocks = int(init_state.pop("num_blocks", NUM_BLOCKS))
        self._env: Any = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_blocksworld import BWRecorder, BlocksWorldEnv

        environment = BlocksWorldEnv(
            table_len=self._table_len,
            num_blocks=self._num_blocks,
            render_mode="rgb_array" if self._record else None,
            symbolic=False,
        )
        environment.expose_snapshot = True
        self._env = (
            BWRecorder(
                environment,
                path=f"/{Orchestrator.SAVE_SUBDIR}",
                single_trace=True,
            )
            if self._record
            else environment
        )
        observation, _ = self._env.reset(seed=self._seed)
        self.state[K_STATE] = np.asarray(observation, dtype=np.uint8).tolist()

    def on_observe(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the complete current observation with a monotonic identity."""

        state["observation_count"] += 1
        state["observation_id"] += 1
        return state, {
            K_OBSERVATION: state[K_STATE],
            "observation_id": state["observation_id"],
        }

    def on_action(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply one atomic action and return legality without retaining history."""

        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            if self._env is not None:
                self._env.close()
            state["closed"] = True
            return state, None
        if not isinstance(action, int) or action not in ATOMIC_ACTIONS:
            if state["failure"] is None:
                state["failure"] = {"code": "invalid-action"}
            return state, {
                K_ACTION: action,
                K_REWARD: None,
                K_LEGAL: False,
                "error": "missing-or-invalid-action",
            }

        observation, reward, terminated, truncated, info = self._env.step(action)
        snapshot = info.get("snapshot")
        legal = bool(snapshot.legal) if snapshot is not None else float(reward) != -0.5
        state[K_STATE] = np.asarray(observation, dtype=np.uint8).tolist()
        state["action_count"] += 1
        if not legal and state["failure"] is None:
            state["failure"] = {"code": "illegal-action", "action": action}
        return state, {
            K_ACTION: action,
            K_REWARD: float(reward),
            K_LEGAL: legal,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()


def environment_initial_state(
    *,
    seed: int | None,
    record: bool,
    treatment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the JSON-native initial state for the numeric environment."""

    return {
        "treatment": dict(treatment or {}),
        "seed": seed,
        "record": record,
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        K_STATE: [],
        "observation_count": 0,
        "observation_id": 0,
        "action_count": 0,
        "closed": False,
        "failure": None,
    }
