"""Self-contained fixed RGB Crafter environment for experiment 2-5-CR."""

from __future__ import annotations

from logging import INFO
from pathlib import Path
from typing import Any

from mhagenta.environment import MHAEnvBase

from .contracts import (
    A_CLOSE,
    K_ACTION,
    K_ATOMIC_ID,
    K_CONTRACT_ERROR,
    K_DEAD,
    K_DONE,
    K_ILLEGAL_ACTION,
    K_LETHAL_MOVEMENT,
    K_NEW_ACHIEVEMENTS,
    K_OBSERVATION,
    K_OBSERVATION_DIGEST,
    K_OBSERVATION_ID,
    K_OWNER_ID,
    K_REQUESTER,
    K_REWARD,
    as_rgb_frame,
    rgb_sha256,
)

MAX_EPISODE_LEN = 1000
_DIRECTION_DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _fail(state: dict[str, Any], code: str, **details: Any) -> None:
    if state["failure"] is None:
        state["failure"] = {"code": code, **details}


class CrafterNeurosymbolicEnvironment(MHAEnvBase):
    """Run one non-resetting episode and expose only RGB and public status."""

    action_context_fields = (K_ATOMIC_ID, K_OWNER_ID, K_REQUESTER, K_ACTION)

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = int(init_state.pop("seed"))
        init_state["environment_seed"] = self._seed
        self._record = bool(init_state.pop("record", False))
        self._artifact_root = str(init_state.pop("artifact_root", "/out"))
        self._recording_prefix = str(init_state.pop("recording_prefix", ""))
        self._expected_agent_id = str(init_state.pop("expected_agent_id"))
        self._crafter: Any = None
        self._env: Any = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_crafter import CrafterEnv, Recorder

        self._crafter = CrafterEnv(
            area=(64, 64), view=(9, 9), size=(64, 64), seed=self._seed,
            length=MAX_EPISODE_LEN, no_mobs=True, symbolic=False,
            daylight_effects=False, sleep_effects=False,
        )
        self._env = self._crafter
        if self._record:
            self._env = Recorder(
                self._crafter, Path(self._artifact_root) / "videos", save_stats=False,
                save_episode=False, video_size=(144, 144), video_fps=2,
            )
        self._env.reset()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_crafter"] = state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()

    def on_observe(
        self, state: dict[str, Any], sender_id: str, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return one transient correlated RGB frame."""

        observation_id = kwargs.get(K_OBSERVATION_ID, kwargs.get("observation_seq"))
        state["observation_count"] += 1
        error: str | None = None
        if sender_id != self._expected_agent_id:
            error = "unexpected-observation-requester"
        elif type(observation_id) is not int or observation_id != state["observation_count"]:
            error = "invalid-observation-id"
        elif state["closed"]:
            error = "observation-after-close"
        frame = None
        digest = None
        if error is None:
            try:
                frame = as_rgb_frame(self._env.render())
                digest = rgb_sha256(frame)
            except (TypeError, ValueError) as exc:
                error = "invalid-rgb-render"
                _fail(state, error, message=str(exc))
        if error is not None:
            _fail(state, error)
        state["last_observation_sha256"] = digest
        if "observation_history" in state:
            state["observation_history"].append(
                {"observation_seq": observation_id, "observation_digest": digest, "error": error}
            )
        return state, {
            K_OBSERVATION: frame.tolist() if frame is not None else [],
            K_OBSERVATION_ID: observation_id,
            K_OBSERVATION_DIGEST: digest,
            K_CONTRACT_ERROR: error,
        }

    def _close(self, state: dict[str, Any], sender_id: str) -> tuple[dict[str, Any], None]:
        if sender_id != self._expected_agent_id:
            _fail(state, "unexpected-close-requester")
            return state, None
        if state["closed"]:
            return state, None
        path = None
        close = getattr(self._env, "close", None)
        if callable(close):
            path = close()
        if path is not None:
            resolved = Path(path).resolve()
            root = Path(self._artifact_root).resolve()
            if not resolved.is_file() or root not in resolved.parents:
                _fail(state, "invalid-recording-path")
            else:
                relative = resolved.relative_to(root).as_posix()
                state["recording_path"] = "/".join(
                    part for part in (self._recording_prefix.strip("/"), relative) if part
                )
        state["close_count"] += 1
        state["closed"] = True
        if self._log_func is not None:
            self.log(INFO, "Crafter RGB environment closed.")
        return state, None

    def on_action(
        self, state: dict[str, Any], sender_id: str, **kwargs: Any
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply an unchanged native action or the operational close command."""

        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            return self._close(state, sender_id)
        state["action_count"] += 1
        atomic_id = kwargs.get(K_ATOMIC_ID)
        context = {field: kwargs.get(field) for field in self.action_context_fields}
        error = self._validate_action_request(state, sender_id, kwargs, action)
        if (
            error is None
            and (type(atomic_id) is not int or atomic_id != state["action_count"])
        ):
            error = "invalid-atomic-id"
        if error is not None:
            _fail(state, error)
            return state, {
                **context, K_REWARD: 0.0, K_DONE: state["terminal"], K_DEAD: False,
                K_ILLEGAL_ACTION: False, K_LETHAL_MOVEMENT: False,
                K_NEW_ACHIEVEMENTS: [], K_CONTRACT_ERROR: error,
            }

        target_material = None
        target_position = None
        delta = _DIRECTION_DELTAS.get(action)
        if delta is not None:
            target = self._crafter._player.pos + delta
            target_position = int(target[0]), int(target[1])
            target_material = self._crafter._world[target][0]
        _, reward, done, info = self._env.step(action)
        illegal = info.get(K_ILLEGAL_ACTION)
        inventory, achievements = info.get("inventory"), info.get("achievements")
        if type(illegal) is not bool or not isinstance(inventory, dict) or not isinstance(achievements, dict):
            raise RuntimeError("Crafter returned an incomplete public status.")
        position = tuple(int(value) for value in self._crafter._player.pos)
        lethal = bool(delta is not None and target_material == "lava" and target_position == position)
        counts = state["achievement_counts"]
        new = sorted(name for name, count in achievements.items() if int(count) > int(counts.get(name, 0)))
        state["native_action_count"] += 1
        state["action_history"].append(dict(context))
        state["status_count"] += 1
        state["illegal_count"] += int(illegal)
        state["lethal_count"] += int(lethal)
        state["terminal_count"] += int(bool(done))
        state["terminal"] = bool(done)
        state["inventory"] = {name: int(count) for name, count in inventory.items()}
        state["achievement_counts"] = {name: int(count) for name, count in achievements.items()}
        state["achievements"] = sorted(name for name, count in achievements.items() if int(count) > 0)
        status = {
            **context, K_REWARD: float(reward), K_DONE: bool(done),
            K_DEAD: int(inventory.get("health", 0)) <= 0,
            K_ILLEGAL_ACTION: illegal, K_LETHAL_MOVEMENT: lethal,
            K_NEW_ACHIEVEMENTS: new, K_CONTRACT_ERROR: None,
        }
        state["status_history"].append(dict(status))
        return state, status

    def _validate_action_request(
        self,
        state: dict[str, Any],
        sender_id: str,
        kwargs: dict[str, Any],
        action: Any,
    ) -> str | None:
        """Validate the payload fields owned by this environment class."""

        if sender_id != self._expected_agent_id:
            return "unexpected-action-requester"
        if set(kwargs) != set(self.action_context_fields):
            return "invalid-action-payload"
        if type(action) is not int or action not in range(len(self._crafter.action_names)):
            return "invalid-native-action"
        if state["terminal"] or state["closed"]:
            return "action-after-terminal"
        return None


def initial_state() -> dict[str, Any]:
    """Return declared compact environment evidence state."""

    return {
        "environment_seed": None,
        "observation_count": 0, "action_count": 0, "native_action_count": 0,
        "status_count": 0, "illegal_count": 0, "lethal_count": 0,
        "terminal_count": 0, "terminal": False, "inventory": {},
        "action_history": [], "status_history": [],
        "achievement_counts": {}, "achievements": [], "last_observation_sha256": None,
        "close_count": 0, "closed": False, "recording_path": None, "failure": None,
    }


__all__ = ["CrafterNeurosymbolicEnvironment", "initial_state"]
