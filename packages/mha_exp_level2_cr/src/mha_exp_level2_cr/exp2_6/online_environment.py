"""Natural RGB worlds with explicit acknowledged 900-action episode resets."""

import json
from pathlib import Path

from mha_exp_common.utils import Seeder
from ..exp2_5.environment import CrafterNeurosymbolicEnvironment, initial_state as baseline_state

RUNS = 20


def episode_seed(run: int, episode: int) -> int:
    """Keep all twenty runs' world streams disjoint, starting at seeds 1000–1019."""
    if not 0 <= run < RUNS or episode < 0:
        raise ValueError("Invalid run or episode index")
    return Seeder(run).environment + Seeder.ENV_MULTIPLIER * episode


def initial_state(run: int, agent_id: str, output: str = "/out") -> dict:
    """Declare cumulative evidence separately from replaceable world state."""
    return {**baseline_state(), "seed": episode_seed(run, 0), "expected_agent_id": agent_id,
            "artifact_root": output, "record": False, "run": run, "episode": 0,
            "episode_actions": 0, "episodes": [], "resets": 0, "controls": 0,
            "episode_reason": None, "last_control": None}


class ContinuingEnvironment(CrafterNeurosymbolicEnvironment):
    """Preserve transport counters across worlds, but never their inventory or map."""

    def _record_episode(self, state: dict, reason: str) -> None:
        if state["episodes"] and state["episodes"][-1]["episode"] == state["episode"]:
            return
        state["episodes"].append({"episode": state["episode"], "seed": self._seed,
                                  "actions": state["episode_actions"], "reason": reason,
                                  "inventory": dict(state["inventory"]),
                                  "achievements": dict(state["achievement_counts"])})

    def on_action(self, state: dict, sender_id: str, **kwargs):
        """Apply one native action, reset a completed world, or acknowledge closure."""
        action = kwargs.get("action")
        if action in {"reset", "close"}:
            control = kwargs.get("control_id")
            error = None
            if (sender_id != self._expected_agent_id or not isinstance(control, str)
                    or set(kwargs) != {"action", "control_id"}):
                error = "invalid-control-request"
            elif state["last_control"] and state["last_control"]["control_id"] == control:
                if state["last_control"]["action"] == action:
                    return state, dict(state["last_control"])
                error = "reused-control-id"
            elif state["closed"] or action == "reset" and not state["terminal"]:
                error = "invalid-control-boundary"
            if error is None:
                state["controls"] += 1
                if action == "close":
                    self._record_episode(state, state["episode_reason"] or "time_limit")
                    self._close(state, sender_id)
                else:
                    self._record_episode(state, state["episode_reason"])
                    state["episode"] += 1
                    self._seed = episode_seed(state["run"], state["episode"])
                    self._build_env()
                    state.update(terminal=False, inventory={}, achievement_counts={}, achievements=[],
                                 episode_actions=0, episode_reason=None, environment_seed=self._seed)
                    state["resets"] += 1
            response = {"action": action, "control_id": control, "episode": state["episode"],
                        "closed": state["closed"], "contract_error": error}
            if error is None:
                state["last_control"] = response
            else:
                state["failure"] = error
            return state, response
        state, status = super().on_action(state, sender_id, **kwargs)
        if status["contract_error"] is None:
            state["episode_actions"] += 1
            reason = ("death" if status["dead"] else "diamond" if state["inventory"].get("diamond", 0)
                      else "action_budget" if state["episode_actions"] >= 900 else
                      "native_terminal" if status["done"] else None)
            state["episode_reason"] = reason
            if reason is not None and not status["done"]:
                state["terminal_count"] += 1
            status["done"] = state["terminal"] = reason is not None
            status.update(episode=state["episode"], episode_reason=reason)
            if reason is not None:
                self._record_episode(state, reason)
            directory = Path(self._artifact_root)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "actions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(status, allow_nan=False) + "\n")
            # The complete trace is streamed once; autosaves stay compact.
            state["action_history"].clear()
            state["status_history"].clear()
        return state, status
