"""Run the rule-based reactive agent in the symbolic Blocks World."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from logging import INFO, WARNING
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Literal, cast

from numpy import random

from mhagenta import ActionStatus, Observation, Orchestrator
from mhagenta.bases import LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, LLState, PerceptorState

import mha_exp_common
from mha_exp_common.batch import normalize_runs, run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import ACTUATOR, LLREASONER, PERCEPTOR
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name

from .reporting import process_execution_metrics
from .treatment import state_digest, treatment_for_run


DURATION = 120.0
ENVIRONMENT_OVERRUN = 30.0
TABLE_LEN = 10
NUM_BLOCKS = 30
SAVE_SUBDIR = Orchestrator.SAVE_SUBDIR
MAX_EPISODE_ACTIONS = 250
RECORD: Literal["all", "first", "none"] = "all"
VERBOSE = True

K_ACTION = "action"
K_STATE = "state"
K_OBSERVATION = "observation"
K_LEGAL = "legal"
K_CLOSED = "closed"
K_ERROR = "error"
K_TASK_SEED = "task_seed"

K_N_SUCCESSES = "n_successes"
K_EP_LENGTHS = "ep_lengths"

A_RESET = "reset"
A_CLOSE = "close"
A_PICK_UP = "PickUp"
A_PUT_DOWN = "PutDown"
A_MOVE_LEFT = "MoveLeft"
A_MOVE_RIGHT = "MoveRight"
NATIVE_ACTIONS = (A_PICK_UP, A_PUT_DOWN, A_MOVE_LEFT, A_MOVE_RIGHT)

F_HAND_EMPTY = "HandEmpty"
F_HOLDING = "Holding"
F_ON = "On"
F_ABOVE = "Above"

PRED_PATTERN = re.compile(r"^(?P<PRED_NAME>[A-Za-z_]\w*)\((?P<ARGS>.*)\)$")
RUNTIME_FAILURE_MARKERS = (
    "[error]",
    "[critical]",
    "traceback",
    "exceptiongroup",
    "caught exception",
    "could not send message",
    "failed to save state",
)


class TestEnvironment(MHAEnvBase):
    """Expose a deterministic symbolic Blocks World through the environment API."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = cast(int | None, init_state.get("seed"))
        self._record = bool(init_state.get("record", False))
        super().__init__(init_state)
        self._env: Any = None
        self._action_values: dict[str, int] = {}
        self._build_env()

    def on_observe(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the current symbolic Blocks World state."""
        state["observation_requests"] += 1
        return state, {K_OBSERVATION: list(state[K_STATE])}

    def on_action(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply one readable action and return a compact execution status."""
        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            if state[K_CLOSED]:
                return state, None
            state["close_requests"] += 1
            state[K_CLOSED] = True
            if self._env is not None:
                self._env.close()
            self.log(
                INFO,
                "Environment closed: "
                f"native={state['native_actions']}, illegal={state['illegal_actions']}, "
                f"status_actions={state['status_actions']}.",
            )
            return state, None

        state["status_actions"] += 1

        if action == A_RESET:
            task_seed = kwargs.get(K_TASK_SEED)
            if task_seed is None:
                state[K_STATE], _ = self._env.reset()
                return state, {K_LEGAL: True}
            if type(task_seed) is not int:
                return state, {K_LEGAL: False, K_ERROR: "invalid task_seed"}
            state[K_STATE], _ = self._env.reset(seed=task_seed)
            state["current_task_seed"] = task_seed
            return state, {K_LEGAL: True, K_TASK_SEED: task_seed}

        native_action = self._action_values.get(action) if isinstance(action, str) else None
        if native_action is None:
            error = f"missing or invalid native action: {action!r}"
            state["illegal_actions"] += 1
            state["errors"].append(error)
            return state, {K_LEGAL: False, K_ERROR: error}

        observation, reward, _, _, info = self._env.step(native_action)
        snapshot = info.get("snapshot")
        legal = bool(snapshot.legal) if snapshot is not None else float(reward) != -0.5
        state[K_STATE] = list(observation)
        state["native_actions"] += 1
        state["illegal_actions"] += int(not legal)
        return state, {K_LEGAL: legal}

    def _build_env(self) -> None:
        from mha_env_blocksworld import BWRecorder, BlocksWorldEnv

        self._action_values = {
            A_PICK_UP: BlocksWorldEnv.Actions.PICK_UP.value,
            A_PUT_DOWN: BlocksWorldEnv.Actions.PUT_DOWN.value,
            A_MOVE_LEFT: BlocksWorldEnv.Actions.MOVE_LEFT.value,
            A_MOVE_RIGHT: BlocksWorldEnv.Actions.MOVE_RIGHT.value,
        }

        environment = BlocksWorldEnv(
            table_len=TABLE_LEN,
            num_blocks=NUM_BLOCKS,
            render_mode="rgb_array" if self._record else None,
            symbolic=True,
        )
        environment.expose_snapshot = True
        self._env = (
            BWRecorder(environment, path=f"/{SAVE_SUBDIR}", single_trace=True)
            if self._record
            else environment
        )
        self.state[K_STATE], _ = self._env.reset(self._seed)
        expected_digest = self.state.get("initial_state_digest")
        actual_digest = state_digest(self.state[K_STATE])
        if expected_digest != actual_digest:
            raise RuntimeError(
                f"Initial state digest mismatch: expected {expected_digest}, got {actual_digest}."
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()


class Position:
    """Column and row of one block in the current symbolic observation."""

    def __init__(self, col: int = -1, row: int = -1) -> None:
        self.col = col
        self.row = row

    def is_on(self, other: Position) -> bool:
        return (
            self.col != -1
            and self.row != -1
            and other.col != -1
            and other.row != -1
            and self.col == other.col
            and self.row == other.row + 1
        )

    def as_list(self) -> list[int]:
        """Return a JSON-serializable representation."""
        return [self.col, self.row]

    def __str__(self) -> str:
        return f"({self.col}, {self.row})"

    def __repr__(self) -> str:
        return f"Position({self.col}, {self.row})"


class TestLLReasoner(LLReasonerBase):
    """Select native actions using the greedy symbolic Blocks World rule tree."""

    def __init__(
        self,
        *args: Any,
        table_len: int = TABLE_LEN,
        num_blocks: int = NUM_BLOCKS,
        max_episode_actions: int = MAX_EPISODE_ACTIONS,
        **kwargs: Any,
    ) -> None:
        if num_blocks < 2:
            raise ValueError("Experiment 2-1-BW requires at least two blocks.")
        if table_len <= 0:
            raise ValueError("Experiment 2-1-BW requires a positive table length.")
        if max_episode_actions <= 0:
            raise ValueError("max_episode_actions must be positive.")
        super().__init__(*args, **kwargs)
        self._table_len = table_len
        self._num_blocks = num_blocks
        self._max_episode_actions = max_episode_actions
        self._rng: random.Generator = random.default_rng()
        self._actuator_id = ""
        self._perceptor_id = ""
        self._clears: set[str] = set()
        self._holding = ""
        self._arm_pos = -1
        self._n_b_digits = len(str(self._num_blocks - 1))
        self._n_t_digits = len(str(self._table_len - 1))
        self._cur_goal_top = ""
        self._cur_goal_bot = ""
        self._episode_started = 0.0
        self._episode_decision_start = 0
        self._episode_reset_pending = False
        self._tasks: list[dict[str, Any]] = []
        self._block_pos = {
            f"B{i:0{self._n_b_digits}d}": Position()
            for i in range(self._num_blocks)
        }

    def on_init(self, **kwargs: Any) -> None:
        if "seed" in kwargs:
            self._rng = random.default_rng(kwargs["seed"])
        tasks = kwargs.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError("Experiment 2-1-BW requires exactly one frozen task per run.")
        self._tasks = [dict(tasks[0])]

    def on_first(self, state: LLState) -> LLState:
        self._actuator_id = state.directory.internal.actuation[0].module_id
        self._perceptor_id = state.directory.internal.perception[0].module_id
        self._episode_started = perf_counter()
        self._episode_decision_start = len(state["decision_trace"])
        state.outbox.request_observation(self._perceptor_id)
        return state

    def on_last(self, state: LLState) -> LLState:
        if state["current_goal"] and not self._episode_reset_pending:
            state["open_episode"] = {
                "episode_id": len(state["episode_results"]),
                "goal": dict(state["current_goal"]),
                "actions": state["current_episode_length"],
                "elapsed_seconds": max(0.0, perf_counter() - self._episode_started),
                "decision_start": self._episode_decision_start,
                "decision_end": len(state["decision_trace"]) - 1,
                "censoring": {"status": "right_censored", "reason": "external_execution_end"},
            }
        self.log(
            INFO,
            "LL summary: "
            f"observations={state['observations']}, statuses={state['action_statuses']}, "
            f"successes={state[K_N_SUCCESSES]}, illegal={state['illegal_actions']}, "
            f"active_seconds={state['active_seconds']:.6f}.",
        )
        return state

    def _record_episode(self, state: LLState, *, outcome: str, failure_reason: str | None) -> None:
        """Persist one completed controller episode before requesting its reset."""

        state["episode_results"].append({
            "episode_id": len(state["episode_results"]),
            "goal": dict(state["current_goal"]),
            "outcome": outcome,
            "failure_reason": failure_reason,
            "actions": state["current_episode_length"],
            "elapsed_seconds": max(0.0, perf_counter() - self._episode_started),
            "decision_start": self._episode_decision_start,
            "decision_end": len(state["decision_trace"]) - 1,
        })
        self._episode_reset_pending = True

    @staticmethod
    def _parse_predicate(predicate: str) -> tuple[str, list[str]]:
        match = PRED_PATTERN.match(predicate)
        if match is None:
            return "", []
        arguments = match.group("ARGS")
        return match.group("PRED_NAME"), arguments.split(", ") if arguments else []

    def _parse_table(self, observation: Sequence[str]) -> None:
        belows: dict[str, str] = {}
        self._clears.clear()
        self._holding = ""
        for position in self._block_pos.values():
            position.col = -1
            position.row = -1

        for predicate in observation:
            fluent, arguments = self._parse_predicate(predicate)
            if fluent == F_ABOVE and len(arguments) == 1:
                self._arm_pos = int(arguments[0][1:])
            elif fluent == F_HAND_EMPTY:
                self._holding = ""
            elif fluent == F_HOLDING and len(arguments) == 1:
                self._holding = arguments[0]
            elif fluent == F_ON and len(arguments) == 2:
                belows[arguments[1]] = arguments[0]

        for location in range(self._table_len):
            current = belows.get(f"t{location:0{self._n_t_digits}d}", "")
            height = 0
            clear_candidate = ""
            while current:
                if current not in self._block_pos:
                    raise ValueError(f"Unknown block in observation: {current!r}")
                clear_candidate = current
                self._block_pos[current].col = location
                self._block_pos[current].row = height
                height += 1
                current = belows.get(current, "")
            if clear_candidate:
                self._clears.add(clear_candidate)

    def _add_goal(self, state: LLState) -> None:
        task_index = state["task_index"]
        if not 0 <= task_index < len(self._tasks):
            raise RuntimeError("Task index is outside the frozen manifest.")
        task = self._tasks[task_index]
        top, bottom = str(task["top"]), str(task["bottom"])
        if top not in self._block_pos or bottom not in self._block_pos or top == bottom:
            raise RuntimeError(f"Invalid frozen Blocks World goal: {top!r} on {bottom!r}.")
        if self._block_pos[top].is_on(self._block_pos[bottom]):
            raise RuntimeError(f"Frozen task {task['task_id']} is initially satisfied.")
        self._cur_goal_top, self._cur_goal_bot = top, bottom
        state["current_goal"] = {
            "top": top, "bottom": bottom, "task_id": task["task_id"],
            "difficulty": task["difficulty"], "seed": task["seed"],
        }
        self.log(INFO, f"Selected frozen task {task['task_id']}: {top} on {bottom}.")

    def _finish_task(self, state: LLState) -> None:
        """Terminate the one-goal run after persisting its outcome."""

        state["task_index"] += 1
        outcome = state["episode_results"][-1]["outcome"]
        if outcome == "success":
            state["phase"] = "complete"
            state["terminal_reason"] = "goal_achieved"
            state.outbox.terminate_agent("2-1-BW fixed goal complete")
        else:
            state["phase"] = "failed"
            state["terminal_reason"] = state["episode_results"][-1]["failure_reason"]
            state.outbox.terminate_agent("2-1-BW fixed goal failed")

    def _add_random_goal(self, state: LLState) -> None:
        """Retained reference implementation for historical protocol reading."""
        for _ in range(self._num_blocks * 2):
            top_index, bottom_index = self._rng.choice(
                self._num_blocks,
                2,
                replace=False,
            ).tolist()
            top = f"B{top_index:0{self._n_b_digits}d}"
            bottom = f"B{bottom_index:0{self._n_b_digits}d}"
            if top != bottom and not self._block_pos[top].is_on(self._block_pos[bottom]):
                break
        else:
            raise RuntimeError("Could not select a distinct unsatisfied Blocks World goal.")

        if top not in self._block_pos or bottom not in self._block_pos or top == bottom:
            raise RuntimeError(f"Invalid Blocks World goal: {top!r} on {bottom!r}.")
        self._cur_goal_top, self._cur_goal_bot = top, bottom
        goal = {"top": top, "bottom": bottom}
        state["current_goal"] = goal
        self.log(INFO, f"Selected goal {top} on {bottom}.")

    def _get_nearest_non_target_col(self) -> int:
        excluded = {
            self._block_pos[self._cur_goal_top].col,
            self._block_pos[self._cur_goal_bot].col,
        }
        for difference in range(self._table_len):
            for direction in (-1, 1):
                target = self._arm_pos + difference * direction
                if 0 <= target < self._table_len and target not in excluded:
                    return target
        raise RuntimeError("No non-target Blocks World column is available.")

    def _is_clear(self, block: str) -> bool:
        return block in self._clears

    def _decision_context(
        self,
        *,
        action: str,
        reason: str,
        episode_step: int,
    ) -> dict[str, Any]:
        return {
            "arm_col": self._arm_pos,
            "holding": self._holding or None,
            "top_position": self._block_pos[self._cur_goal_top].as_list(),
            "bottom_position": self._block_pos[self._cur_goal_bot].as_list(),
            "top_clear": self._is_clear(self._cur_goal_top),
            "bottom_clear": self._is_clear(self._cur_goal_bot),
            K_ACTION: action,
            "reason": reason,
            "episode_step": episode_step,
            K_LEGAL: None,
        }

    def _request_action(
        self,
        state: LLState,
        *,
        action: str,
        reason: str,
        task_seed: int | None = None,
    ) -> None:
        state["decision_trace"].append(
            self._decision_context(
                action=action,
                reason=reason,
                episode_step=state["current_episode_length"],
            )
        )
        kwargs: dict[str, Any] = {"action": action}
        if task_seed is not None:
            kwargs[K_TASK_SEED] = task_seed
        state.outbox.request_action(self._actuator_id, **kwargs)

    def on_observation(
        self,
        state: LLState,
        sender: str,
        observation: Observation,
        **kwargs: Any,
    ) -> LLState:
        started = perf_counter()
        try:
            state["observations"] += 1
            content = observation.content
            if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
                raise ValueError(f"Received an invalid symbolic observation: {content!r}.")
            self._parse_table(content)

            if not self._cur_goal_top:
                self._add_goal(state)

            if self._check_goal():
                state[K_N_SUCCESSES] += 1
                state[K_EP_LENGTHS].append(state["current_episode_length"])
                self._record_episode(state, outcome="success", failure_reason=None)
                self.log(
                    INFO,
                    f"Completed goal {self._cur_goal_top} on {self._cur_goal_bot} "
                    f"in {state['current_episode_length']} actions.",
                )
                self._finish_task(state)
                return state

            if state["current_episode_length"] >= self._max_episode_actions:
                failure = {
                    "reason": "episode action limit reached",
                    "goal": dict(state["current_goal"]),
                    "episode_length": state["current_episode_length"],
                }
                state["logical_failures"].append(failure)
                self._record_episode(
                    state, outcome="failure", failure_reason="episode action limit reached"
                )
                self.log(
                    WARNING,
                    f"Resetting after {self._max_episode_actions} actions without success.",
                )
                self._finish_task(state)
                return state

            action: str | None = None
            reason = ""
            if self._holding and self._holding not in (self._cur_goal_top, self._cur_goal_bot):
                if self._arm_pos not in (
                    self._block_pos[self._cur_goal_top].col,
                    self._block_pos[self._cur_goal_bot].col,
                ):
                    action, reason = A_PUT_DOWN, "put down irrelevant block"
                else:
                    target = self._get_nearest_non_target_col()
                    action = A_MOVE_LEFT if target < self._arm_pos else A_MOVE_RIGHT
                    reason = "move irrelevant block away from target columns"
            elif self._holding == self._cur_goal_bot:
                if self._arm_pos == self._block_pos[self._cur_goal_top].col:
                    action = (
                        A_MOVE_RIGHT
                        if self._arm_pos != self._table_len - 1
                        else A_MOVE_LEFT
                    )
                    reason = "move bottom block away from top block column"
                else:
                    action, reason = A_PUT_DOWN, "relocate bottom block"
            elif not self._is_clear(self._cur_goal_bot):
                if not self._holding:
                    if self._arm_pos == self._block_pos[self._cur_goal_bot].col:
                        action, reason = A_PICK_UP, "excavate bottom block"
                    else:
                        action = (
                            A_MOVE_LEFT
                            if self._arm_pos > self._block_pos[self._cur_goal_bot].col
                            else A_MOVE_RIGHT
                        )
                        reason = "move toward bottom block column"
                elif self._holding == self._cur_goal_top:
                    if self._arm_pos == self._block_pos[self._cur_goal_bot].col:
                        action = A_MOVE_LEFT if self._arm_pos != 0 else A_MOVE_RIGHT
                        reason = "move top block away while excavating bottom block"
                    else:
                        action, reason = A_PUT_DOWN, "relocate top block while excavating bottom block"
            elif self._block_pos[self._cur_goal_bot].col == self._block_pos[self._cur_goal_top].col:
                if self._arm_pos == self._block_pos[self._cur_goal_bot].col:
                    action, reason = A_PICK_UP, "pick up bottom block from above top block"
                else:
                    action = (
                        A_MOVE_LEFT
                        if self._arm_pos > self._block_pos[self._cur_goal_bot].col
                        else A_MOVE_RIGHT
                    )
                    reason = "move toward shared target column"
            elif self._holding == self._cur_goal_top:
                if self._arm_pos == self._block_pos[self._cur_goal_bot].col:
                    action, reason = A_PUT_DOWN, "place top block on bottom block"
                else:
                    action = (
                        A_MOVE_LEFT
                        if self._arm_pos > self._block_pos[self._cur_goal_bot].col
                        else A_MOVE_RIGHT
                    )
                    reason = "carry top block toward bottom block"
            elif not self._is_clear(self._cur_goal_top):
                if self._arm_pos == self._block_pos[self._cur_goal_top].col:
                    action, reason = A_PICK_UP, "excavate top block"
                else:
                    action = (
                        A_MOVE_LEFT
                        if self._arm_pos > self._block_pos[self._cur_goal_top].col
                        else A_MOVE_RIGHT
                    )
                    reason = "move toward top block column"
            else:
                if self._arm_pos == self._block_pos[self._cur_goal_top].col:
                    action, reason = A_PICK_UP, "pick up clear top block"
                else:
                    action = (
                        A_MOVE_LEFT
                        if self._arm_pos > self._block_pos[self._cur_goal_top].col
                        else A_MOVE_RIGHT
                    )
                    reason = "move toward clear top block"

            if action is None:
                failure = {
                    "reason": "no rule selected an action",
                    "goal": dict(state["current_goal"]),
                }
                state["logical_failures"].append(failure)
                self._record_episode(
                    state, outcome="failure", failure_reason="no rule selected an action"
                )
                self.log(WARNING, "No rule selected an action; resetting the environment.")
                self._finish_task(state)
                return state
            else:
                selected_action = action

            self._request_action(
                state,
                action=selected_action,
                reason=reason,
            )
            return state
        finally:
            state["active_seconds"] += perf_counter() - started

    def _check_goal(self) -> bool:
        return bool(
            self._cur_goal_top
            and self._cur_goal_bot
            and self._cur_goal_top != self._cur_goal_bot
            and self._block_pos[self._cur_goal_top].is_on(self._block_pos[self._cur_goal_bot])
        )

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        started = perf_counter()
        try:
            state["action_statuses"] += 1
            status = action_status.status
            if not isinstance(status, dict):
                raise TypeError("Received a non-dictionary action status.")
            if not state["decision_trace"]:
                raise RuntimeError("Received an action status before selecting an action.")

            decision = state["decision_trace"][-1]
            if decision[K_LEGAL] is not None:
                raise RuntimeError("Received more than one status for the latest action.")
            legal = status.get(K_LEGAL)
            if type(legal) is not bool:
                raise ValueError(f"Action status contained invalid legality: {legal!r}.")
            decision[K_LEGAL] = legal

            if error := status.get(K_ERROR):
                raise RuntimeError(f"Environment rejected the action: {error}")

            action = decision[K_ACTION]
            if action == A_RESET:
                if legal:
                    state["current_episode_length"] = 0
                    self._episode_started = perf_counter()
                    self._episode_decision_start = len(state["decision_trace"])
                    self._episode_reset_pending = False
                    state["open_episode"] = None
                    self._cur_goal_top = ""
                    self._cur_goal_bot = ""
                else:
                    state["illegal_actions"] += 1
            elif action in NATIVE_ACTIONS:
                state["illegal_actions"] += int(not legal)
                state["current_episode_length"] += 1
            else:
                raise RuntimeError(f"Decision trace contained an invalid action: {action!r}.")

            state.outbox.request_observation(self._perceptor_id)
            return state
        finally:
            state["active_seconds"] += perf_counter() - started


class TestActuator(RMQActuatorBase):
    """Forward readable actions and compact statuses across the environment edge."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("No Blocks World environment is registered.")
        self._env_id = environment.address["env_id"]
        self._reasoner_id = state.directory.internal.ll_reasoning[0].module_id
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        self.act(self._env_id, action=A_CLOSE)
        self.log(
            INFO,
            "Actuator summary: "
            f"requests={state['requests']}, statuses={state['statuses']}, "
            f"active_seconds={state['active_seconds']:.6f}.",
        )
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        started = perf_counter()
        try:
            state["requests"] += 1
            self.act(self._env_id, **kwargs)
            return state
        finally:
            state["active_seconds"] += perf_counter() - started

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        started = perf_counter()
        try:
            state["statuses"] += 1
            state.outbox.send_status(self._reasoner_id, ActionStatus(dict(kwargs)))
            return state
        finally:
            state["active_seconds"] += perf_counter() - started


class TestPerceptor(RMQPerceptorBase):
    """Request and forward symbolic Blocks World observations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: PerceptorState) -> PerceptorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("No Blocks World environment is registered.")
        self._env_id = environment.address["env_id"]
        self._reasoner_id = state.directory.internal.ll_reasoning[0].module_id
        return state

    def on_last(self, state: PerceptorState) -> PerceptorState:
        self.log(
            INFO,
            "Perceptor summary: "
            f"requests={state['requests']}, observations={state['observations']}, "
            f"active_seconds={state['active_seconds']:.6f}.",
        )
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        started = perf_counter()
        try:
            state["requests"] += 1
            self.observe(self._env_id)
            return state
        finally:
            state["active_seconds"] += perf_counter() - started

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        started = perf_counter()
        try:
            state["observations"] += 1
            state.outbox.send_observation(
                self._reasoner_id,
                Observation(kwargs[K_OBSERVATION]),
            )
            return state
        finally:
            state["active_seconds"] += perf_counter() - started


def _log_has_runtime_error(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in RUNTIME_FAILURE_MARKERS)


def check_results(
    agent_states: dict[str, dict[str, Any]] | None,
    environment_state: dict[str, Any] | None,
    logs: Sequence[str],
    verbose: bool = False,
    expected_treatment: dict[str, Any] | None = None,
) -> bool:
    """Validate scientific success, topology activity, cleanup, and runtime logs."""
    valid = True
    for line in logs:
        if _log_has_runtime_error(line):
            print(f"Runtime error found in logs: {line.rstrip()}")
            valid = False

    if agent_states is None:
        print("Missing saved agent states")
        return False
    if environment_state is None:
        print("Missing saved environment state")
        return False

    required = {
        module_name(PERCEPTOR, 0),
        module_name(ACTUATOR, 0),
        module_name(LLREASONER, 0),
    }
    if missing := required - agent_states.keys():
        print(f"Missing saved module states: {sorted(missing)}")
        return False

    perceptor = agent_states[module_name(PERCEPTOR, 0)]
    actuator = agent_states[module_name(ACTUATOR, 0)]
    reasoner = agent_states[module_name(LLREASONER, 0)]
    if expected_treatment is None:
        run_id = reasoner.get("run_id")
        try:
            expected_treatment = treatment_for_run(run_id)
        except (TypeError, ValueError):
            expected_treatment = {}
    decision_trace = reasoner.get("decision_trace", [])
    trace_valid = isinstance(decision_trace, list) and all(
        isinstance(item, dict) for item in decision_trace
    )
    trace_actions_valid = trace_valid and all(
        item.get(K_ACTION) in {*NATIVE_ACTIONS, A_RESET}
        for item in decision_trace
    )
    native_trace_present = trace_valid and any(
        item.get(K_ACTION) in NATIVE_ACTIONS for item in decision_trace
    )

    completion_count = reasoner.get(K_N_SUCCESSES)
    episode_lengths = reasoner.get(K_EP_LENGTHS)
    completions_valid = (
        type(completion_count) is int
        and completion_count == 1
        and isinstance(episode_lengths, list)
        and len(episode_lengths) == 1
        and all(type(length) is int and 0 <= length <= MAX_EPISODE_ACTIONS for length in episode_lengths)
    )

    episode_results = reasoner.get("episode_results")
    one_goal_valid = (
        isinstance(episode_results, list)
        and len(episode_results) == 1
        and isinstance(episode_results[0], dict)
        and episode_results[0].get("outcome") == "success"
        and episode_results[0].get("failure_reason") is None
        and reasoner.get("phase") == "complete"
        and reasoner.get("terminal_reason") == "goal_achieved"
        and reasoner.get("task_index") == 1
        and reasoner.get("open_episode") is None
    )

    known_blocks = {f"B{index:0{len(str(NUM_BLOCKS - 1))}d}" for index in range(NUM_BLOCKS)}
    goal = reasoner.get("current_goal")
    goal_valid = (
        isinstance(goal, dict)
        and goal.get("top") in known_blocks
        and goal.get("bottom") in known_blocks
        and goal.get("top") != goal.get("bottom")
    )

    identity_keys = (
        "protocol_version", "treatment_id", "manifest_digest", "task_count",
        "run_id", "task_id", "seed", "reasoner_seed", "initial_state_digest",
        "table_len", "num_blocks", "top", "bottom",
    )
    treatment_valid = bool(expected_treatment) and all(
        reasoner.get(key) == expected_treatment.get(key)
        and environment_state.get(key) == expected_treatment.get(key)
        for key in identity_keys
    )
    assigned_goal_valid = bool(expected_treatment) and goal == {
        "top": expected_treatment.get("top"),
        "bottom": expected_treatment.get("bottom"),
        "task_id": expected_treatment.get("task_id"),
        "difficulty": expected_treatment.get("difficulty"),
        "seed": expected_treatment.get("seed"),
    }
    assigned_seed_valid = (
        bool(expected_treatment)
        and environment_state.get("seed") == expected_treatment.get("seed")
        and environment_state.get("current_task_seed") == expected_treatment.get("seed")
    )

    observation_counts = (
        perceptor.get("requests"),
        environment_state.get("observation_requests"),
        perceptor.get("observations"),
        reasoner.get("observations"),
    )
    observation_counts_valid = all(type(count) is int and count >= 0 for count in observation_counts)
    observation_chain_valid = (
        observation_counts_valid
        and all(left >= right for left, right in zip(observation_counts, observation_counts[1:]))
        and observation_counts[0] - observation_counts[-1] <= 1
    )

    reasoner_statuses = reasoner.get("action_statuses")
    action_counts = (
        len(decision_trace) if trace_valid else None,
        actuator.get("requests"),
        environment_state.get("status_actions"),
        actuator.get("statuses"),
        reasoner_statuses,
    )
    action_counts_valid = all(type(count) is int and count >= 0 for count in action_counts)
    action_chain_valid = (
        action_counts_valid
        and all(left >= right for left, right in zip(action_counts, action_counts[1:]))
        and action_counts[0] - action_counts[-1] <= 1
    )
    trace_outcomes_valid = (
        trace_valid
        and type(reasoner_statuses) is int
        and 0 <= reasoner_statuses <= len(decision_trace)
        and all(type(item.get(K_LEGAL)) is bool for item in decision_trace[:reasoner_statuses])
        and all(item.get(K_LEGAL) is None for item in decision_trace[reasoner_statuses:])
    )

    checks = [
        (perceptor.get("requests", 0) > 0, "perceptor received no requests"),
        (perceptor.get("observations", 0) > 0, "perceptor forwarded no observations"),
        (actuator.get("requests", 0) > 0, "actuator received no requests"),
        (actuator.get("statuses", 0) > 0, "actuator forwarded no statuses"),
        (reasoner.get("observations", 0) > 0, "reasoner received no observations"),
        (reasoner.get("action_statuses", 0) > 0, "reasoner received no action statuses"),
        (trace_valid, "reasoner decision trace is malformed"),
        (trace_actions_valid, "reasoner trace contains an unknown action"),
        (native_trace_present, "reasoner selected no native action"),
        (trace_outcomes_valid, "reasoner trace outcomes do not match received statuses"),
        (completions_valid, "completion count and episode lengths are invalid"),
        (one_goal_valid, "run did not terminate after exactly one successful goal"),
        (goal_valid, "current goal is missing or invalid"),
        (treatment_valid, "saved treatment does not match the assigned run treatment"),
        (assigned_goal_valid, "completed goal does not match the assigned run goal"),
        (assigned_seed_valid, "environment seed does not match the assigned run seed"),
        (environment_state.get("observation_requests", 0) > 0, "environment was never observed"),
        (environment_state.get("native_actions", 0) > 0, "environment executed no native action"),
        (environment_state.get("close_requests", 0) == 1, "environment was not closed exactly once"),
        (environment_state.get(K_CLOSED) is True, "environment did not record terminal cleanup"),
        (environment_state.get("illegal_actions", 0) == 0, "environment recorded illegal actions"),
        (reasoner.get("illegal_actions", 0) == 0, "reasoner received illegal action statuses"),
        (not reasoner.get("logical_failures", []), "reasoner recorded logical failures"),
        (not environment_state.get("errors", []), "environment recorded errors"),
        (perceptor.get("active_seconds", 0.0) > 0.0, "perceptor active time was not recorded"),
        (actuator.get("active_seconds", 0.0) > 0.0, "actuator active time was not recorded"),
        (reasoner.get("active_seconds", 0.0) > 0.0, "reasoner active time was not recorded"),
        (observation_chain_valid, "observation topology counts diverged"),
        (action_chain_valid, "action topology counts diverged"),
    ]
    for passed, message in checks:
        if not passed:
            print(message)
            valid = False

    if verbose:
        mean_length = (
            sum(episode_lengths) / len(episode_lengths)
            if isinstance(episode_lengths, list) and episode_lengths
            else float("nan")
        )
        print(
            "Reactive loop: "
            f"observations={reasoner.get('observations', 0)}, "
            f"actions={len(decision_trace) if trace_valid else 0}, "
            f"statuses={reasoner.get('action_statuses', 0)}"
        )
        print(
            f"Goals: successes={reasoner.get(K_N_SUCCESSES, 0)}, "
            f"mean episode length={mean_length:.3f}, "
            f"illegal actions={reasoner.get('illegal_actions', 0)}"
        )
        print(
            "Module active seconds: "
            f"perceptor={perceptor.get('active_seconds', 0.0):.6f}, "
            f"actuator={actuator.get('active_seconds', 0.0):.6f}, "
            f"reasoner={reasoner.get('active_seconds', 0.0):.6f}"
        )
    return valid


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
            return parent
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root.")


def _local_mhagenta_root() -> Path:
    root = (_workspace_root().parent / "mhagenta").resolve()
    pyproject = root / "pyproject.toml"
    if not pyproject.exists() or 'version = "1.4.12"' not in pyproject.read_text(encoding="utf-8"):
        raise RuntimeError(f"Expected local MHAgentA 1.4.12 checkout at {root}.")
    return root


def _perceptor_initial_state() -> dict[str, Any]:
    return {
        "requests": 0,
        "observations": 0,
        "active_seconds": 0.0,
    }


def _actuator_initial_state() -> dict[str, Any]:
    return {
        "requests": 0,
        "statuses": 0,
        "active_seconds": 0.0,
    }


def _reasoner_initial_state(treatment: dict[str, Any]) -> dict[str, Any]:
    return {
        **treatment,
        "phase": "active",
        "terminal_reason": None,
        "task_index": 0,
        "observations": 0,
        "action_statuses": 0,
        "current_episode_length": 0,
        K_N_SUCCESSES: 0,
        K_EP_LENGTHS: [],
        "illegal_actions": 0,
        "logical_failures": [],
        "current_goal": {},
        "decision_trace": [],
        "episode_results": [],
        "open_episode": None,
        "active_seconds": 0.0,
    }


def _environment_initial_state(treatment: dict[str, Any], record: bool) -> dict[str, Any]:
    return {
        **{key: value for key, value in treatment.items() if key != "tasks"},
        "seed": treatment["seed"],
        "record": record,
        "current_task_seed": treatment["seed"],
        K_STATE: [],
        "observation_requests": 0,
        "native_actions": 0,
        "status_actions": 0,
        "illegal_actions": 0,
        "close_requests": 0,
        "errors": [],
        K_CLOSED: False,
    }


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run one deterministic experiment and validate its saved evidence."""
    exp_path = Path(exp_path).resolve()
    bw_spec = find_spec("mha_env_blocksworld")
    if bw_spec is None or bw_spec.origin is None:
        raise ImportError("Could not locate mha_env_blocksworld runtime sources.")
    bw_runtime_source = Path(bw_spec.origin).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    seeder = Seeder(run)
    treatment = treatment_for_run(run)
    if seeder.environment != treatment["seed"] or seeder.ll_reasoner != treatment["reasoner_seed"]:
        raise RuntimeError("Frozen treatment seeds do not match the thesis run-index scheme.")
    exchange_name = "mhagenta"
    run_agent_id = agent_name(run, "2_1")
    run_env_id = env_name(run, "2_1")

    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=0.0,
        control_frequency=0.0,
        status_frequency=5.0,
        agent_start_delay=20.0,
        exec_duration=DURATION,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        no_stdout_logs=False,
        stop_on_agents_term=True,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange_name,
    )
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=TestPerceptor(
            module_id=module_name(PERCEPTOR, 0),
            initial_state=_perceptor_initial_state(),
            exchange_name=exchange_name,
        ),
        actuators=TestActuator(
            module_id=module_name(ACTUATOR, 0),
            initial_state=_actuator_initial_state(),
            exchange_name=exchange_name,
        ),
        ll_reasoners=TestLLReasoner(
            module_id=module_name(LLREASONER, 0),
            init_kwargs={"seed": treatment["reasoner_seed"], "tasks": treatment["tasks"]},
            initial_state=_reasoner_initial_state(treatment),
        ),
        requirements_path=Path(__file__).resolve().with_name("requirements.txt"),
        extra_runtime_sources=common_source,
    )
    record = RECORD == "all" or (RECORD == "first" and run == 0)
    orchestrator.add_environment(
        base=TestEnvironment(_environment_initial_state(treatment, record)),
        env_id=run_env_id,
        exec_duration=DURATION + ENVIRONMENT_OVERRUN,
        requirements_path=Path(__file__).resolve().with_name("requirements-env.txt"),
        exchange_name=exchange_name,
        extra_runtime_sources=[common_source, bw_runtime_source],
    )
    runtime_version = (
        DEFAULT_MHAGENTA_VERSION
        if mha_version in {"", "latest", DEFAULT_MHAGENTA_VERSION}
        else mha_version
    )
    orchestrator.run(
        mhagenta_version=runtime_version,
        force_run=True,
        local_build=_local_mhagenta_root(),
    )

    final_states = gather_states(exp_path, False, no_warnings=True)
    agent_states = final_states.get(run_agent_id)
    environment_states = final_states.get(run_env_id)
    environment_state = environment_states.get(run_env_id) if environment_states else None
    logs: list[str] = []
    for log_id in (run_agent_id, run_env_id):
        log_path = exp_path / f"{log_id}.log"
        if log_path.exists():
            logs.extend(log_path.read_text(encoding="utf-8").splitlines(keepends=True))
        else:
            logs.append(f"[ERROR] Missing runtime log: {log_path}\n")
    result = check_results(
        agent_states, environment_state, logs, verbose=VERBOSE,
        expected_treatment=treatment,
    )
    print(f"Results: {result}")
    return result


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 50,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    process_only: bool = False,
) -> None:
    """Run the configured batch or process its existing results."""

    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [{"execution_id": f"run-{run}", "run_id": run,
                 "factors": treatment_for_run(run)}
                for run in run_ids]
    primary_error: BaseException | None = None
    try:
        run_experiment_batch(
            experiment_id="2-1", title="ARCHITECTURE 2-1-BW TEST",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=run_experiment, process_only=process_only,
            stop_on_error=True,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            process_execution_metrics(root, expected_executions=expected)
        except Exception as reporting_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Execution-metrics processing also failed: {type(reporting_error).__name__}"
            )
