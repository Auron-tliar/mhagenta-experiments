"""Reactive MHAgentA behaviors for the bounded 2-2-BW DQN protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import import_module
from logging import WARNING
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy import random

from mhagenta import ActionStatus, Belief, Goal, Observation
from mhagenta.bases import (
    GoalGraphBase,
    KnowledgeBase,
    LearnerBase,
    LLReasonerBase,
    MemoryBase,
)
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import (
    ActuatorState,
    GoalGraphState,
    KnowledgeState,
    LearnerState,
    LLState,
    MemoryState,
    PerceptorState,
)

from .policy import (
    N_ACTIONS,
    NUM_BLOCKS,
    OBS_SHAPE,
    POLICY_FILENAME,
    TABLE_LEN,
    as_numeric_observation,
    build_q_network,
    goal_achieved,
    goal_conditioned_observation,
    model_state_fingerprint,
    sample_goal,
    legal_action_mask,
    save_policy_checkpoint,
)
from .protocol import DQNProtocol, module_protocol
from .replay import HindsightRelabeler, NStepReturns, PrioritizedReplay


MAX_EP_LENGTH = 200
REPLAY_BUFFER_SIZE = 100_000
REPLAY_BATCH_SIZE = 128
WARMUP_TRANSITIONS = REPLAY_BATCH_SIZE * 10
TOTAL_TRAINING_TRANSITIONS = 10_000
TRAINING_UPDATES = TOTAL_TRAINING_TRANSITIONS - WARMUP_TRANSITIONS + 1
EVALUATION_SEEDS = tuple(range(2_200, 2_210))
BEHAVIOR_WINDOW_TRANSITIONS = 200
OPTIMIZATION_WINDOW_UPDATES = 20

EPS_GREEDY_START = 0.9
EPS_GREEDY_END = 0.05
EPS_GREEDY_STEPS = 10_000
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 10.0
TARGET_SYNC_STEPS = 100
EXPECTED_TARGET_SYNCS = TRAINING_UPDATES // TARGET_SYNC_STEPS
ENV_ILLEGAL_REWARD = -0.5

SYNCHRONIZED_TRAINING = True

K_ACTION = "action"
K_STATE = "state"
K_NEXT_STATE = "next_state"
K_OBSERVATION = "observation"
K_REWARD = "reward"
K_STATUS = "status"
K_TERMINAL = "terminal"
K_GOAL = "goal"
K_PHASE = "phase"
K_UPDATE = "update"
K_FINAL = "final"

A_RESET = "reset"
A_CLOSE = "close"
BELIEF_ILLEGAL = "illegal_action"
BELIEF_GOAL_ACHIEVED = "goal_achieved"


def _single_module_id(entries: Sequence[Any], role: str) -> str:
    """Resolve the only module registered for a required role."""

    if len(entries) != 1:
        raise RuntimeError(
            f"Experiment 2-2-BW requires exactly one {role}; found {len(entries)}."
        )
    return cast(str, entries[0].module_id)


def intrinsic_reward(illegal_action: bool, achieved: bool) -> float:
    """Return the unchanged intrinsic reward used by the original treatment."""

    return (1.0 if achieved else -0.01) + (-0.1 if illegal_action else 0.0)


def goal_to_record(goal: Goal) -> dict[str, int | str]:
    """Convert the experiment's typed goal into its JSON-safe state record."""

    if len(goal.state) != 1 or goal.state[0].predicate != "on":
        raise ValueError("A 2-2-BW goal must contain one 'on' belief.")
    arguments = goal.state[0].arguments
    if not isinstance(arguments, (tuple, list)) or len(arguments) != 2:
        raise ValueError("The 'on' belief must contain two block IDs.")
    extras = goal.extras
    record: dict[str, int | str] = {
        "goal_id": str(extras.get("goal_id", "")),
        "phase": str(extras.get("phase", "")),
        "top_block": int(arguments[0]),
        "bottom_block": int(arguments[1]),
        "status": str(extras.get("status", "")),
    }
    _validate_goal_record(record)
    return record


def goal_from_record(record: Mapping[str, object]) -> Goal:
    """Reconstruct a typed goal from the experiment's persistent record."""

    normalized: dict[str, int | str] = {
        "goal_id": str(record.get("goal_id", "")),
        "phase": str(record.get("phase", "")),
        "top_block": int(cast(Any, record.get("top_block", -1))),
        "bottom_block": int(cast(Any, record.get("bottom_block", -1))),
        "status": str(record.get("status", "")),
    }
    _validate_goal_record(normalized)
    return Goal(
        state=[
            Belief(
                "on",
                (normalized["top_block"], normalized["bottom_block"]),
            )
        ],
        extras={
            "goal_id": normalized["goal_id"],
            "phase": normalized["phase"],
            "status": normalized["status"],
        },
    )


def _validate_goal_record(record: Mapping[str, object]) -> None:
    required = {"goal_id", "phase", "top_block", "bottom_block", "status"}
    if set(record) != required:
        raise ValueError(f"Goal record fields must be exactly {sorted(required)}.")
    if not str(record["goal_id"]):
        raise ValueError("Goal ID must be non-empty.")
    if record["phase"] not in {"training", "evaluation"}:
        raise ValueError("Goal phase must be training or evaluation.")
    if record["status"] not in {
        "active",
        "succeeded",
        "truncated",
        "budget_cutoff",
    }:
        raise ValueError("Unexpected goal status.")
    top = int(cast(Any, record["top_block"]))
    bottom = int(cast(Any, record["bottom_block"]))
    if top == bottom or not (0 <= top < NUM_BLOCKS and 0 <= bottom < NUM_BLOCKS):
        raise ValueError("Goal blocks must be distinct valid block IDs.")


def goal_value(record: Mapping[str, object]) -> tuple[int, int]:
    """Return the ordered block pair from a validated goal record."""

    goal = goal_from_record(record)
    return cast(tuple[int, int], tuple(goal.state[0].arguments))


def make_replay_transition(
    evaluated_observation: Observation,
    previous_observation: Any,
    action: Any,
    goal: Any,
    terminal: bool,
) -> Observation | None:
    """Build one goal-conditioned replay transition when context is complete."""

    if previous_observation is None or action is None or goal is None:
        return None
    pair = tuple(int(value) for value in goal)
    if len(pair) != 2:
        raise ValueError(f"Expected a two-block goal, received {pair}.")
    if evaluated_observation.value is None:
        raise ValueError("Evaluated observations must carry an intrinsic value.")
    reward = float(evaluated_observation.value)
    current = as_numeric_observation(evaluated_observation.content)
    return Observation(
        content={
            K_STATE: goal_conditioned_observation(previous_observation, pair),
            K_ACTION: int(action),
            K_REWARD: reward,
            K_NEXT_STATE: goal_conditioned_observation(current, pair),
            K_TERMINAL: bool(terminal),
        },
        observation_type="dqn_transition",
        value=reward,
    )


class TestEnvironment(MHAEnvBase):
    """Blocks World adapter with compact reset and seed evidence."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed: int = int(init_state.pop("seed"))
        self._record: bool = bool(init_state.pop("record", False))
        super().__init__(init_state)
        self._env: Any = None
        self._current_state = np.zeros(OBS_SHAPE, dtype=np.uint8)
        self._last_info: dict[str, Any] = {}
        self._build_env()

    def on_observe(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state["observation_requests"] += 1
        return state, {K_OBSERVATION: self._current_state.copy()}

    def on_action(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        if K_ACTION not in kwargs:
            return state, {}
        action = kwargs[K_ACTION]
        if action == A_RESET:
            phase = str(kwargs.get(K_PHASE, ""))
            requested_seed = kwargs.get("requested_seed")
            applied_seed = None if requested_seed is None else int(requested_seed)
            if applied_seed is None:
                self._current_state, self._last_info = self._env.reset()
            else:
                self._current_state, self._last_info = self._env.reset(applied_seed)
            state["total_resets"] += 1
            if phase == "training":
                state["training_resets"] += 1
            elif phase == "evaluation":
                state["evaluation_resets"] += 1
                state["applied_evaluation_seeds"].append(applied_seed)
            else:
                state["unclassified_resets"] += 1
            if requested_seed != applied_seed:
                state["reset_seed_mismatches"] += 1
            return state, {
                K_REWARD: 0.0,
                K_PHASE: phase,
                "requested_seed": requested_seed,
                "applied_seed": applied_seed,
            }
        if action == A_CLOSE:
            if self._record:
                self._env.save_and_increment()
            return state, None

        self._current_state, reward, _, _, self._last_info = self._env.step(action)
        state["native_actions"] += 1
        return state, {K_REWARD: float(reward), K_PHASE: kwargs.get(K_PHASE)}

    def _build_env(self) -> None:
        from mha_env_blocksworld import BWRecorder, BlocksWorldEnv

        environment: Any = BlocksWorldEnv(
            table_len=TABLE_LEN,
            num_blocks=NUM_BLOCKS,
            render_mode="rgb_array" if self._record else None,
            symbolic=False,
        )
        if self._record:
            environment = BWRecorder(
                env=cast(BlocksWorldEnv, environment),
                path="/out",
                single_trace=True,
            )
        self._env = environment
        self._current_state, self._last_info = self._env.reset(self._seed)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()


class TestGoalGraph(GoalGraphBase):
    """Trivial graph that exclusively owns valid random goal generation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.default_rng()

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.default_rng(kwargs.get("seed"))

    def on_goal_request(
        self,
        state: GoalGraphState,
        sender: str,
        **kwargs: Any,
    ) -> GoalGraphState:
        if state["active_goal"] is not None:
            self.log(WARNING, "Ignoring a goal request while another goal is active.")
            return state
        phase = str(kwargs.get(K_PHASE, ""))
        observation = as_numeric_observation(kwargs.get(K_OBSERVATION))
        seed = kwargs.get("evaluation_seed")
        generator = self._rng if phase == "training" else random.default_rng(seed)
        pair = sample_goal(generator, observation)
        state["goal_sequence"] += 1
        goal = Goal(
            state=[Belief("on", pair)],
            extras={
                "goal_id": f"{phase}-{state['goal_sequence']:06d}",
                "phase": phase,
                "status": "active",
            },
        )
        record = goal_to_record(goal)
        state["active_goal"] = record
        state[f"{phase}_requests"] += 1
        state[f"{phase}_issued"] += 1
        state.outbox.send_goals(sender, [goal])
        return state

    def on_goal_update(
        self,
        state: GoalGraphState,
        sender: str,
        goals: Sequence[Goal],
        **kwargs: Any,
    ) -> GoalGraphState:
        if len(goals) != 1 or state["active_goal"] is None:
            self.log(WARNING, "Ignoring an unmatched goal update.")
            return state
        update = goal_to_record(goals[0])
        active = cast(dict[str, Any], state["active_goal"])
        if update["goal_id"] != active["goal_id"]:
            self.log(WARNING, "Ignoring a goal update with the wrong goal ID.")
            return state
        phase = str(update["phase"])
        status = str(update["status"])
        if status not in {"succeeded", "truncated", "budget_cutoff"}:
            self.log(WARNING, "Ignoring a non-terminal goal update.")
            return state
        state[f"{phase}_{status}"] += 1
        state["active_goal"] = None
        state.outbox.send_goals(sender, goals, acknowledged=True)
        return state


class TestLLReasoner(LLReasonerBase):
    """Reactive training and fixed-seed evaluation controller."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.default_rng()
        self._torch: Any = None
        self._model: Any = None
        self._actuator_id = ""
        self._perceptor_id = ""
        self._knowledge_id = ""
        self._learner_id = ""
        self._goal_graph_id = ""
        self._previous_observation: np.ndarray | None = None
        self._previous_action: int | None = None
        self._previous_goal: tuple[int, int] | None = None
        self._previous_actor_update = 0
        self._pending_observation: np.ndarray | None = None
        self._last_action_illegal = False
        self._awaiting_reset = False
        self._requested_reset_seed: int | None = None
        self._reset_phase = ""
        self._waiting_for_training = False
        self._pending_after_goal = ""
        self._goal_acknowledged = True
        self._final_model_installed = False
        self._synchronized_training = SYNCHRONIZED_TRAINING
        self._evaluation_started_at = 0.0
        self._evaluation_applied_seed: int | None = None
        self._protocol = DQNProtocol()
        self._training_started_at = 0.0
        self._decision_time = 0.0

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.default_rng(kwargs.get("seed"))
        self._torch = import_module("torch")
        # Small CPU inference/tensor operations must not occupy every host core.
        self._torch.set_num_threads(1)
        self._protocol = module_protocol(kwargs)
        self._synchronized_training = self._protocol.synchronized_training

    def on_first(self, state: LLState) -> LLState:
        self._actuator_id = _single_module_id(state.directory.internal.actuation, "actuator")
        self._perceptor_id = _single_module_id(state.directory.internal.perception, "perceptor")
        self._knowledge_id = _single_module_id(state.directory.internal.knowledge, "knowledge module")
        self._learner_id = _single_module_id(state.directory.internal.learning, "learner")
        self._goal_graph_id = _single_module_id(state.directory.internal.goals, "goal graph")
        state["phase_timestamps"]["training_started"] = float(state.time)
        self._training_started_at = float(state.time)
        state["training_deadline"] = self._training_started_at + self._protocol.training_seconds
        state.outbox.send_learner_task(self._learner_id, {
            "kind": "start", "started_at": self._training_started_at,
            "deadline": state["training_deadline"],
        })
        state.outbox.request_observation(self._perceptor_id)
        return state

    def on_last(self, state: LLState) -> LLState:
        state["training_resets_final"] = state["training_resets"]
        return state

    def _epsilon(self, decisions: int) -> float:
        progress = (
            decisions / self._protocol.total_training_transitions
            if self._synchronized_training else
            (self._decision_time - self._training_started_at) / self._protocol.async_training_seconds
        )
        progress = min(1.0, max(0.0, progress))
        return EPS_GREEDY_START + progress * (EPS_GREEDY_END - EPS_GREEDY_START)

    def _deadline_reached(self, state: LLState) -> bool:
        return float(state.time) >= state["training_deadline"]

    def _close_collection(self, state: LLState) -> None:
        """Close the ordered observation stream without fabricating a transition."""
        if state["collection_closed"]:
            return
        state["collection_closed"] = True
        state["phase"] = "training_draining"
        state["phase_timestamps"]["training_finished"] = float(state.time)
        state["phase_timestamps"]["drain_started"] = float(state.time)
        state["training_resets_at_cutoff"] = state["training_resets"]
        self._flush_behavior_window(state)
        state.outbox.send_beliefs(
            self._knowledge_id, Observation(None), [], phase="training",
            collection_closed=True, transition_index=state["training_transitions"],
        )
        if not self._synchronized_training:
            state.outbox.send_learner_task(self._learner_id, {"kind": "stop"})

    def _stop_before_action(self, state: LLState) -> bool:
        """Honor a deadline reached between an observation and its continuation."""
        if state["phase"] != "training" or not self._deadline_reached(state):
            return False
        if self._synchronized_training:
            state["failure_reason"] = "synchronous_training_timeout"
            state.outbox.terminate_agent(state["failure_reason"])
            return True
        if state["active_goal"] is not None:
            state["training_budget_cutoffs"] += 1
            state["cutoff_episode_length"] = state["current_episode_length"]
            window = state["behavior_current"]
            if window is None and state["behavior_windows"]:
                window = state["behavior_windows"][-1]
            if window is not None:
                window["budget_cutoffs"] += 1
                window["cutoff_episode_length"] = state["current_episode_length"]
            self._close_training_goal(state, "budget_cutoff", final=True, waits_for_update=False)
        self._close_collection(state)
        return True

    def _select_action(
        self,
        observation: np.ndarray,
        goal: tuple[int, int],
        decisions: int,
        *,
        greedy: bool,
    ) -> int:
        if not greedy and (
            self._model is None or self._rng.random() < self._epsilon(decisions)
        ):
            if self._protocol.action_masking:
                return int(self._rng.choice(np.flatnonzero(legal_action_mask(observation))))
            return int(self._rng.integers(N_ACTIONS))
        if self._model is None:
            raise RuntimeError("Evaluation requires the final installed model.")
        tensor = self._torch.as_tensor(
            goal_conditioned_observation(observation, goal),
            dtype=self._torch.float32,
        ).unsqueeze(0)
        with self._torch.no_grad():
            values = self._model(tensor)
            if self._protocol.action_masking:
                valid = self._torch.as_tensor(legal_action_mask(observation), device=values.device)
                values = values.masked_fill(~valid, -self._torch.inf)
            return int(values.argmax(dim=1).item())

    def _request_goal(self, state: LLState, observation: np.ndarray) -> None:
        if state["active_goal"] is not None:
            raise RuntimeError("Cannot request a goal while one is active.")
        self._pending_observation = observation.copy()
        self._goal_acknowledged = False
        kwargs: dict[str, Any] = {
            K_PHASE: state["phase"],
            K_OBSERVATION: observation.copy(),
        }
        if state["phase"] == "evaluation":
            kwargs["evaluation_seed"] = self._protocol.evaluation_seeds[state["evaluation_index"]]
        state.outbox.request_goals(self._goal_graph_id, **kwargs)

    def _request_policy_action(
        self,
        state: LLState,
        observation: np.ndarray,
        *,
        greedy: bool = False,
    ) -> None:
        if self._stop_before_action(state):
            return
        record = state["active_goal"]
        if record is None:
            raise RuntimeError("Cannot select an action without an active goal.")
        pair = goal_value(record)
        self._decision_time = float(state.time)
        action = self._select_action(
            observation,
            pair,
            state["actions"],
            greedy=greedy,
        )
        if self._stop_before_action(state):
            return
        state["actions"] += 1
        if state["phase"] == "training":
            state["last_training_action_started"] = float(state.time)
        state["action_histogram"][action] += 1
        self._previous_observation = observation.copy()
        self._previous_action = action
        self._previous_goal = pair
        self._previous_actor_update = state["actor_update"]
        state.outbox.request_action(
            self._actuator_id,
            action=action,
            phase=state["phase"],
        )

    def _request_reset(
        self,
        state: LLState,
        phase: str,
        seed: int | None = None,
    ) -> None:
        if phase == "training" and self._stop_before_action(state):
            return
        self._awaiting_reset = True
        if phase == "training":
            state["last_training_reset_started"] = float(state.time)
        self._requested_reset_seed = seed
        self._reset_phase = phase
        state.outbox.request_action(
            self._actuator_id,
            action=A_RESET,
            phase=phase,
            requested_seed=seed,
        )

    def _send_training_transition(
        self,
        state: LLState,
        current: np.ndarray,
        achieved: bool,
        terminal: bool,
    ) -> None:
        state.outbox.send_beliefs(
            self._knowledge_id,
            Observation(current.copy()),
            [
                Belief(BELIEF_ILLEGAL, self._last_action_illegal),
                Belief(BELIEF_GOAL_ACHIEVED, achieved),
            ],
            previous_observation=self._previous_observation.copy(),
            action=self._previous_action,
            goal=self._previous_goal,
            terminal=terminal,
            phase="training",
            transition_index=state["training_transitions"],
            actor_update=self._previous_actor_update,
        )

    def _record_behavior_transition(
        self,
        state: LLState,
        *,
        outcome: str | None,
        episode_length: int | None,
    ) -> None:
        current = state["behavior_current"]
        if current is None:
            current = {
                "transition_start": state["training_transitions"],
                "native_actions": 0,
                "illegal_actions": 0,
                "successes": 0,
                "truncations": 0,
                "budget_cutoffs": 0,
                "ordinary_episode_lengths": [],
                "cutoff_episode_length": None,
                "epsilon_start": self._epsilon(state["actions"] - 1),
            }
            state["behavior_current"] = current
        current["native_actions"] += 1
        current["illegal_actions"] += int(self._last_action_illegal)
        if outcome == "succeeded":
            current["successes"] += 1
            current["ordinary_episode_lengths"].append(episode_length)
        elif outcome == "truncated":
            current["truncations"] += 1
            current["ordinary_episode_lengths"].append(episode_length)
        elif outcome == "budget_cutoff":
            current["budget_cutoffs"] += 1
            current["cutoff_episode_length"] = episode_length
        if current["native_actions"] >= self._protocol.behavior_window_transitions:
            self._flush_behavior_window(state)

    def _flush_behavior_window(self, state: LLState) -> None:
        current = state["behavior_current"]
        if current is None:
            return
        completed = current["successes"] + current["truncations"]
        lengths = current.pop("ordinary_episode_lengths")
        row = {
            "window_index": len(state["behavior_windows"]),
            **current,
            "transition_end": state["training_transitions"],
            "elapsed_seconds": float(state.time),
            "completed_episodes": completed,
            "success_rate": (
                None if completed == 0 else current["successes"] / completed
            ),
            "mean_episode_length": (
                None if not lengths else float(np.mean(lengths))
            ),
            "epsilon_end": self._epsilon(state["actions"]),
            "actor_update": state["actor_update"],
        }
        state["behavior_windows"].append(row)
        state["behavior_current"] = None

    def _terminal_goal(self, state: LLState, status: str) -> Goal:
        record = dict(state["active_goal"])
        record["status"] = status
        return goal_from_record(record)

    def _close_training_goal(
        self,
        state: LLState,
        status: str,
        *,
        final: bool,
        waits_for_update: bool,
    ) -> None:
        state.outbox.send_goal_update(
            self._goal_graph_id,
            [self._terminal_goal(state, status)],
        )
        state["active_goal"] = None
        self._goal_acknowledged = False
        self._pending_after_goal = "final" if final else "training_reset"
        self._waiting_for_training = waits_for_update
        self._previous_observation = None
        self._previous_action = None
        self._previous_goal = None
        self._last_action_illegal = False

    def _continue_after_goal(self, state: LLState) -> None:
        if not self._goal_acknowledged or self._waiting_for_training:
            return
        continuation = self._pending_after_goal
        self._pending_after_goal = ""
        if continuation == "training_reset":
            self._request_reset(state, "training")
        elif continuation == "final":
            self._maybe_start_evaluation(state)
        elif continuation == "evaluation_reset":
            next_index = state["evaluation_index"] + 1
            state["evaluation_index"] = next_index
            if next_index == len(self._protocol.evaluation_seeds):
                state["phase"] = "complete"
                state["phase_timestamps"]["complete"] = float(state.time)
                state.outbox.terminate_agent()
            else:
                self._request_reset(state, "evaluation", self._protocol.evaluation_seeds[next_index])

    def _handle_training_observation(
        self,
        state: LLState,
        current: np.ndarray,
    ) -> LLState:
        if self._previous_observation is None:
            if self._stop_before_action(state):
                return state
            self._request_goal(state, current)
            return state

        next_transition = state["training_transitions"] + 1
        pair = cast(tuple[int, int], self._previous_goal)
        achieved = goal_achieved(current, pair)
        natural_truncation = not achieved and state["current_episode_length"] >= self._protocol.max_episode_length
        final = (next_transition == self._protocol.total_training_transitions
                 if self._synchronized_training else self._deadline_reached(state))
        terminal = achieved or natural_truncation
        outcome: str | None = None
        if achieved:
            outcome = "succeeded"
        elif natural_truncation:
            outcome = "truncated"
        elif final:
            outcome = "budget_cutoff"

        state["training_transitions"] = next_transition
        self._send_training_transition(state, current, achieved, terminal)
        self._record_behavior_transition(
            state,
            outcome=outcome,
            episode_length=state["current_episode_length"] if terminal or final else None,
        )

        eligible = next_transition >= self._protocol.warmup_transitions
        if terminal or final:
            if outcome == "succeeded":
                state["training_successes"] += 1
            elif outcome == "truncated":
                state["training_truncations"] += 1
            else:
                state["training_budget_cutoffs"] += 1
                state["cutoff_episode_length"] = state["current_episode_length"]
            state["current_episode_length"] = 0
            if final:
                self._close_collection(state)
            self._close_training_goal(
                state,
                cast(str, outcome),
                final=final,
                waits_for_update=self._synchronized_training and eligible,
            )
            return state

        if self._synchronized_training and eligible:
            self._waiting_for_training = True
            self._pending_observation = current.copy()
        else:
            self._request_policy_action(state, current)
        return state

    def _handle_evaluation_observation(
        self,
        state: LLState,
        current: np.ndarray,
    ) -> LLState:
        if self._previous_observation is None:
            self._request_goal(state, current)
            return state
        pair = cast(tuple[int, int], self._previous_goal)
        achieved = goal_achieved(current, pair)
        truncated = not achieved and state["current_episode_length"] >= self._protocol.max_episode_length
        if not achieved and not truncated:
            self._request_policy_action(state, current, greedy=True)
            return state

        requested_seed = self._protocol.evaluation_seeds[state["evaluation_index"]]
        state["evaluation_cases"].append(
            {
                "requested_seed": requested_seed,
                "applied_seed": self._evaluation_applied_seed,
                "goal": list(pair),
                "success": achieved,
                "steps": state["current_episode_length"],
                "illegal_actions": state["evaluation_illegal_actions"],
                "action_mode": "greedy",
                "exploratory_actions": 0,
                "elapsed_seconds": float(state.time - self._evaluation_started_at),
            }
        )
        status = "succeeded" if achieved else "truncated"
        state.outbox.send_goal_update(
            self._goal_graph_id,
            [self._terminal_goal(state, status)],
        )
        state["active_goal"] = None
        state["current_episode_length"] = 0
        state["evaluation_illegal_actions"] = 0
        self._previous_observation = None
        self._previous_action = None
        self._previous_goal = None
        self._goal_acknowledged = False
        self._pending_after_goal = "evaluation_reset"
        return state

    def on_observation(
        self,
        state: LLState,
        sender: str,
        observation: Observation,
        **kwargs: Any,
    ) -> LLState:
        try:
            current = as_numeric_observation(observation.content)
        except (TypeError, ValueError) as exc:
            self.log(WARNING, f"Received invalid numeric observation from {sender}: {exc}")
            state["invalid_observations"] += 1
            return state
        state["observations"] += 1
        if state["phase"] == "evaluation" and float(state.time) >= state["evaluation_deadline"]:
            state["failure_reason"] = "evaluation_timeout"
            state.outbox.terminate_agent(state["failure_reason"])
            return state
        if state["phase"] == "training":
            return self._handle_training_observation(state, current)
        if state["phase"] == "evaluation":
            return self._handle_evaluation_observation(state, current)
        self.log(WARNING, f"Ignoring observation in phase {state['phase']!r}.")
        return state

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        state["statuses"] += 1
        status = action_status.status if isinstance(action_status.status, dict) else {}
        if self._awaiting_reset:
            self._awaiting_reset = False
            applied = status.get("applied_seed")
            if status.get("requested_seed") != self._requested_reset_seed:
                state["reset_seed_mismatches"] += 1
            if self._reset_phase == "training":
                state["training_resets"] += 1
            else:
                self._evaluation_applied_seed = applied
                self._evaluation_started_at = float(state.time)
                if applied != self._requested_reset_seed:
                    state["reset_seed_mismatches"] += 1
            self._last_action_illegal = False
            state.outbox.request_observation(self._perceptor_id)
            return state

        reward = status.get(K_REWARD)
        self._last_action_illegal = bool(
            reward is not None and np.isclose(float(reward), ENV_ILLEGAL_REWARD)
        )
        state["current_episode_length"] += 1
        if self._protocol.action_masking and self._last_action_illegal:
            state['failure_reason'] = 'masked_action_was_illegal'
            state.outbox.terminate_agent(state['failure_reason'])
            return state
        if state["phase"] == "evaluation":
            state["evaluation_illegal_actions"] += int(self._last_action_illegal)
        state.outbox.request_observation(self._perceptor_id)
        return state

    def on_goal_update(
        self,
        state: LLState,
        sender: str,
        goals: Sequence[Goal],
        **kwargs: Any,
    ) -> LLState:
        if sender != self._goal_graph_id or len(goals) != 1:
            self.log(WARNING, "Ignoring an invalid goal-graph message.")
            return state
        if kwargs.get("acknowledged"):
            self._goal_acknowledged = True
            self._continue_after_goal(state)
            return state
        if state["active_goal"] is not None or self._pending_observation is None:
            self.log(WARNING, "Ignoring an unexpected new goal.")
            return state
        record = goal_to_record(goals[0])
        if record["status"] != "active" or record["phase"] != state["phase"]:
            self.log(WARNING, "Ignoring a goal for the wrong phase or status.")
            return state
        state["active_goal"] = record
        observation = self._pending_observation
        self._pending_observation = None
        self._goal_acknowledged = True
        self._request_policy_action(
            state,
            observation,
            greedy=state["phase"] == "evaluation",
        )
        return state

    def _maybe_start_evaluation(self, state: LLState) -> None:
        if (
            state["phase"] != "training_draining"
            or not state["collection_closed"]
            or not self._final_model_installed
            or not self._goal_acknowledged
            or self._waiting_for_training
        ):
            return
        state["phase"] = "evaluation"
        state["phase_timestamps"]["evaluation_started"] = float(state.time)
        state["evaluation_deadline"] = float(state.time) + self._protocol.evaluation_seconds
        state["evaluation_index"] = 0
        state.outbox.send_learner_task(
            self._learner_id,
            {"kind": "evaluation_started"},
        )
        self._request_reset(state, "evaluation", self._protocol.evaluation_seeds[0])

    def on_model(
        self,
        state: LLState,
        sender: str,
        model: Any,
        **kwargs: Any,
    ) -> LLState:
        update = int(kwargs.get(K_UPDATE, 0))
        final = bool(kwargs.get(K_FINAL, False))
        if model is not None:
            self._model = model.cpu()
            self._model.eval()
            state["models_installed"] += 1
            if state["first_model_install_update"] is None:
                state["first_model_install_update"] = update
            state["final_model_install_update"] = update
            state["actor_update"] = update
        if final:
            if self._model is None or state["actor_update"] != update:
                raise ValueError("Final marker does not identify the installed model")
            state["installed_final_model_fingerprint"] = model_state_fingerprint(self._model)
            state["installed_final_model_update"] = update
            self._final_model_installed = True
        if self._synchronized_training and kwargs.get("completion", not final):
            state["training_completions_received"] += 1
            if self._waiting_for_training:
                self._waiting_for_training = False
                if self._pending_after_goal:
                    self._continue_after_goal(state)
                elif self._pending_observation is not None:
                    observation = self._pending_observation
                    self._pending_observation = None
                    self._request_policy_action(state, observation)
        if final:
            self._maybe_start_evaluation(state)
        return state


class PacedReasoner(TestLLReasoner):
    """Limit collection lead using periodic model updates, without per-action waits."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._paced_observation: np.ndarray | None = None

    def _request_policy_action(self, state: LLState, observation: np.ndarray, *, greedy: bool = False) -> None:
        if self._stop_before_action(state):
            self._paced_observation = None
            return
        eligible = max(0, state['training_transitions'] + 1 - self._protocol.warmup_transitions)
        if (state['phase'] == 'training'
                and eligible * self._protocol.min_updates_per_transition > state['actor_update']):
            self._paced_observation = observation.copy()
            state['collection_pauses'] += 1
            return
        super()._request_policy_action(state, observation, greedy=greedy)

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs: Any) -> LLState:
        state = super().on_model(state, sender, model, **kwargs)
        if self._paced_observation is not None and state['phase'] == 'training':
            observation = self._paced_observation
            self._paced_observation = None
            self._request_policy_action(state, observation)
        return state


class TestActuator(RMQActuatorBase):
    """Forward reasoner actions and reset metadata to Blocks World."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("No Blocks World environment found.")
        self._env_id = environment.address["env_id"]
        self._reasoner_id = _single_module_id(state.directory.internal.ll_reasoning, "LLReasoner")
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        self.act(self._env_id, action=A_CLOSE)
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        if K_ACTION in kwargs:
            state["requests"] += 1
            self.act(self._env_id, **kwargs)
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        state["statuses"] += 1
        state.outbox.send_status(self._reasoner_id, ActionStatus(dict(kwargs)))
        return state


class TestPerceptor(RMQPerceptorBase):
    """Forward numeric observations from Blocks World to the reasoner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: PerceptorState) -> PerceptorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("No Blocks World environment found.")
        self._env_id = environment.address["env_id"]
        self._reasoner_id = _single_module_id(state.directory.internal.ll_reasoning, "LLReasoner")
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        state["requests"] += 1
        self.observe(self._env_id)
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        state["observations"] += 1
        state.outbox.send_observation(
            self._reasoner_id,
            Observation(kwargs.get(K_OBSERVATION)),
        )
        return state


class TestKnowledge(KnowledgeBase):
    """Assign unchanged intrinsic rewards and forward training transitions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._memory_id = ""

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        self._memory_id = _single_module_id(state.directory.internal.memory, "memory module")
        return state

    def on_observed_beliefs(
        self,
        state: KnowledgeState,
        sender: str,
        observation: Observation,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> KnowledgeState:
        if kwargs.get("collection_closed"):
            state.outbox.send_observations(self._memory_id, [], **kwargs)
            return state
        values = {belief.predicate: bool(belief.arguments) for belief in beliefs}
        if set(values) != {BELIEF_ILLEGAL, BELIEF_GOAL_ACHIEVED}:
            self.log(WARNING, f"Received unexpected belief set from {sender}: {values}")
            return state
        reward = intrinsic_reward(values[BELIEF_ILLEGAL], values[BELIEF_GOAL_ACHIEVED])
        evaluated = Observation(
            content=as_numeric_observation(observation.content),
            observation_type=observation.observation_type,
            value=reward,
        )
        state["evaluated_observations"] += 1
        state["intrinsic_reward"] += reward
        state.outbox.send_observations(
            self._memory_id,
            [evaluated],
            illegal_action=values[BELIEF_ILLEGAL],
            **kwargs,
        )
        return state


class TestMemory(MemoryBase):
    """Own n-step assembly, prioritized FIFO replay, and ordered stream closure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._protocol = DQNProtocol()
        self._rng = random.default_rng()
        self._buffer = PrioritizedReplay(self._protocol, self._rng)
        self._returns = NStepReturns(self._protocol.n_steps, self._protocol.discount)
        self._hindsight = HindsightRelabeler(
            self._protocol, random.default_rng(), intrinsic_reward
        )
        self._episode: list[dict[str, Any]] = []
        self._learner_id = ""
        self._request: dict[str, Any] | None = None

    def on_init(self, **kwargs: Any) -> None:
        """Initialize empty transient replay for one treatment."""
        self._protocol = module_protocol(kwargs)
        replay_seed, hindsight_seed = random.SeedSequence(kwargs.get("seed")).spawn(2)
        self._rng = random.default_rng(replay_seed)
        self._buffer = PrioritizedReplay(self._protocol, self._rng)
        self._returns = NStepReturns(self._protocol.n_steps, self._protocol.discount)
        self._hindsight = HindsightRelabeler(
            self._protocol, random.default_rng(hindsight_seed), intrinsic_reward
        )
        self._episode = []
        self._request = None

    def on_first(self, state: MemoryState) -> MemoryState:
        """Resolve the sole learner."""
        self._learner_id = _single_module_id(state.directory.internal.learning, "learner")
        return state

    def _sync_state(self, state: MemoryState) -> None:
        state["buffer_size"] = len(self._buffer)
        state["replay_entries"] = self._buffer.next_id
        state["evictions"] = self._buffer.evictions
        state["stale_priority_feedback"] = self._buffer.stale_feedback
        state["pending_tail"] = len(self._returns.pending)
        state["pending_her_episode"] = len(self._episode)
        state["pending_request"] = self._request is not None

    def _append_hindsight(self, state: MemoryState) -> None:
        """Relabel and admit the completed or collection-truncated episode."""

        if not self._episode:
            return
        entries, sources, candidates = self._hindsight.relabel(self._episode)
        for entry in entries:
            self._buffer.append(entry)
        state["her_episodes"] += 1
        state["her_source_transitions"] += sources
        state["her_candidate_goals"] += candidates
        state["her_entries"] += len(entries)
        state["her_success_entries"] += sum(
            bool(entry["her_goal_achieved"]) for entry in entries
        )
        self._episode = []

    def _send_batch_if_ready(self, state: MemoryState) -> None:
        if state["sampling_closed"]:
            if state["collection_closed"] and not state["closure_acknowledged"]:
                state["closure_acknowledged"] = True
                state.outbox.send_memories(self._learner_id, [], replay_closed=True)
            self._sync_state(state)
            return
        if (self._request is None or state["training_transitions"] < self._protocol.warmup_transitions
                or len(self._buffer) < self._protocol.batch_size
                or (self._protocol.synchronized_training and state["update_credits"] == 0)):
            self._sync_state(state)
            return
        batch, metadata = self._buffer.sample(float(self._request["beta"]))
        self._request = None
        if self._protocol.synchronized_training:
            state["update_credits"] -= 1
            state["credits_drained"] += 1
        state["batches_sent"] += 1
        self._sync_state(state)
        state.outbox.send_memories(self._learner_id, [Observation(item) for item in batch], **metadata)

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs: Any) -> MemoryState:
        """Apply delayed feedback, then serve or cancel the single pending request."""
        if sender != self._learner_id:
            raise ValueError("Unexpected replay requester")
        self._buffer.feedback(kwargs.get("entry_ids", []), kwargs.get("errors", []))
        if kwargs.get("stop"):
            self._request = None
            state["sampling_closed"] = True
        elif not state["sampling_closed"]:
            if self._request is not None:
                raise ValueError("Duplicate outstanding replay request")
            self._request = {"beta": kwargs.get("beta", self._protocol.beta_start)}
        self._send_batch_if_ready(state)
        return state

    def on_observation_update(
        self, state: MemoryState, sender: str, observations: Sequence[Observation], **kwargs: Any,
    ) -> MemoryState:
        """Admit raw transitions or the ordered closure marker from Knowledge."""
        if kwargs.get(K_PHASE) == "evaluation":
            state["evaluation_replay_insertion_attempts"] += 1
            return state
        if kwargs.get(K_PHASE) != "training":
            raise ValueError("Invalid replay phase")
        if kwargs.get("collection_closed"):
            if observations or state["collection_closed"] or kwargs["transition_index"] != state["training_transitions"]:
                raise ValueError("Invalid or out-of-order replay closure")
            for entry in self._returns.flush():
                self._buffer.append(entry)
            self._append_hindsight(state)
            state["collection_closed"] = True
            state["collection_closed_at"] = float(state.time)
            self._send_batch_if_ready(state)
            return state
        if state["collection_closed"] or (self._protocol.synchronized_training
                and state["training_transitions"] >= self._protocol.total_training_transitions):
            state["rejected_post_budget_transitions"] += 1
            raise ValueError("Transition admitted after collection closure")
        if len(observations) != 1 or kwargs.get("transition_index") != state["training_transitions"] + 1:
            raise ValueError("Raw transition sequence is not contiguous")
        transition = make_replay_transition(
            observations[0], kwargs.get("previous_observation"), kwargs.get(K_ACTION),
            kwargs.get(K_GOAL), bool(kwargs.get(K_TERMINAL, False)),
        )
        if transition is None:
            raise ValueError("Incomplete raw transition")
        content = cast(dict[str, Any], transition.content)
        content["actor_update"] = int(kwargs.get("actor_update", 0))
        illegal_action = kwargs.get("illegal_action")
        if type(illegal_action) is not bool:
            raise ValueError("Transition is missing its illegal-action flag")
        self._episode.append({
            "state_observation": as_numeric_observation(kwargs.get("previous_observation")),
            "next_observation": as_numeric_observation(observations[0].content),
            "action": content[K_ACTION],
            "illegal_action": illegal_action,
            "terminal": content[K_TERMINAL],
            "actor_update": content["actor_update"],
            "goal": tuple(kwargs[K_GOAL]),
            "transition_index": int(kwargs["transition_index"]),
        })
        for entry in self._returns.append(content, tuple(kwargs[K_GOAL])):
            self._buffer.append(entry)
        if content[K_TERMINAL]:
            self._append_hindsight(state)
        state["training_transitions"] += 1
        state["final_training_transition_terminal"] = content[K_TERMINAL]
        if self._protocol.synchronized_training and state["training_transitions"] >= self._protocol.warmup_transitions:
            state["credits_earned"] += 1
            state["update_credits"] += 1
        self._send_batch_if_ready(state)
        return state

    def on_last(self, state: MemoryState) -> MemoryState:
        """Persist bounded replay evidence without dumping its tensors."""
        self._sync_state(state)
        return state


class TestLearner(LearnerBase):
    """Double DQN with weighted n-step Huber updates and explicit replay closure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._torch: Any = None
        self._online: Any = None
        self._target: Any = None
        self._optimizer: Any = None
        self._device: Any = None
        self._memory_id = ""
        self._reasoner_id = ""
        self._protocol = DQNProtocol()
        self._synchronized_training = SYNCHRONIZED_TRAINING
        self._policy_path = f"/out/{POLICY_FILENAME}"
        self._feedback: dict[str, Any] = {}
        self._request_outstanding = False

    def on_init(self, **kwargs: Any) -> None:
        """Build independent online and target networks on the requested device."""
        self._torch = import_module("torch")
        # Small CPU inference/tensor operations must not occupy every host core.
        self._torch.set_num_threads(1)
        self._protocol = module_protocol(kwargs)
        self._synchronized_training = self._protocol.synchronized_training
        self._policy_path = str(kwargs.get("policy_path", f"/out/{POLICY_FILENAME}"))
        seed = int(kwargs.get("seed", 0))
        self._torch.manual_seed(seed)
        requested_device = str(kwargs.get("device", "cpu"))
        if requested_device.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for 2-2-BW but is unavailable")
        self._device = self._torch.device(requested_device)
        if self._device.type == "cuda":
            self._torch.cuda.manual_seed_all(seed)
        self._online = build_q_network(self._torch, self._protocol.network_architecture).to(self._device)
        self._target = deepcopy(self._online)
        self._target.eval()
        self._optimizer = self._torch.optim.AdamW(self._online.parameters(), lr=LEARNING_RATE)

    def on_first(self, state: LearnerState) -> LearnerState:
        """Resolve memory and reasoner recipients."""
        state["device"] = str(self._device)
        state["torch_version"] = str(self._torch.__version__)
        state["cuda_runtime"] = self._torch.version.cuda
        state["cuda_available"] = bool(self._torch.cuda.is_available())
        state["cuda_device_name"] = (
            self._torch.cuda.get_device_name(self._device)
            if self._device.type == "cuda"
            else None
        )
        self._memory_id = _single_module_id(state.directory.internal.memory, "memory module")
        self._reasoner_id = _single_module_id(state.directory.internal.ll_reasoning, "LLReasoner")
        return state

    def _beta(self, state: LearnerState) -> float:
        progress = (state["training_updates"] / max(1, self._protocol.training_updates - 1)
                    if self._synchronized_training else
                    (float(state.time) - state["training_started_at"]) / self._protocol.async_training_seconds)
        return self._protocol.beta_start + (1 - self._protocol.beta_start) * min(1.0, max(0.0, progress))

    def _request_batch(self, state: LearnerState) -> None:
        if self._request_outstanding:
            raise ValueError("Learner already has an outstanding batch")
        state.outbox.request_memories(self._memory_id, beta=self._beta(state), **self._feedback)
        self._feedback = {}
        self._request_outstanding = True

    def _stop_sampling(self, state: LearnerState) -> None:
        if state["stopping"]:
            return
        state["stopping"] = True
        if self._protocol.min_updates_per_transition:
            # Wake a paced actor at the deadline so it can close its ordered stream.
            state.outbox.send_model(self._reasoner_id, None, update=state['training_updates'],
                                    final=False, completion=False)
        state.outbox.request_memories(self._memory_id, stop=True, **self._feedback)
        self._feedback = {}

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs: Any) -> LearnerState:
        """Start from the reasoner's clock or stop asynchronously at its deadline."""
        kind = task.get("kind") if isinstance(task, dict) else None
        if kind == "start" and not state["training_started"]:
            state["training_started"] = True
            self._reasoner_id = sender
            state["training_started_at"] = float(task.get("started_at", state.time))
            state["training_deadline"] = float(task.get("deadline", state.time + self._protocol.training_seconds))
            self._request_batch(state)
        elif kind == "stop" and not self._synchronized_training:
            self._stop_sampling(state)
        elif kind == "evaluation_started":
            state["evaluation_start_update"] = state["training_updates"]
        return state

    def _publish_model(self, state: LearnerState, *, final: bool, completion: bool = False) -> None:
        update = state["training_updates"]
        published = None
        if state["final_model_publication_update"] != update:
            published = deepcopy(self._online).cpu()
            published.eval()
            state["models_published"] += 1
            if state["first_model_publication_update"] is None:
                state["first_model_publication_update"] = update
            state["final_model_publication_update"] = update
        state.outbox.send_model(self._reasoner_id, published, update=update, final=final,
                                completion=completion, phase="frozen" if final else "training")

    def _save_model(self, state: LearnerState) -> None:
        state["final_save_update"] = state["training_updates"]
        state["model_artifact"] = POLICY_FILENAME
        state["saved_training_updates"] = state["training_updates"]
        save_policy_checkpoint(self._torch, self._online, self._policy_path, state["training_updates"],
                               training_protocol=self._protocol.record(), frozen=state["frozen"])
        state["model_saved"] = True

    def _freeze(self, state: LearnerState) -> None:
        if state["frozen"]:
            raise ValueError("Duplicate replay closure acknowledgment")
        if state["training_updates"] == 0:
            state["failure_reason"] = "training_deadline_before_first_update"
            state.outbox.terminate_agent(state["failure_reason"])
            return
        state["frozen"] = True
        state["freeze_update"] = state["training_updates"]
        state["final_model_update"] = state["training_updates"]
        state["final_model_fingerprint"] = model_state_fingerprint(self._online)
        self._flush_optimization(state)
        self._save_model(state)
        self._publish_model(state, final=True)

    def _record_optimization(self, state: LearnerState, loss: float, synced: bool, metrics: dict[str, float]) -> None:
        current = state["optimization_current"]
        if current is None:
            current = {"update_start": state["training_updates"], "losses": [], "target_syncs": 0,
                       "td_errors": [], "betas": [], "replay_ages": [], "policy_lags": [],
                       "diagnostics": []}
            state["optimization_current"] = current
        current["losses"].append(loss)
        current["target_syncs"] += int(synced)
        current['diagnostics'].append({key: metrics[key] for key in
            ('unweighted_loss', 'gradient_norm', 'weight_min', 'weight_mean', 'weight_max')})
        for key, metric in (("td_errors", "mean_abs_td_error"), ("betas", "beta"),
                            ("replay_ages", "mean_replay_age"), ("policy_lags", "mean_policy_lag")):
            current[key].append(metrics[metric])
        if len(current["losses"]) >= self._protocol.optimization_window_updates:
            self._flush_optimization(state)

    def _flush_optimization(self, state: LearnerState) -> None:
        current = state["optimization_current"]
        if current is None:
            return
        losses = current["losses"]
        state["optimization_windows"].append({
            "window_index": len(state["optimization_windows"]), "update_start": current["update_start"],
            "update_end": state["training_updates"], "elapsed_seconds": float(state.time),
            "mean_loss": float(np.mean(losses)), "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)), "final_loss": float(losses[-1]),
            "target_syncs": current["target_syncs"],
            "mean_abs_td_error": float(np.mean(current["td_errors"])),
            "beta": float(np.mean(current["betas"])),
            "mean_replay_age": float(np.mean(current["replay_ages"])),
            "mean_policy_lag": float(np.mean(current["policy_lags"])),
            **{key: float(np.mean([row[key] for row in current['diagnostics']]))
               for key in ('unweighted_loss', 'gradient_norm', 'weight_min', 'weight_mean', 'weight_max')},
        })
        state["optimization_current"] = None

    def on_memories(self, state: LearnerState, sender: str,
                    memories: Sequence[Belief | Observation], **kwargs: Any) -> LearnerState:
        """Perform at most one update, or complete the ordered freeze handshake."""
        if kwargs.get("replay_closed"):
            if memories or not state["stopping"]:
                raise ValueError("Unexpected replay closure acknowledgment")
            self._request_outstanding = False
            state["replay_closed"] = True
            self._freeze(state)
            return state
        if state["frozen"]:
            state["post_freeze_update_attempts"] += 1
            raise ValueError("Batch arrived after replay closure")
        if not self._request_outstanding:
            raise ValueError("Unsolicited replay batch")
        self._request_outstanding = False
        state["batches_received"] += 1
        if state["stopping"] or (not self._synchronized_training and float(state.time) >= state["training_deadline"]):
            state["unused_batches"] += 1
            self._stop_sampling(state)
            return state
        try:
            transitions = [cast(dict[str, Any], item.content) for item in memories if isinstance(item, Observation)]
            if len(transitions) != self._protocol.batch_size:
                raise ValueError("Replay batch has the wrong size")
            ids, weights = kwargs["entry_ids"], kwargs["weights"]
            if len(ids) != len(transitions) or len(weights) != len(transitions):
                raise ValueError("Replay metadata has the wrong size")
            if any(not np.isfinite(w) or not 0 < w <= 1 for w in weights):
                raise ValueError("Invalid importance weight")
            torch = self._torch
            states = torch.as_tensor(
                np.stack([t[K_STATE] for t in transitions]), dtype=torch.float32, device=self._device
            )
            next_states = torch.as_tensor(
                np.stack([t[K_NEXT_STATE] for t in transitions]), dtype=torch.float32, device=self._device
            )
            actions = torch.as_tensor(
                [t[K_ACTION] for t in transitions], dtype=torch.int64, device=self._device
            ).unsqueeze(1)
            rewards = torch.as_tensor(
                [t[K_REWARD] for t in transitions], dtype=torch.float32, device=self._device
            )
            discounts = torch.as_tensor(
                [t["bootstrap_discount"] for t in transitions],
                dtype=torch.float32,
                device=self._device,
            )
            importance = torch.as_tensor(weights, dtype=torch.float32, device=self._device)
            if not self._synchronized_training and float(state.time) >= state["training_deadline"]:
                state["unused_batches"] += 1
                self._stop_sampling(state)
                return state
            state["last_update_started"] = float(state.time)
            predicted = self._online(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_q = self._online(next_states)
                if self._protocol.action_masking:
                    masks = np.stack([legal_action_mask(t[K_NEXT_STATE][:OBS_SHAPE[0]]) for t in transitions])
                    next_q = next_q.masked_fill(~torch.as_tensor(masks, device=self._device), -torch.inf)
                bootstrap_actions = next_q.argmax(dim=1, keepdim=True)
                next_values = self._target(next_states).gather(1, bootstrap_actions).squeeze(1)
                targets = rewards + discounts * next_values
            errors = (targets - predicted).detach().abs()
            unweighted = torch.nn.functional.smooth_l1_loss(predicted, targets, reduction="none")
            loss = (importance * unweighted).mean()
            if not torch.isfinite(loss) or not torch.isfinite(errors).all():
                raise ValueError("Non-finite DQN loss or TD error")
            self._optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self._online.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            self._optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in self._online.parameters()):
                raise ValueError("Non-finite DQN parameters")
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            state["failure_reason"] = f"invalid_training_update: {exc}"
            state.outbox.terminate_agent(state["failure_reason"])
            return state
        state["training_updates"] += 1
        state["last_update_finished"] = float(state.time)
        update = state["training_updates"]
        synced = update % self._protocol.target_sync_steps == 0
        if synced:
            self._target.load_state_dict(self._online.state_dict())
            state["target_syncs"] += 1
        self._feedback = {"entry_ids": ids, "errors": errors.cpu().tolist()}
        self._record_optimization(state, float(loss.item()), synced, {
            "mean_abs_td_error": float(errors.mean().item()), "beta": float(kwargs["beta"]),
            "mean_replay_age": float(kwargs["mean_replay_age"]),
            "mean_policy_lag": float(np.mean([update - 1 - t["actor_update"] for t in transitions])),
            'unweighted_loss': float(unweighted.detach().mean().item()),
            'gradient_norm': float(gradient_norm.item()),
            'weight_min': float(importance.min().item()), 'weight_mean': float(importance.mean().item()),
            'weight_max': float(importance.max().item()),
        })
        interval = self._protocol.evaluation_interval_seconds
        elapsed = float(state.time) - state['training_started_at']
        if interval and elapsed >= state['next_evaluation_snapshot'] and elapsed < self._protocol.training_seconds:
            name = f"evaluation-{int(state['next_evaluation_snapshot']):06d}.pt"
            save_policy_checkpoint(torch, self._online, Path(self._policy_path).with_name(name), update,
                                   training_protocol=self._protocol.record(), frozen=True)
            state['evaluation_snapshots'].append({'filename': name, 'elapsed_seconds': elapsed, 'training_updates': update})
            state['next_evaluation_snapshot'] += interval
        if update == 1 or update % self._protocol.actor_publish_steps == 0:
            self._publish_model(state, final=False, completion=self._synchronized_training)
        elif self._synchronized_training:
            state.outbox.send_model(self._reasoner_id, None, update=update, final=False, completion=True)
        if self._synchronized_training:
            state["training_completions_sent"] += 1
        stop = (update == self._protocol.training_updates if self._synchronized_training
                else float(state.time) >= state["training_deadline"])
        if stop:
            self._stop_sampling(state)
        else:
            self._request_batch(state)
        return state

    def on_model_request(self, state: LearnerState, sender: str, **kwargs: Any) -> LearnerState:
        """Only scheduled publications are part of this treatment."""
        raise ValueError("Unscheduled model request")

    def on_last(self, state: LearnerState) -> LearnerState:
        """Save an explicitly unfrozen diagnostic checkpoint after interrupted training."""
        self._flush_optimization(state)
        if not state["model_saved"]:
            self._save_model(state)
        return state
