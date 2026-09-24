"""Reactive six-module DQN protocol for experiment 2-2-CR."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from copy import deepcopy
import hashlib
from importlib import import_module
from logging import INFO, WARNING
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy import random

from mhagenta import ActionStatus, Belief, Observation
from mhagenta.bases import KnowledgeBase, LearnerBase, LLReasonerBase, MemoryBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import (
    ActuatorState, KnowledgeState, LearnerState, LLState, MemoryState,
    PerceptorState,
)

from .policy import (
    ACHIEVEMENTS, FRAME_SHAPE, ILLEGAL_ACTION_PENALTY, N_ACTIONS,
    POLICY_FILENAME, STEP_REWARD, TARGET_ACHIEVEMENT_REWARD, as_rgb_frame,
    build_q_network, goal_achieved, initial_frame_stack, save_policy_checkpoint,
    shift_frame_stack, stack_batch,
)
from .replay import PrioritizedReplay, endpoint_stack
from .masking import action_mask, validate_mask
from .rewards import evaluate_reward, validate_evidence
from .treatment import (
    DEFAULT_WORKLOAD, DQNWorkload, DISCOUNT, EPS_GREEDY_END, EPS_GREEDY_START,
    EPS_GREEDY_STEPS, GRADIENT_CLIP, LEARNING_RATE, REPLAY_BATCH_SIZE,
    REPLAY_BUFFER_SIZE, TARGET_SYNC_STEPS, TRAINING_START_THRESHOLD,
)


DURATION = DEFAULT_WORKLOAD.duration_seconds
ENVIRONMENT_DURATION = DURATION + 30.0
MAX_EP_LENGTH = DEFAULT_WORKLOAD.episode_action_limit
TOTAL_TRAINING_TRANSITIONS = DEFAULT_WORKLOAD.training_transitions
TRAINING_UPDATES = DEFAULT_WORKLOAD.training_updates
EVALUATION_SEEDS = DEFAULT_WORKLOAD.evaluation_seeds
TRAINING_SHUTDOWN_MARGIN = 10.0
OPTIMIZATION_WINDOW_UPDATES = 100

K_ACTION = "action"
K_ACHIEVEMENTS = "achievements"
K_CLASSIFICATION = "classification"
K_CYCLE_ID = "cycle_id"
K_DONE = "done"
K_ILLEGAL_ACTION = "illegal_action"
K_NEXT_FRAME = "next_frame"
K_OBSERVATION = "observation"
K_REWARD = "reward"
K_STATE_STACK = "state_stack"
K_TERMINAL = "terminal"
K_TRANSITION_ORDINAL = "transition_ordinal"
K_PHASE = "phase"
K_REQUESTED_SEED = "requested_seed"
K_APPLIED_SEED = "applied_seed"

WARMUP = "warmup"
UPDATE_ELIGIBLE = "update_eligible"
SHUTDOWN_INELIGIBLE = "shutdown_ineligible"
TRANSITION_CLASSES = (WARMUP, UPDATE_ELIGIBLE, SHUTDOWN_INELIGIBLE)

A_RESET = "reset"
A_CLOSE = "close"
BELIEF_ILLEGAL = "illegal_action"
BELIEF_GOAL_ACHIEVED = "goal_achieved"


def _single_module_id(entries: Sequence[Any], role: str) -> str:
    """Resolve the only module registered for a required role."""
    if len(entries) != 1:
        raise RuntimeError(f"Experiment 2-2-CR requires one {role}; found {len(entries)}.")
    return cast(str, entries[0].module_id)


def validate_achievements(value: Any) -> dict[str, int]:
    """Validate Crafter's complete achievement-count mapping."""
    if not isinstance(value, dict) or set(value) != set(ACHIEVEMENTS):
        raise ValueError("Crafter achievement keys do not match the expected enum.")
    result: dict[str, int] = {}
    for name in ACHIEVEMENTS:
        count = value[name]
        if not isinstance(count, (int, np.integer)):
            raise TypeError(f"Achievement {name!r} must have an integer count.")
        result[name] = int(count)
    return result


def parse_action_status(status: Any) -> tuple[bool, bool, dict[str, int]]:
    """Validate the Crafter fields used by the experiment."""
    if not isinstance(status, dict):
        raise TypeError("Crafter action status must be a dictionary.")
    illegal = status.get(K_ILLEGAL_ACTION)
    done = status.get(K_DONE)
    if type(illegal) is not bool:
        raise TypeError("Crafter status requires Boolean 'illegal_action'.")
    if type(done) is not bool:
        raise TypeError("Crafter status requires Boolean 'done'.")
    evidence = validate_evidence(status.get("reward_evidence"))
    achievements = validate_achievements(status.get(K_ACHIEVEMENTS))
    if achievements != evidence["after"]["achievements"]:
        raise ValueError("Status achievements disagree with reward evidence.")
    dead = evidence["after"]["needs"]["health"] == 0
    if dead and not done:
        raise ValueError("Dead player must have a terminal environment status.")
    return illegal, dead, achievements


def make_replay_transition(
    evaluated_observation: Observation, state_stack: Any, action: Any, terminal: bool,
) -> Observation:
    """Build one compact four-frame replay transition."""
    if evaluated_observation.value is None:
        raise ValueError("Evaluated observations must carry an intrinsic value.")
    action_value = int(action)
    if not 0 <= action_value < N_ACTIONS:
        raise ValueError(f"Invalid action: {action_value}.")
    stack = np.asarray(state_stack)
    next_frame = as_rgb_frame(evaluated_observation.content)
    shift_frame_stack(stack, next_frame)
    reward = float(evaluated_observation.value)
    return Observation(
        content={
            K_STATE_STACK: stack.astype(np.uint8, copy=True), K_ACTION: action_value,
            K_REWARD: reward, K_NEXT_FRAME: next_frame, K_TERMINAL: bool(terminal),
        },
        observation_type="dqn_transition", value=reward,
    )


class TestEnvironment(MHAEnvBase):
    """Crafter adapter with compact environment-side evidence."""
    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed: int | None = init_state.pop("seed", None)
        self._action_masking = bool(init_state.get("action_masking", False))
        super().__init__(init_state)
        self._env: Any = None
        self._current_frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self._build_env()

    @staticmethod
    def _reset_info() -> dict[str, Any]:
        return {
            K_DONE: False, K_ILLEGAL_ACTION: False,
            K_ACHIEVEMENTS: {name: 0 for name in ACHIEVEMENTS},
        }

    def _snapshot(self) -> dict[str, Any]:
        """Capture reward-only context from the local Crafter instance."""
        player = self._env._player
        return {
            "needs": {name: int(player.inventory[name]) for name in ("health", "food", "drink", "energy")},
            "achievements": {name: int(value) for name, value in player.achievements.items()},
            "sleeping": bool(player.sleeping),
        }

    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if state["observation_requests"] == 0:
            self.log(INFO, "Crafter environment serving four-frame DQN run.")
        state["observation_requests"] += 1
        return state, {K_OBSERVATION: self._current_frame.copy(),
                       "action_mask": action_mask(self._env)}

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs: Any) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        if K_ACTION not in kwargs:
            state["contract_errors"] += 1
            return state, {}
        action = kwargs[K_ACTION]
        if action == A_RESET:
            requested_seed = kwargs.get(K_REQUESTED_SEED)
            applied_seed = None if requested_seed is None else int(requested_seed)
            if applied_seed is None:
                self._current_frame = as_rgb_frame(self._env.reset())
            else:
                self._seed = applied_seed
                self._build_env()
            state["resets"] += 1
            state["statuses"] += 1
            if kwargs.get(K_PHASE) == "frozen_evaluation":
                state["evaluation_resets"] += 1
                state["applied_evaluation_seeds"].append(applied_seed)
            return state, {
                **self._reset_info(),
                K_REQUESTED_SEED: requested_seed,
                K_APPLIED_SEED: applied_seed,
                K_REWARD: 0.0,
            }
        if action == A_CLOSE:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            return state, None
        if isinstance(action, str):
            state["contract_errors"] += 1
            return state, {}

        before = self._snapshot()
        if self._action_masking and not action_mask(self._env)[int(action)]:
            raise ValueError('Masked policy requested an inapplicable action.')
        observation, reward, done, info = self._env.step(int(action))
        evidence = {"before": before, "after": self._snapshot()}
        if K_ILLEGAL_ACTION not in info:
            raise RuntimeError("Crafter info is missing 'illegal_action'.")
        achievements = validate_achievements(info.get(K_ACHIEVEMENTS))
        illegal = info[K_ILLEGAL_ACTION]
        if type(illegal) is not bool:
            raise TypeError("Crafter info field 'illegal_action' must be Boolean.")
        if self._action_masking and illegal:
            raise ValueError('Native action validity disagrees with the mask.')
        self._current_frame = as_rgb_frame(observation)
        state["native_actions"] += 1
        state["statuses"] += 1
        return state, {
            K_DONE: bool(done), K_ILLEGAL_ACTION: illegal,
            K_ACHIEVEMENTS: achievements,
            K_REWARD: float(reward),
            "reward_evidence": evidence,
        }

    def _build_env(self) -> None:
        from mha_env_crafter import CrafterEnv

        self._env = CrafterEnv(seed=self._seed, length=10_000, symbolic=False, no_mobs=True)
        if self._env._no_mobs is not True:
            raise ValueError('CR requires no_mobs=True.')
        self._current_frame = as_rgb_frame(self._env.reset())

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()


class TestPerceptor(RMQPerceptorBase):
    """Forward environment observations to the low-level reasoner."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: PerceptorState) -> PerceptorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("No Crafter environment found.")
        self._env_id = environment.address["env_id"]
        self._reasoner_id = _single_module_id(state.directory.internal.ll_reasoning, "LL reasoner")
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        state["requests"] += 1
        self.observe(self._env_id)
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        state["observations_forwarded"] += 1
        state.outbox.send_observation(self._reasoner_id, Observation(kwargs.get(K_OBSERVATION)),
                                      action_mask=kwargs.get("action_mask"))
        return state


class TestActuator(RMQActuatorBase):
    """Forward reasoner actions and Crafter statuses."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("No Crafter environment found.")
        self._env_id = environment.address["env_id"]
        self._reasoner_id = _single_module_id(state.directory.internal.ll_reasoning, "LL reasoner")
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        self.act(self._env_id, action=A_CLOSE)
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        if K_ACTION not in kwargs:
            state["contract_errors"] += 1
            return state
        state["requests"] += 1
        self.act(
            self._env_id,
            action=kwargs[K_ACTION],
            phase=kwargs.get(K_PHASE),
            requested_seed=kwargs.get(K_REQUESTED_SEED),
        )
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        state["statuses_forwarded"] += 1
        state.outbox.send_status(
            self._reasoner_id,
            ActionStatus({K_DONE: kwargs.get(K_DONE),
                          K_ILLEGAL_ACTION: kwargs.get(K_ILLEGAL_ACTION),
                          K_ACHIEVEMENTS: kwargs.get(K_ACHIEVEMENTS),
                          K_REWARD: kwargs.get(K_REWARD),
                          K_REQUESTED_SEED: kwargs.get(K_REQUESTED_SEED),
                          K_APPLIED_SEED: kwargs.get(K_APPLIED_SEED),
                          "reward_evidence": kwargs.get("reward_evidence")}),
        )
        return state


class TestLLReasoner(LLReasonerBase):
    """Own frame assembly, action choice, and synchronized-cycle eligibility."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.default_rng()
        self._torch: Any = None
        self._model: Any = None
        self._actuator_id = ""
        self._perceptor_id = ""
        self._knowledge_id = ""
        self._frame_stack: np.ndarray | None = None
        self._previous_action: int | None = None
        self._last_illegal = False
        self._last_achieved = False
        self._last_done = False
        self._awaiting_reset = False
        self._episode_length = 0
        self._pending_cycle_id: int | None = None
        self._pending_stack: np.ndarray | None = None
        self._behavior_closed = False
        self._duration = DURATION
        self._workload = DEFAULT_WORKLOAD
        self._last_evidence: dict[str, Any] | None = None
        self._last_env_done = False
        self._training_progression: list[dict[str, Any]] = []
        self._training_achievements = {name: 0 for name in ACHIEVEMENTS}
        self._evaluation_seed: int | None = None
        self._evaluation_return = 0.0
        self._evaluation_progression: list[dict[str, Any]] = []
        self._previous_achievements = {name: 0 for name in ACHIEVEMENTS}

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.default_rng(kwargs.get("seed"))
        self._torch = import_module("torch")
        # Small CPU inference/tensor operations must not occupy every host core.
        self._torch.set_num_threads(1)
        self._duration = float(kwargs.get("duration", DURATION))
        self._workload = DQNWorkload(**kwargs.get("workload", DEFAULT_WORKLOAD.dump()))

    def on_first(self, state: LLState) -> LLState:
        self._actuator_id = _single_module_id(state.directory.internal.actuation, "actuator")
        self._perceptor_id = _single_module_id(state.directory.internal.perception, "perceptor")
        self._knowledge_id = _single_module_id(state.directory.internal.knowledge, "knowledge module")
        state["phase_timestamps"]["warmup_started"] = float(state.time)
        self._request_observation(state)
        return state

    def _epsilon(self, decisions: int) -> float:
        progress = min(decisions, EPS_GREEDY_STEPS) / EPS_GREEDY_STEPS
        return EPS_GREEDY_START + progress * (EPS_GREEDY_END - EPS_GREEDY_START)

    def _select_action(
        self, stack: np.ndarray, decisions: int, *, greedy: bool = False,
    ) -> int:
        if not greedy and (
            self._model is None or self._rng.random() < self._epsilon(decisions)
        ):
            if self._workload.action_masking:
                return int(self._rng.choice(np.flatnonzero(self._action_mask)))
            return int(self._rng.integers(N_ACTIONS))
        if self._model is None:
            raise RuntimeError("Frozen evaluation requires an installed model.")
        images = self._torch.as_tensor(
            stack_batch([stack]), dtype=self._torch.float32
        ) / 255.0
        with self._torch.no_grad():
            values = self._model(images)
            if self._workload.action_masking:
                valid = self._torch.as_tensor(self._action_mask, device=values.device)
                values = values.masked_fill(~valid, -self._torch.inf)
            return int(values.argmax(dim=1).item())

    def _environment_open(self, state: LLState) -> bool:
        if not self._behavior_closed:
            return True
        state["post_closure_environment_request_attempts"] += 1
        self.log(WARNING, "Blocked an environment request after behavior closure.")
        return False

    def _request_observation(self, state: LLState) -> None:
        if self._environment_open(state):
            state.outbox.request_observation(self._perceptor_id)

    def _request_reset(self, state: LLState, seed: int | None = None) -> None:
        if self._environment_open(state):
            self._awaiting_reset = True
            state.outbox.request_action(
                self._actuator_id,
                action=A_RESET,
                phase=state["phase"],
                requested_seed=seed,
            )

    def _request_policy_action(self, state: LLState, stack: np.ndarray) -> None:
        if not self._environment_open(state):
            return
        evaluating = state["phase"] == "frozen_evaluation"
        decisions = state["evaluation_actions"] if evaluating else state["actions"]
        action = self._select_action(stack, decisions, greedy=evaluating)
        if evaluating:
            state["evaluation_actions"] += 1
        else:
            state["actions"] += 1
            state["action_histogram"][action] += 1
        if not evaluating:
            state["policy_inferences"] += 1
        self._frame_stack = stack.copy()
        self._previous_action = action
        state.outbox.request_action(self._actuator_id, action=action)

    def _clear_episode(self) -> None:
        self._frame_stack = None
        self._previous_action = None
        self._episode_length = 0
        self._training_progression = []
        self._training_achievements = {name: 0 for name in ACHIEVEMENTS}

    def _record_outcome(self, state: LLState, terminal: bool) -> None:
        if not terminal:
            return
        if self._last_achieved:
            state["target_successes"] += 1
        elif self._last_done:
            state["deaths"] += 1
        else:
            state["truncations"] += 1
        state["completed_episodes"] += 1
        state["total_episode_length"] += self._episode_length
        state["training_episodes"].append({
            "episode_id": state["episodes_started"], "length": self._episode_length,
            "success": self._last_achieved, "death": self._last_done,
            "truncation": not self._last_achieved and not self._last_done,
            "achievement_progression": list(self._training_progression),
        })

    def _classify_transition(self, state: LLState, ordinal: int) -> tuple[str, int | None]:
        if ordinal <= TRAINING_START_THRESHOLD:
            return WARMUP, None
        if ordinal > self._workload.training_transitions:
            state["shutdown_ineligible_transitions"] += 1
            return SHUTDOWN_INELIGIBLE, None
        cycle_id = state["cycles_started"] + 1
        state["cycles_started"] = cycle_id
        state["update_eligible_transitions"] += 1
        if ordinal == TRAINING_START_THRESHOLD + 1:
            state["phase"] = "training"
            state["phase_timestamps"]["training_started"] = float(state.time)
        return UPDATE_ELIGIBLE, cycle_id

    def _handle_evaluation_observation(
        self, state: LLState, current: np.ndarray,
    ) -> LLState:
        """Advance one frozen evaluation episode without touching replay."""

        if self._frame_stack is None:
            self._frame_stack = initial_frame_stack(current)
            self._request_policy_action(state, self._frame_stack)
            return state
        next_stack = shift_frame_stack(self._frame_stack, current)
        terminal = (
            self._last_achieved
            or self._last_done
            or self._last_env_done
            or self._episode_length >= self._workload.episode_action_limit
        )
        if not terminal:
            self._request_policy_action(state, next_stack)
            return state

        outcome = (
            "success" if self._last_achieved
            else "death" if self._last_done
            else "truncation"
        )
        state["evaluation_cases"].append({
            "seed": self._evaluation_seed,
            "success": outcome == "success",
            "death": outcome == "death",
            "truncation": outcome == "truncation",
            "return": float(self._evaluation_return),
            "length": self._episode_length,
            "achievement_progression": list(self._evaluation_progression),
            "checkpoint_digest": state["frozen_checkpoint_digest"],
        })
        state["evaluation_index"] += 1
        self._clear_episode()
        if state["evaluation_index"] == len(self._workload.evaluation_seeds):
            state["phase"] = "complete"
            state["phase_timestamps"]["complete"] = float(state.time)
            self._behavior_closed = True
            state.outbox.terminate_agent()
            return state
        self._request_reset(state, self._workload.evaluation_seeds[state["evaluation_index"]])
        return state

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs: Any) -> LLState:
        if self._behavior_closed:
            state["invalid_inputs"] += 1
            self.log(WARNING, "Ignored an observation after behavior closure.")
            return state
        try:
            current = as_rgb_frame(observation.content)
            if self._workload.action_masking:
                self._action_mask = validate_mask(kwargs.get("action_mask"))
        except (TypeError, ValueError) as exc:
            state["invalid_inputs"] += 1
            state["stack_contract_errors"] += 1
            self.log(WARNING, f"Invalid RGB observation from {sender}: {exc}")
            return state
        state["observations"] += 1

        if state["phase"] == "frozen_evaluation":
            state["evaluation_observations"] += 1
            return self._handle_evaluation_observation(state, current)

        if self._frame_stack is None:
            self._frame_stack = initial_frame_stack(current)
            state["stack_initializations"] += 1
            state["episodes_started"] += 1
            self._request_policy_action(state, self._frame_stack)
            return state
        if self._previous_action is None or self._pending_cycle_id is not None:
            state["stack_contract_errors"] += 1
            self.log(WARNING, "Observation arrived without valid action context.")
            return state

        state_stack = self._frame_stack.copy()
        next_stack = shift_frame_stack(state_stack, current)
        state["stack_shifts"] += 1
        next_ordinal = state["transitions_emitted"] + 1
        terminal = (
            self._last_achieved
            or self._last_done
            or self._last_env_done
            or self._episode_length >= self._workload.episode_action_limit
            or next_ordinal == self._workload.training_transitions
        )
        state["transitions_emitted"] += 1
        ordinal = state["transitions_emitted"]
        classification, cycle_id = self._classify_transition(state, ordinal)
        state.outbox.send_beliefs(
            self._knowledge_id,
            Observation(current),
            [Belief(BELIEF_ILLEGAL, self._last_illegal),
             Belief(BELIEF_GOAL_ACHIEVED, self._last_achieved)],
            state_stack=state_stack,
            next_action_mask=self._action_mask.tolist() if self._workload.action_masking else None,
            action=self._previous_action,
            terminal=self._last_done or self._last_achieved,
            boundary=terminal,
            episode_id=state["episodes_started"],
            reward_evidence=self._last_evidence,
            transition_ordinal=ordinal,
            classification=classification,
            cycle_id=cycle_id,
        )
        self._record_outcome(state, terminal)

        if ordinal == self._workload.training_transitions:
            state["phase"] = "drain"
            state["phase_timestamps"]["drain_started"] = float(state.time)
            state["training_closed_at_transition"] = ordinal
            state["training_closed_at_elapsed_seconds"] = float(state.time)

        if classification == SHUTDOWN_INELIGIBLE:
            self._clear_episode()
            return state
        if classification == UPDATE_ELIGIBLE:
            self._pending_cycle_id = cycle_id
            self._pending_stack = None if terminal else next_stack
            return state
        if terminal:
            self._clear_episode()
            self._request_reset(state)
        else:
            self._request_policy_action(state, next_stack)
        return state

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs: Any) -> LLState:
        if self._behavior_closed:
            state["invalid_inputs"] += 1
            self.log(WARNING, "Ignored an action status after behavior closure.")
            return state
        state["statuses"] += 1
        if self._awaiting_reset:
            self._awaiting_reset = False
            self._clear_episode()
            self._last_illegal = self._last_achieved = self._last_done = False
            self._last_env_done = False
            if state["phase"] == "frozen_evaluation":
                self._evaluation_seed = action_status.status.get(K_APPLIED_SEED)
                if self._evaluation_seed != self._workload.evaluation_seeds[state["evaluation_index"]]:
                    state["invalid_inputs"] += 1
                    state.outbox.terminate_agent("Evaluation reset applied the wrong seed.")
                    return state
                self._evaluation_return = 0.0
                self._evaluation_progression = []
                self._previous_achievements = {name: 0 for name in ACHIEVEMENTS}
            self._request_observation(state)
            return state
        try:
            illegal, done, achievements = parse_action_status(action_status.status)
            achieved = goal_achieved(achievements) and not done
        except (TypeError, ValueError) as exc:
            state["invalid_inputs"] += 1
            self.log(WARNING, f"Invalid Crafter action status from {sender}: {exc}")
            return state
        self._last_illegal = illegal
        self._last_achieved = achieved
        self._last_done = done
        self._last_env_done = action_status.status[K_DONE]
        self._last_evidence = action_status.status["reward_evidence"]
        self._episode_length += 1
        if state["phase"] == "frozen_evaluation":
            self._evaluation_return += float(action_status.status.get(K_REWARD, 0.0))
            for name, count in achievements.items():
                previous = self._previous_achievements[name]
                if count > previous:
                    self._evaluation_progression.append({
                        "action": self._episode_length,
                        "achievement": name,
                        "count": count,
                    })
            self._previous_achievements = achievements
        else:
            for name, count in achievements.items():
                if count > 0 and self._training_achievements[name] == 0:
                    self._training_progression.append({"action": self._episode_length, "achievement": name})
            self._training_achievements = achievements
        self._request_observation(state)
        return state

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs: Any) -> LLState:
        cycle_id = kwargs.get(K_CYCLE_ID)
        if self._pending_cycle_id is None or cycle_id != self._pending_cycle_id:
            state["cycle_errors"] += 1
            self.log(WARNING, f"Unexpected training completion {cycle_id!r}.")
            state.outbox.terminate_agent("Unexpected training completion.")
            return state
        if model is not None:
            self._model = model.cpu()
            self._model.eval()
            state["models_installed"] += 1
        state["training_completions_received"] += 1
        state["cycles_completed"] += 1
        pending_stack = self._pending_stack
        self._pending_cycle_id = None
        self._pending_stack = None
        if state["phase"] == "drain" and state["cycles_completed"] == self._workload.training_updates:
            if model is None:
                state["cycle_errors"] += 1
                state.outbox.terminate_agent("Final completion omitted the frozen model.")
                return state
            state["phase"] = "frozen_evaluation"
            state["phase_timestamps"]["frozen_evaluation_started"] = float(state.time)
            state["frozen_checkpoint_digest"] = kwargs.get("checkpoint_digest")
            self._clear_episode()
            self._request_reset(state, self._workload.evaluation_seeds[0])
        elif pending_stack is None:
            self._clear_episode()
            self._request_reset(state)
        else:
            self._request_policy_action(state, pending_stack)
        return state


class TestKnowledge(KnowledgeBase):
    """Assign intrinsic reward and forward each classified transition."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._memory_id = ""

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        self._memory_id = _single_module_id(state.directory.internal.memory, "memory module")
        return state

    def _fail(self, state: KnowledgeState, message: str) -> KnowledgeState:
        state["contract_errors"] += 1
        self.log(WARNING, message)
        state.outbox.terminate_agent(message)
        return state

    def on_observed_beliefs(
        self, state: KnowledgeState, sender: str, observation: Observation,
        beliefs: Sequence[Belief], **kwargs: Any,
    ) -> KnowledgeState:
        values = {belief.predicate: bool(belief.arguments) for belief in beliefs}
        if set(values) != {BELIEF_ILLEGAL, BELIEF_GOAL_ACHIEVED}:
            return self._fail(state, f"Unexpected belief set from {sender}: {values}")
        classification = kwargs.get(K_CLASSIFICATION)
        if classification not in TRANSITION_CLASSES:
            return self._fail(state, f"Invalid transition class {classification!r}.")
        ordinal = kwargs.get(K_TRANSITION_ORDINAL)
        if ordinal != state["evaluated_transitions"] + 1:
            return self._fail(state, "Knowledge transition ordinal is out of order.")
        try:
            components = evaluate_reward(
                state["reward_tracker"], kwargs.get("episode_id"), values[BELIEF_ILLEGAL],
                kwargs.get("reward_evidence"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(state, f"Invalid reward transition: {exc}")
        reward = sum(components.values())
        for name, amount in components.items():
            state["reward_components"][name] = state["reward_components"].get(name, 0.0) + amount
        episode_id = kwargs["episode_id"]
        if not state["episode_rewards"] or state["episode_rewards"][-1]["episode_id"] != episode_id:
            state["episode_rewards"].append({"episode_id": episode_id, "return": 0.0, "components": {}})
        episode = state["episode_rewards"][-1]
        episode["return"] += reward
        for name, amount in components.items():
            episode["components"][name] = episode["components"].get(name, 0.0) + amount
        evaluated = Observation(
            content=as_rgb_frame(observation.content),
            observation_type=observation.observation_type,
            value=reward,
        )
        state["evaluated_transitions"] += 1
        state[f"{classification}_transitions"] += 1
        state["cumulative_intrinsic_reward"] += reward
        replay_kwargs = {key: value for key, value in kwargs.items() if key != "reward_evidence"}
        state.outbox.send_observations(self._memory_id, [evaluated], **replay_kwargs)
        return state


class TestMemory(MemoryBase):
    """Finalize three-step experiences and acknowledge priority feedback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replay = PrioritizedReplay(random.default_rng())
        self._workload = DEFAULT_WORKLOAD
        self._learner_id = ""
        self._pending_cycle: int | None = None
        self._sample_ids: list[int] = []

    def on_init(self, **kwargs: Any) -> None:
        self._replay = PrioritizedReplay(random.default_rng(kwargs.get("seed")))
        self._workload = DQNWorkload(**kwargs.get("workload", DEFAULT_WORKLOAD.dump()))

    def on_first(self, state: MemoryState) -> MemoryState:
        self._learner_id = _single_module_id(state.directory.internal.learning, "learner")
        return state

    def _fail(self, state: MemoryState, message: str) -> MemoryState:
        state["contract_errors"] += 1
        self.log(WARNING, message)
        state.outbox.terminate_agent(message)
        return state

    def on_observation_update(
        self, state: MemoryState, sender: str, observations: Sequence[Observation],
        **kwargs: Any,
    ) -> MemoryState:
        """Admit one raw transition; sample mature replay independently of its horizon."""
        ordinal = kwargs.get(K_TRANSITION_ORDINAL)
        cycle_id = kwargs.get(K_CYCLE_ID)
        classification = kwargs.get(K_CLASSIFICATION)
        expected_class = WARMUP if type(ordinal) is int and ordinal <= TRAINING_START_THRESHOLD else UPDATE_ELIGIBLE
        expected_cycle = None if expected_class == WARMUP else state["next_cycle_id"]
        if (len(observations) != 1 or type(ordinal) is not int
                or ordinal != state["transitions_admitted"] + 1
                or ordinal > self._workload.training_transitions
                or classification != expected_class or cycle_id != expected_cycle
                or self._pending_cycle is not None):
            return self._fail(state, "Invalid transition admission or cycle token.")
        try:
            transition = make_replay_transition(
                observations[0], kwargs.get(K_STATE_STACK), kwargs.get(K_ACTION),
                bool(kwargs.get(K_TERMINAL, False)),
            ).content
            transition["replay_id"] = ordinal
            if self._workload.action_masking:
                transition["next_action_mask"] = validate_mask(kwargs.get("next_action_mask")).tolist()
            boundary = kwargs.get("boundary")
            if type(boundary) is not bool or (transition["terminal"] and not boundary):
                raise ValueError("Invalid episode boundary.")
            if ordinal == self._workload.training_transitions and not boundary:
                raise ValueError("Final transition must flush replay.")
            self._replay.append(transition, boundary=boundary)
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(state, f"Invalid replay transition: {exc}")
        state["transitions_admitted"] += 1
        state[f"{classification}_transitions"] += 1
        state["buffer_size"] = len(self._replay.items)
        state["experiences_finalized"] = self._replay.finalized
        state["pending_transitions"] = len(self._replay.pending)
        state["horizon_counts"] = list(self._replay.horizons)
        if cycle_id is not None:
            batch, self._sample_ids, weights = self._replay.sample(cycle_id, self._workload.training_updates)
            self._pending_cycle = cycle_id
            state.outbox.send_memories(
                self._learner_id, [Observation(item, observation_type="dqn_n_step") for item in batch],
                kind="batch", cycle_id=cycle_id, replay_ids=self._sample_ids, weights=weights,
            )
            state["batches_sent"] += 1
            state["next_cycle_id"] += 1
        return state

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs: Any) -> MemoryState:
        """Apply exactly the outstanding batch's priorities before acknowledging it."""
        cycle_id = kwargs.get(K_CYCLE_ID)
        if (kwargs.get("kind") != "priority_update" or self._pending_cycle is None
                or cycle_id != self._pending_cycle or kwargs.get("replay_ids") != self._sample_ids):
            return self._fail(state, "Unexpected priority update.")
        try:
            self._replay.update_priorities(self._sample_ids, kwargs.get("td_errors", []))
        except (TypeError, ValueError) as exc:
            return self._fail(state, f"Invalid priority feedback: {exc}")
        self._pending_cycle = None
        self._sample_ids = []
        state["priority_updates"] += 1
        state.outbox.send_memories(self._learner_id, [], kind="priority_ack", cycle_id=cycle_id)
        return state


def double_dqn_targets(torch: Any, online: Any, target: Any, next_states: Any,
                       rewards: Any, discounts: Any, next_masks: Any = None) -> Any:
    """Select with the online network and evaluate with the target network."""
    with torch.no_grad():
        values = online(next_states)
        if next_masks is not None:
            if next_masks.shape != values.shape or not next_masks.any(dim=1).all().item():
                raise ValueError('Every target needs a matching nonempty action mask.')
            values = values.masked_fill(~next_masks, -torch.inf)
        actions = values.argmax(dim=1, keepdim=True)
        return rewards + discounts * target(next_states).gather(1, actions).squeeze(1)


class TestLearner(LearnerBase):
    """Optimize weighted Double DQN, then wait for replay-priority acknowledgement."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._torch: Any = None
        self._online: Any = None
        self._target: Any = None
        self._optimizer: Any = None
        self._device: Any = None
        self._reasoner_id = ""
        self._memory_id = ""
        self._pending_cycle: int | None = None
        self._policy_path = f"/out/{POLICY_FILENAME}"
        self._workload = DEFAULT_WORKLOAD

    def on_init(self, **kwargs: Any) -> None:
        self._torch = import_module("torch")
        # Small CPU inference/tensor operations must not occupy every host core.
        self._torch.set_num_threads(1)
        seed = int(kwargs.get("seed", 0))
        self._torch.manual_seed(seed)
        self._policy_path = str(kwargs.get("policy_path", self._policy_path))
        self._workload = DQNWorkload(**kwargs.get("workload", DEFAULT_WORKLOAD.dump()))
        requested_device = str(kwargs.get("device", "cpu"))
        if requested_device.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for 2-2-CR but is unavailable")
        self._device = self._torch.device(requested_device)
        if self._device.type == "cuda":
            self._torch.cuda.manual_seed_all(seed)
        self._online = build_q_network(self._torch).to(self._device)
        self._target = deepcopy(self._online)
        self._target.eval()
        self._optimizer = self._torch.optim.AdamW(self._online.parameters(), lr=LEARNING_RATE)

    def on_first(self, state: LearnerState) -> LearnerState:
        state["device"] = str(self._device)
        state["torch_version"] = str(self._torch.__version__)
        state["cuda_runtime"] = self._torch.version.cuda
        state["cuda_available"] = bool(self._torch.cuda.is_available())
        state["cuda_device_name"] = (
            self._torch.cuda.get_device_name(self._device)
            if self._device.type == "cuda"
            else None
        )
        self._reasoner_id = _single_module_id(state.directory.internal.ll_reasoning, "LL reasoner")
        self._memory_id = _single_module_id(state.directory.internal.memory, "memory")
        return state

    def _fail(self, state: LearnerState, message: str) -> LearnerState:
        state["contract_errors"] += 1
        self.log(WARNING, message)
        state.outbox.terminate_agent(message)
        return state

    def on_memories(self, state: LearnerState, sender: str,
                    memories: Sequence[Belief | Observation], **kwargs: Any) -> LearnerState:
        """Dispatch a replay batch or the acknowledgement of its priority update."""
        if state["frozen"]:
            state["post_freeze_update_attempts"] += 1
            return self._fail(state, "Replay message received after freezing.")
        cycle_id = kwargs.get(K_CYCLE_ID)
        if kwargs.get("kind") == "priority_ack":
            if memories or self._pending_cycle is None or cycle_id != self._pending_cycle:
                return self._fail(state, "Unexpected priority acknowledgement.")
            self._pending_cycle = None
            state["priority_acknowledgements"] += 1
            return self._complete_update(state, cycle_id)
        if (kwargs.get("kind") != "batch" or self._pending_cycle is not None
                or cycle_id != state["training_updates"] + 1
                or (self._workload.synchronized_training
                    and state["training_updates"] >= self._workload.training_updates)):
            return self._fail(state, "Unexpected replay batch.")
        try:
            transitions = [memory.content for memory in memories if isinstance(memory, Observation)]
            ids, weights = kwargs.get("replay_ids"), kwargs.get("weights")
            if (len(transitions) != REPLAY_BATCH_SIZE or not isinstance(ids, list)
                    or ids != [item["replay_id"] for item in transitions]
                    or not isinstance(weights, list) or len(weights) != REPLAY_BATCH_SIZE
                    or any(not np.isfinite(weight) or not 0 < weight <= 1 for weight in weights)):
                raise ValueError("Invalid replay batch IDs or importance weights.")
            for item in transitions:
                horizon = item.get("horizon")
                if (type(horizon) is not int or not 1 <= horizon <= 3
                        or len(item.get("next_frames", [])) != horizon
                        or type(item.get("action")) is not int or not 0 <= item["action"] < N_ACTIONS
                        or not np.isfinite(item["reward"])
                        or item["bootstrap_discount"] not in (0.0, DISCOUNT ** horizon)):
                    raise ValueError("Invalid n-step target contract.")
            def tensor(values: Any, dtype: Any = None) -> Any:
                return self._torch.as_tensor(
                    values, dtype=dtype or self._torch.float32, device=self._device
                )

            states = tensor(stack_batch([item[K_STATE_STACK] for item in transitions])) / 255.0
            next_states = tensor(stack_batch([endpoint_stack(item) for item in transitions])) / 255.0
            actions = tensor([item[K_ACTION] for item in transitions], self._torch.int64).unsqueeze(1)
            rewards = tensor([item[K_REWARD] for item in transitions])
            discounts = tensor([item["bootstrap_discount"] for item in transitions])
            predicted = self._online(states).gather(1, actions).squeeze(1)
            next_masks = (tensor(np.stack([validate_mask(item.get('next_action_mask')) for item in transitions]),
                                 self._torch.bool) if self._workload.action_masking else None)
            targets = double_dqn_targets(self._torch, self._online, self._target, next_states,
                                        rewards, discounts, next_masks)
            td_error = targets - predicted
            losses = self._torch.nn.functional.smooth_l1_loss(predicted, targets, reduction="none")
            loss = (tensor(weights) * losses).mean()
            if not self._torch.isfinite(loss).item():
                raise ValueError("Non-finite DQN loss.")
            self._optimizer.zero_grad()
            loss.backward()
            self._torch.nn.utils.clip_grad_norm_(self._online.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            self._optimizer.step()
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return self._fail(state, f"Could not train from replay: {exc}")
        state["training_updates"] += 1
        if state["training_updates"] % TARGET_SYNC_STEPS == 0:
            self._target.load_state_dict(self._online.state_dict())
            state["target_syncs"] += 1
        self._record_optimization(state, float(loss.detach().item()), float(td_error.detach().abs().mean().item()))
        self._pending_cycle = cycle_id
        state.outbox.request_memories(
            self._memory_id, kind="priority_update", cycle_id=cycle_id, replay_ids=ids,
            td_errors=td_error.detach().abs().cpu().tolist(),
        )
        state["priority_updates_sent"] += 1
        return state

    def _record_optimization(self, state: LearnerState, loss: float, error: float) -> None:
        """Persist bounded optimization summaries rather than unbounded loss history."""
        current = state["optimization_current"]
        if current is None:
            current = {"update_start": state["training_updates"], "losses": [],
                       "mean_absolute_td_errors": [], "elapsed_start": float(state.time)}
            state["optimization_current"] = current
        current["losses"].append(loss)
        current["mean_absolute_td_errors"].append(error)
        if len(current["losses"]) >= OPTIMIZATION_WINDOW_UPDATES or state["training_updates"] == self._workload.training_updates:
            state["optimization_windows"].append({
                "update_start": current["update_start"], "update_end": state["training_updates"],
                "elapsed_start": current["elapsed_start"], "elapsed_end": float(state.time),
                "mean_loss": float(np.mean(current["losses"])),
                "mean_absolute_td_error": float(np.mean(current["mean_absolute_td_errors"])),
            })
            state["optimization_current"] = None

    def _complete_update(self, state: LearnerState, cycle_id: int) -> LearnerState:
        """Publish a completed update only after its priorities were installed."""
        final = state["training_updates"] == self._workload.training_updates
        model = None
        if state["training_updates"] == 1 or state["training_updates"] % TARGET_SYNC_STEPS == 0 or final:
            model = deepcopy(self._online).cpu()
            model.eval()
            state["models_published"] += 1
        if final:
            checkpoint = save_policy_checkpoint(self._torch, self._online, self._policy_path,
                                                state["training_updates"], self._workload)
            state["model_saved"] = True
            state["model_artifact"] = POLICY_FILENAME
            state["saved_training_steps"] = state["training_updates"]
            state["checkpoint_digest"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state["frozen"] = True
            self._online.eval()
        state.outbox.send_model(self._reasoner_id, model, cycle_id=cycle_id,
                                final=final, checkpoint_digest=state["checkpoint_digest"])
        state["training_completions_sent"] += 1
        return state

    def on_last(self, state: LearnerState) -> LearnerState:
        """Retain an incomplete diagnostic checkpoint if execution was interrupted."""
        if state["model_saved"]:
            return state
        state["model_artifact"] = POLICY_FILENAME
        state["saved_training_steps"] = state["training_updates"]
        try:
            save_policy_checkpoint(self._torch, self._online, self._policy_path,
                                   state["training_updates"], self._workload)
            state["model_saved"] = True
        except Exception as exc:
            self.log(WARNING, f"Could not save final DQN policy: {exc}")
        return state
