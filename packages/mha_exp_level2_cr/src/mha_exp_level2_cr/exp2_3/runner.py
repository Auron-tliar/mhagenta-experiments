"""Experiment 2-3-CR: full-domain BDI planning in symbolic Crafter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.util import find_spec
import json
from logging import ERROR, INFO, WARNING
import os
from pathlib import Path
import random
import re
from time import perf_counter
from typing import Any, Literal, cast

from mhagenta import ActionStatus, Belief, Observation, Orchestrator
from mhagenta.bases import HLReasonerBase, KnowledgeBase, LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, HLState, KnowledgeState, LLState, PerceptorState

import mha_exp_common
from mha_exp_common.batch import normalize_runs, run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import ACTUATOR, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name

from .beliefs import (
    BeliefRevisionError,
    CrafterAction,
    ObservationError,
    beliefs_to_percept,
    cell_key,
    initial_belief_state,
    parse_symbolic_observation,
    percept_to_beliefs,
    revise_belief_state,
    summarize_belief_state,
    support_signature,
)
from .planning import (
    GroundedAction,
    PlanningOutcome,
    PlanningService,
    TechnologyStage,
    action_sound,
    choose_exploration_action,
    derive_technology_stage,
    highest_milestone,
    plan_goal_observed,
    recovery_complete,
    recovery_supported,
    select_recovery_intention,
    stage_supported,
)
from .reporting import process_execution_metrics
from .treatment import ACTION_BUDGET, EPISODE_LENGTH, treatment_for_run


DURATION = 900.0
STARTUP_DELAY = 20.0
ENVIRONMENT_OVERRUN = 30.0
MAX_EPISODE_LEN = EPISODE_LENGTH
MAX_AGENT_STEPS = ACTION_BUDGET
PLANNER_TIMEOUT_SECONDS = 10.0
SHUTDOWN_MARGIN_SECONDS = 45.0
PLAN_CHUNK_SIZE = 8
PLANNING_RETRY_LIMIT = 2
EXPLORATION_BURST_SIZE = 4
EXPLORATION_STALL_LIMIT = 64
RECORD: Literal["all", "first", "none"] = "first"
VERBOSE = True

K_ACTION = "action"
K_OBSERVATION = "observation"
K_REWARD = "reward"
K_DONE = "done"
K_DEAD = "dead"
K_ILLEGAL_ACTION = "illegal_action"
K_NEW_ACHIEVEMENTS = "new_achievements"
A_CLOSE = "close"

LOG_PATTERN = re.compile(
    r"^\[(?P<time>[^\]]+)\]\[(?P<level>[^\]]+)\]::"
    r"(?P<tags>(?:\[[^\]]+\])+)::(?P<message>.*)$"
)
FINAL_PHASES = {"complete", "scientific-terminal", "experiment-error"}


@contextmanager
def _active_time(state: Any) -> Iterator[None]:
    """Accumulate time spent in one experiment callback."""

    started = perf_counter()
    try:
        yield
    finally:
        state["active_seconds"] += perf_counter() - started


def _require_one(items: Sequence[Any], label: str) -> Any:
    if len(items) != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {len(items)}")
    return items[0]


class CrafterBDIEnvironment(MHAEnvBase):
    """Expose one non-resetting symbolic Crafter episode through public APIs."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = int(init_state.pop("seed"))
        self._record = bool(init_state.pop("record", False))
        self._artifact_root = str(
            init_state.pop("artifact_root", f"/{Orchestrator.SAVE_SUBDIR}")
        )
        self._env: Any = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_crafter import CrafterEnv, Recorder

        environment = CrafterEnv(
            seed=self._seed,
            length=MAX_EPISODE_LEN,
            no_mobs=True,
            symbolic=True,
            daylight_effects=False,
        )
        self._env = (
            Recorder(
                environment,
                Path(self._artifact_root) / "videos",
                save_stats=False,
                save_episode=False,
                video_size=(144, 144),
                video_fps=2,
            )
            if self._record
            else environment
        )
        self._env.reset()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()

    def on_observe(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the current public symbolic observation."""

        state["observation_requests"] += 1
        if state["terminal"]:
            state["post_terminal_observations"] += 1
        return state, {K_OBSERVATION: list(self._env.symbolic_observation())}

    def on_action(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """Execute one native action and return public outcome evidence."""

        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            if not state["closed"]:
                state["close_requests"] += 1
                close = getattr(self._env, "close", None)
                if callable(close):
                    close()
                video_dir = Path(self._artifact_root) / "videos"
                if self._record and video_dir.exists() and any(video_dir.glob("*.mp4")):
                    state["video_path"] = "videos"
                state["closed"] = True
            return state, None
        if type(action) is not int or action not in range(len(CrafterAction)):
            raise ValueError(f"Invalid native Crafter action: {action!r}")
        if state["terminal"]:
            raise RuntimeError("Action requested after terminal Crafter state")

        _, reward, done, info = self._env.step(action)
        illegal = info.get(K_ILLEGAL_ACTION)
        inventory = info.get("inventory")
        achievements = info.get("achievements")
        if type(illegal) is not bool:
            raise RuntimeError("Crafter info must contain Boolean illegal_action")
        if not isinstance(inventory, dict) or not isinstance(achievements, dict):
            raise RuntimeError("Crafter info is missing inventory or achievements")
        counts = {name: int(count) for name, count in achievements.items()}
        new_achievements = sorted(
            name
            for name, count in counts.items()
            if count > int(state["achievement_counts"].get(name, 0))
        )
        dead = int(inventory.get("health", 0)) <= 0
        state["native_actions"] += 1
        state["illegal_actions"] += int(illegal)
        state["terminal_events"] += int(bool(done))
        state["inventory"] = {
            name: int(count) for name, count in sorted(inventory.items())
        }
        state["achievement_counts"] = counts
        state["achievements"] = sorted(
            name for name, count in counts.items() if count > 0
        )
        state["terminal"] = bool(done)
        state["dead"] = dead
        return state, {
            K_REWARD: float(reward),
            K_DONE: bool(done),
            K_DEAD: dead,
            K_ILLEGAL_ACTION: illegal,
            K_NEW_ACHIEVEMENTS: new_achievements,
        }


class CrafterBDIPerceptor(RMQPerceptorBase):
    """Forward public symbolic observations to the supporting LL reasoner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""
        self._awaiting_observation = False

    def _resolve_ids(self, state: PerceptorState) -> None:
        if not self._env_id:
            environment = _require_one(
                state.directory.external.environments, "external environment"
            )
            self._env_id = environment.address["env_id"]
        if not self._ll_id:
            self._ll_id = _require_one(
                state.directory.internal.ll_reasoning, "low-level reasoner"
            ).module_id

    def on_first(self, state: PerceptorState) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
        return state

    def on_request(
        self,
        state: PerceptorState,
        sender: str,
        **kwargs: Any,
    ) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._ll_id:
                raise RuntimeError(
                    f"Observation request came from unexpected module {sender!r}"
                )
            if self._awaiting_observation:
                raise RuntimeError("An observation request is already in flight")
            self._awaiting_observation = True
            state["requests"] += 1
            self.observe(self._env_id)
        return state

    def on_observation(
        self,
        state: PerceptorState,
        env_id: str,
        **kwargs: Any,
    ) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
            if env_id != self._env_id or not self._awaiting_observation:
                raise RuntimeError("Received an unexpected environment observation")
            content = kwargs.get(K_OBSERVATION)
            if not isinstance(content, list) or not all(
                isinstance(item, str) for item in content
            ):
                raise ValueError("Crafter observation must be a symbolic string list")
            self._awaiting_observation = False
            state["observations"] += 1
            state.outbox.send_observation(
                self._ll_id,
                Observation(content, observation_type="crafter-symbolic"),
            )
        return state


class CrafterBDIActuator(RMQActuatorBase):
    """Execute one HLR-originated action and associate its public result."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._hl_id = ""
        self._ll_id = ""
        self._pending_action_id: str | None = None

    def _resolve_ids(self, state: ActuatorState) -> None:
        if not self._env_id:
            environment = _require_one(
                state.directory.external.environments, "external environment"
            )
            self._env_id = environment.address["env_id"]
        if not self._hl_id:
            self._hl_id = _require_one(
                state.directory.internal.hl_reasoning, "high-level reasoner"
            ).module_id
        if not self._ll_id:
            self._ll_id = _require_one(
                state.directory.internal.ll_reasoning, "low-level reasoner"
            ).module_id

    def on_first(self, state: ActuatorState) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
        return state

    def on_request(
        self,
        state: ActuatorState,
        sender: str,
        **kwargs: Any,
    ) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._hl_id:
                raise RuntimeError(f"Action request came from non-HLR module {sender!r}")
            if self._pending_action_id is not None:
                raise RuntimeError("An action request is already in flight")
            action = kwargs.get(K_ACTION)
            action_id = kwargs.get("action_id")
            if type(action) is not int or action not in range(len(CrafterAction)):
                raise ValueError(f"Invalid native Crafter action: {action!r}")
            if not isinstance(action_id, str):
                raise ValueError("Action request requires a string action_id")
            self._pending_action_id = action_id
            state["requests"] += 1
            self.act(self._env_id, action=action)
        return state

    def on_status(
        self,
        state: ActuatorState,
        env_id: str,
        **kwargs: Any,
    ) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
            if env_id != self._env_id or self._pending_action_id is None:
                raise RuntimeError("Received an unexpected environment action status")
            result = {
                "action_id": self._pending_action_id,
                K_REWARD: float(kwargs[K_REWARD]),
                K_DONE: kwargs.get(K_DONE),
                K_DEAD: kwargs.get(K_DEAD),
                K_ILLEGAL_ACTION: kwargs.get(K_ILLEGAL_ACTION),
                K_NEW_ACHIEVEMENTS: kwargs.get(K_NEW_ACHIEVEMENTS),
            }
            if any(
                type(result[key]) is not bool
                for key in (K_DONE, K_DEAD, K_ILLEGAL_ACTION)
            ):
                raise ValueError("Crafter action status contains a non-Boolean outcome")
            achievements = result[K_NEW_ACHIEVEMENTS]
            if not isinstance(achievements, list) or not all(
                isinstance(item, str) for item in achievements
            ):
                raise ValueError("Crafter action status has invalid achievements")
            self._pending_action_id = None
            state["statuses"] += 1
            state.outbox.send_status(self._ll_id, ActionStatus(result))
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        with _active_time(state):
            if self._env_id:
                self.act(self._env_id, action=A_CLOSE)
        return state


class SupportLLReasoner(LLReasonerBase):
    """Extract beliefs and maintain the action-result/observation loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._perceptor_id = ""
        self._knowledge_id = ""
        self._actuator_id = ""
        self._pending_result: dict[str, Any] | None = None

    def _resolve_ids(self, state: LLState) -> None:
        if not self._perceptor_id:
            self._perceptor_id = _require_one(
                state.directory.internal.perception, "perceptor"
            ).module_id
        if not self._knowledge_id:
            self._knowledge_id = _require_one(
                state.directory.internal.knowledge, "knowledge module"
            ).module_id
        if not self._actuator_id:
            self._actuator_id = _require_one(
                state.directory.internal.actuation, "actuator"
            ).module_id

    def _request_observation(self, state: LLState) -> None:
        state["observation_requests"] += 1
        state.outbox.request_observation(self._perceptor_id)

    def on_first(self, state: LLState) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            self._request_observation(state)
        return state

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._actuator_id:
                raise RuntimeError(
                    f"Action status came from unexpected module {sender!r}"
                )
            if self._pending_result is not None:
                raise RuntimeError("An action result is already awaiting observation")
            result = action_status.status
            if not isinstance(result, dict) or not isinstance(
                result.get("action_id"), str
            ):
                raise ValueError("Malformed or uncorrelated action status")
            self._pending_result = dict(result)
            state["action_statuses"] += 1
            self._request_observation(state)
        return state

    def on_observation(
        self,
        state: LLState,
        sender: str,
        observation: Observation,
        **kwargs: Any,
    ) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._perceptor_id:
                raise RuntimeError(
                    f"Observation came from unexpected module {sender!r}"
                )
            try:
                beliefs = percept_to_beliefs(
                    parse_symbolic_observation(observation.content)
                )
            except (ObservationError, TypeError, ValueError) as exc:
                state["failure"] = f"{type(exc).__name__}: {exc}"
                raise
            state["observations_received"] += 1
            observation_seq = state["observations_received"]
            state["belief_updates_sent"] += 1
            state.outbox.send_beliefs(
                self._knowledge_id,
                observation,
                beliefs,
                observation_seq=observation_seq,
                action_result=self._pending_result,
            )
            self._pending_result = None
        return state


class ForwardingKnowledge(KnowledgeBase):
    """Forward normalized beliefs without duplicating their payload."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ll_id = ""
        self._hl_id = ""

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        with _active_time(state):
            self._ll_id = _require_one(
                state.directory.internal.ll_reasoning, "low-level reasoner"
            ).module_id
            self._hl_id = _require_one(
                state.directory.internal.hl_reasoning, "high-level reasoner"
            ).module_id
        return state

    def on_observed_beliefs(
        self,
        state: KnowledgeState,
        sender: str,
        observation: Observation,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> KnowledgeState:
        with _active_time(state):
            if sender != self._ll_id:
                raise RuntimeError(f"Beliefs came from unexpected module {sender!r}")
            state["revisions"] += 1
            state["updates_forwarded"] += 1
            state.outbox.send_beliefs(
                self._hl_id,
                beliefs,
                observation_seq=kwargs.get("observation_seq"),
                action_result=kwargs.get("action_result"),
            )
        return state


class CrafterBDIReasoner(HLReasonerBase):
    """Own full-domain deliberation, plan monitoring, and atomic action choice."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random()
        self._planning: PlanningService | None = None
        self._actuator_id = ""
        self._knowledge_id = ""

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.Random(kwargs.get("seed"))
        timeout = float(kwargs.get("planner_timeout", PLANNER_TIMEOUT_SECONDS))
        self._planning = PlanningService(
            Path(__file__).with_name("crafter-domain.pddl"), timeout
        )

    def on_first(self, state: HLState) -> HLState:
        with _active_time(state):
            self._actuator_id = _require_one(
                state.directory.internal.actuation, "actuator"
            ).module_id
            self._knowledge_id = _require_one(
                state.directory.internal.knowledge, "knowledge module"
            ).module_id
            state["run"]["phase"] = "awaiting-beliefs"
        return state

    @staticmethod
    def _execution_time(state: HLState) -> float | None:
        elapsed = getattr(state, "time", None)
        return None if elapsed is None else float(elapsed)

    @classmethod
    def _duration_complete(cls, state: HLState) -> bool:
        elapsed = cls._execution_time(state)
        return elapsed is not None and elapsed >= (
            DURATION - PLANNER_TIMEOUT_SECONDS - SHUTDOWN_MARGIN_SECONDS
        )

    @staticmethod
    def _reset_planning_failures(run: dict[str, Any]) -> None:
        run["planning_failure_signature"] = None
        run["planning_failure_count"] = 0
        run["forced_exploration_remaining"] = 0

    def _terminate(
        self,
        state: HLState,
        row: dict[str, Any],
        reason: str,
        *,
        operational_error: str | None = None,
    ) -> None:
        run = state["run"]
        run["phase"] = (
            "experiment-error"
            if operational_error is not None
            else "complete"
            if reason == "diamond_obtained"
            else "scientific-terminal"
        )
        run["terminal_reason"] = reason
        run["failure"] = operational_error
        run["active_plan"] = None
        run["pending_action"] = None
        run["intention"] = None
        run["explore_reason"] = None
        self._reset_planning_failures(run)
        row["primary_goal"]["status"] = (
            "complete" if reason == "diamond_obtained" else "right-censored"
        )
        row["decision"] = {
            "kind": "experiment-error"
            if operational_error is not None
            else "diamond-obtained"
            if reason == "diamond_obtained"
            else "scientific-terminal",
            "reason": reason,
        }
        if operational_error is not None:
            self.log(ERROR, operational_error)
        state.outbox.terminate_agent(f"2-3-CR terminal: {reason}")

    def _fail(
        self,
        state: HLState,
        row: dict[str, Any],
        reason: str,
    ) -> None:
        self._terminate(
            state,
            row,
            "experiment_error",
            operational_error=reason,
        )

    @staticmethod
    def _grounded(data: Mapping[str, Any]) -> GroundedAction:
        return GroundedAction(
            name=str(data["name"]),
            arguments=tuple(data["arguments"]),
            native_action=int(data["native_action"]),
            movement_kind=cast(Any, data["movement_kind"]),
            source_cell=data.get("source_cell"),
            destination_cell=data.get("destination_cell"),
            recovery_attempt=bool(data.get("recovery_attempt")),
            cow_cell=data.get("cow_cell"),
        )

    @staticmethod
    def _plan_record(
        run: Mapping[str, Any],
        active: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        row = run["trace"][int(active["trace_index"])]
        planning = row.get("planning")
        if not isinstance(planning, Mapping):
            raise RuntimeError("Active plan does not resolve to a planning record")
        return planning

    @classmethod
    def _plan_actions(
        cls,
        run: Mapping[str, Any],
        active: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        actions = cls._plan_record(run, active).get("actions")
        if not isinstance(actions, list):
            raise RuntimeError("Active plan does not resolve to serialized actions")
        return actions

    @staticmethod
    def _exploration_signature(
        belief_state: Mapping[str, Any],
        supported: bool,
    ) -> str:
        terrain = belief_state.get("terrain", {})
        payload = {
            "known": len(belief_state.get("known", ())),
            "reachable": len(belief_state.get("reachable", ())),
            "resources": {
                material: sum(value == material for value in terrain.values())
                for material in (
                    "water",
                    "tree",
                    "stone",
                    "coal",
                    "iron",
                    "diamond",
                    "table",
                    "furnace",
                )
            },
            "supported": supported,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _new_intention(
        self,
        run: dict[str, Any],
        value: Mapping[str, Any],
        selected_revision: int,
    ) -> dict[str, Any]:
        intention = dict(value)
        intention["id"] = f"intention-{run['next_intention_seq']}"
        intention.setdefault("selected_revision", selected_revision)
        run["next_intention_seq"] += 1
        return intention

    def _select_intention(
        self,
        run: dict[str, Any],
        belief_state: Mapping[str, Any],
        stage: TechnologyStage,
        row: dict[str, Any],
    ) -> tuple[dict[str, Any], bool, str | None]:
        current = run["intention"]
        current_recovery = (
            current
            if isinstance(current, Mapping) and current.get("kind") == "recovery"
            else None
        )
        completed_recovery = bool(
            current_recovery is not None
            and recovery_complete(current_recovery, belief_state)
        )
        recovery = select_recovery_intention(belief_state, current_recovery)
        changed = False
        if recovery is not None:
            if (
                current_recovery is not None
                and recovery.get("need") == current_recovery.get("need")
            ):
                intention = dict(current_recovery)
                row["intentions"].append(
                    {
                        "kind": "retained",
                        "id": intention["id"],
                        "name": intention["name"],
                    }
                )
            else:
                if isinstance(current, Mapping):
                    row["intentions"].append(
                        {
                            "kind": (
                                "completed"
                                if completed_recovery
                                else "preempted"
                            ),
                            "id": current["id"],
                            "name": current["name"],
                            "by": recovery["name"],
                        }
                    )
                recovery["suspended_name"] = stage.name
                intention = self._new_intention(
                    run, recovery, int(row["revision"])
                )
                row["intentions"].append(
                    {
                        "kind": "selected",
                        "id": intention["id"],
                        "name": intention["name"],
                    }
                )
                changed = True
        else:
            stage_value = {**stage.as_dict(), "kind": "technology"}
            if (
                isinstance(current, Mapping)
                and current.get("kind") == "technology"
                and current.get("name") == stage.name
            ):
                intention = {**current, **stage_value}
                row["intentions"].append(
                    {
                        "kind": "retained",
                        "id": intention["id"],
                        "name": intention["name"],
                    }
                )
            else:
                resumed = (
                    isinstance(current, Mapping)
                    and current.get("kind") == "recovery"
                )
                if isinstance(current, Mapping):
                    row["intentions"].append(
                        {
                            "kind": "completed",
                            "id": current["id"],
                            "name": current["name"],
                        }
                    )
                intention = self._new_intention(
                    run, stage_value, int(row["revision"])
                )
                row["intentions"].append(
                    {
                        "kind": "resumed" if resumed else "selected",
                        "id": intention["id"],
                        "name": intention["name"],
                    }
                )
                changed = True
        if changed:
            active = run.get("active_plan")
            if isinstance(active, Mapping):
                row["monitoring"]["plan_invalidated"] = {
                    "plan_id": active["id"],
                    "reason": "intention-changed",
                }
            run["active_plan"] = None
            run["explore_reason"] = None
            self._reset_planning_failures(run)
        if intention["kind"] == "recovery":
            supported, reason = recovery_supported(intention, belief_state)
        else:
            supported, reason = stage_supported(stage, belief_state)
        intention["supported"] = supported
        intention["support_reason"] = reason
        run["intention"] = intention
        return intention, supported, reason

    def _dispatch(
        self,
        state: HLState,
        row: dict[str, Any],
        action: GroundedAction,
        provenance: Literal["pddl", "recovery", "explore"],
        intention: Mapping[str, Any],
        *,
        reason: str | None = None,
        forced_exploration: bool = False,
        support_value: str | None = None,
        exploration_signature: str | None = None,
    ) -> None:
        run = state["run"]
        active = run["active_plan"] if provenance == "pddl" else None
        action_id = f"action-{run['next_action_seq']}"
        run["next_action_seq"] += 1
        pending = {
            "action": action.native_action,
            "action_id": action_id,
            "operator": {
                "name": action.name,
                "arguments": list(action.arguments),
            },
            "movement_kind": action.movement_kind,
            "source_cell": action.source_cell,
            "destination_cell": action.destination_cell,
            "recovery_attempt": action.recovery_attempt,
            "provenance": provenance,
            "intention_id": intention["id"],
            "plan_id": active.get("id") if active else None,
            "plan_index": active.get("next_index") if active else None,
            "explore_reason": reason,
            "forced_exploration": forced_exploration,
            "support_signature": support_value,
            "exploration_signature": exploration_signature,
        }
        if active is not None:
            active["next_index"] += 1
        run["pending_action"] = pending
        run["phase"] = "running"
        kind = (
            "planned-action"
            if provenance == "pddl"
            else "recovery-action"
            if provenance == "recovery"
            else "explore-action"
        )
        row["decision"] = {
            "kind": kind,
            "action_id": action_id,
            "native_action": action.native_action,
            "operator": pending["operator"],
            "plan_id": pending["plan_id"],
            "plan_index": pending["plan_index"],
            "intention_id": pending["intention_id"],
            "sound": True,
            "provenance": provenance,
            "reason": reason,
            "movement_kind": action.movement_kind,
            "destination_cell": action.destination_cell,
        }
        state.outbox.request_action(
            self._actuator_id,
            action=action.native_action,
            action_id=action_id,
        )

    def _dispatch_explore(
        self,
        state: HLState,
        row: dict[str, Any],
        intention: Mapping[str, Any],
        *,
        reason: Literal["missing-knowledge", "planning-failure"],
        forced: bool,
    ) -> None:
        run = state["run"]
        belief_state = state["belief_state"]
        action = choose_exploration_action(belief_state, self._rng)
        sound, unsound_reason = action_sound(action, belief_state)
        if not sound:
            player = cell_key(tuple(belief_state["player"]))
            action = GroundedAction(
                "explore-noop",
                (),
                int(CrafterAction.NOOP),
                "none",
                player,
                None,
                False,
                None,
            )
            row["monitoring"]["exploration_action_replaced"] = unsound_reason
        current_signature = support_signature(
            belief_state, str(intention["name"])
        )
        progress_signature = self._exploration_signature(
            belief_state, bool(intention["supported"])
        )
        if run["explore_reason"] != reason:
            row["intentions"].append(
                {
                    "kind": "explore-selected",
                    "id": intention["id"],
                    "name": intention["name"],
                    "reason": reason,
                }
            )
        run["explore_reason"] = reason
        run["last_exploration_signature"] = progress_signature
        self._dispatch(
            state,
            row,
            action,
            "explore",
            intention,
            reason=reason,
            forced_exploration=forced,
            support_value=current_signature,
            exploration_signature=progress_signature,
        )

    def _dispatch_sleep_recovery(
        self,
        state: HLState,
        row: dict[str, Any],
        intention: Mapping[str, Any],
    ) -> None:
        """Dispatch the only legal sleep action for the current recovery."""

        belief_state = state["belief_state"]
        player = cell_key(tuple(belief_state["player"]))
        sleeping = bool(belief_state.get("sleeping"))
        if not sleeping and intention.get("need") != "energy":
            raise RuntimeError("Only energy recovery may initiate sleep")
        action = GroundedAction(
            "recovery-noop" if sleeping else "recovery-sleep",
            (),
            int(CrafterAction.NOOP if sleeping else CrafterAction.SLEEP),
            "none",
            player,
            None,
            False,
            None,
        )
        sound, reason = action_sound(action, belief_state)
        if not sound:
            raise RuntimeError(f"Unsound energy recovery action: {reason}")
        self._dispatch(
            state,
            row,
            action,
            "recovery",
            intention,
            reason=str(intention["name"]),
        )

    def _append_row(self, state: HLState, row: dict[str, Any]) -> None:
        run = state["run"]
        elapsed = self._execution_time(state)
        if elapsed is not None:
            run["elapsed_seconds"] = elapsed
        row["elapsed_seconds"] = float(run["elapsed_seconds"])
        run["trace"].append(row)
        decision = row.get("decision") or {}
        intention = run.get("intention")
        self.log(
            INFO,
            "BDI_REVISION "
            f"revision={row['revision']} observation_seq={row['observation_seq']} "
            f"stage={row['stage'].get('derived') or 'none'} "
            f"intention={intention.get('name') if isinstance(intention, Mapping) else 'none'} "
            f"decision={decision.get('kind', 'none')}",
        )

    def _consume_action(
        self,
        state: HLState,
        row: dict[str, Any],
        action_result: Mapping[str, Any] | None,
        consumed: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        run = state["run"]
        if not consumed:
            return None, None
        pending = cast(dict[str, Any], run["pending_action"])
        result = cast(Mapping[str, Any], action_result)
        run["agent_steps"] += 1
        row["monitoring"]["action_result"] = {
            "action_id": result["action_id"],
            "reward": float(result.get(K_REWARD, 0.0)),
            "done": bool(result.get(K_DONE)),
            "dead": bool(result.get(K_DEAD)),
            "illegal_action": bool(result.get(K_ILLEGAL_ACTION)),
            "new_achievements": list(result.get(K_NEW_ACHIEVEMENTS, ())),
            "provenance": pending["provenance"],
            "plan_id": pending["plan_id"],
            "plan_index": pending["plan_index"],
        }
        completed_plan: dict[str, Any] | None = None
        active = run.get("active_plan")
        if (
            pending.get("provenance") == "pddl"
            and isinstance(active, Mapping)
            and pending.get("plan_id") == active.get("id")
        ):
            actions = self._plan_actions(run, active)
            if active["next_index"] == active["chunk_end_index"]:
                if active["next_index"] < len(actions):
                    row["monitoring"]["plan_chunk"] = {
                        "plan_id": active["id"],
                        "kind": "chunk-complete",
                        "dispatched": active["next_index"],
                    }
                else:
                    planning = self._plan_record(run, active)
                    expected = planning.get("expected_goal")
                    if not isinstance(expected, Mapping):
                        raise RuntimeError("Accepted plan has no expected goal")
                    observed = plan_goal_observed(
                        expected, state["belief_state"], result
                    )
                    row["monitoring"]["plan_goal"] = {
                        "plan_id": active["id"],
                        "status": (
                            "plan-goal-complete"
                            if observed
                            else "plan-exhausted-without-goal-progress"
                        ),
                    }
                    completed_plan = {
                        "intention_id": active["intention_id"],
                        "intention_name": active["intention_name"],
                        "observed": observed,
                    }
                run["active_plan"] = None

        if pending.get("provenance") == "explore":
            current = self._exploration_signature(
                state["belief_state"],
                bool(run.get("intention", {}).get("supported"))
                if isinstance(run.get("intention"), Mapping)
                else False,
            )
            if current == pending.get("exploration_signature"):
                run["exploration_no_progress"] += 1
            else:
                run["exploration_no_progress"] = 0
            run["last_exploration_signature"] = current
            if pending.get("forced_exploration"):
                run["forced_exploration_remaining"] = max(
                    0, int(run["forced_exploration_remaining"]) - 1
                )
        run["pending_action"] = None
        return pending, completed_plan

    def _planning_rejected(
        self,
        state: HLState,
        row: dict[str, Any],
        intention: Mapping[str, Any],
        rejection: object,
    ) -> None:
        run = state["run"]
        signature = support_signature(state["belief_state"], str(intention["name"]))
        if run["planning_failure_signature"] != signature:
            run["planning_failure_signature"] = signature
            run["planning_failure_count"] = 0
        run["planning_failure_count"] += 1
        forced = run["planning_failure_count"] >= PLANNING_RETRY_LIMIT
        if forced:
            run["forced_exploration_remaining"] = EXPLORATION_BURST_SIZE
        self.log(WARNING, f"LPG failed for {intention['name']}: {rejection}")
        self._dispatch_explore(
            state,
            row,
            intention,
            reason="planning-failure",
            forced=forced,
        )

    def on_belief_update(
        self,
        state: HLState,
        sender: str,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> HLState:
        """Revise beliefs and make exactly one reactive controller decision."""

        with _active_time(state):
            run = state["run"]
            if run["phase"] in FINAL_PHASES:
                return state
            if sender != self._knowledge_id:
                raise RuntimeError(f"Beliefs came from unexpected module {sender!r}")
            state["belief_updates"] += 1
            observation_seq = kwargs.get("observation_seq")
            if not isinstance(observation_seq, int):
                observation_seq = state["belief_updates"]
            action_result = kwargs.get("action_result")
            if action_result is not None and not isinstance(action_result, Mapping):
                action_result = {"invalid": repr(action_result)}
            previous_stage = run.get("current_stage")
            row: dict[str, Any] = {
                "revision": state["belief_updates"],
                "observation_seq": observation_seq,
                "belief": {},
                "primary_goal": {"name": "obtain-diamond", "status": "active"},
                "stage": {
                    "previous": (
                        previous_stage.get("name")
                        if isinstance(previous_stage, Mapping)
                        else None
                    ),
                    "derived": None,
                    "target_count": None,
                    "support": None,
                    "transition": None,
                },
                "intentions": [],
                "planning": None,
                "monitoring": {},
                "decision": None,
            }
            try:
                revision = revise_belief_state(
                    state["belief_state"],
                    beliefs_to_percept(beliefs),
                    action_result=cast(Mapping[str, Any] | None, action_result),
                    pending_action=run["pending_action"],
                )
                state["belief_state"] = revision.belief_state
                belief_state = state["belief_state"]
                row["revision"] = belief_state["revision"]
                row["belief"] = summarize_belief_state(
                    belief_state,
                    cow_target_contradicted=revision.cow_target_contradicted,
                )
                pending, completed_plan = self._consume_action(
                    state,
                    row,
                    cast(Mapping[str, Any] | None, action_result),
                    revision.consumed_action,
                )
            except (
                ObservationError,
                BeliefRevisionError,
                TypeError,
                ValueError,
                KeyError,
                RuntimeError,
            ) as exc:
                self._fail(
                    state,
                    row,
                    f"Belief/monitoring failure: {type(exc).__name__}: {exc}",
                )
                self._append_row(state, row)
                return state

            elapsed = self._execution_time(state)
            if elapsed is not None:
                run["elapsed_seconds"] = elapsed
            run["highest_milestone"] = highest_milestone(
                belief_state.get("achievements", ())
            )

            if (
                int(belief_state.get("inventory", {}).get("diamond", 0)) >= 1
                or "collect_diamond" in belief_state.get("achievements", ())
            ):
                self._terminate(state, row, "diamond_obtained")
                self._append_row(state, row)
                return state
            if belief_state.get("dead"):
                self._terminate(state, row, "death")
                self._append_row(state, row)
                return state
            if belief_state.get("terminal"):
                self._terminate(state, row, "episode_limit")
                self._append_row(state, row)
                return state
            if run["agent_steps"] >= MAX_AGENT_STEPS:
                self._terminate(state, row, "action_budget_exhausted")
                self._append_row(state, row)
                return state
            if self._duration_complete(state):
                self._terminate(state, row, "time_budget_exhausted")
                self._append_row(state, row)
                return state
            if run["exploration_no_progress"] >= EXPLORATION_STALL_LIMIT:
                self._terminate(state, row, "exploration_stalled")
                self._append_row(state, row)
                return state

            stage = derive_technology_stage(belief_state)
            if stage is None:
                self._fail(
                    state,
                    row,
                    "Diamond stage resolved without public diamond evidence",
                )
                self._append_row(state, row)
                return state
            technology_supported, technology_reason = stage_supported(
                stage, belief_state
            )
            run["current_stage"] = stage.as_dict()
            row["stage"].update(
                derived=stage.name,
                target_count=stage.target_count,
                support={
                    "supported": technology_supported,
                    "reason": technology_reason,
                },
                transition=(
                    "retained"
                    if row["stage"]["previous"] == stage.name
                    else "derived"
                ),
            )
            intention, supported, support_reason = self._select_intention(
                run, belief_state, stage, row
            )

            if completed_plan is not None:
                if completed_plan["observed"]:
                    same = (
                        intention["id"] == completed_plan["intention_id"]
                        and intention["name"] == completed_plan["intention_name"]
                    )
                    row["monitoring"]["plan_goal"]["progress"] = (
                        "unit-progress" if same else "intention-complete"
                    )
                else:
                    row["monitoring"]["plan_goal"]["progress"] = "none"

            if (
                intention["kind"] == "recovery"
                and belief_state.get("sleeping")
            ):
                self._dispatch_sleep_recovery(state, row, intention)
                self._append_row(state, row)
                return state

            if isinstance(pending, Mapping) and pending.get("provenance") == "explore":
                pending_reason = pending.get("explore_reason")
                current_signature = support_signature(
                    belief_state, str(intention["name"])
                )
                if pending_reason == "missing-knowledge":
                    if not supported:
                        self._dispatch_explore(
                            state,
                            row,
                            intention,
                            reason="missing-knowledge",
                            forced=False,
                        )
                        self._append_row(state, row)
                        return state
                    run["explore_reason"] = None
                elif pending_reason == "planning-failure":
                    if current_signature != run["planning_failure_signature"]:
                        self._reset_planning_failures(run)
                        run["explore_reason"] = None
                    elif (
                        pending.get("forced_exploration")
                        and run["forced_exploration_remaining"] > 0
                    ):
                        self._dispatch_explore(
                            state,
                            row,
                            intention,
                            reason="planning-failure",
                            forced=True,
                        )
                        self._append_row(state, row)
                        return state
                    else:
                        run["explore_reason"] = None

            active = run.get("active_plan")
            if isinstance(active, Mapping):
                if active.get("intention_id") != intention["id"]:
                    row["monitoring"]["plan_invalidated"] = {
                        "plan_id": active["id"],
                        "reason": "intention-changed",
                    }
                    run["active_plan"] = None
                else:
                    actions = self._plan_actions(run, active)
                    if (
                        active["next_index"] < active["chunk_end_index"]
                        and active["next_index"] < len(actions)
                    ):
                        next_action = self._grounded(actions[active["next_index"]])
                        sound, reason = action_sound(next_action, belief_state)
                        if sound:
                            row["monitoring"]["plan_status"] = {
                                "kind": "retained",
                                "plan_id": active["id"],
                            }
                            self._dispatch(
                                state,
                                row,
                                next_action,
                                "pddl",
                                intention,
                            )
                            self._append_row(state, row)
                            return state
                        row["monitoring"]["plan_invalidated"] = {
                            "plan_id": active["id"],
                            "reason": reason,
                        }
                        run["active_plan"] = None

            if not supported:
                row["planning"] = {
                    "intention_id": intention["id"],
                    "stage": intention["name"],
                    "classification": "missing-knowledge",
                    "accepted": False,
                    "rejection": support_reason,
                }
                self._dispatch_explore(
                    state,
                    row,
                    intention,
                    reason="missing-knowledge",
                    forced=False,
                )
                self._append_row(state, row)
                return state

            if intention["kind"] == "recovery" and intention["need"] == "energy":
                self._dispatch_sleep_recovery(state, row, intention)
                self._append_row(state, row)
                return state
            if self._planning is None:
                self._fail(state, row, "Planning service was not initialized")
                self._append_row(state, row)
                return state

            problem_name = (
                f"crafter-{run['next_plan_seq']}-r{belief_state['revision']}"
            )
            outcome: PlanningOutcome = self._planning.solve(
                belief_state,
                intention,
                problem_name,
                MAX_AGENT_STEPS - run["agent_steps"],
            )
            planning = dict(outcome.attempt)
            planning["intention_id"] = intention["id"]
            planning["problem_name"] = outcome.problem_name
            planning["selected_cow"] = outcome.selected_cow
            planning["remaining_steps_at_acceptance"] = (
                MAX_AGENT_STEPS - run["agent_steps"]
            )
            row["planning"] = planning
            classification = outcome.attempt.get("classification")
            if classification in {
                "model-error",
                "malformed-action",
                "planner-exception",
            }:
                self._fail(
                    state,
                    row,
                    "Planning failure: "
                    f"{outcome.attempt.get('error') or classification}",
                )
                self._append_row(state, row)
                return state
            if outcome.accepted:
                plan_id = f"plan-{run['next_plan_seq']}"
                run["next_plan_seq"] += 1
                actions = [action.as_dict() for action in outcome.actions]
                planning.update(
                    plan_id=plan_id,
                    actions=actions,
                    expected_goal=outcome.expected_goal,
                )
                if outcome.selected_cow is not None:
                    target = belief_state.get("cow_target")
                    same = (
                        isinstance(target, Mapping)
                        and target.get("cell") == outcome.selected_cow
                    )
                    belief_state["cow_target"] = {
                        "cell": outcome.selected_cow,
                        "hits": int(target.get("hits", 0)) if same else 0,
                    }
                run["active_plan"] = {
                    "id": plan_id,
                    "trace_index": len(run["trace"]),
                    "intention_id": intention["id"],
                    "intention_name": intention["name"],
                    "next_index": 0,
                    "chunk_end_index": min(
                        len(outcome.actions), PLAN_CHUNK_SIZE
                    ),
                }
                self._reset_planning_failures(run)
                run["explore_reason"] = None
                first = outcome.actions[0]
                sound, reason = action_sound(first, belief_state)
                if sound:
                    self._dispatch(
                        state, row, first, "pddl", intention
                    )
                else:
                    row["monitoring"]["plan_invalidated"] = {
                        "plan_id": plan_id,
                        "reason": reason,
                    }
                    run["active_plan"] = None
                    self._planning_rejected(
                        state,
                        row,
                        intention,
                        f"unsound-first-action:{reason}",
                    )
                self._append_row(state, row)
                return state

            self._planning_rejected(
                state,
                row,
                intention,
                outcome.attempt.get("rejection"),
            )
            self._append_row(state, row)
        return state

    def on_last(self, state: HLState) -> HLState:
        with _active_time(state):
            self._planning = None
        return state


def _initial_states(
    treatment: dict[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return fresh JSON-native states for the five reactive modules."""

    selected_treatment = dict(treatment or treatment_for_run(0))
    return {
        PERCEPTOR: {
            "requests": 0,
            "observations": 0,
            "failure": None,
            "active_seconds": 0.0,
        },
        ACTUATOR: {
            "requests": 0,
            "statuses": 0,
            "failure": None,
            "active_seconds": 0.0,
        },
        LLREASONER: {
            "observation_requests": 0,
            "observations_received": 0,
            "belief_updates_sent": 0,
            "action_statuses": 0,
            "failure": None,
            "active_seconds": 0.0,
        },
        KNOWLEDGE: {
            "revisions": 0,
            "updates_forwarded": 0,
            "failure": None,
            "active_seconds": 0.0,
        },
        HLREASONER: {
            "belief_updates": 0,
            "belief_state": initial_belief_state(),
            "failure": None,
            "active_seconds": 0.0,
            "run": {
                "treatment": selected_treatment,
                "phase": "created",
                "primary_goal": "obtain-diamond",
                "current_stage": None,
                "intention": None,
                "active_plan": None,
                "pending_action": None,
                "next_intention_seq": 1,
                "next_plan_seq": 1,
                "next_action_seq": 1,
                "agent_steps": 0,
                "elapsed_seconds": 0.0,
                "planning_failure_signature": None,
                "planning_failure_count": 0,
                "explore_reason": None,
                "forced_exploration_remaining": 0,
                "exploration_no_progress": 0,
                "last_exploration_signature": None,
                "failure": None,
                "terminal_reason": None,
                "highest_milestone": None,
                "trace": [],
            },
        },
    }


def deliberative_initial_states() -> dict[str, dict[str, Any]]:
    """Expose fresh initial states for focused host tests."""

    return _initial_states()


def _environment_initial_state() -> dict[str, Any]:
    """Return the public environment evidence accumulated during one episode."""

    return {
        "observation_requests": 0,
        "post_terminal_observations": 0,
        "native_actions": 0,
        "illegal_actions": 0,
        "terminal_events": 0,
        "close_requests": 0,
        "inventory": {},
        "achievement_counts": {},
        "achievements": [],
        "terminal": False,
        "dead": False,
        "closed": False,
        "video_path": None,
    }


def _runtime_failure(line: str) -> bool:
    """Recognize fatal records, including MHAgentA warning-level failures."""

    match = LOG_PATTERN.match(line)
    level = match.group("level").lower() if match else ""
    message = match.group("message").lower() if match else line.lower()
    if level in {"error", "critical"}:
        return True
    return any(
        marker in message
        for marker in (
            "traceback",
            "exceptiongroup",
            "caught exception",
            "failed to save state",
            "could not send message",
        )
    )


def _result_errors(
    agent_states: Mapping[str, Mapping[str, Any]] | None,
    environment_state: Mapping[str, Any] | None,
    agent_logs: Sequence[str],
    environment_logs: Sequence[str] = (),
) -> list[str]:
    """Return operational-contract violations without requiring task success."""

    errors: list[str] = []
    if not agent_logs:
        errors.append("configured agent log is missing or empty")
    for line in (*agent_logs, *environment_logs):
        if _runtime_failure(line):
            errors.append(f"runtime failure in logs: {line.rstrip()}")
            break
    if not isinstance(agent_states, Mapping):
        return [*errors, "agent module states are missing"]
    if not isinstance(environment_state, Mapping):
        return [*errors, "environment state is missing"]

    expected = {
        module_name(kind, 0): kind
        for kind in (PERCEPTOR, ACTUATOR, LLREASONER, KNOWLEDGE, HLREASONER)
    }
    if set(agent_states) != set(expected):
        errors.append(
            f"module state set differs: expected={sorted(expected)} "
            f"actual={sorted(agent_states)}"
        )
        if not set(expected) <= set(agent_states):
            return errors
    modules = {kind: agent_states[name] for name, kind in expected.items()}
    try:
        json.dumps([*modules.values(), environment_state])
    except (TypeError, ValueError) as exc:
        errors.append(f"saved state is not JSON-serializable: {exc}")
    for kind, module in modules.items():
        active = module.get("active_seconds") if isinstance(module, Mapping) else None
        if not isinstance(active, (int, float)) or active <= 0:
            errors.append(f"{kind} did not record positive active time")
        if isinstance(module, Mapping) and module.get("failure") is not None:
            errors.append(f"{kind} failed: {module.get('failure')}")
    if not all(isinstance(module, Mapping) for module in modules.values()):
        return errors

    perceptor = modules[PERCEPTOR]
    actuator = modules[ACTUATOR]
    ll = modules[LLREASONER]
    knowledge = modules[KNOWLEDGE]
    hl = modules[HLREASONER]
    run = hl.get("run")
    belief_state = hl.get("belief_state")
    if not isinstance(run, Mapping) or not isinstance(belief_state, Mapping):
        return [*errors, "HL run or belief state is missing"]
    if run.get("failure") is not None:
        errors.append(f"HL run failed: {run.get('failure')}")
    if run.get("phase") not in FINAL_PHASES:
        errors.append(f"HL did not save a final phase: {run.get('phase')!r}")
    if run.get("terminal_reason") not in {
        "diamond_obtained",
        "death",
        "episode_limit",
        "action_budget_exhausted",
        "time_budget_exhausted",
        "exploration_stalled",
        "experiment_error",
    }:
        errors.append(f"invalid terminal reason: {run.get('terminal_reason')!r}")
    if run.get("pending_action") is not None:
        errors.append("HL retained a pending action")
    if environment_state.get("closed") is not True or environment_state.get(
        "close_requests"
    ) != 1:
        errors.append("environment did not close exactly once")
    if environment_state.get("illegal_actions") != 0:
        errors.append("environment recorded an illegal action")
    if environment_state.get("treatment") != run.get("treatment"):
        errors.append("HL and environment treatments differ")

    trace_value = run.get("trace")
    trace = trace_value if isinstance(trace_value, list) else []
    if trace_value is not trace:
        errors.append("HL trace is missing or malformed")
    observation_counts = [
        ll.get("observation_requests"),
        perceptor.get("requests"),
        environment_state.get("observation_requests"),
        perceptor.get("observations"),
        ll.get("observations_received"),
        ll.get("belief_updates_sent"),
        knowledge.get("revisions"),
        knowledge.get("updates_forwarded"),
        hl.get("belief_updates"),
    ]
    if any(type(value) is not int for value in observation_counts):
        errors.append("observation chain contains non-integer counters")
    elif len(set(observation_counts)) != 1 or cast(
        int, observation_counts[0]
    ) <= 0:
        errors.append(f"observation chain is incomplete: {observation_counts}")

    dispatches = [
        row.get("decision")
        for row in trace
        if isinstance(row, Mapping)
        and isinstance(row.get("decision"), Mapping)
        and row["decision"].get("kind")
        in {"planned-action", "recovery-action", "explore-action"}
    ]
    action_counts = [
        len(dispatches),
        run.get("agent_steps"),
        actuator.get("requests"),
        environment_state.get("native_actions"),
        actuator.get("statuses"),
        ll.get("action_statuses"),
    ]
    if any(type(value) is not int for value in action_counts):
        errors.append("action chain contains non-integer counters")
    elif len(set(action_counts)) != 1:
        errors.append(f"action chain is incomplete: {action_counts}")
    if (
        type(observation_counts[0]) is int
        and type(action_counts[0]) is int
        and observation_counts[0] != action_counts[0] + 1
    ):
        errors.append("observation count is not exactly action count plus one")
    if len(trace) != hl.get("belief_updates"):
        errors.append("trace does not contain one row per belief update")

    plans: dict[str, list[Mapping[str, Any]]] = {}
    plan_next: dict[str, int] = {}
    plan_dispatches: dict[str, int] = {}
    ids: dict[str, set[str]] = {
        "action": set(),
        "plan": set(),
        "intention": set(),
    }
    pending: Mapping[str, Any] | None = None
    revisions: list[int] = []
    sequences: list[int] = []
    stopped = False
    for index, row in enumerate(trace):
        if not isinstance(row, Mapping):
            errors.append(f"trace row {index} is malformed")
            continue
        revision = row.get("revision")
        sequence = row.get("observation_seq")
        if type(revision) is int:
            revisions.append(revision)
        else:
            errors.append(f"trace row {index} has invalid revision")
        if type(sequence) is int:
            sequences.append(sequence)
        else:
            errors.append(f"trace row {index} has invalid observation sequence")
        for transition in row.get("intentions", ()):
            if not isinstance(transition, Mapping):
                errors.append(f"trace row {index} has malformed intention transition")
                continue
            intention_id = transition.get("id")
            if transition.get("kind") in {"selected", "resumed"}:
                if not isinstance(intention_id, str) or intention_id in ids["intention"]:
                    errors.append(f"trace row {index} has duplicate intention ID")
                else:
                    ids["intention"].add(intention_id)

        monitoring = row.get("monitoring")
        monitoring = monitoring if isinstance(monitoring, Mapping) else {}
        result = monitoring.get("action_result")
        if result is not None:
            if pending is None or not isinstance(result, Mapping):
                errors.append(f"trace row {index} has an uncorrelated action result")
            else:
                if result.get("action_id") != pending.get("action_id"):
                    errors.append(f"trace row {index} action result mismatch")
                if result.get("illegal_action") is not False:
                    errors.append(f"trace row {index} observed an illegal action")
                pending = None

        planning = row.get("planning")
        if isinstance(planning, Mapping) and planning.get("accepted") is True:
            plan_id = planning.get("plan_id")
            actions = planning.get("actions")
            expected_goal = planning.get("expected_goal")
            remaining = planning.get("remaining_steps_at_acceptance")
            if not isinstance(plan_id, str) or plan_id in ids["plan"]:
                errors.append(f"trace row {index} has a duplicate plan ID")
            elif not isinstance(actions, list) or not actions or not all(
                isinstance(action, Mapping) for action in actions
            ):
                errors.append(f"trace row {index} accepted an empty plan")
            else:
                ids["plan"].add(plan_id)
                plans[plan_id] = actions
                plan_next[plan_id] = 0
                plan_dispatches[plan_id] = 0
            if not isinstance(expected_goal, Mapping) or expected_goal.get(
                "kind"
            ) not in {
                "inventory-at-least",
                "station-established",
                "recovery-interaction",
            }:
                errors.append(f"trace row {index} has an invalid expected goal")
            if type(remaining) is not int or (
                isinstance(actions, list) and len(actions) > remaining
            ):
                errors.append(f"trace row {index} accepted a plan beyond budget")
            if (
                planning.get("classification") != "accepted"
                or planning.get("engine") != "lpg"
                or planning.get("status") not in {
                    "SOLVED_SATISFICING",
                    "SOLVED_OPTIMALLY",
                }
                or planning.get("validation_status") != "VALID"
            ):
                errors.append(f"trace row {index} accepted an invalid plan result")

        decision = row.get("decision")
        decision = decision if isinstance(decision, Mapping) else {}
        kind = decision.get("kind")
        if kind in {"planned-action", "recovery-action", "explore-action"}:
            if stopped or pending is not None:
                errors.append(f"trace row {index} dispatched out of sequence")
            action_id = decision.get("action_id")
            if not isinstance(action_id, str) or action_id in ids["action"]:
                errors.append(f"trace row {index} has a duplicate action ID")
            else:
                ids["action"].add(action_id)
            if decision.get("sound") is not True:
                errors.append(f"trace row {index} dispatched an unsound action")
            if kind == "planned-action":
                plan_id = decision.get("plan_id")
                plan_index = decision.get("plan_index")
                if (
                    not isinstance(plan_id, str)
                    or type(plan_index) is not int
                    or plan_id not in plans
                    or plan_index != plan_next.get(plan_id)
                    or plan_index >= len(plans[plan_id])
                ):
                    errors.append(
                        f"trace row {index} is not the next accepted-plan action"
                    )
                else:
                    expected_action = plans[plan_id][plan_index]
                    operator = decision.get("operator")
                    if (
                        not isinstance(operator, Mapping)
                        or operator.get("name") != expected_action.get("name")
                        or operator.get("arguments")
                        != expected_action.get("arguments")
                        or decision.get("native_action")
                        != expected_action.get("native_action")
                    ):
                        errors.append(
                            f"trace row {index} action differs from accepted plan"
                        )
                    plan_next[plan_id] += 1
                    plan_dispatches[plan_id] += 1
                    if plan_dispatches[plan_id] > PLAN_CHUNK_SIZE:
                        errors.append(
                            f"trace row {index} crossed the plan chunk boundary"
                        )
            pending = decision
        elif kind in {
            "diamond-obtained",
            "scientific-terminal",
            "experiment-error",
        }:
            stopped = True
    if revisions != list(range(1, len(trace) + 1)):
        errors.append("trace revisions are not unique and monotonic")
    if sequences != list(range(1, len(trace) + 1)):
        errors.append("observation sequences are not unique and monotonic")
    if pending is not None:
        errors.append("final trace dispatch has no observed result")

    reason = run.get("terminal_reason")
    inventories = {
        "HL belief": belief_state.get("inventory"),
        "environment": environment_state.get("inventory"),
    }
    diamond_counts: dict[str, int] = {}
    for owner, inventory in inventories.items():
        if not isinstance(inventory, Mapping):
            errors.append(f"{owner} final inventory is not a mapping")
            continue
        diamond = inventory.get("diamond")
        if type(diamond) is not int:
            errors.append(f"{owner} final diamond count is not an integer")
            continue
        diamond_counts[owner] = diamond

    achievements = {
        "HL belief": belief_state.get("achievements"),
        "environment": environment_state.get("achievements"),
    }
    achievement_sets: dict[str, set[str]] = {}
    for owner, value in achievements.items():
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, (Sequence, set, frozenset))
            or not all(isinstance(item, str) for item in value)
        ):
            errors.append(
                f"{owner} final achievements are not a sequence or set of strings"
            )
            continue
        achievement_sets[owner] = set(value)

    if len(diamond_counts) == 2 and len(achievement_sets) == 2:
        diamond_values = (
            diamond_counts["HL belief"] >= 1,
            "collect_diamond" in achievement_sets["HL belief"],
            diamond_counts["environment"] >= 1,
            "collect_diamond" in achievement_sets["environment"],
            reason == "diamond_obtained",
        )
        if any(diamond_values) and not all(diamond_values):
            errors.append(f"diamond evidence disagrees: {diamond_values}")
    if reason == "death" and environment_state.get("dead") is not True:
        errors.append("death reason lacks public environment evidence")
    if reason == "episode_limit" and (
        environment_state.get("terminal") is not True
        or environment_state.get("dead") is True
    ):
        errors.append("episode-limit reason lacks public environment evidence")
    return errors


def check_results(
    agent_states: Mapping[str, Mapping[str, Any]] | None,
    environment_state: Mapping[str, Any] | None,
    agent_logs: Sequence[str],
    environment_logs: Sequence[str] = (),
    *,
    verbose: bool = False,
) -> bool:
    """Apply the operational contract to compact saved evidence."""

    errors = _result_errors(
        agent_states, environment_state, agent_logs, environment_logs
    )
    for error in errors:
        print(f"Acceptance check failed: {error}")
    if verbose and isinstance(agent_states, Mapping):
        high = agent_states.get(module_name(HLREASONER, 0), {})
        run = high.get("run", {}) if isinstance(high, Mapping) else {}
        trace = run.get("trace", []) if isinstance(run, Mapping) else []
        accepted = sum(
            isinstance(row, Mapping)
            and isinstance(row.get("planning"), Mapping)
            and row["planning"].get("accepted") is True
            for row in trace
        )
        planned = sum(
            isinstance(row, Mapping)
            and isinstance(row.get("decision"), Mapping)
            and row["decision"].get("kind") == "planned-action"
            for row in trace
        )
        explored = sum(
            isinstance(row, Mapping)
            and isinstance(row.get("decision"), Mapping)
            and row["decision"].get("kind") == "explore-action"
            for row in trace
        )
        print(
            f"2-3-CR: phase={run.get('phase')} terminal={run.get('terminal_reason')} "
            f"revisions={len(trace)} accepted_plans={accepted} "
            f"planned_actions={planned} explore_actions={explored}"
        )
    return not errors


def _workspace_root() -> Path:
    for candidate in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (candidate, *candidate.parents):
            pyproject = parent / "pyproject.toml"
            if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(
                encoding="utf-8"
            ):
                return parent
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root")


def _read_log(path: Path) -> list[str]:
    return (
        path.read_text(encoding="utf-8").splitlines(keepends=True)
        if path.is_file()
        else []
    )


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Build and execute one deterministic full-domain 2-3-CR run."""

    version = (
        DEFAULT_MHAGENTA_VERSION
        if not mha_version or mha_version == "latest"
        else mha_version
    )
    if version != "1.4.12":
        raise RuntimeError(
            f"Experiment 2-3-CR requires MHAgentA 1.4.12, got {version!r}"
        )
    exp_path = Path(exp_path).resolve()
    workspace_root = _workspace_root()
    mha_root = (workspace_root.parent / "mhagenta").resolve()
    version_file = mha_root / "pyproject.toml"
    if (
        not version_file.is_file()
        or 'version = "1.4.12"'
        not in version_file.read_text(encoding="utf-8")
    ):
        raise RuntimeError(f"Expected local MHAgentA 1.4.12 checkout at {mha_root}")
    crafter_spec = find_spec("mha_env_crafter")
    if crafter_spec is None or crafter_spec.origin is None:
        raise ImportError("Could not locate mha_env_crafter runtime sources")
    crafter_source = Path(crafter_spec.origin).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    seeder = Seeder(run)
    treatment = treatment_for_run(run)
    exchange_name = "mhagenta"
    run_agent_id = agent_name(run, "2_3")
    run_env_id = env_name(run, "2_3")
    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=0.0,
        control_frequency=0.0,
        status_frequency=5.0,
        agent_start_delay=STARTUP_DELAY,
        exec_duration=DURATION,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        no_stdout_logs=False,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange_name,
        state_autosave_interval=30.0,
        module_term_timeout=45.0,
        stop_on_agents_term=True,
    )
    initial = _initial_states(treatment)
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=CrafterBDIPerceptor(
            module_id=module_name(PERCEPTOR, 0),
            initial_state=initial[PERCEPTOR],
            exchange_name=exchange_name,
        ),
        actuators=CrafterBDIActuator(
            module_id=module_name(ACTUATOR, 0),
            initial_state=initial[ACTUATOR],
            exchange_name=exchange_name,
        ),
        ll_reasoners=SupportLLReasoner(
            module_id=module_name(LLREASONER, 0),
            initial_state=initial[LLREASONER],
        ),
        knowledge=ForwardingKnowledge(
            module_id=module_name(KNOWLEDGE, 0),
            initial_state=initial[KNOWLEDGE],
        ),
        hl_reasoners=CrafterBDIReasoner(
            module_id=module_name(HLREASONER, 0),
            initial_state=initial[HLREASONER],
            init_kwargs={
                "seed": seeder.hl_reasoner,
                "planner_timeout": PLANNER_TIMEOUT_SECONDS,
            },
        ),
        requirements_path=Path(__file__).with_name("requirements.txt"),
        extra_runtime_sources=common_source,
    )
    record = RECORD == "all" or (RECORD == "first" and run == 0)
    environment_state = {
        **_environment_initial_state(),
        "seed": treatment["seed"],
        "treatment": treatment,
        "record": record,
        "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}",
    }
    orchestrator.add_environment(
        base=CrafterBDIEnvironment(environment_state),
        env_id=run_env_id,
        exec_duration=DURATION + ENVIRONMENT_OVERRUN,
        requirements_path=Path(__file__).with_name("requirements-env.txt"),
        exchange_name=exchange_name,
        extra_runtime_sources=[common_source, crafter_source],
    )
    orchestrator.run(
        mhagenta_version="1.4.12",
        local_build=mha_root,
        force_run=True,
    )

    states = gather_states(exp_path, False, no_warnings=True)
    return check_results(
        states.get(run_agent_id),
        states.get(run_env_id, {}).get(run_env_id),
        _read_log((exp_path / run_agent_id).with_suffix(".log")),
        _read_log((exp_path / run_env_id).with_suffix(".log")),
        verbose=VERBOSE,
    )


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 50,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    process_only: bool = False,
) -> None:
    """Run the frozen experiment batch or process existing results."""

    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [
        {
            "execution_id": f"run-{run}",
            "run_id": run,
            "factors": treatment_for_run(run),
        }
        for run in run_ids
    ]
    primary_error: BaseException | None = None
    try:
        run_experiment_batch(
            experiment_id="2-3-CR",
            title="PRACTICAL REASONING IN CRAFTER",
            runs=run_ids,
            exp_path=exp_path,
            mha_version=mha_version,
            runner=run_experiment,
            process_only=process_only,
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
                "Execution-metrics processing also failed: "
                f"{type(reporting_error).__name__}"
            )


__all__ = [
    "CrafterBDIActuator",
    "CrafterBDIEnvironment",
    "CrafterBDIPerceptor",
    "CrafterBDIReasoner",
    "ForwardingKnowledge",
    "SupportLLReasoner",
    "check_results",
    "deliberative_initial_states",
    "run_batch",
    "run_experiment",
]
