"""Six-module full-domain hybrid-symbolic agent for Experiment 2-4-CR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Literal

from mhagenta import ActionStatus, Belief, Goal, Observation, Orchestrator
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import (
    ActuatorState,
    GoalGraphState,
    HLState,
    KnowledgeState,
    LLState,
    PerceptorState,
)

from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)

from .activities import (
    MILESTONES,
    NEED_THRESHOLDS,
    ActionDecision,
    ActivityContractError,
    ActivityName,
    ActivityUnavailable,
    PlanStep,
    activity_complete,
    activity_from_goal,
    derive_stage_plan,
    desired_value,
    goal_from_dict,
    goal_to_dict,
    highest_milestone,
    make_activity_goal,
    next_activity_action,
    outcome_from_goal,
    planning_summary,
    prerequisite_available,
    protected_needs,
    technology_stage,
    terminal_activity_goal,
    urgent_need,
)
from .beliefs import (
    AbstractState,
    BeliefRevisionError,
    CrafterAction,
    ObservationError,
    abstract_beliefs,
    initial_belief_state,
    parse_abstract_beliefs,
    parse_symbolic_observation,
    revise_belief_state,
)


DURATION = 900.0
SHUTDOWN_MARGIN = 55.0  # Match the 845-second decision cutoff in 2-3-CR.
STARTUP_DELAY = 20.0
ENVIRONMENT_OVERRUN = 30.0
MAX_EPISODE_LEN = 1000
MAX_TOTAL_ACTIONS = 1000
ACTIVITY_ACTION_LIMITS: Mapping[str, int] = {
    ActivityName.EXPLORE.value: 128,
    ActivityName.EAT.value: 96,
    ActivityName.DRINK.value: 96,
    ActivityName.SLEEP.value: 96,
    ActivityName.GET_WOOD.value: 128,
    ActivityName.GET_STONE.value: 128,
    ActivityName.GET_COAL.value: 128,
    ActivityName.GET_IRON.value: 128,
    ActivityName.GET_DIAMOND.value: 128,
    ActivityName.PLACE_TABLE.value: 64,
    ActivityName.PLACE_FURNACE.value: 64,
    ActivityName.MAKE_WOOD_PICKAXE.value: 32,
    ActivityName.MAKE_STONE_PICKAXE.value: 32,
    ActivityName.MAKE_IRON_PICKAXE.value: 32,
}
RECORD: Literal["all", "first", "none"] = "first"


def _one(items: Sequence[Any], label: str) -> Any:
    if len(items) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(items)}")
    return items[0]


@contextmanager
def _active_time(state: Any) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        state["active_seconds"] += perf_counter() - started


def _failure(state: Any, message: str) -> None:
    if state["failure"] is None:
        state["failure"] = message


def _abstract_as_dict(snapshot: AbstractState) -> dict[str, Any]:
    return {
        "revision": snapshot.revision,
        "sleeping": snapshot.sleeping,
        "inventory": dict(snapshot.inventory),
        "known_target_kinds": sorted(snapshot.known_target_kinds),
        "reachable_target_kinds": sorted(snapshot.reachable_target_kinds),
        "station_counts": dict(snapshot.station_counts),
        "usable_station_sets": sorted(snapshot.usable_station_sets),
        "placement_opportunities": sorted(snapshot.placement_opportunities),
        "known_cell_count": snapshot.known_cell_count,
        "terminal": snapshot.terminal,
        "dead": snapshot.dead,
        "experiment_error": snapshot.experiment_error,
    }


def _snapshot(data: Mapping[str, Any]) -> AbstractState:
    return AbstractState(
        int(data["revision"]),
        bool(data["sleeping"]),
        dict(data["inventory"]),
        frozenset(data["known_target_kinds"]),
        frozenset(data["reachable_target_kinds"]),
        dict(data["station_counts"]),
        frozenset(data["usable_station_sets"]),
        frozenset(data["placement_opportunities"]),
        int(data["known_cell_count"]),
        bool(data["terminal"]),
        bool(data["dead"]),
        data.get("experiment_error"),
    )


def _valid_status(status: Mapping[str, Any]) -> bool:
    reward = status.get("reward")
    achievements = status.get("new_achievements")
    return (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and all(type(status.get(key)) is bool for key in ("done", "dead", "illegal_action"))
        and isinstance(achievements, list)
        and all(isinstance(item, str) for item in achievements)
    )


class CrafterHybridEnvironment(MHAEnvBase):
    """Run one ordinary non-resetting Crafter episode through public APIs."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = int(init_state.pop("seed"))
        self._record = bool(init_state.pop("record", False))
        self._artifact_root = str(init_state.pop("artifact_root", f"/{Orchestrator.SAVE_SUBDIR}"))
        self._expected_agent_id = str(init_state.pop("expected_agent_id"))
        self._no_mobs = bool(init_state.pop("no_mobs"))
        self._daylight_effects = bool(init_state.pop("daylight_effects"))
        self._episode_length = int(init_state.pop("episode_length"))
        self._env: Any = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_crafter import CrafterEnv, Recorder

        env: Any = CrafterEnv(
            seed=self._seed,
            length=self._episode_length,
            no_mobs=self._no_mobs,
            symbolic=True,
            daylight_effects=self._daylight_effects,
        )
        if self._record:
            env = Recorder(
                env,
                Path(self._artifact_root) / "videos",
                save_stats=False,
                save_episode=False,
                video_size=(144, 144),
                video_fps=2,
            )
        self._env = env
        self._env.reset()

    def __getstate__(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["_env"] = None
        return data

    def __setstate__(self, data: dict[str, Any]) -> None:
        self.__dict__.update(data)
        self._build_env()

    def on_observe(
        self, state: dict[str, Any], sender_id: str, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the current public symbolic fluent list."""

        state["observation_requests"] += 1
        if sender_id != self._expected_agent_id or kwargs or state["closed"]:
            state["failure"] = "invalid observation request"
            raise RuntimeError(state["failure"])
        return state, {"observation": list(self._env.symbolic_observation())}

    def on_action(
        self, state: dict[str, Any], sender_id: str, **kwargs: Any
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply one native action or close the episode exactly once."""

        action = kwargs.get("action")
        if action == "close":
            if sender_id != self._expected_agent_id:
                state["failure"] = "invalid close requester"
                raise RuntimeError(state["failure"])
            state["close_requests"] += 1
            if not state["closed"]:
                close = getattr(self._env, "close", None)
                video = close() if callable(close) else None
                if self._record and isinstance(video, Path) and video.is_file():
                    state["video_path"] = video.relative_to(Path(self._artifact_root)).as_posix()
                state["closed"] = True
            return state, None

        state["action_requests"] += 1
        if (
            sender_id != self._expected_agent_id
            or set(kwargs) != {"action"}
            or type(action) is not int
            or action not in range(len(CrafterAction))
            or state["terminal"]
            or state["closed"]
        ):
            state["failure"] = "invalid native action request"
            raise RuntimeError(state["failure"])
        _, reward, done, info = self._env.step(action)
        illegal = info.get("illegal_action")
        inventory = info.get("inventory")
        achievements = info.get("achievements")
        if type(illegal) is not bool or not isinstance(inventory, dict) or not isinstance(achievements, dict):
            state["failure"] = "Crafter returned malformed public info"
            raise RuntimeError(state["failure"])
        previous = state["achievement_counts"]
        new = sorted(
            name
            for name, count in achievements.items()
            if int(count) > int(previous.get(name, 0))
        )
        dead = int(inventory.get("health", 0)) <= 0
        state["native_actions"] += 1
        state["illegal_actions"] += int(illegal)
        state["terminal"] = bool(done)
        state["dead"] = dead
        state["achievement_counts"] = {name: int(count) for name, count in achievements.items()}
        state["achievements"] = sorted(name for name, count in achievements.items() if int(count) > 0)
        return state, {
            "reward": float(reward),
            "done": bool(done),
            "dead": dead,
            "illegal_action": illegal,
            "new_achievements": new,
        }


class CrafterHybridPerceptor(RMQPerceptorBase):
    """Bridge one observation request without carrying action context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""

    def on_first(self, state: PerceptorState) -> PerceptorState:
        self._env_id = _one(state.directory.external.environments, "external environment").address["env_id"]
        self._ll_id = _one(state.directory.internal.ll_reasoning, "low-level reasoner").module_id
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            state["requests"] += 1
            if sender != self._ll_id or kwargs or state["pending"]:
                _failure(state, "invalid or overlapping observation request")
                return state
            state["pending"] = True
            self.observe(self._env_id)
            return state

    def on_observation(
        self, state: PerceptorState, env_id: str, **kwargs: Any
    ) -> PerceptorState:
        with _active_time(state):
            content = kwargs.get("observation")
            valid = (
                env_id == self._env_id
                and state["pending"]
                and set(kwargs) == {"observation"}
                and isinstance(content, list)
                and all(isinstance(item, str) for item in content)
            )
            state["pending"] = False
            state["observations"] += 1
            if not valid:
                _failure(state, "invalid environment observation")
                content = []
            state.outbox.send_observation(
                self._ll_id, Observation(content, observation_type="crafter-symbolic")
            )
            return state


class CrafterHybridActuator(RMQActuatorBase):
    """Bridge one native action and retain only its correlation ID."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        self._env_id = _one(state.directory.external.environments, "external environment").address["env_id"]
        self._ll_id = _one(state.directory.internal.ll_reasoning, "low-level reasoner").module_id
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        with _active_time(state):
            state["requests"] += 1
            action, action_id = kwargs.get("action"), kwargs.get("action_id")
            if (
                sender != self._ll_id
                or set(kwargs) != {"action", "action_id"}
                or state["pending_action_id"] is not None
                or type(action) is not int
                or action not in range(len(CrafterAction))
                or not isinstance(action_id, str)
                or not action_id
            ):
                _failure(state, "invalid or overlapping action request")
                return state
            state["pending_action_id"] = action_id
            self.act(self._env_id, action=action)
            return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        with _active_time(state):
            action_id = state["pending_action_id"]
            if env_id != self._env_id or action_id is None or not _valid_status(kwargs):
                _failure(state, "invalid environment action status")
                return state
            state["pending_action_id"] = None
            state["statuses"] += 1
            state.outbox.send_status(self._ll_id, ActionStatus(dict(kwargs)), action_id=action_id)
            return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        if state["pending_action_id"] is not None:
            _failure(state, "actuator stopped with a pending action")
        if self._env_id and not state["close_sent"]:
            self.act(self._env_id, action="close")
            state["close_sent"] = True
        return state


class ActivityLLReasoner(LLReasonerBase):
    """Realize full-domain compound activities from fresh detailed beliefs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._perceptor_id = ""
        self._actuator_id = ""
        self._knowledge_id = ""
        self._goal_graph_id = ""
        self._activity_limits = dict(ACTIVITY_ACTION_LIMITS)
        self._total_action_limit = MAX_TOTAL_ACTIONS

    def on_init(self, **kwargs: Any) -> None:
        limits = kwargs.get("activity_action_limits", ACTIVITY_ACTION_LIMITS)
        total = kwargs.get("total_action_limit", MAX_TOTAL_ACTIONS)
        if (
            not isinstance(limits, Mapping)
            or set(limits) != {activity.value for activity in ActivityName}
            or any(type(value) is not int or value <= 0 for value in limits.values())
        ):
            raise ValueError("activity_action_limits must cover every activity")
        if total is not None and (type(total) is not int or total <= 0):
            raise ValueError("total_action_limit must be a positive integer or None")
        self._activity_limits = {str(key): int(value) for key, value in limits.items()}
        self._total_action_limit = total

    def on_first(self, state: LLState) -> LLState:
        self._perceptor_id = _one(state.directory.internal.perception, "perceptor").module_id
        self._actuator_id = _one(state.directory.internal.actuation, "actuator").module_id
        self._knowledge_id = _one(state.directory.internal.knowledge, "knowledge").module_id
        self._goal_graph_id = _one(state.directory.internal.goals, "goal graph").module_id
        self._request_observation(state)
        return state

    def _request_observation(self, state: LLState) -> None:
        if state["awaiting_observation"]:
            self._fail(state, "overlapping LL observation request")
            return
        state["observation_requests"] += 1
        state["awaiting_observation"] = True
        state["phase"] = "awaiting_observation"
        state.outbox.request_observation(self._perceptor_id)

    def _fail(self, state: LLState, message: str) -> None:
        _failure(state, message)
        if state["active_activity"] is not None:
            self._finish(state, "failed", "experiment_error")
        else:
            state.outbox.terminate_agent(message)
        state["phase"] = "failed"

    def _finish(
        self,
        state: LLState,
        status: str,
        reason: str = "",
        interruption: Mapping[str, Any] | None = None,
    ) -> None:
        active = state["active_activity"]
        if active is None:
            return
        requested = goal_from_dict(active["goal"])
        completed = terminal_activity_goal(
            requested,
            status=status,
            completion_revision=int(state["belief_state"]["revision"]),
            atomic=active["atomic"],
            failure_reason=reason,
            interruption=interruption,
        )
        state["active_activity"] = None
        state["pending_atomic"] = None
        state["awaiting_observation"] = False
        state["terminal_updates"] += 1
        state["phase"] = "idle"
        state.outbox.send_goal_update(self._goal_graph_id, [completed])

    def _dispatch(self, state: LLState, decision: Any) -> None:
        sequence = state["actions"] + 1
        action_id = f"action-{sequence}"
        state["pending_atomic"] = {
            "action_id": action_id,
            "action": int(decision.action),
            "movement_kind": decision.movement_kind,
            "source_cell": list(decision.source_cell),
            "destination_cell": list(decision.destination_cell),
            "dispatch_revision": int(state["belief_state"]["revision"]),
            "facing_before": state["belief_state"]["facing"],
            "status": None,
        }
        state["actions"] += 1
        state["phase"] = "awaiting_action_status"
        state.outbox.request_action(
            self._actuator_id, action=int(decision.action), action_id=action_id
        )

    def _continue(self, state: LLState) -> None:
        active = state["active_activity"]
        if active is None:
            state["phase"] = "idle"
            return
        spec = activity_from_goal(goal_from_dict(active["goal"]))
        satisfied = activity_complete(spec, state["belief_state"])
        if state["belief_state"]["terminal"] or state["belief_state"]["dead"]:
            self._finish(state, "succeeded" if satisfied else "failed", "" if satisfied else "environment_terminal")
            return
        if satisfied:
            self._finish(state, "succeeded")
            return
        if self._total_action_limit is not None and state["actions"] >= self._total_action_limit:
            state["total_action_bound_reached"] = True
            self._finish(state, "failed", "total_action_bound")
            return
        if float(getattr(state, "time", 0.0)) >= DURATION - SHUTDOWN_MARGIN:
            state["time_bound_reached"] = True
            self._finish(state, "failed", "time_budget_exhausted")
            return
        inventory = state["belief_state"].get("inventory", {})
        need = urgent_need(inventory)
        if need is not None and need not in protected_needs(spec):
            interruption = {
                "need": need,
                "observed_value": int(inventory.get(need, 0)),
                "interrupted_activity": spec.activity.value,
                "interrupted_goal_id": spec.goal_id,
                "belief_revision": int(state["belief_state"]["revision"]),
            }
            state["need_interruptions"] += 1
            self._finish(state, "failed", "need_interruption", interruption)
            return
        if len(active["atomic"]) >= self._activity_limits[spec.activity.value]:
            reason = "exploration_bound" if spec.activity is ActivityName.EXPLORE else "activity_action_bound"
            self._finish(state, "failed", reason)
            return
        if (
            spec.activity is not ActivityName.SLEEP
            and bool(state["belief_state"].get("sleeping"))
        ):
            player = tuple(int(value) for value in state["belief_state"]["player"])
            self._dispatch(
                state,
                ActionDecision(
                    CrafterAction.NOOP,
                    "none",
                    player,
                    player,
                    None,
                    None,
                    "wait_until_awake",
                ),
            )
            return
        try:
            decision = next_activity_action(spec, state["belief_state"])
        except ActivityUnavailable as exc:
            self._finish(state, "failed", exc.reason)
            return
        except ActivityContractError as exc:
            self._fail(state, str(exc))
            return
        self._dispatch(state, decision)

    def on_goal_update(
        self, state: LLState, sender: str, goals: Sequence[Goal], **kwargs: Any
    ) -> LLState:
        with _active_time(state):
            if (
                sender != self._goal_graph_id
                or len(goals) != 1
                or kwargs
                or state["active_activity"] is not None
                or state["pending_atomic"] is not None
                or state["awaiting_observation"]
                or state["phase"] != "idle"
            ):
                self._fail(state, "invalid or overlapping activity request")
                return state
            try:
                spec = activity_from_goal(goals[0])
            except ActivityContractError as exc:
                self._fail(state, str(exc))
                return state
            state["goal_activations"] += 1
            state["active_activity"] = {"goal": goal_to_dict(goals[0]), "atomic": []}
            if spec.based_on_revision != state["belief_state"]["revision"]:
                self._fail(state, "activity belief revision mismatch")
            elif activity_complete(spec, state["belief_state"]):
                self._fail(state, "activity was already satisfied at dispatch")
            elif not prerequisite_available(spec, state["belief_state"]):
                self._finish(state, "failed", "missing_prerequisite")
            else:
                state["phase"] = "active"
                self._continue(state)
            return state

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        with _active_time(state):
            pending = state["pending_atomic"]
            status = action_status.status
            if (
                sender != self._actuator_id
                or pending is None
                or not isinstance(status, Mapping)
                or set(kwargs) != {"action_id"}
                or kwargs["action_id"] != pending["action_id"]
                or not _valid_status(status)
            ):
                self._fail(state, "invalid action status")
                return state
            state["action_statuses"] += 1
            pending["status"] = dict(status)
            if status["illegal_action"]:
                self._fail(state, "Crafter rejected a selected action")
                return state
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
            if sender != self._perceptor_id or kwargs or not state["awaiting_observation"]:
                self._fail(state, "invalid LL observation")
                return state
            state["awaiting_observation"] = False
            revision = int(state["belief_state"]["revision"]) + 1
            try:
                percept = parse_symbolic_observation(observation.content)
                result = revise_belief_state(
                    state["belief_state"],
                    percept,
                    revision=revision,
                    pending_action=state["pending_atomic"],
                )
                summary = planning_summary(result.belief_state)
                beliefs = abstract_beliefs(result.belief_state, summary)
            except (ObservationError, BeliefRevisionError, ActivityContractError, TypeError, ValueError) as exc:
                self._fail(state, f"belief revision failed: {exc}")
                return state
            state["belief_state"] = result.belief_state
            state["observations"] += 1
            if result.consumed_action:
                pending = state["pending_atomic"]
                assert pending is not None and state["active_activity"] is not None
                state["active_activity"]["atomic"].append(
                    {
                        key: pending[key]
                        for key in (
                            "action_id",
                            "action",
                            "movement_kind",
                            "source_cell",
                            "destination_cell",
                            "dispatch_revision",
                        )
                    }
                    | {"legal": True, "confirmation_revision": revision}
                )
                state["pending_atomic"] = None
            state["belief_sends"] += 1
            state.outbox.send_beliefs(self._knowledge_id, observation, beliefs)
            self._continue(state)
            return state


class ForwardingKnowledge(KnowledgeBase):
    """Retain and forward only the latest complete abstract revision."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ll_id = ""

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        self._ll_id = _one(state.directory.internal.ll_reasoning, "LL reasoner").module_id
        _one(state.directory.internal.hl_reasoning, "HL reasoner")
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
            if sender != self._ll_id or kwargs:
                _failure(state, "invalid Knowledge belief sender")
                return state
            try:
                snapshot = parse_abstract_beliefs(beliefs)
            except ObservationError as exc:
                _failure(state, str(exc))
                return state
            if snapshot.revision != state["last_revision"] + 1:
                _failure(state, "non-sequential Knowledge belief revision")
                return state
            state["observed"] += 1
            state["forwarded"] += 1
            state["last_revision"] = snapshot.revision
            state["beliefs"] = _abstract_as_dict(snapshot)
            return super().on_observed_beliefs(state, sender, observation, beliefs, **kwargs)


class ActivityGoalGraph(GoalGraphBase):
    """Relay one requested activity and its terminal update by goal ID."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ll_id = ""
        self._hl_id = ""

    def on_first(self, state: GoalGraphState) -> GoalGraphState:
        self._ll_id = _one(state.directory.internal.ll_reasoning, "LL reasoner").module_id
        self._hl_id = _one(state.directory.internal.hl_reasoning, "HL reasoner").module_id
        return state

    def on_goal_update(
        self, state: GoalGraphState, sender: str, goals: Sequence[Goal], **kwargs: Any
    ) -> GoalGraphState:
        with _active_time(state):
            if len(goals) != 1 or kwargs:
                _failure(state, "goal graph requires one plain Goal")
                return state
            goal = goals[0]
            try:
                if sender == self._hl_id:
                    spec = activity_from_goal(goal)
                    if state["active_goal_id"] is not None:
                        raise ActivityContractError("overlapping goal graph request")
                    state["active_goal_id"] = spec.goal_id
                    state["dispatches"] += 1
                    state.outbox.send_goals(self._ll_id, goals)
                elif sender == self._ll_id:
                    outcome = outcome_from_goal(goal)
                    if outcome.goal_id != state["active_goal_id"]:
                        raise ActivityContractError("terminal goal ID mismatch")
                    state["active_goal_id"] = None
                    state["terminals"] += 1
                    state.outbox.send_goals(self._hl_id, goals)
                else:
                    raise ActivityContractError("invalid goal graph sender")
            except ActivityContractError as exc:
                _failure(state, str(exc))
            return state


class HybridSymbolicReasoner(HLReasonerBase):
    """Own the persistent diamond intention, replanning, and recovery control."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._knowledge_id = ""
        self._goal_graph_id = ""

    def on_first(self, state: HLState) -> HLState:
        self._knowledge_id = _one(state.directory.internal.knowledge, "knowledge").module_id
        self._goal_graph_id = _one(state.directory.internal.goals, "goal graph").module_id
        return state

    def _terminate(self, state: HLState, status: str, reason: str) -> None:
        hierarchy = state["hierarchy"]
        if hierarchy is not None:
            hierarchy["status"] = status
            hierarchy["intention"]["status"] = (
                "succeeded" if status == "succeeded" else "invalid" if status == "invalid" else "unachieved"
            )
            hierarchy["final_revision"] = state["latest_revision"]
            hierarchy["terminal_reason"] = reason
        state["phase"] = status
        state["terminal_reason"] = reason
        state.outbox.terminate_agent(f"2-4-CR {status}: {reason}")

    def _fail(self, state: HLState, message: str) -> None:
        _failure(state, message)
        self._terminate(state, "invalid", message)

    def _active_row(self, state: HLState) -> dict[str, Any] | None:
        hierarchy = state["hierarchy"]
        if hierarchy is None or not hierarchy["activities"]:
            return None
        row = hierarchy["activities"][-1]
        return row if row["status"] == "active" else None

    def _create_hierarchy(self, snapshot: AbstractState) -> dict[str, Any]:
        stage = technology_stage(snapshot)
        return {
            "intention": {"name": "obtain_diamond", "target_value": 1, "status": "active"},
            "current_stage": stage,
            "highest_milestone": highest_milestone(snapshot),
            "override": None,
            "current_plan": None,
            "plan_revision": 0,
            "activities": [],
            "final_revision": None,
            "terminal_reason": None,
            "status": "active",
        }

    def _refresh_stage(self, state: HLState, snapshot: AbstractState) -> None:
        hierarchy = state["hierarchy"]
        assert hierarchy is not None
        previous_stage = hierarchy["current_stage"]
        stage = technology_stage(snapshot)
        hierarchy["current_stage"] = stage
        milestone = highest_milestone(snapshot)
        if MILESTONES.index(milestone) > MILESTONES.index(hierarchy["highest_milestone"]):
            hierarchy["highest_milestone"] = milestone
        episode = state["exploration_episode"]
        if (
            episode is not None
            and episode["owner"] == f"technology:{previous_stage}"
            and stage != previous_stage
        ):
            state["exploration_episode"] = None

    @staticmethod
    def _explore_purpose(spec: Any) -> str | None:
        if spec.activity is not ActivityName.EXPLORE:
            return None
        prefix = "target" if spec.desired_predicate == "reachable_target_kind" else "placement"
        return f"{prefix}:{spec.desired_arguments[0]}"

    def _dispatch_plan(self, state: HLState, steps: Sequence[PlanStep]) -> None:
        snapshot = _snapshot(state["abstract_state"])
        hierarchy = state["hierarchy"]
        assert hierarchy is not None and steps
        hierarchy["plan_revision"] += 1
        plan_revision = hierarchy["plan_revision"]
        hierarchy["current_plan"] = {
            "based_on_revision": snapshot.revision,
            "steps": [step.as_dict() for step in steps],
        }
        step = steps[0]
        goal = make_activity_goal(
            goal_id=f"activity-{state['dispatches'] + 1}",
            activity=step.activity,
            based_on_revision=snapshot.revision,
            desired_predicate=step.desired_predicate,
            desired_arguments=step.desired_arguments,
        )
        spec = activity_from_goal(goal)
        purpose = self._explore_purpose(spec)
        if purpose is not None:
            override = hierarchy["override"]
            owner = (
                f"recovery:{override['need']}"
                if override is not None
                else f"technology:{hierarchy['current_stage']}"
            )
            episode = state["exploration_episode"]
            if episode is None or episode["owner"] != owner or episode["purpose"] != purpose:
                state["exploration_episode"] = {
                    "owner": owner,
                    "purpose": purpose,
                    "consecutive_bounds": 0,
                }
        hierarchy["activities"].append(
            {
                "goal_id": spec.goal_id,
                "stage": hierarchy["current_stage"],
                "plan_revision": plan_revision,
                "activity": spec.activity.value,
                "desired": goal_to_dict(goal)["state"][0],
                "based_on_revision": spec.based_on_revision,
                "known_cell_count_start": snapshot.known_cell_count,
                "known_cell_count_end": None,
                "atomic": [],
                "completion_revision": None,
                "final_value": None,
                "status": "active",
                "failure_reason": "",
                "interruption": None,
            }
        )
        state["dispatches"] += 1
        state["phase"] = "active"
        state.outbox.send_goals(self._goal_graph_id, [goal])

    def _recovery_step(self, snapshot: AbstractState, need: str) -> PlanStep:
        target = NEED_THRESHOLDS[need][1]
        if need == "food":
            if "food" in snapshot.reachable_target_kinds:
                return PlanStep(ActivityName.EAT, "inventory_at_least", ("food", target))
            return PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("food",))
        if need == "drink":
            if "water" in snapshot.reachable_target_kinds:
                return PlanStep(ActivityName.DRINK, "inventory_at_least", ("drink", target))
            return PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("water",))
        return PlanStep(ActivityName.SLEEP, "inventory_at_least", (need, target))

    def _plan_or_terminate(self, state: HLState) -> None:
        snapshot = _snapshot(state["abstract_state"])
        hierarchy = state["hierarchy"]
        assert hierarchy is not None
        if int(snapshot.inventory.get("diamond", 0)) >= 1:
            self._terminate(state, "succeeded", "diamond_obtained")
            return
        if snapshot.terminal or snapshot.dead:
            self._terminate(state, "environment_terminal", "death" if snapshot.dead else "episode_limit")
            return
        override = hierarchy["override"]
        if override is not None:
            need = override["need"]
            if int(snapshot.inventory.get(need, 0)) >= NEED_THRESHOLDS[need][1]:
                if state["exploration_episode"] is not None and state["exploration_episode"]["owner"] == f"recovery:{need}":
                    state["exploration_episode"] = None
                hierarchy["override"] = None
                state["recoveries"] += 1
                override = None
            else:
                self._dispatch_plan(state, (self._recovery_step(snapshot, need),))
                return
        if override is None:
            need = urgent_need(snapshot.inventory)
            if need is not None:
                hierarchy["override"] = {
                    "kind": "need_recovery",
                    "need": need,
                    "source_goal_id": None,
                    "started_revision": snapshot.revision,
                }
                self._dispatch_plan(state, (self._recovery_step(snapshot, need),))
                return
        self._refresh_stage(state, snapshot)
        plan = derive_stage_plan(snapshot)
        if not plan:
            self._fail(state, "diamond plan ended without diamond evidence")
            return
        self._dispatch_plan(state, plan)

    def _row_desired_value(self, row: Mapping[str, Any], snapshot: AbstractState) -> int | bool:
        desired = row["desired"]
        arguments = desired["arguments"]
        if desired["predicate"] == "inventory_at_least":
            return int(snapshot.inventory.get(arguments[0], 0))
        values = {
            "usable_station_set": snapshot.usable_station_sets,
            "reachable_target_kind": snapshot.reachable_target_kinds,
            "placement_opportunity": snapshot.placement_opportunities,
        }[desired["predicate"]]
        return arguments[0] in values

    def _finish_expected_failure(
        self, state: HLState, row: Mapping[str, Any], reason: str
    ) -> bool:
        hierarchy = state["hierarchy"]
        assert hierarchy is not None
        if reason == "total_action_bound":
            self._terminate(state, "budget_exhausted", "total_action_bound")
            return True
        if reason == "time_budget_exhausted":
            self._terminate(state, "budget_exhausted", "time_budget_exhausted")
            return True
        if reason == "environment_terminal":
            snapshot = _snapshot(state["abstract_state"])
            self._terminate(state, "environment_terminal", "death" if snapshot.dead else "episode_limit")
            return True
        if reason == "no_frontier":
            self._terminate(state, "blocked", f"no_frontier:{self._row_purpose(row)}")
            return True
        if reason == "activity_action_bound":
            return False
        if reason == "exploration_bound":
            episode = state["exploration_episode"]
            if episode is None or episode["purpose"] != self._row_purpose(row):
                self._fail(state, "exploration bound has no matching episode")
                return True
            episode["consecutive_bounds"] += 1
            return False
        if reason in {"no_reachable_target", "no_placement_opportunity", "no_usable_station"}:
            return False
        self._fail(state, reason)
        return True

    @staticmethod
    def _row_purpose(row: Mapping[str, Any]) -> str:
        desired = row["desired"]
        prefix = "target" if desired["predicate"] == "reachable_target_kind" else "placement"
        return f"{prefix}:{desired['arguments'][0]}"

    def _try_join(self, state: HLState) -> None:
        serialized = state["pending_outcome"]
        if serialized is None:
            return
        goal = goal_from_dict(serialized)
        outcome = outcome_from_goal(goal)
        if state["latest_revision"] < outcome.completion_revision:
            return
        row = self._active_row(state)
        if row is None or outcome.goal_id != row["goal_id"]:
            self._fail(state, "terminal activity has no matching active row")
            return
        if goal_to_dict(goal)["state"][0] != row["desired"]:
            self._fail(state, "terminal activity changed its desired belief")
            return
        if outcome.completion_revision < row["based_on_revision"] or (
            outcome.status == "succeeded" and outcome.completion_revision == row["based_on_revision"]
        ):
            self._fail(state, "activity completion has an invalid belief revision")
            return
        snapshot = _snapshot(state["abstract_state"])
        final_value = self._row_desired_value(row, snapshot)
        target = row["desired"]["arguments"][-1]
        satisfied = final_value is True if isinstance(target, str) else int(final_value) >= int(target)
        if outcome.status == "succeeded" and (
            snapshot.revision != outcome.completion_revision or not satisfied
        ):
            self._fail(state, "successful activity lacks matching completion beliefs")
            return
        if outcome.interruption is not None:
            if (
                outcome.interruption["belief_revision"] != outcome.completion_revision
                or outcome.interruption["interrupted_goal_id"] != row["goal_id"]
                or outcome.interruption["interrupted_activity"] != row["activity"]
            ):
                self._fail(state, "interruption correlation mismatch")
                return
        row["atomic"] = [dict(item) for item in outcome.atomic]
        row["completion_revision"] = outcome.completion_revision
        row["known_cell_count_end"] = snapshot.known_cell_count
        row["final_value"] = final_value
        row["failure_reason"] = outcome.failure_reason
        row["interruption"] = dict(outcome.interruption) if outcome.interruption is not None else None
        row["status"] = "interrupted" if outcome.interruption is not None else outcome.status
        state["pending_outcome"] = None
        state["terminals"] += 1
        if outcome.interruption is not None:
            state["interrupted_activities"] += 1
            state["need_interruptions"] += 1
            need = str(outcome.interruption["need"])
            hierarchy = state["hierarchy"]
            assert hierarchy is not None
            hierarchy["override"] = {
                "kind": "need_recovery",
                "need": need,
                "source_goal_id": outcome.goal_id,
                "started_revision": outcome.completion_revision,
            }
        elif outcome.status == "succeeded":
            state["completed_activities"] += 1
            if row["activity"] == ActivityName.EXPLORE.value:
                state["exploration_episode"] = None
        else:
            state["failed_activities"] += 1
            if self._finish_expected_failure(state, row, outcome.failure_reason):
                return
        self._refresh_stage(state, snapshot)
        self._plan_or_terminate(state)

    def on_belief_update(
        self, state: HLState, sender: str, beliefs: Sequence[Belief], **kwargs: Any
    ) -> HLState:
        with _active_time(state):
            if state["phase"] in {"succeeded", "blocked", "budget_exhausted", "environment_terminal", "invalid"}:
                return state
            if sender != self._knowledge_id or kwargs:
                self._fail(state, "invalid HL belief sender")
                return state
            try:
                snapshot = parse_abstract_beliefs(beliefs)
            except ObservationError as exc:
                self._fail(state, str(exc))
                return state
            if snapshot.revision != state["latest_revision"] + 1:
                self._fail(state, "non-sequential HL belief revision")
                return state
            state["latest_revision"] = snapshot.revision
            state["belief_updates"] += 1
            state["abstract_state"] = _abstract_as_dict(snapshot)
            if state["hierarchy"] is None and snapshot.experiment_error is None:
                state["hierarchy"] = self._create_hierarchy(snapshot)
            if snapshot.experiment_error:
                self._fail(state, snapshot.experiment_error)
            elif self._active_row(state) is not None:
                self._try_join(state)
            else:
                assert state["hierarchy"] is not None
                self._refresh_stage(state, snapshot)
                self._plan_or_terminate(state)
            return state

    def on_goal_update(
        self, state: HLState, sender: str, goals: Sequence[Goal], **kwargs: Any
    ) -> HLState:
        with _active_time(state):
            terminal = state["phase"] in {
                "succeeded", "blocked", "budget_exhausted", "environment_terminal", "invalid"
            }
            if terminal:
                return state
            if (
                sender != self._goal_graph_id
                or len(goals) != 1
                or kwargs
                or state["pending_outcome"] is not None
            ):
                self._fail(state, "invalid HL terminal activity")
                return state
            try:
                outcome_from_goal(goals[0])
            except ActivityContractError as exc:
                self._fail(state, str(exc))
                return state
            state["pending_outcome"] = goal_to_dict(goals[0])
            self._try_join(state)
            return state


def initial_states(treatment: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Build fresh JSON-compatible state for all six reactive modules."""

    common = {"active_seconds": 0.0, "failure": None}
    return {
        PERCEPTOR: {**common, "requests": 0, "observations": 0, "pending": False},
        ACTUATOR: {
            **common,
            "requests": 0,
            "statuses": 0,
            "pending_action_id": None,
            "close_sent": False,
        },
        LLREASONER: {
            **common,
            "phase": "starting",
            "belief_state": initial_belief_state(),
            "observation_requests": 0,
            "observations": 0,
            "belief_sends": 0,
            "goal_activations": 0,
            "terminal_updates": 0,
            "actions": 0,
            "action_statuses": 0,
            "need_interruptions": 0,
            "active_activity": None,
            "pending_atomic": None,
            "awaiting_observation": False,
            "total_action_bound_reached": False,
            "time_bound_reached": False,
        },
        KNOWLEDGE: {
            **common,
            "observed": 0,
            "forwarded": 0,
            "last_revision": 0,
            "beliefs": None,
        },
        GOALGRAPH: {
            **common,
            "active_goal_id": None,
            "dispatches": 0,
            "terminals": 0,
        },
        HLREASONER: {
            **common,
            "treatment": dict(treatment or {}),
            "phase": "waiting_beliefs",
            "latest_revision": 0,
            "belief_updates": 0,
            "abstract_state": None,
            "hierarchy": None,
            "pending_outcome": None,
            "dispatches": 0,
            "terminals": 0,
            "completed_activities": 0,
            "interrupted_activities": 0,
            "failed_activities": 0,
            "need_interruptions": 0,
            "recoveries": 0,
            "exploration_episode": None,
            "terminal_reason": None,
        },
    }


def environment_initial_state(treatment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build compact environment lifecycle and public-result state."""

    return {
        "treatment": dict(treatment or {}),
        "observation_requests": 0,
        "action_requests": 0,
        "native_actions": 0,
        "illegal_actions": 0,
        "terminal": False,
        "dead": False,
        "achievement_counts": {},
        "achievements": [],
        "close_requests": 0,
        "closed": False,
        "video_path": None,
        "failure": None,
    }
