"""Compact six-module hybrid agent for Experiment 2-4-BW."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from logging import ERROR
from pathlib import Path
import os
import random
import signal
from threading import Timer
from typing import Any

import numpy as np
from mhagenta import ActionStatus, Belief, Goal, Observation, Orchestrator
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, GoalGraphState, HLState, KnowledgeState, LLState, PerceptorState

from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR

from .planning import (
    GoalSpec,
    PlanningOutcome,
    PlanningService,
    TransferSpec,
    beliefs_to_dicts,
    beliefs_to_facts,
    block_names,
    format_fact,
    generate_options,
    location_index,
    location_names,
    missing_transfer_preconditions,
    observed_transfer_targets,
    parse_symbolic_observation,
    project_abstract_facts,
    split_fact,
    transfer_from_goal,
    transfer_goal,
)


def _terminate_agent(state: Any, reason: str) -> None:
    """Request termination when running with a framework-backed outbox."""

    terminate = getattr(state.outbox, "terminate_agent", None)
    if callable(terminate):
        terminate(reason)

DURATION = 1800.0
SHUTDOWN_MARGIN = 5.0
TABLE_LEN = 5
NUM_BLOCKS = 8
PLANNER_TIMEOUT = 15.0
SAVE_SUBDIR = Orchestrator.SAVE_SUBDIR
K_ACTION, K_OBSERVATION, K_REWARD, K_STATE, K_LEGAL = "action", "observation", "reward", "state", "legal"
A_PICK_UP, A_PUT_DOWN, A_MOVE_LEFT, A_MOVE_RIGHT, A_CLOSE = 0, 1, 2, 3, "close"
IDENTITY_KEYS = ("hierarchy_id", "goal_id", "plan_id", "step_index")


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _one_id(entries: Iterable[Any], label: str) -> str:
    ids = [entry.module_id for entry in entries]
    if len(ids) != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {ids!r}.")
    return ids[0]


def _identity(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json(data[key]) for key in IDENTITY_KEYS if data.get(key) is not None}


def _facts_with(facts: set[str], predicate: str) -> list[tuple[str, ...]]:
    return [arguments for fact in facts for name, arguments in [split_fact(fact)] if name == predicate]


def _fail(state: Any, reason: str) -> None:
    if state["failure"] is None:
        state["failure"] = reason


class TestEnvironment(MHAEnvBase):
    """Expose the deterministic symbolic Blocks World over RabbitMQ."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = init_state.pop("seed", None)
        self._record = bool(init_state.pop("record", False))
        self._table_len = int(init_state.pop("table_len", TABLE_LEN))
        self._num_blocks = int(init_state.pop("num_blocks", NUM_BLOCKS))
        self._env: Any = None
        self._stop_timer: Timer | None = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_blocksworld import BWRecorder, BlocksWorldEnv

        env = BlocksWorldEnv(
            table_len=self._table_len,
            num_blocks=self._num_blocks,
            render_mode="rgb_array" if self._record else None,
            symbolic=True,
        )
        env.expose_snapshot = True
        self._env = BWRecorder(env, path=f"/{SAVE_SUBDIR}", single_trace=True) if self._record else env
        observation, _ = self._env.reset(seed=self._seed)
        self.state[K_STATE] = list(observation)
        self.state["initial_state"] = list(observation)

    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        state["observation_count"] += 1
        return state, {K_OBSERVATION: list(state[K_STATE])}

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs: Any) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            state["close_requests"] = state.get("close_requests", 0) + 1
            state["closed"] = True
            if self._env is not None:
                self._env.close()
            if self._stop_timer is None:
                # Match 2-3-BW: save through graceful shutdown before Docker stop.
                self._stop_timer = Timer(0.05, os.kill, args=(os.getpid(), signal.SIGTERM))
                self._stop_timer.daemon = True
                self._stop_timer.start()
            return state, None
        if not isinstance(action, int):
            state["illegal_actions"] += 1
            return state, {K_REWARD: None, K_LEGAL: False, "error": "invalid-action"}
        observation, reward, terminated, truncated, info = self._env.step(action)
        snapshot = info.get("snapshot")
        legal = bool(snapshot.legal) if snapshot is not None else float(reward) != -0.5
        state[K_STATE] = list(observation)
        state["actions"] += 1
        state["illegal_actions"] += int(not legal)
        return state, {
            K_ACTION: action,
            K_REWARD: float(reward),
            K_LEGAL: legal,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def __getstate__(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["_env"] = None
        data["_stop_timer"] = None
        return data

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()


class HybridBlocksWorldPerceptor(RMQPerceptorBase):
    """Forward one complete correlated observation at a time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = self._ll_id = ""

    def _resolve(self, state: PerceptorState) -> None:
        if not self._env_id:
            env = state.directory.external.environment
            if env is None:
                raise RuntimeError("No Blocks World environment is registered.")
            self._env_id = env.address["env_id"]
        if not self._ll_id:
            self._ll_id = _one_id(state.directory.internal.ll_reasoning, "LL reasoner")

    def on_first(self, state: PerceptorState) -> PerceptorState:
        self._resolve(state)
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        self._resolve(state)
        if sender != self._ll_id or state["pending"] is not None:
            _fail(state, "invalid-observation-request")
            return state
        state["pending"] = _json(dict(kwargs))
        state["requests"] += 1
        self.observe(self._env_id)
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        self._resolve(state)
        pending = state["pending"]
        observation = kwargs.get(K_OBSERVATION)
        if env_id != self._env_id or not isinstance(pending, dict) or not isinstance(observation, list):
            _fail(state, "invalid-observation-response")
            return state
        state["pending"] = None
        state["observations"] += 1
        state.outbox.send_observation(self._ll_id, Observation(observation), **pending)
        return state


class HybridBlocksWorldActuator(RMQActuatorBase):
    """Execute one correlated atomic action at a time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = self._ll_id = ""
        self._close_sent = False

    def _resolve(self, state: ActuatorState) -> None:
        if not self._env_id:
            env = state.directory.external.environment
            if env is None:
                raise RuntimeError("No Blocks World environment is registered.")
            self._env_id = env.address["env_id"]
        if not self._ll_id:
            self._ll_id = _one_id(state.directory.internal.ll_reasoning, "LL reasoner")

    def on_first(self, state: ActuatorState) -> ActuatorState:
        self._resolve(state)
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        self._resolve(state)
        action = kwargs.get(K_ACTION)
        if sender != self._ll_id or state["pending"] is not None or action not in {0, 1, 2, 3}:
            _fail(state, "invalid-action-request")
            return state
        state["pending"] = _json(dict(kwargs))
        state["requests"] += 1
        self.act(self._env_id, action=action)
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        self._resolve(state)
        pending = state["pending"]
        if env_id != self._env_id or not isinstance(pending, dict):
            _fail(state, "invalid-action-status")
            return state
        status = {**pending, **_json(dict(kwargs))}
        state["pending"] = None
        state["statuses"] += 1
        state.outbox.send_status(self._ll_id, ActionStatus(status))
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        """Close the environment once even if shutdown hooks are repeated."""
        if self._env_id and not self._close_sent:
            self._close_sent = True
            self.act(self._env_id, action=A_CLOSE)
        return state


class TransferLLReasoner(LLReasonerBase):
    """Realize each transfer with one direct fail-fast atomic FSM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._perceptor_id = self._actuator_id = self._knowledge_id = self._graph_id = ""

    def _resolve(self, state: LLState) -> None:
        directory = state.directory.internal
        if not self._perceptor_id:
            self._perceptor_id = _one_id(directory.perception, "perceptor")
            self._actuator_id = _one_id(directory.actuation, "actuator")
            self._knowledge_id = _one_id(directory.knowledge, "knowledge module")
            self._graph_id = _one_id(directory.goals, "goal graph")

    @staticmethod
    def _context(state: LLState) -> dict[str, Any]:
        active = state["active"]
        return _identity(active) if isinstance(active, dict) else {}

    def _request_observation(self, state: LLState) -> None:
        if state["awaiting_observation"]:
            self._abort(state, "invalid-observation")
            return
        state["awaiting_observation"] = True
        state["observation_requests"] += 1
        state.outbox.request_observation(self._perceptor_id, **self._context(state))

    def on_first(self, state: LLState) -> LLState:
        self._resolve(state)
        self._request_observation(state)
        return state

    def _dispatch(self, state: LLState, action: int, phase: str) -> None:
        active = state["active"]
        if not isinstance(active, dict) or state["pending_action"] is not None or state["awaiting_observation"]:
            self._abort(state, "invalid-status")
            return
        row = {"phase": phase, K_ACTION: action, K_LEGAL: None, "observation_seq": None}
        active["atomic_rows"].append(row)
        state["pending_action"] = {"phase": phase, K_ACTION: action}
        state["action_requests"] += 1
        state.outbox.request_action(self._actuator_id, **self._context(state), phase=phase, action=action)

    def _finish(self, state: LLState, status: str, targets: Sequence[str] = (), reason: str | None = None) -> None:
        active = state["active"]
        if not isinstance(active, dict):
            return
        spec = TransferSpec.from_mapping(active["transfer"])
        goal = transfer_goal(
            spec,
            status=status,
            **_identity(active),
            based_on_observation_seq=active["dispatch_observation_seq"],
            completion_observation_seq=state["observation_seq"],
            observed_target_facts=targets,
            atomic_rows=active["atomic_rows"],
            failure_reason=reason,
        )
        state["terminal_updates"] += 1
        state.outbox.send_goal_update(self._graph_id, [goal])
        state["active"] = state["pending_action"] = None
        state["awaiting_observation"] = False

    def _abort(self, state: LLState, reason: str) -> None:
        _fail(state, reason)
        if state["active"] is not None:
            self._finish(state, "failed", reason=reason)

    def _advance(self, state: LLState, facts: set[str]) -> None:
        active = state["active"]
        if not isinstance(active, dict):
            return
        spec = TransferSpec.from_mapping(active["transfer"])
        phase = active["phase"]
        if phase == "verify":
            targets = observed_transfer_targets(facts, spec)
            if set(targets) == spec.target_facts and format_fact("hand-empty") in facts:
                self._finish(state, "succeeded", targets)
            else:
                self._abort(state, "target-mismatch")
            return
        if float(getattr(state, "time", 0.0)) >= DURATION - SHUTDOWN_MARGIN:
            self._finish(state, "failed", reason="time_budget_exhausted")
            return
        above = _facts_with(facts, "above")
        holding = _facts_with(facts, "holding")
        if len(above) != 1 or len(above[0]) != 1:
            self._abort(state, "invalid-observation")
            return
        arm = above[0][0]
        if phase in {"pick-up", "navigate-destination", "put-down"}:
            if holding != [(spec.block,)]:
                self._abort(state, "invalid-observation")
            elif arm != spec.destination:
                move = A_MOVE_RIGHT if location_index(arm) < location_index(spec.destination) else A_MOVE_LEFT
                active["phase"] = "navigate-destination"
                self._dispatch(state, move, "navigate-destination")
            elif format_fact("clear", (spec.destination_support,)) not in facts:
                self._abort(state, "invalid-observation")
            else:
                active["phase"] = "put-down"
                self._dispatch(state, A_PUT_DOWN, "put-down")
            return
        if format_fact("hand-empty") not in facts or missing_transfer_preconditions(facts, spec):
            self._abort(state, "invalid-observation")
        elif arm != spec.source:
            move = A_MOVE_RIGHT if location_index(arm) < location_index(spec.source) else A_MOVE_LEFT
            active["phase"] = "navigate-source"
            self._dispatch(state, move, "navigate-source")
        else:
            active["phase"] = "pick-up"
            self._dispatch(state, A_PICK_UP, "pick-up")

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs: Any) -> LLState:
        self._resolve(state)
        if sender != self._perceptor_id or not state["awaiting_observation"] or not isinstance(observation.content, list):
            self._abort(state, "invalid-observation")
            return state
        active = state["active"]
        if isinstance(active, dict) and any(kwargs.get(key) != active[key] for key in IDENTITY_KEYS):
            self._abort(state, "invalid-observation")
            return state
        try:
            beliefs = parse_symbolic_observation(observation.content)
            facts = beliefs_to_facts(beliefs)
        except ValueError as exc:
            self.log(ERROR, str(exc))
            self._abort(state, "invalid-observation")
            return state
        state["awaiting_observation"] = False
        state["observation_seq"] += 1
        state["observations"] += 1
        state["current_facts"] = sorted(facts)
        context = {**self._context(state), "observation_seq": state["observation_seq"]}
        state.outbox.send_beliefs(self._knowledge_id, observation, beliefs, **context)
        if isinstance(active, dict):
            rows = active["atomic_rows"]
            if rows and rows[-1][K_LEGAL] is True and rows[-1]["observation_seq"] is None:
                rows[-1]["observation_seq"] = state["observation_seq"]
            self._advance(state, facts)
        return state

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs: Any) -> LLState:
        self._resolve(state)
        active, pending = state["active"], state["pending_action"]
        raw = action_status.status
        status = _json(raw if isinstance(raw, Mapping) else {})
        matches = (
            sender == self._actuator_id
            and isinstance(active, dict)
            and isinstance(pending, dict)
            and all(status.get(key) == active[key] for key in IDENTITY_KEYS)
            and status.get("phase") == pending["phase"]
            and status.get(K_ACTION) == pending[K_ACTION]
        )
        if not matches:
            self._abort(state, "invalid-status")
            return state
        state["pending_action"] = None
        state["action_statuses"] += 1
        active["atomic_rows"][-1][K_LEGAL] = status.get(K_LEGAL) is True
        if status.get(K_LEGAL) is not True:
            self._abort(state, "illegal-action")
            return state
        if pending["phase"] == "put-down":
            active["phase"] = "verify"
        self._request_observation(state)
        return state

    def on_goal_update(self, state: LLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> LLState:
        self._resolve(state)
        if state["failure"] is not None:
            return state
        try:
            goal = goals[0]
            spec = transfer_from_goal(goal)
            extras = goal.extras
            valid = (
                sender == self._graph_id
                and len(goals) == 1
                and extras.get("status") == "requested"
                and all(extras.get(key) is not None for key in IDENTITY_KEYS)
                and state["active"] is None
                and state["pending_action"] is None
            )
        except (IndexError, ValueError):
            valid = False
        if not valid:
            self._abort(state, "invalid-goal")
            return state
        state["active"] = {
            **_identity(extras),
            "dispatch_observation_seq": int(extras.get("based_on_observation_seq") or 0),
            "transfer": spec.as_dict(),
            "phase": "navigate-source",
            "atomic_rows": [],
        }
        state["transfer_activations"] += 1
        basis = state["active"]["dispatch_observation_seq"]
        if state["current_facts"] and state["observation_seq"] >= basis:
            self._advance(state, set(state["current_facts"]))
        else:
            self._request_observation(state)
        return state


class ClosedWorldKnowledge(KnowledgeBase):
    """Replace complete beliefs and forward each revision to HL."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ll_id = self._hl_id = ""

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        self._ll_id = _one_id(state.directory.internal.ll_reasoning, "LL reasoner")
        self._hl_id = _one_id(state.directory.internal.hl_reasoning, "HL reasoner")
        return state

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation, beliefs: Sequence[Belief], **kwargs: Any) -> KnowledgeState:
        if sender != self._ll_id:
            _fail(state, "invalid-belief-update")
            return state
        state["revisions"] += 1
        state["current_beliefs"] = beliefs_to_dicts(beliefs)
        state["forwards"] += 1
        state.outbox.send_beliefs(self._hl_id, beliefs, **_json(dict(kwargs)), revision=state["revisions"])
        return state


class TransferGoalGraph(GoalGraphBase):
    """Relay one correlated transfer goal in each typed direction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ll_id = self._hl_id = ""

    def on_first(self, state: GoalGraphState) -> GoalGraphState:
        self._ll_id = _one_id(state.directory.internal.ll_reasoning, "LL reasoner")
        self._hl_id = _one_id(state.directory.internal.hl_reasoning, "HL reasoner")
        return state

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> GoalGraphState:
        try:
            goal = goals[0]
            transfer_from_goal(goal)
            context = _identity(goal.extras)
            valid = len(goals) == 1 and len(context) == len(IDENTITY_KEYS)
        except (IndexError, ValueError):
            valid = False
            context = {}
        if not valid:
            _fail(state, "invalid-goal")
        elif sender == self._hl_id and goal.extras.get("status") == "requested" and state["active"] is None:
            state["active"] = context
            state["dispatched"] += 1
            state.outbox.send_goals(self._ll_id, goals)
        elif sender == self._ll_id and context == state["active"] and goal.extras.get("status") in {"succeeded", "failed"}:
            state["terminal"] += 1
            state.outbox.send_goals(self._hl_id, goals)
            state["active"] = None
        else:
            _fail(state, "invalid-goal")
        return state


class HybridBDIReasoner(HLReasonerBase):
    """Plan transfers and reconcile target evidence with complete beliefs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random()
        self._blocks: list[str] = []
        self._planning: PlanningService | None = None
        self._graph_id = self._knowledge_id = ""
        self._goal_limit: int | None = None
        self._fixed_goal: GoalSpec | None = None

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.Random(int(kwargs["seed"]))
        self._blocks = block_names(int(kwargs["num_blocks"]))
        locations = location_names(int(kwargs["table_len"]))
        limit = kwargs.get("goal_completion_limit")
        self._goal_limit = int(limit) if limit is not None else None
        if self._goal_limit is not None and self._goal_limit <= 0:
            raise ValueError("goal_completion_limit must be positive.")
        fixed_goal = kwargs.get("fixed_goal")
        if fixed_goal is not None:
            if not isinstance(fixed_goal, dict):
                raise ValueError("fixed_goal must be a dictionary.")
            self._fixed_goal = GoalSpec(str(fixed_goal["top"]), str(fixed_goal["bottom"]))
        self._planning = PlanningService(
            domain_path=Path(__file__).with_name("blocksworld-transfer-domain.pddl"),
            blocks=self._blocks,
            locations=locations,
            timeout=float(kwargs["planner_timeout"]),
        )

    def on_first(self, state: HLState) -> HLState:
        self._graph_id = _one_id(state.directory.internal.goals, "goal graph")
        self._knowledge_id = _one_id(state.directory.internal.knowledge, "knowledge module")
        state["phase"] = "awaiting-beliefs"
        return state

    @staticmethod
    def _mark_failed(state: HLState, reason: str) -> None:
        _fail(state, reason)
        state["phase"] = "failed"
        current = state["current_hierarchy"]
        if isinstance(current, dict) and current.get("status") == "active":
            current["status"] = "failed"
        _terminate_agent(state, f"2-4-BW failed: {reason}")

    @staticmethod
    def _mark_unsolved(state: HLState, reason: str) -> None:
        """Record a bounded scientific outcome without a runtime failure."""
        state["phase"] = "unsolved"
        state["terminal_reason"] = reason
        hierarchy = state["current_hierarchy"]
        if isinstance(hierarchy, dict):
            hierarchy["status"] = "unsolved"
        _terminate_agent(state, f"2-4-BW unachieved: {reason}")

    def _select_intention(self, state: HLState, facts: set[str]) -> GoalSpec | None:
        options = generate_options(self._blocks, facts, state["completed_intentions"])
        if not options:
            state["completed_intentions"] = []
            options = generate_options(self._blocks, facts)
        if self._fixed_goal is not None:
            return self._fixed_goal if self._fixed_goal in options else None
        return self._rng.choice(options) if options else None

    def _plan(self, state: HLState, facts: set[str], intention: GoalSpec) -> bool:
        assert self._planning is not None
        old = state["current_hierarchy"]
        if isinstance(old, dict) and old.get("status") == "completed":
            state["retained_hierarchy"] = old
        state["hierarchy_counter"] += 1
        hierarchy_id = f"hierarchy-{state['hierarchy_counter']}"
        state["current_hierarchy"] = {
            "hierarchy_id": hierarchy_id,
            "intention": intention.as_dict(),
            "plan": None,
            "execution": [],
            "final_goal_observation_seq": None,
            "status": "active",
        }
        state["phase"] = "deliberating"
        try:
            outcome: PlanningOutcome = self._planning.solve(
                facts, intention, f"exp2_4_{state['observation_seq']}_{state['plan_count'] + 1}"
            )
        except Exception as exc:
            state["planner_failures"] += 1
            self._mark_failed(state, "planning-exception")
            self.log(ERROR, f"Planning failed: {type(exc).__name__}: {exc}")
            return False
        if not outcome.accepted or not outcome.actions:
            state["planner_failures"] += 1
            failures = (outcome.failure or "").split(";")
            if failures and all(item.split(":", 1)[-1] in {
                "planner-did-not-solve", "plan-exceeds-sanity-bound"
            } for item in failures):
                self._mark_unsolved(state, "planner_did_not_solve")
            else:
                self._mark_failed(state, outcome.failure or "planning-failed")
                self.log(ERROR, f"Both planners failed: {outcome.failure}")
            return False
        state["plan_count"] += 1
        transfers = [TransferSpec.from_mapping(action).as_dict() for action in outcome.actions]
        state["current_hierarchy"]["plan"] = {
            "plan_id": f"plan-{state['plan_count']}",
            "engine": outcome.engine,
            "validation": outcome.validation_status,
            "bound": outcome.sanity_bound,
            "transfers": transfers,
        }
        if outcome.engine == "lpg":
            state["lpg_successes"] += 1
        else:
            state["fallback_successes"] += 1
        return True

    def _dispatch(self, state: HLState) -> bool:
        hierarchy = state["current_hierarchy"]
        if not isinstance(hierarchy, dict) or hierarchy.get("status") != "active" or not isinstance(hierarchy.get("plan"), dict):
            return False
        plan = hierarchy["plan"]
        index = len(hierarchy["execution"])
        if index >= len(plan["transfers"]):
            return False
        spec = TransferSpec.from_mapping(plan["transfers"][index])
        if missing_transfer_preconditions(set(state["current_abstract_facts"]), spec):
            state["soundness_failures"] += 1
            return False
        goal_id = f"compound-{state['transfer_dispatches'] + 1}"
        goal = transfer_goal(
            spec,
            status="requested",
            hierarchy_id=hierarchy["hierarchy_id"],
            goal_id=goal_id,
            plan_id=plan["plan_id"],
            step_index=index,
            based_on_observation_seq=state["observation_seq"],
        )
        hierarchy["execution"].append({
            "step_index": index,
            "goal_id": goal_id,
            "dispatch_observation_seq": state["observation_seq"],
            "atomic_rows": [],
            "completion_observation_seq": None,
            "observed_target_facts": [],
            "completion_belief_observation_seq": None,
            "status": "dispatched",
        })
        state["transfer_dispatches"] += 1
        state.outbox.send_goals(self._graph_id, [goal])
        state["phase"] = "awaiting-transfer-result"
        return True

    @staticmethod
    def _pending(state: HLState) -> tuple[dict[str, Any], dict[str, Any]] | None:
        hierarchy = state["current_hierarchy"]
        if not isinstance(hierarchy, dict) or not hierarchy.get("execution"):
            return None
        row = hierarchy["execution"][-1]
        return (hierarchy, row) if row.get("status") in {"dispatched", "awaiting-completion-beliefs"} else None

    def _complete_goal(self, state: HLState, facts: set[str]) -> bool:
        hierarchy = state["current_hierarchy"]
        if not isinstance(hierarchy, dict):
            return False
        intention = GoalSpec(**hierarchy["intention"])
        if intention.fact not in facts:
            return False
        hierarchy["final_goal_observation_seq"] = state["observation_seq"]
        hierarchy["status"] = "completed"
        state["completed_intentions"].append(intention.as_dict())
        state["goal_completions"] += 1
        state["phase"] = "succeeded"
        return True

    def _continue(self, state: HLState, facts: set[str]) -> None:
        current = state["current_hierarchy"]
        if isinstance(current, dict) and current.get("status") == "active":
            plan = current.get("plan")
            if isinstance(plan, dict) and len(current["execution"]) < len(plan["transfers"]):
                if self._dispatch(state):
                    return
                self._mark_failed(state, "dispatch-preconditions-missing")
                return
            if not self._complete_goal(state, facts):
                self._mark_failed(state, "validated-plan-did-not-reach-intention")
                return
        if self._goal_limit is not None and state["goal_completions"] >= self._goal_limit:
            state["phase"] = "completed"
            state["terminal_reason"] = "goal-completion-limit"
            _terminate_agent(state, "2-4-BW forced goal complete")
            return
        if float(getattr(state, "time", 0.0)) >= DURATION - 2 * PLANNER_TIMEOUT - SHUTDOWN_MARGIN:
            self._mark_unsolved(state, "time_budget_exhausted")
            return
        intention = self._select_intention(state, facts)
        if intention is None:
            state["planner_failures"] += 1
            self._mark_failed(state, "no-unsatisfied-intention")
        elif self._plan(state, facts, intention):
            self._dispatch(state)

    def _reconcile(self, state: HLState) -> bool:
        pending = self._pending(state)
        if pending is None or pending[1]["status"] != "awaiting-completion-beliefs":
            return False
        hierarchy, row = pending
        context = state["latest_belief_context"]
        matches = (
            context.get("hierarchy_id") == hierarchy["hierarchy_id"]
            and context.get("goal_id") == row["goal_id"]
            and context.get("plan_id") == hierarchy["plan"]["plan_id"]
            and context.get("step_index") == row["step_index"]
            and int(context.get("observation_seq") or -1) >= int(row["completion_observation_seq"])
        )
        if not matches:
            return False
        row["completion_belief_observation_seq"] = int(context["observation_seq"])
        if row.get("failure_reason") == "time_budget_exhausted":
            row["status"] = "failed"
            state["reconciled_outcomes"] += 1
            self._mark_unsolved(state, "time_budget_exhausted")
            return True
        row["status"] = "completed"
        state["reconciled_outcomes"] += 1
        state["completed_transfers"] += 1
        state["phase"] = "deliberating"
        self._continue(state, set(state["current_abstract_facts"]))
        return True

    def on_belief_update(self, state: HLState, sender: str, beliefs: Sequence[Belief], **kwargs: Any) -> HLState:
        if sender != self._knowledge_id:
            self._mark_failed(state, "invalid-belief-update")
            return state
        if state["phase"] in {"failed", "completed", "unsolved"}:
            return state
        incoming = int(kwargs.get("observation_seq") or state["belief_updates"] + 1)
        if incoming < state["observation_seq"]:
            return state
        facts = project_abstract_facts(beliefs_to_facts(beliefs))
        state["belief_updates"] += 1
        state["observation_seq"] = incoming
        state["current_abstract_facts"] = sorted(facts)
        state["latest_belief_context"] = {**_identity(kwargs), "observation_seq": incoming}
        if not self._reconcile(state) and self._pending(state) is None:
            self._continue(state, facts)
        return state

    def on_goal_update(self, state: HLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> HLState:
        if state["phase"] in {"failed", "completed", "unsolved"}:
            return state
        pending = self._pending(state)
        if sender != self._graph_id or len(goals) != 1 or pending is None:
            self._mark_failed(state, "invalid-terminal-goal")
            return state
        hierarchy, row = pending
        if row["status"] == "awaiting-completion-beliefs":
            return state
        goal = goals[0]
        try:
            spec = transfer_from_goal(goal)
            plan = hierarchy["plan"]
            planned = TransferSpec.from_mapping(plan["transfers"][row["step_index"]])
        except (IndexError, KeyError, TypeError, ValueError):
            self._mark_failed(state, "invalid-terminal-goal")
            return state
        extras = goal.extras
        context = _identity(extras)
        if context != {
            "hierarchy_id": hierarchy["hierarchy_id"],
            "goal_id": row["goal_id"],
            "plan_id": plan["plan_id"],
            "step_index": row["step_index"],
        }:
            self._mark_failed(state, "invalid-terminal-goal")
            return state
        if extras.get("status") == "failed":
            state["compound_failures"] += 1
            if extras.get("failure_reason") == "time_budget_exhausted" and spec == planned:
                row["atomic_rows"] = _json(extras.get("atomic_rows", []))
                row["completion_observation_seq"] = int(extras["completion_observation_seq"])
                row["failure_reason"] = "time_budget_exhausted"
                row["status"] = "awaiting-completion-beliefs"
                state["terminal_results"] += 1
                self._reconcile(state)
            else:
                self._mark_failed(state, str(extras.get("failure_reason") or "transfer-failed"))
            return state
        targets = sorted({str(fact).lower() for fact in extras.get("observed_target_facts", [])})
        completion = int(extras.get("completion_observation_seq") or -1)
        rows = extras.get("atomic_rows")
        if (
            extras.get("status") != "succeeded"
            or spec != planned
            or targets != sorted(planned.target_facts)
            or completion <= int(row["dispatch_observation_seq"])
            or not isinstance(rows, list)
        ):
            self._mark_failed(state, "invalid-terminal-evidence")
            return state
        row["atomic_rows"] = _json(rows)
        row["completion_observation_seq"] = completion
        row["observed_target_facts"] = targets
        row["status"] = "awaiting-completion-beliefs"
        state["terminal_results"] += 1
        state["phase"] = "awaiting-completion-beliefs"
        self._reconcile(state)
        return state


def transfer_initial_states(
    treatment: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return fresh JSON-native states for the six reactive modules."""

    return {
        PERCEPTOR: {"requests": 0, "observations": 0, "pending": None, "failure": None},
        ACTUATOR: {"requests": 0, "statuses": 0, "pending": None, "failure": None},
        LLREASONER: {
            "observation_requests": 0, "observations": 0, "action_requests": 0,
            "action_statuses": 0, "transfer_activations": 0, "terminal_updates": 0,
            "observation_seq": 0, "current_facts": [], "active": None,
            "awaiting_observation": False, "pending_action": None, "failure": None,
        },
        KNOWLEDGE: {"revisions": 0, "forwards": 0, "current_beliefs": [], "failure": None},
        GOALGRAPH: {"active": None, "dispatched": 0, "terminal": 0, "failure": None},
        HLREASONER: {
            "treatment": dict(treatment or {}),
            "phase": "awaiting-beliefs", "observation_seq": 0, "belief_updates": 0,
            "transfer_dispatches": 0, "terminal_results": 0, "reconciled_outcomes": 0,
            "current_abstract_facts": [], "latest_belief_context": {},
            "completed_intentions": [], "goal_completions": 0, "terminal_reason": None,
            "hierarchy_counter": 0, "plan_count": 0, "current_hierarchy": None,
            "retained_hierarchy": None, "completed_transfers": 0, "lpg_successes": 0,
            "fallback_successes": 0, "planner_failures": 0, "compound_failures": 0,
            "soundness_failures": 0, "failure": None,
        },
    }


_initial_states = transfer_initial_states
