"""Six compact MHAgentA module behaviors for experiment 2-5-BW."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from importlib import import_module
from logging import ERROR
from pathlib import Path
import random
from time import perf_counter
from typing import Any, cast

import numpy as np
from mhagenta import ActionStatus, Belief, Goal, Observation
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.states import (
    ActuatorState,
    GoalGraphState,
    HLState,
    KnowledgeState,
    LLState,
    PerceptorState,
)

from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR

from .contracts import (
    GoalSpec,
    TransferSpec,
    beliefs_to_dicts,
    beliefs_to_facts,
    block_names,
    generate_options,
    goal_to_dict,
    location_names,
    project_abstract_facts,
    transfer_from_goal,
    transfer_goal,
)
from .environment import (
    A_CLOSE,
    ATOMIC_ACTIONS,
    K_ACTION,
    K_ACTION_CONTEXT,
    K_LEGAL,
)
from .grounding import ground_observation, transfer_succeeded
from .planning import PlanningOutcome, PlanningService
from .policy import (
    MAX_POLICY_STEPS,
    array_sha256,
    artifact_paths,
    greedy_inference,
    load_policy_checkpoint,
    validate_manifest,
)


def _terminate_agent(state: Any, reason: str) -> None:
    """Request termination when running with a framework-backed outbox."""

    terminate = getattr(state.outbox, "terminate_agent", None)
    if callable(terminate):
        terminate(reason)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _single_id(entries: Iterable[Any], label: str) -> str:
    module_ids = [entry.module_id for entry in entries]
    if len(module_ids) != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {module_ids!r}.")
    return cast(str, module_ids[0])


@contextmanager
def _active_time(state: Any):
    started = perf_counter()
    try:
        yield
    finally:
        state["active_seconds"] += perf_counter() - started


def _fail(state: Any, code: str, **details: Any) -> None:
    """Record only the first compact runtime failure."""

    if state["failure"] is None:
        state["failure"] = {"code": code, **_json_value(details)}


def _identity(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_value(data[key])
        for key in ("goal_id", "plan_id", "step_index")
        if key in data
    }


class BlocksWorldPerceptor(RMQPerceptorBase):
    """Forward one correlated complete observation at a time to the neural LL."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""

    def _resolve_ids(self, state: PerceptorState) -> None:
        if not self._env_id:
            environment = state.directory.external.environment
            if environment is None:
                raise RuntimeError("No Blocks World environment is registered.")
            self._env_id = environment.address["env_id"]
        if not self._ll_id:
            self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LL reasoner")

    def on_first(self, state: PerceptorState) -> PerceptorState:
        self._resolve_ids(state)
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._ll_id:
                _fail(state, "unexpected-observation-request-sender", sender=sender)
            elif state["pending"] is not None:
                _fail(state, "overlapping-observation-request")
            else:
                state["pending"] = _json_value(dict(kwargs))
                state["request_count"] += 1
                self.observe(self._env_id)
            return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
            pending = state["pending"]
            if env_id != self._env_id:
                _fail(state, "unexpected-environment", sender=env_id)
                return state
            if not isinstance(pending, dict):
                _fail(state, "observation-without-request")
                return state
            observation_id = kwargs.get("observation_id")
            if not isinstance(observation_id, int) or observation_id <= state["last_observation_id"]:
                _fail(state, "invalid-observation-id", observation_id=observation_id)
                return state
            state["observation_count"] += 1
            state["last_observation_id"] = observation_id
            state["pending"] = None
            state.outbox.send_observation(
                self._ll_id,
                Observation(kwargs.get("observation")),
                observation_id=observation_id,
                **pending,
            )
            return state


class BlocksWorldActuator(RMQActuatorBase):
    """Execute LL-originated atomic actions and return compact correlated status."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""

    def _resolve_ids(self, state: ActuatorState) -> None:
        if not self._env_id:
            environment = state.directory.external.environment
            if environment is None:
                raise RuntimeError("No Blocks World environment is registered.")
            self._env_id = environment.address["env_id"]
        if not self._ll_id:
            self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LL reasoner")

    def on_first(self, state: ActuatorState) -> ActuatorState:
        self._resolve_ids(state)
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
            action = kwargs.get(K_ACTION)
            if sender != self._ll_id:
                _fail(state, "unexpected-action-request-sender", sender=sender)
            elif state["pending"] is not None:
                _fail(state, "overlapping-action-request")
            elif not isinstance(action, int) or action not in ATOMIC_ACTIONS:
                _fail(state, "invalid-atomic-action", action=action)
            else:
                state["pending"] = {
                    **_identity(dict(kwargs)),
                    "atomic_action_id": kwargs.get("atomic_action_id"),
                    K_ACTION: action,
                }
                state["request_count"] += 1
                self.act(self._env_id, action=action)
            return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        with _active_time(state):
            pending = state["pending"]
            if env_id != self._env_id or not isinstance(pending, dict):
                _fail(state, "action-status-without-request", sender=env_id)
                return state
            status = {**pending, **_json_value(dict(kwargs))}
            state["status_count"] += 1
            if status.get(K_LEGAL) is True:
                state["successful_status_count"] += 1
            else:
                _fail(state, "unsuccessful-action-status", action_id=pending.get("atomic_action_id"))
            state["pending"] = None
            state.outbox.send_status(self._ll_id, ActionStatus(status))
            return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        if self._env_id:
            self.act(self._env_id, action=A_CLOSE)
        return state


class ClosedWorldKnowledge(KnowledgeBase):
    """Replace complete beliefs and forward each correlated complete revision."""

    def on_observed_beliefs(
        self,
        state: KnowledgeState,
        sender: str,
        observation: Observation,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> KnowledgeState:
        with _active_time(state):
            state["revision_count"] += 1
            state["current_beliefs"] = beliefs_to_dicts(beliefs)
            state["forwarded_count"] += 1
            correlation = {
                key: kwargs.get(key)
                for key in ("goal_id", "plan_id", "step_index", "observation_id")
                if kwargs.get(key) is not None
            }
            for reasoner in state.directory.internal.hl_reasoning:
                state.outbox.send_beliefs(
                    reasoner.module_id,
                    beliefs,
                    revision=state["revision_count"],
                    **correlation,
                )
            return state

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs: Any) -> KnowledgeState:
        with _active_time(state):
            state["request_count"] += 1
            return state


class TransferGoalGraph(GoalGraphBase):
    """Relay one active transfer goal while retaining only counts and identity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ll_id = ""
        self._hl_id = ""

    def _resolve_ids(self, state: GoalGraphState) -> None:
        if not self._ll_id:
            self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LL reasoner")
        if not self._hl_id:
            self._hl_id = _single_id(state.directory.internal.hl_reasoning, "HL reasoner")

    def on_first(self, state: GoalGraphState) -> GoalGraphState:
        self._resolve_ids(state)
        return state

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs: Any) -> GoalGraphState:
        with _active_time(state):
            state["request_count"] += 1
            return state

    def on_goal_update(
        self,
        state: GoalGraphState,
        sender: str,
        goals: Sequence[Goal],
        **kwargs: Any,
    ) -> GoalGraphState:
        with _active_time(state):
            self._resolve_ids(state)
            if len(goals) != 1:
                _fail(state, "invalid-goal-count", count=len(goals))
                return state
            goal = goals[0]
            try:
                transfer_from_goal(goal)
            except ValueError as exc:
                _fail(state, "invalid-transfer-goal", message=str(exc))
                return state
            goal_id = goal.extras.get("goal_id")
            status = goal.extras.get("status")
            if sender == self._hl_id:
                if status != "requested" or state["active_goal_id"] is not None:
                    _fail(state, "invalid-transfer-dispatch", goal_id=goal_id)
                    return state
                state["active_goal_id"] = goal_id
                state["dispatch_count"] += 1
                state.outbox.send_goals(self._ll_id, goals)
            elif sender == self._ll_id:
                if goal_id != state["active_goal_id"] or status not in {"succeeded", "failed"}:
                    _fail(state, "invalid-terminal-transfer", goal_id=goal_id)
                    return state
                state["terminal_count"] += 1
                if status == "failed":
                    state["failed_count"] += 1
                state.outbox.send_goals(self._hl_id, goals)
                state["active_goal_id"] = None
            else:
                _fail(state, "unexpected-goal-sender", sender=sender)
            return state


class NeuralTransferLLReasoner(LLReasonerBase):
    """Realize each transfer with the unchanged frozen neural policy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._torch: Any = None
        self._model: Any = None
        self._manifest: dict[str, Any] = {}
        self._observation: np.ndarray | None = None
        self._perceptor_id = ""
        self._actuator_id = ""
        self._knowledge_id = ""
        self._goal_graph_id = ""

    def on_init(self, **kwargs: Any) -> None:
        self._torch = import_module("torch")
        manifest_path, checkpoint_path = artifact_paths()
        self._manifest = validate_manifest(manifest_path, checkpoint_path)
        self._model, checkpoint = load_policy_checkpoint(self._torch, checkpoint_path)
        if checkpoint["weight_optimizer_steps"] != self._manifest["training"]["selected_optimizer_steps"]:
            raise ValueError("Checkpoint and manifest optimizer-step counts do not match.")

    def _resolve_ids(self, state: LLState) -> None:
        if not self._perceptor_id:
            self._perceptor_id = _single_id(state.directory.internal.perception, "perceptor")
        if not self._actuator_id:
            self._actuator_id = _single_id(state.directory.internal.actuation, "actuator")
        if not self._knowledge_id:
            self._knowledge_id = _single_id(state.directory.internal.knowledge, "knowledge module")
        if not self._goal_graph_id:
            self._goal_graph_id = _single_id(state.directory.internal.goals, "goal graph")

    def _request_observation(self, state: LLState) -> None:
        if state["awaiting_observation"]:
            _fail(state, "overlapping-observation-request")
            return
        state["awaiting_observation"] = True
        state["observation_request_count"] += 1
        context: dict[str, Any] = {}
        if state["active_transfer"] is not None:
            context = _identity(state["active_transfer"])
        state.outbox.request_observation(
            self._perceptor_id,
            **({K_ACTION_CONTEXT: context} if context else {}),
        )

    def on_first(self, state: LLState) -> LLState:
        self._resolve_ids(state)
        state["policy_loaded"] = True
        state["policy_id"] = self._manifest["architecture"]
        state["checkpoint_sha256"] = self._manifest["checkpoint_sha256"]
        self._request_observation(state)
        return state

    def _infer(self, state: LLState) -> None:
        active = state["active_transfer"]
        if not isinstance(active, dict) or self._observation is None:
            _fail(state, "inference-without-transfer-or-observation")
            return
        if state["pending_action"] is not None or state["awaiting_observation"]:
            _fail(state, "overlapping-atomic-request")
            return
        spec = TransferSpec.from_mapping(active["transfer"])
        action, q_values, model_input = greedy_inference(
            self._torch,
            self._model,
            self._observation,
            spec,
        )
        state["atomic_action_counter"] += 1
        atomic_id = f"atomic-{state['atomic_action_counter']}"
        row = {
            "atomic_action_id": atomic_id,
            "observation_id": state["observation_id"],
            "input_sha256": array_sha256(model_input),
            "selected_action": int(action),
            "q_values": [float(value) for value in q_values],
            "legal": None,
        }
        active["atomic"].append(row)
        pending = {
            **_identity(active),
            "atomic_action_id": atomic_id,
            K_ACTION: int(action),
        }
        state["pending_action"] = pending
        state["inference_count"] += 1
        state["atomic_request_count"] += 1
        state.outbox.request_action(self._actuator_id, **pending)

    def _finish(self, state: LLState, facts: set[str], status: str, reason: str | None = None) -> None:
        active = state["active_transfer"]
        if not isinstance(active, dict):
            return
        spec = TransferSpec.from_mapping(active["transfer"])
        terminal = transfer_goal(
            spec,
            status=status,
            goal_id=active["goal_id"],
            plan_id=active["plan_id"],
            step_index=active["step_index"],
            based_on_observation_seq=active["based_on_observation_id"],
            completion_observation_seq=state["observation_id"],
            observed_facts=sorted(spec.target_facts & facts),
            failure_reason=reason,
            atomic_rows=active["atomic"],
        )
        state["completed_transfer_count"] += int(status == "succeeded")
        state["failed_transfer_count"] += int(status != "succeeded")
        state.outbox.send_goal_update(self._goal_graph_id, [terminal])
        state["active_transfer"] = None
        state["pending_action"] = None
        state["failure_pending"] = None

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
                _fail(state, "unexpected-observation-sender", sender=sender)
                return state
            observation_id = kwargs.get("observation_id")
            if not isinstance(observation_id, int) or observation_id <= state["observation_id"]:
                _fail(state, "invalid-observation-id", observation_id=observation_id)
                return state
            try:
                grounded = ground_observation(observation.content)
            except (TypeError, ValueError) as exc:
                _fail(state, "invalid-numeric-observation", message=str(exc))
                return state
            self._observation = grounded.observation
            state["awaiting_observation"] = False
            state["observation_count"] += 1
            state["observation_id"] = observation_id
            state["belief_count"] += len(grounded.beliefs)
            active = state["active_transfer"]
            correlation = _identity(active) if isinstance(active, dict) else {}
            state.outbox.send_beliefs(
                self._knowledge_id,
                observation,
                grounded.beliefs,
                observation_id=observation_id,
                **correlation,
            )
            if not isinstance(active, dict):
                return state
            if state["pending_action"] is not None:
                _fail(state, "observation-while-action-pending")
                return state
            facts = set(grounded.facts)
            if transfer_succeeded(facts, TransferSpec.from_mapping(active["transfer"])):
                self._finish(state, facts, "succeeded")
            elif state["failure_pending"] is not None:
                self._finish(state, facts, "failed", str(state["failure_pending"]))
            elif len(active["atomic"]) >= MAX_POLICY_STEPS:
                self._finish(state, facts, "failed", "policy-step-limit")
            else:
                self._infer(state)
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
            pending = state["pending_action"]
            raw = action_status.status
            status = _json_value(raw if isinstance(raw, dict) else {"status": raw})
            if sender != self._actuator_id or not isinstance(pending, dict):
                _fail(state, "unexpected-action-status", sender=sender)
                return state
            if status.get("atomic_action_id") != pending["atomic_action_id"] or any(
                status.get(key) != pending.get(key) for key in ("goal_id", "plan_id", "step_index")
            ):
                _fail(state, "action-status-correlation-mismatch")
                return state
            active = state["active_transfer"]
            if not isinstance(active, dict) or not active["atomic"]:
                _fail(state, "status-without-canonical-row")
                return state
            legal = status.get(K_LEGAL) is True
            active["atomic"][-1]["legal"] = legal
            state["status_count"] += 1
            if not legal:
                state["failure_pending"] = "illegal-neural-action"
            state["pending_action"] = None
            self._request_observation(state)
            return state

    def on_goal_update(
        self,
        state: LLState,
        sender: str,
        goals: Sequence[Goal],
        **kwargs: Any,
    ) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._goal_graph_id or len(goals) != 1:
                _fail(state, "invalid-goal-update", sender=sender, count=len(goals))
                return state
            try:
                spec = transfer_from_goal(goals[0])
            except ValueError as exc:
                _fail(state, "invalid-transfer-goal", message=str(exc))
                return state
            extras = goals[0].extras
            if extras.get("status") != "requested" or state["active_transfer"] is not None:
                _fail(state, "transfer-while-active")
                return state
            state["active_transfer"] = {
                **_identity(extras),
                "based_on_observation_id": int(extras["based_on_observation_seq"]),
                "transfer": spec.as_dict(),
                "atomic": [],
            }
            state["activated_transfer_count"] += 1
            based_on = int(extras["based_on_observation_seq"])
            if self._observation is None or based_on > state["observation_id"]:
                self._request_observation(state)
            else:
                self._infer(state)
            return state


class RepeatedGoalHLReasoner(HLReasonerBase):
    """Plan repeated stacking goals and reconcile terminal and belief evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random()
        self._blocks: list[str] = []
        self._planning: PlanningService | None = None
        self._goal_graph_id = ""
        self._completion_limit = 5
        self._fixed_goal: GoalSpec | None = None

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.Random(int(kwargs["seed"]))
        self._blocks = block_names(int(kwargs["num_blocks"]))
        locations = location_names(int(kwargs["table_len"]))
        self._completion_limit = int(kwargs.get("goal_completion_limit", 5))
        if self._completion_limit <= 0:
            raise ValueError("goal_completion_limit must be positive.")
        fixed_goal = kwargs.get("fixed_goal")
        if fixed_goal is not None:
            if not isinstance(fixed_goal, dict):
                raise ValueError("fixed_goal must be a dictionary.")
            self._fixed_goal = GoalSpec(
                str(fixed_goal["top"]), str(fixed_goal["bottom"])
            )
        self._planning = PlanningService(
            domain_path=Path(__file__).resolve().with_name("blocksworld-transfer-domain.pddl"),
            blocks=self._blocks,
            locations=locations,
            timeout=float(kwargs["planner_timeout"]),
        )

    def on_first(self, state: HLState) -> HLState:
        self._goal_graph_id = _single_id(state.directory.internal.goals, "goal graph")
        state["completion_goal_limit"] = self._completion_limit
        state["phase"] = "awaiting-beliefs"
        return state

    def _current_run(self, state: HLState) -> dict[str, Any] | None:
        if state["intention"] is None or not state["goal_runs"]:
            return None
        return state["goal_runs"][-1]

    def _current_transfer(self, state: HLState) -> dict[str, Any] | None:
        run = self._current_run(state)
        if run is None or run["plan"] is None:
            return None
        index = state["plan_index"]
        transfers = run["plan"]["transfers"]
        return transfers[index] if 0 <= index < len(transfers) else None

    def _set_failed(self, state: HLState, code: str, **details: Any) -> None:
        _fail(state, code, **details)
        state["phase"] = "failed"
        self.log(ERROR, f"Experiment 2-5-BW failed: {code}")
        _terminate_agent(state, f"2-5-BW failed: {code}")

    def _select_intention(self, state: HLState, facts: set[str]) -> bool:
        options = generate_options(self._blocks, facts, state["goal_filter"])
        if not options:
            state["goal_filter"] = []
            options = generate_options(self._blocks, facts)
        if not options:
            self._set_failed(state, "no-unsatisfied-stacking-goal")
            return False
        if self._fixed_goal is not None:
            if self._fixed_goal not in options:
                self._set_failed(state, "fixed-goal-unavailable")
                return False
            intention = self._fixed_goal
        else:
            intention = self._rng.choice(options)
        state["intention"] = intention.as_dict()
        state["goal_runs"].append(
            {
                "goal": intention.as_dict(),
                "plan": None,
                "final_goal_fact": None,
                "final_observation_id": None,
                "status": "active",
            }
        )
        return True

    def _plan(self, state: HLState, facts: set[str]) -> bool:
        assert self._planning is not None
        intention = GoalSpec(**state["intention"])
        state["phase"] = "deliberating"
        try:
            outcome: PlanningOutcome = self._planning.solve(
                facts,
                intention,
                f"exp2_5_{state['observation_id']}_{state['plan_count'] + 1}",
            )
        except Exception as exc:
            self._set_failed(state, "planning-exception", message=f"{type(exc).__name__}: {exc}")
            return False
        if not outcome.accepted or not outcome.actions:
            state["planner_failure_count"] += 1
            self._set_failed(state, "planning-failed", attempts=outcome.attempts)
            return False
        state["plan_count"] += 1
        plan_id = f"plan-{state['plan_count']}"
        transfers = []
        for action in outcome.actions:
            spec = TransferSpec.from_mapping(action)
            transfers.append(
                {
                    "spec": spec.as_dict(),
                    "goal_id": None,
                    "target_facts": sorted(spec.target_facts),
                    "terminal_observation_id": None,
                    "revision_observation_id": None,
                    "atomic": [],
                }
            )
        run = self._current_run(state)
        assert run is not None
        run["plan"] = {
            "plan_id": plan_id,
            "engine": outcome.engine,
            "fallback_used": outcome.engine != "lpg",
            "validation_status": outcome.validation_status,
            "transfers": transfers,
        }
        state["current_plan_id"] = plan_id
        state["plan_index"] = 0
        if outcome.engine == "lpg":
            state["lpg_success_count"] += 1
        else:
            state["fallback_success_count"] += 1
            state["planner_degraded"] = True
        return True

    def _dispatch(self, state: HLState) -> bool:
        transfer = self._current_transfer(state)
        if transfer is None:
            self._set_failed(state, "missing-plan-transfer")
            return False
        state["transfer_dispatch_count"] += 1
        goal_id = f"compound-{state['transfer_dispatch_count']}"
        transfer["goal_id"] = goal_id
        spec = TransferSpec.from_mapping(transfer["spec"])
        goal = transfer_goal(
            spec,
            status="requested",
            goal_id=goal_id,
            plan_id=state["current_plan_id"],
            step_index=state["plan_index"],
            based_on_observation_seq=state["observation_id"],
        )
        state["pending_goal"] = {
            "goal_id": goal_id,
            "plan_id": state["current_plan_id"],
            "step_index": state["plan_index"],
        }
        state["pending_terminal"] = None
        state["pending_revision"] = None
        state["phase"] = "awaiting-transfer-evidence"
        state.outbox.send_goals(self._goal_graph_id, [goal])
        return True

    def _finish_goal_if_reached(self, state: HLState, facts: set[str]) -> bool:
        if state["intention"] is None:
            return False
        intention = GoalSpec(**state["intention"])
        if intention.fact not in facts:
            return False
        run = self._current_run(state)
        assert run is not None
        run["final_goal_fact"] = intention.fact
        run["final_observation_id"] = state["observation_id"]
        run["status"] = "succeeded"
        state["goal_filter"].append(intention.as_dict())
        state["goal_completion_count"] += 1
        state["intention"] = None
        state["current_plan_id"] = None
        state["plan_index"] = 0
        return True

    def _continue(self, state: HLState) -> None:
        facts = set(state["current_abstract_facts"])
        self._finish_goal_if_reached(state, facts)
        if state["goal_completion_count"] >= self._completion_limit:
            state["phase"] = "completed"
            state["terminal_reason"] = "goal-completion-limit"
            _terminate_agent(state, "2-5-BW goal treatment complete")
            return
        if state["intention"] is None and not self._select_intention(state, facts):
            return
        if self._current_run(state)["plan"] is None and not self._plan(state, facts):
            return
        self._dispatch(state)

    @staticmethod
    def _matches_pending(pending: dict[str, Any], data: dict[str, Any]) -> bool:
        return all(data.get(key) == pending.get(key) for key in ("goal_id", "plan_id", "step_index"))

    def _reconcile(self, state: HLState) -> None:
        terminal = state["pending_terminal"]
        revision = state["pending_revision"]
        pending = state["pending_goal"]
        if not all(isinstance(item, dict) for item in (terminal, revision, pending)):
            return
        if not self._matches_pending(pending, terminal) or not self._matches_pending(pending, revision):
            self._set_failed(state, "completion-correlation-mismatch")
            return
        if revision["observation_id"] < terminal["observation_id"]:
            return
        transfer = self._current_transfer(state)
        if transfer is None:
            self._set_failed(state, "completion-without-transfer")
            return
        expected = set(transfer["target_facts"])
        if set(terminal["target_facts"]) != expected or not expected.issubset(state["current_abstract_facts"]):
            self._set_failed(state, "transfer-target-mismatch")
            return
        transfer["terminal_observation_id"] = terminal["observation_id"]
        transfer["revision_observation_id"] = revision["observation_id"]
        transfer["atomic"] = terminal["atomic"]
        state["completed_transfer_count"] += 1
        state["last_completed_goal_id"] = pending["goal_id"]
        state["plan_index"] += 1
        state["pending_goal"] = None
        state["pending_terminal"] = None
        state["pending_revision"] = None
        run = self._current_run(state)
        assert run is not None
        if state["plan_index"] < len(run["plan"]["transfers"]):
            self._dispatch(state)
        else:
            state["phase"] = "deliberating"
            self._continue(state)

    def on_belief_update(
        self,
        state: HLState,
        sender: str,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> HLState:
        with _active_time(state):
            observation_id = kwargs.get("observation_id")
            if not isinstance(observation_id, int):
                _fail(state, "missing-belief-observation-id")
                return state
            if observation_id < state["observation_id"]:
                state["stale_revision_count"] += 1
                return state
            facts = beliefs_to_facts(beliefs)
            state["belief_update_count"] += 1
            state["observation_id"] = observation_id
            state["current_abstract_facts"] = sorted(project_abstract_facts(facts))
            pending = state["pending_goal"]
            incoming = {
                key: kwargs.get(key)
                for key in ("goal_id", "plan_id", "step_index")
            }
            if isinstance(pending, dict) and self._matches_pending(pending, incoming):
                revision = {**incoming, "observation_id": observation_id}
                existing = state["pending_revision"]
                if existing is None or existing == revision:
                    state["pending_revision"] = revision
                elif observation_id >= existing["observation_id"]:
                    state["pending_revision"] = revision
                self._reconcile(state)
            elif pending is None and state["phase"] not in {"failed", "completed"}:
                self._continue(state)
            return state

    def on_goal_update(
        self,
        state: HLState,
        sender: str,
        goals: Sequence[Goal],
        **kwargs: Any,
    ) -> HLState:
        with _active_time(state):
            if sender != self._goal_graph_id or len(goals) != 1:
                _fail(state, "invalid-terminal-delivery", sender=sender, count=len(goals))
                return state
            try:
                spec = transfer_from_goal(goals[0])
            except ValueError as exc:
                _fail(state, "invalid-terminal-goal", message=str(exc))
                return state
            extras = goals[0].extras
            pending = state["pending_goal"]
            incoming = _identity(extras)
            if not isinstance(pending, dict):
                if incoming.get("goal_id") == state["last_completed_goal_id"]:
                    return state
                _fail(state, "terminal-without-pending-goal")
                return state
            if not self._matches_pending(pending, incoming):
                _fail(state, "terminal-correlation-mismatch")
                return state
            target_facts = sorted(str(fact).lower() for fact in extras.get("observed_facts", []))
            observation_id = extras.get("completion_observation_seq")
            atomic = _json_value(extras.get("atomic", []))
            terminal = {
                **incoming,
                "observation_id": observation_id,
                "target_facts": target_facts,
                "atomic": atomic,
            }
            if extras.get("status") != "succeeded" or not isinstance(observation_id, int):
                self._set_failed(state, "transfer-failed", reason=extras.get("failure_reason"))
                return state
            expected = self._current_transfer(state)
            if expected is None or set(target_facts) != spec.target_facts or set(target_facts) != set(expected["target_facts"]):
                self._set_failed(state, "terminal-target-mismatch")
                return state
            existing = state["pending_terminal"]
            if existing is None:
                state["pending_terminal"] = terminal
            elif existing != terminal:
                self._set_failed(state, "conflicting-terminal-duplicate")
                return state
            self._reconcile(state)
            return state


def initial_states(
    treatment: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return fresh declared JSON-native states for all six modules."""

    return {
        PERCEPTOR: {
            "active_seconds": 0.0,
            "request_count": 0,
            "observation_count": 0,
            "last_observation_id": 0,
            "pending": None,
            "failure": None,
        },
        ACTUATOR: {
            "active_seconds": 0.0,
            "request_count": 0,
            "status_count": 0,
            "successful_status_count": 0,
            "pending": None,
            "failure": None,
        },
        LLREASONER: {
            "active_seconds": 0.0,
            "policy_loaded": False,
            "policy_id": "",
            "checkpoint_sha256": "",
            "observation_request_count": 0,
            "observation_count": 0,
            "observation_id": 0,
            "awaiting_observation": False,
            "belief_count": 0,
            "activated_transfer_count": 0,
            "completed_transfer_count": 0,
            "failed_transfer_count": 0,
            "inference_count": 0,
            "atomic_request_count": 0,
            "status_count": 0,
            "atomic_action_counter": 0,
            "active_transfer": None,
            "pending_action": None,
            "failure_pending": None,
            "failure": None,
        },
        KNOWLEDGE: {
            "active_seconds": 0.0,
            "revision_count": 0,
            "forwarded_count": 0,
            "request_count": 0,
            "current_beliefs": [],
            "failure": None,
        },
        GOALGRAPH: {
            "active_seconds": 0.0,
            "active_goal_id": None,
            "request_count": 0,
            "dispatch_count": 0,
            "terminal_count": 0,
            "failed_count": 0,
            "failure": None,
        },
        HLREASONER: {
            "treatment": dict(treatment or {}),
            "active_seconds": 0.0,
            "phase": "awaiting-beliefs",
            "terminal_reason": None,
            "completion_goal_limit": 5,
            "belief_update_count": 0,
            "stale_revision_count": 0,
            "observation_id": 0,
            "current_abstract_facts": [],
            "intention": None,
            "goal_filter": [],
            "goal_runs": [],
            "goal_completion_count": 0,
            "plan_count": 0,
            "current_plan_id": None,
            "plan_index": 0,
            "pending_goal": None,
            "pending_terminal": None,
            "pending_revision": None,
            "last_completed_goal_id": None,
            "transfer_dispatch_count": 0,
            "completed_transfer_count": 0,
            "lpg_success_count": 0,
            "fallback_success_count": 0,
            "planner_degraded": False,
            "planner_failure_count": 0,
            "failure": None,
        },
    }
