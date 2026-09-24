"""Six reactive MHAgentA behaviors for frozen-policy survival and crafting."""
from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from logging import INFO
from time import perf_counter
from typing import Any, cast

import numpy as np
from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)
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

from .beliefs import (
    Direction,
    abstract_beliefs,
    available_movement_actions,
    cell_key,
    initial_belief_state,
    known_targets,
    movement_evidence,
    nearest_target,
    novel_movement_actions,
    parse_abstract_beliefs,
    position_from_key,
    revise_belief_state,
)
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
    MAX_TOTAL_ACTIONS,
    ActivityId,
    ActivitySpec,
    activity_from_goal,
    as_rgb_frame,
    goal_from_dict,
    goal_to_dict,
    make_activity_goal,
    passive_goal,
    resource_collected,
    resource_stagnated,
    rgb_sha256,
    terminal_goal,
    validate_requested_goal,
)
from .cow_tracking import associate_cow, consumed_tracked_cow
from .cow_search import discovery_goal
from .deliberation import (
    choose_step,
    failure_key,
    recovery_complete,
    routes,
    urgent_need,
)
from .grounding import (
    GROUNDING_MANIFEST_FILENAME,
    GroundingTemplates,
    ground_observation,
    load_grounding_templates,
    resolve_active_grounding_bundle,
)
from .policy import (
    MANIFEST_FILENAME,
    POLICY_FILENAMES,
    PolicyId,
    encode_context,
    file_sha256,
    inference_evidence,
    load_policy_bundle,
    selection_actions,
)

PUBLIC_STATUS = (K_REWARD, K_DONE, K_DEAD, K_ILLEGAL_ACTION, K_LETHAL_MOVEMENT, K_NEW_ACHIEVEMENTS, K_CONTRACT_ERROR)
ACTION_CONTEXT = (K_ATOMIC_ID, K_OWNER_ID, K_REQUESTER, K_ACTION)


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _single_id(entries: Iterable[Any], label: str) -> str:
    ids = [entry.module_id for entry in entries]
    if len(ids) != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {ids!r}.")
    return cast(str, ids[0])


@contextmanager
def _active_time(state: Any):
    started = perf_counter()
    try:
        yield
    finally:
        state["active_seconds"] += perf_counter() - started


def _fail(state: Any, code: str, **details: Any) -> None:
    """Persist the first contract fault and stop the agent without another action."""

    if state["failure"] is None:
        state["failure"] = {"code": code, **details}
    state.outbox.terminate_agent(code)


def valid_status(status: Mapping[str, Any]) -> bool:
    """Validate the complete public status before any belief revision."""

    reward = status.get(K_REWARD)
    return (
        isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
        and all(type(status.get(key)) is bool for key in (K_DONE, K_DEAD, K_ILLEGAL_ACTION, K_LETHAL_MOVEMENT))
        and isinstance(status.get(K_NEW_ACHIEVEMENTS), list)
        and all(isinstance(name, str) for name in status[K_NEW_ACHIEVEMENTS])
        and status.get(K_CONTRACT_ERROR) is None
    )


class CrafterRGBPerceptor(RMQPerceptorBase):
    """Forward correlated RGB observations without persisting their pixels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""

    def _resolve(self, state: PerceptorState) -> None:
        if not self._env_id:
            environment = state.directory.external.environment
            if environment is None:
                raise RuntimeError("No Crafter environment is registered.")
            self._env_id = environment.address["env_id"]
        if not self._ll_id:
            self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LL reasoner")

    def on_first(self, state: PerceptorState) -> PerceptorState:
        self._resolve(state)
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            self._resolve(state)
            observation_id = kwargs.get(K_OBSERVATION_ID)
            if sender != self._ll_id:
                _fail(state, "unexpected-observation-requester", sender=sender)
            elif state["pending_observation_id"] is not None:
                _fail(state, "overlapping-observation-request")
            elif type(observation_id) is not int or observation_id != state["request_count"] + 1:
                _fail(state, "invalid-observation-id", observation_id=observation_id)
            else:
                state["pending_observation_id"] = observation_id
                state["request_count"] += 1
                self.observe(self._env_id, **{K_OBSERVATION_ID: observation_id})
            return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            self._resolve(state)
            pending = state["pending_observation_id"]
            error = kwargs.get(K_CONTRACT_ERROR)
            try:
                frame = as_rgb_frame(kwargs.get(K_OBSERVATION))
                digest = rgb_sha256(frame)
            except (TypeError, ValueError) as exc:
                frame, digest = np.empty((0, ), dtype=np.uint8), None
                error = error or f"invalid-rgb:{exc}"
            if env_id != self._env_id or pending is None:
                error = error or "observation-without-request"
            elif kwargs.get(K_OBSERVATION_ID) != pending:
                error = error or "observation-id-mismatch"
            elif digest != kwargs.get(K_OBSERVATION_DIGEST):
                error = error or "observation-digest-mismatch"
            if error is not None:
                _fail(state, "perception-pipeline-error", message=str(error))
            state["pending_observation_id"] = None
            state["observation_count"] += 1
            state["last_observation_sha256"] = digest
            state.outbox.send_observation(
                self._ll_id,
                Observation(frame.tolist(), observation_type="crafter-rgb"),
                **{
                    K_OBSERVATION_ID: pending,
                    K_OBSERVATION_DIGEST: digest,
                    K_CONTRACT_ERROR: error,
                },
            )
            return state


class CrafterActivityActuator(RMQActuatorBase):
    """Serialize both owners' actions and allocate the environment sequence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = self._ll_id = self._hl_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        """Resolve the single environment and both action owners."""

        self._env_id = state.directory.external.environment.address["env_id"]
        self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LLR")
        self._hl_id = _single_id(state.directory.internal.hl_reasoning, "HLR")
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        """Dispatch one validated request, or return a correlated contract failure."""

        with _active_time(state):
            owner, action, kind = kwargs.get(K_OWNER_ID), kwargs.get(K_ACTION), kwargs.get(K_REQUESTER)
            expected = "ll_policy" if sender == self._ll_id else "hl_primitive" if sender == self._hl_id else None
            valid = (
                expected is not None and kind == expected and isinstance(owner, str) and bool(owner)
                and owner not in state["owner_ids"]
                and set(kwargs) == {K_OWNER_ID, K_ACTION, K_REQUESTER}
                and state["pending_action"] is None and state["request_count"] < MAX_TOTAL_ACTIONS
                and type(action) is int
                and action in (range(1, 6) if kind == "ll_policy" else (*range(5), *range(6, 17)))
            )
            if not valid:
                if isinstance(owner, str) and owner:
                    status = {
                        K_OWNER_ID: owner, K_ATOMIC_ID: None, K_REQUESTER: kind, K_ACTION: action,
                        K_REWARD: 0.0, K_DONE: False, K_DEAD: False, K_ILLEGAL_ACTION: False,
                        K_LETHAL_MOVEMENT: False, K_NEW_ACHIEVEMENTS: [],
                        K_CONTRACT_ERROR: "invalid-action-request",
                    }
                    state.outbox.send_status(self._ll_id, ActionStatus(status))
                _fail(state, "invalid-action-request", owner_action_id=owner)
                return state
            state["request_count"] += 1
            request = {**kwargs, K_ATOMIC_ID: state["request_count"]}
            state["owner_ids"].append(owner)
            state["pending_action"] = request
            self.act(self._env_id, **request)
            return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        """Forward exactly one matching status to LLR for neural or passive observation."""

        with _active_time(state):
            pending = state["pending_action"]
            if env_id != self._env_id or pending is None or any(kwargs.get(key) != pending[key] for key in ACTION_CONTEXT):
                _fail(state, "status-correlation-error")
                return state
            status = {key: _json(kwargs.get(key)) for key in (*ACTION_CONTEXT, *PUBLIC_STATUS)}
            if not valid_status(status):
                _fail(state, "invalid-public-status")
                return state
            state["status_count"] += 1
            state["illegal_count"] += int(status[K_ILLEGAL_ACTION])
            state["lethal_count"] += int(status[K_LETHAL_MOVEMENT])
            state["pending_action"] = None
            state["statuses"].append(status)
            state.outbox.send_status(self._ll_id, ActionStatus(status))
            return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        """Close the recording without counting a lifecycle command as a native action."""

        if self._env_id and not state["close_sent"]:
            self.act(self._env_id, **{K_ACTION: A_CLOSE})
            state["close_sent"] = True
        return state


class CrafterBeliefKnowledge(KnowledgeBase):
    """Forward complete consecutive public belief revisions to HLR."""

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        """Resolve the sole belief producer."""

        self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LLR")
        return state

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation,
                           beliefs: Sequence[Belief], **kwargs: Any) -> KnowledgeState:
        """Validate revision order before forwarding through the typed knowledge edge."""

        with _active_time(state):
            try:
                snapshot = parse_abstract_beliefs(beliefs)
                if sender != self._ll_id or snapshot["revision"] != state["revision_count"] + 1:
                    raise ValueError("Invalid belief sender or revision.")
            except (TypeError, ValueError, KeyError) as error:
                _fail(state, "invalid-belief-update", message=str(error))
                return state
            state["revision_count"] += 1
            state["latest_beliefs"] = snapshot
            for reasoner in state.directory.internal.hl_reasoning:
                state.outbox.send_beliefs(reasoner.module_id, beliefs)
                state["forward_count"] += 1
            return state


class CrafterActivityGoalGraph(GoalGraphBase):
    """Relay one neural or passive goal and its terminal acknowledgment."""

    def on_first(self, state: GoalGraphState) -> GoalGraphState:
        """Resolve the two reasoners connected through this goal graph."""

        self._ll_id = _single_id(state.directory.internal.ll_reasoning, "LLR")
        self._hl_id = _single_id(state.directory.internal.hl_reasoning, "HLR")
        return state

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> GoalGraphState:
        """Reject overlapping, duplicate, or uncorrelated goal updates."""

        with _active_time(state):
            try:
                if len(goals) != 1:
                    raise ValueError("Exactly one goal is required.")
                goal = goals[0]
                goal_id = goal.extras["goal_id"]
                if sender == self._hl_id:
                    validate_requested_goal(goal)
                    if state["active_goal_id"] is not None or goal_id in state["completed"]:
                        raise ValueError("Overlapping or repeated goal.")
                    state["active_goal_id"] = goal_id
                    state["dispatch_count"] += 1
                    state.outbox.send_goals(self._ll_id, goals)
                elif sender == self._ll_id:
                    if goal_id != state["active_goal_id"] or goal.extras.get("status") not in {"succeeded", "failed", "interrupted"}:
                        raise ValueError("Unmatched terminal goal.")
                    state["active_goal_id"] = None
                    state["completed"].append(goal_id)
                    state["terminal_count"] += 1
                    state.outbox.send_goals(self._hl_id, goals)
                else:
                    raise ValueError("Unexpected goal sender.")
            except (KeyError, ValueError, TypeError) as error:
                _fail(state, "goal-contract-error", message=str(error))
            return state


class NeuralActivityLLReasoner(LLReasonerBase):
    """Execute five frozen skills and join passive primitive observations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._torch: Any = None
        self._models: dict[PolicyId, Any] = {}
        self._templates: GroundingTemplates | None = None
        self._artifact_contract: dict[str, Any] = {}
        self._perceptor_id = self._actuator_id = self._knowledge_id = self._goal_graph_id = ""
        self._latest_frame: np.ndarray | None = None
        self._latest_observation_id = 0
        self._latest_digest = ""

    def on_init(
        self,
        policy_artifacts_root: str | os.PathLike[str] | None = None,
        grounding_artifacts_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().on_init(**kwargs)
        import torch

        policy_bundle, policy_manifest, self._models = load_policy_bundle(torch, policy_artifacts_root)
        grounding_bundle, grounding_manifest = resolve_active_grounding_bundle(grounding_artifacts_root)
        if policy_manifest["environment"] != grounding_manifest["environment"]:
            raise ValueError("Policy and grounding renderer contracts differ.")
        digests: dict[str, str] = {}
        for policy_id in PolicyId:
            path = policy_bundle / POLICY_FILENAMES[policy_id]
            digests[policy_id.value] = file_sha256(path)
        torch.set_num_threads(1)
        self._torch = torch
        self._templates = load_grounding_templates(grounding_bundle)
        self._artifact_contract = {
            "policy_artifact_id": policy_manifest["artifact_id"],
            "policy_manifest_sha256": file_sha256(policy_bundle / MANIFEST_FILENAME),
            "checkpoint_sha256": digests,
            "grounding_artifact_id": grounding_manifest["artifact_id"],
            "grounding_manifest_sha256": file_sha256(grounding_bundle / GROUNDING_MANIFEST_FILENAME),
        }

    def on_first(self, state: LLState) -> LLState:
        state["artifact_contract"] = dict(self._artifact_contract)
        state["policy_loaded"] = True
        self._perceptor_id = _single_id(state.directory.internal.perception, "perceptor")
        self._actuator_id = _single_id(state.directory.internal.actuation, "actuator")
        self._knowledge_id = _single_id(state.directory.internal.knowledge, "knowledge module")
        self._goal_graph_id = _single_id(state.directory.internal.goals, "goal graph")
        self._request_observation(state)
        return state

    def _request_observation(self, state: LLState) -> None:
        if state["awaiting_observation"]:
            _fail(state, "overlapping-observation-request")
            return
        observation_id = state["observation_request_count"] + 1
        state["observation_request_count"] = observation_id
        state["awaiting_observation"] = True
        state["phase"] = "awaiting_observation"
        state.outbox.request_observation(self._perceptor_id, **{K_OBSERVATION_ID: observation_id})

    def _terminal(self, state: LLState, status: str, reason: str = "") -> str:
        """Close the current goal without terminating the ongoing experiment."""

        requested = goal_from_dict(state["current_goal"])
        goal_id = requested.extras["goal_id"]
        revision = state["belief_state"]["revision"]
        result = terminal_goal(requested, revision, status, reason)
        state["terminal_goal_count"] += 1
        state["activities"].append({
            **requested.extras, **result.extras,
            "actions": state["current_goal_action_count"],
            "final_inventory": dict(state["belief_state"]["inventory"]),
        })
        state.outbox.send_goal_update(self._goal_graph_id, [result])
        state["current_goal"] = None
        state["cow_search_goal"] = None
        state["current_goal_action_count"] = 0
        state["phase"] = "awaiting_goal"
        self.log(INFO, f"Goal {goal_id}: {status} {reason}")
        return goal_id

    def _dispatch(self, state: LLState) -> None:
        """Select one neural action, retaining the actual public target and effective mask."""

        if state["current_goal"] is None or state["pending_action"] is not None or state["awaiting_observation"]:
            return
        goal = goal_from_dict(state["current_goal"])
        if goal.extras["kind"] == "passive":
            self._join_passive(state)
            return
        spec = activity_from_goal(goal)
        belief = state["belief_state"]
        policy = PolicyId(spec.activity.value)
        if policy is PolicyId.EAT_COW and not state["enable_eat_cow"]:
            self._terminal(state, "failed", "policy_disabled")
            return
        target = spec.target_cell
        if policy in {PolicyId.EAT_TARGET, PolicyId.EAT_COW}:
            cows = known_targets(belief).get("cow", ())
            previous = state["cow_target"]
            if policy is PolicyId.EAT_TARGET:
                target = tuple(previous) if previous is not None and tuple(previous) in cows else None
            else:
                target = nearest_target(tuple(previous) if previous is not None else tuple(belief["player"]), cows)
            state["cow_target"] = list(target) if target is not None else None
            if target is None and policy is PolicyId.EAT_TARGET:
                self._terminal(state, "failed", "target_lost")
                return
        context = encode_context(policy, belief["player"], target)
        facing = Direction.from_name(belief["facing"]).delta
        interaction_target = target if policy in {
            PolicyId.GET_RESOURCE, PolicyId.EAT_TARGET, PolicyId.EAT_COW,
        } else None
        movement = available_movement_actions(belief, interaction_target)
        if policy in {PolicyId.NAVIGATE_TO, PolicyId.GET_RESOURCE, PolicyId.EXPLORE} or (
            policy is PolicyId.EAT_COW and target is None
        ):
            recent_cells = [
                tuple(row["source_cell"])
                for row in state["trace"][-4:]
                if row.get("goal_id") == spec.goal_id
            ]
            movement = novel_movement_actions(
                movement, tuple(belief["player"]), recent_cells,
            )
        mask_context = context
        search_goal = None
        if policy is PolicyId.EAT_COW and getattr(self._models[policy], "uses_discovery_context", False):
            if target is None:
                previous_goal = state["cow_search_goal"]
                search_goal = discovery_goal(
                    tuple(belief["player"]), map(position_from_key, belief["known_cells"]),
                    map(position_from_key, belief["safe_cells"]),
                    tuple(previous_goal) if previous_goal is not None else None, movement,
                    # Runtime coordinates start at zero; native Crafter starts at (32, 32).
                    world_origin=(-32, -32),
                )
            state["cow_search_goal"] = list(search_goal) if search_goal is not None else None
            if target is None:
                context = encode_context(policy, belief["player"], search_goal)
        if not selection_actions(policy, mask_context, facing, movement):
            self._terminal(state, "failed", "movement_blocked")
            return
        try:
            action, values = inference_evidence(
                self._torch, self._models[policy], self._latest_frame, policy,
                belief["player"], target, facing=facing, movement_actions=movement,
                search_goal_cell=search_goal,
            )
        except (ValueError, RuntimeError) as error:
            _fail(state, "inference-error", message=str(error))
            return
        state["inference_count"] += 1
        state["policy_action_counts"][policy.value] += 1
        owner = f"ll-{state['inference_count']}"
        row = {
            **movement_evidence(action, belief), K_OWNER_ID: owner, K_REQUESTER: "ll_policy",
            "goal_id": spec.goal_id, "policy_id": policy.value, "context": context.tolist(),
            "target_cell": list(target) if target is not None else None,
            "search_goal_cell": list(search_goal) if search_goal is not None else None,
            "facing": list(facing), "available_movements": list(movement),
            "legal_actions": list(selection_actions(policy, mask_context, facing, movement)), "q_values": values,
            "input_observation_id": belief["revision"], "input_observation_sha256": self._latest_digest,
            "status": None, "result_observation_id": None,
        }
        state["pending_action"] = row
        state["current_goal_action_count"] += 1
        state.outbox.request_action(self._actuator_id, **{
            K_OWNER_ID: owner, K_REQUESTER: "ll_policy", K_ACTION: action,
        })
        state["phase"] = "awaiting_status"

    def _join_passive(self, state: LLState) -> None:
        """Handle either arrival order without observing before both matching inputs exist."""

        current, status = state["current_goal"], state["passive_status"]
        if current is None or status is None:
            return
        data = current["extras"]
        if (data["kind"] != "passive" or data[K_OWNER_ID] != status[K_OWNER_ID]
                or data["action"] != status[K_ACTION] or state["pending_action"] is not None):
            _fail(state, "passive-correlation-error")
            return
        belief = state["belief_state"]
        state["pending_action"] = {
            **movement_evidence(status[K_ACTION], belief),
            **{key: status[key] for key in ACTION_CONTEXT},
            "goal_id": data["goal_id"], "policy_id": None,
            "input_observation_id": belief["revision"], "input_observation_sha256": self._latest_digest,
            "status": status, "result_observation_id": None,
        }
        state["passive_status"] = None
        self._request_observation(state)

    def on_goal_update(self, state: LLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> LLState:
        """Start one goal only from the latest fully observed action boundary."""

        with _active_time(state):
            try:
                if sender != self._goal_graph_id or len(goals) != 1 or state["current_goal"] is not None:
                    raise ValueError("Overlapping goal or unexpected sender.")
                validate_requested_goal(goals[0])
                if goals[0].extras["baseline_revision"] != state["belief_state"]["revision"]:
                    raise ValueError("Goal does not use the latest belief revision.")
                state["current_goal"] = goal_to_dict(goals[0])
                state["current_goal_action_count"] = 0
                state["resource_history"] = []
                state["cow_target"] = goals[0].extras.get("target_cell")
                state["cow_search_goal"] = None
                state["cow_search_start_need"] = (
                    urgent_need(state["belief_state"]["inventory"])
                    if goals[0].extras.get("activity") == ActivityId.EAT_COW.value else None
                )
                state["received_goal_count"] += 1
                self._dispatch(state)
            except (KeyError, TypeError, ValueError) as error:
                _fail(state, "invalid-goal", message=str(error))
            return state

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs: Any) -> LLState:
        """Correlate a status before requesting its one fresh observation."""

        with _active_time(state):
            status = action_status.status
            if (sender != self._actuator_id or not isinstance(status, Mapping) or not valid_status(status)
                    or any(key not in status for key in ACTION_CONTEXT)
                    or status[K_REQUESTER] not in {"ll_policy", "hl_primitive"}):
                _fail(state, "invalid-action-status")
                return state
            if status[K_ATOMIC_ID] != state["belief_state"]["native_action_count"] + 1:
                _fail(state, "noncontiguous-action-status")
                return state
            state["status_count"] += 1
            if status[K_REQUESTER] == "hl_primitive":
                if state["passive_status"] is not None or state["pending_action"] is not None:
                    _fail(state, "overlapping-passive-status")
                else:
                    state["passive_status"] = dict(status)
                    self._join_passive(state)
            else:
                pending = state["pending_action"]
                if (pending is None or pending["status"] is not None
                        or status[K_OWNER_ID] != pending[K_OWNER_ID] or status[K_ACTION] != pending["action"]):
                    _fail(state, "neural-status-correlation-error")
                else:
                    pending["status"] = dict(status)
                    pending[K_ATOMIC_ID] = status[K_ATOMIC_ID]
                    self._request_observation(state)
            return state

    def _outcome(self, state: LLState, row: Mapping[str, Any]) -> tuple[str, str] | None:
        """Evaluate a skill after fresh grounding, including survival interruption."""

        goal = goal_from_dict(state["current_goal"])
        if goal.extras["kind"] == "passive":
            return "succeeded", ""
        spec = activity_from_goal(goal)
        belief = state["belief_state"]
        if spec.activity is ActivityId.EXPLORE:
            succeeded = len(belief["known_cells"]) >= spec.target_value
        elif spec.activity is ActivityId.NAVIGATE_TO:
            succeeded = tuple(belief["player"]) == spec.target_cell
        elif spec.activity is ActivityId.GET_RESOURCE:
            succeeded = resource_collected(
                row["action"], row["source_cell"], row["facing"], spec.target_cell,
                spec.target_kind, row["status"][K_NEW_ACHIEVEMENTS], row["status"][K_ILLEGAL_ACTION],
            )
            state["resource_history"].append({
                "source": row["source_cell"], "destination": belief["player"],
                "discovered": row["new_known_cells"] > 0, "collected": succeeded,
            })
            state["resource_history"] = state["resource_history"][-8:]
        elif spec.activity is ActivityId.EAT_TARGET:
            succeeded = consumed_tracked_cow(
                row["action"], row["source_cell"], row["facing"], row["target_cell"],
                row["status"][K_NEW_ACHIEVEMENTS], row["status"][K_ILLEGAL_ACTION],
            )
        else:
            succeeded = belief["achievement_counts"].get("eat_cow", 0) >= spec.target_value
        if succeeded:
            return "succeeded", ""
        if belief["terminal"] or belief["dead"]:
            return "failed", "environment_terminal"
        if belief["inventory"].get("diamond", 0) > 0:
            return "interrupted", "diamond_acquired"
        if belief["native_action_count"] >= MAX_TOTAL_ACTIONS:
            return "interrupted", "total_action_bound"
        if row["status"][K_ILLEGAL_ACTION]:
            return "failed", "illegal_action"
        if spec.activity is ActivityId.GET_RESOURCE and resource_stagnated(state["resource_history"]):
            self.log(INFO, f"Stagnation {spec.goal_id}: {state['resource_history']}")
            return "failed", "stagnation"
        if spec.activity is ActivityId.EAT_TARGET and state["cow_target"] is None:
            return "failed", "target_lost"
        if spec.activity is ActivityId.GET_RESOURCE and spec.target_cell not in known_targets(belief).get(spec.target_kind, ()):
            return "failed", "target_lost"
        if spec.activity is ActivityId.NAVIGATE_TO and spec.target_cell not in routes(belief):
            return "failed", "precondition_invalid"
        if state["current_goal_action_count"] >= spec.max_actions:
            return "failed", "action_bound"
        need = urgent_need(belief["inventory"])
        # Respect a deliberate HLR turn after another need was already deferred.
        if need is not None and (
            not spec.purpose.startswith("recovery:")
            or (spec.activity is ActivityId.EAT_COW and need not in {"food", state["cow_search_start_need"]})
        ):
            return "interrupted", "survival_interrupt"
        return None

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs: Any) -> LLState:
        """Ground once, publish one revision, and then finish or continue the active goal."""

        with _active_time(state):
            try:
                revision = kwargs[K_OBSERVATION_ID]
                frame = as_rgb_frame(observation.content)
                digest = rgb_sha256(frame)
                if (sender != self._perceptor_id or not state["awaiting_observation"]
                        or kwargs.get(K_CONTRACT_ERROR) is not None
                        or revision != state["observation_count"] + 1
                        or digest != kwargs[K_OBSERVATION_DIGEST]):
                    raise ValueError("Invalid observation correlation or digest.")
                previous = state["belief_state"]
                facing = Direction.from_name(previous["facing"]) if previous["facing"] else None
                percept, _ = ground_observation(frame, self._templates, previous_facing=facing)
                row = state["pending_action"]
                belief = revise_belief_state(previous, percept, revision=revision, pending_action=row)
                state["belief_state"] = belief
                state["observation_count"] += 1
                state["awaiting_observation"] = False
                self._latest_frame, self._latest_digest = frame, digest
                completed = None
                if row is not None:
                    if row.get("policy_id") == PolicyId.EAT_TARGET.value:
                        target = associate_cow(
                            tuple(row["target_cell"]) if row["target_cell"] is not None else None,
                            known_targets(previous).get("cow", ()),
                            known_targets(belief).get("cow", ()),
                        )
                        state["cow_target"] = list(target) if target is not None else None
                    row["new_known_cells"] = len(belief["known_cells"]) - len(previous["known_cells"])
                    row["result_observation_id"], row["result_observation_sha256"] = revision, digest
                    state["trace"].append(row)
                    state["pending_action"] = None
                    outcome = self._outcome(state, row)
                    if outcome is not None:
                        completed = self._terminal(state, *outcome)
                state.outbox.send_beliefs(self._knowledge_id, observation, abstract_beliefs(belief, completed))
                state["belief_count"] += 1
                if belief["native_action_count"] and belief["native_action_count"] % 25 == 0:
                    self.log(INFO, f"Progress {belief['native_action_count']}/{MAX_TOTAL_ACTIONS}; "
                             f"needs={belief['inventory']}; goal={state['current_goal']}")
                self._dispatch(state)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                _fail(state, "observation-pipeline-error", message=str(error))
            return state


class CrafterActivityHLReasoner(HLReasonerBase):
    """Maintain needs and explicit crafting progression over successive bounded skills."""

    def on_first(self, state: HLState) -> HLState:
        """Resolve the existing typed goal, belief, and direct primitive edges."""

        self._goal_graph_id = _single_id(state.directory.internal.goals, "goal graph")
        self._knowledge_id = _single_id(state.directory.internal.knowledge, "knowledge")
        self._actuator_id = _single_id(state.directory.internal.actuation, "actuator")
        return state

    def _dispatch(self, state: HLState) -> None:
        """Choose at most one goal or primitive using the latest reconciled public state."""

        belief = state["latest_beliefs"]
        if belief is None or state["active_goal"] is not None or state["terminal_reason"] is not None:
            return
        diamond = belief["inventory"].get("diamond", 0) > 0
        if belief["terminal"] or belief["dead"] or diamond or belief["native_action_count"] >= MAX_TOTAL_ACTIONS:
            state["terminal_reason"] = ("environment_terminal" if belief["terminal"] or belief["dead"]
                                        else "diamond_acquired" if diamond else "action_budget")
            state["survived"] = not belief["dead"] and not belief["terminal"]
            state.outbox.terminate_agent(state["terminal_reason"])
            return
        inventory = belief["inventory"]
        needs = sorted((need for need in ("drink", "food", "energy") if inventory[need] <= 2),
                       key=lambda need: (inventory[need], ("drink", "food", "energy").index(need)))
        if not needs and urgent_need(inventory) == "health":
            needs = ["health"]
        recovery = state["recovery"]
        if recovery is not None and recovery_complete(belief, recovery):
            recovery = None
        if not belief["sleeping"]:
            used = belief["native_action_count"] - state["recovery_start"]
            if recovery is not None and (state["recovery_failed"] or used >= 16):
                if recovery not in state["deferred_needs"]:
                    state["deferred_needs"].append(recovery)
                self.log(INFO, f"Recovery yield: {recovery}, actions={used}, failed={state['recovery_failed']}")
                recovery = None
            if recovery is None and needs:
                eligible = [need for need in needs if need not in state["deferred_needs"]]
                if not eligible:
                    state["deferred_needs"] = []
                    eligible = needs
                recovery = eligible[0]
                state["recovery_start"] = belief["native_action_count"]
            state["recovery_failed"] = False
        elif recovery is None:
            recovery = "energy"
        state["target_deferrals"] = [item for item in state["target_deferrals"] if (
            sum(abs(a - b) for a, b in zip(belief["player"], item["source"], strict=True)) < 3
            and belief["terrain"].get(cell_key(tuple(item["target"]))) == item["material"])]
        excluded = [*state["excluded_targets"], *(item["key"] for item in state["target_deferrals"])]
        if recovery != state["recovery"]:
            self.log(INFO, f"Recovery: {state['recovery']} -> {recovery}")
        state["recovery"] = recovery
        goal_id = f"hl-{state['dispatch_count'] + 1}"
        step, placement = choose_step(belief, goal_id, recovery, excluded, state["placement"],
                                      enable_eat_cow=state["enable_eat_cow"])
        state["placement"] = placement
        if isinstance(step, ActivitySpec):
            # Preserve EatCow's search budget; LLR can yield to a more urgent need.
            if recovery is not None and step.activity is not ActivityId.EAT_COW:
                remaining = max(1, 16 - (belief["native_action_count"] - state["recovery_start"]))
                step = replace(step, max_actions=min(step.max_actions, remaining))
            goal = make_activity_goal(step)
        else:
            goal = passive_goal(goal_id, belief["revision"], step["action"], step["purpose"])
            state["primitive_decisions"].append({**goal.extras, "source_cell": belief["player"]})
        state["active_goal"] = goal_to_dict(goal)
        state["dispatch_count"] += 1
        state.outbox.send_goals(self._goal_graph_id, [goal])
        if not isinstance(step, ActivitySpec):
            state.outbox.request_action(self._actuator_id, **{
                K_OWNER_ID: goal_id, K_REQUESTER: "hl_primitive", K_ACTION: step["action"],
            })
        self.log(INFO, f"Dispatch {goal_id}: {goal.extras}")

    def _reconcile(self, state: HLState) -> None:
        """Wait for terminal acknowledgment and its exact fresh knowledge revision."""

        terminal, active, belief = state["pending_terminal"], state["active_goal"], state["latest_beliefs"]
        if terminal is None or active is None or belief is None:
            return
        result = terminal["extras"]
        if result["goal_id"] != active["extras"]["goal_id"]:
            _fail(state, "terminal-goal-mismatch")
            return
        if result["completion_revision"] != belief["revision"]:
            return
        state["activities"].append({**active["extras"], **result})
        if active["extras"]["kind"] == "passive":
            state["activities"][-1]["primitive_status"] = belief["last_action_status"]
        state["terminal_count"] += 1
        spec = active["extras"]
        primitive_failed = spec["kind"] == "passive" and belief["last_action_status"]["illegal_action"]
        failed = result["status"] == "failed" or primitive_failed
        urgent_cow_interruption = (spec.get("activity") == "eat_cow"
                                   and result["status"] == "interrupted"
                                   and result["reason"] == "survival_interrupt")
        if spec["purpose"].startswith("recovery:") and (failed or urgent_cow_interruption):
            state["recovery_failed"] = True
        if (spec.get("activity") == "get_resource" and result["reason"] in
                {"illegal_action", "stagnation", "action_bound"}):
            key = failure_key(f"resource:{spec['target_kind']}", tuple(spec["target_cell"]))
            state["target_deferrals"] = [item for item in state["target_deferrals"] if item["key"] != key]
            state["target_deferrals"].append({
                "key": key, "target": spec["target_cell"], "source": list(belief["player"]),
                "material": belief["terrain"].get(cell_key(tuple(spec["target_cell"]))),
            })
            self.log(INFO, f"Deferred target: {key}, reason={result['reason']}")
            failed = False
        if failed:
            target = tuple(spec["target_cell"]) if spec.get("target_cell") is not None else None
            key = failure_key(spec["purpose"], target)
            state["failures_by_target"][key] = state["failures_by_target"].get(key, 0) + 1
            if state["failures_by_target"][key] >= 3 and key not in state["excluded_targets"]:
                state["excluded_targets"].append(key)
        elif result["status"] == "succeeded":
            state["failures_by_target"] = {}
            if spec.get("activity") == "explore":
                state["excluded_targets"] = []
        state["active_goal"] = state["pending_terminal"] = None
        self._dispatch(state)

    def on_belief_update(self, state: HLState, sender: str, beliefs: Sequence[Belief], **kwargs: Any) -> HLState:
        """Accept a complete consecutive snapshot, including during neural execution."""

        with _active_time(state):
            try:
                snapshot = parse_abstract_beliefs(beliefs)
                if sender != self._knowledge_id or snapshot["revision"] != state["belief_update_count"] + 1:
                    raise ValueError("Invalid HLR belief sequence.")
                state["latest_beliefs"] = snapshot
                state["belief_update_count"] += 1
                self._reconcile(state)
                self._dispatch(state)
            except (KeyError, TypeError, ValueError) as error:
                _fail(state, "invalid-hlr-beliefs", message=str(error))
            return state

    def on_goal_update(self, state: HLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> HLState:
        """Buffer one terminal acknowledgment until its fresh beliefs arrive."""

        with _active_time(state):
            if (sender != self._goal_graph_id or len(goals) != 1 or state["pending_terminal"] is not None
                    or state["active_goal"] is None):
                _fail(state, "unexpected-terminal-goal")
            else:
                state["pending_terminal"] = goal_to_dict(goals[0])
                self._reconcile(state)
            return state


def initial_states(*, enable_eat_cow: bool = True) -> dict[str, dict[str, Any]]:
    """Declare every persistent field for the six reactive modules."""

    if type(enable_eat_cow) is not bool:
        raise ValueError("enable_eat_cow must be a boolean")
    states = {
        PERCEPTOR: {
            "request_count": 0, "observation_count": 0, "pending_observation_id": None,
            "last_observation_sha256": None,
        },
        ACTUATOR: {
            "request_count": 0, "status_count": 0, "pending_action": None, "owner_ids": [],
            "statuses": [], "illegal_count": 0, "lethal_count": 0, "close_sent": False,
        },
        LLREASONER: {
            "enable_eat_cow": enable_eat_cow,
            "phase": "starting", "policy_loaded": False, "artifact_contract": None,
            "belief_state": initial_belief_state(), "current_goal": None, "current_goal_action_count": 0,
            "cow_target": None, "cow_search_goal": None, "cow_search_start_need": None, "pending_action": None, "passive_status": None,
            "awaiting_observation": False, "observation_request_count": 0,
            "observation_count": 0, "belief_count": 0, "received_goal_count": 0,
            "terminal_goal_count": 0, "inference_count": 0, "status_count": 0,
            "policy_action_counts": {policy.value: 0 for policy in PolicyId},
            "trace": [], "activities": [],
            "resource_history": [],
        },
        KNOWLEDGE: {"revision_count": 0, "forward_count": 0, "latest_beliefs": None},
        GOALGRAPH: {"active_goal_id": None, "dispatch_count": 0, "terminal_count": 0, "completed": []},
        HLREASONER: {
            "enable_eat_cow": enable_eat_cow,
            "latest_beliefs": None, "active_goal": None, "pending_terminal": None,
            "dispatch_count": 0, "terminal_count": 0, "belief_update_count": 0,
            "recovery": None, "placement": None, "failures_by_target": {}, "excluded_targets": [],
            "target_deferrals": [], "deferred_needs": [], "recovery_start": 0, "recovery_failed": False,
            "primitive_decisions": [], "activities": [], "terminal_reason": None, "survived": False,
        },
    }
    for state in states.values():
        state.update(active_seconds=0.0, failure=None)
    return states
