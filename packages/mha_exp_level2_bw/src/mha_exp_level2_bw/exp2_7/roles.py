"""Seven direct MHAgentA cognitive role behaviors for experiment 2-7-BW."""

from __future__ import annotations

from typing import Any, cast

from mhagenta import ActionStatus, Belief, Goal, Observation
from mhagenta.bases import (
    GoalGraphBase,
    HLReasonerBase,
    KnowledgeBase,
    LearnerBase,
    LLReasonerBase,
    MemoryBase,
)
from mhagenta.states import (
    GoalGraphState,
    HLState,
    KnowledgeState,
    LearnerState,
    LLState,
    MemoryState,
)

from .evidence import receive_message, send_message, snapshot_metadata
from .terminal import finalize_module, publish_terminal
from .llm import (
    GoalGraphResponse,
    HLReasonerResponse,
    KnowledgeResponse,
    LLReasonerResponse,
    LearnerResponse,
    MemoryResponse,
    ROLE_RESPONSE_TYPES,
    RoleRuntime,
    belief_from_data,
    belief_from_payload,
    belief_to_data,
    canonical_sha256,
    goal_from_data,
    goal_from_payload,
    goal_to_data,
    json_safe,
    normalized_failure,
    observation_from_data,
    observation_to_data,
)


PERCEPTOR_ID = "perceptor_0"
ACTUATOR_ID = "actuator_0"
LL_ID = "llreasoner_0"
KNOWLEDGE_ID = "knowledge_0"
HL_ID = "hlreasoner_0"
GOAL_GRAPH_ID = "goalgraph_0"
MEMORY_ID = "memory_0"
LL_LEARNER_ID = "learner_0"
HL_LEARNER_ID = "learner_1"


def _changed_goal(state: Any, recipient: str, goal: Goal, *, requested: bool = False) -> bool:
    """Suppress unchanged goal relays while preserving explicit request replies."""
    identity = _goal_id(goal)
    key = recipient + ":" + str(identity)
    value = {"state": goal_to_data(goal)["state"],
             **{name: goal.extras.get(name) for name in ("goal_id", "status", "primary", "order")}}
    if not requested and state["last_sent_payloads"].get(key) == value:
        state["suppressed_messages"] += 1
        return False
    state["last_sent_payloads"][key] = value
    return True


def _goal_id(goal: Goal) -> str | None:
    value = goal.extras.get("goal_id") if goal.extras else None
    return str(value) if value is not None else None


def _goal_hash(goal: Goal) -> str:
    data = goal_to_data(goal)
    extras = dict(data.get("extras", {}))
    extras.pop("status", None)
    extras.pop("goal_payload_sha256", None)
    data["extras"] = extras
    return canonical_sha256(data)


def _metadata_from_snapshot(runtime: RoleRuntime) -> dict[str, Any]:
    return snapshot_metadata(runtime.last_snapshot)


def _mark_dispatch_failure(state: Any, error: BaseException) -> None:
    if state["failure"] is None:
        state["failure"] = normalized_failure(
            kind=type(error).__name__,
            stage="dispatch",
            module_time=float(state.time),
            response_id=state["last_response_id"],
        )
    state["admission_open"] = False
    publish_terminal(state, "execution_error")
    state.outbox.terminate_agent("Experiment 2-7-BW unrecoverable dispatch failure")


def _close_runtime(state: Any, runtime: RoleRuntime) -> Any:
    try:
        runtime.close()
    except Exception as error:  # noqa: BLE001
        if state["failure"] is None:
            state["failure"] = normalized_failure(
                kind=type(error).__name__,
                stage="client_cleanup",
                module_time=float(state.time),
            )
    return finalize_module(state)


def _admission_open(state: Any) -> bool:
    if float(state.time) >= float(state["lifecycle"]["behavior_cutoff"]):
        state["admission_open"] = False
    return bool(state["admission_open"])


class LLMLowLevelReasoner(LLReasonerBase):
    """Low-level reasoning with direct typed MHAgentA dispatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = RoleRuntime("ll_reasoner", ROLE_RESPONSE_TYPES["ll_reasoner"])

    def on_init(self, **kwargs: Any) -> None:
        self._runtime.initialize(**kwargs)

    def on_first(self, state: LLState) -> LLState:
        """Acquire the first sensor reading before asking the model to reason."""
        state["initial_call_pending"] = False
        self._request_observation(state)
        return state

    def _request_observation(self, state: LLState, revision_id: str | None = None) -> None:
        """Request one correlated sensor reading through the native perceptor edge."""
        if not _admission_open(state):
            state["suppressed_dispatches"].append("request_observation")
            return
        sequence = int(state["cycle_sequence"])
        state["cycle_sequence"] = sequence + 1
        cycle_id = f"cycle:{sequence}"
        state["active_cycle_id"] = cycle_id
        send_message(
            state,
            recipient=PERCEPTOR_ID,
            kind="request_observation",
            metadata={"cycle_id": cycle_id, "revision_id": revision_id},
            dispatch=lambda **meta: state.outbox.request_observation(PERCEPTOR_ID, **meta),
        )

    def on_observation(
        self,
        state: LLState,
        sender: str,
        observation: Observation,
        **kwargs: Any,
    ) -> LLState:
        data = observation_to_data(observation)
        state["current_observation"] = data
        freshness = state["action_observation_state"]
        freshness["observation_sequence"] += 1
        freshness["observation_received_at"] = float(state.time)
        last_action = freshness["last_action"]
        request = state["waiting_requests"].get(PERCEPTOR_ID + ":request_observation", {})
        freshness["observation_after_action"] = last_action is None or bool(
            last_action.get("status_received_at") is not None
            and request.get("metadata", {}).get("cycle_id") == kwargs.get("cycle_id")
            and request.get("last_sent_at", -1) >= last_action["status_received_at"]
        )
        state["active_cycle_id"] = kwargs.get("cycle_id")
        receive_message(
            state,
            sender=sender,
            kind="observation",
            payload={"observation": data},
            metadata=kwargs,
        )
        return state

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        receive_message(
            state,
            sender=sender,
            kind="action_status",
            payload={"action_status": {"status": action_status.status}},
            metadata=kwargs,
        )
        state["status_processed"] += 1
        freshness = state["action_observation_state"]
        freshness["last_action_status"] = {"status": json_safe(action_status.status),
                                         "action_id": kwargs.get("action_id"), "received_at": float(state.time)}
        last_action = freshness["last_action"]
        if last_action and kwargs.get("action_id") == last_action["action_id"]:
            last_action["status_received_at"] = float(state.time)
            freshness["observation_after_action"] = False
        status = action_status.status
        if isinstance(status, dict) and status.get("scientific_complete") is True:
            publish_terminal(state, "task_completed")
            state["scientific_complete"] = True
            state["termination_reason"] = "task_completed"
            state["scientific_completion_evidence"] = {
                "cycle_id": kwargs.get("cycle_id"),
                "action_id": kwargs.get("action_id"),
                "eligible_post_goal_discretionary_states": status.get(
                    "eligible_post_goal_discretionary_states"
                ),
                "module_time": float(state.time),
            }
            state["admission_open"] = False
            terminate = getattr(state.outbox, "terminate_agent", None)
            if callable(terminate):
                terminate("Experiment 2-7-BW scientific completion reached")
        return state

    def on_goal_update(
        self,
        state: LLState,
        sender: str,
        goals: list[Goal],
        **kwargs: Any,
    ) -> LLState:
        values = [goal_to_data(goal) for goal in goals]
        state["active_goals"] = values
        for value in values:
            identity = value.get("extras", {}).get("goal_id")
            if identity is not None:
                state["goal_ledger"][str(identity)] = value
        receive_message(
            state,
            sender=sender,
            kind="goal_update",
            payload={"goals": values},
            metadata=kwargs,
        )
        return state

    def on_model(
        self,
        state: LLState,
        sender: str,
        model: Any,
        **kwargs: Any,
    ) -> LLState:
        normalized_model = json_safe(model)
        state["learner_model"] = normalized_model
        receive_message(
            state,
            sender=sender,
            kind="learner_model",
            payload={"model": normalized_model},
            metadata=kwargs,
        )
        return state

    def _dispatch(self, state: LLState, response: LLReasonerResponse) -> None:
        metadata = _metadata_from_snapshot(self._runtime)
        revision_id = metadata.get("revision_id")
        if revision_id and revision_id not in state["used_revision_ids"]:
            state["used_revision_ids"].append(revision_id)

        if response.request_observation:
            self._request_observation(state, revision_id)

        if response.beliefs:
            observation_data = state["current_observation"]
            if observation_data is None:
                raise ValueError("belief dispatch requires a received observation")
            observation = observation_from_data(observation_data)
            beliefs = [belief_from_payload(item) for item in response.beliefs]
            send_message(
                state,
                recipient=KNOWLEDGE_ID,
                kind="send_beliefs",
                metadata={
                    "evidence_id": metadata.get("evidence_id"),
                    "observation_payload_sha256": metadata.get(
                        "observation_payload_sha256"
                    ),
                    "cycle_id": metadata.get("cycle_id"),
                    "source_ids": response.source_ids,
                    "revision_id": revision_id,
                },
                dispatch=lambda **meta: state.outbox.send_beliefs(
                    KNOWLEDGE_ID, observation, beliefs, **meta
                ),
            )

        if response.request_goals:
            send_message(
                state,
                recipient=GOAL_GRAPH_ID,
                kind="request_goals",
                metadata={"revision_id": revision_id},
                dispatch=lambda **meta: state.outbox.request_goals(
                    GOAL_GRAPH_ID, **meta
                ),
            )

        active_by_id = {**state["goal_ledger"], **{
            str(item.get("extras", {}).get("goal_id")): item
            for item in state["active_goals"]
        }}
        for progress in response.goal_progress:
            if progress.goal_id not in active_by_id:
                raise ValueError("unknown goal progress identity")
            goal_data = dict(active_by_id[progress.goal_id])
            goal_data["extras"] = dict(goal_data.get("extras", {}))
            goal_data["extras"]["status"] = progress.status
            goal = goal_from_data(goal_data)
            if not _changed_goal(state, GOAL_GRAPH_ID, goal):
                continue
            goal_hash = str(goal.extras.get("goal_payload_sha256") or _goal_hash(goal))
            send_message(
                state,
                recipient=GOAL_GRAPH_ID,
                kind="send_goal_update",
                metadata={
                    "goal_id": progress.goal_id,
                    "goal_payload_sha256": goal_hash,
                    "cycle_id": state["active_cycle_id"],
                    "revision_id": revision_id,
                },
                dispatch=lambda goal=goal, **meta: state.outbox.send_goal_update(
                    GOAL_GRAPH_ID, [goal], **meta
                ),
            )

        if response.action is not None:
            if _admission_open(state):
                if not state["active_goals"]:
                    raise ValueError("action dispatch requires a delivered goal")
                goal = goal_from_data(state["active_goals"][0])
                goal_id = _goal_id(goal)
                action_sequence = int(state["action_sequence"])
                state["action_sequence"] = action_sequence + 1
                action_id = f"action:{action_sequence}"
                freshness = state["action_observation_state"]
                freshness["last_action"] = {"action_id": action_id, "action": response.action,
                                            "sent_at": float(state.time), "status_received_at": None,
                                            "observation_sequence": freshness["observation_sequence"]}
                freshness["observation_after_action"] = False
                send_message(
                    state,
                    recipient=ACTUATOR_ID,
                    kind="request_action",
                    metadata={
                        "action_id": action_id,
                        "cycle_id": state["active_cycle_id"],
                        "goal_id": goal_id,
                        "goal_payload_sha256": goal.extras.get(
                            "goal_payload_sha256"
                        ),
                        "source_ids": response.source_ids,
                        "revision_id": revision_id,
                    },
                    dispatch=lambda **meta: state.outbox.request_action(
                        ACTUATOR_ID, action=response.action, **meta
                    ),
                )
            else:
                state["suppressed_dispatches"].append("request_action")

        if response.learner_task is not None:
            state["learner_requested"] = True
            send_message(
                state,
                recipient=LL_LEARNER_ID,
                kind="send_learner_task",
                dispatch=lambda **meta: state.outbox.send_learner_task(
                    LL_LEARNER_ID, response.learner_task, **meta
                ),
            )
        if response.request_model:
            state["learner_requested"] = True
            send_message(
                state,
                recipient=LL_LEARNER_ID,
                kind="request_model",
                dispatch=lambda **meta: state.outbox.request_model(
                    LL_LEARNER_ID, **meta
                ),
            )

    def _response_issues(self, state: LLState, response: LLReasonerResponse) -> list[str]:
        """Check required native payloads without prescribing a plan or action."""
        known = set(state["goal_ledger"]) | {
            str(item.get("extras", {}).get("goal_id")) for item in state["active_goals"]}
        issues = [f"Goal {progress.goal_id!r} has never been received; its native progress payload is unavailable."
                  for progress in response.goal_progress if progress.goal_id not in known]
        if response.beliefs and state["current_observation"] is None:
            issues.append("No native observation has been received to accompany beliefs.")
        if response.action is not None and not state["active_goals"]:
            issues.append("No current goal has been delivered for the action's required goal metadata.")
        return issues

    def step(self, state: LLState) -> LLState:
        try:
            scientific_complete = bool(state["scientific_complete"])
        except KeyError:
            scientific_complete = False
        if scientific_complete:
            return state
        previous_text = state["text_state"]
        response = self._runtime.call(state, self.module_id)
        if response is not None:
            response = cast(LLReasonerResponse, response)
            issues = self._response_issues(state, response)
            if issues:
                feedback = {"issues": issues, "previous_response": response.model_dump(mode="json"),
                            "attempt": state["progress_repair_attempts"] + 1,
                            "instruction": "No part of this response was dispatched. Reconsider the received facts and choose your own correction; empty output is valid."}
                state["progress_feedback_history"].append({**feedback, "response_id": state["last_response_id"]})
                if state["progress_repair_attempts"] >= 2:
                    _mark_dispatch_failure(state, ValueError("unresolved native payload after two corrections"))
                    return state
                state["progress_repair_attempts"] += 1
                state["context"]["progress_feedback"] = feedback
                state["text_state"] = previous_text
                state["pending"] = list(self._runtime.last_snapshot) + list(state["pending"])
                state["initial_call_pending"] = True
                return state
            state["progress_repair_attempts"] = 0
            state["context"].pop("progress_feedback", None)
            try:
                self._dispatch(state, response)
            except Exception as error:  # noqa: BLE001
                _mark_dispatch_failure(state, error)
        return state

    def on_last(self, state: LLState) -> LLState:
        return _close_runtime(state, self._runtime)


class LLMKnowledge(KnowledgeBase):
    """Knowledge evaluation with value text isolated to this role."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = RoleRuntime("knowledge", ROLE_RESPONSE_TYPES["knowledge"])

    def on_init(self, **kwargs: Any) -> None:
        self._runtime.initialize(**kwargs)

    def on_observed_beliefs(
        self,
        state: KnowledgeState,
        sender: str,
        observation: Observation,
        beliefs: list[Belief],
        **kwargs: Any,
    ) -> KnowledgeState:
        receive_message(
            state,
            sender=sender,
            kind="observed_beliefs",
            payload={
                "observation": observation_to_data(observation),
                "beliefs": [belief_to_data(item) for item in beliefs],
            },
            metadata=kwargs,
        )
        return state

    def on_belief_update(
        self,
        state: KnowledgeState,
        sender: str,
        beliefs: list[Belief],
        **kwargs: Any,
    ) -> KnowledgeState:
        receive_message(
            state,
            sender=sender,
            kind="belief_update",
            payload={"beliefs": [belief_to_data(item) for item in beliefs]},
            metadata=kwargs,
        )
        return state

    def on_belief_request(
        self, state: KnowledgeState, sender: str, **kwargs: Any
    ) -> KnowledgeState:
        receive_message(
            state,
            sender=sender,
            kind="belief_request",
            payload={},
            metadata=kwargs,
        )
        return state

    def _dispatch(self, state: KnowledgeState, response: KnowledgeResponse) -> None:
        metadata = _metadata_from_snapshot(self._runtime)
        source_id = metadata.get("evidence_id")
        evaluation_id: str | None = None
        if source_id is not None and (
            response.evaluated_observations
            or response.memory_beliefs
            or response.high_level_beliefs
        ):
            sequence = int(state["evaluation_sequence"])
            state["evaluation_sequence"] = sequence + 1
            evaluation_id = f"evaluation:{sequence}"
            state["evaluation_ids"].append(evaluation_id)
        outgoing_meta = {
            "evidence_id": evaluation_id,
            "source_evidence_id": source_id,
            "cycle_id": metadata.get("cycle_id"),
        }
        if response.evaluated_observations:
            observations = [
                Observation(
                    content=item.summary,
                    observation_type="knowledge-evaluation",
                    value=item.value,
                )
                for item in response.evaluated_observations
            ]
            send_message(
                state,
                recipient=MEMORY_ID,
                kind="send_observations",
                metadata=outgoing_meta,
                dispatch=lambda **meta: state.outbox.send_observations(
                    MEMORY_ID, observations, **meta
                ),
            )
        if response.memory_beliefs:
            beliefs = [
                belief_from_payload(item, value_source=KNOWLEDGE_ID)
                for item in response.memory_beliefs
            ]
            send_message(
                state,
                recipient=MEMORY_ID,
                kind="send_belief_memories",
                metadata=outgoing_meta,
                dispatch=lambda **meta: state.outbox.send_belief_memories(
                    MEMORY_ID, beliefs, **meta
                ),
            )
        if response.high_level_beliefs:
            beliefs = [
                belief_from_payload(item, value_source=KNOWLEDGE_ID)
                for item in response.high_level_beliefs
            ]
            send_message(
                state,
                recipient=HL_ID,
                kind="send_beliefs",
                metadata=outgoing_meta,
                dispatch=lambda **meta: state.outbox.send_beliefs(
                    HL_ID, beliefs, **meta
                ),
            )

    def step(self, state: KnowledgeState) -> KnowledgeState:
        response = self._runtime.call(state, self.module_id)
        if response is not None:
            try:
                self._dispatch(state, cast(KnowledgeResponse, response))
            except Exception as error:  # noqa: BLE001
                _mark_dispatch_failure(state, error)
        return state

    def on_last(self, state: KnowledgeState) -> KnowledgeState:
        return _close_runtime(state, self._runtime)


class LLMHighLevelReasoner(HLReasonerBase):
    """High-level reasoner that exclusively owns initial desires."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = RoleRuntime("hl_reasoner", ROLE_RESPONSE_TYPES["hl_reasoner"])

    def on_init(self, **kwargs: Any) -> None:
        self._runtime.initialize(**kwargs)

    def on_belief_update(
        self,
        state: HLState,
        sender: str,
        beliefs: list[Belief],
        **kwargs: Any,
    ) -> HLState:
        receive_message(
            state,
            sender=sender,
            kind="belief_update",
            payload={"beliefs": [belief_to_data(item) for item in beliefs]},
            metadata=kwargs,
        )
        return state

    def on_goal_update(
        self,
        state: HLState,
        sender: str,
        goals: list[Goal],
        **kwargs: Any,
    ) -> HLState:
        receive_message(
            state,
            sender=sender,
            kind="goal_update",
            payload={"goals": [goal_to_data(item) for item in goals]},
            metadata=kwargs,
        )
        return state

    def on_model(
        self, state: HLState, sender: str, model: Any, **kwargs: Any
    ) -> HLState:
        normalized_model = json_safe(model)
        state["learner_model"] = normalized_model
        receive_message(
            state,
            sender=sender,
            kind="learner_model",
            payload={"model": normalized_model},
            metadata=kwargs,
        )
        return state

    def _dispatch(self, state: HLState, response: HLReasonerResponse) -> None:
        metadata = _metadata_from_snapshot(self._runtime)
        input_evaluation_id = metadata.get("evidence_id")
        revision_id = metadata.get("revision_id")
        if revision_id and revision_id not in state["used_revision_ids"]:
            state["used_revision_ids"].append(revision_id)
        if response.request_beliefs:
            send_message(
                state,
                recipient=KNOWLEDGE_ID,
                kind="request_beliefs",
                metadata={"revision_id": revision_id},
                dispatch=lambda **meta: state.outbox.request_beliefs(
                    KNOWLEDGE_ID, **meta
                ),
            )
        if response.belief_updates:
            beliefs = [belief_from_payload(item) for item in response.belief_updates]
            send_message(
                state,
                recipient=KNOWLEDGE_ID,
                kind="send_beliefs",
                metadata={"revision_id": revision_id},
                dispatch=lambda **meta: state.outbox.send_beliefs(
                    KNOWLEDGE_ID, beliefs, **meta
                ),
            )
        for payload in response.goals:
            goal = goal_from_payload(payload)
            if not _changed_goal(state, GOAL_GRAPH_ID, goal):
                continue
            goal_hash = _goal_hash(goal)
            goal.extras.update(
                goal_payload_sha256=goal_hash,
                input_evaluation_id=input_evaluation_id,
            )
            state["authored_goals"].append(goal_to_data(goal))
            send_message(
                state,
                recipient=GOAL_GRAPH_ID,
                kind="send_goals",
                metadata={
                    "goal_id": payload.goal_id,
                    "goal_payload_sha256": goal_hash,
                    "input_evaluation_id": input_evaluation_id,
                    "source_ids": payload.source_ids,
                    "revision_id": revision_id,
                },
                dispatch=lambda goal=goal, **meta: state.outbox.send_goals(
                    GOAL_GRAPH_ID, [goal], **meta
                ),
            )
        if response.learner_task is not None:
            state["learner_requested"] = True
            send_message(
                state,
                recipient=HL_LEARNER_ID,
                kind="send_learner_task",
                dispatch=lambda **meta: state.outbox.send_learner_task(
                    HL_LEARNER_ID, response.learner_task, **meta
                ),
            )
        if response.request_model:
            state["learner_requested"] = True
            send_message(
                state,
                recipient=HL_LEARNER_ID,
                kind="request_model",
                dispatch=lambda **meta: state.outbox.request_model(
                    HL_LEARNER_ID, **meta
                ),
            )

    def step(self, state: HLState) -> HLState:
        response = self._runtime.call(state, self.module_id)
        if response is not None:
            try:
                self._dispatch(state, cast(HLReasonerResponse, response))
            except Exception as error:  # noqa: BLE001
                _mark_dispatch_failure(state, error)
        return state

    def on_last(self, state: HLState) -> HLState:
        return _close_runtime(state, self._runtime)


class LLMGoalGraph(GoalGraphBase):
    """LLM-selected goal routing with bounded feedback for unresolvable relays."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = RoleRuntime("goal_graph", ROLE_RESPONSE_TYPES["goal_graph"])

    def on_init(self, **kwargs: Any) -> None:
        self._runtime.initialize(**kwargs)

    def on_goal_request(
        self, state: GoalGraphState, sender: str, **kwargs: Any
    ) -> GoalGraphState:
        receive_message(
            state,
            sender=sender,
            kind="goal_request",
            payload={},
            metadata=kwargs,
        )
        return state

    def on_goal_update(
        self,
        state: GoalGraphState,
        sender: str,
        goals: list[Goal],
        **kwargs: Any,
    ) -> GoalGraphState:
        values = [goal_to_data(item) for item in goals]
        for goal, value in zip(goals, values, strict=True):
            goal_id = _goal_id(goal)
            if goal_id is not None:
                prior = state["goals"].get(goal_id)
                if prior is not None and sender == HL_ID and prior.get("extras", {}).get("status") == "achieved":
                    value = prior
                state["goals"][goal_id] = value
                if sender == LL_ID:
                    state["progress_ids"].append(goal_id)
        receive_message(
            state,
            sender=sender,
            kind="goal_update",
            payload={"goals": values},
            metadata=kwargs,
        )
        return state

    def _goal(self, state: GoalGraphState, goal_id: str) -> Goal:
        value = state["goals"].get(goal_id)
        if value is None:
            raise ValueError("unknown goal identity")
        return goal_from_data(value)

    def _relay_issues(self, state: GoalGraphState, response: GoalGraphResponse) -> list[str]:
        """Describe existing identity/provenance problems without selecting goals."""
        issues = []
        for goal_id in response.deliver_goal_ids + response.relay_progress_ids:
            if goal_id not in state["goals"]:
                issues.append(f"Goal {goal_id!r} has not been received, so its native payload is unavailable.")
        for goal_id in response.relay_progress_ids:
            if goal_id not in state["progress_ids"]:
                issues.append(f"No low-level progress update has been received for {goal_id!r}. A high-level intention is not a progress report.")
        return issues

    def _dispatch(self, state: GoalGraphState, response: GoalGraphResponse) -> None:
        requested = any(item["kind"] == "goal_request" and item["sender"] == LL_ID
                        for item in self._runtime.last_snapshot)
        if requested and not response.deliver_goal_ids:
            # Transport the model's empty selection so the requester can reconsider.
            send_message(
                state,
                recipient=LL_ID,
                kind="send_goals",
                dispatch=lambda **meta: state.outbox.send_goals(LL_ID, [], **meta),
            )
        for goal_id in response.deliver_goal_ids:
            goal = self._goal(state, goal_id)
            if not _changed_goal(state, LL_ID, goal, requested=requested):
                continue
            send_message(
                state,
                recipient=LL_ID,
                kind="send_goals",
                metadata={
                    "goal_id": goal_id,
                    "goal_payload_sha256": goal.extras.get("goal_payload_sha256"),
                    "input_evaluation_id": goal.extras.get("input_evaluation_id"),
                },
                dispatch=lambda goal=goal, **meta: state.outbox.send_goals(
                    LL_ID, [goal], **meta
                ),
            )
        for goal_id in response.relay_progress_ids:
            if goal_id not in state["progress_ids"]:
                raise ValueError("goal progress was not received from LL")
            goal = self._goal(state, goal_id)
            if not _changed_goal(state, HL_ID, goal):
                continue
            send_message(
                state,
                recipient=HL_ID,
                kind="send_goals",
                metadata={
                    "goal_id": goal_id,
                    "goal_payload_sha256": goal.extras.get("goal_payload_sha256"),
                },
                dispatch=lambda goal=goal, **meta: state.outbox.send_goals(
                    HL_ID, [goal], **meta
                ),
            )

    def step(self, state: GoalGraphState) -> GoalGraphState:
        previous_text = state["text_state"]
        response = self._runtime.call(state, self.module_id)
        if response is not None:
            response = cast(GoalGraphResponse, response)
            issues = self._relay_issues(state, response)
            if issues:
                feedback = {"issues": issues, "previous_response": response.model_dump(mode="json"),
                            "attempt": state["relay_repair_attempts"] + 1,
                            "instruction": "Reconsider your response using the received messages and their senders. Choose the goal selections yourself; an empty selection is valid. No message from the previous response was sent."}
                state["relay_feedback_history"].append({**feedback, "response_id": state["last_response_id"]})
                if state["relay_repair_attempts"] >= 2:
                    _mark_dispatch_failure(state, ValueError("unresolved goal relay after two feedback attempts"))
                    return state
                state["relay_repair_attempts"] += 1
                state["context"]["goal_relay_feedback"] = feedback
                state["text_state"] = previous_text
                # Reconsider the same native inputs; preserve any newer arrivals too.
                state["pending"] = list(self._runtime.last_snapshot) + list(state["pending"])
                state["initial_call_pending"] = True
                return state
            state["context"].pop("goal_relay_feedback", None)
            state["relay_repair_attempts"] = 0
            try:
                self._dispatch(state, response)
            except Exception as error:  # noqa: BLE001
                _mark_dispatch_failure(state, error)
        return state

    def on_last(self, state: GoalGraphState) -> GoalGraphState:
        return _close_runtime(state, self._runtime)


class LLMMemory(MemoryBase):
    """LLM-backed memory with compact received evidence only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = RoleRuntime("memory", ROLE_RESPONSE_TYPES["memory"])

    def on_init(self, **kwargs: Any) -> None:
        self._runtime.initialize(**kwargs)

    def _receive_items(
        self,
        state: MemoryState,
        sender: str,
        kind: str,
        items: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        prepared: list[dict[str, Any]] = []
        for item in items:
            sequence = int(state["memory_sequence"])
            state["memory_sequence"] = sequence + 1
            prepared.append(
                {"memory_id": f"memory:{sequence}", "kind": kind, "value": item}
            )
        receive_message(
            state,
            sender=sender,
            kind=f"{kind}_update",
            payload={"memories": prepared},
            metadata=metadata,
        )

    def on_observation_update(
        self,
        state: MemoryState,
        sender: str,
        observations: list[Observation],
        **kwargs: Any,
    ) -> MemoryState:
        self._receive_items(
            state,
            sender,
            "observation",
            [observation_to_data(item) for item in observations],
            kwargs,
        )
        return state

    def on_belief_update(
        self,
        state: MemoryState,
        sender: str,
        beliefs: list[Belief],
        **kwargs: Any,
    ) -> MemoryState:
        self._receive_items(
            state,
            sender,
            "belief",
            [belief_to_data(item) for item in beliefs],
            kwargs,
        )
        return state

    def on_memory_request(
        self, state: MemoryState, sender: str, **kwargs: Any
    ) -> MemoryState:
        receive_message(
            state,
            sender=sender,
            kind="memory_request",
            payload={},
            metadata=kwargs,
        )
        return state

    def _available_memories(self, state: MemoryState) -> dict[str, dict[str, Any]]:
        """Index received records by their actual memory identity."""
        available = {item["memory_id"]: item for item in state["memories"]}
        for message in self._runtime.last_snapshot:
            for item in message.get("payload", {}).get("memories", []):
                available[item["memory_id"]] = item
        return available

    def _response_issues(self, state: MemoryState, response: MemoryResponse) -> list[str]:
        """Check references before any retention or send, without selecting memories."""
        available = self._available_memories(state)
        requesters = {message["sender"] for message in self._runtime.last_snapshot
                      if message["kind"] == "memory_request"}
        issues = [f"Unknown retained memory_id: {memory_id}"
                  for memory_id in response.retained_memory_ids if memory_id not in available]
        for delivery in response.deliveries:
            if delivery.recipient not in requesters:
                issues.append(f"Recipient {delivery.recipient} has not requested memories in this batch.")
            issues.extend(f"Unknown delivered memory_id: {memory_id}"
                          for memory_id in delivery.memory_ids if memory_id not in available)
        return issues

    def _dispatch(self, state: MemoryState, response: MemoryResponse) -> None:
        available = self._available_memories(state)
        issues = self._response_issues(state, response)
        if issues:
            raise ValueError("invalid memory references")
        for memory_id in response.retained_memory_ids:
            if not any(item["memory_id"] == memory_id for item in state["memories"]):
                state["memories"].append(available[memory_id])
        for delivery in response.deliveries:
            values: list[Belief | Observation] = []
            for memory_id in delivery.memory_ids:
                item = available[memory_id]
                values.append(
                    belief_from_data(item["value"])
                    if item["kind"] == "belief"
                    else observation_from_data(item["value"])
                )
            send_message(
                state,
                recipient=delivery.recipient,
                kind="send_memories",
                metadata=_metadata_from_snapshot(self._runtime),
                dispatch=lambda values=values, recipient=delivery.recipient, **meta: state.outbox.send_memories(
                    recipient, values, **meta
                ),
            )

    def step(self, state: MemoryState) -> MemoryState:
        previous_text = state["text_state"]
        response = self._runtime.call(state, self.module_id)
        if response is not None:
            response = cast(MemoryResponse, response)
            issues = self._response_issues(state, response)
            if issues:
                feedback = {
                    "issues": issues,
                    "available_memory_ids": list(self._available_memories(state)),
                    "previous_response": response.model_dump(mode="json"),
                    "attempt": state["memory_repair_attempts"] + 1,
                    "instruction": "Nothing was retained or sent. Choose your own corrected selection using received memory_id fields; nested source IDs are not memory IDs. Empty selections are valid.",
                }
                state["memory_feedback_history"].append({**feedback, "response_id": state["last_response_id"]})
                if state["memory_repair_attempts"] >= 2:
                    _mark_dispatch_failure(state, ValueError("unresolved memory references after two corrections"))
                    return state
                state["memory_repair_attempts"] += 1
                state["context"]["memory_feedback"] = feedback
                state["text_state"] = previous_text
                state["pending"] = list(self._runtime.last_snapshot) + list(state["pending"])
                state["initial_call_pending"] = True
                return state
            state["memory_repair_attempts"] = 0
            state["context"].pop("memory_feedback", None)
            try:
                self._dispatch(state, response)
            except Exception as error:  # noqa: BLE001
                _mark_dispatch_failure(state, error)
        return state

    def on_last(self, state: MemoryState) -> MemoryState:
        return _close_runtime(state, self._runtime)


class LLMLearner(LearnerBase):
    """Conditional learner shared only as the native learner behavior class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        role = "ll_learner" if self.module_id == LL_LEARNER_ID else "hl_learner"
        self._runtime = RoleRuntime(role, ROLE_RESPONSE_TYPES[role])
        self._paired_reasoner = LL_ID if role == "ll_learner" else HL_ID

    def on_init(self, **kwargs: Any) -> None:
        self._runtime.initialize(**kwargs)

    def _requested(self, state: LearnerState) -> None:
        state["involvement_requested_by_reasoner"] = True
        state["operational_involvement_decision"] = "requested_by_reasoner"
        state["involvement_status"] = "requested_incomplete"

    def on_task(
        self, state: LearnerState, sender: str, task: Any, **kwargs: Any
    ) -> LearnerState:
        self._requested(state)
        receive_message(
            state,
            sender=sender,
            kind="learner_task",
            payload={"task": task},
            metadata=kwargs,
        )
        return state

    def on_model_request(
        self, state: LearnerState, sender: str, **kwargs: Any
    ) -> LearnerState:
        self._requested(state)
        receive_message(
            state,
            sender=sender,
            kind="model_request",
            payload={},
            metadata=kwargs,
        )
        return state

    def on_memories(
        self,
        state: LearnerState,
        sender: str,
        memories: list[Belief | Observation],
        **kwargs: Any,
    ) -> LearnerState:
        payload = [
            {"kind": "belief", "value": belief_to_data(item)}
            if isinstance(item, Belief)
            else {"kind": "observation", "value": observation_to_data(item)}
            for item in memories
        ]
        receive_message(
            state,
            sender=sender,
            kind="memories",
            payload={"memories": payload},
            metadata=kwargs,
        )
        return state

    def _dispatch(self, state: LearnerState, response: LearnerResponse) -> None:
        if response.request_memories:
            send_message(
                state,
                recipient=MEMORY_ID,
                kind="request_memories",
                dispatch=lambda **meta: state.outbox.request_memories(
                    MEMORY_ID, **meta
                ),
            )
        if response.revision is not None:
            sequence = int(state["revision_sequence"])
            state["revision_sequence"] = sequence + 1
            revision_id = f"{self.module_id}:revision:{sequence}"
            state["current_model"] = response.revision
            state["revision_id"] = revision_id
            state["involvement_status"] = "revision_produced"
            model = {
                "status": "revision",
                "model": response.revision,
                "revision_id": revision_id,
            }
            send_message(
                state,
                recipient=self._paired_reasoner,
                kind="send_model",
                metadata={"revision_id": revision_id},
                dispatch=lambda **meta: state.outbox.send_model(
                    self._paired_reasoner, model, **meta
                ),
            )
            state["involvement_status"] = "revision_delivered"
        elif response.no_change:
            state["involvement_status"] = "requested_no_change"
            send_message(
                state,
                recipient=self._paired_reasoner,
                kind="send_model",
                dispatch=lambda **meta: state.outbox.send_model(
                    self._paired_reasoner,
                    {"status": "no_change", "model": None, "revision_id": None},
                    **meta,
                ),
            )

    def step(self, state: LearnerState) -> LearnerState:
        previous_text = state["text_state"]
        response = self._runtime.call(state, self.module_id)
        if response is not None:
            response = cast(LearnerResponse, response)
            if (not response.request_memories and response.revision is None
                    and not response.no_change and not state["waiting_requests"]):
                previous = state["context"].get("learner_reply_feedback", {})
                feedback = {
                    "attempt": previous.get("attempt", 0) + 1,
                    "previous_response": response.model_dump(mode="json"),
                    "instruction": "No message was sent and no reply is pending. You cannot request observations directly. Reconsider the received task: request memories if useful, author your own grounded revision, or explicitly report no_change when no revision is justified. Do not invent evidence.",
                }
                state["context"].setdefault("learner_reply_feedback_history", []).append(
                    {**feedback, "response_id": state["last_response_id"]})
                if feedback["attempt"] > 2:
                    _mark_dispatch_failure(state, ValueError("unanswered learner task after two corrections"))
                    return state
                state["context"]["learner_reply_feedback"] = feedback
                state["text_state"] = previous_text
                state["pending"] = list(self._runtime.last_snapshot) + list(state["pending"])
                state["initial_call_pending"] = True
                return state
            state["context"].pop("learner_reply_feedback", None)
            try:
                self._dispatch(state, response)
            except Exception as error:  # noqa: BLE001
                _mark_dispatch_failure(state, error)
        return state

    def on_last(self, state: LearnerState) -> LearnerState:
        return _close_runtime(state, self._runtime)


__all__ = [
    "LLMGoalGraph",
    "LLMHighLevelReasoner",
    "LLMKnowledge",
    "LLMLearner",
    "LLMLowLevelReasoner",
    "LLMMemory",
]
