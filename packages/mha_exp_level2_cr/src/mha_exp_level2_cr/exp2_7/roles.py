"""Direct MHAgentA role implementations for experiment 2-7-CR."""

from __future__ import annotations

from typing import Any, Sequence

from mhagenta import ActionStatus, Belief, Goal, Observation
from mhagenta.bases import (
    GoalGraphBase, HLReasonerBase, KnowledgeBase, LearnerBase,
    LLReasonerBase, MemoryBase,
)
from mhagenta.states import GoalGraphState, HLState, KnowledgeState, LearnerState, LLState, MemoryState

from .llm import (
    HL_BOOTSTRAP_TASK, LL_BOOTSTRAP_TASK, LL_SURVIVAL_GOAL_ID,
    LL_SURVIVAL_HEALTH_THRESHOLD, GoalGraphResponse, HLReasonerResponse,
    KnowledgeResponse, LearnerResponse, LLReasonerResponse, MemoryResponse,
    RoleRuntime, compact_event, jsonable, normalize_action,
)


def _queue(state: Any, kind: str, sender: str, **payload: Any) -> Any:
    """Queue native work, retaining current context and one pending observation."""
    message = {"type": kind, "sender": sender, **jsonable(payload)}
    # Retain only the newest pending sensor payload; requests/progress are not dropped.
    if "observation" in payload and state["role"] == "knowledge":
        prior = [item for item in state["pending"] if "observation" in item]
        state["coalesced_observations"] += len(prior)
        state["pending"][:] = [item for item in state["pending"] if "observation" not in item]
    if message not in state["pending"]:
        state["pending"].append(message)
    if kind in {"goal_update", "belief_update", "model"} and "observation" not in payload:
        state["context"][kind + ":" + sender] = jsonable(payload)
    state["received"] += 1
    return state


def _changed(state: Any, recipient: str, kind: str, value: Any, *, requested: bool = False) -> bool:
    """Suppress unchanged outgoing content, except explicit request replies."""
    key = recipient + ":" + kind
    value = jsonable(value)
    if not requested and state["last_sent_payloads"].get(key) == value:
        state["suppressed_messages"] += 1
        return False
    state["last_sent_payloads"][key] = value
    return True


def _remember_goals(state: Any, goals: Sequence[Goal], *, progress: bool) -> list[Goal]:
    """Keep received identities and LL progress without selecting an intention."""
    values = []
    for goal in goals:
        value = jsonable(goal)
        goal_id = value["extras"]["goal_id"]
        if goal_id not in state["goal_ledger"]:
            for known_id, known in state["goal_ledger"].items():
                if known["state"] == value["state"]:
                    goal_id = known_id
                    value["extras"]["goal_id"] = known_id
                    break
        prior = state["goal_ledger"].get(goal_id)
        if progress and prior is None:
            state["suppressed_messages"] += 1
            continue
        if prior is not None:
            # Only the status can change on a progress report for an existing goal.
            status = value["extras"].get("status", "pending")
            value = {**prior, "extras": dict(prior["extras"])}
            if progress:
                value["extras"]["status"] = status
        state["goal_ledger"][goal_id] = value
        values.append(Goal([Belief(**item) for item in value["state"]], **value["extras"]))
    return values


def _beliefs(values: Sequence[Any]) -> list[Belief]:
    result: list[Belief] = []
    for item in values:
        extras = {
            name: value
            for name in ("value", "rationale")
            if (value := getattr(item, name, None)) is not None
        }
        result.append(Belief(item.predicate, tuple(item.arguments), extras=extras))
    return result


def _goals(values: Sequence[Any]) -> list[Goal]:
    return [
        Goal(
            [Belief(item.predicate, tuple(item.arguments))],
            goal_id=item.goal_id,
            status=item.status,
            primary=item.primary,
            order=item.order,
        )
        for item in values
    ]


def _observation(value: Any) -> Observation:
    return Observation(value.content, observation_type=value.observation_type, value=value.value)


def _stored_observation(value: Any) -> Observation:
    """Restore a native observation from JSON-native declared state."""

    if isinstance(value, dict):
        return Observation(**value)
    return value


def _memories(values: Sequence[Any]) -> list[Belief | Observation]:
    result: list[Belief | Observation] = []
    for item in values:
        result.append(_observation(item) if hasattr(item, "content") else _beliefs([item])[0])
    return result


def _finish(state: Any, response: Any) -> None:
    state["text_state"] = response.text_state
    state["pending"].clear()
    state["repair_feedback"] = None
    state["semantic_attempts"] = 0


def _sent(runtime: RoleRuntime, state: Any, kind: str, recipient: str, **data: Any) -> None:
    state["sent"] += 1
    compact_event(runtime.events, {"kind": "send", "type": kind, "recipient": recipient, **jsonable(data)})


class LLMLowLevelReasoner(LLReasonerBase):
    """Select atomic actions and maintain the documented local survival goal."""

    runtime: RoleRuntime

    def on_init(self, **kwargs: Any) -> None:
        self.runtime = RoleRuntime(self.module_id, **kwargs)

    def on_first(self, state: LLState) -> LLState:
        state.outbox.request_observation("perceptor_0")
        state.outbox.request_goals("goalgraph_0", request="initial goals")
        state.outbox.send_learner_task("learner_0", task=LL_BOOTSTRAP_TASK)
        state.outbox.request_model("learner_0")
        _sent(self.runtime, state, "request_observation", "perceptor_0")
        _sent(self.runtime, state, "request_goals", "goalgraph_0")
        _sent(self.runtime, state, "send_learner_task", "learner_0")
        _sent(self.runtime, state, "request_model", "learner_0")
        return state

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs: Any) -> LLState:
        needs = kwargs.get("survival_needs")
        if isinstance(needs, dict):
            state["survival_needs"] = {name: int(needs[name]) for name in ("health", "food", "drink", "energy")}
            urgent = state["survival_needs"]["health"] <= LL_SURVIVAL_HEALTH_THRESHOLD or any(
                state["survival_needs"][name] == 0 for name in ("food", "drink", "energy"))
            state["survival_override_active"] = urgent
            if urgent:
                state["active_goals"] = [jsonable(Goal(
                    [Belief("survive", ())], goal_id=LL_SURVIVAL_GOAL_ID,
                    status="active", local_override=True,
                ))]
            else:
                state["active_goals"] = list(state["goal_graph_goals"])
        state["latest_observation"] = jsonable(observation)
        return _queue(state, "observation", sender, observation=observation,
                      survival_needs=needs, active_goals=state["active_goals"])

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs: Any) -> LLState:
        state["last_action_status"] = jsonable(action_status)
        status = action_status.status
        if isinstance(status, dict):
            for achievement in status.get("new_achievements", []):
                if achievement not in state["achievements"]:
                    state["achievements"].append(achievement)
        terminal = isinstance(status, dict) and bool(status.get("terminal"))
        _queue(state, "action_status", sender, action_status=action_status)
        if isinstance(status, dict) and status.get("primary_goal_achieved") is True:
            state["primary_goal_achieved"] = True
            state["termination_reason"] = "task_completed"
            state["terminal_evidence"] = {
                "requested_action": status.get("requested_action"),
                "canonical_action": status.get("canonical_action"),
                "new_achievements": status.get("new_achievements"),
                "module_time": float(getattr(state, "time", 0.0)),
            }
            terminate = getattr(state.outbox, "terminate_agent", None)
            if callable(terminate):
                terminate("Experiment 2-7-CR primary goal achieved")
            return state
        if terminal:
            state["termination_reason"] = "death" if status.get("dead") else "environment_limit"
            state.outbox.terminate_agent(f"Experiment 2-7-CR {state['termination_reason']}")
            return state
        if float(getattr(state, "time", 0.0)) < self.runtime.behavior_duration:
            state.outbox.request_observation("perceptor_0")
            _sent(self.runtime, state, "request_observation", "perceptor_0")
        return state

    def on_goal_update(self, state: LLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> LLState:
        state["goal_graph_goals"] = jsonable(list(goals))
        if not state["survival_override_active"]:
            state["active_goals"] = jsonable(list(goals))
        return _queue(state, "goal_update", sender, goals=goals)

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs: Any) -> LLState:
        state["current_model"] = jsonable(model)
        return _queue(state, "model", sender, model=model)

    def step(self, state: LLState) -> LLState:
        try:
            primary_goal_achieved = bool(state["primary_goal_achieved"])
        except KeyError:
            primary_goal_achieved = False
        if primary_goal_achieved:
            return state
        observation_decision = any(item.get("type") == "observation" for item in state["pending"])
        response = self.runtime.call(state)
        if not isinstance(response, LLReasonerResponse):
            return state
        if observation_decision:
            action, source = normalize_action(response.action)
            if action is None:
                state["repair_feedback"] = {"error": "invalid_action", "value": response.action}
                state["semantic_attempts"] += 1
                if state["semantic_attempts"] > 2:
                    state["halted"] = True
                return state
            state["last_action"] = {"requested": response.action, "canonical": action, "normalization": source}
            state.outbox.request_action("actuator_0", action=action, requested_action=response.action,
                                        normalization=source)
            _sent(self.runtime, state, "request_action", "actuator_0", action=action, normalization=source)
        if observation_decision and response.beliefs and state["latest_observation"] is not None:
            state.outbox.send_beliefs(
                "knowledge_0",
                _stored_observation(state["latest_observation"]),
                _beliefs(response.beliefs),
                achievements=list(state["achievements"]),
                action_status=state["last_action_status"],
            )
            _sent(self.runtime, state, "send_beliefs", "knowledge_0")
        if response.goal_updates and _changed(state, "goalgraph_0", "progress", response.goal_updates):
            state.outbox.send_goal_update("goalgraph_0", _goals(response.goal_updates))
            _sent(self.runtime, state, "send_goal_update", "goalgraph_0")
        if response.learner_task is not None:
            state.outbox.send_learner_task("learner_0", response.learner_task)
            _sent(self.runtime, state, "send_learner_task", "learner_0")
        if response.request_goals:
            state.outbox.request_goals("goalgraph_0")
            _sent(self.runtime, state, "request_goals", "goalgraph_0")
        if response.request_model:
            state.outbox.request_model("learner_0")
            _sent(self.runtime, state, "request_model", "learner_0")
        _finish(state, response)
        return state

    def on_last(self, state: LLState) -> LLState:
        self.runtime.close()
        return state


class LLMKnowledge(KnowledgeBase):
    """Evaluate incoming beliefs/observations for memory and high-level reasoning."""

    def on_init(self, **kwargs: Any) -> None:
        self.runtime = RoleRuntime(self.module_id, **kwargs)

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation,
                            beliefs: Sequence[Belief], **kwargs: Any) -> KnowledgeState:
        state["context"]["environment_achievements"] = kwargs.get("achievements", [])
        return _queue(state, "belief_update", sender, observation=observation, beliefs=beliefs,
                      action_status=kwargs.get("action_status"))

    def on_belief_update(self, state: KnowledgeState, sender: str,
                         beliefs: Sequence[Belief], **kwargs: Any) -> KnowledgeState:
        return _queue(state, "belief_update", sender, beliefs=beliefs)

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs: Any) -> KnowledgeState:
        return _queue(state, "belief_request", sender)

    def step(self, state: KnowledgeState) -> KnowledgeState:
        response = self.runtime.call(state)
        if not isinstance(response, KnowledgeResponse):
            return state
        if response.observations:
            values = [_observation(item) for item in response.observations]
            state.outbox.send_observations("memory_0", values)
            _sent(self.runtime, state, "send_observations", "memory_0")
        if response.memory_beliefs:
            state.outbox.send_belief_memories("memory_0", _beliefs(response.memory_beliefs))
            _sent(self.runtime, state, "send_belief_memories", "memory_0")
        if response.high_level_beliefs:
            state.outbox.send_beliefs("hlreasoner_0", _beliefs(response.high_level_beliefs),
                                     environment_achievements=state["context"].get("environment_achievements", []))
            _sent(self.runtime, state, "send_beliefs", "hlreasoner_0")
        _finish(state, response)
        return state

    def on_last(self, state: KnowledgeState) -> KnowledgeState:
        self.runtime.close()
        return state


class LLMHighLevelReasoner(HLReasonerBase):
    """Translate evaluated beliefs into goals and high-level learning tasks."""

    def on_init(self, **kwargs: Any) -> None:
        self.runtime = RoleRuntime(self.module_id, **kwargs)

    def on_first(self, state: HLState) -> HLState:
        state.outbox.request_beliefs("knowledge_0")
        state.outbox.send_learner_task("learner_1", task=HL_BOOTSTRAP_TASK)
        state.outbox.request_model("learner_1")
        _sent(self.runtime, state, "request_beliefs", "knowledge_0")
        _sent(self.runtime, state, "send_learner_task", "learner_1")
        _sent(self.runtime, state, "request_model", "learner_1")
        return state

    def on_belief_update(self, state: HLState, sender: str, beliefs: Sequence[Belief], **kwargs: Any) -> HLState:
        state["context"]["environment_achievements"] = kwargs.get("environment_achievements", [])
        return _queue(state, "belief_update", sender, beliefs=beliefs)

    def on_goal_update(self, state: HLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> HLState:
        values = _remember_goals(state, goals, progress=True)
        return _queue(state, "goal_update", sender, goals=values)

    def on_model(self, state: HLState, sender: str, model: Any, **kwargs: Any) -> HLState:
        state["current_model"] = jsonable(model)
        return _queue(state, "model", sender, model=model)

    def step(self, state: HLState) -> HLState:
        response = self.runtime.call(state)
        if not isinstance(response, HLReasonerResponse):
            return state
        if response.beliefs:
            state.outbox.send_beliefs("knowledge_0", _beliefs(response.beliefs))
            _sent(self.runtime, state, "send_beliefs", "knowledge_0")
        goals = _remember_goals(state, _goals(response.goals), progress=False)
        if goals and _changed(state, "goalgraph_0", "goals", goals):
            state.outbox.send_goals("goalgraph_0", goals)
            _sent(self.runtime, state, "send_goals", "goalgraph_0")
        if response.learner_task is not None:
            state.outbox.send_learner_task("learner_1", response.learner_task)
            _sent(self.runtime, state, "send_learner_task", "learner_1")
        if response.request_beliefs:
            state.outbox.request_beliefs("knowledge_0")
            _sent(self.runtime, state, "request_beliefs", "knowledge_0")
        if response.request_model:
            state.outbox.request_model("learner_1")
            _sent(self.runtime, state, "request_model", "learner_1")
        _finish(state, response)
        return state

    def on_last(self, state: HLState) -> HLState:
        self.runtime.close()
        return state


class LLMGoalGraph(GoalGraphBase):
    """Maintain and distribute LLM-selected goals."""

    def on_init(self, **kwargs: Any) -> None:
        self.runtime = RoleRuntime(self.module_id, **kwargs)

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> GoalGraphState:
        values = _remember_goals(state, goals, progress=sender == "llreasoner_0")
        return _queue(state, "goal_update", sender, goals=values)

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs: Any) -> GoalGraphState:
        return _queue(state, "goal_request", sender)

    def step(self, state: GoalGraphState) -> GoalGraphState:
        messages = list(state["pending"])
        response = self.runtime.call(state)
        if not isinstance(response, GoalGraphResponse):
            return state
        for recipient, selected, source in (
            ("llreasoner_0", response.low_level_goals, "hlreasoner_0"),
            ("hlreasoner_0", response.high_level_goals, "llreasoner_0"),
        ):
            requested = any(item["type"] == "goal_request" and item["sender"] == recipient for item in messages)
            allowed = requested or any(item["type"] == "goal_update" and item["sender"] == source for item in messages)
            # The model selects IDs; the native received payload remains authoritative.
            values = [state["goal_ledger"][item.goal_id] for item in selected if item.goal_id in state["goal_ledger"]]
            if values and allowed and _changed(state, recipient, "goals", values, requested=requested):
                goals = [Goal([Belief(**belief) for belief in value["state"]], **value["extras"]) for value in values]
                state.outbox.send_goals(recipient, goals)
                _sent(self.runtime, state, "send_goals", recipient)
        _finish(state, response)
        return state

    def on_last(self, state: GoalGraphState) -> GoalGraphState:
        self.runtime.close()
        return state


class LLMMemory(MemoryBase):
    """Select compact memory batches for either learner."""

    def on_init(self, **kwargs: Any) -> None:
        self.runtime = RoleRuntime(self.module_id, **kwargs)

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs: Any) -> MemoryState:
        return _queue(state, "memory_request", sender)

    def on_observation_update(self, state: MemoryState, sender: str,
                              observations: Sequence[Observation], **kwargs: Any) -> MemoryState:
        return _queue(state, "observation_update", sender, observations=observations)

    def on_belief_update(self, state: MemoryState, sender: str,
                         beliefs: Sequence[Belief], **kwargs: Any) -> MemoryState:
        return _queue(state, "belief_update", sender, beliefs=beliefs)

    def step(self, state: MemoryState) -> MemoryState:
        requesters = {item["sender"] for item in state["pending"] if item["type"] == "memory_request"}
        response = self.runtime.call(state)
        if not isinstance(response, MemoryResponse):
            return state
        if "learner_0" in requesters:
            state.outbox.send_memories("learner_0", _memories(response.low_level_memories))
            _sent(self.runtime, state, "send_memories", "learner_0")
        if "learner_1" in requesters:
            state.outbox.send_memories("learner_1", _memories(response.high_level_memories))
            _sent(self.runtime, state, "send_memories", "learner_1")
        _finish(state, response)
        return state

    def on_last(self, state: MemoryState) -> MemoryState:
        self.runtime.close()
        return state


class LLMLearner(LearnerBase):
    """One direct learner class configured as the low- or high-level role."""

    def on_init(self, **kwargs: Any) -> None:
        self.runtime = RoleRuntime(self.module_id, **kwargs)
        self.reasoner_id = str(kwargs["reasoner_id"])

    def on_first(self, state: LearnerState) -> LearnerState:
        state.outbox.request_memories("memory_0")
        _sent(self.runtime, state, "request_memories", "memory_0")
        return state

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs: Any) -> LearnerState:
        state["requester"] = sender
        return _queue(state, "task", sender, task=task)

    def on_memories(self, state: LearnerState, sender: str,
                    memories: Sequence[Belief | Observation], **kwargs: Any) -> LearnerState:
        return _queue(state, "memories", sender, memories=memories)

    def on_model_request(self, state: LearnerState, sender: str, **kwargs: Any) -> LearnerState:
        state["requester"] = sender
        return _queue(state, "model_request", sender)

    def step(self, state: LearnerState) -> LearnerState:
        requested = any(item["type"] in {"task", "model_request"} for item in state["pending"])
        response = self.runtime.call(state)
        if not isinstance(response, LearnerResponse):
            return state
        recipient = state["requester"] or self.reasoner_id
        if _changed(state, recipient, "model", response.model, requested=requested):
            state.outbox.send_model(recipient, response.model)
            _sent(self.runtime, state, "send_model", recipient)
        state["current_model"] = response.model
        if response.request_memories:
            state.outbox.request_memories("memory_0")
            _sent(self.runtime, state, "request_memories", "memory_0")
        _finish(state, response)
        return state

    def on_last(self, state: LearnerState) -> LearnerState:
        self.runtime.close()
        return state


__all__ = ["LLMGoalGraph", "LLMHighLevelReasoner", "LLMKnowledge", "LLMLearner",
           "LLMLowLevelReasoner", "LLMMemory"]
