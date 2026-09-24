"""Frozen model profiles, role schemas, and the small OpenAI call recorder."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from mha_exp_common.openai_budget import BudgetedOpenAIClient, BudgetExceededError, PRICING_POLICY_VERSION
from mha_exp_common.paid_run import (
    decode_api_key,
    encode_api_key,
    read_windows_credential,
)
from mha_exp_common.runtime_secret import load_runtime_credentials
from mhagenta import ActionStatus, Belief, Goal, Observation
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .prompts import compose_prompt
from .terminal import observe_terminal, publish_terminal, terminal_fields


STATE_SCHEMA_VERSION = "2-7-bw-module-state-v3"

MAX_REQUEST_ATTEMPTS = 3
MESSAGE_RESPONSE_TIMEOUT = 30.0
MODEL_POLICY_VERSION = "2-7-bw-nano-role-reasoning-v4"
FAST_MODEL = "gpt-5.4-nano-2026-03-17"
DELIBERATIVE_MODEL = FAST_MODEL
ActionName = Literal["Move-Left", "Move-Right", "Pick-Up", "Put-Down"]
LearnerId = Literal["learner_0", "learner_1"]


@dataclass(frozen=True)
class ModelProfile:
    """One immutable API treatment profile."""

    name: Literal["fast", "deliberative"]
    model: str
    reasoning_effort: Literal["low", "high"] | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ModelPolicy:
    """The exact two-profile model treatment."""

    fast: ModelProfile
    deliberative: ModelProfile
    policy_version: str = MODEL_POLICY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "fast": self.fast.as_dict(),
            "deliberative": self.deliberative.as_dict(),
            "fallback": None,
        }


MODEL_POLICY = ModelPolicy(
    fast=ModelProfile("fast", FAST_MODEL, "low"),
    deliberative=ModelProfile("deliberative", DELIBERATIVE_MODEL, "high"),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BeliefPayload(StrictModel):
    predicate: str = Field(min_length=1, max_length=80)
    arguments: list[str] = Field(default_factory=list, max_length=16)
    value: float | None = None
    rationale: str | None = Field(default=None, max_length=500)
    source_ids: list[str] = Field(default_factory=list, max_length=32)


class GoalPayload(StrictModel):
    goal_id: str = Field(min_length=1, max_length=80)
    predicate: str = Field(min_length=1, max_length=80)
    arguments: list[str] = Field(min_length=1, max_length=16)
    status: Literal["pending", "active", "achieved", "failed"] = "pending"
    source_ids: list[str] = Field(default_factory=list, max_length=32)


class GoalProgress(StrictModel):
    goal_id: str = Field(min_length=1, max_length=80)
    status: Literal["pending", "active", "achieved", "failed"]
    note: str | None = Field(default=None, max_length=500)


class EvaluatedObservation(StrictModel):
    summary: str = Field(min_length=1, max_length=1000)
    value: float | None = None
    rationale: str | None = Field(default=None, max_length=500)


class MemoryDelivery(StrictModel):
    recipient: LearnerId
    memory_ids: list[str] = Field(default_factory=list, max_length=32)


class LLReasonerResponse(StrictModel):
    state: str
    request_observation: bool = False
    beliefs: list[BeliefPayload] = Field(default_factory=list)
    request_goals: bool = False
    action: ActionName | None = None
    goal_progress: list[GoalProgress] = Field(default_factory=list)
    learner_task: str | None = Field(default=None, max_length=1000)
    request_model: bool = False
    source_ids: list[str] = Field(default_factory=list, max_length=32)


class KnowledgeResponse(StrictModel):
    state: str
    evaluated_observations: list[EvaluatedObservation] = Field(default_factory=list)
    memory_beliefs: list[BeliefPayload] = Field(default_factory=list)
    high_level_beliefs: list[BeliefPayload] = Field(default_factory=list)


class HLReasonerResponse(StrictModel):
    state: str
    request_beliefs: bool = False
    belief_updates: list[BeliefPayload] = Field(default_factory=list)
    goals: list[GoalPayload] = Field(default_factory=list)
    learner_task: str | None = Field(default=None, max_length=1000)
    request_model: bool = False


class GoalGraphResponse(StrictModel):
    state: str
    deliver_goal_ids: list[str] = Field(default_factory=list, max_length=32)
    relay_progress_ids: list[str] = Field(default_factory=list, max_length=32)


class MemoryResponse(StrictModel):
    state: str
    retained_memory_ids: list[str] = Field(default_factory=list, max_length=64)
    deliveries: list[MemoryDelivery] = Field(default_factory=list, max_length=8)


class LearnerResponse(StrictModel):
    state: str
    request_memories: bool = False
    revision: str | None = Field(default=None, max_length=4000)
    no_change: bool = False

    @model_validator(mode="after")
    def revision_is_unambiguous(self) -> LearnerResponse:
        if self.revision is not None and self.no_change:
            raise ValueError("revision and no_change are mutually exclusive")
        return self


class PreflightResponse(StrictModel):
    ok: Literal[True]


ROLE_RESPONSE_TYPES: dict[str, type[BaseModel]] = {
    "ll_reasoner": LLReasonerResponse,
    "knowledge": KnowledgeResponse,
    "hl_reasoner": HLReasonerResponse,
    "goal_graph": GoalGraphResponse,
    "memory": MemoryResponse,
    "ll_learner": LearnerResponse,
    "hl_learner": LearnerResponse,
}


def json_safe(value: Any) -> Any:
    """Convert supported typed values to JSON-native data, rejecting surprises."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal")
        return str(value)
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    raise TypeError(f"unsupported persistent value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the frozen canonical JSON representation used by evidence hashes."""

    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def observation_to_data(observation: Observation) -> dict[str, Any]:
    return {
        "content": json_safe(observation.content),
        "observation_type": observation.observation_type,
        "value": observation.value,
    }


def observation_from_data(value: dict[str, Any]) -> Observation:
    return Observation(
        content=value["content"],
        observation_type=value.get("observation_type"),
        value=value.get("value"),
    )


def belief_to_data(belief: Belief) -> dict[str, Any]:
    return {
        "predicate": belief.predicate,
        "arguments": json_safe(belief.arguments),
        "extras": json_safe(belief.extras),
    }


def belief_from_payload(payload: BeliefPayload, *, value_source: str | None = None) -> Belief:
    extras = {
        "value": payload.value,
        "rationale": payload.rationale,
        "source_ids": list(payload.source_ids),
        "value_source": value_source if payload.value is not None else None,
    }
    return Belief(
        predicate=payload.predicate,
        arguments=tuple(payload.arguments),
        extras={key: item for key, item in extras.items() if item not in (None, [])}
        or None,
    )


def belief_from_data(value: dict[str, Any]) -> Belief:
    return Belief(
        predicate=str(value["predicate"]),
        arguments=tuple(value.get("arguments", [])),
        extras=value.get("extras"),
    )


def goal_to_data(goal: Goal) -> dict[str, Any]:
    return {
        "state": [belief_to_data(item) for item in goal.state],
        "extras": json_safe(goal.extras),
    }


def goal_from_payload(payload: GoalPayload) -> Goal:
    belief = Belief(payload.predicate, tuple(payload.arguments))
    return Goal(
        state=[belief],
        extras={
            "goal_id": payload.goal_id,
            "status": payload.status,
            "source_ids": list(payload.source_ids),
        },
    )


def goal_from_data(value: dict[str, Any]) -> Goal:
    return Goal(
        state=[belief_from_data(item) for item in value.get("state", [])],
        extras=dict(value.get("extras", {})),
    )


def action_status_to_data(status: ActionStatus) -> dict[str, Any]:
    return {"status": json_safe(status.status)}


def action_status_from_data(value: dict[str, Any]) -> ActionStatus:
    return ActionStatus(value.get("status"))


def empty_usage(max_budget_usd: str) -> dict[str, Any]:
    return {
        "pricing_policy_version": PRICING_POLICY_VERSION,
        "budget_source": "estimated",
        "max_budget_usd": str(max_budget_usd),
        "forwarded_request_count": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": "0",
        "budget_exhausted": False,
    }


def initial_role_state(
    *,
    role: str,
    text_state: str,
    profile: ModelProfile,
    lifecycle: dict[str, Any],
    max_budget_usd: str,
    initial_call_pending: bool,
    value_condition: str,
) -> dict[str, Any]:
    """Return the complete common JSON persistence schema for one LLM role."""

    return {
        **terminal_fields(),
        "state_schema_version": STATE_SCHEMA_VERSION,
        "role": role,
        "text_state": text_state,
        "pending": [],
        "context": {}, "last_sent_payloads": {}, "coalesced_observations": 0, "suppressed_messages": 0, "goal_ledger": {},
        "initial_call_pending": initial_call_pending,
        "in_flight": None,
        "sent_boundaries": [],
        "processed_boundaries": [],
        "boundary_sequence": 0,
        "received_sequence": 0,
        "input_sequence": 0,
        "processed_input_ids": [],
        "selected_model": profile.model,
        "reasoning_effort": profile.reasoning_effort,
        "actual_models": [],
        "usage": empty_usage(max_budget_usd),
        "last_response_id": None,
        "response_records": 0,
        "repair_count": 0,
        "api_retry_count": 0,
        "api_recovery_count": 0,
        "retry_not_before": 0.0,
        "request_failure_history": [],
        "waiting_requests": {},
        "request_timeout_history": [],
        "failure": None,
        "termination_reason": None,
        "incomplete_reason": None,
        "admission_open": True,
        "lifecycle": json_safe(lifecycle),
        "value_condition": value_condition,
        "response_file": "llm_responses/{module_id}.jsonl",
    }


def normalized_failure(
    *,
    kind: str,
    stage: str,
    module_time: float,
    response_id: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "stage": stage,
        "message": "operation failed",
        "module_time": float(module_time),
        "response_id": response_id,
    }


def _raw_response(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return json_safe(value)


class RoleRuntime:
    """Small composition helper for one role's API call and raw evidence."""

    def __init__(self, role: str, response_type: type[BaseModel]) -> None:
        self.role = role
        self.response_type = response_type
        self._client: Any = None
        self._environment_prompt = ""
        self._value_system = ""
        self._response_root: str | Path = "/out/llm_responses"
        self._request_timeout = 60.0
        self._profile = MODEL_POLICY.fast
        self.last_snapshot: list[dict[str, Any]] = []

    def initialize(
        self,
        *,
        credential_path: str,
        environment_prompt: str,
        value_system: str,
        profile: dict[str, Any],
        max_budget_usd: str,
        response_log_root: str,
        request_timeout: float,
    ) -> None:
        """Create one estimated-budget client from the runtime-only secret."""

        self._environment_prompt = environment_prompt
        self._value_system = value_system
        self._response_root = response_log_root
        self._request_timeout = float(request_timeout)
        self._profile = ModelProfile(**profile)
        encoded_key = ""
        api_key = ""
        try:
            encoded_key, _ = load_runtime_credentials(
                credential_path, budget_source="estimated"
            )
            api_key = decode_api_key(encoded_key)
            self._client = BudgetedOpenAIClient(
                api_key=api_key,
                max_budget_usd=max_budget_usd,
                model_ids=[self._profile.model],
                budget_source="estimated",
                timeout=self._request_timeout,
                max_retries=0,
            )
        finally:
            encoded_key = ""
            api_key = ""

    def set_client_for_testing(
        self,
        client: Any,
        *,
        profile: ModelProfile | None = None,
        response_root: Path | None = None,
        environment: str = "test environment",
        value_system: str = "",
    ) -> None:
        self._client = client
        self._profile = profile or self._profile
        self._environment_prompt = environment
        self._value_system = value_system
        if response_root is not None:
            self._response_root = response_root

    def close(self) -> None:
        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _append_record(self, module_id: str, record: dict[str, Any]) -> None:
        response_root = Path(self._response_root)
        response_root.mkdir(parents=True, exist_ok=True)
        path = response_root / f"{module_id}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")

    def _limits_reached(self, state: Any) -> bool:
        """Preserve scientific time/budget stops before admitting any retry."""
        if observe_terminal(state) is not None:
            return True
        if float(state.time) >= float(state["lifecycle"]["behavior_cutoff"]):
            state["termination_reason"] = "time_limit"
        elif state["usage"].get("budget_exhausted"):
            state["termination_reason"] = "budget_exhausted"
        if state["termination_reason"]:
            publish_terminal(state, state["termination_reason"])
            state["admission_open"] = False
            state.outbox.terminate_agent(f"Experiment 2-7-BW {state['termination_reason']}")
            return True
        return False

    def _wake_missing_replies(self, state: Any) -> None:
        """Give the model timed reply evidence; never replay physical actions."""
        now = float(state.time)
        overdue = []
        for key, waiting in state["waiting_requests"].items():
            if now < waiting["next_reminder_at"]:
                continue
            if waiting["reminders"] >= MAX_REQUEST_ATTEMPTS - 1:
                state["failure"] = normalized_failure(
                    kind="ResponseTimeout", stage="message_reply", module_time=now)
                state["admission_open"] = False
                publish_terminal(state, "execution_error")
                state.outbox.terminate_agent("Experiment 2-7-BW unanswered request after two reminders")
                return
            waiting["reminders"] += 1
            waiting["next_reminder_at"] = now + MESSAGE_RESPONSE_TIMEOUT
            note = {"request": key, "kind": waiting["kind"], "recipient": waiting["recipient"],
                    "elapsed_seconds": now - waiting["first_sent_at"], "reminder": waiting["reminders"]}
            overdue.append(note)
            state["request_timeout_history"].append({**note, "module_time": now})
        if overdue:
            state["context"]["unanswered_requests"] = overdue
            state["initial_call_pending"] = True

    def call(self, state: Any, module_id: str) -> BaseModel | None:
        """Attempt queued work once per step; retain inputs for two timed retries."""
        if self._limits_reached(state):
            return None
        if state["failure"] is not None:
            state.outbox.terminate_agent("Experiment 2-7-BW recorded module failure")
            return None
        self._wake_missing_replies(state)
        if state["failure"] is not None or float(state.time) < state["retry_not_before"]:
            return None
        pending = list(state["pending"])
        self.last_snapshot = pending
        if not pending and not state["initial_call_pending"]:
            return None
        remaining = float(state["lifecycle"]["agent_exec_duration"]) - float(state.time)
        if remaining < self._request_timeout:
            state["admission_open"] = False
            state["incomplete_reason"] = "insufficient_time_for_api_call"
            return None
        input_sequence = int(state["input_sequence"])
        state["input_sequence"] = input_sequence + 1
        input_id = f"{module_id}:input:{input_sequence}"
        state["in_flight"] = {"input_id": input_id,
                              "message_ids": [item["message_id"] for item in pending],
                              "started_at": float(state.time)}
        current_context = dict(state["context"])
        for key in ("active_goals", "goal_ledger", "current_observation", "learner_model", "primary_desires",
                    "goals", "progress_ids", "memories", "current_model", "action_observation_state"):
            if hasattr(state, key):
                current_context[key] = json_safe(state[key])
        current_context["module_time"] = float(state.time)
        current_context["waiting_requests"] = json_safe(state["waiting_requests"])
        attempt = state["api_retry_count"]
        feedback = state["context"].get("request_feedback", {})
        schema_retry = bool(attempt and feedback.get("error") in {"ValidationError", "ValueError"})
        prompt = compose_prompt(role=self.role, environment=self._environment_prompt,
            text_state=state["text_state"], pending_messages=pending, value_system=self._value_system,
            schema_retry=schema_retry, current_context=current_context)
        request = {"model": self._profile.model, "input": prompt,
                   "text_format": self.response_type,
                   "max_output_tokens": 8192 if self._profile.reasoning_effort == "high" else 4096,
                   "store": False}
        if self._profile.reasoning_effort is not None:
            request["reasoning"] = {"effort": self._profile.reasoning_effort}
        started = time.monotonic()
        record = {"schema_version": "2-7-bw-llm-response-v2", "role": self.role,
                  "input_id": input_id, "attempt": attempt, "schema_retry": schema_retry,
                  "prompt": prompt, "prompt_sha256": canonical_sha256(prompt),
                  "requested_model": self._profile.model, "reasoning_effort": self._profile.reasoning_effort,
                  "max_output_tokens": request["max_output_tokens"],
                  "actual_model": None, "response_id": None, "success": False, "raw_response": None}
        try:
            response = self._client.responses.parse(**request)
            record["raw_response"] = _raw_response(response)
            record["response_id"] = getattr(response, "id", None)
            record["actual_model"] = getattr(response, "model", None)
            output = getattr(response, "output_parsed", None)
            if output is None:
                raise ValueError("missing parsed output")
            parsed = self.response_type.model_validate(output)
            record["success"] = True
            if isinstance(record["actual_model"], str) and record["actual_model"] not in state["actual_models"]:
                state["actual_models"].append(record["actual_model"])
            if attempt:
                state["api_recovery_count"] += 1
                if schema_retry:
                    state["repair_count"] += 1
            state["api_retry_count"] = 0
            state["retry_not_before"] = 0.0
            state["context"].pop("request_feedback", None)
            state["text_state"] = str(parsed.state)
            # Remove only consumed identities; preserve arrivals/coalescing during a call.
            consumed = {item["message_id"] for item in pending}
            state["pending"][:] = [item for item in state["pending"] if item["message_id"] not in consumed]
            state["initial_call_pending"] = False
            state["processed_input_ids"].append(input_id)
            state["last_response_id"] = record["response_id"]
            state["usage"] = json_safe(self._client.usage_snapshot())
            if self._limits_reached(state):
                record["cancelled_at_termination"] = True
                return None
            return parsed
        except BudgetExceededError:
            record["error_kind"] = "BudgetExceededError"
            state["usage"] = json_safe(self._client.usage_snapshot())
            state["termination_reason"] = "budget_exhausted"
            self._limits_reached(state)
        except Exception as error:  # noqa: BLE001 - retain only normalized error metadata
            record["success"] = False
            if record["raw_response"] is None and isinstance(error, ValidationError):
                record["raw_response"] = {"validation_errors": error.errors(include_url=False, include_context=False)}
            record["error_kind"] = type(error).__name__
            state["usage"] = json_safe(self._client.usage_snapshot())
            state["api_retry_count"] += 1
            note = {"error": type(error).__name__, "attempt": state["api_retry_count"],
                    "module_time": float(state.time), "input_id": input_id}
            state["request_failure_history"].append(note)
            state["context"]["request_feedback"] = {
                **note, "instruction": "The previous request failed; no output was dispatched. Reconsider the retained inputs and return a complete valid response."}
            state["initial_call_pending"] = True
            if not self._limits_reached(state):
                if state["api_retry_count"] >= MAX_REQUEST_ATTEMPTS:
                    state["failure"] = normalized_failure(kind=type(error).__name__, stage="api_call",
                        module_time=float(state.time), response_id=record["response_id"])
                    state["admission_open"] = False
                    publish_terminal(state, "execution_error")
                    state.outbox.terminate_agent("Experiment 2-7-BW request failed after three attempts")
                else:
                    state["retry_not_before"] = float(state.time) + state["api_retry_count"]
        finally:
            record["latency_seconds"] = time.monotonic() - started
            state["usage"] = json_safe(self._client.usage_snapshot())
            record["usage"] = state["usage"]
            self._append_record(module_id, record)
            state["response_records"] += 1
            state["in_flight"] = None
        return None


def perform_api_preflight(
    api_key: str,
    model_policy: ModelPolicy = MODEL_POLICY,
    *,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Verify the two exact profiles with structured calls and no fallback."""

    client = None
    records: list[dict[str, Any]] = []
    try:
        if client_factory is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
        else:
            client = client_factory(api_key)
    finally:
        api_key = ""
    try:
        for profile in (model_policy.fast, model_policy.deliberative):
            request: dict[str, Any] = {
                "model": profile.model,
                "input": 'Return {"ok": true}.',
                "text_format": PreflightResponse,
                "max_output_tokens": 64,
                "store": False,
            }
            if profile.reasoning_effort is not None:
                request["reasoning"] = {"effort": profile.reasoning_effort}
            try:
                response = client.responses.parse(**request)
                parsed = PreflightResponse.model_validate(response.output_parsed)
                actual = getattr(response, "model", None)
                records.append(
                    {
                        "profile": profile.name,
                        "requested_model": profile.model,
                        "actual_model": actual,
                        "reasoning_effort": profile.reasoning_effort,
                        "success": bool(parsed.ok and actual == profile.model),
                    }
                )
            except Exception as error:  # noqa: BLE001
                records.append(
                    {
                        "profile": profile.name,
                        "requested_model": profile.model,
                        "actual_model": None,
                        "reasoning_effort": profile.reasoning_effort,
                        "success": False,
                        "error_kind": type(error).__name__,
                    }
                )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return {"success": all(item["success"] for item in records), "checks": records}


__all__ = [
    "ActionName",
    "BeliefPayload",
    "DELIBERATIVE_MODEL",
    "FAST_MODEL",
    "GoalGraphResponse",
    "GoalPayload",
    "HLReasonerResponse",
    "KnowledgeResponse",
    "LLReasonerResponse",
    "LearnerResponse",
    "MODEL_POLICY",
    "MODEL_POLICY_VERSION",
    "MemoryResponse",
    "ModelPolicy",
    "ModelProfile",
    "ROLE_RESPONSE_TYPES",
    "RoleRuntime",
    "STATE_SCHEMA_VERSION",
    "action_status_from_data",
    "action_status_to_data",
    "belief_from_data",
    "belief_from_payload",
    "belief_to_data",
    "canonical_json",
    "canonical_sha256",
    "decode_api_key",
    "encode_api_key",
    "goal_from_data",
    "goal_from_payload",
    "goal_to_data",
    "initial_role_state",
    "json_safe",
    "normalized_failure",
    "observation_from_data",
    "observation_to_data",
    "perform_api_preflight",
    "read_windows_credential",
]
