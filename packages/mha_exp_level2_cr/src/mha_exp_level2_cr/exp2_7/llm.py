"""Compact shared LLM boundary for experiment 2-7-CR.

Role behaviour lives in :mod:`roles`; this module only owns structured calls,
approximate budgets, model selection, schemas, and action-name normalisation.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from mha_exp_common.openai_budget import BudgetExceededError, BudgetedOpenAIClient
from mha_exp_common.runtime_secret import load_runtime_credentials
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .prompts import compose_prompt

FAST_MODEL_CANDIDATES = ("gpt-5.4-nano-2026-03-17",)
DELIBERATIVE_MODEL_CANDIDATES = FAST_MODEL_CANDIDATES
MODEL_POLICY_VERSION = "2-7-cr-nano-role-reasoning-v5"
CONTROL_TREATMENT = "cr-ll-local-survival-grounded-v2"
DETERMINISTIC_ACTION_ASSISTANCE = False
DETERMINISTIC_GOAL_ASSISTANCE = True
LL_SURVIVAL_GOAL_ID = "survive"
LL_SURVIVAL_HEALTH_THRESHOLD = 7
MAX_SEMANTIC_REPAIRS = 2
PROMPT_VERSION = "2-7-cr-context-routing-v26"
SYMBOLIC_OBSERVATION_FORMAT_VERSION = "cr-symbolic-predicates-grid-v3"
GOAL_CHAIN_POLICY_VERSION = "2-7-cr-stable-goal-relay-v14"
KNOWLEDGE_ADMISSION_POLICY_VERSION = "2-7-cr-knowledge-bounded-latest-v8"
PROMPT_CACHE_POLICY_VERSION = "2-7-cr-prompt-cache-v1"
RAW_SYMBOLIC_PREDICATES_HEADER = "RAW SYMBOLIC PREDICATES — AUTHORITATIVE"
DERIVED_SYMBOLIC_STATUS_HEADER = "DERIVED HUMAN-READABLE STATUS"
EXACT_FACED_CELL_HEADER = "EXACT FACED CELL — AUTHORITATIVE DO TARGET"
PARSED_SYMBOLIC_GRID_HEADER = "PARSED LOCAL GRID SUPPLEMENT"

PRIMARY_PROGRESS_ACHIEVEMENTS = (
    "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
    "make_stone_pickaxe", "place_furnace", "collect_coal", "collect_iron",
    "make_iron_pickaxe",
)
PRIMARY_COMPLETION_ACHIEVEMENT = "collect_diamond"
PRIMARY_ACHIEVEMENT_ORDER = (*PRIMARY_PROGRESS_ACHIEVEMENTS, PRIMARY_COMPLETION_ACHIEVEMENT)
PRIMARY_GOAL_ID = "primary_collect_diamond"

CRAFTER_ACTIONS = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)
CRAFTER_ACTION_ALIASES = {
    "left": "move_left", "right": "move_right", "up": "move_up", "down": "move_down",
    "move_north": "move_up", "move_south": "move_down", "move_west": "move_left",
    "move_east": "move_right", "interact": "do", "use": "do", "rest": "sleep",
    "wait": "noop",
}
LL_BASELINE_MODEL = (
    "Treat the latest Crafter observation, active Goal Graph goal, recent action "
    "status, and current model as the complete stimulus. Select one atomic action "
    "yourself as an immediate reflex, not a plan. Infer local interaction and "
    "movement opportunities from the observation and adapt after ineffective outcomes."
)
HL_BASELINE_MODEL = (
    "Select intentions from Knowledge beliefs, procedural recipe knowledge, trusted "
    "progress, current needs, and value evaluations. Keep collect_diamond primary "
    "and choose at most one canonical prerequisite beneath it."
)
LL_BOOTSTRAP_TASK = "Refine the low-level baseline only from grounded, evaluated memories."
HL_BOOTSTRAP_TASK = "Refine the high-level baseline only from grounded, evaluated memories."


@dataclass(frozen=True)
class ModelProfile:
    """One ordered model candidate profile."""
    name: Literal["fast", "deliberative"]
    candidates: tuple[str, ...]
    reasoning_effort: Literal["low", "high"] = "low"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"candidates": list(self.candidates)}


@dataclass(frozen=True)
class ModelPolicy:
    """Resolved account-visible model treatment."""
    fast: ModelProfile
    deliberative: ModelProfile
    available_model_ids: tuple[str, ...]
    policy_version: str = MODEL_POLICY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {"policy_version": self.policy_version, "fast": self.fast.as_dict(),
                "deliberative": self.deliberative.as_dict(),
                "available_model_ids": list(self.available_model_ids)}


def select_model_policy(available_model_ids: list[str] | tuple[str, ...]) -> ModelPolicy:
    """Keep visible candidates in their canonical priority order."""
    visible = tuple(dict.fromkeys(str(item) for item in available_model_ids))
    visible_set = set(visible)

    def resolve(name: Literal["fast", "deliberative"], candidates: tuple[str, ...]) -> ModelProfile:
        selected = tuple(model for model in candidates if model in visible_set)
        if not selected:
            raise RuntimeError(f"no account-visible {name} model candidate")
        return ModelProfile(name, selected, "high" if name == "deliberative" else "low")

    return ModelPolicy(resolve("fast", FAST_MODEL_CANDIDATES),
                       resolve("deliberative", DELIBERATIVE_MODEL_CANDIDATES), visible)


def normalize_action(value: Any) -> tuple[str | None, Literal["canonical", "alias", "fuzzy", "rejected"]]:
    """Return a canonical action and transparent normalisation evidence."""
    if not isinstance(value, str):
        return None, "rejected"
    candidate = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    candidate = re.sub(r"_+", "_", candidate)
    if candidate in CRAFTER_ACTIONS:
        return candidate, "canonical"
    if candidate in CRAFTER_ACTION_ALIASES:
        return CRAFTER_ACTION_ALIASES[candidate], "alias"
    match = difflib.get_close_matches(candidate, CRAFTER_ACTIONS, n=1, cutoff=0.5)
    return (match[0], "fuzzy") if match else (None, "rejected")


class BeliefPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicate: str
    arguments: list[str] = Field(default_factory=list)
    value: float | None = None
    rationale: str | None = None


class GoalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal_id: str
    predicate: str
    arguments: list[str] = Field(default_factory=list)
    status: Literal["pending", "active", "achieved", "failed"] = "pending"
    primary: bool = False
    order: int | None = None


class ObservationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    observation_type: str = "text"
    value: float | None = None


class LLReasonerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_state: str
    action: str | None = None
    beliefs: list[BeliefPayload] = Field(default_factory=list)
    goal_updates: list[GoalPayload] = Field(default_factory=list)
    learner_task: str | None = None
    request_goals: bool = False
    request_model: bool = False


class KnowledgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_state: str
    observations: list[ObservationPayload] = Field(default_factory=list)
    memory_beliefs: list[BeliefPayload] = Field(default_factory=list)
    high_level_beliefs: list[BeliefPayload] = Field(default_factory=list)


class HLReasonerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_state: str
    beliefs: list[BeliefPayload] = Field(default_factory=list)
    goals: list[GoalPayload] = Field(default_factory=list)
    learner_task: str | None = None
    request_beliefs: bool = False
    request_model: bool = False


class GoalGraphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_state: str
    low_level_goals: list[GoalPayload] = Field(default_factory=list)
    high_level_goals: list[GoalPayload] = Field(default_factory=list)


class MemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_state: str
    low_level_memories: list[BeliefPayload | ObservationPayload] = Field(default_factory=list)
    high_level_memories: list[BeliefPayload | ObservationPayload] = Field(default_factory=list)


class LearnerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_state: str
    model: str
    request_memories: bool = False


ROLE_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "ll_reasoner": LLReasonerResponse, "knowledge": KnowledgeResponse,
    "hl_reasoner": HLReasonerResponse, "goal_graph": GoalGraphResponse,
    "memory": MemoryResponse, "ll_learner": LearnerResponse,
    "hl_learner": LearnerResponse,
}


def jsonable(value: Any) -> Any:
    """Convert MHAgentA/Pydantic values to JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "__pydantic_serializer__"):
        return value.__pydantic_serializer__.to_python(value, mode="json")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def compact_event(path: Path, event: dict[str, Any]) -> None:
    """Append one compact event; prompts and full responses are not retained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")


class RoleRuntime:
    """Shared OpenAI I/O, composed by each concrete role class."""

    def __init__(self, module_id: str, **kwargs: Any) -> None:
        self.module_id = module_id
        self.role = str(kwargs["role"])
        self.environment = str(kwargs["environment"])
        self.value_system = str(kwargs.get("value_system", ""))
        self.models = tuple(kwargs["model_candidates"])
        self.reasoning_effort = str(kwargs.get("reasoning_effort", "low"))
        self.max_output_tokens = int(kwargs["max_output_tokens"])
        self.behavior_duration = float(kwargs["behavior_duration"])
        self.drain_deadline = float(kwargs["drain_deadline"])
        self.artifact_root = Path(str(kwargs.get("artifact_root", "/out")))
        encoded, _ = load_runtime_credentials(kwargs["credential_path"], budget_source="estimated")
        api_key = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        try:
            self.client = BudgetedOpenAIClient(
                api_key=api_key, max_budget_usd=str(kwargs["max_budget_usd"]),
                model_ids=self.models, budget_source="estimated",
                timeout=float(kwargs.get("request_timeout", 60.0)), max_retries=0)
        finally:
            api_key = ""
            encoded = ""
        self.model_index = 0
        self.events = self.artifact_root / "events" / f"{module_id}.jsonl"

    def close(self) -> None:
        self.client.close()

    def _image_content(self, value: Any) -> list[dict[str, Any]]:
        """Resolve content-addressed frame references under the run output root."""
        images: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if value.get("kind") == "image_reference":
                relative = Path(str(value.get("relative_path", "")))
                path = (self.artifact_root / relative).resolve()
                if self.artifact_root.resolve() not in path.parents:
                    raise ValueError("image reference escapes artifact root")
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != value.get("sha256"):
                    raise ValueError("image reference checksum mismatch")
                encoded = base64.b64encode(payload).decode("ascii")
                images.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
            else:
                for item in value.values():
                    images.extend(self._image_content(item))
        elif isinstance(value, list):
            for item in value:
                images.extend(self._image_content(item))
        return images

    def call(self, state: Any) -> BaseModel | None:
        """Make one structured call for queued native messages, with two repairs."""
        now = float(getattr(state, "time", 0.0))
        if now >= self.behavior_duration:
            state["termination_reason"] = "time_limit"
            state.outbox.terminate_agent("Experiment 2-7-CR time limit")
            return None
        if state["termination_reason"]:
            return None
        pending = list(state["pending"])
        state["phase"] = "behavior"
        if not pending or state["halted"] or state["budget_exhausted"]:
            return None
        snapshot = hashlib.sha256(json.dumps(pending, sort_keys=True, default=str).encode()).hexdigest()[:16]
        prompt = compose_prompt(role=self.role, environment=self.environment,
            text_state=state["text_state"], pending_messages=pending,
            value_system=self.value_system,
            call_requirements={
                "input_snapshot": snapshot, **state["context"],
                **{key: jsonable(state[key]) for key in (
                    "active_goals", "goal_graph_goals", "last_action_status", "current_model",
                    "survival_needs", "survival_override_active", "goal_ledger", "achievements") if hasattr(state, key) or isinstance(state, dict) and key in state},
                "goal_reply_recipients": sorted({
                    item["sender"] if item["type"] == "goal_request" else
                    ("llreasoner_0" if item["sender"] == "hlreasoner_0" else "hlreasoner_0")
                    for item in pending if item["type"] in {"goal_request", "goal_update"}}),
                "memory_reply_recipients": sorted({item["sender"] for item in pending
                    if item["type"] == "memory_request"})},
            repair_feedback=state["repair_feedback"])
        model = self.models[self.model_index]
        started = time.monotonic()
        raw_response = None
        record = {"role": self.role, "model": model, "snapshot": snapshot,
                  "prompt": prompt, "pending_messages": jsonable(pending),
                  "reasoning_effort": self.reasoning_effort,
                  "max_output_tokens": self.max_output_tokens, "store": False}
        try:
            images = self._image_content(pending)
            response = self.client.responses.parse(
                model=model, reasoning={"effort": self.reasoning_effort},
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt}, *images]}],
                text_format=ROLE_RESPONSE_MODELS[self.role],
                max_output_tokens=self.max_output_tokens, store=False)
            raw_response = response.model_dump(mode="json")
            record["outcome"] = "success"
            parsed = ROLE_RESPONSE_MODELS[self.role].model_validate(jsonable(response.output_parsed))
            state["calls"] += 1
            usage = self.client.usage_snapshot()
            state["estimated_cost_usd"] = usage["estimated_cost_usd"]
            compact_event(self.events, {"kind": "llm_call", "role": self.role,
                "model": model, "snapshot": snapshot,
                "latency_seconds": round(time.monotonic() - started, 3),
                "image_inputs": len(images),
                "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                "estimated_cost_usd": usage["estimated_cost_usd"], "outcome": "success"})
            if usage["budget_exhausted"]:
                state["budget_exhausted"] = True
                state["termination_reason"] = "budget_exhausted"
                state.outbox.terminate_agent(f"Experiment 2-7-CR budget exhausted: {self.module_id}")
                return None
            if float(getattr(state, "time", 0.0)) >= self.behavior_duration:
                state["termination_reason"] = "time_limit"
                state.outbox.terminate_agent("Experiment 2-7-CR time limit")
                return None
            return parsed
        except BudgetExceededError:
            record["outcome"] = "budget_exhausted"
            state["budget_exhausted"] = True
            state["termination_reason"] = "budget_exhausted"
            state.outbox.terminate_agent(f"Experiment 2-7-CR budget exhausted: {self.module_id}")
            compact_event(self.events, {"kind": "llm_call", "role": self.role,
                                        "model": model, "outcome": "budget_exhausted"})
        except Exception as error:
            if raw_response is None and isinstance(error, ValidationError):
                raw_response = {"validation_errors": error.errors(include_url=False, include_context=False)}
            record.update(outcome="failure", error=type(error).__name__)
            state["failures"] += 1
            state["semantic_attempts"] += 1
            state["repair_feedback"] = {"error": type(error).__name__, "retry": state["semantic_attempts"]}
            if state["semantic_attempts"] > MAX_SEMANTIC_REPAIRS:
                if self.model_index + 1 < len(self.models):
                    self.model_index += 1
                    state["semantic_attempts"] = 0
                else:
                    state["halted"] = True
            compact_event(self.events, {"kind": "llm_call", "role": self.role,
                "model": model, "outcome": "failure", "error": type(error).__name__})
        finally:
            record["raw_response"] = raw_response
            record["latency_seconds"] = round(time.monotonic() - started, 3)
            record["usage"] = self.client.usage_snapshot()
            state["estimated_cost_usd"] = record["usage"]["estimated_cost_usd"]
            compact_event(self.artifact_root / "llm_responses" / f"{self.module_id}.jsonl", record)
            state["response_records"] += 1
        return None


def initial_role_state(text_state: str, *, profile: ModelProfile, budget: str) -> dict[str, Any]:
    """Return compact, fully predeclared state shared by LLM roles."""
    return {"text_state": text_state, "pending": [], "calls": 0, "failures": 0,
            "semantic_attempts": 0, "repair_feedback": None, "halted": False,
            "budget_exhausted": False, "estimated_cost_usd": "0", "phase": "behavior",
            "termination_reason": None, "response_records": 0,
            "model_profile": profile.as_dict(), "max_budget_usd": budget,
            "context": {}, "last_sent_payloads": {}, "coalesced_observations": 0, "suppressed_messages": 0, "goal_ledger": {},
            "received": 0, "sent": 0}


__all__ = [name for name in globals() if not name.startswith("_")]
