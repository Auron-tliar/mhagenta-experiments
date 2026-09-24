"""Deterministic Blocks World boundary for experiment 2-7-BW."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mha_exp_common.utils import Seeder
from mhagenta import ActionStatus, Observation, Orchestrator
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, PerceptorState

from .evidence import correlation_metadata, receive_message, send_message
from .terminal import finalize_module, terminal_fields
from .llm import (
    STATE_SCHEMA_VERSION,
    canonical_json,
    canonical_sha256,
    normalized_failure,
)


TABLE_LEN = 5
NUM_BLOCKS = 8
N_GOALS = 1
ACTIONS = {
    "Pick-Up": 0,
    "Put-Down": 1,
    "Move-Left": 2,
    "Move-Right": 3,
}
PREDICATE_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)$")


def _parse_predicate(predicate: str) -> tuple[str, list[str]]:
    match = PREDICATE_RE.fullmatch(predicate.strip())
    if match is None:
        return "", []
    return match.group("name").lower(), [
        item.strip() for item in match.group("args").split(",") if item.strip()
    ]


def symbolic_snapshot(
    predicates: Sequence[str], *, table_len: int = TABLE_LEN
) -> dict[str, Any]:
    """Convert symbolic predicates to the exact state used by trace/evaluation."""

    arm_location = "T0"
    held = "empty"
    above: dict[str, str] = {}
    for predicate in predicates:
        name, args = _parse_predicate(predicate)
        if name == "above" and len(args) == 1:
            arm_location = args[0].upper()
        elif name == "holding" and len(args) == 1:
            held = args[0].upper()
        elif name == "on" and len(args) == 2:
            above[args[1].lower()] = args[0].upper()
    stacks: list[list[str]] = []
    for index in range(table_len):
        support = f"t{index}"
        stack: list[str] = []
        seen: set[str] = set()
        while support.lower() in above:
            block = above[support.lower()]
            if block in seen:
                raise ValueError(f"cyclic stack detected at T{index}")
            seen.add(block)
            stack.append(block)
            support = block
        stacks.append(stack)
    return {"arm_location": arm_location, "held_block": held, "stacks": stacks}


def symbolic_observation_to_text(
    predicates: Sequence[str], *, table_len: int = TABLE_LEN
) -> str:
    """Convert symbolic state to the experiment's authoritative text format."""

    snapshot = symbolic_snapshot(predicates, table_len=table_len)
    lines = [
        f"[{snapshot['arm_location']}; holding={snapshot['held_block']}]"
    ]
    lines.extend(
        f"T{index}:{','.join(stack)}"
        for index, stack in enumerate(snapshot["stacks"])
    )
    return "\n".join(lines)


def normalize_action(action: Any) -> tuple[str | None, int | None]:
    """Accept only the four exact model-visible action literals."""

    if not isinstance(action, str) or action not in ACTIONS:
        return None, None
    return action, ACTIONS[action]


def _on_pairs(predicates: Sequence[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for predicate in predicates:
        name, args = _parse_predicate(predicate)
        if name == "on" and len(args) == 2:
            pairs.add((args[0].upper(), args[1].upper()))
    return pairs


def generate_primary_goals(
    *,
    run: int,
    environment_seed: int,
    n_goals: int = N_GOALS,
    table_len: int = TABLE_LEN,
    num_blocks: int = NUM_BLOCKS,
) -> list[dict[str, Any]]:
    """Generate matched, initially unsatisfied ordered goals."""

    if not 1 <= n_goals <= num_blocks * (num_blocks - 1):
        raise ValueError("invalid number of primary goals")
    from mha_env_blocksworld import BlocksWorldEnv

    environment = BlocksWorldEnv(
        table_len=table_len, num_blocks=num_blocks, symbolic=True
    )
    predicates, _ = environment.reset(seed=environment_seed)
    already_true = _on_pairs(cast(Sequence[str], predicates))
    candidates = [
        (f"B{top}", f"B{bottom}")
        for top in range(num_blocks)
        for bottom in range(num_blocks)
        if top != bottom and (f"B{top}", f"B{bottom}") not in already_true
    ]
    random.Random(Seeder(run).hl_reasoner).shuffle(candidates)
    return [
        {
            "goal_id": f"primary_{index}",
            "predicate": "On",
            "arguments": [top, bottom],
            "status": "pending",
            "primary": True,
            "order": index,
        }
        for index, (top, bottom) in enumerate(candidates[:n_goals])
    ]


def adapter_state(*, lifecycle: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Return the complete JSON-native adapter state schema."""

    return {
        **terminal_fields(),
        "state_schema_version": STATE_SCHEMA_VERSION,
        "adapter_kind": kind,
        "pending": [],
        "sent_boundaries": [],
        "processed_boundaries": [],
        "boundary_sequence": 0,
        "received_sequence": 0,
        "failure": None,
        "incomplete_reason": None,
        "lifecycle": dict(lifecycle),
        "admission_open": True,
        "suppressed_dispatches": [],
        "requests": 0,
        "responses": 0,
        "invalid_requests": 0,
        "last_observation": None,
    }


class LLMBlocksWorldEnvironment(MHAEnvBase):
    """Blocks World with one authoritative append-only transition trace."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        super().__init__(init_state)
        self._seed = int(init_state["seed"])
        self._table_len = int(init_state["table_len"])
        self._num_blocks = int(init_state["num_blocks"])
        self._env: Any = None
        self._trace_ready = False
        self._trace_root_override: Path | None = None
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_blocksworld import BlocksWorldEnv

        self._env = BlocksWorldEnv(
            table_len=self._table_len,
            num_blocks=self._num_blocks,
            symbolic=True,
        )
        # The native environment omits legality evidence unless explicitly enabled.
        self._env.expose_snapshot = True
        predicates, _ = self._env.reset(seed=self._seed)
        self.state["state"] = list(predicates)
        self._trace_ready = False

    def __getstate__(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["_env"] = None
        value["_trace_ready"] = False
        value["_trace_root_override"] = None
        return value

    def __setstate__(self, value: dict[str, Any]) -> None:
        self.__dict__.update(value)
        self._build_env()

    @property
    def _trace_path(self) -> Path:
        root = self._trace_root_override or Path(f"/{Orchestrator.SAVE_SUBDIR}")
        return root / self.state["trace_file"]

    def set_trace_root_for_testing(self, root: Path) -> None:
        """Redirect only the transient trace writer in model-free tests."""

        self._trace_root_override = root

    def _append_trace(self, row: dict[str, Any]) -> None:
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(row) + "\n")
        self.state["trace_rows"] += 1

    def _ensure_trace(self) -> None:
        if self._trace_ready:
            return
        self._trace_path.unlink(missing_ok=True)
        snapshot = symbolic_snapshot(self.state["state"], table_len=self._table_len)
        self._append_trace(
            {
                "schema_version": "2-7-bw-environment-transitions-v2",
                "row_type": "initial",
                "action_index": -1,
                "module_time": 0.0,
                "action_id": None,
                "cycle_id": None,
                "boundary_id": None,
                "action": None,
                "accepted": True,
                "legal": True,
                "executed": False,
                "reason": None,
                **snapshot,
                "newly_achieved_goal_ids": [],
                "terminal": False,
                "truncated": False,
            }
        )
        self.state["initial_state_sha256"] = canonical_sha256(snapshot)
        self._trace_ready = True

    def _record_processed(
        self, *, sender: str, kind: str, metadata: Mapping[str, Any]
    ) -> None:
        boundary_id = metadata.get("boundary_id")
        if isinstance(boundary_id, str):
            self.state["processed_boundaries"].append(
                {
                    "boundary_id": boundary_id,
                    "source": sender,
                    "recipient": self.state["environment_id"],
                    "kind": kind,
                    **correlation_metadata(metadata),
                }
            )

    def on_observe(
        self, state: dict[str, Any], sender_id: str, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_trace()
        state["observation_requests"] += 1
        self._record_processed(sender=sender_id, kind="environment_observe", metadata=kwargs)
        return state, {
            "observation": list(state["state"]),
            "boundary_id": kwargs.get("boundary_id"),
            **correlation_metadata(kwargs),
        }

    def on_action(
        self, state: dict[str, Any], sender_id: str, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_trace()
        self._record_processed(sender=sender_id, kind="environment_action", metadata=kwargs)
        requested = kwargs.get("action")
        action_code = kwargs.get("action_code")
        if state["scientific_complete"]:
            return state, {"success": False, "scientific_complete": True,
                           "reason": "primary_goal_already_achieved",
                           "boundary_id": kwargs.get("boundary_id"),
                           **correlation_metadata(kwargs)}
        accepted = (
            isinstance(requested, str)
            and requested in ACTIONS
            and ACTIONS[requested] == action_code
        )
        reward = 0.0
        legal = False
        executed = False
        reason: str | None = None
        if accepted:
            predicates, reward, terminated, truncated, info = self._env.step(action_code)
            state["state"] = list(predicates)
            executed = True
            legal = bool(info["snapshot"].legal)
        else:
            terminated = False
            truncated = False
            reason = "invalid_action_literal"
            state["invalid_environment_requests"] += 1
        if not legal:
            state["illegal_actions"] += 1
        current_pairs = _on_pairs(state["state"])
        newly_achieved: list[str] = []
        for goal in state["primary_goals"]:
            goal_id = str(goal["goal_id"])
            pair = tuple(str(item).upper() for item in goal["arguments"])
            if pair in current_pairs and goal_id not in state["achieved_goal_ids"]:
                state["achieved_goal_ids"].append(goal_id)
                newly_achieved.append(goal_id)
        action_index = int(state["attempted_actions"])
        state["attempted_actions"] = action_index + 1
        all_primary_goals_achieved = len(state["achieved_goal_ids"]) == len(
            state["primary_goals"]
        )
        if (
            all_primary_goals_achieved
            and state["final_goal_completion_action"] is None
        ):
            state["final_goal_completion_action"] = action_index
        if (
            state["final_goal_completion_action"] is not None
            and action_index > state["final_goal_completion_action"]
            and requested == "Put-Down"
            and accepted
            and legal
        ):
            state["eligible_post_goal_discretionary_states"] += 1
        scientific_complete = all_primary_goals_achieved
        state["scientific_complete"] = scientific_complete
        snapshot = symbolic_snapshot(state["state"], table_len=self._table_len)
        self._append_trace(
            {
                "schema_version": "2-7-bw-environment-transitions-v2",
                "row_type": "action",
                "action_index": action_index,
                "module_time": kwargs.get("module_time"),
                "action_id": kwargs.get("action_id"),
                "cycle_id": kwargs.get("cycle_id"),
                "boundary_id": kwargs.get("boundary_id"),
                "goal_id": kwargs.get("goal_id"),
                "action": requested,
                "accepted": accepted,
                "legal": legal,
                "executed": executed,
                "reward": float(reward),
                "reason": reason,
                **snapshot,
                "newly_achieved_goal_ids": newly_achieved,
                "terminal": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        return state, {
            "success": legal,
            "legal": legal,
            "executed": executed,
            "reward": float(reward),
            "reason": reason,
            "all_primary_goals_achieved": all_primary_goals_achieved,
            "eligible_post_goal_discretionary_states": state[
                "eligible_post_goal_discretionary_states"
            ],
            "scientific_complete": scientific_complete,
            "boundary_id": kwargs.get("boundary_id"),
            **correlation_metadata(kwargs),
        }


class TextBlocksWorldPerceptor(RMQPerceptorBase):
    """Convert symbolic environment observations to authoritative text."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._environment_id = ""
        self._reasoner_id = ""

    def on_first(self, state: PerceptorState) -> PerceptorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("Blocks World environment is missing")
        self._environment_id = environment.address["env_id"]
        reasoners = state.directory.internal.ll_reasoning
        if len(reasoners) != 1:
            raise RuntimeError("2-7-BW requires exactly one low-level reasoner")
        self._reasoner_id = reasoners[0].module_id
        return state

    def on_last(self, state: PerceptorState) -> PerceptorState:
        """Save unfinished observation work as terminal cancellation evidence."""
        return finalize_module(state)

    def on_request(
        self, state: PerceptorState, sender: str, **kwargs: Any
    ) -> PerceptorState:
        message = receive_message(
            state, sender=sender, kind="request_observation", payload={}, metadata=kwargs
        )
        state["requests"] += 1
        if float(state.time) >= float(state["lifecycle"]["behavior_cutoff"]):
            state["admission_open"] = False
            state["suppressed_dispatches"].append("environment_observe")
            state["pending"].remove(message)
            return state
        boundary_id = send_message(
            state,
            recipient=self._environment_id,
            kind="environment_observe",
            metadata=message["metadata"],
            dispatch=lambda **meta: self.observe(self._environment_id, **meta),
        )
        message["environment_boundary_id"] = boundary_id
        return state

    def on_observation(
        self, state: PerceptorState, env_id: str, **kwargs: Any
    ) -> PerceptorState:
        predicates = kwargs.get("observation")
        if not isinstance(predicates, list):
            state["invalid_requests"] += 1
            return state
        text = symbolic_observation_to_text(predicates)
        observation = Observation(text, observation_type="blocks-world-text")
        observation_data = {
            "content": text,
            "observation_type": "blocks-world-text",
            "value": None,
        }
        payload_hash = canonical_sha256(observation_data)
        evidence_id = f"observation:{payload_hash[:24]}"
        pending = state["pending"].pop(0) if state["pending"] else None
        metadata = dict(pending["metadata"] if pending else {})
        metadata.update(
            evidence_id=evidence_id,
            observation_payload_sha256=payload_hash,
        )
        send_message(
            state,
            recipient=self._reasoner_id,
            kind="send_observation",
            metadata=metadata,
            dispatch=lambda **meta: state.outbox.send_observation(
                self._reasoner_id, observation, **meta
            ),
        )
        state["responses"] += 1
        state["last_observation"] = text
        return state


class StringBlocksWorldActuator(RMQActuatorBase):
    """Map only exact action literals and return typed status."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._environment_id = ""
        self._reasoner_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        environment = state.directory.external.environment
        if environment is None:
            raise RuntimeError("Blocks World environment is missing")
        self._environment_id = environment.address["env_id"]
        reasoners = state.directory.internal.ll_reasoning
        if len(reasoners) != 1:
            raise RuntimeError("2-7-BW requires exactly one low-level reasoner")
        self._reasoner_id = reasoners[0].module_id
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        """Preserve pending action requests for the terminal integrity audit."""
        return finalize_module(state)

    def on_request(
        self, state: ActuatorState, sender: str, **kwargs: Any
    ) -> ActuatorState:
        requested, action_code = normalize_action(kwargs.get("action"))
        message = receive_message(
            state,
            sender=sender,
            kind="request_action",
            payload={"action": kwargs.get("action")},
            metadata=kwargs,
        )
        if requested is None:
            state["invalid_requests"] += 1
        boundary_id = send_message(
            state,
            recipient=self._environment_id,
            kind="environment_action",
            metadata=message["metadata"],
            dispatch=lambda **meta: self.act(
                self._environment_id,
                action=requested,
                action_code=action_code,
                module_time=float(state.time),
                **meta,
            ),
        )
        message["environment_boundary_id"] = boundary_id
        state["requests"] += 1
        return state

    def on_status(
        self, state: ActuatorState, env_id: str, **kwargs: Any
    ) -> ActuatorState:
        boundary_id = kwargs.get("boundary_id")
        index = next(
            (
                i
                for i, item in enumerate(state["pending"])
                if item.get("environment_boundary_id") == boundary_id
            ),
            None,
        )
        if index is None:
            state["failure"] = normalized_failure(
                kind="UnmatchedEnvironmentStatus",
                stage="actuator_status",
                module_time=float(state.time),
            )
            state["incomplete_reason"] = "unmatched_environment_status"
            return state
        request = state["pending"].pop(index)
        metadata = dict(request["metadata"] if request else {})
        status = {
            "success": bool(kwargs.get("success")),
            "legal": bool(kwargs.get("legal")),
            "executed": bool(kwargs.get("executed")),
            "reward": kwargs.get("reward"),
            "reason": kwargs.get("reason"),
            "action": request["payload"].get("action") if request else None,
            "all_primary_goals_achieved": bool(
                kwargs.get("all_primary_goals_achieved")
            ),
            "eligible_post_goal_discretionary_states": int(
                kwargs.get("eligible_post_goal_discretionary_states", 0)
            ),
            "scientific_complete": bool(kwargs.get("scientific_complete")),
        }
        send_message(
            state,
            recipient=self._reasoner_id,
            kind="send_status",
            metadata=metadata,
            dispatch=lambda **meta: state.outbox.send_status(
                self._reasoner_id, ActionStatus(status), **meta
            ),
        )
        state["responses"] += 1
        return state


def environment_initial_state(
    *,
    seed: int,
    value_condition: str,
    primary_goals: list[dict[str, Any]],
    environment_id: str,
) -> dict[str, Any]:
    """Return the compact JSON state owned by the external environment."""

    return {
        "schema_version": "2-7-bw-environment-state-v2",
        "seed": seed,
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        "environment_id": environment_id,
        "value_condition": value_condition,
        "primary_goals": primary_goals,
        "state": [],
        "achieved_goal_ids": [],
        "final_goal_completion_action": None,
        "eligible_post_goal_discretionary_states": 0,
        "scientific_complete": False,
        "observation_requests": 0,
        "attempted_actions": 0,
        "illegal_actions": 0,
        "invalid_environment_requests": 0,
        "trace_file": "environment-transitions.jsonl",
        "trace_rows": 0,
        "initial_state_sha256": None,
        "processed_boundaries": [],
    }


__all__ = [
    "ACTIONS",
    "LLMBlocksWorldEnvironment",
    "N_GOALS",
    "NUM_BLOCKS",
    "StringBlocksWorldActuator",
    "TABLE_LEN",
    "TextBlocksWorldPerceptor",
    "adapter_state",
    "environment_initial_state",
    "generate_primary_goals",
    "normalize_action",
    "symbolic_observation_to_text",
    "symbolic_snapshot",
]
