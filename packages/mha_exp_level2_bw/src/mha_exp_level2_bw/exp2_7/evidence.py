"""Flat message identity helpers; deliberately not a causal work scheduler."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .llm import MESSAGE_RESPONSE_TIMEOUT, json_safe
from .terminal import observe_terminal


REQUEST_REPLIES = {
    "request_observation": "observation", "request_action": "action_status",
    "request_goals": "goal_update", "request_beliefs": "belief_update",
    "request_model": "learner_model", "send_learner_task": "learner_model",
    "request_memories": "memories",
}


CORRELATION_KEYS = (
    "evidence_id",
    "source_evidence_id",
    "observation_payload_sha256",
    "goal_id",
    "goal_payload_sha256",
    "action_id",
    "cycle_id",
    "revision_id",
    "input_evaluation_id",
    "active_goal_id",
    "source_ids",
)


def correlation_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the small JSON-safe scientific identity envelope."""

    return {
        key: json_safe(value[key])
        for key in CORRELATION_KEYS
        if key in value and value[key] is not None
    }


def receive_message(
    state: Any,
    *,
    sender: str,
    kind: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one normalized callback input and mark its boundary processed."""

    compact = correlation_metadata(metadata)
    boundary_id = metadata.get("boundary_id")
    if isinstance(boundary_id, str):
        message_id = boundary_id
    else:
        sequence = int(state["received_sequence"])
        state["received_sequence"] = sequence + 1
        message_id = f"{state.module_id}:received:{sequence}"
    message = {
        "message_id": message_id,
        "sender": sender,
        "kind": kind,
        "payload": json_safe(dict(payload)),
        "metadata": compact,
        "received_at": float(state.time),
    }
    if kind == "observed_beliefs":
        older = [item for item in state["pending"] if item["kind"] == kind]
        state["coalesced_observations"] += len(older)
        state["pending"][:] = [item for item in state["pending"] if item["kind"] != kind]
    state["pending"].append(message)
    if hasattr(state, "waiting_requests"):
        for key, waiting in list(state["waiting_requests"].items()):
            if waiting["recipient"] != sender or REQUEST_REPLIES[waiting["kind"]] != kind:
                continue
            correlation = "action_id" if kind == "action_status" else "cycle_id" if kind == "observation" else None
            if correlation and waiting["metadata"].get(correlation) != compact.get(correlation):
                continue
            del state["waiting_requests"][key]
        reminders = state["context"].get("unanswered_requests", [])
        reminders = [item for item in reminders if item["request"] in state["waiting_requests"]]
        if reminders:
            state["context"]["unanswered_requests"] = reminders
        else:
            state["context"].pop("unanswered_requests", None)
    if kind in {"goal_update", "learner_model", "belief_update", "action_status"}:
        state["context"][kind + ":" + sender] = json_safe(dict(payload))
    if isinstance(boundary_id, str):
        state["processed_boundaries"].append(
            {
                "boundary_id": boundary_id,
                "source": sender,
                "recipient": state.module_id,
                "kind": kind,
                "module_time": float(state.time),
                **compact,
            }
        )
    return message


def send_message(
    state: Any,
    *,
    recipient: str,
    kind: str,
    dispatch: Callable[..., None],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Queue one typed outbox call and record its unique flat boundary."""

    if observe_terminal(state) is not None:
        state["suppressed_terminal_dispatches"].append({
            "recipient": recipient, "kind": kind, "module_time": float(state.time),
            "metadata": correlation_metadata(metadata or {}),
        })
        return ""
    sequence = int(state["boundary_sequence"])
    state["boundary_sequence"] = sequence + 1
    boundary_id = f"{state.module_id}:boundary:{sequence}"
    compact = correlation_metadata(metadata or {})
    dispatch(boundary_id=boundary_id, **compact)
    if kind in REQUEST_REPLIES and hasattr(state, "waiting_requests"):
        key = recipient + ":" + kind
        if kind == "request_action":
            key += ":" + str(compact.get("action_id"))
        previous = state["waiting_requests"].get(key, {})
        state["waiting_requests"][key] = {
            "kind": kind, "recipient": recipient, "metadata": compact,
            "first_sent_at": previous.get("first_sent_at", float(state.time)),
            "last_sent_at": float(state.time), "reminders": previous.get("reminders", 0),
            "next_reminder_at": previous.get("next_reminder_at", float(state.time) + MESSAGE_RESPONSE_TIMEOUT),
        }
    state["sent_boundaries"].append(
        {
            "boundary_id": boundary_id,
            "source": state.module_id,
            "recipient": recipient,
            "kind": kind,
            "module_time": float(state.time),
            **compact,
        }
    )
    return boundary_id


def snapshot_metadata(snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect identities present on the consumed pending snapshot."""

    result: dict[str, Any] = {}
    source_ids: list[str] = []
    for message in snapshot:
        metadata = message.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        for key in CORRELATION_KEYS:
            value = metadata.get(key)
            if value is None:
                continue
            if key == "source_ids":
                for item in value:
                    if item not in source_ids:
                        source_ids.append(item)
            elif key not in result:
                result[key] = value
    if source_ids:
        result["source_ids"] = source_ids
    return result


__all__ = [
    "CORRELATION_KEYS",
    "correlation_metadata",
    "receive_message",
    "send_message",
    "snapshot_metadata",
]
