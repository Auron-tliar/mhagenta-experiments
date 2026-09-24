"""Shared stop evidence and explicit cancellation at the BW episode boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def terminal_fields() -> dict[str, Any]:
    """Declare persisted shutdown evidence; tests may leave the shared path unset."""
    return {"terminal_file": None, "terminal_observed": None, "shutdown_record": None,
            "cancelled_inputs": [], "cancelled_waits": {}, "suppressed_terminal_dispatches": []}


def observe_terminal(state: Any) -> dict[str, Any] | None:
    """Close admission as soon as any module publishes an episode stop."""
    record = state["terminal_observed"]
    path = state["terminal_file"]
    if record is None and path and Path(path).is_file():
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        state["terminal_observed"] = record
    if record is not None:
        state["admission_open"] = False
    return record


def publish_terminal(state: Any, reason: str) -> dict[str, Any]:
    """Atomically retain the first stop, shared by the agent's module processes."""
    record = observe_terminal(state)
    if record is not None:
        return record
    record = {"reason": reason, "module_id": state.module_id, "module_time": float(state.time)}
    if state["terminal_file"]:
        path = Path(state["terminal_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex)
        try:
            temporary.write_text(json.dumps(record), encoding="utf-8")
            try:
                os.link(temporary, path)
            except FileExistsError:
                record = json.loads(path.read_text(encoding="utf-8"))
        finally:
            temporary.unlink(missing_ok=True)
    state["terminal_observed"] = record
    state["admission_open"] = False
    return record


def finalize_module(state: Any) -> Any:
    """Preserve cancelled work separately from delivered or model-processed work."""
    record = observe_terminal(state)
    state["admission_open"] = False
    state["shutdown_record"] = {"module_time": float(state.time), "terminal": record}
    if record is not None:
        state["cancelled_inputs"].extend(state["pending"])
        state["pending"] = []
        if hasattr(state, "waiting_requests"):
            state["cancelled_waits"] = dict(state["waiting_requests"])
            state["waiting_requests"] = {}
    return state


def terminal_accounting(agent_states: dict[str, Any], environment: dict[str, Any],
                        rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Allow only recent, non-action message cancellation at a verified stop."""
    from collections import Counter

    sent = [item for state in agent_states.values() for item in state.get("sent_boundaries", [])]
    received = [item for state in agent_states.values() for item in state.get("processed_boundaries", [])]
    received += list(environment.get("processed_boundaries", []))
    sent_ids = Counter(item.get("boundary_id") for item in sent)
    received_ids = Counter(item.get("boundary_id") for item in received)
    identities_valid = (all(isinstance(key, str) and count == 1 for key, count in sent_ids.items())
                        and all(count == 1 and key in sent_ids for key, count in received_ids.items()))
    missing = [item for item in sent if item.get("boundary_id") not in received_ids]
    records = [state.get("terminal_observed") for state in agent_states.values()]
    terminal = next((item for item in records if item), None)
    verified = False
    if terminal is not None:
        source = agent_states.get(terminal.get("module_id"), {})
        stop_time = float(terminal["module_time"])
        reason = terminal.get("reason")
        if reason == "task_completed":
            evidence = source.get("scientific_completion_evidence") or {}
            verified = bool(source.get("scientific_complete") and environment.get("scientific_complete")
                            and source.get("termination_reason") == reason
                            and any(row.get("action_id") == evidence.get("action_id")
                                    and row.get("newly_achieved_goal_ids")
                                    and row.get("legal") is True and row.get("executed") is True
                                    and float(row["module_time"]) <= stop_time for row in rows))
        elif reason == "time_limit":
            verified = bool(source.get("termination_reason") == reason
                            and stop_time >= float(source.get("lifecycle", {}).get("behavior_cutoff", float("inf"))))
        elif reason == "budget_exhausted":
            verified = bool(source.get("termination_reason") == reason
                            and source.get("usage", {}).get("budget_exhausted"))
        verified = verified and all(
            item == terminal and state.get("admission_open") is False
            and (state.get("shutdown_record") or {}).get("terminal") == terminal
            and float((state.get("shutdown_record") or {}).get("module_time", -1)) >= stop_time
            and state.get("in_flight") is None and not state.get("pending")
            and state.get("failure") is None and state.get("incomplete_reason") is None
            for state, item in zip(agent_states.values(), records, strict=True))
    cancelled, unresolved = [], []
    for item in missing:
        source = agent_states.get(item.get("source"), {})
        recipient = agent_states.get(item.get("recipient"), {})
        sent_at = float(item.get("module_time", -1))
        timeout = float(source.get("lifecycle", {}).get("request_timeout", 30.0))
        # Do not conceal an old lost message, a post-stop send, or an unconfirmed action.
        allowed = (verified and item.get("kind") not in {"request_action", "environment_action"}
                   and item.get("recipient") in agent_states
                   and stop_time - timeout <= sent_at <= stop_time
                   and float((recipient.get("shutdown_record") or {}).get("module_time", -1)) >= stop_time)
        (cancelled if allowed else unresolved).append(item)
    return {"terminal": terminal, "terminal_verified": bool(verified),
            "identities_valid": identities_valid, "delivered_count": len(received),
            "cancelled_boundaries": cancelled, "unresolved_boundaries": unresolved,
            "all_boundaries_accounted": bool(identities_valid and not unresolved)}
