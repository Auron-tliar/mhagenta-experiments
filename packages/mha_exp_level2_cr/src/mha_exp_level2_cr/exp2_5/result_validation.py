"""Recognize a narrowly evidenced normal time cutoff in retained result files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any


def time_limit_evidence(
    agent: Mapping[str, Any], environment: Mapping[str, Any],
    run_root: Path | None, runtime_ids: tuple[str, str] | None,
) -> dict[str, Any] | None:
    """Recognize the final dispatched-but-unreceived activity at the 600s cap.

This is a reporting rule, not a recovery or state mutation. Native transitions,
artifacts and all other result invariants still require the ordinary checker.
Other in-flight goal/action configurations are intentionally not accepted.
"""
    if run_root is None or runtime_ids is None:
        return None
    try:
        ll, hl, goals = (agent[name] for name in ("llreasoner_0", "hlreasoner_0", "goalgraph_0"))
        count = environment["native_action_count"]
        if (type(count) is not int or not 0 < count < 900 or environment["closed"] is not True
                or environment["terminal"] is not False or environment["inventory"]["health"] <= 0
                or environment["inventory"].get("diamond", 0) != 0
                or environment["achievement_counts"].get("collect_diamond", 0) != 0
                or environment["failure"] is not None or any(state["failure"] is not None for state in agent.values())
                or hl["terminal_reason"] is not None or hl["survived"] is not False
                or hl["pending_terminal"] is not None or ll["current_goal"] is not None
                or ll["pending_action"] is not None or ll["passive_status"] is not None
                or ll["awaiting_observation"] is not False):
            return None
        completed = ll["terminal_goal_count"]
        counts = (completed, ll["received_goal_count"], hl["terminal_count"], goals["terminal_count"],
                  hl["dispatch_count"], goals["dispatch_count"])
        if (any(type(value) is not int for value in counts) or completed < 1
                or counts[:4] != (completed,) * 4 or counts[4:] != (completed + 1,) * 2):
            return None
        extras = hl["active_goal"]["extras"]
        pending_id = f"hl-{completed + 1}"
        if (extras["kind"] != "activity" or extras["status"] != "requested"
                or extras["goal_id"] != pending_id or goals["active_goal_id"] != pending_id
                or extras["baseline_revision"] != ll["observation_count"]
                or ll["activities"][-1]["goal_id"] != f"hl-{completed}"):
            return None
        log = run_root / f"{runtime_ids[0]}.log"
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 262144))
            tail = stream.read().decode("utf-8", errors="replace")
        pattern = (r"^\[[^\]\r\n]+\|(?P<seconds>\d+(?:\.\d+)?)\]\[INFO\]::\["
                   + re.escape(runtime_ids[0])
                   + r"\]\[root\]::Sending stop command \(reason AGENT TIMEOUT CMD\)\r?$")
        matches = list(re.finditer(pattern, tail, re.MULTILINE))
        if len(matches) != 1:
            return None
        elapsed = float(matches[0]["seconds"])
        if not 600.0 <= elapsed <= 605.0:
            return None
        return {"reason": "time_limit", "execution_seconds": elapsed,
                "boundary": "activity_dispatched_to_goal_graph_not_received_by_ll",
                "pending_goal_id": pending_id, "completed_goals": completed}
    except (KeyError, TypeError, ValueError, IndexError, OSError):
        return None
