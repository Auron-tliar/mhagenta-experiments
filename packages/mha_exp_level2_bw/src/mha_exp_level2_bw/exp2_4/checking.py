"""Behavioral acceptance checks for experiment 2-4-BW."""

from __future__ import annotations

import json
from hashlib import sha256
from collections.abc import Sequence
from typing import Any

from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import module_name

from .agent import A_MOVE_LEFT, A_MOVE_RIGHT, A_PICK_UP, A_PUT_DOWN, K_ACTION, K_STATE
from .planning import GoalSpec, TransferSpec, beliefs_to_facts, parse_symbolic_observation


def _atomic_rows_valid(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    actions = [row.get(K_ACTION) for row in rows]
    if actions.count(A_PICK_UP) != 1 or actions.count(A_PUT_DOWN) != 1:
        return False
    pickup, putdown = actions.index(A_PICK_UP), actions.index(A_PUT_DOWN)
    return (
        pickup < putdown == len(actions) - 1
        and all(action in {A_MOVE_LEFT, A_MOVE_RIGHT} for action in actions[:pickup])
        and all(action in {A_MOVE_LEFT, A_MOVE_RIGHT} for action in actions[pickup + 1:putdown])
        and all(row.get("legal") is True and isinstance(row.get("observation_seq"), int) for row in rows)
    )


def _hierarchy_valid(hierarchy: Any) -> bool:
    try:
        plan, execution = hierarchy["plan"], hierarchy["execution"]
        transfers = plan["transfers"]
        intention = GoalSpec(**hierarchy["intention"])
        if (
            hierarchy["status"] != "completed" or plan["validation"] != "VALID"
            or not transfers or len(transfers) > plan["bound"]
            or len(execution) != len(transfers)
            or not isinstance(hierarchy["final_goal_observation_seq"], int)
        ):
            return False
        goal_ids = []
        for index, (mapping, row) in enumerate(zip(transfers, execution, strict=True)):
            spec = TransferSpec.from_mapping(mapping)
            goal_ids.append(row["goal_id"])
            if (
                row["step_index"] != index or row["status"] != "completed"
                or sorted(row["observed_target_facts"]) != sorted(spec.target_facts)
                or int(row["completion_observation_seq"]) <= int(row["dispatch_observation_seq"])
                or int(row["completion_belief_observation_seq"]) < int(row["completion_observation_seq"])
                or not _atomic_rows_valid(row["atomic_rows"])
            ):
                return False
        return intention.fact in {
            f"on({item['block']},{item['destination_support']})" for item in transfers
        } and len(goal_ids) == len(set(goal_ids))
    except (KeyError, TypeError, ValueError):
        return False


def _single_flight(chain: Sequence[int]) -> bool:
    return all(left >= right and left - right <= 1 for left, right in zip(chain, chain[1:]))


def check_results(states: dict[str, dict[str, Any]], logs: Sequence[str], verbose: bool = False,
                  *, environment: dict[str, Any] | None = None,
                  environment_logs: Sequence[str] | None = None,
                  expected_treatment: dict[str, Any] | None = None) -> bool:
    """Certify operational integrity separately from achieving the stacking goal."""

    success = not any(
        "][error]::" in line.lower() or "exception" in line.lower() for line in logs
    )
    names = [PERCEPTOR, ACTUATOR, LLREASONER, KNOWLEDGE, GOALGRAPH, HLREASONER]
    required = {module_name(name, 0) for name in names}
    if required - states.keys():
        return False
    p, a, ll, knowledge, graph, hl = (states[module_name(name, 0)] for name in names)
    modules = (p, a, ll, knowledge, graph, hl)
    try:
        json.dumps(modules)
    except (TypeError, ValueError):
        return False
    current, retained = hl.get("current_hierarchy"), hl.get("retained_hierarchy")
    completed = current if isinstance(current, dict) and current.get("status") == "completed" else retained
    chains = (
        [hl["transfer_dispatches"], graph["dispatched"], ll["transfer_activations"],
         ll["terminal_updates"], graph["terminal"], hl["terminal_results"], hl["reconciled_outcomes"]],
        [ll["action_requests"], a["requests"], a["statuses"], ll["action_statuses"]],
        [ll["observation_requests"], p["requests"], p["observations"], ll["observations"]],
    )
    active = isinstance(current, dict) and current.get("status") == "active"
    pending = any((p.get("pending"), a.get("pending"), ll.get("awaiting_observation"),
                   ll.get("pending_action"), graph.get("active"))) or bool(
        active and current.get("execution") and current["execution"][-1].get("status") != "completed"
    )
    unsolved = (hl.get("phase") == "unsolved" and hl.get("terminal_reason") in {
        "planner_did_not_solve", "time_budget_exhausted"
    } and hl.get("goal_completions") == 0)
    unsolved_evidence = bool(
        unsolved and isinstance(current, dict) and current.get("status") == "unsolved"
        and not pending and p["observations"] > 0 and knowledge["forwards"] > 0
        and all(len(set(chain)) == 1 for chain in chains)
        and all(
            row.get("status") in {"completed", "failed"}
            and all(item.get("legal") is True and type(item.get("observation_seq")) is int
                    for item in row.get("atomic_rows", []))
            for row in current.get("execution", [])
        )
    )
    checks = [
        success, (unsolved_evidence or _hierarchy_valid(completed)), all(item.get("failure") is None for item in modules),
        all(_single_flight(chain) for chain in chains), unsolved_evidence or hl.get("goal_completions", 0) > 0,
        unsolved_evidence or all(item.get(key, 0) > 0 for item, key in (
            (p, "observations"), (a, "statuses"), (ll, "terminal_updates"),
            (knowledge, "forwards"), (graph, "terminal"), (hl, "reconciled_outcomes"),
        )),
        active or all(len(set(chain)) == 1 for chain in chains),
        not active or pending or all(len(set(chain)) == 1 for chain in chains),
    ]
    if expected_treatment is not None:
        checks.append(hl.get("treatment") == expected_treatment)
    if environment is not None:
        final_facts = beliefs_to_facts(parse_symbolic_observation(environment.get(K_STATE, [])))
        initial_facts = beliefs_to_facts(parse_symbolic_observation(environment.get("initial_state", [])))
        digest = sha256(json.dumps(sorted(initial_facts), separators=(",", ":")).encode()).hexdigest()
        checks.extend([
            environment.get("illegal_actions") == 0,
            environment.get("actions") == ll["action_requests"],
            environment.get("observation_count") == ll["observations"],
            environment.get("closed") is True and environment.get("close_requests") == 1,
            bool(environment_logs) and not any(
                marker in line.lower() for line in environment_logs or []
                for marker in ("[error]", "traceback", "caught exception", "failed to save state")
            ),
        ])
        if expected_treatment is not None:
            goal = GoalSpec(**expected_treatment["goal"])
            checks.extend([
                environment.get("treatment") == expected_treatment,
                digest == expected_treatment["initial_state_digest"],
                goal.fact not in initial_facts,
                (goal.fact in final_facts) == (hl.get("goal_completions", 0) > 0),
            ])
    if verbose:
        print(f"Completed stacking goals: {hl.get('goal_completions', 0)}")
        print(f"LPG plans / ENHSP fallback plans: {hl.get('lpg_successes', 0)} / {hl.get('fallback_successes', 0)}")
    return all(checks)
