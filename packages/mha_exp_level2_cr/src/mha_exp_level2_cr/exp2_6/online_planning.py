"""HLR teaching commands and shared observable skill boundary predicates."""

import numpy as np

from ..exp2_5.beliefs import Direction, available_movement_actions, cell_key, known_targets
from ..exp2_5.contracts import resource_collected
from ..exp2_5.cow_tracking import associate_cow, consumed_tracked_cow
from ..exp2_5.policy import PolicyId, encode_context, selection_actions
from .online_deliberation import routes


def safe_to_explore(belief: dict) -> bool:
    """Permit curiosity only with a reserve in every need and no sleeping state."""
    inv = belief["inventory"]
    return not belief["sleeping"] and inv.get("health", 0) >= 6 and all(
        inv.get(need, 0) >= 5 for need in ("food", "drink", "energy"))


def epsilon(progress: float, *, teacher: bool, safe: bool, evaluation: bool = False) -> float:
    """Anneal by elapsed agent time, with no transition-based stopping schedule."""
    if not safe or evaluation:
        return 0.0
    start, end = (0.05, 0.01) if teacher else (0.10, 0.02)
    return start + min(1.0, max(0.0, progress)) * (end - start)


def inputs(belief: dict, skill: str, target: list | tuple | None) -> tuple[np.ndarray, tuple[int, ...]]:
    """Use the same public masks for acting, teacher replay and Double-DQN targets."""
    if target is None:
        return np.zeros(2, dtype=np.float32), ()
    policy = PolicyId(skill)
    context = encode_context(policy, belief["player"], target)
    movements = available_movement_actions(belief, tuple(target))
    return context, selection_actions(policy, context, Direction.from_name(belief["facing"]).delta, movements)


def teacher_command(belief: dict, session: dict) -> dict:
    """HLR explicitly chooses navigation, orientation and native interaction.

    NavigateTo still selects its own native movement. HLR recomputes the
    designated cow's destination after every observed action.
    """
    target = tuple(session["target"])
    source = tuple(belief["player"])
    delta = target[0] - source[0], target[1] - source[1]
    if abs(delta[0]) + abs(delta[1]) == 1:
        action = (5 if Direction.from_name(belief["facing"]).delta == delta else
                  next(int(d.value[2]) for d in Direction if d.delta == delta))
        return {"kind": "primitive", "action": action, "target": list(target)}
    paths = routes(belief)
    neighbors = [(target[0] + d.delta[0], target[1] + d.delta[1]) for d in Direction]
    anchors = [p for p in neighbors if p in paths and p != source]
    if not anchors:
        return {"kind": "finish", "reason": "unreachable"}
    anchor = min(anchors, key=lambda p: (len(paths[p]), p))
    visible = [p for p in paths[anchor][1:11] if cell_key(p) in belief["visible_cells"]]
    if not visible:
        return {"kind": "finish", "reason": "unreachable"}
    return {"kind": "basic", "policy": "navigate_to", "target": list(visible[-1])}


def outcome(previous: dict, belief: dict, row: dict, session: dict) -> tuple[bool, str] | None:
    """Attribute success to the designated pre-action target; interrupt unsafe trials."""
    status, target = row["status"], row["target"]
    facing = Direction.from_name(previous["facing"]).delta
    if session["skill"] == "eat_target":
        success = consumed_tracked_cow(row["action"], previous["player"], facing,
                                       target, status["new_achievements"], status["illegal_action"])
        tracked = associate_cow(tuple(target) if target is not None else None,
                                known_targets(previous).get("cow", ()), known_targets(belief).get("cow", ()))
        session["target"] = list(tracked) if tracked is not None else None
    else:
        success = resource_collected(row["action"], previous["player"], facing, target,
                                     session["resource"], status["new_achievements"], status["illegal_action"])
    if success:
        return True, "success"
    if belief["terminal"]:
        return False, status.get("episode_reason", "environment_terminal")
    if session["target"] is None:
        return False, "target_lost"
    if session["steps"] >= 32:
        return False, "action_bound"
    inv = belief["inventory"]
    # Teacher recovery may finish the need it is actively repairing. All other
    # depleted needs preempt, including during autonomous learned execution.
    repairing = "food" if session["skill"] == "eat_target" else "drink" if session["resource"] == "water" else None
    if any(inv.get(need, 0) <= 2 and need != repairing for need in ("food", "drink", "energy")):
        return False, "survival_interrupt"
    if session["mode"] in {"trial", "probe"} and (inv.get("health", 0) <= 4 or any(
            inv.get(need, 0) <= 3 for need in ("food", "drink", "energy"))):
        return False, "survival_interrupt"
    if session["skill"] == "get_resource":
        target_key = cell_key(tuple(session["target"]))
        if belief["terrain"].get(target_key) != session["resource"]:
            return False, "target_lost"
    return None
