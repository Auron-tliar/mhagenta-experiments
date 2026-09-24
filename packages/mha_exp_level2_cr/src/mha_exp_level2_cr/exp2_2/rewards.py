"""Achievement reward assignment using evidence unavailable to policy inference."""

from typing import Any

from .treatment import (
    ACHIEVEMENT_REWARDS, DEATH_PENALTY, ILLEGAL_ACTION_PENALTY,
    NEED_THRESHOLD, STEP_REWARD, SURVIVAL_REWARDS,
)


def reward_state() -> dict[str, Any]:
    """Return compact episode-local reward bookkeeping."""
    return {"episode_id": None, "awarded": [], "eligible_sleep": False}


def validate_evidence(evidence: Any) -> dict[str, Any]:
    """Validate pre/post achievement, need, and sleep snapshots."""
    if not isinstance(evidence, dict):
        raise ValueError("Reward evidence must be a dictionary.")
    from .policy import ACHIEVEMENTS

    for side in ("before", "after"):
        snapshot = evidence.get(side)
        if not isinstance(snapshot, dict) or type(snapshot.get("sleeping")) is not bool:
            raise ValueError("Reward evidence requires Boolean sleeping state.")
        needs = snapshot.get("needs")
        if not isinstance(needs, dict) or set(needs) != {"health", "food", "drink", "energy"}:
            raise ValueError("Reward evidence requires health and all needs.")
        if any(type(value) is not int or not 0 <= value <= 9 for value in needs.values()):
            raise ValueError("Need levels must be integers from 0 through 9.")
        counts = snapshot.get("achievements")
        if not isinstance(counts, dict) or set(counts) != set(ACHIEVEMENTS):
            raise ValueError("Reward evidence requires complete achievement counts.")
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("Achievement counts must be nonnegative integers.")
    if any(evidence["after"]["achievements"][name] < evidence["before"]["achievements"][name]
           for name in ACHIEVEMENTS):
        raise ValueError("Achievement counts decreased within an action.")
    return evidence


def evaluate_reward(
    tracker: dict[str, Any], episode_id: int, illegal: bool,
    evidence: dict[str, Any],
) -> dict[str, float]:
    """Update episode bookkeeping and return additive named reward components."""
    validate_evidence(evidence)
    if type(episode_id) is not int or episode_id < 1:
        raise ValueError("Episode ID must be a positive integer.")
    if tracker["episode_id"] != episode_id:
        if tracker["episode_id"] is not None and episode_id != tracker["episode_id"] + 1:
            raise ValueError("Reward episodes must be consecutive.")
        tracker.update(episode_id=episode_id, awarded=[], eligible_sleep=False)
    before, after = evidence["before"], evidence["after"]
    components = {"step": STEP_REWARD, "illegal_action": ILLEGAL_ACTION_PENALTY if illegal else 0.0}
    if after["needs"]["health"] == 0:
        tracker["eligible_sleep"] = False
        return {**components, "death": DEATH_PENALTY}
    if not before["sleeping"] and after["sleeping"]:
        tracker["eligible_sleep"] = before["needs"]["energy"] <= NEED_THRESHOLD
    new_events = {name for name, count in after["achievements"].items()
                  if count > before["achievements"][name]}
    eligible = set(ACHIEVEMENT_REWARDS) & new_events
    for achievement, need in (("collect_drink", "drink"), ("eat_cow", "food"), ("eat_plant", "food")):
        if (achievement in new_events and before["needs"][need] <= NEED_THRESHOLD
                and after["needs"][need] > before["needs"][need]):
            eligible.add(achievement)
    if "wake_up" in new_events and before["sleeping"] and tracker["eligible_sleep"]:
        eligible.add("wake_up")
    if not after["sleeping"]:
        tracker["eligible_sleep"] = False
    bonuses = {**ACHIEVEMENT_REWARDS, **SURVIVAL_REWARDS}
    for name in sorted(eligible - set(tracker["awarded"])):
        components[name] = bonuses[name]
        tracker["awarded"].append(name)
    return components
