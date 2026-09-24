"""Fixed DQfD demonstration starts, independent of candidate behavior or evaluation."""

from typing import Any

import numpy as np

from mha_exp_level2_bw.exp2_5.contracts import GoalSpec
from mha_exp_level2_bw.exp2_5.grounding import ground_observation
from mha_exp_level2_bw.exp2_5.policy import legal_action_indices


def obstruction_count(facts: Any, goal: GoalSpec) -> int:
    """Count distinct blocks above either goal block in the initial stacks."""
    above = {}
    for fact in facts:
        if fact.startswith("on("):
            top, bottom = fact[3:-1].split(",")
            above[bottom] = top
    obstacles = set()
    for block in (goal.top, goal.bottom):
        while block in above:
            block = above[block]
            obstacles.add(block)
    return len(obstacles)


def demonstration_start(environment: Any, observation: np.ndarray, seed: int,
                        index: int) -> tuple[np.ndarray, GoalSpec, dict]:
    """Balance goal difficulty; perturb one quarter of starts before choosing a goal.

    Perturbations are fixed random legal actions, never learner decisions or
    imitation labels. End with an empty hand so the Transfer planner applies.
    """
    rng = np.random.default_rng(seed)
    prefix = []
    if (index // 3) % 4 == 3:
        for _ in range(8):
            action = int(rng.choice(legal_action_indices(observation)))
            observation, _, terminated, truncated, info = environment.step(action)
            if not info["snapshot"].legal or terminated or truncated:
                raise RuntimeError("Invalid demonstration perturbation.")
            prefix.append(action)
        if "hand-empty()" not in ground_observation(observation).facts:
            if 1 not in legal_action_indices(observation):
                raise RuntimeError("Cannot settle the demonstration's held block.")
            observation, _, terminated, truncated, info = environment.step(1)
            if not info["snapshot"].legal or terminated or truncated:
                raise RuntimeError("Invalid settling action.")
            prefix.append(1)
    facts = ground_observation(observation).facts
    goals = [GoalSpec(f"b{top}", f"b{bottom}") for top in range(8) for bottom in range(8)
             if top != bottom and f"on(b{top},b{bottom})" not in facts]
    requested = index % 3
    buckets = {level: [goal for goal in goals if min(2, obstruction_count(facts, goal)) == level]
               for level in range(3)}
    actual = min((level for level in buckets if buckets[level]), key=lambda level: (abs(level - requested), level))
    goal = buckets[actual][int(rng.integers(len(buckets[actual])))]
    return observation, goal, {"reset_actions": prefix, "requested_difficulty": requested,
                               "difficulty": actual, "obstructions": obstruction_count(facts, goal)}
