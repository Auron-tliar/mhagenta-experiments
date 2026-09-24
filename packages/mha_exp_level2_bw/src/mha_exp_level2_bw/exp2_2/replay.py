"""Episode-bounded n-step returns and FIFO proportional prioritized replay."""

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from .protocol import DQNProtocol
from .policy import (
    NUM_BLOCKS,
    block_position,
    goal_achieved,
    goal_conditioned_observation,
)


def achieved_on_relations(observation: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return every directed block-on-block relation in one observation."""

    positions = [block_position(observation, block) for block in range(NUM_BLOCKS)]
    occupants = {
        position: block
        for block, position in enumerate(positions)
        if position is not None
    }
    return tuple(
        (top, occupants[(position[0] + 1, position[1])])
        for top, position in enumerate(positions)
        if position is not None and (position[0] + 1, position[1]) in occupants
    )


class HindsightRelabeler:
    """Create n-step entries for future relations newly achieved in an episode."""

    def __init__(
        self,
        protocol: DQNProtocol,
        rng: np.random.Generator,
        reward: Callable[[bool, bool], float],
    ) -> None:
        self.protocol = protocol
        self.rng = rng
        self.reward = reward

    def relabel(
        self,
        episode: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Return relabeled entries, contributing sources, and candidate count."""

        entries: list[dict[str, Any]] = []
        contributing_sources = 0
        candidate_count = 0
        future_relations: set[tuple[int, int]] = set()
        candidates_by_start: list[tuple[tuple[int, int], ...]] = [tuple()] * len(episode)
        for index in range(len(episode) - 1, -1, -1):
            future_relations.update(achieved_on_relations(episode[index]["next_observation"]))
            already_true = set(achieved_on_relations(episode[index]["state_observation"]))
            original_goal = tuple(episode[index]["goal"])
            candidates_by_start[index] = tuple(sorted(
                future_relations - already_true - {original_goal}
            ))

        for start, candidates in enumerate(candidates_by_start):
            candidate_count += len(candidates)
            if not candidates:
                continue
            count = min(self.protocol.her_future_goals, len(candidates))
            selected = self.rng.choice(len(candidates), size=count, replace=False)
            contributing_sources += 1
            for candidate_index in np.atleast_1d(selected):
                entries.append(self._entry(episode, start, candidates[int(candidate_index)]))
        return entries, contributing_sources, candidate_count

    def _entry(
        self,
        episode: list[dict[str, Any]],
        start: int,
        goal: tuple[int, int],
    ) -> dict[str, Any]:
        first = episode[start]
        reward = 0.0
        terminal = False
        hindsight_achieved = False
        horizon = 0
        last = first
        for offset, transition in enumerate(
            episode[start:start + self.protocol.n_steps]
        ):
            achieved = goal_achieved(transition["next_observation"], goal)
            hindsight_achieved = achieved
            reward += self.protocol.discount ** offset * self.reward(
                bool(transition["illegal_action"]), achieved
            )
            horizon += 1
            last = transition
            terminal = achieved or bool(transition["terminal"])
            if terminal:
                break
        return {
            "state": goal_conditioned_observation(first["state_observation"], goal),
            "action": int(first["action"]),
            "reward": reward,
            "next_state": goal_conditioned_observation(last["next_observation"], goal),
            "terminal": terminal,
            "horizon": horizon,
            "bootstrap_discount": 0.0 if terminal else self.protocol.discount ** horizon,
            "actor_update": int(first["actor_update"]),
            "her": True,
            "her_goal_achieved": hindsight_achieved,
            "goal": goal,
            "source_transition_index": int(first["transition_index"]),
        }


class NStepReturns:
    """Emit one replay entry per raw transition without crossing episode boundaries."""

    def __init__(self, steps: int, discount: float) -> None:
        self.steps = steps
        self.discount = discount
        self.pending: deque[dict[str, Any]] = deque()
        self.goal: tuple[int, int] | None = None

    def append(self, transition: dict[str, Any], goal: tuple[int, int]) -> list[dict[str, Any]]:
        """Accumulate one transition, flushing all tails after a terminal step."""
        if self.pending and goal != self.goal:
            raise ValueError("Goal changed before the n-step tail was closed")
        self.goal = goal
        self.pending.append(transition)
        if transition["terminal"]:
            return self.flush()
        return [self._emit()] if len(self.pending) >= self.steps else []

    def _emit(self) -> dict[str, Any]:
        sequence = list(self.pending)[:self.steps]
        first, last = sequence[0], sequence[-1]
        result = dict(first)
        result.update(
            reward=sum(self.discount ** i * item["reward"] for i, item in enumerate(sequence)),
            next_state=last["next_state"], terminal=last["terminal"], horizon=len(sequence),
            bootstrap_discount=0.0 if last["terminal"] else self.discount ** len(sequence),
        )
        self.pending.popleft()
        return result

    def flush(self) -> list[dict[str, Any]]:
        """Emit shortened tails; collection truncation retains bootstrapping."""
        result = []
        while self.pending:
            result.append(self._emit())
        self.goal = None
        return result


class PrioritizedReplay:
    """Bounded ring buffer with monotonic IDs protecting delayed priority feedback."""

    def __init__(self, protocol: DQNProtocol, rng: np.random.Generator) -> None:
        self.protocol = protocol
        self.rng = rng
        self.entries: list[dict[str, Any] | None] = [None] * protocol.replay_capacity
        self.ids = np.full(protocol.replay_capacity, -1, dtype=np.int64)
        self.priorities = np.zeros(protocol.replay_capacity, dtype=np.float64)
        self.count = 0
        self.next_id = 0
        self.max_priority = 1.0
        self.evictions = 0
        self.stale_feedback = 0

    def __len__(self) -> int:
        return self.count

    def append(self, entry: dict[str, Any]) -> None:
        """Insert a fresh entry, evicting the oldest entry when full."""
        slot = self.next_id % len(self.entries)
        self.evictions += int(self.count == len(self.entries))
        self.entries[slot] = entry
        self.ids[slot] = self.next_id
        self.priorities[slot] = self.max_priority ** self.protocol.priority_alpha
        self.next_id += 1
        self.count = min(self.count + 1, len(self.entries))

    def feedback(self, ids: list[int], errors: list[float]) -> None:
        """Apply maximum error per sampled ID; discard feedback for evicted IDs."""
        if len(ids) != len(errors) or any(not np.isfinite(e) or e < 0 for e in errors):
            raise ValueError("Invalid replay priority feedback")
        merged: dict[int, float] = {}
        for entry_id, error in zip(ids, errors):
            if type(entry_id) is not int or entry_id < 0 or entry_id >= self.next_id:
                raise ValueError("Invalid replay entry ID")
            merged[entry_id] = max(merged.get(entry_id, 0.0), error)
        for entry_id, error in merged.items():
            slot = entry_id % len(self.entries)
            if self.ids[slot] != entry_id:
                self.stale_feedback += 1
                continue
            priority = error + self.protocol.priority_epsilon
            self.max_priority = max(self.max_priority, priority)
            self.priorities[slot] = priority ** self.protocol.priority_alpha

    def sample(self, beta: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Sample with replacement and normalize importance weights globally."""
        if not self.count or not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("Cannot sample empty replay or invalid beta")
        probabilities = self.priorities[:self.count] / self.priorities[:self.count].sum()
        slots = self.rng.choice(self.count, size=self.protocol.batch_size, replace=True, p=probabilities)
        ids = self.ids[slots]
        weights = (probabilities[slots] / probabilities.min()) ** -beta
        return [self.entries[int(slot)] for slot in slots], {  # type: ignore[misc]
            "entry_ids": ids.tolist(), "weights": weights.tolist(), "beta": beta,
            "mean_replay_age": float(np.mean(self.next_id - 1 - ids)),
        }
