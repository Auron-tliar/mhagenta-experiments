"""Compact three-step experience assembly and proportional prioritized replay."""

from collections import deque
from typing import Any

import numpy as np

from .policy import shift_frame_stack
from .treatment import (
    BETA_START, DISCOUNT, N_STEP, PRIORITY_ALPHA, PRIORITY_EPSILON,
    REPLAY_BATCH_SIZE, REPLAY_BUFFER_SIZE,
)


def endpoint_stack(item: dict[str, Any]) -> np.ndarray:
    """Reconstruct the endpoint from a starting stack and successor frames."""
    stack = item["state_stack"]
    for frame in item["next_frames"]:
        stack = shift_frame_stack(stack, frame)
    return stack


class PrioritizedReplay:
    """Own a bounded replay ring and an episode-local unfinished suffix."""

    def __init__(self, rng: np.random.Generator, capacity: int = REPLAY_BUFFER_SIZE) -> None:
        self.rng = rng
        self.capacity = capacity
        self.items: list[dict[str, Any]] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pending: deque[dict[str, Any]] = deque()
        self.finalized = 0
        self.horizons = [0] * N_STEP
        self.evictions = 0
        self.stale_feedback = 0

    def append(self, transition: dict[str, Any], *, boundary: bool) -> None:
        """Finalize mature experiences, flushing shortened tails at boundaries."""
        self.pending.append(transition)
        while len(self.pending) >= N_STEP or (boundary and self.pending):
            steps = list(self.pending)[:N_STEP]
            item = {
                "replay_id": steps[0]["replay_id"],
                "state_stack": steps[0]["state_stack"], "action": steps[0]["action"],
                "reward": sum(DISCOUNT ** index * step["reward"] for index, step in enumerate(steps)),
                "next_frames": [step["next_frame"] for step in steps],
                "horizon": len(steps),
                "bootstrap_discount": 0.0 if steps[-1]["terminal"] else DISCOUNT ** len(steps),
            }
            slot = self.finalized % self.capacity
            if "next_action_mask" in steps[-1]:
                item["next_action_mask"] = list(steps[-1]["next_action_mask"])
            priority = float(self.priorities[:len(self.items)].max()) if self.items else 1.0
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                self.items[slot] = item
                self.evictions += 1
            self.priorities[slot] = priority
            self.finalized += 1
            self.horizons[len(steps) - 1] += 1
            self.pending.popleft()

    def flush(self) -> None:
        """Finalize a collection cutoff without marking it as terminal."""
        if self.pending:
            last = self.pending.pop()
            self.append(last, boundary=True)

    def sample(self, cycle_id: int, total_updates: int, *, beta: float | None = None) -> tuple[list[dict[str, Any]], list[int], list[float]]:
        """Sample with replacement and return batch-normalized IS weights."""
        if len(self.items) < REPLAY_BATCH_SIZE:
            raise ValueError("Replay has not reached the minimum batch size.")
        probabilities = self.priorities[:len(self.items)] ** PRIORITY_ALPHA
        probabilities /= probabilities.sum()
        indices = self.rng.choice(len(self.items), size=REPLAY_BATCH_SIZE, replace=True, p=probabilities)
        progress = (cycle_id - 1) / max(1, total_updates - 1)
        beta = BETA_START + (1.0 - BETA_START) * progress if beta is None else beta
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("Invalid importance beta.")
        weights = (len(self.items) * probabilities[indices]) ** -beta
        weights /= weights.max()
        batch = [self.items[int(index)] for index in indices]
        return batch, [item["replay_id"] for item in batch], weights.tolist()

    def update_priorities(self, ids: list[int], errors: list[float], *, allow_evicted: bool = False) -> None:
        """Apply maximum error per ID; optionally discard feedback for evicted IDs."""
        if len(ids) != REPLAY_BATCH_SIZE or len(errors) != len(ids):
            raise ValueError("Priority feedback must match the batch size.")
        updates: dict[int, float] = {}
        for replay_id, error in zip(ids, errors, strict=True):
            if type(replay_id) is not int or replay_id < 1:
                raise ValueError("Invalid replay ID.")
            if not np.isfinite(error) or error < 0:
                raise ValueError("Priority errors must be finite and nonnegative.")
            slot = (replay_id - 1) % self.capacity
            if slot >= len(self.items) or self.items[slot]["replay_id"] != replay_id:
                if allow_evicted and replay_id <= self.finalized - len(self.items):
                    self.stale_feedback += 1
                    continue
                raise ValueError("Stale or unknown replay ID.")
            if not np.isfinite(error) or error < 0:
                raise ValueError("Priority errors must be finite and nonnegative.")
            updates[slot] = max(updates.get(slot, 0.0), float(error))
        for slot, error in updates.items():
            self.priorities[slot] = error + PRIORITY_EPSILON
