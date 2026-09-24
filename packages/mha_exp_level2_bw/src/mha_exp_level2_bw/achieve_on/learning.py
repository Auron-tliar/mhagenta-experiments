"""Shared atomic DQfD-lite updates for offline and runtime AchieveOn learning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from mha_exp_level2_bw.exp2_5.policy import legal_action_indices

@dataclass(frozen=True)
class Config:
    """Shared optimizer settings; each runner supplies its own learning budget."""

    seed: int = 2605
    demonstrations: int = 200
    warm_updates: int = 2000
    online_steps: int = 250_000
    selection_interval: int = 10_000
    selection_cases: int = 100
    action_cap: int = 64
    capacity: int = 50_000
    batch_size: int = 128
    discount: float = 0.99
    n_step: int = 3
    learning_rate: float = 0.00003
    demonstration_fraction: float = 0.5
    target_sync: int = 500
    margin: float = 0.8

    def __post_init__(self) -> None:
        """Reject invalid budgets before creating any output or GPU state."""
        for name in ("seed", "demonstrations", "warm_updates", "online_steps", "selection_interval",
                     "selection_cases", "action_cap", "capacity", "batch_size", "n_step", "target_sync"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid {name}: {value!r}")
        if max(self.demonstrations, self.online_steps, self.selection_cases) > 1_000_000:
            raise ValueError("Budget exceeds the reserved seed window.")
        if self.n_step > self.action_cap:
            raise ValueError("N-step horizon exceeds the episode cap.")
        for name in ("discount", "learning_rate", "margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}.")
        if self.discount > 1:
            raise ValueError("Discount cannot exceed one.")
        if not 0 < self.demonstration_fraction < 1:
            raise ValueError("Demonstration fraction must be strictly between zero and one.")


@dataclass
class Transition:
    """An atomic transition conditioned only on the final compound goal."""

    state: np.ndarray
    action: int
    reward: float
    successor: np.ndarray
    terminal: bool
    legal_next: np.ndarray
    discount: float = 0.99


def legal_mask(observation: np.ndarray) -> np.ndarray:
    """Use the existing observation-derived legality contract."""
    mask = np.zeros(4, dtype=np.bool_)
    mask[legal_action_indices(observation)] = True
    return mask


def emit(pending: deque[Transition], config: Config, flush: bool) -> list[Transition]:
    """Aggregate n-step rewards without crossing a terminal episode boundary."""
    result = []
    while pending and (len(pending) >= config.n_step or flush):
        rows = list(pending)[:config.n_step]
        reward = 0.0
        for index, row in enumerate(rows):
            reward += config.discount ** index * row.reward
            if row.terminal:
                break
        result.append(Transition(pending[0].state, pending[0].action, reward,
                                 row.successor, row.terminal, row.legal_next,
                                 config.discount ** (index + 1)))
        pending.popleft()
    return result


class Replay:
    """Prioritized replay with a protected demonstration prefix."""

    def __init__(self, capacity: int, demonstrations: list[Transition],
                 difficulties: list[int] | None = None) -> None:
        if not 0 < len(demonstrations) < capacity:
            raise ValueError("Demonstrations must fit with space left for online replay.")
        self.rows = list(demonstrations)
        self.protected = len(demonstrations)
        self.capacity = capacity
        self.cursor = self.protected
        self.priorities = np.ones(capacity, dtype=np.float64)
        self.difficulties = np.asarray(difficulties if difficulties is not None else [0] * self.protected)
        if len(self.difficulties) != self.protected or not np.isin(self.difficulties, [0, 1, 2]).all():
            raise ValueError("Every demonstration transition needs an obstacle stratum 0, 1, or 2.")

    def append(self, row: Transition) -> None:
        """Overwrite only ordinary experience when capacity is reached."""
        if self.cursor == len(self.rows):
            self.rows.append(row)
        else:
            self.rows[self.cursor] = row
        self.priorities[self.cursor] = self.priorities[:len(self.rows)].max()
        self.cursor += 1
        if self.cursor == self.capacity:
            self.cursor = self.protected

    def sample(self, rng: Any, count: int, beta: float, demonstration_fraction: float = 0.5,
               max_difficulty: int = 2) -> tuple:
        """Stratify demonstrations and experience, correcting actual sampling probabilities."""
        priority = self.priorities[:len(self.rows)] ** 0.6
        available = np.unique(self.difficulties[self.difficulties <= max_difficulty])
        if not len(available):
            available = np.array([self.difficulties.min()])
        probabilities = np.zeros(len(self.rows))
        for difficulty in available:
            mask = np.flatnonzero(self.difficulties == difficulty)
            probabilities[mask] = priority[mask] / priority[mask].sum() / len(available)
        demo_count = count
        if len(self.rows) > self.protected:
            demo_count = (max(1, min(count - 1, round(count * demonstration_fraction)))
                          if count > 1 else int(rng.random() < demonstration_fraction))
            # For a one-row batch, use the marginal mixture probability.
            mass = demo_count / count if count > 1 else demonstration_fraction
            probabilities[:self.protected] *= mass
            ordinary = priority[self.protected:]
            probabilities[self.protected:] = (1 - mass) * ordinary / ordinary.sum()
        parts = []
        for start, stop, size in ((0, self.protected, demo_count),
                                  (self.protected, len(self.rows), count - demo_count)):
            if size:
                p = probabilities[start:stop]
                parts.append(rng.choice(np.arange(start, stop), size=size, p=p / p.sum()))
        indices = np.concatenate(parts)
        rng.shuffle(indices)
        weights = (len(self.rows) * probabilities[indices]) ** -beta
        weights /= (len(self.rows) * probabilities[probabilities > 0].min()) ** -beta
        return indices, [self.rows[index] for index in indices], weights


def curriculum_limit(update: int, total: int) -> int:
    """Introduce one-obstacle and harder demonstrations after each warm-up third."""
    return min(2, 3 * update // max(1, total))


def optimize(model: Any, target: Any, optimizer: Any, replay: Replay,
             rng: Any, config: Config, device: str, beta: float,
             metrics: dict | None = None, max_difficulty: int = 2) -> float:
    """Perform legal Double-DQN and demonstration-margin optimization."""
    indices, rows, weights = replay.sample(rng, config.batch_size, beta,
                                         config.demonstration_fraction, max_difficulty)

    def tensor(value: Any, dtype: Any = torch.float32) -> Any:
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

    q = model(tensor([row.state for row in rows]))
    actions = tensor([row.action for row in rows], torch.long)
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        successor = tensor([row.successor for row in rows])
        mask = tensor([row.legal_next for row in rows], torch.bool)
        next_actions = model(successor).masked_fill(~mask, -torch.inf).argmax(1)
        next_values = target(successor).gather(1, next_actions[:, None]).squeeze(1)
        expected = tensor([row.reward for row in rows]) + tensor([
            0.0 if row.terminal else row.discount for row in rows]) * next_values
    td = expected - chosen
    td_loss = (tensor(weights) * torch.nn.functional.smooth_l1_loss(chosen, expected, reduction="none")).mean()
    imitation_loss = q.new_zeros(())
    demo = tensor(indices < replay.protected, torch.bool)
    if demo.any():
        margin = torch.full_like(q, config.margin)
        margin.scatter_(1, actions[:, None], 0)
        imitation_loss = ((q + margin).max(1).values - chosen)[demo].mean()
    loss = td_loss + imitation_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss.")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    priorities = td.detach().abs().cpu().numpy() + 1e-5 + (indices < replay.protected) * 0.1
    for index, priority in zip(indices, priorities, strict=True):
        replay.priorities[index] = float(priority)
    if metrics is not None:
        metrics.update(td_loss=float(td_loss.detach().cpu()), imitation_loss=float(imitation_loss.detach().cpu()),
                       demonstration_fraction=float(demo.float().mean().cpu()),
                       mean_absolute_td=float(td.detach().abs().mean().cpu()))
    return float(loss.detach().cpu())
