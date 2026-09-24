"""Replay and optimization primitives for expanded 2-5-CR policies."""

from __future__ import annotations

import random
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from mha_exp_level2_cr.exp2_5.policy import (
    CONTEXT_SIZE,
    OBSERVATION_SHAPE,
    PolicyId,
    legal_actions,
)

REPLAY_FORMAT_VERSION = 2


@dataclass(frozen=True)
class Transition:
    """One row with episode-local one-step and n-step successors."""

    frame: np.ndarray
    context: np.ndarray
    action: int
    reward: float
    next_frame: np.ndarray
    next_context: np.ndarray
    done: bool
    expert: bool
    n_reward: float = 0.0
    n_next_frame: np.ndarray | None = None
    n_next_context: np.ndarray | None = None
    n_done: bool = False
    n_steps: int = 1
    imitation_action: int | None = None


def episode_n_step(
    rows: Sequence[Transition],
    *,
    horizon: int,
    discount: float,
) -> list[Transition]:
    """Attach discounted n-step targets without crossing an episode boundary."""

    if horizon < 1:
        raise ValueError("n-step horizon must be positive.")
    result: list[Transition] = []
    for start, first in enumerate(rows):
        reward, last, steps = 0.0, first, 0
        for offset, row in enumerate(rows[start:start + horizon]):
            reward += discount**offset * row.reward
            last, steps = row, offset + 1
            if row.done:
                break
        result.append(Transition(
            first.frame,
            first.context,
            first.action,
            first.reward,
            first.next_frame,
            first.next_context,
            first.done,
            first.expert,
            reward,
            last.next_frame,
            last.next_context,
            last.done,
            steps,
            first.imitation_action,
        ))
    return result


class CompressedReplay:
    """Prioritized replay that permanently protects its demonstration prefix."""

    def __init__(
        self,
        capacity: int,
        seed: int,
        *,
        priority_alpha: float,
        demonstration_bonus: float,
        demonstration_fraction: float = 0.25,
        policy_id: PolicyId,
    ) -> None:
        if capacity < 2:
            raise ValueError("Replay needs room for both partitions.")
        if not isinstance(demonstration_fraction, float) or not 0 < demonstration_fraction < 1:
            raise ValueError("Demonstration fraction must be strictly between zero and one.")
        self.capacity = capacity
        self.seed = seed
        self.priority_alpha = priority_alpha
        self.demonstration_bonus = demonstration_bonus
        self.demonstration_fraction = demonstration_fraction
        self.policy_id = policy_id
        self.actions = legal_actions(policy_id)
        self.random = random.Random(seed)
        self.rows: list[tuple[Any, ...]] = []
        self.priorities: list[float] = []
        self.insertion_priority = 1.0
        self.demonstration_count: int | None = None
        self.cursor: int | None = None

    @staticmethod
    def _frame(value: np.ndarray) -> bytes:
        frame = np.ascontiguousarray(value, dtype=np.uint8)
        if frame.shape != OBSERVATION_SHAPE:
            raise ValueError("Replay frame violates the policy contract.")
        return zlib.compress(frame.tobytes(), 1)

    @staticmethod
    def _decode(value: bytes) -> np.ndarray:
        return np.frombuffer(zlib.decompress(value), dtype=np.uint8).reshape(OBSERVATION_SHAPE).copy()

    @classmethod
    def _pack(cls, row: Transition) -> tuple[Any, ...]:
        if row.n_next_frame is None or row.n_next_context is None:
            raise ValueError("Replay rows require n-step data.")
        return (
            cls._frame(row.frame),
            row.context.astype(np.float32),
            row.action,
            row.reward,
            cls._frame(row.next_frame),
            row.next_context.astype(np.float32),
            row.done,
            row.expert,
            row.n_reward,
            cls._frame(row.n_next_frame),
            row.n_next_context.astype(np.float32),
            row.n_done,
            row.n_steps,
            row.imitation_action,
        )

    @classmethod
    def _unpack(cls, row: tuple[Any, ...]) -> Transition:
        return Transition(
            cls._decode(row[0]),
            row[1].copy(),
            row[2],
            row[3],
            cls._decode(row[4]),
            row[5].copy(),
            row[6],
            row[7],
            row[8],
            cls._decode(row[9]),
            row[10].copy(),
            row[11],
            row[12],
            row[13],
        )

    def append(self, transition: Transition) -> None:
        """Append to the active partition and replace online rows only."""

        if transition.action not in self.actions:
            raise ValueError("Replay action violates the policy mask.")
        if transition.expert != (transition.imitation_action is not None):
            raise ValueError("Only demonstration rows may carry imitation actions.")
        if transition.imitation_action is not None and transition.imitation_action not in self.actions:
            raise ValueError("Replay imitation action violates the policy mask.")
        sealed = self.demonstration_count is not None
        if transition.expert == sealed:
            raise ValueError("Replay append does not match its active partition.")
        row = self._pack(transition)
        priority = self.insertion_priority + (self.demonstration_bonus if transition.expert else 0.0)
        if len(self.rows) < self.capacity:
            self.rows.append(row)
            self.priorities.append(priority)
            if sealed:
                self.cursor = len(self.rows) if len(self.rows) < self.capacity else self.demonstration_count
            return
        assert self.cursor is not None and self.demonstration_count is not None
        self.rows[self.cursor], self.priorities[self.cursor] = row, priority
        self.cursor += 1
        if self.cursor == self.capacity:
            self.cursor = self.demonstration_count

    def seal_demonstrations(self) -> None:
        """Freeze the nonempty expert prefix and activate online replacement."""

        if self.demonstration_count is not None or not self.rows or len(self.rows) >= self.capacity or not all(row[7] for row in self.rows):
            raise ValueError("Demonstration partition cannot be sealed.")
        self.demonstration_count = len(self.rows)
        self.cursor = len(self.rows)

    def sample(self, size: int, beta: float) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Sample prioritized rows and normalized importance weights."""

        if self.demonstration_count is None or size < 1 or not 0 <= beta <= 1:
            raise ValueError("Replay must be sealed and sampling values valid.")
        count = self.demonstration_count
        priorities = np.asarray(self.priorities, dtype=np.float64)
        demonstration_probabilities = priorities[:count]**self.priority_alpha
        demonstration_probabilities /= demonstration_probabilities.sum()
        if count == len(self.rows):
            selected = self.random.choices(
                range(count), weights=demonstration_probabilities, k=size,
            )
            indices = np.asarray(selected, dtype=np.int64)
            weights = (count * demonstration_probabilities[indices])**-beta
            return (
                [self._unpack(self.rows[index]) for index in indices],
                indices,
                (weights / weights.max()).astype(np.float32),
            )
        demonstration_size = round(size * self.demonstration_fraction)
        online_size = size - demonstration_size
        if demonstration_size < 1 or online_size < 1:
            raise ValueError("Replay sample must include both partitions.")
        online_probabilities = priorities[count:]**self.priority_alpha
        online_probabilities /= online_probabilities.sum()
        selected = self.random.choices(
            range(count),
            weights=demonstration_probabilities,
            k=demonstration_size,
        )
        selected.extend(self.random.choices(
            range(count, len(self.rows)),
            weights=online_probabilities,
            k=online_size,
        ))
        self.random.shuffle(selected)
        indices = np.asarray(selected, dtype=np.int64)
        mixture = np.asarray([self.demonstration_fraction * demonstration_probabilities[index] if index < count else (1 - self.demonstration_fraction) * online_probabilities[index - count] for index in selected])
        weights = (len(self.rows) * mixture)**-beta
        return (
            [self._unpack(self.rows[index]) for index in indices],
            indices,
            (weights / weights.max()).astype(np.float32),
        )

    def update(
        self,
        indices: np.ndarray,
        one_step: np.ndarray,
        n_step: np.ndarray,
        experts: np.ndarray,
    ) -> None:
        """Update priorities from the larger absolute TD error."""

        updates: dict[int, tuple[float, bool]] = {}
        for index, first, multi, expert in zip(indices, one_step, n_step, experts, strict=True):
            base = max(abs(float(first)), abs(float(multi))) + 1e-5
            key = int(index)
            previous = updates.get(key)
            if previous is None or base > previous[0]:
                updates[key] = base, bool(expert)
        for index in sorted(updates):
            base, expert = updates[index]
            self.insertion_priority = max(self.insertion_priority, base)
            self.priorities[index] = base + (self.demonstration_bonus if expert else 0.0)

    def state_dict(self) -> dict[str, Any]:
        """Return the complete exact-resume replay state."""

        return {
            "format_version": REPLAY_FORMAT_VERSION,
            "policy_id": self.policy_id.value,
            "capacity": self.capacity,
            "seed": self.seed,
            "priority_alpha": self.priority_alpha,
            "demonstration_bonus": self.demonstration_bonus,
            "demonstration_fraction": self.demonstration_fraction,
            "insertion_priority": self.insertion_priority,
            "rows": self.rows,
            "priorities": self.priorities,
            "demonstration_count": self.demonstration_count,
            "cursor": self.cursor,
            "random_state": self.random.getstate(),
        }

    @classmethod
    def _row_valid(
        cls,
        row: Any,
        n_step_horizon: int,
        actions: tuple[int, ...],
    ) -> bool:
        if not isinstance(row, tuple) or len(row) != 14:
            return False
        try:
            for index in (0, 4, 9):
                if not isinstance(row[index], bytes):
                    return False
                cls._decode(row[index])
        except (TypeError, ValueError, zlib.error):
            return False
        contexts = (row[1], row[5], row[10])
        rewards = (row[3], row[8])
        return (
            all(isinstance(context, np.ndarray) and context.shape == (CONTEXT_SIZE, ) and context.dtype == np.float32 and np.all(np.isfinite(context)) for context in contexts)
            and type(row[2]) is int and row[2] in actions
            and row[7] == (row[13] is not None)
            and (row[13] is None or type(row[13]) is int and row[13] in actions)
            and all(isinstance(reward, (int, float)) and not isinstance(reward, bool) and np.isfinite(reward) for reward in rewards)
            and all(type(row[index]) is bool for index in (6, 7, 11))
            and type(row[12]) is int and 1 <= row[12] <= n_step_horizon
        )

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, Any],
        *,
        expected_capacity: int | None = None,
        expected_seed: int | None = None,
        expected_priority_alpha: float | None = None,
        expected_demonstration_bonus: float | None = None,
        expected_demonstration_fraction: float | None = None,
        expected_policy_id: PolicyId | None = None,
        n_step_horizon: int = 3,
    ) -> CompressedReplay:
        """Restore and validate a complete replay state."""

        expected = {
            "format_version",
            "policy_id",
            "capacity",
            "seed",
            "priority_alpha",
            "demonstration_bonus",
            "demonstration_fraction",
            "insertion_priority",
            "rows",
            "priorities",
            "demonstration_count",
            "cursor",
            "random_state",
        }
        if set(value) != expected:
            raise ValueError("Working replay schema is invalid.")
        if value["format_version"] != REPLAY_FORMAT_VERSION:
            raise ValueError("Working replay format is incompatible.")
        try:
            policy_id = PolicyId(value["policy_id"])
        except (TypeError, ValueError) as error:
            raise ValueError("Working replay policy identity is invalid.") from error
        capacity, seed = value["capacity"], value["seed"]
        priority_alpha = value["priority_alpha"]
        demonstration_bonus = value["demonstration_bonus"]
        demonstration_fraction = value["demonstration_fraction"]
        if (type(capacity) is not int or type(seed) is not int or not isinstance(priority_alpha, float) or not isinstance(demonstration_bonus, float) or not isinstance(demonstration_fraction, float) or not 0 <= priority_alpha <= 1 or demonstration_bonus < 0 or not 0 < demonstration_fraction < 1 or
            (expected_capacity is not None and capacity != expected_capacity) or (expected_seed is not None and seed != expected_seed) or (expected_priority_alpha is not None and priority_alpha != expected_priority_alpha) or
            (expected_demonstration_bonus is not None and demonstration_bonus != expected_demonstration_bonus) or (expected_demonstration_fraction is not None and demonstration_fraction != expected_demonstration_fraction) or
            (expected_policy_id is not None and policy_id is not expected_policy_id) or type(n_step_horizon) is not int or n_step_horizon < 1):
            raise ValueError("Working replay identity is invalid.")
        replay = cls(
            capacity,
            seed,
            priority_alpha=priority_alpha,
            demonstration_bonus=demonstration_bonus,
            demonstration_fraction=demonstration_fraction,
            policy_id=policy_id,
        )
        replay.rows = list(value["rows"])
        replay.priorities = list(value["priorities"])
        replay.insertion_priority = value["insertion_priority"]
        replay.demonstration_count = value["demonstration_count"]
        replay.cursor = value["cursor"]
        count = replay.demonstration_count
        common_invalid = (len(replay.rows) != len(replay.priorities) or len(replay.rows) > replay.capacity or not isinstance(replay.insertion_priority, (int, float)) or not np.isfinite(replay.insertion_priority) or replay.insertion_priority <= 0 or any(not cls._row_valid(row, n_step_horizon, replay.actions) for row in replay.rows) or
                          any(not isinstance(priority, (int, float)) or not np.isfinite(priority) or priority <= 0 for priority in replay.priorities) or any(priority - (demonstration_bonus if row[7] else 0.0) > replay.insertion_priority + 1e-12 for row, priority in zip(replay.rows, replay.priorities, strict=True)))
        unsealed_invalid = count is None and (replay.cursor is not None or any(not row[7] for row in replay.rows))
        sealed_invalid = count is not None and (type(count) is not int or not 0 < count < replay.capacity or type(replay.cursor) is not int or count > len(replay.rows) or any(not row[7] for row in replay.rows[:count]) or any(row[7] for row in replay.rows[count:]) or
                                                (replay.cursor != len(replay.rows) if len(replay.rows) < replay.capacity else not count <= replay.cursor < replay.capacity))
        if common_invalid or unsealed_invalid or sealed_invalid:
            raise ValueError("Working replay state is invalid.")
        try:
            replay.random.setstate(value["random_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("Working replay random state is invalid.") from error
        return replay


def optimize(
    torch: Any,
    online: Any,
    target: Any,
    optimizer: Any,
    replay: CompressedReplay,
    device: str,
    beta: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Apply weighted Double-DQN one-step, n-step, and margin losses."""

    rows, indices, weights_np = replay.sample(config["batch_size"], beta)

    def images(values: Sequence[np.ndarray]) -> Any:
        data = np.transpose(np.stack(values), (0, 3, 1, 2))
        return torch.as_tensor(data, dtype=torch.uint8, device=device)

    current_images = images([row.frame for row in rows])
    contexts = torch.as_tensor(
        np.stack([row.context for row in rows]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor([row.action for row in rows], dtype=torch.long, device=device)
    weights = torch.as_tensor(weights_np, device=device)
    q_values = online(current_images, contexts)
    chosen = q_values.gather(1, actions[:, None]).squeeze(1)

    def target_values(n_step: bool) -> tuple[Any, Any]:
        next_frames = [row.n_next_frame if n_step else row.next_frame for row in rows]
        next_contexts = torch.as_tensor(
            np.stack([row.n_next_context if n_step else row.next_context for row in rows]),
            dtype=torch.float32,
            device=device,
        )
        next_images = images(next_frames)  # type: ignore[arg-type]
        with torch.no_grad():
            actions = torch.as_tensor(replay.actions, dtype=torch.long, device=device)
            legal_q = online(next_images, next_contexts).index_select(1, actions)
            next_actions = actions[legal_q.argmax(1)]
            bootstrap = target(next_images, next_contexts).gather(1, next_actions[:, None]).squeeze(1)
        rewards = torch.as_tensor(
            [row.n_reward if n_step else row.reward for row in rows],
            dtype=torch.float32,
            device=device,
        )
        done = torch.as_tensor(
            [row.n_done if n_step else row.done for row in rows],
            dtype=torch.float32,
            device=device,
        )
        steps = torch.as_tensor(
            [row.n_steps if n_step else 1 for row in rows],
            dtype=torch.float32,
            device=device,
        )
        expected = rewards + config["discount"]**steps * (1 - done) * bootstrap
        return expected - chosen, (weights * (expected - chosen).square()).mean()

    one_td, one_loss = target_values(False)
    n_td, n_loss = target_values(True)
    imitation_mask = torch.as_tensor(
        [row.expert for row in rows],
        dtype=torch.bool,
        device=device,
    )
    imitation_actions = torch.as_tensor(
        [row.action if row.imitation_action is None else row.imitation_action for row in rows],
        dtype=torch.long,
        device=device,
    )
    experts = torch.as_tensor([row.expert for row in rows], dtype=torch.bool, device=device)
    margin_loss = balanced_margin_loss(
        torch,
        q_values,
        imitation_actions,
        imitation_mask,
        config["large_margin"],
        replay.actions,
    )
    loss = config["one_step_loss_weight"] * one_loss + config["n_step_loss_weight"] * n_loss + config["margin_loss_weight"] * margin_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), config["gradient_clip"])
    optimizer.step()
    replay.update(
        indices,
        one_td.detach().cpu().numpy(),
        n_td.detach().cpu().numpy(),
        experts.detach().cpu().numpy(),
    )
    return {
        "loss": float(loss.item()),
        "one_step": float(one_loss.item()),
        "n_step": float(n_loss.item()),
        "margin": float(margin_loss.item()),
    }


def balanced_margin_loss(
    torch: Any,
    q_values: Any,
    imitation_actions: Any,
    imitation_mask: Any,
    large_margin: float,
    legal_action_mask: Sequence[int],
) -> Any:
    """Return action-balanced margin loss over legal demonstration actions."""

    legal = torch.as_tensor(legal_action_mask, dtype=torch.long, device=q_values.device)
    legal_values = q_values.index_select(1, legal)
    local_actions = torch.stack([
        torch.where(legal == action)[0][0] for action in imitation_actions
    ])
    margins = torch.full_like(legal_values, large_margin)
    margins.scatter_(1, local_actions[:, None], 0.0)
    imitated = legal_values.gather(1, local_actions[:, None]).squeeze(1)
    losses = (legal_values + margins).max(1).values - imitated
    per_action = [
        losses[selected].mean()
        for action in legal_action_mask
        if bool((selected := imitation_mask & imitation_actions.eq(action)).any())
    ]
    return torch.stack(per_action).mean() if per_action else q_values.new_zeros(())
