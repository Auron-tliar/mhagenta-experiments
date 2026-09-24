"""Bounded replay and DQfD updates; wall time, never transitions, ends learning."""

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .online_policy import candidate, weights_hash


@dataclass(frozen=True)
class LearningConfig:
    """Memory capacities and optimizer settings, not execution budgets."""

    capacity: int = 10000
    protected_capacity: int = 2000
    batch_size: int = 32
    updates_per_segment: int = 8
    discount: float = 0.99
    learning_rate: float = 0.0001
    target_interval: int = 200
    margin: float = 0.8
    alpha: float = 0.6


def transitions(segment: dict) -> list[dict]:
    """Build one- and three-step targets without crossing any skill/world boundary."""
    rows = segment["rows"]
    if not rows or not rows[-1]["terminal"] or any(row["terminal"] for row in rows[:-1]):
        raise ValueError("A replay segment must have exactly one terminal suffix")
    output = []
    for start, row in enumerate(rows):
        reward, horizon = 0.0, 0
        last = row
        for last in rows[start:start + 3]:
            reward += 0.99 ** horizon * last["reward"]
            horizon += 1
            if last["terminal"]:
                break
        output.append({**row, "n_reward": reward, "n_discount": 0.99 ** horizon,
                       "n_next": last["next"], "n_context": last["next_context"],
                       "n_mask": last["next_mask"], "n_terminal": last["terminal"],
                       "demo": segment["mode"] == "teacher" and segment["success"]
                               and not row.get("exploratory", False)})
    return output


class Trainer:
    """Own one skill's fresh model, optimizer, protected demonstrations and replay."""

    def __init__(self, torch: Any, basic: Any, seed: int, device: str,
                 config: LearningConfig = LearningConfig()) -> None:
        self.torch, self.device, self.config = torch, device, config
        self.model = candidate(torch, basic).to(device)
        self.target = deepcopy(self.model).eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(seed)
        self.protected: list[dict] = []
        self.rows: list[dict] = []
        self.priorities: list[float] = []
        self.cursor = self.updates = self.segments = self.seen = 0
        self.last_loss: float | None = None

    def ingest(self, segment: dict) -> None:
        """Keep all ordinary failures; preserve a bounded prefix of successful teaching."""
        if segment["mode"] == "probe":
            raise ValueError("Evaluation experience cannot enter replay")
        for row in transitions(segment):
            self.seen += 1
            if row["demo"] and len(self.protected) < self.config.protected_capacity:
                self.protected.append(row)
            else:
                if len(self.rows) < self.config.capacity:
                    self.rows.append(row)
                    self.priorities.append(1.0)
                else:
                    self.rows[self.cursor] = row
                    self.priorities[self.cursor] = max(self.priorities, default=1.0)
                    self.cursor = (self.cursor + 1) % self.config.capacity
        self.segments += 1

    def optimize(self, progress: float) -> None:
        """Perform one legal Double-DQN update with demonstration margin and PER."""
        t, cfg = self.torch, self.config
        demo_count = cfg.batch_size // 2 if self.protected else 0
        ordinary_count = cfg.batch_size - demo_count if self.rows else 0
        if not ordinary_count:
            demo_count = cfg.batch_size if self.protected else 0
        if not demo_count and not ordinary_count:
            return
        sampled = [self.protected[int(i)] for i in self.rng.integers(len(self.protected), size=demo_count)] if demo_count else []
        weights = [1.0] * demo_count
        indices = []
        if ordinary_count:
            probabilities = np.asarray(self.priorities, dtype=np.float64) ** cfg.alpha
            probabilities /= probabilities.sum()
            indices = self.rng.choice(len(self.rows), ordinary_count, p=probabilities)
            sampled.extend(self.rows[int(i)] for i in indices)
            correction = (len(self.rows) * probabilities[indices]) ** -(0.4 + 0.6 * progress)
            weights.extend((correction / correction.max()).tolist())

        def tensor(key: str, dtype: Any = None) -> Any:
            values = np.stack([row[key] for row in sampled])
            if key in {"state", "next", "n_next"}:
                values = values.transpose(0, 3, 1, 2)
            return t.as_tensor(values, device=self.device, dtype=dtype)

        action = tensor("action", t.long)
        q = self.model(tensor("state"), tensor("context", t.float32))
        chosen = q.gather(1, action[:, None]).squeeze(1)
        targets = []
        with t.no_grad():
            for prefix in ("", "n_"):
                images = tensor("next" if not prefix else "n_next")
                context = tensor("next_context" if not prefix else "n_context", t.float32)
                mask = tensor("next_mask" if not prefix else "n_mask", t.bool)
                online = self.model(images, context).masked_fill(~mask, -1e9)
                values = self.target(images, context).gather(1, online.argmax(1)[:, None]).squeeze(1)
                done = tensor("terminal" if not prefix else "n_terminal", t.bool)
                discount = cfg.discount if not prefix else tensor("n_discount", t.float32)
                targets.append(tensor("reward" if not prefix else "n_reward", t.float32)
                               + discount * values * (~done & mask.any(1)))
        error = (chosen - targets[0]).abs()
        losses = sum(t.nn.functional.smooth_l1_loss(chosen, target, reduction="none") for target in targets)
        legal = tensor("mask", t.bool)
        margin = t.full_like(q, cfg.margin)
        margin.scatter_(1, action[:, None], 0)
        imitation = ((q + margin).masked_fill(~legal, -1e9).max(1).values - chosen)
        losses += imitation * tensor("demo", t.float32)
        loss = (losses * t.tensor(weights, device=self.device)).mean()
        if not t.isfinite(loss):
            raise RuntimeError("Nonfinite learner loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0, error_if_nonfinite=True)
        self.optimizer.step()
        for index, priority in zip(indices, error[demo_count:].detach().cpu().tolist(), strict=True):
            self.priorities[int(index)] = float(priority) + 1e-5
        self.updates += 1
        self.last_loss = float(loss.detach().cpu())
        if self.updates % cfg.target_interval == 0:
            self.target.load_state_dict(self.model.state_dict())

    def summary(self) -> dict:
        """Return compact live evidence without replay pixels or optimizer tensors."""
        return {"updates": self.updates, "segments": self.segments, "transitions": self.seen,
                "protected_rows": len(self.protected), "ordinary_rows": len(self.rows),
                "loss": self.last_loss, "sha256": weights_hash(self.model)}

    def checkpoint(self) -> dict:
        """Retain optimizer/RNG/replay as well as exact model and target weights."""
        return {"config": asdict(self.config), "model": self.model.state_dict(),
                "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "rng": self.rng.bit_generator.state, "protected": self.protected,
                "rows": self.rows, "priorities": self.priorities, "cursor": self.cursor,
                "summary": self.summary()}
