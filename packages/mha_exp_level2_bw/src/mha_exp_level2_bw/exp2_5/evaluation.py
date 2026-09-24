"""Deterministic greedy rollout for the frozen 2-5 transfer policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import TransferSpec
from .grounding import NUM_BLOCKS, TABLE_LEN, ground_observation, transfer_succeeded
from .policy import MAX_POLICY_STEPS, greedy_inference


@dataclass(frozen=True)
class EvaluationResult:
    """JSON-safe result of one greedy transfer rollout."""
    environment_seed: int
    goal: dict[str, str]
    outcome: str
    success: bool
    actions: list[int]
    illegal_action: bool
    episode_length: int


def evaluate_transfer(
    torch_module: Any,
    model: Any,
    environment: Any,
    observation: np.ndarray,
    spec: TransferSpec,
    seed: int,
    *, table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> tuple[EvaluationResult, np.ndarray]:
    """Run one transfer greedily from the environment's current state."""
    actions: list[int] = []
    illegal = False
    for _ in range(MAX_POLICY_STEPS):
        action, _, _ = greedy_inference(
            torch_module, model, observation, spec, table_len=table_len, num_blocks=num_blocks,
        )
        actions.append(action)
        observation, _, _, _, info = environment.step(action)
        snapshot = info.get("snapshot")
        illegal = snapshot is not None and not bool(snapshot.legal)
        grounded = ground_observation(observation, table_len=table_len, num_blocks=num_blocks)
        if transfer_succeeded(grounded.facts, spec):
            return EvaluationResult(
                seed, spec.as_dict(), "succeeded", True, actions, illegal, len(actions),
            ), grounded.observation
        if illegal:
            break
    return EvaluationResult(
        seed,
        spec.as_dict(),
        "illegal-action" if illegal else "step-limit",
        False,
        actions,
        illegal,
        len(actions),
    ), ground_observation(observation, table_len=table_len, num_blocks=num_blocks).observation
