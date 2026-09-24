"""Planner-reachable policy regressions for experiment 2-5-BW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mha_exp_level2_bw.exp2_5.contracts import TransferSpec

from mha_exp_level2_bw.exp2_5.grounding import NUM_BLOCKS, TABLE_LEN


@dataclass(frozen=True)
class PlannerRegression:
    """A reproducible environment state and transfer that previously failed."""

    name: str
    environment_seed: int
    action_prefix: tuple[int, ...]
    spec: TransferSpec
    input_sha256: str


OSCILLATING_B7_TRANSFER = PlannerRegression(
    name="run0-b7-b5-to-b2",
    environment_seed=1000,
    action_prefix=(
        0, 2, 2, 1, 3, 3, 3, 3, 0, 2, 2, 1, 2, 0, 3, 3, 3, 1, 2, 2, 2,
        0, 2, 1, 3, 0, 3, 1, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 2, 1, 3, 3,
        3, 3, 0, 2, 2, 2, 2, 1, 3, 0, 3, 3, 3, 1, 2, 2, 2, 2, 0, 3, 3,
        3, 3, 1, 2, 2, 2, 2, 0, 3, 3, 1, 2, 2, 0, 3, 3, 1, 2, 2, 0, 3,
        3, 1, 3, 3, 0, 2, 2, 1, 3, 3, 0, 2, 1, 3, 0, 2, 2, 1, 0, 3, 1,
    ),
    spec=TransferSpec(
        block="b7",
        source_support="b5",
        destination_support="b2",
        source="t2",
        destination="t3",
    ),
    input_sha256="aaa74cb7f36c533590b08031fbc1dc71f06e0c47e50a973f95755197176bf08a",
)

PLANNER_REGRESSIONS = (OSCILLATING_B7_TRANSFER,)


def reconstruct_regression(environment: Any, case: PlannerRegression) -> np.ndarray:
    """Replay a regression prefix and return its numeric observation."""

    observation, _ = environment.reset(seed=case.environment_seed)
    for action in case.action_prefix:
        observation, _, terminated, truncated, info = environment.step(action)
        snapshot = info.get("snapshot")
        if snapshot is not None and not bool(snapshot.legal):
            raise RuntimeError(f"Regression {case.name!r} contains an illegal action.")
        if terminated or truncated:
            raise RuntimeError(f"Regression {case.name!r} ended before its target state.")
    array = np.asarray(observation, dtype=np.uint8)
    expected_shape = (NUM_BLOCKS + 2, TABLE_LEN, NUM_BLOCKS)
    if array.shape != expected_shape:
        raise RuntimeError(
            f"Regression {case.name!r} produced {array.shape}, expected {expected_shape}."
        )
    return array
