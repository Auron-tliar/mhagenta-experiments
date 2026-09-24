"""Frozen full-domain seed strata for experiment 2-3-CR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal


PROTOCOL_VERSION = "2-3-cr-full-domain-diamond-50-v1"
ACTION_BUDGET = 1_000
EPISODE_LENGTH = 1_000
InitialSupport = Literal["wood-supported", "exploration-first"]


@dataclass(frozen=True)
class DiamondTreatment:
    """One ordinary-start Crafter seed and its initial knowledge stratum."""

    task_id: str
    seed: int
    initial_support: InitialSupport
    action_budget: int = ACTION_BUDGET
    episode_length: int = EPISODE_LENGTH

    def as_dict(self) -> dict[str, object]:
        """Return JSON-native treatment data."""

        return asdict(self)


_WOOD_SUPPORTED_SEEDS = (
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1018, 1019, 1020, 1021,
    1022, 1023, 1024, 1025, 1027,
)
_EXPLORATION_FIRST_SEEDS = (
    1009, 1017, 1026, 1028, 1029, 1030, 1034, 1035, 1037, 1038,
    1039, 1040, 1041, 1042, 1047, 1061, 1074, 1082, 1084, 1088,
    1089, 1096, 1101, 1104, 1105,
)
_TASK_ROWS: tuple[tuple[str, int, InitialSupport], ...] = tuple(
    (f"2-3-cr-wood-supported-{index:02d}", seed, "wood-supported")
    for index, seed in enumerate(_WOOD_SUPPORTED_SEEDS, start=1)
) + tuple(
    (f"2-3-cr-exploration-first-{index:02d}", seed, "exploration-first")
    for index, seed in enumerate(_EXPLORATION_FIRST_SEEDS, start=1)
)
TASKS = tuple(
    DiamondTreatment(*row)
    for row in _TASK_ROWS
)


def manifest_digest() -> str:
    """Return the canonical seed-manifest digest."""

    encoded = json.dumps(
        [task.as_dict() for task in TASKS], sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(encoded).hexdigest()


def treatment_for_run(run: int) -> dict[str, object]:
    """Return the frozen treatment for a run index from zero through forty-nine."""

    if type(run) is not int or run < 0 or run >= len(TASKS):
        raise ValueError(f"run must be an integer from 0 through {len(TASKS) - 1}")
    task = TASKS[run]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "treatment_id": f"{PROTOCOL_VERSION}:{task.task_id}",
        "manifest_digest": manifest_digest(),
        **task.as_dict(),
    }


__all__ = [
    "ACTION_BUDGET",
    "DiamondTreatment",
    "EPISODE_LENGTH",
    "PROTOCOL_VERSION",
    "TASKS",
    "manifest_digest",
    "treatment_for_run",
]
