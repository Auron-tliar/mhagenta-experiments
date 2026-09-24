"""Frozen ordinary-from-scratch treatments for Experiment 2-4-CR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


PROTOCOL_VERSION = "2-4-cr-matched-50-budget-v3"
# Same order as the 50 retained 2-3-CR environment treatments.
SEEDS = (
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1018, 1019, 1020, 1021,
    1022, 1023, 1024, 1025, 1027, 1009, 1017, 1026, 1028, 1029,
    1030, 1034, 1035, 1037, 1038, 1039, 1040, 1041, 1042, 1047,
    1061, 1074, 1082, 1084, 1088, 1089, 1096, 1101, 1104, 1105,
)


@dataclass(frozen=True)
class DiamondTreatment:
    """One immutable from-scratch seed and execution-budget assignment."""

    task_id: str
    seed: int
    stratum: str = "ordinary_from_scratch"
    primary_intention: str = "obtain_diamond"
    no_mobs: bool = True
    daylight_effects: bool = False
    duration: int = 900
    total_action_cap: int = 1000

    def as_dict(self) -> dict[str, object]:
        """Return JSON-native treatment data."""

        return asdict(self)


TASKS = tuple(
    DiamondTreatment(task_id=f"2-4-cr-diamond-{index:02d}", seed=seed)
    for index, seed in enumerate(SEEDS, 1)
)


def manifest_digest() -> str:
    """Return the canonical digest of all 50 matched from-scratch treatments."""

    encoded = json.dumps(
        [task.as_dict() for task in TASKS], sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(encoded).hexdigest()


def treatment_for_run(run: int) -> dict[str, object]:
    """Map a run deterministically to one from-scratch diamond treatment."""

    if type(run) is not int or not 0 <= run < len(TASKS):
        raise ValueError("run must be an integer from 0 through 49")
    task = TASKS[run]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "treatment_id": f"{PROTOCOL_VERSION}:{task.task_id}",
        "manifest_digest": manifest_digest(),
        **task.as_dict(),
    }
