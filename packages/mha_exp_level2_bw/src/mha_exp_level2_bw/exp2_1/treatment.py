"""Run-indexed one-goal treatment for experiment 2-1-BW."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal


PROTOCOL_VERSION = "2-1-bw-run-indexed-one-goal-v2"


def state_digest(facts: Iterable[str]) -> str:
    """Hash symbolic facts independently of observation ordering."""

    payload = json.dumps(sorted(set(facts)), separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReactiveTask:
    """One seeded world and its deterministic, initially unsatisfied goal."""

    task_id: str
    run_id: int
    seed: int
    reasoner_seed: int
    initial_state_digest: str
    table_len: int
    num_blocks: int
    top: str
    bottom: str
    arm_distance: int
    horizontal_distance: int
    obstructions: int
    difficulty: Literal["easy", "medium", "hard"]

    def as_dict(self) -> dict[str, object]:
        """Return the persisted treatment fields."""

        return asdict(self)


TASKS = tuple(ReactiveTask(**row) for row in json.loads(
    Path(__file__).with_name("tasks.json").read_text(encoding="utf-8")
))


def manifest_digest() -> str:
    """Hash the frozen run ordering and all treatment fields."""

    payload = [task.as_dict() for task in TASKS]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def treatment_for_run(run: int) -> dict[str, object]:
    """Return the unique one-goal treatment assigned to run IDs 0 through 49."""

    if type(run) is not int or not 0 <= run < len(TASKS):
        raise ValueError("run must be an integer from 0 through 49")
    task = TASKS[run]
    task_data = task.as_dict()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "treatment_id": f"{PROTOCOL_VERSION}:{task.task_id}",
        "manifest_digest": manifest_digest(),
        "task_count": 1,
        "max_episode_actions": 250,
        **task_data,
        "tasks": [task_data],
    }
