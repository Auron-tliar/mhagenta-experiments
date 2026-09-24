"""Fifty distinct seeded arrangements and goals for experiment 2-3-BW."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Iterable

PROTOCOL_VERSION = "2-3-bw-distinct-seeds-50-v2"


def state_digest(facts: Iterable[str]) -> str:
    """Hash canonical facts independently of observation ordering."""
    return sha256(json.dumps(sorted(set(facts)), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class DeliberativeTask:
    """One seeded world and initially unsatisfied stacking goal."""

    task_id: str
    seed: int
    initial_state_digest: str
    table_len: int
    num_blocks: int
    top: str
    bottom: str

    def as_dict(self) -> dict[str, object]:
        """Return the persisted treatment fields."""
        return asdict(self)


TASKS = tuple(DeliberativeTask(**row) for row in json.loads(
    Path(__file__).with_name("tasks.json").read_text(encoding="utf-8")
))


def manifest_digest() -> str:
    """Hash the frozen task ordering and all treatment fields."""
    return sha256(json.dumps([t.as_dict() for t in TASKS], sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def treatment_for_run(run: int) -> dict[str, object]:
    """Select one task; never recycle tasks outside IDs 0-49."""
    if type(run) is not int or not 0 <= run < len(TASKS):
        raise ValueError("run must be an integer from 0 through 49")
    task = TASKS[run]
    return {"protocol_version": PROTOCOL_VERSION, "run_id": run,
            "treatment_id": f"{PROTOCOL_VERSION}:{task.task_id}",
            "manifest_digest": manifest_digest(), **task.as_dict()}
