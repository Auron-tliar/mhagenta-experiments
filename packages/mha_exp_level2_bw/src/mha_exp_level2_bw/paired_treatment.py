"""Shared planner-qualified treatment for the 2-4/2-5-BW comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal


PROTOCOL_VERSION = "2-4-2-5-bw-paired-v1"
SEED_DISJOINTNESS = "unverifiable"


@dataclass(frozen=True)
class PairedTask:
    """One structurally selected 5×8 task shared by both designs."""

    task_id: str
    seed: int
    initial_state_digest: str
    top: int
    bottom: int
    difficulty: Literal["easy", "medium", "hard"]
    atomic_plan_length: int
    transfer_count: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-native task data."""

        return asdict(self)


TASKS = tuple(PairedTask(*row) for row in (
    ("paired-000", 44000, "a44aa830b76cda7bc89dda9eb75e8acf7e8294871bd1d8e780cd582f4c7c60e1", 1, 3, "easy", 26, 3),
    ("paired-002", 44002, "7e80a5b1d73f0dd47a1201f06d63b152b3b67c37b023b5783ffcbfdbc5571b8e", 6, 7, "easy", 20, 2),
    ("paired-003", 44003, "1d4bc9ffb80d8877e0495202c6cf68586784f3f5df88684dec47319149f09495", 0, 4, "easy", 10, 2),
    ("paired-004", 44004, "bc87119d36365bc95312b3e88657d2fe1bf1dcaa62fe1c7f1ea156f9069f8678", 6, 0, "easy", 20, 2),
    ("paired-001", 44001, "52ab5605987c9220f03ce0cfed3b80d5e8db21ac39a49c1bd8d6e0bb6d96bfb7", 1, 0, "medium", 47, 3),
    ("paired-005", 44005, "24df0bbc72302ed2fbe6eb6d90227d5d772c7de48c5aff3897309cb0b836fbd8", 3, 1, "medium", 45, 6),
    ("paired-006", 44006, "0c71275412b27ede59602d26eedc2579cfdace087768181d0229ec04857ab66c", 3, 2, "medium", 33, 4),
    ("paired-007", 44007, "78e0b09760b943fb73b3e427dddecb52958e94d75b6d5e60635ac2d62fa2a0a1", 0, 2, "medium", 33, 2),
    ("paired-008", 44008, "1854558c6e0d12939b5bda63877eb951ed9a68ac843c6aafd0c26176f38fc4fc", 2, 1, "hard", 76, 3),
    ("paired-009", 44009, "50e23312d820f8cd1969f2670b2f2a7e22cba51dfbfb7e63020844b25b41e63c", 3, 4, "hard", 74, 2),
    ("paired-010", 44010, "406d2fbef972466e7888b239ca8959c9fdcff7885661a4a495e5940895d041ca", 0, 7, "hard", 61, 2),
    ("paired-011", 44011, "875bb37c10f2d973e47c0af57aa4767baeea4d9e4314ee69c76858a35f25b9db", 1, 2, "hard", 73, 4),
))


def manifest_digest() -> str:
    """Return the canonical digest of the paired task manifest."""

    encoded = json.dumps(
        [task.as_dict() for task in TASKS], sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(encoded).hexdigest()


def paired_treatment(run: int, design: Literal["2-4-BW", "2-5-BW"]) -> dict[str, object]:
    """Map a run to the same task while retaining the executing design."""

    if run < 0:
        raise ValueError("run must be nonnegative")
    task = TASKS[run % len(TASKS)]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "treatment_id": f"{PROTOCOL_VERSION}:{task.task_id}",
        "design": design,
        "manifest_digest": manifest_digest(),
        "table_len": 5,
        "num_blocks": 8,
        "goal": {"top": f"b{task.top}", "bottom": f"b{task.bottom}"},
        "seed_disjointness": SEED_DISJOINTNESS if design == "2-5-BW" else "not_applicable",
        **task.as_dict(),
    }
