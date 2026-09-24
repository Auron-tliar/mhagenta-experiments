"""One versioned diamond-target treatment for every 2-1-CR run."""

from __future__ import annotations

from hashlib import sha256
import json


PROTOCOL_VERSION = "2-1-cr-diamond-v1"
TARGET_ACHIEVEMENT = "collect_diamond"
ACTION_BUDGET = 1000


def treatment_for_run(run: int) -> dict[str, object]:
    """Return the shared diamond goal and budget for a nonnegative run ID."""

    if run < 0:
        raise ValueError("run must be nonnegative")
    payload = {"target_achievement": TARGET_ACHIEVEMENT, "action_budget": ACTION_BUDGET}
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "treatment_id": PROTOCOL_VERSION,
        "treatment_digest": digest,
        **payload,
    }
