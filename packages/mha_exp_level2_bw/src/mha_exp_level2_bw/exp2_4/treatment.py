"""Fifty distinct 2-3-BW tasks, retaining the first thirty recorded treatments."""
from hashlib import sha256
import json
from pathlib import Path

PROTOCOL_VERSION = "2-4-bw-distinct-seeds-30-v2"
EXTENSION_PROTOCOL_VERSION = "2-4-bw-additional-seeds-20-v1"
TASKS = tuple(json.loads(Path(__file__).with_name("tasks.json").read_text()))


def treatment_for_run(run: int) -> dict[str, object]:
    """Select a frozen world and goal without cycling through the manifest."""
    if type(run) is not int or not 0 <= run < len(TASKS):
        raise ValueError("run must be an integer from 0 through 49")
    task = TASKS[run]
    # Appending tasks must not relabel the already-collected thirty executions.
    manifest = TASKS[:30] if run < 30 else TASKS[30:]
    protocol = PROTOCOL_VERSION if run < 30 else EXTENSION_PROTOCOL_VERSION
    digest = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**task, "protocol_version": protocol, "design": "2-4-BW",
            "treatment_id": f"{protocol}:{task['task_id']}",
            "manifest_digest": digest, "duration": 1800,
            "goal": {"top": task["top"], "bottom": task["bottom"]}}
