"""Frozen source-task matching for the fifty single-goal 2-5-BW executions."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mha_env_blocksworld import BlocksWorldEnv
from .contracts import GoalSpec
from .grounding import ground_observation


PROTOCOL = "2-5-bw-matched-2-4-single-goal-v1"
TASK_FILE = Path(__file__).with_name("matched_tasks.json")


def matched_task(run: int) -> tuple[dict, dict]:
    """Reconstruct the saved comparator's initial world and require its exact digest."""
    records = json.loads(TASK_FILE.read_text())
    if type(run) is not int or not 0 <= run < 50 or len(records) != 50:
        raise ValueError("Matched run must be 0..49 within the frozen fifty-task manifest.")
    record = deepcopy(records[run])
    if record["source_run"] != run:
        raise ValueError("Source run numbering differs from the matched manifest.")
    source = record["source_treatment"]
    dimensions = {key: source[key] for key in ("table_len", "num_blocks")}
    env = BlocksWorldEnv(**dimensions, symbolic=False)
    try:
        observation, _ = env.reset(seed=source["seed"])
        facts = sorted(ground_observation(observation, **dimensions).facts)
    finally:
        env.close()
    checksum = hashlib.sha256(json.dumps(facts, separators=(",", ":")).encode()).hexdigest()
    if checksum != source["initial_state_digest"] or GoalSpec(**source["goal"]).fact in facts:
        raise ValueError("The matched initial state changed or already satisfies the goal.")
    task = {"id": f"matched-2-4-{run:05d}", "mode": "transfer", "seed": source["seed"],
            "goal": source["goal"], "case_index": run, "probe": None, "initial_facts": facts,
            "plan": [], "reset_actions": [], "difficulty": 0}
    record["manifest_sha256"] = hashlib.sha256(TASK_FILE.read_bytes()).hexdigest()
    return task, record
