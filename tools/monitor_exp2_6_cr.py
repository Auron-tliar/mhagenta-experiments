"""Read compact batch/agent/learner progress for a heartbeat; never launch or stop work."""

import argparse
import json
import math
from pathlib import Path
import time


def read_json(path: Path) -> dict | None:
    """Tolerate a temporarily absent or incomplete autosave without inventing a fault."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def snapshot(root: Path) -> dict:
    """Return only progress counters, freshness, exact container IDs and reported faults."""
    batch = read_json(root / "batch.json")
    if batch is None:
        config = read_json(root / "config.json")
        if config is None:
            return {"batch": str(root), "status": "unavailable"}
        final = read_json(root / "result.json")
        batch = {"run_ids": [config["run"]], "current_run": config["run"], "runs": [],
                 "status": "completed" if final and final["execution_valid"] else "running"}
        if final and not final["execution_valid"]:
            batch.update(status="blocked", error=final["errors"])
        directory = root
    else:
        directory = root / f"run-{batch['current_run']:05d}" if batch.get("current_run") is not None else root
    result = {"batch": str(root), "status": batch["status"], "run": batch.get("current_run"),
              "completed": len(batch.get("runs", [])), "total": len(batch["run_ids"]), "faults": []}
    if batch.get("error"):
        result["faults"].append(batch["error"])
    result["needs_inspection"] = bool(result["faults"])
    if result["run"] is None:
        return result
    config = read_json(directory / "config.json")
    if config is None or "agent_id" not in config:
        result["phase"] = "preparing"
        return result
    agent = directory / config["agent_id"] / "out"
    result["containers"] = [config["agent_id"], config["env_id"]]
    for identifier in result["containers"]:
        log = directory / f"{identifier}.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                tail = stream.read().decode("utf-8", errors="replace").lower()
            if any(marker in tail for marker in ("traceback", "caught exception", "failed to save state")):
                result["faults"].append(f"runtime-log:{identifier}")
    result["needs_inspection"] = bool(result["faults"])
    ll_path = agent / f"{config['agent_id']}.ll_reasoner.json"
    ll = read_json(ll_path)
    if ll is None:
        result["phase"] = "building-or-starting"
        return result
    result.update(phase="executing", actions=ll["actions"], episode=ll["episode"],
                  exploration=ll["exploratory_actions"], model_installs=ll["model_installs"],
                  autosave_age_seconds=round(time.time() - ll_path.stat().st_mtime),
                  waiting_model=ll["waiting_model"])
    progress = read_json(agent / "learner-progress.json")
    if progress:
        result["learning"] = {name: {key: values[key] for key in ("updates", "transitions", "loss")}
                              for name, values in progress.items() if name != "updated_at"}
        for name, values in result["learning"].items():
            if values["loss"] is not None and not math.isfinite(values["loss"]):
                result["faults"].append(f"nonfinite-loss:{name}")
                values["loss"] = str(values["loss"])
    for name in ("perceptor", "actuator", "ll_reasoner", "hl_reasoner", "knowledge", "memory", "learner", "goal_graph"):
        state = read_json(agent / f"{config['agent_id']}.{name}.json")
        if state and state.get("failure"):
            result["faults"].append({"module": name, "error": state["failure"]})
    environment = read_json(directory / config["env_id"] / "out" / f"{config['env_id']}.json")
    if environment and environment.get("failure"):
        result["faults"].append({"module": "environment", "error": environment["failure"]})
    # A stale snapshot requires inspecting processes/logs; it is not sufficient
    # grounds to kill a healthy long checkpoint write or a disconnected host.
    result["needs_inspection"] = bool(result["faults"]) or result["autosave_age_seconds"] > 180
    return result


def main() -> None:
    """Print one bounded JSON snapshot per batch directory supplied by the caller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([snapshot(root) for root in args.batches], allow_nan=False))


if __name__ == "__main__":
    main()
