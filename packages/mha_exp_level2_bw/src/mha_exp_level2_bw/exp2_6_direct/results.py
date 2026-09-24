"""Validate and export final hourly BW results without retaining training files."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil

from . import runner, runtime


COMPACT_FORMAT = "bw-hourly-final-v1"
FAULT = re.compile(r"Traceback|PRECONDITION_FAILED|ChannelClosed|ConnectionClosedByBroker|"
                   r"message size.*larger|\[ERROR\]|caught exception", re.IGNORECASE)


def read_json(path: Path) -> dict:
    """Read one saved JSON object without modifying the evidence."""
    return json.loads(path.read_text(encoding="utf-8"))


def run_evidence(directory: Path) -> tuple[dict, dict, dict, list[Path], Path]:
    """Locate the exact final states and logs from the immutable run identity."""
    config = read_json(directory / "config.json")
    identity = runtime.digest({key: value for key, value in config.items()
                               if key not in {"identity", "preparation_seconds"}})
    if config["identity"] != identity:
        raise ValueError(f"Configuration identity changed: {directory.name}")
    short = identity[:12]
    agent, environment = f"exp_direct_bw_{short}", f"exp_direct_env_{short}"
    states, env = runner.read_states(directory, agent, environment)
    out = directory / agent / "out"
    files = [directory / "config.json", directory / "result.json",
             directory / f"{agent}.log", directory / f"{environment}.log",
             directory / environment / "out" / f"{environment}.json"]
    files.extend(out / f"{agent}.{name}.json" for name in states)
    return config, states, env, files, out


def validate_run(directory: Path, *, compact: bool = False) -> dict:
    """Recheck actions, accounting and saved outcomes; never start an agent."""
    config, states, env, files, out = run_evidence(directory)
    receipt_path = directory / "artifact-validation.json"
    receipt = read_json(receipt_path) if compact else None
    result = runner.check_results(states, env, config, out, artifact_validation=receipt)
    if config.get('seed_protocol') == 'thesis-component-seeds-v1':
        run = config['run']
        offsets = {'orchestrator': 100, 'environment': 1000, 'actuator': 10000, 'perceptor': 20000,
                   'llreasoner': 30000, 'knowledge': 40000, 'hlreasoner': 50000,
                   'goalgraph': 60000, 'memory': 70000, 'learner': 80000}
        if (config.get('component_seeds') != {key: offset + run for key, offset in offsets.items()}
                or config['learning']['seed'] != 80000 + run):
            result['failures'].append('thesis-component-seeds')
        for task in config['schedule']:
            offset = 0 if task['mode'] == 'train' else 5000 if task['mode'] == 'demo' else 10000
            if task['seed'] != 1000 + 100 * (offset + task['case_index']) + run:
                result['failures'].append('thesis-environment-seeds')
                break
    for path in files:
        if not path.is_file():
            result["failures"].append(f"missing-final-file:{path.name}")
        elif path.suffix == ".log" and FAULT.search(path.read_text(encoding="utf-8")):
            result["failures"].append(f"error-log:{path.name}")
    metrics = states.get("learner", {}).get("learning_metrics", {})
    if any(not math.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))):
        result["failures"].append("nonfinite-learning-metrics")
    saved = read_json(directory / "result.json")
    if {key: value for key, value in saved.items() if key != "wall_seconds"} != result:
        result["failures"].append("saved-result-disagrees")
    if compact:
        expected = {path.relative_to(directory) for path in [*files, receipt_path]}
        # The preserved historical matched cohort also contains its first native recording.
        # New-family exports keep the strict evidence-only contract.
        recording = Path(f"exp_direct_env_{config['identity'][:12]}") / "out" / "0000.mp4"
        if config.get("matched_2_4") and not config.get("policy_family") and (directory / recording).is_file():
            expected.add(recording)
        actual = {path.relative_to(directory) for path in directory.rglob("*") if path.is_file()}
        if actual != expected:
            result["failures"].append("unexpected-compact-files")
    result["failures"] = sorted(set(result["failures"]))
    result["operationally_valid"] = not result["failures"]
    return result


def process_batch(output: Path, run_ids: list[int]) -> None:
    """Validate selected saved runs and raise on any failure, including omissions."""
    batch = read_json(output / "batch.json")
    entries = {item["run"]: item for item in batch["runs"]}
    if (batch["status"] != "completed" or len(entries) != len(batch["runs"])
            or set(entries) != set(batch["run_ids"]) or not run_ids
            or len(set(run_ids)) != len(run_ids) or not set(run_ids) <= set(entries)):
        raise ValueError("Incomplete batch or invalid requested run selection.")
    compact = batch.get("results_format") == COMPACT_FORMAT
    failed = []
    for index in run_ids:
        directory = output / f"run-{index:05d}"
        try:
            result = validate_run(directory, compact=compact)
            config = read_json(directory / "config.json")
            if (config["run"] != index or config.get("frozen_policy") != batch["frozen_policy"]
                    or config.get("policy_family") != batch.get("policy_family")
                    or config["profile"]["duration_seconds"] != batch["duration_seconds"]
                    or config["sources"] != batch.get('run_source_identities', {}).get(str(index), batch["source_identity"])
                    or entries[index]["result"] != read_json(directory / "result.json")):
                result["failures"].append("batch-run-evidence-disagrees")
                result["operationally_valid"] = False
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {"operationally_valid": False, "failures": [str(error)]}
        print(json.dumps({"run": index, "operationally_valid": result["operationally_valid"],
                          "failures": result["failures"], "production": result.get("production")}), flush=True)
        if not result["operationally_valid"]:
            failed.append(index)
    if failed:
        raise ValueError(f"Saved-results validation failed for runs {failed}")
    print(f"Validated {len(run_ids)}/{len(run_ids)} saved runs; no execution launched.", flush=True)


def export_batch(source: Path, destination: Path) -> None:
    """Copy only final evidence after validating every original artifact.

    Originals remain in the evidence archive. Checkpoints and per-episode
    training files are represented by a receipt bound to all retained states.
    """
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace results: {destination}")
    batch = read_json(source / "batch.json")
    if batch["status"] != "completed" or batch.get("results_format"):
        raise ValueError("Export requires a completed original batch.")
    if [item["run"] for item in batch["runs"]] != batch["run_ids"]:
        raise ValueError("Batch run index is incomplete or inconsistent.")
    destination.mkdir(parents=True)
    for entry in batch["runs"]:
        directory = source / f"run-{entry['run']:05d}"
        result = validate_run(directory)
        if not result["operationally_valid"]:
            raise ValueError(f"Cannot export run {entry['run']}: {result['failures']}")
        config, states, env, files, _ = run_evidence(directory)
        target = destination / directory.name
        for path in files:
            copied = target / path.relative_to(directory)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copied)
            if runner.sha(path) != runner.sha(copied):
                raise ValueError(f"Copy checksum mismatch: {copied}")
        (target / "artifact-validation.json").write_text(
            json.dumps(runner.artifact_receipt(states, env, config), indent=2) + "\n", encoding="utf-8")
        print(f"Exported fully validated run {entry['run']} to {target}", flush=True)
    exported = deepcopy(batch)
    exported["results_format"] = COMPACT_FORMAT
    for entry in exported["runs"]:
        entry["directory"] = f"run-{entry['run']:05d}"
    (destination / "batch.json").write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Export an original batch into a new compact final-results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_batch(args.source, args.destination)


if __name__ == "__main__":
    main()
