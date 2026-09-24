"""CLI collection of paired, duration-limited frozen and learning BW agents."""

import json
from pathlib import Path
import time
from typing import Sequence

from mha_exp_common.batch import normalize_runs
from mha_exp_common.defaults import DEFAULT_TORCH_MHAGENTA_VERSION
from . import runner


def write_status(path: Path, value: dict) -> None:
    """Atomically publish progress without discarding previous run evidence."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_batch(runs: int | tuple[int, int] | Sequence[int] | None = None,
              exp_path: str | Path = ".", mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
              process_only: bool = False, *, duration_seconds: int = 3600,
              pretrained_run: str | Path = "/workspace/results/achieve-on-v2-search-a1",
              assessment: str | Path = "/workspace/results/achieve-on-v2-assessment-a1",
              frozen_policy: str | None = None, device: str = "cuda", gpu_id: int = 0,
              episode_limit: int = 5000, demonstrations: int = 60,
              warm_updates: int = 600, probe_cases: int = 100,
              probe_interval: int = 100, updates_per_episode: int = 32,
              matched_2_4: bool = False, policy_family: str | None = None) -> None:
    """Collect each requested run once, retaining failed attempts and a batch index.

    Scientific failure does not prevent subsequent runs. Existing outputs are
    never overwritten, and source identity is frozen across the entire batch.
    The six-module frozen treatment and eight-module learner share ordinary
    case seeds/goals; learning/probe overhead counts against the latter's hour.
    """
    output = Path(exp_path).resolve()
    if process_only:
        from .results import process_batch
        ids = json.loads((output / "batch.json").read_text())["run_ids"] if runs is None else list(normalize_runs(runs)[0])
        process_batch(output, ids)
        return
    runs = 10 if runs is None else runs
    ids = list(normalize_runs(runs)[0])
    if (not ids or len(set(ids)) != len(ids) or any(not 0 <= index < 50 for index in ids)
            or not 60 <= duration_seconds <= 14400 or not 1 <= episode_limit <= 10000
            or frozen_policy not in {None, "transfer", "achieve_on"}
            or not 1 <= demonstrations <= 1000 or warm_updates < 1
            or not 1 <= probe_cases <= 1000 or probe_interval < 1 or updates_per_episode < 1):
        raise ValueError("Invalid run selection or duration/learning budget.")
    if mha_version != DEFAULT_TORCH_MHAGENTA_VERSION:
        raise ValueError("This collection requires the validated MHAgentA Torch image.")
    if matched_2_4 and (frozen_policy != "transfer" or episode_limit != 1):
        raise ValueError("Matched 2-4 tasks require frozen Transfer and episode_limit=1.")
    if policy_family is not None and not matched_2_4:
        raise ValueError("Explicit policy families are restricted to matched frozen Transfer runs.")
    if output.exists():
        raise FileExistsError(f"Refusing to replace batch output: {output}")
    profile = runner.Profile(0 if frozen_policy else demonstrations, episode_limit,
                             0 if frozen_policy else warm_updates,
                             0 if frozen_policy else updates_per_episode,
                             0 if frozen_policy else probe_cases, probe_interval, duration_seconds,
                             action_cap=None if matched_2_4 else 64)
    source = runner.source_identity()
    output.mkdir(parents=True)
    status = {"status": "running", "run_ids": ids, "frozen_policy": frozen_policy,
              "duration_seconds": duration_seconds, "started_at": time.time(), "runs": [],
              "source_identity": source, "current_run": None}
    if matched_2_4:
        status["matched_2_4"] = True
    if policy_family is not None:
        status["policy_family"] = policy_family
    write_status(output / "batch.json", status)
    for index in ids:
        if runner.source_identity() != source:
            status.update(status="blocked-source-change", current_run=index)
            write_status(output / "batch.json", status)
            raise RuntimeError("Sources changed during collection; remaining runs were not started.")
        directory = output / f"run-{index:05d}"
        status["current_run"] = index
        write_status(output / "batch.json", status)
        started = time.time()
        try:
            config = runner.prepare(Path(pretrained_run), Path(assessment), directory, index, "hourly",
                                    "cpu" if frozen_policy else device, gpu_id, True,
                                    profile_override=profile, frozen_policy=frozen_policy, matched_2_4=matched_2_4,
                                    policy_family=policy_family)
            result = runner.execute(config, directory)
        except Exception as error:
            result = {"operationally_valid": False, "outcome": "execution-error",
                      "failures": [f"{type(error).__name__}: {error}"]}
            directory.mkdir(exist_ok=True)
            write_status(directory / "batch-error.json", result)
        status["runs"].append({"run": index, "started_at": started, "finished_at": time.time(),
                               "result": result, "directory": str(directory)})
        write_status(output / "batch.json", status)
        print(json.dumps({"run": index, "result": result}), flush=True)
        if (matched_2_4 or duration_seconds > 3600) and not result["operationally_valid"]:
            status.update(status="blocked-operational-failure", current_run=index)
            write_status(output / "batch.json", status)
            raise RuntimeError(f"Run {index} failed validation; remaining runs were not started.")
    status.update(status="completed", current_run=None, finished_at=time.time(),
                  operationally_valid_runs=sum(item["result"]["operationally_valid"] for item in status["runs"]))
    write_status(output / "batch.json", status)
