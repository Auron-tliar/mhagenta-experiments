import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

RETRIES = 1  # 3


RunSpec = int | tuple[int, int] | Sequence[int]
ExperimentRunner = Callable[[int, Path, str], bool]


class NonRetryableBatchError(RuntimeError):
    """Mark a failure for which starting another run is unsafe."""


class DockerCleanupError(NonRetryableBatchError):
    """Report a failed exact Docker cleanup operation."""

    def __init__(self, stage: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(f"exact Docker cleanup failed during {stage}")
        self.stage = stage
        self.evidence = dict(evidence)


def _docker_ids(*args: str) -> list[str]:
    result = subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def cleanup_exp_docker(prefix: str = "exp_") -> None:
    containers = _docker_ids(
        "ps", "-a",
        "--format", "{{.ID}} {{.Names}}",
    )
    container_ids = [
        line.split(maxsplit=1)[0]
        for line in containers
        if len(line.split(maxsplit=1)) == 2
        and line.split(maxsplit=1)[1].startswith(prefix)
    ]

    if container_ids:
        subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            check=False,
            stdout=subprocess.DEVNULL
        )

    images = _docker_ids(
        "images",
        "--format", "{{.ID}} {{.Repository}}:{{.Tag}}",
    )
    image_ids = [
        line.split(maxsplit=1)[0]
        for line in images
        if len(line.split(maxsplit=1)) == 2
        and line.split(maxsplit=1)[1].split(":", 1)[-1].startswith(prefix)
    ]

    if image_ids:
        subprocess.run(
            ["docker", "rmi", "-f", *image_ids],
            check=False,
            stdout=subprocess.DEVNULL
        )

    subprocess.run(
        ["docker", "builder", "prune", "--force"],
        check=False,
        stdout=subprocess.DEVNULL
    )


def _exact_docker_cleanup(
        *,
        resource: str,
        requested: Sequence[str],
        phase: str,
) -> dict[str, Any]:
    command = "ps" if resource == "containers" else "images"
    format_value = "{{.Names}}" if resource == "containers" else "{{.Repository}}:{{.Tag}}"
    evidence: dict[str, Any] = {
        "phase": phase,
        "resource": resource,
        "status": "failed",
        "requested": list(requested),
        "resolved": [],
        "verified_absent": [],
        "failure": None,
    }

    def query() -> set[str]:
        result = subprocess.run(
            ["docker", command, "-a", "--format", format_value]
            if resource == "containers"
            else ["docker", command, "--format", format_value],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("query")
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    try:
        existing = query()
        resolved = [target for target in requested if target in existing]
        evidence["resolved"] = resolved
        if resolved:
            remove = subprocess.run(
                [
                    "docker",
                    "container" if resource == "containers" else "image",
                    "rm",
                    "--force",
                    *resolved,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if remove.returncode != 0:
                raise RuntimeError("remove")
        remaining = query()
        present = [target for target in requested if target in remaining]
        evidence["verified_absent"] = [
            target for target in requested if target not in remaining
        ]
        if present:
            raise RuntimeError("verify")
        evidence["status"] = "succeeded"
        return evidence
    except BaseException as error:
        if not isinstance(error, Exception):
            try:
                remaining = query()
                present = [target for target in requested if target in remaining]
                evidence["verified_absent"] = [
                    target for target in requested if target not in remaining
                ]
            except BaseException as verification_error:  # noqa: BLE001
                evidence["failure"] = {
                    "type": type(error).__name__,
                    "verification_type": type(verification_error).__name__,
                }
            else:
                if not present:
                    evidence["status"] = "succeeded"
                    raise
        if evidence["failure"] is None:
            evidence["failure"] = {"type": type(error).__name__}
        raise DockerCleanupError(f"{phase}_{resource}", evidence) from error


def cleanup_run_containers(
        agent_id: str,
        environment_id: str,
        *,
        phase: str,
) -> dict[str, Any]:
    """Remove and verify absence of one run's exact containers."""

    return _exact_docker_cleanup(
        resource="containers",
        requested=(agent_id, environment_id),
        phase=phase,
    )


def cleanup_run_images(
        agent_id: str,
        environment_id: str,
        *,
        phase: str,
) -> dict[str, Any]:
    """Remove and verify absence of one run's exact runtime images."""

    return _exact_docker_cleanup(
        resource="images",
        requested=(f"mhagent:{agent_id}", f"mhagent-env:{environment_id}"),
        phase=phase,
    )


def normalize_runs(runs: RunSpec) -> tuple[Sequence[int], int]:
    if isinstance(runs, int):
        return range(runs), runs
    if isinstance(runs, tuple) and len(runs) == 2:
        return range(*runs), runs[1]
    return runs, len(runs)


def run_batch(
        *,
        experiment_id: str,
        title: str,
        runs: RunSpec = 50,
        exp_path: str | PathLike[str] = '.',
        mha_version: str = 'latest',
        runner: ExperimentRunner,
        process_only: bool = False,
        cleanup_before_run: bool = True,
        stop_on_error: bool = False,
) -> bool:
    """Run a batch or make its existing output available for processing.

    Set ``cleanup_before_run`` to false when the experiment runner owns exact,
    per-run Docker cleanup. The default preserves the historical broad cleanup.
    Set ``stop_on_error`` to stop on exceptions or a failed result check.
    """

    print('===========================================\n'
          f'RUNNING EXPERIMENT {experiment_id} BATCH: {title}\n'
          '===========================================\n')

    normalized_runs, n_runs = normalize_runs(runs)
    path = Path(exp_path)
    if process_only:
        if not path.is_dir() or not any(path.iterdir()):
            print(f'No existing results found at {path}; nothing to process.')
            return False
        print(f'Using existing results at {path}; execution skipped.')
        return True

    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    n_success = 0
    for r in normalized_runs:
        print(f'>>> Experiment {experiment_id}, run {r + 1}/{n_runs}: running...')
        for i in range(RETRIES):
            try:
                if cleanup_before_run:
                    cleanup_exp_docker(prefix="exp_")
                status = runner(r, path, mha_version)
                if not status and stop_on_error:
                    raise RuntimeError(f'Experiment {experiment_id}, run {r}: result check failed')
                n_success += 1 if status else 0
                status_msg = '\033[32mSUCCESS\033[0m' if status else '\033[31mFAIL\033[0m'
                print(f'\r\033[2K>>> Experiment {experiment_id}, run {r + 1}/{n_runs}: done! Status: {status_msg}')
                break
            except NonRetryableBatchError as e:
                print(f'\r\033[2K>>> Experiment {experiment_id}, run {r + 1}/{n_runs}: encountered an exception! Reason: {e}')
                print('\r\033[2K>>> Unsafe cleanup state detected; stopping the batch.')
                raise
            except Exception as e:
                print(f'\r\033[2K>>> Experiment {experiment_id}, run {r + 1}/{n_runs}: encountered an exception! Reason: {e}')
                if i < RETRIES - 1:
                    print('\r\033[2K>>> Retrying...')
                else:
                    print('\r\033[2K>>> Maximal number of retries reached, assigning status to \033[31mFAIL\033[0m...')
                    if stop_on_error:
                        print('\r\033[2K>>> Stopping the batch after the failed run.')
                        raise
        print('-------------------------------------------')

    print('===========================================\n'
          f'{"EXPERIMENT " + experiment_id + " BATCH FINISHED":^43}\n'
          f'{str(n_success) + " / " + str(len(normalized_runs)) + " SUCCESSFUL RUNS":^43}\n'
          '===========================================\n')
    return True
