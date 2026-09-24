"""Orchestration and result gathering for Experiment 2-4-CR."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
from typing import Any, cast

from mhagenta import Orchestrator

import mha_exp_common
from mha_exp_common.batch import (
    cleanup_run_containers, cleanup_run_images, normalize_runs,
    run_batch as run_experiment_batch,
)
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)
from mha_exp_common.utils import agent_name, env_name, gather_states, module_name

from .reporting import (
    PROVENANCE_FILENAME,
    PROVENANCE_PROTOCOL_VERSION,
    process_execution_metrics,
)
from .treatment import treatment_for_run

from .agent import (
    ACTIVITY_ACTION_LIMITS,
    ENVIRONMENT_OVERRUN,
    MAX_EPISODE_LEN,
    RECORD,
    STARTUP_DELAY,
    ActivityGoalGraph,
    ActivityLLReasoner,
    CrafterHybridActuator,
    CrafterHybridEnvironment,
    CrafterHybridPerceptor,
    ForwardingKnowledge,
    HybridSymbolicReasoner,
    environment_initial_state,
    initial_states,
)
from .checking import check_results


REQUIRED_MHAGENTA_VERSION = "1.4.12"
VERBOSE = True


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one required source file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(paths: Sequence[Path]) -> str:
    """Hash runtime source bundles deterministically by logical path and bytes."""

    excluded_names = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    files: list[tuple[str, Path]] = []
    for source_index, source in enumerate(paths):
        if source.is_file():
            files.append((f"{source_index}/{source.name}", source))
            continue
        for path in source.rglob("*"):
            if (
                path.is_file()
                and not any(part in excluded_names for part in path.parts)
                and path.suffix not in {".pyc", ".pyo"}
            ):
                files.append((f"{source_index}/{path.relative_to(source).as_posix()}", path))
    digest = hashlib.sha256()
    for logical_path, path in sorted(files):
        digest.update(logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(root: Path, cohort_root: Path | None = None) -> tuple[str, bool]:
    """Return the commit and dirty flag, excluding generated cohort artifacts."""

    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    commit = subprocess.run(
        [*command, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        [*command, "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if cohort_root is not None and cohort_root.is_relative_to(root):
        cohort_relative = cohort_root.relative_to(root).as_posix().rstrip("/")
        status = [
            row for row in status
            if not row[3:].replace("\\", "/").strip('"').startswith(f"{cohort_relative}/")
            and row[3:].replace("\\", "/").strip('"') != cohort_relative
        ]
    return commit, bool(status)


def _build_execution_provenance(
    workspace_root: Path,
    mha_root: Path,
    crafter_source: Path,
    common_source: Path,
    cohort_root: Path,
) -> dict[str, Any]:
    """Build the exact source identity shared by every execution in a cohort."""

    workspace_commit, workspace_dirty = _git_state(workspace_root, cohort_root)
    mha_commit, mha_dirty = _git_state(mha_root)
    experiment_source = Path(__file__).resolve().parent
    experiment_files = (
        *sorted(experiment_source.glob("*.py")),
        experiment_source / "requirements-env.txt",
    )
    payload: dict[str, Any] = {
        "protocol_version": PROVENANCE_PROTOCOL_VERSION,
        "workspace_git_commit": workspace_commit,
        "workspace_git_dirty": workspace_dirty,
        "mhagenta_git_commit": mha_commit,
        "mhagenta_git_dirty": mha_dirty,
        "mhagenta_version": REQUIRED_MHAGENTA_VERSION,
        "experiment_source_sha256": _source_digest((*experiment_files, common_source)),
        "crafter_source_sha256": _source_digest((crafter_source,)),
        "mhagenta_source_sha256": _source_digest(
            (mha_root / "mhagenta", mha_root / "pyproject.toml")
        ),
        "uv_lock_sha256": _sha256_file(workspace_root / "uv.lock"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["provenance_id"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _ensure_execution_provenance(cohort_root: Path, provenance: dict[str, Any]) -> None:
    """Write the cohort provenance once and reject later source mismatches."""

    cohort_root.mkdir(parents=True, exist_ok=True)
    path = cohort_root / PROVENANCE_FILENAME
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Malformed {PROVENANCE_FILENAME}") from error
        if existing != provenance:
            # Commits elsewhere in the shared workspace do not change runtime sources.
            # Retain the starting metadata, but still compare every content digest.
            current_metadata = dict(existing)
            for key in ("workspace_git_commit", "workspace_git_dirty",
                        "mhagenta_git_commit", "mhagenta_git_dirty"):
                if key in provenance:
                    current_metadata[key] = provenance[key]
            payload = {key: value for key, value in current_metadata.items()
                       if key != "provenance_id"}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            current_metadata["provenance_id"] = hashlib.sha256(canonical).hexdigest()
            if current_metadata != provenance:
                raise RuntimeError("Execution provenance differs from the existing cohort manifest")
        return
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _workspace_root() -> Path:
    """Locate the uv workspace containing this experiment package."""

    for candidate in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (candidate, *candidate.parents):
            pyproject = parent / "pyproject.toml"
            if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(
                encoding="utf-8"
            ):
                return parent
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root")


def _read_logs(paths: dict[str, Path]) -> dict[str, list[str]] | None:
    logs: dict[str, list[str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            print(f"Missing {name} runtime log: {path}")
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            print(f"Empty {name} runtime log: {path}")
            return None
        logs[name] = lines
    return logs


def _readable_video(output_dir: Path, relative_path: Any) -> bool:
    """Return whether an in-run MP4 is nonempty and yields a decoded frame."""

    import imageio

    if not isinstance(relative_path, str) or not relative_path:
        return False
    root = output_dir.resolve()
    candidate = (root / relative_path).resolve()
    if (
        Path(relative_path).is_absolute()
        or not candidate.is_relative_to(root)
        or candidate.suffix.lower() != ".mp4"
        or not candidate.is_file()
        or candidate.stat().st_size < 1
    ):
        return False
    reader = None
    try:
        reader = imageio.get_reader(candidate)
        frame = reader.get_data(0)
        return getattr(frame, "size", 0) > 0
    except (OSError, RuntimeError, ValueError):
        return False
    finally:
        if reader is not None:
            reader.close()


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Build and execute one deterministic 2-4-CR run."""

    version = DEFAULT_MHAGENTA_VERSION if not mha_version or mha_version == "latest" else mha_version
    if version != REQUIRED_MHAGENTA_VERSION:
        raise RuntimeError(
            f"Experiment 2-4-CR requires MHAgentA {REQUIRED_MHAGENTA_VERSION}, got {version!r}"
        )
    exp_path = Path(exp_path).resolve()
    workspace_root = _workspace_root()
    mha_root = (workspace_root.parent / "mhagenta").resolve()
    version_file = mha_root / "pyproject.toml"
    marker = f'version = "{REQUIRED_MHAGENTA_VERSION}"'
    if not version_file.is_file() or marker not in version_file.read_text(encoding="utf-8"):
        raise RuntimeError(
            f"Expected local MHAgentA {REQUIRED_MHAGENTA_VERSION} checkout at {mha_root}"
        )
    crafter_spec = find_spec("mha_env_crafter")
    if crafter_spec is None or crafter_spec.origin is None:
        raise ImportError("Could not locate mha_env_crafter runtime sources")
    crafter_source = Path(crafter_spec.origin).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    provenance = _build_execution_provenance(
        workspace_root, mha_root, crafter_source, common_source, exp_path
    )
    _ensure_execution_provenance(exp_path, provenance)
    treatment = treatment_for_run(run)
    exchange = "mhagenta"
    agent_id = agent_name(run, "2_4")
    environment_id = env_name(run, "2_4")
    duration = float(treatment["duration"])
    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=0.0,
        control_frequency=0.0,
        status_frequency=5.0,
        agent_start_delay=STARTUP_DELAY,
        exec_duration=duration,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        no_stdout_logs=False,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange,
        state_autosave_interval=30,
        stop_on_agents_term=True,
    )
    initial = initial_states(treatment)
    orchestrator.add_agent(
        agent_id=agent_id,
        perceptors=CrafterHybridPerceptor(
            module_id=module_name(PERCEPTOR, 0),
            initial_state=initial[PERCEPTOR],
            exchange_name=exchange,
        ),
        actuators=CrafterHybridActuator(
            module_id=module_name(ACTUATOR, 0),
            initial_state=initial[ACTUATOR],
            exchange_name=exchange,
        ),
        ll_reasoners=ActivityLLReasoner(
            module_id=module_name(LLREASONER, 0), initial_state=initial[LLREASONER],
            init_kwargs={
                "activity_action_limits": dict(ACTIVITY_ACTION_LIMITS),
                "total_action_limit": treatment["total_action_cap"],
            },
        ),
        knowledge=ForwardingKnowledge(
            module_id=module_name(KNOWLEDGE, 0), initial_state=initial[KNOWLEDGE]
        ),
        hl_reasoners=HybridSymbolicReasoner(
            module_id=module_name(HLREASONER, 0), initial_state=initial[HLREASONER],
        ),
        goal_graphs=ActivityGoalGraph(
            module_id=module_name(GOALGRAPH, 0), initial_state=initial[GOALGRAPH]
        ),
        extra_runtime_sources=common_source,
    )
    record = RECORD == "all" or (RECORD == "first" and run == 0)
    environment_state = environment_initial_state(treatment)
    environment_state.update(
        {
            "seed": treatment["seed"],
            "record": record,
            "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}",
            "expected_agent_id": agent_id,
            "no_mobs": treatment["no_mobs"],
            "daylight_effects": treatment["daylight_effects"],
            "episode_length": MAX_EPISODE_LEN,
        }
    )
    orchestrator.add_environment(
        base=CrafterHybridEnvironment(environment_state),
        env_id=environment_id,
        exec_duration=duration + ENVIRONMENT_OVERRUN,
        requirements_path=Path(__file__).with_name("requirements-env.txt"),
        exchange_name=exchange,
        extra_runtime_sources=[common_source, crafter_source],
    )
    cleanup_run_containers(agent_id, environment_id, phase="before")
    cleanup_run_images(agent_id, environment_id, phase="before")
    try:
        orchestrator.run(
            mhagenta_version=REQUIRED_MHAGENTA_VERSION,
            local_build=mha_root,
            force_run=True,
        )
    finally:
        cleanup_run_containers(agent_id, environment_id, phase="after")
        cleanup_run_images(agent_id, environment_id, phase="after")

    states = {
        **gather_states(exp_path / agent_id, True, no_warnings=True),
        **gather_states(exp_path / environment_id, True, no_warnings=True),
    }
    logs = _read_logs(
        {
            "agent": (exp_path / agent_id).with_suffix(".log"),
            "environment": (exp_path / environment_id).with_suffix(".log"),
        }
    )
    if logs is None or agent_id not in states or environment_id not in states:
        return False
    environment = states[environment_id][environment_id]
    result = check_results(
        states[agent_id],
        environment,
        logs,
        expected_recording=record,
        verbose=VERBOSE,
    )
    if record:
        result = result and _readable_video(
            exp_path / environment_id / Orchestrator.SAVE_SUBDIR,
            environment.get("video_path"),
        )
    print(f"Results: {result}")
    return result


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 50,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    process_only: bool = False,
) -> None:
    """Run the experiment batch or process its existing results."""

    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [{"execution_id": f"run-{run}", "run_id": run,
                 "factors": treatment_for_run(run)}
                for run in run_ids]
    primary_error: BaseException | None = None
    try:
        run_experiment_batch(
            experiment_id="2-4-CR", title="HYBRID SYMBOLIC CRAFTER",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=run_experiment, process_only=process_only,
            cleanup_before_run=False, stop_on_error=True,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            process_execution_metrics(root, expected_executions=expected)
        except Exception as reporting_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Execution-metrics processing also failed: {type(reporting_error).__name__}"
            )


__all__ = ["run_batch", "run_experiment"]
