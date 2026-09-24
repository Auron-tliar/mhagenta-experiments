"""Orchestration and scientific acceptance for experiment 2-5-BW."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
import math
import json
import os
from pathlib import Path
import re
from typing import Any, Literal, cast

import mhagenta
from mhagenta import Orchestrator

import mha_exp_common
from mha_exp_common.batch import normalize_runs, run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION, DEFAULT_TORCH_MHAGENTA_VERSION
from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name

from .contracts import format_fact
from .environment import NumericBlocksWorldEnvironment, environment_initial_state
from .grounding import NUM_BLOCKS, TABLE_LEN
from .policy import N_ACTIONS, artifact_paths, load_policy_checkpoint, validate_manifest
from .reporting import process_execution_metrics
from ..paired_treatment import paired_treatment
from .runtime import (
    BlocksWorldActuator,
    BlocksWorldPerceptor,
    ClosedWorldKnowledge,
    NeuralTransferLLReasoner,
    RepeatedGoalHLReasoner,
    TransferGoalGraph,
    initial_states,
)


DURATION = 120.0
ENVIRONMENT_DURATION = 150.0
GOAL_COMPLETION_LIMIT = 5
PLANNER_TIMEOUT = 15.0
RECORD: Literal["all", "first", "none"] = "first"
VERBOSE = True
LOG_PATTERN = re.compile(
    r"^\[(?P<time>[^\]]+)\]\[(?P<level>[^\]]+)\]::"
    r"\[(?P<sender>[^\]]+)\]::(?P<message>.*)$"
)
ATOMIC_ID_PATTERN = re.compile(r"^atomic-(?P<index>[1-9]\d*)$")


def preflight_policy_artifacts(torch_module: Any | None = None) -> dict[str, Any]:
    """Validate and load the unchanged frozen artifact before orchestration."""

    torch_module = import_module("torch") if torch_module is None else torch_module
    manifest_path, checkpoint_path = artifact_paths()
    manifest = validate_manifest(manifest_path, checkpoint_path)
    _, checkpoint = load_policy_checkpoint(torch_module, checkpoint_path)
    if checkpoint["weight_optimizer_steps"] != manifest["training"]["selected_optimizer_steps"]:
        raise ValueError("Checkpoint and manifest optimizer-step counts do not match.")
    return manifest


def _logs_clean(logs: Sequence[str]) -> bool:
    for line in logs:
        match = LOG_PATTERN.match(line)
        level = match.group("level").lower() if match else ""
        message = match.group("message").lower() if match else line.lower()
        if level == "error" or "exception" in message or "traceback" in message:
            print(f"Error message found in logs: {line.rstrip()}")
            return False
    return True


def _canonical_rows(hl: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for run in hl.get("goal_runs", [])
        for transfer in (run.get("plan") or {}).get("transfers", [])
        for row in transfer.get("atomic", [])
    ]


def _canonical_evidence_valid(hl: dict[str, Any]) -> bool:
    rows = _canonical_rows(hl)
    if not rows:
        return False
    ids: list[int] = []
    for row in rows:
        match = ATOMIC_ID_PATTERN.fullmatch(str(row.get("atomic_action_id", "")))
        q_values = row.get("q_values")
        if (
            match is None
            or not isinstance(row.get("observation_id"), int)
            or not isinstance(row.get("input_sha256"), str)
            or len(row["input_sha256"]) != 64
            or not isinstance(row.get("selected_action"), int)
            or row["selected_action"] not in range(N_ACTIONS)
            or not isinstance(q_values, list)
            or len(q_values) != N_ACTIONS
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in q_values)
            or row.get("legal") is not True
        ):
            return False
        ids.append(int(match.group("index")))
    return len(ids) == len(set(ids)) and sorted(ids) == list(range(1, len(ids) + 1))


def _goal_records_valid(hl: dict[str, Any]) -> bool:
    runs = hl.get("goal_runs")
    expected_count = hl.get("completion_goal_limit", GOAL_COMPLETION_LIMIT)
    if not isinstance(runs, list) or len(runs) != expected_count:
        return False
    goal_pairs: set[tuple[str, str]] = set()
    for run in runs:
        goal = run.get("goal")
        plan = run.get("plan")
        if not isinstance(goal, dict) or not isinstance(plan, dict):
            return False
        pair = (goal.get("top"), goal.get("bottom"))
        transfers = plan.get("transfers")
        if (
            None in pair
            or pair in goal_pairs
            or run.get("status") != "succeeded"
            or run.get("final_goal_fact") != format_fact("on", pair)
            or not isinstance(run.get("final_observation_id"), int)
            or not isinstance(transfers, list)
            or not transfers
        ):
            return False
        goal_pairs.add(cast(tuple[str, str], pair))
        for transfer in transfers:
            spec = transfer.get("spec")
            target = transfer.get("target_facts")
            terminal_id = transfer.get("terminal_observation_id")
            revision_id = transfer.get("revision_observation_id")
            if not isinstance(spec, dict) or not isinstance(target, list) or len(target) != 2:
                return False
            expected = {
                format_fact("on", (spec.get("block"), spec.get("destination_support"))),
                format_fact("at-location", (spec.get("block"), spec.get("destination"))),
            }
            observation_ids = [row.get("observation_id") for row in transfer.get("atomic", [])]
            if (
                set(target) != expected
                or not isinstance(terminal_id, int)
                or not isinstance(revision_id, int)
                or revision_id < terminal_id
                or not observation_ids
                or observation_ids != sorted(observation_ids)
            ):
                return False
    return True


def check_results(
    states: dict[str, dict[str, Any]],
    environment: dict[str, Any],
    logs: Sequence[str],
    manifest: dict[str, Any],
    *,
    verbose: bool = False,
) -> bool:
    """Validate scientific outcomes and compact interface-boundary invariants."""

    required = {
        module_name(PERCEPTOR, 0),
        module_name(ACTUATOR, 0),
        module_name(LLREASONER, 0),
        module_name(KNOWLEDGE, 0),
        module_name(GOALGRAPH, 0),
        module_name(HLREASONER, 0),
    }
    missing = required - states.keys()
    if missing or not isinstance(environment, dict):
        print(f"Missing runtime states: {sorted(missing)}")
        return False

    perceptor = states[module_name(PERCEPTOR, 0)]
    actuator = states[module_name(ACTUATOR, 0)]
    ll = states[module_name(LLREASONER, 0)]
    knowledge = states[module_name(KNOWLEDGE, 0)]
    graph = states[module_name(GOALGRAPH, 0)]
    hl = states[module_name(HLREASONER, 0)]
    module_states = [perceptor, actuator, ll, knowledge, graph, hl]
    rows = _canonical_rows(hl)
    row_count = len(rows)
    transfer_count = sum(
        len((run.get("plan") or {}).get("transfers", []))
        for run in hl.get("goal_runs", [])
    )
    expected_goal_count = hl.get("completion_goal_limit", GOAL_COMPLETION_LIMIT)
    checks = {
        "logs are clean": _logs_clean(logs),
        "runtime failures are absent": all(state.get("failure") is None for state in module_states) and environment.get("failure") is None,
        "six-module topology has no online learning": module_name("learner", 0) not in states and module_name("memory", 0) not in states,
        "frozen policy identity matches": ll.get("policy_loaded") is True and ll.get("policy_id") == manifest["architecture"] and ll.get("checkpoint_sha256") == manifest["checkpoint_sha256"],
        "canonical neural evidence is valid": _canonical_evidence_valid(hl),
        "one owner spans the configured goals": _goal_records_valid(hl),
        "atomic ownership counts are exact": row_count == ll.get("inference_count") == ll.get("atomic_request_count") == actuator.get("request_count") == actuator.get("successful_status_count") == environment.get("action_count"),
        "observation boundary counts are exact": ll.get("observation_request_count") == ll.get("observation_count") == perceptor.get("request_count") == perceptor.get("observation_count") == environment.get("observation_count") == environment.get("action_count", -1) + 1,
        "transfer boundary counts are exact": transfer_count == hl.get("transfer_dispatch_count") == hl.get("completed_transfer_count") == graph.get("dispatch_count") == graph.get("terminal_count") == ll.get("activated_transfer_count") == ll.get("completed_transfer_count"),
        "grounded beliefs traversed knowledge": ll.get("belief_count", 0) > 0 and knowledge.get("revision_count") == knowledge.get("forwarded_count") == ll.get("observation_count"),
        "terminal repeated-goal state is valid": hl.get("phase") == "completed" and hl.get("terminal_reason") == "goal-completion-limit" and hl.get("goal_completion_count") == expected_goal_count,
        "all modules are quiescent": perceptor.get("pending") is None and actuator.get("pending") is None and ll.get("active_transfer") is None and ll.get("pending_action") is None and ll.get("awaiting_observation") is False and graph.get("active_goal_id") is None and hl.get("intention") is None and hl.get("current_plan_id") is None and hl.get("pending_goal") is None and hl.get("pending_terminal") is None and hl.get("pending_revision") is None,
        "no training state exists at runtime": not any(key in ll for key in ("optimizer", "optimizer_steps", "replay", "replay_buffer", "training_steps")),
    }
    success = True
    for label, passed in checks.items():
        if not passed:
            print(f"Acceptance check failed: {label}")
            success = False
    if verbose:
        print(f"Completed stacking goals: {hl.get('goal_completion_count', 0)}")
        print(f"Completed neural transfers: {transfer_count}")
        print(f"Neural atomic actions: {row_count}")
        print(f"Frozen training seconds: {manifest['training']['elapsed_seconds']:.3f}")
    return success


def _workspace_root() -> Path:
    for candidate in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (candidate, *candidate.parents):
            pyproject = parent / "pyproject.toml"
            if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
                return parent
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root.")


def _verify_host_mhagenta(mha_root: Path) -> None:
    installed_version = distribution_version("mhagenta")
    imported_source = Path(cast(str, mhagenta.__file__)).resolve()
    if installed_version != DEFAULT_MHAGENTA_VERSION or not imported_source.is_relative_to(mha_root):
        raise RuntimeError(f"Host orchestration requires local MHAgentA {DEFAULT_MHAGENTA_VERSION} at {mha_root}.")


def _run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
    *,
    matched: bool,
) -> bool:
    """Run and validate one pretrained-neurosymbolic Blocks World agent."""

    exp_path = Path(exp_path).resolve()
    workspace_root = _workspace_root()
    mha_root = (workspace_root.parent / "mhagenta").resolve()
    _verify_host_mhagenta(mha_root)
    runtime_version = DEFAULT_TORCH_MHAGENTA_VERSION if mha_version in {"", "latest"} else mha_version
    if runtime_version != DEFAULT_TORCH_MHAGENTA_VERSION:
        raise ValueError(f"Experiment 2-5-BW requires MHAgentA {DEFAULT_TORCH_MHAGENTA_VERSION}.")
    manifest = preflight_policy_artifacts()

    bw_spec = find_spec("mha_env_blocksworld")
    if bw_spec is None or bw_spec.origin is None:
        raise ImportError("Could not locate mha_env_blocksworld runtime sources.")
    exp_dir = Path(__file__).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    bw_source = Path(bw_spec.origin).resolve().parent
    seeder = Seeder(run)
    treatment = (
        paired_treatment(run, "2-5-BW")
        if matched
        else {
            "protocol_version": "2-5-bw-five-goal-v2",
            "treatment_id": "2-5-bw-five-goal-v2",
            "goal_count": GOAL_COMPLETION_LIMIT,
            "seed_disjointness": "unverifiable",
        }
    )
    goal_limit = 1 if matched else GOAL_COMPLETION_LIMIT
    exchange_name = "mhagenta"
    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=0.0,
        control_frequency=0.0,
        status_frequency=5.0,
        agent_start_delay=20.0,
        exec_duration=DURATION,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        no_stdout_logs=False,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange_name,
        state_autosave_interval=30,
    )
    initial = initial_states(treatment)
    run_agent_id = agent_name(run, "2_5")
    run_env_id = env_name(run, "2_5")
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=BlocksWorldPerceptor(module_id=module_name(PERCEPTOR, 0), initial_state=initial[PERCEPTOR], exchange_name=exchange_name),
        actuators=BlocksWorldActuator(module_id=module_name(ACTUATOR, 0), initial_state=initial[ACTUATOR], exchange_name=exchange_name),
        ll_reasoners=NeuralTransferLLReasoner(module_id=module_name(LLREASONER, 0), initial_state=initial[LLREASONER]),
        knowledge=ClosedWorldKnowledge(module_id=module_name(KNOWLEDGE, 0), initial_state=initial[KNOWLEDGE]),
        hl_reasoners=RepeatedGoalHLReasoner(
            module_id=module_name(HLREASONER, 0),
            init_kwargs={
                "seed": seeder.hl_reasoner,
                "num_blocks": NUM_BLOCKS,
                "table_len": TABLE_LEN,
                "planner_timeout": PLANNER_TIMEOUT,
                "goal_completion_limit": goal_limit,
                **({"fixed_goal": treatment["goal"]} if matched else {}),
            },
            initial_state=initial[HLREASONER],
        ),
        goal_graphs=TransferGoalGraph(module_id=module_name(GOALGRAPH, 0), initial_state=initial[GOALGRAPH]),
        init_script=exp_dir / "init_script.sh",
        requirements_path=exp_dir / "requirements.txt",
        extra_runtime_sources=[common_source, exp_dir],
    )
    record = RECORD == "all" or (RECORD == "first" and run == 0)
    orchestrator.add_environment(
        base=NumericBlocksWorldEnvironment(environment_initial_state(
            seed=int(treatment["seed"]) if matched else seeder.environment,
            record=record,
            treatment=treatment,
        )),
        env_id=run_env_id,
        exec_duration=ENVIRONMENT_DURATION,
        requirements_path=exp_dir / "requirements-env.txt",
        exchange_name=exchange_name,
        extra_runtime_sources=[common_source, bw_source, exp_dir],
    )
    orchestrator.run(mhagenta_version=runtime_version, local_build=mha_root, force_run=True)

    final_states = gather_states(exp_path, False, no_warnings=True)
    logs: list[str] = []
    for runtime_id in (run_agent_id, run_env_id):
        log_path = (exp_path / runtime_id).with_suffix(".log")
        if log_path.is_file():
            logs.extend(log_path.read_text(encoding="utf-8").splitlines(keepends=True))
    result = check_results(
        final_states[run_agent_id],
        final_states[run_env_id][run_env_id],
        logs,
        manifest,
        verbose=VERBOSE,
    )
    print(f"Results: {result}")
    return result


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
) -> bool:
    """Run the canonical five-goal pretrained treatment."""

    return _run_experiment(run, exp_path, mha_version, matched=False)


def run_matched_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
) -> bool:
    """Run one paired fixed goal shared with experiment 2-4-BW."""

    return _run_experiment(run, exp_path, mha_version, matched=True)


def _print_batch_summary(exp_path: Path) -> None:
    states = gather_states(exp_path, False, no_warnings=True)
    agents = [
        modules for agent_id, modules in states.items()
        if agent_id.startswith("exp_agent2_5_") and module_name(HLREASONER, 0) in modules
    ]
    if not agents:
        return
    high_level = [modules[module_name(HLREASONER, 0)] for modules in agents]
    goals = sum(state["goal_completion_count"] for state in high_level)
    transfers = sum(state["completed_transfer_count"] for state in high_level)
    actions = sum(len(_canonical_rows(state)) for state in high_level)
    manifest = validate_manifest(*artifact_paths())
    print("2-5-BW pretrained transfer-policy summary")
    print(f"  completed stacking goals: {goals}")
    print(f"  completed neural transfers: {transfers}")
    print(f"  neural atomic actions: {actions}")
    print(f"  frozen training seconds: {manifest['training']['elapsed_seconds']:.3f}")


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] | None = None,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
    process_only: bool = False,
    *, duration_seconds: int | None = None, policy: str = "transfer", **hourly_options: Any,
) -> None:
    """Run the experiment batch or summarize its existing compact results."""

    if process_only and (Path(exp_path) / "batch.json").is_file():
        from ..exp2_6_direct.results import process_batch
        ids = json.loads((Path(exp_path) / "batch.json").read_text())["run_ids"] if runs is None else list(normalize_runs(runs)[0])
        process_batch(Path(exp_path).resolve(), ids)
        return

    runs = 50 if runs is None else runs
    if duration_seconds is not None:
        from ..exp2_6_direct.batch import run_batch as run_hourly
        run_hourly(runs, exp_path, mha_version, process_only, duration_seconds=duration_seconds,
                   frozen_policy=policy, **hourly_options)
        return
    if policy != "transfer" or hourly_options:
        raise ValueError("Additional policy/collection options require duration_seconds.")

    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [{"execution_id": f"run-{run}", "run_id": run, "factors": {
                    "protocol_version": "2-5-bw-five-goal-v2",
                    "treatment_id": "2-5-bw-five-goal-v2",
                    "goal_count": GOAL_COMPLETION_LIMIT,
                    "seed_disjointness": "unverifiable",
                }}
                for run in run_ids]
    primary_error: BaseException | None = None
    try:
        results_available = run_experiment_batch(
            experiment_id="2-5", title="PRETRAINED NEUROSYMBOLIC BLOCKS WORLD",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=run_experiment, process_only=process_only,
        )
        if results_available:
            _print_batch_summary(root)
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


def run_matched_batch(
    runs: int | tuple[int, int] | Sequence[int] = 12,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
    process_only: bool = False,
) -> None:
    """Run or process the fixed one-goal paired treatment."""

    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [
        {"execution_id": f"run-{run}", "run_id": run,
         "factors": paired_treatment(run, "2-5-BW")}
        for run in run_ids
    ]
    primary_error: BaseException | None = None
    try:
        results_available = run_experiment_batch(
            experiment_id="2-5-matched",
            title="PRETRAINED NEUROSYMBOLIC BLOCKS WORLD — MATCHED GOAL",
            runs=run_ids,
            exp_path=exp_path,
            mha_version=mha_version,
            runner=run_matched_experiment,
            process_only=process_only,
        )
        if results_available:
            _print_batch_summary(root)
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
                "Execution-metrics processing also failed: "
                f"{type(reporting_error).__name__}"
            )
