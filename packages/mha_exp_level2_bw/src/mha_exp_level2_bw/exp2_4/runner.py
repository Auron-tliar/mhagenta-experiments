"""Orchestration and public surface for experiment 2-4-BW."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
import os
from pathlib import Path
from typing import Literal, cast

from mhagenta import Orchestrator
import mha_exp_common
from mha_exp_common.batch import (cleanup_run_containers, cleanup_run_images,
                                  normalize_runs, run_batch as run_experiment_batch)
from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name

from .agent import (
    A_CLOSE, A_MOVE_LEFT, A_MOVE_RIGHT, A_PICK_UP, A_PUT_DOWN,
    DURATION, K_ACTION, K_LEGAL, K_OBSERVATION, K_REWARD, K_STATE,
    NUM_BLOCKS, PLANNER_TIMEOUT, TABLE_LEN,
    ClosedWorldKnowledge, HybridBDIReasoner, HybridBlocksWorldActuator,
    HybridBlocksWorldPerceptor, TestEnvironment, TransferGoalGraph,
    TransferLLReasoner, _initial_states, transfer_initial_states,
)
from .checking import check_results
from .reporting import process_execution_metrics
from .treatment import treatment_for_run
from .planning import (
    GoalSpec, PlanningOutcome, PlanningService, TransferSpec, belief_to_fact,
    beliefs_to_dicts, beliefs_to_facts, block_names, build_problem_pddl,
    format_fact, generate_options, goal_to_dict, location_index, location_names,
    missing_transfer_preconditions, observed_transfer_targets,
    parse_symbolic_observation, project_abstract_facts, serialize_transfer_action,
    split_fact, transfer_from_goal, transfer_goal,
)

RECORD: Literal["all", "first", "none"] = "first"
VERBOSE = True
DEFAULT_MHAGENTA_VERSION = "1.4.12"


def _workspace_root() -> Path:
    for candidate in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (candidate, *candidate.parents):
            project = parent / "pyproject.toml"
            if project.is_file() and "[tool.uv.workspace]" in project.read_text(encoding="utf-8"):
                return parent
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root.")


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run one seeded hybrid symbolic Blocks World treatment."""

    exp_path, workspace = Path(exp_path).resolve(), _workspace_root()
    mha_root = (workspace.parent / "mhagenta").resolve()
    version_file = mha_root / "pyproject.toml"
    if not version_file.is_file() or 'version = "1.4.12"' not in version_file.read_text(encoding="utf-8"):
        raise RuntimeError(f"Expected local MHAgentA 1.4.12 checkout at {mha_root}.")
    if mha_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f"Experiment 2-4-BW requires MHAgentA {DEFAULT_MHAGENTA_VERSION}.")
    bw_spec = find_spec("mha_env_blocksworld")
    if bw_spec is None or bw_spec.origin is None:
        raise ImportError("Could not locate mha_env_blocksworld.")
    exp_dir = Path(__file__).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    seeder, exchange = Seeder(run), "mhagenta"
    treatment = treatment_for_run(run)
    run_agent_id, run_env_id = agent_name(run, "2_4"), env_name(run, "2_4")
    orchestrator = Orchestrator(
        save_dir=exp_path, step_frequency=0.0, control_frequency=0.0,
        status_frequency=5.0, agent_start_delay=20.0, exec_duration=DURATION,
        save_format="json", log_level=Orchestrator.INFO, save_logs=True,
        no_stdout_logs=False, mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange, state_autosave_interval=30,
        stop_on_agents_term=True,
    )
    initial = transfer_initial_states(treatment)
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=HybridBlocksWorldPerceptor(
            module_id=module_name(PERCEPTOR, 0), initial_state=initial[PERCEPTOR], exchange_name=exchange),
        actuators=HybridBlocksWorldActuator(
            module_id=module_name(ACTUATOR, 0), initial_state=initial[ACTUATOR], exchange_name=exchange),
        ll_reasoners=TransferLLReasoner(
            module_id=module_name(LLREASONER, 0), initial_state=initial[LLREASONER]),
        knowledge=ClosedWorldKnowledge(
            module_id=module_name(KNOWLEDGE, 0), initial_state=initial[KNOWLEDGE]),
        hl_reasoners=HybridBDIReasoner(
            module_id=module_name(HLREASONER, 0), initial_state=initial[HLREASONER],
            init_kwargs={"seed": seeder.hl_reasoner, "num_blocks": treatment["num_blocks"],
                         "table_len": treatment["table_len"], "planner_timeout": PLANNER_TIMEOUT,
                         "goal_completion_limit": 1,
                         "fixed_goal": treatment["goal"]}),
        goal_graphs=TransferGoalGraph(
            module_id=module_name(GOALGRAPH, 0), initial_state=initial[GOALGRAPH]),
        init_script=exp_dir / "init_script.sh",
        requirements_path=exp_dir / "requirements.txt",
        extra_runtime_sources=common_source,
    )
    record = RECORD == "all" or RECORD == "first" and run == 0
    orchestrator.add_environment(
        base=TestEnvironment({
            "seed": treatment["seed"], "record": record, "table_len": treatment["table_len"],
            "num_blocks": treatment["num_blocks"], K_STATE: [], "observation_count": 0,
            "actions": 0, "illegal_actions": 0, "treatment": treatment,
            "initial_state": [], "close_requests": 0, "closed": False,
        }),
        env_id=run_env_id, exec_duration=DURATION + 30.0,
        requirements_path=exp_dir / "requirements-env.txt", exchange_name=exchange,
        extra_runtime_sources=[common_source, Path(bw_spec.origin).resolve().parent],
    )
    cleanup_run_containers(run_agent_id, run_env_id, phase="before")
    cleanup_run_images(run_agent_id, run_env_id, phase="before")
    try:
        orchestrator.run(mhagenta_version=mha_version, local_build=mha_root, force_run=True)
    finally:
        cleanup_run_containers(run_agent_id, run_env_id, phase="after")
        cleanup_run_images(run_agent_id, run_env_id, phase="after")
    final = gather_states(exp_path, False, no_warnings=True)
    log_path = (exp_path / run_agent_id).with_suffix(".log")
    logs = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    result = (
        check_results(final[run_agent_id], logs, verbose=VERBOSE,
                      environment=final[run_env_id][run_env_id],
                      environment_logs=(exp_path / f"{run_env_id}.log").read_text().splitlines(),
                      expected_treatment=treatment)
        and final[run_env_id][run_env_id].get("illegal_actions") == 0
    )
    print(f"Results: {result}")
    return result


def _print_batch_summary(exp_path: Path) -> None:
    states = gather_states(exp_path, False, no_warnings=True)
    agents = [modules for name, modules in states.items()
              if name.startswith("exp_agent2_4_") and module_name(HLREASONER, 0) in modules]
    if not agents:
        return
    high = [modules[module_name(HLREASONER, 0)] for modules in agents]
    low = [modules[module_name(LLREASONER, 0)] for modules in agents]
    goals = sum(state["goal_completions"] for state in high)
    transfers = sum(state["completed_transfers"] for state in high)
    actions = sum(state["action_requests"] for state in low)
    print("2-4-BW transfer hierarchy summary")
    print(f"  completed stacking goals: {goals}")
    print(f"  completed abstract transfers: {transfers}")
    print(f"  mean atomic actions per transfer: {actions / transfers:.3f}" if transfers else "  mean atomic actions per transfer: n/a")
    print(f"  LPG plans / ENHSP fallback plans: {sum(s['lpg_successes'] for s in high)} / {sum(s['fallback_successes'] for s in high)}")
    print(f"  compound failures: {sum(s['compound_failures'] for s in high)}")


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
        available = run_experiment_batch(
            experiment_id="2-4", title="TRANSFER-BASED HYBRID SYMBOLIC BLOCKS WORLD",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=run_experiment, process_only=process_only,
            cleanup_before_run=False, stop_on_error=True,
        )
        if available:
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
