"""Run and validate the current frozen-policy Crafter survival experiment."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, cast

import mha_exp_common
import mhagenta
from mha_exp_common.batch import (
    NonRetryableBatchError,
    cleanup_run_containers,
    cleanup_run_images,
    normalize_runs,
)
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import (
    DEFAULT_MHAGENTA_VERSION,
    DEFAULT_TORCH_MHAGENTA_VERSION,
)
from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)
from mha_exp_common.utils import (
    Seeder,
    agent_name,
    env_name,
    gather_states,
    module_name,
)

from .contracts import K_ACTION, K_ATOMIC_ID, K_OWNER_ID, K_REQUESTER, MAX_TOTAL_ACTIONS, activity_action_bound
from .environment import CrafterNeurosymbolicEnvironment
from .environment import initial_state as environment_initial_state
from .grounding import load_grounding_templates, resolve_active_grounding_bundle
from .orchestration import ContainerStartOrchestrator as Orchestrator
from .policy import (
    PolicyId,
    encode_context,
    file_sha256,
    load_policy_bundle,
    select_action,
    selection_actions,
)
from .runtime import (
    CrafterActivityActuator,
    CrafterActivityGoalGraph,
    CrafterActivityHLReasoner,
    CrafterBeliefKnowledge,
    CrafterRGBPerceptor,
    NeuralActivityLLReasoner,
    initial_states,
)

DURATION = 600.0
STARTUP_DELAY = 20.0
ENVIRONMENT_OVERRUN = 30.0
RECORD: Literal["all", "first", "none"] = "first"
VERBOSE = True
EXPERIMENT_ID = "2_cr_5"


def preflight_artifacts(torch_module: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load current models and templates before building any containers."""

    if torch_module is None:
        import torch as torch_module
    bundle, policies, _ = load_policy_bundle(torch_module)
    grounding, templates = resolve_active_grounding_bundle()
    if policies["environment"] != templates["environment"]:
        raise ValueError("Policy and grounding renderer contracts differ.")
    load_grounding_templates(grounding)
    return (
        {**policies, "_manifest_sha256": file_sha256(bundle / "manifest.json")},
        {**templates, "_manifest_sha256": file_sha256(grounding / "manifest.json")},
    )


def check_results(
    agent_states: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    run_root: Path | None = None,
    runtime_ids: tuple[str, str] | None = None,
    artifact_evidence: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
    verbose: bool = False,
    require_objective: bool = True,
    allow_time_limit: bool = True,
) -> tuple[bool, list[str]]:
    """Check execution integrity, optionally also requiring survival and the crafting milestone."""

    errors = []
    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        states = {kind: agent_states[module_name(kind, 0)] for kind in initial_states()}
        ll, hl, actuator = states[LLREASONER], states[HLREASONER], states[ACTUATOR]
        from .result_validation import time_limit_evidence
        timeout = (time_limit_evidence(agent_states, environment, run_root, runtime_ids)
                   if allow_time_limit else None)
        rows = ll["trace"]
        count = len(rows)
        require(all(state["failure"] is None for state in states.values()), "module contract failure")
        require(environment["failure"] is None, "environment contract failure")
        if require_objective:
            diamond = (environment["inventory"].get("diamond", 0) > 0
                       and environment["achievement_counts"].get("collect_diamond", 0) > 0)
            diamond_stop = hl["terminal_reason"] == "diamond_acquired" and diamond
            require((diamond_stop and count <= MAX_TOTAL_ACTIONS) or count == MAX_TOTAL_ACTIONS,
                    f"native action budget incomplete without diamond: {count}/{MAX_TOTAL_ACTIONS}")
            require((diamond_stop or hl["terminal_reason"] == "action_budget") and hl["survived"], "survival objective failed")
            require(not environment["terminal"] and environment["inventory"].get("health", 0) > 0, "agent did not survive")
            require(environment["achievement_counts"].get("make_stone_pickaxe", 0) >= 1, "stone pickaxe was not crafted")
        require(count == environment["native_action_count"] == actuator["request_count"] == actuator["status_count"] == ll["status_count"], "action/status counts disagree")
        require(count + 1 == ll["observation_count"] == ll["observation_request_count"] == states[PERCEPTOR]["observation_count"] == environment["observation_count"] == states[KNOWLEDGE]["revision_count"] == hl["belief_update_count"], "observation counts disagree")
        require(ll["pending_action"] is None and ll["passive_status"] is None and not ll["awaiting_observation"], "unfinished LL cycle")
        require(timeout is not None or (hl["active_goal"] is None and hl["pending_terminal"] is None and states[GOALGRAPH]["active_goal_id"] is None), "unfinished goal reconciliation")
        require(timeout is not None or ll["received_goal_count"] == ll["terminal_goal_count"] == hl["dispatch_count"] == hl["terminal_count"] == states[GOALGRAPH]["terminal_count"], "goal counts disagree")
        require(len({row[K_OWNER_ID] for row in rows}) == count, "duplicate owner action IDs")
        neural_counts = {policy.value: 0 for policy in PolicyId}
        primitive_rows = []
        for index, row in enumerate(rows, 1):
            require(row[K_ATOMIC_ID] == index and row["status"][K_ATOMIC_ID] == index, "noncontiguous environment IDs")
            require(row["input_observation_id"] == index and row["result_observation_id"] == index + 1, "action/observation join mismatch")
            require(row["status"] == actuator["statuses"][index - 1] == environment["status_history"][index - 1], "status evidence differs")
            require(all(row["status"][key] == row[key] == environment["action_history"][index - 1][key]
                        for key in (K_ACTION, K_ATOMIC_ID, K_OWNER_ID, K_REQUESTER)), "action owner evidence differs")
            if row[K_REQUESTER] == "ll_policy":
                policy = PolicyId(row["policy_id"])
                mask_context = encode_context(policy, row["source_cell"], row["target_cell"])
                context_target = (row.get("search_goal_cell") if policy is PolicyId.EAT_COW
                                  and row["target_cell"] is None else row["target_cell"])
                expected_context = encode_context(policy, row["source_cell"], context_target)
                require(row["context"] == expected_context.tolist(), "neural context/goal mismatch")
                actions = selection_actions(policy, mask_context, tuple(row["facing"]), row["available_movements"])
                require(list(actions) == row["legal_actions"] and select_action(row["q_values"], actions) == row["action"], "neural action/mask mismatch")
                neural_counts[policy.value] += 1
            else:
                require(row[K_REQUESTER] == "hl_primitive" and row["policy_id"] is None and "q_values" not in row, "primitive counted as neural evidence")
                primitive_rows.append(row)
        require(neural_counts == ll["policy_action_counts"] and sum(neural_counts.values()) == ll["inference_count"], "policy counts disagree")
        require(ll.get("enable_eat_cow", True) == hl.get("enable_eat_cow", True), "policy availability differs")
        if not hl.get("enable_eat_cow", True):
            require(neural_counts["eat_cow"] == 0 and not any(
                item.get("activity") == "eat_cow" for item in ll["activities"]
            ), "disabled EatCow was dispatched")
        require(len(primitive_rows) == len(hl["primitive_decisions"]), "primitive decision count differs")
        for row, decision in zip(primitive_rows, hl["primitive_decisions"], strict=True):
            require(row[K_OWNER_ID] == decision[K_OWNER_ID] and row["action"] == decision["action"], "primitive owner mismatch")
        require(all(0 <= item["actions"] <= item["max_actions"] <= activity_action_bound(item["activity"])
                    for item in ll["activities"] if item["kind"] == "activity"), "policy activity bound exceeded")
        require(environment["closed"], "environment did not close its recording")
        require(ll["belief_state"]["achievement_counts"] == {
            name: count for name, count in environment["achievement_counts"].items() if count
        }, "achievement grounding differs")
        if artifact_evidence is not None:
            policies, grounding = artifact_evidence
            require(ll["artifact_contract"]["checkpoint_sha256"] == {name: record["sha256"] for name, record in policies["policies"].items()}, "checkpoint identity differs")
            require(ll["artifact_contract"]["grounding_manifest_sha256"] == grounding["_manifest_sha256"], "grounding identity differs")
        if run_root is not None and runtime_ids is not None:
            for runtime_id in runtime_ids:
                log = run_root / f"{runtime_id}.log"
                require(log.is_file(), f"missing log: {runtime_id}")
                if log.is_file():
                    content = log.read_text(encoding="utf-8", errors="replace").lower()
                    require(not any(word in content for word in ("traceback", "caught exception", "failed to save state")), f"runtime errors in {runtime_id}")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"incomplete or invalid result evidence: {error}")
    if verbose:
        print("PASS" if not errors else "\n".join(f"FAIL: {error}" for error in errors), flush=True)
    return not errors, errors


def _workspace_root() -> Path:
    for candidate in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (candidate, *candidate.parents):
            pyproject = parent / "pyproject.toml"
            if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
                return parent
    raise RuntimeError("Could not locate the experiment workspace.")


def _verify_host_mhagenta(mha_root: Path) -> None:
    source = Path(cast(str, mhagenta.__file__)).resolve()
    if distribution_version("mhagenta") != DEFAULT_MHAGENTA_VERSION or not source.is_relative_to(mha_root):
        raise RuntimeError(f"Host requires local MHAgentA {DEFAULT_MHAGENTA_VERSION} at {mha_root}.")


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
    *, enable_eat_cow: bool = True, environment_seed: int | None = None,
) -> bool:
    """Run one global run ID, optionally overriding its default environment seed."""

    if environment_seed is not None and (type(environment_seed) is not int or not 0 <= environment_seed < 2**32):
        raise ValueError("environment_seed must be an integer in [0, 2**32)")
    output = Path(exp_path).resolve()
    workspace = _workspace_root()
    mha_root = (workspace.parent / "mhagenta").resolve()
    _verify_host_mhagenta(mha_root)
    # An existing base image bypasses local_build and can retain older same-version code.
    framework_source = mha_root / "mhagenta"
    version = DEFAULT_TORCH_MHAGENTA_VERSION if mha_version in {"", "latest"} else mha_version
    if version != DEFAULT_TORCH_MHAGENTA_VERSION:
        raise ValueError(f"Experiment 2-5-CR requires MHAgentA {DEFAULT_TORCH_MHAGENTA_VERSION}.")
    artifact_evidence = preflight_artifacts()
    crafter_spec = find_spec("mha_env_crafter")
    if crafter_spec is None or crafter_spec.origin is None:
        raise ImportError("Could not locate mha_env_crafter runtime sources.")
    crafter_source = Path(crafter_spec.origin).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    exp_dir = Path(__file__).resolve().parent
    # Use the thesis worked example and shared zero-based run convention.
    seeder = Seeder(run)
    run_agent_id, run_env_id = agent_name(run, EXPERIMENT_ID), env_name(run, EXPERIMENT_ID)
    cleanup_run_containers(run_agent_id, run_env_id, phase="before")
    cleanup_run_images(run_agent_id, run_env_id, phase="before")
    exchange = "mhagenta"
    orchestrator = Orchestrator(
        save_dir=output,
        step_frequency=0.0,
        control_frequency=0.0,
        status_frequency=5.0,
        agent_start_delay=STARTUP_DELAY,
        exec_duration=DURATION,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        no_stdout_logs=False,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange,
        state_autosave_interval=30,
        stop_on_agents_term=True,
    )
    states = initial_states(enable_eat_cow=enable_eat_cow)
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=CrafterRGBPerceptor(
            module_id=module_name(PERCEPTOR, 0),
            initial_state=states[PERCEPTOR],
            exchange_name=exchange,
        ),
        actuators=CrafterActivityActuator(
            module_id=module_name(ACTUATOR, 0),
            initial_state=states[ACTUATOR],
            exchange_name=exchange,
        ),
        ll_reasoners=NeuralActivityLLReasoner(module_id=module_name(LLREASONER, 0), initial_state=states[LLREASONER]),
        knowledge=CrafterBeliefKnowledge(module_id=module_name(KNOWLEDGE, 0), initial_state=states[KNOWLEDGE]),
        hl_reasoners=CrafterActivityHLReasoner(module_id=module_name(HLREASONER, 0), initial_state=states[HLREASONER]),
        goal_graphs=CrafterActivityGoalGraph(module_id=module_name(GOALGRAPH, 0), initial_state=states[GOALGRAPH]),
        requirements_path=exp_dir / "requirements.txt",
        extra_runtime_sources=[common_source, exp_dir, framework_source],
    )
    record = RECORD == "all" or (RECORD == "first" and run == 0)
    environment_state = environment_initial_state()
    environment_state.update({
        "seed": seeder.environment if environment_seed is None else environment_seed,
        "record": record,
        "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}",
        "recording_prefix": f"{run_env_id}/{Orchestrator.SAVE_SUBDIR}",
        "expected_agent_id": run_agent_id,
    })
    orchestrator.add_environment(
        base=CrafterNeurosymbolicEnvironment(environment_state),
        env_id=run_env_id,
        exec_duration=DURATION + ENVIRONMENT_OVERRUN,
        requirements_path=exp_dir / "requirements-env.txt",
        exchange_name=exchange,
        extra_runtime_sources=[crafter_source, exp_dir, framework_source],
    )
    orchestrator.run(mhagenta_version=version, local_build=mha_root, force_run=True)
    try:
        saved = gather_states(output, False, no_warnings=True)
        agent_states = saved[run_agent_id]
        environment = saved[run_env_id][run_env_id]
        if not isinstance(agent_states, Mapping) or not isinstance(environment, Mapping):
            raise ValueError("saved agent and environment states must be mappings")
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise NonRetryableBatchError(f"Run {run}: incomplete saved-state evidence: {error}") from error
    intact, errors = check_results(
        agent_states, environment,
        run_root=output,
        runtime_ids=(run_agent_id, run_env_id),
        artifact_evidence=artifact_evidence,
        require_objective=False,
    )
    expected_seed = seeder.environment if environment_seed is None else environment_seed
    if type(environment.get("environment_seed")) is not int or environment.get("environment_seed") != expected_seed:
        intact = False
        errors.append(f"environment seed differs: expected {expected_seed}, found {environment.get('environment_seed')}")
    if not intact:
        raise NonRetryableBatchError(f"Run {run}: execution integrity failed: {'; '.join(errors)}")
    passed, _ = check_results(
        agent_states,
        environment,
        run_root=output,
        runtime_ids=(run_agent_id, run_env_id),
        artifact_evidence=artifact_evidence,
        verbose=VERBOSE,
    )
    print(f"Results: {passed}")
    cleanup_run_containers(run_agent_id, run_env_id, phase="after")
    cleanup_run_images(run_agent_id, run_env_id, phase="after")
    return passed


def _seed_mapping(
    runs: int | tuple[int, int] | Sequence[int],
    environment_seeds: Mapping[str | int, int] | None,
) -> dict[int, int] | None:
    """Validate a manifest keyed by global run ID before any batch side effects."""

    if environment_seeds is None:
        return None
    if not isinstance(environment_seeds, Mapping):
        raise ValueError("environment_seeds must map global run IDs to integer seeds")
    seeds = {}
    for key, seed in environment_seeds.items():
        if type(key) is int and key >= 0:
            run = key
        elif isinstance(key, str) and key.isascii() and key.isdecimal() and str(int(key)) == key:
            run = int(key)
        else:
            raise ValueError(f"Invalid environment_seeds run ID: {key!r}")
        if run in seeds:
            raise ValueError(f"Duplicate environment_seeds run ID: {run}")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError(f"environment_seeds[{key!r}] must be an integer in [0, 2**32)")
        seeds[run] = seed
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("environment_seeds must contain unique seeds")
    requested, _ = normalize_runs(runs)
    missing = sorted(set(requested) - seeds.keys())
    if missing:
        raise ValueError(f"environment_seeds is missing requested global run IDs: {missing}")
    return seeds


def _run_selected_episode(
    run: int, exp_path: str | os.PathLike[str], mha_version: str,
    *, enable_eat_cow: bool, environment_seeds: Mapping[int, int] | None,
) -> bool:
    """Resolve the selected seed without renumbering a remote shard's runs."""

    return run_experiment(
        run, exp_path, mha_version, enable_eat_cow=enable_eat_cow,
        environment_seed=None if environment_seeds is None else environment_seeds[run],
    )


def process_results(
    runs: int | tuple[int, int] | Sequence[int], exp_path: str | os.PathLike[str],
) -> tuple[bool, dict[str, Any]]:
    """Write one readable run summary without statistical or provenance gates."""

    from .reporting import process_results as process_retained_results
    return process_retained_results(runs, exp_path)


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 50,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_TORCH_MHAGENTA_VERSION,
    process_only: bool = False,
    enable_eat_cow: bool = True,
    stop_on_error: bool = True,
    environment_seeds: Mapping[str | int, int] | None = None,
) -> None:
    """Run frozen policies with an optional global run-ID-to-seed manifest.

    With ``stop_on_error=False``, scientific failures continue the cohort;
    execution-integrity failures always stop it and retain the evidence.
    """

    if type(enable_eat_cow) is not bool:
        raise ValueError("enable_eat_cow must be a boolean")
    if type(stop_on_error) is not bool:
        raise ValueError("stop_on_error must be a boolean")
    seeds = _seed_mapping(runs, environment_seeds)
    path = Path(exp_path).resolve()
    if not process_only:
        workspace = _workspace_root()
        if path == workspace or not path.is_relative_to(workspace) or path == workspace / "agents":
            raise ValueError("Use an experiment-specific output directory inside the workspace.")
        preflight_artifacts()
        if path.exists():
            archive = path.with_name(f"{path.name}-archive-{datetime.now(UTC):%Y%m%d-%H%M%S-%f}")
            path.rename(archive)
            print(f"Previous output preserved at {archive}", flush=True)
        run_experiment_batch(
            experiment_id="2-5-CR", title="FIVE-POLICY SURVIVAL AND CRAFTING",
            runs=runs, exp_path=path, mha_version=mha_version,
            runner=partial(_run_selected_episode, enable_eat_cow=enable_eat_cow, environment_seeds=seeds),
            cleanup_before_run=False, stop_on_error=stop_on_error,
        )
    passed, _ = process_results(runs, path)
    if not passed:
        raise RuntimeError("2-5-CR did not meet its acceptance criteria; see summary.json.")
