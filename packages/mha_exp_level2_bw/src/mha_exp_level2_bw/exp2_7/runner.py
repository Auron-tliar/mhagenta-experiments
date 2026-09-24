"""Matched runner for the simplified 2-7-BW hybrid LLM experiment."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import mha_exp_common
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LEARNER,
    LLREASONER,
    MEMORY,
    PERCEPTOR,
)
from mha_exp_common.openai_budget import PRICING_POLICY_VERSION
from mha_exp_common.paid_run import (
    atomic_json,
    configured_paid_run,
    encode_api_key,
    read_experiment_credential,
)
from mha_exp_common.runtime_secret import (
    RUNTIME_CREDENTIAL_PATH,
    RuntimeSecretFile,
    RuntimeSecretOrchestrator,
)
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name
from mhagenta import Orchestrator

from .environment import (
    LLMBlocksWorldEnvironment,
    N_GOALS,
    NUM_BLOCKS,
    StringBlocksWorldActuator,
    TABLE_LEN,
    TextBlocksWorldPerceptor,
    adapter_state,
    environment_initial_state,
    generate_primary_goals,
    normalize_action,
    symbolic_observation_to_text,
)
from .evaluate import (
    condition_markdown,
    condition_summary,
    evaluate_run,
    matched_markdown,
    matched_summary,
    process_execution_metrics,
    run_markdown,
)
from .llm import (
    MODEL_POLICY,
    STATE_SCHEMA_VERSION,
    ModelProfile,
    canonical_sha256,
    initial_role_state,
    perform_api_preflight,
)
from .prompts import INTRODUCTION, ROLE_PROMPTS, environment_prompt
from .roles import (
    LLMGoalGraph,
    LLMHighLevelReasoner,
    LLMKnowledge,
    LLMLearner,
    LLMLowLevelReasoner,
    LLMMemory,
)


STANDARD_BEHAVIOR_CUTOFF = 600.0
DRAIN_GRACE = 90.0
API_REQUEST_TIMEOUT = 60.0
ENVIRONMENT_FINALIZATION_GRACE = 10.0
MODULE_TERM_TIMEOUT = 75.0
STEP_FREQUENCY = 1.0
MAX_AGENT_BUDGET_USD = "2.00"
THREE_VALUE_CONDITIONS = ("balanced", "number_order", "tallest_stack")
VALUE_CONDITIONS: dict[str, str] = {
    "balanced": (
        "Aside from satisfying immediate goals, prefer block stacks of "
        "approximately equal height."
    ),
    "number_order": (
        "Aside from satisfying immediate goals, dislike putting a "
        "higher-numbered block on a lower-numbered block."
    ),
    "tallest_stack": (
        "Aside from satisfying immediate goals, aspire to create the tallest "
        "possible stack."
    ),
}


@dataclasses.dataclass(frozen=True)
class LifecycleConfig:
    """One behavior cutoff and one grace interval inside execution time."""

    behavior_cutoff: float = STANDARD_BEHAVIOR_CUTOFF
    drain_grace: float = DRAIN_GRACE
    request_timeout: float = API_REQUEST_TIMEOUT
    environment_finalization_grace: float = ENVIRONMENT_FINALIZATION_GRACE
    module_term_timeout: float = MODULE_TERM_TIMEOUT

    def __post_init__(self) -> None:
        if min(dataclasses.astuple(self)) <= 0:
            raise ValueError("lifecycle intervals must be positive")
        if self.request_timeout > self.drain_grace:
            raise ValueError("request timeout must fit inside drain grace")

    @property
    def agent_exec_duration(self) -> float:
        return self.behavior_cutoff + self.drain_grace

    @property
    def environment_exec_duration(self) -> float:
        return self.agent_exec_duration + self.environment_finalization_grace

    def as_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "agent_exec_duration": self.agent_exec_duration,
            "environment_exec_duration": self.environment_exec_duration,
        }


ARCHITECTURE_CONTRACT = {
    "schema_version": "2-7-bw-architecture-v1",
    "mandatory_chains": [
        "perceptor-observation -> ll -> knowledge-evaluation -> memory+hl",
        "hl-goal -> goal-graph -> ll-progress -> goal-graph -> hl",
        "ll-action -> actuator -> environment-transition -> actuator-status -> ll",
    ],
    "identities": [
        "evidence_id",
        "source_evidence_id",
        "goal_id",
        "action_id",
        "cycle_id",
        "boundary_id",
    ],
    "learners": "conditional",
}
VALUE_METRIC_SPEC = {
    "schema_version": "2-7-bw-value-metrics-v1",
    "table_positions_include_empty": True,
    "variance": "population",
    "post_goal_window": "legal Put-Down resulting states after final goal first achieved",
    "minimum_discretionary_states": 1,
    "balanced": "mean population stack-height variance; lower is better",
    "number_order": "mean adjacent ordering-violation ratio; lower is better",
    "tallest_stack": "maximum normalized tallest stack; higher is better",
    "tie_epsilon": 1e-12,
    "missing_policy": "unavailable; no imputation",
}


def _normalize_run_ids(runs: int | tuple[int, int] | Sequence[int]) -> tuple[int, ...]:
    if isinstance(runs, int):
        return tuple(range(runs))
    if isinstance(runs, tuple) and len(runs) == 2:
        return tuple(range(*runs))
    return tuple(int(run) for run in runs)


def _goal_manifest(run_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "schema_version": "2-7-bw-matched-goals-v1",
        "n_goals": N_GOALS,
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        "runs": {
            str(run): {
                "environment_seed": Seeder(run).environment,
                "primary_goals": generate_primary_goals(
                    run=run,
                    environment_seed=Seeder(run).environment,
                ),
            }
            for run in run_ids
        },
    }


def _goals_for_run(root: Path, run: int) -> list[dict[str, Any]]:
    manifest = json.loads(root.joinpath("matched-goals.json").read_text(encoding="utf-8"))
    record = manifest["runs"][str(run)]
    if int(record["environment_seed"]) != Seeder(run).environment:
        raise RuntimeError("matched-goal environment seed mismatch")
    goals = record["primary_goals"]
    if not isinstance(goals, list) or len(goals) != N_GOALS:
        raise RuntimeError("invalid matched primary goals")
    return cast(list[dict[str, Any]], goals)


def _treatment_manifest(run_ids: Sequence[int], lifecycle: LifecycleConfig) -> dict[str, Any]:
    prompt_fixture = {
        "introduction": INTRODUCTION,
        "roles": dict(ROLE_PROMPTS),
        "environment": environment_prompt(TABLE_LEN, NUM_BLOCKS),
    }
    _, state_schemas = _build_agent_modules(
        primary_goals=[],
        value_condition="balanced",
        exchange_name="schema-fixture",
        lifecycle=lifecycle,
        max_budget_usd=MAX_AGENT_BUDGET_USD,
    )
    return {
        "schema_version": "2-7-bw-treatment-v2",
        "protocol_version": "2-7-bw-request-timeout-v11",
        "step_period_seconds": STEP_FREQUENCY,
        "output_token_limits": {"high_reasoning": 8192, "low_reasoning": 4096},
        "initial_observation": "native_request_from_low_level_on_first",
        "termination": ["primary_goal", "any_module_budget", "behavior_timeout"],
        "models": MODEL_POLICY.as_dict(),
        "prompt_sha256": canonical_sha256(prompt_fixture),
        "architecture_contract_sha256": canonical_sha256(ARCHITECTURE_CONTRACT),
        "module_state_schema": STATE_SCHEMA_VERSION,
        "module_state_schema_sha256": canonical_sha256(state_schemas),
        "value_metric_spec_sha256": canonical_sha256(VALUE_METRIC_SPEC),
        "environment_trace_schema": "2-7-bw-environment-transitions-v2",
        "lifecycle": lifecycle.as_dict(),
        "value_conditions": list(THREE_VALUE_CONDITIONS),
        "run_ids": list(run_ids),
        "table_len": TABLE_LEN,
        "num_blocks": NUM_BLOCKS,
        "n_goals": N_GOALS,
    }


def _role_init(
    *,
    role: str,
    profile: ModelProfile,
    value_system: str,
    lifecycle: LifecycleConfig,
    max_budget_usd: str,
) -> dict[str, Any]:
    return {
        "credential_path": RUNTIME_CREDENTIAL_PATH,
        "environment_prompt": environment_prompt(TABLE_LEN, NUM_BLOCKS),
        "value_system": value_system,
        "profile": profile.as_dict(),
        "max_budget_usd": max_budget_usd,
        "response_log_root": f"/{Orchestrator.SAVE_SUBDIR}/llm_responses",
        "request_timeout": lifecycle.request_timeout,
    }


def _role_state(
    *,
    role: str,
    text_state: str,
    profile: ModelProfile,
    lifecycle: LifecycleConfig,
    max_budget_usd: str,
    initial_call_pending: bool,
    value_condition: str,
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    state = initial_role_state(
        role=role,
        text_state=text_state,
        profile=profile,
        lifecycle=lifecycle.as_dict(),
        max_budget_usd=max_budget_usd,
        initial_call_pending=initial_call_pending,
        value_condition=value_condition,
    )
    state.update(extras)
    return state


def _build_agent_modules(
    *,
    primary_goals: list[dict[str, Any]],
    value_condition: str,
    exchange_name: str,
    lifecycle: LifecycleConfig,
    max_budget_usd: str,
    terminal_file: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Construct the fixed nine-module topology and frozen state schemas."""

    fast = MODEL_POLICY.fast
    deliberative = MODEL_POLICY.deliberative
    states = {
        "llreasoner_0": _role_state(
            role="ll_reasoner",
            text_state="No observation or Goal Graph goal has been received.",
            profile=fast,
            lifecycle=lifecycle,
            max_budget_usd=max_budget_usd,
            initial_call_pending=True,
            value_condition=value_condition,
            extras={
                "current_observation": None,
                "action_observation_state": {"last_action": None, "last_action_status": None,
                                             "observation_sequence": 0, "observation_received_at": None,
                                             "observation_after_action": False},
                "progress_repair_attempts": 0,
                "progress_feedback_history": [],
                "active_goals": [],
                "active_cycle_id": None,
                "cycle_sequence": 0,
                "action_sequence": 0,
                "status_processed": 0,
                "scientific_complete": False,
                "scientific_completion_evidence": None,
                "suppressed_dispatches": [],
                "learner_requested": False,
                "learner_model": None,
                "used_revision_ids": [],
            },
        ),
        "knowledge_0": _role_state(
            role="knowledge",
            text_state="No grounded world evidence has been evaluated.",
            profile=deliberative,
            lifecycle=lifecycle,
            max_budget_usd=max_budget_usd,
            initial_call_pending=False,
            value_condition=value_condition,
            extras={"evaluation_sequence": 0, "evaluation_ids": []},
        ),
        "hlreasoner_0": _role_state(
            role="hl_reasoner",
            text_state="Ordered primary desires:\n"
            + json.dumps(primary_goals, ensure_ascii=False),
            profile=deliberative,
            lifecycle=lifecycle,
            max_budget_usd=max_budget_usd,
            initial_call_pending=True,
            value_condition=value_condition,
            extras={
                "primary_desires": primary_goals,
                "authored_goals": [],
                "learner_requested": False,
                "learner_model": None,
                "used_revision_ids": [],
            },
        ),
        "goalgraph_0": _role_state(
            role="goal_graph",
            text_state="The goal graph is initially empty.",
            profile=fast,
            lifecycle=lifecycle,
            max_budget_usd=max_budget_usd,
            initial_call_pending=False,
            value_condition=value_condition,
            extras={"goals": {}, "progress_ids": [], "relay_repair_attempts": 0,
                    "relay_feedback_history": []},
        ),
        "memory_0": _role_state(
            role="memory",
            text_state="Experience memory is initially empty.",
            profile=deliberative,
            lifecycle=lifecycle,
            max_budget_usd=max_budget_usd,
            initial_call_pending=False,
            value_condition=value_condition,
            extras={"memories": [], "memory_sequence": 0, "memory_repair_attempts": 0,
                    "memory_feedback_history": []},
        ),
    }
    for learner_id, role in (("learner_0", "ll_learner"), ("learner_1", "hl_learner")):
        states[learner_id] = _role_state(
            role=role,
            text_state="No learning request has been received.",
            profile=fast,
            lifecycle=lifecycle,
            max_budget_usd=max_budget_usd,
            initial_call_pending=False,
            value_condition=value_condition,
            extras={
                "involvement_requested_by_reasoner": False,
                "involvement_status": "not_requested",
                "revision_used_by_reasoner": False,
                "operational_involvement_decision": "not_requested",
                "counterfactual_necessity": "not_established",
                "current_model": None,
                "revision_id": None,
                "revision_sequence": 0,
            },
        )
    perceptor_state = adapter_state(lifecycle=lifecycle.as_dict(), kind="perceptor")
    actuator_state = adapter_state(lifecycle=lifecycle.as_dict(), kind="actuator")
    states["perceptor_0"] = perceptor_state
    states["actuator_0"] = actuator_state

    for state in states.values():
        state["terminal_file"] = terminal_file

    value_text = VALUE_CONDITIONS[value_condition]
    modules = {
        "perceptor": TextBlocksWorldPerceptor(
            module_id=module_name(PERCEPTOR, 0),
            initial_state=perceptor_state,
            exchange_name=exchange_name,
        ),
        "actuator": StringBlocksWorldActuator(
            module_id=module_name(ACTUATOR, 0),
            initial_state=actuator_state,
            exchange_name=exchange_name,
        ),
        "ll_reasoner": LLMLowLevelReasoner(
            module_id=module_name(LLREASONER, 0),
            initial_state=states["llreasoner_0"],
            init_kwargs=_role_init(
                role="ll_reasoner",
                profile=fast,
                value_system="",
                lifecycle=lifecycle,
                max_budget_usd=max_budget_usd,
            ),
        ),
        "knowledge": LLMKnowledge(
            module_id=module_name(KNOWLEDGE, 0),
            initial_state=states["knowledge_0"],
            init_kwargs=_role_init(
                role="knowledge",
                profile=deliberative,
                value_system=value_text,
                lifecycle=lifecycle,
                max_budget_usd=max_budget_usd,
            ),
        ),
        "hl_reasoner": LLMHighLevelReasoner(
            module_id=module_name(HLREASONER, 0),
            initial_state=states["hlreasoner_0"],
            init_kwargs=_role_init(
                role="hl_reasoner",
                profile=deliberative,
                value_system="",
                lifecycle=lifecycle,
                max_budget_usd=max_budget_usd,
            ),
        ),
        "goal_graph": LLMGoalGraph(
            module_id=module_name(GOALGRAPH, 0),
            initial_state=states["goalgraph_0"],
            init_kwargs=_role_init(
                role="goal_graph",
                profile=fast,
                value_system="",
                lifecycle=lifecycle,
                max_budget_usd=max_budget_usd,
            ),
        ),
        "memory": LLMMemory(
            module_id=module_name(MEMORY, 0),
            initial_state=states["memory_0"],
            init_kwargs=_role_init(
                role="memory",
                profile=deliberative,
                value_system="",
                lifecycle=lifecycle,
                max_budget_usd=max_budget_usd,
            ),
        ),
        "learners": [
            LLMLearner(
                module_id=module_name(LEARNER, index),
                initial_state=states[f"learner_{index}"],
                init_kwargs=_role_init(
                    role="ll_learner" if index == 0 else "hl_learner",
                    profile=fast,
                    value_system="",
                    lifecycle=lifecycle,
                    max_budget_usd=max_budget_usd,
                ),
            )
            for index in range(2)
        ],
    }
    schemas = {module_id: sorted(state) for module_id, state in states.items()}
    return modules, schemas


def _local_mhagenta_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[5]
    mha_root = (workspace_root.parent / "mhagenta").resolve()
    version_file = mha_root / "pyproject.toml"
    if not version_file.is_file() or 'version = "1.4.12"' not in version_file.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(f"Expected local MHAgentA 1.4.12 at {mha_root}")
    return mha_root


def _run_agent(
    *,
    run: int,
    exp_path: Path,
    mha_version: str,
    value_condition: str,
    runtime_secret: RuntimeSecretFile,
    on_secret_mounted: Any,
    max_budget_usd: str,
    treatment_digest: str,
) -> bool:
    if mha_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(
            f"2-7-BW requires MHAgentA {DEFAULT_MHAGENTA_VERSION}, got {mha_version!r}"
        )
    mha_root = _local_mhagenta_root()
    lifecycle = LifecycleConfig()
    exchange_name = "mhagenta"
    run_agent_id = agent_name(run, f"2_7_{value_condition}")
    run_env_id = env_name(run, f"2_7_{value_condition}")
    primary_goals = _goals_for_run(exp_path.parent, run)
    modules, state_schemas = _build_agent_modules(
        primary_goals=primary_goals,
        value_condition=value_condition,
        exchange_name=exchange_name,
        lifecycle=lifecycle,
        max_budget_usd=max_budget_usd,
        terminal_file=f"/{Orchestrator.SAVE_SUBDIR}/control/terminal.json",
    )
    atomic_json(exp_path / f"state-schemas-{run}.json", state_schemas)

    bw_spec = find_spec("mha_env_blocksworld")
    if bw_spec is None or bw_spec.origin is None:
        raise ImportError("Could not locate mha_env_blocksworld runtime sources")
    bw_runtime_source = Path(bw_spec.origin).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    orchestrator = RuntimeSecretOrchestrator(
        save_dir=exp_path,
        step_frequency=STEP_FREQUENCY,
        control_frequency=0.05,
        status_frequency=5.0,
        agent_start_delay=20.0,
        exec_duration=lifecycle.agent_exec_duration,
        module_term_timeout=lifecycle.module_term_timeout,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        stop_on_agents_term=True,
        no_stdout_logs=False,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange_name,
        state_autosave_interval=30,
        runtime_secret_mounts={run_agent_id: runtime_secret},
        runtime_secret_mounted_callback=on_secret_mounted,
    )
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=modules["perceptor"],
        actuators=modules["actuator"],
        ll_reasoners=modules["ll_reasoner"],
        learners=modules["learners"],
        knowledge=modules["knowledge"],
        hl_reasoners=modules["hl_reasoner"],
        goal_graphs=modules["goal_graph"],
        memory=modules["memory"],
        requirements_path=Path(__file__).resolve().with_name("requirements.txt"),
        extra_runtime_sources=common_source,
    )
    orchestrator.add_environment(
        base=LLMBlocksWorldEnvironment(
            environment_initial_state(
                seed=Seeder(run).environment,
                value_condition=value_condition,
                primary_goals=primary_goals,
                environment_id=run_env_id,
            )
        ),
        env_id=run_env_id,
        exec_duration=lifecycle.environment_exec_duration,
        requirements_path=Path(__file__).resolve().with_name("requirements-env.txt"),
        exchange_name=exchange_name,
        extra_runtime_sources=[common_source, bw_runtime_source],
    )
    orchestrator.run(
        mhagenta_version=mha_version,
        local_build=mha_root,
        force_run=True,
    )

    states = gather_states(exp_path, False, no_warnings=True)
    log_path = (exp_path / run_agent_id).with_suffix(".log")
    logs = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
    agent_out = exp_path / run_agent_id / Orchestrator.SAVE_SUBDIR
    env_out = exp_path / run_env_id / Orchestrator.SAVE_SUBDIR
    result = evaluate_run(
        agent_states=states[run_agent_id],
        environment_state=states[run_env_id][run_env_id],
        trace_path=env_out / "environment-transitions.jsonl",
        response_dir=agent_out / "llm_responses",
        primary_goals=primary_goals,
        value_condition=value_condition,
        state_schemas=state_schemas,
        treatment_digest=treatment_digest,
        logs=logs,
    )
    result["run"] = run
    result["result_path"] = f"{run_agent_id}/out/result.json"
    atomic_json(agent_out / "result.json", result)
    (agent_out / "result.md").write_text(run_markdown(result), encoding="utf-8")
    return bool(result["execution_valid"]
                and result["architecture_checks"]["all_sent_boundaries_accounted"]
                and result["comparability_checks"]["lifecycle_complete"])


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str,
    *,
    value_condition: str,
    encoded_key: str,
    max_budget_usd: str = MAX_AGENT_BUDGET_USD,
    treatment_digest: str,
) -> bool:
    """Run one configured-key execution through the common exact cleanup boundary."""

    if value_condition not in VALUE_CONDITIONS:
        raise ValueError(f"unknown value condition {value_condition!r}")
    path = Path(exp_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    run_agent_id = agent_name(run, f"2_7_{value_condition}")
    run_env_id = env_name(run, f"2_7_{value_condition}")
    workspace_root = Path(__file__).resolve().parents[5]

    def body(secret: RuntimeSecretFile, mounted: Any) -> bool:
        return _run_agent(
            run=run,
            exp_path=path,
            mha_version=mha_version,
            value_condition=value_condition,
            runtime_secret=secret,
            on_secret_mounted=lambda _agent_id: mounted(),
            max_budget_usd=max_budget_usd,
            treatment_digest=treatment_digest,
        )

    paid_evidence = path / f"paid-run-{run}.json"
    try:
        return configured_paid_run(
            encoded_key=encoded_key,
            evidence_path=paid_evidence,
            agent_id=run_agent_id,
            environment_id=run_env_id,
            forbidden_roots=(workspace_root, path, path / run_agent_id),
            body=body,
        )
    finally:
        _bind_paid_run_evidence(
            result_path=path / run_agent_id / Orchestrator.SAVE_SUBDIR / "result.json",
            evidence_path=paid_evidence,
        )


def _bind_paid_run_evidence(*, result_path: Path, evidence_path: Path) -> None:
    """Make cleanup validity part of the run's matched-inclusion decision."""

    if not result_path.is_file() or not evidence_path.is_file():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    paid = json.loads(evidence_path.read_text(encoding="utf-8"))
    cleanup_valid = bool(paid.get("cleanup_success") and not paid.get("failures"))
    checks = result["comparability_checks"]
    checks["runtime_cleanup_valid"] = cleanup_valid
    result["cohort_comparable"] = bool(result.get("execution_valid", True) and all(
        bool(value) for key, value in checks.items() if key != "no_schema_repairs"))
    result["matched_inclusion"] = bool(
        result["architecture_valid"] and result["cohort_comparable"]
    )
    result["evidence"]["paid_run"] = {
        "path": evidence_path.name,
        "schema_version": paid.get("schema_version"),
        "sha256": canonical_sha256(paid),
        "cleanup_success": cleanup_valid,
    }
    atomic_json(result_path, result)
    if "scientific_outcomes" in result:
        result_path.with_suffix(".md").write_text(
            run_markdown(result), encoding="utf-8"
        )


def _collect_condition_results(path: Path, condition: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for result_path in sorted(path.glob("*/out/result.json")):
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if value.get("schema_version") not in {"2-7-bw-result-v1", "2-7-bw-result-v2"}:
            continue
        results.append(value)
    summary = condition_summary(condition, results)
    atomic_json(path / "condition-summary.json", summary)
    (path / "condition-summary.md").write_text(
        condition_markdown(summary), encoding="utf-8"
    )
    return summary


def _rebuild_summaries(root: Path, conditions: Sequence[str]) -> None:
    summaries = {
        condition: _collect_condition_results(root / condition, condition)
        for condition in conditions
    }
    matched = matched_summary(summaries)
    atomic_json(root / "matched-condition-summary.json", matched)
    (root / "matched-condition-summary.md").write_text(
        matched_markdown(matched), encoding="utf-8"
    )


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 5,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    conditions: Sequence[str] = THREE_VALUE_CONDITIONS,
    max_budget_usd: str = MAX_AGENT_BUDGET_USD,
    process_only: bool = False,
) -> None:
    """Run or reprocess the frozen 5Ãƒâ€”3 matched scientific treatment."""

    if mha_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(
            f"2-7-BW requires MHAgentA {DEFAULT_MHAGENTA_VERSION}, got {mha_version!r}"
        )
    unknown = set(conditions) - VALUE_CONDITIONS.keys()
    if unknown:
        raise ValueError(f"unknown value conditions: {sorted(unknown)}")
    root = Path(exp_path).resolve()
    run_ids = _normalize_run_ids(runs)
    expected = [{
        "execution_id": f"run-{run}/value-{condition}",
        "run_id": run,
        "factors": {"value_condition": condition},
    } for condition in conditions for run in run_ids]
    api_key = ""
    encoded_key = ""
    primary_error: BaseException | None = None
    try:
        if process_only:
            if not root.is_dir():
                raise FileNotFoundError(root)
            _rebuild_summaries(root, conditions)
            return

        lifecycle = LifecycleConfig()
        treatment = _treatment_manifest(run_ids, lifecycle)
        treatment["per_module_budget_usd"] = str(max_budget_usd)
        treatment_digest = canonical_sha256(treatment)
        api_key = read_experiment_credential()
        preflight = perform_api_preflight(api_key)
        if not preflight["success"]:
            failure_path = root.parent / f"{root.name}-api-preflight-failed.json"
            atomic_json(failure_path, preflight)
            raise RuntimeError("2-7-BW exact-model API preflight failed")
        encoded_key = encode_api_key(api_key)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "matched-goals.json", _goal_manifest(run_ids))
        atomic_json(
            root / "treatment-manifest.json",
            {**treatment, "treatment_sha256": treatment_digest},
        )
        atomic_json(root / "architecture-contract.json", ARCHITECTURE_CONTRACT)
        atomic_json(root / "value-metric-spec.json", VALUE_METRIC_SPEC)
        atomic_json(root / "api-preflight.json", preflight)
        atomic_json(
            root / "budget-policy.json",
            {
                "pricing_policy_version": PRICING_POLICY_VERSION,
                "max_agent_budget_usd": str(max_budget_usd),
                "budget_source": "estimated",
                "scope": "per_module_instance",
                "credential_mode": "configured_key_runtime_mount",
            },
        )
        for condition in conditions:
            runner = partial(
                run_experiment,
                value_condition=condition,
                encoded_key=encoded_key,
                max_budget_usd=max_budget_usd,
                treatment_digest=treatment_digest,
            )
            run_experiment_batch(
                experiment_id=f"2-7-BW-{condition}",
                title=f"HYBRID LLM BLOCKS WORLD ({condition})",
                runs=list(run_ids),
                exp_path=root / condition,
                mha_version=mha_version,
                runner=runner,
                cleanup_before_run=False,
                stop_on_error=True,
            )
            _collect_condition_results(root / condition, condition)
        _rebuild_summaries(root, conditions)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        api_key = ""
        encoded_key = ""
        try:
            process_execution_metrics(root, expected_executions=expected)
        except Exception as reporting_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "Execution-metrics processing also failed: "
                f"{type(reporting_error).__name__}"
            )


def check_results(**kwargs: Any) -> bool:
    """Compatibility wrapper returning scientific matched-inclusion validity."""

    return bool(evaluate_run(**kwargs)["matched_inclusion"])


__all__ = [
    "ARCHITECTURE_CONTRACT",
    "LifecycleConfig",
    "MAX_AGENT_BUDGET_USD",
    "N_GOALS",
    "THREE_VALUE_CONDITIONS",
    "VALUE_CONDITIONS",
    "VALUE_METRIC_SPEC",
    "check_results",
    "generate_primary_goals",
    "normalize_action",
    "run_batch",
    "run_experiment",
    "symbolic_observation_to_text",
]
