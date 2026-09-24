"""Experiment 2-7-CR profiles, assembly, execution, and batch entry points."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, cast

import mha_exp_common
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.names import ACTUATOR, GOALGRAPH, HLREASONER, KNOWLEDGE, LEARNER, LLREASONER, MEMORY, PERCEPTOR
from mha_exp_common.paid_run import atomic_json, configured_paid_run, encode_api_key, read_experiment_credential
from mha_exp_common.runtime_secret import RUNTIME_CREDENTIAL_PATH, RuntimeSecretFile, RuntimeSecretOrchestrator
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name
from mhagenta import Belief, Goal, Orchestrator

from .environment import CrafterPerceptor, DAYLIGHT_EFFECTS, LLMCrafterEnvironment, StringCrafterActuator, png_bytes, symbolic_observation_to_text
from .evaluate import evaluate_run, process_execution_metrics
from .llm import (
    CONTROL_TREATMENT, DETERMINISTIC_ACTION_ASSISTANCE, DETERMINISTIC_GOAL_ASSISTANCE,
    GOAL_CHAIN_POLICY_VERSION, HL_BASELINE_MODEL, KNOWLEDGE_ADMISSION_POLICY_VERSION,
    LL_BASELINE_MODEL, LL_SURVIVAL_HEALTH_THRESHOLD, MODEL_POLICY_VERSION,
    PRIMARY_ACHIEVEMENT_ORDER, PRIMARY_COMPLETION_ACHIEVEMENT, PRIMARY_GOAL_ID,
    PROMPT_VERSION, SYMBOLIC_OBSERVATION_FORMAT_VERSION, ModelPolicy, ModelProfile,
    initial_role_state, jsonable, select_model_policy,
)
from .prompts import ENVIRONMENT_PROMPT, ROLE_PROMPTS
from .roles import LLMGoalGraph, LLMHighLevelReasoner, LLMKnowledge, LLMLearner, LLMLowLevelReasoner, LLMMemory

DEFAULT_MHAGENTA_VERSION = "1.4.12"
PROTOCOL_VERSION = "2-7-cr-single-episode-stop-v2"
STANDARD_BEHAVIOR_DURATION = 1_200.0
EXTENDED_BEHAVIOR_DURATION = 2_400.0
DRAIN_DURATION = 180.0
ENVIRONMENT_FINALIZATION_GRACE = 10.0
API_REQUEST_TIMEOUT = 60.0
STEP_FREQUENCY = 0.25
CONTROL_FREQUENCY = 0.05
STATUS_FREQUENCY = 5.0
STARTUP_DELAY = 20.0
LL_MAX_OUTPUT_TOKENS = 2_048
DELIBERATIVE_MAX_OUTPUT_TOKENS = 4_096
HIGH_REASONING_MAX_OUTPUT_TOKENS = 8_192
OUTPUT_LIMIT_POLICY_VERSION = "2-7-cr-output-allowances-v3"
OBSERVATION_FORMATS = ("image", "symbolic")
VALUE_CONDITIONS: dict[str, str] = {
    "comfort": "You value comfort and dislike working hungry, thirsty, or tired; you would choose farming.",
    "exploration": "You like seeing new places and spending spare time in nature, especially near water.",
    "preparedness": "You dislike lacking needed items and value a safety supply for later.",
}


@dataclass(frozen=True)
class RunProfile:
    """Frozen duration and per-module approximate budget treatment."""
    name: Literal["standard", "extended"]
    behavior_duration: float
    knowledge_budget_usd: str
    other_budget_usd: str
    cohort_included: bool

    @property
    def agent_duration(self) -> float:
        return self.behavior_duration + DRAIN_DURATION

    @property
    def environment_duration(self) -> float:
        return self.agent_duration + ENVIRONMENT_FINALIZATION_GRACE


STANDARD_PROFILE = RunProfile("standard", STANDARD_BEHAVIOR_DURATION, "2", "2", True)
EXTENDED_PROFILE = RunProfile("extended", EXTENDED_BEHAVIOR_DURATION, "4", "2", False)


def primary_goal() -> Goal:
    """Return the single canonical experimental objective."""
    return Goal([Belief("Achievement", (PRIMARY_COMPLETION_ACHIEVEMENT,))],
                goal_id=PRIMARY_GOAL_ID, status="pending", primary=True, order=0)


def recipe_knowledge_text() -> str:
    """Load the exact canonical Crafter rules shipped with the environment."""
    from mha_env_crafter.crafter import constants
    return json.dumps({"collect": constants.collect, "place": constants.place,
                       "make": constants.make,
                       "notes": ["do affects only the faced tile", "crafting needs listed resources and nearby structures"]},
                      indent=2, sort_keys=True)


def _role_kwargs(role: str, profile: ModelProfile, run_profile: RunProfile,
                 *, value_system: str = "", reasoner_id: str | None = None) -> dict[str, Any]:
    budget = run_profile.knowledge_budget_usd if role == "knowledge" else run_profile.other_budget_usd
    result = {
        "role": role, "environment": ENVIRONMENT_PROMPT, "value_system": value_system,
        "model_candidates": list(profile.candidates), "reasoning_effort": profile.reasoning_effort,
        "max_output_tokens": (HIGH_REASONING_MAX_OUTPUT_TOKENS if profile.reasoning_effort == "high"
                              else LL_MAX_OUTPUT_TOKENS if role == "ll_reasoner"
                              else DELIBERATIVE_MAX_OUTPUT_TOKENS),
        "credential_path": RUNTIME_CREDENTIAL_PATH, "max_budget_usd": budget,
        "request_timeout": API_REQUEST_TIMEOUT, "behavior_duration": run_profile.behavior_duration,
        "drain_deadline": run_profile.agent_duration, "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}",
    }
    if reasoner_id is not None:
        result["reasoner_id"] = reasoner_id
    return result


def _state(role: str, text: str, profile: ModelProfile, run_profile: RunProfile) -> dict[str, Any]:
    budget = run_profile.knowledge_budget_usd if role == "knowledge" else run_profile.other_budget_usd
    state = initial_role_state(text, profile=profile, budget=budget)
    state.update({"role": role, "control_treatment": CONTROL_TREATMENT,
                  "protocol_version": PROTOCOL_VERSION,
                  "deterministic_action_assistance": DETERMINISTIC_ACTION_ASSISTANCE,
                  "deterministic_goal_assistance": DETERMINISTIC_GOAL_ASSISTANCE,
                  "ll_survival_health_threshold": LL_SURVIVAL_HEALTH_THRESHOLD,
                  "prompt_version": PROMPT_VERSION,
                  "symbolic_observation_format_version": SYMBOLIC_OBSERVATION_FORMAT_VERSION})
    return state


def build_agent_modules(observation_format: str, value_condition: str,
                        model_policy: ModelPolicy, run_profile: RunProfile,
                        exchange_name: str = "mhagenta") -> dict[str, Any]:
    """Build exactly nine modules with compact predeclared state."""
    value_system = VALUE_CONDITIONS[value_condition]
    fast, deliberate = model_policy.fast, model_policy.deliberative
    ll_state = _state("ll_reasoner", "Await the first observation. Prioritize the primary goal.", fast, run_profile)
    initial_goal = jsonable(primary_goal())
    ll_state.update({"latest_observation": None, "last_action_status": None,
                     "last_action": None, "current_model": LL_BASELINE_MODEL,
                     "primary_goal_achieved": False, "terminal_evidence": None,
                     "goal_graph_goals": [initial_goal], "active_goals": [dict(initial_goal)],
                     "survival_needs": {"health": 9, "food": 9, "drink": 9, "energy": 9},
                     "survival_override_active": False})
    hl_state = _state("hl_reasoner", f"Primary goal: {PRIMARY_COMPLETION_ACHIEVEMENT}", deliberate, run_profile)
    hl_state["current_model"] = HL_BASELINE_MODEL
    learner0 = _state("ll_learner", "No grounded memories yet.", fast, run_profile)
    learner0["requester"] = None
    learner0["current_model"] = LL_BASELINE_MODEL
    learner1 = _state("hl_learner", "No grounded memories yet.", fast, run_profile)
    learner1["requester"] = None
    learner1["current_model"] = HL_BASELINE_MODEL
    modules = {
        "perceptor": CrafterPerceptor(module_id=module_name(PERCEPTOR, 0), exchange_name=exchange_name,
            observation_format=observation_format, artifact_root=f"/{Orchestrator.SAVE_SUBDIR}",
            initial_state={"requests": 0, "observations": 0, "last_reference": None,
                           "observation_format": observation_format}),
        "actuator": StringCrafterActuator(module_id=module_name(ACTUATOR, 0), exchange_name=exchange_name,
            initial_state={"requests": 0, "statuses": 0, "rejected": 0, "terminal": False,
                           "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}"}),
        "ll_reasoner": LLMLowLevelReasoner(module_id=module_name(LLREASONER, 0), initial_state=ll_state,
            init_kwargs=_role_kwargs("ll_reasoner", fast, run_profile)),
        "knowledge": LLMKnowledge(module_id=module_name(KNOWLEDGE, 0),
            initial_state=_state("knowledge", recipe_knowledge_text() + "\nValue system: " + value_system, deliberate, run_profile),
            init_kwargs=_role_kwargs("knowledge", deliberate, run_profile, value_system=value_system)),
        "hl_reasoner": LLMHighLevelReasoner(module_id=module_name(HLREASONER, 0), initial_state=hl_state,
            init_kwargs=_role_kwargs("hl_reasoner", deliberate, run_profile)),
        "goal_graph": LLMGoalGraph(module_id=module_name(GOALGRAPH, 0),
            initial_state=_state("goal_graph", f"Primary goal: {PRIMARY_COMPLETION_ACHIEVEMENT}", fast, run_profile),
            init_kwargs=_role_kwargs("goal_graph", fast, run_profile)),
        "memory": LLMMemory(module_id=module_name(MEMORY, 0),
            initial_state=_state("memory", "Experience memory is initially empty.", deliberate, run_profile),
            init_kwargs=_role_kwargs("memory", deliberate, run_profile)),
        "learners": [
            LLMLearner(module_id=module_name(LEARNER, 0), initial_state=learner0,
                init_kwargs=_role_kwargs("ll_learner", fast, run_profile, reasoner_id="llreasoner_0")),
            LLMLearner(module_id=module_name(LEARNER, 1), initial_state=learner1,
                init_kwargs=_role_kwargs("hl_learner", fast, run_profile, reasoner_id="hlreasoner_0")),
        ],
    }
    ll_state["achievements"] = []
    for name in ("goal_graph", "hl_reasoner"):
        modules[name].initial_state["goal_ledger"] = {PRIMARY_GOAL_ID: initial_goal}
    return modules


def _environment_state(seed: int, observation_format: str, value_condition: str,
                       run_profile: RunProfile) -> dict[str, Any]:
    return {"seed": seed, "observation_format": observation_format,
            "protocol_version": PROTOCOL_VERSION,
            "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}", "value_condition": value_condition,
            "profile": asdict(run_profile), "control_treatment": CONTROL_TREATMENT,
            "primary_achievement_order": list(PRIMARY_ACHIEVEMENT_ORDER),
            "step_count": 0, "observation_requests": 0, "native_return": 0.0,
            "illegal_actions": 0, "terminal": False, "dead": False,
            "primary_goal_achieved": False, "highest_primary_achievement": None,
            "final_inventory": {}, "final_achievements": {}, "final_position": [0, 0],
            "video_path": None, "video_frames": 0}


def _local_mhagenta_root() -> Path:
    root = (Path(__file__).resolve().parents[6] / "mhagenta").resolve()
    version_file = root / "pyproject.toml"
    if not version_file.is_file() or 'version = "1.4.12"' not in version_file.read_text(encoding="utf-8"):
        raise RuntimeError(f"expected local MHAgentA 1.4.12 at {root}")
    return root


def discover_model_policy(api_key: str) -> ModelPolicy:
    """Resolve the canonical policy against models visible to the configured account."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
    try:
        return select_model_policy([item.id for item in client.models.list().data])
    finally:
        client.close()


def _runtime(run: int, subset: Path, mha_version: str, observation_format: str,
             value_condition: str, model_policy: ModelPolicy, run_profile: RunProfile,
             secret: RuntimeSecretFile, agent_id: str, environment_id: str,
             mounted: Any = None) -> bool:
    if mha_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f"2-7-CR requires MHAgentA {DEFAULT_MHAGENTA_VERSION}")
    runtime_root = subset / f"run-{run:03d}"
    runtime_root.mkdir(parents=True, exist_ok=True)
    modules = build_agent_modules(observation_format, value_condition, model_policy, run_profile)
    orchestrator = RuntimeSecretOrchestrator(
        save_dir=runtime_root, step_frequency=STEP_FREQUENCY, control_frequency=CONTROL_FREQUENCY,
        status_frequency=STATUS_FREQUENCY, agent_start_delay=STARTUP_DELAY,
        exec_duration=run_profile.agent_duration, save_format="json", log_level=Orchestrator.INFO,
        save_logs=True, no_stdout_logs=False, mas_rmq_uri="localhost:5672",
        stop_on_agents_term=True,
        mas_rmq_exchange_name="mhagenta", state_autosave_interval=30,
        runtime_secret_mounts={agent_id: secret},
        runtime_secret_mounted_callback=(lambda _agent_id: mounted()) if mounted else None)
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    orchestrator.add_agent(agent_id=agent_id, perceptors=modules["perceptor"],
        actuators=modules["actuator"], ll_reasoners=modules["ll_reasoner"],
        learners=modules["learners"], knowledge=modules["knowledge"],
        hl_reasoners=modules["hl_reasoner"], goal_graphs=modules["goal_graph"],
        memory=modules["memory"], requirements_path=Path(__file__).with_name("requirements.txt"),
        extra_runtime_sources=common_source)
    crafter_spec = find_spec("mha_env_crafter")
    if crafter_spec is None or crafter_spec.origin is None:
        raise ImportError("could not locate mha_env_crafter")
    orchestrator.add_environment(base=LLMCrafterEnvironment(_environment_state(
        Seeder(run).environment, observation_format, value_condition, run_profile)),
        env_id=environment_id, exec_duration=run_profile.environment_duration,
        requirements_path=Path(__file__).with_name("requirements-env.txt"), exchange_name="mhagenta",
        extra_runtime_sources=[common_source, Path(crafter_spec.origin).resolve().parent])
    orchestrator.run(mhagenta_version=mha_version, local_build=_local_mhagenta_root(), force_run=True)
    gathered = gather_states(runtime_root, False, no_warnings=True)
    agent_states = gathered.get(agent_id, {})
    environment_state = gathered.get(environment_id, {}).get(environment_id, {})
    logs: list[str] = []
    for runtime_id in (agent_id, environment_id):
        path = runtime_root / f"{runtime_id}.log"
        if path.is_file():
            logs.extend(path.read_text(encoding="utf-8").splitlines())
    evaluation = evaluate_run(agent_states, environment_state,
        runtime_root / agent_id / Orchestrator.SAVE_SUBDIR / "events", logs)
    video = runtime_root / environment_id / Orchestrator.SAVE_SUBDIR / str(environment_state.get("video_path", ""))
    evidence_valid = video.is_file() and int(environment_state.get("video_frames", 0)) == int(environment_state.get("step_count", 0)) + 1
    for module_id, state in agent_states.items():
        count = int(state.get("response_records", 0))
        if count:
            records = runtime_root / agent_id / Orchestrator.SAVE_SUBDIR / "llm_responses" / f"{module_id}.jsonl"
            evidence_valid = evidence_valid and records.is_file() and len(records.read_text().splitlines()) == count
    evaluation["evidence_valid"] = evidence_valid
    evaluation["execution_valid"] = evaluation["execution_valid"] and evidence_valid
    evaluation["cohort_comparable"] = evaluation["execution_valid"]
    evaluation.update({"run": run, "observation_format": observation_format,
                       "value_condition": value_condition, "profile": asdict(run_profile),
                       "protocol_version": PROTOCOL_VERSION,
                       "model_policy": model_policy.as_dict(),
                       "cohort_comparable": evaluation["cohort_comparable"] and run_profile.cohort_included})
    atomic_json(runtime_root / "evaluation.json", evaluation)
    return bool(evaluation["execution_valid"])


def run_experiment(run: int, exp_path: str | os.PathLike[str], mha_version: str,
                   *, observation_format: str, value_condition: str,
                   model_policy: ModelPolicy, encoded_key: str,
                   run_profile: RunProfile = STANDARD_PROFILE) -> bool:
    """Run one configured-key execution with no experiment-specific attempt protocol."""
    subset = Path(exp_path).resolve()
    suffix = f"2_7_{observation_format}_{value_condition}_{run_profile.name}"
    agent_id, environment_id = agent_name(run, suffix), env_name(run, suffix)
    return configured_paid_run(encoded_key=encoded_key,
        evidence_path=subset / f"run-{run:03d}" / "paid-run.json",
        agent_id=agent_id, environment_id=environment_id,
        forbidden_roots=[subset],
        body=lambda secret, mounted: _runtime(run, subset, mha_version,
            observation_format, value_condition, model_policy, run_profile,
            secret, agent_id, environment_id, mounted))


def _normalize_runs(runs: int | tuple[int, int] | Sequence[int]) -> tuple[int, ...]:
    if isinstance(runs, int):
        return tuple(range(runs))
    if isinstance(runs, tuple) and len(runs) == 2:
        return tuple(range(*runs))
    return tuple(int(item) for item in runs)


def initial_context(run: int) -> dict[str, Any]:
    """Hash both matched observation treatments for one environment seed."""
    from mha_env_crafter import CrafterEnv
    seed = Seeder(run).environment
    env = CrafterEnv(seed=seed, length=10_000, no_mobs=True, symbolic=False,
                     daylight_effects=DAYLIGHT_EFFECTS)
    env.reset()
    image = png_bytes(env.render())
    symbolic = symbolic_observation_to_text(env.symbolic_observation())
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return {"run": run, "crafter_seed": seed,
            "initial_raw_sha256": hashlib.sha256(image).hexdigest(),
            "initial_symbolic_sha256": hashlib.sha256(symbolic.encode()).hexdigest()}


def summarize(root: Path, run_profile: RunProfile) -> dict[str, Any]:
    """Write one small aggregate without reconstructing module internals."""
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"{run_profile.name}-*/run-*/evaluation.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            records.append(value)
    by_design: dict[str, dict[str, int]] = {}
    for record in records:
        key = f"{record.get('observation_format')}:{record.get('value_condition')}"
        item = by_design.setdefault(key, {"runs": 0, "architecture_valid": 0,
                                          "alive": 0, "primary_goal_achieved": 0})
        outcomes = record.get("scientific_outcomes", {})
        item["runs"] += 1
        item["architecture_valid"] += int(record.get("architecture_valid") is True)
        item["alive"] += int(outcomes.get("alive") is True)
        item["primary_goal_achieved"] += int(outcomes.get("primary_goal_achieved") is True)
    summary = {"schema_version": "2-7-cr-summary-v10", "profile": asdict(run_profile),
               "runs": len(records),
               "architecture_valid": sum(item["architecture_valid"] for item in by_design.values()),
               "cohort_comparable": sum(record.get("cohort_comparable") is True for record in records),
               "by_design": by_design}
    atomic_json(root / f"summary-{run_profile.name}.json", summary)
    return summary


def _batch(runs: int | tuple[int, int] | Sequence[int], exp_path: str | os.PathLike[str],
           mha_version: str, process_only: bool, *, run_profile: RunProfile,
           designs: Sequence[tuple[str, str]]) -> None:
    root = Path(exp_path).resolve()
    run_ids = _normalize_runs(runs)
    expected = [{
        "execution_id": (
            f"profile-{run_profile.name}/run-{run}/"
            f"observation-{observation_format}/value-{value_condition}"
        ),
        "run_id": run,
        "factors": {"profile": run_profile.name,
                    "protocol_version": PROTOCOL_VERSION,
                    "observation_format": observation_format,
                    "value_condition": value_condition},
    } for observation_format, value_condition in designs for run in run_ids]
    primary_error: BaseException | None = None
    try:
        if process_only:
            for observation_format, value_condition in designs:
                run_experiment_batch(experiment_id="2-7-CR", title="LLM CRAFTER",
                    runs=list(run_ids), exp_path=root / f"{run_profile.name}-{observation_format}-{value_condition}",
                    mha_version=mha_version, runner=lambda *_: True, process_only=True)
            summarize(root, run_profile)
            return
        api_key = read_experiment_credential()
        try:
            policy = discover_model_policy(api_key)
            encoded_key = encode_api_key(api_key)
        finally:
            api_key = ""
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / f"matched-context-{run_profile.name}.json", {
            "schema_version": "2-7-cr-matched-context-v10", "profile": asdict(run_profile),
            "protocol_version": PROTOCOL_VERSION,
            "model_policy_version": MODEL_POLICY_VERSION, "goal_chain_policy_version": GOAL_CHAIN_POLICY_VERSION,
            "knowledge_admission_policy_version": KNOWLEDGE_ADMISSION_POLICY_VERSION,
            "output_limit_policy_version": OUTPUT_LIMIT_POLICY_VERSION,
            "max_output_tokens": {"high_reasoning": HIGH_REASONING_MAX_OUTPUT_TOKENS,
                                  "low_level": LL_MAX_OUTPUT_TOKENS,
                                  "other_low_reasoning": DELIBERATIVE_MAX_OUTPUT_TOKENS},
            "control_treatment": CONTROL_TREATMENT,
            "deterministic_action_assistance": False, "deterministic_goal_assistance": True,
            "ll_survival_health_threshold": LL_SURVIVAL_HEALTH_THRESHOLD,
            "prompt_version": PROMPT_VERSION,
            "symbolic_observation_format_version": SYMBOLIC_OBSERVATION_FORMAT_VERSION,
            "designs": [list(item) for item in designs], "runs": [initial_context(run) for run in run_ids]})
        for observation_format, value_condition in designs:
            subset = root / f"{run_profile.name}-{observation_format}-{value_condition}"
            runner = partial(run_experiment, observation_format=observation_format,
                value_condition=value_condition, model_policy=policy, encoded_key=encoded_key,
                run_profile=run_profile)
            run_experiment_batch(experiment_id="2-7-CR", title="LLM CRAFTER",
                runs=list(run_ids), exp_path=subset, mha_version=mha_version, runner=runner,
                process_only=False, cleanup_before_run=False, stop_on_error=True)
        summarize(root, run_profile)
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


def run_batch(runs: int | tuple[int, int] | Sequence[int] = 5,
              exp_path: str | os.PathLike[str] = ".",
              mha_version: str = DEFAULT_MHAGENTA_VERSION,
              process_only: bool = False,
              observation_formats: Sequence[str] = OBSERVATION_FORMATS,
              conditions: Sequence[str] = tuple(VALUE_CONDITIONS)) -> None:
    """Run selected standard-profile cells through the public CLI."""
    if not observation_formats or not conditions:
        raise ValueError("At least one observation format and value condition is required")
    if set(observation_formats) - set(OBSERVATION_FORMATS):
        raise ValueError("Unknown observation format")
    if set(conditions) - set(VALUE_CONDITIONS):
        raise ValueError("Unknown value condition")
    if len(set(observation_formats)) != len(observation_formats) or len(set(conditions)) != len(conditions):
        raise ValueError("Duplicate design cells are not allowed")
    designs = [(observation, condition) for observation in observation_formats for condition in conditions]
    _batch(runs, exp_path, mha_version, process_only, run_profile=STANDARD_PROFILE, designs=designs)


def run_extended_batch(runs: int | tuple[int, int] | Sequence[int] = 1,
                       exp_path: str | os.PathLike[str] = ".",
                       mha_version: str = DEFAULT_MHAGENTA_VERSION,
                       process_only: bool = False) -> None:
    """Run the fixed symbolic Ã— comfort 40-minute demonstration profile."""
    _batch(runs, exp_path, mha_version, process_only, run_profile=EXTENDED_PROFILE,
           designs=[("symbolic", "comfort")])


__all__ = ["EXTENDED_PROFILE", "STANDARD_PROFILE", "RunProfile", "build_agent_modules",
           "discover_model_policy", "initial_context", "primary_goal", "recipe_knowledge_text",
           "run_batch", "run_experiment", "run_extended_batch", "summarize"]
