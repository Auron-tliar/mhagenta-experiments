"""Contract tests for the simplified 2-7-CR scientific treatment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mhagenta import ActionStatus, Belief, Goal, Observation
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LearnerBase, LLReasonerBase, MemoryBase

from mha_exp_level2_cr.exp2_7 import runner
from mha_exp_level2_cr.exp2_7.environment import (
    LLMCrafterEnvironment,
    frame_reference,
    symbolic_observation_to_text,
)
from mha_exp_level2_cr.exp2_7.evaluate import LLM_MODULE_IDS, MODULE_IDS, evaluate_run
from mha_exp_level2_cr.exp2_7.llm import (
    CRAFTER_ACTIONS, CONTROL_TREATMENT, DETERMINISTIC_ACTION_ASSISTANCE,
    DETERMINISTIC_GOAL_ASSISTANCE, LL_SURVIVAL_HEALTH_THRESHOLD,
    BeliefPayload, LLReasonerResponse, ModelPolicy, ModelProfile,
    jsonable, normalize_action, select_model_policy,
)
from mha_exp_level2_cr.exp2_7.roles import (
    LLMGoalGraph, LLMHighLevelReasoner, LLMKnowledge, LLMLearner,
    LLMLowLevelReasoner, LLMMemory,
)
from mha_exp_level2_cr.exp2_7.prompts import ROLE_PROMPTS, compose_prompt


def policy() -> ModelPolicy:
    return ModelPolicy(
        fast=ModelProfile("fast", ("gpt-5.4-nano-2026-03-17",)),
        deliberative=ModelProfile("deliberative", ("gpt-5.4-nano-2026-03-17",), "high"),
        available_model_ids=(
            "gpt-5.4-nano-2026-03-17",
        ),
    )


def test_profiles_freeze_duration_budget_and_cohort_treatment() -> None:
    standard, extended = runner.STANDARD_PROFILE, runner.EXTENDED_PROFILE
    assert (standard.behavior_duration, standard.agent_duration, standard.environment_duration) == (1200, 1380, 1390)
    assert (extended.behavior_duration, extended.agent_duration, extended.environment_duration) == (2400, 2580, 2590)
    assert (standard.other_budget_usd, standard.knowledge_budget_usd, standard.cohort_included) == ("2", "2", True)
    assert (extended.other_budget_usd, extended.knowledge_budget_usd, extended.cohort_included) == ("2", "4", False)


def test_standard_and_extended_entrypoints_have_fixed_designs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_batch", lambda *args, **kwargs: calls.append(kwargs))
    runner.run_batch(runs=5)
    runner.run_extended_batch(runs=1)
    assert len(calls[0]["designs"]) == 6
    assert calls[0]["run_profile"] is runner.STANDARD_PROFILE
    assert calls[1]["designs"] == [("symbolic", "comfort")]
    assert calls[1]["run_profile"] is runner.EXTENDED_PROFILE


def test_nano_reasoning_levels_match_every_module_state_and_request() -> None:
    """Ensure both saved treatment state and actual API kwargs use the requested roles."""
    selected = select_model_policy(["gpt-5.4-nano-2026-03-17"])
    modules = runner.build_agent_modules("symbolic", "comfort", selected, runner.STANDARD_PROFILE)
    cognitive = [modules[role] for role in ("ll_reasoner", "knowledge", "hl_reasoner", "goal_graph", "memory")]
    for module in [*cognitive, *modules["learners"]]:
        expected_effort = "high" if module.module_id in {"knowledge_0", "hlreasoner_0", "memory_0"} else "low"
        assert module.initial_state["model_profile"]["candidates"] == ["gpt-5.4-nano-2026-03-17"]
        assert module.initial_state["model_profile"]["reasoning_effort"] == expected_effort
        assert module.init_kwargs["model_candidates"] == ["gpt-5.4-nano-2026-03-17"]
        assert module.init_kwargs["reasoning_effort"] == expected_effort
        expected_limit = 8192 if expected_effort == "high" else 2048 if module.module_id == "llreasoner_0" else 4096
        assert module.init_kwargs["max_output_tokens"] == expected_limit
        assert module.init_kwargs["max_budget_usd"] == "2"
        assert module.init_kwargs["behavior_duration"] == 1200


def test_model_policy_keeps_canonical_visible_priority() -> None:
    selected = select_model_policy([
        "gpt-5.4-2026-03-05",
        "gpt-5.4-nano-2026-03-17",
    ])
    assert selected.fast.candidates == ("gpt-5.4-nano-2026-03-17",)
    assert selected.deliberative.candidates == ("gpt-5.4-nano-2026-03-17",)
    assert selected.fast.reasoning_effort == "low"
    assert selected.deliberative.reasoning_effort == "high"
    with pytest.raises(RuntimeError):
        select_model_policy(["unrelated-model"])


@pytest.mark.parametrize(
    ("value", "expected", "source"),
    [("move_left", "move_left", "canonical"), (" MOVE NORTH ", "move_up", "alias"),
     ("interact", "do", "alias"), ("make iron pickax", "make_iron_pickaxe", "fuzzy"),
     ("not an action at all", None, "rejected")],
)
def test_action_boundary_preserves_canonical_alias_and_fuzzy_behavior(value: str, expected: str | None, source: str) -> None:
    action, evidence = normalize_action(value)
    assert (action, evidence) == (expected, source)
    assert action is None or action in CRAFTER_ACTIONS


class FakeOutbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


class FakeState(dict):
    time = 0.0

    def __init__(self, value: dict[str, object]) -> None:
        super().__init__({"context": {}, "last_sent_payloads": {}, "goal_ledger": {},
                         "suppressed_messages": 0, "coalesced_observations": 0,
                         "role": "ll_reasoner", "achievements": [], "last_action_status": None,
                         **value})
        self.outbox = FakeOutbox()


def ll_state() -> FakeState:
    primary = jsonable(Goal(
        [Belief("Achievement", ("collect_diamond",))],
        goal_id="primary_collect_diamond",
    ))
    return FakeState({"survival_needs": {"health": 9, "food": 9, "drink": 9, "energy": 9},
        "survival_override_active": False, "goal_graph_goals": [primary], "active_goals": [primary],
        "latest_observation": None, "pending": [], "received": 0})


def test_local_survival_goal_overrides_and_restores_goal_graph_intention() -> None:
    module = LLMLowLevelReasoner(module_id="llreasoner_0", initial_state={})
    state = ll_state()
    module.on_observation(state, "perceptor_0", Observation("urgent"),
                          survival_needs={"health": 7, "food": 9, "drink": 9, "energy": 9})
    assert state["survival_override_active"] is True
    assert state["active_goals"][0]["extras"]["goal_id"] == "survive"
    module.on_observation(state, "perceptor_0", Observation("clear"),
                          survival_needs={"health": 8, "food": 1, "drink": 1, "energy": 1})
    assert state["survival_override_active"] is False
    assert state["active_goals"] == state["goal_graph_goals"]


def test_low_level_response_remains_llm_selected_and_uses_native_outboxes(tmp_path: Path) -> None:
    class Runtime:
        events = tmp_path / "events.jsonl"
        behavior_duration = 1200.0

        @staticmethod
        def call(state: FakeState) -> LLReasonerResponse:
            return LLReasonerResponse(text_state="updated", action="interact",
                beliefs=[BeliefPayload(predicate="visible", arguments=["tree"])])

    module = LLMLowLevelReasoner(module_id="llreasoner_0", initial_state={})
    module.runtime = Runtime()
    state = ll_state()
    state.update({"pending": [{"type": "observation"}], "text_state": "old",
                  "last_action": None, "sent": 0, "repair_feedback": None,
                  "semantic_attempts": 0, "halted": False})
    state["latest_observation"] = jsonable(Observation("current"))
    module.step(state)
    calls = state.outbox.calls
    assert calls[0][0] == "request_action"
    assert calls[0][2]["action"] == "do"
    assert calls[0][2]["normalization"] == "alias"
    assert calls[1][0] == "send_beliefs"
    assert state["pending"] == []
    assert state["text_state"] == "updated"
    json.dumps(state)


def test_primary_goal_status_stops_before_another_llm_call() -> None:
    module = LLMLowLevelReasoner(module_id="llreasoner_0", initial_state={})
    state = ll_state()
    state.update({"primary_goal_achieved": False, "terminal_evidence": None})

    module.on_action_status(
        state,
        "actuator_0",
        ActionStatus({
            "primary_goal_achieved": True,
            "requested_action": "interact",
            "canonical_action": "do",
            "new_achievements": ["collect_diamond"],
        }),
    )

    assert state["primary_goal_achieved"] is True
    assert state["terminal_evidence"]["new_achievements"] == ["collect_diamond"]
    assert [name for name, _, _ in state.outbox.calls] == ["terminate_agent"]
    module.step(state)
    assert [name for name, _, _ in state.outbox.calls] == ["terminate_agent"]


def symbolic_fixture() -> list[str]:
    values = ["Facing(R1) = true", "Sleeping() = false"]
    values.extend(f"Have({item}) = 9" for item in ("health", "food", "drink", "energy"))
    values.extend(["MadeOf(R1,tree) = true", "OccupiedBy(R1,none) = true"])
    return values


def test_symbolic_treatment_preserves_authoritative_predicates_and_faced_cell() -> None:
    source = symbolic_fixture()
    text = symbolic_observation_to_text(source, agent_position=(2, -1))
    assert json.dumps(source, ensure_ascii=False) in text
    assert "Facing location: R1 (dx=+1, dy=+0)" in text
    assert "collect wood from the faced tree" in text
    assert "Agent-relative position: (2, -1)" in text


def test_image_treatment_is_lossless_and_content_addressed(tmp_path: Path) -> None:
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    first = frame_reference(image, tmp_path, (0, 0))
    second = frame_reference(image, tmp_path, (0, 0))
    assert first == second
    paths = list((tmp_path / "observations").glob("*.png"))
    assert len(paths) == 1
    assert paths[0].stem == first["sha256"]


def test_module_assembly_is_nine_native_roles_with_per_module_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "recipe_knowledge_text", lambda: "rules")
    modules = runner.build_agent_modules("symbolic", "comfort", policy(), runner.STANDARD_PROFILE)
    flat = [modules[name] for name in ("perceptor", "actuator", "ll_reasoner", "knowledge",
                                       "hl_reasoner", "goal_graph", "memory")] + modules["learners"]
    assert len(flat) == 9
    assert isinstance(modules["ll_reasoner"], LLReasonerBase)
    assert isinstance(modules["knowledge"], KnowledgeBase)
    assert isinstance(modules["hl_reasoner"], HLReasonerBase)
    assert isinstance(modules["goal_graph"], GoalGraphBase)
    assert isinstance(modules["memory"], MemoryBase)
    assert all(isinstance(item, LearnerBase) for item in modules["learners"])
    assert modules["knowledge"].initial_state["max_budget_usd"] == "2"
    assert all(item.initial_state["max_budget_usd"] == "2" for item in
               [modules["ll_reasoner"], modules["hl_reasoner"], modules["goal_graph"],
                modules["memory"], *modules["learners"]])
    assert isinstance(modules["perceptor"].artifact_root, str)
    assert all(json.dumps(item.initial_state) for item in flat)


def test_environment_artifact_path_is_container_portable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LLMCrafterEnvironment, "_build", lambda self: None)
    environment = LLMCrafterEnvironment(runner._environment_state(
        1,
        "symbolic",
        "comfort",
        runner.STANDARD_PROFILE,
    ))

    assert isinstance(environment.artifact_root, str)


def test_host_constructed_artifact_paths_are_container_portable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "recipe_knowledge_text", lambda: "rules")
    modules = runner.build_agent_modules(
        "symbolic", "comfort", policy(), runner.STANDARD_PROFILE
    )
    monkeypatch.setattr(LLMCrafterEnvironment, "_build", lambda self: None)
    environment = LLMCrafterEnvironment(
        runner._environment_state(  # noqa: SLF001 - container payload fixture
            1, "symbolic", "comfort", runner.STANDARD_PROFILE
        )
    )

    assert isinstance(modules["perceptor"].artifact_root, str)
    assert isinstance(environment.artifact_root, str)


def test_treatment_flags_are_explicit_and_canonical() -> None:
    assert CONTROL_TREATMENT == "cr-ll-local-survival-grounded-v2"
    assert DETERMINISTIC_ACTION_ASSISTANCE is False
    assert DETERMINISTIC_GOAL_ASSISTANCE is True
    assert LL_SURVIVAL_HEALTH_THRESHOLD == 7


def test_prompts_match_direct_native_schemas_and_survival_setup() -> None:
    prompt = compose_prompt(role="ll_reasoner", environment="environment", text_state="state",
                            pending_messages=[], value_system="comfort")
    assert "health <= 7 OR food, drink, or energy equals 0" in prompt
    assert "no tactical" in prompt and "controller" in prompt
    assert "outbox routes" not in prompt
    assert "required route" not in prompt
    assert set(ROLE_PROMPTS) == {"ll_reasoner", "knowledge", "hl_reasoner", "goal_graph",
                                 "memory", "ll_learner", "hl_learner"}


def test_evaluation_separates_architecture_from_scientific_outcomes(tmp_path: Path) -> None:
    events = tmp_path / "events"
    events.mkdir()
    for module_id in LLM_MODULE_IDS:
        (events / f"{module_id}.jsonl").write_text(json.dumps({"kind": "llm_call", "outcome": "success"}) + "\n")
    edges = [("perceptor_0", "llreasoner_0", "send_observation"),
             ("llreasoner_0", "actuator_0", "request_action"),
             ("actuator_0", "llreasoner_0", "send_status"),
             ("llreasoner_0", "learner_0", "send_learner_task"),
             ("llreasoner_0", "learner_0", "request_model"),
             ("hlreasoner_0", "knowledge_0", "request_beliefs"),
             ("hlreasoner_0", "learner_1", "send_learner_task"),
             ("hlreasoner_0", "learner_1", "request_model"),
             ("learner_0", "memory_0", "request_memories"),
             ("learner_1", "memory_0", "request_memories")]
    for module_id, recipient, kind in edges:
        with (events / f"{module_id}.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": "send", "recipient": recipient, "type": kind}) + "\n")
    states = {module_id: {"calls": 1, "estimated_cost_usd": "0.1",
                          "budget_exhausted": False} for module_id in MODULE_IDS}
    result = evaluate_run(states, {"dead": False, "final_achievements": {"collect_wood": 1},
        "highest_primary_achievement": "collect_wood", "native_return": 1, "step_count": 3,
        "illegal_actions": 0, "final_inventory": {"health": 9}}, events)
    assert result["architecture_valid"] is True
    assert result["scientific_outcomes"]["primary_goal_achieved"] is False
    assert result["scientific_outcomes"]["highest_primary_achievement"] == "collect_wood"


@pytest.mark.parametrize("line", [
    "PydanticSerializationUnexpectedValue(input_value=KnowledgeResponse(text='comfort-critical'))",
    "PydanticSerializationUnexpectedValue(input_value='fatal traceback [ERROR]:: quoted text')",
    "[2026-09-06 12:58:49,445|3.815100|3.815000|0.0][INFO]::[root]::critical goal; fatal risk",
    "[2026-09-06 12:58:49,445|3.815100|3.815000|0.0][WARNING]::[root]::[ERROR]:: quoted text",
])
def test_log_checker_preserves_valid_runs_with_error_words(tmp_path: Path, line: str) -> None:
    """Model prose inside warnings must not invalidate a scientific failure."""
    states = {module_id: {} for module_id in MODULE_IDS}
    result = evaluate_run(states, {"dead": False}, tmp_path, [line])
    assert result["execution_valid"]
    assert "runtime_errors" not in result["architecture_reasons"]
    assert result["scientific_outcomes"]["termination_reason"] == "time_limit"


@pytest.mark.parametrize("line", [
    "Traceback (most recent call last):",
    "  + Exception Group Traceback (most recent call last):",
    *[f"[2026-09-06 12:58:49,445|3.815100|3.815000|0.0][{level}]::[root]::failed"
      for level in ("ERROR", "CRITICAL", "FATAL")],
])
def test_log_checker_rejects_actual_runtime_errors(tmp_path: Path, line: str) -> None:
    """Structural error records still stop the batch even without error prose."""
    result = evaluate_run({module_id: {} for module_id in MODULE_IDS}, {"dead": False}, tmp_path, [line])
    assert not result["execution_valid"]
    assert "runtime_errors" in result["architecture_reasons"]


def test_summary_aggregates_only_architecture_and_scientific_outcomes(tmp_path: Path) -> None:
    target = tmp_path / "standard-symbolic-comfort" / "run-000"
    target.mkdir(parents=True)
    (target / "evaluation.json").write_text(json.dumps({
        "observation_format": "symbolic", "value_condition": "comfort",
        "architecture_valid": True, "cohort_comparable": True,
        "scientific_outcomes": {"alive": True, "primary_goal_achieved": False}}))
    summary = runner.summarize(tmp_path, runner.STANDARD_PROFILE)
    assert summary["runs"] == 1
    assert summary["architecture_valid"] == 1
    assert summary["by_design"]["symbolic:comfort"] == {
        "runs": 1, "architecture_valid": 1, "alive": 1, "primary_goal_achieved": 0}


def test_retired_protocol_and_tactical_controller_symbols_are_absent() -> None:
    root = Path(runner.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for retired in ("LLMModuleMixin", "_TransportCommand", "MANDATORY_CAUSAL_EDGES",
                    "deterministic_action_assistance = True", "organization_cost"):
        assert retired not in source


def test_python_size_gates() -> None:
    root = Path(runner.__file__).parent
    production = list(root.glob("*.py"))
    tests = list(Path(__file__).parent.glob("test_exp2_7*.py"))
    prod_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in production)
    test_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in tests)
    assert prod_lines <= 6000
    assert test_lines <= 2500
    assert prod_lines + test_lines <= 8500
    assert max(len(path.read_text(encoding="utf-8").splitlines()) for path in production) <= 1500


def test_death_stops_agent_without_another_observation() -> None:
    """Death ends the one seeded episode and requests whole-agent termination."""
    module = LLMLowLevelReasoner(module_id="llreasoner_0")
    state = ll_state()
    module.on_action_status(state, "actuator_0", ActionStatus({"terminal": True, "dead": True}))
    assert state["termination_reason"] == "death"
    assert [name for name, _, _ in state.outbox.calls] == ["terminate_agent"]


def test_full_response_evidence_and_budget_stop(tmp_path: Path) -> None:
    """Retain exact text/response and image references, never inline image payloads."""
    from mha_exp_level2_cr.exp2_7.llm import RoleRuntime, initial_role_state
    runtime = RoleRuntime.__new__(RoleRuntime)
    runtime.module_id = "llreasoner_0"
    runtime.role = "ll_reasoner"
    runtime.environment = "test"
    runtime.value_system = ""
    runtime.models = ("gpt-5.4-nano-2026-03-17",)
    runtime.model_index = 0
    runtime.reasoning_effort = "low"
    runtime.max_output_tokens = 768
    runtime.behavior_duration = 1200
    runtime.drain_deadline = 1380
    runtime.artifact_root = tmp_path
    runtime.events = tmp_path / "events" / "llreasoner_0.jsonl"
    output = LLReasonerResponse(text_state="test", action="noop")
    response = SimpleNamespace(output_parsed=output, model_dump=lambda **_: {"id": "response-test", "output": output.model_dump()})
    usage = {"estimated_cost_usd": "2.01", "budget_exhausted": True, "input_tokens": 10, "output_tokens": 20}
    runtime.client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: response), usage_snapshot=lambda: usage)
    state = FakeState(initial_role_state("test", profile=policy().fast, budget="2"))
    reference = frame_reference(np.zeros((64, 64, 3), dtype=np.uint8), tmp_path, (0, 0))
    state["pending"] = [{"type": "observation", "content": reference}]
    assert runtime.call(state) is None
    assert state["termination_reason"] == "budget_exhausted"
    assert state["calls"] == 1
    raw = (tmp_path / "llm_responses" / "llreasoner_0.jsonl").read_text()
    record = json.loads(raw)
    assert record["raw_response"]["id"] == "response-test"
    assert record["prompt"] and record["pending_messages"][0]["content"] == reference
    assert "data:image" not in raw
    assert state.outbox.calls[0][0] == "terminate_agent"


@pytest.mark.parametrize("observation_format", ["symbolic", "image"])
def test_video_finalizes_single_partial_episode(tmp_path: Path, observation_format: str) -> None:
    """Both observation treatments retain initial and every action frame."""
    import imageio.v2 as imageio
    from mha_exp_level2_cr.exp2_7.environment import A_CLOSE
    initial = runner._environment_state(1000, observation_format, "comfort", runner.STANDARD_PROFILE)
    initial["artifact_root"] = str(tmp_path)
    environment = LLMCrafterEnvironment(initial)
    state = environment.state
    environment.on_observe(state, "agent")
    environment.on_action(state, "agent", action=0)
    environment.on_action(state, "agent", action=0)
    environment.on_action(state, "agent", action=A_CLOSE)
    environment.on_action(state, "agent", action=A_CLOSE)
    video = tmp_path / state["video_path"]
    reader = imageio.get_reader(video)
    assert reader.count_frames() == state["video_frames"] == 3
    assert reader.get_data(0).shape == (512, 512, 3)
    reader.close()
    assert len(list(tmp_path.glob("videos/*.mp4"))) == 1


@pytest.mark.parametrize("reason", ["death", "budget_exhausted", "time_limit"])
def test_scientific_limits_remain_valid_without_completed_cognitive_routes(tmp_path: Path, reason: str) -> None:
    """Natural termination does not require every cognitive route to finish."""
    states = {module_id: {"calls": 0, "budget_exhausted": reason == "budget_exhausted"} for module_id in MODULE_IDS}
    result = evaluate_run(states, {"dead": reason == "death", "final_achievements": {}}, tmp_path)
    assert result["execution_valid"] is True
    assert result["cohort_comparable"] is True
    assert result["scientific_outcomes"]["primary_goal_achieved"] is False
    assert result["scientific_outcomes"]["termination_reason"] == reason
    states["llreasoner_0"]["halted"] = True
    assert evaluate_run(states, {"dead": False}, tmp_path)["execution_valid"] is False


def test_public_cli_can_select_one_symbolic_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    """The requested checkpoint batch has ten seeds and exactly one design cell."""
    calls = []
    monkeypatch.setattr(runner, "_batch", lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.run_batch(runs=10, observation_formats=["symbolic"], conditions=["comfort"])
    assert calls[0][0][0] == 10
    assert calls[0][1]["designs"] == [("symbolic", "comfort")]
    assert calls[0][1]["run_profile"] is runner.STANDARD_PROFILE


def test_knowledge_coalesces_observations_without_losing_requests() -> None:
    """Retain one newest sensor payload and preserve an explicit belief request."""
    from mhagenta import Directory, State
    modules = runner.build_agent_modules("symbolic", "comfort", policy(), runner.STANDARD_PROFILE)
    module = modules["knowledge"]
    state = State(agent_id="test", module_id=module.module_id, time_func=lambda: 0,
                  directory=Directory(), outbox=FakeOutbox(), **module.initial_state)
    module.on_belief_request(state, "hlreasoner_0")
    for index in range(5):
        module.on_observed_beliefs(state, "llreasoner_0", Observation(str(index)), [], achievements=["place_table"])
    assert [item["type"] for item in state["pending"]] == ["belief_request", "belief_update"]
    assert state["pending"][-1]["observation"]["content"] == "4"
    assert state["coalesced_observations"] == 4
    assert state.dump()["context"]["environment_achievements"] == ["place_table"]


def test_goal_relay_preserves_identity_progress_direction_and_request_replies(tmp_path: Path) -> None:
    """A model cannot rename received goals, undo progress, or echo to its sender."""
    from mha_exp_level2_cr.exp2_7.llm import GoalGraphResponse, GoalPayload
    module = runner.build_agent_modules("symbolic", "comfort", policy(), runner.STANDARD_PROFILE)["goal_graph"]
    state = FakeState(module.initial_state)
    goal = Goal([Belief("Achievement", ("place_table",))], goal_id="table", status="active")
    response = GoalGraphResponse(text_state="stored", low_level_goals=[GoalPayload(goal_id="table", predicate="wrong")],
                                 high_level_goals=[GoalPayload(goal_id="table", predicate="wrong")])
    module.runtime = SimpleNamespace(call=lambda _: response, events=tmp_path / "events.jsonl")
    module.on_goal_update(state, "hlreasoner_0", [goal])
    module.step(state)
    assert len(state.outbox.calls) == 1
    assert state.outbox.calls[0][1][0] == "llreasoner_0"
    assert jsonable(state.outbox.calls[0][1][1][0]) == jsonable(goal)
    module.on_goal_update(state, "hlreasoner_0", [goal])
    module.step(state)
    assert len(state.outbox.calls) == 1
    module.on_goal_request(state, "llreasoner_0")
    module.step(state)
    assert len(state.outbox.calls) == 2
    achieved = Goal([Belief("Achievement", ("place_table",))], goal_id="table", status="achieved")
    module.on_goal_update(state, "llreasoner_0", [achieved])
    module.step(state)
    assert state.outbox.calls[-1][1][0] == "hlreasoner_0"
    module.on_goal_update(state, "hlreasoner_0", [goal])
    assert state["goal_ledger"]["table"]["extras"]["status"] == "achieved"


def test_memory_only_answers_requested_learner(tmp_path: Path) -> None:
    """Knowledge updates must not create unsolicited learner calls."""
    from mha_exp_level2_cr.exp2_7.llm import MemoryResponse
    module = runner.build_agent_modules("symbolic", "comfort", policy(), runner.STANDARD_PROFILE)["memory"]
    state = FakeState(module.initial_state)
    response = MemoryResponse(text_state="retained", low_level_memories=[BeliefPayload(predicate="fact")],
                              high_level_memories=[BeliefPayload(predicate="fact")])
    module.runtime = SimpleNamespace(call=lambda _: response, events=tmp_path / "events.jsonl")
    module.on_belief_update(state, "knowledge_0", [Belief("fact", ())])
    module.step(state)
    assert not state.outbox.calls
    module.on_memory_request(state, "learner_1")
    module.step(state)
    assert len(state.outbox.calls) == 1
    assert state.outbox.calls[0][1][0] == "learner_1"


def test_current_context_is_present_when_no_observation_arrives() -> None:
    """Instructions must carry the actual model and goal, not just refer to them."""
    prompt = compose_prompt(role="ll_reasoner", environment="test", text_state="old",
                            pending_messages=[{"type": "model", "sender": "learner_0"}],
                            call_requirements={"active_goals": ["primary_collect_diamond"], "current_model": "current-model"})
    assert "current-model" in prompt and "primary_collect_diamond" in prompt


@pytest.mark.parametrize("formats,conditions", [([], ["comfort"]), (["video"], ["comfort"]),
    (["symbolic"], ["unknown"]), (["symbolic", "symbolic"], ["comfort"])])
def test_public_cli_rejects_empty_unknown_or_duplicate_cells(formats, conditions) -> None:
    """Invalid design selection must fail before any account access or execution."""
    with pytest.raises(ValueError):
        runner.run_batch(runs=10, observation_formats=formats, conditions=conditions)
