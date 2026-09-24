"""Scientific-contract tests for the finite deliberative BDI experiment."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable

import pytest
from mhagenta import ActionStatus, Observation
from mhagenta.outboxes import HLOutbox
from mhagenta.utils import Directory, State
from unified_planning.plans import SequentialPlan

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_common.names import ACTUATOR, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import module_name
from mha_exp_level2_bw.exp2_3 import planning, runner


REPRESENTATIVE_OBSERVATION = [
    "HandEmpty()",
    "On(B0,t0)",
    "AtLoc(B0,t0)",
    "Clear(B0)",
    "On(B1,t1)",
    "AtLoc(B1,t1)",
    "Clear(B1)",
    "AtLoc(t0,t0)",
    "AtLoc(t1,t1)",
    "LeftOf(t0,t1)",
    "Above(t0)",
]


class RecordingOutbox:
    """Capture typed outbox calls without emulating MHAgentA transport."""

    def __init__(self) -> None:
        self.observation_requests: list[tuple[str, dict[str, Any]]] = []
        self.belief_updates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.action_requests: list[tuple[str, dict[str, Any]]] = []

    def request_observation(self, recipient: str, **kwargs: Any) -> None:
        self.observation_requests.append((recipient, dict(kwargs)))

    def send_beliefs(self, *args: Any, **kwargs: Any) -> None:
        self.belief_updates.append((args, dict(kwargs)))

    def request_action(self, recipient: str, **kwargs: Any) -> None:
        self.action_requests.append((recipient, dict(kwargs)))


class FakeState(dict[str, Any]):
    """Small state double exposing only the typed hook dependencies."""

    def __init__(self, values: dict[str, Any], directory: Any | None = None) -> None:
        super().__init__(deepcopy(values))
        self.outbox = RecordingOutbox()
        self.directory = directory


def _card(module_id: str) -> SimpleNamespace:
    return SimpleNamespace(module_id=module_id)


def _directory() -> SimpleNamespace:
    return SimpleNamespace(
        internal=SimpleNamespace(
            perception=[_card("perceptor_0")],
            actuation=[_card("actuator_0")],
            ll_reasoning=[_card("llreasoner_0")],
            knowledge=[_card("knowledge_0")],
            hl_reasoning=[_card("hlreasoner_0")],
        )
    )


def _facts(observation: list[str] = REPRESENTATIVE_OBSERVATION) -> set[str]:
    return planning.beliefs_to_facts(planning.parse_symbolic_observation(observation))


def _fixed_selector(goal: planning.GoalSpec) -> Callable[..., planning.GoalSpec]:
    def select(self: Any, state: FakeState, facts: set[str]) -> planning.GoalSpec:
        run = state["run"]
        run["option_count"] = 1
        run["intention"] = goal.as_dict()
        run["intention_initially_satisfied"] = False
        run["goal_fact"] = goal.fact
        return goal

    return select


def _reasoner_state() -> FakeState:
    return FakeState(runner.deliberative_initial_states()[HLREASONER], _directory())


def _reasoner(num_blocks: int = 2, table_len: int = 2) -> runner.DeliberativeBDIReasoner:
    reasoner = runner.DeliberativeBDIReasoner(module_id="hlreasoner_0", initial_state={})
    reasoner._log_func = lambda *_: None
    reasoner.on_init(
        seed=17,
        num_blocks=num_blocks,
        table_len=table_len,
        planner_timeout=5.0,
    )
    reasoner._actuator_id = "actuator_0"
    return reasoner


def test_closed_world_observation_conversion_is_typed_and_canonical() -> None:
    beliefs = planning.parse_symbolic_observation(REPRESENTATIVE_OBSERVATION)
    facts = planning.beliefs_to_facts(beliefs)

    assert facts == {
        "hand-empty()",
        "on(b0,t0)",
        "at-location(b0,t0)",
        "clear(b0)",
        "on(b1,t1)",
        "at-location(b1,t1)",
        "clear(b1)",
        "at-location(t0,t0)",
        "at-location(t1,t1)",
        "left-of(t0,t1)",
        "above(t0)",
    }
    assert beliefs[0].predicate == "HandEmpty"


@pytest.mark.parametrize(
    "observation",
    [["Unknown(B0)"], ["On(B0)"], ["not a fact"], [3]],
)
def test_observation_conversion_rejects_unknown_or_malformed_facts(
    observation: list[Any],
) -> None:
    with pytest.raises(ValueError):
        planning.parse_symbolic_observation(observation)  # type: ignore[arg-type]


def test_representative_problem_parses_against_the_atomic_domain() -> None:
    goal = planning.GoalSpec("b0", "b1")
    text = planning.build_problem_pddl(
        problem_name="representative",
        blocks=("b0", "b1"),
        locations=("t0", "t1"),
        facts=_facts(),
        goal=goal,
    )
    assert "(:goal (on b0 b1))" in text
    assert "(:metric minimize (elapsed-steps))" in text
    problem = planning._parse_problem(
        _facts(),
        goal,
        blocks=("b0", "b1"),
        locations=("t0", "t1"),
        problem_name="representative",
    )
    assert problem.action("pickup") is not None
    assert problem.action("putdown") is not None


def test_lpg_returns_one_valid_bounded_serialized_plan() -> None:
    result = planning.plan_blocks_world(
        _facts(),
        planning.GoalSpec("b0", "b1"),
        blocks=("b0", "b1"),
        locations=("t0", "t1"),
        timeout=5.0,
        problem_name="focused_lpg",
    )

    assert result.accepted
    assert result.planner == "lpg"
    assert result.sequential
    assert result.validation_status == "VALID"
    assert 0 < len(result.actions) <= result.sanity_bound
    assert {action["name"] for action in result.actions} <= set(planning.ACTION_TO_ENV)
    assert json.loads(json.dumps(result.as_dict())) == result.as_dict()


def test_private_lpg_boundary_uses_one_fixed_search_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Planner:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def __enter__(self) -> Planner:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def solve(self, problem: Any, timeout: float) -> str:
            assert problem == "problem"
            assert timeout == 2.0
            return "result"

    monkeypatch.setattr(planning, "OneshotPlanner", Planner)

    assert planning._run_lpg("problem", 2.0) == "result"
    assert calls == [{"name": "lpg", "params": planning.LPG_PARAMETERS}]


def test_plan_result_json_boundary_normalizes_action_argument_tuples() -> None:
    result = _accepted_result(
        ({"name": "pickup", "arguments": ("b0", "t0", "t0"), "env_action": 0},)
    )
    persisted = result.as_dict()
    assert persisted["actions"][0]["arguments"] == ["b0", "t0", "t0"]
    assert isinstance(persisted["actions"][0]["arguments"], list)
    json.dumps(persisted)


def _planning_result(status: str, plan: Any) -> SimpleNamespace:
    return SimpleNamespace(status=SimpleNamespace(name=status), plan=plan)


def _patch_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planning,
        "_parse_problem",
        lambda *args, **kwargs: SimpleNamespace(kind="problem-kind"),
    )


def _plan_with_patches() -> planning.PlanResult:
    return planning.plan_blocks_world(
        set(),
        planning.GoalSpec("b0", "b1"),
        blocks=("b0", "b1"),
        locations=("t0", "t1"),
        timeout=1.0,
        problem_name="patched",
    )


def test_planner_timeout_is_one_compact_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_problem(monkeypatch)
    calls = 0

    def timeout(*args: Any, **kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _planning_result("TIMEOUT", None)

    monkeypatch.setattr(planning, "_run_lpg", timeout)
    result = _plan_with_patches()
    assert calls == 1
    assert result.failure == "planner-did-not-solve"
    assert result.status == "TIMEOUT"


def test_planner_exception_is_one_compact_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_problem(monkeypatch)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(planning, "_run_lpg", explode)
    result = _plan_with_patches()
    assert result.failure is not None and result.failure.startswith("planner-exception:")


def test_nonsequential_plan_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_problem(monkeypatch)
    monkeypatch.setattr(
        planning,
        "_run_lpg",
        lambda *args, **kwargs: _planning_result("SOLVED_SATISFICING", object()),
    )
    assert _plan_with_patches().failure == "non-sequential-plan"


def test_overbound_plan_is_rejected_before_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_problem(monkeypatch)
    monkeypatch.setattr(planning, "plan_length_bound", lambda *args: -1)
    monkeypatch.setattr(
        planning,
        "_run_lpg",
        lambda *args, **kwargs: _planning_result(
            "SOLVED_SATISFICING", SequentialPlan([])
        ),
    )
    assert _plan_with_patches().failure == "plan-too-long"


def test_invalid_plan_is_rejected_by_the_single_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem(monkeypatch)
    monkeypatch.setattr(
        planning,
        "_run_lpg",
        lambda *args, **kwargs: _planning_result(
            "SOLVED_SATISFICING", SequentialPlan([])
        ),
    )

    class InvalidValidator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> InvalidValidator:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def validate(self, problem: Any, plan: Any) -> SimpleNamespace:
            return SimpleNamespace(status=SimpleNamespace(name="INVALID"))

    monkeypatch.setattr(planning, "PlanValidator", InvalidValidator)
    result = _plan_with_patches()
    assert result.failure == "plan-invalid"
    assert result.validation_status == "INVALID"


@pytest.mark.parametrize(
    ("action", "facts"),
    [
        (
            {"name": "pickup", "arguments": ["b0", "t0", "t0"], "env_action": 0},
            {
                "hand-empty()",
                "on(b0,t0)",
                "clear(b0)",
                "at-location(b0,t0)",
                "at-location(t0,t0)",
                "above(t0)",
            },
        ),
        (
            {"name": "putdown", "arguments": ["b0", "b1", "t1"], "env_action": 1},
            {"holding(b0)", "clear(b1)", "at-location(b1,t1)", "above(t1)"},
        ),
        (
            {"name": "moveleft", "arguments": ["t1", "t0"], "env_action": 2},
            {"above(t1)", "left-of(t0,t1)"},
        ),
        (
            {"name": "moveright", "arguments": ["t0", "t1"], "env_action": 3},
            {"above(t0)", "left-of(t0,t1)"},
        ),
    ],
)
def test_next_action_soundness_covers_every_atomic_action(
    action: dict[str, Any],
    facts: set[str],
) -> None:
    assert planning.action_soundness(action, facts) == (True, [])
    missing_fact = next(iter(facts))
    sound, missing = planning.action_soundness(action, facts - {missing_fact})
    assert not sound
    assert missing_fact in missing


def test_pickup_soundness_requires_the_support_at_the_action_location() -> None:
    action = {"name": "pickup", "arguments": ["b0", "b1", "t0"], "env_action": 0}
    facts = {
        "hand-empty()",
        "on(b0,b1)",
        "clear(b0)",
        "at-location(b0,t0)",
        "above(t0)",
    }
    sound, missing = planning.action_soundness(action, facts)
    assert not sound
    assert missing == ["at-location(b1,t0)"]


def test_supporting_modules_preserve_the_typed_observation_cycle() -> None:
    directory = _directory()
    ll = runner.SupportLLReasoner(module_id="llreasoner_0", initial_state={})
    ll._log_func = lambda *_: None
    ll._perceptor_id = "perceptor_0"
    ll._knowledge_id = "knowledge_0"
    ll_state = FakeState(runner.deliberative_initial_states()[LLREASONER], directory)

    ll.on_first(ll_state)
    assert ll_state.outbox.observation_requests == [("perceptor_0", {})]
    observation = Observation(REPRESENTATIVE_OBSERVATION)
    ll.on_observation(ll_state, "perceptor_0", observation)
    assert len(ll_state.outbox.belief_updates) == 1
    args, metadata = ll_state.outbox.belief_updates[0]
    assert args[0] == "knowledge_0"
    assert args[1] is observation
    assert metadata["observation_seq"] == 1

    knowledge = runner.ClosedWorldKnowledge(module_id="knowledge_0", initial_state={})
    knowledge_state = FakeState(
        runner.deliberative_initial_states()[KNOWLEDGE], directory
    )
    knowledge.on_observed_beliefs(
        knowledge_state,
        "llreasoner_0",
        observation,
        args[2],
        **metadata,
    )
    assert knowledge_state["revisions"] == 1
    assert knowledge_state["updates_forwarded"] == 1
    assert knowledge_state.outbox.belief_updates[0][0][0] == "hlreasoner_0"

    ll.on_action_status(
        ll_state,
        "actuator_0",
        ActionStatus({"action_id": "action-1", "legal": True}),
    )
    assert ll_state.outbox.observation_requests[-1] == (
        "perceptor_0",
        {"action_id": "action-1", "legal": True},
    )
    assert not ll_state.outbox.action_requests


def test_finite_bdi_lifecycle_completes_one_goal_in_the_real_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = BlocksWorldEnv(table_len=3, num_blocks=3, symbolic=True)
    environment.expose_snapshot = True
    observation, _ = environment.reset(seed=1000)
    facts = _facts(list(observation))
    goal = planning.generate_options(planning.block_names(3), facts)[0]
    result = planning.plan_blocks_world(
        facts,
        goal,
        blocks=planning.block_names(3),
        locations=planning.location_names(3),
        timeout=5.0,
        problem_name="finite_lifecycle",
    )
    assert result.accepted
    monkeypatch.setattr(runner, "plan_blocks_world", lambda *args, **kwargs: result)

    reasoner = _reasoner(num_blocks=3, table_len=3)
    reasoner._select_intention = MethodType(_fixed_selector(goal), reasoner)
    state = _reasoner_state()
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        planning.parse_symbolic_observation(list(observation)),
        observation_seq=1,
    )

    consumed = 0
    observation_seq = 1
    while state["run"]["phase"] == "executing":
        _, request = state.outbox.action_requests[consumed]
        consumed += 1
        observation, _, _, _, info = environment.step(request[runner.K_ACTION])
        observation_seq += 1
        reasoner.on_belief_update(
            state,
            "knowledge_0",
            planning.parse_symbolic_observation(list(observation)),
            observation_seq=observation_seq,
            action_id=request["action_id"],
            legal=bool(info["snapshot"].legal),
        )

    requests_at_completion = len(state.outbox.action_requests)
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        planning.parse_symbolic_observation(list(observation)),
        observation_seq=observation_seq + 1,
    )
    environment.close()

    assert state["run"]["phase"] == "complete"
    assert state["run"]["goal_fact"] == goal.fact
    assert state["run"]["pending_action"] is None
    assert state["run"]["failure"] is None
    assert len(state.outbox.action_requests) == requests_at_completion


def _accepted_result(actions: tuple[dict[str, Any], ...]) -> planning.PlanResult:
    return planning.PlanResult(
        planner="lpg",
        status="SOLVED_SATISFICING",
        elapsed_seconds=0.01,
        sequential=True,
        validation_status="VALID",
        sanity_bound=20,
        actions=actions,
        failure=None,
    )


def _start_reasoner(
    monkeypatch: pytest.MonkeyPatch,
    result: planning.PlanResult,
) -> tuple[runner.DeliberativeBDIReasoner, FakeState]:
    goal = planning.GoalSpec("b0", "b1")
    reasoner = _reasoner()
    reasoner._select_intention = MethodType(_fixed_selector(goal), reasoner)
    monkeypatch.setattr(runner, "plan_blocks_world", lambda *args, **kwargs: result)
    state = _reasoner_state()
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        planning.parse_symbolic_observation(REPRESENTATIVE_OBSERVATION),
        observation_seq=1,
    )
    return reasoner, state


def test_planner_failure_is_terminal_and_does_not_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> planning.PlanResult:
        nonlocal calls
        calls += 1
        return planning.PlanResult(
            "lpg", "TIMEOUT", 0.01, False, None, 20, (), "planner-did-not-solve"
        )

    monkeypatch.setattr(runner, "plan_blocks_world", fail)
    reasoner = _reasoner()
    reasoner._select_intention = MethodType(
        _fixed_selector(planning.GoalSpec("b0", "b1")), reasoner
    )
    state = _reasoner_state()
    beliefs = planning.parse_symbolic_observation(REPRESENTATIVE_OBSERVATION)
    reasoner.on_belief_update(state, "knowledge_0", beliefs, observation_seq=1)
    reasoner.on_belief_update(state, "knowledge_0", beliefs, observation_seq=2)
    assert calls == 1
    assert state["run"]["phase"] == "failed"
    assert not state.outbox.action_requests


def test_unsound_action_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    action = {"name": "putdown", "arguments": ["b0", "b1", "t1"], "env_action": 1}
    _, state = _start_reasoner(monkeypatch, _accepted_result((action,)))
    assert state["run"]["phase"] == "failed"
    assert state["run"]["failure"]["code"] == "unsound-next-action"
    assert not state.outbox.action_requests


@pytest.mark.parametrize(
    ("action_id", "legal", "code"),
    [("wrong-action", True, "correlation-error"), ("action-1", False, "illegal-action")],
)
def test_action_outcome_failures_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
    action_id: str,
    legal: bool,
    code: str,
) -> None:
    action = {"name": "pickup", "arguments": ["b0", "t0", "t0"], "env_action": 0}
    reasoner, state = _start_reasoner(monkeypatch, _accepted_result((action,)))
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        planning.parse_symbolic_observation(REPRESENTATIVE_OBSERVATION),
        observation_seq=2,
        action_id=action_id,
        legal=legal,
    )
    assert state["run"]["phase"] == "failed"
    assert state["run"]["failure"]["code"] == code
    assert len(state.outbox.action_requests) == 1


def test_plan_exhaustion_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    _, state = _start_reasoner(monkeypatch, _accepted_result(()))
    assert state["run"]["phase"] == "failed"
    assert state["run"]["failure"]["code"] == "plan-exhausted-before-goal"


def _valid_states() -> dict[str, dict[str, Any]]:
    treatment = runner.treatment_for_run(0)
    action = {"name": "pickup", "arguments": ["b0", "t0", "t0"], "env_action": 0}
    return {
        module_name(PERCEPTOR, 0): {
            "requests": 2,
            "observations": 2,
            "active_seconds": 0.01,
        },
        module_name(ACTUATOR, 0): {
            "requests": 1,
            "statuses": 1,
            "active_seconds": 0.01,
        },
        module_name(LLREASONER, 0): {
            "observation_requests": 2,
            "observations_received": 2,
            "observation_seq": 2,
            "belief_updates_sent": 2,
            "action_statuses": 1,
            "failure": None,
            "active_seconds": 0.01,
        },
        module_name(KNOWLEDGE, 0): {
            "revisions": 2,
            "updates_forwarded": 2,
            "active_seconds": 0.01,
        },
        module_name(HLREASONER, 0): {
            "treatment": treatment,
            "belief_updates": 2,
            "active_seconds": 0.01,
            "run": {
                "phase": "complete",
                "option_count": 1,
                "intention": {"top": treatment["top"], "bottom": treatment["bottom"]},
                "intention_initially_satisfied": False,
                "planner": _accepted_result((action,)).as_dict(),
                "plan_index": 1,
                "executions": [
                    {
                        "plan_index": 0,
                        "action_id": "action-1",
                        "sound": True,
                        "correlated": True,
                        "legal": True,
                        "observation_seq": 2,
                    }
                ],
                "pending_action": None,
                "goal_fact": f"on({treatment['top']},{treatment['bottom']})",
                "goal_observed_at": 2,
                "failure": None,
            },
        },
    }


def _valid_environment() -> dict[str, Any]:
    """Build independent world evidence for the synthetic checker fixture."""
    treatment = runner.treatment_for_run(0)
    env = BlocksWorldEnv(table_len=treatment["table_len"], num_blocks=treatment["num_blocks"], symbolic=True)
    observation, _ = env.reset(seed=treatment["seed"])
    env.close()
    goal = f"On({treatment['top']},{treatment['bottom']})"
    return {"treatment": treatment, "initial_world": list(observation),
            "world": [*observation, goal], "actions": 1, "observations": 2,
            "illegal_actions": 0, "closed": True, "close_requests": 1}


def _check_fixture(states: Any, logs: list[str], verbose: bool = False) -> bool:
    """Supply complete environment evidence to the existing agent mutations."""
    return runner.check_results(states, logs or ["fixture"], _valid_environment(), ["fixture"], verbose=verbose)


def test_checker_accepts_the_clean_finite_scientific_contract() -> None:
    assert _check_fixture(_valid_states(), [], verbose=False)


@pytest.mark.parametrize("field,value", [
    ("world", []), ("initial_world", []), ("actions", 2), ("observations", 3),
    ("illegal_actions", 1), ("closed", False), ("close_requests", 0),
    ("treatment", {}),
])
def test_checker_rejects_inconsistent_environment(field: str, value: Any) -> None:
    environment = _valid_environment()
    environment[field] = value
    assert not runner.check_results(_valid_states(), ["fixture"], environment, ["fixture"])


def test_checker_requires_environment_and_both_logs() -> None:
    states, environment = _valid_states(), _valid_environment()
    assert not runner.check_results(states, ["fixture"], None, ["fixture"])
    assert not runner.check_results(states, [], environment, ["fixture"])
    assert not runner.check_results(states, ["fixture"], environment, [])
    assert not runner.check_results(states, ["fixture"], environment, ["Traceback (most recent call last):"])
    assert not runner.check_results(states, ["fixture"], environment, ["fixture"], expected_treatment=runner.treatment_for_run(1))


def test_repeated_shutdown_closes_environment_and_sends_request_once(monkeypatch: pytest.MonkeyPatch) -> None:
    closes: list[bool] = []
    scheduled = []
    def timer(delay: float, callback: Any, args: tuple) -> Any:
        scheduled.append((delay, callback, args))
        return SimpleNamespace(daemon=False, start=lambda: None)
    monkeypatch.setattr(runner, "Timer", timer)
    environment = object.__new__(runner.TestEnvironment)
    environment._env = SimpleNamespace(close=lambda: closes.append(True))
    state = {"closed": False, "close_requests": 0}
    environment.on_action(state, "agent", action=runner.A_CLOSE)
    environment.on_action(state, "agent", action=runner.A_CLOSE)
    assert closes == [True]
    assert state == {"closed": True, "close_requests": 1}
    assert scheduled == [(0.05, runner.os.kill, (runner.os.getpid(), runner.signal.SIGTERM))]
    with pytest.raises(RuntimeError, match="after the environment closed"):
        environment.on_action(state, "agent", action=0)
    actuator = runner.BDIBlocksWorldActuator(module_id="actuator_0", initial_state={})
    actuator._env_id = "env"
    requests = []
    actuator.act = lambda *args, **kwargs: requests.append((args, kwargs))
    actuator.on_last({})
    actuator.on_last({})
    assert len(requests) == 1


def test_report_uses_persisted_treatment_and_environment(tmp_path: Path) -> None:
    from mha_exp_level2_bw.exp2_3.reporting import process_execution_metrics

    states, environment = _valid_states(), _valid_environment()
    for entity, values in (("exp_agent2_3_0", states), ("exp_env2_3_0", {"environment": environment})):
        out = tmp_path / entity / "out"
        out.mkdir(parents=True)
        for key, value in values.items():
            (out / f"{entity}.{key}.json").write_text(json.dumps(value))
        (tmp_path / f"{entity}.log").write_text("fixture\n")
    specs = [{"execution_id": "run-0", "run_id": 0, "factors": runner.treatment_for_run(0)}]
    report = process_execution_metrics(tmp_path, expected_executions=specs)
    assert report["executions"][0]["operationally_valid"]
    assert report["executions"][0]["certificate_passed"]
    assert report["analyses"]["planning_tasks"]["rows"][0]["task_dimensions"] == {"table_len": 4, "num_blocks": 6}
    (tmp_path / "exp_env2_3_0.log").unlink()
    report = process_execution_metrics(tmp_path, expected_executions=specs)
    assert not report["executions"][0]["operationally_valid"]


@pytest.mark.parametrize(
    "path",
    [
        (module_name(LLREASONER, 0), "observation_requests"),
        (module_name(PERCEPTOR, 0), "requests"),
        (module_name(PERCEPTOR, 0), "observations"),
        (module_name(LLREASONER, 0), "observations_received"),
        (module_name(LLREASONER, 0), "belief_updates_sent"),
        (module_name(KNOWLEDGE, 0), "revisions"),
        (module_name(KNOWLEDGE, 0), "updates_forwarded"),
        (module_name(HLREASONER, 0), "belief_updates"),
    ],
)
def test_checker_rejects_every_broken_observation_chain_node(
    path: tuple[str, str],
) -> None:
    states = _valid_states()
    states[path[0]][path[1]] += 1
    assert not _check_fixture(states, [], verbose=False)


@pytest.mark.parametrize(
    "path",
    [
        (module_name(ACTUATOR, 0), "requests"),
        (module_name(ACTUATOR, 0), "statuses"),
        (module_name(LLREASONER, 0), "action_statuses"),
    ],
)
def test_checker_rejects_every_broken_action_chain_node(path: tuple[str, str]) -> None:
    states = _valid_states()
    states[path[0]][path[1]] += 1
    assert not _check_fixture(states, [], verbose=False)


def test_checker_rejects_a_broken_observation_to_action_relation() -> None:
    states = _valid_states()
    for module_id, field in (
        (module_name(LLREASONER, 0), "observation_requests"),
        (module_name(PERCEPTOR, 0), "requests"),
        (module_name(PERCEPTOR, 0), "observations"),
        (module_name(LLREASONER, 0), "observations_received"),
        (module_name(LLREASONER, 0), "belief_updates_sent"),
        (module_name(KNOWLEDGE, 0), "revisions"),
        (module_name(KNOWLEDGE, 0), "updates_forwarded"),
        (module_name(HLREASONER, 0), "belief_updates"),
    ):
        states[module_id][field] += 1
    assert not _check_fixture(states, [], verbose=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda states: states.pop(module_name(PERCEPTOR, 0)),
        lambda states: states[module_name(ACTUATOR, 0)].update(active_seconds=0.0),
        lambda states: states[module_name(HLREASONER, 0)]["run"]["planner"].update(planner="enhsp"),
        lambda states: states[module_name(HLREASONER, 0)]["run"]["planner"].update(failure="timeout"),
        lambda states: states[module_name(HLREASONER, 0)]["run"]["planner"].update(sanity_bound=0),
        lambda states: states[module_name(HLREASONER, 0)]["run"]["executions"][0].update(sound=False),
        lambda states: states[module_name(HLREASONER, 0)]["run"]["executions"][0].update(correlated=False),
        lambda states: states[module_name(HLREASONER, 0)]["run"]["executions"][0].update(legal=False),
        lambda states: states[module_name(HLREASONER, 0)]["run"].update(goal_observed_at=None),
        lambda states: states[module_name(HLREASONER, 0)]["run"].update(pending_action={"action_id": "action-2"}),
        lambda states: states[module_name(HLREASONER, 0)]["run"].update(phase="executing"),
    ],
)
def test_checker_rejects_principal_scientific_failures(
    mutate: Callable[[dict[str, dict[str, Any]]], Any],
) -> None:
    states = _valid_states()
    mutate(states)
    assert not _check_fixture(states, [], verbose=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda states: states.update({module_name(PERCEPTOR, 0): []}),
        lambda states: states[module_name(HLREASONER, 0)].update(run=[]),
        lambda states: states[module_name(HLREASONER, 0)]["run"].update(executions=["bad-row"]),
    ],
)
def test_checker_reports_malformed_evidence_without_raising(
    mutate: Callable[[dict[str, dict[str, Any]]], Any],
) -> None:
    states = _valid_states()
    mutate(states)
    assert not _check_fixture(states, [], verbose=False)


def test_checker_rejects_runtime_failures_but_not_known_planner_warnings() -> None:
    assert _check_fixture(
        _valid_states(),
        ["[0][WARNING]::[planner]::pkg_resources is deprecated as an API."],
    )
    assert not _check_fixture(
        _valid_states(),
        ["[0][ERROR]::[hlreasoner_0]::Failed to save state"],
    )
    assert not _check_fixture(
        _valid_states(),
        ["[0][WARNING]::[hlreasoner_0]::Failed to send action request"],
    )


@pytest.mark.parametrize("phase", ["complete", "failed", "executing"])
def test_real_hl_state_dump_is_json_native_in_every_persistence_phase(
    phase: str,
) -> None:
    hl = deepcopy(_valid_states()[module_name(HLREASONER, 0)])
    run = hl["run"]
    run["phase"] = phase
    if phase == "failed":
        run["failure"] = {"code": "test-failure", "stage": "test", "message": "failure"}
    elif phase == "executing":
        run["pending_action"] = {"action_id": "action-2", "plan_index": 1}
    state = State(
        agent_id="agent",
        module_id="hlreasoner_0",
        time_func=lambda: 0.0,
        directory=Directory(),
        outbox=HLOutbox(),
        **hl,
    )
    dumped = state.dump()
    json.dumps(dumped)

    def assert_native(value: Any) -> None:
        assert not isinstance(value, (tuple, planning.GoalSpec, planning.PlanResult))
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                assert_native(item)
        elif isinstance(value, list):
            for item in value:
                assert_native(item)
        else:
            assert value is None or isinstance(value, (str, int, float, bool))

    assert_native(dumped)
    serialized = json.dumps(dumped)
    for removed in (
        "planning_attempts",
        "belief_history",
        "fact_snapshots",
        "trace",
        "action_history",
        "replans",
    ):
        assert removed not in serialized


def test_batch_summary_reads_only_the_compact_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "gather_states",
        lambda *args, **kwargs: {"exp_agent2_3_0": _valid_states()},
    )
    monkeypatch.setattr(runner, "read_run_evidence", lambda *args: (_valid_states(), ["fixture"], _valid_environment(), ["fixture"]))
    runner._print_batch_summary(tmp_path)
    output = capsys.readouterr().out
    assert "1 / 1 / 0 / 1" in output
    assert "mean LPG plan length: 1.000" in output
    assert "failures by code: {}" in output
