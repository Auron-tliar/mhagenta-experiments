"""Focused validation for the reactive Blocks World experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from mhagenta import ActionStatus, Observation

from mha_exp_common.names import ACTUATOR, LLREASONER, PERCEPTOR
from mha_exp_common.utils import module_name
from mha_exp_level2_bw.exp2_1 import runner
from mha_exp_level2_bw.exp2_1.treatment import treatment_for_run


class FakeOutbox:
    """Capture typed outbox operations without emulating a transport protocol."""

    def __init__(self) -> None:
        self.action_requests: list[tuple[str, dict[str, Any]]] = []
        self.observation_requests: list[tuple[str, dict[str, Any]]] = []
        self.observations: list[tuple[str, Observation, dict[str, Any]]] = []
        self.statuses: list[tuple[str, ActionStatus, dict[str, Any]]] = []
        self.terminations: list[str] = []

    def request_action(self, recipient: str, **kwargs: Any) -> None:
        self.action_requests.append((recipient, dict(kwargs)))

    def request_observation(self, recipient: str, **kwargs: Any) -> None:
        self.observation_requests.append((recipient, dict(kwargs)))

    def send_observation(
        self,
        recipient: str,
        observation: Observation,
        **kwargs: Any,
    ) -> None:
        self.observations.append((recipient, observation, dict(kwargs)))

    def send_status(
        self,
        recipient: str,
        status: ActionStatus,
        **kwargs: Any,
    ) -> None:
        self.statuses.append((recipient, status, dict(kwargs)))

    def terminate_agent(self, reason: str) -> None:
        self.terminations.append(reason)


class FakeState(dict[str, Any]):
    """Dictionary state with the typed operations required by module hooks."""

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.outbox = FakeOutbox()


def _silent(*args: Any, **kwargs: Any) -> None:
    pass


def _environment(run: int) -> runner.TestEnvironment:
    treatment = treatment_for_run(run)
    environment = runner.TestEnvironment(runner._environment_initial_state(treatment, False))
    environment._log_func = _silent
    return environment


def _reasoner(run: int) -> runner.TestLLReasoner:
    treatment = treatment_for_run(run)
    reasoner = runner.TestLLReasoner(module_id="ll", initial_state={})
    reasoner._log_func = _silent
    reasoner.on_init(seed=treatment["reasoner_seed"], tasks=treatment["tasks"])
    reasoner._actuator_id = "actuator"
    reasoner._perceptor_id = "perceptor"
    return reasoner


def test_environment_close_is_idempotent() -> None:
    environment = _environment(21)

    environment.on_action(environment.state, "agent", action=runner.A_CLOSE)
    environment.on_action(environment.state, "agent", action=runner.A_CLOSE)

    assert environment.state[runner.K_CLOSED] is True
    assert environment.state["close_requests"] == 1


def _simulate(run: int) -> tuple[FakeState, dict[str, Any]]:
    treatment = treatment_for_run(run)
    environment = _environment(run)
    reasoner = _reasoner(run)
    state = FakeState(runner._reasoner_initial_state(treatment))

    for _ in range(runner.MAX_EPISODE_ACTIONS + 2):
        reasoner.on_observation(
            state,
            "perceptor",
            Observation(list(environment.state[runner.K_STATE])),
        )
        if state[runner.K_N_SUCCESSES] > 0:
            break
        _, request = state.outbox.action_requests[-1]
        _, status = environment.on_action(environment.state, "agent", **request)
        assert status is not None
        reasoner.on_action_status(state, "actuator", ActionStatus(status))

    return state, environment.state


def test_same_run_selects_the_same_distinct_goal() -> None:
    first, _ = _simulate(21)
    second, _ = _simulate(21)

    assert first["current_goal"] == second["current_goal"]
    assert first["current_goal"]["top"] != first["current_goal"]["bottom"]


def test_fifty_treatments_follow_the_thesis_seed_rotation() -> None:
    treatments = [treatment_for_run(run) for run in range(50)]

    assert [item["seed"] for item in treatments] == list(range(1000, 1050))
    assert [item["reasoner_seed"] for item in treatments] == list(range(30000, 30050))
    assert len({item["initial_state_digest"] for item in treatments}) == 50
    assert len({(item["top"], item["bottom"]) for item in treatments}) == 50
    assert all(item["task_count"] == 1 and len(item["tasks"]) == 1 for item in treatments)


def test_fifty_run_treatments_reach_goals_without_illegal_actions() -> None:
    reasons: set[str] = set()
    for run in range(50):
        state, environment = _simulate(run)
        reasons.update(item["reason"] for item in state["decision_trace"])

        assert state[runner.K_N_SUCCESSES] == 1, run
        assert len(state[runner.K_EP_LENGTHS]) == 1, run
        assert state[runner.K_EP_LENGTHS][0] <= runner.MAX_EPISODE_ACTIONS, run
        assert len(state["episode_results"]) == 1, run
        episode = state["episode_results"][0]
        assert episode["outcome"] == "success", run
        assert episode["failure_reason"] is None, run
        assert episode["actions"] == state[runner.K_EP_LENGTHS][0], run
        assert episode["decision_start"] <= episode["decision_end"], run
        assert state["current_goal"]["top"] != state["current_goal"]["bottom"], run
        assert state["phase"] == "complete", run
        assert state["terminal_reason"] == "goal_achieved", run
        assert state.outbox.terminations == ["2-1-BW fixed goal complete"], run
        assert not state["logical_failures"], run
        assert state["illegal_actions"] == 0, run
        assert environment["illegal_actions"] == 0, run
        assert all(
            item[runner.K_ACTION] in runner.NATIVE_ACTIONS
            for item in state["decision_trace"]
        ), run

    assert "excavate bottom block" in reasons
    assert "relocate bottom block" in reasons
    assert "excavate top block" in reasons
    assert "place top block on bottom block" in reasons


def test_episode_action_limit_records_failure_and_terminates() -> None:
    treatment = treatment_for_run(7)
    environment = _environment(7)
    reasoner = _reasoner(7)
    state = FakeState(runner._reasoner_initial_state(treatment))
    state["current_episode_length"] = runner.MAX_EPISODE_ACTIONS
    state["decision_trace"] = [
        {runner.K_ACTION: runner.A_MOVE_LEFT, "reason": "fixture"}
        for _ in range(runner.MAX_EPISODE_ACTIONS)
    ]

    reasoner.on_observation(
        state,
        "perceptor",
        Observation(list(environment.state[runner.K_STATE])),
    )

    assert state.outbox.action_requests == []
    assert state.outbox.terminations == ["2-1-BW fixed goal failed"]
    assert state["phase"] == "failed"
    assert state["terminal_reason"] == "episode action limit reached"
    assert state["logical_failures"][0]["reason"] == "episode action limit reached"
    assert state["episode_results"] == [{
        "episode_id": 0,
        "goal": state["current_goal"],
        "outcome": "failure",
        "failure_reason": "episode action limit reached",
        "actions": runner.MAX_EPISODE_ACTIONS,
        "elapsed_seconds": state["episode_results"][0]["elapsed_seconds"],
        "decision_start": 0,
        "decision_end": runner.MAX_EPISODE_ACTIONS - 1,
    }]


def _arm_column(environment: runner.TestEnvironment) -> int:
    above = next(
        item
        for item in environment.state[runner.K_STATE]
        if item.startswith(f"{runner.F_ABOVE}(")
    )
    return int(above.split("t", 1)[1].removesuffix(")"))


def _occupied_columns(environment: runner.TestEnvironment) -> list[int]:
    occupied: list[int] = []
    for predicate in environment.state[runner.K_STATE]:
        fluent, arguments = runner.TestLLReasoner._parse_predicate(predicate)
        if fluent == runner.F_ON and len(arguments) == 2 and arguments[1].startswith("t"):
            occupied.append(int(arguments[1][1:]))
    return occupied


def test_all_readable_actions_reach_the_real_environment_as_integer_values() -> None:
    environment = _environment(11)

    def apply(action: str) -> dict[str, Any]:
        _, status = environment.on_action(environment.state, "agent", action=action)
        assert status is not None
        assert environment._env._snapshot.action == environment._action_values[action]
        assert type(environment._env._snapshot.action) is int
        return status

    start = _arm_column(environment)
    if start > 0:
        assert apply(runner.A_MOVE_LEFT)[runner.K_LEGAL] is True
        assert apply(runner.A_MOVE_RIGHT)[runner.K_LEGAL] is True
    else:
        assert apply(runner.A_MOVE_RIGHT)[runner.K_LEGAL] is True
        assert apply(runner.A_MOVE_LEFT)[runner.K_LEGAL] is True

    target = _occupied_columns(environment)[0]
    while _arm_column(environment) < target:
        assert apply(runner.A_MOVE_RIGHT)[runner.K_LEGAL] is True
    while _arm_column(environment) > target:
        assert apply(runner.A_MOVE_LEFT)[runner.K_LEGAL] is True

    assert apply(runner.A_PICK_UP)[runner.K_LEGAL] is True
    assert apply(runner.A_PUT_DOWN)[runner.K_LEGAL] is True


def test_unknown_action_records_authoritative_failure_evidence() -> None:
    environment = _environment(9)

    _, status = environment.on_action(environment.state, "agent", action="Unknown")

    assert status is not None
    assert status[runner.K_LEGAL] is False
    assert runner.K_ERROR in status
    assert environment.state["status_actions"] == 1
    assert environment.state["native_actions"] == 0
    assert environment.state["illegal_actions"] == 1
    assert environment.state["errors"] == [status[runner.K_ERROR]]


def test_reset_and_close_remain_explicit_environment_controls() -> None:
    environment = _environment(13)
    close_calls: list[bool] = []
    environment._env.close = lambda: close_calls.append(True)

    _, reset = environment.on_action(environment.state, "agent", action=runner.A_RESET)
    _, closed = environment.on_action(environment.state, "agent", action=runner.A_CLOSE)

    assert reset == {runner.K_LEGAL: True}
    assert closed is None
    assert environment.state["status_actions"] == 1
    assert environment.state["native_actions"] == 0
    assert environment.state["close_requests"] == 1
    assert environment.state[runner.K_CLOSED] is True
    assert close_calls == [True]


def test_reactive_modules_preserve_one_typed_cycle_without_correlation_metadata() -> None:
    perceptor = runner.TestPerceptor(module_id="perceptor", initial_state={})
    perceptor._env_id = "environment"
    perceptor._reasoner_id = "reasoner"
    observed_requests: list[tuple[str, dict[str, Any]]] = []
    perceptor.observe = lambda env_id, **kwargs: observed_requests.append((env_id, kwargs))
    perceptor_state = FakeState(runner._perceptor_initial_state())

    perceptor.on_request(perceptor_state, "reasoner")
    perceptor.on_observation(perceptor_state, "environment", observation=["HandEmpty()"])

    assert observed_requests == [("environment", {})]
    assert perceptor_state.outbox.observations[0][2] == {}

    actuator = runner.TestActuator(module_id="actuator", initial_state={})
    actuator._env_id = "environment"
    actuator._reasoner_id = "reasoner"
    action_requests: list[tuple[str, dict[str, Any]]] = []
    actuator.act = lambda env_id, **kwargs: action_requests.append((env_id, kwargs))
    actuator_state = FakeState(runner._actuator_initial_state())

    actuator.on_request(actuator_state, "reasoner", action=runner.A_PICK_UP)
    actuator.on_status(actuator_state, "environment", legal=True)

    assert action_requests == [("environment", {runner.K_ACTION: runner.A_PICK_UP})]
    assert actuator_state.outbox.statuses[0][1].status == {runner.K_LEGAL: True}
    assert actuator_state.outbox.statuses[0][2] == {}

    reasoner = _reasoner(3)
    reasoner_state = FakeState(runner._reasoner_initial_state(treatment_for_run(3)))
    environment = _environment(3)
    reasoner.on_observation(
        reasoner_state,
        "perceptor",
        Observation(list(environment.state[runner.K_STATE])),
    )
    _, request = reasoner_state.outbox.action_requests[-1]
    _, status = environment.on_action(environment.state, "agent", **request)
    assert status is not None
    reasoner.on_action_status(reasoner_state, "actuator", ActionStatus(status))

    assert reasoner_state.outbox.observation_requests == [("perceptor", {})]
    assert type(reasoner_state["decision_trace"][-1][runner.K_LEGAL]) is bool


def _valid_result_fixture() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reasoner, environment = _simulate(5)
    observations = reasoner["observations"]
    actions = len(reasoner["decision_trace"])
    reasoner["active_seconds"] = max(reasoner["active_seconds"], 0.01)
    environment["observation_requests"] = observations
    environment["close_requests"] = 1
    environment[runner.K_CLOSED] = True

    states = {
        module_name(PERCEPTOR, 0): {
            **runner._perceptor_initial_state(),
            "requests": observations,
            "observations": observations,
            "active_seconds": 0.01,
        },
        module_name(ACTUATOR, 0): {
            **runner._actuator_initial_state(),
            "requests": actions,
            "statuses": actions,
            "active_seconds": 0.01,
        },
        module_name(LLREASONER, 0): dict(reasoner),
    }
    return states, environment


def test_result_checker_accepts_minimal_scientific_evidence() -> None:
    states, environment = _valid_result_fixture()
    assert runner.check_results(states, environment, [], verbose=False)


@pytest.mark.parametrize(
    ("requests_ahead", "environment_ahead", "perceptor_ahead"),
    [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
)
def test_result_checker_accepts_one_in_flight_observation_phase(
    requests_ahead: int,
    environment_ahead: int,
    perceptor_ahead: int,
) -> None:
    states, environment = _valid_result_fixture()
    reasoner = states[module_name(LLREASONER, 0)]
    perceptor = states[module_name(PERCEPTOR, 0)]
    completed = reasoner["observations"]
    perceptor["requests"] = completed + requests_ahead
    environment["observation_requests"] = completed + environment_ahead
    perceptor["observations"] = completed + perceptor_ahead

    assert runner.check_results(states, environment, [], verbose=False)


@pytest.mark.parametrize("phase", range(4))
def test_result_checker_accepts_one_in_flight_action_phase(phase: int) -> None:
    states, environment = _valid_result_fixture()
    reasoner = states[module_name(LLREASONER, 0)]
    actuator = states[module_name(ACTUATOR, 0)]
    completed = reasoner["action_statuses"]
    reasoner["decision_trace"].append(
        {
            runner.K_ACTION: runner.A_MOVE_LEFT,
            "reason": "pending test action",
            runner.K_LEGAL: None,
        }
    )
    actuator["requests"] = completed + int(phase >= 1)
    environment["status_actions"] = completed + int(phase >= 2)
    actuator["statuses"] = completed + int(phase >= 3)

    assert runner.check_results(states, environment, [], verbose=False)


def test_result_checker_rejects_missing_state_and_scientific_failures() -> None:
    states, environment = _valid_result_fixture()
    assert not runner.check_results(None, environment, [])
    assert not runner.check_results(states, None, [])

    missing_module = deepcopy(states)
    missing_module.pop(module_name(PERCEPTOR, 0))
    assert not runner.check_results(missing_module, environment, [])

    no_completion = deepcopy(states)
    no_completion[module_name(LLREASONER, 0)][runner.K_N_SUCCESSES] = 0
    no_completion[module_name(LLREASONER, 0)][runner.K_EP_LENGTHS] = []
    assert not runner.check_results(no_completion, environment, [])

    invalid_goal = deepcopy(states)
    invalid_goal[module_name(LLREASONER, 0)]["current_goal"] = {
        "top": "B01",
        "bottom": "B01",
    }
    assert not runner.check_results(invalid_goal, environment, [])

    logical_failure = deepcopy(states)
    logical_failure[module_name(LLREASONER, 0)]["logical_failures"] = [{"reason": "stuck"}]
    assert not runner.check_results(logical_failure, environment, [])

    illegal_environment = deepcopy(environment)
    illegal_environment["illegal_actions"] = 1
    illegal_environment["errors"] = ["bad action"]
    assert not runner.check_results(states, illegal_environment, [])

    inactive = deepcopy(states)
    inactive[module_name(ACTUATOR, 0)]["active_seconds"] = 0.0
    assert not runner.check_results(inactive, environment, [])


def test_result_checker_rejects_multiple_in_flight_actions_and_bad_cleanup() -> None:
    states, environment = _valid_result_fixture()
    divergent = deepcopy(states)
    reasoner = divergent[module_name(LLREASONER, 0)]
    for _ in range(2):
        reasoner["decision_trace"].append(
            {
                runner.K_ACTION: runner.A_MOVE_LEFT,
                "reason": "extra pending action",
                runner.K_LEGAL: None,
            }
        )
    assert not runner.check_results(divergent, environment, [])

    not_closed = deepcopy(environment)
    not_closed["close_requests"] = 0
    not_closed[runner.K_CLOSED] = False
    assert not runner.check_results(states, not_closed, [])

    closed_twice = deepcopy(environment)
    closed_twice["close_requests"] = 2
    assert not runner.check_results(states, closed_twice, [])


@pytest.mark.parametrize(
    "log_line",
    [
        "[2026-08-10][ERROR]::[agent][module]::plain failure\n",
        "[2026-08-10][CRITICAL]::[agent][module]::plain failure\n",
        "Traceback (most recent call last):\n",
        "ExceptionGroup: grouped failure\n",
        "[WARNING]::Caught exception while processing action\n",
        "[WARNING]::Could not send message to actuator\n",
        "[WARNING]::Failed to save state\n",
    ],
)
def test_result_checker_rejects_runtime_failure_logs(log_line: str) -> None:
    states, environment = _valid_result_fixture()
    assert not runner.check_results(states, environment, [log_line])
