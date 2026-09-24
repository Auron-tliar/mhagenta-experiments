from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mhagenta import ActionStatus, Observation

from mha_exp_level2_cr.exp2_1.policy import (
    ACTIONS,
    Direction,
    ObservationError,
    ReactiveCrafterPolicy,
    RESET_ACTION,
)
from mha_exp_level2_cr.exp2_1.runner import (
    DIAMOND_PATH_ACHIEVEMENTS,
    CrafterActuator,
    CrafterEnvironment,
    CrafterPerceptor,
    CrafterReactiveReasoner,
    MAX_EPISODE_LEN,
    _reasoner_initial_state,
    check_results,
)
from mha_exp_level2_cr.exp2_1.treatment import treatment_for_run


INVENTORY = {
    "health": 9, "food": 9, "drink": 9, "energy": 9, "sapling": 0,
    "wood": 0, "stone": 0, "coal": 0, "iron": 0, "diamond": 0,
    "wood_pickaxe": 0, "stone_pickaxe": 0, "iron_pickaxe": 0,
    "wood_sword": 0, "stone_sword": 0, "iron_sword": 0,
}


def _label(position: tuple[int, int]) -> str:
    x, y = position
    parts = []
    if x:
        parts.append(f"{'L' if x < 0 else 'R'}{abs(x)}")
    if y:
        parts.append(f"{'U' if y < 0 else 'D'}{abs(y)}")
    return "_".join(parts)


def observation(
    *,
    facing: str = "R1",
    sleeping: bool = False,
    inventory: dict[str, int] | None = None,
    materials: dict[tuple[int, int], str] | None = None,
    occupants: dict[tuple[int, int], str] | None = None,
    radius: int = 2,
) -> list[str]:
    items = {**INVENTORY, **(inventory or {})}
    materials = materials or {}
    occupants = occupants or {}
    predicates = [
        f"Sleeping() = {'true' if sleeping else 'false'}",
        f"Facing({facing}) = true",
        *(f"Have({item}) = {count}" for item, count in items.items()),
    ]
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if (x, y) == (0, 0):
                continue
            position = (x, y)
            label = _label(position)
            predicates.extend((
                f"MadeOf({label}, {materials.get(position, 'grass')}) = true",
                f"OccupiedBy({label}, {occupants.get(position, 'none')}) = true",
            ))
    return predicates


def test_action_vocabulary_is_plain_readable_and_native_ordered() -> None:
    assert ACTIONS == (
        "noop", "move_left", "move_right", "move_up", "move_down", "do",
        "sleep", "place_stone", "place_table", "place_furnace", "place_plant",
        "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
        "make_wood_sword", "make_stone_sword", "make_iron_sword",
    )
    assert all(type(action) is str for action in ACTIONS)


def test_current_fluent_boundary_parses_live_and_rejects_malformed_input() -> None:
    from mha_env_crafter import CrafterEnv

    policy = ReactiveCrafterPolicy(1)
    result = policy._parse_observation(
        CrafterEnv(seed=7, no_mobs=True, symbolic=True).reset()
    )
    assert result.facing is Direction.DOWN
    assert result.inventory["health"] == 9
    assert result.tiles[(0, 0)].occupant == "player"
    assert len(result.tiles) == 63

    incomplete = [
        item for item in observation(radius=1)
        if not item.startswith("OccupiedBy(R1,")
    ]
    with pytest.raises(ObservationError, match="do not match"):
        policy._parse_observation(incomplete)
    with pytest.raises(ObservationError, match="Invalid tile location"):
        policy._parse_location("R0")
    with pytest.raises(ObservationError, match="Unknown symbolic predicate"):
        policy._parse_observation([*observation(radius=1), "Mystery() = true"])


@pytest.mark.parametrize(
    ("kwargs", "expected_action", "expected_reason"),
    [
        ({"sleeping": True}, "noop", "sleeping"),
        (
            {"facing": "R1", "inventory": {"drink": 1},
             "materials": {(1, 0): "water"}},
            "do", "seek water",
        ),
        ({"facing": "R1", "materials": {(1, 0): "tree"}},
         "do", "collect wood"),
        (
            {"inventory": {"wood": 1}, "materials": {(1, 1): "table"}},
            "make_wood_pickaxe", "make wood pickaxe",
        ),
        (
            {"inventory": {"wood": 1, "stone": 1, "wood_pickaxe": 1},
             "materials": {(1, 1): "table"}},
            "make_stone_pickaxe", "make stone pickaxe",
        ),
        (
            {"inventory": {"wood": 1, "coal": 1, "iron": 1,
                           "wood_pickaxe": 1, "stone_pickaxe": 1},
             "materials": {(1, 0): "table", (0, 1): "furnace"}},
            "make_iron_pickaxe", "make iron pickaxe",
        ),
        (
            {"facing": "R1", "inventory": {"iron_pickaxe": 1},
             "materials": {(1, 0): "diamond"}},
            "do", "collect diamond",
        ),
    ],
)
def test_representative_policy_priorities_are_unchanged(
    kwargs: dict[str, Any], expected_action: str, expected_reason: str
) -> None:
    decision = ReactiveCrafterPolicy(4).choose_action(observation(**kwargs))
    assert (decision.action, decision.reason) == (expected_action, expected_reason)


def test_diamond_rush_precedes_self_care_but_unreachable_diamond_does_not() -> None:
    policy = ReactiveCrafterPolicy(4)
    reachable = policy.choose_action(observation(
        inventory={"drink": 1, "iron_pickaxe": 1},
        materials={(1, 0): "diamond", (0, -1): "water"},
    ))
    unreachable = policy.choose_action(observation(
        inventory={"drink": 1, "iron_pickaxe": 1},
        materials={(1, 0): "water", (2, 2): "diamond",
                   (1, 2): "lava", (2, 1): "lava"},
    ))
    assert (reachable.action, reachable.reason) == ("do", "collect diamond")
    assert (unreachable.action, unreachable.reason) == ("do", "seek water")


def test_fallback_keeps_fixed_eighty_twenty_rule_and_entrapment_reset() -> None:
    policy = ReactiveCrafterPolicy(1)
    obs = policy._parse_observation(observation(
        inventory={"wood_pickaxe": 1, "stone_pickaxe": 1, "iron_pickaxe": 1}
    ))
    policy._rng = SimpleNamespace(  # type: ignore[assignment]
        random=lambda: 0.79, choice=lambda values: Direction.UP)
    assert policy._fallback_action(obs).action == "move_right"
    policy._rng = SimpleNamespace(  # type: ignore[assignment]
        random=lambda: 0.81, choice=lambda values: Direction.UP)
    assert policy._fallback_action(obs).action == "move_up"
    with pytest.raises(TypeError):
        ReactiveCrafterPolicy(1, maintain_dir_prob=0.9)  # type: ignore[call-arg]

    trapped = policy._parse_observation(observation(
        inventory={"wood_pickaxe": 1, "stone_pickaxe": 1, "iron_pickaxe": 1},
        materials={(1, 0): "water", (-1, 0): "water",
                   (0, 1): "water", (0, -1): "water"},
    ))
    assert policy._fallback_action(trapped).action == RESET_ACTION
    assert set(policy.__dict__) == {"_rng"}


class FakeOutbox:
    def __init__(self) -> None:
        self.observation_requests: list[str] = []
        self.action_requests: list[tuple[str, Any]] = []
        self.action_metadata: list[dict[str, Any]] = []
        self.observations: list[tuple[str, Observation]] = []
        self.statuses: list[tuple[str, ActionStatus]] = []
        self.terminations: list[str] = []

    def request_observation(self, recipient: str, **kwargs: Any) -> None:
        self.observation_requests.append(recipient)

    def request_action(self, recipient: str, **kwargs: Any) -> None:
        self.action_requests.append((recipient, kwargs["action"]))
        self.action_metadata.append(dict(kwargs))

    def send_observation(self, recipient: str, value: Observation) -> None:
        self.observations.append((recipient, value))

    def send_status(self, recipient: str, value: ActionStatus) -> None:
        self.statuses.append((recipient, value))

    def terminate_agent(self, reason: str) -> None:
        self.terminations.append(reason)


class FakeState(dict[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.outbox = FakeOutbox()


def reasoner() -> CrafterReactiveReasoner:
    value = CrafterReactiveReasoner(module_id="ll", initial_state={})
    value._actuator_id = "act"
    value._perceptor_id = "per"
    value._policy = ReactiveCrafterPolicy(1)
    return value


def reasoner_state() -> FakeState:
    return FakeState(_reasoner_initial_state())


def native_status(
    *, action: str = "noop", done: bool = False, dead: bool = False
) -> ActionStatus:
    return ActionStatus({
        "action": action, "reward": 0.0, "done": done,
        "illegal_action": False, "dead": dead, "reset": False,
        "target_achieved": False, "highest_milestone": None,
        "native_actions": 1,
        "terminal_reason": "environment_terminal" if done else None,
    })


def test_complete_typed_cycle_keeps_strings_on_both_agent_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perceptor = CrafterPerceptor(module_id="per", initial_state={})
    perceptor._env_id = "env"
    perceptor._reasoner_id = "ll"
    observed: list[str] = []
    monkeypatch.setattr(perceptor, "observe", lambda env_id: observed.append(env_id))
    perceptor_state = FakeState({"requests": 0, "observations": 0})
    perceptor.on_request(perceptor_state, "ll")
    perceptor.on_observation(
        perceptor_state, "env", observation=observation(materials={(1, 0): "tree"})
    )

    ll = reasoner()
    ll_state = reasoner_state()
    envelope = perceptor_state.outbox.observations[0][1]
    ll.on_observation(ll_state, "per", envelope)
    action = ll_state.outbox.action_requests[0][1]

    actuator = CrafterActuator(module_id="act", initial_state={})
    actuator._env_id = "env"
    actuator._reasoner_id = "ll"
    forwarded: list[str] = []
    monkeypatch.setattr(
        actuator, "act", lambda env_id, **kwargs: forwarded.append(kwargs["action"])
    )
    actuator_state = FakeState({"requests": 0, "statuses": 0})
    actuator.on_request(actuator_state, "ll", **ll_state.outbox.action_metadata[0])
    actuator.on_status(
        actuator_state, "env", **native_status(action=action).status,
        request_id=0, episode_id=0,
    )
    ll.on_action_status(ll_state, "act", actuator_state.outbox.statuses[0][1])

    assert observed == ["env"]
    assert type(action) is str and forwarded == [action]
    assert actuator_state.outbox.statuses[0][1].status["action"] == action
    assert ll_state.outbox.observation_requests == ["per"]
    assert ll_state["last_decision"] == {"action": "do", "reason": "collect wood"}


@pytest.mark.parametrize("terminal", [native_status(done=True, dead=True), native_status(done=True)])
def test_native_death_and_truncation_terminate_without_reset(
    terminal: ActionStatus,
) -> None:
    ll = reasoner()
    state = reasoner_state()
    ll._request_action(state, action="noop", reason="test")
    ll.on_action_status(
        state, "act", ActionStatus({**terminal.status, "request_id": 0, "episode_id": 0})
    )
    assert state.outbox.action_requests == [("act", "noop")]
    assert not state.outbox.observation_requests
    assert state.outbox.terminations == ["2-1-CR environment_terminal"]


@pytest.mark.parametrize("target,actions,expected", [
    (False, 999, None),
    (True, 999, "target_achieved"),
    (False, 1000, "action_budget_exhausted"),
    (True, 1000, "target_achieved"),
])
def test_diamond_and_action_budget_stop_without_another_action(
    target: bool, actions: int, expected: str | None,
) -> None:
    """Goal success wins at the budget boundary; earlier progress continues."""
    ll = reasoner()
    state = reasoner_state()
    ll._request_action(state, action="do", reason="collect")
    ll.on_action_status(state, "act", ActionStatus({
        **native_status(action="do").status, "request_id": 0, "episode_id": 0,
        "target_achieved": target, "native_actions": actions,
        "highest_milestone": "collect_diamond" if target else "collect_stone",
        "terminal_reason": expected,
    }))
    assert state.outbox.terminations == ([] if expected is None else [f"2-1-CR {expected}"])
    assert state.outbox.observation_requests == (["per"] if expected is None else [])


@pytest.mark.parametrize("elapsed,phase,terminal,expected", [
    (600.1, "active", None, "timeout"),
    (20.0, "active", None, "external_stop"),
    (600.1, "complete", "target_achieved", "target_achieved"),
    (600.1, "complete", "death", "death"),
])
def test_shutdown_records_timeout_without_overwriting_terminal_outcome(
    elapsed: float, phase: str, terminal: str | None, expected: str,
) -> None:
    """Shutdown preserves an observed outcome even at the wall-time boundary."""
    state = reasoner_state()
    state.time = elapsed
    state.update(phase=phase, terminal_reason=terminal)
    reasoner().on_last(state)
    assert state["terminal_reason"] == expected
    assert state["execution_seconds"] == elapsed


@pytest.mark.parametrize("case", ["diamond", "entrapment"])
def test_policy_terminal_observations_request_exactly_one_reset(case: str) -> None:
    content = (
        observation(inventory={"diamond": 1})
        if case == "diamond"
        else observation(
            inventory={"wood_pickaxe": 1, "stone_pickaxe": 1, "iron_pickaxe": 1},
            materials={(1, 0): "water", (-1, 0): "water",
                       (0, 1): "water", (0, -1): "water"},
        )
    )
    ll = reasoner()
    state = reasoner_state()
    ll.on_observation(state, "per", Observation(content))
    assert state.outbox.action_requests == [("act", RESET_ACTION)]
    assert state["action_requests"] == 0


def test_reset_acknowledgement_and_normal_status_continue_observation_loop() -> None:
    ll = reasoner()
    reset_state = reasoner_state()
    ll._request_reset(reset_state, cause="policy_requested", reason="test")
    ll.on_action_status(reset_state, "act", ActionStatus({
        "action": RESET_ACTION, "reset": True, "request_id": 0,
        "episode_id": 0, "next_episode_id": 1,
    }))
    assert reset_state.outbox.observation_requests == ["per"]
    assert reset_state.outbox.action_requests == [("act", RESET_ACTION)]

    normal_state = reasoner_state()
    ll._request_action(normal_state, action="noop", reason="test")
    ll.on_action_status(normal_state, "act", ActionStatus({
        **native_status().status, "request_id": 0, "episode_id": 0,
    }))
    assert normal_state.outbox.observation_requests == ["per"]
    assert normal_state.outbox.action_requests == [("act", "noop")]


def _environment_state() -> dict[str, Any]:
    return {
        "seed": 7, "record": False, "artifact_root": "/tmp",
        **treatment_for_run(0),
        "native_actions": 0, "illegal_actions": 0,
        "highest_diamond_path_achievement": None,
        "target_achieved": False, "terminal_reason": None,
        "next_event_id": 0, "episode_id": 0, "execution_trace": [],
        "closed": False, "video_path": None, "video_error": None,
    }


class SpyCrafter:
    action_names = ACTIONS

    def __init__(self, achievements: dict[str, int]) -> None:
        self.achievements = achievements
        self.native_actions: list[int] = []
        self.resets = 0

    def step(self, action: int) -> tuple[None, float, bool, dict[str, Any]]:
        self.native_actions.append(action)
        return None, 0.0, False, {
            "illegal_action": False,
            "inventory": {"health": 9, "food": 9, "drink": 9, "energy": 9},
            "achievements": self.achievements,
            "player_pos": [len(self.native_actions), 0],
        }

    def reset(self) -> None:
        self.resets += 1


def achievement_counts(**positive: int) -> dict[str, int]:
    return {name: positive.get(name, 0) for name in DIAMOND_PATH_ACHIEVEMENTS}


def test_environment_owns_monotonic_path_result_and_native_conversion() -> None:
    bridge = CrafterEnvironment(_environment_state())
    assert bridge._env._length == MAX_EPISODE_LEN
    _, observed = bridge.on_observe(bridge.state, "agent")
    assert isinstance(observed["observation"], list)


def test_close_marks_environment_and_schedules_clean_self_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The close callback exits cleanly without waiting for Docker to stop it."""
    bridge = CrafterEnvironment(_environment_state())
    started: list[bool] = []
    monkeypatch.setattr(
        "mha_exp_level2_cr.exp2_1.runner.Timer.start",
        lambda timer: started.append(timer.daemon),
    )
    state, response = bridge.on_action(bridge.state, "agent", action="close")
    assert state["closed"] is True
    assert response == {"action": "close", "closed": True}
    assert started == [True]

    spy = SpyCrafter(achievement_counts(collect_wood=1))
    bridge._env = spy
    _, status = bridge.on_action(
        bridge.state, "agent", action="move_right", request_id=0, episode_id=0
    )
    assert spy.native_actions == [2]
    assert status["action"] == "move_right" and "achievements" not in status
    assert bridge.state["highest_diamond_path_achievement"] == "collect_wood"

    spy.achievements = achievement_counts(place_table=1)
    bridge.on_action(bridge.state, "agent", action="noop", request_id=1, episode_id=0)
    bridge.on_action(
        bridge.state, "agent", action=RESET_ACTION, request_id=2, episode_id=0,
        reset_cause="policy_requested", policy_reason="test",
    )
    assert bridge.state["highest_diamond_path_achievement"] == "place_table"
    assert spy.resets == 1
    with pytest.raises(ValueError, match="Invalid readable environment action"):
        bridge.on_action(bridge.state, "agent", action=2, request_id=3, episode_id=1)


def test_environment_trace_uses_fake_clock_and_preserves_reset_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = CrafterEnvironment(_environment_state())
    bridge._env = SpyCrafter(achievement_counts())
    bridge._started = 10.0
    ticks = iter((11.0, 12.0, 13.0, 14.0))
    monkeypatch.setattr(
        "mha_exp_level2_cr.exp2_1.runner.time.monotonic", lambda: next(ticks)
    )

    bridge.on_action(
        bridge.state, "agent", action="noop", request_id=0, episode_id=0
    )
    bridge.on_action(
        bridge.state, "agent", action=RESET_ACTION, request_id=1, episode_id=0,
        reset_cause="environment_terminal",
    )
    bridge.on_action(
        bridge.state, "agent", action="noop", request_id=2, episode_id=1
    )
    bridge.on_action(
        bridge.state, "agent", action=RESET_ACTION, request_id=3, episode_id=1,
        reset_cause="policy_requested",
    )

    trace = bridge.state["execution_trace"]
    assert [row["event_id"] for row in trace] == [0, 1, 2, 3]
    assert [row["episode_id"] for row in trace] == [0, 0, 1, 1]
    assert [row["elapsed_seconds"] for row in trace] == [1.0, 2.0, 3.0, 4.0]
    assert [trace[1]["reset_cause"], trace[3]["reset_cause"]] == [
        "environment_terminal", "policy_requested",
    ]


def test_environment_rejects_malformed_path_counts() -> None:
    bridge = CrafterEnvironment(_environment_state())
    bridge._env = SpyCrafter({"collect_wood": -1})
    with pytest.raises(ValueError, match="achievement count"):
        bridge.on_action(
            bridge.state, "agent", action="noop", request_id=0, episode_id=0
        )


def test_native_crafter_truncates_at_its_configured_length() -> None:
    from mha_env_crafter import CrafterEnv

    env = CrafterEnv(seed=7, length=3, no_mobs=True, symbolic=True)
    env.reset()
    assert [env.step(0)[2] for _ in range(3)] == [False, False, True]


def valid_states() -> dict[str, dict[str, Any]]:
    return {
        "perceptor_0": {"requests": 4, "observations": 4},
        "actuator_0": {"requests": 3, "statuses": 3},
        "llreasoner_0": {
            "observations": 4, "action_requests": 2, "action_statuses": 3,
            "last_decision": {"action": "move_right", "reason": "explore"},
        },
    }


def valid_environment_state() -> dict[str, Any]:
    return {"native_actions": 2, "illegal_actions": 0,
            "highest_diamond_path_achievement": "collect_wood"}


def test_checker_accepts_environment_owned_minimum_and_stronger_results() -> None:
    assert check_results(valid_states(), valid_environment_state(), [])
    stronger = valid_environment_state()
    stronger["highest_diamond_path_achievement"] = "collect_diamond"
    assert check_results(valid_states(), stronger, [])


def test_no_achievement_is_a_valid_unsuccessful_trial() -> None:
    """Absence of progress alone is not a runtime or evidence failure."""
    environment = valid_environment_state()
    environment["highest_diamond_path_achievement"] = None
    assert check_results(valid_states(), environment, [])

@pytest.mark.parametrize("mutation", [
    "off_path", "illegal", "bad_decision", "mixed_counters",
    "missing_agent_state", "missing_environment_state", "missing_agent_log",
])
def test_checker_rejects_invalid_scientific_or_runtime_evidence(mutation: str) -> None:
    states = valid_states()
    environment = valid_environment_state()
    logs: list[str] | None = []
    if mutation == "off_path":
        environment["highest_diamond_path_achievement"] = "eat_cow"
    elif mutation == "illegal":
        environment["illegal_actions"] = 1
    elif mutation == "bad_decision":
        states["llreasoner_0"]["last_decision"] = {"action": 2, "reason": "bad"}
    elif mutation == "mixed_counters":
        states["actuator_0"]["requests"] = 5
    elif mutation == "missing_agent_state":
        states.pop("perceptor_0")
    elif mutation == "missing_environment_state":
        environment = None  # type: ignore[assignment]
    else:
        logs = None
    assert not check_results(states, environment, logs)


@pytest.mark.parametrize("source,line", [
    ("agent", "Traceback (most recent call last):\n"),
    ("agent", "[t][ERROR]::[agent]::failure\n"),
    ("agent", "[t][CRITICAL]::[agent]::failure\n"),
    ("agent", "[t][WARNING]::[agent]::Caught exception in callback\n"),
    ("agent", "[t][WARNING]::[agent]::Failed to save state\n"),
    ("agent", "[t][WARNING]::[agent]::Could not send message\n"),
    ("environment", "[t][WARNING]::[env]::Caught exception: bad achievements\n"),
])
def test_checker_rejects_runtime_failures_from_both_logs(source: str, line: str) -> None:
    agent_logs = [line] if source == "agent" else []
    environment_logs = [line] if source == "environment" else []
    assert not check_results(
        valid_states(), valid_environment_state(), agent_logs, environment_logs
    )
