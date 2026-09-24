"""Focused full-domain contract tests for Experiment 2-4-CR."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mhagenta import ActionStatus, Belief, Observation
import pytest

from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)
from mha_exp_common.utils import module_name
from mha_exp_level2_cr.exp2_4.activities import (
    MILESTONES,
    ActivityContractError,
    ActivityName,
    PlanStep,
    activity_complete,
    activity_from_goal,
    derive_stage_plan,
    find_path,
    make_activity_goal,
    next_activity_action,
    outcome_from_goal,
    placement_options,
    planning_summary,
    terminal_activity_goal,
    technology_stage,
    usable_station_sets,
)
from mha_exp_level2_cr.exp2_4.agent import (
    ACTIVITY_ACTION_LIMITS,
    ActivityGoalGraph,
    ActivityLLReasoner,
    CrafterHybridActuator,
    CrafterHybridEnvironment,
    CrafterHybridPerceptor,
    HybridSymbolicReasoner,
    environment_initial_state,
    initial_states,
)
from mha_exp_level2_cr.exp2_4.beliefs import (
    AbstractState,
    CrafterAction,
    abstract_beliefs,
    initial_belief_state,
    parse_abstract_beliefs,
    parse_symbolic_observation,
    revise_belief_state,
)
from mha_exp_level2_cr.exp2_4.checking import check_results
from mha_exp_level2_cr.exp2_4 import reporting
from mha_exp_level2_cr.exp2_4.runner import (
    _ensure_execution_provenance,
    _readable_video,
    _source_digest,
)
from mha_exp_level2_cr.exp2_4.treatment import PROTOCOL_VERSION, treatment_for_run


INVENTORY = {
    "health": 9,
    "food": 9,
    "drink": 9,
    "energy": 9,
    "sapling": 0,
    "wood": 0,
    "stone": 0,
    "coal": 0,
    "iron": 0,
    "diamond": 0,
    "wood_pickaxe": 0,
    "stone_pickaxe": 0,
    "iron_pickaxe": 0,
    "wood_sword": 0,
    "stone_sword": 0,
    "iron_sword": 0,
}


def symbolic_observation(
    *, facing: str = "L1", inventory: dict[str, int] | None = None
) -> list[str]:
    values = INVENTORY | (inventory or {})
    return [
        "Sleeping() = false",
        f"Facing({facing}) = true",
        *(f"Have({item}) = {value}" for item, value in values.items()),
        "MadeOf(R1, tree) = true",
        "OccupiedBy(R1, none) = true",
        "MadeOf(L1, grass) = true",
        "OccupiedBy(L1, none) = true",
    ]


def public_status(*, done: bool = False) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "done": done,
        "dead": False,
        "illegal_action": False,
        "new_achievements": [],
    }


def detailed_state(
    *,
    revision: int = 1,
    player: tuple[int, int] = (0, 0),
    facing: str = "left",
    inventory: dict[str, int] | None = None,
    terrain: dict[str, str] | None = None,
    safe: list[str] | None = None,
    visible: list[str] | None = None,
    occupants: dict[str, dict[str, Any]] | None = None,
    offsets: list[list[int]] | None = None,
    terminal: bool = False,
) -> dict[str, Any]:
    state = initial_belief_state()
    state.update(
        revision=revision,
        player=list(player),
        facing=facing,
        inventory=INVENTORY | (inventory or {}),
        terrain=dict(terrain or {}),
        occupants=dict(occupants or {}),
        visible_cells=list(visible or (terrain or {}).keys()),
        known_cells=sorted({f"{player[0]},{player[1]}", *(terrain or {}).keys()}),
        safe_cells=list(safe or [f"{player[0]},{player[1]}"]),
        blocked_cells=sorted(set((terrain or {})) - set(safe or [])),
        view_offsets=list(offsets or [[-1, 0], [1, 0]]),
        terminal=terminal,
    )
    return state


def abstract_snapshot(
    revision: int,
    *,
    sleeping: bool = False,
    inventory: dict[str, int] | None = None,
    reachable: tuple[str, ...] = (),
    known: tuple[str, ...] | None = None,
    usable: tuple[str, ...] = (),
    placements: tuple[str, ...] = (),
    station_counts: dict[str, int] | None = None,
    known_cells: int = 10,
    terminal: bool = False,
) -> list[Belief]:
    values = INVENTORY | (inventory or {})
    known_values = known if known is not None else reachable
    counts = {"table": 0, "furnace": 0} | (station_counts or {})
    return [
        Belief("observation_revision", (revision,)),
        Belief("sleeping", (sleeping,)),
        *(Belief("inventory", (item, value)) for item, value in values.items()),
        *(Belief("known_target_kind", (kind,)) for kind in known_values),
        *(Belief("reachable_target_kind", (kind,)) for kind in reachable),
        *(Belief("station_count", (kind, value)) for kind, value in counts.items()),
        *(Belief("usable_station_set", (value,)) for value in usable),
        *(Belief("placement_opportunity", (value,)) for value in placements),
        Belief("known_cell_count", (known_cells,)),
        Belief("terminal", (terminal,)),
        Belief("dead", (False,)),
    ]


def goal(step: PlanStep, revision: int = 1, goal_id: str = "activity-1"):
    return make_activity_goal(
        goal_id=goal_id,
        activity=step.activity,
        based_on_revision=revision,
        desired_predicate=step.desired_predicate,
        desired_arguments=step.desired_arguments,
    )


def atomic_row(index: int, dispatch: int, confirmation: int) -> dict[str, Any]:
    return {
        "action_id": f"action-{index}",
        "action": int(CrafterAction.NOOP),
        "movement_kind": "none",
        "source_cell": [0, 0],
        "destination_cell": [0, 0],
        "dispatch_revision": dispatch,
        "legal": True,
        "confirmation_revision": confirmation,
    }


class FakeOutbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return call


class FakeState(dict[str, Any]):
    def __init__(self, values: dict[str, Any], directory: Any | None = None) -> None:
        super().__init__(deepcopy(values))
        self.outbox = FakeOutbox()
        self.directory = directory


def directory() -> Any:
    entry = lambda name: SimpleNamespace(module_id=name)
    internal = SimpleNamespace(
        perception=[entry("perceptor")],
        actuation=[entry("actuator")],
        ll_reasoning=[entry("ll")],
        knowledge=[entry("knowledge")],
        goals=[entry("goals")],
        hl_reasoning=[entry("hl")],
    )
    external = SimpleNamespace(
        environments=[SimpleNamespace(address={"env_id": "environment"})]
    )
    return SimpleNamespace(internal=internal, external=external)


def test_treatment_is_from_scratch_and_frozen() -> None:
    treatment = treatment_for_run(0)
    assert treatment["protocol_version"] == PROTOCOL_VERSION
    assert treatment["primary_intention"] == "obtain_diamond"
    assert treatment["duration"] == 900 and treatment["total_action_cap"] == 1000
    assert not {"activity", "inventory_fixture", "required_target"} & set(treatment)


def test_symbolic_revision_and_coordinate_free_abstract_round_trip() -> None:
    percept = parse_symbolic_observation(symbolic_observation())
    state = revise_belief_state(
        initial_belief_state(), percept, revision=1, pending_action=None
    ).belief_state
    summary = planning_summary(state)
    snapshot = parse_abstract_beliefs(abstract_beliefs(state, summary))
    assert snapshot.inventory["health"] == 9
    assert "tree" in snapshot.known_target_kinds
    assert not hasattr(snapshot, "player") and not hasattr(snapshot, "facing")


@pytest.mark.parametrize(
    ("kind", "action", "destination", "expected_player"),
    [
        ("walk", CrafterAction.MOVE_LEFT, [-1, 0], [-1, 0]),
        ("turn", CrafterAction.MOVE_RIGHT, [1, 0], [0, 0]),
        ("none", CrafterAction.PLACE_TABLE, [1, 0], [0, 0]),
        ("none", CrafterAction.MAKE_WOOD_PICKAXE, [0, 0], [0, 0]),
        ("none", CrafterAction.SLEEP, [0, 0], [0, 0]),
        ("none", CrafterAction.NOOP, [0, 0], [0, 0]),
    ],
)
def test_localization_covers_all_native_action_shapes(
    kind: str,
    action: CrafterAction,
    destination: list[int],
    expected_player: list[int],
) -> None:
    state = detailed_state(terrain={"-1,0": "grass", "1,0": "tree"}, safe=["0,0", "-1,0"])
    pending = {
        "action": int(action),
        "movement_kind": kind,
        "source_cell": [0, 0],
        "destination_cell": destination,
        "facing_before": "left",
        "status": public_status(),
    }
    facing = "L1" if action is CrafterAction.MOVE_LEFT else "R1"
    result = revise_belief_state(
        state,
        parse_symbolic_observation(symbolic_observation(facing=facing)),
        revision=2,
        pending_action=pending,
    )
    assert result.belief_state["player"] == expected_player


def placement_state(*, facing: str = "right") -> dict[str, Any]:
    terrain = {
        "-1,0": "grass",
        "0,-1": "grass",
        "1,0": "grass",
        "0,1": "grass",
    }
    return detailed_state(
        facing=facing,
        inventory={"wood": 2, "stone": 4},
        terrain=terrain,
        safe=["0,0", "-1,0", "0,-1", "1,0", "0,1"],
        visible=["1,0", "0,1"],
        offsets=[[-1, 0], [0, -1], [1, 0], [0, 1]],
    )


def test_table_and_furnace_geometry_preserves_shared_craft_anchor() -> None:
    state = placement_state()
    table_options = placement_options(state, "table")
    assert ((-1, 0), (0, 0), (1, 0)) in table_options
    assert placement_options(state, "table", first_only=True) == table_options[:1]
    state["terrain"]["1,0"] = "table"
    state["safe_cells"].remove("1,0")
    furnace_options = placement_options(state, "furnace")
    assert any(anchor == (0, 0) and placement != (1, 0) for _, anchor, placement in furnace_options)
    assert placement_options(state, "furnace", first_only=True) == furnace_options[:1]
    assert usable_station_sets(state) == frozenset({"table"})


def test_wrong_facing_at_placement_anchor_routes_through_orientation() -> None:
    request = goal(PlanStep(ActivityName.PLACE_TABLE, "usable_station_set", ("table",)))
    decision = next_activity_action(activity_from_goal(request), placement_state(facing="left"))
    assert decision.movement_kind == "walk"
    assert decision.destination_cell == (-1, 0)
    assert decision.destination_cell != decision.target_cell


def test_reachability_and_boundary_exploration_are_deterministic() -> None:
    assert find_path((0, 0), {(2, 0)}, {(0, 0), (1, 0), (2, 0)}) == ((1, 0), (2, 0))
    state = detailed_state(
        terrain={"1,0": "tree"},
        safe=["0,0"],
        visible=["1,0"],
        offsets=[[1, 0]],
    )
    request = goal(PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("diamond",)))
    decision = next_activity_action(activity_from_goal(request), state)
    assert decision.movement_kind == "turn" and decision.target_kind == "tree"
    state["facing"] = "right"
    assert next_activity_action(activity_from_goal(request), state).action is CrafterAction.DO


@pytest.mark.parametrize(
    "step",
    [
        PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("tree",)),
        PlanStep(ActivityName.EAT, "inventory_at_least", ("food", 5)),
        PlanStep(ActivityName.DRINK, "inventory_at_least", ("drink", 5)),
        PlanStep(ActivityName.SLEEP, "inventory_at_least", ("energy", 5)),
        PlanStep(ActivityName.GET_WOOD, "inventory_at_least", ("wood", 2)),
        PlanStep(ActivityName.GET_STONE, "inventory_at_least", ("stone", 1)),
        PlanStep(ActivityName.GET_COAL, "inventory_at_least", ("coal", 1)),
        PlanStep(ActivityName.GET_IRON, "inventory_at_least", ("iron", 1)),
        PlanStep(ActivityName.GET_DIAMOND, "inventory_at_least", ("diamond", 1)),
        PlanStep(ActivityName.PLACE_TABLE, "usable_station_set", ("table",)),
        PlanStep(ActivityName.PLACE_FURNACE, "usable_station_set", ("table_furnace",)),
        PlanStep(ActivityName.MAKE_WOOD_PICKAXE, "inventory_at_least", ("wood_pickaxe", 1)),
        PlanStep(ActivityName.MAKE_STONE_PICKAXE, "inventory_at_least", ("stone_pickaxe", 1)),
        PlanStep(ActivityName.MAKE_IRON_PICKAXE, "inventory_at_least", ("iron_pickaxe", 1)),
    ],
)
def test_every_activity_goal_round_trips(step: PlanStep) -> None:
    request = goal(step)
    assert set(request.extras) == {"goal_id", "activity", "status", "based_on_revision"}
    terminal = terminal_activity_goal(
        request,
        status="failed",
        completion_revision=1,
        atomic=[],
        failure_reason="no_frontier",
    )
    assert outcome_from_goal(terminal).goal_id == "activity-1"
    assert set(terminal.extras) == {
        "goal_id", "status", "completion_revision", "atomic", "failure_reason", "interruption"
    }


def test_interruption_round_trip_is_self_identifying() -> None:
    request = goal(PlanStep(ActivityName.GET_STONE, "inventory_at_least", ("stone", 1)))
    interruption = {
        "need": "drink",
        "observed_value": 2,
        "interrupted_activity": "get_stone",
        "interrupted_goal_id": "activity-1",
        "belief_revision": 3,
    }
    outcome = outcome_from_goal(
        terminal_activity_goal(
            request,
            status="failed",
            completion_revision=3,
            atomic=[atomic_row(1, 1, 2)],
            failure_reason="need_interruption",
            interruption=interruption,
        )
    )
    assert outcome.interruption == interruption


def snapshot_model(
    *,
    inventory: dict[str, int] | None = None,
    reachable: tuple[str, ...] = (),
    usable: tuple[str, ...] = (),
    placements: tuple[str, ...] = (),
) -> AbstractState:
    return parse_abstract_beliefs(
        abstract_snapshot(
            1,
            inventory=inventory,
            reachable=reachable,
            usable=usable,
            placements=placements,
        )
    )


@pytest.mark.parametrize(
    ("inventory", "usable", "expected"),
    [
        ({}, (), "table"),
        ({}, ("table",), "wood_pickaxe"),
        ({"wood_pickaxe": 1}, ("table",), "stone_pickaxe"),
        ({"stone_pickaxe": 1}, ("table",), "furnace"),
        ({"stone_pickaxe": 1}, ("table", "table_furnace"), "iron_pickaxe"),
        ({"iron_pickaxe": 1}, (), "diamond"),
    ],
)
def test_technology_stage_is_derived_from_prerequisites(
    inventory: dict[str, int], usable: tuple[str, ...], expected: str
) -> None:
    assert technology_stage(snapshot_model(inventory=inventory, usable=usable)) == expected


def test_stage_plans_replace_missing_gather_target_with_explore() -> None:
    snapshot = snapshot_model(inventory={"wood_pickaxe": 1}, usable=("table",))
    plan = derive_stage_plan(snapshot)
    assert plan[0] == PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("stone",))
    reachable = snapshot_model(
        inventory={"wood_pickaxe": 1}, reachable=("stone",), usable=("table",)
    )
    assert derive_stage_plan(reachable)[0] == PlanStep(
        ActivityName.GET_STONE, "inventory_at_least", ("stone", 1)
    )


def test_sleep_finishes_only_after_threshold_and_clean_wake() -> None:
    request = goal(PlanStep(ActivityName.SLEEP, "inventory_at_least", ("energy", 5)))
    spec = activity_from_goal(request)
    state = detailed_state(inventory={"energy": 5})
    state["sleeping"] = True
    assert not activity_complete(spec, state)
    state["sleeping"] = False
    assert activity_complete(spec, state)


def test_sleep_is_interrupted_immediately_by_a_different_urgent_need() -> None:
    belief_state = detailed_state(
        inventory={"health": 3, "energy": 4, "drink": 1}
    )
    belief_state["sleeping"] = True
    _, state = ll_state_for_goal(
        PlanStep(ActivityName.SLEEP, "inventory_at_least", ("health", 4)),
        belief_state,
    )
    terminal = next(call for call in state.outbox.calls if call[0] == "send_goal_update")
    outcome = outcome_from_goal(terminal[1][1][0])
    assert outcome.failure_reason == "need_interruption"
    assert outcome.interruption["need"] == "drink"
    assert state["active_activity"] is None
    assert state["pending_atomic"] is None


def test_non_sleep_recovery_waits_with_noop_until_fresh_beliefs_show_awake() -> None:
    belief_state = detailed_state(
        inventory={"drink": 1, "energy": 4},
        terrain={"1,0": "water"},
        safe=["0,0"],
        visible=["1,0"],
    )
    belief_state["sleeping"] = True
    ll, state = ll_state_for_goal(
        PlanStep(ActivityName.DRINK, "inventory_at_least", ("drink", 5)),
        belief_state,
    )
    waiting = state["pending_atomic"]
    assert waiting["action"] == int(CrafterAction.NOOP)
    assert waiting["movement_kind"] == "none"
    assert waiting["source_cell"] == waiting["destination_cell"] == [0, 0]

    ll.on_action_status(
        state,
        "actuator",
        ActionStatus(public_status()),
        action_id=waiting["action_id"],
    )
    awake_observation = [
        value.replace("MadeOf(R1, tree)", "MadeOf(R1, water)")
        for value in symbolic_observation(inventory={"drink": 1, "energy": 5})
    ]
    ll.on_observation(
        state,
        "perceptor",
        Observation(awake_observation, observation_type="crafter-symbolic"),
    )
    assert state["active_activity"]["atomic"][0]["action"] == int(CrafterAction.NOOP)
    assert state["pending_atomic"]["action"] == int(CrafterAction.MOVE_RIGHT)


def ll_state_for_goal(step: PlanStep, state_data: dict[str, Any]) -> tuple[ActivityLLReasoner, FakeState]:
    ll = ActivityLLReasoner(module_id="ll", initial_state={})
    ll._perceptor_id, ll._actuator_id = "perceptor", "actuator"
    ll._knowledge_id, ll._goal_graph_id = "knowledge", "goals"
    ll.on_init(activity_action_limits=ACTIVITY_ACTION_LIMITS, total_action_limit=900)
    state = FakeState(initial_states()[LLREASONER])
    state["belief_state"] = state_data
    state["phase"] = "idle"
    ll.on_goal_update(state, "goals", [goal(step, revision=state_data["revision"])])
    return ll, state


def frontier_state(*, food: int = 5, drink: int = 5) -> dict[str, Any]:
    return detailed_state(
        inventory={"food": food, "drink": drink},
        terrain={"-1,0": "grass"},
        safe=["0,0", "-1,0"],
        visible=["-1,0"],
        offsets=[[-1, 0], [1, 0]],
    )


def test_food_explore_protects_its_need_at_intervention_threshold() -> None:
    _, state = ll_state_for_goal(
        PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("food",)),
        frontier_state(food=2),
    )
    assert state["pending_atomic"] is not None
    assert state["need_interruptions"] == 0


def test_survival_explore_continues_until_actual_death_or_global_budget() -> None:
    """Missing food alone does not abandon a live run."""
    _, state = ll_state_for_goal(
        PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("food",)),
        frontier_state(food=0),
    )
    assert state["pending_atomic"] is not None
    assert not [call for call in state.outbox.calls if call[0] == "send_goal_update"]


def test_higher_priority_different_need_interrupts_survival_explore() -> None:
    _, state = ll_state_for_goal(
        PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("food",)),
        frontier_state(food=2, drink=1),
    )
    terminal = next(call for call in state.outbox.calls if call[0] == "send_goal_update")
    outcome = outcome_from_goal(terminal[1][1][0])
    assert outcome.failure_reason == "need_interruption"
    assert outcome.interruption["need"] == "drink"


def test_total_action_cap_precedes_need_interruption() -> None:
    ll = ActivityLLReasoner(module_id="ll", initial_state={})
    ll._goal_graph_id = "goals"
    ll.on_init(activity_action_limits=ACTIVITY_ACTION_LIMITS, total_action_limit=1)
    state = FakeState(initial_states()[LLREASONER])
    state["belief_state"] = frontier_state(food=1)
    request = goal(PlanStep(ActivityName.GET_WOOD, "inventory_at_least", ("wood", 1)))
    state["active_activity"] = {"goal": {"state": [{"predicate": "inventory_at_least", "arguments": ["wood", 1], "extras": None}], "extras": dict(request.extras)}, "atomic": []}
    state["actions"] = 1
    ll._continue(state)
    terminal = next(call for call in state.outbox.calls if call[0] == "send_goal_update")
    assert outcome_from_goal(terminal[1][1][0]).failure_reason == "total_action_bound"


def test_bridges_and_goal_graph_keep_typed_single_flight_edges(monkeypatch) -> None:
    values = initial_states()
    perceptor = CrafterHybridPerceptor(module_id="perceptor", initial_state={})
    pstate = FakeState(values[PERCEPTOR], directory())
    perceptor.on_first(pstate)
    observed = []
    monkeypatch.setattr(perceptor, "observe", lambda *args, **kwargs: observed.append((args, kwargs)))
    perceptor.on_request(pstate, "ll")
    perceptor.on_observation(pstate, "environment", observation=symbolic_observation())
    assert observed == [(("environment",), {})]

    actuator = CrafterHybridActuator(module_id="actuator", initial_state={})
    astate = FakeState(values[ACTUATOR], directory())
    actuator.on_first(astate)
    acted = []
    monkeypatch.setattr(actuator, "act", lambda *args, **kwargs: acted.append((args, kwargs)))
    actuator.on_request(astate, "ll", action=1, action_id="action-1")
    actuator.on_status(astate, "environment", **public_status())
    actuator.on_last(astate)
    actuator.on_last(astate)
    assert acted[-1] == (("environment",), {"action": "close"})

    graph = ActivityGoalGraph(module_id="goals", initial_state={})
    gstate = FakeState(values[GOALGRAPH], directory())
    graph.on_first(gstate)
    request = goal(PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("tree",)))
    graph.on_goal_update(gstate, "hl", [request])
    graph.on_goal_update(
        gstate,
        "ll",
        [terminal_activity_goal(request, status="failed", completion_revision=1, atomic=[], failure_reason="no_frontier")],
    )
    assert gstate["active_goal_id"] is None


@pytest.mark.parametrize("outcome_first", [False, True])
def test_hl_joins_terminal_and_belief_in_either_order(outcome_first: bool) -> None:
    hl = HybridSymbolicReasoner(module_id="hl", initial_state={})
    hl._knowledge_id, hl._goal_graph_id = "knowledge", "goals"
    state = FakeState(initial_states()[HLREASONER])
    hl.on_belief_update(
        state,
        "knowledge",
        abstract_snapshot(1, reachable=("tree",), placements=("table",)),
    )
    request = next(call for call in state.outbox.calls if call[0] == "send_goals")[1][1][0]
    terminal = terminal_activity_goal(
        request,
        status="succeeded",
        completion_revision=2,
        atomic=[atomic_row(1, 1, 2)],
    )
    if outcome_first:
        hl.on_goal_update(state, "goals", [terminal])
        hl.on_belief_update(
            state,
            "knowledge",
            abstract_snapshot(
                2,
                inventory={"wood": 2},
                reachable=("tree",),
                placements=("table",),
                known_cells=14,
            ),
        )
    else:
        hl.on_belief_update(
            state,
            "knowledge",
            abstract_snapshot(
                2,
                inventory={"wood": 2},
                reachable=("tree",),
                placements=("table",),
                known_cells=14,
            ),
        )
        hl.on_goal_update(state, "goals", [terminal])
    assert state["completed_activities"] == 1
    row = state["hierarchy"]["activities"][0]
    assert row["final_value"] == 2
    assert row["known_cell_count_start"] == 10
    assert row["known_cell_count_end"] == 14


def test_hl_replaces_interrupted_sleep_with_the_selected_need_recovery() -> None:
    hl = HybridSymbolicReasoner(module_id="hl", initial_state={})
    hl._knowledge_id, hl._goal_graph_id = "knowledge", "goals"
    state = FakeState(initial_states()[HLREASONER])
    hl.on_belief_update(
        state,
        "knowledge",
        abstract_snapshot(
            1,
            inventory={"health": 2, "food": 5, "drink": 5, "energy": 3},
            reachable=("water",),
        ),
    )
    request = next(call for call in state.outbox.calls if call[0] == "send_goals")[1][1][0]
    assert activity_from_goal(request).activity is ActivityName.SLEEP
    interruption = {
        "need": "drink",
        "observed_value": 1,
        "interrupted_activity": "sleep",
        "interrupted_goal_id": "activity-1",
        "belief_revision": 2,
    }
    terminal = terminal_activity_goal(
        request,
        status="failed",
        completion_revision=2,
        atomic=[atomic_row(1, 1, 2)],
        failure_reason="need_interruption",
        interruption=interruption,
    )
    hl.on_goal_update(state, "goals", [terminal])
    hl.on_belief_update(
        state,
        "knowledge",
        abstract_snapshot(
            2,
            sleeping=True,
            inventory={"health": 2, "food": 5, "drink": 1, "energy": 4},
            reachable=("water",),
        ),
    )

    row = state["hierarchy"]["activities"][0]
    assert row["status"] == "interrupted"
    assert state["hierarchy"]["override"]["need"] == "drink"
    replacement = [call for call in state.outbox.calls if call[0] == "send_goals"][-1][1][1][0]
    assert activity_from_goal(replacement).activity is ActivityName.DRINK


def test_hl_persistent_intention_replans_through_diamond() -> None:
    hl = HybridSymbolicReasoner(module_id="hl", initial_state={})
    hl._knowledge_id, hl._goal_graph_id = "knowledge", "goals"
    state = FakeState(initial_states()[HLREASONER])
    inventory = dict(INVENTORY)
    usable: tuple[str, ...] = ()
    placements = ("table", "furnace")
    reachable = ("tree", "stone", "coal", "iron", "diamond")
    revision = 1
    hl.on_belief_update(
        state, "knowledge", abstract_snapshot(revision, inventory=inventory, reachable=reachable, usable=usable, placements=placements)
    )
    action_index = 1
    while state["phase"] != "succeeded":
        request = [call for call in state.outbox.calls if call[0] == "send_goals"][-1][1][1][0]
        spec = activity_from_goal(request)
        if spec.desired_predicate == "inventory_at_least":
            inventory[str(spec.desired_arguments[0])] = int(spec.desired_arguments[1])
        elif spec.desired_predicate == "usable_station_set":
            if spec.desired_arguments == ("table",):
                usable = ("table",)
                inventory["wood"] = 0
            else:
                usable = ("table", "table_furnace")
                inventory["stone"] = 0
        else:
            raise AssertionError(f"unexpected Explore in fully supported chain: {spec}")
        revision += 1
        terminal = terminal_activity_goal(
            request,
            status="succeeded",
            completion_revision=revision,
            atomic=[atomic_row(action_index, revision - 1, revision)],
        )
        action_index += 1
        hl.on_goal_update(state, "goals", [terminal])
        hl.on_belief_update(
            state,
            "knowledge",
            abstract_snapshot(
                revision,
                inventory=inventory,
                reachable=reachable,
                usable=usable,
                placements=placements,
            ),
        )
        assert len(state["hierarchy"]["activities"]) < 20
    assert state["hierarchy"]["intention"] == {
        "name": "obtain_diamond", "target_value": 1, "status": "succeeded"
    }
    assert state["hierarchy"]["highest_milestone"] == "diamond"
    assert tuple(MILESTONES) == (
        "start", "table", "wood_pickaxe", "stone_pickaxe", "furnace", "iron_pickaxe", "diamond"
    )


class FakeCrafter:
    def reset(self):
        return None

    def symbolic_observation(self):
        return symbolic_observation()

    def step(self, action):
        return None, 1.0, False, {
            "illegal_action": False,
            "inventory": INVENTORY,
            "achievements": {"collect_wood": 1},
        }

    def close(self):
        return None


def test_environment_uses_only_public_crafter_surface(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        CrafterHybridEnvironment,
        "_build_env",
        lambda self: setattr(self, "_env", FakeCrafter()),
    )
    state = environment_initial_state()
    init = {
        **state,
        "seed": 1,
        "expected_agent_id": "agent",
        "artifact_root": str(tmp_path),
        "no_mobs": True,
        "daylight_effects": False,
        "episode_length": 1000,
    }
    env = CrafterHybridEnvironment(init)
    env.on_observe(state, "agent")
    env.on_action(state, "agent", action=int(CrafterAction.DO))
    env.on_action(state, "agent", action="close")
    assert state["native_actions"] == 1 and state["close_requests"] == 1
    assert "applied_inventory_fixture" not in state


def valid_checker_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    treatment = treatment_for_run(0)
    states = initial_states(treatment)
    for component in states.values():
        component["active_seconds"] = 0.1
    states[PERCEPTOR].update(requests=2, observations=2)
    states[ACTUATOR].update(requests=1, statuses=1)
    states[LLREASONER].update(
        phase="idle",
        observation_requests=2,
        observations=2,
        belief_sends=2,
        goal_activations=1,
        terminal_updates=1,
        actions=1,
        action_statuses=1,
    )
    final = {
        "revision": 2,
        "sleeping": False,
        "inventory": dict(INVENTORY),
        "known_target_kinds": [],
        "reachable_target_kinds": [],
        "station_counts": {"table": 0, "furnace": 0},
        "usable_station_sets": [],
        "placement_opportunities": [],
        "known_cell_count": 2,
        "terminal": False,
        "dead": False,
        "experiment_error": None,
    }
    states[KNOWLEDGE].update(observed=2, forwarded=2, last_revision=2, beliefs=final)
    states[GOALGRAPH].update(dispatches=1, terminals=1)
    row = {
        "goal_id": "activity-1",
        "stage": "table",
        "plan_revision": 1,
        "activity": "explore",
        "desired": {
            "predicate": "placement_opportunity",
            "arguments": ["table"],
            "extras": None,
        },
        "based_on_revision": 1,
        "known_cell_count_start": 1,
        "known_cell_count_end": 2,
        "atomic": [atomic_row(1, 1, 2)],
        "completion_revision": 2,
        "final_value": False,
        "status": "failed",
        "failure_reason": "no_frontier",
        "interruption": None,
    }
    states[HLREASONER].update(
        phase="blocked",
        latest_revision=2,
        belief_updates=2,
        abstract_state=final,
        dispatches=1,
        terminals=1,
        failed_activities=1,
        terminal_reason="no_frontier:placement:table",
        exploration_episode={
            "owner": "technology:table",
            "purpose": "placement:table",
            "consecutive_bounds": 0,
        },
        hierarchy={
            "intention": {"name": "obtain_diamond", "target_value": 1, "status": "unachieved"},
            "current_stage": "table",
            "highest_milestone": "start",
            "override": None,
            "current_plan": {
                "based_on_revision": 1,
                "steps": [{"activity": "explore", "desired": {"predicate": "placement_opportunity", "arguments": ["table"]}}],
            },
            "plan_revision": 1,
            "activities": [row],
            "final_revision": 2,
            "terminal_reason": "no_frontier:placement:table",
            "status": "blocked",
        },
    )
    agent = {module_name(role, 0): state for role, state in states.items()}
    environment = environment_initial_state(treatment)
    environment.update(
        observation_requests=2,
        action_requests=1,
        native_actions=1,
        close_requests=1,
        closed=True,
    )
    logs = {"agent": ["[INFO] blocked"], "environment": ["[INFO] closed"]}
    return agent, environment, logs


def test_checker_accepts_clean_blocked_run_without_task_success() -> None:
    agent, environment, logs = valid_checker_fixture()
    assert check_results(agent, environment, logs, expected_recording=False)


def test_checker_rejects_non_mapping_treatment_without_raising() -> None:
    agent, environment, logs = valid_checker_fixture()
    agent[module_name(HLREASONER, 0)]["treatment"] = "bad"

    assert not check_results(
        agent,
        environment,
        logs,
        expected_recording=False,
        verbose=False,
    )


def test_malformed_treatment_does_not_abort_later_reported_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reporting,
        "_read_provenance",
        lambda _root: ({"provenance_id": "test-provenance"}, None),
    )
    expected = []
    for run_id in range(2):
        agent, environment, logs = valid_checker_fixture()
        treatment = treatment_for_run(run_id)
        agent[module_name(HLREASONER, 0)]["treatment"] = (
            "bad" if run_id == 0 else treatment
        )
        environment["treatment"] = treatment
        agent_id = f"exp_agent2_4_{run_id}"
        environment_id = f"exp_env2_4_{run_id}"
        agent_out = tmp_path / agent_id / "out"
        environment_out = tmp_path / environment_id / "out"
        agent_out.mkdir(parents=True)
        environment_out.mkdir(parents=True)
        for module_id, state in agent.items():
            (agent_out / f"{agent_id}.{module_id}.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
        (environment_out / f"{environment_id}.json").write_text(
            json.dumps(environment), encoding="utf-8"
        )
        for name, lines in logs.items():
            log_id = agent_id if name == "agent" else environment_id
            (tmp_path / f"{log_id}.log").write_text(
                "\n".join(lines), encoding="utf-8"
            )
        expected.append({
            "execution_id": f"run-{run_id}",
            "run_id": run_id,
            "factors": treatment,
        })

    report = reporting.process_execution_metrics(
        tmp_path, expected_executions=expected
    )

    assert report["executions"][0]["operationally_valid"] is False
    assert "treatment_identity_missing" in report["executions"][0][
        "operational_reasons"
    ]
    assert report["executions"][1]["operationally_valid"] is True
    rows = report["analyses"]["hierarchy_runs"]["rows"]
    assert any(row["execution_id"] == "run-1" for row in rows)
    assert report["analyses"]["hierarchy_runs"]["summary"][
        "diamond_success"
    ]["denominator"] == 1
    assert (tmp_path / "execution-metrics.json").is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_module",
        "illegal",
        "pending",
        "module_failure",
        "bad_status",
        "bad_interruption",
        "bad_log",
        "duplicate_close",
        "negative_known_cells",
        "decreasing_known_cells",
        "missing_known_end",
    ],
)
def test_checker_rejects_principal_invalidity(mutation: str) -> None:
    agent, environment, logs = valid_checker_fixture()
    high = agent[module_name(HLREASONER, 0)]
    if mutation == "missing_module":
        agent.pop(module_name(PERCEPTOR, 0))
    elif mutation == "illegal":
        environment["illegal_actions"] = 1
    elif mutation == "pending":
        agent[module_name(ACTUATOR, 0)]["pending_action_id"] = "action-2"
    elif mutation == "module_failure":
        agent[module_name(KNOWLEDGE, 0)]["failure"] = "failed"
    elif mutation == "bad_status":
        high["hierarchy"]["status"] = "succeeded"
    elif mutation == "bad_interruption":
        high["hierarchy"]["activities"][0]["status"] = "interrupted"
    elif mutation == "bad_log":
        logs["environment"] = ["[ERROR] failed"]
    elif mutation == "duplicate_close":
        environment["close_requests"] = 2
    elif mutation == "negative_known_cells":
        high["hierarchy"]["activities"][0]["known_cell_count_start"] = -1
    elif mutation == "decreasing_known_cells":
        high["hierarchy"]["activities"][0]["known_cell_count_end"] = 0
    else:
        high["hierarchy"]["activities"][0]["known_cell_count_end"] = None
    assert not check_results(agent, environment, logs, expected_recording=False, verbose=False)


@pytest.mark.parametrize("path", ["missing.mp4", "../escape.mp4", "empty.mp4", "bad.mp4"])
def test_video_gate_rejects_invalid_artifacts(tmp_path: Path, path: str) -> None:
    if path in {"empty.mp4", "bad.mp4"}:
        (tmp_path / path).write_bytes(b"" if path == "empty.mp4" else b"not video")
    assert not _readable_video(tmp_path, path)


def test_source_digest_is_deterministic_and_excludes_generated_bytecode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.py").write_text("b = 2\n", encoding="utf-8")
    (source / "a.py").write_text("a = 1\n", encoding="utf-8")
    before = _source_digest((source,))
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "a.pyc").write_bytes(b"generated")

    assert _source_digest((source,)) == before
    (source / "a.py").write_text("a = 3\n", encoding="utf-8")
    assert _source_digest((source,)) != before


def test_provenance_manifest_is_reused_and_rejects_malformed_or_mixed_sources(
    tmp_path: Path,
) -> None:
    manifest = {"protocol_version": "test", "provenance_id": "a" * 64}
    _ensure_execution_provenance(tmp_path, manifest)
    _ensure_execution_provenance(tmp_path, manifest)
    path = tmp_path / "execution-provenance.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Malformed"):
        _ensure_execution_provenance(tmp_path, manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        _ensure_execution_provenance(
            tmp_path, {**manifest, "provenance_id": "b" * 64}
        )


def test_fifty_matched_seeds_do_not_wrap() -> None:
    """The full CLI cohort never silently repeats a seed."""
    from mha_exp_level2_cr.exp2_3.treatment import TASKS
    seeds = [treatment_for_run(run)["seed"] for run in range(50)]
    assert seeds == [task.seed for task in TASKS]
    assert len(set(seeds)) == 50
    with pytest.raises(ValueError):
        treatment_for_run(50)


def test_timeout_finishes_activity_without_dispatching_another_action() -> None:
    """A time limit uses the ordinary correlated terminal pipeline."""
    ll = ActivityLLReasoner(module_id="ll", initial_state={})
    ll._goal_graph_id = "goals"
    ll.on_init(activity_action_limits=ACTIVITY_ACTION_LIMITS, total_action_limit=None)
    state = FakeState(initial_states()[LLREASONER])
    state.time = 845.0
    state["phase"] = "idle"
    state["belief_state"] = frontier_state()
    request = goal(PlanStep(ActivityName.GET_WOOD, "inventory_at_least", ("wood", 1)))
    ll.on_goal_update(state, "goals", [request])
    terminal = next(call for call in state.outbox.calls if call[0] == "send_goal_update")
    assert outcome_from_goal(terminal[1][1][0]).failure_reason == "time_budget_exhausted"
    assert state["time_bound_reached"] and state["actions"] == 0
    assert state["active_activity"] is None


@pytest.mark.parametrize("changed_field", [None, "experiment_source_sha256", "crafter_source_sha256", "mhagenta_source_sha256", "uv_lock_sha256"])
def test_provenance_allows_only_git_metadata_changes(tmp_path: Path, changed_field: str | None) -> None:
    """Unrelated commits cannot abort a cohort; changed runtime bytes still do."""
    import hashlib
    def signed(**updates):
        payload = {"protocol_version": "test", "workspace_git_commit": "a" * 40,
                   "workspace_git_dirty": True, "mhagenta_git_commit": "b" * 40,
                   "mhagenta_git_dirty": False,
                   **{key: "c" * 64 for key in ("experiment_source_sha256", "crafter_source_sha256", "mhagenta_source_sha256", "uv_lock_sha256")},
                   **updates}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {**payload, "provenance_id": digest}
    original = signed()
    _ensure_execution_provenance(tmp_path, original)
    updates = {"workspace_git_commit": "d" * 40, "workspace_git_dirty": False}
    if changed_field:
        updates[changed_field] = "e" * 64
        with pytest.raises(RuntimeError, match="differs"):
            _ensure_execution_provenance(tmp_path, signed(**updates))
    else:
        _ensure_execution_provenance(tmp_path, signed(**updates))
    assert json.loads((tmp_path / "execution-provenance.json").read_text()) == original


@pytest.mark.parametrize("reachable, reason", [((), "exploration_bound"), (("diamond",), "activity_action_bound")])
def test_diamond_search_retries_beyond_old_activity_limits(reachable, reason) -> None:
    """A rare diamond can require repeated activities without ending the run."""
    hl = HybridSymbolicReasoner(module_id="hl", initial_state={})
    hl._knowledge_id, hl._goal_graph_id = "knowledge", "goals"
    state = FakeState(initial_states()[HLREASONER])
    inventory = {"iron_pickaxe": 1}
    hl.on_belief_update(state, "knowledge", abstract_snapshot(1, inventory=inventory, reachable=reachable))
    for revision in range(2, 8):
        request = [call for call in state.outbox.calls if call[0] == "send_goals"][-1][1][1][0]
        terminal = terminal_activity_goal(request, status="failed", completion_revision=revision,
                                         atomic=[atomic_row(revision-1, revision-1, revision)], failure_reason=reason)
        hl.on_goal_update(state, "goals", [terminal])
        hl.on_belief_update(state, "knowledge", abstract_snapshot(revision, inventory=inventory, reachable=reachable))
        assert state["phase"] == "active" and state["terminal_reason"] is None
    assert state["dispatches"] == 7 and state["failed_activities"] == 6
    assert not [call for call in state.outbox.calls if call[0] == "terminate_agent"]


def test_global_thousand_action_cap_prevents_next_dispatch() -> None:
    """Retrying a compound activity never resets the run's global action cap."""
    ll = ActivityLLReasoner(module_id="ll", initial_state={})
    ll._goal_graph_id = "goals"
    ll.on_init(activity_action_limits=ACTIVITY_ACTION_LIMITS)
    state = FakeState(initial_states()[LLREASONER])
    state.update(phase="idle", actions=1000, belief_state=frontier_state())
    ll.on_goal_update(state, "goals", [goal(PlanStep(ActivityName.EXPLORE, "reachable_target_kind", ("diamond",)))])
    terminal = next(call for call in state.outbox.calls if call[0] == "send_goal_update")
    assert outcome_from_goal(terminal[1][1][0]).failure_reason == "total_action_bound"
    assert state["actions"] == 1000 and state["total_action_bound_reached"]
    assert not [call for call in state.outbox.calls if call[0] == "request_action"]


@pytest.mark.parametrize("dead, reason", [(True, "death"), (False, "episode_limit")])
def test_environment_termination_records_actual_reason(dead: bool, reason: str) -> None:
    """Separate death from the native episode limit in final outcomes."""
    hl = HybridSymbolicReasoner(module_id="hl", initial_state={})
    hl._knowledge_id, hl._goal_graph_id = "knowledge", "goals"
    state = FakeState(initial_states()[HLREASONER])
    beliefs = abstract_snapshot(1, terminal=True)
    beliefs = [Belief("dead", (dead,)) if b.predicate == "dead" else b for b in beliefs]
    hl.on_belief_update(state, "knowledge", beliefs)
    assert state["terminal_reason"] == reason
    assert state["phase"] == "environment_terminal"
