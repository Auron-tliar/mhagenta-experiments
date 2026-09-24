from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mha_exp_common.names import ACTUATOR, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import module_name
from mha_exp_level2_cr.exp2_3.beliefs import (
    BeliefRevisionError,
    CrafterAction,
    ObservationError,
    beliefs_to_percept,
    initial_belief_state,
    parse_symbolic_observation,
    percept_to_beliefs,
    revise_belief_state,
    summarize_belief_state,
    support_signature,
)
from mha_exp_level2_cr.exp2_3.planning import (
    GroundedAction,
    PlanningOutcome,
    PlanningService,
    TechnologyStage,
    action_sound,
    build_problem_pddl,
    choose_exploration_action,
    derive_technology_stage,
    plan_goal_observed,
    recovery_complete,
    recovery_entry_threshold,
    recovery_supported,
    select_recovery_intention,
    stage_supported,
)
from mha_exp_level2_cr.exp2_3.runner import (
    EXPLORATION_BURST_SIZE,
    CrafterBDIEnvironment,
    CrafterBDIReasoner,
    _environment_initial_state,
    _initial_states,
    _result_errors,
)
from mha_exp_level2_cr.exp2_3.reporting import process_execution_metrics
from mha_exp_level2_cr.exp2_3.treatment import TASKS, treatment_for_run


def symbolic_observation(
    *,
    facing: str = "R1",
    sleeping: bool = False,
    inventory: dict[str, int] | None = None,
    tiles: dict[str, tuple[str, str]] | None = None,
) -> list[str]:
    values = {
        "health": 9,
        "food": 9,
        "drink": 9,
        "energy": 9,
        "wood": 0,
        "stone": 0,
        "coal": 0,
        "iron": 0,
        "diamond": 0,
        "wood_pickaxe": 0,
        "stone_pickaxe": 0,
        "iron_pickaxe": 0,
    }
    values.update(inventory or {})
    tiles = tiles or {
        "L1": ("grass", "none"),
        "R1": ("grass", "none"),
        "U1": ("grass", "none"),
        "D1": ("grass", "none"),
    }
    result = [
        f"Sleeping() = {'true' if sleeping else 'false'}",
        f"Facing({facing}) = true",
        *(f"Have({name}) = {count}" for name, count in values.items()),
    ]
    for location, (material, occupant) in tiles.items():
        result.extend(
            (
                f"MadeOf({location}, {material}) = true",
                f"OccupiedBy({location}, {occupant}) = true",
            )
        )
    return result


def revised_state(
    *,
    inventory: dict[str, int] | None = None,
    tiles: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    return revise_belief_state(
        initial_belief_state(),
        parse_symbolic_observation(
            symbolic_observation(inventory=inventory, tiles=tiles)
        ),
        action_result=None,
        pending_action=None,
    ).belief_state


def direct_state(
    *,
    inventory: dict[str, int] | None = None,
    terrain: dict[str, str] | None = None,
    occupants: dict[str, dict[str, Any]] | None = None,
    facing: str = "right",
) -> dict[str, Any]:
    values = {
        "health": 9,
        "food": 9,
        "drink": 9,
        "energy": 9,
        "wood": 0,
        "stone": 0,
        "coal": 0,
        "iron": 0,
        "diamond": 0,
        "wood_pickaxe": 0,
        "stone_pickaxe": 0,
        "iron_pickaxe": 0,
    }
    values.update(inventory or {})
    terrain = dict(
        terrain
        or {
            "-1,0": "grass",
            "1,0": "grass",
            "0,-1": "grass",
            "0,1": "grass",
        }
    )
    occupants = dict(occupants or {})
    safe = sorted(
        key
        for key, material in terrain.items()
        if material in {"grass", "path", "sand"} and key not in occupants
    )
    blocked = sorted(
        key
        for key, material in terrain.items()
        if key not in safe and material not in {"lava", "unknown"}
    )
    clear = sorted(key for key in terrain if key not in occupants)
    state = initial_belief_state()
    state.update(
        revision=1,
        facing=facing,
        inventory=values,
        terrain=terrain,
        occupants=occupants,
        visible_cells=sorted(terrain),
        known=sorted({"0,0", *terrain}),
        clear=clear,
        safe_to_enter=safe,
        reachable=sorted({"0,0", *safe}),
        movement_blocked=blocked,
        visit_counts={"0,0": 1},
    )
    return state


def action_result(
    action_id: str,
    *,
    achievement: str | None = None,
    done: bool = False,
    dead: bool = False,
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "reward": 0.0,
        "done": done,
        "dead": dead,
        "illegal_action": False,
        "new_achievements": [achievement] if achievement else [],
    }


def domain_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "src/mha_exp_level2_cr/exp2_3/crafter-domain.pddl"
    )


def test_treatment_manifest_is_strict_and_frozen() -> None:
    assert len(TASKS) == 50
    assert [task.seed for task in TASKS[:6]] == list(range(1000, 1006))
    assert [task.initial_support for task in TASKS].count("exploration-first") == 25
    assert treatment_for_run(0)["protocol_version"] == "2-3-cr-full-domain-diamond-50-v1"
    with pytest.raises(ValueError):
        treatment_for_run(50)


def test_symbolic_percept_round_trip_and_strict_validation() -> None:
    percept = parse_symbolic_observation(symbolic_observation())
    assert beliefs_to_percept(percept_to_beliefs(percept)) == percept
    with pytest.raises(ObservationError, match="Duplicate Sleeping"):
        parse_symbolic_observation(symbolic_observation() + ["Sleeping() = false"])


def test_revision_tracks_known_center_reachability_and_visits() -> None:
    state = revised_state(
        tiles={
            "L1": ("grass", "none"),
            "R1": ("tree", "none"),
            "U1": ("lava", "none"),
            "D1": ("grass", "cow"),
        }
    )
    assert "0,0" in state["known"]
    assert state["visit_counts"] == {"0,0": 1}
    assert set(state["reachable"]) == {"0,0", "-1,0"}
    assert "1,0" in state["movement_blocked"]
    summary = summarize_belief_state(state)
    assert summary["known_terrain"] == len(state["known"])
    assert summary["unknown_frontier"] > 0


def test_revision_replaces_static_contradictions_and_discards_stale_occupants() -> None:
    state = revised_state(
        tiles={
            "R1": ("tree", "none"),
            "D1": ("grass", "cow"),
        }
    )
    state["cow_target"] = {"cell": "0,1", "hits": 1}
    revised = revise_belief_state(
        state,
        parse_symbolic_observation(
            symbolic_observation(tiles={"R1": ("grass", "none")})
        ),
        action_result=None,
        pending_action=None,
    )
    assert revised.belief_state["terrain"]["1,0"] == "grass"
    assert revised.belief_state["occupants"] == {}
    assert revised.belief_state["cow_target"] is None


def test_revision_requires_exact_action_correlation() -> None:
    state = revised_state(tiles={"D1": ("grass", "none")})
    pending = {
        "action": int(CrafterAction.MOVE_DOWN),
        "action_id": "action-1",
        "movement_kind": "walk",
        "source_cell": "0,0",
        "destination_cell": "0,1",
        "operator": {"name": "walk-down", "arguments": []},
    }
    percept = parse_symbolic_observation(
        symbolic_observation(facing="D1", tiles={"U1": ("grass", "none")})
    )
    result = revise_belief_state(
        state,
        percept,
        action_result=action_result("action-1"),
        pending_action=pending,
    )
    assert result.belief_state["player"] == [0, 1]
    with pytest.raises(BeliefRevisionError, match="action_id"):
        revise_belief_state(
            state,
            percept,
            action_result=action_result("wrong"),
            pending_action=pending,
        )


def test_support_signature_ignores_irrelevant_terrain_but_tracks_targets() -> None:
    state = direct_state(terrain={"1,0": "grass"})
    signature = support_signature(state, "collect-table-wood")
    state["terrain"]["4,4"] = "grass"
    assert support_signature(state, "collect-table-wood") == signature
    state["terrain"]["4,4"] = "tree"
    assert support_signature(state, "collect-table-wood") != signature


@pytest.mark.parametrize(
    ("inventory", "terrain", "expected"),
    [
        ({}, None, "collect-table-wood"),
        ({"wood": 2}, None, "place-table"),
        ({"wood": 0}, {"1,0": "table"}, "collect-wood-pickaxe-wood"),
        ({"wood": 1}, {"1,0": "table"}, "make-wood-pickaxe"),
        (
            {"wood_pickaxe": 1},
            {"1,0": "table"},
            "collect-stone-pickaxe-wood",
        ),
        (
            {"wood_pickaxe": 1, "wood": 1},
            {"1,0": "table"},
            "collect-stone-pickaxe-stone",
        ),
        (
            {"wood_pickaxe": 1, "wood": 1, "stone": 1},
            {"1,0": "table"},
            "make-stone-pickaxe",
        ),
        (
            {"wood_pickaxe": 1, "stone_pickaxe": 1},
            {"1,0": "table"},
            "collect-furnace-stone",
        ),
        (
            {"wood_pickaxe": 1, "stone_pickaxe": 1, "stone": 4},
            {"1,0": "table", "-1,0": "grass", "0,1": "grass"},
            "place-furnace",
        ),
        (
            {"wood_pickaxe": 1, "stone_pickaxe": 1},
            {"1,0": "table", "0,1": "furnace"},
            "collect-iron-pickaxe-wood",
        ),
        (
            {"wood_pickaxe": 1, "stone_pickaxe": 1, "wood": 1},
            {"1,0": "table", "0,1": "furnace"},
            "collect-iron-pickaxe-coal",
        ),
        (
            {
                "wood_pickaxe": 1,
                "stone_pickaxe": 1,
                "wood": 1,
                "coal": 1,
            },
            {"1,0": "table", "0,1": "furnace"},
            "collect-iron-pickaxe-iron",
        ),
        (
            {
                "wood_pickaxe": 1,
                "stone_pickaxe": 1,
                "wood": 1,
                "coal": 1,
                "iron": 1,
            },
            {"1,0": "table", "0,1": "furnace"},
            "make-iron-pickaxe",
        ),
        (
            {
                "wood_pickaxe": 1,
                "stone_pickaxe": 1,
                "iron_pickaxe": 1,
            },
            {"1,0": "table", "0,1": "furnace"},
            "collect-diamond",
        ),
    ],
)
def test_fourteen_technology_stages(
    inventory: dict[str, int],
    terrain: dict[str, str] | None,
    expected: str,
) -> None:
    state = direct_state(inventory=inventory, terrain=terrain)
    assert derive_technology_stage(state).name == expected


def test_stage_derivation_completes_only_with_diamond_inventory() -> None:
    state = direct_state(inventory={"diamond": 1})
    assert derive_technology_stage(state) is None


def test_station_support_requires_cardinal_usable_layouts() -> None:
    state = direct_state(inventory={"wood": 2})
    assert stage_supported(
        TechnologyStage("place-table", "place", "(table-established)"), state
    ) == (True, None)
    state = direct_state(
        inventory={"stone": 4},
        terrain={"1,1": "table", "1,0": "grass", "0,1": "grass"},
    )
    assert stage_supported(
        TechnologyStage("place-furnace", "place", "(furnace-established)"),
        state,
    ) == (False, "missing-site")


def test_recovery_hysteresis_ties_and_cross_need_preemption() -> None:
    state = direct_state(inventory={"food": 2, "drink": 2})
    selected = select_recovery_intention(state, None)
    assert selected["need"] == "drink"
    selected["id"] = "intention-1"
    state["inventory"]["food"] = 0
    assert select_recovery_intention(state, selected)["need"] == "food"
    state["inventory"]["food"] = 5
    assert recovery_complete(
        {
            "kind": "recovery",
            "need": "food",
            "completion_threshold": 5,
        },
        state,
    )


def test_energy_recovery_requires_waking_at_full_energy() -> None:
    state = direct_state(inventory={"energy": 1})
    selected = select_recovery_intention(state, None)
    assert selected["need"] == "energy"
    state["inventory"]["drink"] = 0
    state["inventory"]["energy"] = 6
    state["sleeping"] = True
    assert select_recovery_intention(state, selected)["need"] == "drink"
    state["inventory"]["energy"] = 9
    assert not recovery_complete(selected, state)
    state["sleeping"] = False
    assert recovery_complete(selected, state)


def test_recovery_entry_thresholds_increase_only_at_low_health() -> None:
    assert recovery_entry_threshold("food", 4) == 2
    assert recovery_entry_threshold("drink", 4) == 2
    assert recovery_entry_threshold("energy", 4) == 1
    assert recovery_entry_threshold("food", 3) == 4
    assert recovery_entry_threshold("drink", 3) == 4
    assert recovery_entry_threshold("energy", 3) == 3


def test_recovery_support_uses_fresh_targets_and_reachable_water() -> None:
    drink = {"kind": "recovery", "need": "drink", "name": "restore-drink"}
    state = direct_state(terrain={"1,0": "water"})
    assert recovery_supported(drink, state) == (True, None)
    food = {"kind": "recovery", "need": "food", "name": "restore-food"}
    state = direct_state(
        occupants={"1,0": {"kind": "ripe-plant", "last_seen_revision": 0}}
    )
    assert recovery_supported(food, state) == (False, "missing-target")

    state = direct_state(
        terrain={"0,0": "grass", "1,0": "grass", "9,9": "grass"},
        occupants={
            "1,0": {
                "kind": "cow",
                "last_seen_revision": 1,
            },
            "9,9": {
                "kind": "ripe-plant",
                "last_seen_revision": 1,
            },
        },
    )
    assert recovery_supported(food, state) == (True, None)


@pytest.mark.parametrize(
    ("state", "intention", "expected_action"),
    [
        (
            direct_state(terrain={"1,0": "tree"}),
            TechnologyStage(
                "collect-table-wood", "collect", "(>= (wood) 1)", 2
            ),
            "do-tree",
        ),
        (
            direct_state(
                inventory={"wood": 2},
                terrain={
                    "1,0": "grass",
                    "-1,0": "grass",
                    "0,1": "grass",
                },
            ),
            TechnologyStage("place-table", "place", "(table-established)"),
            "place-table",
        ),
        (
            direct_state(
                inventory={"wood": 1},
                terrain={"1,0": "table"},
            ),
            TechnologyStage(
                "make-wood-pickaxe",
                "craft",
                "(>= (wood-pickaxe) 1)",
                1,
            ),
            "make-wood-pickaxe",
        ),
        (
            direct_state(
                inventory={"stone_pickaxe": 1},
                terrain={"1,0": "iron"},
            ),
            TechnologyStage(
                "collect-iron-pickaxe-iron",
                "collect",
                "(>= (iron-count) 1)",
                1,
            ),
            "do-iron",
        ),
        (
            direct_state(
                inventory={
                    "wood": 1,
                    "coal": 1,
                    "iron": 1,
                    "stone_pickaxe": 1,
                },
                terrain={"1,0": "table", "0,1": "furnace"},
            ),
            TechnologyStage(
                "make-iron-pickaxe",
                "craft",
                "(>= (iron-pickaxe) 1)",
                1,
            ),
            "make-iron-pickaxe",
        ),
        (
            direct_state(
                inventory={"iron_pickaxe": 1},
                terrain={"1,0": "diamond"},
            ),
            TechnologyStage(
                "collect-diamond",
                "collect",
                "(>= (diamond-count) 1)",
                1,
            ),
            "do-diamond",
        ),
    ],
)
def test_real_lpg_solves_representative_full_domain_stages(
    state: dict[str, Any],
    intention: TechnologyStage,
    expected_action: str,
) -> None:
    value = {"id": "intention-1", "kind": "technology", **intention.as_dict()}
    outcome = PlanningService(domain_path(), timeout=10.0).solve(
        state, value, f"{intention.name}-test", remaining_steps=100
    )
    assert outcome.accepted, outcome.attempt
    assert outcome.attempt["classification"] == "accepted"
    assert expected_action in [action.name for action in outcome.actions]


def test_real_lpg_solves_drink_recovery() -> None:
    state = direct_state(terrain={"1,0": "water"})
    intention = {
        "id": "intention-1",
        "kind": "recovery",
        "name": "restore-drink",
        "need": "drink",
        "goal_predicate": "(drink-restored)",
    }
    outcome = PlanningService(domain_path(), timeout=10.0).solve(
        state, intention, "restore-drink-test", remaining_steps=100
    )
    assert outcome.accepted, outcome.attempt
    assert "do-water" in [action.name for action in outcome.actions]


def test_planning_scope_prevents_loopy_movement_plans() -> None:
    terrain = {f"{x},0": "grass" for x in range(1, 10)}
    terrain["10,0"] = "water"
    state = direct_state(terrain=terrain)
    intention = {
        "id": "intention-1",
        "kind": "recovery",
        "name": "restore-drink",
        "need": "drink",
        "goal_predicate": "(drink-restored)",
    }
    outcome = PlanningService(domain_path(), timeout=10.0).solve(
        state, intention, "directed-route-test", remaining_steps=100
    )
    assert outcome.accepted, outcome.attempt
    assert len(outcome.actions) == 10
    assert outcome.actions[-1].name == "do-water"


def test_placement_markers_start_false_and_disconnected_cells_are_omitted() -> None:
    state = direct_state(
        inventory={"wood": 2},
        terrain={
            "1,0": "grass",
            "-1,0": "grass",
            "0,1": "grass",
            "20,20": "table",
        },
    )
    intention = {
        "name": "place-table",
        "kind": "technology",
        "goal_predicate": "(table-established)",
    }
    problem, _ = build_problem_pddl(
        problem_name="placement", belief_state=state, intention=intention
    )
    initial = problem.split("(:goal", 1)[0]
    assert "(table-established)" not in initial
    assert "c_p20_p20" not in problem
    assert "(future-station-layout" in problem


def test_runtime_soundness_requires_visible_walks_tools_and_coupling() -> None:
    state = direct_state(terrain={"1,0": "grass"})
    walk = GroundedAction(
        "walk-right",
        ("c_p0_p0", "c_p1_p0"),
        int(CrafterAction.MOVE_RIGHT),
        "walk",
        "0,0",
        "1,0",
        False,
        None,
    )
    assert action_sound(walk, state) == (True, None)
    state["visible_cells"] = []
    assert action_sound(walk, state)[1] == "walk-destination-not-visible"
    iron = GroundedAction(
        "do-iron",
        ("c_p0_p0", "c_p1_p0", "right"),
        int(CrafterAction.DO),
        "none",
        "0,0",
        "1,0",
        False,
        None,
    )
    state = direct_state(terrain={"1,0": "iron"})
    assert action_sound(iron, state)[1] == "missing-stone-pickaxe"


def test_expected_goals_distinguish_unit_progress_from_failure() -> None:
    state = direct_state(inventory={"wood": 1})
    expected = {"kind": "inventory-at-least", "item": "wood", "count": 1}
    assert plan_goal_observed(expected, state, action_result("action-1"))
    expected["count"] = 2
    assert not plan_goal_observed(expected, state, action_result("action-1"))
    station = {
        "kind": "station-established",
        "marker": "table-established",
        "cell": "1,0",
    }
    state["terrain"]["1,0"] = "table"
    assert plan_goal_observed(station, state, action_result("action-2"))


def test_exploration_uses_known_frontiers_and_tool_aware_boundaries() -> None:
    state = direct_state(
        inventory={"wood_pickaxe": 1},
        terrain={"1,0": "stone"},
    )
    state["known"] = sorted(
        f"{x},{y}" for x in range(-4, 5) for y in range(-3, 4)
    )
    action = choose_exploration_action(state, __import__("random").Random(0))
    assert action.name in {"turn-right", "do-stone"}
    assert action.destination_cell == "1,0"


class FakeOutbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return call


class FakeState(dict[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(deepcopy(values))
        self.outbox = FakeOutbox()
        self.time = 0.0


class StubPlanning:
    def __init__(
        self,
        actions: list[GroundedAction] | None = None,
        *,
        reject: bool = False,
        error: bool = False,
    ) -> None:
        self.actions = actions or []
        self.reject = reject
        self.error = error
        self.calls = 0

    def solve(self, *args: Any, **kwargs: Any) -> PlanningOutcome:
        self.calls += 1
        if self.error:
            return PlanningOutcome(
                [],
                {
                    "stage": "collect-table-wood",
                    "engine": "lpg",
                    "status": "ERROR",
                    "elapsed_seconds": 0.01,
                    "plan_length": None,
                    "validation_status": None,
                    "accepted": False,
                    "classification": "planner-exception",
                    "rejection": "planner-error",
                    "error": "fixture failure",
                },
                "error",
                None,
                None,
            )
        if self.reject:
            return PlanningOutcome(
                [],
                {
                    "stage": "collect-table-wood",
                    "engine": "lpg",
                    "status": "TIMEOUT",
                    "elapsed_seconds": 0.01,
                    "plan_length": None,
                    "validation_status": None,
                    "accepted": False,
                    "classification": "timeout-or-unsolved",
                    "rejection": "planner-did-not-solve",
                    "error": None,
                },
                "rejected",
                None,
                None,
            )
        return PlanningOutcome(
            self.actions,
            {
                "stage": "collect-table-wood",
                "engine": "lpg",
                "status": "SOLVED_SATISFICING",
                "elapsed_seconds": 0.01,
                "plan_length": len(self.actions),
                "validation_status": "VALID",
                "accepted": True,
                "classification": "accepted",
                "rejection": None,
                "error": None,
            },
            "accepted",
            None,
            {"kind": "inventory-at-least", "item": "wood", "count": 1},
        )


def reasoner_with(planning: StubPlanning) -> tuple[CrafterBDIReasoner, FakeState]:
    reasoner = CrafterBDIReasoner(module_id="hlreasoner_0", initial_state={})
    reasoner._log_func = lambda *_: None
    reasoner._actuator_id = "actuator_0"
    reasoner._knowledge_id = "knowledge_0"
    reasoner._planning = planning  # type: ignore[assignment]
    state = FakeState(_initial_states()[HLREASONER])
    state["run"]["phase"] = "awaiting-beliefs"
    return reasoner, state


def tree_percept(*, wood: int = 0, material: str = "tree") -> Any:
    return parse_symbolic_observation(
        symbolic_observation(
            inventory={"wood": wood},
            tiles={
                "R1": (material, "none"),
                "D1": ("grass", "none"),
                "L1": ("grass", "none"),
            },
        )
    )


def test_reasoner_records_unit_progress_and_plans_the_next_wood() -> None:
    action = GroundedAction(
        "do-tree",
        ("c_p0_p0", "c_p1_p0", "right"),
        int(CrafterAction.DO),
        "none",
        "0,0",
        "1,0",
        False,
        None,
    )
    planner = StubPlanning([action])
    reasoner, state = reasoner_with(planner)
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(tree_percept()),
        observation_seq=1,
        action_result=None,
    )
    pending = state["run"]["pending_action"]
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(tree_percept(wood=1)),
        observation_seq=2,
        action_result=action_result(
            pending["action_id"], achievement="collect_wood"
        ),
    )
    row = state["run"]["trace"][-1]
    assert row["monitoring"]["plan_goal"]["progress"] == "unit-progress"
    assert row["decision"]["kind"] == "planned-action"
    assert planner.calls == 2
    assert all(
        row["planning"]["intention_id"] == "intention-1"
        for row in state["run"]["trace"]
        if row["planning"] is not None
    )


def test_first_planning_failure_probes_then_second_starts_finite_burst() -> None:
    planner = StubPlanning(reject=True)
    reasoner, state = reasoner_with(planner)
    percept = tree_percept()
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(percept),
        observation_seq=1,
        action_result=None,
    )
    assert state["run"]["planning_failure_count"] == 1
    assert state["run"]["trace"][-1]["decision"]["kind"] == "explore-action"
    pending = state["run"]["pending_action"]
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(percept),
        observation_seq=2,
        action_result=action_result(pending["action_id"]),
    )
    assert planner.calls == 2
    assert state["run"]["planning_failure_count"] == 2
    assert state["run"]["forced_exploration_remaining"] == EXPLORATION_BURST_SIZE
    assert state["run"]["pending_action"]["forced_exploration"] is True
    assert all(
        row["planning"]["intention_id"] == "intention-1"
        for row in state["run"]["trace"]
        if row["planning"] is not None
    )


def test_missing_support_planning_record_carries_intention_id() -> None:
    planning = StubPlanning()
    reasoner, state = reasoner_with(planning)

    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(tree_percept(material="grass")),
        observation_seq=1,
        action_result=None,
    )

    row = state["run"]["trace"][-1]
    assert row["planning"]["classification"] == "missing-knowledge"
    assert row["planning"]["intention_id"] == row["decision"]["intention_id"]
    assert planning.calls == 0


def test_planner_error_record_carries_intention_id() -> None:
    reasoner, state = reasoner_with(StubPlanning(error=True))

    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(tree_percept()),
        observation_seq=1,
        action_result=None,
    )

    row = state["run"]["trace"][-1]
    assert row["planning"]["classification"] == "planner-exception"
    assert row["planning"]["intention_id"] == "intention-1"


def test_sleeping_recovery_preempts_but_defers_planning_until_awake() -> None:
    action = GroundedAction(
        "do-water",
        ("c_p0_p0", "c_p1_p0", "right"),
        int(CrafterAction.DO),
        "none",
        "0,0",
        "1,0",
        True,
        None,
    )
    planner = StubPlanning([action])
    reasoner, state = reasoner_with(planner)
    tiles = {
        "L1": ("grass", "none"),
        "R1": ("water", "none"),
        "U1": ("grass", "none"),
        "D1": ("grass", "none"),
    }
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(parse_symbolic_observation(symbolic_observation(
            inventory={"energy": 1},
            tiles=tiles,
        ))),
        observation_seq=1,
        action_result=None,
    )
    sleep_action = state["run"]["pending_action"]
    assert sleep_action["action"] == int(CrafterAction.SLEEP)

    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(parse_symbolic_observation(symbolic_observation(
            sleeping=True,
            inventory={"drink": 0, "energy": 6},
            tiles=tiles,
        ))),
        observation_seq=2,
        action_result=action_result(sleep_action["action_id"]),
    )
    sleeping_row = state["run"]["trace"][-1]
    drink_intention = state["run"]["intention"]
    assert [item["kind"] for item in sleeping_row["intentions"]] == [
        "preempted",
        "selected",
    ]
    assert drink_intention["need"] == "drink"
    assert sleeping_row["decision"]["native_action"] == int(CrafterAction.NOOP)
    assert sleeping_row["decision"]["intention_id"] == drink_intention["id"]
    assert sleeping_row["planning"] is None
    assert planner.calls == 0

    noop_action = state["run"]["pending_action"]
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(parse_symbolic_observation(symbolic_observation(
            inventory={"drink": 0, "energy": 9},
            tiles=tiles,
        ))),
        observation_seq=3,
        action_result=action_result(noop_action["action_id"]),
    )
    awake_row = state["run"]["trace"][-1]
    assert planner.calls == 1
    assert awake_row["planning"]["intention_id"] == drink_intention["id"]
    assert awake_row["decision"]["native_action"] == int(CrafterAction.DO)


def test_episode_terminal_precedes_action_budget() -> None:
    action = GroundedAction(
        "do-tree",
        ("c_p0_p0", "c_p1_p0", "right"),
        int(CrafterAction.DO),
        "none",
        "0,0",
        "1,0",
        False,
        None,
    )
    reasoner, state = reasoner_with(StubPlanning([action]))
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(tree_percept()),
        observation_seq=1,
        action_result=None,
    )
    state["run"]["agent_steps"] = 999
    pending = state["run"]["pending_action"]
    reasoner.on_belief_update(
        state,
        "knowledge_0",
        percept_to_beliefs(tree_percept(material="grass")),
        observation_seq=2,
        action_result=action_result(pending["action_id"], done=True),
    )
    assert state["run"]["terminal_reason"] == "episode_limit"


def test_public_environment_adapter_uses_only_public_state() -> None:
    source = inspect.getsource(CrafterBDIEnvironment)
    assert "._player" not in source and "._world" not in source

    class PublicEnv:
        def symbolic_observation(self) -> list[str]:
            return symbolic_observation()

        def step(self, action: int) -> tuple[None, float, bool, dict[str, Any]]:
            return None, 1.0, False, {
                "illegal_action": False,
                "inventory": {"health": 9, "diamond": 0},
                "achievements": {"collect_wood": 1},
            }

    adapter = object.__new__(CrafterBDIEnvironment)
    adapter._env = PublicEnv()
    state = _environment_initial_state()
    _, status = adapter.on_action(
        state, "actuator", action=int(CrafterAction.DO)
    )
    assert status is not None and status["new_achievements"] == ["collect_wood"]
    assert state["inventory"] == {"diamond": 0, "health": 9}


def valid_scientific_failure_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    states = _initial_states()
    named = {
        module_name(kind, 0): deepcopy(states[kind])
        for kind in (PERCEPTOR, ACTUATOR, LLREASONER, KNOWLEDGE, HLREASONER)
    }
    for value in named.values():
        value["active_seconds"] = 1.0
    named[module_name(PERCEPTOR, 0)].update(requests=1, observations=1)
    named[module_name(LLREASONER, 0)].update(
        observation_requests=1,
        observations_received=1,
        belief_updates_sent=1,
        action_statuses=0,
    )
    named[module_name(KNOWLEDGE, 0)].update(revisions=1, updates_forwarded=1)
    high = named[module_name(HLREASONER, 0)]
    high["belief_updates"] = 1
    high["belief_state"]["inventory"] = {"diamond": 0}
    high["run"].update(
        phase="scientific-terminal",
        terminal_reason="time_budget_exhausted",
        trace=[
            {
                "revision": 1,
                "observation_seq": 1,
                "belief": {},
                "primary_goal": {
                    "name": "obtain-diamond",
                    "status": "right-censored",
                },
                "stage": {},
                "intentions": [],
                "planning": None,
                "monitoring": {},
                "decision": {
                    "kind": "scientific-terminal",
                    "reason": "time_budget_exhausted",
                },
            }
        ],
    )
    environment = _environment_initial_state()
    environment.update(
        observation_requests=1,
        close_requests=1,
        inventory={"diamond": 0},
        treatment=deepcopy(high["run"]["treatment"]),
        closed=True,
    )
    return named, environment


def test_checker_accepts_clean_task_failure_without_behavioral_quotas() -> None:
    states, environment = valid_scientific_failure_fixture()
    assert _result_errors(states, environment, ["[0][INFO]::[x]::ok"]) == []
    json.dumps([states, environment])


def test_checker_rejects_inconsistent_diamond_evidence() -> None:
    states, environment = valid_scientific_failure_fixture()
    high = states[module_name(HLREASONER, 0)]
    high["belief_state"]["inventory"]["diamond"] = 1
    assert any(
        "diamond evidence disagrees" in error
        for error in _result_errors(
            states, environment, ["[0][INFO]::[x]::ok"]
        )
    )


@pytest.mark.parametrize(
    ("target", "value", "expected_error"),
    (
        ("belief_inventory", [], "HL belief final inventory is not a mapping"),
        (
            "environment_inventory",
            [],
            "environment final inventory is not a mapping",
        ),
        (
            "belief_diamond",
            "1",
            "HL belief final diamond count is not an integer",
        ),
        (
            "environment_diamond",
            True,
            "environment final diamond count is not an integer",
        ),
        (
            "belief_achievements",
            {},
            "HL belief final achievements are not a sequence or set of strings",
        ),
        (
            "environment_achievements",
            [1],
            "environment final achievements are not a sequence or set of strings",
        ),
    ),
)
def test_checker_reports_malformed_final_world_evidence(
    target: str,
    value: Any,
    expected_error: str,
) -> None:
    states, environment = valid_scientific_failure_fixture()
    belief_state = states[module_name(HLREASONER, 0)]["belief_state"]
    if target == "belief_inventory":
        belief_state["inventory"] = value
    elif target == "environment_inventory":
        environment["inventory"] = value
    elif target == "belief_diamond":
        belief_state["inventory"]["diamond"] = value
    elif target == "environment_diamond":
        environment["inventory"]["diamond"] = value
    elif target == "belief_achievements":
        belief_state["achievements"] = value
    else:
        environment["achievements"] = value

    assert expected_error in _result_errors(
        states, environment, ["[0][INFO]::[x]::ok"]
    )


def test_reporting_isolates_malformed_full_state_execution(tmp_path: Path) -> None:
    """A checker failure must not prevent a later execution being reported."""

    expected = []
    for run_id in (0, 1):
        states, environment = valid_scientific_failure_fixture()
        trace_row = states[module_name(HLREASONER, 0)]["run"]["trace"][0]
        trace_row["elapsed_seconds"] = 1.0
        trace_row["belief"] = {
            "achievements": [],
            "needs": {"health": 9, "food": 9, "drink": 9, "energy": 9},
            "known_terrain": 1,
            "reachable_cells": 1,
            "unknown_frontier": 0,
        }
        if run_id == 0:
            states[module_name(HLREASONER, 0)]["belief_state"]["inventory"] = []
        agent_out = tmp_path / f"exp_agent2_3_{run_id}" / "out"
        environment_out = tmp_path / f"exp_env2_3_{run_id}" / "out"
        agent_out.mkdir(parents=True)
        environment_out.mkdir(parents=True)
        for name, state in states.items():
            (agent_out / f"exp_agent2_3_{run_id}.{name}.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
        (environment_out / f"exp_env2_3_{run_id}.environment.json").write_text(
            json.dumps(environment), encoding="utf-8"
        )
        (tmp_path / f"exp_agent2_3_{run_id}.log").write_text(
            "[0][INFO]::[x]::ok\n", encoding="utf-8"
        )
        expected.append({
            "execution_id": f"run-{run_id}",
            "run_id": run_id,
            "factors": {},
        })

    report = process_execution_metrics(tmp_path, expected_executions=expected)

    malformed, valid = report["executions"]
    assert malformed["certificate_status"] == "failed"
    assert malformed["operational_reasons"] == ["operational_contract_failed"]
    assert valid["certificate_status"] == "passed"
    assert valid["operationally_valid"] is True
    assert report["execution_accounting"]["operationally_valid"] == 1
    assert report["analyses"]["execution_outcomes"]["summary"]["count"] == 1


def test_checker_rejects_agent_step_counter_mismatch() -> None:
    states, environment = valid_scientific_failure_fixture()
    states[module_name(HLREASONER, 0)]["run"]["agent_steps"] = 1

    assert any(
        "action chain is incomplete" in error
        for error in _result_errors(
            states, environment, ["[0][INFO]::[x]::ok"]
        )
    )
