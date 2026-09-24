from types import SimpleNamespace

import numpy as np
import pytest

from mha_env_crafter.crafter import constants, objects
from mha_env_crafter.crafter.env import Env


@pytest.fixture
def env() -> Env:
    environment = Env(seed=7, no_mobs=True)
    environment.reset()
    return environment


def _action(name: str) -> int:
    return constants.actions.index(name)


def _step(env: Env, action: str) -> dict:
    _, _, _, info = env.step(_action(action))
    assert type(info["illegal_action"]) is bool
    return info


def _clear_target(
    env: Env,
    direction: tuple[int, int] | None = None,
    material: str = "grass",
) -> np.ndarray:
    player = env._player
    direction = tuple(player.facing) if direction is None else direction
    target = player.pos + direction
    _, obj = env._world[target]
    if obj is not None:
        env._world.remove(obj)
    env._world[target] = material
    return target


def _clear_nearby_materials(env: Env) -> None:
    player = env._player
    for x in range(-1, 2):
        for y in range(-1, 2):
            env._world[player.pos + (x, y)] = "grass"


def test_noop_is_legal_and_info_value_is_a_strict_bool(env: Env) -> None:
    info = _step(env, "noop")

    assert info["illegal_action"] is False


@pytest.mark.parametrize(
    ("material", "illegal"),
    [
        ("grass", False),
        ("tree", True),
    ],
)
def test_movement_legality_depends_on_destination(
    env: Env,
    material: str,
    illegal: bool,
) -> None:
    env._player.facing = (1, 0)
    start = env._player.pos.copy()
    _clear_target(env, direction=(1, 0), material=material)

    info = _step(env, "move_right")

    assert info["illegal_action"] is illegal
    assert tuple(env._player.facing) == (1, 0)
    if illegal:
        assert np.array_equal(env._player.pos, start)
    else:
        assert np.array_equal(env._player.pos, start + (1, 0))


def test_do_is_legal_for_collectable_material_with_satisfied_requirements(
    env: Env,
) -> None:
    _clear_target(env, material="tree")

    info = _step(env, "do")

    assert info["illegal_action"] is False
    assert env._player.inventory["wood"] == 1


def test_do_is_illegal_when_material_requirements_are_not_satisfied(
    env: Env,
) -> None:
    _clear_target(env, material="stone")
    env._player.inventory["wood_pickaxe"] = 0

    info = _step(env, "do")

    assert info["illegal_action"] is True
    assert env._player.inventory["stone"] == 0


def test_probabilistic_collection_is_legal_when_it_yields_nothing(
    env: Env,
) -> None:
    _clear_target(env, material="grass")
    env._player.random = SimpleNamespace(uniform=lambda: 1.0)

    info = _step(env, "do")

    assert info["illegal_action"] is False
    assert env._player.inventory["sapling"] == 0


@pytest.mark.parametrize(
    ("ripe", "illegal"),
    [
        (True, False),
        (False, True),
    ],
)
def test_do_legality_for_plants(env: Env, ripe: bool, illegal: bool) -> None:
    target = _clear_target(env)
    plant = objects.Plant(env._world, target)
    plant.grown = 301 if ripe else 0
    env._world.add(plant)

    info = _step(env, "do")

    assert info["illegal_action"] is illegal


@pytest.mark.parametrize(
    ("energy", "illegal", "sleeping"),
    [
        (8, False, True),
        (9, True, False),
    ],
)
def test_sleep_requires_missing_energy(
    env: Env,
    energy: int,
    illegal: bool,
    sleeping: bool,
) -> None:
    env._player.inventory["energy"] = energy

    info = _step(env, "sleep")

    assert info["illegal_action"] is illegal
    assert env._player.sleeping is sleeping


@pytest.mark.parametrize(
    ("material", "wood", "occupied", "illegal"),
    [
        ("grass", 2, False, False),
        ("water", 2, False, True),
        ("grass", 1, False, True),
        ("grass", 2, True, True),
    ],
)
def test_place_requires_valid_target_and_inventory(
    env: Env,
    material: str,
    wood: int,
    occupied: bool,
    illegal: bool,
) -> None:
    target = _clear_target(env, material=material)
    env._player.inventory["wood"] = wood
    if occupied:
        env._world.add(objects.Plant(env._world, target))

    info = _step(env, "place_table")

    assert info["illegal_action"] is illegal


@pytest.mark.parametrize(
    ("near_table", "wood", "illegal"),
    [
        (True, 1, False),
        (False, 1, True),
        (True, 0, True),
    ],
)
def test_make_requires_nearby_utility_and_inventory(
    env: Env,
    near_table: bool,
    wood: int,
    illegal: bool,
) -> None:
    _clear_nearby_materials(env)
    if near_table:
        env._world[env._player.pos + (1, 0)] = "table"
    env._player.inventory["wood"] = wood

    info = _step(env, "make_wood_pickaxe")

    assert info["illegal_action"] is illegal


@pytest.mark.parametrize(
    ("action", "illegal"),
    [
        ("noop", False),
        ("sleep", False),
        ("move_right", True),
    ],
)
def test_sleeping_player_reports_ignored_actions(
    env: Env,
    action: str,
    illegal: bool,
) -> None:
    start = env._player.pos.copy()
    env._player.sleeping = True
    env._player.inventory["energy"] = 8
    _clear_target(env, direction=(1, 0))

    info = _step(env, action)

    assert info["illegal_action"] is illegal
    assert np.array_equal(env._player.pos, start)


def test_selected_action_is_evaluated_after_automatic_wake_up(env: Env) -> None:
    start = env._player.pos.copy()
    env._player.sleeping = True
    env._player.inventory["energy"] = 9
    _clear_target(env, direction=(1, 0))

    info = _step(env, "move_right")

    assert info["illegal_action"] is False
    assert np.array_equal(env._player.pos, start + (1, 0))


def test_out_of_range_action_behavior_is_unchanged(env: Env) -> None:
    with pytest.raises(IndexError):
        env.step(len(constants.actions))
