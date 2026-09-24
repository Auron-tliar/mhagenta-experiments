"""Read-only action applicability for the explicitly state-assisted CR treatment."""

from typing import Any

import numpy as np


def validate_mask(value: Any) -> np.ndarray:
    """Require seventeen Boolean entries with at least one applicable action."""
    mask = np.asarray(value)
    if mask.shape != (17,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Action mask must contain 17 Booleans and a valid action.')
    return mask.copy()


def action_mask(environment: Any) -> list[bool]:
    """Mirror Player.update applicability without changing world state or RNG.

    This exposes environment-side knowledge beyond RGB observations. Applicable
    actions can still be unproductive or dangerous (including walking into lava).
    """
    from mha_env_crafter.crafter import constants, objects

    player = environment._player
    inventory = player.inventory
    world = player.world
    target = player.pos + player.facing
    material, obj = world[target]
    tired = inventory['energy'] < constants.items['energy']['max']
    directions = dict(left=(-1, 0), right=(1, 0), up=(0, -1), down=(0, 1))
    result = []
    for action in constants.actions:
        if player.sleeping and tired:
            valid = action in ('noop', 'sleep')
        elif action == 'noop':
            valid = True
        elif action.startswith('move_'):
            direction = directions[action[5:]]
            valid = tuple(player.facing) != direction or player.is_free(player.pos + direction)
        elif action == 'sleep':
            valid = tired
        elif action == 'do':
            if obj is not None:
                valid = (isinstance(obj, (objects.Fence, objects.Zombie, objects.Skeleton, objects.Cow))
                         or isinstance(obj, objects.Plant) and obj.ripe)
            else:
                rule = constants.collect.get(material)
                valid = bool(rule) and all(inventory[k] >= v for k, v in rule['require'].items())
        elif action.startswith('place_'):
            rule = constants.place[action[6:]]
            valid = (obj is None and material in rule['where']
                     and all(inventory[k] >= v for k, v in rule['uses'].items()))
        elif action.startswith('make_'):
            rule = constants.make[action[5:]]
            nearby, _ = world.nearby(player.pos, 1)
            valid = (all(name in nearby for name in rule['nearby'])
                     and all(inventory[k] >= v for k, v in rule['uses'].items()))
        else:
            raise ValueError(f'Unrecognized Crafter action: {action}')
        result.append(bool(valid))
    return validate_mask(result).tolist()
