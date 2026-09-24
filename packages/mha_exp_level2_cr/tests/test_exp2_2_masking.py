"""Applicability, state isolation, n-step mask alignment and mob-free regressions."""

from copy import deepcopy
import pickle

import numpy as np
import pytest

from mha_env_crafter import CrafterEnv
from mha_env_crafter.crafter import constants, objects
from mha_exp_level2_cr.exp2_2.masking import action_mask, validate_mask
from mha_exp_level2_cr.exp2_2.modules import TestLLReasoner as Reasoner, double_dqn_targets
from mha_exp_level2_cr.exp2_2.policy import initial_frame_stack
from mha_exp_level2_cr.exp2_2.replay import PrioritizedReplay
from mha_exp_level2_cr.exp2_2.treatment import DQNWorkload


def test_mask_matches_player_and_does_not_mutate_state_or_randomness():
    """Compare all actions against real Player.update over diverse states."""
    env = CrafterEnv(seed=23, no_mobs=True)
    env.reset()
    rng = np.random.default_rng(8)
    for iteration in range(24):
        player = env._player
        if iteration % 4 == 0:
            for name in player.inventory:
                player.inventory[name] = constants.items[name]['max']
            target = player.pos + player.facing
            obj = env._world[target][1]
            if obj is not None:
                env._world.remove(obj)
            env._world[target] = ('tree', 'stone', 'water', 'grass', 'coal', 'iron')[iteration // 4]
        player.sleeping = iteration % 6 == 0
        player.inventory['energy'] = 3 if iteration % 2 else 9
        before = pickle.dumps(env)
        mask = action_mask(env)
        assert pickle.dumps(env) == before
        assert mask[0]
        for index, expected in enumerate(mask):
            clone = deepcopy(env)
            clone._player.action = constants.actions[index]
            clone._player.update()
            assert expected == (not clone._player.illegal_action), (iteration, constants.actions[index])
        env.step(int(rng.choice(np.flatnonzero(mask))))


def test_blocked_move_that_turns_remains_available():
    env = CrafterEnv(seed=11, no_mobs=True)
    env.reset()
    player = env._player
    player.facing = (0, 1)
    env._world[player.pos + (-1, 0)] = 'stone'
    left = constants.actions.index('move_left')
    assert action_mask(env)[left]
    player.facing = (-1, 0)
    assert not action_mask(env)[left]


def test_no_hostile_mobs_generate_or_respawn_across_resets_and_time():
    env = CrafterEnv(seed=0, no_mobs=True)
    for _ in range(3):
        env.reset()
        for step in range(650):
            assert not any(isinstance(obj, (objects.Zombie, objects.Skeleton)) for obj in env._world.objects)
            env._player.inventory.update(health=9, food=9, drink=9, energy=9)
            env.step(0)


def test_n_step_masks_use_endpoint_including_truncated_tail_and_wraparound():
    replay = PrioritizedReplay(np.random.default_rng(1), capacity=4)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    masks = []
    for ordinal in range(1, 8):
        mask = [True] + [index == ordinal for index in range(1, 17)]
        masks.append(mask)
        replay.append(dict(replay_id=ordinal, state_stack=initial_frame_stack(frame),
                           action=0, reward=0., next_frame=frame, terminal=False,
                           next_action_mask=mask), boundary=False)
    replay.flush()
    assert sorted(item['replay_id'] for item in replay.items) == [4, 5, 6, 7]
    for item in replay.items:
        endpoint = item['replay_id'] + item['horizon'] - 1
        assert item['next_action_mask'] == masks[endpoint - 1]
        assert item['bootstrap_discount'] > 0


def test_exploration_greedy_playback_and_double_dqn_exclude_invalid_maximum():
    torch = pytest.importorskip('torch')
    from mha_exp_level2_cr.exp2_2.play_policy import greedy_action
    reasoner = Reasoner('reasoner', {})
    reasoner.on_init(seed=2, workload=DQNWorkload(action_masking=True).dump())
    mask = [False] * 17
    mask[3] = True
    reasoner._action_mask = validate_mask(mask)
    stack = initial_frame_stack(np.zeros((64, 64, 3), dtype=np.uint8))
    assert {reasoner._select_action(stack, 0) for _ in range(30)} == {3}

    class Fixed:
        def __call__(self, images):
            return torch.arange(17, dtype=torch.float32).repeat(len(images), 1)

    reasoner._model = Fixed()
    assert reasoner._select_action(stack, 0, greedy=True) == 3
    assert greedy_action(torch, Fixed(), stack, mask=mask) == 3
    result = double_dqn_targets(torch, Fixed(), Fixed(), torch.zeros((2, 1)),
                               torch.tensor([1., 2.]), torch.tensor([.99, 0.]),
                               torch.tensor([mask, mask]))
    assert result.tolist() == pytest.approx([3.97, 2.])
    with pytest.raises(ValueError):
        validate_mask([False] * 17)


def test_checkpoint_rejects_legacy_or_mobs_enabled_provenance():
    torch = pytest.importorskip('torch')
    from mha_exp_level2_cr.exp2_2.policy import build_q_network, policy_checkpoint, _validate_checkpoint
    payload = policy_checkpoint(build_q_network(torch), 0, DQNWorkload(action_masking=True))
    assert _validate_checkpoint(payload)['environment'] == {'no_mobs': True}
    payload['environment']['no_mobs'] = False
    with pytest.raises(ValueError, match='environment'):
        _validate_checkpoint(payload)
    payload['format_version'] = 3
    with pytest.raises(ValueError, match='legacy'):
        _validate_checkpoint(payload)


def test_adapter_enforces_masks_and_mob_free_seeded_resets():
    from mha_exp_level2_cr.exp2_2.modules import TestEnvironment as Adapter
    from mha_exp_level2_cr.exp2_2.runner import environment_initial_state
    state = environment_initial_state(7, action_masking=True)
    adapter = Adapter({'seed': 7, **state})
    assert adapter._env._no_mobs is True
    mask = action_mask(adapter._env)
    forbidden = mask.index(False)
    with pytest.raises(ValueError, match='inapplicable'):
        adapter.on_action(state, 'actuator', action=forbidden)
    adapter.on_action(state, 'actuator', action=0)
    adapter.on_action(state, 'actuator', action='reset', requested_seed=920000,
                      phase='frozen_evaluation')
    assert adapter._env._no_mobs is True
    assert not any(isinstance(obj, (objects.Zombie, objects.Skeleton)) for obj in adapter._env._world.objects)
