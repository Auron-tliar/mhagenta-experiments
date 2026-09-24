"""Geometry, identity symmetry, training and serialization of BW alternatives."""

import pickle

import numpy as np
import pytest

from mha_exp_level2_bw.exp2_2.policy import (
    build_q_network, goal_conditioned_observation, load_policy_checkpoint,
    save_policy_checkpoint,
)
from mha_exp_level2_bw.exp2_2.protocol import DQNProtocol


def example():
    """Place a held source above a destination with a separate distractor."""
    obs = np.zeros((32, 10, 30), dtype=np.uint8)
    obs[0, 3, :] = 1
    obs[1, 3, 7] = 1
    obs[31, 3, 9] = 1
    obs[30, 3, 4] = 1
    return obs, (7, 9)


def test_spatial_geometry_and_identity_symmetry():
    torch = pytest.importorskip('torch')
    from mha_exp_level2_bw.exp2_2.networks import SpatialObservation
    obs, goal = example()
    values = torch.tensor(goal_conditioned_observation(obs, goal)).float()[None]
    planes = SpatialObservation()(values)
    assert planes.shape == (1, 7, 30, 10)
    assert planes[0, 0].sum() == 2
    assert planes[0, 2, 29, 3] == 1
    assert planes[0, 1].sum() == 0  # Source is held, not in a stack.
    assert planes[0, 3, :, 3].sum() == 30
    assert planes[0, 5, :, 3].sum() == 30
    assert planes[0, 4].sum() == planes[0, 6].sum() == 0
    permutation = np.random.default_rng(42).permutation(30)
    renamed = np.zeros_like(obs)
    renamed[:, :, permutation] = obs
    renamed_goal = tuple(int(permutation[b]) for b in goal)
    other = torch.tensor(goal_conditioned_observation(renamed, renamed_goal)).float()[None]
    torch.testing.assert_close(SpatialObservation()(other), planes, rtol=0, atol=0)


def test_volume_retains_original_axes_and_separate_goal_channels():
    torch = pytest.importorskip('torch')
    from mha_exp_level2_bw.exp2_2.networks import VolumeObservation
    obs, goal = example()
    values = torch.tensor(goal_conditioned_observation(obs, goal)).float()[None]
    volume = VolumeObservation()(values)
    assert volume.shape == (1, 3, 32, 10, 30)
    torch.testing.assert_close(volume[0, 0], torch.tensor(obs).float())
    assert volume[0, 1, :, :, 7].sum() == 320
    assert volume[0, 2, :, :, 9].sum() == 320


@pytest.mark.parametrize('architecture', ['goal_spatial_cnn_v1', 'goal_volume_cnn3d_v1'])
def test_alternative_trains_serializes_and_evaluates(tmp_path, architecture):
    torch = pytest.importorskip('torch')
    from mha_exp_level2_bw.exp2_2.evaluate_snapshot import evaluate
    torch.set_num_threads(1)
    obs, goal = example()
    values = torch.tensor(goal_conditioned_observation(obs, goal)).float()[None]
    model = build_q_network(torch, architecture)
    before = model(values).detach().clone()
    assert before.shape == (1, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = torch.nn.functional.smooth_l1_loss(model(values)[:, 1], torch.ones(1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    after = model(values).detach()
    assert not torch.equal(before, after)
    # Published models cross a process boundary, independently of checkpoints.
    torch.testing.assert_close(pickle.loads(pickle.dumps(model))(values), after)
    protocol = DQNProtocol(network_architecture=architecture, evaluation_seeds=(2200,),
                           max_episode_length=5, action_masking=True)
    assert DQNProtocol.from_record(protocol.record()) == protocol
    path = tmp_path/'policy.pt'
    save_policy_checkpoint(torch, model, path, 1, training_protocol=protocol.record(), frozen=True)
    loaded, metadata = load_policy_checkpoint(torch, path)
    assert metadata['architecture'] == architecture
    torch.testing.assert_close(loaded(values), after)
    result = evaluate(path, tmp_path/'evaluation.json')
    assert result['episodes'][0]['illegal_actions'] == 0


def test_legacy_protocol_schema_and_unknown_architecture():
    protocol = DQNProtocol()
    assert 'network_architecture' not in protocol.record()['config']
    assert DQNProtocol.from_record(protocol.record()) == protocol
    with pytest.raises(ValueError, match='architecture'):
        DQNProtocol(network_architecture='unrecognized')
