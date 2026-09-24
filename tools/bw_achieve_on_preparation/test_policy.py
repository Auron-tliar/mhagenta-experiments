"""Checks for transferred behavior and the new direct-action goal contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import GoalSpec
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_5.policy import artifact_paths, greedy_inference, load_policy_checkpoint
from tools.bw_achieve_on_preparation.policy import condition_goal, goal_succeeded, infer, warm_start
from mha_exp_level2_bw.achieve_on.policy import (
    CONV3D_ARCHITECTURE, build_network, checkpoint_payload, conv3d_volume,
)


@pytest.fixture
def environment():
    """Provide an isolated numeric environment without recording or CUDA."""
    env = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    yield env
    env.close()


def test_warm_start_preserves_complete_transfer_rollout(environment) -> None:
    """New bottom identity must not perturb copied behavior before training."""
    torch.set_num_threads(1)
    observation, _ = environment.reset(seed=1000)
    spec = next(spec for spec in enumerate_transfer_targets(ground_observation(observation).facts)
                if spec.destination_support.startswith("b"))
    goal = GoalSpec(spec.block, spec.destination_support)
    teacher, _ = load_policy_checkpoint(torch, artifact_paths()[1])
    model, lineage = warm_start(torch)
    assert lineage["copied_parameters"] == 24836
    assert sum(parameter.numel() for parameter in model.parameters()) == 24900
    assert all(parameter.requires_grad for parameter in model.parameters())
    for _ in range(32):
        expected, expected_values, _ = greedy_inference(torch, teacher, observation, spec)
        action, values = infer(torch, model, observation, goal)
        assert action == expected
        np.testing.assert_allclose(values, expected_values, atol=1e-5, rtol=1e-5)
        observation, *_ = environment.step(action)
        if goal_succeeded(observation, goal):
            break
    assert goal_succeeded(observation, goal)


def test_bottom_identity_distinguishes_same_column_targets(environment) -> None:
    """Final support identity must survive even when two targets share a stack."""
    observation, _ = environment.reset(seed=1000)
    locations = {}
    for block in range(8):
        column = int(np.flatnonzero(np.any(observation[1:, :, block], axis=0))[0])
        locations.setdefault(column, []).append(f"b{block}")
    bottom_a, bottom_b = next(blocks[:2] for blocks in locations.values() if len(blocks) >= 2)
    top = next(f"b{index}" for index in range(8) if f"b{index}" not in (bottom_a, bottom_b))
    first = condition_goal(observation, GoalSpec(top, bottom_a))
    second = condition_goal(observation, GoalSpec(top, bottom_b))
    np.testing.assert_array_equal(first[:-1], second[:-1])
    assert not np.array_equal(first[-1], second[-1])


def test_bottom_encoding_follows_held_block_and_new_weights_can_learn(environment) -> None:
    """Candidate-induced held-bottom states stay representable and trainable."""
    observation, _ = environment.reset(seed=1000)
    spec = enumerate_transfer_targets(ground_observation(observation).facts)[0]
    teacher, _ = load_policy_checkpoint(torch, artifact_paths()[1])
    while ground_observation(observation).held_block is None:
        action, _, _ = greedy_inference(torch, teacher, observation, spec)
        observation, *_ = environment.step(action)
    top = next(f"b{index}" for index in range(8) if f"b{index}" != spec.block)
    goal = GoalSpec(top, spec.block)
    encoded = condition_goal(observation, goal)
    arm = int(ground_observation(observation).arm_location[1:])
    assert np.all(encoded[11, arm] == 1)
    assert not goal_succeeded(observation, goal)
    model, _ = warm_start(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    prediction = model(torch.as_tensor(encoded, dtype=torch.float32).unsqueeze(0))
    loss = prediction.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.entity_encoder[0].weight.grad[:, -1].abs().sum() > 0
    optimizer.step()
    assert model.entity_encoder[0].weight[:, -1].abs().sum() > 0


@pytest.mark.parametrize("goal", [GoalSpec("b0", "b0"), GoalSpec("b8", "b1"), GoalSpec("b0", "t1")])
def test_invalid_goals_are_rejected(environment, goal) -> None:
    observation, _ = environment.reset(seed=1000)
    with pytest.raises(ValueError, match="distinct canonical"):
        condition_goal(observation, goal)


def test_conv3d_volume_separates_occupancy_and_context(environment) -> None:
    """Only stack planes are height; holding and goal roles stay distinct channels."""
    observation, _ = environment.reset(seed=1000)
    encoded = torch.tensor(condition_goal(observation, GoalSpec("b0", "b1")), dtype=torch.float32)[None]
    volume = conv3d_volume(torch, encoded)
    assert volume.shape == (1, 6, 8, 5, 8)
    assert torch.equal(volume[:, 0], encoded[:, 2:10])
    for channel, plane in enumerate((0, 1, 10, 12, 11), start=1):
        for height in range(8):
            assert torch.equal(volume[:, channel, height], encoded[:, plane])
    # A consistent relabeling changes the categorical axis, never height/column.
    permutation = torch.tensor([7, 1, 4, 0, 3, 6, 2, 5])
    assert torch.equal(conv3d_volume(torch, encoded[..., permutation]), volume[..., permutation])
    encoded[:, 1, 2, 1] = 1
    assert torch.all(conv3d_volume(torch, encoded)[:, 2, :, 2, 1] == 1)


def test_conv3d_preserves_warm_start_then_learns_and_round_trips(environment, tmp_path) -> None:
    """Zero residual preserves Transfer values, then gradients reach both Conv3d layers."""
    torch.set_num_threads(1)
    torch.manual_seed(2605)
    model, provenance = warm_start(torch, CONV3D_ARCHITECTURE)
    base, _ = warm_start(torch)
    observations = [environment.reset(seed=1000 + index)[0] for index in range(4)]
    inputs = torch.tensor(np.stack([condition_goal(obs, GoalSpec("b0", "b1")) for obs in observations]), dtype=torch.float32)
    assert torch.equal(model(inputs), base(inputs))
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in model.base.state_dict().items())
    assert provenance["copied_parameters"] == 24836
    assert sum(p.numel() for p in model.parameters()) == provenance["copied_parameters"] + provenance["added_parameters"]
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (model(inputs) - torch.tensor([1., -1., .5, -.5])).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    for layer in (model.convolutions[0], model.convolutions[2]):
        assert layer.weight.grad.abs().sum() > 0
    payload = checkpoint_payload(model, provenance)
    path = tmp_path / "conv3d.pt"
    torch.save(payload, path)
    saved = torch.load(path, weights_only=True)
    restored = build_network(torch, saved["architecture"])
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    assert torch.equal(restored(inputs), model(inputs))
    # Full kernels use identity ordering: expose, rather than hide, sensitivity.
    reordered = inputs[..., torch.arange(7, -1, -1)]
    assert not torch.equal(model(inputs), model(reordered))
    assert torch.isfinite(model(reordered)).all()
