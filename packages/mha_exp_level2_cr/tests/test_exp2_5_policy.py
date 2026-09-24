"""Current five-policy weights, contexts, and execution masks."""

import numpy as np
import pytest
from mha_exp_level2_cr.exp2_5.policy import (
    PolicyId,
    build_q_network,
    checkpoint_data,
    encode_context,
    legal_actions,
    load_policy_checkpoint,
    select_action,
    selection_actions,
    validate_checkpoint,
)


@pytest.mark.parametrize("policy", list(PolicyId))
def test_current_checkpoint_loads_frozen_models(policy, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    torch.save(checkpoint_data(policy, build_q_network(torch).state_dict()), path)
    model, _ = load_policy_checkpoint(torch, path, expected_policy_id=policy)
    assert all(not value.requires_grad for value in model.parameters())
    validate_checkpoint(checkpoint_data(policy, model.state_dict()), policy)


def test_navigation_and_resource_checkpoints_declare_hud_mask():
    torch = pytest.importorskip("torch")
    for policy in (PolicyId.NAVIGATE_TO, PolicyId.GET_RESOURCE):
        checkpoint = checkpoint_data(policy, build_q_network(torch).state_dict())
        assert checkpoint["input"]["image_preprocessing"].endswith("hud49_zero_v1")


def test_hud_mask_is_invariant_and_does_not_mutate_input():
    torch = pytest.importorskip("torch")
    model = build_q_network(torch, mask_hud=True)
    first = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    other = first.clone()
    other[:, :, 49:] = 255 - other[:, :, 49:]
    context = torch.zeros((1, 2))
    torch.testing.assert_close(model(first, context), model(other, context), rtol=0, atol=0)
    assert not torch.equal(first, other)


def test_persistent_context_and_zero_context_skills():
    assert np.array_equal(encode_context(PolicyId.EXPLORE, (0, 0), None), [0, 0])
    assert np.array_equal(encode_context(PolicyId.EAT_COW, (0, 0), None), [0, 0])
    np.testing.assert_allclose(encode_context(PolicyId.NAVIGATE_TO, (1, 0), (4, 2)), [3 / 63, 2 / 63])
    np.testing.assert_allclose(encode_context(PolicyId.NAVIGATE_TO, (2, 0), (4, 2)), [2 / 63, 2 / 63])


def test_static_and_observable_eat_cow_masks():
    assert legal_actions(PolicyId.EXPLORE) == (1, 2, 3, 4)
    assert selection_actions(PolicyId.EAT_COW, [0, 0], (1, 0), [1, 3, 4]) == (1, 3, 4)
    assert 5 in selection_actions(PolicyId.EAT_COW, [1 / 63, 0], (1, 0))
    assert 5 not in selection_actions(PolicyId.EAT_COW, [1 / 63, 0], (0, 1))
    values = np.array([100, 1, 2, 3, 4, 99], dtype=float)
    assert select_action(values, legal_actions(PolicyId.EXPLORE)) == 4
    assert values.tolist() == [100, 1, 2, 3, 4, 99]
    with pytest.raises(ValueError):
        select_action(np.full(6, np.nan), (1, 2))
    with pytest.raises(ValueError):
        select_action(values, ())


def test_navigation_and_resource_masks_exclude_observable_illegal_actions():
    assert selection_actions(PolicyId.NAVIGATE_TO, [2 / 63, 0], movement_actions=[1, 3]) == (1, 3)
    assert selection_actions(PolicyId.GET_RESOURCE, [2 / 63, 0], (1, 0), [2, 4]) == (2, 4)
    assert selection_actions(PolicyId.GET_RESOURCE, [1 / 63, 0], (0, 1), [2, 4]) == (2,)
    assert selection_actions(PolicyId.GET_RESOURCE, [1 / 63, 0], (1, 0), [2, 4]) == (5,)


def test_explore_cannot_choose_an_excluded_move_even_with_highest_q_value():
    """A blocked high-value action must not escape the shared runtime mask."""
    values = np.array([0, 100, 2, 3, 4, 0], dtype=float)
    allowed = selection_actions(PolicyId.EXPLORE, [0, 0], movement_actions=[2, 4])
    assert allowed == (2, 4)
    assert select_action(values, allowed) == 4
    assert selection_actions(PolicyId.EXPLORE, [0, 0], movement_actions=[]) == ()


def test_eat_target_masks_blocked_moves_and_unavailable_interactions():
    """Cow pursuit must respect movement exclusions and grounded interaction."""
    assert selection_actions(PolicyId.EAT_TARGET, [2 / 63, 0], (1, 0), [1, 3]) == (1, 3)
    assert selection_actions(PolicyId.EAT_TARGET, [1 / 63, 0], (0, 1), [2, 4]) == (2, 4)
    assert selection_actions(PolicyId.EAT_TARGET, [1 / 63, 0], (1, 0), []) == (5,)
    assert selection_actions(
        PolicyId.EAT_TARGET, [2 / 63, 0], (1, 0), [], interaction_available=True,
    ) == (5,)
    assert selection_actions(
        PolicyId.EAT_TARGET, [1 / 63, 0], (1, 0), [3], interaction_available=False,
    ) == (3,)


def test_old_checkpoint_contract_is_not_supported():
    with pytest.raises(ValueError):
        validate_checkpoint({"format_version": 3}, PolicyId.EXPLORE)
