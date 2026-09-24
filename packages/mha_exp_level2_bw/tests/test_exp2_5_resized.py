"""Semantic warm-start and native-world checks for the two resized Transfer models."""

from pathlib import Path
import sys

import numpy as np
import pytest

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import RUNTIME_TO_CANONICAL, format_fact
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_5.policy import (
    artifact_paths, condition_observation, legal_action_indices, load_policy_checkpoint,
)
from mha_exp_level2_bw.exp2_5.resized import feature_keys, warm_start

PREPARATION = Path(__file__).resolve().parents[3] / "tools/exp2_5_policy_preparation"
sys.path.insert(0, str(PREPARATION))
from selected_dqfd import SelectedReplayBuffer, legal_action_mask, _optimize, CONFIGS, VARIANT_DQFD_LITE
from train_resized import collect_demonstrations


@pytest.mark.parametrize("columns,blocks,count", [(4, 6, 23620), (5, 8, 24836), (7, 12, 28036)])
def test_warm_start_preserves_semantically_matching_weights(columns, blocks, count):
    """Check height alignment and each feature group independently of model shapes."""
    torch = pytest.importorskip("torch")
    source, _ = load_policy_checkpoint(torch, artifact_paths()[1])
    model, provenance = warm_start(torch, artifact_paths()[1], table_len=columns, num_blocks=blocks, device="cpu")
    old, new = source.state_dict(), model.state_dict()
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert provenance["trainable_parameters"] == count
    for name in old:
        if name != "entity_encoder.0.weight":
            assert torch.equal(old[name], new[name])
    weights, original = new["entity_encoder.0.weight"], old["entity_encoder.0.weight"]
    # Bottom-most stack cells must stay bottom-most, although their flat indices change.
    for height in range(min(blocks, 8)):
        for column in range(min(columns, 5)):
            assert torch.equal(weights[:, (blocks - 1 - height) * columns + column],
                               original[:, (7 - height) * 5 + column])
    old_keys = {key: index for index, key in enumerate(feature_keys(5, 8))}
    for index, key in enumerate(feature_keys(columns, blocks)):
        if key in old_keys:
            assert torch.equal(weights[:, index], original[:, old_keys[key]])
        else:
            assert not bool(weights[:, index].any())
    if (columns, blocks) == (5, 8):
        inputs = torch.randn(3, 12, 5, 8)
        assert torch.equal(source(inputs), model(inputs))


@pytest.mark.parametrize("columns,blocks", [(4, 6), (7, 12)])
def test_sized_grounding_and_masks_match_native_environment(columns, blocks):
    """Cross-check numeric grounding against the environment's independent symbolic state."""
    torch = pytest.importorskip("torch")
    dimensions = dict(table_len=columns, num_blocks=blocks)
    numeric = BlocksWorldEnv(**dimensions, symbolic=False)
    symbolic = BlocksWorldEnv(**dimensions, symbolic=True)
    numeric.expose_snapshot = True
    rng = np.random.default_rng(92)
    try:
        observation, _ = numeric.reset(seed=83)
        symbols, _ = symbolic.reset(seed=83)
        for _ in range(100):
            expected = set()
            for text in symbols:
                name, _, raw = text.partition("(")
                arguments = raw.rstrip(")").split(",") if raw else []
                expected.add(format_fact(RUNTIME_TO_CANONICAL[name], [value.strip() for value in arguments]))
            assert ground_observation(observation, **dimensions).facts == expected
            # Conditioning only needs a valid target index for the mask check.
            from mha_exp_level2_bw.exp2_5.contracts import TransferSpec
            spec = TransferSpec("b00" if blocks == 12 else "b0", "t0", "t1", "t0", "t1")
            conditioned = condition_observation(observation, spec, **dimensions)
            mask = legal_action_mask(torch, torch.tensor(conditioned)[None])[0]
            legal = legal_action_indices(observation, **dimensions)
            assert torch.nonzero(mask).flatten().tolist() == legal
            action = int(rng.choice(legal))
            observation, _, _, _, info = numeric.step(action)
            symbols, *_ = symbolic.step(action)
            assert info["snapshot"].legal
    finally:
        numeric.close()
        symbolic.close()


@pytest.mark.parametrize("columns,blocks", [(4, 6), (7, 12)])
def test_demonstrations_train_resized_network_and_preserve_protected_replay(columns, blocks):
    """Exercise actual expert episodes, replay sampling and finite optimizer updates."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    dimensions = dict(table_len=columns, num_blocks=blocks)
    rng = np.random.default_rng(2505)
    replay = SelectedReplayBuffer(1024, prioritized=True, protected_capacity=288,
                                  input_shape=(blocks + 4, columns, blocks))
    result = collect_demonstrations(replay, dimensions, rng, 256)
    assert 256 <= result["steps"] < 288
    assert result["episodes"] > 0
    assert replay.demonstrations[:len(replay)].all()
    model, _ = warm_start(torch, artifact_paths()[1], **dimensions, device="cpu")
    from copy import deepcopy
    target = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    before = model.entity_encoder[0].weight.detach().clone()
    for _ in range(3):
        loss, margin = _optimize(torch, model, target, optimizer, replay, rng, "cpu",
                                 CONFIGS[VARIANT_DQFD_LITE], 0.4, mask_legal_actions=True)
        assert np.isfinite(loss) and np.isfinite(margin)
    assert not torch.equal(before, model.entity_encoder[0].weight)
