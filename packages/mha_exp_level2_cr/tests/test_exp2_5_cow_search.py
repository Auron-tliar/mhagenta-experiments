"""Discovery-context contract, coordinate parity, persistence and mask regressions."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from exp2_5_harness import HarnessState
from mha_exp_common.names import LLREASONER
from mha_exp_level2_cr.exp2_5 import policy, runtime
from mha_exp_level2_cr.exp2_5.beliefs import initial_belief_state
from mha_exp_level2_cr.exp2_5.contracts import ActivityId, ActivitySpec, goal_to_dict, make_activity_goal
from mha_exp_level2_cr.exp2_5.cow_search import camera_cells, discovery_goal


@pytest.mark.parametrize("player", [(32, 32), (1, 1), (62, 62)])
def test_discovery_goal_is_invariant_to_runtime_origin(player):
    """The relative runtime map and absolute preparation map select the same goal."""
    known = camera_cells(player)
    safe = known - {(player[0] - 1, player[1])}
    goal = discovery_goal(player, known, safe, None, (2, 3, 4))
    shift = lambda cell: (cell[0] - 32, cell[1] - 32)
    shifted = discovery_goal(shift(player), map(shift, known), map(shift, safe), None,
                             (2, 3, 4), world_origin=(-32, -32))
    assert shifted == (shift(goal) if goal is not None else None)


def test_discovery_goal_retains_reachable_goal_but_respects_mask_and_blockage():
    """An old goal persists only while a masked route still exists."""
    player = (32, 32)
    known = camera_cells(player)
    safe = {player, (33, 32), (34, 32), (31, 32), (30, 32)}
    assert discovery_goal(player, known, safe, (34, 32), (2,)) == (34, 32)
    assert discovery_goal(player, known, safe, (34, 32), (1,)) == (31, 32)
    assert discovery_goal(player, known, {player}, (34, 32), (1, 2)) is None


def test_discovery_contract_is_explicit_and_specific_to_eat_cow(tmp_path):
    """Old EatCow remains loadable; only explicit new weights enable discovery input."""
    torch = pytest.importorskip("torch")
    weights = policy.build_q_network(torch).state_dict()
    for enabled in (False, True):
        path = tmp_path / f"cow-{enabled}.pt"
        data = policy.checkpoint_data(policy.PolicyId.EAT_COW, weights, discovery_context=enabled)
        torch.save(data, path)
        model, _ = policy.load_policy_checkpoint(torch, path, expected_policy_id=policy.PolicyId.EAT_COW)
        assert model.uses_discovery_context is enabled
        altered = deepcopy(data)
        altered["input"]["context_encoding"] = "unknown"
        with pytest.raises(ValueError):
            policy.validate_checkpoint(altered, policy.PolicyId.EAT_COW)
    with pytest.raises(ValueError):
        policy.checkpoint_data(policy.PolicyId.EAT_TARGET, weights, discovery_context=True)
    with pytest.raises(ValueError):
        policy.validate_checkpoint({"experimental_contract": "probe", "weights": data}, policy.PolicyId.EAT_COW)


def test_search_waypoint_cannot_enable_do_but_visible_cow_can():
    """A high DO logit cannot confuse a directly faced search goal with a cow."""
    torch = pytest.importorskip("torch")
    class RecordingModel(torch.nn.Module):
        uses_discovery_context = True

        def forward(self, images, contexts):
            self.context = contexts.cpu().numpy().copy()
            return torch.tensor([[0., 1., 2., 3., 4., 100.]])

    model = RecordingModel()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    args = (torch, model, frame, policy.PolicyId.EAT_COW, (0, 0))
    action, _ = policy.inference_evidence(*args, None, facing=(1, 0), search_goal_cell=(1, 0))
    assert action == 4
    np.testing.assert_allclose(model.context, [[1 / 63, 0]])
    action, _ = policy.inference_evidence(*args, (1, 0), facing=(1, 0), search_goal_cell=(4, 0))
    assert action == 5
    np.testing.assert_allclose(model.context, [[1 / 63, 0]])


def test_bundle_must_declare_discovery_context(tmp_path):
    """Updating a weight hash alone cannot silently change the runtime input contract."""
    torch = pytest.importorskip("torch")
    bundle = tmp_path / "current"
    bundle.mkdir()
    weights = policy.build_q_network(torch).state_dict()
    records = {}
    for identity, filename in policy.POLICY_FILENAMES.items():
        torch.save(policy.checkpoint_data(identity, weights,
                   discovery_context=identity is policy.PolicyId.EAT_COW), bundle / filename)
        records[identity.value] = {"filename": filename, "sha256": policy.file_sha256(bundle / filename)}
    manifest = {"format_version": policy.MANIFEST_FORMAT_VERSION, "artifact_id": "current",
                "environment": policy.ENVIRONMENT_CONTRACT, "input": policy.INPUT_CONTRACT, "policies": records}
    path = bundle / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest and checkpoint context differ"):
        policy.load_policy_bundle(torch, tmp_path)
    records["eat_cow"]["context_encoding"] = policy.EAT_COW_WAYPOINT_ENCODING
    path.write_text(json.dumps(manifest))
    _, _, models = policy.load_policy_bundle(torch, tmp_path)
    assert models[policy.PolicyId.EAT_COW].uses_discovery_context


def test_old_trainer_cannot_silently_discard_discovery_context(tmp_path):
    """A new-contract parent cannot start a long old-contract run by accident."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).parents[3] / "tools/exp2_5_cr_preparation"))
    from train import _warm_model

    path = tmp_path / "discovery.pt"
    torch.save(policy.checkpoint_data(policy.PolicyId.EAT_COW,
               policy.build_q_network(torch).state_dict(), discovery_context=True), path)
    with pytest.raises(ValueError, match="zero search context"):
        _warm_model(torch, policy.PolicyId.EAT_COW, "cpu", path)


def test_runtime_persists_resets_and_logs_the_actual_discovery_input(monkeypatch):
    """Exercise real dispatch and goal callbacks, including search-to-pursuit reset."""
    belief = initial_belief_state()
    belief.update(revision=1, facing="right", inventory=dict(health=9, food=5, drink=9, energy=9))
    cells = camera_cells((0, 0), (-32, -32))
    belief["terrain"] = {f"{x},{y}": "grass" for x, y in cells}
    for key in ("known_cells", "safe_cells", "visible_cells"):
        belief[key] = sorted(belief["terrain"])
    spec = ActivitySpec("search", ActivityId.EAT_COW, "recovery:food", "cow", None, 1, 0, 1)
    state = HarnessState(runtime.initial_states()[LLREASONER])
    state.update(belief_state=belief, current_goal=goal_to_dict(make_activity_goal(spec)),
                 cow_search_goal=[2, 0])
    state.outbox = SimpleNamespace(request_action=lambda *args, **kwargs: None)
    module = runtime.NeuralActivityLLReasoner(module_id=LLREASONER, initial_state=state)
    module._models = {policy.PolicyId.EAT_COW: SimpleNamespace(uses_discovery_context=True)}
    module._torch, module._latest_frame, module._latest_digest = None, None, "frame"
    module._actuator_id, module._goal_graph_id = "actuator", "goals"
    received = []

    def infer(*args, **kwargs):
        received.append(kwargs["search_goal_cell"])
        return 2, [0., 1., 9., 3., 4., 100.]

    monkeypatch.setattr(runtime, "inference_evidence", infer)
    module._dispatch(state)
    assert received[-1] == (2, 0) and state["cow_search_goal"] == [2, 0]
    assert 5 not in state["pending_action"]["legal_actions"]
    np.testing.assert_allclose(state["pending_action"]["context"], [2 / 63, 0])
    state["pending_action"] = None
    belief["occupants"]["2,0"] = {"kind": "cow", "last_seen_revision": 1}
    module._dispatch(state)
    assert state["cow_search_goal"] is None and received[-1] is None
    assert state["pending_action"]["target_cell"] == [2, 0]
    state.update(current_goal=None, pending_action=None, cow_search_goal=[3, 0])
    monkeypatch.setattr(module, "_dispatch", lambda state: None)
    module.on_goal_update(state, "goals", [make_activity_goal(spec)])
    assert state["failure"] is None and state["cow_search_goal"] is None
