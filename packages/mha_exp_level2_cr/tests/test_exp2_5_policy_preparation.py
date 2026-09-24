"""Host-only tests for five-policy v3 preparation."""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from mha_exp_level2_cr.exp2_5.policy import (
    checkpoint_data,
    legal_actions,
)

TOOL_ROOT = Path(__file__).parents[3] / "tools" / "exp2_5_cr_preparation"
sys.path.insert(0, str(TOOL_ROOT))
cases = importlib.import_module("cases")
cli = importlib.import_module("cli")
protocol = importlib.import_module("protocol")
replay_module = importlib.import_module("replay")
train = importlib.import_module("train")


def _transition(index: int, policy, *, expert: bool):
    action = legal_actions(policy)[index % len(legal_actions(policy))]
    frame = np.full((64, 64, 3), index, dtype=np.uint8)
    context = np.asarray((index / 63, -index / 63), dtype=np.float32)
    return replay_module.Transition(
        frame,
        context,
        action,
        -0.01,
        frame,
        context,
        False,
        expert,
        imitation_action=action if expert else None,
    )


def _completed_rows(policy, *, expert: bool) -> list:
    rows = [_transition(index, policy, expert=expert) for index in range(4)]
    last = rows[-1]
    rows[-1] = replay_module.Transition(
        last.frame,
        last.context,
        last.action,
        1.0,
        last.next_frame,
        last.next_context,
        True,
        expert,
        imitation_action=last.action if expert else None,
    )
    return replay_module.episode_n_step(rows, horizon=3, discount=0.99)


def _replay(policy=train.PolicyId.EXPLORE):
    return replay_module.CompressedReplay(
        64,
        2508,
        priority_alpha=0.6,
        demonstration_bonus=0.1,
        demonstration_fraction=0.25,
        policy_id=policy,
    )


def test_schedules_cover_frozen_structural_strata() -> None:
    navigate = protocol.training_requests(train.PolicyId.NAVIGATE_TO)
    assert {(request.distance_band, request.first_direction, request.variant) for request in navigate} == {
        (band, direction, variant)
        for band in protocol.NAVIGATION_BANDS
        for direction in range(1, 5)
        for variant in ("clear", "obstructed")
    }
    resources = protocol.training_requests(train.PolicyId.GET_RESOURCE)
    assert {request.target_kind for request in resources} == set(protocol.RESOURCE_KINDS)
    assert all(sum(request.target_kind == kind for request in resources) == 4 for kind in protocol.RESOURCE_KINDS)
    eat_target = protocol.training_requests(train.PolicyId.EAT_TARGET)
    assert sum(request.variant == "distractor" for request in eat_target) == 12
    eat_cow = protocol.training_requests(train.PolicyId.EAT_COW)
    assert sum(request.variant == "reacquisition" for request in eat_cow) == 12


def test_all_navigate_to_strata_have_exact_deterministic_geometry() -> None:
    requests = protocol.training_requests(train.PolicyId.NAVIGATE_TO)
    short_obstructed = [
        request
        for request in requests
        if request.distance_band == "short" and request.variant == "obstructed"
    ]
    assert all(request.maximum_distance == 4 for request in short_obstructed)
    for request in requests:
        env = cases.make_env(5_240_200)
        env.reset()
        case = cases.setup_case(env, request)
        assert request.minimum_distance <= case.safe_path_length <= request.maximum_distance
        assert case.shortest_path_first_direction == request.first_direction
        assert (case.safe_path_length > case.manhattan_distance) is (
            request.variant == "obstructed"
        )


def test_navigate_to_corridor_blocks_off_path_moves_without_damage() -> None:
    request = next(
        request
        for request in protocol.training_requests(train.PolicyId.NAVIGATE_TO)
        if request.first_direction == 1 and request.variant == "obstructed"
    )
    env = cases.make_env(5_240_200)
    env.reset()
    cases.setup_case(env, request)
    initial_position = cases.position(env)
    _, _, done, info = env.step(2)
    assert cases.position(env) == initial_position
    assert not done
    assert info["inventory"]["health"] > 0


def test_resource_corridor_blocks_barrier_moves_without_damage() -> None:
    request = next(
        request
        for request in protocol.training_requests(train.PolicyId.GET_RESOURCE)
        if request.first_direction == 1 and request.variant.startswith("obstructed")
    )
    env = cases.make_env(5_240_400)
    env.reset()
    cases.setup_case(env, request)
    initial_position = cases.position(env)
    _, _, done, info = env.step(3)
    assert cases.position(env) == initial_position
    assert not done
    assert info["inventory"]["health"] > 0


def test_case_request_rejects_legacy_composition() -> None:
    with pytest.raises(TypeError):
        cases.CaseRequest(
            train.PolicyId.EAT_TARGET,
            "cow",
            execution_mode="composed",
            variant="distractor",
        )
    with pytest.raises(ValueError, match="variant"):
        cases.CaseRequest(train.PolicyId.NAVIGATE_TO, "safe_cell", variant="ordinary")


def test_replay_rejects_masked_or_online_imitation_actions() -> None:
    replay = _replay()
    invalid = _transition(0, train.PolicyId.EXPLORE, expert=True)
    invalid = replay_module.Transition(
        invalid.frame,
        invalid.context,
        0,
        invalid.reward,
        invalid.next_frame,
        invalid.next_context,
        invalid.done,
        True,
        imitation_action=0,
    )
    with pytest.raises(ValueError, match="mask"):
        replay.append(invalid)
    online = _transition(0, train.PolicyId.EXPLORE, expert=False)
    online = replay_module.Transition(
        online.frame,
        online.context,
        online.action,
        online.reward,
        online.next_frame,
        online.next_context,
        online.done,
        False,
        imitation_action=online.action,
    )
    with pytest.raises(ValueError, match="Only demonstration"):
        replay.append(online)


def test_replay_round_trip_binds_policy_and_rng() -> None:
    replay = _replay(train.PolicyId.EAT_TARGET)
    for row in _completed_rows(train.PolicyId.EAT_TARGET, expert=True):
        replay.append(row)
    replay.seal_demonstrations()
    for row in _completed_rows(train.PolicyId.EAT_TARGET, expert=False):
        replay.append(row)
    replay.sample(4, 0.4)
    restored = replay_module.CompressedReplay.from_state_dict(
        replay.state_dict(),
        expected_policy_id=train.PolicyId.EAT_TARGET,
    )
    assert restored.policy_id is train.PolicyId.EAT_TARGET
    assert restored.random.getstate() == replay.random.getstate()
    with pytest.raises(ValueError, match="identity"):
        replay_module.CompressedReplay.from_state_dict(
            replay.state_dict(),
            expected_policy_id=train.PolicyId.EXPLORE,
        )


def test_margin_ignores_illegal_q_outputs() -> None:
    torch = pytest.importorskip("torch")
    q_values = torch.tensor([[100.0, 5.0, 0.0, 0.0, 0.0, 90.0]])
    imitation = torch.tensor([1])
    selected = torch.tensor([True])
    explore = replay_module.balanced_margin_loss(
        torch,
        q_values,
        imitation,
        selected,
        0.8,
        legal_actions(train.PolicyId.EXPLORE),
    )
    resource = replay_module.balanced_margin_loss(
        torch,
        q_values,
        imitation,
        selected,
        0.8,
        legal_actions(train.PolicyId.GET_RESOURCE),
    )
    assert explore.item() == 0.0
    assert resource.item() > 80


def test_one_masked_optimizer_step_is_finite() -> None:
    torch = pytest.importorskip("torch")
    replay = _replay()
    for row in _completed_rows(train.PolicyId.EXPLORE, expert=True):
        replay.append(row)
    replay.seal_demonstrations()
    for row in _completed_rows(train.PolicyId.EXPLORE, expert=False):
        replay.append(row)
    online = train.build_q_network(torch)
    target = train.build_q_network(torch)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
    config = {**train.CONFIG, "batch_size": 4}
    losses = replay_module.optimize(
        torch, online, target, optimizer, replay, "cpu", 0.4, config
    )
    assert all(np.isfinite(value) for value in losses.values())


@pytest.mark.parametrize(
    ("policy", "seed"),
    [
        (train.PolicyId.EXPLORE, 5_240_000),
        (train.PolicyId.NAVIGATE_TO, 5_240_200),
        (train.PolicyId.GET_RESOURCE, 5_240_403),
        (train.PolicyId.EAT_TARGET, 5_240_600),
        (train.PolicyId.EAT_COW, 5_240_800),
    ],
)
def test_complete_expert_cases(policy, seed) -> None:
    torch = pytest.importorskip("torch")
    env = cases.make_env(seed)
    env.reset()
    case = cases.setup_case(env, protocol.training_requests(policy)[0])
    initial_player = cases.position(env)
    original_target = case.target_cell
    rows, success, stats = cases.run_case(
        torch,
        train.build_q_network(torch),
        env,
        case,
        expert=True,
        epsilon=0.0,
        device="cpu",
        seed=seed,
        discount=0.99,
        horizon=3,
    )
    assert rows and success
    assert all(row.action in legal_actions(policy) for row in rows)
    assert all(row.expert and row.imitation_action == row.action for row in rows)
    assert all(np.isfinite(row.context).all() for row in rows)
    if policy is train.PolicyId.NAVIGATE_TO:
        assert original_target == case.target_cell
        assert case.safe_path_length >= 2
        assert any(np.count_nonzero(row.context) for row in rows)
    if policy in {
        train.PolicyId.GET_RESOURCE,
        train.PolicyId.EAT_TARGET,
        train.PolicyId.EAT_COW,
    }:
        assert stats["movement_actions"] > 0
        assert stats["do_actions"] > 0
    if policy is train.PolicyId.EAT_TARGET:
        assert stats["target_cow_consumed"] == 1
    if policy is train.PolicyId.EAT_COW:
        assert original_target not in cases.visible_cells(initial_player)
        assert cases._cow_sector(initial_player, original_target) == case.request.first_direction
        assert case.request.minimum_distance <= case.safe_path_length <= case.request.maximum_distance
        assert np.array_equal(rows[0].context, np.zeros(2, dtype=np.float32))
        assert stats["search_actions"] > 0
        assert stats["acquisition_events"] == 1
        assert stats["pursuit_actions"] > 0


def test_eat_cow_search_expert_does_not_route_to_private_target(monkeypatch) -> None:
    request = protocol.training_requests(train.PolicyId.EAT_COW)[0]
    env = cases.make_env(5_240_800)
    env.reset()
    case = cases.setup_case(env, request)

    def forbidden_private_route(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Search expert queried an unrestricted private route.")

    monkeypatch.setattr(cases, "shortest_path", forbidden_private_route)
    assert cases.expert_action(env, case) in cases.DIRECTIONS


@pytest.mark.parametrize(
    "variant",
    ["clear_aligned", "clear_misaligned", "obstructed_aligned", "obstructed_misaligned"],
)
def test_resource_facing_variants_are_structural(variant) -> None:
    obstructed = variant.startswith("obstructed")
    request = cases.CaseRequest(
        train.PolicyId.GET_RESOURCE,
        "tree",
        "long" if obstructed else "short",
        7 if obstructed else 2,
        10,
        first_direction=1,
        variant=variant,
    )
    env = cases.make_env(5_240_400)
    env.reset()
    case = cases.setup_case(env, request)
    path = cases.shortest_path(env, cases.position(env), {case.approach_cell}, 10)
    assert path is not None and len(path) > 1
    first_direction = cases.DIRECTIONS[cases._direction(path[0], path[1])]
    assert (case.initial_facing == first_direction) is ("misaligned" not in variant)
    assert (case.safe_path_length > case.manhattan_distance) is obstructed


def test_wrong_cow_reward_is_nonterminal_and_identity_private() -> None:
    torch = pytest.importorskip("torch")
    request = next(
        request
        for request in protocol.training_requests(train.PolicyId.EAT_TARGET)
        if request.variant == "distractor"
    )
    seed = next(
        seed
        for seed in range(5_241_000, 5_241_200)
        if _case_exists(seed, request)
    )
    env = cases.make_env(seed)
    env.reset()
    case = cases.setup_case(env, request)
    distractor = case.distractor_objects[0]
    distractor.health = 1
    approach = next(cell for cell in sorted(cases.neighbors(tuple(distractor.pos))) if cases.walkable(env, cell))
    cases._move_player(env, approach, cases._direction(approach, tuple(distractor.pos)))

    class Do(torch.nn.Module):
        def forward(self, images, contexts):
            del images, contexts
            return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    rows, _, stats = cases.run_case(
        torch,
        Do(),
        env,
        case,
        expert=False,
        epsilon=0.0,
        device="cpu",
        seed=seed,
        discount=0.99,
        horizon=3,
    )
    assert rows[0].reward == 0.3
    assert rows[0].done is False
    assert stats["wrong_cows_consumed"] == 1
    assert stats["target_cow_consumed"] == 0
    assert all(row.imitation_action is None and not row.expert for row in rows)
    assert all(not hasattr(row, "target_object") for row in rows)


def test_eat_target_private_goal_tracks_cow_outside_visible_area() -> None:
    request = protocol.training_requests(train.PolicyId.EAT_TARGET)[0]
    env = cases.make_env(5_241_200)
    env.reset()
    case = cases.setup_case(env, request)
    visible = cases.visible_cells(cases.position(env))
    hidden = next(
        cell
        for cell in ((x, y) for x in range(64) for y in range(64))
        if cell not in visible and cases.walkable(env, cell)
    )

    env._world.move(case.target_object, hidden)

    assert cases.current_target(case) == hidden


def test_eat_target_fixture_has_only_controlled_cows_and_requires_consumption() -> None:
    request = next(
        request
        for request in protocol.training_requests(train.PolicyId.EAT_TARGET)
        if request.variant == "distractor"
    )
    env = cases.make_env(5_241_201)
    env.reset()
    case = cases.setup_case(env, request)
    controlled = {case.target_object, *case.distractor_objects}

    assert {
        obj for obj in env._world.objects if type(obj).__name__ == "Cow"
    } == controlled
    env._world.remove(case.target_object)
    assert not cases.succeeded(env, case)


def test_eat_target_epsilon_actions_follow_the_observable_goal() -> None:
    rng = random.Random(2508)
    context = np.asarray((2 / 63, -3 / 63), dtype=np.float32)

    assert {
        cases.exploratory_action(rng, train.PolicyId.EAT_TARGET, context, (0, 1))
        for _ in range(20)
    } == {2, 3}
    adjacent = np.asarray((1 / 63, 0), dtype=np.float32)
    assert cases.exploratory_action(
        rng, train.PolicyId.EAT_TARGET, adjacent, (0, 1)
    ) == 2
    assert cases.exploratory_action(
        rng, train.PolicyId.EAT_TARGET, adjacent, (1, 0)
    ) == 5


def test_get_resource_epsilon_does_not_restore_masked_movements() -> None:
    rng = random.Random(2508)
    adjacent = np.asarray((1 / 63, 0), dtype=np.float32)

    assert cases.exploratory_action(
        rng, train.PolicyId.GET_RESOURCE, adjacent, (1, 0), (),
    ) == 5


@pytest.mark.parametrize("policy", [train.PolicyId.EAT_TARGET, train.PolicyId.EAT_COW])
def test_cow_epsilon_respects_blocked_goal_directions(policy) -> None:
    """A preferred goal direction cannot restore an excluded movement."""
    rng = random.Random(2508)
    context = np.asarray((-2 / 63, 0), dtype=np.float32)
    assert {
        cases.exploratory_action(rng, policy, context, (-1, 0), (2, 3, 4))
        for _ in range(30)
    } == {2, 3, 4}
    adjacent = np.asarray((-1 / 63, 0), dtype=np.float32)
    assert cases.exploratory_action(rng, policy, adjacent, (0, 1), (2,)) == 2
    assert cases.exploratory_action(rng, policy, adjacent, (-1, 0), ()) == 5


def test_eat_cow_epsilon_actions_search_then_follow_the_observable_goal() -> None:
    rng = random.Random(2508)

    assert {
        cases.exploratory_action(
            rng,
            train.PolicyId.EAT_COW,
            np.zeros(2, dtype=np.float32),
            (0, 1),
        )
        for _ in range(20)
    } == set(cases.DIRECTIONS)
    context = np.asarray((-2 / 63, 3 / 63), dtype=np.float32)
    assert {
        cases.exploratory_action(
            rng, train.PolicyId.EAT_COW, context, (0, 1)
        )
        for _ in range(20)
    } == {1, 4}
    adjacent = np.asarray((0, -1 / 63), dtype=np.float32)
    assert cases.exploratory_action(
        rng, train.PolicyId.EAT_COW, adjacent, (0, 1)
    ) == 3
    assert cases.exploratory_action(
        rng, train.PolicyId.EAT_COW, adjacent, (0, -1)
    ) == 5


def test_eat_cow_network_masks_do_until_cow_is_adjacent_and_faced() -> None:
    torch = pytest.importorskip("torch")

    class PreferDo(torch.nn.Module):
        def forward(self, images, contexts):
            del images, contexts
            return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0]])

    frame = np.zeros((9, 9, 3), dtype=np.uint8)
    searching = np.zeros(2, dtype=np.float32)
    pursuing = np.asarray((1 / 63, 0), dtype=np.float32)

    action, values = cases.network_action(
        torch,
        PreferDo(),
        frame,
        searching,
        "cpu",
        train.PolicyId.EAT_COW,
        (0, 1),
        (1, 2, 3, 4),
    )
    assert action == 4
    assert values[5] == 100.0
    assert cases.network_action(
        torch,
        PreferDo(),
        frame,
        pursuing,
        "cpu",
        train.PolicyId.EAT_COW,
        (0, 1),
        (1, 2, 3, 4),
    )[0] == 4
    assert cases.network_action(
        torch,
        PreferDo(),
        frame,
        pursuing,
        "cpu",
        train.PolicyId.EAT_COW,
        (1, 0),
        (1, 2, 3, 4),
    )[0] == 5
    assert cases.network_action(
        torch,
        PreferDo(),
        frame,
        searching,
        "cpu",
        train.PolicyId.EAT_COW,
        (0, 1),
        (1, 2, 3),
    )[0] == 3


def test_eat_cow_reacquisition_returns_to_zero_search_context() -> None:
    torch = pytest.importorskip("torch")
    request = next(
        request
        for request in protocol.training_requests(train.PolicyId.EAT_COW)
        if request.variant == "reacquisition"
    )
    env = cases.make_env(5_242_000)
    env.reset()
    case = cases.setup_case(env, request)
    rows, success, stats = cases.run_case(
        torch,
        train.build_q_network(torch),
        env,
        case,
        expert=True,
        epsilon=0.0,
        device="cpu",
        seed=5_242_000,
        discount=0.99,
        horizon=3,
    )
    zero = np.zeros(2, dtype=np.float32)
    assert success and case.forced_reacquisition
    assert stats["reacquisition_events"] == 1
    assert any(np.count_nonzero(row.context) for row in rows)
    first_visible = next(index for index, row in enumerate(rows) if np.count_nonzero(row.context))
    assert any(np.array_equal(row.context, zero) for row in rows[first_visible + 1:])


def _case_exists(seed: int, request) -> bool:
    env = cases.make_env(seed)
    env.reset()
    try:
        cases.setup_case(env, request)
    except ValueError:
        return False
    return True


def test_optional_warm_start_resets_learning_state(tmp_path):
    torch = pytest.importorskip("torch")
    spec = protocol.POLICY_SPEC_BY_ID[train.PolicyId.GET_RESOURCE]
    parent = train.build_q_network(torch)
    path = tmp_path / "parent.pt"
    torch.save(checkpoint_data(train.PolicyId.NAVIGATE_TO, parent.state_dict()), path)
    state, online, target, optimizer, replay = train._new_state(torch, spec, "cpu", path)
    assert state["online_steps"] == state["optimizer_steps"] == state["demonstration_steps"] == 0
    assert optimizer.state_dict()["state"] == {} and replay.rows == []
    transferred = online.state_dict()
    for name, value in parent.state_dict().items():
        if name not in {"head.2.weight", "head.2.bias"}:
            assert torch.equal(value, transferred[name])
    assert torch.equal(parent.state_dict()["head.2.weight"][1:5], transferred["head.2.weight"][1:5])
    assert torch.equal(parent.state_dict()["head.2.bias"][1:5], transferred["head.2.bias"][1:5])
    assert not torch.equal(parent.state_dict()["head.2.weight"][[0, 5]], transferred["head.2.weight"][[0, 5]])
    assert all(torch.equal(value, target.state_dict()[name]) for name, value in transferred.items())


def test_current_working_state_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    spec = protocol.POLICY_SPEC_BY_ID[train.PolicyId.EXPLORE]
    state, online, target, optimizer, replay = train._new_state(torch, spec, "cpu")
    path = tmp_path / "work.pt"
    torch.save({
        "state": state, "online": online.state_dict(), "target": target.state_dict(),
        "optimizer": optimizer.state_dict(), "replay": replay.state_dict(), "rng": train._rng_state(torch),
    }, path)
    restored, model, _, optim, memory = train._load_state(torch, path, spec, "cpu")
    assert restored == state
    assert optim.state_dict() == optimizer.state_dict()
    assert memory.state_dict() == replay.state_dict()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in online.state_dict().items())
    state["format_version"] = 4
    torch.save({"state": state}, path)
    with pytest.raises(ValueError, match="incompatible"):
        train._load_state(torch, path, spec, "cpu")


def test_cli_has_no_certification_or_dependencies():
    args = cli.parse_args(["train", "run", "--device", "cpu", "--policy", "eat_cow", "--init-checkpoint", "parent.pt"])
    assert args.policy == "eat_cow" and args.init_checkpoint == Path("parent.pt")
    with pytest.raises(SystemExit):
        cli.parse_args(["certify", "run"])


def test_no_implicit_dependencies_and_invalid_resume_fails_before_writing(tmp_path):
    with pytest.raises(ValueError, match="requires one policy"):
        train.train(tmp_path / "run", "cpu", init_checkpoint=Path("missing.pt"))
    assert not (tmp_path / "run").exists()


def test_export_requires_five_current_models(tmp_path):
    with pytest.raises(ValueError, match="five policies"):
        train.export_models({}, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
