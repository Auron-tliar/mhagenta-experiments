"""Behavioral and leakage regressions for natural remaining-policy audits."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "tools" / "exp2_5_cr_preparation"))
audit = importlib.import_module("remaining_evaluation")
cases = importlib.import_module("cases")


def test_world_ranges_and_cohort_balance() -> None:
    """No world belongs to two policies/splits, including generation retries."""
    blocks = sorted((base, base + 10000 * 64) for splits in audit.SPLIT_SEEDS.values() for base in splits.values())
    assert all(left[1] < right[0] for left, right in zip(blocks, blocks[1:]))
    for policy in audit.POLICIES:
        schedule = [audit.request(index, policy) for index in range(240)]
        assert {sum(row['direction'] == d for row in schedule) for d in range(1,5)} == {60}
        assert {sum(row['band'] == b for row in schedule) for b in range(3)} == {80}
        assert sum(row['source'] == 'history' for row in schedule) == 120
        if policy is audit.PolicyId.EAT_TARGET:
            assert sum(row['variant'] == 'distractor' for row in schedule) == 120


def test_gate_rejects_small_cohorts_subgroup_collapse_and_missing_search() -> None:
    """An aggregate pass cannot conceal unsafe or structurally invalid behavior."""
    rows = [{**audit.request(i, audit.PolicyId.EAT_COW), 'policy':'eat_cow', 'success':True,
             'search_actions':2, 'acquisition_events':1, 'pursuit_actions':3, 'do_actions':3}
            for i in range(240)]
    assert audit.summarize(rows,240)['passed']
    assert not audit.summarize(rows[:24],240)['passed']
    rows[0]['search_actions']=0
    assert not audit.summarize(rows,240)['passed']
    rows[0]['search_actions']=2
    rows[0]['illegal_actions']=1
    assert not audit.summarize(rows,240)['passed']
    rows[0]['illegal_actions']=0
    for row in [r for r in rows if r['direction']==1][:7]:
        row['success']=False
    assert sum(row['success'] for row in rows)/240 > .95
    assert not audit.summarize(rows,240)['passed']


@pytest.mark.parametrize('policy', audit.POLICIES)
def test_native_descriptors_replay_without_terrain_or_cow_edits(tmp_path, policy) -> None:
    """A freshly regenerated descriptor matches both terrain and native objects."""
    source = audit.RemainingCases(tmp_path,'pilot',policy)
    env, case, record = source.get(0)
    restored, known = source._restore(record)
    assert case.native_dynamics
    assert known == case.known
    assert np.array_equal(env.render(), restored.render())
    assert not record['terrain_modified'] and not record['cow_placement_modified']
    cow_positions = lambda world: sorted(tuple(obj.pos) for obj in world.objects if type(obj).__name__=='Cow')
    assert cow_positions(env._world) == cow_positions(restored._world)
    env.step(1)
    replayed, _, second = audit.RemainingCases(tmp_path,'pilot',policy).get(0)
    assert np.array_equal(replayed.render(), restored.render())
    assert second == record
    json.dumps(record)


def test_native_target_has_no_invisible_identity_oracle(tmp_path) -> None:
    """Moving a target out of view cannot leak its private coordinates as context."""
    env, case, _ = audit.RemainingCases(tmp_path,'pilot',audit.PolicyId.EAT_TARGET).get(0)
    assert cases.current_target(case) is not None
    for obj in list(env._world.objects):
        if type(obj).__name__=='Cow':
            env._world.remove(obj)
    assert cases.current_target(case) is None


def test_native_target_association_uses_positions_and_never_reacquires() -> None:
    """Public association follows a vacated cell without reading target identity."""
    player = type("Player", (), {"pos": (0, 0)})()
    first = type("Cow", (), {"pos": (1, 0)})()
    second = type("Cow", (), {"pos": (2, 0)})()
    world = SimpleNamespace(objects=[player, first, second])
    identity = SimpleNamespace(world=world)  # Deliberately has no position or health.
    case = cases.Case(cases.CaseRequest(audit.PolicyId.EAT_TARGET, "cow", variant="single"),
                      (2, 0), 0, set(), target_object=identity,
                      native_dynamics=True, observed_target=(2, 0))
    assert cases.current_target(case) == (2, 0)
    first.pos, second.pos = (2, 0), (3, 0)
    assert cases.current_target(case) == (3, 0)
    assert cases.current_target(case) == (3, 0)
    world.objects = [player, first]
    assert cases.current_target(case) is None
    world.objects.append(second)
    assert cases.current_target(case) is None
    assert case.target_object is identity


@pytest.mark.parametrize("player,target,expected", [
    ((10, 10), (9, 7), (10, 9)),
    ((10, 10), (9, 13), (10, 11)),
    ((10, 10), (14, 9), (11, 10)),
    ((10, 10), (6, 9), (9, 10)),
])
def test_visible_pursuit_prefers_camera_margin_on_shortest_routes(player, target, expected):
    """Four camera edges favor inward tracking without increasing route length."""
    safe = cases.visible_cells(player) - {target}
    path = cases.visible_cow_pursuit_path(player, target, safe)
    baseline = cases._visible_path(player, cases.neighbors(target) & safe, safe)
    assert path[1] == expected and len(path) == len(baseline)


def test_visible_pursuit_uses_returned_step_and_retains_obstacle_detour():
    """BFS preference is not a forced action; only actual safe path steps are valid."""
    player, target = (10, 10), (10, 7)
    safe = {(10, 10), (11, 10), (11, 9), (11, 8), (10, 8)}
    path = cases.visible_cow_pursuit_path(player, target, safe)
    assert path == [(10, 10), (11, 10), (11, 9), (11, 8), (10, 8)]
    assert cases.visible_cow_pursuit_path(player, target, {player}) is None


@pytest.mark.parametrize("target,expected", [((9, 7), 3), ((9, 13), 4)])
def test_native_eat_cow_teacher_preserves_visibility_on_route_ties(monkeypatch, target, expected):
    """The actual EatCow teacher must avoid equally short camera-edge escapes."""
    player = (10, 10)
    safe = cases.visible_cells(player) - {target}
    env = SimpleNamespace(_player=SimpleNamespace(pos=player, facing=(0, 1)))
    case = cases.Case(cases.CaseRequest(audit.PolicyId.EAT_COW, "cow"), target,
                      0, safe, native_dynamics=True)
    monkeypatch.setattr(cases, "current_target", lambda case: case.target_cell)
    monkeypatch.setattr(cases, "_visible_safe", lambda env: safe)
    monkeypatch.setattr(cases, "walkable", lambda env, cell: cell in safe)
    assert cases.expert_action(env, case) == expected


def test_legacy_training_cannot_start_for_remaining_policies(tmp_path, monkeypatch) -> None:
    """The production CLI cannot silently fall back to the small fixture cohort."""
    train = importlib.import_module('train')
    randomized = importlib.import_module('resource_training')
    calls = []
    def capture(torch, output, policy, *args, **kwargs):
        calls.append(policy)
        return {}
    monkeypatch.setattr(randomized, 'train_randomized', capture)
    for policy in audit.POLICIES:
        spec = train.POLICY_SPEC_BY_ID[policy]
        train._train_policy(None, tmp_path, spec, 'cpu', False, None)
        assert isinstance(randomized.randomized_cases(tmp_path,'train',policy), audit.RemainingCases)
        assert randomized.RANDOMIZED_CONFIGS[policy]['validation_cases'] == 240
        assert randomized.RANDOMIZED_CONFIGS[policy]['test_cases'] == 600
    assert calls == list(audit.POLICIES)


def test_explore_epsilon_uses_the_same_movement_exclusions() -> None:
    """Training exploration cannot reintroduce the diagnosed blocked action."""
    rng = random.Random(0)
    for _ in range(100):
        assert cases.exploratory_action(rng, audit.PolicyId.EXPLORE, np.zeros(2), (1,0), (2,4)) in (2,4)


@pytest.mark.parametrize('change', ({'expert':True}, {'required':24}, {'passed':False}, {'checkpoint_sha256':'wrong'}))
def test_test_split_rejects_invalid_validation_before_loading_model(tmp_path, change) -> None:
    """Neither a small cohort nor an expert run can unlock the untouched test."""
    checkpoint = tmp_path/'placeholder.pt'
    checkpoint.write_bytes(b'not a torch model; must never be loaded')
    validation = tmp_path/'validation.json'
    audit.atomic_json({'passed':True,'expert':False,'required':240,'split':'validation',
                       'checkpoint_sha256':audit.file_sha256(checkpoint),
                       'protocol_version':audit.VERSION,**change}, validation)
    with pytest.raises(ValueError, match='Test requires a passing current validation'):
        audit.evaluate(checkpoint,audit.PolicyId.EXPLORE,tmp_path/'audit','test','cpu',validation=validation)


def test_native_replay_can_start_at_original_player_cell(tmp_path) -> None:
    """Relocating to the current occupied cell must not invoke world.move."""
    env = cases.make_env(1110002368, environment_type=audit.ResourceEnvironment)
    env.reset()
    record = {'seed':1110002368, 'start':list(cases.position(env)),
              'facing':[0,1], 'inventory':dict(env._player.inventory), 'prefix':[]}
    restored, _ = audit.RemainingCases(tmp_path,'pilot',audit.PolicyId.EXPLORE)._restore(record)
    assert cases.position(restored) == cases.position(env)


def test_native_training_commits_and_stops_on_an_unsafe_episode(tmp_path, monkeypatch) -> None:
    """An unsafe episode must not silently continue toward another boundary."""
    torch = pytest.importorskip('torch')
    randomized = importlib.import_module('resource_training')
    replay = importlib.import_module('replay')
    frame = np.zeros((64,64,3), dtype=np.uint8)
    context = np.zeros(2, dtype=np.float32)
    row = replay.Transition(frame, context, 1, -.01, frame, context, True, True, imitation_action=1)
    row = replay.episode_n_step([row], horizon=3, discount=.99)[0]
    stats = {'case_seconds':0., 'execution_seconds':0., 'world_seed':1,
             'source':'fresh','kind':'frontier','distance_band':'2-4',
             'illegal_actions':1,'lethal_actions':0,'environment_terminal_failures':0}
    monkeypatch.setattr(randomized, 'randomized_cases', lambda *args: SimpleNamespace(split='pilot'))
    monkeypatch.setattr(randomized, 'randomized_episode', lambda *args, **kwargs: ([row],False,stats))
    state = randomized.train_randomized(torch,tmp_path,audit.PolicyId.EXPLORE,'cpu',False,
                                       SimpleNamespace(requested=False),None,diagnostic=True)
    assert state['phase']=='failed_safety'
    assert state['episodes']==1 and state['demonstration_steps']==1
    assert state['failure_episode']['illegal_actions']==1
    saved = torch.load(tmp_path/'work/explore.pt',map_location='cpu',weights_only=False)
    assert saved['state']['phase']=='failed_safety'


def test_native_training_selection_requires_real_search_structure(monkeypatch) -> None:
    """The production trainer must enforce the same behavioral gate as audits."""
    randomized = importlib.import_module('resource_training')
    def fake_episode(*args, **kwargs):
        return [None],True,{'distance':2,'source':'fresh','kind':'cow','distance_band':'1-2',
                           'direction':1,'variant':'ordinary','illegal_actions':0,'lethal_actions':0,
                           'environment_terminal_failures':0,'movement_blocked':0,
                           'search_actions':0,'acquisition_events':1,'pursuit_actions':1,'do_actions':1}
    monkeypatch.setattr(randomized,'randomized_episode',fake_episode)
    report = randomized.evaluate_randomized(None,None,SimpleNamespace(policy=audit.PolicyId.EAT_COW,split='validation'),'cpu',240)
    assert report['succeeded']==240
    assert not report['passed']


@pytest.mark.parametrize('policy', audit.POLICIES)
def test_snapshot_and_regeneration_produce_identical_complete_episodes(tmp_path, policy) -> None:
    """The speed optimization must preserve native RNG, frames and outcomes."""
    first = audit.RemainingCases(tmp_path,'pilot',policy)
    cached_env, cached_case, descriptor = first.get(0)
    (first.root/'case-0.state.pkl').unlink()
    (first.root/'case-0.state.json').unlink()
    regenerated_env, regenerated_case, _ = audit.RemainingCases(tmp_path,'pilot',policy).get(0)
    args = dict(expert=True,epsilon=0.,device='cpu',seed=descriptor['seed'],discount=.99,horizon=3)
    a, success_a, stats_a = cases.run_case(None,None,cached_env,cached_case,**args)
    b, success_b, stats_b = cases.run_case(None,None,regenerated_env,regenerated_case,**args)
    assert success_a == success_b and stats_a == stats_b and len(a)==len(b)
    assert all(x.action==y.action and x.reward==y.reward and np.array_equal(x.frame,y.frame)
               and np.array_equal(x.next_frame,y.next_frame) for x,y in zip(a,b))


def test_corrupted_native_snapshot_is_rejected_before_deserialization(tmp_path) -> None:
    """Damaged cached scenes must not silently become different evaluation cases."""
    source = audit.RemainingCases(tmp_path,'pilot',audit.PolicyId.EXPLORE)
    source.get(0)
    (source.root/'case-0.state.pkl').write_bytes(b'corrupted snapshot')
    with pytest.raises(ValueError,match='snapshot checksum'):
        audit.RemainingCases(tmp_path,'pilot',audit.PolicyId.EXPLORE).get(0)


def test_cow_teacher_keeps_frontier_without_hidden_target_access(tmp_path) -> None:
    """Private target identity cannot change an ungrounded search action or goal."""
    from copy import deepcopy
    env, case, _ = audit.RemainingCases(tmp_path, 'pilot', audit.PolicyId.EAT_COW).get(0)
    assert cases.current_target(case) is None
    action = cases.expert_action(env, case)
    goal = case.search_goal
    assert goal is not None
    changed = deepcopy(case)
    changed.target_cell = (-1000, -1000)
    changed.target_object = SimpleNamespace(world=env._world)
    assert cases.expert_action(env, changed) == action
    assert changed.search_goal == goal
    env.step(action)
    case.known.update(cases.visible_cells(cases.position(env)))
    if cases.current_target(case) is None and cases.position(env) != goal:
        cases.expert_action(env, case)
        assert case.search_goal == goal


def test_cow_teacher_can_reveal_terrain_without_reachable_boundary(tmp_path) -> None:
    """A safe camera shift is useful even when no terrain frontier is reachable."""
    env, case, _ = audit.RemainingCases(tmp_path, 'train', audit.PolicyId.EAT_COW).get(27)
    player = cases.position(env)
    safe = {cell for cell in case.known if cases.walkable(env, cell)} | {player}
    frontiers = set(cases.frontier_cells(case.known, safe))
    assert cases._visible_path(player, frontiers, safe) is None
    action = cases.observable_cow_search(env, case)
    env.step(action)
    assert cases.visible_cells(cases.position(env)) - case.known


@pytest.mark.parametrize('split,index', [('pilot', 32), ('train', 3), ('train', 13), ('train', 9)])
def test_cow_demonstrations_respect_execution_mask(tmp_path, split, index) -> None:
    """Demonstrations cannot teach recent-cell moves forbidden to the network."""
    env, case, descriptor = audit.RemainingCases(tmp_path, split, audit.PolicyId.EAT_COW).get(index)
    rows, success, stats = cases.run_case(None, None, env, case, expert=True, epsilon=0.,
        device='cpu', seed=descriptor['seed'], discount=.99, horizon=3)
    assert rows
    assert all(row['action'] == 5 or row['action'] in row['available_movements']
               for row in stats['action_trace'])
    assert all(stats[key] == 0 for key in audit.SAFETY_FIELDS)
    if split == 'train' and index == 9:
        assert success  # Previously stopped after 14 actions at an obstructed cow.


def test_eat_cow_training_budget_is_isolated() -> None:
    """The approved budget cannot silently extend EatTarget or accepted policies."""
    from mha_exp_level2_cr.exp2_5.contracts import activity_action_bound
    trainer = importlib.import_module('resource_training')
    for policy in audit.PolicyId:
        assert activity_action_bound(policy.value) == (96 if policy is audit.PolicyId.EAT_COW else 32)
    assert trainer.RANDOMIZED_CONFIGS[audit.PolicyId.EAT_COW]['version'] == 34
    assert trainer.RANDOMIZED_CONFIGS[audit.PolicyId.EAT_COW]['teacher'] == 'masked_discovery_visibility_pursuit_v3'
    assert trainer.RANDOMIZED_CONFIGS[audit.PolicyId.EAT_TARGET]['version'] == 33
    assert trainer.RANDOMIZED_CONFIGS[audit.PolicyId.EAT_TARGET]['target_tracking'] == 'observable_joint_association_v2'
    assert trainer.RANDOMIZED_CONFIGS[audit.PolicyId.EAT_TARGET]['teacher'] == 'observable_visibility_tie_v1'


def test_eat_cow_run_records_effective_budget(tmp_path, monkeypatch) -> None:
    """The saved run configuration must describe actual 96-action episodes."""
    train = importlib.import_module('train')
    monkeypatch.setattr(train, '_set_determinism', lambda *args: 'test-cpu')
    monkeypatch.setattr(train, '_train_policy', lambda *args, **kwargs: {'phase':'diagnostic_complete'})
    train.train(tmp_path / 'run', 'cpu', audit.PolicyId.EAT_COW, diagnostic=True)
    config = json.loads((tmp_path / 'run' / 'config.json').read_text())
    assert config['configuration']['activity_action_bound'] == 96
    assert config['configuration']['maximum_demo_episode_overshoot'] == 95
    assert config['configuration']['maximum_committed_online_steps'] == 20095
    assert config['randomized_configuration']['activity_action_bound'] == 96
    assert train.CONFIG['activity_action_bound'] == 32


@pytest.mark.parametrize('index', [5, 17])
def test_integrated_cow_teacher_completes_long_native_search(tmp_path, index) -> None:
    """Previously failing 32-action searches finish safely under the approved limit."""
    env, case, descriptor = audit.RemainingCases(tmp_path, 'pilot', audit.PolicyId.EAT_COW).get(index)
    rows, success, stats = cases.run_case(None, None, env, case, expert=True, epsilon=0.,
        device='cpu', seed=descriptor['seed'], discount=.99, horizon=3)
    assert success and 32 < len(rows) <= 64
    assert all(stats[key] == 0 for key in audit.SAFETY_FIELDS)
    assert all(stats[key] > 0 for key in ('search_actions', 'acquisition_events', 'pursuit_actions', 'do_actions'))


@pytest.mark.parametrize('policy', audit.POLICIES)
def test_native_blocked_route_logging_preserves_episode(policy, monkeypatch) -> None:
    """A blocked native teacher route is an outcome, not a missing-key crash."""
    trainer = importlib.import_module('resource_training')
    descriptor = {'source':'history', 'seed':123, 'distance':3, 'bounds':[2,3],
                  'generation_rejections':0, 'direction':1, 'variant':'ordinary'}
    case = SimpleNamespace(target_kind='frontier' if policy is audit.PolicyId.EXPLORE else 'cow')
    cohort = SimpleNamespace(get=lambda index: (None, case, descriptor))
    transitions = [object()]
    monkeypatch.setattr(trainer, 'run_case', lambda *args, **kwargs: (transitions, False, {'expert_route_blocked':1}))
    events = []
    monkeypatch.setattr(trainer, '_emit_event', lambda event, **fields: events.append((event,fields)))
    rows, success, stats = trainer.randomized_episode(None,None,cohort,0,'cpu',expert=True)
    assert rows is transitions and not success and stats['expert_route_blocked'] == 1
    assert events == [('expert_route_blocked', {'index':0, 'world_seed':123,
        'retained_transitions':1, 'source':'history', 'kind':case.target_kind})]


@pytest.mark.parametrize('writer_name', ('audit_json', 'train_json', 'torch_checkpoint'))
def test_atomic_report_retries_transient_reader_lock(tmp_path, monkeypatch, writer_name) -> None:
    """A transient Windows sharing failure must not abort a complete episode."""
    io = importlib.import_module('atomic_io')
    train = importlib.import_module('train')
    original = Path.replace
    attempts = []
    def locked_once(source, destination):
        attempts.append(source)
        if len(attempts) == 1:
            raise PermissionError('simulated sharing violation')
        return original(source, destination)
    monkeypatch.setattr(Path, 'replace', locked_once)
    monkeypatch.setattr(io, 'sleep', lambda duration: None)
    destination = tmp_path / 'report.json'
    destination.write_text('old complete report')
    if writer_name == 'audit_json':
        audit.atomic_json({'complete':True}, destination)
    elif writer_name == 'train_json':
        train._atomic_json({'complete':True}, destination)
    else:
        fake_torch = SimpleNamespace(save=lambda value, path: Path(path).write_text(json.dumps(value)))
        train._atomic_torch_save(fake_torch, {'complete':True}, destination)
    assert len(attempts) == 2 and json.loads(destination.read_text()) == {'complete':True}


def test_atomic_report_preserves_both_files_on_permanent_lock(tmp_path, monkeypatch) -> None:
    """A persistent error propagates with the previous and pending reports intact."""
    io = importlib.import_module('atomic_io')
    attempts = []
    def always_locked(source, destination):
        attempts.append(source)
        raise PermissionError('persistent sharing violation')
    monkeypatch.setattr(Path, 'replace', always_locked)
    monkeypatch.setattr(io, 'sleep', lambda duration: None)
    destination = tmp_path / 'report.json'
    destination.write_text('old complete report')
    with pytest.raises(PermissionError):
        audit.atomic_json({'complete':True}, destination)
    assert len(attempts) == 11
    assert destination.read_text() == 'old complete report'
    assert json.loads(destination.with_suffix('.tmp').read_text()) == {'complete':True}
