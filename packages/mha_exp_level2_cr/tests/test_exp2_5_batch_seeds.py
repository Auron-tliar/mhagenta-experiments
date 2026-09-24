"""Cover thesis seed wiring and full-cohort main-CLI execution options."""
from types import SimpleNamespace

import pytest

from mha_exp_common.batch import NonRetryableBatchError
from mha_exp_common.names import HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import module_name
from mha_exp_level2_cr.exp2_5 import runner


SELECTED_SEEDS = (
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1010, 1011, 1012, 1013,
    1014, 1015, 1016, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1027, 1009,
    1017, 1026, 1028, 1029, 1030, 1034, 1035, 1037, 1038, 1039, 1040, 1041,
    1042, 1047, 1061, 1074, 1082, 1084, 1088, 1089, 1096, 1101, 1104, 1105,
)


@pytest.fixture
def runner_harness(monkeypatch):
    """Exercise native environment construction while replacing container execution."""
    import mha_env_crafter
    native_seeds, environments, cleanup = [], [], []

    def native_environment(**kwargs):
        native_seeds.append(kwargs['seed'])
        return SimpleNamespace(reset=lambda: None)

    class FakeOrchestrator:
        INFO = 20
        SAVE_SUBDIR = 'out'

        def __init__(self, **kwargs):
            pass

        def add_agent(self, **kwargs):
            assert kwargs['hl_reasoners'] is not None

        def add_environment(self, **kwargs):
            self.base = kwargs['base']
            environments.append(self.base)

        def run(self, **kwargs):
            # Rebuilding the packaged environment must keep its native seed.
            self.base.__setstate__(self.base.__getstate__())

    monkeypatch.setattr(mha_env_crafter, 'CrafterEnv', native_environment)
    monkeypatch.setattr(mha_env_crafter, 'Recorder', lambda env, *args, **kwargs: env)
    monkeypatch.setattr(runner, 'Orchestrator', FakeOrchestrator)
    monkeypatch.setattr(runner, 'preflight_artifacts', lambda: ({}, {}))
    monkeypatch.setattr(runner, 'check_results', lambda *args, **kwargs: (True, []))
    monkeypatch.setattr(runner, 'cleanup_run_containers', lambda *args, **kwargs: cleanup.append(('containers', args, kwargs['phase'])))
    monkeypatch.setattr(runner, 'cleanup_run_images', lambda *args, **kwargs: cleanup.append(('images', args, kwargs['phase'])))
    return SimpleNamespace(native_seeds=native_seeds, environments=environments, cleanup=cleanup)


@pytest.mark.parametrize('seeds', [None, SELECTED_SEEDS])
def test_seeds_reach_native_crafter_and_survive_packaging(monkeypatch, tmp_path, runner_harness, seeds):
    """Both historical defaults and all selected seeds retain their global run IDs."""
    expected = tuple(range(1000, 1020)) if seeds is None else seeds
    for run, seed in enumerate(expected):
        agent_id, env_id = f'exp_agent2_cr_5_{run}', f'exp_env2_cr_5_{run}'
        monkeypatch.setattr(runner, 'gather_states', lambda *args, a=agent_id, e=env_id, s=seed, **kwargs: {a: {}, e: {e: {'environment_seed': s}}})
        assert runner.run_experiment(run, tmp_path, enable_eat_cow=True, environment_seed=None if seeds is None else seed)
        assert runner_harness.environments[-1].state['environment_seed'] == seed
        assert runner_harness.cleanup[-2:] == [('containers', (agent_id, env_id), 'after'), ('images', (agent_id, env_id), 'after')]
    assert runner_harness.native_seeds == [seed for seed in expected for _ in range(2)]


@pytest.mark.parametrize('stop_on_error', [True, False])
def test_main_cli_forwards_twenty_runs_and_failure_policy(monkeypatch, tmp_path, stop_on_error):
    """The public CLI forwards run count and explicit cohort options unchanged."""
    import json
    from mha_exp_cli.__main__ import main
    captured = {}
    monkeypatch.setattr(runner, '_workspace_root', lambda: tmp_path)
    monkeypatch.setattr(runner, 'preflight_artifacts', lambda: ({}, {}))
    monkeypatch.setattr(runner, 'run_experiment_batch', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runner, 'process_results', lambda *args: (True, {}))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'enable_eat_cow': True, 'stop_on_error': stop_on_error}))
    assert main(['2.cr.5', '-n', '20', '--work-dir', str(tmp_path / 'cohort'), '--config-file', str(config)]) == 0
    assert captured['runs'] == 20
    assert captured['stop_on_error'] is stop_on_error
    assert captured['runner'].keywords['enable_eat_cow'] is True
    assert captured['cleanup_before_run'] is False


def test_batch_rejects_non_boolean_failure_policy(tmp_path):
    """Reject a string false rather than silently changing cohort behavior."""
    with pytest.raises(ValueError, match='stop_on_error'):
        runner.run_batch(runs=20, exp_path=tmp_path, stop_on_error='false')


@pytest.mark.parametrize('start,end', [(0, 12), (13, 25), (26, 37), (38, 49)])
def test_main_cli_preserves_global_shard_ids_and_manifest(monkeypatch, tmp_path, start, end):
    """JSON configuration selects the same seeds regardless of the host shard."""
    import json
    from mha_exp_cli.__main__ import main

    captured, episodes = {}, []
    monkeypatch.setattr(runner, '_workspace_root', lambda: tmp_path)
    monkeypatch.setattr(runner, 'preflight_artifacts', lambda: ({}, {}))
    monkeypatch.setattr(runner, 'run_experiment_batch', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runner, 'process_results', lambda *args: (True, {}))
    monkeypatch.setattr(runner, 'run_experiment', lambda run, *args, **kwargs: episodes.append((run, kwargs)))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'enable_eat_cow': True, 'stop_on_error': False,
        'environment_seeds': {str(run): seed for run, seed in enumerate(SELECTED_SEEDS)},
    }))
    assert main(['2.cr.5', '-r', f'{start}-{end}', '--work-dir', str(tmp_path / 'cohort'), '--config-file', str(config)]) == 0
    assert captured['runs'] == list(range(start, end + 1))
    assert captured['stop_on_error'] is False
    for run in captured['runs']:
        captured['runner'](run, tmp_path, runner.DEFAULT_TORCH_MHAGENTA_VERSION)
    assert episodes == [(run, {'enable_eat_cow': True, 'environment_seed': SELECTED_SEEDS[run]}) for run in range(start, end + 1)]


@pytest.mark.parametrize('mapping', [
    [1000, 1001], {'0': 1000}, {'0': 1000, '1': 1000},
    {'0': 1000, '1': True}, {'0': 1000, '1': '1001'},
    {'0': 1000, '1': -1}, {'0': 1000, '1': 2**32},
    {'0': 1000, '01': 1001}, {'0': 1000, '-1': 1001},
    {False: 1000, '1': 1001}, {0: 1000, '0': 1001, '1': 1002},
])
def test_invalid_manifest_fails_before_preflight_or_output_changes(monkeypatch, tmp_path, mapping):
    """Bad manifest data cannot archive existing output or start container work."""
    output = tmp_path / 'cohort'
    output.mkdir()
    retained = output / 'retained.txt'
    retained.write_text('evidence')
    monkeypatch.setattr(runner, 'preflight_artifacts', lambda: pytest.fail('preflight must not run'))
    with pytest.raises(ValueError, match='environment_seeds'):
        runner.run_batch(runs=2, exp_path=output, environment_seeds=mapping)
    assert retained.read_text() == 'evidence'
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('seed', [True, '1000', -1, 2**32])
def test_single_run_rejects_invalid_seed_before_work(monkeypatch, tmp_path, seed):
    """Direct callers receive the same seed validation as batch callers."""
    monkeypatch.setattr(runner, '_workspace_root', lambda: pytest.fail('workspace lookup must not run'))
    with pytest.raises(ValueError, match='environment_seed'):
        runner.run_experiment(0, tmp_path, environment_seed=seed)


@pytest.mark.parametrize('fault', ['integrity', 'seed', 'missing'])
def test_invalid_execution_stops_without_retry_or_cleanup(monkeypatch, tmp_path, runner_harness, fault):
    """An invalid run retains its evidence even when scientific failures may continue."""
    from mha_exp_common.batch import run_batch

    env_id, agent_id = 'exp_env2_cr_5_13', 'exp_agent2_cr_5_13'
    saved = {} if fault == 'missing' else {agent_id: {}, env_id: {env_id: {'environment_seed': 1000 if fault == 'seed' else 1014}}}
    monkeypatch.setattr(runner, 'gather_states', lambda *args, **kwargs: saved)
    if fault == 'integrity':
        monkeypatch.setattr(runner, 'check_results', lambda *args, **kwargs: (False, ['action/status counts disagree']))
    with pytest.raises(NonRetryableBatchError):
        run_batch(
            experiment_id='2-5-CR', title='test', runs=[13, 14], exp_path=tmp_path / 'cohort',
            runner=lambda run, path, version: runner.run_experiment(run, path, environment_seed=SELECTED_SEEDS[run]),
            cleanup_before_run=False, stop_on_error=False,
        )
    assert len(runner_harness.environments) == 1
    assert all(call[2] == 'before' for call in runner_harness.cleanup)


def test_scientific_failure_continues_without_retry(monkeypatch, tmp_path, runner_harness):
    """A valid failed objective is retained and followed by the next scheduled run."""
    from mha_exp_common.batch import run_batch

    def saved_states(*args, **kwargs):
        seed = runner_harness.environments[-1].state['environment_seed']
        run = SELECTED_SEEDS.index(seed)
        env_id, agent_id = f'exp_env2_cr_5_{run}', f'exp_agent2_cr_5_{run}'
        return {agent_id: {}, env_id: {env_id: {'environment_seed': seed}}}

    monkeypatch.setattr(runner, 'gather_states', saved_states)
    monkeypatch.setattr(runner, 'check_results', lambda *args, require_objective=True, **kwargs: (not require_objective, ['survival objective failed'] if require_objective else []))
    assert run_batch(
        experiment_id='2-5-CR', title='test', runs=[13, 14], exp_path=tmp_path / 'cohort',
        runner=lambda run, path, version: runner.run_experiment(run, path, environment_seed=SELECTED_SEEDS[run]),
        cleanup_before_run=False, stop_on_error=False,
    )
    assert [env.state['environment_seed'] for env in runner_harness.environments] == [1014, 1015]


def test_death_is_a_scientific_failure_not_an_integrity_failure():
    """The actual checker accepts reconciled death evidence while rejecting its objective."""
    states = runner.initial_states()
    states[LLREASONER].update(observation_count=1, observation_request_count=1)
    states[PERCEPTOR]['observation_count'] = 1
    states[KNOWLEDGE]['revision_count'] = 1
    states[HLREASONER].update(belief_update_count=1, terminal_reason='environment_terminal', survived=False)
    environment = runner.environment_initial_state()
    environment.update(observation_count=1, terminal=True, inventory={'health': 0}, closed=True)
    saved = {module_name(kind, 0): state for kind, state in states.items()}
    assert runner.check_results(saved, environment, require_objective=False) == (True, [])
    passed, errors = runner.check_results(saved, environment)
    assert not passed
    assert 'survival objective failed' in errors
