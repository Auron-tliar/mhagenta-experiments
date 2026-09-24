"""Native-size six-module integration of an explicitly qualified policy family."""

from concurrent.futures import Future
from copy import deepcopy
import json
import shutil
from types import SimpleNamespace

import pytest

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5 import family, matched
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_6_direct import results, runner, runtime
from test_exp2_6_direct import exercise

FAMILY = 'structured-dqfd-family-20260920-v1'


def test_family_rejects_changed_parent_lineage(tmp_path, monkeypatch):
    """A complete family cannot silently claim a different parent for either child."""
    source = family.artifact_root() / FAMILY
    if not source.exists():
        pytest.skip('The qualified family is not present.')
    target = tmp_path / FAMILY
    shutil.copytree(source, target)
    manifest = target / 'family.json'
    data = json.loads(manifest.read_text())
    data['policies']['7x12']['parent_sha256'] = '0' * 64
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(family, 'artifact_root', lambda: tmp_path)
    with pytest.raises(ValueError, match='lineage'):
        family.resolve(FAMILY, table_len=7, num_blocks=12)


def test_historical_recording_allowance_does_not_relax_new_exports(tmp_path, monkeypatch):
    """Only the known old recording path is optional; arbitrary and new-family extras fail."""
    config = {'identity': 'f' * 64, 'matched_2_4': {'source_run': 0}}
    result = {'operationally_valid': True, 'failures': []}
    saved = tmp_path / 'result.json'
    saved.write_text(json.dumps(result))
    (tmp_path / 'artifact-validation.json').write_text('{}')
    monkeypatch.setattr(results, 'run_evidence', lambda directory: (config, {}, {}, [saved], tmp_path))
    monkeypatch.setattr(runner, 'check_results', lambda *args, **kwargs: deepcopy(result))
    recording = tmp_path / ('exp_direct_env_' + 'f' * 12) / 'out' / '0000.mp4'
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b'retained video fixture')
    assert results.validate_run(tmp_path, compact=True)['operationally_valid']
    config['policy_family'] = FAMILY
    assert 'unexpected-compact-files' in results.validate_run(tmp_path, compact=True)['failures']
    config.pop('policy_family')
    (tmp_path / 'unexpected.txt').write_text('not a retained recording')
    assert 'unexpected-compact-files' in results.validate_run(tmp_path, compact=True)['failures']


@pytest.mark.parametrize('columns,blocks', [(4, 6), (5, 8), (7, 12)])
def test_qualified_family_through_six_native_modules(columns, blocks, tmp_path, monkeypatch):
    """Use a development-only goal to check routing, identity and saved-action replay."""
    if not (family.artifact_root() / FAMILY / 'family.json').exists():
        pytest.skip('All three family policies must qualify before this integration check.')
    dimensions = {'table_len': columns, 'num_blocks': blocks}
    environment = BlocksWorldEnv(**dimensions, symbolic=False)
    try:
        observation, _ = environment.reset(seed=9_000_123)
        facts = sorted(ground_observation(observation, **dimensions).facts)
        spec = next(item for item in enumerate_transfer_targets(facts) if item.destination_support.startswith('b'))
    finally:
        environment.close()
    goal = {'top': spec.block, 'bottom': spec.destination_support}
    task = {'id': 'development-family-integration', 'mode': 'transfer', 'seed': 9_000_123,
            'goal': goal, 'case_index': 0, 'probe': None, 'initial_facts': facts,
            'plan': [], 'reset_actions': [], 'difficulty': 0}
    record = {'source_run': 0, 'source_treatment': {**dimensions, 'seed': task['seed'], 'goal': goal}}
    monkeypatch.setattr(matched, 'matched_task', lambda run: (deepcopy(task), deepcopy(record)))

    class InlinePool:
        """Resolve the fixed development plan without replacing any runtime message edges."""

        def __init__(self, **kwargs):
            pass

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(runtime, 'ThreadPoolExecutor', InlinePool)
    monkeypatch.setattr(runtime, 'production_planner', lambda **kwargs: SimpleNamespace(
        solve=lambda *args: SimpleNamespace(actions=[spec.as_dict()], accepted=True, elapsed_seconds=0.01, engine='fixture')))
    config = runner.prepare(tmp_path, tmp_path, tmp_path / 'prepared', 0, 'matched', 'cpu', 0, True,
                            profile_override=runner.Profile(0, 1, 0, 0, 0, 1, 600, action_cap=None),
                            frozen_policy='transfer', matched_2_4=True, policy_family=FAMILY)
    states, env, config = exercise(tmp_path, True, production=True, frozen_policy='transfer', config_override=config)
    assert len(states) == 6 and env['resets'] == 1 and env['closed']
    assert states['hlreasoner']['results'][0]['success']
    assert not states['llreasoner']['model_installs']
    result = runner.check_results(states, env, config, tmp_path)
    assert result['operationally_valid'], result
    changed = deepcopy(config)
    changed['reference']['transfer_artifact']['family_sha256'] = '0' * 64
    assert 'matched-task-or-artifact-identity' in runner.check_results(states, env, changed, tmp_path)['failures']
