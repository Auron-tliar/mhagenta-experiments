"""Retain a proved wall-time cutoff without weakening native transition checks."""

from copy import deepcopy

import pytest

from mha_exp_common.utils import module_name
from mha_exp_level2_cr.exp2_5.contracts import K_ACTION, K_ATOMIC_ID, K_OWNER_ID, K_REQUESTER
from mha_exp_level2_cr.exp2_5.result_validation import time_limit_evidence
from mha_exp_level2_cr.exp2_5.runner import check_results
from mha_exp_level2_cr.exp2_5.runtime import initial_states


@pytest.fixture
def cutoff(tmp_path):
    """A complete native transition followed by one undelivered activity at timeout."""
    agent = {module_name(name, 0): values for name, values in initial_states().items()}
    ll, hl, goals, actuator = (agent[name] for name in
        ('llreasoner_0', 'hlreasoner_0', 'goalgraph_0', 'actuator_0'))
    status = {K_ACTION: 0, K_ATOMIC_ID: 1, K_OWNER_ID: 'primitive-1', K_REQUESTER: 'hl_primitive'}
    ll.update(status_count=1, observation_count=2, observation_request_count=2,
              received_goal_count=2, terminal_goal_count=2,
              trace=[{**status, 'status': status, 'input_observation_id': 1,
                      'result_observation_id': 2, 'policy_id': None}],
              activities=[{'kind': 'activity', 'activity': 'explore', 'goal_id': 'hl-2',
                           'actions': 0, 'max_actions': 8}])
    hl.update(dispatch_count=3, terminal_count=2, belief_update_count=2,
              primitive_decisions=[{K_OWNER_ID: 'primitive-1', 'action': 0}],
              active_goal={'extras': {'goal_id': 'hl-3', 'kind': 'activity', 'status': 'requested',
                                      'baseline_revision': 2}})
    goals.update(dispatch_count=3, terminal_count=2, active_goal_id='hl-3')
    actuator.update(request_count=1, status_count=1, statuses=[status])
    agent['perceptor_0']['observation_count'] = 2
    agent['knowledge_0']['revision_count'] = 2
    environment = {'failure': None, 'native_action_count': 1, 'observation_count': 2,
                   'status_history': [status], 'action_history': [status],
                   'inventory': {'health': 9, 'diamond': 0}, 'achievement_counts': {},
                   'closed': True, 'terminal': False}
    ids = ('exp_agent2_cr_5_41', 'exp_env2_cr_5_41')
    log = tmp_path / f'{ids[0]}.log'
    log.write_text('[2026-09-20 18:16:07,377|620.000000|620.000000|600.0][INFO]::'
                   f'[{ids[0]}][root]::Sending stop command (reason AGENT TIMEOUT CMD)\n')
    (tmp_path / f'{ids[1]}.log').write_text('Environment closed.\n')
    return agent, environment, tmp_path, ids


def test_timeout_is_operationally_valid_but_not_objective_success(cutoff):
    """Only the two proved cutoff differences are accepted, with no raw-state mutation."""
    agent, environment, root, ids = cutoff
    before = deepcopy((agent, environment))
    kwargs = dict(run_root=root, runtime_ids=ids)
    assert check_results(agent, environment, require_objective=False, allow_time_limit=False, **kwargs) == (
        False, ['unfinished goal reconciliation', 'goal counts disagree'])
    assert check_results(agent, environment, require_objective=False, **kwargs) == (True, [])
    passed, errors = check_results(agent, environment, **kwargs)
    assert not passed and 'survival objective failed' in errors
    assert (agent, environment) == before


@pytest.mark.parametrize('change', ['early', 'late', 'other-agent', 'other-reason', 'other-module', 'duplicate'])
def test_timeout_requires_exact_root_clock_evidence(cutoff, change):
    """Elapsed wall time alone, another agent, or another shutdown reason is insufficient."""
    agent, environment, root, ids = cutoff
    log = root / f'{ids[0]}.log'
    text = log.read_text()
    modified = {'early': text.replace('|600.0]', '|599.9]'),
                'late': text.replace('|600.0]', '|605.1]'),
                'other-agent': text.replace(ids[0], 'exp_agent2_cr_5_40'),
                'other-reason': text.replace('AGENT TIMEOUT CMD', 'USER COMMAND'),
                'other-module': text.replace('[root]', '[llreasoner_0]'),
                'duplicate': text + text}[change]
    log.write_text(modified)
    assert time_limit_evidence(agent, environment, root, ids) is None
    assert not check_results(agent, environment, run_root=root, runtime_ids=ids, require_objective=False)[0]


@pytest.mark.parametrize('module,key,value', [
    ('llreasoner_0', 'received_goal_count', 3),
    ('llreasoner_0', 'pending_action', {}),
    ('llreasoner_0', 'current_goal', {}),
    ('hlreasoner_0', 'dispatch_count', 4),
    ('hlreasoner_0', 'pending_terminal', {}),
    ('goalgraph_0', 'active_goal_id', 'hl-99'),
    ('goalgraph_0', 'dispatch_count', 2),
])
def test_other_goal_boundaries_are_not_accepted(cutoff, module, key, value):
    """Do not generalize a single observed cutoff into permission for unfinished work."""
    agent, environment, root, ids = cutoff
    agent[module][key] = value
    assert time_limit_evidence(agent, environment, root, ids) is None
    assert not check_results(agent, environment, run_root=root, runtime_ids=ids, require_objective=False)[0]


def test_native_mismatch_and_runtime_error_still_fail(cutoff):
    """A valid timeout marker cannot excuse trace corruption or runtime exceptions."""
    agent, environment, root, ids = cutoff
    environment['observation_count'] = 1
    valid, errors = check_results(agent, environment, run_root=root, runtime_ids=ids, require_objective=False)
    assert not valid and 'observation counts disagree' in errors
    environment['observation_count'] = 2
    with (root / f'{ids[0]}.log').open('a') as stream:
        stream.write('Caught exception: broken pipeline\n')
    valid, errors = check_results(agent, environment, run_root=root, runtime_ids=ids, require_objective=False)
    assert not valid and f'runtime errors in {ids[0]}' in errors
