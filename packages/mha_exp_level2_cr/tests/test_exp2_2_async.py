"""Asynchronous lifecycle, deadline, and circular replay regression checks."""

from collections import deque
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from mhagenta import ActionStatus, Observation
from mhagenta.outboxes import KnowledgeOutbox, LearnerOutbox, LLOutbox, MemoryOutbox
from mhagenta.utils import State

from mha_exp_level2_cr.exp2_2.async_modules import AsyncReasoner, AsyncKnowledge, AsyncMemory, AsyncLearner
from mha_exp_level2_cr.exp2_2.policy import initial_frame_stack, load_policy_checkpoint
from mha_exp_level2_cr.exp2_2.replay import PrioritizedReplay
from mha_exp_level2_cr.exp2_2.runner import initial_states, _reward_trace_valid
from mha_exp_level2_cr.exp2_2.treatment import DQNWorkload
from test_exp2_2_dqn import frame, FakeState
from test_exp2_2_rainbow import evidence


def workload(seconds=6):
    return DQNWorkload(episode_action_limit=3, evaluation_seeds=(920000, 920001),
        duration_seconds=seconds + 330, synchronized_training=False,
        async_training_seconds=seconds, evaluation_interval_seconds=2)


def test_circular_overwrite_and_evicted_feedback_preserves_new_occupant():
    replay = PrioritizedReplay(np.random.default_rng(7), capacity=4)
    for ordinal in range(1, 14):
        replay.append({'replay_id': ordinal, 'state_stack': initial_frame_stack(frame()),
            'action': 0, 'reward': 1., 'next_frame': frame(), 'terminal': True}, boundary=True)
    assert sorted(item['replay_id'] for item in replay.items) == [10, 11, 12, 13]
    assert replay.evictions == 9
    before = replay.priorities.copy()
    replay.update_priorities([1] * 128, [99.] * 128, allow_evicted=True)
    np.testing.assert_equal(replay.priorities, before)
    assert replay.stale_feedback == 128
    replay.update_priorities([13] * 128, [.5] * 128, allow_evicted=True)
    assert replay.priorities[0] == pytest.approx(.500001)
    with pytest.raises(ValueError):
        replay.update_priorities([14] * 128, [1.] * 128, allow_evicted=True)


@pytest.mark.parametrize('closure_first', [True, False])
def test_stop_closure_order_flushes_tail_once(closure_first):
    w = workload()
    memory = AsyncMemory('memory', {})
    memory.on_init(seed=1, workload=w.dump())
    memory._learner_id = 'learner'
    state = FakeState(initial_states(w)['memory'])
    memory.on_observation_update(state, 'knowledge', [Observation(frame(1), value=-.001)],
        transition_ordinal=1, classification='warmup', cycle_id=None,
        state_stack=initial_frame_stack(frame()), action=0, terminal=False, boundary=False)
    close = lambda: memory.on_observation_update(state, 'knowledge', [], collection_closed=True, transition_ordinal=1)
    stop = lambda: memory.on_memory_request(state, 'learner', kind='stop')
    (close if closure_first else stop)()
    (stop if closure_first else close)()
    assert state['collection_closed'] and state['sampling_closed'] and state['closure_acknowledged']
    assert state['pending_transitions'] == 0 and state['experiences_finalized'] == 1
    assert memory._replay.items[0]['bootstrap_discount'] == .99
    assert sum(call[2].get('kind') == 'replay_closed' for call in state.outbox.calls) == 1


def test_reasoner_continues_beyond_old_transition_cap():
    w = workload(60)
    reasoner = AsyncReasoner('ll', {})
    reasoner.on_init(seed=1, workload=w.dump())
    reasoner._knowledge_id = 'knowledge'
    reasoner._actuator_id = 'actuator'
    reasoner._frame_stack = initial_frame_stack(frame())
    reasoner._previous_action = 0
    reasoner._episode_length = 1
    state = FakeState(initial_states(w)['ll_reasoner'], time=1)
    state.update(phase='training', training_started_at=0., training_deadline=60.,
                 transitions_emitted=10000, actions=10000, episodes_started=1)
    reasoner.on_observation(state, 'perceptor', Observation(frame(1)))
    assert state['transitions_emitted'] == 10001
    assert state['actions'] == 10001 and not state['collection_closed']
    assert any(call[0] == 'request_action' for call in state.outbox.calls)


def test_runner_ignores_periodic_results_directory(tmp_path, monkeypatch):
    """Sidecar evaluation directories must never be treated as agent states."""
    from mha_exp_level2_cr.exp2_2 import runner

    class Orchestrator:
        INFO = 20

        def __init__(self, **kwargs):
            pass

        def add_agent(self, **kwargs):
            pass

        def add_environment(self, **kwargs):
            pass

        def run(self, **kwargs):
            pass

    agent, env, _ = runner.runtime_identity(0)
    for name in (agent, env):
        folder = tmp_path / name / 'out'
        folder.mkdir(parents=True)
        (folder / (name + '.json')).write_text('{}')
    (tmp_path / 'periodic-evaluations').mkdir()
    monkeypatch.setattr(runner, 'Orchestrator', Orchestrator)
    monkeypatch.setattr(runner, 'TestEnvironment', lambda **kwargs: None)
    monkeypatch.setattr(runner, 'check_results', lambda *args, **kwargs: True)
    assert runner.run_experiment(0, tmp_path)


def test_reset_observation_after_deadline_is_accounted_without_an_action():
    """A reset already in flight may return after collection time expires."""
    w = workload()
    reasoner = AsyncReasoner('ll', {})
    reasoner.on_init(seed=1, workload=w.dump())
    reasoner._knowledge_id = 'knowledge'
    state = FakeState(initial_states(w)['ll_reasoner'], time=7)
    state.update(training_started_at=0., training_deadline=6.)
    reasoner.on_observation(state, 'perceptor', Observation(frame()))
    assert state['observations'] == state['cutoff_reset_observations'] == 1
    assert state['actions'] == state['stack_initializations'] == state['episodes_started'] == 0
    assert state['collection_closed']
    assert not any(call[0] == 'request_action' for call in state.outbox.calls)


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('cancel_last_batch', [False, True])
def test_typed_async_lifecycle_deadline_snapshots_freeze_and_evaluate(tmp_path, cancel_last_batch, masked):
    pytest.importorskip('torch')
    w = replace(workload(), action_masking=masked)
    clock = [0.]
    directory = SimpleNamespace(internal=SimpleNamespace(**{
        key: [SimpleNamespace(module_id=value)] for key, value in {
            'actuation': 'actuator', 'perception': 'perceptor', 'knowledge': 'knowledge',
            'memory': 'memory', 'learning': 'learner', 'll_reasoning': 'll_reasoner'}.items()}))
    behaviors = {'ll_reasoner': AsyncReasoner('ll_reasoner', {}),
        'knowledge': AsyncKnowledge('knowledge', {}), 'memory': AsyncMemory('memory', {}),
        'learner': AsyncLearner('learner', {})}
    boxes = dict(ll_reasoner=LLOutbox, knowledge=KnowledgeOutbox, memory=MemoryOutbox, learner=LearnerOutbox)
    states = {name: State(agent_id='agent', module_id=name, time_func=lambda: clock[0],
        directory=directory, outbox=boxes[name](), **initial_states(w)[name]) for name in behaviors}
    queue = deque()
    terminated = []

    def flush(name):
        out = states[name].outbox
        if out:
            queue.extend((name, receiver, dict(body)) for receiver, _, _, body in out)
        out.clear()
        term, reason = out.pop_term_request()
        if term:
            terminated.append(reason)

    for name, behavior in behaviors.items():
        behavior.log = lambda *_: None
        behavior.on_init(seed=4, workload=w.dump(), policy_path=str(tmp_path / 'policy.pt'))
        behavior.on_first(states[name])
        flush(name)
    snapshot = evidence()['before']
    episode_action = 0
    callbacks = 0
    while queue:
        callbacks += 1
        assert callbacks < 12000, 'Asynchronous callbacks did not terminate'
        sender, receiver, body = queue.popleft()
        if receiver == 'perceptor':
            behaviors['ll_reasoner'].on_observation(states['ll_reasoner'], receiver, Observation(frame(episode_action)),
                action_mask=[True] + [False] * 16)
            flush('ll_reasoner')
            continue
        if receiver == 'actuator':
            if body['action'] == 'reset':
                episode_action = 0
                snapshot = evidence()['before']
                status = {'applied_seed': body.get('requested_seed')}
            else:
                if masked:
                    assert body['action'] == 0
                clock[0] += .01
                before = deepcopy(snapshot)
                episode_action += 1
                if episode_action == 3:
                    snapshot['achievements']['collect_diamond'] = 1
                status = {'illegal_action': False, 'done': False, 'reward': 1.,
                    'achievements': dict(snapshot['achievements']),
                    'reward_evidence': {'before': before, 'after': deepcopy(snapshot)}}
            behaviors['ll_reasoner'].on_action_status(states['ll_reasoner'], receiver, ActionStatus(status))
            flush('ll_reasoner')
            continue
        behavior, state = behaviors[receiver], states[receiver]
        if receiver == 'knowledge':
            behavior.on_observed_beliefs(state, sender, **body)
        elif receiver == 'memory' and sender == 'knowledge':
            behavior.on_observation_update(state, sender, **body)
        elif receiver == 'memory':
            behavior.on_memory_request(state, sender, **body)
        elif receiver == 'learner' and sender == 'll_reasoner':
            behavior.on_task(state, sender, **body)
        elif receiver == 'learner':
            if cancel_last_batch and body.get('kind') == 'batch' and state['training_updates'] >= 3:
                clock[0] = max(clock[0], 6.01)
            behavior.on_memories(state, sender, **body)
        else:
            behavior.on_model(state, sender, **body)
        flush(receiver)
    ll, mem, lea = (states[k] for k in ('ll_reasoner', 'memory', 'learner'))
    assert len(terminated) == 1 and ll['phase'] == 'complete'
    assert ll['transitions_emitted'] > 512
    assert mem['experiences_finalized'] == mem['transitions_admitted'] == ll['actions']
    assert lea['training_updates'] != ll['update_eligible_transitions']
    assert mem['pending_transitions'] == 0 and not mem['pending_batch']
    assert mem['closure_acknowledged'] and lea['frozen']
    assert lea['training_updates'] == mem['priority_updates'] == lea['priority_acknowledgements']
    assert mem['batches_sent'] == lea['training_updates'] + lea['unused_batches']
    assert lea['unused_batches'] == mem['cancelled_batches']
    assert lea['post_freeze_update_attempts'] == 0
    assert lea['last_update_started'] < lea['training_deadline']
    assert ll['last_training_action_started'] < ll['training_deadline']
    assert _reward_trace_valid(ll.dump(), states['knowledge'].dump(), w)
    assert all(case['success'] for case in ll['evaluation_cases'])
    import torch
    _, metadata = load_policy_checkpoint(torch, tmp_path / 'policy.pt')
    assert metadata['workload']['training_transitions'] is None
    assert not metadata['algorithm']['synchronized']
    assert metadata['environment']['no_mobs'] is True
    assert metadata['workload']['action_masking'] is masked
    if masked:
        assert all(item['next_action_mask'] == [True] + [False] * 16
                   for item in behaviors['memory']._replay.items)
    assert list(tmp_path.glob('evaluation-*.pt'))
