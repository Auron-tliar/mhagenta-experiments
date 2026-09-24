"""Legal masking, asynchronous pacing, deadline wakeup and learning diagnostics."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from mhagenta import Observation
from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_2.modules import PacedReasoner, TestLearner as Learner
from mha_exp_level2_bw.exp2_2.policy import (
    TABLE_LEN, NUM_BLOCKS, goal_conditioned_observation, legal_action_mask,
    load_policy_checkpoint, build_q_network, save_policy_checkpoint,
)
from mha_exp_level2_bw.exp2_2.runner import initial_states
from test_exp2_2_dqn import FakeState, active_goal_record
from test_exp2_2_rainbow import small_protocol


def test_observation_mask_matches_physical_environment_actions():
    env = BlocksWorldEnv(table_len=TABLE_LEN, num_blocks=NUM_BLOCKS, symbolic=False)
    obs, _ = env.reset(seed=6)
    rng = np.random.default_rng(4)
    for _ in range(250):
        mask = legal_action_mask(obs)
        assert mask.any()
        for action in range(4):
            clone = deepcopy(env)
            clone.step(action)
            assert mask[action] == clone._last_legal
        # Relabeling the goal cannot change applicability at the same endpoint.
        conditioned = goal_conditioned_observation(obs, (4, 8))
        np.testing.assert_array_equal(mask, legal_action_mask(conditioned[:32]))
        obs, _, _, _, _ = env.step(int(rng.integers(4)))
    env.close()


def test_pacing_resumes_on_publication_and_deadline_without_lockstep():
    protocol = small_protocol(synchronized_training=False, min_updates_per_transition=.25,
                              action_masking=True, async_training_seconds=10)
    reasoner = PacedReasoner('reasoner', {})
    reasoner.on_init(seed=1, protocol=protocol)
    state = FakeState(initial_states(protocol)['ll_reasoner'], time=1)
    state.update(active_goal=active_goal_record(), training_transitions=8, actor_update=1)
    env = BlocksWorldEnv(table_len=TABLE_LEN, num_blocks=NUM_BLOCKS, symbolic=False)
    obs, _ = env.reset(seed=4)
    reasoner._request_policy_action(state, obs)
    assert reasoner._paced_observation is not None
    assert not state.outbox.calls
    state['actor_update'] = 10
    reasoner.on_model(state, 'learner', None, update=10, final=False)
    assert state.outbox.calls[-1][0] == 'request_action'
    assert not reasoner._waiting_for_training
    state['training_transitions'] = 44
    reasoner._request_policy_action(state, obs)
    assert reasoner._paced_observation is not None
    state.time = 10
    reasoner.on_model(state, 'learner', None, update=11, final=False, completion=False)
    assert state['collection_closed'] and state['phase'] == 'training_draining'
    assert reasoner._paced_observation is None
    env.close()


def test_learner_masks_targets_records_strength_saves_snapshot_and_wakes_actor(tmp_path):
    torch = pytest.importorskip('torch')
    protocol = small_protocol(synchronized_training=False, min_updates_per_transition=.25,
                              action_masking=True, evaluation_interval_seconds=1,
                              async_training_seconds=4, optimization_window_updates=1)
    learner = Learner('learner', {})
    learner.on_init(protocol=protocol, policy_path=str(tmp_path/'policy.pt'))
    learner._memory_id = 'memory'
    state = FakeState(initial_states(protocol)['learner'], time=0)
    learner.on_task(state, 'reasoner', {'kind': 'start', 'started_at': 0., 'deadline': 4.})
    env = BlocksWorldEnv(table_len=TABLE_LEN, num_blocks=NUM_BLOCKS, symbolic=False)
    obs, _ = env.reset(seed=4)
    obs[0] = 0
    obs[0, 0, :] = 1
    obs[1] = 0
    next_state = goal_conditioned_observation(obs, (4, 8))

    class Constant(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values))
        def forward(self, images):
            return self.values.repeat(len(images), 1)

    learner._online = Constant([1., 90., 100., 3.])
    learner._target = Constant([10., 20., 30., 40.])
    learner._optimizer = torch.optim.AdamW(learner._online.parameters(), lr=1e-4)
    entry = dict(state=next_state, action=3, next_state=next_state, reward=0.,
                 bootstrap_discount=.5, horizon=1, actor_update=0)
    state.time = 1.1
    learner.on_memories(state, 'memory', [Observation(entry)]*2, entry_ids=[0,1],
                        weights=[.25,1.], beta=.4, mean_replay_age=1.)
    assert not state['failure_reason']
    assert learner._feedback == {}  # Feedback was sent with the next batch request.
    row = state['optimization_windows'][0]
    assert row['mean_abs_td_error'] == pytest.approx(17.)
    assert row['unweighted_loss'] == pytest.approx(16.5)
    assert row['mean_loss'] == pytest.approx(10.3125)
    assert row['weight_mean'] == pytest.approx(.625) and row['gradient_norm'] > 0
    assert len(state['evaluation_snapshots']) == 1
    assert (tmp_path/'evaluation-000001.pt').is_file()
    state.time = 4.1
    learner.on_memories(state, 'memory', [Observation(entry)]*2)
    assert any(name == 'send_model' and args[1] is None for name,args,kw in state.outbox.calls)
    assert state['stopping']
    env.close()


def test_periodic_cpu_evaluation_records_masking_and_preserves_weights(tmp_path):
    torch = pytest.importorskip('torch')
    from mha_exp_level2_bw.exp2_2.evaluate_snapshot import evaluate
    protocol = small_protocol(synchronized_training=False, action_masking=True,
                              min_updates_per_transition=.25, evaluation_seeds=(2200,), max_episode_length=5)
    checkpoint = tmp_path/'snapshot.pt'
    model = build_q_network(torch)
    save_policy_checkpoint(torch, model, checkpoint, 3, training_protocol=protocol.record(), frozen=True)
    before = checkpoint.read_bytes()
    result = evaluate(checkpoint, tmp_path/'evaluation.json')
    assert checkpoint.read_bytes() == before
    assert result['episodes'][0]['illegal_actions'] == 0
    assert result['training_updates'] == 3 and not result['training_replay_connected']
    assert result['protocol']['config']['action_masking']


def test_runner_gathers_only_named_agent_and_environment(tmp_path, monkeypatch):
    """Use the real state reader so auxiliary evaluation folders cannot break it."""
    from mha_exp_level2_bw.exp2_2 import runner

    class Orchestrator:
        INFO = 20
        def __init__(self, **kwargs): pass
        def add_agent(self, **kwargs): pass
        def add_environment(self, **kwargs): pass
        def run(self, **kwargs): pass

    agent, env, _ = runner.runtime_resources(0)
    for name in (agent, env):
        directory = tmp_path/name/'out'
        directory.mkdir(parents=True)
        filename = name+'.learner_0.json' if name == agent else name+'.json'
        (directory/filename).write_text('{"device":"cuda:0"}')
    (tmp_path/'periodic-evaluations').mkdir()
    monkeypatch.setattr(runner, 'Orchestrator', Orchestrator)
    monkeypatch.setattr(runner, 'TestEnvironment', lambda **kwargs: None)
    monkeypatch.setattr(runner, 'cleanup_run_containers', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'cleanup_run_images', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'check_results_detailed', lambda *a, **k: (True, [], {}))
    monkeypatch.setattr(runner, 'build_run_summary', lambda **kwargs: {})
    monkeypatch.setattr(runner, 'write_run_summary', lambda *a: None)
    monkeypatch.setattr(runner, 'print_run_summary', lambda *a: None)
    assert runner.run_experiment(0, tmp_path, gpu_device=0)
