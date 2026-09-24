"""Time-bounded CR collection and learning with ordered replay closure."""

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mhagenta import Belief, Observation
from mhagenta.states import KnowledgeState, LearnerState, LLState, MemoryState

from . import modules as base
from .policy import as_rgb_frame, initial_frame_stack, shift_frame_stack, save_policy_checkpoint
from .treatment import BETA_START, EPS_GREEDY_START, EPS_GREEDY_END, TARGET_SYNC_STEPS, TRAINING_START_THRESHOLD


class AsyncReasoner(base.TestLLReasoner):
    """Collect without update acknowledgements; evaluate only after final freeze."""

    def on_first(self, state: LLState) -> LLState:
        """Start collection and give the learner the same absolute deadline."""
        state = super().on_first(state)
        state['training_started_at'] = float(state.time)
        state['training_deadline'] = float(state.time) + self._workload.async_training_seconds
        learner = base._single_module_id(state.directory.internal.learning, 'learner')
        state.outbox.send_learner_task(learner, {'kind': 'start',
            'started_at': state['training_started_at'], 'deadline': state['training_deadline']})
        return state

    def _epsilon(self, decisions: int) -> float:
        progress = min(1.0, max(0.0, self._decision_progress))
        return EPS_GREEDY_START + progress * (EPS_GREEDY_END - EPS_GREEDY_START)

    def _close_collection(self, state: LLState) -> None:
        if state['collection_closed']:
            return
        if self._episode_length:
            self._record_outcome(state, True)
        self._clear_episode()
        state['collection_closed'] = True
        state['phase'] = 'drain'
        state['phase_timestamps']['drain_started'] = float(state.time)
        state['training_closed_at_transition'] = state['transitions_emitted']
        state['training_closed_at_elapsed_seconds'] = float(state.time)
        state.outbox.send_beliefs(self._knowledge_id, Observation(None), [],
            collection_closed=True, transition_ordinal=state['transitions_emitted'])

    def _request_policy_action(self, state: LLState, stack: np.ndarray) -> None:
        if state['phase'] == 'frozen_evaluation':
            super()._request_policy_action(state, stack)
            return
        if float(state.time) >= state['training_deadline']:
            self._close_collection(state)
            return
        self._decision_progress = (float(state.time) - state['training_started_at']) / self._workload.async_training_seconds
        action = self._select_action(stack, state['actions'])
        if float(state.time) >= state['training_deadline']:
            self._close_collection(state)
            return
        state['last_training_action_started'] = float(state.time)
        state['actions'] += 1
        state['policy_inferences'] += 1
        state['action_histogram'][action] += 1
        self._frame_stack = stack.copy()
        self._previous_action = action
        state.outbox.request_action(self._actuator_id, action=action)

    def _request_reset(self, state: LLState, seed: int | None = None) -> None:
        if state['phase'] != 'frozen_evaluation' and float(state.time) >= state['training_deadline']:
            self._close_collection(state)
            return
        super()._request_reset(state, seed)

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs: Any) -> LLState:
        """Admit the completed action, then continue or close at the deadline."""
        if state['phase'] == 'frozen_evaluation':
            return super().on_observation(state, sender, observation, **kwargs)
        if state['collection_closed']:
            state['invalid_inputs'] += 1
            state.outbox.terminate_agent('Observation after collection closure')
            return state
        current = as_rgb_frame(observation.content)
        if self._workload.action_masking:
            self._action_mask = base.validate_mask(kwargs.get('action_mask'))
        state['observations'] += 1
        if self._frame_stack is None:
            if float(state.time) >= state['training_deadline']:
                state['cutoff_reset_observations'] += 1
                self._close_collection(state)
                return state
            self._frame_stack = initial_frame_stack(current)
            state['stack_initializations'] += 1
            state['episodes_started'] += 1
            self._request_policy_action(state, self._frame_stack)
            return state
        if self._previous_action is None:
            raise ValueError('Observation without an outstanding action')
        next_stack = shift_frame_stack(self._frame_stack, current)
        state['stack_shifts'] += 1
        ordinal = state['transitions_emitted'] + 1
        deadline = float(state.time) >= state['training_deadline']
        boundary = (self._last_done or self._last_achieved or self._last_env_done
                    or self._episode_length >= self._workload.episode_action_limit or deadline)
        classification = base.WARMUP if ordinal <= TRAINING_START_THRESHOLD else base.UPDATE_ELIGIBLE
        if classification == base.UPDATE_ELIGIBLE:
            state['update_eligible_transitions'] += 1
            if state['phase'] == 'warmup':
                state['phase'] = 'training'
                state['phase_timestamps']['training_started'] = float(state.time)
        state['transitions_emitted'] = ordinal
        state.outbox.send_beliefs(self._knowledge_id, Observation(current),
            [Belief(base.BELIEF_ILLEGAL, self._last_illegal),
             Belief(base.BELIEF_GOAL_ACHIEVED, self._last_achieved)],
            state_stack=self._frame_stack.copy(), action=self._previous_action,
            next_action_mask=self._action_mask.tolist() if self._workload.action_masking else None,
            terminal=self._last_done or self._last_achieved, boundary=boundary,
            episode_id=state['episodes_started'], reward_evidence=self._last_evidence,
            transition_ordinal=ordinal, classification=classification, cycle_id=None)
        self._record_outcome(state, boundary)
        if boundary:
            self._clear_episode()
        if deadline:
            self._close_collection(state)
        elif boundary:
            self._request_reset(state)
        else:
            self._request_policy_action(state, next_stack)
        return state

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs: Any) -> LLState:
        """Install updates without controlling collection; start evaluation on final."""
        update = int(kwargs['update'])
        if model is not None:
            self._model = model.cpu().eval()
            state['models_installed'] += 1
            state['installed_update'] = update
        if kwargs.get('final'):
            if not state['collection_closed'] or self._model is None or state['installed_update'] != update:
                raise ValueError('Final model arrived without closed collection or matching weights')
            state['phase'] = 'frozen_evaluation'
            state['phase_timestamps']['frozen_evaluation_started'] = float(state.time)
            state['frozen_checkpoint_digest'] = kwargs['checkpoint_digest']
            self._clear_episode()
            self._request_reset(state, self._workload.evaluation_seeds[0])
        return state


class AsyncKnowledge(base.TestKnowledge):
    """Keep the closure marker ordered behind all evaluated observations."""

    def on_observed_beliefs(self, state: KnowledgeState, sender: str,
                           observation: Observation, beliefs: Sequence[Belief], **kwargs: Any) -> KnowledgeState:
        """Forward an ordered closure marker or apply the unchanged shaped reward."""
        if kwargs.get('collection_closed'):
            if state['collection_closed'] or kwargs['transition_ordinal'] != state['evaluated_transitions']:
                return self._fail(state, 'Out-of-order collection closure')
            state['collection_closed'] = True
            state.outbox.send_observations(self._memory_id, [], **kwargs)
            return state
        if state['collection_closed']:
            return self._fail(state, 'Transition after collection closure')
        return super().on_observed_beliefs(state, sender, observation, beliefs, **kwargs)


class AsyncMemory(base.TestMemory):
    """Admit continuously while serving one learner request at a time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_beta: float | None = None

    def _sync(self, state: MemoryState) -> None:
        state['buffer_size'] = len(self._replay.items)
        state['experiences_finalized'] = self._replay.finalized
        state['pending_transitions'] = len(self._replay.pending)
        state['horizon_counts'] = list(self._replay.horizons)
        state['evictions'] = self._replay.evictions
        state['stale_priority_feedback'] = self._replay.stale_feedback
        state['pending_batch'] = self._pending_cycle is not None
        state['oldest_replay_id'] = self._replay.finalized - len(self._replay.items) + 1
        state['newest_replay_id'] = self._replay.finalized

    def _serve(self, state: MemoryState) -> None:
        self._sync(state)
        if state['collection_closed']:
            self._request_beta = None
            if self._pending_cycle is None and not state['closure_acknowledged']:
                state['sampling_closed'] = True
                state['closure_acknowledged'] = True
                state.outbox.send_memories(self._learner_id, [], kind='replay_closed')
            return
        if (state['sampling_closed'] or self._request_beta is None or self._pending_cycle is not None
                or state['transitions_admitted'] <= TRAINING_START_THRESHOLD):
            return
        cycle = state['next_cycle_id']
        batch, self._sample_ids, weights = self._replay.sample(cycle, 1, beta=self._request_beta)
        self._request_beta = None
        self._pending_cycle = cycle
        state['batches_sent'] += 1
        state['next_cycle_id'] += 1
        state.outbox.send_memories(self._learner_id,
            [Observation(item, observation_type='dqn_n_step') for item in batch],
            kind='batch', cycle_id=cycle, replay_ids=self._sample_ids, weights=weights)
        self._sync(state)

    def on_observation_update(self, state: MemoryState, sender: str,
                              observations: Sequence[Observation], **kwargs: Any) -> MemoryState:
        """Admit transitions even while the learner owns a sampled batch."""
        ordinal = kwargs.get('transition_ordinal')
        if kwargs.get('collection_closed'):
            if observations or state['collection_closed'] or ordinal != state['transitions_admitted']:
                return self._fail(state, 'Invalid replay closure')
            self._replay.flush()
            state['collection_closed'] = True
            self._serve(state)
            return state
        classification = base.WARMUP if type(ordinal) is int and ordinal <= TRAINING_START_THRESHOLD else base.UPDATE_ELIGIBLE
        if (state['collection_closed'] or len(observations) != 1 or type(ordinal) is not int
                or ordinal != state['transitions_admitted'] + 1 or kwargs.get('classification') != classification
                or kwargs.get('cycle_id') is not None):
            return self._fail(state, 'Invalid asynchronous admission')
        item = base.make_replay_transition(observations[0], kwargs.get('state_stack'),
            kwargs.get('action'), bool(kwargs.get('terminal', False))).content
        item['replay_id'] = ordinal
        if self._workload.action_masking:
            item['next_action_mask'] = base.validate_mask(kwargs.get('next_action_mask')).tolist()
        boundary = kwargs.get('boundary')
        if type(boundary) is not bool or (item['terminal'] and not boundary):
            return self._fail(state, 'Invalid episode boundary')
        self._replay.append(item, boundary=boundary)
        state['transitions_admitted'] += 1
        state[f'{classification}_transitions'] += 1
        self._serve(state)
        return state

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs: Any) -> MemoryState:
        """Serve a request, apply delayed feedback, or cancel the final unused batch."""
        if sender != self._learner_id:
            return self._fail(state, 'Unknown replay requester')
        kind = kwargs.get('kind')
        if kind == 'batch_request':
            if not state['closure_acknowledged']:
                if self._pending_cycle is not None or self._request_beta is not None:
                    return self._fail(state, 'Duplicate batch request')
                self._request_beta = float(kwargs['beta'])
        elif kind == 'priority_update':
            if (self._pending_cycle is None or kwargs.get('cycle_id') != self._pending_cycle
                    or kwargs.get('replay_ids') != self._sample_ids):
                return self._fail(state, 'Unexpected priority feedback')
            self._replay.update_priorities(self._sample_ids, kwargs['td_errors'], allow_evicted=True)
            cycle = self._pending_cycle
            self._pending_cycle = None
            self._sample_ids = []
            state['priority_updates'] += 1
            state.outbox.send_memories(self._learner_id, [], kind='priority_ack', cycle_id=cycle)
        elif kind == 'stop':
            if self._pending_cycle is not None:
                if kwargs.get('cancel_cycle') != self._pending_cycle:
                    return self._fail(state, 'Stop did not identify the outstanding batch')
                self._pending_cycle = None
                self._sample_ids = []
                state['cancelled_batches'] += 1
            state['sampling_closed'] = True
            self._request_beta = None
        else:
            return self._fail(state, 'Unknown asynchronous replay request')
        self._serve(state)
        return state


class AsyncLearner(base.TestLearner):
    """Learn independently until deadline; freeze only after replay closure."""

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs: Any) -> LearnerState:
        """Initialize the shared deadline and request the first replay batch."""
        if task.get('kind') != 'start' or state['training_deadline'] is not None:
            return self._fail(state, 'Unexpected learner task')
        state['training_started_at'] = float(task['started_at'])
        state['training_deadline'] = float(task['deadline'])
        state['next_evaluation_snapshot'] = self._workload.evaluation_interval_seconds
        self._request_batch(state)
        return state

    def _request_batch(self, state: LearnerState) -> None:
        if float(state.time) >= state['training_deadline']:
            state.outbox.request_memories(self._memory_id, kind='stop')
        else:
            progress = min(1.0, max(0.0, (float(state.time) - state['training_started_at']) / self._workload.async_training_seconds))
            state.outbox.request_memories(self._memory_id, kind='batch_request',
                beta=BETA_START + (1 - BETA_START) * progress)

    def _publish(self, state: LearnerState, *, final: bool) -> None:
        model = deepcopy(self._online).cpu().eval()
        state['models_published'] += 1
        state.outbox.send_model(self._reasoner_id, model, update=state['training_updates'],
            final=final, checkpoint_digest=state['checkpoint_digest'])

    def _complete_update(self, state: LearnerState, cycle_id: int) -> LearnerState:
        elapsed = float(state.time) - state['training_started_at']
        if state['training_updates'] == 1 or state['training_updates'] % TARGET_SYNC_STEPS == 0:
            self._publish(state, final=False)
        interval = state['next_evaluation_snapshot']
        if elapsed >= interval and interval < self._workload.async_training_seconds:
            path = Path(self._policy_path).with_name(f'evaluation-{int(interval):06d}.pt')
            save_policy_checkpoint(self._torch, self._online, path, state['training_updates'], self._workload)
            state['evaluation_snapshots'].append({'filename': path.name, 'elapsed_seconds': elapsed,
                                                'training_updates': state['training_updates']})
            state['next_evaluation_snapshot'] += self._workload.evaluation_interval_seconds
        self._request_batch(state)
        return state

    def on_memories(self, state: LearnerState, sender: str,
                    memories: Sequence[Belief | Observation], **kwargs: Any) -> LearnerState:
        """Train before deadline, discard late batches, and freeze on replay closure."""
        kind = kwargs.get('kind')
        if kind == 'replay_closed':
            if memories or self._pending_cycle is not None or state['frozen'] or state['training_updates'] == 0:
                return self._fail(state, 'Invalid replay closure or no learning updates')
            current = state['optimization_current']
            if current is not None:
                state['optimization_windows'].append({
                    'update_start': current['update_start'], 'update_end': state['training_updates'],
                    'elapsed_start': current['elapsed_start'], 'elapsed_end': float(state.time),
                    'mean_loss': float(np.mean(current['losses'])),
                    'mean_absolute_td_error': float(np.mean(current['mean_absolute_td_errors']))})
                state['optimization_current'] = None
            path = save_policy_checkpoint(self._torch, self._online, self._policy_path,
                                           state['training_updates'], self._workload)
            state['checkpoint_digest'] = hashlib.sha256(path.read_bytes()).hexdigest()
            state['model_saved'] = True
            state['model_artifact'] = base.POLICY_FILENAME
            state['saved_training_steps'] = state['training_updates']
            state['frozen'] = True
            state['freeze_elapsed'] = float(state.time)
            self._online.eval()
            self._publish(state, final=True)
            return state
        if kind == 'batch' and float(state.time) >= state['training_deadline']:
            if state['frozen'] or self._pending_cycle is not None:
                return self._fail(state, 'Batch received in invalid learner state')
            state['unused_batches'] += 1
            state.outbox.request_memories(self._memory_id, kind='stop', cancel_cycle=kwargs['cycle_id'])
            return state
        if kind == 'batch':
            state['last_update_started'] = float(state.time)
        return super().on_memories(state, sender, memories, **kwargs)
