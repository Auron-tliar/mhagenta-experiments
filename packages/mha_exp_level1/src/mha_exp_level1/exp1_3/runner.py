#########################################################
# Checking RabbitMQ-based internal module communication #
#########################################################
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from logging import WARNING
from pathlib import Path
from typing import Any

import mha_exp_common
import numpy as np
from mha_exp_common.batch import normalize_runs
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import *
from mha_exp_common.utils import Seeder, agent_name, gather_states, module_name
from mhagenta import ActionStatus, Belief, Goal, Observation, Orchestrator, State
from mhagenta.bases import *
from mhagenta.states import *
from mhagenta.utils.common.classes import ICard
from numpy import random

DURATION = 10
N_MODULES = 5
# Ordered fan_out calls in each step method. Historical records have no route
# label, so reconstruction depends on this execution order being preserved.
SEND_ROUTES = {
    ACTUATOR: ((LLREASONER, 'send_status'),),
    PERCEPTOR: ((LLREASONER, 'send_observation'),),
    LLREASONER: (
        (ACTUATOR, 'request_action'),
        (PERCEPTOR, 'request_observation'),
        (KNOWLEDGE, 'send_beliefs'),
        (GOALGRAPH, 'send_goal_update'),
        (GOALGRAPH, 'request_goals'),
        (LEARNER, 'send_learner_task'),
        (LEARNER, 'request_model'),
    ),
    KNOWLEDGE: (
        (HLREASONER, 'send_beliefs'),
        (MEMORY, 'send_observations'),
        (MEMORY, 'send_belief_memories'),
    ),
    HLREASONER: (
        (KNOWLEDGE, 'request_beliefs'),
        (KNOWLEDGE, 'send_beliefs'),
        (GOALGRAPH, 'send_goals'),
        (ACTUATOR, 'request_action'),
        (LEARNER, 'send_learner_task'),
        (LEARNER, 'request_model'),
    ),
    GOALGRAPH: ((HLREASONER, 'send_goals'), (LLREASONER, 'send_goals')),
    MEMORY: ((LEARNER, 'send_memories'),),
    LEARNER: (
        (MEMORY, 'request_memories'),
        (LLREASONER, 'send_model'),
        (HLREASONER, 'send_model'),
    ),
}
EXPECTED_EDGES = {
    (source, target)
    for source, routes in SEND_ROUTES.items()
    for target, _ in routes
}
EXPECTED_CHANNELS = {
    (module_name(source, i), module_name(target, j), route)
    for source, routes in SEND_ROUTES.items()
    for target, route in routes
    for i in range(N_MODULES)
    for j in range(N_MODULES)
}


def fan_out(
        rng: random.Generator,
        recipients: list[ICard],
        send_func: Callable[..., None],
        payload_wrapper: Callable[[str], dict[str, Any]],
        log: Any
) -> None:
    """Send one timestamped payload to every recipient and record each send."""

    for recipient in recipients:
        payload = str(rng.integers(0, 2**32).item())
        kwargs = payload_wrapper(payload)
        sent_ns = time.time_ns()
        kwargs['sent_ns'] = sent_ns
        send_func(recipient.module_id, **kwargs)
        log.append((recipient.module_id, payload, sent_ns))


def record_received(state: State, sender: str, payload: Any, kwargs: dict[str, Any]) -> State:
    """Record a received payload and its one-way wall-clock latency inputs."""

    state['received'].append((sender, payload, kwargs['sent_ns'], time.time_ns()))
    return state


# Outgoing: LLReasoner
class TestActuator(ActuatorBase):
    """Exercise actuator-to-reasoner messages and incoming action requests."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: ActuatorState) -> ActuatorState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state
        fan_out(
            self._rng,
            state.directory.internal.ll_reasoning,
            state.outbox.send_status,
            lambda p: {'status': ActionStatus(p)},
            state['sent']
        )
        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        action = kwargs.get('action', None)

        if action is None:
            self.log(WARNING, f'Received an empty action request from {sender}: {kwargs}!')

        return record_received(state, sender, action, kwargs)


# Outgoing: LLReasoner
class TestPerceptor(PerceptorBase):
    """Exercise perceptor-to-reasoner messages and observation requests."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: PerceptorState) -> PerceptorState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state
        fan_out(
            self._rng,
            state.directory.internal.ll_reasoning,
            state.outbox.send_observation,
            lambda p: {'observation': Observation(p)},
            state['sent']
        )
        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs) -> PerceptorState:
        request = kwargs.get('request', None)

        if request is None:
            self.log(WARNING, f'Received an empty observation request from {sender}: {kwargs}!')

        return record_received(state, sender, request, kwargs)


class TestLLReasoner(LLReasonerBase):
    """Exercise all supported low-level reasoner communication edges."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: LLState) -> LLState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state

        fan_out(
            self._rng,
            state.directory.internal.actuation,
            state.outbox.request_action,
            lambda p: {'action': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.perception,
            state.outbox.request_observation,
            lambda p: {'request': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.knowledge,
            state.outbox.send_beliefs,
            lambda p: {'observation': Observation(p), 'beliefs': list()},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.goals,
            state.outbox.send_goal_update,
            lambda p: {'goals': [Goal(list(), payload=p)]},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.goals,
            state.outbox.request_goals,
            lambda p: {'request': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.learning,
            state.outbox.send_learner_task,
            lambda p: {'task': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.learning,
            state.outbox.request_model,
            lambda p: {'request': p},
            state['sent']
        )

        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs) -> LLState:
        return record_received(state, sender, action_status.status, kwargs)

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs) -> LLState:
        return record_received(state, sender, observation.content, kwargs)

    def on_goal_update(self, state: LLState, sender: str, goals: list[Goal], **kwargs) -> LLState:
        if goals[0].extras is None:
            self.log(WARNING, f'Received goals collection from {sender} with empty extras, expected the payload!')
            payload = None
        else:
            payload = goals[0].extras['payload']
        return record_received(state, sender, payload, kwargs)

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs) -> LLState:
        return record_received(state, sender, model, kwargs)


class TestKnowledge(KnowledgeBase):
    """Exercise knowledge communication with reasoning and memory modules."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: KnowledgeState) -> KnowledgeState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state

        fan_out(
            self._rng,
            state.directory.internal.hl_reasoning,
            state.outbox.send_beliefs,
            lambda p: {'beliefs': [Belief('payload', p)]},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.memory,
            state.outbox.send_observations,
            lambda p: {'observations': [Observation(p)]},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.memory,
            state.outbox.send_belief_memories,
            lambda p: {'beliefs': [Belief('payload', p)]},
            state['sent']
        )

        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation, beliefs: Sequence[Belief], **kwargs) -> KnowledgeState:
        return record_received(state, sender, observation.content, kwargs)

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs) -> KnowledgeState:
        request = kwargs.get('request', None)

        if request is None:
            self.log(WARNING, f'Received an empty belief request from {sender}: {kwargs}!')

        return record_received(state, sender, request, kwargs)

    def on_belief_update(self, state: KnowledgeState, sender: str, beliefs: Sequence[Belief], **kwargs) -> KnowledgeState:
        return record_received(state, sender, list(beliefs)[0].arguments, kwargs)


class TestHLReasoner(HLReasonerBase):
    """Exercise all supported high-level reasoner communication edges."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: HLState) -> HLState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state

        fan_out(
            self._rng,
            state.directory.internal.knowledge,
            state.outbox.request_beliefs,
            lambda p: {'request': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.knowledge,
            state.outbox.send_beliefs,
            lambda p: {'beliefs': [Belief('payload', p)]},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.goals,
            state.outbox.send_goals,
            lambda p: {'goals': [Goal(list(), payload=p)]},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.actuation,
            state.outbox.request_action,
            lambda p: {'action': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.learning,
            state.outbox.send_learner_task,
            lambda p: {'task': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.learning,
            state.outbox.request_model,
            lambda p: {'request': p},
            state['sent']
        )

        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_belief_update(self, state: HLState, sender: str, beliefs: Sequence[Belief], **kwargs) -> HLState:
        return record_received(state, sender, beliefs[0].arguments, kwargs)

    def on_goal_update(self, state: HLState, sender: str, goals: Sequence[Goal], **kwargs) -> HLState:
        if goals[0].extras is None:
            self.log(WARNING, f'Received goals collection from {sender} with empty extras, expected the payload!')
            payload = None
        else:
            payload = goals[0].extras['payload']
        return record_received(state, sender, payload, kwargs)

    def on_model(self, state: HLState, sender: str, model: Any, **kwargs) -> HLState:
        return record_received(state, sender, model, kwargs)


class TestGoalGraph(GoalGraphBase):
    """Exercise goal updates and requests across both reasoning levels."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: GoalGraphState) -> GoalGraphState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state

        fan_out(
            self._rng,
            state.directory.internal.hl_reasoning,
            state.outbox.send_goals,
            lambda p: {'goals': [Goal(list(), payload=p)]},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.ll_reasoning,
            state.outbox.send_goals,
            lambda p: {'goals': [Goal(list(), payload=p)]},
            state['sent']
        )

        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Sequence[Goal], **kwargs) -> GoalGraphState:
        goals = list(goals)
        if goals[0].extras is None:
            self.log(WARNING, f'Received goals collection from {sender} with empty extras, expected the payload!')
            payload = None
        else:
            payload = goals[0].extras['payload']
        return record_received(state, sender, payload, kwargs)

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs) -> GoalGraphState:
        request = kwargs.get('request', None)

        if request is None:
            self.log(WARNING, f'Received an empty goal request from {sender}: {kwargs}!')

        return record_received(state, sender, request, kwargs)


class TestMemory(MemoryBase):
    """Exercise observation, belief, request, and memory deliveries."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: MemoryState) -> MemoryState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state

        fan_out(
            self._rng,
            state.directory.internal.learning,
            state.outbox.send_memories,
            lambda p: {'memories': [Observation(p)]},
            state['sent']
        )

        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_observation_update(self, state: MemoryState, sender: str, observations: Sequence[Observation], **kwargs) -> MemoryState:
        return record_received(state, sender, observations[0].content, kwargs)

    def on_belief_update(self, state: MemoryState, sender: str, beliefs: Sequence[Belief], **kwargs) -> MemoryState:
        return record_received(state, sender, beliefs[0].arguments, kwargs)

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs) -> MemoryState:
        request = kwargs.get('request', None)

        if request is None:
            self.log(WARNING, f'Received an empty memory request from {sender}: {kwargs}!')

        return record_received(state, sender, request, kwargs)


class TestLearner(LearnerBase):
    """Exercise learner task, memory, and model communication edges."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            self._rng = random.default_rng(kwargs['seed'])

    def step(self, state: LearnerState) -> LearnerState:
        if state.time < self._next_msg_ts or DURATION - state.time < 2:
            return state

        fan_out(
            self._rng,
            state.directory.internal.memory,
            state.outbox.request_memories,
            lambda p: {'request': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.ll_reasoning,
            state.outbox.send_model,
            lambda p: {'model': p},
            state['sent']
        )
        fan_out(
            self._rng,
            state.directory.internal.hl_reasoning,
            state.outbox.send_model,
            lambda p: {'model': p},
            state['sent']
        )

        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_memories(self, state: LearnerState, sender: str, memories: Sequence[Belief | Observation], **kwargs) -> LearnerState:
        obs = memories[0]
        if not isinstance(obs, Observation):
            self.log(WARNING, f'The first memory element is not an observation: {obs}: {type(obs)}!')
            return state
        return record_received(state, sender, obs.content, kwargs)

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs) -> LearnerState:
        return record_received(state, sender, task, kwargs)

    def on_model_request(self, state: LearnerState, sender: str, **kwargs) -> LearnerState:
        request = kwargs.get('request', None)

        if request is None:
            self.log(WARNING, f'Received an empty memory request from {sender}: {kwargs}!')

        return record_received(state, sender, request, kwargs)


def _runtime_version(mha_version: str) -> str:
    runtime_version = DEFAULT_MHAGENTA_VERSION if mha_version in {'', 'latest'} else mha_version
    if runtime_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f'Experiment 1-3 requires MHAgentA {DEFAULT_MHAGENTA_VERSION}.')
    return runtime_version


def _local_mhagenta_root() -> Path:
    root = Path(__file__).resolve().parents[6] / 'mhagenta'
    project = root / 'pyproject.toml'
    if not project.is_file() or f'version = "{DEFAULT_MHAGENTA_VERSION}"' not in project.read_text(encoding='utf-8'):
        raise RuntimeError(f'Expected local MHAgentA {DEFAULT_MHAGENTA_VERSION} at {root}.')
    return root


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {'count': 0, 'p50_ms': None, 'p95_ms': None, 'p99_ms': None, 'max_ms': None}
    return {
        'count': len(values),
        'p50_ms': float(np.percentile(values, 50)),
        'p95_ms': float(np.percentile(values, 95)),
        'p99_ms': float(np.percentile(values, 99)),
        'max_ms': float(max(values)),
    }


def _module_type(module_id: str) -> str:
    return module_id.rsplit('_', 1)[0]


def _reconstruct_channels(
        states: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[Any, ...], tuple[str, str, str]], dict[str, int]]:
    """Attribute historical sends to routes using complete, ordered send waves.

    Match receipts by sender, recipient, payload and timestamp afterwards. This
    reconstructs the sending route; it cannot identify the receiving callback
    independently because the saved schema does not record that information.
    """

    message_channels: dict[tuple[Any, ...], tuple[str, str, str]] = {}
    waves: dict[str, int] = {}
    for sender, state in states.items():
        sequence = [
            (module_name(target, index), route)
            for target, route in SEND_ROUTES[_module_type(sender)]
            for index in range(N_MODULES)
        ]
        sends = state['sent']
        if not sends or len(sends) % len(sequence):
            raise ValueError(f'{sender}: expected nonempty, complete waves of {len(sequence)} sends.')
        waves[sender] = len(sends) // len(sequence)
        previous_ns = None
        for index, (recipient, payload, sent_ns) in enumerate(sends):
            expected_recipient, route = sequence[index % len(sequence)]
            if recipient != expected_recipient:
                raise ValueError(f'{sender}: send {index} targets {recipient}, expected {expected_recipient}.')
            if previous_ns is not None and sent_ns < previous_ns:
                raise ValueError(f'{sender}: send timestamps are out of order at {index}.')
            previous_ns = sent_ns
            key = (sender, recipient, payload, sent_ns)
            if key in message_channels:
                raise ValueError(f'{sender}: duplicate message identity prevents unambiguous route attribution.')
            message_channels[key] = (sender, recipient, route)
    return message_channels, waves


def _communication_metrics(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Match exact deliveries and measure instance-pair and reconstructed-route coverage."""

    message_channels, waves = _reconstruct_channels(states)
    sent: Counter[tuple[Any, ...]] = Counter()
    received: Counter[tuple[Any, ...]] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    channel_counts = Counter(message_channels.values())
    received_channels: Counter[tuple[str, str, str]] = Counter()
    latencies: list[float] = []
    for sender, state in states.items():
        for recipient, payload, sent_ns in state['sent']:
            sent[(sender, recipient, payload, sent_ns)] += 1
            edge_counts[(_module_type(sender), _module_type(recipient))] += 1
    for recipient, state in states.items():
        for sender, payload, sent_ns, received_ns in state['received']:
            key = (sender, recipient, payload, sent_ns)
            received[key] += 1
            if key in message_channels:
                received_channels[message_channels[key]] += 1
            latencies.append((received_ns - sent_ns) / 1_000_000)
    expected_pairs = {(sender, recipient) for sender, recipient, _ in EXPECTED_CHANNELS}
    return {
        'sent': sum(sent.values()),
        'received': sum(received.values()),
        'missing': sum((sent - received).values()),
        'unexpected': sum((received - sent).values()),
        'edges': edge_counts,
        'channels': channel_counts,
        'coverage': {
            'expected_module_pairs': len(expected_pairs),
            'sent_module_pairs': len({key[:2] for key in sent}),
            'received_module_pairs': len({key[:2] for key in received}),
            'expected_channels': len(EXPECTED_CHANNELS),
            'sent_channels': len(channel_counts),
            'received_channels': len(received_channels),
            'missing_sent_channels': len(EXPECTED_CHANNELS - channel_counts.keys()),
            'missing_received_channels': len(EXPECTED_CHANNELS - received_channels.keys()),
            'unexpected_sent_channels': len(channel_counts.keys() - EXPECTED_CHANNELS),
            'min_received_per_channel': min(received_channels[channel] for channel in EXPECTED_CHANNELS),
            'max_received_per_channel': max(received_channels.values(), default=0),
        },
        'waves_per_module': waves,
        'latencies_ms': latencies,
    }


def check_results(states: dict[str, dict[str, Any]]) -> bool:
    """Validate complete send waves, all reconstructed channels and exact deliveries."""

    module_ids = {
        module_name(module, index)
        for module in MODULES
        for index in range(N_MODULES)
    }
    if set(states) != module_ids:
        print(f'ERROR: expected modules {sorted(module_ids)}, received {sorted(states)}')
        return False
    try:
        metrics = _communication_metrics(states)
    except (KeyError, TypeError, ValueError) as error:
        print(f'ERROR: invalid or old communication state schema: {error}')
        return False
    success = (
        metrics['sent'] > 0
        and metrics['missing'] == 0
        and metrics['unexpected'] == 0
        and set(metrics['edges']) == EXPECTED_EDGES
        and metrics['coverage']['missing_sent_channels'] == 0
        and metrics['coverage']['missing_received_channels'] == 0
        and metrics['coverage']['unexpected_sent_channels'] == 0
        and bool(metrics['latencies_ms'])
        and all(math.isfinite(value) and value >= 0 for value in metrics['latencies_ms'])
    )
    if success:
        print(
            f'Exactly matched {metrics["received"]} internal message deliveries across '
            f'{metrics["coverage"]["received_module_pairs"]} module pairs and '
            f'{metrics["coverage"]["received_channels"]} reconstructed channels.'
        )
    else:
        print(
            f'ERROR: sent={metrics["sent"]}, received={metrics["received"]}, '
            f'missing={metrics["missing"]}, unexpected={metrics["unexpected"]}.'
        )
    return success


def _process_results(exp_path: Path, runs: Sequence[int]) -> None:
    """Validate saved runs, write metrics, and render topology figures."""

    records: list[dict[str, Any]] = []
    aggregate_edges: Counter[tuple[str, str]] = Counter()
    aggregate_channels: Counter[tuple[str, str, str]] = Counter()
    aggregate_latencies: list[float] = []
    for run in runs:
        agent_id = agent_name(run, '1_3')
        try:
            states = gather_states(exp_path / agent_id, True, no_warnings=True)[agent_id]
            passed = check_results(states)
            metrics = _communication_metrics(states) if passed else None
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f'ERROR: Run {run} cannot be processed: {error}')
            passed = False
            metrics = None
        record: dict[str, Any] = {'run': run, 'success': passed}
        if metrics:
            aggregate_edges.update(metrics['edges'])
            aggregate_channels.update(metrics['channels'])
            aggregate_latencies.extend(metrics['latencies_ms'])
            record.update({
                'sent': metrics['sent'],
                'received': metrics['received'],
                'missing': metrics['missing'],
                'unexpected': metrics['unexpected'],
                'coverage': metrics['coverage'],
                'waves_per_module': metrics['waves_per_module'],
                'latency': _latency_summary(metrics['latencies_ms']),
            })
        records.append(record)

    successes = sum(record['success'] for record in records)
    summary = {
        'experiment': '1-3',
        'mhagenta_version': DEFAULT_MHAGENTA_VERSION,
        'channel_validation': {
            'method': 'Reconstruct routes from complete ordered send waves; match receipts by exact message identity.',
            'limitation': 'Channel and callback labels were not saved; receiving callback identity is not independently verified.',
        },
        'runs': records,
        'aggregate_edges': {
            f'{source}->{target}': count
            for (source, target), count in sorted(aggregate_edges.items())
        },
        'aggregate_channels': {
            f'{source}->{target}:{route}': count
            for (source, target, route), count in sorted(aggregate_channels.items())
        },
        'aggregate_latency': _latency_summary(aggregate_latencies),
        'successes': successes,
        'total_runs': len(records),
        'success_rate': successes / len(records) if records else None,
        'success_rate_wilson_95': _wilson_interval(successes, len(records)),
    }
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    import matplotlib.pyplot as plt

    matrix = np.asarray([
        [aggregate_edges[(source, target)] for target in MODULES]
        for source in MODULES
    ])
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap='viridis', aspect='auto')
    axis.set_title(f'Experiment 1-3 communication topology — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    axis.set_xlabel('Destination module type')
    axis.set_ylabel('Source module type')
    axis.set_xticks(range(len(MODULES)), MODULES, rotation=35, ha='right')
    axis.set_yticks(range(len(MODULES)), MODULES)
    for row in range(len(MODULES)):
        for column in range(len(MODULES)):
            axis.text(column, row, f'{matrix[row, column]:,}', ha='center', va='center', fontsize=7)
    figure.colorbar(image, ax=axis, label='Messages sent')
    figure.tight_layout()
    figure.savefig(exp_path / 'communication-heatmap.svg')
    figure.savefig(exp_path / 'communication-heatmap.png', dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    ordered = np.sort(aggregate_latencies)
    if len(ordered):
        axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post', label='Internal messages')
        p50, p95 = np.percentile(ordered, [50, 95])
        axis.axvline(p50, color='#009E73', linestyle='--', label=f'P50 {p50:.3f} ms')
        axis.axvline(p95, color='#D55E00', linestyle=':', label=f'P95 {p95:.3f} ms')
    axis.set_title(f'Experiment 1-3 latency ECDF — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    axis.set_xlabel('One-way latency (ms)')
    axis.set_ylabel('Empirical cumulative probability')
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(exp_path / 'latency-ecdf.svg')
    figure.savefig(exp_path / 'latency-ecdf.png', dpi=200)
    plt.close(figure)

    if successes != len(records):
        raise RuntimeError('Experiment 1-3 contains missing, invalid, or old-schema results; rerun required.')


def run_experiment(
        run: int,
        exp_path: str | os.PathLike[str],
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run one complete internal communication-topology experiment."""

    exp_path = Path(exp_path).resolve()
    seeder = Seeder(run)
    start_delta: float = 30
    log_level: int = WARNING  # INFO
    orchestrator = Orchestrator(
        save_dir=Path(exp_path),
        step_frequency=0.1,
        control_frequency=1.,
        agent_start_delay=0,
        exec_start_time=time.time() + start_delta,
        exec_duration=DURATION,
        save_format='json',
        log_level=log_level,  # 15,  # INFO,
        save_logs=True,
        no_stdout_logs=False,
    )
    agent_id = agent_name(run, '1_3')
    orchestrator.add_agent(
        agent_id=agent_id,
        perceptors=[
            TestPerceptor(
                module_id=module_name(PERCEPTOR, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.perceptor}
            ) for i in range(N_MODULES)
        ],
        actuators=[
            TestActuator(
                module_id=module_name(ACTUATOR, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.actuator}
            ) for i in range(N_MODULES)
        ],
        ll_reasoners=[
            TestLLReasoner(
                module_id=module_name(LLREASONER, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.ll_reasoner}
            ) for i in range(N_MODULES)
        ],
        knowledge=[
            TestKnowledge(
                module_id=module_name(KNOWLEDGE, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.knowledge}
            ) for i in range(N_MODULES)
        ],
        hl_reasoners=[
            TestHLReasoner(
                module_id=module_name(HLREASONER, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.hl_reasoner}
            ) for i in range(N_MODULES)
        ],
        goal_graphs=[
            TestGoalGraph(
                module_id=module_name(GOALGRAPH, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.goal_graph}
            ) for i in range(N_MODULES)
        ],
        memory=[
            TestMemory(
                module_id=module_name(MEMORY, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.memory}
            ) for i in range(N_MODULES)
        ],
        learners=[
            TestLearner(
                module_id=module_name(LEARNER, i),
                initial_state={'sent': list(), 'received': list()},
                init_kwargs={'seed': seeder.learner}
            ) for i in range(N_MODULES)
        ],
        requirements_path=Path(__file__).resolve().with_name("requirements.txt"),
        extra_runtime_sources=Path(mha_exp_common.__file__).resolve().parent
    )

    orchestrator.run(
        mhagenta_version=_runtime_version(mha_version),
        local_build=_local_mhagenta_root(),
        force_run=True,
    )
    final_states = gather_states(exp_path / agent_id, True, no_warnings=True)
    final_states = final_states[agent_id]
    # print(f'Printing logs for agent \"{agent_name(run, '1_3')}\" at {exp_path / agent_name(run, '1_3')}:\n---{json.dumps(final_states, indent=2)}\n---')
    return check_results(final_states)


def run_batch(
        runs: int | tuple[int, int] | Sequence[int] = 50,
        exp_path: str | os.PathLike[str] = '.',
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
        process_only: bool = False,
) -> None:
    """Run the experiment batch or process its existing results."""

    completed = run_experiment_batch(
        experiment_id='1-3',
        title='COMMUNICATION',
        runs=runs,
        exp_path=exp_path,
        mha_version=_runtime_version(mha_version),
        runner=run_experiment,
        process_only=process_only,
    )
    if completed:
        _process_results(Path(exp_path).resolve(), list(normalize_runs(runs)[0]))
