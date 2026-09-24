#########################################################
# Checking RabbitMQ-based inter-container communication #
#########################################################
import json
import math
import os
import time
from collections import Counter
from collections.abc import Sequence
from logging import INFO, WARNING
from pathlib import Path
from typing import Any, cast

import mha_exp_common
import numpy as np
from mha_exp_common.batch import normalize_runs
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states
from mhagenta import Orchestrator
from mhagenta.bases import LLReasonerBase
from mhagenta.defaults.communication import (
    RMQActuatorBase,
    RMQPerceptorBase,
    RMQReceiverBase,
    RMQSenderBase,
)
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, PerceptorState
from numpy import random

DURATION = 10
N_AGENTS = 5
N_ENVS = 2
STOP_DELTA = 2.


class TestEnvironment(MHAEnvBase):
    """Echo timestamped observation and action requests over RabbitMQ."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator | None = None

    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._rng is None:
            self._rng = random.default_rng(state.pop('seed', None))
        payload = kwargs['payload']
        sent_ns = kwargs['sent_ns']
        state['observations'].append((sender_id, payload, sent_ns, time.time_ns()))
        response = f'{payload}_{self._rng.integers(0, 2**32).item()}'
        state['sent'].append(('observation', sender_id, response, sent_ns))
        return state, {'observation': response, 'sent_ns': sent_ns}

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._rng is None:
            self._rng = random.default_rng(state.pop('seed', None))
        payload = kwargs['action']
        sent_ns = kwargs['sent_ns']
        state['actions'].append((sender_id, payload, sent_ns, time.time_ns()))
        response = f'{payload}_{self._rng.integers(0, 2**32).item()}'
        state['sent'].append(('action', sender_id, response, sent_ns))
        return state, {'status': response, 'sent_ns': sent_ns}


class TestRMQSender(RMQSenderBase):
    """Send timestamped messages to every distinct peer agent."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0.

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            agent_idx = int(self.agent_id.split('_')[-1])
            self._rng = random.default_rng(kwargs['seed'] + Seeder.AGENT_MULTIPLIER * agent_idx)

    def step(self, state: ActuatorState) -> ActuatorState:
        if state.time < self._next_msg_ts or DURATION - state.time < STOP_DELTA:
            return state
        for agent in state.directory.external.agents:
            if agent.agent_id == self.agent_id:
                continue
            payload = str(self._rng.integers(0, 2**32).item())
            sent_ns = time.time_ns()
            self.send(
                recipient_id=agent.agent_id,
                msg={'payload': payload, 'sent_ns': sent_ns},
                performative='inform',
            )
            cast(list[tuple[str, str, int]], state['sent']).append((agent.agent_id, payload, sent_ns))
        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state


class TestRMQReceiver(RMQReceiverBase):
    """Record timestamped messages received from peer agents."""

    def on_message(self, state: PerceptorState, sender: str, msg: dict[str, Any]) -> PerceptorState:
        state['received'].append((sender, msg['payload'], msg['sent_ns'], time.time_ns()))
        return state


class TestRMQActuator(RMQActuatorBase):
    """Send timestamped actions and record their round-trip responses."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0.

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            agent_idx = int(self.agent_id.split('_')[-1])
            self._rng = random.default_rng(
                kwargs['seed'] + Seeder.AGENT_MULTIPLIER * agent_idx + Seeder.MODULE_MULTIPLIER
            )

    def step(self, state: ActuatorState) -> ActuatorState:
        if state.time < self._next_msg_ts or DURATION - state.time < STOP_DELTA:
            return state
        for env in state.directory.external.environments:
            payload = str(self._rng.integers(0, 2**32).item())
            sent_ns = time.time_ns()
            self.act(env_id=env.agent_id, action=payload, sent_ns=sent_ns)
            cast(list[tuple[str, str, int]], state['sent']).append((env.agent_id, payload, sent_ns))
        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs) -> ActuatorState:
        status = kwargs.get('status')
        if status is None:
            self.log(INFO, f'Received an empty status from {env_id}: {kwargs}!')
        state['received'].append((env_id, status, kwargs['sent_ns'], time.time_ns()))
        return state


class TestRMQPerceptor(RMQPerceptorBase):
    """Send timestamped observations and record their round-trip responses."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng: random.Generator = random.default_rng()
        self._next_msg_ts = 0.

    def on_init(self, **kwargs) -> None:
        if 'seed' in kwargs:
            agent_idx = int(self.agent_id.split('_')[-1])
            self._rng = random.default_rng(kwargs['seed'] + Seeder.AGENT_MULTIPLIER * agent_idx)

    def step(self, state: PerceptorState) -> PerceptorState:
        if state.time < self._next_msg_ts or DURATION - state.time < STOP_DELTA:
            return state
        for env in state.directory.external.environments:
            payload = str(self._rng.integers(0, 2**32).item())
            sent_ns = time.time_ns()
            self.observe(env_id=env.agent_id, payload=payload, sent_ns=sent_ns)
            cast(list[tuple[str, str, int]], state['sent']).append((env.agent_id, payload, sent_ns))
        self._next_msg_ts = state.time + 1 + self._rng.random()
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs) -> PerceptorState:
        observation = kwargs.get('observation')
        if observation is None:
            self.log(INFO, f'Received an empty observation from {env_id}: {kwargs}!')
        state['received'].append((env_id, observation, kwargs['sent_ns'], time.time_ns()))
        return state


def _runtime_version(mha_version: str) -> str:
    runtime_version = DEFAULT_MHAGENTA_VERSION if mha_version in {'', 'latest'} else mha_version
    if runtime_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f'Experiment 1-2 requires MHAgentA {DEFAULT_MHAGENTA_VERSION}.')
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


def _communication_metrics(
        states: dict[str, dict[str, dict[str, Any]]],
        agent_ids: list[str],
        env_ids: list[str],
) -> dict[str, Any]:
    peer_sent: Counter[tuple[Any, ...]] = Counter()
    peer_received: Counter[tuple[Any, ...]] = Counter()
    observation_sent: Counter[tuple[Any, ...]] = Counter()
    observation_received: Counter[tuple[Any, ...]] = Counter()
    observation_responses: Counter[tuple[Any, ...]] = Counter()
    observation_callbacks: Counter[tuple[Any, ...]] = Counter()
    action_sent: Counter[tuple[Any, ...]] = Counter()
    action_received: Counter[tuple[Any, ...]] = Counter()
    action_responses: Counter[tuple[Any, ...]] = Counter()
    action_callbacks: Counter[tuple[Any, ...]] = Counter()
    latencies = {'peer': [], 'observation': [], 'action': []}

    for agent_id in agent_ids:
        agent = states[agent_id]
        for recipient, payload, sent_ns in agent['sender']['sent']:
            peer_sent[(agent_id, recipient, payload, sent_ns)] += 1
        for sender, payload, sent_ns, received_ns in agent['receiver']['received']:
            peer_received[(sender, agent_id, payload, sent_ns)] += 1
            latencies['peer'].append((received_ns - sent_ns) / 1_000_000)
        for recipient, payload, sent_ns in agent['perceptor']['sent']:
            observation_sent[(agent_id, recipient, payload, sent_ns)] += 1
        for sender, response, sent_ns, received_ns in agent['perceptor']['received']:
            observation_callbacks[(sender, agent_id, response, sent_ns)] += 1
            latencies['observation'].append((received_ns - sent_ns) / 1_000_000)
        for recipient, payload, sent_ns in agent['actuator']['sent']:
            action_sent[(agent_id, recipient, payload, sent_ns)] += 1
        for sender, response, sent_ns, received_ns in agent['actuator']['received']:
            action_callbacks[(sender, agent_id, response, sent_ns)] += 1
            latencies['action'].append((received_ns - sent_ns) / 1_000_000)

    for env_id in env_ids:
        environment = states[env_id][env_id]
        for sender, payload, sent_ns, _ in environment['observations']:
            observation_received[(sender, env_id, payload, sent_ns)] += 1
        for sender, payload, sent_ns, _ in environment['actions']:
            action_received[(sender, env_id, payload, sent_ns)] += 1
        for kind, recipient, response, sent_ns in environment['sent']:
            event = (env_id, recipient, response, sent_ns)
            if kind == 'observation':
                observation_responses[event] += 1
            elif kind == 'action':
                action_responses[event] += 1
            else:
                raise ValueError(f'Unexpected environment response kind: {kind!r}')

    channel_pairs = {
        'peer': (peer_sent, peer_received),
        'observation_requests': (observation_sent, observation_received),
        'observation_responses': (observation_responses, observation_callbacks),
        'action_requests': (action_sent, action_received),
        'action_responses': (action_responses, action_callbacks),
    }
    channels = {
        name: {
            'sent': sum(sent.values()),
            'received': sum(received.values()),
            'missing': sum((sent - received).values()),
            'unexpected': sum((received - sent).values()),
        }
        for name, (sent, received) in channel_pairs.items()
    }
    return {
        'channels': channels,
        'latencies_ms': latencies,
        'self_addressed': sum(count for event, count in peer_sent.items() if event[0] == event[1]),
    }


def check_results(states: dict[str, dict[str, dict[str, Any]]], agent_ids: list[str], env_ids: list[str]) -> bool:
    """Validate entities, exact message multisets, and timestamp samples."""

    expected_ids = set(agent_ids) | set(env_ids)
    if set(states) != expected_ids:
        print(f'ERROR: expected IDs {sorted(expected_ids)}, received {sorted(states)}')
        return False
    for agent_id in agent_ids:
        if set(states[agent_id]) != {'sender', 'receiver', 'perceptor', 'actuator'}:
            print(f'ERROR: unexpected module state for {agent_id}: {sorted(states[agent_id])}')
            return False
    for env_id in env_ids:
        if set(states[env_id]) != {env_id}:
            print(f'ERROR: unexpected environment state for {env_id}: {sorted(states[env_id])}')
            return False
    try:
        metrics = _communication_metrics(states, agent_ids, env_ids)
    except (KeyError, TypeError, ValueError) as error:
        print(f'ERROR: invalid or old communication state schema: {error}')
        return False

    success = metrics['self_addressed'] == 0
    for channel, counts in metrics['channels'].items():
        if counts['missing'] or counts['unexpected'] or counts['sent'] == 0:
            success = False
            print(f'ERROR: invalid {channel} delivery counts: {counts}')
    for channel, values in metrics['latencies_ms'].items():
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            success = False
            print(f'ERROR: invalid {channel} latency samples.')
    if metrics['self_addressed']:
        print(f'ERROR: found {metrics["self_addressed"]} self-addressed messages.')
    if success:
        delivered = sum(counts['received'] for counts in metrics['channels'].values())
        print(f'Exactly matched {delivered} deliveries over {len(agent_ids)} agents and {len(env_ids)} environments.')
    return success


def _load_states(exp_path: Path, run: int) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str], list[str]]:
    run_path = exp_path / f'run_{run}'
    states = gather_states(run_path, False, no_warnings=True)
    template_id = agent_name(run, '1_2')
    template_state = states.pop(template_id, None)
    if template_state not in (None, {}):
        raise ValueError(f'Template agent {template_id} unexpectedly contains state.')
    agent_ids = [f'{template_id}_{index}' for index in range(N_AGENTS)]
    env_ids = [f'{env_name(run, "1_2")}_{index}' for index in range(N_ENVS)]
    for agent_id in agent_ids:
        if agent_id in states:
            states[agent_id].pop('dummy_llr', None)
    return states, agent_ids, env_ids


def _process_results(exp_path: Path, runs: Sequence[int]) -> None:
    """Validate saved runs, write metrics, and render communication figures."""

    records: list[dict[str, Any]] = []
    aggregate_channels: dict[str, Counter[str]] = {}
    aggregate_latencies = {'peer': [], 'observation': [], 'action': []}
    for run in runs:
        try:
            states, agent_ids, env_ids = _load_states(exp_path, run)
            passed = check_results(states, agent_ids, env_ids)
            metrics = _communication_metrics(states, agent_ids, env_ids) if passed else None
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f'ERROR: Run {run} cannot be processed: {error}')
            passed = False
            metrics = None
        record: dict[str, Any] = {'run': run, 'success': passed}
        if metrics:
            record['channels'] = metrics['channels']
            record['self_addressed'] = metrics['self_addressed']
            record['latency'] = {
                channel: _latency_summary(values)
                for channel, values in metrics['latencies_ms'].items()
            }
            for channel, counts in metrics['channels'].items():
                aggregate_channels.setdefault(channel, Counter()).update(counts)
            for channel, values in metrics['latencies_ms'].items():
                aggregate_latencies[channel].extend(values)
        records.append(record)

    successes = sum(record['success'] for record in records)
    summary = {
        'experiment': '1-2',
        'mhagenta_version': DEFAULT_MHAGENTA_VERSION,
        'runs': records,
        'aggregate_channels': {name: dict(counts) for name, counts in aggregate_channels.items()},
        'aggregate_latency': {
            channel: _latency_summary(values)
            for channel, values in aggregate_latencies.items()
        },
        'self_addressed': sum(record.get('self_addressed', 0) for record in records),
        'successes': successes,
        'total_runs': len(records),
        'success_rate': successes / len(records) if records else None,
        'success_rate_wilson_95': _wilson_interval(successes, len(records)),
    }
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis('off')
    axis.text(2, 3, 'Agents', ha='center', va='center', fontsize=14, bbox={'boxstyle': 'round', 'facecolor': '#E69F00', 'alpha': .25})
    axis.text(8, 3, 'Environments', ha='center', va='center', fontsize=14, bbox={'boxstyle': 'round', 'facecolor': '#56B4E9', 'alpha': .25})
    flow_specs = [
        ((2.8, 4.2), (7.2, 4.2), 'observation requests', '#0072B2', .12, 4.7),
        ((7.2, 3.55), (2.8, 3.55), 'observation responses', '#0072B2', -.12, 3.1),
        ((2.8, 2.45), (7.2, 2.45), 'action requests', '#D55E00', -.12, 2.05),
        ((7.2, 1.8), (2.8, 1.8), 'action responses', '#D55E00', .12, 1.35),
    ]
    for start, end, channel, color, curve, label_y in flow_specs:
        count = aggregate_channels.get(channel.replace(' ', '_'), Counter()).get('sent', 0)
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14, color=color, connectionstyle=f'arc3,rad={curve}'))
        axis.text(
            5,
            label_y,
            f'{channel}: {count:,}',
            ha='center',
            va='center',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .1},
        )
    peer_count = aggregate_channels.get('peer', Counter()).get('sent', 0)
    axis.add_patch(FancyArrowPatch((1.5, 3.7), (2.5, 3.7), arrowstyle='-|>', mutation_scale=14, color='#009E73', connectionstyle='arc3,rad=-1.2'))
    axis.text(2, 5.25, f'peer deliveries: {peer_count:,}\nself-addressed: {summary["self_addressed"]:,}', ha='center', va='center')
    axis.set_title(f'Experiment 1-2 communication flow — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    figure.tight_layout()
    figure.savefig(exp_path / 'communication-flow.svg')
    figure.savefig(exp_path / 'communication-flow.png', dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    labels = {'peer': 'Peer one-way', 'observation': 'Observation round trip', 'action': 'Action round trip'}
    for channel, values in aggregate_latencies.items():
        ordered = np.sort(values)
        if len(ordered):
            axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post', label=labels[channel])
    axis.set_title(f'Experiment 1-2 latency ECDF — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    axis.set_xlabel('Latency (ms)')
    axis.set_ylabel('Empirical cumulative probability')
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(exp_path / 'latency-ecdf.svg')
    figure.savefig(exp_path / 'latency-ecdf.png', dpi=200)
    plt.close(figure)

    if successes != len(records):
        raise RuntimeError('Experiment 1-2 contains missing, invalid, or old-schema results; rerun required.')


def run_experiment(
        run: int,
        exp_path: str | os.PathLike[str],
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run one external communication experiment."""

    exp_path = Path(exp_path).resolve() / f'run_{run}'
    exp_path.mkdir(parents=True, exist_ok=True)
    exchange_name = 'mhagenta'
    seeder = Seeder(run)
    start_delta = 120 if run == 0 else 90
    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=.1,
        control_frequency=1.,
        agent_start_delay=0,
        exec_start_time=time.time() + start_delta,
        exec_duration=DURATION,
        save_format='json',
        log_level=WARNING,
        save_logs=True,
        mas_rmq_uri='localhost:5672',
        no_stdout_logs=False,
        mas_rmq_exchange_name=exchange_name,
    )
    template_id = agent_name(run, '1_2')
    orchestrator.add_agent(
        agent_id=template_id,
        perceptors=[
            TestRMQReceiver(module_id='receiver', initial_state={'received': []}, exchange_name=exchange_name),
            TestRMQPerceptor(
                module_id='perceptor',
                initial_state={'sent': [], 'received': []},
                init_kwargs={'seed': seeder.perceptor},
                exchange_name=exchange_name,
            ),
        ],
        actuators=[
            TestRMQSender(
                module_id='sender',
                initial_state={'sent': []},
                init_kwargs={'seed': seeder.actuator},
                exchange_name=exchange_name,
            ),
            TestRMQActuator(
                module_id='actuator',
                initial_state={'sent': [], 'received': []},
                init_kwargs={'seed': seeder.actuator},
                exchange_name=exchange_name,
            ),
        ],
        ll_reasoners=LLReasonerBase(module_id='dummy_llr'),
        requirements_path=Path(__file__).resolve().with_name('requirements.txt'),
        num_copies=N_AGENTS,
        extra_runtime_sources=Path(mha_exp_common.__file__).resolve().parent,
    )
    for index in range(N_ENVS):
        environment_id = f'{env_name(run, "1_2")}_{index}'
        orchestrator.add_environment(
            base=TestEnvironment(init_state={
                'seed': seeder.environment + Seeder.ENV_MULTIPLIER * index,
                'sent': [],
                'actions': [],
                'observations': [],
            }),
            env_id=environment_id,
            exec_duration=DURATION * 2,
            requirements_path=Path(__file__).resolve().with_name('requirements.txt'),
            exchange_name=exchange_name,
            extra_runtime_sources=Path(mha_exp_common.__file__).resolve().parent,
        )
    orchestrator.run(
        mhagenta_version=_runtime_version(mha_version),
        local_build=_local_mhagenta_root(),
        force_run=True,
    )
    states, agent_ids, env_ids = _load_states(exp_path.parent, run)
    return check_results(states, agent_ids, env_ids)


def run_batch(
        runs: int | tuple[int, int] | Sequence[int] = 50,
        exp_path: str | os.PathLike[str] = '.',
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
        process_only: bool = False,
) -> None:
    """Run the experiment batch or process its existing results."""

    completed = run_experiment_batch(
        experiment_id='1-2',
        title='COMMUNICATION',
        runs=runs,
        exp_path=exp_path,
        mha_version=_runtime_version(mha_version),
        runner=run_experiment,
        process_only=process_only,
    )
    if completed:
        _process_results(Path(exp_path).resolve(), list(normalize_runs(runs)[0]))
