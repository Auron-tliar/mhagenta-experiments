#########################################
# Checking module execution concurrency #
#########################################
import json
import math
import os
import re
import time
from collections.abc import Sequence
from logging import INFO
from pathlib import Path
from typing import Any, cast

import mha_exp_common
import numpy as np
from mha_exp_common.batch import normalize_runs
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import *
from mha_exp_common.utils import agent_name, gather_states, module_name
from mhagenta import Orchestrator, State
from mhagenta.bases import *
from mhagenta.states import *
from scipy.stats import t as student_t

INIT_LOOP_LEN = 100_000_000
TOLERANCE = .1
TARGET_DURATION = 10.


def simulate_load(length: int, duration: float) -> tuple[int, float, float]:
    """Run a CPU-bound loop for at most the requested wall-clock duration."""

    counter = 0
    start_ts = time.time()
    start_thread_ts = time.thread_time()
    for _ in range(length):
        counter += 1
        if time.time() - start_ts > duration:
            break
    return counter, time.thread_time() - start_thread_ts, time.time() - start_ts


def run_step(state: State) -> State:
    """Run and persist one CPU-load measurement per module."""

    if state['done_ts'] is None:
        counter, thread_sec, elapsed_sec = simulate_load(state['loop_len'], TARGET_DURATION)
        state.done_ts = time.time()
        state.elapsed_sec = elapsed_sec
        state.thread_sec = thread_sec
        state.counter = counter
    return state


class TestActuator(ActuatorBase):
    """Measure CPU time in an actuator process."""

    def step(self, state: ActuatorState) -> ActuatorState:
        return run_step(state)


class TestPerceptor(PerceptorBase):
    """Measure CPU time in a perceptor process."""

    def step(self, state: PerceptorState) -> PerceptorState:
        return run_step(state)


class TestLLReasoner(LLReasonerBase):
    """Measure CPU time in a low-level reasoner process."""

    def step(self, state: LLState) -> LLState:
        return run_step(state)


class TestKnowledge(KnowledgeBase):
    """Measure CPU time in a knowledge process."""

    def step(self, state: KnowledgeState) -> KnowledgeState:
        return run_step(state)


class TestHLReasoner(HLReasonerBase):
    """Measure CPU time in a high-level reasoner process."""

    def step(self, state: HLState) -> HLState:
        return run_step(state)


class TestGoalGraph(GoalGraphBase):
    """Measure CPU time in a goal-graph process."""

    def step(self, state: GoalGraphState) -> GoalGraphState:
        return run_step(state)


class TestMemory(MemoryBase):
    """Measure CPU time in a memory process."""

    def step(self, state: MemoryState) -> MemoryState:
        return run_step(state)


class TestLearners(LearnerBase):
    """Measure CPU time in a learner process."""

    def step(self, state: LearnerState) -> LearnerState:
        return run_step(state)


MODULE_CLASSES: dict[str, type[ModuleBase]] = {
    ACTUATOR: TestActuator,
    PERCEPTOR: TestPerceptor,
    LLREASONER: TestLLReasoner,
    KNOWLEDGE: TestKnowledge,
    HLREASONER: TestHLReasoner,
    GOALGRAPH: TestGoalGraph,
    MEMORY: TestMemory,
    LEARNER: TestLearners,
}


def _runtime_version(mha_version: str) -> str:
    runtime_version = DEFAULT_MHAGENTA_VERSION if mha_version in {'', 'latest'} else mha_version
    if runtime_version != DEFAULT_MHAGENTA_VERSION:
        raise ValueError(f'Experiment 1-4 requires MHAgentA {DEFAULT_MHAGENTA_VERSION}.')
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


def _mean_confidence_interval(values: Sequence[float]) -> tuple[float, float | None, float | None]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    margin = float(student_t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def _module_multiples(cpu_count: int) -> list[int]:
    multiples = [index + 1 for index in range(cpu_count // len(MODULES) + 1)]
    if len(multiples) == 1:
        multiples.append(2)
    if multiples[-2] * len(MODULES) < cpu_count:
        multiples.append(multiples[-1] + 1)
    return multiples


def _clean_log(logs: Sequence[str]) -> bool:
    lowered = ''.join(logs).lower()
    return bool(logs) and 'traceback' not in lowered and re.search(r'\berror\b', lowered) is None


def check_results(
        states: dict[str, dict[str, Any]],
        expected_modules: int,
        cpu_count: int,
        logs: Sequence[str],
        verbose: bool = False,
) -> bool:
    """Validate complete, finite load measurements without a performance gate."""

    if len(states) != expected_modules:
        print(f'ERROR: expected {expected_modules} module states, received {len(states)}.')
        return False
    success = _clean_log(logs)
    elapsed: list[float] = []
    done: list[float] = []
    for module_id, state in states.items():
        try:
            values = (
                float(state['elapsed_sec']),
                float(state['thread_sec']),
                float(state['done_ts']),
                int(state['counter']),
                int(state['cpu_count']),
            )
        except (KeyError, TypeError, ValueError):
            print(f'ERROR: incomplete or old state schema for {module_id}: {state}')
            return False
        elapsed_sec, thread_sec, done_ts, counter, saved_cpu_count = values
        module_success = (
            all(math.isfinite(value) for value in (elapsed_sec, thread_sec, done_ts))
            and elapsed_sec > 0
            and 0 < thread_sec <= elapsed_sec + TOLERANCE
            and counter > 0
            and saved_cpu_count == cpu_count
        )
        if not module_success:
            success = False
            print(f'ERROR: invalid load measurement for {module_id}: {state}')
        elapsed.append(elapsed_sec)
        done.append(done_ts)
    if not _clean_log(logs):
        print('ERROR: Agent log is empty or contains an error/traceback entry.')
    if verbose and elapsed:
        print(
            f'Elapsed spread={max(elapsed) - min(elapsed):.6f} sec; '
            f'finish spread={max(done) - min(done):.6f} sec.'
        )
    return success


def test_duration() -> float:
    """Measure the host baseline used to size the bounded load loop."""

    start_ts = time.time()
    counter = 0
    for _ in range(INIT_LOOP_LEN):
        counter += 1
    return time.time() - start_ts


def setup_and_run(
        n_copies: int,
        loop_len: int,
        agent_id: str,
        start_delay: float,
        duration: float,
        cpu_count: int,
        mha_version: str,
        exp_path: str | os.PathLike[str],
        log_level: int = INFO,
) -> None:
    """Configure and run one module-count load level."""

    modules: dict[str, list[ModuleBase]] = {}
    for module in MODULES:
        modules[module] = [
            MODULE_CLASSES[module](
                module_id=module_name(module, index),
                initial_state={
                    'done_ts': None,
                    'elapsed_sec': None,
                    'thread_sec': None,
                    'counter': None,
                    'loop_len': loop_len,
                    'cpu_count': cpu_count,
                },
            )
            for index in range(n_copies)
        ]

    orchestrator = Orchestrator(
        save_dir=Path(exp_path),
        step_frequency=1.,
        control_frequency=1.,
        agent_start_delay=start_delay,
        exec_duration=duration,
        save_format='json',
        log_level=log_level,
        save_logs=True,
        no_stdout_logs=False,
    )
    orchestrator.add_agent(
        agent_id=agent_id,
        perceptors=cast(list[PerceptorBase], modules[PERCEPTOR]),
        actuators=cast(list[ActuatorBase], modules[ACTUATOR]),
        ll_reasoners=cast(list[LLReasonerBase], modules[LLREASONER]),
        knowledge=cast(list[KnowledgeBase], modules[KNOWLEDGE]),
        hl_reasoners=cast(list[HLReasonerBase], modules[HLREASONER]),
        goal_graphs=cast(list[GoalGraphBase], modules[GOALGRAPH]),
        memory=cast(list[MemoryBase], modules[MEMORY]),
        learners=cast(list[LearnerBase], modules[LEARNER]),
        requirements_path=Path(__file__).resolve().with_name('requirements.txt'),
        extra_runtime_sources=Path(mha_exp_common.__file__).resolve().parent,
    )
    log_path = (Path(exp_path) / agent_id).with_suffix('.log')
    if log_path.exists():
        log_path.unlink()
    orchestrator.run(
        mhagenta_version=_runtime_version(mha_version),
        local_build=_local_mhagenta_root(),
        force_run=True,
    )


def _load_metrics(states: dict[str, dict[str, Any]]) -> tuple[float, float]:
    shares = [float(state['thread_sec']) / float(state['elapsed_sec']) for state in states.values()]
    done = [float(state['done_ts']) for state in states.values()]
    return float(np.mean(shares)), max(done) - min(done)


def _process_results(exp_path: Path, runs: Sequence[int]) -> None:
    """Aggregate independent runs and render saturation and finish figures."""

    records: list[dict[str, Any]] = []
    load_shares: dict[int, list[float]] = {}
    finish_spreads: dict[int, list[float]] = {}
    common_cpu_count: int | None = None
    for run in runs:
        run_loads: list[dict[str, Any]] = []
        passed = True
        try:
            first_id = agent_name(run, '1_4_0')
            first_states = gather_states(exp_path / first_id, True, no_warnings=True)[first_id]
            cpu_count = int(next(iter(first_states.values()))['cpu_count'])
            if common_cpu_count not in (None, cpu_count):
                raise ValueError(f'CPU count changed from {common_cpu_count} to {cpu_count}.')
            common_cpu_count = cpu_count
            multiples = _module_multiples(cpu_count)
            for index, n_copies in enumerate(multiples):
                agent_id = agent_name(run, f'1_4_{index}')
                states = first_states if index == 0 else gather_states(exp_path / agent_id, True, no_warnings=True)[agent_id]
                logs = (exp_path / agent_id).with_suffix('.log').read_text(encoding='utf-8').splitlines()
                module_count = n_copies * len(MODULES)
                load_passed = check_results(states, module_count, cpu_count, logs)
                passed = passed and load_passed
                if load_passed:
                    share, finish_spread = _load_metrics(states)
                    load_shares.setdefault(module_count, []).append(share)
                    finish_spreads.setdefault(module_count, []).append(finish_spread)
                    run_loads.append({
                        'module_count': module_count,
                        'mean_cpu_time_share': share,
                        'finish_spread_sec': finish_spread,
                    })
        except (FileNotFoundError, KeyError, StopIteration, TypeError, ValueError) as error:
            print(f'ERROR: Run {run} cannot be processed: {error}')
            passed = False
        records.append({'run': run, 'success': passed, 'loads': run_loads})

    successes = sum(record['success'] for record in records)
    load_summaries: list[dict[str, Any]] = []
    for module_count in sorted(load_shares):
        mean, lower, upper = _mean_confidence_interval(load_shares[module_count])
        spreads = finish_spreads[module_count]
        load_summaries.append({
            'module_count': module_count,
            'runs': len(load_shares[module_count]),
            'mean_cpu_time_share': mean,
            'mean_cpu_time_share_ci_95': [lower, upper],
            'ideal_cpu_time_share': min(1., (common_cpu_count or 1) / module_count),
            'finish_spread_sec': {
                'p50': float(np.percentile(spreads, 50)),
                'p95': float(np.percentile(spreads, 95)),
                'max': float(max(spreads)),
            },
        })
    summary = {
        'experiment': '1-4',
        'mhagenta_version': DEFAULT_MHAGENTA_VERSION,
        'logical_cpu_count': common_cpu_count,
        'runs': records,
        'loads': load_summaries,
        'successes': successes,
        'total_runs': len(records),
        'success_rate': successes / len(records) if records else None,
        'success_rate_wilson_95': _wilson_interval(successes, len(records)),
    }
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    import matplotlib.pyplot as plt

    module_counts = [load['module_count'] for load in load_summaries]
    means = np.asarray([load['mean_cpu_time_share'] for load in load_summaries]) * 100
    lowers = np.asarray([
        load['mean_cpu_time_share_ci_95'][0]
        if load['mean_cpu_time_share_ci_95'][0] is not None else load['mean_cpu_time_share']
        for load in load_summaries
    ]) * 100
    uppers = np.asarray([
        load['mean_cpu_time_share_ci_95'][1]
        if load['mean_cpu_time_share_ci_95'][1] is not None else load['mean_cpu_time_share']
        for load in load_summaries
    ]) * 100
    ideal = np.asarray([load['ideal_cpu_time_share'] for load in load_summaries]) * 100
    figure, axis = plt.subplots(figsize=(8, 5))
    if module_counts:
        axis.errorbar(module_counts, means, yerr=[means - lowers, uppers - means], marker='o', capsize=4, label='Observed mean and 95% CI')
        axis.plot(module_counts, ideal, linestyle='--', marker='s', label='Ideal scheduling ceiling')
    if common_cpu_count is not None:
        axis.axvline(common_cpu_count, color='#D55E00', linestyle=':', label=f'{common_cpu_count} logical CPUs')
    axis.set_title(f'Experiment 1-4 concurrency saturation — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    axis.set_xlabel('Concurrent module processes')
    axis.set_ylabel('Mean CPU-time share per module (%)')
    axis.set_ylim(bottom=0)
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(exp_path / 'concurrency-saturation.svg')
    figure.savefig(exp_path / 'concurrency-saturation.png', dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    if module_counts:
        axis.boxplot([finish_spreads[count] for count in module_counts], tick_labels=module_counts)
    axis.set_title(f'Experiment 1-4 finish spread — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    axis.set_xlabel('Concurrent module processes')
    axis.set_ylabel('Within-run finish-time spread (s)')
    axis.grid(axis='y', alpha=.25)
    figure.tight_layout()
    figure.savefig(exp_path / 'finish-spread.svg')
    figure.savefig(exp_path / 'finish-spread.png', dpi=200)
    plt.close(figure)

    if successes != len(records):
        raise RuntimeError('Experiment 1-4 contains missing, invalid, or old-schema results; rerun required.')


def run_experiment(
        run: int,
        exp_path: str | os.PathLike[str],
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run all concurrency load levels for one independent repetition."""

    exp_path = Path(exp_path).resolve()
    cpu_count = os.cpu_count() or 1
    print(f'>>>>> Number of logical CPUs detected: {cpu_count}')
    multiples = _module_multiples(cpu_count)
    print(f'>>>>> Module multiples: {multiples}')

    baseline_duration = test_duration()
    loop_len = int(INIT_LOOP_LEN * TARGET_DURATION / baseline_duration)
    normal_duration = baseline_duration * 1.5
    for index, n_copies in enumerate(multiples):
        agent_id = agent_name(run, f'1_4_{index}')
        module_count = n_copies * len(MODULES)
        print(f'>>>>> Running {agent_id} with {module_count} modules...')
        execution_duration = normal_duration * (2 if index >= len(multiples) - 2 else 1)
        setup_and_run(
            n_copies,
            loop_len,
            agent_id,
            40,
            execution_duration,
            cpu_count,
            _runtime_version(mha_version),
            exp_path,
            Orchestrator.WARNING,
        )
        states = gather_states(exp_path / agent_id, True)[agent_id]
        logs = (exp_path / agent_id).with_suffix('.log').read_text(encoding='utf-8').splitlines()
        if not check_results(states, module_count, cpu_count, logs, verbose=True):
            return False
    return True


def run_batch(
        runs: int | tuple[int, int] | Sequence[int] = 50,
        exp_path: str | os.PathLike[str] = '.',
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
        process_only: bool = False,
) -> None:
    """Run the experiment batch or process its existing results."""

    completed = run_experiment_batch(
        experiment_id='1-4',
        title='CONCURRENCY',
        runs=runs,
        exp_path=exp_path,
        mha_version=_runtime_version(mha_version),
        runner=run_experiment,
        process_only=process_only,
    )
    if completed:
        _process_results(Path(exp_path).resolve(), list(normalize_runs(runs)[0]))
