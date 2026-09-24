################################
# Checking basic functionality #
################################
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from logging import INFO
from pathlib import Path
from typing import Any, cast

import mha_exp_common
import numpy as np
from mha_exp_common.batch import normalize_runs
from mha_exp_common.batch import run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import *
from mha_exp_common.utils import Seeder, agent_name, gather_states, module_name
from mhagenta import Orchestrator, State
from mhagenta.bases import *
from mhagenta.states import *
from numpy import random

DURATION = 10.
FREQUENCY = 2.
EXPECTED_COUNT = int(DURATION // FREQUENCY + 1)
RE_PATTERN = r'^.*?::\[[^\[\]]*\]\[([^\]]+)\].*?::.*$'
MODULE_COUNT_RANGE = (1, 11)
OPTIONAL_MODULES = (KNOWLEDGE, HLREASONER, GOALGRAPH, MEMORY, LEARNER)


def msg(counter: int) -> str:
    """Return the canonical counter log message."""

    return f'Current counter value={counter}, incrementing...'


def increment(state: State, log: Callable[[int, str], None]) -> State:
    """Log and apply one lifecycle counter increment."""

    log(INFO, msg(state.counter))
    state.counter += 1
    return state


def initialize(module: ModuleBase) -> None:
    """Record that the module completed its stateless initialization hook."""

    module._exp_initialized = True


def first_step(module: ModuleBase, state: State) -> State:
    """Persist that initialization preceded the first stateful hook."""

    state.initialized = bool(getattr(module, '_exp_initialized', False))
    state.first_called = state.initialized
    return state


def last_step(module: ModuleBase, state: State) -> State:
    """Persist ordered shutdown and perform the final expected increment."""

    state.last_called = bool(state.first_called and getattr(module, '_exp_initialized', False))
    if state.counter == EXPECTED_COUNT - 1:
        increment(state, module.log)
    return state


class TestActuator(ActuatorBase):
    """Exercise the actuator lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: ActuatorState) -> ActuatorState:
        return first_step(self, state)

    def step(self, state: ActuatorState) -> ActuatorState:
        return increment(state, self.log)

    def on_last(self, state: ActuatorState) -> ActuatorState:
        return last_step(self, state)


class TestPerceptor(PerceptorBase):
    """Exercise the perceptor lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: PerceptorState) -> PerceptorState:
        return first_step(self, state)

    def step(self, state: PerceptorState) -> PerceptorState:
        return increment(state, self.log)

    def on_last(self, state: PerceptorState) -> PerceptorState:
        return last_step(self, state)


class TestLLReasoner(LLReasonerBase):
    """Exercise the low-level reasoner lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: LLState) -> LLState:
        return first_step(self, state)

    def step(self, state: LLState) -> LLState:
        return increment(state, self.log)

    def on_last(self, state: LLState) -> LLState:
        return last_step(self, state)


class TestKnowledge(KnowledgeBase):
    """Exercise the knowledge lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: KnowledgeState) -> KnowledgeState:
        return first_step(self, state)

    def step(self, state: KnowledgeState) -> KnowledgeState:
        return increment(state, self.log)

    def on_last(self, state: KnowledgeState) -> KnowledgeState:
        return last_step(self, state)


class TestHLReasoner(HLReasonerBase):
    """Exercise the high-level reasoner lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: HLState) -> HLState:
        return first_step(self, state)

    def step(self, state: HLState) -> HLState:
        return increment(state, self.log)

    def on_last(self, state: HLState) -> HLState:
        return last_step(self, state)


class TestGoalGraph(GoalGraphBase):
    """Exercise the goal-graph lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: GoalGraphState) -> GoalGraphState:
        return first_step(self, state)

    def step(self, state: GoalGraphState) -> GoalGraphState:
        return increment(state, self.log)

    def on_last(self, state: GoalGraphState) -> GoalGraphState:
        return last_step(self, state)


class TestMemory(MemoryBase):
    """Exercise the memory lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: MemoryState) -> MemoryState:
        return first_step(self, state)

    def step(self, state: MemoryState) -> MemoryState:
        return increment(state, self.log)

    def on_last(self, state: MemoryState) -> MemoryState:
        return last_step(self, state)


class TestLearners(LearnerBase):
    """Exercise the learner lifecycle and periodic step hook."""

    def on_init(self, **kwargs) -> None:
        initialize(self)

    def on_first(self, state: LearnerState) -> LearnerState:
        return first_step(self, state)

    def step(self, state: LearnerState) -> LearnerState:
        return increment(state, self.log)

    def on_last(self, state: LearnerState) -> LearnerState:
        return last_step(self, state)


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
        raise ValueError(f'Experiment 1-1 requires MHAgentA {DEFAULT_MHAGENTA_VERSION}.')
    return runtime_version


def _local_mhagenta_root() -> Path:
    root = Path(__file__).resolve().parents[6] / 'mhagenta'
    project = root / 'pyproject.toml'
    if not project.is_file() or f'version = "{DEFAULT_MHAGENTA_VERSION}"' not in project.read_text(encoding='utf-8'):
        raise RuntimeError(f'Expected local MHAgentA {DEFAULT_MHAGENTA_VERSION} at {root}.')
    return root


def _module_counts(run: int) -> dict[str, int]:
    rng = random.default_rng(Seeder(run).orchestrator)
    omitted = OPTIONAL_MODULES[run % len(OPTIONAL_MODULES)]
    return {
        module: 0 if module == omitted else rng.integers(*MODULE_COUNT_RANGE).item()
        for module in MODULES
    }


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def check_agent(states: dict[str, dict[str, Any]], module_count: dict[str, int], logs: Sequence[str]) -> bool:
    """Validate exact module membership, lifecycle state, counters, and logs."""

    actual_modules = set(states)
    expected_modules = {
        module_name(module, index)
        for module, count in module_count.items()
        for index in range(count)
    }
    module_logs: dict[str, set[str]] = {module: set() for module in expected_modules}
    for line in logs:
        match = re.match(RE_PATTERN, line.lower())
        if match and match.group(1) in module_logs:
            module_logs[match.group(1)].add(line.lower().split('::')[2].strip())

    success = actual_modules == expected_modules
    if actual_modules - expected_modules:
        print(f'\tERROR: Unexpected modules: {", ".join(sorted(actual_modules - expected_modules))}')
    if expected_modules - actual_modules:
        print(f'\tERROR: Missing modules: {", ".join(sorted(expected_modules - actual_modules))}')

    expected_messages = {msg(counter).lower() for counter in range(EXPECTED_COUNT)}
    passed_modules = 0
    for module in sorted(actual_modules & expected_modules):
        state = states[module]
        module_success = (
            state.get('initialized') is True
            and state.get('first_called') is True
            and state.get('last_called') is True
            and state.get('counter') == EXPECTED_COUNT
            and expected_messages <= module_logs[module]
        )
        if not module_success:
            success = False
            print(f'\tERROR: Invalid lifecycle, counter, or logs for {module}: {state}')
        else:
            passed_modules += 1

    lowered_logs = ''.join(logs).lower()
    if not logs or 'traceback' in lowered_logs or re.search(r'\berror\b', lowered_logs):
        success = False
        print('\tERROR: Agent log is empty or contains an error/traceback entry.')
    print(f'Received expected results for {passed_modules}/{len(expected_modules)} modules!')
    return success


def _process_results(exp_path: Path, runs: Sequence[int]) -> None:
    """Validate saved runs, write aggregate metrics, and plot module counts."""

    records: list[dict[str, Any]] = []
    count_rows: list[list[int]] = []
    for run in runs:
        counts = _module_counts(run)
        count_rows.append([counts[module] for module in MODULES])
        agent_id = agent_name(run, '1_1')
        lifecycle_successes = 0
        counter_successes = 0
        try:
            gathered = gather_states(exp_path / agent_id, True, no_warnings=True)
            states = gathered[agent_id]
            logs = (exp_path / agent_id).with_suffix('.log').read_text(encoding='utf-8').splitlines()
            lifecycle_successes = sum(
                state.get('initialized') is True
                and state.get('first_called') is True
                and state.get('last_called') is True
                for state in states.values()
            )
            counter_successes = sum(state.get('counter') == EXPECTED_COUNT for state in states.values())
            passed = check_agent(states, counts, logs)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            passed = False
            print(f'ERROR: Run {run} cannot be processed: {error}')
        records.append({
            'run': run,
            'module_counts': counts,
            'modules_expected': sum(counts.values()),
            'lifecycle_successes': lifecycle_successes,
            'counter_successes': counter_successes,
            'success': passed,
        })

    successes = sum(record['success'] for record in records)
    summary = {
        'experiment': '1-1',
        'mhagenta_version': DEFAULT_MHAGENTA_VERSION,
        'runs': records,
        'modules_expected': sum(record['modules_expected'] for record in records),
        'lifecycle_successes': sum(record['lifecycle_successes'] for record in records),
        'counter_successes': sum(record['counter_successes'] for record in records),
        'successes': successes,
        'total_runs': len(records),
        'success_rate': successes / len(records) if records else None,
        'success_rate_wilson_95': _wilson_interval(successes, len(records)),
    }
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, max(3.5, len(records) * 0.18)))
    image = axis.imshow(np.asarray(count_rows), aspect='auto', cmap='viridis', vmin=0, vmax=MODULE_COUNT_RANGE[1] - 1)
    axis.set_title(f'Experiment 1-1 module counts — MHAgentA {DEFAULT_MHAGENTA_VERSION}, n={len(records)} runs')
    axis.set_xlabel('Module type')
    axis.set_ylabel('Run')
    axis.set_xticks(range(len(MODULES)), MODULES, rotation=35, ha='right')
    axis.set_yticks(range(len(records)), [record['run'] for record in records])
    figure.colorbar(image, ax=axis, label='Module instances')
    figure.tight_layout()
    figure.savefig(exp_path / 'module-counts.svg')
    figure.savefig(exp_path / 'module-counts.png', dpi=200)
    plt.close(figure)

    if successes != len(records):
        raise RuntimeError('Experiment 1-1 contains missing, invalid, or old-schema results; rerun required.')


def run_experiment(
        run: int,
        exp_path: str | os.PathLike[str],
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run one partial-topology lifecycle experiment."""

    exp_path = Path(exp_path).resolve()
    counts = _module_counts(run)
    modules: dict[str, list[ModuleBase]] = {}
    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=FREQUENCY,
        control_frequency=1.,
        agent_start_delay=20,
        exec_duration=DURATION,
        save_format='json',
        log_level=INFO,
        save_logs=True,
        no_stdout_logs=True,
    )
    for module in MODULES:
        modules[module] = [
            MODULE_CLASSES[module](
                module_id=module_name(module, index),
                initial_state={
                    'counter': 0,
                    'initialized': False,
                    'first_called': False,
                    'last_called': False,
                },
            )
            for index in range(counts[module])
        ]
    print('\tUsing ' + ', '.join(f'{counts[module]} {module}(s)' for module in MODULES) + '.')
    orchestrator.add_agent(
        agent_id=agent_name(run, '1_1'),
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
    log_path = (exp_path / agent_name(run, '1_1')).with_suffix('.log')
    if log_path.exists():
        log_path.unlink()
    orchestrator.run(
        mhagenta_version=_runtime_version(mha_version),
        local_build=_local_mhagenta_root(),
        force_run=True,
    )
    final_states = gather_states(exp_path / agent_name(run, '1_1'), True)[agent_name(run, '1_1')]
    logs = log_path.read_text(encoding='utf-8').splitlines()
    return check_agent(final_states, counts, logs)


def run_batch(
        runs: int | tuple[int, int] | Sequence[int] = 50,
        exp_path: str | os.PathLike[str] = '.',
        mha_version: str = DEFAULT_MHAGENTA_VERSION,
        process_only: bool = False,
) -> None:
    """Run the experiment batch or process its existing results."""

    completed = run_experiment_batch(
        experiment_id='1-1',
        title='FUNCTIONALITY',
        runs=runs,
        exp_path=exp_path,
        mha_version=_runtime_version(mha_version),
        runner=run_experiment,
        process_only=process_only,
    )
    if completed:
        _process_results(Path(exp_path).resolve(), list(normalize_runs(runs)[0]))
