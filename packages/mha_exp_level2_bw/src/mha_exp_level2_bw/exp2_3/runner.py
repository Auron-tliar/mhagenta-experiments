##################################################
# Experiment 2-3-BW: deliberative BDI planning  #
##################################################

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.util import find_spec
import json
from logging import ERROR
import os
from pathlib import Path
import random
import re
import signal
from statistics import fmean
from threading import Timer
from time import perf_counter
from typing import Any, Literal, cast

from mhagenta import ActionStatus, Belief, Observation, Orchestrator
from mhagenta.bases import HLReasonerBase, KnowledgeBase, LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, HLState, KnowledgeState, LLState, PerceptorState

import mha_exp_common
from mha_exp_common.batch import (
    cleanup_run_containers, cleanup_run_images, normalize_runs,
    run_batch as run_experiment_batch,
)
from mha_exp_common.names import ACTUATOR, HLREASONER, KNOWLEDGE, LLREASONER, PERCEPTOR
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name

from .reporting import process_execution_metrics
from .treatment import state_digest, treatment_for_run

from .planning import (
    GoalSpec,
    PlanResult,
    action_soundness,
    beliefs_to_facts,
    block_names,
    generate_options,
    location_names,
    parse_symbolic_observation,
    plan_blocks_world,
)


DURATION = 120.0
TABLE_LEN = 5
NUM_BLOCKS = 8
PLANNER_TIMEOUT = 30.0
SAVE_SUBDIR = Orchestrator.SAVE_SUBDIR
RECORD: Literal["all", "first", "none"] = "first"
VERBOSE = True
DEFAULT_MHAGENTA_VERSION = "1.4.12"


def _terminate_agent(state: Any, reason: str) -> None:
    """Request termination when running with a framework-backed outbox."""

    terminate = getattr(state.outbox, "terminate_agent", None)
    if callable(terminate):
        terminate(reason)

K_ACTION = "action"
K_OBSERVATION = "observation"
K_LEGAL = "legal"

A_CLOSE = "close"

LOG_PATTERN = re.compile(
    r"^\[(?P<time>[^\]]+)\]\[(?P<level>[^\]]+)\]::"
    r"\[(?P<sender>[^\]]+)\]::(?P<message>.*)$"
)


@contextmanager
def _active_time(state: Any) -> Iterator[None]:
    """Accumulate time spent in one experiment callback."""

    started = perf_counter()
    try:
        yield
    finally:
        state["active_seconds"] += perf_counter() - started


class TestEnvironment(MHAEnvBase):
    """Deterministic symbolic Blocks World exposed through MHAgentA."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = init_state.pop("seed", None)
        self._record = bool(init_state.pop("record", False))
        self._table_len = int(init_state.pop("table_len", TABLE_LEN))
        self._num_blocks = int(init_state.pop("num_blocks", NUM_BLOCKS))
        self._env: Any = None
        self._stop_timer: Timer | None = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_blocksworld import BWRecorder, BlocksWorldEnv

        environment = BlocksWorldEnv(
            table_len=self._table_len,
            num_blocks=self._num_blocks,
            render_mode="rgb_array" if self._record else None,
            symbolic=True,
        )
        environment.expose_snapshot = True
        self._env = (
            BWRecorder(environment, path=f"/{SAVE_SUBDIR}", single_trace=True)
            if self._record
            else environment
        )
        observation, _ = self._env.reset(seed=self._seed)
        self.state["world"] = list(observation)
        self.state["initial_world"] = list(observation)
        digest = state_digest(beliefs_to_facts(parse_symbolic_observation(observation)))
        if digest != self.state["treatment"]["initial_state_digest"]:
            raise ValueError("Seeded arrangement differs from the frozen treatment")

    def on_observe(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the current complete symbolic observation."""

        state["observations"] += 1
        return state, {K_OBSERVATION: list(state["world"])}

    def on_action(
        self,
        state: dict[str, Any],
        sender_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """Execute one atomic action or close the optional recorder."""

        action = kwargs.get(K_ACTION)
        if action == A_CLOSE:
            if not state["closed"]:
                state["close_requests"] += 1
                if self._env is not None:
                    self._env.close()
                state["closed"] = True
                # Trigger the framework's graceful save from inside the environment,
                # before external Docker stop can race with its shutdown.
                self._stop_timer = Timer(0.05, os.kill, args=(os.getpid(), signal.SIGTERM))
                self._stop_timer.daemon = True
                self._stop_timer.start()
            return state, None
        if state["closed"]:
            raise RuntimeError("Action requested after the environment closed")
        state["actions"] += 1
        if not isinstance(action, int):
            state["illegal_actions"] += 1
            return state, {K_LEGAL: False, "error": "missing-or-invalid-action"}

        observation, _, _, _, info = self._env.step(action)
        snapshot = info.get("snapshot")
        legal = bool(snapshot.legal) if snapshot is not None else False
        state["world"] = list(observation)
        if not legal:
            state["illegal_actions"] += 1
        return state, {K_LEGAL: legal}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        state["_stop_timer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_env()


class BDIBlocksWorldPerceptor(RMQPerceptorBase):
    """Forward complete symbolic observations to the supporting LL reasoner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""
        self._pending_context: dict[str, Any] | None = None

    def _resolve_ids(self, state: PerceptorState) -> None:
        if not self._env_id:
            environment = state.directory.external.environment
            if environment is None:
                raise RuntimeError("No Blocks World environment is registered.")
            self._env_id = environment.address["env_id"]
        if not self._ll_id:
            self._ll_id = state.directory.internal.ll_reasoning[0].module_id

    def on_first(self, state: PerceptorState) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender != self._ll_id:
                raise RuntimeError(f"Observation request came from unexpected module {sender!r}.")
            if self._pending_context is not None:
                raise RuntimeError("An observation request is already in flight.")
            self._pending_context = {
                key: kwargs[key]
                for key in ("action_id", K_LEGAL)
                if key in kwargs
            }
            state["requests"] += 1
            self.observe(self._env_id)
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        with _active_time(state):
            self._resolve_ids(state)
            if self._pending_context is None:
                raise RuntimeError("Received an observation with no pending request.")
            observation = kwargs.get(K_OBSERVATION)
            if not isinstance(observation, list):
                raise ValueError("Blocks World observation must be a symbolic fact list.")
            context = self._pending_context
            self._pending_context = None
            state["observations"] += 1
            state.outbox.send_observation(self._ll_id, Observation(observation), **context)
        return state


class BDIBlocksWorldActuator(RMQActuatorBase):
    """Execute only HL-originated atomic actions and report them to the LL."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._ll_id = ""
        self._hl_ids: set[str] = set()
        self._pending_action_id: str | None = None
        self._close_sent = False

    def _resolve_ids(self, state: ActuatorState) -> None:
        if not self._env_id:
            environment = state.directory.external.environment
            if environment is None:
                raise RuntimeError("No Blocks World environment is registered.")
            self._env_id = environment.address["env_id"]
        if not self._ll_id:
            self._ll_id = state.directory.internal.ll_reasoning[0].module_id
        if not self._hl_ids:
            self._hl_ids = {entry.module_id for entry in state.directory.internal.hl_reasoning}

    def on_first(self, state: ActuatorState) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
            if sender not in self._hl_ids:
                raise RuntimeError(f"Action request came from non-HL module {sender!r}.")
            if self._pending_action_id is not None:
                raise RuntimeError("An action request is already in flight.")
            action = kwargs.get(K_ACTION)
            action_id = kwargs.get("action_id")
            if not isinstance(action, int) or not isinstance(action_id, str):
                raise ValueError("Action requests require an integer action and string action_id.")
            self._pending_action_id = action_id
            state["requests"] += 1
            self.act(self._env_id, action=action)
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        with _active_time(state):
            self._resolve_ids(state)
            if self._pending_action_id is None:
                raise RuntimeError("Received an action status with no pending action.")
            action_id = self._pending_action_id
            self._pending_action_id = None
            state["statuses"] += 1
            state.outbox.send_status(
                self._ll_id,
                ActionStatus(
                    {
                        "action_id": action_id,
                        K_LEGAL: kwargs.get(K_LEGAL),
                        "error": kwargs.get("error"),
                    }
                ),
            )
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        """Send the recorder close once, even if shutdown hooks are repeated."""
        if self._env_id and not self._close_sent:
            self._close_sent = True
            self.act(self._env_id, action=A_CLOSE)
        return state


class SupportLLReasoner(LLReasonerBase):
    """Extract and forward beliefs without selecting actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._perceptor_id = ""
        self._knowledge_id = ""

    def _resolve_ids(self, state: LLState) -> None:
        if not self._perceptor_id:
            self._perceptor_id = state.directory.internal.perception[0].module_id
        if not self._knowledge_id:
            self._knowledge_id = state.directory.internal.knowledge[0].module_id

    def _request_observation(
        self,
        state: LLState,
        *,
        action_id: str | None = None,
        legal: bool | None = None,
    ) -> None:
        metadata: dict[str, Any] = {}
        if action_id is not None:
            metadata["action_id"] = action_id
            metadata[K_LEGAL] = legal
        state["observation_requests"] += 1
        state.outbox.request_observation(self._perceptor_id, **metadata)

    def _fail(self, state: LLState, message: str) -> None:
        if state["failure"] is None:
            state["failure"] = message
            self.log(ERROR, message)

    def on_first(self, state: LLState) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            self._request_observation(state)
        return state

    def on_observation(
        self,
        state: LLState,
        sender: str,
        observation: Observation,
        **kwargs: Any,
    ) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            if state["failure"] is not None:
                return state
            content = observation.content
            if not isinstance(content, list):
                self._fail(state, f"Expected list observation, got {type(content).__name__}.")
                return state
            try:
                beliefs = parse_symbolic_observation(content)
            except ValueError as exc:
                self._fail(state, str(exc))
                return state

            state["observations_received"] += 1
            state["observation_seq"] += 1
            state["belief_updates_sent"] += 1
            state.outbox.send_beliefs(
                self._knowledge_id,
                observation,
                beliefs,
                observation_seq=state["observation_seq"],
                action_id=kwargs.get("action_id"),
                legal=kwargs.get(K_LEGAL),
            )
        return state

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        with _active_time(state):
            self._resolve_ids(state)
            if state["failure"] is not None:
                return state
            status = action_status.status
            if not isinstance(status, dict) or not isinstance(status.get("action_id"), str):
                self._fail(state, "Malformed or uncorrelated action status.")
                return state
            legal = status.get(K_LEGAL)
            if not isinstance(legal, bool):
                self._fail(state, "Action status did not contain Boolean legality.")
                return state
            state["action_statuses"] += 1
            self._request_observation(
                state,
                action_id=status["action_id"],
                legal=legal,
            )
        return state


class ClosedWorldKnowledge(KnowledgeBase):
    """Forward complete belief snapshots and action metadata to HL reasoners."""

    def on_observed_beliefs(
        self,
        state: KnowledgeState,
        sender: str,
        observation: Observation,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> KnowledgeState:
        with _active_time(state):
            state["revisions"] += 1
            for reasoner in state.directory.internal.hl_reasoning:
                state.outbox.send_beliefs(
                    reasoner.module_id,
                    beliefs,
                    observation_seq=kwargs.get("observation_seq"),
                    action_id=kwargs.get("action_id"),
                    legal=kwargs.get(K_LEGAL),
                )
                state["updates_forwarded"] += 1
        return state


class DeliberativeBDIReasoner(HLReasonerBase):
    """Select, plan, and execute one finite Blocks World intention."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random()
        self._blocks: list[str] = []
        self._locations: list[str] = []
        self._planner_timeout = PLANNER_TIMEOUT
        self._actuator_id = ""
        self._fixed_goal: GoalSpec | None = None

    def on_init(self, **kwargs: Any) -> None:
        self._rng = random.Random(int(kwargs["seed"]))
        self._blocks = block_names(int(kwargs["num_blocks"]))
        self._locations = location_names(int(kwargs["table_len"]))
        self._planner_timeout = float(kwargs["planner_timeout"])
        fixed_goal = kwargs.get("fixed_goal")
        if fixed_goal is not None:
            if not isinstance(fixed_goal, dict):
                raise ValueError("fixed_goal must be a dictionary")
            self._fixed_goal = GoalSpec(str(fixed_goal["top"]), str(fixed_goal["bottom"]))

    def on_first(self, state: HLState) -> HLState:
        with _active_time(state):
            self._actuator_id = state.directory.internal.actuation[0].module_id
            state["run"]["phase"] = "awaiting-beliefs"
        return state

    def _fail(self, state: HLState, code: str, stage: str, message: str) -> None:
        run = state["run"]
        if run["failure"] is not None:
            return
        run["failure"] = {"code": code, "stage": stage, "message": message}
        run["phase"] = "failed"
        self.log(ERROR, f"{code}: {message}")
        _terminate_agent(state, f"2-3-BW failed: {code}")

    def _select_intention(self, state: HLState, facts: set[str]) -> GoalSpec | None:
        options = generate_options(self._blocks, facts)
        run = state["run"]
        run["option_count"] = len(options)
        if not options:
            return None
        intention = self._fixed_goal or self._rng.choice(options)
        if intention not in options:
            return None
        run["intention"] = intention.as_dict()
        run["intention_initially_satisfied"] = intention.fact in facts
        run["goal_fact"] = intention.fact
        return intention

    def _plan(self, state: HLState, facts: set[str], intention: GoalSpec) -> bool:
        run = state["run"]
        run["phase"] = "planning"
        result: PlanResult = plan_blocks_world(
            facts,
            intention,
            blocks=self._blocks,
            locations=self._locations,
            timeout=self._planner_timeout,
            problem_name=f"exp2_3_{state['belief_updates']}",
        )
        run["planner"] = result.as_dict()
        if not result.accepted:
            failure = result.failure or "planner-failed"
            code = failure.split(":", 1)[0]
            self._fail(state, code, "planning", failure)
            return False
        run["phase"] = "executing"
        return True

    def _dispatch_next(self, state: HLState, facts: set[str]) -> bool:
        run = state["run"]
        planner = run["planner"]
        actions = planner.get("actions", []) if isinstance(planner, dict) else []
        index = run["plan_index"]
        if not isinstance(actions, list) or index >= len(actions):
            self._fail(
                state,
                "plan-exhausted-before-goal",
                "executing",
                "The accepted plan ended before the selected goal was observed.",
            )
            return False
        action = actions[index]
        sound, missing = action_soundness(action, facts)
        if not sound:
            self._fail(
                state,
                "unsound-next-action",
                "executing",
                f"Plan action {index} is missing preconditions: {missing}.",
            )
            return False

        action_id = f"action-{index + 1}"
        run["pending_action"] = {"action_id": action_id, "plan_index": index}
        run["executions"].append(
            {
                "plan_index": index,
                "action_id": action_id,
                "sound": True,
                "correlated": None,
                "legal": None,
                "observation_seq": None,
            }
        )
        state.outbox.request_action(
            self._actuator_id,
            action=action["env_action"],
            action_id=action_id,
        )
        return True

    def _monitor_pending(
        self,
        state: HLState,
        *,
        observation_seq: int,
        action_id: Any,
        legal: Any,
    ) -> bool:
        run = state["run"]
        pending = run["pending_action"]
        if pending is None:
            if action_id is not None or legal is not None:
                self._fail(
                    state,
                    "correlation-error",
                    "executing",
                    "An action outcome arrived with no pending action.",
                )
                return False
            return True

        execution = run["executions"][-1]
        correlated = action_id == pending["action_id"]
        execution["correlated"] = correlated
        execution["legal"] = legal
        execution["observation_seq"] = observation_seq
        if not correlated:
            self._fail(
                state,
                "correlation-error",
                "executing",
                f"Expected {pending['action_id']!r}, received {action_id!r}.",
            )
            return False

        run["pending_action"] = None
        if legal is not True:
            self._fail(
                state,
                "illegal-action",
                "executing",
                f"Environment rejected {pending['action_id']}.",
            )
            return False
        run["plan_index"] += 1
        return True

    def on_belief_update(
        self,
        state: HLState,
        sender: str,
        beliefs: Sequence[Belief],
        **kwargs: Any,
    ) -> HLState:
        with _active_time(state):
            run = state["run"]
            if run["phase"] in {"complete", "failed"}:
                return state
            try:
                facts = beliefs_to_facts(beliefs)
            except ValueError as exc:
                self._fail(state, "belief-error", "belief-revision", str(exc))
                return state
            state["belief_updates"] += 1
            observation_seq = kwargs.get("observation_seq")
            if not isinstance(observation_seq, int):
                observation_seq = state["belief_updates"]

            if run["phase"] == "awaiting-beliefs":
                if kwargs.get("action_id") is not None or kwargs.get(K_LEGAL) is not None:
                    self._fail(
                        state,
                        "correlation-error",
                        "belief-revision",
                        "The initial observation contained an action outcome.",
                    )
                    return state
                intention = self._select_intention(state, facts)
                if intention is None:
                    self._fail(
                        state,
                        "no-intention",
                        "selecting-intention",
                        "No unsatisfied distinct-block stacking intention is available.",
                    )
                    return state
                if not self._plan(state, facts, intention):
                    return state
                self._dispatch_next(state, facts)
                return state

            if not self._monitor_pending(
                state,
                observation_seq=observation_seq,
                action_id=kwargs.get("action_id"),
                legal=kwargs.get(K_LEGAL),
            ):
                return state
            if run["goal_fact"] in facts:
                run["phase"] = "complete"
                run["goal_observed_at"] = observation_seq
                _terminate_agent(state, "2-3-BW fixed goal complete")
                return state
            self._dispatch_next(state, facts)
        return state


def _initial_states(treatment: dict[str, object] | None = None) -> dict[str, dict[str, Any]]:
    treatment = treatment or treatment_for_run(0)
    return {
        PERCEPTOR: {
            "requests": 0,
            "observations": 0,
            "active_seconds": 0.0,
        },
        ACTUATOR: {
            "requests": 0,
            "statuses": 0,
            "active_seconds": 0.0,
        },
        LLREASONER: {
            "observation_requests": 0,
            "observations_received": 0,
            "observation_seq": 0,
            "belief_updates_sent": 0,
            "action_statuses": 0,
            "failure": None,
            "active_seconds": 0.0,
        },
        KNOWLEDGE: {
            "revisions": 0,
            "updates_forwarded": 0,
            "active_seconds": 0.0,
        },
        HLREASONER: {
            "treatment": dict(treatment),
            "belief_updates": 0,
            "active_seconds": 0.0,
            "run": {
                "phase": "awaiting-beliefs",
                "option_count": 0,
                "intention": None,
                "intention_initially_satisfied": None,
                "planner": None,
                "plan_index": 0,
                "executions": [],
                "pending_action": None,
                "goal_fact": None,
                "goal_observed_at": None,
                "failure": None,
            },
        },
    }


def deliberative_initial_states() -> dict[str, dict[str, Any]]:
    """Return fresh initial states for the five-module deliberative topology."""

    return _initial_states()


def _runtime_failure(line: str) -> bool:
    match = LOG_PATTERN.match(line)
    level = match.group("level").lower() if match else ""
    message = match.group("message").lower() if match else line.lower()
    if level in {"error", "critical"}:
        return True
    return any(
        marker in message
        for marker in (
            "traceback",
            "exceptiongroup",
            "caught exception",
            "could not send message",
            "failed to send",
            "failed to save state",
        )
    )


def _result_errors(
    states: dict[str, dict[str, Any]] | None,
    logs: Sequence[str],
    environment_state: dict[str, Any] | None,
    environment_logs: Sequence[str],
    expected_treatment: dict[str, object] | None = None,
) -> list[str]:
    """Cross-check agent execution against independently saved world evidence."""
    errors: list[str] = []
    if not isinstance(states, dict):
        return ["agent module states are missing"]
    if not logs or not environment_logs:
        errors.append("required agent or environment log is missing")
    for line in (*logs, *environment_logs):
        if _runtime_failure(line):
            errors.append(f"runtime failure in logs: {line.rstrip()}")
            break

    required = {
        module_name(PERCEPTOR, 0),
        module_name(ACTUATOR, 0),
        module_name(LLREASONER, 0),
        module_name(KNOWLEDGE, 0),
        module_name(HLREASONER, 0),
    }
    if set(states) != required:
        return [*errors, "agent module state set differs from the required five modules"]

    perceptor = states[module_name(PERCEPTOR, 0)]
    actuator = states[module_name(ACTUATOR, 0)]
    ll = states[module_name(LLREASONER, 0)]
    knowledge = states[module_name(KNOWLEDGE, 0)]
    hl = states[module_name(HLREASONER, 0)]
    modules = (perceptor, actuator, ll, knowledge, hl)
    try:
        json.dumps(modules)
    except (TypeError, ValueError) as exc:
        errors.append(f"module state is not JSON-serializable: {exc}")

    for name, state in zip(
        (PERCEPTOR, ACTUATOR, LLREASONER, KNOWLEDGE, HLREASONER),
        modules,
        strict=True,
    ):
        if not isinstance(state, dict):
            errors.append(f"{name} state is not a dictionary")
        elif not isinstance(state.get("active_seconds"), (int, float)) or state["active_seconds"] <= 0:
            errors.append(f"{name} did not record positive active time")
    if not all(isinstance(state, dict) for state in modules):
        return errors

    if ll.get("failure") is not None:
        errors.append(f"supporting LL failed: {ll.get('failure')}")

    run = hl.get("run")
    if not isinstance(run, dict):
        return [*errors, "HL compact run record is missing"]
    executions = run.get("executions")
    if not isinstance(executions, list):
        executions = []
        errors.append("HL executions are missing or malformed")

    observation_counts = [
        ll.get("observation_requests"),
        perceptor.get("requests"),
        perceptor.get("observations"),
        ll.get("observations_received"),
        ll.get("belief_updates_sent"),
        knowledge.get("revisions"),
        knowledge.get("updates_forwarded"),
        hl.get("belief_updates"),
    ]
    if any(not isinstance(value, int) for value in observation_counts):
        errors.append("observation/update chain contains non-integer counters")
    elif observation_counts[0] <= 0 or len(set(observation_counts)) != 1:
        errors.append(f"observation/update count chain is incomplete: {observation_counts}")

    action_counts = [
        len(executions),
        actuator.get("requests"),
        actuator.get("statuses"),
        ll.get("action_statuses"),
    ]
    if any(not isinstance(value, int) for value in action_counts):
        errors.append("action/status chain contains non-integer counters")
    elif action_counts[0] <= 0 or len(set(action_counts)) != 1:
        errors.append(f"action/status count chain is incomplete: {action_counts}")
    if (
        isinstance(observation_counts[0], int)
        and isinstance(action_counts[0], int)
        and observation_counts[0] != action_counts[0] + 1
    ):
        errors.append("observation count is not exactly action count plus one")

    if run.get("phase") != "complete":
        errors.append(f"HL phase is not complete: {run.get('phase')!r}")
    if run.get("failure") is not None:
        errors.append(f"HL recorded failure: {run.get('failure')}")
    if run.get("pending_action") is not None:
        errors.append("HL retained a pending action")
    if not isinstance(run.get("option_count"), int) or run["option_count"] <= 0:
        errors.append("HL did not record a non-empty option set")
    if run.get("intention_initially_satisfied") is not False:
        errors.append("selected intention was not recorded as initially unsatisfied")

    intention = run.get("intention")
    treatment = hl.get("treatment")
    treatment_num_blocks = (
        treatment.get("num_blocks") if isinstance(treatment, dict) else NUM_BLOCKS
    )
    valid_blocks = set(block_names(int(treatment_num_blocks)))
    if not isinstance(intention, dict):
        errors.append("selected intention is missing")
        intention = {}
    top = intention.get("top")
    bottom = intention.get("bottom")
    if top not in valid_blocks or bottom not in valid_blocks or top == bottom:
        errors.append(f"selected intention is invalid: {intention}")
    expected_goal = f"on({top},{bottom})"
    if run.get("goal_fact") != expected_goal:
        errors.append("recorded goal fact does not match the selected intention")

    planner = run.get("planner")
    actions: list[Any] = []
    if not isinstance(planner, dict):
        errors.append("planner record is missing")
    else:
        actions_value = planner.get("actions")
        if isinstance(actions_value, list):
            actions = actions_value
        if planner.get("planner") != "lpg":
            errors.append("accepted planner is not LPG")
        if planner.get("status") not in {"SOLVED_SATISFICING", "SOLVED_OPTIMALLY"}:
            errors.append("LPG did not record a solved status")
        if planner.get("sequential") is not True:
            errors.append("accepted plan is not sequential")
        if planner.get("validation_status") != "VALID":
            errors.append("accepted plan did not pass independent validation")
        if planner.get("failure") is not None:
            errors.append(f"planner record contains failure: {planner.get('failure')}")
        if not actions:
            errors.append("accepted plan has no actions")
        if planner.get("length") != len(actions):
            errors.append("planner length does not match serialized actions")
        if not isinstance(planner.get("sanity_bound"), int) or len(actions) > planner.get(
            "sanity_bound", -1
        ):
            errors.append("accepted plan exceeds or lacks the sanity bound")

    for index, execution in enumerate(executions):
        if not isinstance(execution, dict):
            errors.append(f"execution row {index} is malformed")
            continue
        if execution.get("plan_index") != index:
            errors.append(f"execution row {index} is not the next plan action")
        if execution.get("action_id") != f"action-{index + 1}":
            errors.append(f"execution row {index} has an invalid action identity")
        if execution.get("sound") is not True:
            errors.append(f"execution row {index} was not sound")
        if execution.get("correlated") is not True:
            errors.append(f"execution row {index} was not correlated")
        if execution.get("legal") is not True:
            errors.append(f"execution row {index} was illegal")
        if not isinstance(execution.get("observation_seq"), int):
            errors.append(f"execution row {index} lacks an observation sequence")
    if len(executions) > len(actions):
        errors.append("executed actions are not a prefix of the accepted plan")
    if run.get("plan_index") != len(executions):
        errors.append("plan index does not match completed execution rows")
    final_execution = executions[-1] if executions and isinstance(executions[-1], dict) else None
    if (
        final_execution is None
        or run.get("goal_observed_at") != final_execution.get("observation_seq")
    ):
        errors.append("goal completion was not established by the final action observation")
    if [row.get("observation_seq") for row in executions if isinstance(row, dict)] != list(range(2, len(executions) + 2)):
        errors.append("action observations are not sequential")
    if not isinstance(environment_state, dict):
        return [*errors, "environment state is missing"]
    if not isinstance(treatment, dict):
        return [*errors, "agent treatment is missing"]
    try:
        frozen = treatment_for_run(treatment.get("run_id"))
    except (ValueError, TypeError):
        return [*errors, "agent treatment has invalid run ID"]
    if treatment != frozen or environment_state.get("treatment") != frozen:
        errors.append("agent/environment treatment differs from frozen manifest")
    if expected_treatment is not None and treatment != expected_treatment:
        errors.append("saved treatment does not match requested execution")
    if (top, bottom) != (frozen["top"], frozen["bottom"]):
        errors.append("selected intention differs from treatment goal")
    if environment_state.get("actions") != len(executions):
        errors.append("environment and agent action counts differ")
    if environment_state.get("observations") != hl.get("belief_updates"):
        errors.append("environment and agent observation counts differ")
    if environment_state.get("illegal_actions") != 0:
        errors.append("environment recorded illegal actions")
    if environment_state.get("closed") is not True or environment_state.get("close_requests") != 1:
        errors.append("environment did not close exactly once")
    try:
        initial_facts = beliefs_to_facts(parse_symbolic_observation(environment_state["initial_world"]))
        final_facts = beliefs_to_facts(parse_symbolic_observation(environment_state["world"]))
    except (KeyError, TypeError, ValueError):
        errors.append("environment initial/final symbolic state is malformed")
    else:
        if state_digest(initial_facts) != frozen["initial_state_digest"]:
            errors.append("initial arrangement differs from the frozen treatment")
        if expected_goal in initial_facts or expected_goal not in final_facts:
            errors.append("environment does not prove initially unsatisfied goal completion")
    return errors


def check_results(
    states: dict[str, dict[str, Any]] | None,
    logs: Sequence[str],
    environment_state: dict[str, Any] | None,
    environment_logs: Sequence[str],
    *,
    expected_treatment: dict[str, object] | None = None,
    verbose: bool = False,
) -> bool:
    """Apply the finite experiment's scientific acceptance contract."""

    errors = _result_errors(states, logs, environment_state, environment_logs, expected_treatment)
    for error in errors:
        print(f"Acceptance check failed: {error}")
    if verbose and isinstance(states, dict):
        hl = states.get(module_name(HLREASONER, 0), {})
        run = hl.get("run", {}) if isinstance(hl, dict) else {}
        planner = run.get("planner", {}) if isinstance(run, dict) else {}
        executions = run.get("executions", []) if isinstance(run, dict) else []
        print(f"Terminal phase: {run.get('phase')}")
        print(f"Selected intention: {run.get('intention')}")
        print(f"LPG plan length: {planner.get('length')}")
        print(f"Executed prefix length: {len(executions)}")
        print(f"Failure: {run.get('failure')}")
    return not errors


def read_run_evidence(root: Path, run: int) -> tuple[dict, list[str], dict, list[str]]:
    """Read exactly five agent states, one environment state, and both logs."""
    agent_id, environment_id = agent_name(run, "2_3"), env_name(run, "2_3")
    agent_files = list((root / agent_id / "out").glob("*.json"))
    environment_files = list((root / environment_id / "out").glob("*.json"))
    if len(agent_files) != 5 or len(environment_files) != 1:
        raise ValueError(f"Run {run}: expected five agent states and one environment state")
    states = {p.stem.split(".", 1)[1]: json.loads(p.read_text(encoding="utf-8")) for p in agent_files}
    environment = json.loads(environment_files[0].read_text(encoding="utf-8"))
    logs = [(root / f"{entity}.log").read_text(encoding="utf-8").splitlines()
            for entity in (agent_id, environment_id)]
    return states, logs[0], environment, logs[1]


def _workspace_root() -> Path:
    candidates = [Path(__file__).resolve(), Path.cwd().resolve()]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            pyproject = parent / "pyproject.toml"
            if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
                return parent
    raise RuntimeError("Could not locate the mhagenta-experiments workspace root.")


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run one isolated 2-3-BW lifecycle and check its saved evidence."""

    exp_path = Path(exp_path).resolve()
    workspace_root = _workspace_root()
    mha_root = (workspace_root.parent / "mhagenta").resolve()
    version_file = mha_root / "pyproject.toml"
    if not version_file.is_file() or 'version = "1.4.12"' not in version_file.read_text(encoding="utf-8"):
        raise RuntimeError(f"Expected local MHAgentA 1.4.12 checkout at {mha_root}.")

    bw_spec = find_spec("mha_env_blocksworld")
    if bw_spec is None or bw_spec.origin is None:
        raise ImportError("Could not locate mha_env_blocksworld for environment runtime sources.")
    bw_runtime_source = Path(bw_spec.origin).resolve().parent

    seeder = Seeder(run)
    exchange_name = "mhagenta"
    orchestrator = Orchestrator(
        save_dir=exp_path,
        step_frequency=0.0,
        control_frequency=0.0,
        status_frequency=5.0,
        agent_start_delay=20.0,
        exec_duration=DURATION,
        save_format="json",
        log_level=Orchestrator.INFO,
        save_logs=True,
        no_stdout_logs=False,
        mas_rmq_uri="localhost:5672",
        mas_rmq_exchange_name=exchange_name,
        stop_on_agents_term=True,
    )

    treatment = treatment_for_run(run)
    initial = _initial_states(treatment)
    orchestrator.add_agent(
        agent_id=agent_name(run, "2_3"),
        perceptors=BDIBlocksWorldPerceptor(
            module_id=module_name(PERCEPTOR, 0),
            initial_state=initial[PERCEPTOR],
            exchange_name=exchange_name,
        ),
        actuators=BDIBlocksWorldActuator(
            module_id=module_name(ACTUATOR, 0),
            initial_state=initial[ACTUATOR],
            exchange_name=exchange_name,
        ),
        ll_reasoners=SupportLLReasoner(
            module_id=module_name(LLREASONER, 0),
            initial_state=initial[LLREASONER],
        ),
        knowledge=ClosedWorldKnowledge(
            module_id=module_name(KNOWLEDGE, 0),
            initial_state=initial[KNOWLEDGE],
        ),
        hl_reasoners=DeliberativeBDIReasoner(
            module_id=module_name(HLREASONER, 0),
            init_kwargs={
                "seed": seeder.hl_reasoner,
                "num_blocks": treatment["num_blocks"],
                "table_len": treatment["table_len"],
                "planner_timeout": PLANNER_TIMEOUT,
                "fixed_goal": {"top": treatment["top"], "bottom": treatment["bottom"]},
            },
            initial_state=initial[HLREASONER],
        ),
        requirements_path=Path(__file__).resolve().with_name("requirements.txt"),
        extra_runtime_sources=Path(cast(str, mha_exp_common.__file__)).resolve().parent,
    )

    record = RECORD == "all" or (RECORD == "first" and run == 0)
    orchestrator.add_environment(
        base=TestEnvironment(
            init_state={
                "seed": treatment["seed"],
                "record": record,
                "table_len": treatment["table_len"],
                "num_blocks": treatment["num_blocks"],
                "treatment": treatment,
                "world": [],
                "initial_world": [],
                "observations": 0,
                "actions": 0,
                "illegal_actions": 0,
                "close_requests": 0,
                "closed": False,
            }
        ),
        env_id=env_name(run, "2_3"),
        exec_duration=DURATION + 30.0,
        requirements_path=Path(__file__).resolve().with_name("requirements-env.txt"),
        exchange_name=exchange_name,
        extra_runtime_sources=[
            Path(cast(str, mha_exp_common.__file__)).resolve().parent,
            bw_runtime_source,
        ],
    )
    run_agent_id = agent_name(run, "2_3")
    run_env_id = env_name(run, "2_3")
    cleanup_run_containers(run_agent_id, run_env_id, phase="before_run")
    cleanup_run_images(run_agent_id, run_env_id, phase="before_run")
    try:
        orchestrator.run(mhagenta_version=mha_version, local_build=mha_root, force_run=True)
    finally:
        cleanup_run_containers(run_agent_id, run_env_id, phase="after_run")
        cleanup_run_images(run_agent_id, run_env_id, phase="after_run")
    result = check_results(*read_run_evidence(exp_path, run),
                           expected_treatment=treatment, verbose=VERBOSE)
    print(f"Results: {result}")
    return result


def _print_batch_summary(exp_path: Path) -> None:
    states = gather_states(exp_path, False, no_warnings=True)
    records: list[tuple[dict[str, Any], bool]] = []
    for agent_id, modules in states.items():
        if not agent_id.startswith("exp_agent2_3_"):
            continue
        hl = modules.get(module_name(HLREASONER, 0), {})
        run = hl.get("run") if isinstance(hl, dict) else None
        if not isinstance(run, dict):
            continue
        log_path = (exp_path / agent_id).with_suffix(".log")
        logs = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
        run_id = int(agent_id.rsplit("_", 1)[1])
        records.append((run, not _result_errors(*read_run_evidence(exp_path, run_id), treatment_for_run(run_id))))
    if not records:
        return

    complete = sum(run.get("phase") == "complete" for run, _ in records)
    failed = sum(run.get("phase") == "failed" for run, _ in records)
    passed = sum(passed for _, passed in records)
    plan_lengths = [
        run["planner"]["length"]
        for run, _ in records
        if isinstance(run.get("planner"), dict) and isinstance(run["planner"].get("length"), int)
    ]
    execution_lengths = [len(run.get("executions", [])) for run, _ in records]
    failure_codes = Counter(
        run["failure"]["code"]
        for run, _ in records
        if isinstance(run.get("failure"), dict) and isinstance(run["failure"].get("code"), str)
    )
    print("2-3-BW finite deliberative summary")
    print(f"  discovered / complete / failed / accepted: {len(records)} / {complete} / {failed} / {passed}")
    print(f"  mean LPG plan length: {fmean(plan_lengths):.3f}" if plan_lengths else "  mean LPG plan length: n/a")
    print(f"  mean executed prefix: {fmean(execution_lengths):.3f}")
    print(f"  failures by code: {dict(sorted(failure_codes.items()))}")


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 50,
    exp_path: str | os.PathLike[str] = ".",
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
    process_only: bool = False,
) -> None:
    """Run the experiment batch or process its existing results."""

    run_ids, _ = normalize_runs(runs)
    root = Path(exp_path).resolve()
    expected = [{"execution_id": f"run-{run}", "run_id": run,
                 "factors": treatment_for_run(run)}
                for run in run_ids]
    primary_error: BaseException | None = None
    try:
        results_available = run_experiment_batch(
            experiment_id="2-3", title="DELIBERATIVE BDI BLOCKS WORLD",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=run_experiment, process_only=process_only,
            stop_on_error=True, cleanup_before_run=False,
        )
        if results_available:
            _print_batch_summary(root)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            process_execution_metrics(root, expected_executions=expected)
        except Exception as reporting_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Execution-metrics processing also failed: {type(reporting_error).__name__}"
            )
