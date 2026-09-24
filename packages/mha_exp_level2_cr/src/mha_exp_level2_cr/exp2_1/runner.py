from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.util import find_spec
import os
from pathlib import Path
import time
from threading import Timer
import signal
from typing import Any, cast

from mhagenta import ActionStatus, Observation, Orchestrator
from mhagenta.bases import LLReasonerBase
from mhagenta.defaults.communication import RMQActuatorBase, RMQPerceptorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import ActuatorState, LLState, PerceptorState

import mha_exp_common
from mha_exp_common.batch import normalize_runs, run_batch as run_experiment_batch
from mha_exp_common.defaults import DEFAULT_MHAGENTA_VERSION
from mha_exp_common.names import ACTUATOR, LLREASONER, PERCEPTOR
from mha_exp_common.utils import Seeder, agent_name, env_name, gather_states, module_name

from .policy import ACTIONS, RESET_ACTION, ReactiveCrafterPolicy
from .reporting import process_execution_metrics
from .treatment import treatment_for_run


DURATION = 600.0
STARTUP_DELAY = 20.0
ENVIRONMENT_OVERRUN = 15.0
MAX_EPISODE_LEN = 1000
RECORD = "first"
VERBOSE = True
DIAMOND_PATH_ACHIEVEMENTS = (
    "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
    "make_stone_pickaxe", "collect_coal", "collect_iron", "place_furnace",
    "make_iron_pickaxe", "collect_diamond",
)
_ACHIEVEMENT_RANK = {
    name: rank for rank, name in enumerate(DIAMOND_PATH_ACHIEVEMENTS)
}
K_ACTION = "action"
K_OBSERVATION = "observation"
K_DONE = "done"
K_ILLEGAL_ACTION = "illegal_action"
K_RESET = "reset"
K_CLOSED = "closed"
K_REQUEST_ID = "request_id"
K_EPISODE_ID = "episode_id"
A_CLOSE = "close"
_RUNTIME_ERROR_MARKERS = (
    "[error]", "[critical]", "traceback", "exceptiongroup",
    "caught exception", "failed to save state", "could not send message",
)


def _environment_reasoner_ids(state: Any) -> tuple[str, str]:
    environment = state.directory.external.environment
    if environment is None:
        raise RuntimeError("Expected one external environment")
    return environment.address["env_id"], state.directory.internal.ll_reasoning[0].module_id


def _highest_path_achievement(value: Any) -> str | None:
    """Return the highest positive diamond-path achievement in Crafter counts."""
    if not isinstance(value, Mapping):
        raise ValueError("Crafter achievements must be a mapping")
    highest = None
    for name in DIAMOND_PATH_ACHIEVEMENTS:
        count = value.get(name)
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid Crafter achievement count for {name!r}")
        if count > 0:
            highest = name
    return highest


class CrafterEnvironment(MHAEnvBase):
    """Crafter bridge that owns native conversion and achievement evidence."""

    def __init__(self, init_state: dict[str, Any]) -> None:
        self._seed = int(init_state.pop("seed"))
        self._record = bool(init_state.pop("record", False))
        self._target_achievement = str(init_state["target_achievement"])
        # Serialized on Windows and reconstructed in a Linux container.
        self._artifact_root = str(
            init_state.pop("artifact_root", f"/{Orchestrator.SAVE_SUBDIR}")
        )
        self._env: Any = None
        self._action_ids: dict[str, int] = {}
        self._started = time.monotonic()
        self._achievement_counts: dict[str, int] = {}
        self._stop_timer: Timer | None = None
        super().__init__(init_state)
        self._build_env()

    def _build_env(self) -> None:
        from mha_env_crafter import CrafterEnv, Recorder
        environment = CrafterEnv(
            seed=self._seed, length=MAX_EPISODE_LEN, no_mobs=True,
            symbolic=True, daylight_effects=False,
        )
        if self._record:
            environment = Recorder(
                environment, Path(self._artifact_root) / "videos",
                save_stats=False, save_episode=False,
                video_size=(144, 144), video_fps=2,
            )
        if tuple(environment.action_names) != ACTIONS:
            raise RuntimeError("Crafter action order does not match the experiment contract")
        self._env = environment
        self._action_ids = {name: index for index, name in enumerate(ACTIONS)}
        self._env.reset()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        state["_action_ids"] = {}
        state["_stop_timer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._started = time.monotonic()
        self._achievement_counts = {}
        self._build_env()

    def _trace(self, state: dict[str, Any], **row: Any) -> None:
        state["execution_trace"].append({
            "event_id": state["next_event_id"],
            "episode_id": state["episode_id"],
            "elapsed_seconds": max(0.0, time.monotonic() - self._started),
            **row,
        })
        state["next_event_id"] += 1

    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs: Any
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
        return state, {K_OBSERVATION: list(self._env.symbolic_observation())}

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs: Any
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
        action = kwargs.get(K_ACTION)
        if action == RESET_ACTION:
            request_id = kwargs.get(K_REQUEST_ID)
            episode_id = kwargs.get(K_EPISODE_ID)
            if type(request_id) is not int or episode_id != state["episode_id"]:
                raise ValueError("Reset request correlation is invalid")
            next_episode_id = state["episode_id"] + 1
            self._trace(
                state, event_kind="reset", action=RESET_ACTION,
                request_id=request_id, reset_cause=kwargs.get("reset_cause"),
                policy_reason=kwargs.get("policy_reason"),
                next_episode_id=next_episode_id,
            )
            self._env.reset()
            self._achievement_counts = {}
            state["episode_id"] = next_episode_id
            return state, {
                K_ACTION: RESET_ACTION,
                K_DONE: False,
                K_ILLEGAL_ACTION: False,
                K_RESET: True,
                K_REQUEST_ID: request_id,
                K_EPISODE_ID: episode_id,
                "next_episode_id": next_episode_id,
                "reset_cause": kwargs.get("reset_cause"),
            }
        if action == A_CLOSE:
            self._trace(
                state, event_kind="close", action=A_CLOSE, request_id=None,
                reset_cause=None, next_episode_id=None,
            )
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            video_dir = Path(self._artifact_root) / "videos"
            if self._record and any(video_dir.glob("*.mp4")):
                state["video_path"] = "videos"
            state["closed"] = True
            # Let this container terminate itself cleanly after the callback returns.
            # This avoids a race between agent shutdown and orchestrator Docker stop.
            self._stop_timer = Timer(0.05, os.kill, args=(os.getpid(), signal.SIGTERM))
            self._stop_timer.daemon = True
            self._stop_timer.start()
            return state, {K_ACTION: A_CLOSE, K_CLOSED: True}
        if type(action) is not str or action not in self._action_ids:
            raise ValueError(f"Invalid readable environment action: {action!r}")
        request_id = kwargs.get(K_REQUEST_ID)
        episode_id = kwargs.get(K_EPISODE_ID)
        if type(request_id) is not int or episode_id != state["episode_id"]:
            raise ValueError("Native action request correlation is invalid")
        native_action = self._action_ids[action]
        _, _, done, info = self._env.step(native_action)
        illegal = info.get(K_ILLEGAL_ACTION)
        if type(illegal) is not bool:
            raise ValueError("Crafter info must contain a strict Boolean illegal_action")
        highest = _highest_path_achievement(info.get("achievements"))
        current = state["highest_diamond_path_achievement"]
        if highest is not None and (
            current is None or _ACHIEVEMENT_RANK[highest] > _ACHIEVEMENT_RANK[current]
        ):
            state["highest_diamond_path_achievement"] = highest
        achievements = info.get("achievements")
        if not isinstance(achievements, Mapping):
            raise ValueError("Crafter info must contain achievement counts")
        newly_achieved = sorted(
            name for name, count in achievements.items()
            if isinstance(count, int) and count > self._achievement_counts.get(str(name), 0)
        )
        self._achievement_counts = {
            str(name): int(count) for name, count in achievements.items()
            if isinstance(count, int)
        }
        target_achieved = int(achievements.get(self._target_achievement, 0)) > 0
        inventory = info.get("inventory")
        if not isinstance(inventory, Mapping):
            raise ValueError("Crafter info must contain inventory")
        selected_inventory = {
            key: int(inventory.get(key, 0)) for key in (
                "health", "food", "drink", "energy", "wood", "stone", "coal",
                "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
            )
        }
        player_pos = info.get("player_pos")
        try:
            player_position = [int(value) for value in player_pos]
        except (TypeError, ValueError):
            player_position = []
        if len(player_position) != 2:
            raise ValueError("Crafter info must contain player_pos")
        self._trace(
            state, event_kind="native_action", action=action, request_id=request_id,
            reset_cause=None, next_episode_id=None, illegal_action=illegal,
            environment_done=bool(done), player_pos=player_position,
            inventory=selected_inventory, newly_achieved=newly_achieved,
            highest_milestone=state["highest_diamond_path_achievement"],
            target_achievement=self._target_achievement,
            target_achieved=target_achieved,
        )
        state["native_actions"] += 1
        state["illegal_actions"] += int(illegal)
        state["target_achieved"] = target_achieved
        if target_achieved:
            state["terminal_reason"] = "target_achieved"
        elif bool(done) and selected_inventory["health"] <= 0:
            state["terminal_reason"] = "death"
        elif state["native_actions"] >= int(state["action_budget"]):
            state["terminal_reason"] = "action_budget_exhausted"
        elif bool(done):
            state["terminal_reason"] = "environment_terminal"
        return state, {
            K_ACTION: action,
            K_DONE: bool(done),
            K_ILLEGAL_ACTION: illegal,
            K_RESET: False,
            K_REQUEST_ID: request_id,
            K_EPISODE_ID: episode_id,
            "target_achievement": self._target_achievement,
            "target_achieved": target_achieved,
            "highest_milestone": state["highest_diamond_path_achievement"],
            "native_actions": state["native_actions"],
            "terminal_reason": state["terminal_reason"],
        }


class CrafterPerceptor(RMQPerceptorBase):
    """Forward current symbolic fluent observations to the reasoner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: PerceptorState) -> PerceptorState:
        self._env_id, self._reasoner_id = _environment_reasoner_ids(state)
        return state

    def on_request(self, state: PerceptorState, sender: str,
                   **kwargs: Any) -> PerceptorState:
        state["requests"] += 1
        self.observe(self._env_id)
        return state

    def on_observation(self, state: PerceptorState, env_id: str,
                       **kwargs: Any) -> PerceptorState:
        content = kwargs.get(K_OBSERVATION)
        if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
            raise ValueError("Environment returned an invalid symbolic observation")
        state["observations"] += 1
        state.outbox.send_observation(
            self._reasoner_id,
            Observation(content, observation_type="crafter-symbolic"),
        )
        return state


class CrafterActuator(RMQActuatorBase):
    """Forward readable scientific actions and reset controls to Crafter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env_id = ""
        self._reasoner_id = ""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        self._env_id, self._reasoner_id = _environment_reasoner_ids(state)
        return state

    def on_request(self, state: ActuatorState, sender: str,
                   **kwargs: Any) -> ActuatorState:
        action = kwargs.get(K_ACTION)
        if type(action) is not str or action not in (*ACTIONS, RESET_ACTION):
            raise ValueError(f"Invalid readable actuator action: {action!r}")
        state["requests"] += 1
        self.act(self._env_id, **kwargs)
        return state

    def on_status(self, state: ActuatorState, env_id: str,
                  **kwargs: Any) -> ActuatorState:
        if kwargs.get(K_CLOSED):
            return state
        state["statuses"] += 1
        state.outbox.send_status(self._reasoner_id, ActionStatus(dict(kwargs)))
        return state

    def on_last(self, state: ActuatorState) -> ActuatorState:
        self.act(self._env_id, action=A_CLOSE)
        return state


class CrafterReactiveReasoner(LLReasonerBase):
    """Apply the world-stateless policy and maintain the reactive loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._policy = ReactiveCrafterPolicy()
        self._actuator_id = ""
        self._perceptor_id = ""

    def on_init(self, **kwargs: Any) -> None:
        self._policy = ReactiveCrafterPolicy(kwargs.get("seed"))

    def on_first(self, state: LLState) -> LLState:
        self._actuator_id = state.directory.internal.actuation[0].module_id
        self._perceptor_id = state.directory.internal.perception[0].module_id
        state.outbox.request_observation(self._perceptor_id)
        return state

    def on_last(self, state: LLState) -> LLState:
        """Persist elapsed execution time and distinguish timeout from external stop."""
        state["execution_seconds"] = state.time
        if state["phase"] == "active":
            state["phase"] = "complete"
            state["terminal_reason"] = "timeout" if state.time >= DURATION else "external_stop"
        return state

    def _request_action(self, state: LLState, *, action: str, reason: str,
                        reset_cause: str | None = None) -> None:
        request_id = state["next_request_id"]
        state["next_request_id"] += 1
        state["pending_request"] = {
            K_REQUEST_ID: request_id, K_EPISODE_ID: state["episode_id"],
            K_ACTION: action, "reset_cause": reset_cause,
        }
        state["decision_trace"].append({
            "decision_id": len(state["decision_trace"]),
            K_REQUEST_ID: request_id, K_EPISODE_ID: state["episode_id"],
            "event_kind": "reset" if action == RESET_ACTION else "native_action",
            K_ACTION: action, "reason": reason, "reset_cause": reset_cause,
        })
        state.outbox.request_action(
            self._actuator_id, action=action, request_id=request_id,
            episode_id=state["episode_id"], reset_cause=reset_cause,
            policy_reason=reason,
        )

    def _request_reset(self, state: LLState, *, cause: str, reason: str) -> None:
        self._request_action(state, action=RESET_ACTION, reason=reason, reset_cause=cause)

    def on_observation(
        self,
        state: LLState,
        sender: str,
        observation: Observation,
        **kwargs: Any,
    ) -> LLState:
        state["observations"] += 1
        content = observation.content
        if not isinstance(content, list):
            raise ValueError("Reasoner received a non-list symbolic observation")
        decision = self._policy.choose_action(content)
        if decision.action == RESET_ACTION:
            self._request_reset(
                state, cause="policy_requested", reason=decision.reason or "policy_requested"
            )
            return state
        if type(decision.action) is not str or decision.action not in ACTIONS:
            raise ValueError(f"Policy returned an invalid action: {decision.action!r}")
        state["action_requests"] += 1
        state["last_decision"] = {
            "action": decision.action, "reason": decision.reason or "",
        }
        self._request_action(
            state, action=decision.action, reason=decision.reason or ""
        )
        return state

    def on_action_status(
        self,
        state: LLState,
        sender: str,
        action_status: ActionStatus,
        **kwargs: Any,
    ) -> LLState:
        state["action_statuses"] += 1
        status = action_status.status
        if not isinstance(status, dict):
            raise ValueError("Reasoner received a non-dict action status")
        pending = state["pending_request"]
        if not isinstance(pending, Mapping):
            raise ValueError("Reasoner received a status without a pending request")
        if (status.get(K_REQUEST_ID) != pending.get(K_REQUEST_ID)
                or status.get(K_EPISODE_ID) != pending.get(K_EPISODE_ID)):
            raise ValueError("Reasoner received an uncorrelated action status")
        state["pending_request"] = None
        if status.get(K_RESET):
            next_episode_id = status.get("next_episode_id")
            if next_episode_id != state["episode_id"] + 1:
                raise ValueError("Reasoner received an invalid reset boundary")
            state["episode_id"] = next_episode_id
            state.outbox.request_observation(self._perceptor_id)
            return state
        action = status.get(K_ACTION)
        if type(action) is not str or action not in ACTIONS:
            raise ValueError("Reasoner received an invalid readable action status")
        if type(status.get(K_ILLEGAL_ACTION)) is not bool:
            raise ValueError("Reasoner received an invalid illegal-action status")
        done = status.get(K_DONE, False)
        if type(done) is not bool:
            raise ValueError("Reasoner received a non-Boolean done status")
        target_achieved = status.get("target_achieved")
        native_actions = status.get("native_actions")
        if type(target_achieved) is not bool or type(native_actions) is not int:
            raise ValueError("Reasoner received invalid authoritative treatment status")
        state["target_achieved"] = target_achieved
        state["highest_milestone"] = status.get("highest_milestone")
        state["native_actions"] = native_actions
        terminal_reason = status.get("terminal_reason")
        if target_achieved or done or native_actions >= state["action_budget"]:
            state["phase"] = "complete"
            state["terminal_reason"] = (
                terminal_reason
                or ("target_achieved" if target_achieved else "environment_terminal")
            )
            state.outbox.terminate_agent(f"2-1-CR {state['terminal_reason']}")
            return state
        state.outbox.request_observation(self._perceptor_id)
        return state


def _counter(state: Mapping[str, Any], key: str) -> int:
    value = state[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def check_results(
    agent_states: Mapping[str, dict[str, Any]] | None,
    environment_state: Mapping[str, Any] | None,
    agent_logs: Sequence[str] | None,
    environment_logs: Sequence[str] = (),
    verbose: bool = False,
) -> bool:
    """Validate the reactive loop and environment-owned scientific result."""
    valid = True
    if agent_logs is None:
        print("Missing required agent log")
        agent_logs = ()
        valid = False
    for source, lines in (("agent", agent_logs), ("environment", environment_logs)):
        for line in lines:
            if any(marker in line.lower() for marker in _RUNTIME_ERROR_MARKERS):
                print(f"Runtime error found in {source} log: {line.rstrip()}")
                valid = False
    if not isinstance(agent_states, Mapping):
        print("Missing saved agent state")
        return False
    required = {
        module_name(PERCEPTOR, 0), module_name(ACTUATOR, 0),
        module_name(LLREASONER, 0),
    }
    if missing := required - agent_states.keys():
        print(f"Missing saved module states: {sorted(missing)}")
        return False
    if not isinstance(environment_state, Mapping):
        print("Missing saved environment state")
        return False
    perceptor = agent_states[module_name(PERCEPTOR, 0)]
    actuator = agent_states[module_name(ACTUATOR, 0)]
    reasoner = agent_states[module_name(LLREASONER, 0)]
    try:
        counts = {
            "perceptor requests": _counter(perceptor, "requests"),
            "perceptor observations": _counter(perceptor, "observations"),
            "reasoner observations": _counter(reasoner, "observations"),
            "scientific actions": _counter(reasoner, "action_requests"),
            "actuator requests": _counter(actuator, "requests"),
            "actuator statuses": _counter(actuator, "statuses"),
            "reasoner statuses": _counter(reasoner, "action_statuses"),
            "native actions": _counter(environment_state, "native_actions"),
        }
        illegal_actions = _counter(environment_state, "illegal_actions")
        highest = environment_state["highest_diamond_path_achievement"]
        decision = reasoner["last_decision"]
        valid_decision = (
            isinstance(decision, Mapping)
            and set(decision) == {"action", "reason"}
            and type(decision["action"]) is str
            and decision["action"] in ACTIONS
            and type(decision["reason"]) is str
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Invalid saved-state contract: {exc}")
        return False
    for label, count in counts.items():
        if count <= 0:
            print(f"No {label} recorded")
            valid = False
    if not valid_decision:
        print("Reasoner recorded an invalid readable decision")
        valid = False
    if illegal_actions:
        print("Environment reported illegal actions")
        valid = False
    if highest is not None and (type(highest) is not str or highest not in DIAMOND_PATH_ACHIEVEMENTS):
        print("Invalid highest diamond-path achievement")
        valid = False
    chains = (
        ("perceptor request/observation", "perceptor requests", "perceptor observations"),
        ("perceptor/reasoner observation", "perceptor observations", "reasoner observations"),
        ("scientific action/environment", "scientific actions", "native actions"),
        ("actuator request/status", "actuator requests", "actuator statuses"),
        ("actuator/reasoner status", "actuator statuses", "reasoner statuses"),
    )
    for label, upstream, downstream in chains:
        if not 0 <= counts[upstream] - counts[downstream] <= 1:
            print(f"{label} counts diverged")
            valid = False
    if verbose:
        print(
            f"Reactive loop: observations={counts['reasoner observations']}, "
            f"actions={counts['scientific actions']}, "
            f"statuses={counts['reasoner statuses']}"
        )
        print(f"Highest diamond-path achievement: {highest}")
        print(f"Crafter-reported illegal actions: {illegal_actions}")
    return valid


def _local_mhagenta_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        workspace_file = parent / "pyproject.toml"
        if (workspace_file.is_file()
                and "[tool.uv.workspace]" in workspace_file.read_text(encoding="utf-8")):
            root = (parent.parent / "mhagenta").resolve()
            version_file = root / "pyproject.toml"
            if (version_file.is_file()
                    and 'version = "1.4.12"' in version_file.read_text(encoding="utf-8")):
                return root
            raise RuntimeError(f"Expected local MHAgentA 1.4.12 at {root}")
    raise RuntimeError("Could not locate mhagenta-experiments workspace root")


def _reasoner_initial_state(
    treatment: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if treatment is None:
        treatment = treatment_for_run(0)
    return {**treatment, "observations": 0, "action_requests": 0,
            "action_statuses": 0, "last_decision": None,
            "next_request_id": 0, "episode_id": 0,
            "pending_request": None, "decision_trace": [],
            "phase": "active", "terminal_reason": None,
            "execution_seconds": 0.0,
            "target_achieved": False, "highest_milestone": None,
            "native_actions": 0}


def run_experiment(
    run: int,
    exp_path: str | os.PathLike[str],
    mha_version: str = DEFAULT_MHAGENTA_VERSION,
) -> bool:
    """Run and validate one 2-1-CR agent/environment pair."""
    exp_path = Path(exp_path).resolve()
    crafter_spec = find_spec("mha_env_crafter")
    if crafter_spec is None or crafter_spec.origin is None:
        raise ImportError("Could not locate mha_env_crafter runtime sources")
    crafter_source = Path(crafter_spec.origin).resolve().parent
    common_source = Path(cast(str, mha_exp_common.__file__)).resolve().parent
    seeder = Seeder(run)
    treatment = treatment_for_run(run)
    exchange_name = "mhagenta"
    run_agent_id = agent_name(run, "2_1")
    run_env_id = env_name(run, "2_1")
    orchestrator = Orchestrator(
        save_dir=exp_path, step_frequency=0.0, control_frequency=0.0,
        status_frequency=5.0, agent_start_delay=STARTUP_DELAY,
        exec_duration=DURATION, save_format="json",
        stop_on_agents_term=True,
        log_level=Orchestrator.INFO, save_logs=True, no_stdout_logs=False,
        mas_rmq_uri="localhost:5672", mas_rmq_exchange_name=exchange_name,
    )
    orchestrator.add_agent(
        agent_id=run_agent_id,
        perceptors=CrafterPerceptor(
            module_id=module_name(PERCEPTOR, 0), exchange_name=exchange_name,
            initial_state={"requests": 0, "observations": 0}),
        actuators=CrafterActuator(
            module_id=module_name(ACTUATOR, 0), exchange_name=exchange_name,
            initial_state={"requests": 0, "statuses": 0}),
        ll_reasoners=CrafterReactiveReasoner(
            module_id=module_name(LLREASONER, 0),
            init_kwargs={"seed": seeder.ll_reasoner},
            initial_state=_reasoner_initial_state(treatment)),
        extra_runtime_sources=common_source,
    )
    existing_runs = any(exp_path.glob("exp_agent2_1_*"))
    record = RECORD == "all" or (RECORD == "first" and not existing_runs)
    orchestrator.add_environment(
        base=CrafterEnvironment({
            "seed": seeder.environment, "record": record,
            **treatment,
            "artifact_root": f"/{Orchestrator.SAVE_SUBDIR}",
            "native_actions": 0, "illegal_actions": 0,
            "highest_diamond_path_achievement": None,
            "next_event_id": 0, "episode_id": 0, "execution_trace": [],
            "closed": False, "video_path": None, "video_error": None,
            "target_achieved": False, "terminal_reason": None,
        }),
        env_id=run_env_id,
        exec_duration=DURATION + ENVIRONMENT_OVERRUN,
        requirements_path=Path(__file__).resolve().with_name("requirements-env.txt"),
        exchange_name=exchange_name,
        extra_runtime_sources=[common_source, crafter_source],
    )
    runtime_version = (DEFAULT_MHAGENTA_VERSION
                       if mha_version in {"", "latest", DEFAULT_MHAGENTA_VERSION}
                       else mha_version)
    orchestrator.run(
        mhagenta_version=runtime_version,
        local_build=_local_mhagenta_root(),
        force_run=True,
    )

    states = gather_states(exp_path, False, no_warnings=True)
    agent_log_path = exp_path / f"{run_agent_id}.log"
    environment_log_path = exp_path / f"{run_env_id}.log"
    agent_logs = (agent_log_path.read_text(encoding="utf-8").splitlines(keepends=True)
                  if agent_log_path.is_file() else None)
    environment_logs = (
        environment_log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if environment_log_path.is_file() else ())
    environment_states = states.get(run_env_id, {})
    result = check_results(
        states.get(run_agent_id, {}),
        environment_states.get(run_env_id),
        agent_logs,
        environment_logs,
        verbose=VERBOSE,
    )
    print(f"Results: {result}")
    return result


def run_batch(
    runs: int | tuple[int, int] | Sequence[int] = 40,
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
        run_experiment_batch(
            experiment_id="2-1-CR", title="REACTIVE SYMBOLIC CRAFTER AGENT",
            runs=run_ids, exp_path=exp_path, mha_version=mha_version,
            runner=run_experiment, process_only=process_only,
            stop_on_error=True,
        )
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
