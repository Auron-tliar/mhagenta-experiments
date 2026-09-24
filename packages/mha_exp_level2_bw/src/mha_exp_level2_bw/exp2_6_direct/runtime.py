"""Eight-module direct AchieveOn learning with CPU actors and a GPU learner.

Only Learner performs optimizer work. Completed training episodes traverse
Knowledge and Memory before Learner may consume them. Model installation is
acknowledged through GoalGraph before the high level dispatches another task.
"""

from collections import deque
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib import import_module
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from mhagenta import ActionStatus, Belief, Goal, Observation, Orchestrator
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LLReasonerBase, LearnerBase, MemoryBase
from mhagenta.defaults.communication.rabbitmq import RMQActuatorBase, RMQPerceptorBase
from mhagenta.states import ActuatorState, GoalGraphState, HLState, KnowledgeState, LLState, LearnerState, MemoryState, PerceptorState

from mha_exp_level2_bw.achieve_on.policy import (
    ARCHITECTURE, build_network, checkpoint_payload, condition_goal, goal_succeeded, infer, warm_start,
)
from mha_exp_level2_bw.exp2_5.contracts import GoalSpec, TransferSpec, beliefs_to_facts, block_names, location_names
from mha_exp_level2_bw.exp2_5.grounding import ground_observation, transfer_succeeded
from mha_exp_level2_bw.exp2_5.policy import artifact_paths, greedy_inference, legal_action_indices, load_policy_checkpoint
from mha_exp_level2_bw.exp2_5.resized import load_frozen


def digest(value: Any) -> str:
    """Hash a JSON-native protocol value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def one(state: Any, name: str) -> str:
    """Resolve the experiment's one module of the requested kind."""
    entries = list(getattr(state.directory.internal, name))
    if len(entries) != 1:
        raise ValueError(f"Expected one {name} module.")
    return entries[0].module_id


def require(condition: bool, message: str) -> None:
    """Fail visibly on a violated protocol invariant."""
    if not condition:
        raise ValueError(message)


def folder(kwargs: dict) -> Path:
    """Use the container's persisted output directory, or a test directory."""
    path = Path(kwargs.get("output_dir", f"/{Orchestrator.SAVE_SUBDIR}"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def production_planner(*, table_len: int = 5, num_blocks: int = 8) -> Any:
    """Create the existing validated Transfer planner inside the runtime HLR."""
    from mha_exp_level2_bw.exp2_5.planning import PlanningService
    return PlanningService(domain_path=artifact_paths()[0].parents[2] / "blocksworld-transfer-domain.pddl",
                           blocks=block_names(num_blocks), locations=location_names(table_len))


def adoption_evidence(results: list[dict], milestone: int, count: int) -> dict:
    """Require perfect paired probes and strictly fewer joint-success actions."""
    candidate = {r["case_index"]: r for r in results if r["mode"] == "probe" and r["probe"] == milestone}
    baseline = {r["case_index"]: r for r in results if r["mode"] == "transfer"}
    pairs = [(r, baseline[i]) for i, r in candidate.items() if i in baseline]
    successes = sum(a["success"] for a, _ in pairs)
    saved = sum(len(b["actions"]) - len(a["actions"]) for a, b in pairs if a["success"] and b["success"])
    identities = {(a["revision"], a["model_sha256"]) for a, _ in pairs}
    passed = (milestone >= 100 and len(pairs) == count and successes == count and saved > 0
              and len(identities) == 1 and all(a["seed"] == b["seed"] and a["goal"] == b["goal"] for a, b in pairs))
    revision, checksum = next(iter(identities)) if len(identities) == 1 else (None, None)
    return {"milestone": milestone, "cases": len(pairs), "successes": successes,
            "actions_saved": saved, "passed": passed, "revision": revision, "sha256": checksum}


class Perceptor(RMQPerceptorBase):
    """Route correlated numeric observations from the external environment."""

    def on_request(self, state: PerceptorState, sender: str, **kwargs: Any) -> PerceptorState:
        require(sender == one(state, "ll_reasoning") and state.pending is None, "Overlapping observation request.")
        state.pending = kwargs["request_id"]
        state.requests += 1
        self.observe(state.directory.external.environment.address["env_id"], **kwargs)
        return state

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs: Any) -> PerceptorState:
        require(env_id == state.directory.external.environment.address["env_id"] and kwargs["request_id"] == state.pending,
                "Observation identity mismatch.")
        state.pending = None
        state.responses += 1
        state.outbox.send_observation(one(state, "ll_reasoning"), Observation(kwargs["observation"]), request_id=kwargs["request_id"])
        return state


class Actuator(RMQActuatorBase):
    """Route resets, atomic actions, and acknowledged environment closure."""

    def on_request(self, state: ActuatorState, sender: str, **kwargs: Any) -> ActuatorState:
        require(sender == one(state, "ll_reasoning") and state.pending is None, "Overlapping action request.")
        state.pending = kwargs["request_id"]
        state.requests += 1
        self.act(state.directory.external.environment.address["env_id"], **kwargs)
        return state

    def on_status(self, state: ActuatorState, env_id: str, **kwargs: Any) -> ActuatorState:
        require(env_id == state.directory.external.environment.address["env_id"] and kwargs["request_id"] == state.pending,
                "Action status identity mismatch.")
        state.pending = None
        state.responses += 1
        state.outbox.send_status(one(state, "ll_reasoning"), ActionStatus(kwargs))
        return state


class GoalGraph(GoalGraphBase):
    """Keep one active goal and route model acknowledgements and terminal facts."""

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> GoalGraphState:
        require(len(goals) == 1, "Expected one goal envelope.")
        extras = goals[0].extras
        if sender == one(state, "hl_reasoning"):
            require(state.pending is None, "A goal is already active.")
            state.pending = extras["task"]["id"]
            state.requests += 1
            state.outbox.send_goals(one(state, "ll_reasoning"), goals)
        else:
            require(sender == one(state, "ll_reasoning"), "Unknown goal sender.")
            if "model_ready" not in extras:
                require(extras["result"]["id"] == state.pending, "Terminal goal identity mismatch.")
                state.pending = None
                state.responses += 1
            else:
                require(state.pending is None, "Model update during an active goal.")
            state.outbox.send_goals(one(state, "hl_reasoning"), goals)
        return state


class LLReasoner(LLReasonerBase):
    """Execute a complete On goal atomically with a CPU inference snapshot."""

    def on_init(self, **kwargs: Any) -> None:
        self.torch = import_module("torch")
        self.torch.set_num_threads(1)
        self.frozen_policy = kwargs.get("frozen_policy")
        self.dimensions = kwargs.get("dimensions", {"table_len": 5, "num_blocks": 8})
        require(self.frozen_policy == "transfer" or self.dimensions == {"table_len": 5, "num_blocks": 8},
                "Only frozen Transfer supports resized worlds.")
        if kwargs.get("policy_family") is None:
            self.teacher, reference = load_frozen(self.torch, **self.dimensions)
        else:
            require(self.frozen_policy == "transfer", "Policy families require frozen Transfer.")
            from mha_exp_level2_bw.exp2_5.family import load
            self.teacher, reference = load(self.torch, kwargs["policy_family"], **self.dimensions)
        self.teacher_sha256 = reference["checkpoint_sha256"]
        if kwargs.get("transfer_sha256"):
            require(self.teacher_sha256 == kwargs["transfer_sha256"], "Runtime Transfer differs from the prepared artifact.")
        self.model = self.pretrained = None
        self.pretrained_sha256 = None
        if self.frozen_policy != "transfer":
            self.model = build_network(self.torch).eval()
            source = Path(kwargs["pretrained_path"]) if "pretrained_path" in kwargs else Path(import_module("bw_atomic_input").__file__).with_name("pretrained.pt")
            self.pretrained = build_network(self.torch).eval()
            payload = self.torch.load(source, map_location="cpu", weights_only=True)
            require(payload["architecture"] == ARCHITECTURE, "Pretrained architecture mismatch.")
            self.pretrained.load_state_dict(payload["model_state_dict"], strict=True)
            self.pretrained_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            for model in (self.model, self.pretrained):
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
        self.rng = np.random.default_rng(kwargs["seed"])
        self.cap = kwargs["action_cap"]
        self.training_episodes = kwargs["training_episodes"]
        self._observation = None
        self._adopted = None

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs: Any) -> LLState:
        require(sender == one(state, "learning") and state.active is None, "Unsafe model installation.")
        require(digest(model["weights"]) == model["sha256"] and model["revision"] > state.revision, "Invalid model publication.")
        tensors = {key: self.torch.tensor(value, dtype=self.torch.float32) for key, value in model["weights"].items()}
        self.model.load_state_dict(tensors, strict=True)
        state.revision, state.model_sha256 = model["revision"], model["sha256"]
        state.model_installs += 1
        state.outbox.send_goal_update(one(state, "goals"), [Goal([], model_ready={
            "revision": state.revision, "sha256": state.model_sha256, "learning_id": model["learning_id"]})])
        return state

    def _act(self, state: LLState, action: int | str, **kwargs: Any) -> None:
        state.requests += 1
        state.pending = f"action-{state.requests}"
        state.outbox.request_action(one(state, "actuation"), action=action, request_id=state.pending, **kwargs)

    def on_goal_update(self, state: LLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> LLState:
        require(sender == one(state, "goals") and state.active is None and state.pending is None, "Unsafe goal activation.")
        self._episode_started = time.monotonic()
        state.active = dict(goals[0].extras["task"])
        self._deadline = (self._episode_started + state.active["remaining_seconds"]
                          if "remaining_seconds" in state.active else None)
        state.rows, state.actions = [], []
        state.teacher_index, state.teacher_steps = 0, 0
        state.last_action, state.last_input = None, None
        state.tasks_started += 1
        if state.active.get("executor") == "adopted":
            admitted = state.active["adoption"]
            if state.adopted_sha256 != admitted["sha256"]:
                require(state.model_sha256 == admitted["sha256"] and state.revision == admitted["revision"],
                        "Adoption must freeze the exact evaluated model.")
                self._adopted = deepcopy(self.model).eval()
                state.adopted_sha256 = admitted["sha256"]
        if state.active["mode"] == "close":
            self._act(state, "close")
        else:
            if self.frozen_policy:
                state.revision = 0
                state.model_sha256 = self.teacher_sha256 if self.frozen_policy == "transfer" else self.pretrained_sha256
            require(state.revision >= 0, "Task dispatched before initialization.")
            self._act(state, "reset", seed=state.active["seed"], mode=state.active["mode"],
                      reset_actions=state.active.get("reset_actions", []))
        return state

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs: Any) -> LLState:
        status = action_status.status
        require(sender == one(state, "actuation") and status["request_id"] == state.pending, "Unexpected action status.")
        require(status["legal"] is True, "Rejected environment action.")
        state.pending = None
        state.statuses += 1
        if status["action"] == "close":
            result = {"id": state.active["id"], "mode": "close", "success": True}
            state.active = None
            state.outbox.send_goal_update(one(state, "goals"), [Goal([], result=result)])
            return state
        state.environment_terminal = status.get("terminated", False) or status.get("truncated", False)
        state.observation_requests += 1
        state.pending = f"observation-{state.observation_requests}"
        state.outbox.request_observation(one(state, "perception"), request_id=state.pending)
        return state

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs: Any) -> LLState:
        require(sender == one(state, "perception") and kwargs["request_id"] == state.pending, "Unexpected observation.")
        state.pending = None
        state.observations += 1
        task = state.active
        goal = GoalSpec(**task["goal"])
        current = np.asarray(observation.content, dtype=np.uint8)
        grounded = ground_observation(current, **self.dimensions)
        if state.last_action is None and "initial_facts" in task:
            require(set(task["initial_facts"]) == set(grounded.facts), "Reset differs from planned initial facts.")
        success = goal.fact in grounded.facts and "hand-empty()" in grounded.facts
        timed_out = self._deadline is not None and time.monotonic() >= self._deadline
        done = success or (self.cap is not None and len(state.actions) >= self.cap) or state.environment_terminal or timed_out
        encoded = None if self.frozen_policy == "transfer" else condition_goal(current, goal)
        if state.last_action is not None and encoded is not None:
            mask = [index in legal_action_indices(current) for index in range(4)]
            state.rows.append({"state": state.last_input, "action": state.last_action,
                               "reward": 1.0 if success else (-1.0 if done else -0.01),
                               "successor": encoded.tolist(), "terminal": done, "legal_next": mask})
        self._observation = current
        if done:
            return self._finish(state, grounded, success, "time-limit" if timed_out else None)
        if task["mode"] in {"demo", "transfer"} or task.get("executor") == "transfer":
            if not task["plan"]:
                return self._finish(state, grounded, False, "planner-failed")
            spec = TransferSpec.from_mapping(task["plan"][state.teacher_index])
            if transfer_succeeded(grounded.facts, spec):
                state.teacher_index += 1
                state.teacher_steps = 0
                if state.teacher_index == len(task["plan"]):
                    return self._finish(state, grounded, False, "teacher-plan-incomplete")
                spec = TransferSpec.from_mapping(task["plan"][state.teacher_index])
            if state.teacher_steps >= 32:
                return self._finish(state, grounded, False, "teacher-transfer-cap")
            action, _, _ = greedy_inference(self.torch, self.teacher, current, spec, **self.dimensions)
            state.teacher_steps += 1
            state.teacher_inferences += 1
        else:
            model = self._adopted if task.get("executor") == "adopted" else self.pretrained if task["mode"] == "pretrained" else self.model
            action, _ = infer(self.torch, model, current, goal)
            if task["mode"] == "train" and not task.get("executor"):
                epsilon = 0.30 - 0.25 * min(task["train_index"] / max(1, self.training_episodes - 1), 1.0)
                if self.rng.random() < epsilon:
                    action = int(self.rng.choice(legal_action_indices(current)))
            state.direct_inferences += 1
        state.actions.append(action)
        state.last_action, state.last_input = action, None if encoded is None else encoded.tolist()
        self._act(state, action)
        return state

    def _finish(self, state: LLState, grounded: Any, success: bool, reason: str | None = None) -> LLState:
        task = state.active
        if state.rows and not state.rows[-1]["terminal"]:
            state.rows[-1].update(terminal=True, reward=-1.0)
        result = {"id": task["id"], "mode": task["mode"], "seed": task["seed"], "goal": task["goal"],
                  "success": success, "reason": "goal" if success else (reason or "action-cap"),
                  "actions": list(state.actions), "revision": state.revision, "model_sha256": state.model_sha256,
                  "execution_seconds": time.monotonic() - self._episode_started,
                  "policy_sha256": (self.teacher_sha256 if task["mode"] in {"demo", "transfer"} or task.get("executor") == "transfer" else
                                    state.adopted_sha256 if task.get("executor") == "adopted" else
                                    self.pretrained_sha256 if task["mode"] == "pretrained" else state.model_sha256),
                  "facts": sorted(grounded.facts), "probe": task.get("probe"), "case_index": task.get("case_index"),
                  "reset_actions": task.get("reset_actions", []), "difficulty": task.get("difficulty", 0),
                  "executor": task.get("executor"), "execution_plan": task.get("plan", []),
                  "planner_seconds": task.get("planner_seconds", 0), "adoption": task.get("adoption")}
        episode = {"result": result, "rows": state.rows if task["mode"] in {"demo", "train"} else []}
        state.outbox.send_beliefs(one(state, "knowledge"), Observation(episode, "atomic-achieve-on-episode"), grounded.beliefs)
        state.outbox.send_goal_update(one(state, "goals"), [Goal([], result=result)])
        state.completed += 1
        state.active, state.rows = None, []
        return state


class Knowledge(KnowledgeBase):
    """Forward grounded outcomes to HL and learning episodes to Memory."""

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation,
                           beliefs: Sequence[Belief], **kwargs: Any) -> KnowledgeState:
        require(sender == one(state, "ll_reasoning"), "Unknown belief source.")
        result = observation.content["result"]
        state.revisions += 1
        if result["mode"] in {"demo", "train"}:
            state.outbox.send_observations(one(state, "memory"), [observation])
            state.learning_episodes += 1
        state.outbox.send_beliefs(one(state, "hl_reasoning"), beliefs, result=result)
        return state


class Memory(MemoryBase):
    """Persist complete episodes and satisfy requests in either arrival order."""

    def on_init(self, **kwargs: Any) -> None:
        self.output = folder(kwargs)
        self._episodes = {}

    def _reply(self, state: MemoryState) -> None:
        request = state.pending
        if request is None or not all(identifier in self._episodes for identifier in request["ids"]):
            return
        # A main-run batch exceeds RabbitMQ's 16 MiB message limit. Each
        # capped episode is small; retain ordering within one logical request.
        for index, identifier in enumerate(request["ids"]):
            episode = json.loads(self._episodes[identifier].read_text())
            state.outbox.send_memories(one(state, "learning"), [Observation(episode)],
                                      learning_id=request["learning_id"], memory_index=index)
        state.pending = None
        state.responses += 1

    def on_observation_update(self, state: MemoryState, sender: str, observations: Sequence[Observation], **kwargs: Any) -> MemoryState:
        require(sender == one(state, "knowledge"), "Unknown memory source.")
        for observation in observations:
            episode = observation.content
            result = episode["result"]
            identifier = result["id"]
            require(result["mode"] in {"demo", "train"} and identifier not in self._episodes, "Invalid or duplicate episode.")
            path = self.output / f"episode-{identifier}.json"
            with path.open("x", encoding="utf-8") as target:
                json.dump(episode, target)
            self._episodes[identifier] = path
            state.episodes.append({"id": identifier, "mode": result["mode"], "rows": len(episode["rows"]), "sha256": digest(episode)})
        self._reply(state)
        return state

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs: Any) -> MemoryState:
        require(sender == one(state, "learning") and state.pending is None, "Overlapping memory request.")
        state.pending = dict(kwargs)
        state.requests += 1
        self._reply(state)
        return state


class Learner(LearnerBase):
    """Own the GPU optimizer; publish frozen CPU weights at episode boundaries."""

    def on_init(self, **kwargs: Any) -> None:
        self.torch = import_module("torch")
        self.torch.set_num_threads(1)
        self.core = import_module("mha_exp_level2_bw.achieve_on.learning")
        self.config = self.core.Config(**kwargs["learning"])
        self.device = kwargs["device"]
        require(self.device in {"cpu", "cuda"}, "Invalid learner device.")
        if self.device == "cuda":
            require(str(self.torch.__version__) == "2.14.0+cu130", "CUDA learner requires the validated Torch 2.14.0+cu130 build.")
            require(self.torch.cuda.is_available(), "CUDA learner requires an exposed compatible GPU.")
        self.torch.manual_seed(self.config.seed)
        self.model, self.provenance = warm_start(self.torch)
        self.model.to(self.device)
        self.target = deepcopy(self.model).eval()
        self.optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.rng = np.random.default_rng(self.config.seed)
        self.replay = None
        self.remaining = 0
        self.per_episode = kwargs["updates_per_episode"]
        self.per_callback = kwargs.get("updates_per_callback", 4)
        self.output = folder(kwargs)
        self.identity = kwargs["identity"]
        self._started = time.monotonic()
        self.hardware = {"torch": str(self.torch.__version__), "cuda": self.torch.version.cuda,
                         "device": self.device, "device_name": self.torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"}

    def _publish(self, state: LearnerState) -> None:
        weights = {key: value.detach().cpu().tolist() for key, value in self.model.state_dict().items()}
        checksum = digest(weights)
        state.revision += 1
        checkpoint = self.output / f"achieve-on-r{state.revision:05d}.pt"
        require(not checkpoint.exists(), "Checkpoint already exists.")
        payload = checkpoint_payload(self.model, self.provenance)
        payload.update(status="runtime-trained" if state.optimizer_steps else "initialized-only",
                       optimizer_steps=state.optimizer_steps, identity=self.identity, revision=state.revision)
        self.torch.save(payload, checkpoint)
        state.checkpoints.append({"revision": state.revision, "optimizer_steps": state.optimizer_steps,
                                  "elapsed_seconds": time.monotonic() - self._started,
                                  "learning_metrics": dict(state.learning_metrics),
                                  "model_sha256": checksum, "file": checkpoint.name,
                                  "file_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        state.outbox.send_model(one(state, "ll_reasoning"), {"weights": weights, "sha256": checksum,
                              "revision": state.revision, "learning_id": state.pending["id"]})
        state.pending = None
        state.device = self.device
        state.hardware = self.hardware

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs: Any) -> LearnerState:
        require(sender == one(state, "hl_reasoning") and state.pending is None, "Overlapping learner task.")
        state.pending = task
        if task["kind"] == "initialize":
            require(state.revision == -1, "Duplicate initialization.")
            self._publish(state)
        else:
            require(task["kind"] in {"warm", "episode"}, "Unknown learning task.")
            require(bool(task["ids"]), "Empty memory request.")
            self._memory_received = 0
            self._memory_rows = []
            self._memory_difficulties = []
            state.outbox.request_memories(one(state, "memory"), ids=task["ids"], learning_id=task["id"])
        return state

    def on_memories(self, state: LearnerState, sender: str, memories: Sequence[Observation], **kwargs: Any) -> LearnerState:
        require(sender == one(state, "memory") and state.pending is not None and kwargs["learning_id"] == state.pending["id"], "Unexpected memories.")
        require(kwargs.get("memory_index") == self._memory_received, "Memory chunk is duplicate or out of order.")
        require([item.content["result"]["id"] for item in memories] == state.pending["ids"][self._memory_received:self._memory_received + 1], "Memory episode identities differ.")
        rows = []
        difficulties = []
        for item in memories:
            episode = item.content
            result = episode["result"]
            expected_mode = "demo" if state.pending["kind"] == "warm" else "train"
            require(result["mode"] == expected_mode, "Probe or wrong split reached replay.")
            if expected_mode == "demo" and not result["success"]:
                continue
            pending = deque(self.core.Transition(
                np.asarray(row["state"], dtype=np.uint8), row["action"], row["reward"],
                np.asarray(row["successor"], dtype=np.uint8), row["terminal"],
                np.asarray(row["legal_next"], dtype=np.bool_)) for row in episode["rows"])
            require(bool(pending) and pending[-1].terminal, "Learning episode is incomplete.")
            emitted = self.core.emit(pending, self.config, True)
            rows.extend(emitted)
            difficulties.extend([result.get("difficulty", 0)] * len(emitted))
        self._memory_rows.extend(rows)
        self._memory_difficulties.extend(difficulties)
        self._memory_received += 1
        if self._memory_received < len(state.pending["ids"]):
            return state
        rows, difficulties = self._memory_rows, self._memory_difficulties
        self._memory_rows, self._memory_difficulties = [], []
        if state.pending["kind"] == "warm":
            require(self.replay is None, "Duplicate warm start.")
            self.replay = self.core.Replay(self.config.capacity, rows, difficulties)
            state.expert_transitions = len(rows)
            self.remaining = self.config.warm_updates
        else:
            require(self.replay is not None, "Online learning before demonstration warm-up.")
            for row in rows:
                self.replay.append(row)
            state.online_transitions += len(rows)
            state.trained_episodes += 1
            self.remaining = self.per_episode
        return state

    def step(self, state: LearnerState) -> LearnerState:
        """Bound optimizer work per callback so runtime messages remain serviceable."""
        if self.remaining:
            for _ in range(min(self.per_callback, self.remaining)):
                self.core.optimize(self.model, self.target, self.optimizer, self.replay,
                                   self.rng, self.config, self.device, min(1.0, 0.4 + 0.6 * state.online_transitions / self.config.online_steps),
                                   state.learning_metrics,
                                   self.core.curriculum_limit(state.optimizer_steps, self.config.warm_updates)
                                   if state.pending["kind"] == "warm" else 2)
                state.optimizer_steps += 1
                self.remaining -= 1
                if state.optimizer_steps % self.config.target_sync == 0:
                    self.target.load_state_dict(self.model.state_dict())
            if not self.remaining:
                self._publish(state)
        return state


class HLReasoner(HLReasonerBase):
    """Own the fixed schedule and reconcile grounded outcomes before advancing."""

    def on_init(self, **kwargs: Any) -> None:
        self.schedule = kwargs["schedule"]
        self.demo_ids = [task["id"] for task in self.schedule if task["mode"] == "demo"]
        self.production = kwargs.get("production", False)
        self.frozen_policy = kwargs.get("frozen_policy")
        self.probe_cases = len({task["case_index"] for task in self.schedule if task["mode"] == "probe"})
        self._planner = production_planner(**kwargs.get("dimensions", {})) if self.production else None
        self._pool = ThreadPoolExecutor(max_workers=1) if self.production else None
        self._planning = None
        self._planned_task = None
        self._duration = kwargs.get("duration_seconds")
        self._matched = kwargs.get("matched_single_goal", False)

    def on_first(self, state: HLState) -> HLState:
        self._started = time.monotonic()
        if self.frozen_policy:
            self._dispatch(state)
            return state
        state.learning_id = "initialize"
        state.outbox.send_learner_task(one(state, "learning"), {"kind": "initialize", "id": "initialize"})
        return state

    def _dispatch(self, state: HLState) -> None:
        task = dict(self.schedule[state.index]) if state.index < len(self.schedule) else {"id": "close", "mode": "close"}
        state.execution_seconds = time.monotonic() - self._started
        if not self._matched and self._duration is not None and task["mode"] != "close" and state.execution_seconds >= max(0, self._duration - 45):
            state.time_limited = True
            task = {"id": "close", "mode": "close"}
        if self.production and (task["mode"] == "train" or (task["mode"] in {"demo", "transfer"} and not task.get("plan"))):
            if state.adopted is not None and task["mode"] == "train":
                task.update(executor="adopted", adoption=dict(state.adopted))
                state.planner_bypasses += 1
            else:
                state.pending, state.phase = task["id"], "planning"
                self._planned_task = task
                self._planning = self._pool.submit(self._planner.solve, set(task["initial_facts"]),
                                                   GoalSpec(**task["goal"]), f"runtime-{task['id']}")
                return
        self._send_task(state, task)

    def _send_task(self, state: HLState, task: dict) -> None:
        """Dispatch one planned Transfer sequence or admitted AchieveOn macro."""
        if self._matched and task["mode"] != "close":
            task["remaining_seconds"] = max(0.0, self._duration - (time.monotonic() - self._started) - 2.0)
        state.pending = task["id"]
        state.phase = "executing"
        beliefs = [] if task["mode"] == "close" else [Belief("On", (task["goal"]["top"], task["goal"]["bottom"]))]
        state.outbox.send_goals(one(state, "goals"), [Goal(beliefs, task=task)])

    def step(self, state: HLState) -> HLState:
        """Poll bounded background planning without blocking module messages."""
        if self._planning is not None and self._planning.done():
            outcome = self._planning.result()
            task = self._planned_task
            task.update(executor="transfer", plan=outcome.actions if outcome.accepted else [],
                        planner_seconds=outcome.elapsed_seconds)
            state.planning_records.append({"id": task["id"], "mode": task["mode"], "accepted": outcome.accepted,
                                           "engine": outcome.engine, "seconds": outcome.elapsed_seconds,
                                           "actions": task["plan"]})
            self._planning = self._planned_task = None
            self._send_task(state, task)
        return state

    def on_last(self, state: HLState) -> HLState:
        """Release the runtime planner worker during normal or timed shutdown."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        return state

    def _advance(self, state: HLState) -> None:
        if state.goal_result is None or state.belief_result is None:
            return
        require(state.goal_result == state.belief_result, "Goal and Knowledge results differ.")
        result = state.goal_result
        require(result["id"] == state.pending, "Unexpected completed episode.")
        expected = self.schedule[state.index]
        require(result["mode"] == expected["mode"] and result["seed"] == expected["seed"] and result["goal"] == expected["goal"], "Completed task differs from schedule.")
        goal = GoalSpec(**result["goal"])
        observed_success = goal.fact in result["facts"] and "hand-empty()" in result["facts"]
        require(result["success"] == observed_success, "Success is not grounded.")
        state.results.append(result)
        if self._matched and result["reason"] == "time-limit":
            state.time_limited = True
        state.index += 1
        if self.production:
            if result["mode"] == "train" and result.get("executor") == "adopted" and not result["success"]:
                state.adoption_records.append({"revoked_after": result["id"], "sha256": state.adopted["sha256"]})
                state.adopted = None
                state.adoption_revoked = True
            next_task = self.schedule[state.index] if state.index < len(self.schedule) else None
            milestone = result.get("probe")
            if (result["mode"] == "probe" and isinstance(milestone, int) and milestone >= 100
                    and (next_task is None or next_task["mode"] != "probe")):
                evidence = adoption_evidence(state.results, milestone, self.probe_cases)
                state.adoption_records.append(evidence)
                if evidence["passed"] and state.adopted is None and not state.adoption_revoked:
                    state.adopted = evidence
        state.pending = state.goal_result = state.belief_result = None
        task = None
        if self.demo_ids and result["id"] == self.demo_ids[-1]:
            task = {"kind": "warm", "id": "warm", "ids": self.demo_ids}
        elif result["mode"] == "train":
            task = {"kind": "episode", "id": f"learn-{result['id']}", "ids": [result["id"]]}
        if task:
            state.learning_id = task["id"]
            state.phase = "learning"
            state.outbox.send_learner_task(one(state, "learning"), task)
        else:
            self._dispatch(state)

    def on_belief_update(self, state: HLState, sender: str, beliefs: Sequence[Belief], **kwargs: Any) -> HLState:
        require(sender == one(state, "knowledge") and state.belief_result is None, "Unexpected Knowledge result.")
        require(set(kwargs["result"]["facts"]) == beliefs_to_facts(beliefs), "Knowledge facts differ from result.")
        state.belief_result = kwargs["result"]
        self._advance(state)
        return state

    def on_goal_update(self, state: HLState, sender: str, goals: Sequence[Goal], **kwargs: Any) -> HLState:
        require(sender == one(state, "goals"), "Unexpected GoalGraph sender.")
        extras = goals[0].extras
        if "model_ready" in extras:
            require(extras["model_ready"]["learning_id"] == state.learning_id, "Unexpected model acknowledgement.")
            state.learning_id = None
            self._dispatch(state)
        elif extras["result"]["mode"] == "close":
            require((state.index == len(self.schedule) or state.time_limited) and state.pending == "close", "Premature closure.")
            state.pending = None
            state.phase = "completed"
            state.outbox.terminate_agent("direct-achieve-on-time-limit" if state.time_limited else "direct-achieve-on-schedule-completed")
        else:
            require(state.goal_result is None, "Duplicate terminal goal.")
            state.goal_result = extras["result"]
            self._advance(state)
        return state


def initial_states() -> dict[str, dict]:
    """Declare every persisted module field explicitly."""
    return {
        "perceptor": {"pending": None, "requests": 0, "responses": 0},
        "actuator": {"pending": None, "requests": 0, "responses": 0},
        "goalgraph": {"pending": None, "requests": 0, "responses": 0},
        "knowledge": {"revisions": 0, "learning_episodes": 0},
        "memory": {"episodes": [], "pending": None, "requests": 0, "responses": 0},
        "learner": {"pending": None, "revision": -1, "optimizer_steps": 0, "trained_episodes": 0,
                    "online_transitions": 0, "expert_transitions": 0, "checkpoints": [], "device": None, "hardware": None,
                    "learning_metrics": {}},
        "hlreasoner": {"index": 0, "pending": None, "phase": "initializing", "learning_id": None,
                       "goal_result": None, "belief_result": None, "results": [], "planning_records": [],
                       "planner_bypasses": 0, "adopted": None, "adoption_revoked": False, "adoption_records": [],
                       "time_limited": False, "execution_seconds": 0.0},
        "llreasoner": {"active": None, "pending": None, "requests": 0, "statuses": 0,
                       "observation_requests": 0, "observations": 0, "tasks_started": 0, "completed": 0,
                       "revision": -1, "model_sha256": None, "model_installs": 0, "rows": [], "actions": [],
                       "teacher_index": 0, "teacher_steps": 0, "last_action": None, "last_input": None,
                       "environment_terminal": False, "teacher_inferences": 0, "direct_inferences": 0,
                       "adopted_sha256": None},
    }
