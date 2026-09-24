"""Eight MHAgentA modules for continuous, goal-directed native-policy learning."""

from copy import deepcopy
from functools import wraps
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from mhagenta import ActionStatus, Belief, Goal, Observation
from mhagenta.states import ActuatorState, GoalGraphState, KnowledgeState, MemoryState, LearnerState, LLState, HLState
from mhagenta.bases import GoalGraphBase, HLReasonerBase, KnowledgeBase, LLReasonerBase, LearnerBase, MemoryBase
from mhagenta.defaults.communication import RMQActuatorBase

from ..exp2_5.beliefs import (Direction, initial_belief_state, revise_belief_state,
                            movement_evidence, available_movement_actions, novel_movement_actions)
from ..exp2_5.contracts import as_rgb_frame, rgb_sha256
from ..exp2_5.grounding import load_grounding_templates, ground_observation, resolve_active_grounding_bundle
from ..exp2_5.policy import PolicyId, inference_evidence, select_action
from ..exp2_5.runtime import CrafterRGBPerceptor, _single_id
from .online_deliberation import choose_step, failure_key, recovery_complete
from .online_learning import LearningConfig, Trainer
from .online_planning import safe_to_explore, epsilon, inputs, teacher_command, outcome
from .online_policy import SKILLS, load_basics, candidate, weights_hash


def checked(callback):
    """Persist operational faults and terminate instead of leaving a silent stalled loop."""
    @wraps(callback)
    def wrapped(self, state, *args, **kwargs):
        try:
            return callback(self, state, *args, **kwargs)
        except Exception as error:
            import traceback
            state["failure"] = f"{callback.__name__}: {type(error).__name__}: {error}"
            self.log(40, traceback.format_exc())
            state.outbox.terminate_agent(state["failure"])
            return state
    return wrapped


def require(condition: bool, message: str) -> None:
    """Raise an experiment contract fault, including under optimized Python."""
    if not condition:
        raise ValueError(message)


def one(state, role: str) -> str:
    """Resolve the single registered module for one typed role."""
    return _single_id(getattr(state.directory.internal, role), role)


def mask(actions) -> np.ndarray:
    """Represent a native action set consistently in replay and action evidence."""
    return np.asarray([index in actions for index in range(6)], dtype=np.bool_)


def write_json(path: Path, value: Any) -> None:
    """Atomically publish JSON progress/checkpoint metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Actuator(RMQActuatorBase):
    """Preserve native action ownership and correlate reset/close acknowledgements."""

    def on_first(self, state: ActuatorState) -> ActuatorState:
        """Resolve the external environment and the two authorized action owners."""
        self.env = state.directory.external.environment.address["env_id"]
        self.ll, self.hl = one(state, "ll_reasoning"), one(state, "hl_reasoning")
        return state

    @checked
    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        """Assign a native action ID or forward an explicitly correlated control."""
        require(sender in {self.ll, self.hl} and state["pending"] is None, "Overlapping or foreign action")
        request = dict(kwargs)
        if type(request["action"]) is int:
            require(request["requester_kind"] == ("hl_primitive" if sender == self.hl else "ll_policy"), "Action owner mismatch")
            state["requests"] += 1
            request["environment_atomic_id"] = state["requests"]
        else:
            require(sender == self.ll and request["action"] in {"reset", "close"}, "Invalid control owner")
            state["controls"] += 1
        state["pending"] = request
        self.act(self.env, **request)
        return state

    @checked
    def on_status(self, state: ActuatorState, env_id: str, **kwargs) -> ActuatorState:
        """Validate the environment reply before forwarding it to LL."""
        request = state["pending"]
        require(env_id == self.env and request is not None, "Status without matching action")
        require(all(kwargs.get(key) == value for key, value in request.items()), "Status correlation differs")
        require(kwargs.get("contract_error") is None, "Environment contract failure")
        if type(request["action"]) is int:
            require(not kwargs["illegal_action"] and not kwargs["lethal_movement"], "Prohibited native action")
            state["statuses"] += 1
        else:
            state["control_statuses"] += 1
        state["pending"] = None
        state.outbox.send_status(self.ll, ActionStatus(status=kwargs))
        return state


class GoalGraph(GoalGraphBase):
    """Serialize HLR commands and the corresponding observed/model-installed replies."""

    @checked
    def on_goal_update(self, state: GoalGraphState, sender: str, goals, **kwargs) -> GoalGraphState:
        """Route a fresh command or its matching completed reply."""
        require(len(goals) == 1, "Exactly one goal required")
        data = goals[0].extras
        if sender == one(state, "hl_reasoning"):
            require(state["active"] is None and data["id"] == state["sent"] + 1, "Overlapping or out-of-order goal")
            state["active"] = data["id"]
            state["sent"] += 1
            state.outbox.send_goals(one(state, "ll_reasoning"), goals)
        else:
            require(sender == one(state, "ll_reasoning") and data["id"] == state["active"], "Unmatched goal reply")
            state["active"] = None
            state["received"] += 1
            state.outbox.send_goals(one(state, "hl_reasoning"), goals)
        return state


class Knowledge(KnowledgeBase):
    """Authenticate fresh RGB boundaries, forward beliefs and archive completed experience."""

    @checked
    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation, beliefs, **kwargs) -> KnowledgeState:
        """Verify RGB identity and observation ordering before forwarding evidence."""
        require(sender == one(state, "ll_reasoning"), "Unexpected belief source")
        require(kwargs["observation_id"] == state["observations"] + 1, "Noncontiguous knowledge observations")
        require(rgb_sha256(observation.content) == kwargs["digest"], "Knowledge RGB digest mismatch")
        state["observations"] += 1
        segment = kwargs.get("segment")
        if segment is not None:
            state["segments"] += 1
            state.outbox.send_observations(one(state, "memory"), [observation], segment=segment)
        state.outbox.send_beliefs(one(state, "hl_reasoning"), beliefs, snapshot=kwargs["snapshot"])
        return state


class Memory(MemoryBase):
    """Join either message order and persist one compact binary experience segment."""

    def on_init(self, output="/out", **kwargs) -> None:
        """Create the experience archive and an empty two-order delivery join."""
        self.output = Path(output) / "experience"
        self.output.mkdir(parents=True, exist_ok=True)
        self.segments, self.request = {}, None

    def _deliver(self, state: MemoryState):
        """Evict an archived segment only when its matching request is queued."""
        if self.request is None or self.request not in self.segments:
            return
        segment = self.segments.pop(self.request)
        state.outbox.send_memories(one(state, "learning"), [Observation(segment)], segment_id=self.request)
        state["delivered"] += 1
        self.request = None
        state["pending"] = None
        state["stored"] = len(self.segments)

    @checked
    def on_observation_update(self, state: MemoryState, sender: str, observations, **kwargs) -> MemoryState:
        """Archive exactly one completed segment supplied by Knowledge."""
        import torch
        require(sender == one(state, "knowledge"), "Unexpected memory source")
        segment = kwargs["segment"]
        path = self.output / f"{segment['id']}.pt"
        require(not path.exists() and segment["id"] not in self.segments, "Duplicate segment")
        require(segment["rows"] and segment["rows"][-1]["terminal"], "Incomplete experience segment")
        torch.save(segment, path)
        self.segments[segment["id"]] = segment
        state["received"] += 1
        state["stored"] = len(self.segments)
        self._deliver(state)
        return state

    @checked
    def on_memory_request(self, state: MemoryState, sender: str, **kwargs) -> MemoryState:
        """Retain an early request until its completed experience arrives."""
        require(sender == one(state, "learning") and self.request is None, "Overlapping memory request")
        self.request = kwargs["segment_id"]
        state["pending"] = self.request
        self._deliver(state)
        return state


class Learner(LearnerBase):
    """Update independent skill networks in short callbacks and acknowledge installation."""

    def on_init(self, seed=0, device="cpu", output="/out", learning=None, **kwargs) -> None:
        """Create fresh independent optimizers and replay on the requested device."""
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        require(torch.__version__.split("+")[0] == "2.14.0", "Torch 2.14.0 is required")
        require(device == "cpu" or torch.cuda.is_available(), "CUDA unavailable in learner container")
        self.torch, self.output, self.device = torch, Path(output), device
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = LearningConfig(**(learning or {}))
        basic = load_basics(torch)["navigate_to"]
        self.trainers = {skill: Trainer(torch, basic, seed + i, device, self.config) for i, skill in enumerate(SKILLS)}
        self.task, self.remaining, self.progress = None, 0, 0.0
        self.saved_at = time.monotonic()

    def on_first(self, state: LearnerState) -> LearnerState:
        """Expose actual Torch/device information and initial model identities."""
        state["hardware"] = {"torch": self.torch.__version__, "device": self.device,
                             "cuda": self.torch.version.cuda,
                             "name": self.torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"}
        state["skills"] = {name: trainer.summary() for name, trainer in self.trainers.items()}
        return state

    @checked
    def on_task(self, state: LearnerState, sender: str, task, **kwargs) -> LearnerState:
        """Request the exact completed skill segment through Memory."""
        require(sender == one(state, "ll_reasoning") and self.task is None, "Overlapping learner task")
        self.task = task
        state["pending"] = task["id"]
        state.outbox.request_memories(one(state, "memory"), segment_id=task["id"])
        return state

    @checked
    def on_memories(self, state: LearnerState, sender: str, memories, **kwargs) -> LearnerState:
        """Exclude probes, ingest ordinary experience and schedule bounded updates."""
        require(sender == one(state, "memory") and self.task is not None, "Unexpected memories")
        require(len(memories) == 1 and memories[0].content["id"] == self.task["id"], "Memory/task mismatch")
        segment = memories[0].content
        require(segment["skill"] == self.task["skill"], "Wrong skill in memory response")
        self.progress = self.task["progress"]
        state["segments"] += 1
        if segment["mode"] != "probe":
            self.trainers[segment["skill"]].ingest(segment)
            self.remaining = self.config.updates_per_segment
        else:
            state["excluded_probes"] += 1
            self._publish(state)
        return state

    def _publish(self, state: LearnerState):
        """Checkpoint the trained revision and send its exact weights to LL."""
        trainer = self.trainers[self.task["skill"]]
        summary = trainer.summary()
        state["skills"][self.task["skill"]] = summary
        checkpoint = self.output / f"{self.task['skill']}-r{trainer.updates:07d}.pt"
        if not checkpoint.exists():
            self.torch.save({"weights": trainer.model.state_dict(), "summary": summary}, checkpoint)
            state["checkpoints"].append({"skill": self.task["skill"], "file": checkpoint.name,
                                         "updates": trainer.updates, "sha256": summary["sha256"]})
        payload = {"id": self.task["id"], "skill": self.task["skill"], "summary": summary,
                   "weights": {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}}
        state.outbox.send_model(one(state, "ll_reasoning"), payload)
        state["publications"] += 1
        self.task = None
        state["pending"] = None
        write_json(self.output / "learner-progress.json", {"updated_at": time.time(), **state["skills"]})
        if time.monotonic() - self.saved_at >= 300:
            self._save_trainers()

    def _save_trainers(self):
        """Atomically checkpoint bounded replay and optimizer state every five minutes."""
        for skill, trainer in self.trainers.items():
            path = self.output / f"{skill}-trainer.pt"
            temporary = path.with_suffix(".tmp")
            self.torch.save(trainer.checkpoint(), temporary)
            temporary.replace(path)
        self.saved_at = time.monotonic()

    @checked
    def step(self, state: LearnerState) -> LearnerState:
        """Run one optimizer update per callback so lifecycle/control messages can execute."""
        if self.remaining:
            self.trainers[self.task["skill"]].optimize(self.progress)
            self.remaining -= 1
            if self.remaining == 0:
                self._publish(state)
        return state

    @checked
    def on_last(self, state: LearnerState) -> LearnerState:
        """Retain full training state for audit; each main run still starts afresh."""
        self._save_trainers()
        for skill, trainer in self.trainers.items():
            state["skills"][skill] = trainer.summary()
        return state


class LLReasoner(LLReasonerBase):
    """Ground RGB, execute base/learned policies, and attribute native experience."""

    def on_init(self, seed=0, output="/out", duration_seconds=3600, **kwargs) -> None:
        """Load only allowed base policies and create unadmitted warm-start candidates."""
        import torch
        torch.set_num_threads(1)
        self.torch, self.models = torch, load_basics(torch)
        self.candidates = {name: candidate(torch, self.models["navigate_to"]).eval() for name in SKILLS}
        self.probes = {}
        self.templates = load_grounding_templates(resolve_active_grounding_bundle()[0])
        self.frame, self.command, self.pending, self.early_status = None, None, None, None
        self.rows, self.segment_result = [], None
        self.output, self.duration = Path(output), duration_seconds
        self.output.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def on_first(self, state: LLState) -> LLState:
        """Start the first RGB observation cycle."""
        self.started = time.monotonic()
        self._observe(state)
        return state

    def _observe(self, state: LLState):
        """Request the next globally correlated RGB frame."""
        require(not state["awaiting_observation"], "Duplicate observation request")
        state["awaiting_observation"] = True
        state.outbox.request_observation(one(state, "perception"), observation_id=state["observations"] + 1)

    def _reply(self, state: LLState):
        """Acknowledge a command only after observation and any model installation."""
        data = {"id": self.command["id"], "kind": self.command["kind"], "result": self.segment_result,
                "observation_id": state["observations"]}
        state.outbox.send_goal_update(one(state, "goals"), [Goal([], extras=data)])
        state["replies"] += 1
        self.command, self.segment_result = None, None
        state["command"] = None

    @checked
    def on_goal_update(self, state: LLState, sender: str, goals, **kwargs) -> LLState:
        """Install the latest HLR command and freeze new probe revisions when needed."""
        require(sender == one(state, "goals") and len(goals) == 1 and self.command is None, "Invalid LL goal")
        command = dict(goals[0].extras)
        require(command["basis"] == state["observations"], "Stale HLR decision")
        self.command = command
        state["command"] = command
        if command.get("session") is not None:
            incoming = command["session"]
            if state["session"] is None:
                require(not self.rows, "Unclosed previous skill trajectory")
                state["session"] = deepcopy(incoming)
                if incoming["mode"] == "probe":
                    old = self.probes.get(incoming["skill"])
                    if old is None or old[0] != incoming["probe_token"]:
                        self.probes[incoming["skill"]] = (incoming["probe_token"], deepcopy(self.candidates[incoming["skill"]]))
                        self.torch.save(self.probes[incoming["skill"]][1].state_dict(),
                                        self.output / f"{incoming['skill']}-probe-{incoming['probe_token']}.pt")
            else:
                require(incoming == state["session"], "HLR/LL skill context mismatch")
        if command["kind"] in {"reset", "close"}:
            require(state["session"] is None, "Control crossed unfinished skill")
            state.outbox.request_action(one(state, "actuation"), action=command["kind"], control_id=str(command["id"]))
        else:
            self._act(state)
        return state

    def _act(self, state: LLState):
        """Select a masked native action or await the HLR-owned primitive status."""
        belief, session = state["belief"], state["session"]
        require(self.pending is None and not state["awaiting_observation"], "Overlapping native action")
        command = self.command
        progress = min(1.0, (time.monotonic() - self.started) / self.duration)
        safe = safe_to_explore(belief)
        target = session["target"] if session else command.get("target")
        skill = session["skill"] if session else None
        context, allowed = inputs(belief, skill, target) if session else (np.zeros(2), ())
        mode = session["mode"] if session else "teacher"
        q_values, model_hash = None, None
        if command["kind"] == "primitive":
            action = command["action"]
            exploration = 0.0
        elif command["kind"] == "learned":
            if mode in {"probe", "adopted"}:
                token, model = self.probes[skill]
                require(token == session["probe_token"], "Wrong frozen skill revision")
            else:
                model = self.candidates[skill]
            require(bool(allowed), "Learned skill has no legal actions")
            with self.torch.inference_mode():
                q_values = model(self.torch.as_tensor(self.frame.transpose(2, 0, 1)[None]),
                                 self.torch.as_tensor(context[None], dtype=self.torch.float32))[0].numpy().tolist()
            action = select_action(q_values, allowed)
            model_hash = weights_hash(model)
            exploration = epsilon(progress, teacher=False, safe=safe, evaluation=mode in {"probe", "adopted"})
        else:
            policy = PolicyId(command["policy"])
            movement = available_movement_actions(belief, tuple(target) if session else None)
            if policy is PolicyId.EXPLORE:
                movement = novel_movement_actions(movement, tuple(belief["player"]), map(tuple, state["recent"]))
            # Do not label a move that the learned skill's runtime mask forbids.
            if session:
                movement = tuple(action for action in movement if action in allowed)
            require(bool(movement), "Basic policy has no legal movements")
            action, q_values = inference_evidence(self.torch, self.models[policy.value], self.frame, policy,
                                                 belief["player"], command.get("target"),
                                                 facing=Direction.from_name(belief["facing"]).delta, movement_actions=movement)
            if not session:
                allowed = movement
            exploration = epsilon(progress, teacher=True, safe=safe)
        exploratory = exploration > 0 and self.rng.random() < exploration
        if exploratory:
            action = int(self.rng.choice(allowed))
        if session:
            require(action in allowed, "Teacher/candidate action outside shared replay mask")
        requester = "hl_primitive" if command["kind"] == "primitive" else "ll_policy"
        owner = str(command["id"]) if requester == "hl_primitive" else f"{command['id']}/{state['actions'] + 1}"
        self.pending = {**movement_evidence(action, belief), "status": None,
                        "owner_action_id": owner, "requester_kind": requester,
                        "command_id": command["id"], "purpose": command.get("purpose"),
                        "policy": skill if command["kind"] == "learned" else command.get("policy"),
                        "inventory": dict(belief["inventory"]),
                        "target": deepcopy(target), "context": context, "mask": mask(allowed),
                        "frame": self.frame, "epsilon": exploration, "exploratory": exploratory,
                        "model_sha256": model_hash, "q_values": q_values}
        state["exploratory_actions"] += int(exploratory)
        state["recent"] = (state["recent"] + [list(belief["player"])])[-4:]
        if requester == "ll_policy":
            state.outbox.request_action(one(state, "actuation"), action=action,
                                       owner_action_id=owner, requester_kind=requester)
        if self.early_status is not None:
            status, self.early_status = self.early_status, None
            self._status(state, status)

    def _status(self, state: LLState, status):
        """Join the originating native action before asking for its successor frame."""
        require(self.pending is not None and self.pending["status"] is None, "Unsolicited native status")
        require(status["environment_atomic_id"] == state["actions"] + 1, "Noncontiguous native action")
        require(all(status[key] == self.pending[key] for key in ("action", "owner_action_id", "requester_kind")), "Native action ownership mismatch")
        self.pending["status"] = status
        state["actions"] += 1
        self._observe(state)

    @checked
    def on_action_status(self, state: LLState, sender: str, action_status, **kwargs) -> LLState:
        """Handle native, early passive, reset and close replies without crossing epochs."""
        require(sender == one(state, "actuation"), "Unexpected actuator")
        status = action_status.status
        require(status.get("contract_error") is None, "Environment contract fault")
        if type(status["action"]) is str:
            require(self.command is not None and status["control_id"] == str(self.command["id"]), "Control acknowledgement mismatch")
            if status["action"] == "close":
                require(status["closed"], "Close not acknowledged")
                state["closed"] = True
                self._reply(state)
            else:
                require(status["episode"] == state["episode"] + 1, "Reset episode mismatch")
                state["episode"] = status["episode"]
                state["belief"] = initial_belief_state()
                state["recent"] = []
                self._observe(state)
        elif self.command is None or self.pending is None:
            require(self.early_status is None and status["requester_kind"] == "hl_primitive", "Unexpected early status")
            self.early_status = status
        else:
            self._status(state, status)
        return state

    def _finish_segment(self, state: LLState, success, reason):
        """Close the actual trajectory and request learning without inventing actions."""
        session = state["session"]
        segment = {**deepcopy(session), "success": success, "reason": reason, "rows": self.rows,
                   "model_sha256": self.rows[-1].get("model_sha256")}
        self.segment_result = {key: value for key, value in segment.items() if key != "rows"}
        state["segments"] += 1
        state["waiting_model"] = segment["id"]
        state["session"] = None
        self.rows = []
        state.outbox.send_learner_task(one(state, "learning"), {
            "id": segment["id"], "skill": segment["skill"],
            "progress": min(1.0, (time.monotonic() - self.started) / self.duration)})
        return segment

    def _send_beliefs(self, state: LLState, segment=None):
        """Send grounded state and optional completed experience on the Knowledge edge."""
        snapshot = {"belief": state["belief"], "episode": state["episode"], "actions": state["actions"],
                    "observation_id": state["observations"], "command_id": self.command["id"] if self.command else None,
                    "session": deepcopy(state["session"])}
        state.outbox.send_beliefs(one(state, "knowledge"), Observation(self.frame, observation_type="crafter-rgb"),
                                 [Belief("grounded_revision", (state["episode"], state["belief"]["revision"]))],
                                 observation_id=state["observations"], digest=rgb_sha256(self.frame),
                                 snapshot=snapshot, segment=segment)

    @checked
    def on_observation(self, state: LLState, sender: str, observation, **kwargs) -> LLState:
        """Ground the fresh frame, attribute skill outcomes and advance the control loop."""
        require(sender == one(state, "perception") and state["awaiting_observation"], "Unrequested observation")
        require(kwargs["observation_id"] == state["observations"] + 1, "Observation ID mismatch")
        frame = as_rgb_frame(observation.content)
        require(kwargs["observation_digest"] == rgb_sha256(frame) and kwargs.get("contract_error") is None, "Invalid RGB delivery")
        previous = state["belief"]
        percept, _ = ground_observation(frame, self.templates,
                                        previous_facing=Direction.from_name(previous["facing"]) if previous["facing"] else None)
        belief = revise_belief_state(previous, percept, revision=previous["revision"] + 1, pending_action=self.pending)
        state["belief"], self.frame = belief, frame
        state["observations"] += 1
        state["awaiting_observation"] = False
        segment = None
        if self.pending is not None:
            row, self.pending = self.pending, None
            session = state["session"]
            if session is not None:
                session["steps"] += 1
                result = outcome(previous, belief, row, session)
                context, allowed = inputs(belief, session["skill"], session["target"])
                if result is None and not allowed:
                    result = False, "movement_blocked"
                if result is None and session["mode"] == "teacher" and teacher_command(belief, session)["kind"] == "finish":
                    result = False, "unreachable"
                self.rows.append({"state": row["frame"], "context": row["context"], "mask": row["mask"],
                                  "action": row["action"], "next": frame, "next_context": context,
                                  "next_mask": mask(allowed), "terminal": result is not None,
                                  "reward": (1.0 if result[0] else -1.0) if result else -0.01,
                                  "exploratory": row["exploratory"], "epsilon": row["epsilon"],
                                  "model_sha256": row["model_sha256"]})
                if result is not None:
                    segment = self._finish_segment(state, *result)
            evidence = {key: value for key, value in row.items() if key not in {"frame", "context", "mask"}}
            evidence.update(episode=state["episode"], observation_id=state["observations"],
                            result_inventory=dict(belief["inventory"]),
                            context=row["context"].tolist(), legal_actions=np.flatnonzero(row["mask"]).tolist(),
                            input_sha256=rgb_sha256(row["frame"]), result_sha256=rgb_sha256(frame),
                            session_id=session["id"] if session else None)
            with (self.output / "actions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(evidence, allow_nan=False) + "\n")
        self._send_beliefs(state, segment)
        if state["waiting_model"] is not None:
            return state
        if self.command is not None:
            if self.command["kind"] == "learned" and state["session"] is not None:
                self._act(state)
            else:
                self._reply(state)
        return state

    @checked
    def on_model(self, state: LLState, sender: str, model, **kwargs) -> LLState:
        """Authenticate exact installed weights before acknowledging the completed goal."""
        require(sender == one(state, "learning") and model["id"] == state["waiting_model"], "Unmatched model publication")
        network = self.candidates[model["skill"]]
        network.load_state_dict(model["weights"], strict=True)
        require(weights_hash(network) == model["summary"]["sha256"], "Installed model hash differs")
        state["models"][model["skill"]] = model["summary"]
        state["model_installs"] += 1
        state["waiting_model"] = None
        self._reply(state)
        return state


class HLReasoner(HLReasonerBase):
    """Control survival, crafting, explicit acquisition, skill trials and admission."""

    def on_init(self, duration_seconds=3600, **kwargs) -> None:
        """Set the wall-clock execution ceiling."""
        self.duration = duration_seconds

    def on_first(self, state: HLState) -> HLState:
        """Start the agent execution clock."""
        self.started = time.monotonic()
        return state

    @checked
    def on_belief_update(self, state: HLState, sender: str, beliefs, **kwargs) -> HLState:
        """Join fresh grounded state with the corresponding goal completion."""
        require(sender == one(state, "knowledge"), "Unexpected HLR belief source")
        state["snapshot"] = kwargs["snapshot"]
        self._advance(state)
        return state

    @checked
    def on_goal_update(self, state: HLState, sender: str, goals, **kwargs) -> HLState:
        """Reconcile a command reply or finish after acknowledged environment closure."""
        require(sender == one(state, "goals") and len(goals) == 1, "Unexpected HLR goal update")
        reply = goals[0].extras
        require(reply["id"] == state["pending"], "Unmatched HLR reply")
        state["replies"] += 1
        if reply["kind"] == "close":
            state["closed"] = True
            state["pending"] = None
            state["execution_seconds"] = time.monotonic() - self.started
            state.outbox.terminate_agent("execution-time-complete")
            return state
        state["reply"] = reply
        self._advance(state)
        return state

    def _account(self, state: HLState, result):
        """Record outcomes and admit or revoke a specific frozen skill revision."""
        skill = result["skill"]
        stats = state["skills"][skill]
        stats["segments"] += 1
        stats["successes"] += int(result["success"])
        if result["mode"] == "teacher":
            stats["teacher_successes"] += int(result["success"])
        if result["mode"] == "probe":
            stats["probe_results"].append({"success": result["success"], "sha256": result["model_sha256"], "actions": result["steps"]})
            if len(stats["probe_results"]) == 10:
                rows = stats["probe_results"]
                require(len({row["sha256"] for row in rows}) == 1, "Probe cohort mixed model revisions")
                accepted = sum(row["success"] for row in rows) >= 9
                stats["adopted"] = stats["probe_token"] if accepted else None
                stats["admissions"].append({"token": stats["probe_token"], "accepted": accepted, "results": rows})
                if not accepted:
                    stats["probe_token"] += 1
                stats["probe_results"] = []
        if result["mode"] == "adopted" and not result["success"]:
            stats["admissions"].append({"revoked": stats["adopted"], "reason": result["reason"]})
            stats["adopted"] = None
            stats["probe_token"] += 1
        if not result["success"] and result["reason"] not in {"survival_interrupt", "action_budget", "death", "diamond"}:
            key = failure_key(result["purpose"], tuple(result["initial_target"]))
            state["excluded"][key] = state["snapshot"]["actions"] + 32

    def _advance(self, state: HLState):
        """Choose one next command only after the observation/reply join completes."""
        snapshot = state["snapshot"]
        if snapshot is None or state["closed"]:
            return
        if state["pending"] is not None:
            reply = state["reply"]
            if reply is None or snapshot["observation_id"] != reply["observation_id"]:
                return
            if reply["result"] is not None:
                self._account(state, reply["result"])
            if reply["kind"] == "reset":
                state["recovery"], state["placement"], state["excluded"] = None, None, {}
            state["pending"], state["reply"] = None, None
        belief = snapshot["belief"]
        elapsed = time.monotonic() - self.started
        state["execution_seconds"] = elapsed
        session = snapshot["session"]
        if elapsed >= max(1.0, self.duration - 45) and session is None:
            command = {"kind": "close"}
        elif belief["terminal"]:
            command = {"kind": "reset"}
        elif session is not None:
            command = teacher_command(belief, session)
            command.update(session=session, purpose=session["purpose"])
        else:
            command = self._choose(state, belief)
        state["commands"] += 1
        command.update(id=state["commands"], basis=snapshot["observation_id"])
        state["pending"] = command["id"]
        state.outbox.send_goals(one(state, "goals"), [Goal([], extras=command)])
        if command["kind"] == "primitive":
            state["primitive_actions"] += 1
            state.outbox.request_action(one(state, "actuation"), action=command["action"],
                                       owner_action_id=str(command["id"]), requester_kind="hl_primitive")

    def _choose(self, state: HLState, belief):
        """Prioritize needs and technology, then select explicit teaching or skill execution."""
        inv = belief["inventory"]
        recovery = state["recovery"]
        if recovery is not None and recovery_complete(belief, recovery):
            recovery = None
        # Begin food work while there is still a reserve. This supplies real,
        # useful EatTarget opportunities on which bounded trials are safe.
        depleted = sorted((inv[need], i, need) for i, need in enumerate(("drink", "food", "energy"))
                          if inv[need] <= (6 if need == "food" else 4))
        if depleted and (recovery is None or depleted[0][0] <= 2):
            recovery = depleted[0][2]
        if recovery is None and inv["health"] <= 4:
            recovery = "health"
        state["recovery"] = recovery
        state["excluded"] = {key: expiry for key, expiry in state["excluded"].items() if expiry > state["snapshot"]["actions"]}
        decision, state["placement"] = choose_step(belief, f"intent-{state['commands']}", recovery,
                                                    list(state["excluded"]), state["placement"], enable_eat_cow=False)
        if isinstance(decision, dict):
            return {"kind": "primitive", **decision}
        skill = decision.activity.value
        if skill not in SKILLS:
            if not available_movement_actions(belief):
                return {"kind": "primitive", "action": 0, "purpose": "blocked_wait"}
            return {"kind": "basic", "policy": skill, "target": decision.target_cell, "purpose": decision.purpose}
        stats = state["skills"][skill]
        stats["opportunities"] += 1
        mode = "teacher"
        if stats["adopted"] is not None:
            mode = "adopted"
        elif stats["teacher_successes"] >= 4 and safe_to_explore(belief):
            mode = "probe" if stats["opportunities"] % 4 == 0 else "trial" if stats["opportunities"] % 2 == 0 else "teacher"
        session = {"id": f"segment-{state['commands'] + 1}", "episode": state["snapshot"]["episode"],
                   "skill": skill, "resource": decision.target_kind, "target": list(decision.target_cell),
                   "initial_target": list(decision.target_cell), "purpose": decision.purpose,
                   "mode": mode, "steps": 0, "probe_token": stats["probe_token"]}
        command = teacher_command(belief, session) if mode == "teacher" else {"kind": "learned"}
        # Do not invent empty trajectories for targets that cannot be reached.
        if command["kind"] == "finish":
            return {"kind": "basic", "policy": "explore", "target": None, "purpose": decision.purpose}
        return {**command, "session": session, "purpose": decision.purpose}


def initial_states() -> dict:
    """Declare every persistent field; keep binary replay outside module autosaves."""
    stats = {"segments": 0, "successes": 0, "teacher_successes": 0, "opportunities": 0,
             "adopted": None, "probe_token": 0, "probe_results": [], "admissions": []}
    states = {
        "perceptor": {"active_seconds": 0.0, "pending_observation_id": None, "request_count": 0,
                      "observation_count": 0, "last_observation_sha256": None},
        "actuator": {"pending": None, "requests": 0, "statuses": 0, "controls": 0, "control_statuses": 0},
        "ll_reasoner": {"belief": initial_belief_state(), "episode": 0, "observations": 0, "actions": 0,
                        "awaiting_observation": False, "session": None, "command": None, "recent": [],
                        "waiting_model": None, "segments": 0, "models": {}, "model_installs": 0,
                        "exploratory_actions": 0, "replies": 0, "closed": False},
        "knowledge": {"observations": 0, "segments": 0},
        "memory": {"received": 0, "delivered": 0, "stored": 0, "pending": None},
        "learner": {"pending": None, "segments": 0, "publications": 0, "excluded_probes": 0,
                    "skills": {}, "hardware": None, "checkpoints": []},
        "goal_graph": {"active": None, "sent": 0, "received": 0},
        "hl_reasoner": {"snapshot": None, "pending": None, "reply": None, "commands": 0, "replies": 0,
                        "skills": {skill: deepcopy(stats) for skill in SKILLS}, "excluded": {}, "placement": None,
                        "recovery": None, "closed": False, "execution_seconds": 0.0, "primitive_actions": 0},
    }
    for state in states.values():
        state["failure"] = None
    return states


MODULES = {"perceptor": CrafterRGBPerceptor, "actuator": Actuator, "ll_reasoner": LLReasoner,
           "knowledge": Knowledge, "memory": Memory, "learner": Learner,
           "goal_graph": GoalGraph, "hl_reasoner": HLReasoner}
