"""Queued delivery of the eight real module callbacks against native Crafter."""

from collections import deque
from types import SimpleNamespace
from mhagenta import State
from mhagenta.outboxes import (PerceptorOutbox, ActuatorOutbox, LLOutbox, HLOutbox,
                              KnowledgeOutbox, MemoryOutbox, LearnerOutbox, GoalGraphOutbox)

from mha_exp_level2_cr.exp2_6 import online_runtime as runtime
from mha_exp_level2_cr.exp2_6.online_environment import ContinuingEnvironment, initial_state


def outbox(harness, sender):
    """Use actual typed framework outboxes, replacing only their transport sink."""
    bases = {"perceptor": PerceptorOutbox, "actuator": ActuatorOutbox, "ll_reasoner": LLOutbox,
             "hl_reasoner": HLOutbox, "knowledge": KnowledgeOutbox, "memory": MemoryOutbox,
             "learner": LearnerOutbox, "goal_graph": GoalGraphOutbox}

    class Capture(bases[sender]):
        """Route the exact public outbox payload through a deterministic queue."""

        def terminate_agent(self, reason=None):
            harness.terminated = reason

        def _add(self, receiver, connection, body, *args):
            if receiver in {"perceptor", "actuator"}:
                callback = "on_request"
            elif "goals" in body:
                callback = "on_goal_update"
            elif "action_status" in body:
                callback = "on_action_status"
            elif "task" in body:
                callback = "on_task"
            elif "model" in body:
                callback = "on_model"
            elif "memories" in body:
                callback = "on_memories"
            elif receiver == "memory":
                callback = "on_memory_request" if sender == "learner" else "on_observation_update"
            elif sender == "perceptor":
                callback = "on_observation"
            elif sender == "ll_reasoner":
                callback = "on_observed_beliefs"
            else:
                callback = "on_belief_update"
            harness.queue.append((receiver, callback, {"sender": sender, **body}))

    return Capture()


class RuntimeHarness:
    """Run the actual learning/acting loop with deterministic in-process transport."""

    def __init__(self, output, *, run=0, updates=1):
        self.queue, self.terminated = deque(), None
        self.logs = []
        self.output = output
        self.env = ContinuingEnvironment(initial_state(run, "agent", str(output / "environment")))
        roles = {"perception": "perceptor", "actuation": "actuator", "ll_reasoning": "ll_reasoner",
                 "hl_reasoning": "hl_reasoner", "knowledge": "knowledge", "learning": "learner",
                 "memory": "memory", "goals": "goal_graph"}
        directory = SimpleNamespace(internal=SimpleNamespace(**{
            role: [SimpleNamespace(module_id=name)] for role, name in roles.items()}),
            external=SimpleNamespace(environment=SimpleNamespace(address={"env_id": "environment"})))
        initial = runtime.initial_states()
        self.modules, self.states = {}, {}
        for name, cls in runtime.MODULES.items():
            self.modules[name] = cls(module_id=name, initial_state=initial[name])
            self.modules[name].log = lambda *args: self.logs.append(args)
            self.states[name] = State(agent_id="agent", module_id=name, time_func=lambda: 0.0,
                                      **initial[name], directory=directory, outbox=outbox(self, name))
        self.modules["perceptor"].observe = self.observe
        self.modules["actuator"].act = self.act
        for name in ("ll_reasoner", "hl_reasoner", "learner", "memory"):
            self.modules[name].on_init(output=str(output / "agent"), seed=run,
                                        learning={"updates_per_segment": updates, "batch_size": 4})
        for name in ("perceptor", "actuator", "knowledge", "memory", "goal_graph", "learner", "hl_reasoner", "ll_reasoner"):
            self.modules[name].on_first(self.states[name])

    def observe(self, env_id, **kwargs):
        _, response = self.env.on_observe(self.env.state, "agent", **kwargs)
        self.queue.append(("perceptor", "on_observation", {"env_id": env_id, **response}))

    def act(self, env_id, **kwargs):
        _, response = self.env.on_action(self.env.state, "agent", **kwargs)
        self.queue.append(("actuator", "on_status", {"env_id": env_id, **response}))

    def deliver(self):
        if self.queue:
            name, callback, kwargs = self.queue.popleft()
            self.states[name] = getattr(self.modules[name], callback)(self.states[name], **kwargs)
        else:
            self.modules["learner"].step(self.states["learner"])

    def run(self, actions=80, *, limit=100000):
        """End a focused validation at the next reconciled boundary, not a main-run budget."""
        for _ in range(limit):
            if self.states["ll_reasoner"]["actions"] >= actions:
                self.modules["hl_reasoner"].started = min(self.modules["hl_reasoner"].started,
                                                         self.modules["ll_reasoner"].started - 3600)
            if self.terminated:
                for name in self.modules:
                    self.modules[name].on_last(self.states[name])
                return
            self.deliver()
        raise AssertionError(f"Loop did not finish: {self.states['ll_reasoner']}")

    def clean_states(self):
        """Remove transport-only harness fields before result reconstruction."""
        return {name: state.dump() for name, state in self.states.items()}
