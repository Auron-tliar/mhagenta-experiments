"""In-process delivery of the real six-module callbacks against real Crafter."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from mha_exp_common.names import (
    ACTUATOR,
    GOALGRAPH,
    HLREASONER,
    KNOWLEDGE,
    LLREASONER,
    PERCEPTOR,
)
from mha_exp_common.utils import Seeder
from mha_exp_level2_cr.exp2_5 import runtime
from mha_exp_level2_cr.exp2_5.environment import (
    CrafterNeurosymbolicEnvironment,
    initial_state,
)


class HarnessState(dict):
    """Expose the state/outbox surface used by synchronous behavior callbacks."""

    def __getattr__(self, name):
        return self[name]


class Outbox:
    """Queue typed message bodies instead of bypassing receiver callbacks."""

    def __init__(self, harness, sender):
        self.harness, self.sender = harness, sender

    def terminate_agent(self, reason):
        self.harness.terminated = reason

    def __getattr__(self, method):
        callbacks = {
            "request_observation": ("on_request", ()),
            "send_observation": ("on_observation", ("observation",)),
            "request_action": ("on_request", ()),
            "send_status": ("on_action_status", ("action_status",)),
            "send_goal_update": ("on_goal_update", ("goals",)),
            "send_goals": ("on_goal_update", ("goals",)),
        }
        if method == "send_beliefs":
            callback, fields = (("on_observed_beliefs", ("observation", "beliefs"))
                                if self.sender == LLREASONER else ("on_belief_update", ("beliefs",)))
        else:
            callback, fields = callbacks[method]

        def send(receiver, *args, **kwargs):
            self.harness.queue.append((receiver, callback, {
                "sender": self.sender, **dict(zip(fields, args, strict=True)), **kwargs,
            }))
        return send


class RuntimeHarness:
    """Exercise identical runtime callbacks with deterministic queued delivery."""

    def __init__(self, *, load_models=False):
        self.queue, self.terminated, self.logs = deque(), None, []
        self.env = CrafterNeurosymbolicEnvironment({
            **initial_state(), "seed": Seeder(819_000).environment, "expected_agent_id": "agent",
        })
        classes = {
            PERCEPTOR: runtime.CrafterRGBPerceptor,
            ACTUATOR: runtime.CrafterActivityActuator,
            LLREASONER: runtime.NeuralActivityLLReasoner,
            KNOWLEDGE: runtime.CrafterBeliefKnowledge,
            GOALGRAPH: runtime.CrafterActivityGoalGraph,
            HLREASONER: runtime.CrafterActivityHLReasoner,
        }
        directory = SimpleNamespace(
            internal=SimpleNamespace(**{
                key: [SimpleNamespace(module_id=value)] for key, value in {
                    "perception": PERCEPTOR, "actuation": ACTUATOR, "ll_reasoning": LLREASONER,
                    "knowledge": KNOWLEDGE, "goals": GOALGRAPH, "hl_reasoning": HLREASONER,
                }.items()
            }),
            external=SimpleNamespace(environment=SimpleNamespace(address={"env_id": "environment"})),
        )
        self.states, self.modules = {}, {}
        for name, data in runtime.initial_states().items():
            self.states[name] = HarnessState(**data, directory=directory, outbox=Outbox(self, name))
            self.modules[name] = classes[name](module_id=name, initial_state=data)
            self.modules[name].log = lambda level, message: self.logs.append(message)
        self.modules[PERCEPTOR].observe = self.observe
        self.modules[ACTUATOR].act = self.act
        if load_models:
            self.modules[LLREASONER].on_init()
        else:
            from mha_exp_level2_cr.exp2_5.grounding import (
                artifact_paths,
                load_grounding_templates,
            )
            self.modules[LLREASONER]._templates = load_grounding_templates(artifact_paths()[0])
        for name in (PERCEPTOR, ACTUATOR, KNOWLEDGE, GOALGRAPH, HLREASONER, LLREASONER):
            self.modules[name].on_first(self.states[name])

    def observe(self, env_id, **kwargs):
        _, response = self.env.on_observe(self.env.state, "agent", **kwargs)
        self.queue.append((PERCEPTOR, "on_observation", {"env_id": env_id, **response}))

    def act(self, env_id, **kwargs):
        _, response = self.env.on_action(self.env.state, "agent", **kwargs)
        if response is not None:
            self.queue.append((ACTUATOR, "on_status", {"env_id": env_id, **response}))

    def deliver(self):
        """Deliver one queued message and preserve its returned state."""

        name, callback, kwargs = self.queue.popleft()
        self.states[name] = getattr(self.modules[name], callback)(self.states[name], **kwargs)

    def run(self, limit=20000):
        """Run until termination or fail on a stalled/overlong callback chain."""

        for _ in range(limit):
            if self.terminated:
                self.modules[ACTUATOR].on_last(self.states[ACTUATOR])
                return
            if not self.queue:
                raise AssertionError("Reactive callback chain stalled.")
            self.deliver()
        raise AssertionError("Callback limit exceeded.")
