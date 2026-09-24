"""Immutable treatment and stopping budgets for Rainbow with HER."""

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping

from .policy import POLICY_ARCHITECTURE, POLICY_ARCHITECTURES


PROTOCOL_VERSION = "2-2-bw-rainbow-her-v3"


@dataclass(frozen=True)
class DQNProtocol:
    """Define one reproducible training treatment, including smoke overrides."""

    synchronized_training: bool = True
    total_training_transitions: int = 10_000
    warmup_transitions: int = 1_280
    replay_capacity: int = 100_000
    batch_size: int = 128
    n_steps: int = 3
    her_future_goals: int = 4
    discount: float = 0.99
    priority_alpha: float = 0.6
    priority_epsilon: float = 1e-6
    beta_start: float = 0.4
    actor_publish_steps: int = 10
    target_sync_steps: int = 100
    async_training_seconds: float = 3_600.0
    sync_safety_seconds: float = 10_800.0
    evaluation_seconds: float = 300.0
    shutdown_seconds: float = 30.0
    max_episode_length: int = 200
    evaluation_seeds: tuple[int, ...] = tuple(range(2_200, 2_210))
    behavior_window_transitions: int = 200
    optimization_window_updates: int = 20
    smoke: bool = False
    action_masking: bool = False
    min_updates_per_transition: float = 0.0
    evaluation_interval_seconds: float = 0.0
    network_architecture: str = POLICY_ARCHITECTURE

    def __post_init__(self) -> None:
        """Reject combinations that cannot supply a synchronous warm-up batch."""
        if self.network_architecture not in POLICY_ARCHITECTURES:
            raise ValueError("Unknown BW network architecture")
        if type(self.synchronized_training) is not bool or type(self.smoke) is not bool:
            raise ValueError("Scheduling and smoke flags must be booleans")
        if type(self.action_masking) is not bool:
            raise ValueError('Action masking must be Boolean')
        for value in (self.min_updates_per_transition, self.evaluation_interval_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Pacing ratio and evaluation interval must be finite and nonnegative')
        if self.synchronized_training and self.min_updates_per_transition:
            raise ValueError('Collection pacing requires asynchronous scheduling')
        for name in ("total_training_transitions", "warmup_transitions", "replay_capacity",
                     "batch_size", "n_steps", "her_future_goals", "actor_publish_steps", "target_sync_steps",
                     "max_episode_length", "behavior_window_transitions", "optimization_window_updates"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.warmup_transitions < self.batch_size + self.n_steps - 1:
            raise ValueError("Warm-up must leave a full batch of mature n-step entries")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must hold a batch")
        if self.total_training_transitions < self.warmup_transitions:
            raise ValueError("Synchronous collection budget must reach warm-up")
        for name in ("async_training_seconds", "sync_safety_seconds", "evaluation_seconds",
                     "shutdown_seconds", "priority_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in
                   (self.discount, self.priority_alpha, self.beta_start)):
            raise ValueError("Discount, alpha, and beta must be in [0, 1]")
        if (not isinstance(self.evaluation_seeds, tuple) or not self.evaluation_seeds
                or any(type(seed) is not int or seed < 0 for seed in self.evaluation_seeds)
                or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds)):
            raise ValueError("Evaluation seeds must be non-empty and unique")

    @property
    def training_updates(self) -> int:
        """Return the synchronous optimizer budget."""
        return self.total_training_transitions - self.warmup_transitions + 1

    @property
    def training_seconds(self) -> float:
        """Return the training deadline, a failure ceiling in synchronous mode."""
        return self.sync_safety_seconds if self.synchronized_training else self.async_training_seconds

    @property
    def duration(self) -> float:
        """Return the outer agent safety ceiling."""
        return self.training_seconds + self.evaluation_seconds + self.shutdown_seconds

    def publication_count(self, updates: int) -> int:
        """Count distinct weight publications, including an off-cadence final model."""
        return len({1, updates} | set(range(self.actor_publish_steps, updates + 1, self.actor_publish_steps))) if updates else 0

    def record(self) -> dict[str, Any]:
        """Return complete JSON-safe cohort provenance."""
        config = asdict(self)
        # Preserve the exact provenance schema of existing v3 baseline archives.
        if self.network_architecture == POLICY_ARCHITECTURE:
            config.pop("network_architecture")
        config["evaluation_seeds"] = list(self.evaluation_seeds)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "scheduling_mode": "synchronous" if self.synchronized_training else "asynchronous",
            "algorithm": "double-dqn-3step-per-her-future",
            "her_strategy": "future-new-on-relations",
            "her_future_goals": self.her_future_goals,
            "config": config,
            "warmup_transitions": self.warmup_transitions,
            "total_training_transitions": self.total_training_transitions if self.synchronized_training else None,
            "training_updates": self.training_updates if self.synchronized_training else None,
            "target_sync_steps": self.target_sync_steps,
            "evaluation_seeds": list(self.evaluation_seeds),
            "behavior_window_transitions": self.behavior_window_transitions,
            "optimization_window_updates": self.optimization_window_updates,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DQNProtocol":
        """Reconstruct current provenance, rejecting inconsistent derived fields."""
        config = dict(record["config"])
        config["evaluation_seeds"] = tuple(config["evaluation_seeds"])
        protocol = cls(**config)
        if protocol.record() != dict(record):
            raise ValueError("Inconsistent Rainbow protocol record")
        return protocol


def module_protocol(kwargs: Mapping[str, Any]) -> DQNProtocol:
    """Resolve explicit protocol or the existing module-level scheduling override."""
    protocol = kwargs.get("protocol", DQNProtocol())
    if not isinstance(protocol, DQNProtocol):
        raise TypeError("protocol must be a DQNProtocol")
    if "synchronized_training" in kwargs:
        protocol = replace(protocol, synchronized_training=bool(kwargs["synchronized_training"]))
    return protocol
