"""Fixed Rainbow and reward treatment, with bounded execution workloads."""

from dataclasses import asdict, dataclass
import math


PROTOCOL_VERSION = "2-2-cr-rainbow-diamond-v1"
RUNTIME_NAMESPACE = "2_2_cr"
STEP_REWARD = -0.001
ILLEGAL_ACTION_PENALTY = -0.1
DEATH_PENALTY = -5.0
TARGET_ACHIEVEMENT_REWARD = 10.0
NEED_THRESHOLD = 4
ACHIEVEMENT_REWARDS = {
    "collect_wood": 0.25, "place_table": 0.25,
    "collect_stone": 0.25, "collect_coal": 0.25,
    "make_wood_pickaxe": 0.5, "collect_iron": 0.5, "place_furnace": 0.5,
    "make_stone_pickaxe": 0.75, "make_iron_pickaxe": 1.0,
    "collect_diamond": TARGET_ACHIEVEMENT_REWARD,
}
SURVIVAL_REWARDS = dict.fromkeys(("collect_drink", "eat_cow", "eat_plant", "wake_up"), 0.1)
N_STEP = 3
DISCOUNT = 0.99
REPLAY_BUFFER_SIZE = 20_000
REPLAY_BATCH_SIZE = 128
TRAINING_START_THRESHOLD = 512
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 10.0
TARGET_SYNC_STEPS = 100
EPS_GREEDY_START = 0.9
EPS_GREEDY_END = 0.05
EPS_GREEDY_STEPS = 10_000
PRIORITY_ALPHA = 0.5
PRIORITY_EPSILON = 1e-6
BETA_START = 0.4


def runtime_identity(run: int) -> tuple[str, str, str]:
    """Return Docker/RabbitMQ names isolated from the parallel 2-2-BW run."""
    return (
        f"exp_agent{RUNTIME_NAMESPACE}_{run}",
        f"exp_env{RUNTIME_NAMESPACE}_{run}",
        f"mhagenta-{RUNTIME_NAMESPACE}-{run}",
    )


@dataclass(frozen=True)
class DQNWorkload:
    """Finite training/evaluation limits; algorithm settings remain fixed."""

    training_transitions: int | None = 10_000
    episode_action_limit: int = 1_000
    evaluation_seeds: tuple[int, ...] = tuple(range(920_000, 920_010))
    duration_seconds: float = 21_600.0
    synchronized_training: bool = True
    async_training_seconds: float = 3600.0
    evaluation_interval_seconds: float = 900.0
    action_masking: bool = False

    def __post_init__(self) -> None:
        if type(self.action_masking) is not bool:
            raise ValueError('Action masking must be Boolean.')
        if type(self.synchronized_training) is not bool:
            raise ValueError("Scheduling mode must be Boolean.")
        for value in (self.async_training_seconds, self.evaluation_interval_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Asynchronous time budgets must be positive and finite.")
        if not self.synchronized_training and self.duration_seconds < self.async_training_seconds + 330:
            raise ValueError("Allow 300 seconds evaluation and 30 seconds shutdown after training.")
        if not self.synchronized_training:
            object.__setattr__(self, 'training_transitions', None)
        elif type(self.training_transitions) is not int or self.training_transitions <= TRAINING_START_THRESHOLD:
            raise ValueError("Training must exceed the 512-transition warm-up.")
        if type(self.episode_action_limit) is not int or not 0 < self.episode_action_limit <= 10_000:
            raise ValueError("Episode action limit must be between 1 and 10,000.")
        if not self.evaluation_seeds or any(type(seed) is not int or seed < 0 for seed in self.evaluation_seeds):
            raise ValueError("Evaluation requires nonnegative integer seeds.")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("Evaluation seeds must be distinct.")
        object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("Duration must be finite and positive.")

    @property
    def training_updates(self) -> int | None:
        """One update per raw transition after warm-up."""
        return self.training_transitions - TRAINING_START_THRESHOLD if self.synchronized_training else None

    def dump(self) -> dict:
        """Return JSON-native workload metadata."""
        return {**asdict(self), "evaluation_seeds": list(self.evaluation_seeds)}


DEFAULT_WORKLOAD = DQNWorkload()
SMOKE_WORKLOAD = DQNWorkload(640, 100, (920_000, 920_001), 900.0)


def reward_metadata() -> dict:
    """Describe the complete reward contract for checkpoint validation."""
    return {
        "step": STEP_REWARD, "illegal_action": ILLEGAL_ACTION_PENALTY,
        "death": DEATH_PENALTY, "achievement_bonuses": dict(ACHIEVEMENT_REWARDS),
        "survival_bonuses": dict(SURVIVAL_REWARDS), "need_threshold": NEED_THRESHOLD,
        "bonus_limit": "once_per_achievement_per_episode",
        "replenishment": "achievement_delta_and_pre_need_at_most_threshold_and_post_need_increase",
        "sleep": "qualifying_sleep_entry_then_uninterrupted_natural_wake_up",
        "death_suppresses_bonuses": True,
    }


def algorithm_metadata(synchronized_training: bool = True) -> dict:
    """Describe fixed learning and exploration settings."""
    return {
        "double_dqn": True, "dueling": True, "n_step": N_STEP,
        "discount": DISCOUNT, "replay_capacity": REPLAY_BUFFER_SIZE,
        "batch_size": REPLAY_BATCH_SIZE, "warmup_transitions": TRAINING_START_THRESHOLD,
        "priority_alpha": PRIORITY_ALPHA, "priority_epsilon": PRIORITY_EPSILON,
        "sampling_replacement": True, "beta_start": BETA_START, "beta_end": 1.0,
        "importance_normalization": "sample_max", "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE, "weight_decay": 0.01,
        "gradient_clip": GRADIENT_CLIP, "loss": "importance_weighted_huber",
        "target_sync_updates": TARGET_SYNC_STEPS,
        "epsilon_start": EPS_GREEDY_START, "epsilon_end": EPS_GREEDY_END,
        "epsilon_decisions": EPS_GREEDY_STEPS if synchronized_training else None,
        "synchronized": synchronized_training,
        "truncation_bootstrap": True,
    }
