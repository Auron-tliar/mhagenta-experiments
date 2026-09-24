"""Selected structured DQfD-lite preparation implementation for 2-5-BW."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from mha_exp_level2_bw.exp2_5.contracts import TransferSpec

from mha_exp_level2_bw.exp2_5.grounding import (
    GroundedObservation,
    enumerate_transfer_targets,
    ground_observation,
    transfer_phase,
    transfer_succeeded,
)
from mha_exp_level2_bw.exp2_5.policy import (
    MAX_POLICY_STEPS,
    MODEL_INPUT_SHAPE,
    N_ACTIONS,
    array_sha256,
    build_q_network,
    condition_observation,
    greedy_inference,
)
from regression import PLANNER_REGRESSIONS, reconstruct_regression
from evaluation import (
    BATCH_SIZE,
    DISCOUNT,
    GRADIENT_CLIP,
    LEARNING_RATE,
    REPLAY_CAPACITY,
    REPLAY_WARMUP,
    TARGET_SYNC_STEPS,
    epsilon,
    evaluate_case,
    evaluate_sequential_case,
    evaluate_transfer,
    make_environment,
    require_torch,
    transition_reward,
)


VARIANT_DQFD_LITE = "structured-ddqn-per-n3-dqfd-lite"
DEFAULT_SEEDS = (2505, 2506, 2507)
TRAINING_ENV_SEED_START = 250_500
TRAINING_ENV_SEED_STRIDE = 100_000
DEMONSTRATION_SEED_OFFSET = 60_000
DEMONSTRATION_STEPS = 2_000
N_STEP_RETURN = 3
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_EPSILON = 1e-5
DEMONSTRATION_PRIORITY_BONUS = 0.1
LARGE_MARGIN = 0.8
LARGE_MARGIN_WEIGHT = 1.0
SELECTION_INTERVAL = 10_000
SELECTION_SINGLE_SEEDS = tuple(range(330_000, 330_020))
SELECTION_SEQUENCE_SEEDS = tuple(range(331_000, 331_010))
VALIDATION_SINGLE_SEEDS = tuple(range(340_000, 340_050))
VALIDATION_SEQUENCE_SEEDS = tuple(range(341_000, 341_020))
VALIDATION_SEQUENCE_LENGTH = 5
REGRESSION_DEMONSTRATION_REPETITIONS = 32


@dataclass(frozen=True)
class RawTransition:
    """One environment transition before n-step aggregation."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool
    demonstration: bool = False


@dataclass(frozen=True)
class ReplayTransition:
    """One replay transition with its bootstrapping discount."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool
    discount: float
    demonstration: bool = False


@dataclass(frozen=True)
class SelectedConfig:
    """Frozen choices for the selected preparation algorithm."""

    name: str
    structured_model: bool
    double_dqn: bool
    prioritized_replay: bool
    n_step: int
    dqfd_lite: bool


CONFIGS = {
    VARIANT_DQFD_LITE: SelectedConfig(
        VARIANT_DQFD_LITE,
        structured_model=True,
        double_dqn=True,
        prioritized_replay=True,
        n_step=N_STEP_RETURN,
        dqfd_lite=True,
    ),
}


class SelectedReplayBuffer:
    """Replay storage supporting uniform or sum-tree prioritized sampling."""

    def __init__(
        self,
        capacity: int,
        *,
        prioritized: bool,
        protected_capacity: int = 0,
        input_shape: tuple[int, int, int] = MODEL_INPUT_SHAPE,
    ) -> None:
        if protected_capacity < 0 or protected_capacity >= capacity:
            raise ValueError("protected_capacity must be in [0, capacity).")
        self.capacity = capacity
        self.prioritized = prioritized
        self.protected_capacity = protected_capacity
        self.states = np.empty((capacity, *input_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *input_shape), dtype=np.uint8)
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.demonstrations = np.empty(capacity, dtype=np.bool_)
        self.size = 0
        self.next_index = protected_capacity
        self.max_priority = 1.0
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity
        self.sum_tree = np.zeros(2 * tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * tree_capacity, np.inf, dtype=np.float64)

    def __len__(self) -> int:
        return self.size

    def append(self, transition: ReplayTransition) -> None:
        """Append a transition without overwriting protected demonstrations."""

        if transition.demonstration:
            if self.size >= self.protected_capacity:
                raise ValueError("Protected demonstration capacity is exhausted.")
            index = self.size
        else:
            index = self.next_index
            self.next_index += 1
            if self.next_index >= self.capacity:
                self.next_index = self.protected_capacity
        self.states[index] = transition.state
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.next_states[index] = transition.next_state
        self.terminals[index] = transition.terminal
        self.discounts[index] = transition.discount
        self.demonstrations[index] = transition.demonstration
        self.size = min(self.size + 1, self.capacity)
        self._set_priority(index, self.max_priority)

    def _set_priority(self, index: int, priority: float) -> None:
        scaled = max(float(priority), PRIORITY_EPSILON) ** PRIORITY_ALPHA
        tree_index = self.tree_capacity + index
        self.sum_tree[tree_index] = scaled
        self.min_tree[tree_index] = scaled
        tree_index //= 2
        while tree_index:
            left = 2 * tree_index
            self.sum_tree[tree_index] = self.sum_tree[left] + self.sum_tree[left + 1]
            self.min_tree[tree_index] = min(
                self.min_tree[left],
                self.min_tree[left + 1],
            )
            tree_index //= 2

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update sampled priorities after a TD-error calculation."""

        raw_priorities = np.maximum(priorities, PRIORITY_EPSILON)
        self.max_priority = max(self.max_priority, float(raw_priorities.max()))
        order = np.argsort(indices)
        sorted_indices = indices[order]
        sorted_priorities = raw_priorities[order]
        unique_indices, starts = np.unique(sorted_indices, return_index=True)
        unique_priorities = np.maximum.reduceat(sorted_priorities, starts)
        tree_indices = self.tree_capacity + unique_indices
        scaled = unique_priorities**PRIORITY_ALPHA
        self.sum_tree[tree_indices] = scaled
        self.min_tree[tree_indices] = scaled
        while tree_indices.size:
            tree_indices = np.unique(tree_indices // 2)
            tree_indices = tree_indices[tree_indices > 0]
            if not tree_indices.size:
                break
            left = 2 * tree_indices
            self.sum_tree[tree_indices] = self.sum_tree[left] + self.sum_tree[left + 1]
            self.min_tree[tree_indices] = np.minimum(
                self.min_tree[left],
                self.min_tree[left + 1],
            )

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample replay indices and normalized importance weights."""

        if not self.prioritized:
            indices = rng.choice(self.size, size=batch_size, replace=False)
            return indices, np.ones(batch_size, dtype=np.float32)
        total = self.sum_tree[1]
        segment = total / batch_size
        masses = (np.arange(batch_size) + rng.random(batch_size)) * segment
        tree_indices = np.ones(batch_size, dtype=np.int64)
        while tree_indices[0] < self.tree_capacity:
            left = 2 * tree_indices
            left_sums = self.sum_tree[left]
            go_right = masses > left_sums
            masses = masses - go_right * left_sums
            tree_indices = left + go_right
        indices = tree_indices - self.tree_capacity
        probabilities = self.sum_tree[self.tree_capacity + indices] / total
        minimum_probability = self.min_tree[1] / total
        weights = (self.size * probabilities) ** (-beta)
        maximum_weight = (self.size * minimum_probability) ** (-beta)
        return indices, (weights / maximum_weight).astype(np.float32)

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the arrays associated with sampled indices."""

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
            self.discounts[indices],
            self.demonstrations[indices],
        )


def aggregate_n_step(
    transitions: Sequence[RawTransition],
    n_step: int,
) -> ReplayTransition:
    """Aggregate up to ``n_step`` consecutive transitions into one target."""

    if not transitions or n_step <= 0:
        raise ValueError("At least one transition and a positive n_step are required.")
    reward = 0.0
    count = 0
    final = transitions[0]
    for count, transition in enumerate(transitions[:n_step], start=1):
        reward += (DISCOUNT ** (count - 1)) * transition.reward
        final = transition
        if transition.terminal:
            break
    return ReplayTransition(
        state=transitions[0].state,
        action=transitions[0].action,
        reward=reward,
        next_state=final.next_state,
        terminal=final.terminal,
        discount=DISCOUNT**count,
        demonstration=transitions[0].demonstration,
    )


def emit_n_step(
    pending: deque[RawTransition],
    n_step: int,
    *,
    flush: bool,
) -> list[ReplayTransition]:
    """Emit ready n-step transitions and optionally flush a terminal queue."""

    emitted: list[ReplayTransition] = []
    while pending and (len(pending) >= n_step or flush):
        emitted.append(aggregate_n_step(tuple(pending), n_step))
        pending.popleft()
    return emitted


def expert_action(observation: np.ndarray, spec: TransferSpec) -> int:
    """Return the direct controller action used only for DQfD-lite data."""

    grounded = ground_observation(observation)
    arm_column = int(grounded.arm_location[1:])
    source_column = int(spec.source[1:])
    destination_column = int(spec.destination[1:])
    if grounded.held_block is None:
        if arm_column < source_column:
            return 3
        if arm_column > source_column:
            return 2
        return 0
    if arm_column < destination_column:
        return 3
    if arm_column > destination_column:
        return 2
    return 1


def _collect_demonstrations(
    replay: SelectedReplayBuffer,
    rng: np.random.Generator,
    environment_seed_start: int,
) -> dict[str, int]:
    environment = make_environment()
    pending: deque[RawTransition] = deque()
    environment_steps = 0
    completed_transfers = 0
    reset_index = 0
    regression_transitions = 0
    grounded: GroundedObservation | None = None
    try:
        for case in PLANNER_REGRESSIONS:
            for _ in range(REGRESSION_DEMONSTRATION_REPETITIONS):
                observation = reconstruct_regression(environment, case)
                grounded = ground_observation(observation)
                highest_phase = transfer_phase(grounded, case.spec)
                pending.clear()
                while not transfer_succeeded(grounded.facts, case.spec):
                    model_input = condition_observation(grounded.observation, case.spec)
                    action = expert_action(grounded.observation, case.spec)
                    next_observation, _, _, _, info = environment.step(action)
                    next_grounded = ground_observation(next_observation)
                    snapshot = info.get("snapshot")
                    illegal = snapshot is not None and not bool(snapshot.legal)
                    reward, outcome, highest_phase = transition_reward(
                        next_grounded,
                        case.spec,
                        highest_phase,
                        illegal,
                    )
                    environment_steps += 1
                    terminal = outcome is not None
                    pending.append(
                        RawTransition(
                            model_input,
                            action,
                            reward,
                            condition_observation(next_grounded.observation, case.spec),
                            terminal,
                            demonstration=True,
                        )
                    )
                    emitted = emit_n_step(
                        pending,
                        N_STEP_RETURN,
                        flush=terminal,
                    )
                    for transition in emitted:
                        replay.append(transition)
                    regression_transitions += len(emitted)
                    grounded = next_grounded
                completed_transfers += 1
        grounded = None
        while environment_steps < DEMONSTRATION_STEPS:
            if grounded is None:
                observation, _ = environment.reset(seed=environment_seed_start + reset_index)
                reset_index += 1
                grounded = ground_observation(observation)
            candidates = enumerate_transfer_targets(grounded.facts)
            if not candidates:
                grounded = None
                continue
            spec = candidates[int(rng.integers(len(candidates)))]
            highest_phase = transfer_phase(grounded, spec)
            pending.clear()
            for episode_step in range(1, MAX_POLICY_STEPS + 1):
                model_input = condition_observation(grounded.observation, spec)
                action = expert_action(grounded.observation, spec)
                next_observation, _, _, _, info = environment.step(action)
                next_grounded = ground_observation(next_observation)
                snapshot = info.get("snapshot")
                illegal = snapshot is not None and not bool(snapshot.legal)
                reward, outcome, highest_phase = transition_reward(
                    next_grounded,
                    spec,
                    highest_phase,
                    illegal,
                )
                environment_steps += 1
                terminal = (
                    outcome is not None
                    or episode_step == MAX_POLICY_STEPS
                    or environment_steps == DEMONSTRATION_STEPS
                )
                pending.append(
                    RawTransition(
                        model_input,
                        action,
                        reward,
                        condition_observation(next_grounded.observation, spec),
                        terminal,
                        demonstration=True,
                    )
                )
                for transition in emit_n_step(
                    pending,
                    N_STEP_RETURN,
                    flush=terminal,
                ):
                    replay.append(transition)
                grounded = next_grounded
                if terminal:
                    if outcome == "succeeded":
                        completed_transfers += 1
                    else:
                        grounded = None
                    break
    finally:
        environment.close()
    return {
        "environment_steps": environment_steps,
        "transitions": len(replay),
        "completed_transfers": completed_transfers,
        "environment_resets": reset_index,
        "regression_transitions": regression_transitions,
        "regression_repetitions": REGRESSION_DEMONSTRATION_REPETITIONS,
    }


def legal_action_mask(torch_module: Any, states: Any) -> Any:
    """Return legal native actions for a batch of conditioned numeric states."""
    num_blocks = states.shape[3]
    columns = states.shape[2]
    arm = states[:, 0].sum(2).argmax(1)
    holding = states[:, 1].sum((1, 2)) > 0
    occupied = states[:, 2 : num_blocks + 2].sum((1, 3)) > 0
    pickup = occupied.gather(1, arm[:, None]).squeeze(1) & ~holding
    return torch_module.stack((pickup, holding, arm > 0, arm < columns - 1), dim=1)


def _optimize(
    torch_module: Any,
    online: Any,
    target: Any,
    optimizer: Any,
    replay: SelectedReplayBuffer,
    rng: np.random.Generator,
    device: str,
    config: SelectedConfig,
    beta: float,
    *,
    mask_legal_actions: bool = False,
) -> tuple[float, float]:
    indices, importance_weights = replay.sample(rng, BATCH_SIZE, beta)
    (
        batch_states,
        batch_actions,
        batch_rewards,
        batch_next_states,
        batch_terminals,
        batch_discounts,
        batch_demonstrations,
    ) = replay.batch(indices)
    states = torch_module.as_tensor(batch_states, dtype=torch_module.float32, device=device)
    actions = torch_module.as_tensor(batch_actions, dtype=torch_module.int64, device=device)
    rewards = torch_module.as_tensor(batch_rewards, dtype=torch_module.float32, device=device)
    next_states = torch_module.as_tensor(
        batch_next_states,
        dtype=torch_module.float32,
        device=device,
    )
    terminals = torch_module.as_tensor(
        batch_terminals,
        dtype=torch_module.float32,
        device=device,
    )
    discounts = torch_module.as_tensor(
        batch_discounts,
        dtype=torch_module.float32,
        device=device,
    )
    weights = torch_module.as_tensor(
        importance_weights,
        dtype=torch_module.float32,
        device=device,
    )

    q_values = online(states)
    predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch_module.no_grad():
        if config.double_dqn:
            next_q = online(next_states)
            if mask_legal_actions:
                next_q = next_q.masked_fill(~legal_action_mask(torch_module, next_states), -torch_module.inf)
            next_actions = next_q.argmax(1, keepdim=True)
            next_values = target(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_values = target(next_states).max(1).values
        expected = rewards + discounts * (1.0 - terminals) * next_values
    td_errors = expected - predicted
    td_loss = torch_module.nn.functional.smooth_l1_loss(
        predicted,
        expected,
        reduction="none",
    )
    loss = (weights * td_loss).mean()
    margin_loss = torch_module.zeros((), dtype=torch_module.float32, device=device)
    if config.dqfd_lite and np.any(batch_demonstrations):
        demo_mask = torch_module.as_tensor(batch_demonstrations, device=device)
        margins = torch_module.full_like(q_values, LARGE_MARGIN)
        margins.scatter_(1, actions.unsqueeze(1), 0.0)
        alternatives = q_values + margins
        if mask_legal_actions:
            alternatives = alternatives.masked_fill(~legal_action_mask(torch_module, states), -torch_module.inf)
        violations = alternatives.max(1).values - predicted
        margin_loss = torch_module.relu(violations[demo_mask]).mean()
        loss = loss + LARGE_MARGIN_WEIGHT * margin_loss

    if not bool(torch_module.isfinite(loss)):
        raise FloatingPointError("Non-finite DQfD loss; refusing to update weights.")
    optimizer.zero_grad()
    loss.backward()
    torch_module.nn.utils.clip_grad_norm_(online.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
    optimizer.step()
    if config.prioritized_replay:
        priorities = np.abs(td_errors.detach().cpu().numpy()) + PRIORITY_EPSILON
        priorities += batch_demonstrations * DEMONSTRATION_PRIORITY_BONUS
        replay.update_priorities(indices, priorities)
    return float(loss.detach().cpu().item()), float(margin_loss.detach().cpu().item())


def _evaluate_seed_sets(
    torch_module: Any,
    model: Any,
    single_seeds: Sequence[int],
    sequence_seeds: Sequence[int],
) -> dict[str, Any]:
    environment = make_environment()
    model.eval()
    try:
        single = [
            evaluate_case(
                torch_module,
                model,
                environment,
                seed,
                seed,
            )
            for seed in single_seeds
        ]
        sequences = [
            evaluate_sequential_case(
                torch_module,
                model,
                environment,
                seed,
                VALIDATION_SEQUENCE_LENGTH,
            )
            for seed in sequence_seeds
        ]
        regressions: list[dict[str, Any]] = []
        for case in PLANNER_REGRESSIONS:
            observation = reconstruct_regression(environment, case)
            if array_sha256(condition_observation(observation, case.spec)) != case.input_sha256:
                raise RuntimeError(f"Regression {case.name!r} input digest changed.")
            result, _ = evaluate_transfer(
                torch_module,
                model,
                environment,
                observation,
                case.spec,
                case.environment_seed,
            )
            regressions.append({"name": case.name, **asdict(result)})
    finally:
        environment.close()
    return {
        "single_transfer_successes": sum(result.success for result in single),
        "single_transfer_cases": len(single),
        "five_transfer_sequence_successes": sum(result.success for result in sequences),
        "five_transfer_sequence_cases": len(sequences),
        "successful_transfers_in_sequences": sum(
            sum(transfer.success for transfer in result.transfers)
            for result in sequences
        ),
        "requested_transfers_in_sequences": len(sequences) * VALIDATION_SEQUENCE_LENGTH,
        "regression_successes": sum(result["success"] for result in regressions),
        "regression_cases": len(regressions),
        "regression_results": regressions,
        "single_transfer_results": [asdict(result) for result in single],
        "five_transfer_sequence_results": [asdict(result) for result in sequences],
    }


def _selection_score(evaluation: dict[str, Any]) -> tuple[int, int, int, int, int]:
    action_count = sum(
        transfer["episode_length"]
        for result in evaluation["five_transfer_sequence_results"]
        for transfer in result["transfers"]
    )
    return (
        evaluation["regression_successes"],
        evaluation["five_transfer_sequence_successes"],
        evaluation["successful_transfers_in_sequences"],
        evaluation["single_transfer_successes"],
        -action_count,
    )


def _training_environment_start(seed: int) -> int:
    return TRAINING_ENV_SEED_START + abs(seed - DEFAULT_SEEDS[0]) * TRAINING_ENV_SEED_STRIDE


def train_selected(
    *,
    variant: str,
    seed: int,
    training_steps: int,
    output_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and validate the selected method without touching certification cases."""

    if variant not in CONFIGS:
        raise ValueError(f"Unknown selected method {variant!r}.")
    if training_steps <= 0:
        raise ValueError("training_steps must be positive.")
    config = CONFIGS[variant]
    torch_module = require_torch()
    selected_device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    rng = np.random.default_rng(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    online = build_q_network(torch_module).to(selected_device)
    target = deepcopy(online).to(selected_device)
    target.eval()
    optimizer = torch_module.optim.AdamW(online.parameters(), lr=LEARNING_RATE)
    protected_capacity = DEMONSTRATION_STEPS if config.dqfd_lite else 0
    replay = SelectedReplayBuffer(
        REPLAY_CAPACITY + protected_capacity,
        prioritized=config.prioritized_replay,
        protected_capacity=protected_capacity,
    )
    training_environment_start = _training_environment_start(seed)
    demonstration = None
    if config.dqfd_lite:
        demonstration = _collect_demonstrations(
            replay,
            rng,
            training_environment_start + DEMONSTRATION_SEED_OFFSET,
        )

    environment = make_environment()
    pending: deque[RawTransition] = deque()
    grounded: GroundedObservation | None = None
    environment_steps = 0
    optimizer_steps = 0
    reset_index = 0
    episode_count = 0
    outcomes = {"succeeded": 0, "illegal-action": 0, "truncated": 0}
    sampled_losses: list[dict[str, Any]] = []
    selection_history: list[dict[str, Any]] = []
    best_score: tuple[int, int, int, int, int] | None = None
    best_state: dict[str, Any] | None = None
    best_step = 0
    best_optimizer_step = 0
    started = time.perf_counter()
    try:
        while environment_steps < training_steps:
            if grounded is None:
                observation, _ = environment.reset(
                    seed=training_environment_start + reset_index
                )
                reset_index += 1
                grounded = ground_observation(observation)
            candidates = enumerate_transfer_targets(grounded.facts)
            if not candidates:
                grounded = None
                continue
            spec = candidates[int(rng.integers(len(candidates)))]
            highest_phase = transfer_phase(grounded, spec)
            pending.clear()
            episode_count += 1
            outcome = "truncated"
            for episode_step in range(1, MAX_POLICY_STEPS + 1):
                model_input = condition_observation(grounded.observation, spec)
                if rng.random() < epsilon(environment_steps):
                    action = int(rng.integers(N_ACTIONS))
                else:
                    action, _, _ = greedy_inference(
                        torch_module,
                        online,
                        grounded.observation,
                        spec,
                    )
                next_observation, _, _, _, info = environment.step(action)
                next_grounded = ground_observation(next_observation)
                snapshot = info.get("snapshot")
                illegal = snapshot is not None and not bool(snapshot.legal)
                reward, terminal_outcome, highest_phase = transition_reward(
                    next_grounded,
                    spec,
                    highest_phase,
                    illegal,
                )
                if terminal_outcome is not None:
                    outcome = terminal_outcome
                environment_steps += 1
                terminal = (
                    terminal_outcome is not None
                    or episode_step == MAX_POLICY_STEPS
                    or environment_steps == training_steps
                )
                pending.append(
                    RawTransition(
                        model_input,
                        action,
                        reward,
                        condition_observation(next_grounded.observation, spec),
                        terminal,
                    )
                )
                for transition in emit_n_step(pending, config.n_step, flush=terminal):
                    replay.append(transition)
                    if len(replay) >= REPLAY_WARMUP:
                        beta_progress = environment_steps / training_steps
                        beta = PRIORITY_BETA_START + beta_progress * (
                            1.0 - PRIORITY_BETA_START
                        )
                        loss, margin_loss = _optimize(
                            torch_module,
                            online,
                            target,
                            optimizer,
                            replay,
                            rng,
                            selected_device,
                            config,
                            beta,
                        )
                        optimizer_steps += 1
                        if optimizer_steps % 500 == 0:
                            sampled_losses.append(
                                {
                                    "optimizer_step": optimizer_steps,
                                    "environment_step": environment_steps,
                                    "loss": loss,
                                    "margin_loss": margin_loss,
                                }
                            )
                        if optimizer_steps % TARGET_SYNC_STEPS == 0:
                            target.load_state_dict(online.state_dict())
                grounded = next_grounded

                if (
                    environment_steps % SELECTION_INTERVAL == 0
                    or environment_steps == training_steps
                ):
                    selection = _evaluate_seed_sets(
                        torch_module,
                        online,
                        SELECTION_SINGLE_SEEDS,
                        SELECTION_SEQUENCE_SEEDS,
                    )
                    score = _selection_score(selection)
                    print(
                        f"Selection at environment step {environment_steps}, "
                        f"optimizer step {optimizer_steps}: {score}",
                        flush=True,
                    )
                    selection_history.append(
                        {
                            "environment_step": environment_steps,
                            "optimizer_step": optimizer_steps,
                            "score": list(score),
                            **selection,
                        }
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_step = environment_steps
                        best_optimizer_step = optimizer_steps
                        best_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in online.state_dict().items()
                        }
                    online.train()
                if terminal:
                    outcomes[outcome] += 1
                    if outcome != "succeeded":
                        grounded = None
                    break
    finally:
        environment.close()

    if best_state is None:
        raise RuntimeError("Pilot did not produce a selectable checkpoint.")
    elapsed_seconds = time.perf_counter() - started
    online.load_state_dict(best_state)
    online.to(selected_device)
    validation = _evaluate_seed_sets(
        torch_module,
        online,
        VALIDATION_SINGLE_SEEDS,
        VALIDATION_SEQUENCE_SEEDS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch_module.save(
        {
            "variant": variant,
            "seed": seed,
            "search_environment_steps": training_steps,
            "selected_environment_step": best_step,
            "selected_optimizer_step": best_optimizer_step,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    report: dict[str, Any] = {
        "variant": variant,
        "config": asdict(config),
        "seed": seed,
        "device": selected_device,
        "training_environment_seed_start": training_environment_start,
        "training_environment_steps": environment_steps,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed_seconds,
        "episode_count": episode_count,
        "outcomes": outcomes,
        "demonstration": demonstration,
        "selected_environment_step": best_step,
        "selected_optimizer_step": best_optimizer_step,
        "selection_seed_ranges": {
            "single_transfer": [SELECTION_SINGLE_SEEDS[0], SELECTION_SINGLE_SEEDS[-1]],
            "five_transfer_sequence": [
                SELECTION_SEQUENCE_SEEDS[0],
                SELECTION_SEQUENCE_SEEDS[-1],
            ],
        },
        "validation_seed_ranges": {
            "single_transfer": [VALIDATION_SINGLE_SEEDS[0], VALIDATION_SINGLE_SEEDS[-1]],
            "five_transfer_sequence": [
                VALIDATION_SEQUENCE_SEEDS[0],
                VALIDATION_SEQUENCE_SEEDS[-1],
            ],
        },
        "selection_history": selection_history,
        "sampled_losses": sampled_losses,
        "validation": validation,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


