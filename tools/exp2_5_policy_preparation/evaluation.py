"""Shared environment and evaluation support for 2-5-BW policy preparation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from mha_exp_level2_bw.exp2_5.contracts import TransferSpec
from mha_exp_level2_bw.exp2_5.evaluation import EvaluationResult, evaluate_transfer

from mha_exp_level2_bw.exp2_5.grounding import (
    GroundedObservation,
    NUM_BLOCKS,
    TABLE_LEN,
    enumerate_transfer_targets,
    ground_observation,
    transfer_phase,
    transfer_succeeded,
)


HELD_OUT_SEED_START = 370_000
HELD_OUT_CASES = 100
INTEGRATION_SEED = 1000
SEQUENTIAL_EVALUATION_SEEDS = tuple(range(371_000, 371_020))
SEQUENTIAL_TRANSFERS = 10
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_024
BATCH_SIZE = 128
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 10.0
TARGET_SYNC_STEPS = 500
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000


@dataclass(frozen=True)
class SequentialEvaluationResult:
    """Result of a deterministic sequence of legal abstract transfers."""

    environment_seed: int
    requested_transfers: int
    success: bool
    transfers: list[EvaluationResult]


def require_torch() -> Any:
    """Import Torch or raise an actionable host-dependency error."""

    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch is required for policy pretraining. Run `uv sync --extra policies`."
        ) from exc


def make_environment() -> Any:
    """Construct the fixed numeric Blocks World training environment."""

    environment_module = import_module("mha_env_blocksworld")
    environment = environment_module.BlocksWorldEnv(
        table_len=TABLE_LEN,
        num_blocks=NUM_BLOCKS,
        symbolic=False,
    )
    environment.expose_snapshot = True
    return environment


def deterministic_case(
    environment: Any,
    seed: int,
    index: int,
) -> tuple[np.ndarray, TransferSpec]:
    """Reset an environment and select a reproducible legal transfer target."""

    observation, _ = environment.reset(seed=seed)
    grounded = ground_observation(observation)
    candidates = enumerate_transfer_targets(grounded.facts)
    if not candidates:
        raise RuntimeError(f"Environment seed {seed} produced no legal transfer target.")
    return grounded.observation, candidates[index % len(candidates)]


def epsilon(environment_steps: int) -> float:
    """Return the linearly annealed exploration rate."""

    progress = min(environment_steps, EPSILON_DECAY_STEPS) / EPSILON_DECAY_STEPS
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def transition_reward(
    grounded: GroundedObservation,
    spec: TransferSpec,
    highest_phase: int,
    illegal: bool,
) -> tuple[float, str | None, int]:
    """Calculate the fixed transfer reward and terminal outcome."""

    if transfer_succeeded(grounded.facts, spec):
        return 1.0, "succeeded", highest_phase
    if illegal:
        return -0.5, "illegal-action", highest_phase
    phase = transfer_phase(grounded, spec)
    progress = max(0, phase - highest_phase)
    return -0.01 + 0.10 * progress, None, max(highest_phase, phase)


def evaluate_case(
    torch_module: Any,
    model: Any,
    environment: Any,
    seed: int,
    index: int,
) -> EvaluationResult:
    """Evaluate a greedy policy on one deterministic transfer case."""

    observation, spec = deterministic_case(environment, seed, index)
    result, _ = evaluate_transfer(
        torch_module,
        model,
        environment,
        observation,
        spec,
        seed,
    )
    return result


def evaluate_sequential_case(
    torch_module: Any,
    model: Any,
    environment: Any,
    seed: int,
    transfer_count: int = SEQUENTIAL_TRANSFERS,
) -> SequentialEvaluationResult:
    """Evaluate greedy inference across evolving planner-reachable states."""

    observation, _ = environment.reset(seed=seed)
    current = ground_observation(observation).observation
    transfers: list[EvaluationResult] = []
    for transfer_index in range(transfer_count):
        candidates = enumerate_transfer_targets(ground_observation(current).facts)
        if not candidates:
            break
        spec = candidates[(seed + transfer_index) % len(candidates)]
        result, current = evaluate_transfer(
            torch_module,
            model,
            environment,
            current,
            spec,
            seed,
        )
        transfers.append(result)
        if not result.success:
            break
    return SequentialEvaluationResult(
        environment_seed=seed,
        requested_transfers=transfer_count,
        success=len(transfers) == transfer_count
        and all(result.success for result in transfers),
        transfers=transfers,
    )
