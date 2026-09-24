"""Frozen schedules, dependencies, and seed accounting for v3 preparation."""

from __future__ import annotations

from dataclasses import dataclass

from cases import CaseRequest
from mha_exp_level2_cr.exp2_5.contracts import RESOURCE_ITEMS
from mha_exp_level2_cr.exp2_5.policy import (
    POLICY_FILENAMES,
    TRAINING_CONFIG,
    PolicyId,
)

CONFIG = TRAINING_CONFIG
RESOURCE_KINDS = tuple(RESOURCE_ITEMS)
POLICY_ORDER = tuple(PolicyId)
NAVIGATION_BANDS = {"short": (2, 3), "mid": (4, 6), "long": (7, 10)}
DISCOVERY_BANDS = {"short": (2, 4), "mid": (5, 8), "long": (9, 12)}
DETERMINISM_SETTINGS = {
    "python_seed": CONFIG["master_seed"],
    "numpy_seed": CONFIG["master_seed"],
    "torch_seed": CONFIG["master_seed"],
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
}
@dataclass(frozen=True)
class PolicySpec:
    """Training curriculum and selection requirements for one independent policy."""

    policy_id: PolicyId
    filename: str
    target_kinds: tuple[str, ...]
    selection_attempts: int
    selection_successes: int


POLICY_SPECS = tuple(
    PolicySpec(policy, POLICY_FILENAMES[policy], kinds, attempts, successes)
    for policy, kinds, attempts, successes in (
        (PolicyId.EXPLORE, ("frontier",), 12, 11),
        (PolicyId.NAVIGATE_TO, ("safe_cell",), 24, 22),
        (PolicyId.GET_RESOURCE, RESOURCE_KINDS, 24, 22),
        (PolicyId.EAT_TARGET, ("cow",), 24, 20),
        (PolicyId.EAT_COW, ("cow",), 24, 18),
    )
)
POLICY_SPEC_BY_ID = {spec.policy_id: spec for spec in POLICY_SPECS}


def _request(
    policy: PolicyId,
    target_kind: str,
    band: str,
    direction: int,
    variant: str,
    *,
    discovery: bool = False,
) -> CaseRequest:
    bounds = (DISCOVERY_BANDS if discovery else NAVIGATION_BANDS)[band]
    if policy is PolicyId.NAVIGATE_TO and band == "short" and variant == "obstructed":
        # A four-neighbor detour adds at least two steps to a non-adjacent target.
        bounds = bounds[0], 4
    return CaseRequest(
        policy,
        target_kind,
        distance_band=band,
        minimum_distance=bounds[0],
        maximum_distance=bounds[1],
        first_direction=direction,
        variant=variant,
    )


def training_requests(policy: PolicyId) -> tuple[CaseRequest, ...]:
    """Return the exact repeating demonstration and online case cycle."""

    if policy is PolicyId.EXPLORE:
        return tuple(CaseRequest(policy, "frontier", first_direction=direction) for direction in range(1, 5))
    if policy is PolicyId.NAVIGATE_TO:
        return tuple(
            _request(policy, "safe_cell", band, direction, variant)
            for band in NAVIGATION_BANDS
            for direction in range(1, 5)
            for variant in ("clear", "obstructed")
        )
    if policy is PolicyId.GET_RESOURCE:
        variants = (
            ("short", "clear_aligned"),
            ("mid", "clear_misaligned"),
            ("long", "obstructed_aligned"),
            ("long", "obstructed_misaligned"),
        )
        return tuple(
            _request(
                policy,
                kind,
                band,
                1 + ((kind_index + variant_index) % 4),
                variant,
            )
            for kind_index, kind in enumerate(RESOURCE_KINDS)
            for variant_index, (band, variant) in enumerate(variants)
        )
    if policy is PolicyId.EAT_TARGET:
        return tuple(
            _request(policy, "cow", band, direction, variant)
            for band in NAVIGATION_BANDS
            for direction in range(1, 5)
            for variant in ("single", "distractor")
        )
    return tuple(
        _request(policy, "cow", band, sector, variant, discovery=True)
        for band in DISCOVERY_BANDS
        for sector in range(1, 5)
        for variant in ("ordinary", "reacquisition")
    )


def pilot_requests(policy: PolicyId) -> tuple[CaseRequest, ...]:
    """Return the exact feasibility schedule for one isolated pilot."""

    requests = training_requests(policy)
    if policy is PolicyId.EXPLORE:
        return requests
    if policy is PolicyId.GET_RESOURCE:
        return tuple(
            _request(
                policy,
                kind,
                band,
                1 + (emitted_index % 4),
                variant,
            )
            for kind_index, kind in enumerate(RESOURCE_KINDS)
            for variant_index, (band, variant) in enumerate((
                ("short", "clear_aligned"),
                ("long", "obstructed_misaligned"),
            ))
            for emitted_index in (kind_index * 2 + variant_index,)
        )
    return tuple(
        requests[pair_index * 2 + (band_index + direction) % 2]
        for band_index in range(3)
        for direction in range(1, 5)
        for pair_index in (band_index * 4 + direction - 1,)
    )


def selection_requests(policy: PolicyId) -> tuple[CaseRequest, ...]:
    """Return the frozen candidate-selection schedule."""

    requests = training_requests(policy)
    return requests * 3 if policy is PolicyId.EXPLORE else requests


class SeedConsumptionError(RuntimeError):
    """Carry the next unused seed when case construction exhausts its window."""

    def __init__(self, next_seed: int, error: Exception) -> None:
        super().__init__(str(error))
        self.next_seed = next_seed


def seed_windows() -> dict[str, tuple[int, int]]:
    """Return every disjoint seed-2508 preparation window."""

    windows: dict[str, tuple[int, int]] = {}
    for index, policy in enumerate(POLICY_ORDER):
        windows[f"pilot_{policy.value}"] = (4_800_000 + index * 2_000, 4_801_999 + index * 2_000)
        windows[f"demo_{policy.value}"] = (4_820_000 + index * 20_000, 4_839_999 + index * 20_000)
        windows[f"online_{policy.value}"] = (4_920_000 + index * 60_000, 4_979_999 + index * 60_000)
        windows[f"selection_{policy.value}"] = (5_220_000 + index * 2_000, 5_221_999 + index * 2_000)
    windows["probe"] = (5_240_000, 5_249_999)
    return windows


def priority_beta(environment_steps: int) -> float:
    """Return beta at an actual committed online step."""

    progress = min(1.0, max(0, environment_steps) / CONFIG["priority_beta_steps"])
    return CONFIG["priority_beta_start"] + progress * (
        CONFIG["priority_beta_end"] - CONFIG["priority_beta_start"]
    )


def epsilon_at(environment_steps: int) -> float:
    """Return epsilon at an actual committed online step."""

    progress = min(1.0, max(0, environment_steps) / CONFIG["epsilon_steps"])
    return CONFIG["epsilon_start"] + progress * (
        CONFIG["epsilon_end"] - CONFIG["epsilon_start"]
    )


def demonstration_step_target(policy: PolicyId) -> int:
    """Return the expert-transition target shared by all five policies."""

    del policy
    return int(CONFIG["nominal_demonstration_steps"])
