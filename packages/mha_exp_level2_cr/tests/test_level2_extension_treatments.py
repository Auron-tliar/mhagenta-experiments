"""Model-free checks for the frozen Level 2 Crafter treatments."""

from __future__ import annotations

from collections import Counter

from mha_exp_level2_cr.exp2_1.treatment import treatment_for_run
from mha_exp_level2_cr.exp2_2.modules import (
    EVALUATION_SEEDS,
    TOTAL_TRAINING_TRANSITIONS,
    TRAINING_START_THRESHOLD,
    TRAINING_UPDATES,
)
from mha_exp_level2_cr.exp2_3.treatment import TASKS as BDI_TASKS
from mha_exp_level2_cr.exp2_4.treatment import TASKS as ACTIVITY_TASKS


def test_reactive_runs_share_one_diamond_treatment() -> None:
    """Run IDs vary the seed, without assigning different success goals."""
    treatment = treatment_for_run(0)
    assert treatment["target_achievement"] == "collect_diamond"
    assert treatment["action_budget"] == 1000
    assert "tier" not in treatment
    assert all(treatment_for_run(run) == treatment for run in range(50))


def test_bdi_seed_manifest_is_balanced_by_initial_support() -> None:
    assert len(BDI_TASKS) == 50
    assert Counter(task.initial_support for task in BDI_TASKS) == {
        "wood-supported": 25,
        "exploration-first": 25,
    }


def test_diamond_manifest_has_sixteen_from_scratch_seeds() -> None:
    assert len(ACTIVITY_TASKS) == 16
    assert all(task.stratum == "ordinary_from_scratch" for task in ACTIVITY_TASKS)
    assert all(task.primary_intention == "obtain_diamond" for task in ACTIVITY_TASKS)
    assert all(task.duration == 600 and task.total_action_cap == 900 for task in ACTIVITY_TASKS)


def test_crafter_learning_budget_and_frozen_evaluation_are_exact() -> None:
    assert TOTAL_TRAINING_TRANSITIONS == 10_000
    assert TRAINING_START_THRESHOLD == 512
    assert TRAINING_UPDATES == 9_488
    assert EVALUATION_SEEDS == tuple(range(920_000, 920_010))
