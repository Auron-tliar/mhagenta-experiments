"""Model-free checks for the frozen Level 2 Blocks World treatments."""

from __future__ import annotations

from collections import Counter

from mha_exp_level2_bw.exp2_1.treatment import TASKS as REACTIVE_TASKS
from mha_exp_level2_bw.exp2_2.modules import (
    TOTAL_TRAINING_TRANSITIONS,
    TRAINING_UPDATES,
    WARMUP_TRANSITIONS,
)
from mha_exp_level2_bw.exp2_3.treatment import TASKS as BDI_TASKS
from mha_exp_level2_bw.paired_treatment import TASKS as PAIRED_TASKS


def test_reactive_manifest_has_balanced_unique_strata() -> None:
    assert len(REACTIVE_TASKS) == 24
    assert len({task.task_id for task in REACTIVE_TASKS}) == 24
    assert Counter(task.difficulty for task in REACTIVE_TASKS) == {
        "easy": 8, "medium": 8, "hard": 8,
    }


def test_bdi_manifest_has_distinct_worlds_and_goals_in_each_layout() -> None:
    assert len(BDI_TASKS) == 50
    assert len({task.seed for task in BDI_TASKS}) == 50
    assert len({task.initial_state_digest for task in BDI_TASKS}) == 50
    assert len({(task.top, task.bottom) for task in BDI_TASKS}) == 50
    assert Counter(
        (task.table_len, task.num_blocks) for task in BDI_TASKS
    ) == {(4, 6): 17, (5, 8): 17, (7, 12): 16}


def test_bdi_seeds_reproduce_distinct_arrangements_and_unsatisfied_goals() -> None:
    from mha_env_blocksworld import BlocksWorldEnv
    from mha_exp_level2_bw.exp2_3.planning import beliefs_to_facts, parse_symbolic_observation
    from mha_exp_level2_bw.exp2_3.treatment import state_digest, treatment_for_run
    import pytest

    arrangements = set()
    for run_id, task in enumerate(BDI_TASKS):
        env = BlocksWorldEnv(table_len=task.table_len, num_blocks=task.num_blocks, symbolic=True)
        observation, _ = env.reset(seed=task.seed)
        env.close()
        facts = beliefs_to_facts(parse_symbolic_observation(observation))
        assert state_digest(facts) == task.initial_state_digest
        assert f"on({task.top},{task.bottom})" not in facts
        arrangements.add(tuple(sorted(f for f in facts if f.startswith("on("))))
        assert treatment_for_run(run_id)["seed"] == task.seed
    assert len(arrangements) == 50
    for invalid in (-1, 50, True, 1.5):
        with pytest.raises(ValueError):
            treatment_for_run(invalid)


def test_paired_manifest_has_four_tasks_per_band_and_two_transfers() -> None:
    assert len(PAIRED_TASKS) == 12
    assert Counter(task.difficulty for task in PAIRED_TASKS) == {
        "easy": 4, "medium": 4, "hard": 4,
    }
    assert all(task.transfer_count >= 2 for task in PAIRED_TASKS)


def test_bw_learning_budget_has_one_update_per_eligible_transition() -> None:
    assert TOTAL_TRAINING_TRANSITIONS == 10_000
    assert WARMUP_TRANSITIONS == 1_280
    assert TRAINING_UPDATES == 8_721
