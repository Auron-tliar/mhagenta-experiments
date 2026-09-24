"""Regression tests for third-party startup and planning noise."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import warnings

import pytest

from mha_exp_level2_bw.exp2_3 import planning as atomic
from mha_exp_level2_bw.exp2_4 import planning as transfer


REPRESENTATIVE_OBSERVATION = [
    "HandEmpty()",
    "On(B0,t0)",
    "AtLoc(B0,t0)",
    "Clear(B0)",
    "On(B1,t1)",
    "AtLoc(B1,t1)",
    "Clear(B1)",
    "AtLoc(t0,t0)",
    "AtLoc(t1,t1)",
    "LeftOf(t0,t1)",
    "Above(t0)",
]


def test_blocksworld_import_hides_pygame_support_prompt() -> None:
    env = os.environ.copy()
    env.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)
    result = subprocess.run(
        [sys.executable, "-c", "import mha_env_blocksworld"],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
    assert "Hello from the pygame community" not in result.stdout
    assert "pkg_resources is deprecated as an API" not in result.stderr


def test_atomic_lpg_call_hides_third_party_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning_lpg(problem: Any, timeout: float) -> Any:
        warnings.warn(
            "pkg_resources is deprecated as an API. Remove it.",
            UserWarning,
        )
        warnings.warn(
            "We cannot establish whether lpg can solve this problem!",
            UserWarning,
        )
        return SimpleNamespace(status=SimpleNamespace(name="TIMEOUT"), plan=None)

    monkeypatch.setattr(atomic, "_run_lpg", warning_lpg)
    facts = atomic.beliefs_to_facts(
        atomic.parse_symbolic_observation(REPRESENTATIVE_OBSERVATION)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = atomic.plan_blocks_world(
            facts,
            atomic.GoalSpec("b0", "b1"),
            blocks=("b0", "b1"),
            locations=("t0", "t1"),
            timeout=1.0,
            problem_name="warning_probe",
        )

    assert result.failure == "planner-did-not-solve"
    assert not any(
        "We cannot establish whether" in str(item.message)
        or "pkg_resources is deprecated as an API" in str(item.message)
        for item in caught
    )


def test_transfer_planning_service_hides_third_party_warnings() -> None:
    class WarningPlanner:
        def __init__(self) -> None:
            warnings.warn(
                "pkg_resources is deprecated as an API. Remove it.",
                UserWarning,
            )

        def __enter__(self) -> WarningPlanner:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def solve(self, problem: Any, timeout: float) -> Any:
            warnings.warn(
                "We cannot establish whether lpg can solve this problem!",
                UserWarning,
            )
            return SimpleNamespace(status=SimpleNamespace(name="TIMEOUT"), plan=None)

    domain_path = Path(transfer.__file__).resolve().with_name(
        "blocksworld-transfer-domain.pddl"
    )
    service = transfer.PlanningService(
        domain_path=domain_path,
        blocks=("b0", "b1"),
        locations=("t0", "t1"),
        engines=("lpg",),
        planner_factory=lambda **_: WarningPlanner(),
    )
    facts = transfer.beliefs_to_facts(
        transfer.parse_symbolic_observation(REPRESENTATIVE_OBSERVATION)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service.solve(facts, transfer.GoalSpec("b0", "b1"), "warning_probe")

    assert not any(
        "We cannot establish whether" in str(item.message)
        or "pkg_resources is deprecated as an API" in str(item.message)
        for item in caught
    )
