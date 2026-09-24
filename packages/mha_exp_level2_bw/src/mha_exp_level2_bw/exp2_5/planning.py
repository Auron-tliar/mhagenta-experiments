"""Experiment-local abstract planning service for experiment 2-5-BW."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
import warnings

from unified_planning.io import PDDLReader
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import OneshotPlanner, PlanValidator, get_environment

from .contracts import GoalSpec, build_problem_pddl, serialize_transfer_action


SOLVED_STATUSES = {"SOLVED_SATISFICING", "SOLVED_OPTIMALLY"}

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    get_environment().credits_stream = None


@dataclass(frozen=True)
class PlanningOutcome:
    """Compact result of trying the configured abstract planners."""

    accepted: bool
    engine: str | None
    status: str
    elapsed_seconds: float
    actions: list[dict[str, str]]
    validation_status: str | None
    attempts: list[dict[str, Any]]
    sanity_bound: int
    failure: str | None


class PlanningService:
    """Build, solve, and independently validate abstract transfer problems."""

    def __init__(
        self,
        *,
        domain_path: Path,
        blocks: Sequence[str],
        locations: Sequence[str],
        timeout: float = 15.0,
        engines: Sequence[str] = ("lpg", "enhsp"),
        planner_factory: Callable[..., Any] = OneshotPlanner,
    ) -> None:
        self.domain_text = domain_path.read_text(encoding="utf-8")
        self.blocks = tuple(name.lower() for name in blocks)
        self.locations = tuple(name.lower() for name in locations)
        self.timeout = timeout
        self.engines = tuple(engines)
        self._planner_factory = planner_factory
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API\..*",
                category=UserWarning,
            )
            self._reader = PDDLReader()

    def build_problem(self, facts: set[str], goal: GoalSpec, problem_name: str) -> Any:
        """Build a Unified Planning problem from the current abstract facts."""

        problem_pddl = build_problem_pddl(
            problem_name=problem_name,
            blocks=self.blocks,
            locations=self.locations,
            facts=facts,
            goal=goal,
        )
        return self._reader.parse_problem_string(self.domain_text, problem_pddl)

    def solve(self, facts: set[str], goal: GoalSpec, problem_name: str) -> PlanningOutcome:
        """Try LPG and then ENHSP, accepting only a validated transfer-only plan."""

        problem = self.build_problem(facts, goal, problem_name)
        sanity_bound = 4 * len(self.blocks)
        attempts: list[dict[str, Any]] = []

        for engine_name in self.engines:
            started = monotonic()
            attempt: dict[str, Any] = {
                "engine": engine_name,
                "status": "NOT_RUN",
                "elapsed_seconds": 0.0,
                "plan_length": None,
                "validated": False,
                "goal_reached": False,
                "transfer_only": False,
                "sanity_bound": sanity_bound,
                "accepted": False,
                "rejection": None,
                "error": None,
            }
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"pkg_resources is deprecated as an API\..*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"We cannot establish whether .* can solve this problem!",
                        category=UserWarning,
                    )
                    with self._planner_factory(name=engine_name) as planner:
                        result = planner.solve(problem, timeout=self.timeout)
                attempt["status"] = result.status.name
                plan = result.plan
                if result.status.name not in SOLVED_STATUSES or plan is None:
                    attempt["rejection"] = "planner-did-not-solve"
                elif not isinstance(plan, SequentialPlan):
                    attempt["rejection"] = "non-sequential-plan"
                else:
                    attempt["plan_length"] = len(plan.actions)
                    valid, transfer_only, validation_status, rejection = self._assess_plan(
                        problem=problem,
                        plan=plan,
                        sanity_bound=sanity_bound,
                    )
                    attempt.update(
                        validated=valid,
                        goal_reached=valid,
                        transfer_only=transfer_only,
                        validation_status=validation_status,
                        rejection=rejection,
                        accepted=valid and transfer_only and rejection is None,
                    )
                    if attempt["accepted"]:
                        attempt["elapsed_seconds"] = monotonic() - started
                        attempts.append(attempt)
                        return PlanningOutcome(
                            accepted=True,
                            engine=engine_name,
                            status=str(attempt["status"]),
                            elapsed_seconds=float(attempt["elapsed_seconds"]),
                            actions=[serialize_transfer_action(action) for action in plan.actions],
                            validation_status=validation_status,
                            attempts=attempts,
                            sanity_bound=sanity_bound,
                            failure=None,
                        )
            except Exception as exc:
                attempt["status"] = "EXCEPTION"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                attempt["rejection"] = "planner-exception"
            attempt["elapsed_seconds"] = monotonic() - started
            attempts.append(attempt)

        failure = str(attempts[-1].get("rejection") or "planner-did-not-solve") if attempts else "no-planner"
        return PlanningOutcome(
            accepted=False,
            engine=None,
            status=str(attempts[-1].get("status", "NOT_RUN")) if attempts else "NOT_RUN",
            elapsed_seconds=sum(float(item["elapsed_seconds"]) for item in attempts),
            actions=[],
            validation_status=None,
            attempts=attempts,
            sanity_bound=sanity_bound,
            failure=failure,
        )

    @staticmethod
    def _assess_plan(
        *,
        problem: Any,
        plan: SequentialPlan,
        sanity_bound: int,
    ) -> tuple[bool, bool, str | None, str | None]:
        if len(plan.actions) > sanity_bound:
            return False, False, None, "plan-exceeds-sanity-bound"

        try:
            for action in plan.actions:
                serialize_transfer_action(action)
        except ValueError:
            return False, False, None, "plan-contains-non-transfer"

        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            validation_result = validator.validate(problem, plan)
        valid = validation_result.status.name == "VALID"
        if not valid:
            return (
                False,
                True,
                validation_result.status.name,
                f"validation-{validation_result.status.name.lower()}",
            )
        return True, True, validation_result.status.name, None
