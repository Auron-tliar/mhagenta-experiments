"""Focused symbolic planning for Experiment 2-3-BW."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from time import monotonic
from typing import Any
import warnings

from mhagenta import Belief
from unified_planning.io import PDDLReader
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import OneshotPlanner, PlanValidator


PREDICATE_PATTERN = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\((?P<args>.*)\)$")

RUNTIME_TO_PDDL = {
    "HandEmpty": "hand-empty",
    "Holding": "holding",
    "On": "on",
    "Clear": "clear",
    "Above": "above",
    "AtLoc": "at-location",
    "LeftOf": "left-of",
}
PDDL_ARITIES = {
    "hand-empty": 0,
    "holding": 1,
    "on": 2,
    "clear": 1,
    "above": 1,
    "at-location": 2,
    "left-of": 2,
}
ACTION_TO_ENV = {
    "pickup": 0,
    "putdown": 1,
    "moveleft": 2,
    "moveright": 3,
}
SOLVED_STATUSES = {"SOLVED_SATISFICING", "SOLVED_OPTIMALLY"}
LPG_PARAMETERS = {
    "-seed": 0,
    "-search_steps": 100,
    "-restarts": 100,
    "-repeats": 3,
    "-noise": 0.3,
    "-static_noise": "",
}


@dataclass(frozen=True)
class GoalSpec:
    """A single Blocks World stacking intention."""

    top: str
    bottom: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-native representation of the goal."""

        return {"top": self.top, "bottom": self.bottom}

    @property
    def fact(self) -> str:
        """Return the canonical PDDL fact achieved by the goal."""

        return format_fact("on", (self.top, self.bottom))


@dataclass(frozen=True)
class PlanResult:
    """Transient result of the experiment's one LPG planning attempt."""

    planner: str
    status: str
    elapsed_seconds: float
    sequential: bool
    validation_status: str | None
    sanity_bound: int
    actions: tuple[dict[str, Any], ...]
    failure: str | None

    @property
    def accepted(self) -> bool:
        """Whether LPG returned the accepted validated sequential plan."""

        return self.failure is None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-native planner record for persistent module state."""

        return {
            "planner": self.planner,
            "status": self.status,
            "elapsed_seconds": float(self.elapsed_seconds),
            "sequential": self.sequential,
            "validation_status": self.validation_status,
            "sanity_bound": self.sanity_bound,
            "length": len(self.actions),
            "actions": [
                {
                    "name": action["name"],
                    "arguments": list(action["arguments"]),
                    "env_action": action["env_action"],
                }
                for action in self.actions
            ],
            "failure": self.failure,
        }


def block_names(num_blocks: int) -> list[str]:
    """Return canonical lower-case block object names."""

    width = len(str(num_blocks - 1))
    return [f"b{i:0{width}d}" for i in range(num_blocks)]


def location_names(table_len: int) -> list[str]:
    """Return canonical lower-case table-location object names."""

    width = len(str(table_len - 1))
    return [f"t{i:0{width}d}" for i in range(table_len)]


def format_fact(predicate: str, arguments: Iterable[str] = ()) -> str:
    """Format one canonical lower-case symbolic fact."""

    return f"{predicate.lower()}({','.join(str(arg).lower() for arg in arguments)})"


def parse_symbolic_observation(content: Sequence[str]) -> list[Belief]:
    """Convert a complete environment observation to typed MHAgentA beliefs."""

    beliefs: list[Belief] = []
    for raw_fact in content:
        if not isinstance(raw_fact, str):
            raise ValueError(f"Symbolic fact must be a string, got {type(raw_fact).__name__}.")
        stripped = raw_fact.strip()
        if stripped in RUNTIME_TO_PDDL and PDDL_ARITIES[RUNTIME_TO_PDDL[stripped]] == 0:
            beliefs.append(Belief(predicate=stripped, arguments=()))
            continue
        match = PREDICATE_PATTERN.fullmatch(stripped)
        if match is None:
            raise ValueError(f"Malformed symbolic fact: {raw_fact!r}.")
        runtime_name = match.group("name")
        try:
            pddl_name = RUNTIME_TO_PDDL[runtime_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Blocks World predicate: {runtime_name!r}.") from exc
        raw_args = match.group("args").strip()
        arguments = tuple(arg.strip() for arg in raw_args.split(",") if arg.strip()) if raw_args else ()
        expected_arity = PDDL_ARITIES[pddl_name]
        if len(arguments) != expected_arity:
            raise ValueError(
                f"Predicate {runtime_name!r} expects {expected_arity} arguments, got {len(arguments)}."
            )
        beliefs.append(Belief(predicate=runtime_name, arguments=arguments))
    return beliefs


def belief_to_fact(belief: Belief) -> str:
    """Convert one typed belief to its canonical PDDL fact."""

    try:
        predicate = RUNTIME_TO_PDDL[belief.predicate]
    except KeyError as exc:
        raise ValueError(f"Unknown belief predicate: {belief.predicate!r}.") from exc
    arguments = belief.arguments
    if arguments is None:
        normalized: tuple[Any, ...] = ()
    elif isinstance(arguments, tuple):
        normalized = arguments
    elif isinstance(arguments, list):
        normalized = tuple(arguments)
    else:
        normalized = (arguments,)
    if len(normalized) != PDDL_ARITIES[predicate]:
        raise ValueError(f"Invalid arity for belief {belief!r}.")
    return format_fact(predicate, (str(arg) for arg in normalized))


def beliefs_to_facts(beliefs: Sequence[Belief]) -> set[str]:
    """Convert a complete typed belief collection to canonical facts."""

    return {belief_to_fact(belief) for belief in beliefs}


def generate_options(blocks: Sequence[str], facts: set[str]) -> list[GoalSpec]:
    """Generate the unsatisfied distinct-block stacking options."""

    options = [
        GoalSpec(top=top, bottom=bottom)
        for top in blocks
        for bottom in blocks
        if top != bottom and format_fact("on", (top, bottom)) not in facts
    ]
    return sorted(options, key=lambda goal: (goal.top, goal.bottom))


def blocks_above(target: str, facts: set[str]) -> set[str]:
    """Return all blocks transitively above a target block."""

    directly_above: dict[str, str] = {}
    for fact in facts:
        match = PREDICATE_PATTERN.fullmatch(fact)
        if match is None or match.group("name").lower() != "on":
            continue
        args = tuple(arg.strip().lower() for arg in match.group("args").split(","))
        if len(args) == 2:
            directly_above[args[1]] = args[0]

    result: set[str] = set()
    cursor = target.lower()
    while cursor in directly_above:
        cursor = directly_above[cursor]
        if cursor in result:
            break
        result.add(cursor)
    return result


def plan_length_bound(goal: GoalSpec, facts: set[str], table_len: int) -> int:
    """Return the thesis-inspired loose safeguard for accepted plan length."""

    blockers = blocks_above(goal.top, facts) | blocks_above(goal.bottom, facts)
    work_items = len(blockers) + 1
    base = 2 * work_items + 2 * (table_len - 1) * work_items
    return 4 * base


def build_problem_pddl(
    *,
    problem_name: str,
    blocks: Sequence[str],
    locations: Sequence[str],
    facts: set[str],
    goal: GoalSpec,
) -> str:
    """Build one complete closed-world Blocks World PDDL problem."""

    literals: list[str] = []
    for fact in sorted(facts):
        match = PREDICATE_PATTERN.fullmatch(fact)
        if match is None:
            raise ValueError(f"Invalid canonical fact: {fact!r}.")
        arguments = [arg.strip() for arg in match.group("args").split(",") if arg.strip()]
        suffix = f" {' '.join(arguments)}" if arguments else ""
        literals.append(f"({match.group('name')}{suffix})")
    init_facts = "\n    ".join(literals)
    return f"""(define (problem {problem_name})
  (:domain blocksworld)
  (:objects
    {' '.join(blocks)} - block
    {' '.join(locations)} - location
  )
  (:init
    (= (elapsed-steps) 0)
    {init_facts}
  )
  (:goal (on {goal.top} {goal.bottom}))
  (:metric minimize (elapsed-steps))
)"""


def serialize_action(action_instance: Any) -> dict[str, Any]:
    """Serialize one grounded UP action to the experiment action vocabulary."""

    name = action_instance.action.name.lower()
    arguments = [str(parameter).lower() for parameter in action_instance.actual_parameters]
    try:
        env_action = ACTION_TO_ENV[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Blocks World plan action: {name!r}.") from exc
    return {"name": name, "arguments": arguments, "env_action": env_action}


def action_soundness(action: dict[str, Any], facts: set[str]) -> tuple[bool, list[str]]:
    """Check one serialized action's PDDL preconditions against observed facts."""

    name = action.get("name")
    arguments = action.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        return False, ["malformed-action"]

    required: list[str]
    if name == "pickup" and len(arguments) == 3:
        block, support, location = arguments
        required = [
            format_fact("hand-empty"),
            format_fact("on", (block, support)),
            format_fact("clear", (block,)),
            format_fact("at-location", (block, location)),
            format_fact("at-location", (support, location)),
            format_fact("above", (location,)),
        ]
    elif name == "putdown" and len(arguments) == 3:
        block, support, location = arguments
        required = [
            format_fact("holding", (block,)),
            format_fact("clear", (support,)),
            format_fact("at-location", (support, location)),
            format_fact("above", (location,)),
        ]
    elif name == "moveleft" and len(arguments) == 2:
        source, destination = arguments
        required = [
            format_fact("above", (source,)),
            format_fact("left-of", (destination, source)),
        ]
    elif name == "moveright" and len(arguments) == 2:
        source, destination = arguments
        required = [
            format_fact("above", (source,)),
            format_fact("left-of", (source, destination)),
        ]
    else:
        return False, ["unknown-or-wrong-arity-action"]

    missing = [fact for fact in required if fact not in facts]
    return not missing, missing


def _parse_problem(
    facts: set[str],
    goal: GoalSpec,
    *,
    blocks: Sequence[str],
    locations: Sequence[str],
    problem_name: str,
) -> Any:
    domain_text = Path(__file__).resolve().with_name("blocksworld-domain.pddl").read_text(
        encoding="utf-8"
    )
    problem_pddl = build_problem_pddl(
        problem_name=problem_name,
        blocks=blocks,
        locations=locations,
        facts=facts,
        goal=goal,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        return PDDLReader().parse_problem_string(domain_text, problem_pddl)


def _run_lpg(problem: Any, timeout: float) -> Any:
    """Call LPG once; kept private as the focused test seam."""

    with OneshotPlanner(name="lpg", params=LPG_PARAMETERS) as planner:
        return planner.solve(problem, timeout=timeout)


def _failure_result(
    *,
    status: str,
    elapsed_seconds: float,
    sanity_bound: int,
    failure: str,
    sequential: bool = False,
    validation_status: str | None = None,
    actions: Sequence[dict[str, Any]] = (),
) -> PlanResult:
    return PlanResult(
        planner="lpg",
        status=status,
        elapsed_seconds=elapsed_seconds,
        sequential=sequential,
        validation_status=validation_status,
        sanity_bound=sanity_bound,
        actions=tuple(dict(action) for action in actions),
        failure=failure,
    )


def plan_blocks_world(
    facts: set[str],
    goal: GoalSpec,
    *,
    blocks: Sequence[str],
    locations: Sequence[str],
    timeout: float,
    problem_name: str,
) -> PlanResult:
    """Build, solve with LPG, bound, and independently validate one problem."""

    sanity_bound = plan_length_bound(goal, facts, len(locations))
    try:
        problem = _parse_problem(
            facts,
            goal,
            blocks=blocks,
            locations=locations,
            problem_name=problem_name,
        )
    except Exception as exc:
        return _failure_result(
            status="PROBLEM_ERROR",
            elapsed_seconds=0.0,
            sanity_bound=sanity_bound,
            failure=f"problem-generation-error: {type(exc).__name__}: {exc}",
        )

    started = monotonic()
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
            result = _run_lpg(problem, timeout)
    except Exception as exc:
        return _failure_result(
            status="EXCEPTION",
            elapsed_seconds=monotonic() - started,
            sanity_bound=sanity_bound,
            failure=f"planner-exception: {type(exc).__name__}: {exc}",
        )
    elapsed_seconds = monotonic() - started
    status = result.status.name
    plan = result.plan
    if status not in SOLVED_STATUSES or plan is None:
        return _failure_result(
            status=status,
            elapsed_seconds=elapsed_seconds,
            sanity_bound=sanity_bound,
            failure="planner-did-not-solve",
        )
    if not isinstance(plan, SequentialPlan):
        return _failure_result(
            status=status,
            elapsed_seconds=elapsed_seconds,
            sanity_bound=sanity_bound,
            failure="non-sequential-plan",
        )

    try:
        actions = tuple(serialize_action(action) for action in plan.actions)
    except Exception as exc:
        return _failure_result(
            status=status,
            elapsed_seconds=elapsed_seconds,
            sanity_bound=sanity_bound,
            sequential=True,
            failure=f"plan-serialization-error: {type(exc).__name__}: {exc}",
        )
    if len(actions) > sanity_bound:
        return _failure_result(
            status=status,
            elapsed_seconds=elapsed_seconds,
            sanity_bound=sanity_bound,
            sequential=True,
            actions=actions,
            failure="plan-too-long",
        )

    try:
        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            validation = validator.validate(problem, plan)
        validation_status = validation.status.name
    except Exception as exc:
        return _failure_result(
            status=status,
            elapsed_seconds=elapsed_seconds,
            sanity_bound=sanity_bound,
            sequential=True,
            actions=actions,
            failure=f"validation-exception: {type(exc).__name__}: {exc}",
        )
    if validation_status != "VALID":
        return _failure_result(
            status=status,
            elapsed_seconds=elapsed_seconds,
            sanity_bound=sanity_bound,
            sequential=True,
            validation_status=validation_status,
            actions=actions,
            failure="plan-invalid",
        )

    return PlanResult(
        planner="lpg",
        status=status,
        elapsed_seconds=elapsed_seconds,
        sequential=True,
        validation_status=validation_status,
        sanity_bound=sanity_bound,
        actions=actions,
        failure=None,
    )
