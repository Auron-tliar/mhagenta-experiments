"""Pure symbolic planning and transfer-goal support for experiment 2-4-BW."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from time import monotonic
from typing import Any
import warnings

from mhagenta import Belief, Goal
from unified_planning.io import PDDLReader
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import OneshotPlanner, PlanValidator, get_environment


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API\..*", category=UserWarning)
    get_environment().credits_stream = None


PREDICATE_PATTERN = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\((?P<args>.*)\)$")
LOCATION_PATTERN = re.compile(r"^t(?P<index>\d+)$", re.IGNORECASE)

RUNTIME_TO_CANONICAL = {
    "HandEmpty": "hand-empty", "Holding": "holding", "On": "on", "Clear": "clear",
    "Above": "above", "AtLoc": "at-location", "LeftOf": "left-of",
}
CANONICAL_ARITIES = {
    "hand-empty": 0, "holding": 1, "on": 2, "clear": 1,
    "above": 1, "at-location": 2, "left-of": 2,
}
ABSTRACT_PREDICATES = {"on", "clear", "at-location"}
SOLVED_STATUSES = {"SOLVED_SATISFICING", "SOLVED_OPTIMALLY"}


@dataclass(frozen=True)
class GoalSpec:
    """A single high-level Blocks World stacking intention."""

    top: str
    bottom: str

    def as_dict(self) -> dict[str, str]:
        return {"top": self.top, "bottom": self.bottom}

    @property
    def fact(self) -> str:
        return format_fact("on", (self.top, self.bottom))
@dataclass(frozen=True)
class TransferSpec:
    """A fully grounded abstract transfer action."""

    block: str
    source_support: str
    destination_support: str
    source: str
    destination: str

    def as_dict(self) -> dict[str, str]:
        return {
            "block": self.block, "source_support": self.source_support,
            "destination_support": self.destination_support,
            "source": self.source, "destination": self.destination,
        }

    @property
    def target_facts(self) -> set[str]:
        return {
            format_fact("on", (self.block, self.destination_support)),
            format_fact("at-location", (self.block, self.destination))}

    @property
    def target_beliefs(self) -> list[Belief]:
        return [
            Belief(predicate="On", arguments=(self.block, self.destination_support)),
            Belief(predicate="AtLoc", arguments=(self.block, self.destination))]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TransferSpec":
        try:
            values = {
                key: str(data[key]).lower()
                for key in ("block", "source_support", "destination_support", "source", "destination")
            }
        except KeyError as exc:
            raise ValueError(f"Missing transfer field {exc.args[0]!r}.") from exc
        spec = cls(**values)
        if spec.block == spec.destination_support:
            raise ValueError("A transfer block cannot support itself.")
        if spec.source == spec.destination:
            raise ValueError("A transfer must change locations.")
        location_index(spec.source)
        location_index(spec.destination)
        return spec
@dataclass(frozen=True)
class PlanningOutcome:
    """Compact result of trying the configured abstract planners."""

    accepted: bool
    engine: str | None
    status: str
    elapsed_seconds: float
    actions: list[dict[str, str]]
    validation_status: str | None
    sanity_bound: int
    failure: str | None
def block_names(num_blocks: int) -> list[str]:
    width = len(str(num_blocks - 1))
    return [f"b{i:0{width}d}" for i in range(num_blocks)]
def location_names(table_len: int) -> list[str]:
    width = len(str(table_len - 1))
    return [f"t{i:0{width}d}" for i in range(table_len)]
def location_index(location: str) -> int:
    match = LOCATION_PATTERN.fullmatch(location)
    if match is None:
        raise ValueError(f"Invalid Blocks World location {location!r}.")
    return int(match.group("index"))
def format_fact(predicate: str, arguments: Iterable[str] = ()) -> str:
    return f"{predicate.lower()}({','.join(str(arg).lower() for arg in arguments)})"


def split_fact(fact: str) -> tuple[str, tuple[str, ...]]:
    match = PREDICATE_PATTERN.fullmatch(fact)
    if match is None:
        raise ValueError(f"Malformed canonical fact {fact!r}.")
    raw_args = match.group("args").strip()
    arguments = tuple(arg.strip().lower() for arg in raw_args.split(",") if arg.strip())
    return match.group("name").lower(), arguments


def parse_symbolic_observation(content: Sequence[str]) -> list[Belief]:
    """Convert a complete symbolic environment observation to typed beliefs."""
    beliefs: list[Belief] = []
    for raw_fact in content:
        if not isinstance(raw_fact, str):
            raise ValueError(f"Symbolic fact must be a string, got {type(raw_fact).__name__}.")
        stripped = raw_fact.strip()
        if stripped in RUNTIME_TO_CANONICAL and CANONICAL_ARITIES[RUNTIME_TO_CANONICAL[stripped]] == 0:
            beliefs.append(Belief(predicate=stripped, arguments=()))
            continue
        match = PREDICATE_PATTERN.fullmatch(stripped)
        if match is None:
            raise ValueError(f"Malformed symbolic fact: {raw_fact!r}.")
        runtime_name = match.group("name")
        try:
            canonical_name = RUNTIME_TO_CANONICAL[runtime_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Blocks World predicate: {runtime_name!r}.") from exc
        raw_args = match.group("args").strip()
        arguments = tuple(arg.strip().lower() for arg in raw_args.split(",") if arg.strip()) if raw_args else ()
        expected_arity = CANONICAL_ARITIES[canonical_name]
        if len(arguments) != expected_arity:
            raise ValueError(f"Predicate {runtime_name!r} expects {expected_arity} arguments, got {len(arguments)}.")
        beliefs.append(Belief(predicate=runtime_name, arguments=arguments))
    return beliefs


def belief_to_fact(belief: Belief) -> str:
    try:
        predicate = RUNTIME_TO_CANONICAL[belief.predicate]
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
    if len(normalized) != CANONICAL_ARITIES[predicate]:
        raise ValueError(f"Invalid arity for belief {belief!r}.")
    return format_fact(predicate, (str(arg) for arg in normalized))


def beliefs_to_facts(beliefs: Sequence[Belief]) -> set[str]:
    return {belief_to_fact(belief) for belief in beliefs}
def project_abstract_facts(facts: Iterable[str]) -> set[str]:
    projected: set[str] = set()
    for fact in facts:
        predicate, _ = split_fact(fact)
        if predicate in ABSTRACT_PREDICATES:
            projected.add(fact.lower())
    return projected


def missing_transfer_preconditions(
    facts: set[str],
    transfer: TransferSpec,
) -> tuple[str, ...]:
    """Return missing abstract preconditions for one grounded transfer."""
    required = {
        format_fact("on", (transfer.block, transfer.source_support)),
        format_fact("clear", (transfer.block,)),
        format_fact("at-location", (transfer.block, transfer.source)),
        format_fact("clear", (transfer.destination_support,)),
        format_fact("at-location", (transfer.destination_support, transfer.destination)),
    }
    if transfer.source == transfer.destination:
        required.add("different(source,destination)")
    return tuple(sorted(required - facts))


def observed_transfer_targets(
    facts: set[str],
    transfer: TransferSpec,
) -> list[str]:
    """Return the transfer target facts present in a fresh observation."""
    return sorted(transfer.target_facts & facts)
def beliefs_to_dicts(beliefs: Sequence[Belief]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for belief in beliefs:
        arguments = belief.arguments
        if arguments is None:
            args: list[Any] = []
        elif isinstance(arguments, (tuple, list)):
            args = list(arguments)
        else:
            args = [arguments]
        result.append({"predicate": belief.predicate, "arguments": args})
    return result


def generate_options(
    blocks: Sequence[str],
    facts: set[str],
    completed: Sequence[dict[str, str]] = (),
) -> list[GoalSpec]:
    completed_pairs = {(entry["top"].lower(), entry["bottom"].lower()) for entry in completed}
    options = [
        GoalSpec(top=top, bottom=bottom)
        for top in blocks
        for bottom in blocks
        if top != bottom
        and format_fact("on", (top, bottom)) not in facts
        and (top, bottom) not in completed_pairs
    ]
    return sorted(options, key=lambda goal: (goal.top, goal.bottom))


def build_problem_pddl(
    *,
    problem_name: str,
    blocks: Sequence[str],
    locations: Sequence[str],
    facts: set[str],
    goal: GoalSpec,
) -> str:
    abstract_facts = project_abstract_facts(facts)
    literals: list[str] = []
    for fact in sorted(abstract_facts):
        predicate, arguments = split_fact(fact)
        suffix = f" {' '.join(arguments)}" if arguments else ""
        literals.append(f"({predicate}{suffix})")
    init_facts = "\n    ".join(literals)
    return f"""(define (problem {problem_name})
  (:domain abstract-blocksworld)
  (:objects
    {' '.join(blocks)} - block
    {' '.join(locations)} - location
  )
  (:init
    (= (abstract-steps) 0)
    {init_facts}
  )
  (:goal (on {goal.top} {goal.bottom}))
  (:metric minimize (abstract-steps))
)"""


def serialize_transfer_action(action_instance: Any) -> dict[str, Any]:
    name = action_instance.action.name.lower()
    arguments = [str(parameter).lower() for parameter in action_instance.actual_parameters]
    if name != "transfer" or len(arguments) != 5:
        raise ValueError(f"Expected one grounded transfer action, got {name}{tuple(arguments)!r}.")
    return {"name": name, **TransferSpec(*arguments).as_dict()}


def transfer_goal(
    spec: TransferSpec,
    *,
    status: str,
    goal_id: str,
    plan_id: str,
    step_index: int,
    based_on_observation_seq: int,
    hierarchy_id: str | None = None,
    completion_observation_seq: int | None = None,
    observed_facts: Iterable[str] = (),
    observed_target_facts: Iterable[str] = (),
    atomic_rows: Sequence[Mapping[str, Any]] = (),
    failure_reason: str | None = None,
) -> Goal:
    extras: dict[str, Any] = {
        "kind": "transfer",
        "status": status,
        "goal_id": goal_id,
        "plan_id": plan_id,
        "step_index": int(step_index),
        "based_on_observation_seq": int(based_on_observation_seq),
        **spec.as_dict(),
    }
    if hierarchy_id is not None:
        extras["hierarchy_id"] = str(hierarchy_id)
    if completion_observation_seq is not None:
        extras["completion_observation_seq"] = int(completion_observation_seq)
    if observed_facts:
        extras["observed_facts"] = sorted(str(fact).lower() for fact in observed_facts)
    if observed_target_facts:
        extras["observed_target_facts"] = sorted(str(fact).lower() for fact in observed_target_facts)
    if atomic_rows:
        extras["atomic_rows"] = [dict(row) for row in atomic_rows]
    if failure_reason is not None:
        extras["failure_reason"] = failure_reason
    return Goal(state=spec.target_beliefs, extras=extras)


def transfer_from_goal(goal: Goal) -> TransferSpec:
    extras = goal.extras
    if not isinstance(extras, dict) or extras.get("kind") != "transfer":
        raise ValueError("Expected a transfer goal.")
    spec = TransferSpec.from_mapping(extras)
    target_facts = {belief_to_fact(belief) for belief in goal.state}
    if target_facts != spec.target_facts:
        raise ValueError("Transfer goal state does not match its grounded target.")
    return spec


def goal_to_dict(goal: Goal) -> dict[str, Any]:
    return {"state": beliefs_to_dicts(goal.state), "extras": dict(goal.extras)}


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
            warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API\..*", category=UserWarning)
            self._reader = PDDLReader()

    def build_problem(self, facts: set[str], goal: GoalSpec, problem_name: str) -> Any:
        return self._reader.parse_problem_string(self.domain_text, build_problem_pddl(
            problem_name=problem_name,
            blocks=self.blocks,
            locations=self.locations,
            facts=facts,
            goal=goal,
        ))

    def solve(self, facts: set[str], goal: GoalSpec, problem_name: str) -> PlanningOutcome:
        problem = self.build_problem(facts, goal, problem_name)
        sanity_bound = 4 * len(self.blocks)
        started, status, failures = monotonic(), "NOT_RUN", []
        for engine_name in self.engines:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API\..*", category=UserWarning)
                    warnings.filterwarnings("ignore", message=r"We cannot establish whether .* can solve this problem!", category=UserWarning)
                    with self._planner_factory(name=engine_name) as planner:
                        result = planner.solve(problem, timeout=self.timeout)
                status = result.status.name
                plan = result.plan
                if status not in SOLVED_STATUSES or plan is None:
                    rejection = "planner-did-not-solve"
                elif not isinstance(plan, SequentialPlan):
                    rejection = "non-sequential-plan"
                else:
                    actions, validation_status, rejection = self._assess_plan(problem, plan, sanity_bound)
                    if rejection is None:
                        return PlanningOutcome(
                            accepted=True,
                            engine=engine_name,
                            status=status,
                            elapsed_seconds=monotonic() - started,
                            actions=actions,
                            validation_status=validation_status,
                            sanity_bound=sanity_bound,
                            failure=None,
                        )
            except Exception as exc:
                status, rejection = "EXCEPTION", f"planner-exception:{type(exc).__name__}:{exc}"
            failures.append(f"{engine_name}:{rejection}")
        return PlanningOutcome(
            accepted=False,
            engine=None,
            status=status,
            elapsed_seconds=monotonic() - started,
            actions=[],
            validation_status=None,
            sanity_bound=sanity_bound,
            failure=";".join(failures) or "no-planner",
        )

    @staticmethod
    def _assess_plan(problem: Any, plan: SequentialPlan, sanity_bound: int) -> tuple[list[dict[str, str]], str | None, str | None]:
        if len(plan.actions) > sanity_bound:
            return [], None, "plan-exceeds-sanity-bound"
        try:
            actions = [serialize_transfer_action(action) for action in plan.actions]
        except ValueError:
            return [], None, "plan-contains-non-transfer"
        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            status = validator.validate(problem, plan).status.name
        rejection = None if status == "VALID" else f"validation-{status.lower()}"
        return actions, status, rejection
