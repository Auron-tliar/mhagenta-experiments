"""Experiment-local Blocks World planning and transfer contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import re
from typing import Any

from mhagenta import Belief, Goal


PREDICATE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\((?P<args>.*)\)$"
)
LOCATION_PATTERN = re.compile(r"^t(?P<index>\d+)$", re.IGNORECASE)
RUNTIME_TO_CANONICAL = {
    "HandEmpty": "hand-empty",
    "Holding": "holding",
    "On": "on",
    "Clear": "clear",
    "Above": "above",
    "AtLoc": "at-location",
    "LeftOf": "left-of",
}
CANONICAL_ARITIES = {
    "hand-empty": 0,
    "holding": 1,
    "on": 2,
    "clear": 1,
    "above": 1,
    "at-location": 2,
    "left-of": 2,
}
ABSTRACT_PREDICATES = {"on", "clear", "at-location"}


@dataclass(frozen=True)
class GoalSpec:
    """One high-level Blocks World stacking intention."""

    top: str
    bottom: str

    def as_dict(self) -> dict[str, str]:
        """Return the JSON-safe goal representation."""

        return {"top": self.top, "bottom": self.bottom}

    @property
    def fact(self) -> str:
        """Return the canonical fact that satisfies this goal."""

        return format_fact("on", (self.top, self.bottom))


@dataclass(frozen=True)
class TransferSpec:
    """One fully grounded abstract transfer action."""

    block: str
    source_support: str
    destination_support: str
    source: str
    destination: str

    def as_dict(self) -> dict[str, str]:
        """Return the JSON-safe transfer representation."""

        return {
            "block": self.block,
            "source_support": self.source_support,
            "destination_support": self.destination_support,
            "source": self.source,
            "destination": self.destination,
        }

    @property
    def target_facts(self) -> set[str]:
        """Return the two observable facts produced by the transfer."""

        return {
            format_fact("on", (self.block, self.destination_support)),
            format_fact("at-location", (self.block, self.destination)),
        }

    @property
    def target_beliefs(self) -> list[Belief]:
        """Return the typed MHAgentA target beliefs."""

        return [
            Belief(predicate="On", arguments=(self.block, self.destination_support)),
            Belief(predicate="AtLoc", arguments=(self.block, self.destination)),
        ]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TransferSpec":
        """Validate and construct a transfer from a mapping."""

        try:
            values = {
                key: str(data[key]).lower()
                for key in (
                    "block",
                    "source_support",
                    "destination_support",
                    "source",
                    "destination",
                )
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


def block_names(num_blocks: int) -> list[str]:
    """Return canonical block names for a fixed inventory."""

    width = len(str(num_blocks - 1))
    return [f"b{i:0{width}d}" for i in range(num_blocks)]


def location_names(table_len: int) -> list[str]:
    """Return canonical table-location names."""

    width = len(str(table_len - 1))
    return [f"t{i:0{width}d}" for i in range(table_len)]


def location_index(location: str) -> int:
    """Return the numeric index encoded by a location name."""

    match = LOCATION_PATTERN.fullmatch(location)
    if match is None:
        raise ValueError(f"Invalid Blocks World location {location!r}.")
    return int(match.group("index"))


def format_fact(predicate: str, arguments: Iterable[str] = ()) -> str:
    """Format one lowercase canonical fact."""

    return f"{predicate.lower()}({','.join(str(arg).lower() for arg in arguments)})"


def split_fact(fact: str) -> tuple[str, tuple[str, ...]]:
    """Split a canonical fact into its predicate and arguments."""

    match = PREDICATE_PATTERN.fullmatch(fact)
    if match is None:
        raise ValueError(f"Malformed canonical fact {fact!r}.")
    raw_args = match.group("args").strip()
    arguments = tuple(
        arg.strip().lower() for arg in raw_args.split(",") if arg.strip()
    )
    return match.group("name").lower(), arguments


def belief_to_fact(belief: Belief) -> str:
    """Convert one typed runtime belief to a canonical fact."""

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
    """Convert typed beliefs to canonical facts."""

    return {belief_to_fact(belief) for belief in beliefs}


def project_abstract_facts(facts: Iterable[str]) -> set[str]:
    """Return the facts used by abstract transfer planning."""

    projected: set[str] = set()
    for fact in facts:
        predicate, _ = split_fact(fact)
        if predicate in ABSTRACT_PREDICATES:
            projected.add(fact.lower())
    return projected


def beliefs_to_dicts(beliefs: Sequence[Belief]) -> list[dict[str, Any]]:
    """Convert beliefs to JSON-safe mappings."""

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
    """Return unachieved stacking goals not already completed this run."""

    completed_pairs = {
        (entry["top"].lower(), entry["bottom"].lower()) for entry in completed
    }
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
    """Build the abstract transfer planning problem."""

    literals: list[str] = []
    for fact in sorted(project_abstract_facts(facts)):
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
    """Serialize one grounded Unified Planning transfer action."""

    name = action_instance.action.name.lower()
    arguments = [
        str(parameter).lower() for parameter in action_instance.actual_parameters
    ]
    if name != "transfer" or len(arguments) != 5:
        raise ValueError(
            f"Expected one grounded transfer action, got {name}{tuple(arguments)!r}."
        )
    return {"name": name, **TransferSpec(*arguments).as_dict()}


def transfer_goal(
    spec: TransferSpec,
    *,
    status: str,
    goal_id: str,
    plan_id: str,
    step_index: int,
    based_on_observation_seq: int,
    completion_observation_seq: int | None = None,
    observed_facts: Iterable[str] = (),
    failure_reason: str | None = None,
    atomic_rows: Sequence[Mapping[str, Any]] = (),
) -> Goal:
    """Build one typed transfer goal or terminal transfer result."""

    extras: dict[str, Any] = {
        "kind": "transfer",
        "status": status,
        "goal_id": goal_id,
        "plan_id": plan_id,
        "step_index": int(step_index),
        "based_on_observation_seq": int(based_on_observation_seq),
        **spec.as_dict(),
    }
    if completion_observation_seq is not None:
        extras["completion_observation_seq"] = int(completion_observation_seq)
    if observed_facts:
        extras["observed_facts"] = sorted(
            str(fact).lower() for fact in observed_facts
        )
    if failure_reason is not None:
        extras["failure_reason"] = failure_reason
    if atomic_rows:
        extras["atomic"] = [dict(row) for row in atomic_rows]
    return Goal(state=spec.target_beliefs, extras=extras)


def transfer_from_goal(goal: Goal) -> TransferSpec:
    """Validate and recover a transfer specification from a typed goal."""

    extras = goal.extras
    if not isinstance(extras, dict) or extras.get("kind") != "transfer":
        raise ValueError("Expected a transfer goal.")
    spec = TransferSpec.from_mapping(extras)
    target_facts = {belief_to_fact(belief) for belief in goal.state}
    if target_facts != spec.target_facts:
        raise ValueError("Transfer goal state does not match its grounded target.")
    return spec


def goal_to_dict(goal: Goal) -> dict[str, Any]:
    """Convert a typed goal to JSON-safe data."""

    return {"state": beliefs_to_dicts(goal.state), "extras": dict(goal.extras)}


def observable_facts_from_state(problem: Any, state: Any) -> set[str]:
    """Return observable Boolean facts from a Unified Planning state."""

    facts: set[str] = set()
    for fluent in problem.fluents:
        if not fluent.type.is_bool_type():
            continue
        domains = [list(problem.objects(parameter.type)) for parameter in fluent.signature]
        combinations = product(*domains) if domains else [()]
        for arguments in combinations:
            expression = fluent(*arguments)
            if state.get_value(expression).is_true():
                facts.add(
                    format_fact(
                        fluent.name,
                        (str(argument) for argument in arguments),
                    )
                )
    return facts
