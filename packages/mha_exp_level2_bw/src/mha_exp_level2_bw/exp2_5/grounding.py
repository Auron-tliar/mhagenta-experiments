"""Numeric observation grounding for experiment 2-5-BW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from mhagenta import Belief

from .contracts import TransferSpec, beliefs_to_facts, block_names, format_fact, split_fact


TABLE_LEN = 5
NUM_BLOCKS = 8
OBSERVATION_SHAPE = (NUM_BLOCKS + 2, TABLE_LEN, NUM_BLOCKS)


@dataclass(frozen=True)
class GroundedObservation:
    """Validated numeric observation and its symbolic interpretation."""

    observation: np.ndarray
    beliefs: tuple[Belief, ...]
    facts: frozenset[str]
    arm_location: str
    held_block: str | None


def as_numeric_observation(
    content: Any, *, table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> np.ndarray:
    """Return a validated copy of a Blocks World numeric observation."""

    observation = np.asarray(content)
    shape = (num_blocks + 2, table_len, num_blocks)
    if table_len < 2 or num_blocks < 1:
        raise ValueError("Transfer requires at least two columns and one block.")
    if observation.shape != shape:
        raise ValueError(
            f"Expected observation shape {shape}, received {observation.shape}."
        )
    if not np.all((observation == 0) | (observation == 1)):
        raise ValueError("Blocks World observations must contain only binary values.")
    return observation.astype(np.uint8, copy=True)


def _arm_column(observation: np.ndarray) -> int:
    arm_plane = observation[0]
    encoded_columns = [
        column
        for column in range(observation.shape[1])
        if np.all(arm_plane[column] == 1)
    ]
    if len(encoded_columns) != 1:
        raise ValueError("The arm plane must identify exactly one table column.")
    arm_column = encoded_columns[0]
    if np.any(np.delete(arm_plane, arm_column, axis=0)):
        raise ValueError("The arm plane contains values outside its encoded column.")
    return arm_column


def _held_block(observation: np.ndarray, arm_column: int) -> int | None:
    holding_cells = np.argwhere(observation[1] == 1)
    if len(holding_cells) > 1:
        raise ValueError("The holding plane contains more than one block.")
    if not len(holding_cells):
        return None
    column, block = (int(value) for value in holding_cells[0])
    if column != arm_column:
        raise ValueError("The held block is not encoded at the arm column.")
    return block


def _stacks(observation: np.ndarray) -> tuple[tuple[int, ...], ...]:
    stacks: list[tuple[int, ...]] = []
    seen_blocks: set[int] = set()
    for column in range(observation.shape[1]):
        bottom_to_top: list[int] = []
        found_empty = False
        for row in range(observation.shape[2] - 1, -1, -1):
            blocks = np.flatnonzero(observation[row + 2, column])
            if len(blocks) > 1:
                raise ValueError(
                    f"Stack cell row={row}, column={column} contains multiple blocks."
                )
            if not len(blocks):
                found_empty = True
                continue
            if found_empty:
                raise ValueError(f"Stack column {column} contains a gap.")
            block = int(blocks[0])
            if block in seen_blocks:
                raise ValueError(f"Block {block} occurs more than once in the stacks.")
            seen_blocks.add(block)
            bottom_to_top.append(block)
        stacks.append(tuple(bottom_to_top))
    return tuple(stacks)


def ground_observation(
    content: Any, *, table_len: int = TABLE_LEN, num_blocks: int = NUM_BLOCKS,
) -> GroundedObservation:
    """Convert a complete numeric observation into typed closed-world beliefs."""

    observation = as_numeric_observation(content, table_len=table_len, num_blocks=num_blocks)
    names = block_names(num_blocks)
    arm_column = _arm_column(observation)
    held_block = _held_block(observation, arm_column)
    stacks = _stacks(observation)

    observed_blocks = {block for stack in stacks for block in stack}
    if held_block is not None:
        if held_block in observed_blocks:
            raise ValueError(f"Held block {held_block} also occurs in a stack.")
        observed_blocks.add(held_block)
    expected_blocks = set(range(num_blocks))
    if observed_blocks != expected_blocks:
        missing = sorted(expected_blocks - observed_blocks)
        extra = sorted(observed_blocks - expected_blocks)
        raise ValueError(f"Observation block inventory mismatch; missing={missing}, extra={extra}.")

    beliefs: list[Belief] = []
    if held_block is None:
        beliefs.append(Belief(predicate="HandEmpty", arguments=()))
    else:
        beliefs.append(Belief(predicate="Holding", arguments=(names[held_block].upper(),)))

    for column, stack in enumerate(stacks):
        location = f"t{column}"
        beliefs.append(Belief(predicate="AtLoc", arguments=(location, location)))
        if not stack:
            beliefs.append(Belief(predicate="Clear", arguments=(location,)))
            continue
        support = location
        for block in stack:
            block_name = names[block].upper()
            beliefs.append(Belief(predicate="On", arguments=(block_name, support)))
            beliefs.append(Belief(predicate="AtLoc", arguments=(block_name, location)))
            support = block_name
        beliefs.append(Belief(predicate="Clear", arguments=(support,)))

    for column in range(table_len - 1):
        beliefs.append(
            Belief(predicate="LeftOf", arguments=(f"t{column}", f"t{column + 1}"))
        )
    beliefs.append(Belief(predicate="Above", arguments=(f"t{arm_column}",)))

    facts = beliefs_to_facts(beliefs)
    return GroundedObservation(
        observation=observation,
        beliefs=tuple(beliefs),
        facts=frozenset(facts),
        arm_location=f"t{arm_column}",
        held_block=None if held_block is None else names[held_block],
    )


def enumerate_transfer_targets(facts: set[str] | frozenset[str]) -> list[TransferSpec]:
    """Enumerate fully grounded transfers legal in the supplied symbolic state."""

    clear = {
        arguments[0]
        for fact in facts
        for predicate, arguments in [split_fact(fact)]
        if predicate == "clear" and len(arguments) == 1
    }
    locations = {
        arguments[0]: arguments[1]
        for fact in facts
        for predicate, arguments in [split_fact(fact)]
        if predicate == "at-location" and len(arguments) == 2
    }
    supports = {
        arguments[0]: arguments[1]
        for fact in facts
        for predicate, arguments in [split_fact(fact)]
        if predicate == "on" and len(arguments) == 2
    }
    candidates: list[TransferSpec] = []
    for block in sorted(item for item in clear if item.startswith("b")):
        source_support = supports.get(block)
        source = locations.get(block)
        if source_support is None or source is None:
            continue
        for destination_support in sorted(clear):
            destination = locations.get(destination_support)
            if (
                destination is None
                or destination == source
                or destination_support == block
            ):
                continue
            candidates.append(
                TransferSpec(
                    block=block,
                    source_support=source_support,
                    destination_support=destination_support,
                    source=source,
                    destination=destination,
                )
            )
    return sorted(
        candidates,
        key=lambda spec: (
            spec.block,
            spec.source_support,
            spec.destination_support,
            spec.source,
            spec.destination,
        ),
    )


def transfer_succeeded(facts: set[str] | frozenset[str], spec: TransferSpec) -> bool:
    """Return whether observed facts confirm a completed transfer."""

    return spec.target_facts.issubset(facts) and format_fact("hand-empty") in facts


def transfer_phase(grounded: GroundedObservation, spec: TransferSpec) -> int:
    """Return the highest shaped-reward phase reached by a transfer episode."""

    if transfer_succeeded(grounded.facts, spec):
        return 4
    if grounded.held_block == spec.block and grounded.arm_location == spec.destination:
        return 3
    if grounded.held_block == spec.block:
        return 2
    if (
        grounded.held_block is None
        and grounded.arm_location == spec.source
        and format_fact("hand-empty") in grounded.facts
    ):
        return 1
    return 0
