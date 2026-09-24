"""Observable cow association and target-specific interaction evidence."""

from collections.abc import Iterable, Sequence
from functools import lru_cache

Position = tuple[int, int]


def associate_cow(
    target: Position | None,
    previous: Iterable[Position],
    current: Iterable[Position],
) -> Position | None:
    """Match visible positions under one-cell motion; reject ambiguous targets.

    Prefer the most retained tracks. Use movement cost only for a complete
    bijection: births or losses can otherwise impersonate a stationary cow.
    Only the connected association component
    containing the target matters. A component larger than 12 cows is
    conservatively untracked to bound runtime work. No private IDs are used.
    """
    old, new = set(previous), set(current)
    if target is None or target not in old:
        return None

    def distance(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    old_component, new_component = {target}, set()
    while True:
        visible = {cell for cell in new if any(distance(cell, prior) <= 1 for prior in old_component)}
        prior = {cell for cell in old if any(distance(cell, now) <= 1 for now in visible)} | {target}
        if prior == old_component and visible == new_component:
            break
        old_component, new_component = prior, visible
        if max(len(old_component), len(new_component)) > 12:
            return None
    others, destinations = sorted(old_component - {target}), sorted(new_component)

    @lru_cache(maxsize=None)
    def score(index: int, used: int) -> tuple[int, int]:
        """Maximize retained tracks and then minimize total displacement."""
        if index == len(others):
            return 0, 0
        best = score(index + 1, used)
        for slot, cell in enumerate(destinations):
            movement = distance(others[index], cell)
            if not used & (1 << slot) and movement <= 1:
                retained, cost = score(index + 1, used | (1 << slot))
                best = max(best, (retained + 1, cost - movement))
        return best

    possibilities = [(None, score(0, 0))]
    for slot, cell in enumerate(destinations):
        movement = distance(target, cell)
        if movement <= 1:
            retained, cost = score(0, 1 << slot)
            possibilities.append((cell, (retained + 1, cost - movement)))
    retained = max(value[0] for _, value in possibilities)
    complete = retained == len(old_component) == len(new_component)
    scores = [(cell, value if complete else (value[0], 0)) for cell, value in possibilities]
    best = max(value for _, value in scores)
    matches = [cell for cell, value in scores if value == best]
    return matches[0] if len(matches) == 1 else None


def consumed_tracked_cow(
    action: int,
    source: Sequence[int],
    facing: Sequence[int],
    target: Sequence[int] | None,
    new_achievements: Iterable[str],
    illegal: bool,
) -> bool:
    """Require fresh consumption at the designated pre-action interaction cell."""
    return (action == 5 and not illegal and target is not None
            and tuple(a + b for a, b in zip(source, facing, strict=True)) == tuple(target)
            and "eat_cow" in new_achievements)
