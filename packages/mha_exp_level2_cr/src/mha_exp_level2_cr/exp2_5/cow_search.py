"""Observable discovery goals shared by EatCow preparation and execution."""

from collections import deque
from collections.abc import Iterable, Sequence

from .beliefs import Position, frontier_cells

_MOVES = ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1)))


def camera_cells(player: Position, world_origin: Position = (0, 0)) -> set[Position]:
    """Return the clipped 9-by-7 visible footprint in the 64-by-64 world."""
    left, top = world_origin
    return {(x, y) for x in range(max(left, player[0] - 4), min(left + 64, player[0] + 5))
            for y in range(max(top, player[1] - 3), min(top + 64, player[1] + 4))}


def discovery_goal(
    player: Position, known_cells: Iterable[Position], safe_cells: Iterable[Position],
    previous: Position | None, movement_actions: Sequence[int],
    *, world_origin: Position = (0, 0),
) -> Position | None:
    """Keep a reachable public goal or select camera gain per movement.

    Inputs contain observed terrain and current visible occupancy only. The
    first route step respects the execution mask; at most 31 moves are searched.
    This chooses a destination, never the neural policy's action.
    """
    known, safe = set(known_cells), set(safe_cells) | {player}
    distances = {player: 0}
    queue = deque([player])
    while queue:
        cell = queue.popleft()
        if distances[cell] >= 31:
            continue
        for action, (dx, dy) in _MOVES:
            if cell == player and action not in movement_actions:
                continue
            other = cell[0] + dx, cell[1] + dy
            if other in safe and other not in distances:
                distances[other] = distances[cell] + 1
                queue.append(other)
    if previous in distances and previous != player:
        return previous
    for candidates in (frontier_cells(known, safe), distances):
        choices = []
        for cell in candidates:
            distance = distances.get(cell, 0)
            gain = len(camera_cells(cell, world_origin) - known)
            if distance and gain:
                choices.append((-gain / distance, distance, cell))
        if choices:
            return min(choices)[2]
    return None
