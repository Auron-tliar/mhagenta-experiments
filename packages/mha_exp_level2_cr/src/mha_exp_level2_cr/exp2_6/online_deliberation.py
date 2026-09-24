"""Full-world HLR goals, derived from 2-5 with remembered station placement."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..exp2_5.beliefs import Direction, cell_key, known_targets, position_from_key
from ..exp2_5.contracts import RESOURCE_ITEMS, RESOURCE_TOOLS, ActivityId, ActivitySpec, activity_action_bound


def urgent_need(inventory: Mapping[str, int]) -> str | None:
    """Choose the lowest depleted necessity, with stable drink/food/energy ties."""

    needs = [(inventory.get(need, 0), order, need)
             for order, need in enumerate(("drink", "food", "energy"))
             if inventory.get(need, 0) <= 2]
    if needs:
        return min(needs)[2]
    return "health" if inventory.get("health", 0) <= 2 and all(
        inventory.get(need, 0) > 0 for need in ("food", "drink", "energy")) else None


def recovery_complete(state: Mapping[str, Any], need: str) -> bool:
    """Keep natural sleeping recovery active until the player is awake."""

    target = {"food": 5, "drink": 5, "energy": 9, "health": 4}[need]
    return state["inventory"].get(need, 0) >= target and not state["sleeping"]


def routes(state: Mapping[str, Any]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Find known-safe routes for HLR target selection, never neural action selection."""

    player = tuple(state["player"])
    safe = {position_from_key(key) for key in state["safe_cells"]}
    paths = {player: [player]}
    queue = deque([player])
    while queue:
        cell = queue.popleft()
        for direction in Direction:
            other = cell[0] + direction.delta[0], cell[1] + direction.delta[1]
            if other in safe and other not in paths:
                paths[other] = [*paths[cell], other]
                queue.append(other)
    return paths


def local_destination(state: Mapping[str, Any], path: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Select a persistent visible-local goal at most ten known-safe edges away."""

    visible = set(state["visible_cells"])
    candidates = [cell for cell in path[1:11] if cell_key(cell) in visible]
    return candidates[-1] if candidates else None


def failure_key(purpose: str, target: tuple[int, int] | None) -> str:
    """Identify repeated failures without coupling them to observation revision numbers."""

    return f"{purpose}:{target}"


def choose_step(
    state: Mapping[str, Any],
    goal_id: str,
    recovery: str | None,
    excluded: list[str],
    placement: dict[str, Any] | None,
    *, enable_eat_cow: bool = True,
) -> tuple[ActivitySpec | dict[str, Any], dict[str, Any] | None]:
    """Choose one skill or primitive, plus any bounded placement setup state."""

    player = tuple(state["player"])
    inventory = state["inventory"]
    paths = routes(state)
    targets = known_targets(state)

    def activity(name: ActivityId, purpose: str, kind: str = "", target: tuple[int, int] | None = None) -> ActivitySpec:
        if name is ActivityId.EXPLORE:
            baseline, value = len(state["known_cells"]), len(state["known_cells"]) + 10
        elif name is ActivityId.NAVIGATE_TO:
            baseline, value = 0, 1
        elif name is ActivityId.GET_RESOURCE:
            baseline = inventory.get(RESOURCE_ITEMS[kind], 0)
            value = baseline + 1
        else:
            baseline = state["achievement_counts"].get("eat_cow", 0)
            value = baseline + 1
        return ActivitySpec(goal_id, name, purpose, kind, target, state["revision"], baseline, value,
                            max_actions=activity_action_bound(name))

    def explore(purpose: str) -> ActivitySpec:
        return replace(activity(ActivityId.EXPLORE, purpose), max_actions=8)

    def navigate(destination: tuple[int, int], purpose: str) -> ActivitySpec:
        local = local_destination(state, paths[destination])
        return activity(ActivityId.NAVIGATE_TO, purpose, "safe_cell", local) if local else explore(purpose)

    def primitive(action: int, purpose: str) -> dict[str, Any]:
        return {"action": action, "purpose": purpose}

    def gather(kind: str, purpose: str) -> ActivitySpec:
        if state["sleeping"] or inventory.get(RESOURCE_ITEMS[kind], 0) >= 9:
            return explore(purpose)
        if kind in RESOURCE_TOOLS and not inventory.get(RESOURCE_TOOLS[kind], 0):
            return explore(purpose)
        eligible = []
        for target in targets.get(kind, ()):
            if failure_key(f"resource:{kind}", target) in excluded or failure_key(purpose, target) in excluded:
                continue
            neighbors = [(target[0] + d.delta[0], target[1] + d.delta[1]) for d in Direction]
            anchors = [cell for cell in neighbors if cell in paths]
            if anchors:
                anchor = min(anchors, key=lambda cell: (len(paths[cell]), cell))
                eligible.append((len(paths[anchor]), target, anchor))
        if not eligible:
            return explore(purpose)
        distance, target, anchor = min(eligible)
        if cell_key(target) not in state["visible_cells"] or distance > 10:
            return navigate(anchor, purpose)
        return activity(ActivityId.GET_RESOURCE, purpose, kind, target)

    if state["sleeping"]:
        return primitive(6 if inventory["energy"] < 9 else 0, "recovery:energy"), None
    if recovery is not None:
        purpose = f"recovery:{recovery}"
        if recovery == "drink":
            return gather("water", purpose), None
        if recovery == "food":
            cows = [target for target in targets.get("cow", ())
                    if failure_key(purpose, target) not in excluded]
            if cows:
                cow = min(cows, key=lambda cell: (abs(cell[0] - player[0]) + abs(cell[1] - player[1]), cell))
                return activity(ActivityId.EAT_TARGET, purpose, "cow", cow), None
            if enable_eat_cow:
                return activity(ActivityId.EAT_COW, purpose, "cow"), None
            return explore(purpose), None
        return primitive(6 if inventory["energy"] < 9 else 0, purpose), None

    def craft_anchors(furnace: bool) -> list[tuple[int, int]]:
        def near(cell: tuple[int, int], kind: str) -> bool:
            return any(max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) <= 1 for p in targets.get(kind, ()))
        return sorted((cell for cell in paths if near(cell, "table") and (not furnace or near(cell, "furnace"))),
                      key=lambda cell: (len(paths[cell]), cell))

    def place(station: str) -> tuple[ActivitySpec | dict[str, Any], dict[str, Any] | None]:
        purpose = f"technology:place_{station}"
        nonlocal placement
        if placement is not None and placement["station"] == station:
            anchor, destination = tuple(placement["anchor"]), tuple(placement["target"])
            if destination not in paths or anchor not in paths:
                placement = None
            elif player == anchor:
                delta = destination[0] - player[0], destination[1] - player[1]
                if Direction.from_name(state["facing"]).delta == delta:
                    return primitive(8 if station == "table" else 9, purpose), None
                placement = None
            elif placement["moves"] < 2 and abs(anchor[0] - player[0]) + abs(anchor[1] - player[1]) == 1:
                placement["moves"] += 1
                delta = anchor[0] - player[0], anchor[1] - player[1]
                action = next(int(d.value[2]) for d in Direction if d.delta == delta)
                return primitive(action, purpose), placement
            else:
                placement = None
        options = []
        for destination in paths:
            if failure_key(purpose, destination) in excluded:
                continue
            for direction in Direction:
                anchor = destination[0] - direction.delta[0], destination[1] - direction.delta[1]
                orientation = anchor[0] - direction.delta[0], anchor[1] - direction.delta[1]
                if anchor not in paths or orientation not in paths:
                    continue
                if station == "furnace" and not any(max(abs(anchor[0] - t[0]), abs(anchor[1] - t[1])) <= 1 for t in targets.get("table", ())):
                    continue
                options.append((len(paths[orientation]), orientation, anchor, destination))
        if not options:
            return explore(purpose), None
        _, orientation, anchor, destination = min(options)
        placement = {"station": station, "anchor": list(anchor), "target": list(destination), "moves": 0}
        if player != orientation:
            return navigate(orientation, purpose), placement
        placement["moves"] = 1
        delta = anchor[0] - player[0], anchor[1] - player[1]
        action = next(int(d.value[2]) for d in Direction if d.delta == delta)
        return primitive(action, purpose), placement

    if inventory.get("diamond", 0):
        return explore("exploration"), None
    stages = (
        ("wood_pickaxe", 11, {"wood": 1}, False),
        ("stone_pickaxe", 12, {"stone": 1, "wood": 1}, False),
        ("iron_pickaxe", 13, {"coal": 1, "iron": 1, "wood": 1}, True),
    )
    stage, action, needs, furnace = next(
        (row for row in stages if not inventory.get(row[0], 0)),
        ("diamond", 0, {"diamond": 1}, True),
    )
    purpose = f"technology:{stage}"
    anchors = craft_anchors(False)
    if not anchors:
        if inventory.get("wood", 0) < 2:
            return gather("tree", "technology:table"), None
        return place("table")
    if furnace and not craft_anchors(True):
        if inventory.get("stone", 0) < 4:
            return gather("stone", "technology:furnace"), None
        return place("furnace")
    for item, count in needs.items():
        if inventory.get(item, 0) < count:
            return gather("tree" if item == "wood" else item, purpose), None
    anchors = craft_anchors(furnace)
    if player not in anchors:
        return navigate(anchors[0], purpose), None
    return primitive(action, purpose), None
