"""Small dependency-free helpers for experiment execution summaries."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping


def summarize_numbers(
    values: Iterable[float | int | None],
    *,
    include_iqr: bool = False,
    p95_min_count: int | None = None,
) -> dict[str, int | float | None]:
    """Summarize present finite numbers without imputing missing observations."""

    materialized = list(values)
    present: list[float] = []
    for value in materialized:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Numeric summaries require numbers or None.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Numeric summaries require finite values.")
        present.append(number)

    summary: dict[str, int | float | None] = {
        "count": len(present),
        "missing": len(materialized) - len(present),
        "median": statistics.median(present) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }
    if include_iqr and len(present) >= 4:
        q1, _, q3 = statistics.quantiles(present, n=4, method="inclusive")
        summary.update({"q1": q1, "q3": q3, "iqr": q3 - q1})
    if p95_min_count is not None:
        if p95_min_count <= 0:
            raise ValueError("p95_min_count must be positive.")
        if len(present) >= p95_min_count:
            summary["p95"] = statistics.quantiles(
                present, n=100, method="inclusive",
            )[94]
    return summary


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float] | None:
    """Return a Wilson score interval for a binomial proportion."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("Wilson counts must be integers.")
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total.")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a positive finite value.")
    if total == 0:
        return None
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )
    return centre - radius, centre + radius


def empirical_action_diversity(counts: Mapping[str, int]) -> float | None:
    """Return Shannon entropy in bits for an observed action distribution."""

    total = 0
    for count in counts.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Action counts must be nonnegative integers.")
        total += count
    if total == 0:
        return None
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )
