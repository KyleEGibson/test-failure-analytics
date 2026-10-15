"""Statistical process control: control limits, run rules and capability.

Constants and rules follow standard SPC practice (Montgomery, *Introduction to
Statistical Quality Control*; Western Electric *Statistical Quality Control
Handbook*, 1956).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Moving range of 2 consecutive points: d2 = 1.128, D4 = 3.267.
D2 = 1.128
D4 = 3.267


@dataclass(frozen=True)
class Limits:
    center: float
    lcl: float
    ucl: float
    sigma: float


def imr_limits(values: pd.Series | np.ndarray) -> tuple[Limits, Limits]:
    """Individuals and moving-range chart limits.

    Sigma is estimated from the average moving range (MR-bar / d2), not the
    sample standard deviation, so a shift in the data does not inflate the
    limits that are meant to detect it.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        raise ValueError("need at least 2 points for an I-MR chart")
    mr_bar = np.mean(np.abs(np.diff(x)))
    sigma = mr_bar / D2
    center = float(np.mean(x))
    individuals = Limits(center, center - 3 * sigma, center + 3 * sigma, sigma)
    moving_range = Limits(mr_bar, 0.0, D4 * mr_bar, sigma)
    return individuals, moving_range


def p_chart(defectives: pd.Series, sample_sizes: pd.Series) -> pd.DataFrame:
    """p-chart with variable subgroup sizes.

    Returns one row per subgroup with the fraction defective, the overall
    center line and that subgroup's own 3-sigma limits.
    """
    d = np.asarray(defectives, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    if np.any(n <= 0):
        raise ValueError("sample sizes must be positive")
    p_bar = d.sum() / n.sum()
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    return pd.DataFrame(
        {
            "defectives": d,
            "n": n,
            "p": d / n,
            "center": p_bar,
            "sigma": sigma,
            "lcl": np.clip(p_bar - 3 * sigma, 0, None),
            "ucl": np.clip(p_bar + 3 * sigma, None, 1),
        },
        index=getattr(defectives, "index", None),
    )


def western_electric(values, center, sigma) -> pd.DataFrame:
    """Flag points that violate the four Western Electric run rules.

    ``center`` and ``sigma`` may be scalars or per-point arrays (p-charts with
    variable n). Returns one row per (position, rule) violation; the flagged
    position is the point that completes the pattern.

    Rule 1: one point beyond 3 sigma.
    Rule 2: 2 of 3 consecutive points beyond 2 sigma, same side.
    Rule 3: 4 of 5 consecutive points beyond 1 sigma, same side.
    Rule 4: 8 consecutive points on the same side of the center line.
    """
    x = np.asarray(values, dtype=float)
    z = (x - np.broadcast_to(center, x.shape)) / np.broadcast_to(sigma, x.shape)
    found: list[tuple[int, int, str]] = []
    descriptions = {
        1: "1 point beyond 3 sigma",
        2: "2 of 3 beyond 2 sigma",
        3: "4 of 5 beyond 1 sigma",
        4: "8 in a row on one side",
    }

    def window_rule(rule: int, width: int, needed: int, threshold: float) -> None:
        for end in range(width - 1, len(z)):
            window = z[end - width + 1 : end + 1]
            for sign in (1, -1):
                if np.sum(sign * window > threshold) >= needed and sign * z[end] > threshold:
                    found.append((end, rule, descriptions[rule]))
                    break

    for i, value in enumerate(z):
        if abs(value) > 3:
            found.append((i, 1, descriptions[1]))
    window_rule(2, 3, 2, 2.0)
    window_rule(3, 5, 4, 1.0)
    for end in range(7, len(z)):
        window = z[end - 7 : end + 1]
        if np.all(window > 0) or np.all(window < 0):
            found.append((end, 4, descriptions[4]))

    result = pd.DataFrame(found, columns=["position", "rule", "description"])
    return result.sort_values(["position", "rule"]).reset_index(drop=True)


def cpk(values, lsl: float | None = None, usl: float | None = None) -> float:
    """Process capability index; one-sided when only one spec limit is given."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    mean, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if sd == 0:
        return float("inf")
    sides = []
    if usl is not None:
        sides.append((usl - mean) / (3 * sd))
    if lsl is not None:
        sides.append((mean - lsl) / (3 * sd))
    if not sides:
        raise ValueError("at least one spec limit is required")
    return float(min(sides))
