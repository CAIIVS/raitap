"""Binomial confidence intervals for average-case (statistical-sampling) accuracy.

``wilson`` is closed-form (no scipy). ``clopper_pearson`` is the exact
Beta-quantile interval and needs ``scipy.stats.beta`` — available through the
``imagecorruptions`` extra (skimage/opencv pull scipy in), so it sits behind the
same optional gate as the adapter that uses it.
"""

from __future__ import annotations

import math
from typing import Literal

CIMethod = Literal["wilson", "clopper_pearson"]

# z for a two-sided normal interval at common levels; 0.95 -> 1.959963985.
_Z_BY_LEVEL: dict[float, float] = {0.90: 1.6448536269, 0.95: 1.9599639845, 0.99: 2.5758293035}


def _z_for_level(level: float) -> float:
    z = _Z_BY_LEVEL.get(round(level, 2))
    if z is not None:
        return z
    # General fallback via the inverse error function.
    return math.sqrt(2.0) * _erfinv(level)


def _erfinv(x: float) -> float:
    # Winitzki approximation; only used for non-tabulated levels.
    a = 0.147
    ln = math.log(1.0 - x * x)
    term = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), x)


def binomial_ci(
    n_correct: int,
    n: int,
    *,
    level: float = 0.95,
    method: CIMethod = "wilson",
) -> tuple[float, float]:
    """Return the ``level`` two-sided CI for the success probability.

    ``n == 0`` returns the uninformative ``(0.0, 1.0)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    if method == "wilson":
        return _wilson(n_correct, n, level)
    if method == "clopper_pearson":
        return _clopper_pearson(n_correct, n, level)
    raise ValueError(f"Unknown ci_method {method!r}; expected 'wilson' or 'clopper_pearson'.")


def _wilson(k: int, n: int, level: float) -> tuple[float, float]:
    z = _z_for_level(level)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _clopper_pearson(k: int, n: int, level: float) -> tuple[float, float]:
    from scipy.stats import beta  # lazy: only the exact path needs scipy

    alpha = 1.0 - level
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)
