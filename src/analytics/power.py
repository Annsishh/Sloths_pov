"""Minimum detectable effect (MDE) and power helpers (normal approximation).

MDE answers: "how large would a true difference have to be for this test to detect it with
`power` probability at level `alpha`?" A non-significant result from a test whose MDE exceeds
the effect of interest is inconclusive, not evidence of no effect.
"""

from __future__ import annotations

import math
from typing import Optional

from scipy import stats

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def _z_sum(alpha: float, power: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))


def mde_proportions(
    p_baseline: float, n1: int, n2: int, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> Optional[float]:
    """Absolute MDE for a two-proportion z-test around baseline rate `p_baseline`.

    A baseline of exactly 0 or 1 has zero variance; it is clamped to 1/n (n = smaller sample) so
    the MDE is not reported as 0.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    p = min(max(p_baseline, 0.0), 1.0)
    floor = 1.0 / min(n1, n2)
    p = min(max(p, floor), 1.0 - floor) if floor < 0.5 else 0.5
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    return _z_sum(alpha, power) * se


def mde_means(
    sd: float, n1: int, n2: int, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> Optional[float]:
    """Absolute MDE for a two-sample comparison of means with common standard deviation `sd`."""
    if n1 <= 0 or n2 <= 0 or sd is None or sd < 0 or math.isnan(sd):
        return None
    return _z_sum(alpha, power) * sd * math.sqrt(1.0 / n1 + 1.0 / n2)


def power_for_effect(se: float, effect: float, alpha: float = DEFAULT_ALPHA) -> float:
    """Power of a two-sided z-test with standard error `se` to detect a true difference `effect`."""
    if se <= 0:
        return 1.0 if effect != 0 else alpha
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    d = abs(effect) / se
    return float(stats.norm.cdf(d - z) + stats.norm.cdf(-d - z))
