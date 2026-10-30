"""Multiple-comparison correction (Benjamini–Hochberg FDR)."""

from __future__ import annotations

from typing import List, Sequence


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Return BH-adjusted p-values (q-values) in the input order.

    Sort ascending, scale p_(i) by m/i, then enforce monotonicity from the largest rank down.
    """
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        if p is None or not (0.0 <= p <= 1.0):
            raise ValueError(f"p-values must lie in [0, 1]; got {p!r}")

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running_min = min(running_min, p_values[idx] * m / rank)
        adjusted[idx] = running_min
    return adjusted
