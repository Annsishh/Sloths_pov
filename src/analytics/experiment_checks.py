"""Experiment integrity checks: sample ratio mismatch (SRM)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats

SRM_ALPHA = 0.001  # conventional, deliberately strict: SRM signals a broken assignment, not a subtle effect


def sample_ratio_mismatch(
    observed: Sequence[int],
    expected_ratios: Optional[Sequence[float]] = None,
    alpha: float = SRM_ALPHA,
) -> Dict[str, Any]:
    """Chi-square goodness-of-fit of assigned counts against the intended allocation.

    Returns chi2, p_value, expected counts and `mismatch` (p < alpha). With a mismatch the experiment's
    assignment is suspect and effect estimates must not be trusted.
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 1 or len(obs) < 2:
        raise ValueError("At least two groups are required for a sample ratio check.")
    if (obs < 0).any():
        raise ValueError("Observed counts must be non-negative.")
    total = obs.sum()
    if total <= 0:
        raise ValueError("Total sample size must be positive.")
    ratios = np.full(len(obs), 1.0 / len(obs)) if expected_ratios is None else np.asarray(expected_ratios, dtype=float)
    if len(ratios) != len(obs) or (ratios <= 0).any():
        raise ValueError("expected_ratios must be positive and match the number of groups.")
    ratios = ratios / ratios.sum()
    expected = total * ratios
    chi2 = float(((obs - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=len(obs) - 1))
    return {
        "observed": [int(o) for o in obs], "expected": [float(e) for e in expected],
        "chi2": chi2, "p_value": p, "alpha": alpha, "mismatch": bool(p < alpha),
        "small_sample": bool((expected < 5).any()),
    }
