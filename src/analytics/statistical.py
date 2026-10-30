"""Statistical hypothesis testing and uncertainty estimates.

Provides rigorous, calibrated comparisons:
1. Two-proportion z-test (conversion rates, failure rates, pass rates)
   with pooled standard error, two-tailed p-value, and 95% confidence interval.
2. Welch's two-sample t-test (order values, revenue, feedback ratings)
   with unpooled variances, Welch-Satterthwaite degrees of freedom, and confidence intervals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus


def compare_proportions(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    label1: str = "sample_1",
    label2: str = "sample_2",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform a two-proportion z-test comparing k1/n1 vs k2/n2."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive. Found n1={n1}, n2={n2}")
    if k1 < 0 or k1 > n1 or k2 < 0 or k2 > n2:
        raise ValueError(f"Success counts must be between 0 and n. Found k1={k1}/{n1}, k2={k2}/{n2}")

    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2

    p_pool = (k1 + k2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))

    # Standard error for confidence interval (unpooled)
    se_ci = np.sqrt((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2))

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = diff - z_crit * se_ci
    ci_upper = diff + z_crit * se_ci

    if se_pool > 0:
        z_stat = float(diff / se_pool)
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    else:
        z_stat = 0.0
        p_val = 1.0

    # Rule of thumb for normal approximation: n*p >= 5 and n*(1-p) >= 5
    warnings = []
    if min(k1, n1 - k1, k2, n2 - k2) < 5:
        warnings.append(
            "Low expected count (< 5) in at least one cell; normal approximation may have reduced accuracy."
        )

    return {
        f"rate_{label1}": round(p1, 4),
        f"rate_{label2}": round(p2, 4),
        "count_1": k1,
        "n_1": n1,
        "count_2": k2,
        "n_2": n2,
        "absolute_difference": round(diff, 4),
        "relative_difference": round(diff / p2, 4) if p2 != 0 else None,
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_val, 6),
        "ci_lower_95": round(ci_lower, 4),
        "ci_upper_95": round(ci_upper, 4),
        "is_significant_p05": bool(p_val < 0.05),
        "is_significant_p01": bool(p_val < 0.01),
        "warnings": warnings,
    }


def compare_means(
    values1: Sequence[float] | np.ndarray,
    values2: Sequence[float] | np.ndarray,
    label1: str = "sample_1",
    label2: str = "sample_2",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform Welch's two-sample t-test (unequal variances)."""
    arr1 = np.asarray(values1, dtype=float)
    arr2 = np.asarray(values2, dtype=float)

    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    n1, n2 = len(arr1), len(arr2)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"At least 2 observations required per sample. Found n1={n1}, n2={n2}")

    m1, m2 = float(np.mean(arr1)), float(np.mean(arr2))
    s1, s2 = float(np.std(arr1, ddof=1)), float(np.std(arr2, ddof=1))
    diff = m1 - m2

    se = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    warnings: List[str] = []

    if n1 < 10 or n2 < 10:
        warnings.append(
            f"Small sample size (n1={n1}, n2={n2}); normality is critical for Welch's t-test validity."
        )

    # Welch-Satterthwaite degrees of freedom
    if se > 0:
        df_num = ((s1 ** 2) / n1 + (s2 ** 2) / n2) ** 2
        df_den = (((s1 ** 2) / n1) ** 2) / (n1 - 1) + (((s2 ** 2) / n2) ** 2) / (n2 - 1)
        dof = float(df_num / df_den) if df_den > 0 else 1.0

        t_stat = float(diff / se)
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se
    else:
        # Zero sample variance
        dof = float(n1 + n2 - 2)
        ci_lower = diff
        ci_upper = diff
        if abs(diff) < 1e-9:
            t_stat = 0.0
            p_val = 1.0
            warnings.append("Zero variance in both samples; groups are identical constants.")
        else:
            t_stat = None
            p_val = None
            warnings.append("Zero variance observed with unequal means; t-statistic cannot be standardly evaluated.")

    return {
        f"mean_{label1}": round(m1, 4),
        f"mean_{label2}": round(m2, 4),
        f"std_{label1}": round(s1, 4),
        f"std_{label2}": round(s2, 4),
        "n_1": n1,
        "n_2": n2,
        "absolute_difference": round(diff, 4),
        "relative_difference": round(diff / m2, 4) if m2 != 0 else None,
        "t_statistic": round(t_stat, 4) if t_stat is not None else None,
        "degrees_of_freedom": round(dof, 2),
        "p_value": round(p_val, 6) if p_val is not None else None,
        "ci_lower_95": round(ci_lower, 4),
        "ci_upper_95": round(ci_upper, 4),
        "is_significant_p05": bool(p_val < 0.05) if p_val is not None else None,
        "is_significant_p01": bool(p_val < 0.01) if p_val is not None else None,
        "warnings": warnings,
    }


def run_statistical_test(request: AnalysisRequest) -> AnalysisResult:
    """Analysis entry-point for standalone statistical tests.

    Parameters in ``request.params``
    ---------------------------------
    test_type : "proportions" | "means"
    k1, n1, k2, n2 : required if test_type="proportions"
    values1, values2 : required if test_type="means"
    """
    test_type = request.params.get("test_type", "proportions")

    try:
        if test_type == "proportions":
            res_dict = compare_proportions(
                k1=int(request.params["k1"]),
                n1=int(request.params["n1"]),
                k2=int(request.params["k2"]),
                n2=int(request.params["n2"]),
                label1=request.params.get("label1", "sample_1"),
                label2=request.params.get("label2", "sample_2"),
            )
        elif test_type == "means":
            res_dict = compare_means(
                values1=request.params["values1"],
                values2=request.params["values2"],
                label1=request.params.get("label1", "sample_1"),
                label2=request.params.get("label2", "sample_2"),
            )
        else:
            return AnalysisResult(
                request_id=request.request_id,
                operation="statistical_test",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=f"Unknown test_type '{test_type}'. Supported: 'proportions', 'means'.",
            )
    except Exception as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="statistical_test",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    warnings = res_dict.pop("warnings", [])
    df = pd.DataFrame([res_dict])

    return AnalysisResult(
        request_id=request.request_id,
        operation="statistical_test",
        status=AnalysisStatus.OK,
        data=df,
        metadata={
            "test_type": test_type,
            "window": str(request.window),
        },
        limitations=["Assumes independent observations and random sampling."],
        warnings=warnings,
    )
