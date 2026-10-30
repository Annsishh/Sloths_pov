"""Ratio metrics (e.g. revenue per session) with delta-method variance.

A ratio of totals R = sum(x_i) / sum(y_i) over independent units i is NOT a mean of independent values, so
Var(R) cannot be computed as for a simple mean. The delta method gives

    Var(R) ~= (1 / (n * ybar^2)) * (s_x^2 + R^2 * s_y^2 - 2 R * cov(x, y))
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.utils import apply_time_filter

RATIO_METRICS = {
    "revenue_per_session": "completed-order revenue / sessions",
    "orders_per_session": "completed orders / sessions",
}


def delta_method_ratio(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Ratio estimate sum(x)/sum(y) with delta-method standard error over per-unit pairs (x_i, y_i)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    n = len(xa)
    if n < 2:
        raise ValueError("At least 2 units are required for a ratio-metric variance.")
    ybar = float(ya.mean())
    if ybar == 0:
        raise ValueError("Denominator total is zero; the ratio is undefined.")
    ratio = float(xa.sum() / ya.sum())
    sx2, sy2 = float(xa.var(ddof=1)), float(ya.var(ddof=1))
    cov = float(np.cov(xa, ya, ddof=1)[0, 1])
    var = (sx2 + ratio ** 2 * sy2 - 2 * ratio * cov) / (n * ybar ** 2)
    var = max(var, 0.0)   # numerical noise can push a perfectly proportional pair slightly negative
    return {"ratio": ratio, "se": math.sqrt(var), "variance": var, "n_units": n}


def compare_ratio_metrics(x1, y1, x0, y0, alpha: float = 0.05) -> Dict[str, Any]:
    """Difference of two independent ratio metrics (group/period 1 minus 0) with z-test and CI."""
    a, b = delta_method_ratio(x1, y1), delta_method_ratio(x0, y0)
    diff = a["ratio"] - b["ratio"]
    se = math.sqrt(a["variance"] + b["variance"])
    if se == 0:
        z, p = (0.0, 1.0) if diff == 0 else (math.inf, 0.0)
    else:
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
    zc = float(stats.norm.ppf(1 - alpha / 2))
    return {"ratio_1": a["ratio"], "ratio_0": b["ratio"], "se_1": a["se"], "se_0": b["se"], "difference": diff,
            "se_difference": se, "z": z, "p_value": p, "ci_lower": diff - zc * se, "ci_upper": diff + zc * se,
            "n_1": a["n_units"], "n_0": b["n_units"]}


def _per_user(orders: pd.DataFrame, events: pd.DataFrame, window, metric: str) -> pd.DataFrame:
    ev = apply_time_filter(events, "timestamp", window)
    sessions = ev.groupby("user_id")["session_id"].nunique().rename("y")
    od = apply_time_filter(orders, "timestamp", window)
    od = od[od["status"] == "completed"]
    num = (od.groupby("user_id")["amount"].sum() if metric == "revenue_per_session"
           else od.groupby("user_id").size().astype(float)).rename("x")
    df = pd.concat([sessions, num], axis=1).fillna({"x": 0.0}).dropna(subset=["y"])
    return df[df["y"] > 0].reset_index()


def ratio_metric_comparison(orders: pd.DataFrame, events: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
    """Compare a per-session ratio metric between periods using the delta method (user = independent unit)."""
    def bad(msg, status=AnalysisStatus.INVALID_PARAMS):
        return AnalysisResult(request.request_id, "ratio_metric_comparison", status, None, error_message=msg)

    metric = request.params.get("metric_name", "revenue_per_session")
    if metric not in RATIO_METRICS:
        return bad(f"Unknown ratio metric '{metric}'. Supported: {sorted(RATIO_METRICS)}")
    if request.comparison_window is None:
        return bad("ratio_metric_comparison requires a comparison_window (baseline period).")
    cur = _per_user(orders, events, request.window, metric)
    base = _per_user(orders, events, request.comparison_window, metric)
    if len(cur) < 2 or len(base) < 2:
        return bad("Each period needs at least 2 users with sessions.", AnalysisStatus.INSUFFICIENT_DATA)
    try:
        r = compare_ratio_metrics(cur["x"], cur["y"], base["x"], base["y"])
    except ValueError as e:
        return bad(str(e), AnalysisStatus.INSUFFICIENT_DATA)
    df = pd.DataFrame([
        {"period": "baseline", "ratio": r["ratio_0"], "se": r["se_0"], "users": r["n_0"]},
        {"period": "comparison", "ratio": r["ratio_1"], "se": r["se_1"], "users": r["n_1"]},
        {"period": "change", "ratio": r["difference"], "se": r["se_difference"], "users": None},
    ])
    warnings: List[str] = []
    if min(r["n_0"], r["n_1"]) < 30:
        warnings.append("Fewer than 30 users in a period; the delta-method normal approximation is unreliable.")
    return AnalysisResult(
        request_id=request.request_id, operation="ratio_metric_comparison", status=AnalysisStatus.OK, data=df,
        metadata={"metric_name": metric, "definition": RATIO_METRICS[metric], "absolute_change": r["difference"],
                  "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"], "z": r["z"],
                  "p_values": {"ratio_change": r["p_value"]}, "window": str(request.window)},
        limitations=["The user is the independent unit; sessions of one user are treated as one cluster (delta method).",
                     "Ratio of totals, not mean of per-user ratios."],
        warnings=warnings,
    )
