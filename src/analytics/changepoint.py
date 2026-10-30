"""Change-point detection on a daily metric series (single split scan + seeded permutation test).

For every admissible split k the two-sample t-like statistic |mean_after - mean_before| / pooled SE is
computed; the split with the largest statistic is the estimated onset. Because the maximum over many splits is
inflated under the null, significance comes from a permutation test of the MAXIMUM statistic (seeded,
reproducible). A step model is also compared with a linear ramp to hint at a gradual onset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus, TimeWindow
from analytics.seasonality import _daily
from analytics.utils import apply_filters, apply_time_filter, enrich_with_user_attrs, validate_aggregation


def _scan(x: np.ndarray, min_seg: int) -> Tuple[int, float]:
    """Return (best split index k, max |t|) where segments are x[:k] and x[k:]."""
    n = len(x)
    best_k, best_t = -1, 0.0
    for k in range(min_seg, n - min_seg + 1):
        a, b = x[:k], x[k:]
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va / len(a) + vb / len(b))
        if se == 0:
            t = 0.0 if a.mean() == b.mean() else np.inf
        else:
            t = abs(b.mean() - a.mean()) / se
        if t > best_t:
            best_k, best_t = k, float(t)
    return best_k, best_t


def detect_change_point(
    x: np.ndarray, min_segment: int = 3, n_permutations: int = 499, seed: int = 0
) -> Dict[str, Any]:
    """Locate the most likely single change point in `x` and test it with a seeded permutation test."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_segment:
        raise ValueError(f"Need at least {2 * min_segment} points for a change-point scan (got {n}).")
    k, t_obs = _scan(x, min_segment)
    if k < 0:
        raise ValueError("No admissible split found.")
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        if _scan(rng.permutation(x), min_segment)[1] >= t_obs:
            exceed += 1
    p = (exceed + 1) / (n_permutations + 1)

    before, after = x[:k], x[k:]
    # Step vs linear-ramp fit: a ramp that fits better than a step suggests a gradual onset.
    step_sse = float(((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum())
    t_idx = np.arange(n)
    ramp_coef = np.polyfit(t_idx, x, 1)
    ramp_sse = float(((x - np.polyval(ramp_coef, t_idx)) ** 2).sum())
    shape = "gradual" if ramp_sse < step_sse else "step"
    return {
        "change_index": int(k), "statistic": t_obs, "p_value": float(p),
        "mean_before": float(before.mean()), "mean_after": float(after.mean()),
        "shift": float(after.mean() - before.mean()), "onset_shape": shape,
        "step_sse": step_sse, "ramp_sse": ramp_sse, "ramp_slope": float(ramp_coef[0]),
    }


def change_point(primary_df: pd.DataFrame, users_df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
    """Estimate when a metric shifted. The scan covers the baseline and comparison windows together."""
    def bad(msg, status=AnalysisStatus.INVALID_PARAMS):
        return AnalysisResult(request.request_id, "change_point", status, None, error_message=msg)

    p = request.params
    aggregation = str(p.get("aggregation", "count")).lower()
    metric_column, metric_value = p.get("metric_column"), p.get("metric_value")
    time_column = p.get("time_column", "timestamp")
    min_segment = int(p.get("min_segment", 3))
    err = validate_aggregation(aggregation)
    if err:
        return bad(err)
    if aggregation in ("sum", "mean") and not metric_column:
        return bad(f"aggregation='{aggregation}' requires 'metric_column'.")
    if aggregation == "rate" and (not metric_column or metric_value is None):
        return bad("aggregation='rate' requires 'metric_column' and 'metric_value'.")
    if time_column not in primary_df.columns:
        return bad(f"time_column '{time_column}' not found.")
    if min_segment < 2:
        return bad("min_segment must be >= 2.")

    start = min(request.window.start, request.comparison_window.start) if request.comparison_window else request.window.start
    # (the controller already applies params["lookback_days"] to the window start)
    end = max(request.window.end, request.comparison_window.end) if request.comparison_window else request.window.end
    scoped = apply_time_filter(primary_df, time_column, TimeWindow(start, end))
    need = [c for c in request.filters if c in users_df.columns]
    if need:
        scoped = enrich_with_user_attrs(scoped, users_df, need)
    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return bad(str(e))
    if scoped.empty:
        return bad("No data in the scan range.", AnalysisStatus.INSUFFICIENT_DATA)

    series = _daily(scoped, time_column, aggregation, metric_column, metric_value)
    days = pd.date_range(pd.Timestamp(start).tz_convert("UTC").floor("D"), pd.Timestamp(end).tz_convert("UTC").floor("D"),
                         freq="D", inclusive="left")
    if aggregation in ("count", "sum"):
        full = series.reindex(days).fillna(0.0)
    else:
        full = series.reindex(days)
    warnings: List[str] = []
    if full.isna().any():
        warnings.append(f"{int(full.isna().sum())} day(s) had no observations and were interpolated.")
        full = full.interpolate(limit_direction="both")
    if len(full) < 2 * min_segment:
        return bad(f"Only {len(full)} day(s) in range; a change-point scan needs at least {2 * min_segment}.",
                   AnalysisStatus.INSUFFICIENT_DATA)
    if float(full.std(ddof=1)) == 0.0:
        return bad("The series is constant; there is no change point to detect.", AnalysisStatus.INSUFFICIENT_DATA)

    r = detect_change_point(full.to_numpy(), min_segment=min_segment, n_permutations=int(p.get("n_permutations", 499)))
    from analytics.power import mde_means
    half = len(full) // 2
    mde = mde_means(float(full.std(ddof=1)), half, len(full) - half)
    onset = full.index[r["change_index"]]
    df = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in full.index], "value": full.to_numpy(),
                       "segment": ["before" if i < r["change_index"] else "after" for i in range(len(full))]})
    if r["onset_shape"] == "gradual":
        warnings.append("A linear ramp fits better than a step: the shift looks gradual, so the estimated onset date is approximate.")
    return AnalysisResult(
        request_id=request.request_id,
        operation="change_point",
        status=AnalysisStatus.OK,
        data=df,
        metadata={
            "estimated_onset": onset.strftime("%Y-%m-%d"), "change_index": r["change_index"],
            "mean_before": r["mean_before"], "mean_after": r["mean_after"], "shift": r["shift"],
            "onset_shape": r["onset_shape"], "ramp_slope_per_day": r["ramp_slope"],
            "p_values": {"change_point_permutation": r["p_value"]},
            "mde": mde,    # smallest mean shift (metric units) a half/half split would reliably detect
            "n_days": len(full), "window": f"[{start.date()} – {end.date()})",
        },
        limitations=[
            "Assumes a single change point; multiple shifts are not separated.",
            "The permutation p-value treats days as exchangeable, ignoring autocorrelation and weekly seasonality.",
        ],
        warnings=warnings,
    )
