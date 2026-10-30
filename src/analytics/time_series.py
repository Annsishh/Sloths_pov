"""Time-series and period-over-period metric comparisons.

Computes chronological aggregations and mathematically sound period comparisons
for counts, sums, means, and rates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.utils import (
    apply_filters,
    apply_time_filter,
    enrich_with_user_attrs,
    validate_aggregation,
    validate_numeric_column,
)


def compute_time_series(
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Aggregate a metric chronologically across the request window.

    Parameters in ``request.params``
    ---------------------------------
    metric_column : str, optional
        Column to aggregate.
    aggregation : str
        "count" (default) | "sum" | "mean" | "rate" | "std" | "median".
    metric_value : Any, optional
        Target value when aggregation="rate".
    time_column : str
        Timestamp column (default "timestamp").
    freq : str
        Pandas frequency string — "D" (day, default), "W" (week).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    metric_column: Optional[str] = request.params.get("metric_column")
    aggregation: str = request.params.get("aggregation", "count").lower()
    metric_value: Optional[Any] = request.params.get("metric_value")
    time_column: str = request.params.get("time_column", "timestamp")
    freq: str = request.params.get("freq", "D")

    agg_err = validate_aggregation(aggregation)
    if agg_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="time_series",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=agg_err,
        )

    if aggregation == "rate" and metric_value is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="time_series",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="aggregation='rate' requires 'metric_value'.",
        )

    if time_column not in primary_df.columns:
        return AnalysisResult(
            request_id=request.request_id,
            operation="time_series",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=f"time_column '{time_column}' not found in DataFrame.",
        )

    scoped = apply_time_filter(primary_df, time_column, request.window)

    needed_user_cols = [c for c in request.filters if c in users_df.columns]
    if needed_user_cols:
        scoped = enrich_with_user_attrs(scoped, users_df, needed_user_cols)

    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="time_series",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if scoped.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="time_series",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No data in the specified window after applying filters.",
        )

    if aggregation in {"sum", "mean", "std", "median"}:
        if metric_column is None:
            return AnalysisResult(
                request_id=request.request_id,
                operation="time_series",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=f"aggregation='{aggregation}' requires 'metric_column'.",
            )
        num_err = validate_numeric_column(scoped, metric_column)
        if num_err:
            return AnalysisResult(
                request_id=request.request_id,
                operation="time_series",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=num_err,
            )

    scoped = scoped.copy()
    ts_utc = scoped[time_column].dt.tz_convert("UTC") if scoped[time_column].dt.tz is not None else scoped[time_column]
    if freq.upper() == "D":
        scoped["_period"] = ts_utc.dt.floor("D")
    elif freq.upper() == "W":
        scoped["_period"] = ts_utc.dt.floor("D") - pd.to_timedelta(ts_utc.dt.weekday, unit="D")
    else:
        with pd.option_context("mode.chained_assignment", None):
            scoped["_period"] = ts_utc.dt.to_period(freq).dt.start_time

    if aggregation == "count" or metric_column is None:
        agg_df = scoped.groupby("_period").size().reset_index(name="value")
        agg_df["count"] = agg_df["value"]
    elif aggregation == "sum":
        counts = scoped.groupby("_period").size().rename("count")
        sums = scoped.groupby("_period")[metric_column].sum().rename("value")
        agg_df = pd.concat([counts, sums], axis=1).reset_index()
    elif aggregation == "mean":
        counts = scoped.groupby("_period").size().rename("count")
        means = scoped.groupby("_period")[metric_column].mean().rename("value")
        agg_df = pd.concat([counts, means], axis=1).reset_index()
    elif aggregation == "std":
        counts = scoped.groupby("_period").size().rename("count")
        stds = scoped.groupby("_period")[metric_column].std(ddof=1).fillna(0.0).rename("value")
        agg_df = pd.concat([counts, stds], axis=1).reset_index()
    elif aggregation == "median":
        counts = scoped.groupby("_period").size().rename("count")
        meds = scoped.groupby("_period")[metric_column].median().rename("value")
        agg_df = pd.concat([counts, meds], axis=1).reset_index()
    elif aggregation == "rate":
        counts = scoped.groupby("_period").size().rename("denominator")
        hits = scoped[scoped[metric_column] == metric_value].groupby("_period").size().rename("numerator")
        agg_df = pd.concat([counts, hits], axis=1).fillna(0).reset_index()
        agg_df["numerator"] = agg_df["numerator"].astype(int)
        agg_df["count"] = agg_df["denominator"]
        agg_df["value"] = np.where(
            agg_df["denominator"] > 0,
            (agg_df["numerator"] / agg_df["denominator"]).round(4),
            0.0,
        )

    agg_df = agg_df.rename(columns={"_period": "period"})
    agg_df["period"] = agg_df["period"].dt.strftime("%Y-%m-%d")
    agg_df["value"] = agg_df["value"].round(4)
    agg_df["metric_value"] = agg_df["value"]

    return AnalysisResult(
        request_id=request.request_id,
        operation="time_series",
        status=AnalysisStatus.OK,
        data=agg_df,
        metadata={
            "freq": freq,
            "aggregation": aggregation,
            "metric_column": metric_column,
            "metric_value": metric_value,
            "window": str(request.window),
            "total_records": len(scoped),
        },
        limitations=limitations,
        warnings=warnings,
    )


def compare_periods(
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare a metric between two periods with mathematically pooled aggregations.

    Correctly handles counts, sums, means, and rates:
    - Counts & Sums: total volume and daily average volume.
    - Means: true pooled mean (total sum / total count).
    - Rates: true pooled rate (total hits / total eligible rows).
    """
    if request.comparison_window is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="period_comparison",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="compare_periods requires request.comparison_window to be set.",
        )

    metric_column: Optional[str] = request.params.get("metric_column")
    aggregation: str = request.params.get("aggregation", "count").lower()
    metric_value: Optional[Any] = request.params.get("metric_value")
    time_column: str = request.params.get("time_column", "timestamp")

    agg_err = validate_aggregation(aggregation)
    if agg_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="period_comparison",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=agg_err,
        )

    def _calc_period_pool(window, label: str):
        scoped = apply_time_filter(primary_df, time_column, window)
        needed_cols = [c for c in request.filters if c in users_df.columns]
        if needed_cols:
            scoped = enrich_with_user_attrs(scoped, users_df, needed_cols)
        scoped = apply_filters(scoped, request.filters)

        n = len(scoped)
        days = max(1.0, window.duration_days())

        if n == 0:
            return {
                "period": label,
                "window": str(window),
                "duration_days": days,
                "count": 0,
                "metric_value": 0.0,
                "daily_avg": 0.0,
            }

        if aggregation == "count" or metric_column is None:
            m_val = n
            daily_avg = round(n / days, 4)
        elif aggregation == "sum":
            m_val = float(scoped[metric_column].sum())
            daily_avg = round(m_val / days, 4)
        elif aggregation == "mean":
            m_val = float(scoped[metric_column].mean())
            daily_avg = None  # mean has no daily additive volume
        elif aggregation == "std":
            m_val = float(scoped[metric_column].std(ddof=1)) if n > 1 else 0.0
            daily_avg = None
        elif aggregation == "median":
            m_val = float(scoped[metric_column].median())
            daily_avg = None
        elif aggregation == "rate":
            hits = (scoped[metric_column] == metric_value).sum()
            m_val = round(float(hits / n), 4) if n > 0 else 0.0
            daily_avg = None
        else:
            m_val = n
            daily_avg = round(n / days, 4)

        return {
            "period": label,
            "window": str(window),
            "duration_days": days,
            "count": n,
            "metric_value": round(m_val, 4) if m_val is not None else None,
            "daily_avg": daily_avg,
        }

    try:
        base_stats = _calc_period_pool(request.comparison_window, "baseline")
        comp_stats = _calc_period_pool(request.window, "comparison")
    except Exception as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="period_comparison",
            status=AnalysisStatus.ERROR,
            data=None,
            error_message=str(e),
        )

    b_val = base_stats["metric_value"]
    c_val = comp_stats["metric_value"]

    abs_change = round(c_val - b_val, 4) if (b_val is not None and c_val is not None) else None
    rel_change = round(abs_change / b_val, 4) if (b_val and b_val != 0 and abs_change is not None) else None

    # Change row
    delta_stats = {
        "period": "change",
        "window": "—",
        "duration_days": None,
        "count": comp_stats["count"] - base_stats["count"],
        "metric_value": abs_change,
        "daily_avg": (
            round(comp_stats["daily_avg"] - base_stats["daily_avg"], 4)
            if (comp_stats["daily_avg"] is not None and base_stats["daily_avg"] is not None)
            else None
        ),
    }

    result_df = pd.DataFrame([base_stats, comp_stats, delta_stats])
    result_df["relative_change"] = [None, None, rel_change]

    # Correct test per aggregation: z-test for rates, Welch for means. Counts/sums have no valid test here.
    p_values: Dict[str, float] = {}
    mde = None
    warnings: List[str] = []
    try:
        from analytics.statistical import compare_means, compare_proportions
        def _scoped(window):
            sc = apply_time_filter(primary_df, time_column, window)
            need = [c for c in request.filters if c in users_df.columns]
            if need:
                sc = enrich_with_user_attrs(sc, users_df, need)
            return apply_filters(sc, request.filters)
        if aggregation == "rate":
            b_s, c_s = _scoped(request.comparison_window), _scoped(request.window)
            if len(b_s) > 0 and len(c_s) > 0:
                r = compare_proportions(int((c_s[metric_column] == metric_value).sum()), len(c_s),
                                        int((b_s[metric_column] == metric_value).sum()), len(b_s))
                p_values["rate_change"], mde = r["p_value"], r["mde_80pct_power"]
                warnings += r["warnings"]
        elif aggregation == "mean" and metric_column:
            b_s, c_s = _scoped(request.comparison_window), _scoped(request.window)
            if len(b_s) > 1 and len(c_s) > 1:
                r = compare_means(c_s[metric_column], b_s[metric_column])
                if r["p_value"] is not None:
                    p_values["mean_change"] = r["p_value"]
                mde = r["mde_80pct_power"]
                warnings += r["warnings"]
        elif aggregation in ("count", "sum"):
            warnings.append("No significance test is computed for counts/sums (volume depends on window length and traffic).")
    except Exception as e:  # a failed side-test must never break the comparison itself
        warnings.append(f"Significance test skipped: {e}")

    return AnalysisResult(
        request_id=request.request_id,
        operation="period_comparison",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "baseline_window": str(request.comparison_window),
            "comparison_window": str(request.window),
            "aggregation": aggregation,
            "metric_column": metric_column,
            "metric_value": metric_value,
            "absolute_change": abs_change,
            "relative_change": rel_change,
            "p_values": p_values,
            "mde": mde,
        },
        limitations=[],
        warnings=warnings,
    )
