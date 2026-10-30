"""Seasonality-aware baseline: expected vs observed using the same weekday in prior weeks."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus, TimeWindow
from analytics.utils import (
    apply_filters,
    apply_time_filter,
    enrich_with_user_attrs,
    validate_aggregation,
)

DEFAULT_BASELINE_WEEKS = 4
DEFAULT_MIN_WEEKS = 2


def _daily(df: pd.DataFrame, time_column: str, aggregation: str, metric_column: Optional[str], metric_value) -> pd.Series:
    """Per-UTC-date aggregate. Missing days are 0 for count/sum and NaN for mean/rate."""
    d = df.copy()
    d["_day"] = d[time_column].dt.tz_convert("UTC").dt.floor("D")
    g = d.groupby("_day")
    if aggregation == "count" or metric_column is None:
        s = g.size().astype(float)
    elif aggregation == "sum":
        s = g[metric_column].sum().astype(float)
    elif aggregation == "mean":
        s = g[metric_column].mean()
    elif aggregation == "rate":
        s = g.apply(lambda x: float((x[metric_column] == metric_value).mean()), include_groups=False)
    else:
        raise ValueError(f"Unsupported aggregation '{aggregation}' for seasonal_baseline.")
    return s


def seasonal_baseline(
    primary_df: pd.DataFrame, users_df: pd.DataFrame, request: AnalysisRequest
) -> AnalysisResult:
    """Compare each day of the window with the same weekday in up to `baseline_weeks` prior weeks.

    Params: table-level metric (aggregation, metric_column, metric_value, time_column), baseline_weeks (default 4),
    min_weeks (default 2). Reports per-day expected value and a prediction range, plus a window-level p-value
    from the mean standardized residual.
    """
    def bad(msg, status=AnalysisStatus.INVALID_PARAMS):
        return AnalysisResult(request.request_id, "seasonal_baseline", status, None, error_message=msg)

    p = request.params
    aggregation = str(p.get("aggregation", "count")).lower()
    metric_column, metric_value = p.get("metric_column"), p.get("metric_value")
    time_column = p.get("time_column", "timestamp")
    weeks = int(p.get("baseline_weeks", DEFAULT_BASELINE_WEEKS))
    min_weeks = int(p.get("min_weeks", DEFAULT_MIN_WEEKS))
    err = validate_aggregation(aggregation)
    if err:
        return bad(err)
    if weeks < 1 or min_weeks < 1 or min_weeks > weeks:
        return bad("baseline_weeks must be >= min_weeks >= 1.")
    if aggregation in {"sum", "mean"} and not metric_column:
        return bad(f"aggregation='{aggregation}' requires 'metric_column'.")
    if aggregation == "rate" and (not metric_column or metric_value is None):
        return bad("aggregation='rate' requires 'metric_column' and 'metric_value'.")
    if time_column not in primary_df.columns:
        return bad(f"time_column '{time_column}' not found.")

    data_start = primary_df[time_column].min() if len(primary_df) else None
    if data_start is None or pd.isna(data_start):
        return bad("No data available.", AnalysisStatus.INSUFFICIENT_DATA)
    first_day = data_start.tz_convert("UTC").floor("D")

    lookback_start = request.window.start - timedelta(days=7 * weeks)
    scoped = apply_time_filter(primary_df, time_column, TimeWindow(lookback_start, request.window.end))
    need = [c for c in request.filters if c in users_df.columns]
    if need:
        scoped = enrich_with_user_attrs(scoped, users_df, need)
    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return bad(str(e))
    if scoped.empty:
        return bad("No data in the lookback + comparison range.", AnalysisStatus.INSUFFICIENT_DATA)
    series = _daily(scoped, time_column, aggregation, metric_column, metric_value)
    zero_fill = aggregation in ("count", "sum")

    def val(day: pd.Timestamp) -> float:
        if day in series.index:
            return float(series[day])
        return 0.0 if zero_fill else float("nan")

    rows: List[dict] = []
    zs: List[float] = []
    day = pd.Timestamp(request.window.start).tz_convert("UTC").floor("D")
    last = pd.Timestamp(request.window.end).tz_convert("UTC")
    warnings: List[str] = []
    while day < last:
        priors = [day - pd.Timedelta(days=7 * k) for k in range(1, weeks + 1)]
        priors = [d for d in priors if d >= first_day]       # only weeks the data actually covers
        pv = [val(d) for d in priors]
        pv = [v for v in pv if not math.isnan(v)]
        obs = val(day)
        row = {"date": day.strftime("%Y-%m-%d"), "weekday": day.day_name(), "observed": obs, "weeks_used": len(pv)}
        if len(pv) >= min_weeks and not math.isnan(obs):
            exp = float(np.mean(pv))
            sd = float(np.std(pv, ddof=1))
            half = float(stats.t.ppf(0.975, len(pv) - 1) * sd * math.sqrt(1 + 1 / len(pv)))
            row.update(expected=exp, lower=exp - half, upper=exp + half,
                       outside_range=bool(obs < exp - half or obs > exp + half))
            if sd > 0:
                zs.append((obs - exp) / (sd * math.sqrt(1 + 1 / len(pv))))
        else:
            row.update(expected=None, lower=None, upper=None, outside_range=None)
        rows.append(row)
        day += pd.Timedelta(days=1)

    df = pd.DataFrame(rows)
    usable = df[df["expected"].notna()]
    if usable.empty:
        return bad(
            f"No day in the window has {min_weeks}+ prior same-weekday observations within the data "
            f"(data starts {first_day.date()}); a seasonality-aware baseline needs more history.",
            AnalysisStatus.INSUFFICIENT_DATA,
        )
    if len(usable) < len(df):
        warnings.append(f"{len(df) - len(usable)} day(s) lacked {min_weeks}+ prior same-weekday observations and were skipped.")
    if usable["weeks_used"].min() < weeks:
        warnings.append(f"Fewer than the requested {weeks} prior weeks were available for some days (min {int(usable['weeks_used'].min())}).")

    # Is there a weekday pattern at all? Kruskal-Wallis across weekdays in the lookback + comparison range.
    pattern_p = None
    weekdays = pd.Series([pd.Timestamp(d).weekday() for d in series.index], index=series.index)
    groups = [series[weekdays == wd].to_numpy() for wd in range(7) if (weekdays == wd).sum() >= 2]
    if len(groups) >= 3 and len(set(np.concatenate(groups).tolist())) > 1:
        pattern_p = float(stats.kruskal(*groups).pvalue)

    p_value = None
    if zs:
        zbar = float(np.sum(zs) / math.sqrt(len(zs)))
        p_value = float(2 * (1 - stats.norm.cdf(abs(zbar))))
    else:
        warnings.append("Prior weeks had zero variance; no p-value can be computed.")

    # How much of the baseline->comparison change does seasonality predict? (daily counts/sums only)
    effect_estimate = None
    if request.comparison_window is not None and aggregation in ("count", "sum"):
        base_days = series.reindex(pd.date_range(
            pd.Timestamp(request.comparison_window.start).tz_convert("UTC").floor("D"),
            pd.Timestamp(request.comparison_window.end).tz_convert("UTC").floor("D"), freq="D", inclusive="left"))
        if base_days.notna().sum() >= 1:
            expected_shift = float(usable["expected"].mean() - base_days.fillna(0.0).mean())
            sd_pred = float(np.nanmean([max(0.0, (r["upper"] - r["expected"])) / 3.0 for _, r in usable.iterrows()]))
            effect_estimate = {
                "value": expected_shift, "method": "seasonal_same_weekday_expected_shift",
                "se": sd_pred / math.sqrt(max(1, len(usable))),
                "window": {"start": request.window.start.isoformat(), "end": request.window.end.isoformat()},
                "metric": f"{aggregation}/day",
            }

    p_values = {}
    if p_value is not None:
        p_values["seasonal_deviation"] = p_value      # LOW => NOT explained by seasonality
    if pattern_p is not None:
        p_values["weekday_pattern"] = pattern_p       # LOW => a real weekday pattern exists

    return AnalysisResult(
        request_id=request.request_id,
        operation="seasonal_baseline",
        status=AnalysisStatus.OK,
        data=df,
        metadata={
            "baseline_weeks_requested": weeks,
            "days_compared": int(len(usable)),
            "days_outside_range": int(usable["outside_range"].sum()),
            "observed_mean": float(usable["observed"].mean()),
            "expected_mean": float(usable["expected"].mean()),
            "mean_deviation": float((usable["observed"] - usable["expected"]).mean()),
            "p_values": p_values,
            "effect_estimate": effect_estimate,
            "window": str(request.window),
        },
        limitations=[
            "Expected values assume the prior weeks were themselves normal (no earlier anomaly or holiday).",
            "The window-level p-value treats days as independent and normally distributed around their weekday baseline.",
        ],
        warnings=warnings,
    )
