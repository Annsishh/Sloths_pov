"""Spike detection: is any single period's rate abnormal compared with all other periods?

For each period (day) the rate of `metric_column == metric_value` is compared with the pooled rate of the OTHER
periods using a two-proportion z-test. Every period's p-value is returned RAW and labelled (e.g. "2026-09-20");
all of them join the investigation-wide Benjamini-Hochberg family, which is what corrects for scanning many days.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.statistical import compare_proportions
from analytics.utils import apply_filters, apply_time_filter, enrich_with_user_attrs

MIN_PERIODS = 5


def spike_check(primary_df: pd.DataFrame, users_df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
    def bad(msg, status=AnalysisStatus.INVALID_PARAMS):
        return AnalysisResult(request.request_id, "spike_check", status, None, error_message=msg)

    p = request.params
    metric_column, metric_value = p.get("metric_column"), p.get("metric_value")
    time_column = p.get("time_column", "timestamp")
    if not metric_column or metric_value is None:
        return bad("Parameters 'metric_column' and 'metric_value' are required.")
    if time_column not in primary_df.columns or metric_column not in primary_df.columns:
        return bad("time_column or metric_column not found in the table.")

    scoped = apply_time_filter(primary_df, time_column, request.window)
    need = [c for c in request.filters if c in users_df.columns]
    if need:
        scoped = enrich_with_user_attrs(scoped, users_df, need)
    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return bad(str(e))
    if scoped.empty:
        return bad("No data in the window.", AnalysisStatus.INSUFFICIENT_DATA)

    day = scoped[time_column].dt.tz_convert("UTC").dt.floor("D")
    g = scoped.assign(_hit=(scoped[metric_column] == metric_value)).groupby(day)["_hit"].agg(["size", "sum"])
    g.columns = ["n", "k"]
    if len(g) < MIN_PERIODS:
        return bad(f"Only {len(g)} period(s) in the window; a spike scan needs at least {MIN_PERIODS} "
                   "(use lookback_days to extend the window).", AnalysisStatus.INSUFFICIENT_DATA)

    total_n, total_k = int(g["n"].sum()), int(g["k"].sum())
    rows: List[Dict] = []
    p_values: Dict[str, float] = {}
    for d, r in g.iterrows():
        n, k = int(r["n"]), int(r["k"])
        rest_n, rest_k = total_n - n, total_k - k
        pv = None
        if n > 0 and rest_n > 0:
            pv = compare_proportions(k, n, rest_k, rest_n)["p_value"]
            p_values[d.strftime("%Y-%m-%d")] = pv
        rows.append({"period": d.strftime("%Y-%m-%d"), "n": n, "hits": k, "rate": round(k / n, 4) if n else None,
                     "rate_other_periods": round(rest_k / rest_n, 4) if rest_n else None, "p_value": pv})
    df = pd.DataFrame(rows)
    top = df.sort_values("rate", ascending=False).iloc[0]
    return AnalysisResult(
        request_id=request.request_id, operation="spike_check", status=AnalysisStatus.OK, data=df,
        metadata={"top_period": top["period"], "top_rate": float(top["rate"]), "overall_rate": total_k / total_n,
                  "periods": len(df), "p_values": p_values, "window": str(request.window)},
        limitations=["One test per period, each compared with the pooled other periods; correction for scanning many periods "
                     "happens in the investigation-wide BH family, not here.",
                     "A spike shows WHEN something happened, not why."],
        warnings=[],
    )
