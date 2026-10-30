"""Segmentation and population comparison analysis.

Segments any metric by one or more categorical dimensions and
computes counts, rates, means, and period-over-period comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.utils import (
    apply_filters,
    apply_time_filter,
    check_min_sample,
    enrich_with_user_attrs,
    validate_aggregation,
    validate_dimensions,
    validate_numeric_column,
)


def segment_metric(
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compute a metric broken down by dimensions.

    Parameters in ``request.params``
    ---------------------------------
    metric_column : str, optional
        Column to aggregate (e.g. "amount", "status").
    metric_value : Any, optional
        Target value when aggregation="rate" (e.g. "completed").
    aggregation : str
        "count" (default) | "sum" | "mean" | "rate" | "std" | "median".
    time_column : str
        Timestamp column to use for window scoping (default "timestamp").
    min_segment_size : int
        Minimum segment size threshold for warnings (default 10).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    metric_column: Optional[str] = request.params.get("metric_column")
    metric_value: Optional[Any] = request.params.get("metric_value")
    aggregation: str = request.params.get("aggregation", "count").lower()
    time_column: str = request.params.get("time_column", "timestamp")
    min_seg: int = request.params.get("min_segment_size", 10)

    agg_err = validate_aggregation(aggregation)
    if agg_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=agg_err,
        )

    if aggregation == "rate" and metric_value is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="aggregation='rate' requires 'metric_value' to be set.",
        )

    # Scope to window
    if time_column in primary_df.columns:
        scoped = apply_time_filter(primary_df, time_column, request.window)
    else:
        limitations.append(f"Time column '{time_column}' not found; window filter not applied.")
        scoped = primary_df.copy()

    # Safely enrich with user dimensions/filters
    needed_user_cols = [
        c for c in list(request.filters.keys()) + request.dimensions
        if c in users_df.columns
    ]
    if needed_user_cols:
        scoped = enrich_with_user_attrs(scoped, users_df, needed_user_cols)

    # Apply filters
    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if scoped.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No data in the specified window after applying filters.",
        )

    # Validate dimensions
    dim_err = validate_dimensions(scoped, request.dimensions)
    if dim_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=dim_err,
        )

    # Validate numeric requirement for sum/mean/std/median
    if aggregation in {"sum", "mean", "std", "median"}:
        if metric_column is None:
            return AnalysisResult(
                request_id=request.request_id,
                operation="segment",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=f"aggregation='{aggregation}' requires 'metric_column'.",
            )
        num_err = validate_numeric_column(scoped, metric_column)
        if num_err:
            return AnalysisResult(
                request_id=request.request_id,
                operation="segment",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=num_err,
            )

    dimensions = request.dimensions

    def _calc_row(group: pd.DataFrame, dim_vals: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(dim_vals)
        n = len(group)
        row["count"] = n

        if aggregation == "count" or metric_column is None:
            row["value"] = n
        elif aggregation == "sum":
            row["value"] = round(float(group[metric_column].sum()), 4)
        elif aggregation == "mean":
            row["value"] = round(float(group[metric_column].mean()), 4)
        elif aggregation == "std":
            row["value"] = round(float(group[metric_column].std(ddof=1)), 4) if n > 1 else 0.0
        elif aggregation == "median":
            row["value"] = round(float(group[metric_column].median()), 4)
        elif aggregation == "rate":
            hits = int((group[metric_column] == metric_value).sum())
            row["numerator"] = hits
            row["denominator"] = n
            row["rate"] = round(hits / n, 4) if n > 0 else 0.0
            row["value"] = row["rate"]
        return row

    if not dimensions:
        res_row = _calc_row(scoped, {})
        result_df = pd.DataFrame([res_row])
    else:
        rows = []
        for keys, grp in scoped.groupby(dimensions, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            d_vals = dict(zip(dimensions, keys))
            r = _calc_row(grp, d_vals)
            warnings.extend(check_min_sample(r["count"], min_seg, context=str(d_vals)))
            rows.append(r)
        result_df = pd.DataFrame(rows)

    return AnalysisResult(
        request_id=request.request_id,
        operation="segment",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "dimensions": dimensions,
            "metric_column": metric_column,
            "metric_value": metric_value,
            "aggregation": aggregation,
            "window": str(request.window),
            "total_rows_in_window": len(scoped),
        },
        limitations=limitations,
        warnings=warnings,
    )


def compare_segments(
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare segment metrics across two time windows (baseline vs comparison)."""
    if request.comparison_window is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment_comparison",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="compare_segments requires request.comparison_window to be set.",
        )

    base_req = AnalysisRequest(
        operation="segment",
        window=request.comparison_window,
        params=request.params,
        dimensions=request.dimensions,
        filters=request.filters,
        request_id=request.request_id + "_baseline",
    )
    base_res = segment_metric(primary_df, users_df, base_req)
    comp_res = segment_metric(primary_df, users_df, request)

    if not base_res.ok() or not comp_res.ok():
        return AnalysisResult(
            request_id=request.request_id,
            operation="segment_comparison",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message=(
                f"Baseline: {base_res.error_message}; "
                f"Comparison: {comp_res.error_message}"
            ),
        )

    dims = request.dimensions
    b_df = base_res.data
    c_df = comp_res.data

    if not dims:
        # Scalar comparison
        b_val = b_df["value"].iloc[0]
        c_val = c_df["value"].iloc[0]
        abs_change = round(c_val - b_val, 4)
        rel_change = round(abs_change / b_val, 4) if b_val != 0 else None
        merged = pd.DataFrame([{
            "baseline_value": b_val,
            "comparison_value": c_val,
            "absolute_change": abs_change,
            "relative_change": rel_change,
        }])
    else:
        merged = b_df.merge(c_df, on=dims, how="outer", suffixes=("_baseline", "_comparison"))
        merged["count_baseline"] = merged["count_baseline"].fillna(0).astype(int)
        merged["count_comparison"] = merged["count_comparison"].fillna(0).astype(int)
        merged["value_baseline"] = merged["value_baseline"].fillna(0.0)
        merged["value_comparison"] = merged["value_comparison"].fillna(0.0)

        merged["absolute_change"] = (merged["value_comparison"] - merged["value_baseline"]).round(4)
        merged["relative_change"] = np.where(
            merged["value_baseline"] != 0,
            (merged["absolute_change"] / merged["value_baseline"]).round(4),
            np.nan,
        )

    return AnalysisResult(
        request_id=request.request_id,
        operation="segment_comparison",
        status=AnalysisStatus.OK,
        data=merged,
        metadata={
            "baseline_window": str(request.comparison_window),
            "comparison_window": str(request.window),
            "dimensions": dims,
            "aggregation": request.params.get("aggregation", "count"),
        },
        limitations=base_res.limitations + comp_res.limitations,
        warnings=base_res.warnings + comp_res.warnings,
    )
