"""Error analysis and customer feedback analysis.

Provides diagnostic analytical operations for:
1. Feedback analysis: Rating distributions, category breakdowns (billing, product, general),
   negative feedback surges, and correlation with user attributes.
2. Error analysis: Breakdown of failed transactions by failure_reason, failure rates
   segmented by device, browser, and app version, and failure timeline tracking.
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
    validate_dimensions,
)


def analyze_feedback(
    feedback_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Analyze customer ratings, categories, and negative feedback patterns.

    Parameters in ``request.params``
    ---------------------------------
    breakdown : List[str], optional
        User dimensions to segment feedback by (e.g. ["device_type"]).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    breakdown: List[str] = request.params.get("breakdown", [])

    scoped = apply_time_filter(feedback_df, "timestamp", request.window)

    needed_user_cols = [c for c in list(request.filters.keys()) + breakdown if c in users_df.columns]
    if needed_user_cols:
        scoped = enrich_with_user_attrs(scoped, users_df, needed_user_cols)

    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="feedback_analysis",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if scoped.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="feedback_analysis",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No feedback records found in the specified window.",
        )

    dim_err = validate_dimensions(scoped, breakdown)
    if dim_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="feedback_analysis",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=dim_err,
        )

    def _calc_summary(grp: pd.DataFrame) -> Dict[str, Any]:
        n = len(grp)
        avg_r = float(grp["rating"].mean())
        std_r = float(grp["rating"].std(ddof=1)) if n > 1 else 0.0

        low_ratings = int((grp["rating"] <= 2).sum())
        high_ratings = int((grp["rating"] >= 4).sum())

        cat_counts = grp["category"].value_counts().to_dict()

        return {
            "total_feedback": n,
            "avg_rating": round(avg_r, 2),
            "std_rating": round(std_r, 2),
            "negative_feedback_count": low_ratings,
            "negative_rate": round(low_ratings / n, 4) if n > 0 else 0.0,
            "positive_rate": round(high_ratings / n, 4) if n > 0 else 0.0,
            "billing_category_count": cat_counts.get("billing", 0),
            "product_category_count": cat_counts.get("product", 0),
            "general_category_count": cat_counts.get("general", 0),
        }

    if not breakdown:
        overall = _calc_summary(scoped)
        result_df = pd.DataFrame([overall])
    else:
        rows = []
        for keys, grp in scoped.groupby(breakdown, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            d_vals = dict(zip(breakdown, keys))
            s = _calc_summary(grp)
            d_vals.update(s)
            warnings.extend(check_min_sample(s["total_feedback"], 5, context=str(keys)))
            rows.append(d_vals)
        result_df = pd.DataFrame(rows)

    return AnalysisResult(
        request_id=request.request_id,
        operation="feedback_analysis",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "breakdown": breakdown,
            "window": str(request.window),
            "total_records": len(scoped),
        },
        limitations=limitations,
        warnings=warnings,
    )


def analyze_errors(
    orders_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Analyze failed transactions, failure reasons, and error distributions.

    Parameters in ``request.params``
    ---------------------------------
    breakdown : List[str], optional
        Dimensions to slice error rates by (e.g. ["device_type", "failure_reason"]).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    breakdown: List[str] = request.params.get("breakdown", ["failure_reason"])

    scoped = apply_time_filter(orders_df, "timestamp", request.window)

    needed_user_cols = [c for c in list(request.filters.keys()) + breakdown if c in users_df.columns]
    if needed_user_cols:
        scoped = enrich_with_user_attrs(scoped, users_df, needed_user_cols)

    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="error_analysis",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if scoped.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="error_analysis",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No order records found in the specified window.",
        )

    dim_err = validate_dimensions(scoped, breakdown)
    if dim_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="error_analysis",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=dim_err,
        )

    total_orders = len(scoped)
    failed_orders = scoped[scoped["status"] == "failed"].copy()
    total_failures = len(failed_orders)

    if not breakdown:
        overall = {
            "total_orders": total_orders,
            "failed_orders": total_failures,
            "failure_rate": round(total_failures / total_orders, 4) if total_orders > 0 else 0.0,
        }
        result_df = pd.DataFrame([overall])
    else:
        # Group by requested breakdown
        # If failure_reason is in breakdown, we slice failed orders
        if "failure_reason" in breakdown:
            grp_cols = breakdown
            target_df = failed_orders if not failed_orders.empty else pd.DataFrame(columns=scoped.columns)
            if target_df.empty:
                result_df = pd.DataFrame(columns=breakdown + ["failed_count", "share_of_failures"])
            else:
                counts = target_df.groupby(grp_cols, sort=True).size().reset_index(name="failed_count")
                counts["share_of_failures"] = (counts["failed_count"] / total_failures).round(4) if total_failures > 0 else 0.0
                result_df = counts
        else:
            # Slicing by user dimension (e.g. device_type) -> report both total orders, failed orders, and rate
            rows = []
            for keys, grp in scoped.groupby(breakdown, sort=True):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                d_vals = dict(zip(breakdown, keys))
                n_grp = len(grp)
                n_fail = int((grp["status"] == "failed").sum())
                rate = round(n_fail / n_grp, 4) if n_grp > 0 else 0.0
                d_vals.update({
                    "total_orders": n_grp,
                    "failed_orders": n_fail,
                    "failure_rate": rate,
                })
                rows.append(d_vals)
            result_df = pd.DataFrame(rows)

    return AnalysisResult(
        request_id=request.request_id,
        operation="error_analysis",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "breakdown": breakdown,
            "window": str(request.window),
            "total_orders": total_orders,
            "total_failed_orders": total_failures,
            "overall_failure_rate": round(total_failures / total_orders, 4) if total_orders > 0 else 0.0,
        },
        limitations=limitations,
        warnings=warnings,
    )

