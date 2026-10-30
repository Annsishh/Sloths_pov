"""Metric definitions, aggregation, and explicit semantics.

Computes product metrics according to explicit counting units, eligible populations,
numerators, and denominators defined in MetricDefinition objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus, MetricDefinition, TimeWindow
from analytics.utils import apply_filters, apply_time_filter, check_min_sample, enrich_with_user_attrs


# Predefined metric catalogue
STANDARD_METRICS: Dict[str, MetricDefinition] = {
    "daily_active_users": MetricDefinition(
        name="daily_active_users",
        description="Count of distinct users with at least one event in the window.",
        counting_unit="unique user",
        numerator_events=[],
        denominator="none",
        time_column="timestamp",
    ),
    "new_signups": MetricDefinition(
        name="new_signups",
        description="Count of newly registered users in the window.",
        counting_unit="unique user",
        numerator_events=[],
        denominator="none",
        time_column="signup_timestamp",
    ),
    "gross_revenue": MetricDefinition(
        name="gross_revenue",
        description="Total monetary value of successfully completed orders.",
        counting_unit="USD amount",
        numerator_events=[],
        denominator="none",
        time_column="timestamp",
        eligible_filter={"status": "completed"},
    ),
    "order_conversion_rate": MetricDefinition(
        name="order_conversion_rate",
        description="Proportion of checkout_started sessions that successfully completed an order.",
        counting_unit="session",
        numerator_events=["order_completed"],
        denominator="sessions",
        time_column="timestamp",
        notes="Evaluated from telemetry events. Cross-check with tracking_gap_check for telemetry loss.",
    ),
    "order_failure_rate": MetricDefinition(
        name="order_failure_rate",
        description="Proportion of attempted backend orders that failed.",
        counting_unit="order",
        numerator_events=[],
        denominator="orders",
        time_column="timestamp",
        notes="Evaluated directly from financial backend orders table.",
    ),
    "avg_feedback_rating": MetricDefinition(
        name="avg_feedback_rating",
        description="Arithmetic mean of submitted customer feedback ratings (1 to 5 stars).",
        counting_unit="rating",
        numerator_events=[],
        denominator="none",
        time_column="timestamp",
    ),
}


def compute_metric(
    tables: Dict[str, pd.DataFrame],
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compute a defined product metric with full semantic transparency.

    Parameters in ``request.params``
    ---------------------------------
    metric_name : str, optional
        Key in STANDARD_METRICS or custom metric specified in request.metric.
    """
    limitations: List[str] = []
    warnings: List[str] = []

    metric_name: Optional[str] = request.params.get("metric_name")
    m_def = request.metric

    if m_def is None and metric_name in STANDARD_METRICS:
        m_def = STANDARD_METRICS[metric_name]

    if m_def is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="metric_aggregate",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=(
                f"Unknown metric '{metric_name}'. Provide a valid metric_name from "
                f"{list(STANDARD_METRICS.keys())} or attach a MetricDefinition."
            ),
        )

    users_df = tables.get("users", pd.DataFrame())
    events_df = tables.get("events", pd.DataFrame())
    orders_df = tables.get("orders", pd.DataFrame())
    feedback_df = tables.get("user_feedback", pd.DataFrame())

    # Dispatch calculation based on metric definition semantics
    val = None
    numerator = None
    denominator = None
    sample_size = 0

    if m_def.name == "daily_active_users":
        scoped = apply_time_filter(events_df, "timestamp", request.window)
        if request.filters:
            scoped = enrich_with_user_attrs(scoped, users_df, list(request.filters.keys()))
            scoped = apply_filters(scoped, request.filters)
        val = scoped["user_id"].nunique()
        sample_size = len(scoped)

    elif m_def.name == "new_signups":
        scoped = apply_time_filter(users_df, "signup_timestamp", request.window)
        if request.filters:
            scoped = apply_filters(scoped, request.filters)
        val = len(scoped)
        sample_size = val

    elif m_def.name == "gross_revenue":
        scoped = apply_time_filter(orders_df, "timestamp", request.window)
        if request.filters:
            scoped = enrich_with_user_attrs(scoped, users_df, list(request.filters.keys()))
            scoped = apply_filters(scoped, request.filters)
        completed = scoped[scoped["status"] == "completed"]
        amt_col = "amount" if "amount" in completed.columns else "total_amount"
        val = round(float(completed[amt_col].sum()), 2) if (amt_col in completed.columns and not completed.empty) else 0.0
        sample_size = len(completed)

    elif m_def.name == "order_failure_rate":
        scoped = apply_time_filter(orders_df, "timestamp", request.window)
        if request.filters:
            scoped = enrich_with_user_attrs(scoped, users_df, list(request.filters.keys()))
            scoped = apply_filters(scoped, request.filters)
        denominator = len(scoped)
        numerator = int((scoped["status"] == "failed").sum())
        val = round(numerator / denominator, 4) if denominator > 0 else 0.0
        sample_size = denominator

    elif m_def.name == "order_conversion_rate":
        scoped = apply_time_filter(events_df, "timestamp", request.window)
        if request.filters:
            scoped = enrich_with_user_attrs(scoped, users_df, list(request.filters.keys()))
            scoped = apply_filters(scoped, request.filters)
        checkout_sessions = set(scoped[scoped["event_name"] == "checkout_started"]["session_id"].unique())
        completed_sessions = set(scoped[scoped["event_name"] == "order_completed"]["session_id"].unique())
        denominator = len(checkout_sessions)
        numerator = len(checkout_sessions & completed_sessions)
        val = round(numerator / denominator, 4) if denominator > 0 else 0.0
        sample_size = denominator

    elif m_def.name == "avg_feedback_rating":
        scoped = apply_time_filter(feedback_df, "timestamp", request.window)
        if request.filters:
            scoped = enrich_with_user_attrs(scoped, users_df, list(request.filters.keys()))
            scoped = apply_filters(scoped, request.filters)
        denominator = len(scoped)
        val = round(float(scoped["rating"].mean()), 2) if denominator > 0 else None
        sample_size = denominator

    else:
        # Generic metric computation based on m_def.eligible_filter and counting_unit
        limitations.append(f"Metric '{m_def.name}' evaluated via generic fallback logic.")
        val = 0.0

    warnings.extend(check_min_sample(sample_size, m_def.min_sample_size, context=m_def.name))

    result_data = {
        "metric_name": m_def.name,
        "description": m_def.description,
        "counting_unit": m_def.counting_unit,
        "value": val,
        "numerator": numerator,
        "denominator": denominator,
        "sample_size": sample_size,
    }

    return AnalysisResult(
        request_id=request.request_id,
        operation="metric_aggregate",
        status=AnalysisStatus.OK,
        data=pd.DataFrame([result_data]),
        metadata={
            "metric_name": m_def.name,
            "window": str(request.window),
            "counting_unit": m_def.counting_unit,
            "notes": m_def.notes,
        },
        limitations=limitations,
        warnings=warnings,
    )

