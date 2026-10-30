"""Tests for feedback analysis and error analysis."""

import pandas as pd
import pytest

from analytics.feedback_analysis import analyze_errors, analyze_feedback
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow


@pytest.fixture
def sample_feedback_and_orders():
    users_df = pd.DataFrame([
        {"user_id": f"u{i}", "device_type": "mobile" if i % 2 == 0 else "desktop"}
        for i in range(10)
    ])

    feedback_records = [
        {"feedback_id": "f1", "user_id": "u0", "rating": 1, "category": "billing", "timestamp": pd.Timestamp("2026-09-02T10:00:00Z")},
        {"feedback_id": "f2", "user_id": "u1", "rating": 5, "category": "product", "timestamp": pd.Timestamp("2026-09-02T11:00:00Z")},
        {"feedback_id": "f3", "user_id": "u2", "rating": 2, "category": "general", "timestamp": pd.Timestamp("2026-09-03T12:00:00Z")},
        {"feedback_id": "f4", "user_id": "u3", "rating": 4, "category": "billing", "timestamp": pd.Timestamp("2026-09-03T13:00:00Z")},
    ]
    feedback_df = pd.DataFrame(feedback_records)

    order_records = [
        {"order_id": "o1", "user_id": "u0", "status": "completed", "failure_reason": None, "timestamp": pd.Timestamp("2026-09-02T10:00:00Z")},
        {"order_id": "o2", "user_id": "u1", "status": "failed", "failure_reason": "card_declined", "timestamp": pd.Timestamp("2026-09-02T11:00:00Z")},
        {"order_id": "o3", "user_id": "u2", "status": "failed", "failure_reason": "card_declined", "timestamp": pd.Timestamp("2026-09-03T12:00:00Z")},
        {"order_id": "o4", "user_id": "u3", "status": "failed", "failure_reason": "gateway_timeout", "timestamp": pd.Timestamp("2026-09-03T13:00:00Z")},
    ]
    orders_df = pd.DataFrame(order_records)

    return feedback_df, orders_df, users_df


def test_analyze_feedback_summary(sample_feedback_and_orders):
    feedback_df, _, users_df = sample_feedback_and_orders
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")

    req = AnalysisRequest(
        operation="feedback_analysis",
        window=window,
        params={},
    )
    res = analyze_feedback(feedback_df, users_df, req)
    assert res.ok()
    df = res.data

    assert len(df) == 1
    row = df.iloc[0]
    assert row["total_feedback"] == 4
    # Ratings: 1, 5, 2, 4 -> mean 3.0
    assert row["avg_rating"] == 3.0
    # Negative (<=2): 1, 2 -> 2 out of 4 (50%)
    assert row["negative_feedback_count"] == 2
    assert row["negative_rate"] == 0.50
    assert row["billing_category_count"] == 2


def test_analyze_errors_breakdown(sample_feedback_and_orders):
    _, orders_df, users_df = sample_feedback_and_orders
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")

    req = AnalysisRequest(
        operation="error_analysis",
        window=window,
        params={"breakdown": ["device_type"]},
    )
    res = analyze_errors(orders_df, users_df, req)
    assert res.ok()
    df = res.data

    # Should report failed orders count and failure rate
    assert res.metadata.get("total_failed_orders") == 3
    assert res.metadata.get("overall_failure_rate") == 0.75
