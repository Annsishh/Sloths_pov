"""Tests for segmentation and population comparison."""

import pytest
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.segmentation import compare_segments, segment_metric


def test_segment_metric_rate_and_count():
    users_df = pd.DataFrame([
        {"user_id": "u1", "device_type": "mobile", "country": "US"},
        {"user_id": "u2", "device_type": "mobile", "country": "US"},
        {"user_id": "u3", "device_type": "desktop", "country": "US"},
        {"user_id": "u4", "device_type": "desktop", "country": "US"},
    ])
    orders_df = pd.DataFrame([
        {"order_id": "o1", "user_id": "u1", "status": "completed", "amount": 100.0, "timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"order_id": "o2", "user_id": "u2", "status": "failed", "amount": 50.0, "timestamp": pd.Timestamp("2026-09-01T11:00:00Z")},
        {"order_id": "o3", "user_id": "u3", "status": "completed", "amount": 200.0, "timestamp": pd.Timestamp("2026-09-01T12:00:00Z")},
        {"order_id": "o4", "user_id": "u4", "status": "completed", "amount": 150.0, "timestamp": pd.Timestamp("2026-09-01T13:00:00Z")},
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
    req = AnalysisRequest(
        operation="segment",
        window=window,
        dimensions=["device_type"],
        params={"metric_column": "status", "metric_value": "completed", "aggregation": "rate"},
    )

    res = segment_metric(orders_df, users_df, req)
    assert res.ok()
    df = res.data

    desktop_row = df[df["device_type"] == "desktop"].iloc[0]
    assert desktop_row["denominator"] == 2
    assert desktop_row["numerator"] == 2
    assert desktop_row["rate"] == 1.0

    mobile_row = df[df["device_type"] == "mobile"].iloc[0]
    assert mobile_row["denominator"] == 2
    assert mobile_row["numerator"] == 1
    assert mobile_row["rate"] == 0.5


def test_segment_metric_no_column_collision():
    # If events table already has 'device_type' and users table also has 'device_type',
    # enrich_with_user_attrs must NOT create 'device_type_x' / 'device_type_y'!
    users_df = pd.DataFrame([{"user_id": "u1", "device_type": "mobile", "plan_tier": "pro"}])
    events_df = pd.DataFrame([
        {"event_id": "e1", "user_id": "u1", "device_type": "mobile", "timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
    req = AnalysisRequest(
        operation="segment",
        window=window,
        dimensions=["device_type", "plan_tier"],
        params={"aggregation": "count"},
    )

    res = segment_metric(events_df, users_df, req)
    assert res.ok()
    df = res.data
    assert "device_type" in df.columns
    assert "plan_tier" in df.columns
    assert "device_type_x" not in df.columns
    assert "device_type_y" not in df.columns

