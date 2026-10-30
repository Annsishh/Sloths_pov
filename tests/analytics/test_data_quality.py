"""Tests for data quality and event tracking investigation."""

import pandas as pd
import pytest

from analytics.data_quality import (
    duplicate_event_check,
    event_volume_check,
    tracking_gap_check,
)
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow


@pytest.fixture
def tracking_failure_sample():
    users_df = pd.DataFrame([
        {"user_id": f"u_chrome_{i}", "browser": "Chrome"} for i in range(10)
    ] + [
        {"user_id": f"u_safari_{i}", "browser": "Safari"} for i in range(10)
    ])

    orders = []
    events = []

    # Chrome: 10 backend orders, all 10 have telemetry events
    for i in range(10):
        uid = f"u_chrome_{i}"
        t = pd.Timestamp(f"2026-09-05T12:{i:02d}:00Z")
        orders.append({"order_id": f"ord_chr_{i}", "user_id": uid, "status": "completed", "timestamp": t})
        events.append({"event_id": f"ev_chr_{i}", "user_id": uid, "event_name": "order_completed", "timestamp": t + pd.Timedelta(seconds=2)})

    # Safari: 10 backend orders, but 0 telemetry events (tracking failure!)
    for i in range(10):
        uid = f"u_safari_{i}"
        t = pd.Timestamp(f"2026-09-05T14:{i:02d}:00Z")
        orders.append({"order_id": f"ord_saf_{i}", "user_id": uid, "status": "completed", "timestamp": t})

    orders_df = pd.DataFrame(orders)
    events_df = pd.DataFrame(events)

    return events_df, orders_df, users_df


def test_tracking_gap_check_detects_safari_failure(tracking_failure_sample):
    events_df, orders_df, users_df = tracking_failure_sample
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")

    req = AnalysisRequest(
        operation="tracking_gap_check",
        window=window,
        params={"breakdown": ["browser"]},
    )

    res = tracking_gap_check(events_df, orders_df, users_df, req)
    assert res.ok()
    df = res.data

    chrome_row = df[df["browser"] == "Chrome"].iloc[0]
    safari_row = df[df["browser"] == "Safari"].iloc[0]

    assert chrome_row["backend_orders"] == 10
    assert chrome_row["telemetry_events"] == 10
    assert chrome_row["matched_orders"] == 10
    assert chrome_row["match_rate"] == 1.0

    assert safari_row["backend_orders"] == 10
    assert safari_row["telemetry_events"] == 0
    assert safari_row["matched_orders"] == 0
    assert safari_row["match_rate"] == 0.0

    # Severe tracking warning should be triggered for Safari
    assert any("Severe tracking failure detected in {'browser': 'Safari'}" in w for w in res.warnings)


def test_event_volume_check(tracking_failure_sample):
    events_df, _, users_df = tracking_failure_sample
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")

    req = AnalysisRequest(
        operation="event_volume_check",
        window=window,
        params={"freq": "D"},
    )
    res = event_volume_check(events_df, users_df, req)
    assert res.ok()
    df = res.data
    assert len(df) == 1
    assert df["event_count"].iloc[0] == 10


def test_duplicate_event_check():
    events = [
        {"session_id": "s1", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T10:00:00Z")},
        {"session_id": "s1", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T10:01:00Z")},
        {"session_id": "s1", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T10:02:00Z")},
        {"session_id": "s1", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T10:03:00Z")},  # 4 > 3
        {"session_id": "s2", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T11:00:00Z")},
    ]
    events_df = pd.DataFrame(events)
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")

    req = AnalysisRequest(
        operation="duplicate_event_check",
        window=window,
        params={"event_name": "order_completed", "max_per_session": 3},
    )
    res = duplicate_event_check(events_df, req)
    assert res.ok()
    df = res.data
    assert len(df) == 1
    assert df["session_id"].iloc[0] == "s1"
    assert df["event_count"].iloc[0] == 4
    assert len(res.warnings) == 1


def test_tracking_gap_1to1_matching_nearby_orders():
    # 1 user places TWO orders within grace period, but telemetry only fires ONE event
    users_df = pd.DataFrame([{"user_id": "u_dual", "browser": "Chrome"}])
    orders_df = pd.DataFrame([
        {"order_id": "ord_1", "user_id": "u_dual", "status": "completed", "timestamp": pd.Timestamp("2026-09-05T12:00:00Z")},
        {"order_id": "ord_2", "user_id": "u_dual", "status": "completed", "timestamp": pd.Timestamp("2026-09-05T12:00:05Z")},
    ])
    events_df = pd.DataFrame([
        {"event_id": "ev_1", "user_id": "u_dual", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T12:00:02Z")},
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="tracking_gap_check",
        window=window,
        params={"breakdown": []},
    )
    res = tracking_gap_check(events_df, orders_df, users_df, req)
    assert res.ok()
    df = res.data

    row = df.iloc[0]
    assert row["backend_orders"] == 2
    assert row["telemetry_events"] == 1
    # Crucial 1-to-1 assertion: 1 event cannot match both orders!
    assert row["matched_orders"] == 1
    assert row["unmatched_orders"] == 1
    assert row["match_rate"] == 0.5


def test_tracking_gap_duplicate_events_and_inflation():
    # 1 order, but 3 duplicate telemetry events
    users_df = pd.DataFrame([{"user_id": "u_dup", "browser": "Chrome"}])
    orders_df = pd.DataFrame([
        {"order_id": "ord_1", "user_id": "u_dup", "status": "completed", "timestamp": pd.Timestamp("2026-09-05T12:00:00Z")},
    ])
    events_df = pd.DataFrame([
        {"event_id": "ev_1", "user_id": "u_dup", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T12:00:01Z")},
        {"event_id": "ev_2", "user_id": "u_dup", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T12:00:02Z")},
        {"event_id": "ev_3", "user_id": "u_dup", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-05T12:00:03Z")},
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="tracking_gap_check",
        window=window,
        params={"breakdown": []},
    )
    res = tracking_gap_check(events_df, orders_df, users_df, req)
    assert res.ok()
    row = res.data.iloc[0]

    assert row["backend_orders"] == 1
    assert row["telemetry_events"] == 3
    assert row["matched_orders"] == 1
    assert row["unmatched_orders"] == 0
    assert row["tracking_ratio"] == 3.0
    assert row["match_rate"] == 1.0


def test_tracking_gap_zero_event_and_zero_order_segments():
    users_df = pd.DataFrame([
        {"user_id": "u_zero_ev", "browser": "BrowserA"},
        {"user_id": "u_zero_ord", "browser": "BrowserB"},
    ])
    # BrowserA has 5 orders, 0 events
    orders_df = pd.DataFrame([
        {"order_id": f"o_{i}", "user_id": "u_zero_ev", "status": "completed", "timestamp": pd.Timestamp(f"2026-09-05T12:{i:02d}:00Z")}
        for i in range(5)
    ])
    # BrowserB has 0 orders, 2 events
    events_df = pd.DataFrame([
        {"event_id": f"e_{i}", "user_id": "u_zero_ord", "event_name": "order_completed", "timestamp": pd.Timestamp(f"2026-09-05T13:{i:02d}:00Z")}
        for i in range(2)
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="tracking_gap_check",
        window=window,
        params={"breakdown": ["browser"]},
    )
    res = tracking_gap_check(events_df, orders_df, users_df, req)
    assert res.ok()
    df = res.data

    row_a = df[df["browser"] == "BrowserA"].iloc[0]
    assert row_a["backend_orders"] == 5
    assert row_a["telemetry_events"] == 0
    assert row_a["matched_orders"] == 0
    assert row_a["match_rate"] == 0.0
    assert row_a["tracking_ratio"] == 0.0

    row_b = df[df["browser"] == "BrowserB"].iloc[0]
    assert row_b["backend_orders"] == 0
    assert row_b["telemetry_events"] == 2
    assert row_b["matched_orders"] == 0
    assert pd.isna(row_b["match_rate"])
    assert pd.isna(row_b["tracking_ratio"])

