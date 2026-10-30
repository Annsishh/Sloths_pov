"""Tests for funnel conversion and chronological step ordering."""

import pytest
import pandas as pd

from analytics.funnel import compare_funnel, compute_funnel
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow


def test_funnel_chronological_ordering():
    # Session 1: page_view (t=10) -> checkout_started (t=20) -> order_completed (t=30) -> CONVERTS FULLY
    # Session 2: order_completed (t=10) -> checkout_started (t=20) -> page_view (t=30) (REVERSED) -> SHOULD NOT CONVERT PAST STEP 0
    # Session 3: page_view (t=10) -> checkout_started (t=20) -> abandons -> CONVERTS STEP 0 & 1, DROPS AT STEP 2
    users_df = pd.DataFrame([
        {"user_id": "u1", "browser": "Chrome"},
        {"user_id": "u2", "browser": "Safari"},
        {"user_id": "u3", "browser": "Chrome"},
    ])

    events_df = pd.DataFrame([
        # Session 1
        {"event_id": "e1", "session_id": "s1", "user_id": "u1", "event_name": "page_view", "timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"event_id": "e2", "session_id": "s1", "user_id": "u1", "event_name": "checkout_started", "timestamp": pd.Timestamp("2026-09-01T10:05:00Z")},
        {"event_id": "e3", "session_id": "s1", "user_id": "u1", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-01T10:10:00Z")},

        # Session 2 (Reverse order)
        {"event_id": "e4", "session_id": "s2", "user_id": "u2", "event_name": "order_completed", "timestamp": pd.Timestamp("2026-09-01T11:00:00Z")},
        {"event_id": "e5", "session_id": "s2", "user_id": "u2", "event_name": "checkout_started", "timestamp": pd.Timestamp("2026-09-01T11:05:00Z")},
        {"event_id": "e6", "session_id": "s2", "user_id": "u2", "event_name": "page_view", "timestamp": pd.Timestamp("2026-09-01T11:10:00Z")},

        # Session 3 (Drop off at checkout)
        {"event_id": "e7", "session_id": "s3", "user_id": "u3", "event_name": "page_view", "timestamp": pd.Timestamp("2026-09-01T12:00:00Z")},
        {"event_id": "e8", "session_id": "s3", "user_id": "u3", "event_name": "checkout_started", "timestamp": pd.Timestamp("2026-09-01T12:05:00Z")},
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
    req = AnalysisRequest(
        operation="funnel",
        window=window,
        params={"steps": ["page_view", "checkout_started", "order_completed"]},
    )

    res = compute_funnel(events_df, users_df, req)
    assert res.ok()
    df = res.data

    # Step 0 (page_view): s1, s2 (at t=11:10), s3 -> all 3 had page_view
    assert df.loc[0, "count"] == 3
    assert df.loc[0, "conversion_from_top"] == 1.0

    # Step 1 (checkout_started after page_view):
    # s1: page_view(10:00) -> checkout_started(10:05) -> YES
    # s2: page_view(11:10), checkout_started was at 11:05 (BEFORE page_view) -> NO
    # s3: page_view(12:00) -> checkout_started(12:05) -> YES
    # Total step 1 count: 2
    assert df.loc[1, "count"] == 2
    assert df.loc[1, "conversion_from_top"] == pytest.approx(2 / 3, abs=1e-4)

    # Step 2 (order_completed after checkout_started):
    # s1: order_completed(10:10) after checkout_started(10:05) -> YES
    # s3: no order_completed -> NO
    # Total step 2 count: 1
    assert df.loc[2, "count"] == 1
    assert df.loc[2, "conversion_from_top"] == pytest.approx(1 / 3, abs=1e-4)
    assert df.loc[2, "conversion_from_prev"] == 0.50
    assert df.loc[2, "drop_off_from_prev"] == 0.50
    assert df.loc[2, "drop_count_from_prev"] == 1


def test_funnel_invalid_params():
    users_df = pd.DataFrame()
    events_df = pd.DataFrame()
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
    req = AnalysisRequest(
        operation="funnel",
        window=window,
        params={"steps": ["only_one_step"]},
    )
    res = compute_funnel(events_df, users_df, req)
    assert res.status == AnalysisStatus.INVALID_PARAMS

