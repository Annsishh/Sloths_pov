"""Tests for retention and cohort analysis with controlled datasets and exact expected results."""

from datetime import datetime, timezone
import pytest
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.retention import compute_retention, compare_retention


def test_retention_exact_calculation():
    # 3 users sign up on Sept 1:
    # u1 returns on Day 1 (Sept 2) and Day 3 (Sept 4)
    # u2 returns only on Day 1 (Sept 2)
    # u3 does not return
    users_df = pd.DataFrame([
        {"user_id": "u1", "signup_timestamp": pd.Timestamp("2026-09-01T08:00:00Z")},
        {"user_id": "u2", "signup_timestamp": pd.Timestamp("2026-09-01T12:00:00Z")},
        {"user_id": "u3", "signup_timestamp": pd.Timestamp("2026-09-01T15:00:00Z")},
    ])
    events_df = pd.DataFrame([
        # Day 1: u1 at 24h+4h = 28h from signup
        {"event_id": "e1", "user_id": "u1", "timestamp": pd.Timestamp("2026-09-02T12:00:00Z"), "event_name": "page_view"},
        # Day 1: u2 at 24h+1h = 25h from signup
        {"event_id": "e2", "user_id": "u2", "timestamp": pd.Timestamp("2026-09-02T13:00:00Z"), "event_name": "page_view"},
        # Day 3: u1 at 72h+1h = 73h from signup
        {"event_id": "e3", "user_id": "u1", "timestamp": pd.Timestamp("2026-09-04T09:00:00Z"), "event_name": "page_view"},
    ])

    # Observation window extends to Sept 10
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
    req = AnalysisRequest(
        operation="retention",
        window=window,
        params={
            "retention_days": [1, 3],
            "cohort_period": "day",
            "max_observation_date": "2026-09-10T00:00:00Z",
        },
    )

    res = compute_retention(users_df, events_df, req)
    assert res.ok()
    df = res.data
    assert len(df) == 1
    row = df.iloc[0]
    assert row["cohort"] == "2026-09-01"
    assert row["cohort_size"] == 3
    # Day 1: u1 and u2 retained -> 2 / 3 = 0.6667
    assert row["eligible_d1"] == 3
    assert row["retained_d1"] == 2
    assert row["retention_d1"] == pytest.approx(2 / 3, abs=1e-4)

    # Day 3: only u1 retained -> 1 / 3 = 0.3333
    assert row["eligible_d3"] == 3
    assert row["retained_d3"] == 1
    assert row["retention_d3"] == pytest.approx(1 / 3, abs=1e-4)


def test_retention_incomplete_observation_window():
    # User u1 signs up on Sept 1 -> complete for Day 1 and Day 7
    # User u2 signs up on Sept 8 -> complete for Day 1 (by Sept 10), but INCOMPLETE for Day 7 (needs Sept 16)
    users_df = pd.DataFrame([
        {"user_id": "u1", "signup_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"user_id": "u2", "signup_timestamp": pd.Timestamp("2026-09-08T10:00:00Z")},
    ])
    events_df = pd.DataFrame([
        {"event_id": "e1", "user_id": "u1", "timestamp": pd.Timestamp("2026-09-02T12:00:00Z"), "event_name": "page_view"},
        {"event_id": "e2", "user_id": "u2", "timestamp": pd.Timestamp("2026-09-09T12:00:00Z"), "event_name": "page_view"},
    ])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-11T00:00:00Z")
    req = AnalysisRequest(
        operation="retention",
        window=window,
        params={
            "retention_days": [1, 7],
            "cohort_period": "day",
            "max_observation_date": "2026-09-11T00:00:00Z",
        },
    )

    res = compute_retention(users_df, events_df, req)
    assert res.ok()
    df = res.data

    # Cohort 2026-09-08
    c2 = df[df["cohort"] == "2026-09-08"].iloc[0]
    # Day 1 is complete: eligible=1, retained=1
    assert c2["eligible_d1"] == 1
    assert c2["retention_d1"] == 1.0
    # Day 7 is incomplete: eligible=0, retention=None
    assert c2["eligible_d7"] == 0
    assert pd.isna(c2["retention_d7"])

    # Limitation documented
    assert any("Day 7" in lim for lim in res.limitations)


def test_compare_retention_weighted_aggregation():
    # Cohort A (10 users, 1 retained -> 10%)
    # Cohort B (1 user, 1 retained -> 100%)
    # Unweighted average: (10% + 100%) / 2 = 55% (WRONG)
    # Weighted average: 2 / 11 = 18.18% (CORRECT)
    users_a = [{"user_id": f"ua_{i}", "signup_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")} for i in range(10)]
    users_b = [{"user_id": "ub_0", "signup_timestamp": pd.Timestamp("2026-09-02T10:00:00Z")}]
    users_df = pd.DataFrame(users_a + users_b)

    events_df = pd.DataFrame([
        {"event_id": "e1", "user_id": "ua_0", "timestamp": pd.Timestamp("2026-09-02T12:00:00Z"), "event_name": "page_view"},
        {"event_id": "e2", "user_id": "ub_0", "timestamp": pd.Timestamp("2026-09-03T12:00:00Z"), "event_name": "page_view"},
    ])

    base_window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-03T00:00:00Z")
    comp_window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-03T00:00:00Z")

    req = AnalysisRequest(
        operation="retention_comparison",
        window=comp_window,
        comparison_window=base_window,
        params={"retention_days": [1], "cohort_period": "day", "max_observation_date": "2026-09-10T00:00:00Z"},
    )

    res = compare_retention(users_df, events_df, req)
    assert res.ok()
    df = res.data
    base_row = df[df["period"] == "baseline"].iloc[0]
    # Check that baseline retention_d1 is 2 / 11 = 0.1818, NOT 0.55!
    assert base_row["eligible_d1"] == 11
    assert base_row["retained_d1"] == 2
    assert base_row["retention_d1"] == pytest.approx(2 / 11, abs=1e-4)
