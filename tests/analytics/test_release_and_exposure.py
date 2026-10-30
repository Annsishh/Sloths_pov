"""Tests for release impact and feature exposure analysis."""

import pandas as pd
import pytest

from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.release_analysis import exposure_effect, release_impact


@pytest.fixture
def sample_release_data():
    releases_df = pd.DataFrame([
        {
            "release_id": "rel_v1_1",
            "version": "1.1.0",
            "component": "checkout",
            "deployed_timestamp": pd.Timestamp("2026-09-10T00:00:00Z"),
        }
    ])

    users_df = pd.DataFrame([
        {"user_id": f"u{i}", "device_type": "desktop"} for i in range(10)
    ])

    # 10 orders per day before release (Sept 5 to Sept 10 = 5 days -> 50 orders)
    # 5 orders per day after release (Sept 10 to Sept 15 = 5 days -> 25 orders)
    orders = []
    for day in range(5, 10):
        for o in range(10):
            orders.append({
                "order_id": f"pre_{day}_{o}",
                "user_id": f"u{o}",
                "timestamp": pd.Timestamp(f"2026-09-{day:02d}T12:00:00Z"),
                "total_amount": 100.0,
            })
    for day in range(10, 15):
        for o in range(5):
            orders.append({
                "order_id": f"post_{day}_{o}",
                "user_id": f"u{o}",
                "timestamp": pd.Timestamp(f"2026-09-{day:02d}T12:00:00Z"),
                "total_amount": 100.0,
            })
    orders_df = pd.DataFrame(orders)

    return releases_df, orders_df, users_df


def test_release_impact_counts_and_limitation(sample_release_data):
    releases_df, orders_df, users_df = sample_release_data
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-20T00:00:00Z")

    req = AnalysisRequest(
        operation="release_impact",
        window=window,
        params={
            "release_id": "rel_v1_1",
            "window_days": 5,
            "aggregation": "count",
        },
    )
    res = release_impact(releases_df, orders_df, users_df, req)
    assert res.ok()
    df = res.data

    pre_val = df.loc[df["period"] == "pre_release", "value"].iloc[0]
    post_val = df.loc[df["period"] == "post_release", "value"].iloc[0]
    chg_val = df.loc[df["period"] == "change", "value"].iloc[0]

    assert pre_val == 50
    assert post_val == 25
    assert chg_val == -25
    assert df.loc[df["period"] == "change", "relative_change"].iloc[0] == -0.5

    # Check that temporal correlation limitation is explicitly reported
    assert any("Temporal correlation only" in lim for lim in res.limitations)


def test_release_impact_unknown_release(sample_release_data):
    releases_df, orders_df, users_df = sample_release_data
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-20T00:00:00Z")

    req = AnalysisRequest(
        operation="release_impact",
        window=window,
        params={"release_id": "rel_nonexistent", "window_days": 5},
    )
    res = release_impact(releases_df, orders_df, users_df, req)
    assert res.status == AnalysisStatus.INVALID_PARAMS
    assert "not found" in res.error_message


def test_exposure_effect_followup_and_stats():
    # 20 control users, 20 treatment users exposed on Sept 1
    # 1 control user exposed on Sept 9 (post_days=3 -> requires Sept 12 -> beyond max_obs Sept 10)
    users = []
    exposures = []
    events = []

    # Control: 20 users exposed Sept 1, 4 convert (20%)
    for i in range(20):
        uid = f"ctrl_{i}"
        users.append({"user_id": uid})
        exposures.append({
            "user_id": uid,
            "feature_key": "new_checkout",
            "variant": "control",
            "first_exposed_timestamp": pd.Timestamp("2026-09-01T10:00:00Z"),
        })
        if i < 4:
            events.append({
                "user_id": uid,
                "event_name": "checkout_completed",
                "timestamp": pd.Timestamp("2026-09-02T12:00:00Z"),
            })

    # Treatment: 20 users exposed Sept 1, 12 convert (60%)
    for i in range(20):
        uid = f"treat_{i}"
        users.append({"user_id": uid})
        exposures.append({
            "user_id": uid,
            "feature_key": "new_checkout",
            "variant": "treatment",
            "first_exposed_timestamp": pd.Timestamp("2026-09-01T10:00:00Z"),
        })
        if i < 12:
            events.append({
                "user_id": uid,
                "event_name": "checkout_completed",
                "timestamp": pd.Timestamp("2026-09-02T12:00:00Z"),
            })

    # Incomplete user: exposed Sept 9, follow-up ends Sept 12 (beyond Sept 10)
    uid_late = "late_user"
    users.append({"user_id": uid_late})
    exposures.append({
        "user_id": uid_late,
        "feature_key": "new_checkout",
        "variant": "control",
        "first_exposed_timestamp": pd.Timestamp("2026-09-09T10:00:00Z"),
    })

    users_df = pd.DataFrame(users)
    exposures_df = pd.DataFrame(exposures)
    events_df = pd.DataFrame(events)

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="exposure_effect",
        window=window,
        params={
            "feature_key": "new_checkout",
            "outcome_event": "checkout_completed",
            "post_exposure_days": 3,
            "max_observation_date": "2026-09-10T00:00:00Z",
        },
    )

    res = exposure_effect(exposures_df, events_df, users_df, req)
    assert res.ok()
    df = res.data

    ctrl = df[df["variant"] == "control"].iloc[0]
    treat = df[df["variant"] == "treatment"].iloc[0]

    # Total assigned in control was 21, but 1 had incomplete follow-up -> eligible = 20
    assert ctrl["total_assigned"] == 21
    assert ctrl["eligible_users"] == 20
    assert ctrl["converted_users"] == 4
    assert ctrl["conversion_rate"] == 0.20

    assert treat["total_assigned"] == 20
    assert treat["eligible_users"] == 20
    assert treat["converted_users"] == 12
    assert treat["conversion_rate"] == 0.60

    # Incomplete follow up warning/limitation
    assert any("incomplete follow-up" in lim for lim in res.limitations)
    # Association vs causation limitation
    assert any("Observational distinction" in lim for lim in res.limitations)

    # Statistical test check
    assert res.metadata["p_value"] is not None
    assert res.metadata["p_value"] < 0.05
    assert res.metadata["is_significant_p05"] is True


def test_exposure_effect_repeated_users_and_crossover():
    # User u1 has duplicate exposure records in treatment
    # User u_cross appears in BOTH control and treatment (crossover contamination)
    # Users u2, u3 in control
    users_df = pd.DataFrame([
        {"user_id": "u1"}, {"user_id": "u2"}, {"user_id": "u3"}, {"user_id": "u_cross"}
    ])
    exposures_df = pd.DataFrame([
        # u1 repeated exposures
        {"user_id": "u1", "feature_key": "btn", "variant": "treatment", "first_exposed_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"user_id": "u1", "feature_key": "btn", "variant": "treatment", "first_exposed_timestamp": pd.Timestamp("2026-09-02T10:00:00Z")},
        # u_cross in both
        {"user_id": "u_cross", "feature_key": "btn", "variant": "treatment", "first_exposed_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"user_id": "u_cross", "feature_key": "btn", "variant": "control", "first_exposed_timestamp": pd.Timestamp("2026-09-01T11:00:00Z")},
        # control
        {"user_id": "u2", "feature_key": "btn", "variant": "control", "first_exposed_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"user_id": "u3", "feature_key": "btn", "variant": "control", "first_exposed_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
    ])
    events_df = pd.DataFrame([])

    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="exposure_effect",
        window=window,
        params={"feature_key": "btn", "post_exposure_days": 3},
    )
    res = exposure_effect(exposures_df, events_df, users_df, req)
    assert res.ok()
    df = res.data

    # Crossover warning should be reported
    assert any("crossover contamination" in w for w in res.warnings)
    # Deduplication limitation should be reported
    assert any("repeated exposure record" in lim for lim in res.limitations)

    # u_cross is excluded. u1 is deduplicated to 1 user. u2 and u3 are 2 users.
    treat = df[df["variant"] == "treatment"].iloc[0]
    ctrl = df[df["variant"] == "control"].iloc[0]
    assert treat["total_assigned"] == 1
    assert ctrl["total_assigned"] == 2

