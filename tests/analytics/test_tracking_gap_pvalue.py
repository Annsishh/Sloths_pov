"""tracking_gap_check reports a p-value so reconciliation evidence can enter the BH-corrected support rule."""

import pandas as pd
import pytest
from scipy import stats

from analytics.data_quality import tracking_gap_check
from analytics.models import AnalysisRequest, TimeWindow
from analytics.statistical import compare_proportions
from investigation.models import extract_tests

WINDOW = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z")
T0 = pd.Timestamp("2026-09-10T10:00:00Z")


def _frames(captured):
    """captured: {browser: (orders, events_that_match_orders)}"""
    users, orders, events = [], [], []
    for b, (n_orders, n_events) in captured.items():
        for i in range(n_orders):
            uid = f"{b}_{i}"
            users.append({"user_id": uid, "browser": b})
            orders.append({"order_id": f"o_{uid}", "user_id": uid, "status": "completed", "timestamp": T0})
            if i < n_events:
                events.append({"event_id": f"e_{uid}", "user_id": uid, "event_name": "order_completed", "timestamp": T0})
    return (pd.DataFrame(users), pd.DataFrame(orders),
            pd.DataFrame(events, columns=["event_id", "user_id", "event_name", "timestamp"]))


def _run(captured, breakdown=("browser",)):
    users, orders, events = _frames(captured)
    req = AnalysisRequest(operation="tracking_gap_check", window=WINDOW, params={"breakdown": list(breakdown)})
    return tracking_gap_check(events, orders, users, req)


def test_tool_returns_raw_per_segment_pvalues_with_no_internal_correction():
    # Safari: 10/20 orders matched; Chrome: 20/20. Each segment is compared with the rest (2 tests).
    # pooled p = 30/40 = 0.75 ; se = sqrt(0.75*0.25*(1/20+1/20)) = 0.136931 ; z = -0.5/se = -3.6515
    # p = 2*(1 - Phi(3.6515)) = 2.607e-4
    res = _run({"Safari": (20, 10), "Chrome": (20, 20)})
    z = -0.5 / (0.75 * 0.25 * 0.1) ** 0.5
    assert z == pytest.approx(-3.6515, abs=1e-4)
    expected_row = 2 * stats.norm.sf(abs(z))
    assert expected_row == pytest.approx(2.607e-4, abs=2e-7)
    assert res.data["p_value"].tolist() == pytest.approx([expected_row] * 2, abs=1e-6)  # rows are rounded to 6 dp
    # Raw per-segment p-values, labelled; NO Bonferroni (or any) correction inside the tool.
    assert "p_value" not in res.metadata
    assert set(res.metadata["p_values"]) == {"capture_rate[browser=Chrome]", "capture_rate[browser=Safari]"}
    for v in res.metadata["p_values"].values():
        assert v == pytest.approx(expected_row, abs=1e-6)
    assert extract_tests(res) == res.metadata["p_values"]


def test_healthy_tracking_gives_a_large_pvalue():
    res = _run({"Safari": (30, 30), "Chrome": (30, 30)})
    assert all(v == pytest.approx(1.0) for v in res.metadata["p_values"].values())


def test_single_segment_has_no_pvalue():
    res = _run({"Safari": (10, 0)})
    assert res.metadata["p_values"] == {}
    assert "p_value" not in res.data.columns
    assert extract_tests(res) == {}


def test_segment_with_no_orders_is_skipped_not_crashed():
    users, orders, events = _frames({"Safari": (10, 0), "Chrome": (10, 10)})
    extra = pd.DataFrame([{"event_id": "stray", "user_id": "ghost", "event_name": "order_completed", "timestamp": T0}])
    users = pd.concat([users, pd.DataFrame([{"user_id": "ghost", "browser": "Firefox"}])], ignore_index=True)
    events = pd.concat([events, extra], ignore_index=True)
    req = AnalysisRequest(operation="tracking_gap_check", window=WINDOW, params={"breakdown": ["browser"]})
    res = tracking_gap_check(events, orders, users, req)
    firefox = res.data[res.data["browser"] == "Firefox"].iloc[0]
    assert pd.isna(firefox["p_value"])
    assert all(0.0 <= v <= 1.0 for v in res.metadata["p_values"].values())
    assert not any("Firefox" in k for k in res.metadata["p_values"])
