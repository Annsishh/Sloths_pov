"""Fixes and additions found while building the evaluation harness."""

import numpy as np
import pandas as pd
import pytest

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.spikes import spike_check
from controller.capability_catalogue import CONTROLLED_OPERATIONS, validate_analysis_plan
from controller.controller import InvestigationController
from controller.llm_client import MockLLMClient
from controller.schemas import AnalysisDecision, HypothesesGenerationOutput, HypothesisProposal
from investigation import Observation

U = pd.DataFrame(columns=["user_id"])
W = TimeWindow.from_iso


def _events():
    rows = []
    def ev(i, uid, sid, day, name, browser="Chrome"):
        rows.append({"event_id": f"e{i}", "user_id": uid, "session_id": sid, "timestamp": pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z"),
                     "event_name": name, "device_type": "mobile", "browser": browser, "app_version": "v1"})
    n = 0
    for sid, uid, day, names in [("s1", "u1", 2, ["page_view", "checkout_started", "order_completed"]), ("s2", "u1", 2, ["page_view"]),
                                 ("s3", "u2", 3, ["page_view", "checkout_started"]), ("s4", "u2", 12, ["page_view", "order_completed"])]:
        for nm in names:
            n += 1
            ev(n, uid, sid, day, nm)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ catalogue <-> tool parameter names

def test_funnel_works_with_the_catalogue_parameter_names():
    """The catalogue advertises funnel_steps/breakdown; the tool used to read steps/dimensions and always failed."""
    users = pd.DataFrame([{"user_id": "u1", "acquisition_channel": "a"}, {"user_id": "u2", "acquisition_channel": "b"}])
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"events": _events(), "users": users}))
    res = ex.run(AnalysisRequest("funnel", W("2026-09-01", "2026-09-30"),
                                 params={"funnel_steps": ["checkout_started", "order_completed"], "breakdown": ["acquisition_channel"]}))
    assert res.ok(), res.error_message
    assert "acquisition_channel" in res.data.columns and set(res.data["acquisition_channel"]) == {"a", "b"}
    cmp_ = ex.run(AnalysisRequest("funnel_comparison", W("2026-09-10", "2026-09-30"), comparison_window=W("2026-09-01", "2026-09-10"),
                                  params={"funnel_steps": ["page_view", "order_completed"]}))
    assert cmp_.status in (AnalysisStatus.OK, AnalysisStatus.INSUFFICIENT_DATA) and "requires 'steps'" not in (cmp_.error_message or "")


def test_legacy_steps_parameter_still_works():
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"events": _events(), "users": U}))
    assert ex.run(AnalysisRequest("funnel", W("2026-09-01", "2026-09-30"), params={"steps": ["page_view", "order_completed"]})).ok()


def test_error_analysis_can_break_down_by_failure_reason_through_the_catalogue():
    assert validate_analysis_plan("error_analysis", {"breakdown": ["failure_reason"]})[0]
    orders = pd.DataFrame([{"order_id": f"o{i}", "user_id": "u", "status": "failed", "failure_reason": r,
                            "timestamp": pd.Timestamp("2026-09-12T10:00:00Z")} for i, r in enumerate(["a", "a", "b"])])
    res = AnalyticsExecutor(DataLoader.from_dataframes({"orders": orders, "users": U})).run(
        AnalysisRequest("error_analysis", W("2026-09-10", "2026-09-20"), params={"breakdown": ["failure_reason"]}))
    assert dict(zip(res.data["failure_reason"], res.data["failed_count"])) == {"a": 2, "b": 1}


# ------------------------------------------------------------------ derived sessions table

def test_sessions_table_is_derived_from_events():
    ld = DataLoader.from_dataframes({"events": _events(), "users": U})
    s = ld.load("sessions").set_index("session_id")
    assert len(s) == 4
    assert bool(s.loc["s1", "converted"]) and bool(s.loc["s1", "checkout_started"])
    assert not bool(s.loc["s2", "converted"]) and bool(s.loc["s3", "checkout_started"]) and not bool(s.loc["s3", "converted"])
    assert bool(s.loc["s4", "converted"]) and not bool(s.loc["s4", "checkout_started"])
    assert "sessions" not in ld.available_tables()                 # derived, never listed as on-disk data
    assert ld.load("sessions") is ld.load("sessions")              # cached


def test_sessions_table_handles_empty_events():
    ld = DataLoader.from_dataframes({"events": _events().iloc[0:0], "users": U})
    assert ld.load("sessions").empty and "converted" in ld.load("sessions").columns


def test_mix_decomposition_can_use_sessions_as_the_unit():
    rows, users = [], []
    n = 0
    for chan, base_sess, comp_sess, rate in (("organic", 100, 60, 0.30), ("paid", 20, 80, 0.05)):
        for period, day, cnt in (("b", 3, base_sess), ("c", 13, comp_sess)):
            for i in range(cnt):
                n += 1
                uid = f"{chan}{period}{i}"
                users.append({"user_id": uid, "acquisition_channel": chan})
                rows.append({"event_id": f"p{n}", "user_id": uid, "session_id": f"s{n}", "timestamp": pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z"),
                             "event_name": "page_view", "device_type": "d", "browser": "b", "app_version": "v"})
                if i < int(cnt * rate):
                    n += 1
                    rows.append({**rows[-1], "event_id": f"p{n}", "event_name": "order_completed"})
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"events": pd.DataFrame(rows), "users": pd.DataFrame(users)}))
    res = ex.run(AnalysisRequest("mix_shift_decomposition", W("2026-09-10", "2026-09-20"), comparison_window=W("2026-09-01", "2026-09-10"),
                                 params={"table": "sessions", "metric_column": "converted", "metric_value": True,
                                         "breakdown": ["acquisition_channel"]}))
    assert res.ok(), res.error_message
    m = res.metadata
    assert m["mix_effect"] < 0 and abs(m["mix_effect"]) > 0.5 * abs(m["total_change"])        # the shift to 'paid' explains it
    assert abs(m["identity_residual"]) < 1e-12 and m["p_values"]["mix_homogeneity"] < 0.001


# ------------------------------------------------------------------ lookback + spikes

def _orders_failing_on(day_failures, days=16, per_day=20):
    rows = []
    for d in range(days):
        for i in range(per_day):
            fail = i < day_failures.get(d, 1)
            rows.append({"order_id": f"o{d}_{i}", "user_id": "u", "status": "failed" if fail else "completed",
                         "timestamp": pd.Timestamp("2026-09-01T10:00:00Z") + pd.Timedelta(days=d, minutes=i)})
    return pd.DataFrame(rows)


def test_spike_check_flags_the_one_abnormal_day_with_raw_pvalues():
    df = _orders_failing_on({9: 15})
    res = spike_check(df, U, AnalysisRequest("spike_check", W("2026-09-01", "2026-09-17"),
                                             params={"metric_column": "status", "metric_value": "failed"}))
    assert res.ok()
    m = res.metadata
    assert m["top_period"] == "2026-09-10" and m["top_rate"] == pytest.approx(0.75)
    assert len(m["p_values"]) == 16                               # one RAW p-value per day, no correction inside the tool
    assert m["p_values"]["2026-09-10"] < 1e-6
    assert sum(1 for p in m["p_values"].values() if p < 0.001) == 1
    flat = spike_check(_orders_failing_on({}), U, AnalysisRequest("spike_check", W("2026-09-01", "2026-09-17"),
                                                                  params={"metric_column": "status", "metric_value": "failed"}))
    assert all(p > 0.5 for p in flat.metadata["p_values"].values())


def test_spike_check_edge_cases():
    req = lambda w, **p: AnalysisRequest("spike_check", w, params=p)
    df = _orders_failing_on({}, days=3)
    assert spike_check(df, U, req(W("2026-09-01", "2026-09-04"), metric_column="status", metric_value="failed")).status == AnalysisStatus.INSUFFICIENT_DATA
    assert spike_check(df, U, req(W("2026-09-01", "2026-09-04"))).status == AnalysisStatus.INVALID_PARAMS
    assert spike_check(df, U, req(W("2030-01-01", "2030-01-09"), metric_column="status", metric_value="failed")).status == AnalysisStatus.INSUFFICIENT_DATA
    assert spike_check(df, U, req(W("2026-09-01", "2026-09-04"), metric_column="nope", metric_value=1)).status == AnalysisStatus.INVALID_PARAMS


def test_lookback_days_extends_the_window_start():
    orders = _orders_failing_on({9: 15})
    loader = DataLoader.from_dataframes({"orders": orders, "users": U})
    ex = AnalyticsExecutor(loader)
    obs = Observation("order_success_rate", W("2026-09-09", "2026-09-10"), W("2026-09-10", "2026-09-11"), 0.95, 0.25, -0.7, -0.74, "d")
    canned = [HypothesesGenerationOutput(
        hypotheses=[HypothesisProposal(title="Outage hypothesis", explanation="An external outage happened.", category="third_party_outage"),
                    HypothesisProposal(title="Release hypothesis", explanation="A release broke things.", category="code_regression")],
        reasoning="two candidates"),
        AnalysisDecision(hypothesis_id_to_test="hyp_1", operation="spike_check", rationale="Look for an abnormal day.",
                         parameters={"table": "orders", "metric_column": "status", "metric_value": "failed", "lookback_days": 7})]
    ctrl = InvestigationController(executor=ex, llm_client=MockLLMClient(canned))
    state = ctrl.start_investigation(obs)
    ctrl.step_formulate_hypotheses(state)
    seen = []
    orig = ex.run
    ex.run = lambda r: (seen.append(r), orig(r))[1]
    result = ctrl.execute_analysis(state, ctrl.step_select_analysis(state, 1))
    assert seen[0].window.start == pd.Timestamp("2026-09-03", tz="UTC") and seen[0].window.end == pd.Timestamp("2026-09-11", tz="UTC")
    assert result.ok() and result.metadata["top_period"] == "2026-09-10"


@pytest.mark.parametrize("params", [{"lookback_days": 0}, {"lookback_days": 9999}, {"lookback_days": "7"}])
def test_lookback_days_is_validated(params):
    base = {"metric_column": "status", "metric_value": "failed", **params}
    assert not validate_analysis_plan("spike_check", base)[0]


def test_lookback_is_only_accepted_where_supported():
    assert not validate_analysis_plan("funnel", {"funnel_steps": ["a", "b"], "lookback_days": 7})[0]
    assert CONTROLLED_OPERATIONS["spike_check"].supports_lookback and CONTROLLED_OPERATIONS["time_series"].supports_lookback
    assert not CONTROLLED_OPERATIONS["error_analysis"].supports_lookback


def test_third_party_outage_is_testable_via_spike_detection():
    from analytics.loader import CATALOGUE
    from investigation.testability import assess_testability
    ta = assess_testability(CONTROLLED_OPERATIONS, CATALOGUE.keys())["third_party_outage"]
    assert ta.testable and "spike_check" in ta.provided_by


# ------------------------------------------------------------------ seasonality additions & change-point MDE

def test_seasonal_baseline_reports_weekday_pattern_and_expected_shift():
    base = [10, 12, 14, 16, 18, 4, 3]
    rows = []
    for w in range(6):
        for d in range(7):
            day = pd.Timestamp("2026-08-03", tz="UTC") + pd.Timedelta(days=7 * w + d)
            rows += [{"order_id": f"o{len(rows)}", "user_id": "u", "timestamp": day + pd.Timedelta(minutes=i)}
                     for i in range(base[d] + [1, -1, 0, 1, -1, 0][w])]
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"orders": pd.DataFrame(rows), "users": U}))
    # compare the weekend (Sat-Sun 2026-09-12..13, week 6 start 09-07) with the preceding weekdays (09-07..09-12)
    res = ex.run(AnalysisRequest("seasonal_baseline", W("2026-09-12", "2026-09-14"), comparison_window=W("2026-09-07", "2026-09-12"),
                                 params={"table": "orders", "aggregation": "count", "baseline_weeks": 4}))
    assert res.ok(), res.error_message
    p = res.metadata["p_values"]
    assert p["weekday_pattern"] < 0.01 and p["seasonal_deviation"] > 0.05        # a real weekday pattern; weekend is normal
    est = res.metadata["effect_estimate"]
    expected_weekend = (4 + 3) / 2                                                # same-weekday means
    expected_weekdays = (10 + 12 + 14 + 16 + 18) / 5
    assert est["value"] == pytest.approx(expected_weekend - expected_weekdays, abs=1.0)
    assert est["method"] == "seasonal_same_weekday_expected_shift" and est["se"] >= 0 and est["window"]["start"].startswith("2026-09-12")


def test_change_point_reports_an_mde_in_metric_units():
    rng = np.random.default_rng(1)
    counts = rng.poisson(20, 24)
    rows = []
    for i, c in enumerate(counts):
        rows += [{"order_id": f"o{len(rows)}", "user_id": "u", "timestamp": pd.Timestamp("2026-09-01", tz="UTC") + pd.Timedelta(days=i, minutes=j)} for j in range(int(c))]
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"orders": pd.DataFrame(rows), "users": U}))
    res = ex.run(AnalysisRequest("change_point", W("2026-09-13", "2026-09-25"), comparison_window=W("2026-09-01", "2026-09-13"),
                                 params={"table": "orders", "aggregation": "count"}))
    sd = float(np.std(counts, ddof=1))
    assert res.metadata["mde"] == pytest.approx(2.801585 * sd * np.sqrt(2 / 12), rel=0.05)
