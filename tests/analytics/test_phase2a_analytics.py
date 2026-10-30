"""Phase 2a analytics: MDE/power, test selection, mix-shift decomposition, seasonal baseline, retention p-values."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.decomposition import decompose_rate_change, mix_shift_decomposition
from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.power import mde_means, mde_proportions, power_for_effect
from analytics.seasonality import seasonal_baseline
from analytics.statistical import compare_means, compare_proportions, run_statistical_test

DATA = Path(__file__).resolve().parents[2] / "data"


# ------------------------------------------------------------------ power / MDE (hand computed)

def test_mde_proportions_hand_computed():
    # (1.959964 + 0.841621) * sqrt(0.1*0.9*(1/1000+1/1000)) = 2.801585 * 0.0134164 = 0.037587
    assert mde_proportions(0.1, 1000, 1000) == pytest.approx(0.037587, abs=2e-6)


def test_mde_means_hand_computed():
    # 2.801585 * 10 * sqrt(1/100+1/100) = 2.801585 * 1.414214 = 3.96200
    assert mde_means(10.0, 100, 100) == pytest.approx(3.9620, abs=1e-3)


def test_mde_edges():
    assert mde_proportions(0.1, 0, 10) is None
    assert mde_means(5.0, 10, 0) is None
    assert mde_means(float("nan"), 10, 10) is None
    # zero-variance baseline is clamped, not reported as "can detect anything"
    assert mde_proportions(0.0, 50, 50) > 0
    assert mde_proportions(1.0, 50, 50) > 0


def test_power_at_mde_is_about_80_percent():
    n = 1000
    mde = mde_proportions(0.1, n, n)
    se = math.sqrt(0.1 * 0.9 * 2 / n)
    assert power_for_effect(se, mde) == pytest.approx(0.80, abs=0.005)
    assert power_for_effect(se, 0.0) == pytest.approx(0.05, abs=1e-9)


def test_underpowered_is_judged_against_the_effect_of_interest():
    r = compare_proportions(10, 100, 12, 100, target_effect=0.05)   # 10% vs 12%, want to see 5 points
    assert r["p_value"] > 0.05 and r["mde_80pct_power"] > 0.05
    assert r["underpowered"] is True
    assert any("Underpowered" in w for w in r["warnings"])


def test_adequately_powered_nonsignificant_result_is_not_flagged():
    r = compare_proportions(1000, 10000, 1005, 10000, target_effect=0.05)   # n=10k each: MDE ~ 1.2 points
    assert r["p_value"] > 0.05 and r["mde_80pct_power"] < 0.05
    assert r["underpowered"] is False


def test_without_a_target_underpowered_is_unknown_not_vacuously_true():
    r = compare_proportions(10, 100, 12, 100)
    assert r["underpowered"] is None and r["mde_80pct_power"] > 0
    assert any("could only reliably detect" in w for w in r["warnings"])


def test_significant_result_is_never_marked_underpowered():
    assert compare_proportions(10, 100, 40, 100, target_effect=0.01)["underpowered"] is None


def test_compare_means_reports_mde_and_target_based_underpowered():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 5, 20), rng.normal(10.2, 5, 20)
    r = compare_means(a, b, target_effect=1.0)
    assert r["mde_80pct_power"] > 1.0 and r["underpowered"] is True


# ------------------------------------------------------------------ test selection

def test_means_test_on_binary_data_is_flagged_as_misuse():
    r = compare_means([0, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1])
    assert r["misuse_binary_data"] is True
    assert any("Test-selection misuse" in w for w in r["warnings"])


def test_means_test_on_continuous_data_is_not_flagged():
    assert compare_means([1.5, 2.5, 3.5], [2.0, 3.0, 4.5])["misuse_binary_data"] is False


def test_statistical_test_op_flags_misuse_and_marks_inputs_caller_supplied():
    w = TimeWindow.from_iso("2026-09-01", "2026-09-02")
    res = run_statistical_test(AnalysisRequest("statistical_test", w, params={
        "test_type": "means", "values1": [0, 1, 0, 1], "values2": [1, 1, 1, 0]}))
    assert res.ok() and res.metadata["misuse_flags"] == ["binary_data_with_means_test"]
    assert res.metadata["p_value_provenance"] == "caller_supplied"
    prop = run_statistical_test(AnalysisRequest("statistical_test", w, params={
        "test_type": "proportions", "k1": 5, "n1": 50, "k2": 9, "n2": 50}))
    assert prop.metadata["misuse_flags"] == [] and prop.metadata["p_value_provenance"] == "caller_supplied"


# ------------------------------------------------------------------ mix-shift decomposition

def test_decomposition_hand_computed_and_sums_to_total():
    base = {("A",): (100, 50), ("B",): (100, 10)}     # w .5/.5, r .5/.1, R0 = .30
    comp = {("A",): (50, 30), ("B",): (150, 15)}      # w .25/.75, r .6/.1, R1 = .225
    d = decompose_rate_change(base, comp)
    assert d["total_change"] == pytest.approx(-0.075)
    assert d["mix_effect"] == pytest.approx(-0.10)          # (-.25*.5) + (.25*.1)
    assert d["rate_effect"] == pytest.approx(0.05)          # .5*(.6-.5) + .5*0
    assert d["interaction"] == pytest.approx(-0.025)        # (-.25*.1) + (.25*0)
    assert d["mix_effect"] + d["rate_effect"] + d["interaction"] == pytest.approx(d["total_change"])


def test_decomposition_identity_holds_with_new_and_vanished_segments():
    d = decompose_rate_change({("A",): (100, 50)}, {("A",): (50, 25), ("B",): (50, 0)})
    assert d["total_change"] == pytest.approx(-0.25)
    assert d["mix_effect"] + d["rate_effect"] + d["interaction"] == pytest.approx(-0.25)
    d2 = decompose_rate_change({("A",): (50, 25), ("B",): (50, 5)}, {("A",): (100, 40)})
    assert d2["mix_effect"] + d2["rate_effect"] + d2["interaction"] == pytest.approx(d2["total_change"])


def test_decomposition_requires_data_in_both_periods():
    with pytest.raises(ValueError):
        decompose_rate_change({}, {("A",): (1, 1)})


def _orders(rows):
    return pd.DataFrame(rows)


def test_mix_shift_operation_end_to_end_pure_mix_change():
    """Segment rates are constant; only the composition shifts => rate effect ~ 0, mix explains it all."""
    users, orders = [], []
    uid = 0

    def add(seg, n, k, day):
        nonlocal uid
        for i in range(n):
            uid += 1
            users.append({"user_id": f"u{uid}", "acquisition_channel": seg})
            orders.append({"order_id": f"o{uid}", "user_id": f"u{uid}", "status": "completed" if i < k else "failed",
                           "timestamp": pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z")})
    add("organic", 200, 180, 3)       # baseline: 90% good, 200 of 300 rows
    add("paid", 100, 30, 3)           # 30% good
    add("organic", 100, 90, 13)       # comparison: same segment rates, mix moves to paid
    add("paid", 200, 60, 13)
    loader = DataLoader.from_dataframes({"users": pd.DataFrame(users), "orders": pd.DataFrame(orders)})
    req = AnalysisRequest(
        "mix_shift_decomposition", TimeWindow.from_iso("2026-09-10", "2026-09-20"),
        comparison_window=TimeWindow.from_iso("2026-09-01", "2026-09-10"),
        params={"metric_column": "status", "metric_value": "completed", "breakdown": ["acquisition_channel"]})
    res = AnalyticsExecutor(loader).run(req)
    assert res.ok()
    m = res.metadata
    assert m["rate_effect"] == pytest.approx(0.0, abs=1e-12) and m["interaction"] == pytest.approx(0.0, abs=1e-12)
    assert m["total_change"] == pytest.approx(0.5 - 0.6, abs=1e-12) or m["mix_effect"] == pytest.approx(m["total_change"])
    assert abs(m["identity_residual"]) < 1e-12
    est = m["effect_estimate"]
    assert est["method"].startswith("mix_rate_decomposition") and est["se"] >= 0 and est["window"]["start"].startswith("2026-09-10")
    assert m["p_values"]["mix_homogeneity"] < 1e-6                  # composition clearly changed
    assert m["p_values"]["overall_rate_change"] < 0.05


def test_mix_shift_is_reproducible():
    base = {("A",): (100, 50), ("B",): (100, 10)}
    comp = {("A",): (50, 30), ("B",): (150, 15)}
    from analytics.decomposition import bootstrap_se
    assert bootstrap_se(base, comp) == bootstrap_se(base, comp)


def test_mix_shift_validation_errors():
    users = pd.DataFrame([{"user_id": "u1", "device_type": "m"}])
    orders = pd.DataFrame([{"order_id": "o", "user_id": "u1", "status": "completed",
                            "timestamp": pd.Timestamp("2026-09-12T00:00:00Z")}])
    w = TimeWindow.from_iso("2026-09-10", "2026-09-20")
    cw = TimeWindow.from_iso("2026-09-01", "2026-09-10")
    ok = {"metric_column": "status", "metric_value": "completed", "breakdown": ["device_type"]}
    assert mix_shift_decomposition(orders, users, AnalysisRequest("m", w, params=ok)).status == AnalysisStatus.INVALID_PARAMS  # no baseline
    assert mix_shift_decomposition(orders, users, AnalysisRequest("m", w, comparison_window=cw, params={**ok, "breakdown": []})).status == AnalysisStatus.INVALID_PARAMS
    assert mix_shift_decomposition(orders, users, AnalysisRequest("m", w, comparison_window=cw, params={"breakdown": ["device_type"]})).status == AnalysisStatus.INVALID_PARAMS
    assert mix_shift_decomposition(orders, users, AnalysisRequest("m", w, comparison_window=cw, params=ok)).status == AnalysisStatus.INSUFFICIENT_DATA  # empty baseline


@pytest.mark.skipif(not (DATA / "acquisition_shift" / "investigator_data").exists(), reason="tracked dataset missing")
def test_mix_shift_on_the_acquisition_shift_scenario_finds_a_mix_effect():
    ex = AnalyticsExecutor(DataLoader(data_dir=DATA / "acquisition_shift" / "investigator_data"))
    req = AnalysisRequest(
        "mix_shift_decomposition", TimeWindow.from_iso("2026-09-15", "2026-10-01"),
        comparison_window=TimeWindow.from_iso("2026-09-01", "2026-09-15"),
        params={"table": "events", "metric_column": "event_name", "metric_value": "order_completed",
                "breakdown": ["acquisition_channel"]})
    res = ex.run(req)
    assert res.ok(), res.error_message
    assert abs(res.metadata["identity_residual"]) < 1e-12
    assert res.metadata["p_values"]["mix_homogeneity"] < 0.05


# ------------------------------------------------------------------ seasonal baseline

def _daily_counts(start, weeks, fn):
    rows = []
    for w in range(weeks):
        for d in range(7):
            day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=7 * w + d)
            for i in range(fn(w, d)):
                rows.append({"order_id": f"o{len(rows)}", "user_id": "u", "timestamp": day + pd.Timedelta(hours=1, minutes=i % 50)})
    return pd.DataFrame(rows)


BASE = [10, 12, 14, 16, 18, 8, 6]


def _season_req(week_start, weeks=4, **kw):
    s = pd.Timestamp(week_start)
    return AnalysisRequest("seasonal_baseline", TimeWindow(s.tz_localize("UTC"), (s + pd.Timedelta(days=7)).tz_localize("UTC")),
                           params={"aggregation": "count", "baseline_weeks": weeks, **kw})


def test_seasonal_baseline_detects_a_shift_against_same_weekday_history():
    jitter = [1, -1, 1, -1]
    df = _daily_counts("2026-08-03", 5, lambda w, d: BASE[d] + (jitter[w] if w < 4 else -5))
    res = seasonal_baseline(df, pd.DataFrame(columns=["user_id"]), _season_req("2026-08-31"))
    assert res.ok()
    m = res.metadata
    assert m["days_compared"] == 7 and m["days_outside_range"] == 7
    assert m["mean_deviation"] == pytest.approx(-5.0)
    # expected = same weekday mean of 4 weeks = BASE exactly (jitter sums to 0)
    assert res.data["expected"].tolist() == pytest.approx([float(b) for b in BASE])
    # t(0.975, 3) * sd(1.1547) * sqrt(1 + 1/4) = 4.1186 => range
    assert res.data.iloc[0]["upper"] - res.data.iloc[0]["expected"] == pytest.approx(3.1824 * 1.1547 * math.sqrt(1.25), rel=1e-3)
    assert m["p_values"]["seasonal_deviation"] < 1e-6


def test_weekend_dip_is_not_flagged_when_it_is_normal_for_that_weekday():
    jitter = [1, -1, 1, -1, 0]
    df = _daily_counts("2026-08-03", 5, lambda w, d: BASE[d] + jitter[w])
    res = seasonal_baseline(df, pd.DataFrame(columns=["user_id"]), _season_req("2026-08-31"))
    assert res.metadata["days_outside_range"] == 0
    assert res.metadata["p_values"]["seasonal_deviation"] > 0.5


def test_seasonal_baseline_uses_fewer_weeks_with_a_warning_and_requires_min_history():
    df = _daily_counts("2026-08-17", 3, lambda w, d: BASE[d] + [1, -1, 0][w])   # 2 prior weeks only
    res = seasonal_baseline(df, pd.DataFrame(columns=["user_id"]), _season_req("2026-08-31", weeks=4))
    assert res.ok() and any("Fewer than the requested 4" in w for w in res.warnings)
    short = _daily_counts("2026-08-24", 2, lambda w, d: BASE[d])                 # 1 prior week
    res2 = seasonal_baseline(short, pd.DataFrame(columns=["user_id"]), _season_req("2026-08-31"))
    assert res2.status == AnalysisStatus.INSUFFICIENT_DATA and "more history" in res2.error_message


def test_seasonal_baseline_parameter_validation():
    df = _daily_counts("2026-08-03", 5, lambda w, d: 5)
    u = pd.DataFrame(columns=["user_id"])
    assert seasonal_baseline(df, u, _season_req("2026-08-31", weeks=0)).status == AnalysisStatus.INVALID_PARAMS
    assert seasonal_baseline(df, u, _season_req("2026-08-31", min_weeks=9)).status == AnalysisStatus.INVALID_PARAMS
    bad = _season_req("2026-08-31"); bad.params["aggregation"] = "mean"
    assert seasonal_baseline(df, u, bad).status == AnalysisStatus.INVALID_PARAMS
    assert seasonal_baseline(df.iloc[0:0], u, _season_req("2026-08-31")).status != AnalysisStatus.OK


# ------------------------------------------------------------------ compare_periods / retention p-values

def _period_frame(day, n_good, n_bad):
    rows = [{"user_id": f"u{day}_{i}", "status": "completed" if i < n_good else "failed",
             "timestamp": pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z")} for i in range(n_good + n_bad)]
    return pd.DataFrame(rows)


def test_period_comparison_rate_uses_ztest_with_mde():
    df = pd.concat([_period_frame(3, 90, 10), _period_frame(13, 60, 40)], ignore_index=True)
    req = AnalysisRequest("period_comparison", TimeWindow.from_iso("2026-09-10", "2026-09-20"),
                          comparison_window=TimeWindow.from_iso("2026-09-01", "2026-09-10"),
                          params={"aggregation": "rate", "metric_column": "status", "metric_value": "completed", "table": "orders"})
    res = AnalyticsExecutor(DataLoader.from_dataframes({"orders": df, "users": pd.DataFrame(columns=["user_id"])})).run(req)
    expected = compare_proportions(60, 100, 90, 100)
    assert res.metadata["p_values"]["rate_change"] == pytest.approx(expected["p_value"])
    assert res.metadata["mde"] == pytest.approx(expected["mde_80pct_power"])


def test_period_comparison_counts_produce_no_pvalue():
    df = pd.concat([_period_frame(3, 90, 10), _period_frame(13, 60, 40)], ignore_index=True)
    req = AnalysisRequest("period_comparison", TimeWindow.from_iso("2026-09-10", "2026-09-20"),
                          comparison_window=TimeWindow.from_iso("2026-09-01", "2026-09-10"),
                          params={"aggregation": "count", "table": "orders"})
    res = AnalyticsExecutor(DataLoader.from_dataframes({"orders": df, "users": pd.DataFrame(columns=["user_id"])})).run(req)
    assert res.metadata["p_values"] == {} and any("No significance test" in w for w in res.warnings)


@pytest.mark.skipif(not (DATA / "product_regression" / "investigator_data").exists(), reason="tracked dataset missing")
def test_retention_comparison_reports_raw_pvalues_per_horizon():
    ex = AnalyticsExecutor(DataLoader(data_dir=DATA / "baseline" / "investigator_data"))
    req = AnalysisRequest("retention_comparison", TimeWindow.from_iso("2026-09-12", "2026-09-26"),
                          comparison_window=TimeWindow.from_iso("2026-09-01", "2026-09-12"),
                          params={"retention_days": [1, 3], "cohort_period": "week", "max_observation_date": "2026-10-01"})
    res = ex.run(req)
    assert res.ok(), res.error_message
    for k, v in res.metadata["p_values"].items():
        assert k.startswith("retention_d") and 0.0 <= v <= 1.0
