"""Phase 2b: change-point detection, difference-in-differences, SRM, delta-method ratios, edge-case sweep."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.changepoint import change_point, detect_change_point
from analytics.executor import AnalyticsExecutor
from analytics.experiment_checks import sample_ratio_mismatch
from analytics.loader import CATALOGUE, DataLoader
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.ratio_metrics import compare_ratio_metrics, delta_method_ratio
from controller.capability_catalogue import CONTROLLED_OPERATIONS

U = pd.DataFrame(columns=["user_id"])


def W(a, b):
    return TimeWindow.from_iso(a, b)


# ------------------------------------------------------------------ change point

def _orders_by_day(counts, start="2026-09-01"):
    rows = []
    for i, c in enumerate(counts):
        day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=i)
        rows += [{"order_id": f"o{len(rows)}", "user_id": "u", "timestamp": day + pd.Timedelta(minutes=j)} for j in range(int(c))]
    return pd.DataFrame(rows)


def _cp_req(n_days, **params):
    return AnalysisRequest("change_point", W("2026-09-01", f"2026-09-{1 + n_days:02d}"),
                           params={"aggregation": "count", "n_permutations": 499, **params})


def test_step_change_is_located_to_the_right_day():
    counts = [10, 11, 9, 10, 12, 10, 9, 11, 10, 11] + [20, 21, 19, 20, 22, 20, 19, 21, 20, 21]
    res = change_point(_orders_by_day(counts), U, _cp_req(20))
    assert res.ok(), res.error_message
    m = res.metadata
    assert m["estimated_onset"] == "2026-09-11" and m["change_index"] == 10
    assert m["shift"] == pytest.approx(10.0, abs=0.5) and m["onset_shape"] == "step"
    assert m["p_values"]["change_point_permutation"] <= 0.01
    assert res.data["segment"].tolist() == ["before"] * 10 + ["after"] * 10


def test_noise_only_series_has_no_significant_change_point():
    rng = np.random.default_rng(3)
    counts = rng.poisson(15, 24)
    res = change_point(_orders_by_day(counts), U, _cp_req(24))
    assert res.ok() and res.metadata["p_values"]["change_point_permutation"] > 0.05


def test_gradual_ramp_is_flagged_as_gradual():
    counts = [10 + 2 * i for i in range(24)]
    res = change_point(_orders_by_day(counts), U, _cp_req(24))
    assert res.metadata["onset_shape"] == "gradual"
    assert any("gradual" in w for w in res.warnings)


def test_change_point_is_reproducible():
    counts = [10, 12, 9, 11, 10, 13, 20, 22, 19, 21, 20, 23]
    a = detect_change_point(np.array(counts), 3)
    b = detect_change_point(np.array(counts), 3)
    assert a == b


def test_change_point_edge_cases():
    assert change_point(_orders_by_day([5] * 12), U, _cp_req(12)).status == AnalysisStatus.INSUFFICIENT_DATA   # constant
    assert change_point(_orders_by_day([5, 9]), U, _cp_req(2)).status == AnalysisStatus.INSUFFICIENT_DATA      # too short
    assert change_point(_orders_by_day([5, 9, 5, 9, 5, 9]).iloc[0:0], U, _cp_req(6)).status != AnalysisStatus.OK
    bad = _cp_req(12); bad.params["aggregation"] = "rate"
    assert change_point(_orders_by_day([5, 9] * 6), U, bad).status == AnalysisStatus.INVALID_PARAMS
    with pytest.raises(ValueError):
        detect_change_point(np.array([1.0, 2.0, 3.0]), 3)


def test_change_point_spans_baseline_and_comparison_windows():
    counts = [10, 11, 9, 10, 12, 10] + [20, 21, 19, 20, 22, 20]
    req = AnalysisRequest("change_point", W("2026-09-07", "2026-09-13"), comparison_window=W("2026-09-01", "2026-09-07"),
                          params={"aggregation": "count"})
    res = change_point(_orders_by_day(counts), U, req)
    assert res.ok() and res.metadata["estimated_onset"] == "2026-09-07"


# ------------------------------------------------------------------ difference-in-differences (hand computed)

DEPLOY = pd.Timestamp("2026-09-15T10:00:00Z")


def _did_tables(cells):
    """cells: {(group, period): (n, k)}; group 'e' = users seen on v9 after deploy."""
    users, orders, events = [], [], []
    for (g, per), (n, k) in cells.items():
        ts = DEPLOY + pd.Timedelta(days=3 if per == "post" else -3)
        for i in range(n):
            uid = f"{g}_{per}_{i}"
            users.append({"user_id": uid})
            orders.append({"order_id": f"o_{uid}", "user_id": uid, "timestamp": ts,
                           "status": "completed" if i < k else "failed"})
            if g == "e" and per == "post":
                events.append({"event_id": f"e_{uid}", "user_id": uid, "session_id": "s", "timestamp": ts, "app_version": "v9"})
    # exposed users must also appear pre-release: re-use the same ids is not required (groups are defined by id set)
    exposed_pre_ids = [f"e_pre_{i}" for i in range(cells[("e", "pre")][0])]
    for uid in exposed_pre_ids:       # users who were exposed later: give them a post event so they join the exposed group
        events.append({"event_id": f"x_{uid}", "user_id": uid, "session_id": "s", "timestamp": DEPLOY + pd.Timedelta(days=1),
                       "app_version": "v9"})
    releases = pd.DataFrame([{"release_id": "rel_x", "version": "v9", "deployed_timestamp": DEPLOY}])
    return releases, pd.DataFrame(events), pd.DataFrame(orders), pd.DataFrame(users)


def test_did_matches_hand_computation():
    # exposed: .8 -> .5 ; other: .8 -> .75   =>   DiD = (-.30) - (-.05) = -.25
    cells = {("e", "pre"): (100, 80), ("e", "post"): (100, 50), ("o", "pre"): (100, 80), ("o", "post"): (100, 75)}
    rel, ev, od, us = _did_tables(cells)
    loader = DataLoader.from_dataframes({"releases": rel, "events": ev, "orders": od, "users": us})
    res = AnalyticsExecutor(loader).run(AnalysisRequest(
        "release_did", W("2026-09-01", "2026-10-01"), params={"release_id": "rel_x", "window_days": 7, "table": "orders"}))
    assert res.ok(), res.error_message
    m = res.metadata
    var = 0.8 * 0.2 / 100 + 0.5 * 0.5 / 100 + 0.8 * 0.2 / 100 + 0.75 * 0.25 / 100          # 0.007575
    assert m["did"] == pytest.approx(-0.25) and m["se"] == pytest.approx(math.sqrt(var))
    assert m["z"] == pytest.approx(-0.25 / math.sqrt(var))                                   # -2.8724
    assert m["p_values"]["did_interaction"] == pytest.approx(2 * stats.norm.sf(2.8724), rel=2e-3)   # 0.00407
    assert m["pre_period_gap"] == pytest.approx(0.0)
    est = m["effect_estimate"]
    assert est["value"] == pytest.approx(-0.25 * 0.5) and est["se"] == pytest.approx(math.sqrt(var) * 0.5)
    assert est["method"].startswith("difference_in_differences") and est["window"]["start"].startswith("2026-09-15")


def test_did_nets_out_a_shock_that_hit_everyone():
    """Both groups fall by 0.3: plain pre/post would blame the release, DiD correctly finds no release effect."""
    cells = {("e", "pre"): (200, 160), ("e", "post"): (200, 100), ("o", "pre"): (200, 160), ("o", "post"): (200, 100)}
    rel, ev, od, us = _did_tables(cells)
    loader = DataLoader.from_dataframes({"releases": rel, "events": ev, "orders": od, "users": us})
    ex = AnalyticsExecutor(loader)
    did = ex.run(AnalysisRequest("release_did", W("2026-09-01", "2026-10-01"),
                                 params={"release_id": "rel_x", "window_days": 7, "table": "orders"}))
    assert did.metadata["did"] == pytest.approx(0.0, abs=1e-12)
    assert did.metadata["p_values"]["did_interaction"] == pytest.approx(1.0)
    prepost = ex.run(AnalysisRequest("release_impact", W("2026-09-01", "2026-10-01"),
                                     params={"release_id": "rel_x", "window_days": 7, "table": "orders",
                                             "aggregation": "rate", "metric_column": "status", "metric_value": "completed"}))
    assert prepost.metadata["absolute_change"] < -0.25       # the naive method sees a big "release effect"


def test_did_edge_cases():
    rel = pd.DataFrame([{"release_id": "rel_x", "version": "v9", "deployed_timestamp": DEPLOY}])
    empty_ev = pd.DataFrame(columns=["event_id", "user_id", "session_id", "timestamp", "app_version"])
    od = pd.DataFrame([{"order_id": "o", "user_id": "u", "timestamp": DEPLOY, "status": "completed"}])
    ex = lambda **kw: AnalyticsExecutor(DataLoader.from_dataframes({"releases": rel, "events": kw.get("ev", empty_ev), "orders": od, "users": U}))
    base = {"release_id": "rel_x", "table": "orders"}
    run = lambda e, **p: e.run(AnalysisRequest("release_did", W("2026-09-01", "2026-10-01"), params={**base, **p}))
    assert run(ex()).status == AnalysisStatus.INSUFFICIENT_DATA                       # nobody on the new version
    assert run(ex(), release_id="nope").status == AnalysisStatus.INVALID_PARAMS
    assert run(ex(), window_days=0).status == AnalysisStatus.INVALID_PARAMS
    assert run(ex(), exposure_column="nope").status == AnalysisStatus.INVALID_PARAMS
    ev = pd.DataFrame([{"event_id": "e", "user_id": "u", "session_id": "s", "timestamp": DEPLOY + pd.Timedelta(hours=2), "app_version": "v9"}])
    assert run(ex(ev=ev)).status == AnalysisStatus.INSUFFICIENT_DATA                  # empty cells


# ------------------------------------------------------------------ SRM

def test_srm_hand_computed():
    r = sample_ratio_mismatch([600, 400])                      # expected 500/500 : chi2 = 2 * 100^2 / 500 = 40
    assert r["chi2"] == pytest.approx(40.0) and r["p_value"] == pytest.approx(stats.chi2.sf(40, 1))
    assert r["mismatch"] is True
    ok = sample_ratio_mismatch([503, 497])                      # chi2 = 0.036
    assert ok["chi2"] == pytest.approx(0.036) and ok["mismatch"] is False
    uneven = sample_ratio_mismatch([800, 200], expected_ratios=[0.8, 0.2])
    assert uneven["chi2"] == pytest.approx(0.0) and uneven["mismatch"] is False


def test_srm_validation():
    for bad in ([5], [-1, 5], [0, 0]):
        with pytest.raises(ValueError):
            sample_ratio_mismatch(bad)
    with pytest.raises(ValueError):
        sample_ratio_mismatch([5, 5], expected_ratios=[1.0])
    with pytest.raises(ValueError):
        sample_ratio_mismatch([5, 5], expected_ratios=[1.0, 0.0])
    assert sample_ratio_mismatch([3, 2])["small_sample"] is True


def _exposures(n_control, n_treat):
    rows = []
    for v, n in (("control", n_control), ("treatment", n_treat)):
        for i in range(n):
            rows.append({"exposure_id": f"x_{v}_{i}", "user_id": f"{v}_{i}", "feature_key": "f", "variant": v,
                         "first_exposed_timestamp": pd.Timestamp("2026-09-02T00:00:00Z")})
    return pd.DataFrame(rows)


def _events(n_control, n_treat, treat_rate=0.5, ctrl_rate=0.3):
    rows = []
    for v, n, rate in (("control", n_control, ctrl_rate), ("treatment", n_treat, treat_rate)):
        for i in range(int(n * rate)):
            rows.append({"event_id": f"e_{v}_{i}", "user_id": f"{v}_{i}", "session_id": "s", "event_name": "order_completed",
                         "timestamp": pd.Timestamp("2026-09-03T00:00:00Z")})
    return pd.DataFrame(rows, columns=["event_id", "user_id", "session_id", "event_name", "timestamp"])


def _effect(n_c, n_t):
    loader = DataLoader.from_dataframes({"feature_exposures": _exposures(n_c, n_t), "events": _events(n_c, n_t), "users": U})
    return AnalyticsExecutor(loader).run(AnalysisRequest(
        "exposure_effect", W("2026-09-01", "2026-09-30"),
        params={"feature_key": "f", "outcome_event": "order_completed", "post_exposure_days": 3}))


def test_exposure_effect_withholds_the_pvalue_after_a_sample_ratio_mismatch():
    res = _effect(600, 400)
    assert res.ok() and res.metadata["srm"]["mismatch"] is True and res.metadata["trusted"] is False
    assert res.metadata["p_value"] is None and res.metadata["is_significant_p05"] is None
    assert any("SAMPLE RATIO MISMATCH" in w for w in res.warnings)


def test_exposure_effect_reports_pvalue_when_assignment_is_balanced():
    res = _effect(500, 500)
    assert res.metadata["srm"]["mismatch"] is False and res.metadata["trusted"] is True
    assert res.metadata["p_value"] is not None and res.metadata["p_value"] < 0.001


def test_sample_ratio_check_operation():
    loader = DataLoader.from_dataframes({"feature_exposures": _exposures(600, 400), "users": U})
    ex = AnalyticsExecutor(loader)
    res = ex.run(AnalysisRequest("sample_ratio_check", W("2026-09-01", "2026-09-30"), params={"feature_key": "f"}))
    assert res.ok() and res.metadata["srm"]["mismatch"] and res.warnings
    assert res.metadata["p_values"]["srm_chi_square"] < 1e-6
    assert ex.run(AnalysisRequest("sample_ratio_check", W("2026-09-01", "2026-09-30"), params={})).status == AnalysisStatus.INVALID_PARAMS
    assert ex.run(AnalysisRequest("sample_ratio_check", W("2026-09-01", "2026-09-30"),
                                  params={"feature_key": "zzz"})).status == AnalysisStatus.INSUFFICIENT_DATA
    assert ex.run(AnalysisRequest("sample_ratio_check", W("2026-09-01", "2026-09-30"),
                                  params={"feature_key": "f", "expected_ratio": [1.0]})).status == AnalysisStatus.INVALID_PARAMS


# ------------------------------------------------------------------ delta method (hand computed)

def test_delta_method_equals_standard_error_of_the_mean_when_denominator_is_constant():
    r = delta_method_ratio([1, 3], [1, 1])      # ratio = 2 ; mean of x with sd sqrt(2), n=2 -> SE = 1
    assert r["ratio"] == pytest.approx(2.0) and r["se"] == pytest.approx(1.0)


def test_delta_method_hand_computed_with_covariance():
    # x=[2,4,9], y=[1,2,3]: R=2.5, ybar=2, sx2=13, sy2=1, cov=3.5 -> var=(13+6.25-17.5)/(3*4)=0.145833
    r = delta_method_ratio([2, 4, 9], [1, 2, 3])
    assert r["ratio"] == pytest.approx(2.5) and r["variance"] == pytest.approx(1.75 / 12) and r["se"] == pytest.approx(0.381881, abs=1e-6)


def test_perfectly_proportional_units_have_zero_variance():
    assert delta_method_ratio([2, 4, 6], [1, 2, 3])["se"] == pytest.approx(0.0, abs=1e-9)


def test_naive_mean_of_ratios_is_not_the_delta_method_ratio():
    x, y = [1, 10], [1, 10]
    assert delta_method_ratio([1, 10], [10, 10])["ratio"] == pytest.approx(0.55)      # ratio of totals 11/20
    assert np.mean([a / b for a, b in zip([1, 10], [10, 10])]) == pytest.approx(0.55)  # coincide here...
    assert delta_method_ratio([1, 10], [1, 100])["ratio"] == pytest.approx(11 / 101)   # ...but not in general
    assert np.mean([1 / 1, 10 / 100]) != pytest.approx(11 / 101)


def test_ratio_comparison_hand_computed():
    r = compare_ratio_metrics([1, 3], [1, 1], [1, 1], [1, 1])    # 2.0 vs 1.0 ; se1 = 1, se0 = 0
    assert r["difference"] == pytest.approx(1.0) and r["se_difference"] == pytest.approx(1.0)
    assert r["z"] == pytest.approx(1.0) and r["p_value"] == pytest.approx(2 * stats.norm.sf(1.0))
    assert (r["ci_lower"], r["ci_upper"]) == pytest.approx((1.0 - 1.959964, 1.0 + 1.959964), abs=1e-5)


def test_delta_method_edge_cases():
    with pytest.raises(ValueError):
        delta_method_ratio([1], [1])
    with pytest.raises(ValueError):
        delta_method_ratio([1, 2], [0, 0])
    with pytest.raises(ValueError):
        delta_method_ratio([1, 2], [1])
    same = compare_ratio_metrics([1, 3], [1, 1], [1, 3], [1, 1])
    assert same["difference"] == 0 and same["p_value"] == pytest.approx(1.0)
    flat = compare_ratio_metrics([2, 4], [1, 2], [1, 2], [1, 2])      # zero variance both, unequal ratios
    assert flat["p_value"] == 0.0


def test_ratio_metric_operation_end_to_end():
    rng = np.random.default_rng(5)
    ev, od = [], []
    for period, day, scale in (("b", 3, 30.0), ("c", 13, 20.0)):
        for u in range(60):
            uid = f"{period}{u}"
            for s in range(1 + u % 3):
                ts = pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z")
                ev.append({"event_id": f"e{len(ev)}", "user_id": uid, "session_id": f"{uid}_{s}", "timestamp": ts})
                if rng.random() < 0.5:
                    od.append({"order_id": f"o{len(od)}", "user_id": uid, "timestamp": ts, "status": "completed", "amount": scale})
    loader = DataLoader.from_dataframes({"events": pd.DataFrame(ev), "orders": pd.DataFrame(od), "users": U})
    res = AnalyticsExecutor(loader).run(AnalysisRequest(
        "ratio_metric_comparison", W("2026-09-10", "2026-09-20"), comparison_window=W("2026-09-01", "2026-09-10"),
        params={"metric_name": "revenue_per_session"}))
    assert res.ok(), res.error_message
    assert res.metadata["absolute_change"] < 0 and res.metadata["p_values"]["ratio_change"] < 0.05
    bad = AnalyticsExecutor(loader).run(AnalysisRequest(
        "ratio_metric_comparison", W("2026-09-10", "2026-09-20"), comparison_window=W("2026-09-01", "2026-09-10"),
        params={"metric_name": "nonsense"}))
    assert bad.status == AnalysisStatus.INVALID_PARAMS


# ------------------------------------------------------------------ catalogue exposure

@pytest.mark.parametrize("op", ["change_point", "release_did", "sample_ratio_check", "ratio_metric_comparison"])
def test_phase2b_operations_are_exposed(op):
    from controller.capability_catalogue import validate_analysis_plan
    assert op in CONTROLLED_OPERATIONS
    ok, err = validate_analysis_plan(op, CONTROLLED_OPERATIONS[op].example)
    assert ok, err


def test_phase2b_parameter_validation():
    from controller.capability_catalogue import validate_analysis_plan
    assert not validate_analysis_plan("change_point", {"aggregation": "count", "min_segment": 1})[0]
    assert not validate_analysis_plan("change_point", {"aggregation": "count", "n_permutations": 10**6})[0]
    assert not validate_analysis_plan("release_did", {"release_id": "r", "window_days": 0})[0]
    assert not validate_analysis_plan("release_did", {})[0]
    assert not validate_analysis_plan("sample_ratio_check", {})[0]


# ------------------------------------------------------------------ edge-case sweep over EVERY operation

EXAMPLE_PARAMS = {
    **{name: dict(spec.example) for name, spec in CONTROLLED_OPERATIONS.items()},
    "metric_aggregate": {"metric_name": "order_conversion_rate"},
}


def _empty_tables():
    return {name: pd.DataFrame({f.name: pd.Series(dtype="object") for f in info.fields}) for name, info in CATALOGUE.items()}


def _ok_status(res):
    return res.status in (AnalysisStatus.OK, AnalysisStatus.INSUFFICIENT_DATA, AnalysisStatus.INVALID_PARAMS,
                          AnalysisStatus.DATA_QUALITY)


@pytest.mark.parametrize("op", sorted(CONTROLLED_OPERATIONS))
def test_every_operation_survives_empty_tables(op):
    ex = AnalyticsExecutor(DataLoader.from_dataframes(_empty_tables()))
    spec = CONTROLLED_OPERATIONS[op]
    req = AnalysisRequest(op, W("2026-09-10", "2026-09-20"), params=dict(EXAMPLE_PARAMS[op]),
                          comparison_window=W("2026-09-01", "2026-09-10"))
    res = ex.run(req)
    assert _ok_status(res), f"{op}: {res.status} {res.error_message}"


@pytest.mark.parametrize("op", sorted(CONTROLLED_OPERATIONS))
def test_every_operation_survives_a_single_day_window_on_real_data(op):
    from pathlib import Path
    d = Path(__file__).resolve().parents[2] / "data" / "baseline" / "investigator_data"
    ex = AnalyticsExecutor(DataLoader(data_dir=d))
    req = AnalysisRequest(op, W("2026-09-20", "2026-09-21"), params=dict(EXAMPLE_PARAMS[op]),
                          comparison_window=W("2026-09-19", "2026-09-20"))
    res = ex.run(req)
    assert _ok_status(res), f"{op}: {res.status} {res.error_message}"


@pytest.mark.parametrize("op", sorted(CONTROLLED_OPERATIONS))
def test_every_operation_survives_a_window_with_no_data(op):
    from pathlib import Path
    d = Path(__file__).resolve().parents[2] / "data" / "baseline" / "investigator_data"
    ex = AnalyticsExecutor(DataLoader(data_dir=d))
    req = AnalysisRequest(op, W("2027-01-02", "2027-01-09"), params=dict(EXAMPLE_PARAMS[op]),
                          comparison_window=W("2027-01-01", "2027-01-02"))
    res = ex.run(req)
    assert _ok_status(res), f"{op}: {res.status} {res.error_message}"


def test_zero_denominator_rates_do_not_crash():
    df = pd.DataFrame([{"order_id": "o1", "user_id": "u", "status": "failed", "timestamp": pd.Timestamp("2026-09-12T00:00:00Z")}])
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"orders": df, "users": U}))
    res = ex.run(AnalysisRequest("period_comparison", W("2026-09-10", "2026-09-20"), comparison_window=W("2026-09-01", "2026-09-10"),
                                 params={"table": "orders", "aggregation": "rate", "metric_column": "status", "metric_value": "completed"}))
    assert _ok_status(res)
    assert res.metadata["p_values"] == {}                    # baseline period is empty: no test, no crash
    assert res.metadata["relative_change"] is None           # zero baseline: relative change undefined, not inf
