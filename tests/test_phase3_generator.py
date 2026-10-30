"""Phase 3: generator realism knobs, the eight new scenarios, release timing, validators, ground-truth separation."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from investigator_data.config import GeneratorConfig
from investigator_data.engine import BOT_BROWSER, SyntheticDataEngine
from investigator_data.scenarios import (
    ALERT_METRICS,
    ORIGINAL_SCENARIOS,
    SCENARIO_REGISTRY,
    build_config,
    get_baseline_scenario,
    get_product_regression_scenario,
)
from investigator_data.validator import DataValidator

DATA = Path(__file__).resolve().parents[1] / "data"
NEW = ["noise_only", "seasonal_dip", "pricing_change", "decoy_release", "overlapping_causes",
       "gradual_decay", "denominator_effect", "third_party_outage"]


def _gen(scn, cfg=None, seed=42):
    cfg = cfg or build_config(scn, seed=seed)
    eng = SyntheticDataEngine(cfg, scn)
    return eng.generate_all(), eng


@pytest.fixture(scope="module")
def worlds():
    out = {}
    for key, factory in SCENARIO_REGISTRY.items():
        scn = factory()
        tables, eng = _gen(scn)
        out[key] = (scn, tables, eng)
    return out


def _ts(df, col="timestamp"):
    return pd.to_datetime(df[col], utc=True)


def _daily(df):
    return df.groupby(_ts(df).dt.date).size()


# ------------------------------------------------------------------ registry & validation

def test_registry_has_all_scenarios():
    assert set(SCENARIO_REGISTRY) == set(ORIGINAL_SCENARIOS) | set(NEW)
    assert len(SCENARIO_REGISTRY) == 12


@pytest.mark.parametrize("key", sorted(SCENARIO_REGISTRY))
def test_every_scenario_validates_clean(worlds, key):
    scn, tables, eng = worlds[key]
    ok, errors = DataValidator(tables, scn, eng.config).validate_all()
    assert ok, errors


@pytest.mark.parametrize("key", sorted(SCENARIO_REGISTRY))
def test_every_scenario_declares_an_expected_answer_and_alert(worlds, key):
    scn = worlds[key][0]
    assert "primary_cause" in scn.expected_answer and "insufficient_evidence" in scn.expected_answer
    assert scn.alert["metric"] in ALERT_METRICS
    assert ("baseline_window" in scn.alert and "comparison_window" in scn.alert) or scn.alert.get("mode") == "worst_adjacent_week"


def test_new_scenarios_have_the_required_answers():
    r = {k: SCENARIO_REGISTRY[k]().expected_answer for k in NEW}
    assert r["noise_only"]["primary_cause"] == "natural_variation"
    assert r["pricing_change"]["insufficient_evidence"] is True and r["pricing_change"]["primary_cause"] is None
    assert r["decoy_release"]["primary_cause"] == "traffic_mix_shift" and "code_regression" in r["decoy_release"]["decoys"]
    assert set([r["overlapping_causes"]["primary_cause"]] + r["overlapping_causes"]["secondary_causes"]) == {"code_regression", "tracking_loss"}
    assert r["gradual_decay"]["primary_cause"] == "gradual_or_cohort_effect"
    assert r["denominator_effect"]["primary_cause"] == "denominator_effect"
    assert r["third_party_outage"]["primary_cause"] == "third_party_outage"
    assert r["seasonal_dip"]["primary_cause"] == "seasonality_or_calendar"


# ------------------------------------------------------------------ reproducibility & separation

def test_generation_is_seeded_and_reproducible():
    scn = SCENARIO_REGISTRY["overlapping_causes"]()
    a, ea = _gen(scn)
    b, eb = _gen(scn)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
    assert ea.truth == eb.truth
    c, _ = _gen(scn, seed=7)
    assert not a["events"].equals(c["events"])


def test_all_knobs_on_is_still_reproducible():
    cfg = GeneratorConfig(noise_level=0.2, late_arrival_rate=0.1, duplicate_event_rate=0.05, bot_traffic_share=0.1,
                          identity_split_share=0.1, event_renames=(("v2.4.0", "order_completed", "purchase_success"),),
                          effect_onset="ramp", holiday_effects=(("2026-09-10", 0.5),))
    scn = SCENARIO_REGISTRY["product_regression"]()
    a, _ = _gen(scn, cfg)
    b, _ = _gen(scn, cfg)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


@pytest.mark.skipif(not (DATA / "noise_only").exists(), reason="tracked data missing")
@pytest.mark.parametrize("key", sorted(SCENARIO_REGISTRY))
def test_ground_truth_is_physically_separate_and_alert_is_not_ground_truth(key):
    base = DATA / key
    assert (base / "ground_truth" / "scenario_ground_truth.json").exists()
    assert (base / "alert.json").exists()
    assert not (base / "investigator_data" / "scenario_ground_truth.json").exists()
    alert = json.loads((base / "alert.json").read_text(encoding="utf-8"))
    assert set(alert) <= {"metric", "baseline_window", "comparison_window", "mode"}      # nothing about causes
    gt = json.loads((base / "ground_truth" / "scenario_ground_truth.json").read_text(encoding="utf-8"))
    assert "expected_answer" in gt and "alert" not in gt
    for table in (base / "investigator_data").glob("*.csv"):
        text = table.read_text(encoding="utf-8")
        assert "ground_truth" not in text and key not in text.replace("_", "") or key in ("baseline",)


# ------------------------------------------------------------------ release timing varies by scenario

def test_no_hard_coded_release_cutoff_in_the_engine():
    src = (Path(__file__).resolve().parents[1] / "src" / "investigator_data" / "engine.py").read_text(encoding="utf-8")
    assert "2026, 9, 15, 10" not in src and "2026, 9, 8, 14" not in src


def test_app_versions_follow_the_scenarios_own_release_log(worlds):
    for key in ("baseline", "decoy_release"):
        scn, tables, _ = worlds[key]
        rel = tables["releases"]
        assert set(tables["events"]["app_version"]) <= set(rel["version"])
    assert "v2.5.0" in set(worlds["decoy_release"][1]["events"]["app_version"])
    assert "v2.5.0" not in set(worlds["baseline"][1]["events"]["app_version"])


def test_moving_the_release_moves_the_regression():
    scn = get_product_regression_scenario()
    moved = copy.deepcopy(scn)
    moved.parameters["deploy_time"] = "2026-09-22T10:00:00Z"
    moved.releases = [dict(r, deployed_timestamp="2026-09-22T10:00:00+00:00") if r["release_id"] == "rel_240" else r
                      for r in __import__("investigator_data.scenarios", fromlist=["x"]).default_releases()]
    tables, _ = _gen(moved)
    gate = tables["orders"][tables["orders"]["failure_reason"] == "payment_gateway_timeout"]
    assert not gate.empty and _ts(gate).min() >= pd.Timestamp("2026-09-22T10:00:00Z")
    base_tables, _ = _gen(scn)
    base_gate = base_tables["orders"][base_tables["orders"]["failure_reason"] == "payment_gateway_timeout"]
    assert _ts(base_gate).min() < pd.Timestamp("2026-09-18T00:00:00Z")


def test_mobile_users_adopt_the_new_version_gradually(worlds):
    ev = worlds["product_regression"][1]["events"]
    after = ev[(_ts(ev) >= pd.Timestamp("2026-09-15T10:00:00Z")) & (_ts(ev) < pd.Timestamp("2026-09-16T10:00:00Z"))]
    mobile, desktop = after[after.device_type == "mobile"], after[after.device_type == "desktop"]
    assert (desktop.app_version == "v2.4.0").mean() > 0.95            # web updates instantly
    assert 0.2 < (mobile.app_version == "v2.4.0").mean() < 0.9        # mobile app updates trickle in


def test_instant_adoption_when_configured():
    scn = get_product_regression_scenario()
    tables, _ = _gen(scn, replace(build_config(scn), app_adoption_days_mobile=0.0))
    ev = tables["events"]
    after = ev[(_ts(ev) >= pd.Timestamp("2026-09-15T10:00:00Z")) & (_ts(ev) < pd.Timestamp("2026-09-16T10:00:00Z"))]
    assert (after.app_version == "v2.4.0").all()


# ------------------------------------------------------------------ knobs

STATIONARY = dict(existing_user_share=0.8, num_users=600)


def _base(**kw):
    scn = get_baseline_scenario()
    return _gen(scn, GeneratorConfig(**{**STATIONARY, **kw}))[0]


def test_baseline_has_the_weekly_seasonality_it_claims():
    ev = _base()["events"]
    d = _daily(ev)
    wd = np.array([x.weekday() for x in d.index])
    assert d.to_numpy()[wd >= 5].mean() < 0.85 * d.to_numpy()[wd < 5].mean()
    flat = _base(weekly_seasonality=0.0)["events"]
    df = _daily(flat)
    wf = np.array([x.weekday() for x in df.index])
    assert abs(df.to_numpy()[wf >= 5].mean() / df.to_numpy()[wf < 5].mean() - 1) < 0.2


def test_seasonality_strength_scales_the_weekend_dip():
    ratios = []
    for strength in (0.0, 0.5, 1.0):
        d = _daily(_base(weekly_seasonality=strength)["events"])
        wd = np.array([x.weekday() for x in d.index])
        ratios.append(d.to_numpy()[wd >= 5].mean() / d.to_numpy()[wd < 5].mean())
    assert ratios[0] > ratios[1] > ratios[2]


def test_noise_level_increases_day_to_day_dispersion():
    cv = lambda t: (lambda d: d.std() / d.mean())(_daily(t["events"]).iloc[3:-3])
    assert cv(_base(noise_level=0.5, weekly_seasonality=0.0)) > 1.5 * cv(_base(noise_level=0.0, weekly_seasonality=0.0))


def test_holiday_dip_and_spike():
    d = _daily(_base(weekly_seasonality=0.0, holiday_effects=(("2026-09-12", 0.2), ("2026-09-20", 2.0)))["events"])
    neighbours = lambda day: d[[x for x in d.index if abs((x - day).days) in (2, 3)]].mean()
    dip, spike = pd.Timestamp("2026-09-12").date(), pd.Timestamp("2026-09-20").date()
    assert d[dip] < 0.6 * neighbours(dip)
    assert d[spike] > 1.2 * neighbours(spike)


def test_late_arriving_events_vanish_from_the_tail():
    on, off = _base(late_arrival_rate=0.5, late_arrival_mean_hours=24.0)["events"], _base()["events"]
    last_day = lambda e: _daily(e).iloc[-1] / _daily(e).iloc[-8:-1].mean()
    assert last_day(on) < 0.9 * last_day(off)
    assert len(on) < len(off)


def test_duplicate_events_are_generated_at_the_configured_rate():
    ev = _base(duplicate_event_rate=0.1)["events"]
    dup = ev.duplicated(subset=["user_id", "session_id", "timestamp", "event_name"]).sum()
    assert 0.07 < dup / len(ev) < 0.13
    assert ev["event_id"].is_unique
    assert not _base()["events"].duplicated(subset=["user_id", "session_id", "timestamp", "event_name"]).any()


def test_bot_traffic_share_and_signature():
    t = _base(bot_traffic_share=0.2)
    bots = t["users"][t["users"].browser == BOT_BROWSER]
    assert 0.17 < len(bots) / len(t["users"]) < 0.23
    bot_events = t["events"][t["events"].browser == BOT_BROWSER]
    assert len(bot_events) > 0 and set(bot_events.event_name) == {"page_view"}
    assert not t["orders"]["user_id"].isin(bots.user_id).any()
    assert not t["feature_exposures"]["user_id"].isin(bots.user_id).any()


def test_identity_split_creates_twin_accounts():
    t = _base(identity_split_share=0.3)
    u = t["users"]
    twins = u.groupby(["signup_timestamp", "acquisition_channel", "country", "plan_tier"]).filter(lambda g: len(g) > 1)
    assert 0.2 < len(twins) / 2 / (len(u) - len(twins) / 2) < 0.4
    for _, g in twins.groupby(["signup_timestamp", "acquisition_channel", "country", "plan_tier"]):
        assert g.device_type.nunique() == 2


def test_event_names_change_across_app_versions():
    t = _base(event_renames=(("v2.4.0", "order_completed", "purchase_success"),))
    ev = t["events"]
    new, old = ev[ev.event_name == "purchase_success"], ev[ev.event_name == "order_completed"]
    key = lambda v: tuple(int(x) for x in v.lstrip("v").split("."))
    assert len(new) > 0 and len(old) > 0
    assert all(key(v) >= (2, 4, 0) for v in new.app_version.unique())
    assert all(key(v) < (2, 4, 0) for v in old.app_version.unique())


def test_effect_size_knob_scales_the_failure_effect():
    scn = get_product_regression_scenario()
    counts = {}
    for size in (0.1, 0.4, 0.9):
        t, e = _gen(scn, replace(build_config(scn), effect_size=size))
        counts[size] = e.truth["regression_failed_orders"]
    assert counts[0.1] < counts[0.4] < counts[0.9]
    t0, e0 = _gen(scn, replace(build_config(scn), effect_size=0.0))
    assert e0.truth["regression_failed_orders"] == 0


def test_ramp_onset_starts_gently_and_step_starts_at_once():
    scn = get_product_regression_scenario()
    def first_day_failure_share(onset):
        t, _ = _gen(scn, replace(build_config(scn), effect_onset=onset, effect_ramp_days=8.0, app_adoption_days_mobile=0.0))
        o = t["orders"]
        d = o[(_ts(o) >= pd.Timestamp("2026-09-15T10:00:00Z")) & (_ts(o) < pd.Timestamp("2026-09-16T10:00:00Z"))]
        late = o[(_ts(o) >= pd.Timestamp("2026-09-23T10:00:00Z")) & (_ts(o) < pd.Timestamp("2026-09-24T10:00:00Z"))]
        f = lambda x: (x.failure_reason == "payment_gateway_timeout").mean()
        return f(d), f(late)
    s_first, s_late = first_day_failure_share("step")
    r_first, r_late = first_day_failure_share("ramp")
    assert r_first < 0.5 * s_first          # ramp: small at the start
    assert r_late > 0.5 * s_late            # ...but reaches full strength after the ramp length


def test_config_validation_rejects_bad_knobs():
    for bad in (dict(effect_onset="sudden"), dict(late_arrival_rate=1.0), dict(bot_traffic_share=-0.1),
                dict(weekly_seasonality=2.0), dict(noise_level=-1), dict(effect_size=1.5),
                dict(effect_ramp_days=0), dict(existing_user_share=1.0)):
        with pytest.raises(ValueError):
            GeneratorConfig(**bad)


# ------------------------------------------------------------------ new scenarios leave their fingerprints

def test_pricing_change_cuts_conversion_with_no_other_signal(worlds):
    scn, t, _ = worlds["pricing_change"]
    cut = pd.Timestamp("2026-09-16T00:00:00Z")
    ev = t["events"]
    cs = ev[ev.event_name == "checkout_started"]
    pre = (_ts(cs) < cut).sum() / 15
    post = (_ts(cs) >= cut).sum() / 15
    assert post < 0.75 * pre
    assert set(t["orders"].failure_reason.dropna()) <= {"card_declined"}            # no failures to blame
    assert set(t["releases"].release_id) == {"rel_238", "rel_239", "rel_240", "rel_241"}
    assert not any(c in scn.parameters for c in ("surge_start", "deploy_time", "tracking_drop_start"))


def test_third_party_outage_is_short_and_spans_all_versions(worlds):
    _, t, e = worlds["third_party_outage"]
    out = t["orders"][t["orders"].failure_reason == "payment_processor_unavailable"]
    assert len(out) >= 10 and e.truth["outage_failed_orders"] == len(out)
    ts = _ts(out)
    assert ts.min() >= pd.Timestamp("2026-09-20T10:00:00Z") and ts.max() < pd.Timestamp("2026-09-20T22:00:00Z")
    ev = t["events"][t["events"].user_id.isin(out.user_id)]
    assert ev.app_version.nunique() >= 2 and ev.device_type.nunique() >= 2


def test_denominator_effect_overall_rate_rises_but_segment_rates_do_not(worlds):
    _, t, _ = worlds["denominator_effect"]
    cut = pd.Timestamp("2026-09-15T00:00:00Z")
    users = t["users"].set_index("user_id")
    ev, od = t["events"], t["orders"]
    sess = ev.drop_duplicates(["user_id", "session_id"])
    sess = sess.assign(channel=sess.user_id.map(users.acquisition_channel))
    comp = od[od.status == "completed"].assign(channel=lambda d: d.user_id.map(users.acquisition_channel))
    def rate(lo, hi, ch=None):
        s = sess[(_ts(sess) >= lo) & (_ts(sess) < hi)]; o = comp[(_ts(comp) >= lo) & (_ts(comp) < hi)]
        if ch: s, o = s[s.channel == ch], o[o.channel == ch]
        return len(o) / len(s)
    start, end = pd.Timestamp("2026-09-01T00:00:00Z"), pd.Timestamp("2026-10-01T00:00:00Z")
    assert rate(cut, end) > 1.08 * rate(start, cut)
    other = lambda lo, hi: sum(rate(lo, hi, c) for c in ("organic_search", "direct")) / 2
    assert abs(other(cut, end) / other(start, cut) - 1) < 0.2


def test_gradual_decay_declines_without_a_single_step(worlds):
    _, t, _ = worlds["gradual_decay"]
    d = _daily(t["events"]).to_numpy().astype(float)
    n = len(d) // 3
    assert d[n:2 * n].mean() < d[:n].mean() and d[2 * n:].mean() < d[n:2 * n].mean()
    assert np.polyfit(np.arange(len(d)), d, 1)[0] < 0
    from analytics.changepoint import detect_change_point
    cp = detect_change_point(d, 3, 99)
    assert cp["onset_shape"] == "gradual"


def test_seasonal_dip_weekend_and_holiday(worlds):
    _, t, _ = worlds["seasonal_dip"]
    d = _daily(t["events"])
    wd = np.array([x.weekday() for x in d.index])
    assert d.to_numpy()[wd >= 5].mean() < 0.6 * d.to_numpy()[wd < 5].mean()
    hol = pd.Timestamp("2026-09-21").date()
    same = d[[x for x in d.index if x.weekday() == hol.weekday() and x != hol]]
    assert d[hol] < 0.7 * same.mean()


def test_overlapping_causes_record_both_effects_for_share_ground_truth(worlds):
    _, t, e = worlds["overlapping_causes"]
    assert e.truth["regression_failed_orders"] > 20 and e.truth["tracking_dropped_events"] > 20
    safari = t["orders"].merge(t["users"][["user_id", "browser"]], on="user_id")
    assert (safari[(safari.browser == "Safari")].status == "completed").any()      # Safari backend orders still succeed


def test_decoy_release_does_not_change_checkout_failures(worlds):
    _, t, e = worlds["decoy_release"]
    assert "rel_250" in set(t["releases"].release_id)
    assert e.truth["regression_failed_orders"] == 0
    users = t["users"]
    su = _ts(users, "signup_timestamp")
    assert (users[su >= pd.Timestamp("2026-09-14T00:00:00Z")].acquisition_channel == "paid_social").mean() > 0.2


def test_noise_only_has_no_true_shift_but_large_dispersion(worlds):
    _, t, e = worlds["noise_only"]
    assert sum(e.truth.values()) == 0
    d = _daily(t["orders"])
    assert d.std() / d.mean() > 0.2


# ------------------------------------------------------------------ validators catch broken data

def _valid(key="overlapping_causes"):
    scn = SCENARIO_REGISTRY[key]()
    tables, eng = _gen(scn)
    return scn, {k: v.copy() for k, v in tables.items()}, eng.config


def test_validator_flags_a_leaked_extra_column():
    scn, t, cfg = _valid("baseline")
    t["orders"]["browser"] = "Chrome"
    ok, errs = DataValidator(t, scn, cfg).validate_all()
    assert not ok and any("unexpected columns" in e for e in errs)


def test_validator_flags_versions_missing_from_the_release_log():
    scn, t, cfg = _valid("baseline")
    t["events"].loc[0, "app_version"] = "v9.9.9"
    assert any("missing from releases" in e for e in DataValidator(t, scn, cfg).check_release_consistency())


def test_validator_flags_events_before_their_release():
    scn, t, cfg = _valid("baseline")
    i = t["events"].index[t["events"].app_version == "v2.4.1"][0]
    t["events"].loc[i, "timestamp"] = "2026-09-02T00:00:00+00:00"
    assert any("before its deployment" in e for e in DataValidator(t, scn, cfg).check_release_consistency())


@pytest.mark.parametrize("knob,msg", [
    (dict(duplicate_event_rate=0.1), "no duplicated events"),
    (dict(bot_traffic_share=0.1), "no bot users"),
    (dict(identity_split_share=0.1), "no split identities"),
    (dict(event_renames=(("v2.4.0", "order_completed", "purchase_success"),)), "never appears"),
])
def test_validator_flags_configured_knobs_that_left_no_fingerprint(knob, msg):
    scn, t, cfg = _valid("baseline")
    errs = DataValidator(t, scn, replace(cfg, **knob)).check_config_knobs()
    assert any(msg in e for e in errs), errs


def test_validator_flags_fingerprints_without_the_knob():
    scn = get_baseline_scenario()
    t, eng = _gen(scn, replace(build_config(scn), bot_traffic_share=0.1, duplicate_event_rate=0.1))
    errs = DataValidator(t, scn, replace(eng.config, bot_traffic_share=0.0, duplicate_event_rate=0.0)).check_config_knobs()
    assert any("bot users present" in e for e in errs) and any("duplicated events present" in e for e in errs)


def test_validator_flags_outage_outside_its_window():
    scn, t, cfg = _valid("third_party_outage")
    i = t["orders"].index[t["orders"].failure_reason == "payment_processor_unavailable"][0]
    t["orders"].loc[i, "timestamp"] = "2026-09-25T00:00:00+00:00"
    ok, errs = DataValidator(t, scn, cfg).validate_all()
    assert any("outside the configured window" in e for e in errs)


def test_validator_flags_a_flat_world_that_claims_seasonality():
    scn = get_baseline_scenario()
    t, eng = _gen(scn, GeneratorConfig(weekly_seasonality=0.0, **STATIONARY))
    errs = DataValidator(t, scn, replace(eng.config, weekly_seasonality=1.0)).check_config_knobs()
    assert any("weekly seasonality" in e for e in errs)
