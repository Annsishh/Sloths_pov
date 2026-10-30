"""Scenario observable effect test suite.

Ensures that every test scenario generates its intended observable fingerprints in data,
verified with rigorous quantitative statistical thresholds.
"""

from datetime import datetime
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import (
    get_acquisition_shift_scenario,
    get_baseline_scenario,
    get_product_regression_scenario,
    get_tracking_failure_scenario,
)
from investigator_data.validator import DataValidator


def test_product_regression_scenario_quantitative():
    """Quantitative verification of Mobile Checkout Gateway Bug:
    1. Overall mobile order failure rate after deploy_time must be close to configured rate (75% +/- 10%).
    2. Non-mobile order failure rate after deploy_time must remain at baseline (< 10%).
    3. Mobile order failure rate before deploy_time must remain at baseline (< 10%).
    4. Negative feedback (ratings 1-2) with category='billing' must surge on mobile post-deploy.
    """
    scenario = get_product_regression_scenario()
    config = GeneratorConfig(seed=42, num_users=1000)
    engine = SyntheticDataEngine(config=config, scenario=scenario)
    tables = engine.generate_all()

    # Pass validator check with explicit generator config
    validator = DataValidator(tables=tables, scenario=scenario, config=config)
    is_valid, errors = validator.validate_all()
    assert is_valid, f"Scenario validation failed: {errors}"

    orders_df = tables["orders"]
    users_df = tables["users"]
    feedback_df = tables["user_feedback"]

    user_device = dict(zip(users_df["user_id"], users_df["device_type"]))
    deploy_time_str = scenario.parameters["deploy_time"]
    target_failure_prob = scenario.parameters["mobile_order_failure_prob"]

    orders_df["device"] = orders_df["user_id"].map(user_device)

    # Pre-deployment vs Post-deployment splits
    mobile_post = orders_df[(orders_df["device"] == "mobile") & (orders_df["timestamp"] >= deploy_time_str)]
    mobile_pre = orders_df[(orders_df["device"] == "mobile") & (orders_df["timestamp"] < deploy_time_str)]
    desktop_post = orders_df[(orders_df["device"] != "mobile") & (orders_df["timestamp"] >= deploy_time_str)]

    assert len(mobile_post) >= 20, "Insufficient mobile orders post-deployment for statistical validation"

    # 1. Post-deploy mobile failure rate quantitative check
    mobile_post_fail_rate = (mobile_post["status"] == "failed").mean()
    assert abs(mobile_post_fail_rate - target_failure_prob) < 0.12, (
        f"Mobile post-deploy failure rate {mobile_post_fail_rate:.2f} differs significantly from target {target_failure_prob}"
    )

    # 2. Specific failure reason check
    timeout_failures = mobile_post[mobile_post["failure_reason"] == "payment_gateway_timeout"]
    assert len(timeout_failures) / len(mobile_post) > 0.60, "Expected majority of mobile failures to be payment_gateway_timeout"

    # 3. Non-mobile orders and pre-deployment mobile orders must remain healthy (baseline failure ~4%)
    desktop_post_fail_rate = (desktop_post["status"] == "failed").mean()
    assert desktop_post_fail_rate < 0.10, f"Desktop post-deploy failure rate {desktop_post_fail_rate:.2f} too high"

    if len(mobile_pre) > 0:
        mobile_pre_fail_rate = (mobile_pre["status"] == "failed").mean()
        assert mobile_pre_fail_rate < 0.10, f"Mobile pre-deploy failure rate {mobile_pre_fail_rate:.2f} too high"

    # 4. User feedback quantitative impact
    feedback_df["device"] = feedback_df["user_id"].map(user_device)
    mobile_post_fb = feedback_df[(feedback_df["device"] == "mobile") & (feedback_df["timestamp"] >= deploy_time_str)]
    low_ratings = mobile_post_fb[mobile_post_fb["rating"] <= 2]
    assert len(low_ratings) >= 5, "Expected noticeable count of negative ratings from mobile users post-bug"
    billing_issues = low_ratings[low_ratings["category"] == "billing"]
    assert len(billing_issues) > 0, "Expected billing category complaints in mobile feedback"


def test_acquisition_shift_scenario_quantitative():
    """Quantitative verification of Paid Social Traffic Surge:
    1. paid_social_checkout_start_prob directly controls checkout initiation for paid_social sessions.
    2. Post-surge paid_social share of signups increases significantly (> 1.8x pre-surge share).
    3. Checkout initiation rate per session for paid_social is distinctly lower than baseline (organic/direct).
    4. Paid social conversion to completed orders drops quantitatively to ~1.2%.
    5. Organic/direct maintain steady conversion (> 5%).
    """
    scenario = get_acquisition_shift_scenario()
    config = GeneratorConfig(seed=42, num_users=1000)
    engine = SyntheticDataEngine(config=config, scenario=scenario)
    tables = engine.generate_all()

    validator = DataValidator(tables=tables, scenario=scenario, config=config)
    is_valid, errors = validator.validate_all()
    assert is_valid, f"Scenario validation failed: {errors}"

    users_df = tables["users"]
    orders_df = tables["orders"]
    events_df = tables["events"]

    surge_start = scenario.parameters["surge_start"]
    configured_checkout_prob = scenario.parameters["paid_social_checkout_start_prob"]

    pre_users = users_df[users_df["signup_timestamp"] < surge_start]
    post_users = users_df[users_df["signup_timestamp"] >= surge_start]

    # 1. Share of paid_social users
    pre_social_pct = (pre_users["acquisition_channel"] == "paid_social").mean()
    post_social_pct = (post_users["acquisition_channel"] == "paid_social").mean()

    assert 0.05 <= pre_social_pct <= 0.16, f"Pre-surge paid_social share {pre_social_pct:.2f} out of expected bounds"
    assert post_social_pct >= pre_social_pct * 1.8, (
        f"Post-surge paid_social share {post_social_pct:.2f} did not increase sufficiently over pre {pre_social_pct:.2f}"
    )

    # 2. Checkout initiation rate verification
    user_channel = dict(zip(users_df["user_id"], users_df["acquisition_channel"]))
    events_df["channel"] = events_df["user_id"].map(user_channel)

    # Measure checkout start rate across total sessions per channel
    social_sessions = events_df[events_df["channel"] == "paid_social"].groupby("session_id")
    organic_sessions = events_df[events_df["channel"].isin(["organic_search", "direct"])].groupby("session_id")

    social_checkout_starts = events_df[
        (events_df["channel"] == "paid_social") & (events_df["event_name"] == "checkout_started")
    ]["session_id"].nunique()
    organic_checkout_starts = events_df[
        (events_df["channel"].isin(["organic_search", "direct"])) & (events_df["event_name"] == "checkout_started")
    ]["session_id"].nunique()

    social_checkout_rate = social_checkout_starts / len(social_sessions)
    organic_checkout_rate = organic_checkout_starts / len(organic_sessions)

    # Verify paid social checkout initiation is close to configured prob (0.03 +/- 0.02)
    assert abs(social_checkout_rate - configured_checkout_prob) < 0.02, (
        f"Paid social checkout rate {social_checkout_rate:.3f} deviates from configured {configured_checkout_prob}"
    )
    # Organic checkout initiation should be significantly higher (~12% baseline)
    assert organic_checkout_rate > social_checkout_rate * 2.5, (
        f"Organic checkout rate {organic_checkout_rate:.3f} not sufficiently higher than social {social_checkout_rate:.3f}"
    )

    # 3. Conversion rate comparison by channel
    completed_orders = orders_df[orders_df["status"] == "completed"]
    completed_user_ids = set(completed_orders["user_id"])

    post_social_users = post_users[post_users["acquisition_channel"] == "paid_social"]["user_id"]
    post_organic_users = post_users[post_users["acquisition_channel"].isin(["organic_search", "direct"])]["user_id"]

    social_conv_rate = sum(u in completed_user_ids for u in post_social_users) / len(post_social_users)
    organic_conv_rate = sum(u in completed_user_ids for u in post_organic_users) / len(post_organic_users)

    assert social_conv_rate < 0.035, f"Paid social conversion rate {social_conv_rate:.3f} is too high"
    assert organic_conv_rate >= 0.05, f"Organic conversion rate {organic_conv_rate:.3f} degraded unexpectedly"
    assert organic_conv_rate > social_conv_rate * 2.0


def test_tracking_failure_scenario_quantitative():
    """Quantitative verification of Safari Event Tracking Failure:
    Uses real backend successful orders as denominator in comparable time windows and browser segments.

    1. Pre-drop window: Both Chrome and Safari have telemetry tracking ratio >= 0.90
       (tracked order_completed events / backend completed orders).
    2. Post-drop window: Chrome maintains tracking ratio >= 0.90.
    3. Post-drop window: Safari tracking ratio drops dramatically to <= 0.25 (target 90% drop).
    4. Discrepancy metric: Backend completed orders on Safari occur normally, proving it is a tracking bug.
    """
    scenario = get_tracking_failure_scenario()
    config = GeneratorConfig(seed=42, num_users=1000)
    engine = SyntheticDataEngine(config=config, scenario=scenario)
    tables = engine.generate_all()

    validator = DataValidator(tables=tables, scenario=scenario, config=config)
    is_valid, errors = validator.validate_all()
    assert is_valid, f"Scenario validation failed: {errors}"

    events_df = tables["events"]
    orders_df = tables["orders"]
    users_df = tables["users"]
    user_browser = dict(zip(users_df["user_id"], users_df["browser"]))
    orders_df["browser"] = orders_df["user_id"].map(user_browser)

    drop_start = scenario.parameters["tracking_drop_start"]

    # --- Pre-drop window verification (Sept 1 to Sept 18) ---
    pre_orders = orders_df[orders_df["timestamp"] < drop_start]
    pre_events = events_df[events_df["timestamp"] < drop_start]

    safari_pre_orders = len(pre_orders[(pre_orders["browser"] == "Safari") & (pre_orders["status"] == "completed")])
    safari_pre_events = len(pre_events[(pre_events["browser"] == "Safari") & (pre_events["event_name"] == "order_completed")])

    chrome_pre_orders = len(pre_orders[(pre_orders["browser"] == "Chrome") & (pre_orders["status"] == "completed")])
    chrome_pre_events = len(pre_events[(pre_events["browser"] == "Chrome") & (pre_events["event_name"] == "order_completed")])

    assert safari_pre_orders > 0 and chrome_pre_orders > 0
    # Before drop, telemetry tracking is intact across both segments
    assert (safari_pre_events / safari_pre_orders) >= 0.90, "Safari pre-drop tracking ratio should be healthy (>= 0.90)"
    assert (chrome_pre_events / chrome_pre_orders) >= 0.90, "Chrome pre-drop tracking ratio should be healthy (>= 0.90)"

    # --- Post-drop window verification (Sept 18 to Oct 1) ---
    post_orders = orders_df[orders_df["timestamp"] >= drop_start]
    post_events = events_df[events_df["timestamp"] >= drop_start]

    safari_post_orders = len(post_orders[(post_orders["browser"] == "Safari") & (post_orders["status"] == "completed")])
    safari_post_events = len(post_events[(post_events["browser"] == "Safari") & (post_events["event_name"] == "order_completed")])

    chrome_post_orders = len(post_orders[(post_orders["browser"] == "Chrome") & (post_orders["status"] == "completed")])
    chrome_post_events = len(post_events[(post_events["browser"] == "Chrome") & (post_events["event_name"] == "order_completed")])

    assert safari_post_orders >= 5, "Expected real Safari backend orders in post-drop window"
    assert chrome_post_orders >= 15, "Expected real Chrome backend orders in post-drop window"

    # Chrome post-drop remains intact
    chrome_post_ratio = chrome_post_events / chrome_post_orders
    assert chrome_post_ratio >= 0.90, f"Chrome post-drop ratio {chrome_post_ratio:.2f} unexpectedly low"

    # Safari post-drop drops to <= 0.25
    safari_post_ratio = safari_post_events / safari_post_orders
    assert safari_post_ratio <= 0.25, (
        f"Safari post-drop ratio {safari_post_ratio:.2f} should be <= 0.25 due to tracking failure"
    )

    # Telemetry gap comparison: Chrome captures nearly all orders; Safari misses > 75%
    assert (safari_post_orders - safari_post_events) > 0
