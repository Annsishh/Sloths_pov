"""Synthetic Data Generation Engine.

Generates realistic, interconnected tabular SaaS datasets:
- users
- releases
- feature_exposures
- events
- orders
- user_feedback
- daily_product_metrics

Ensures strict reproducibility, temporal consistency, simulation-window bounds, and referential integrity.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from investigator_data.config import GeneratorConfig
from investigator_data.scenarios import ScenarioDefinition, get_baseline_scenario


class SyntheticDataEngine:
    """Generates synthetic SaaS data based on config and scenario rules."""

    def __init__(self, config: GeneratorConfig, scenario: ScenarioDefinition | None = None):
        self.config = config
        self.scenario = scenario or get_baseline_scenario()
        # Initialize deterministic RNG
        self.rng = np.random.default_rng(self.config.seed)
        self.simulation_start = self.config.start_time
        self.simulation_end = self.simulation_start + timedelta(days=self.config.num_days)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all interconnected tables and return as dictionary of DataFrames."""
        users_df = self._generate_users()
        releases_df = self._generate_releases()
        exposures_df = self._generate_feature_exposures(users_df, releases_df)
        events_df, orders_df = self._generate_events_and_orders(users_df, releases_df)
        feedback_df = self._generate_user_feedback(users_df, orders_df)
        metrics_df = self._generate_daily_product_metrics(users_df, events_df, orders_df, feedback_df)

        return {
            "users": users_df,
            "releases": releases_df,
            "feature_exposures": exposures_df,
            "events": events_df,
            "orders": orders_df,
            "user_feedback": feedback_df,
            "daily_product_metrics": metrics_df,
        }

    def _generate_users(self) -> pd.DataFrame:
        channels = ["organic_search", "direct", "paid_search", "paid_social", "referral"]
        channel_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        countries = ["US", "GB", "DE", "FR", "CA", "IN", "JP", "AU"]
        country_weights = [0.45, 0.15, 0.10, 0.08, 0.07, 0.07, 0.04, 0.04]
        devices = ["desktop", "mobile", "tablet"]
        device_weights = [0.60, 0.35, 0.05]
        browsers = ["Chrome", "Safari", "Firefox", "Edge"]
        browser_weights = [0.62, 0.24, 0.08, 0.06]
        tiers = ["starter", "pro", "enterprise"]
        tier_weights = [0.70, 0.25, 0.05]

        num_base_users = self.config.num_users
        records = []
        user_id_counter = 1000

        # Check if acquisition_shift scenario applies
        is_acq_shift = self.scenario.category == "acquisition_shift"
        surge_start = None
        social_mult = 1.0
        if is_acq_shift:
            surge_start = datetime.fromisoformat(self.scenario.parameters["surge_start"])
            social_mult = float(self.scenario.parameters.get("paid_social_volume_multiplier", 3.0))

        # Simulation window parameters
        total_seconds = int((self.simulation_end - self.simulation_start).total_seconds())
        # Keep signup timestamps strictly within [start_time, simulation_end)
        max_signup_offset = max(1, total_seconds - 3600)

        # Baseline users distributed uniformly across window
        for _ in range(num_base_users):
            user_id_counter += 1
            offset_seconds = self.rng.integers(0, max_signup_offset)
            signup_time = self.simulation_start + timedelta(seconds=int(offset_seconds))

            channel = self.rng.choice(channels, p=channel_weights)
            device = self.rng.choice(devices, p=device_weights)
            browser = self.rng.choice(browsers, p=browser_weights)
            country = self.rng.choice(countries, p=country_weights)
            tier = self.rng.choice(tiers, p=tier_weights)

            records.append({
                "user_id": f"usr_{user_id_counter}",
                "signup_timestamp": signup_time.isoformat(),
                "acquisition_channel": channel,
                "country": country,
                "device_type": device,
                "browser": browser,
                "plan_tier": tier,
            })

        # If acquisition shift scenario: generate the additional surging paid_social users
        # based on paid_social_volume_multiplier
        if is_acq_shift and surge_start and social_mult > 1.0:
            post_surge_duration = (self.simulation_end - surge_start).total_seconds()
            if post_surge_duration > 0:
                post_surge_fraction = post_surge_duration / total_seconds
                # Baseline expected paid_social users in post-surge window
                expected_baseline_social = num_base_users * 0.10 * post_surge_fraction
                # Additional users to achieve total multiplier: extra = (multiplier - 1) * expected_baseline
                extra_social_users = int(round((social_mult - 1.0) * expected_baseline_social))

                for _ in range(extra_social_users):
                    user_id_counter += 1
                    # Offset within the post-surge window, strictly < simulation_end
                    offset_within_surge = self.rng.integers(0, max(1, int(post_surge_duration - 1800)))
                    signup_time = surge_start + timedelta(seconds=int(offset_within_surge))

                    device = self.rng.choice(devices, p=device_weights)
                    browser = self.rng.choice(browsers, p=browser_weights)
                    country = self.rng.choice(countries, p=country_weights)
                    # Paid social low-intent skew towards starter plan
                    tier = self.rng.choice(tiers, p=[0.90, 0.09, 0.01])

                    records.append({
                        "user_id": f"usr_{user_id_counter}",
                        "signup_timestamp": signup_time.isoformat(),
                        "acquisition_channel": "paid_social",
                        "country": country,
                        "device_type": device,
                        "browser": browser,
                        "plan_tier": tier,
                    })

        df = pd.DataFrame(records)
        df.sort_values(by="signup_timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _generate_releases(self) -> pd.DataFrame:
        releases = [
            {
                "release_id": "rel_238",
                "version": "v2.3.8",
                "deployed_timestamp": (self.simulation_start + timedelta(days=2, hours=10)).isoformat(),
                "component": "core_web",
                "description": "Performance improvements and caching updates for dashboard.",
                "status": "success",
            },
            {
                "release_id": "rel_239",
                "version": "v2.3.9",
                "deployed_timestamp": (self.simulation_start + timedelta(days=8, hours=14)).isoformat(),
                "component": "notifications_service",
                "description": "Email delivery and in-app alert latency optimizations.",
                "status": "success",
            },
            {
                "release_id": "rel_240",
                "version": "v2.4.0",
                "deployed_timestamp": (self.simulation_start + timedelta(days=14, hours=10)).isoformat(),
                "component": "mobile_checkout",
                "description": "Upgraded mobile payment processing SDK & one-click checkout.",
                "status": "success",
            },
            {
                "release_id": "rel_241",
                "version": "v2.4.1",
                "deployed_timestamp": (self.simulation_start + timedelta(days=22, hours=16)).isoformat(),
                "component": "reporting_engine",
                "description": "Export CSV/PDF feature overhaul and analytics widgets.",
                "status": "success",
            },
        ]
        df = pd.DataFrame(releases)
        df.sort_values(by="deployed_timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _generate_feature_exposures(self, users_df: pd.DataFrame, releases_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        exposure_counter = 10000
        features = [
            ("new_onboarding_flow", datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)),
            ("one_click_checkout", datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)),
            ("dark_mode_ui", datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)),
        ]

        for _, user in users_df.iterrows():
            user_signup = datetime.fromisoformat(user["signup_timestamp"])
            for feature_key, rollout_date in features:
                # 60% probability of exposure
                if self.rng.random() < 0.60:
                    base_exposure = max(user_signup, rollout_date)
                    max_delay = int((self.simulation_end - base_exposure).total_seconds())
                    if max_delay <= 60:
                        continue
                    delay_seconds = int(self.rng.integers(60, min(86400, max_delay)))
                    exposure_time = base_exposure + timedelta(seconds=delay_seconds)
                    if exposure_time >= self.simulation_end:
                        continue

                    variant = self.rng.choice(["control", "treatment"], p=[0.5, 0.5])
                    exposure_counter += 1
                    records.append({
                        "exposure_id": f"exp_{exposure_counter}",
                        "user_id": user["user_id"],
                        "feature_key": feature_key,
                        "variant": variant,
                        "first_exposed_timestamp": exposure_time.isoformat(),
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by="first_exposed_timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _generate_events_and_orders(
        self, users_df: pd.DataFrame, releases_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        event_records = []
        order_records = []
        event_counter = 100000
        order_counter = 50000

        pages = ["landing_page", "pricing", "dashboard", "settings", "checkout"]

        # Parse scenario parameters
        is_regression = self.scenario.category == "product_regression"
        regression_deploy_time = None
        mobile_fail_rate = 0.0
        if is_regression:
            regression_deploy_time = datetime.fromisoformat(self.scenario.parameters["deploy_time"])
            mobile_fail_rate = float(self.scenario.parameters.get("mobile_order_failure_prob", 0.75))

        is_acq_shift = self.scenario.category == "acquisition_shift"
        social_events_mult = 1.0
        social_checkout_prob = None
        social_order_conv_prob = None
        if is_acq_shift:
            social_events_mult = float(self.scenario.parameters.get("paid_social_events_multiplier", 0.4))
            social_checkout_prob = float(self.scenario.parameters.get("paid_social_checkout_start_prob", 0.03))
            social_order_conv_prob = float(self.scenario.parameters.get("paid_social_conversion_rate", 0.012))

        is_tracking_failure = self.scenario.category == "tracking_failure"
        tracking_drop_start = None
        tracking_drop_rate = 0.0
        if is_tracking_failure:
            tracking_drop_start = datetime.fromisoformat(self.scenario.parameters["tracking_drop_start"])
            tracking_drop_rate = float(self.scenario.parameters.get("drop_probability", 0.90))

        for _, user in users_df.iterrows():
            user_id = user["user_id"]
            user_signup = datetime.fromisoformat(user["signup_timestamp"])
            channel = user["acquisition_channel"]
            device = user["device_type"]
            browser = user["browser"]

            remaining_seconds = (self.simulation_end - user_signup).total_seconds()
            if remaining_seconds <= 120:
                continue

            days_active = max(1.0, remaining_seconds / 86400.0)

            # Activity multiplier based on acquisition channel
            activity_mult = social_events_mult if (is_acq_shift and channel == "paid_social") else 1.0

            # Checkout initiation probability: controlled directly by paid_social_checkout_start_prob in acquisition scenario
            if is_acq_shift and channel == "paid_social":
                checkout_start_prob = social_checkout_prob
                # Payment attempt conditional probability given checkout start
                if social_checkout_prob and social_checkout_prob > 0:
                    payment_attempt_prob = min(0.95, social_order_conv_prob / social_checkout_prob)
                else:
                    payment_attempt_prob = 0.50
            else:
                checkout_start_prob = self.config.baseline_checkout_start_prob
                # For baseline: ~67% of users who start checkout complete an order attempt (0.08 / 0.12)
                payment_attempt_prob = self.config.baseline_order_conversion_rate / self.config.baseline_checkout_start_prob

            # Expected sessions
            num_sessions = int(self.rng.poisson(lam=max(0.5, 0.4 * days_active * activity_mult)))

            for sess_idx in range(num_sessions):
                max_session_offset = max(1, int(remaining_seconds - 300))
                session_offset = self.rng.integers(0, max_session_offset)
                session_time = user_signup + timedelta(seconds=int(session_offset))
                if session_time >= self.simulation_end - timedelta(seconds=60):
                    continue

                session_id = f"sess_{user_id}_{sess_idx + 1}"

                # App version depends on deployment date
                app_version = "v2.3.8"
                if session_time >= datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc):
                    app_version = "v2.4.0"
                elif session_time >= datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc):
                    app_version = "v2.3.9"

                # Generate browsing events
                num_events_in_sess = int(self.rng.integers(2, 6))
                curr_t = session_time
                for ev_i in range(num_events_in_sess):
                    if curr_t >= self.simulation_end:
                        break
                    event_counter += 1
                    page = pages[min(ev_i, len(pages) - 1)]
                    ev_name = "page_view" if ev_i == 0 else "feature_click"

                    event_records.append({
                        "event_id": f"evt_{event_counter}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "timestamp": curr_t.isoformat(),
                        "event_name": ev_name,
                        "page": page,
                        "device_type": device,
                        "browser": browser,
                        "app_version": app_version,
                    })
                    curr_t += timedelta(seconds=int(self.rng.integers(10, 60)))

                # Checkout funnel generation:
                # 1. Checkout started probability: directly controlled by checkout_start_prob
                if self.rng.random() < checkout_start_prob and curr_t < self.simulation_end - timedelta(seconds=40):
                    event_counter += 1
                    checkout_start_time = curr_t
                    event_records.append({
                        "event_id": f"evt_{event_counter}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "timestamp": checkout_start_time.isoformat(),
                        "event_name": "checkout_started",
                        "page": "checkout",
                        "device_type": device,
                        "browser": browser,
                        "app_version": app_version,
                    })
                    curr_t += timedelta(seconds=int(self.rng.integers(8, 20)))

                    # 2. Checkout payment attempt
                    checkout_attempt_payment = (self.rng.random() < payment_attempt_prob)

                    if not checkout_attempt_payment:
                        # User abandoned checkout without attempting payment
                        if curr_t < self.simulation_end:
                            event_counter += 1
                            event_records.append({
                                "event_id": f"evt_{event_counter}",
                                "user_id": user_id,
                                "session_id": session_id,
                                "timestamp": curr_t.isoformat(),
                                "event_name": "checkout_abandoned",
                                "page": "checkout",
                                "device_type": device,
                                "browser": browser,
                                "app_version": app_version,
                            })
                    else:
                        # User attempted payment -> generates order record
                        order_counter += 1
                        order_time = curr_t
                        order_id = f"ord_{order_counter}"
                        amount = round(float(self.rng.choice([29.0, 79.0, 199.0, 499.0], p=[0.4, 0.35, 0.2, 0.05])), 2)

                        order_status = "completed"
                        failure_reason = None

                        # Check Product Regression scenario (mobile checkout bug)
                        if (
                            is_regression
                            and device == "mobile"
                            and regression_deploy_time
                            and order_time >= regression_deploy_time
                        ):
                            if self.rng.random() < mobile_fail_rate:
                                order_status = "failed"
                                failure_reason = "payment_gateway_timeout"

                        # Natural baseline small failure rate (e.g. 4% card decline)
                        if order_status == "completed" and self.rng.random() < 0.04:
                            order_status = "failed"
                            failure_reason = "card_declined"

                        if order_time < self.simulation_end:
                            order_records.append({
                                "order_id": order_id,
                                "user_id": user_id,
                                "timestamp": order_time.isoformat(),
                                "amount": amount,
                                "currency": self.config.default_currency,
                                "status": order_status,
                                "failure_reason": failure_reason,
                            })

                            # Telemetry event for order outcome
                            curr_t = order_time + timedelta(seconds=2)
                            if curr_t < self.simulation_end:
                                if order_status == "completed":
                                    # Check tracking failure scenario (Safari drops order_completed)
                                    should_drop = False
                                    if (
                                        is_tracking_failure
                                        and browser == "Safari"
                                        and tracking_drop_start
                                        and order_time >= tracking_drop_start
                                    ):
                                        if self.rng.random() < tracking_drop_rate:
                                            should_drop = True

                                    if not should_drop:
                                        event_counter += 1
                                        event_records.append({
                                            "event_id": f"evt_{event_counter}",
                                            "user_id": user_id,
                                            "session_id": session_id,
                                            "timestamp": curr_t.isoformat(),
                                            "event_name": "order_completed",
                                            "page": "checkout",
                                            "device_type": device,
                                            "browser": browser,
                                            "app_version": app_version,
                                        })
                                else:
                                    # Payment failure event logged in telemetry
                                    event_counter += 1
                                    event_records.append({
                                        "event_id": f"evt_{event_counter}",
                                        "user_id": user_id,
                                        "session_id": session_id,
                                        "timestamp": curr_t.isoformat(),
                                        "event_name": "checkout_failed",
                                        "page": "checkout",
                                        "device_type": device,
                                        "browser": browser,
                                        "app_version": app_version,
                                    })

        events_df = pd.DataFrame(event_records)
        orders_df = pd.DataFrame(order_records)

        if not events_df.empty:
            events_df.sort_values(by="timestamp", inplace=True)
            events_df.reset_index(drop=True, inplace=True)
        if not orders_df.empty:
            orders_df.sort_values(by="timestamp", inplace=True)
            orders_df.reset_index(drop=True, inplace=True)

        return events_df, orders_df

    def _generate_user_feedback(self, users_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        feedback_counter = 8000
        user_lookup = users_df.set_index("user_id").to_dict("index")

        # Feedback from order experiences
        for _, order in orders_df.iterrows():
            user_id = order["user_id"]
            user_info = user_lookup.get(user_id, {})
            order_time = datetime.fromisoformat(order["timestamp"])

            # Failed orders trigger negative feedback with higher probability
            if order["status"] == "failed":
                if self.rng.random() < 0.65:
                    max_sec = int((self.simulation_end - order_time).total_seconds())
                    if max_sec <= 60:
                        continue
                    fb_offset = int(self.rng.integers(60, min(3600, max_sec)))
                    fb_time = order_time + timedelta(seconds=fb_offset)
                    if fb_time >= self.simulation_end:
                        continue

                    feedback_counter += 1
                    rating = 1 if self.rng.random() < 0.8 else 2
                    if order.get("failure_reason") == "payment_gateway_timeout":
                        comment = "Checkout timed out while processing payment on mobile app. Very frustrating!"
                    else:
                        comment = "Payment declined unexpectedly at checkout."

                    records.append({
                        "feedback_id": f"fb_{feedback_counter}",
                        "user_id": user_id,
                        "timestamp": fb_time.isoformat(),
                        "rating": rating,
                        "category": "billing",
                        "comment": comment,
                    })
            else:
                # Normal successful order feedback
                if self.rng.random() < self.config.baseline_feedback_rate:
                    max_sec = int((self.simulation_end - order_time).total_seconds())
                    if max_sec <= 60:
                        continue
                    fb_offset = int(self.rng.integers(60, min(21600, max_sec)))
                    fb_time = order_time + timedelta(seconds=fb_offset)
                    if fb_time >= self.simulation_end:
                        continue

                    feedback_counter += 1
                    rating = int(self.rng.choice([4, 5], p=[0.3, 0.7]))
                    records.append({
                        "feedback_id": f"fb_{feedback_counter}",
                        "user_id": user_id,
                        "timestamp": fb_time.isoformat(),
                        "rating": rating,
                        "category": "product",
                        "comment": "Quick and seamless checkout experience.",
                    })

        # General unsolicited feedback
        for _, user in users_df.iterrows():
            if self.rng.random() < 0.03:
                user_signup = datetime.fromisoformat(user["signup_timestamp"])
                max_sec = int((self.simulation_end - user_signup).total_seconds())
                if max_sec <= 3600:
                    continue
                fb_offset = int(self.rng.integers(3600, max_sec))
                fb_time = user_signup + timedelta(seconds=fb_offset)
                if fb_time >= self.simulation_end:
                    continue

                feedback_counter += 1
                rating = int(self.rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
                records.append({
                    "feedback_id": f"fb_{feedback_counter}",
                    "user_id": user["user_id"],
                    "timestamp": fb_time.isoformat(),
                    "rating": rating,
                    "category": "general",
                    "comment": "App interface is intuitive and fast.",
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by="timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _generate_daily_product_metrics(
        self,
        users_df: pd.DataFrame,
        events_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        feedback_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Roll up daily aggregate product metrics from granular event and order tables."""
        metrics_records = []
        start_date = self.simulation_start.date()

        for day_idx in range(self.config.num_days):
            current_date = start_date + timedelta(days=day_idx)
            date_str = current_date.isoformat()

            # Filter records within day range
            daily_signups = 0
            if not users_df.empty:
                daily_signups = int(users_df["signup_timestamp"].str.startswith(date_str).sum())

            daily_dau = 0
            daily_events = 0
            if not events_df.empty:
                day_events = events_df[events_df["timestamp"].str.startswith(date_str)]
                daily_events = len(day_events)
                daily_dau = day_events["user_id"].nunique()

            completed_orders = 0
            failed_orders = 0
            gross_revenue = 0.0
            if not orders_df.empty:
                day_orders = orders_df[orders_df["timestamp"].str.startswith(date_str)]
                completed = day_orders[day_orders["status"] == "completed"]
                failed = day_orders[day_orders["status"] == "failed"]
                completed_orders = len(completed)
                failed_orders = len(failed)
                gross_revenue = round(float(completed["amount"].sum()), 2)

            avg_rating = 4.2
            if not feedback_df.empty:
                day_fb = feedback_df[feedback_df["timestamp"].str.startswith(date_str)]
                if not day_fb.empty:
                    avg_rating = round(float(day_fb["rating"].mean()), 2)

            metrics_records.append({
                "date": date_str,
                "daily_active_users": daily_dau,
                "new_signups": daily_signups,
                "total_events": daily_events,
                "completed_orders": completed_orders,
                "failed_orders": failed_orders,
                "gross_revenue": gross_revenue,
                "avg_feedback_rating": avg_rating,
            })

        df = pd.DataFrame(metrics_records)
        return df
