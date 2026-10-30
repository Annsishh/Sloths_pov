"""Synthetic Data Generation Engine.

Generates realistic, interconnected tabular SaaS datasets:
users, releases, feature_exposures, events, orders, user_feedback, daily_product_metrics.

Ensures strict reproducibility (all randomness is seeded), temporal consistency, simulation-window bounds and
referential integrity. Realism knobs (noise, weekly seasonality, holidays, effect size/onset, late and duplicate
events, bots, identity splits, event renames) live in GeneratorConfig; causes live in the scenario's parameters.
Random streams: `rng` drives the core simulation; `aux` drives the realism knobs, so toggling a knob does not
reshuffle unrelated draws.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from investigator_data.config import GeneratorConfig
from investigator_data.scenarios import ScenarioDefinition, default_releases, get_baseline_scenario

BOT_BROWSER = "HeadlessChrome"
# Mon..Sun deviation from the weekly mean; scaled by config.weekly_seasonality.
WEEKDAY_PATTERN = (0.15, 0.25, 0.25, 0.15, 0.0, -0.6, -0.7)


def _parse(ts: Any) -> datetime:
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _version_key(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in v.lstrip("v").split("."))


class SyntheticDataEngine:
    """Generates synthetic SaaS data based on config and scenario rules."""

    def __init__(self, config: GeneratorConfig, scenario: ScenarioDefinition | None = None):
        self.scenario = scenario or get_baseline_scenario()
        # Explicit config wins; the scenario's world overrides fill fields the config left at their defaults.
        default = GeneratorConfig()
        fill = {k: v for k, v in self.scenario.config_overrides.items()
                if getattr(config, k, None) == getattr(default, k, None)}
        self.config = config.with_overrides(fill)
        self.rng = np.random.default_rng(self.config.seed)
        self.aux = np.random.default_rng([self.config.seed, 7919])
        self.simulation_start = self.config.start_time
        self.simulation_end = self.simulation_start + timedelta(days=self.config.num_days)
        self.params: Dict[str, Any] = self.scenario.parameters
        # ground-truth bookkeeping written by generate_data (never visible to the investigator)
        self.truth: Dict[str, Any] = {"regression_failed_orders": 0, "tracking_dropped_events": 0, "outage_failed_orders": 0}
        self._event_counter = 100000
        self._events: List[Dict[str, Any]] = []
        self._release_log = self._build_release_log()
        self._day_mult = self._build_day_multipliers()
        self._max_mult = max(self._day_mult.values())
        self._session_scale = self._max_mult / (sum(self._day_mult.values()) / len(self._day_mult))

    # ------------------------------------------------------------------ world model
    def _build_release_log(self) -> List[Dict[str, Any]]:
        releases = self.scenario.releases if self.scenario.releases is not None else default_releases()
        # releases are expressed in absolute time; shift default log if the simulation starts on another date
        return sorted(releases, key=lambda r: r["deployed_timestamp"])

    def _build_day_multipliers(self) -> Dict[Any, float]:
        c = self.config
        holidays = {datetime.fromisoformat(d).date(): m for d, m in c.holiday_effects}
        mult: Dict[Any, float] = {}
        for i in range(c.num_days + 1):
            day = (self.simulation_start + timedelta(days=i)).date()
            wd = 1.0 + c.weekly_seasonality * WEEKDAY_PATTERN[day.weekday()]
            hol = holidays.get(day, 1.0)
            noise = float(self.aux.lognormal(-c.noise_level ** 2 / 2, c.noise_level)) if c.noise_level > 0 else 1.0
            mult[day] = max(0.05, wd * hol * noise)
        return mult

    def _accept_session(self, t: datetime, signup: datetime, channel: str = "") -> bool:
        p = self._day_mult.get(t.date(), 1.0) / self._max_mult
        cut = self.params.get("paid_social_cut_start")
        if cut and channel == "paid_social" and t >= _parse(cut):
            p *= 1.0 - float(self.params.get("paid_social_cut_fraction", 0.0))   # campaign traffic switched off
        decay = float(self.params.get("tenure_decay_rate", 0.0))
        if decay > 0:
            p *= math.exp(-decay * max(0.0, (t - signup).total_seconds() / 86400.0))
        return bool(self.aux.random() < p)

    def _intensity(self, t: datetime, start: datetime, end: Optional[datetime] = None) -> float:
        """Effect strength in [0, 1] at time t: step or linear ramp from `start` (zero after `end`)."""
        if t < start or (end is not None and t >= end):
            return 0.0
        if self.config.effect_onset == "step":
            return 1.0
        return min(1.0, (t - start).total_seconds() / (self.config.effect_ramp_days * 86400.0))

    def _size(self, default: float) -> float:
        return default if self.config.effect_size is None else float(self.config.effect_size)

    def _version_at(self, t: datetime, adoption_lag: timedelta) -> str:
        """App version a session runs: latest release deployed at least `adoption_lag` ago (first release if none)."""
        eff = t - adoption_lag
        version = self._release_log[0]["version"]
        for r in self._release_log:
            if _parse(r["deployed_timestamp"]) <= eff:
                version = r["version"]
        return version

    def _renamed(self, name: str, app_version: str) -> str:
        for from_version, old, new in self.config.event_renames:
            if name == old and _version_key(app_version) >= _version_key(from_version):
                return new
        return name

    # ------------------------------------------------------------------ event emission
    def _emit(self, name: str, page: str, ts: datetime, user_id: str, session_id: str,
              device: str, browser: str, app_version: str) -> None:
        c = self.config
        if c.late_arrival_rate > 0 and self.aux.random() < c.late_arrival_rate:
            lag = float(self.aux.exponential(c.late_arrival_mean_hours * 3600.0))
            if ts + timedelta(seconds=lag) >= self.simulation_end:
                return          # arrives after the extraction cut-off: missing from the table
        record = {"user_id": user_id, "session_id": session_id, "timestamp": ts.isoformat(),
                  "event_name": self._renamed(name, app_version), "page": page, "device_type": device,
                  "browser": browser, "app_version": app_version}
        self._event_counter += 1
        self._events.append({"event_id": f"evt_{self._event_counter}", **record})
        if c.duplicate_event_rate > 0 and self.aux.random() < c.duplicate_event_rate:
            self._event_counter += 1
            self._events.append({"event_id": f"evt_{self._event_counter}", **record})

    # ------------------------------------------------------------------ tables
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all interconnected tables and return as dictionary of DataFrames."""
        users_df = self._generate_users()
        releases_df = self._generate_releases()
        exposures_df = self._generate_feature_exposures(users_df, releases_df)
        events_df, orders_df = self._generate_events_and_orders(users_df, releases_df)
        feedback_df = self._generate_user_feedback(users_df, orders_df)
        metrics_df = self._generate_daily_product_metrics(users_df, events_df, orders_df, feedback_df)
        return {
            "users": users_df, "releases": releases_df, "feature_exposures": exposures_df,
            "events": events_df, "orders": orders_df, "user_feedback": feedback_df,
            "daily_product_metrics": metrics_df,
        }

    def _generate_users(self) -> pd.DataFrame:
        channels = ["organic_search", "direct", "paid_search", "paid_social", "referral"]
        channel_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        share = self.params.get("paid_social_share")
        if share is not None:                    # re-weight so paid_social has the requested share
            rest = [w for ch, w in zip(channels, channel_weights) if ch != "paid_social"]
            scale = (1.0 - share) / sum(rest)
            channel_weights = [share if ch == "paid_social" else w * scale for ch, w in zip(channels, channel_weights)]
        countries = ["US", "GB", "DE", "FR", "CA", "IN", "JP", "AU"]
        country_weights = [0.45, 0.15, 0.10, 0.08, 0.07, 0.07, 0.04, 0.04]
        devices = ["desktop", "mobile", "tablet"]
        device_weights = [0.60, 0.35, 0.05]
        browsers = ["Chrome", "Safari", "Firefox", "Edge"]
        browser_weights = [0.62, 0.24, 0.08, 0.06]
        tiers = ["starter", "pro", "enterprise"]
        tier_weights = [0.70, 0.25, 0.05]

        num_base_users = self.config.num_users
        records: List[Dict[str, Any]] = []
        user_id_counter = 1000

        surge_start_raw = self.params.get("surge_start")
        surge_start = _parse(surge_start_raw) if surge_start_raw else None
        social_mult = float(self.params.get("paid_social_volume_multiplier", 3.0))

        total_seconds = int((self.simulation_end - self.simulation_start).total_seconds())
        max_signup_offset = max(1, total_seconds - 3600)

        for _ in range(num_base_users):
            user_id_counter += 1
            offset_seconds = self.rng.integers(0, max_signup_offset)
            signup_time = self.simulation_start + timedelta(seconds=int(offset_seconds))
            if self.config.existing_user_share > 0 and self.aux.random() < self.config.existing_user_share:
                # pre-existing user: signed up 1-90 days before the window opens
                signup_time = self.simulation_start - timedelta(seconds=int(self.aux.integers(86400, 90 * 86400)))

            channel = self.rng.choice(channels, p=channel_weights)
            device = self.rng.choice(devices, p=device_weights)
            browser = self.rng.choice(browsers, p=browser_weights)
            country = self.rng.choice(countries, p=country_weights)
            tier = self.rng.choice(tiers, p=tier_weights)

            records.append({
                "user_id": f"usr_{user_id_counter}", "signup_timestamp": signup_time.isoformat(),
                "acquisition_channel": channel, "country": country, "device_type": device,
                "browser": browser, "plan_tier": tier,
            })

        if surge_start and social_mult > 1.0:
            post_surge_duration = (self.simulation_end - surge_start).total_seconds()
            if post_surge_duration > 0:
                post_surge_fraction = post_surge_duration / total_seconds
                expected_baseline_social = num_base_users * 0.10 * post_surge_fraction
                extra_social_users = int(round((social_mult - 1.0) * expected_baseline_social))
                for _ in range(extra_social_users):
                    user_id_counter += 1
                    offset_within_surge = self.rng.integers(0, max(1, int(post_surge_duration - 1800)))
                    signup_time = surge_start + timedelta(seconds=int(offset_within_surge))
                    device = self.rng.choice(devices, p=device_weights)
                    browser = self.rng.choice(browsers, p=browser_weights)
                    country = self.rng.choice(countries, p=country_weights)
                    tier = self.rng.choice(tiers, p=[0.90, 0.09, 0.01])
                    records.append({
                        "user_id": f"usr_{user_id_counter}", "signup_timestamp": signup_time.isoformat(),
                        "acquisition_channel": "paid_social", "country": country, "device_type": device,
                        "browser": browser, "plan_tier": tier,
                    })

        # --- identity splits: some people appear under a second user_id on another device
        if self.config.identity_split_share > 0:
            twins = []
            for rec in records:
                if self.aux.random() < self.config.identity_split_share:
                    user_id_counter += 1
                    other_device = "desktop" if rec["device_type"] == "mobile" else "mobile"
                    twins.append({**rec, "user_id": f"usr_{user_id_counter}", "device_type": other_device,
                                  "browser": str(self.aux.choice(browsers, p=browser_weights))})
            records.extend(twins)

        # --- bots: unlabeled in the data except for an unusual browser string
        if self.config.bot_traffic_share > 0:
            n_bots = int(round(len(records) * self.config.bot_traffic_share / (1 - self.config.bot_traffic_share)))
            for _ in range(n_bots):
                user_id_counter += 1
                signup_time = self.simulation_start + timedelta(seconds=int(self.aux.integers(0, max_signup_offset)))
                records.append({
                    "user_id": f"usr_{user_id_counter}", "signup_timestamp": signup_time.isoformat(),
                    "acquisition_channel": "direct", "country": "US", "device_type": "desktop",
                    "browser": BOT_BROWSER, "plan_tier": "starter",
                })

        df = pd.DataFrame(records)
        df.sort_values(by="signup_timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _generate_releases(self) -> pd.DataFrame:
        df = pd.DataFrame(self._release_log)
        df.sort_values(by="deployed_timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _generate_feature_exposures(self, users_df: pd.DataFrame, releases_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        exposure_counter = 10000
        # one_click_checkout ships with the mobile_checkout release in THIS scenario's release log
        checkout_release = next((_parse(r["deployed_timestamp"]) for r in self._release_log if r["component"] == "mobile_checkout"),
                                self.simulation_start + timedelta(days=14, hours=10))
        features = [
            ("new_onboarding_flow", self.simulation_start + timedelta(days=2)),
            ("one_click_checkout", checkout_release),
            ("dark_mode_ui", self.simulation_start + timedelta(days=6)),
        ]
        for _, user in users_df.iterrows():
            if user["browser"] == BOT_BROWSER:
                continue
            user_signup = datetime.fromisoformat(user["signup_timestamp"])
            for feature_key, rollout_date in features:
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
                        "exposure_id": f"exp_{exposure_counter}", "user_id": user["user_id"],
                        "feature_key": feature_key, "variant": variant,
                        "first_exposed_timestamp": exposure_time.isoformat(),
                    })
        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by="first_exposed_timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------ cause mechanisms
    def _failure_effects(self) -> List[Dict[str, Any]]:
        """Payment-failure effects: the mobile-release bug plus any generic scenario effects."""
        effects: List[Dict[str, Any]] = []
        if self.params.get("deploy_time") and "mobile_order_failure_prob" in self.params:
            effects.append({
                "kind": "regression", "start": _parse(self.params["deploy_time"]), "end": None,
                "prob": self._size(float(self.params["mobile_order_failure_prob"])),
                "reason": "payment_gateway_timeout", "device": "mobile",
                "min_version": self.params.get("release_version", "v2.4.0"),
            })
        for e in self.params.get("order_failure_effects", []):
            effects.append({
                "kind": "outage", "start": _parse(e["start"]), "end": _parse(e["end"]) if e.get("end") else None,
                "prob": self._size(float(e["prob"])), "reason": e.get("reason", "payment_failure"),
                "device": e.get("device"), "min_version": None,
            })
        return effects

    def _conversion_multiplier(self, t: datetime, channel: str) -> float:
        mult = 1.0
        for e in self.params.get("conversion_effects", []):
            if e.get("channel") and e["channel"] != channel:
                continue
            m = float(e["multiplier"])
            if self.config.effect_size is not None:       # effect_size = fractional cut in conversion
                m = 1.0 - float(self.config.effect_size)
            mult *= 1.0 + (m - 1.0) * self._intensity(t, _parse(e["start"]))
        return mult

    def _generate_events_and_orders(
        self, users_df: pd.DataFrame, releases_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._events = []
        order_records: List[Dict[str, Any]] = []
        order_counter = 50000
        pages = ["landing_page", "pricing", "dashboard", "settings", "checkout"]

        failure_effects = self._failure_effects()

        low_intent_social = bool(self.params.get("surge_start") or self.params.get("paid_social_low_intent"))
        social_events_mult = float(self.params.get("paid_social_events_multiplier", 0.4))
        social_checkout_prob = float(self.params.get("paid_social_checkout_start_prob", 0.03))
        social_order_conv_prob = float(self.params.get("paid_social_conversion_rate", 0.012))

        tracking_drop_start = _parse(self.params["tracking_drop_start"]) if self.params.get("tracking_drop_start") else None
        tracking_drop_rate = self._size(float(self.params.get("drop_probability", 0.90)))
        tracking_browser = self.params.get("affected_browser", "Safari")

        adoption_days = self.config.app_adoption_days_mobile
        end_minus_60 = self.simulation_end - timedelta(seconds=60)

        for _, user in users_df.iterrows():
            user_id = user["user_id"]
            user_signup = max(datetime.fromisoformat(user["signup_timestamp"]), self.simulation_start)  # activity starts at window open
            channel = user["acquisition_channel"]
            device = user["device_type"]
            browser = user["browser"]

            remaining_seconds = (self.simulation_end - user_signup).total_seconds()
            if remaining_seconds <= 120:
                continue
            days_active = max(1.0, remaining_seconds / 86400.0)

            if browser == BOT_BROWSER:
                self._generate_bot_sessions(user_id, user_signup, days_active, remaining_seconds)
                continue

            activity_mult = social_events_mult if (low_intent_social and channel == "paid_social") else 1.0

            if low_intent_social and channel == "paid_social":
                checkout_start_prob = social_checkout_prob
                payment_attempt_prob = min(0.95, social_order_conv_prob / social_checkout_prob) if social_checkout_prob > 0 else 0.50
            else:
                checkout_start_prob = self.config.baseline_checkout_start_prob
                payment_attempt_prob = self.config.baseline_order_conversion_rate / self.config.baseline_checkout_start_prob

            adoption_lag = timedelta(days=float(self.aux.exponential(adoption_days))) if (device == "mobile" and adoption_days > 0) else timedelta(0)

            num_sessions = int(self.rng.poisson(lam=max(0.5, 0.4 * days_active * activity_mult) * self._session_scale))

            for sess_idx in range(num_sessions):
                max_session_offset = max(1, int(remaining_seconds - 300))
                session_offset = self.rng.integers(0, max_session_offset)
                session_time = user_signup + timedelta(seconds=int(session_offset))
                if session_time >= end_minus_60:
                    continue
                if not self._accept_session(session_time, user_signup, channel):
                    continue

                session_id = f"sess_{user_id}_{sess_idx + 1}"
                app_version = self._version_at(session_time, adoption_lag)

                num_events_in_sess = int(self.rng.integers(2, 6))
                curr_t = session_time
                for ev_i in range(num_events_in_sess):
                    if curr_t >= self.simulation_end:
                        break
                    page = pages[min(ev_i, len(pages) - 1)]
                    self._emit("page_view" if ev_i == 0 else "feature_click", page, curr_t, user_id, session_id, device, browser, app_version)
                    curr_t += timedelta(seconds=int(self.rng.integers(10, 60)))

                eff_checkout = checkout_start_prob * self._conversion_multiplier(session_time, channel)
                if self.rng.random() < eff_checkout and curr_t < self.simulation_end - timedelta(seconds=40):
                    self._emit("checkout_started", "checkout", curr_t, user_id, session_id, device, browser, app_version)
                    curr_t += timedelta(seconds=int(self.rng.integers(8, 20)))

                    if not (self.rng.random() < payment_attempt_prob):
                        if curr_t < self.simulation_end:
                            self._emit("checkout_abandoned", "checkout", curr_t, user_id, session_id, device, browser, app_version)
                        continue

                    order_counter += 1
                    order_time = curr_t
                    amount = round(float(self.rng.choice([29.0, 79.0, 199.0, 499.0], p=[0.4, 0.35, 0.2, 0.05])), 2)
                    order_status, failure_reason = "completed", None

                    for eff in failure_effects:
                        if order_status != "completed":
                            break
                        if eff["device"] and device != eff["device"]:
                            continue
                        if eff["min_version"] and _version_key(app_version) < _version_key(eff["min_version"]):
                            continue
                        intensity = self._intensity(order_time, eff["start"], eff["end"])
                        if intensity > 0 and self.rng.random() < eff["prob"] * intensity:
                            order_status, failure_reason = "failed", eff["reason"]
                            key = "regression_failed_orders" if eff["kind"] == "regression" else "outage_failed_orders"
                            self.truth[key] += 1

                    if order_status == "completed" and self.rng.random() < 0.04:
                        order_status, failure_reason = "failed", "card_declined"

                    if order_time < self.simulation_end:
                        order_records.append({
                            "order_id": f"ord_{order_counter}", "user_id": user_id, "timestamp": order_time.isoformat(),
                            "amount": amount, "currency": self.config.default_currency,
                            "status": order_status, "failure_reason": failure_reason,
                        })
                        curr_t = order_time + timedelta(seconds=2)
                        if curr_t < self.simulation_end:
                            if order_status == "completed":
                                dropped = False
                                if (tracking_drop_start and browser == tracking_browser and order_time >= tracking_drop_start):
                                    inten = self._intensity(order_time, tracking_drop_start)
                                    if self.rng.random() < tracking_drop_rate * inten:
                                        dropped = True
                                        self.truth["tracking_dropped_events"] += 1
                                if not dropped:
                                    self._emit("order_completed", "checkout", curr_t, user_id, session_id, device, browser, app_version)
                            else:
                                self._emit("checkout_failed", "checkout", curr_t, user_id, session_id, device, browser, app_version)

        events_df = pd.DataFrame(self._events)
        orders_df = pd.DataFrame(order_records)
        if not events_df.empty:
            events_df.sort_values(by="timestamp", kind="stable", inplace=True)
            events_df.reset_index(drop=True, inplace=True)
        if not orders_df.empty:
            orders_df.sort_values(by="timestamp", inplace=True)
            orders_df.reset_index(drop=True, inplace=True)
        return events_df, orders_df

    def _generate_bot_sessions(self, user_id: str, signup: datetime, days_active: float, remaining_seconds: float) -> None:
        """Bots: many fast page views from HeadlessChrome, no checkout, no orders (aux stream only)."""
        version = self._release_log[-1]["version"]
        for s in range(int(self.aux.poisson(2.0 * days_active))):
            t = signup + timedelta(seconds=int(self.aux.integers(0, max(1, int(remaining_seconds - 300)))))
            if t >= self.simulation_end - timedelta(seconds=60):
                continue
            version = self._version_at(t, timedelta(0))
            for i in range(int(self.aux.integers(8, 16))):
                ts = t + timedelta(seconds=i * 2)
                if ts >= self.simulation_end:
                    break
                self._emit("page_view", "landing_page", ts, user_id, f"sess_{user_id}_{s + 1}", "desktop", BOT_BROWSER, version)

    # ------------------------------------------------------------------ feedback & metrics (unchanged semantics)
    def _generate_user_feedback(self, users_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        feedback_counter = 8000
        for _, order in orders_df.iterrows():
            user_id = order["user_id"]
            order_time = datetime.fromisoformat(order["timestamp"])
            if order["status"] == "failed":
                if self.rng.random() < 0.65:
                    max_sec = int((self.simulation_end - order_time).total_seconds())
                    if max_sec <= 60:
                        continue
                    fb_time = order_time + timedelta(seconds=int(self.rng.integers(60, min(3600, max_sec))))
                    if fb_time >= self.simulation_end:
                        continue
                    feedback_counter += 1
                    rating = 1 if self.rng.random() < 0.8 else 2
                    reason = order.get("failure_reason")
                    if reason == "payment_gateway_timeout":
                        comment = "Checkout timed out while processing payment on mobile app. Very frustrating!"
                    else:
                        comment = "Payment declined unexpectedly at checkout."
                    records.append({"feedback_id": f"fb_{feedback_counter}", "user_id": user_id,
                                    "timestamp": fb_time.isoformat(), "rating": rating, "category": "billing", "comment": comment})
            else:
                if self.rng.random() < self.config.baseline_feedback_rate:
                    max_sec = int((self.simulation_end - order_time).total_seconds())
                    if max_sec <= 60:
                        continue
                    fb_time = order_time + timedelta(seconds=int(self.rng.integers(60, min(21600, max_sec))))
                    if fb_time >= self.simulation_end:
                        continue
                    feedback_counter += 1
                    rating = int(self.rng.choice([4, 5], p=[0.3, 0.7]))
                    records.append({"feedback_id": f"fb_{feedback_counter}", "user_id": user_id,
                                    "timestamp": fb_time.isoformat(), "rating": rating, "category": "product",
                                    "comment": "Quick and seamless checkout experience."})

        for _, user in users_df.iterrows():
            if user["browser"] == BOT_BROWSER:
                continue
            if self.rng.random() < 0.03:
                user_signup = max(datetime.fromisoformat(user["signup_timestamp"]), self.simulation_start)
                max_sec = int((self.simulation_end - user_signup).total_seconds())
                if max_sec <= 3600:
                    continue
                fb_time = user_signup + timedelta(seconds=int(self.rng.integers(3600, max_sec)))
                if fb_time >= self.simulation_end:
                    continue
                feedback_counter += 1
                rating = int(self.rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
                records.append({"feedback_id": f"fb_{feedback_counter}", "user_id": user["user_id"],
                                "timestamp": fb_time.isoformat(), "rating": rating, "category": "general",
                                "comment": "App interface is intuitive and fast."})
        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by="timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _generate_daily_product_metrics(
        self, users_df: pd.DataFrame, events_df: pd.DataFrame, orders_df: pd.DataFrame, feedback_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Roll up daily aggregate product metrics from granular event and order tables."""
        metrics_records = []
        start_date = self.simulation_start.date()
        for day_idx in range(self.config.num_days):
            date_str = (start_date + timedelta(days=day_idx)).isoformat()
            daily_signups = int(users_df["signup_timestamp"].str.startswith(date_str).sum()) if not users_df.empty else 0
            daily_dau = daily_events = 0
            if not events_df.empty:
                day_events = events_df[events_df["timestamp"].str.startswith(date_str)]
                daily_events, daily_dau = len(day_events), day_events["user_id"].nunique()
            completed_orders = failed_orders = 0
            gross_revenue = 0.0
            if not orders_df.empty:
                day_orders = orders_df[orders_df["timestamp"].str.startswith(date_str)]
                completed = day_orders[day_orders["status"] == "completed"]
                completed_orders = len(completed)
                failed_orders = int((day_orders["status"] == "failed").sum())
                gross_revenue = round(float(completed["amount"].sum()), 2)
            avg_rating = 4.2
            if not feedback_df.empty:
                day_fb = feedback_df[feedback_df["timestamp"].str.startswith(date_str)]
                if not day_fb.empty:
                    avg_rating = round(float(day_fb["rating"].mean()), 2)
            metrics_records.append({
                "date": date_str, "daily_active_users": daily_dau, "new_signups": daily_signups,
                "total_events": daily_events, "completed_orders": completed_orders, "failed_orders": failed_orders,
                "gross_revenue": gross_revenue, "avg_feedback_rating": avg_rating,
            })
        return pd.DataFrame(metrics_records)
