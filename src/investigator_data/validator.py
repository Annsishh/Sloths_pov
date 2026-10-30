"""Validation module to verify integrity, temporal consistency, simulation boundaries, and ground-truth separation."""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
import pandas as pd
from investigator_data.schemas import TABLE_COLUMNS
from investigator_data.scenarios import ScenarioDefinition
from investigator_data.config import GeneratorConfig


class DataValidator:
    """Runs programmatic validation checks across synthetic datasets."""

    def __init__(
        self,
        tables: Dict[str, pd.DataFrame],
        scenario: ScenarioDefinition | None = None,
        config: GeneratorConfig | None = None,
    ):
        self.tables = tables
        self.scenario = scenario
        self.config = config

    def validate_all(self) -> Tuple[bool, List[str]]:
        """Run all validation checks and return (is_valid, list_of_errors)."""
        errors = []
        errors.extend(self.check_schema_columns())
        errors.extend(self.check_referential_integrity())
        errors.extend(self.check_temporal_consistency())
        errors.extend(self.check_simulation_window_bounds())
        errors.extend(self.check_release_consistency())
        if self.config:
            errors.extend(self.check_config_knobs())
        if self.scenario:
            errors.extend(self.check_scenario_effects())
            errors.extend(self.check_phase3_scenario_effects())
        return len(errors) == 0, errors

    def check_schema_columns(self) -> List[str]:
        errors = []
        for table_name, expected_cols in TABLE_COLUMNS.items():
            if table_name not in self.tables:
                errors.append(f"Missing table: {table_name}")
                continue
            df = self.tables[table_name]
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                errors.append(f"Table '{table_name}' missing expected columns: {missing_cols}")
            extra_cols = [c for c in df.columns if c not in expected_cols]
            if extra_cols:
                errors.append(f"Table '{table_name}' has unexpected columns (schema leak): {extra_cols}")
        return errors

    def check_referential_integrity(self) -> List[str]:
        errors = []
        users_df = self.tables.get("users", pd.DataFrame())
        if users_df.empty or "user_id" not in users_df.columns:
            return ["Cannot verify referential integrity: users table is empty or missing user_id"]

        valid_user_ids = set(users_df["user_id"])

        for table_name in ["events", "orders", "user_feedback", "feature_exposures"]:
            df = self.tables.get(table_name, pd.DataFrame())
            if df.empty or "user_id" not in df.columns:
                continue
            invalid_ids = set(df["user_id"]) - valid_user_ids
            if invalid_ids:
                errors.append(
                    f"Referential integrity violation in '{table_name}': "
                    f"{len(invalid_ids)} orphan user_id(s) found (e.g. {list(invalid_ids)[:3]})"
                )
        return errors

    def check_simulation_window_bounds(self) -> List[str]:
        """Ensure all timestamps across all tables stay strictly within [start_time, end_time)

        Validates against actual generator configuration window using parsed datetimes
        with an exclusive upper boundary.
        """
        errors = []
        if self.config:
            start_dt = self.config.start_time
            from datetime import timedelta
            end_dt = self.config.start_time + timedelta(days=self.config.num_days)
        elif self.scenario:
            start_dt = self.scenario.start_date
            end_dt = self.scenario.end_date
        else:
            return errors

        timestamp_columns = {
            "users": "signup_timestamp",
            "events": "timestamp",
            "orders": "timestamp",
            "user_feedback": "timestamp",
            "feature_exposures": "first_exposed_timestamp",
            "releases": "deployed_timestamp",
        }

        for table_name, ts_col in timestamp_columns.items():
            df = self.tables.get(table_name, pd.DataFrame())
            if df.empty or ts_col not in df.columns:
                continue

            parsed_ts = pd.to_datetime(df[ts_col], utc=True)

            # Check lower bound (inclusive): timestamp >= start_dt. Pre-existing users may legitimately have
            # signed up before the window opens (existing_user_share > 0).
            if table_name == "users" and self.config and self.config.existing_user_share > 0:
                before_start_mask = parsed_ts < start_dt - pd.Timedelta(days=91)
            else:
                before_start_mask = parsed_ts < start_dt
            if before_start_mask.any():
                count = before_start_mask.sum()
                min_val = parsed_ts[before_start_mask].min()
                errors.append(
                    f"Simulation boundary violation: table '{table_name}' has {count} "
                    f"records before start {start_dt.isoformat()} (earliest: {min_val.isoformat()})"
                )

            # Check upper bound (exclusive): timestamp < end_dt
            after_or_at_end_mask = parsed_ts >= end_dt
            if after_or_at_end_mask.any():
                count = after_or_at_end_mask.sum()
                max_val = parsed_ts[after_or_at_end_mask].max()
                errors.append(
                    f"Simulation boundary violation: table '{table_name}' has {count} "
                    f"records at or after exclusive end {end_dt.isoformat()} (latest: {max_val.isoformat()})"
                )

        return errors

    def check_temporal_consistency(self) -> List[str]:
        errors = []
        users_df = self.tables.get("users", pd.DataFrame())
        if users_df.empty:
            return ["Users table empty for temporal validation"]

        user_signups = dict(zip(users_df["user_id"], users_df["signup_timestamp"]))

        # Events, orders and feedback cannot occur before their user signed up (vectorised; reports the first offender).
        for table, id_col, label in (("events", "event_id", "event"), ("orders", "order_id", "order"),
                                     ("user_feedback", "feedback_id", "feedback")):
            df = self.tables.get(table, pd.DataFrame())
            if df.empty:
                continue
            signup = df["user_id"].map(user_signups)
            bad = df[(signup.notna()) & (df["timestamp"] < signup)]
            if not bad.empty:
                row = bad.iloc[0]
                errors.append(
                    f"Temporal violation: {label} {row[id_col]} timestamp {row['timestamp']} "
                    f"is earlier than user {row['user_id']} signup {user_signups[row['user_id']]}"
                )

        return errors

    def check_scenario_effects(self) -> List[str]:
        """Check whether injected scenario creates its expected observable effects."""
        errors = []
        if not self.scenario:
            return errors

        orders_df = self.tables.get("orders", pd.DataFrame())
        events_df = self.tables.get("events", pd.DataFrame())
        users_df = self.tables.get("users", pd.DataFrame())

        if self.scenario.category == "product_regression":
            if orders_df.empty:
                errors.append("Product regression check failed: orders table empty")
            else:
                users_lookup = dict(zip(users_df["user_id"], users_df["device_type"]))
                mobile_orders = orders_df[orders_df["user_id"].map(users_lookup) == "mobile"]
                failed_mobile = mobile_orders[mobile_orders["status"] == "failed"]
                if len(failed_mobile) == 0:
                    errors.append("Product regression check failed: expected mobile order failures but found 0")

        elif self.scenario.category == "acquisition_shift":
            surge_start = self.scenario.parameters.get("surge_start")
            if surge_start and not users_df.empty:
                post_surge_users = users_df[users_df["signup_timestamp"] >= surge_start]
                paid_social_pct = (post_surge_users["acquisition_channel"] == "paid_social").mean()
                if paid_social_pct < 0.20:
                    errors.append(
                        f"Acquisition shift check failed: expected >20% paid_social post-surge, found {paid_social_pct:.2%}"
                    )

        elif self.scenario.category == "tracking_failure":
            if not events_df.empty and not orders_df.empty:
                drop_start = self.scenario.parameters.get("tracking_drop_start")
                user_browser = dict(zip(users_df["user_id"], users_df["browser"]))
                
                # Comparable segment & window: Safari post-drop
                # Work on a copy: mutating the shared table leaked a derived `browser` column into orders.csv.
                orders_df = orders_df.assign(browser=orders_df["user_id"].map(user_browser))
                safari_orders = orders_df[
                    (orders_df["browser"] == "Safari")
                    & (orders_df["status"] == "completed")
                    & (orders_df["timestamp"] >= drop_start)
                ]
                safari_events = events_df[
                    (events_df["browser"] == "Safari")
                    & (events_df["event_name"] == "order_completed")
                    & (events_df["timestamp"] >= drop_start)
                ]
                if len(safari_orders) > 0:
                    safari_tracking_ratio = len(safari_events) / len(safari_orders)
                    if safari_tracking_ratio > 0.35:
                        errors.append(
                            f"Tracking failure check failed: Safari tracking ratio ({safari_tracking_ratio:.2f}) "
                            f"is too high given backend orders ({len(safari_orders)}) vs events ({len(safari_events)})"
                        )

        return errors

    # ------------------------------------------------------------------ Phase 3 checks
    @staticmethod
    def _utc(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, utc=True)

    def check_release_consistency(self) -> List[str]:
        """Every app_version seen in events must exist in the release log, and never before its deploy time
        (minus nothing: adoption only delays updates)."""
        errors: List[str] = []
        events, releases = self.tables.get("events", pd.DataFrame()), self.tables.get("releases", pd.DataFrame())
        if events.empty or releases.empty or "app_version" not in events.columns:
            return errors
        known = dict(zip(releases["version"], self._utc(releases["deployed_timestamp"])))
        first_version = releases.sort_values("deployed_timestamp").iloc[0]["version"]
        unknown = set(events["app_version"]) - set(known)
        if unknown:
            errors.append(f"events reference app_version(s) missing from releases: {sorted(unknown)}")
            return errors
        ts = self._utc(events["timestamp"])
        for version, deployed in known.items():
            if version == first_version:
                continue
            early = ts[(events["app_version"] == version)] < deployed
            if early.any():
                errors.append(f"{int(early.sum())} event(s) on {version} occur before its deployment {deployed.isoformat()}")
        return errors

    def check_config_knobs(self) -> List[str]:
        """Each realism knob that is switched on must leave its fingerprint in the data."""
        errors: List[str] = []
        c = self.config
        events = self.tables.get("events", pd.DataFrame())
        users = self.tables.get("users", pd.DataFrame())
        if events.empty:
            return errors
        if c.duplicate_event_rate > 0:
            dup = events.duplicated(subset=["user_id", "session_id", "timestamp", "event_name"], keep=False).sum()
            if dup == 0:
                errors.append("duplicate_event_rate > 0 but no duplicated events were generated")
        elif events.duplicated(subset=["user_id", "session_id", "timestamp", "event_name"]).any():
            errors.append("duplicated events present although duplicate_event_rate == 0")
        if c.bot_traffic_share > 0:
            if not (users["browser"] == "HeadlessChrome").any():
                errors.append("bot_traffic_share > 0 but no bot users were generated")
        elif (users["browser"] == "HeadlessChrome").any():
            errors.append("bot users present although bot_traffic_share == 0")
        for from_version, old, new in c.event_renames:
            names = set(events["event_name"])
            if new not in names:
                errors.append(f"event rename {old}->{new} configured but '{new}' never appears")
            renamed_rows = events[events["event_name"] == new]
            if not renamed_rows.empty and any(
                tuple(int(x) for x in v.lstrip("v").split(".")) < tuple(int(x) for x in from_version.lstrip("v").split("."))
                for v in renamed_rows["app_version"].unique()
            ):
                errors.append(f"'{new}' appears on versions older than {from_version}")
        if c.identity_split_share > 0 and not users.empty:
            keys = users.groupby(["signup_timestamp", "acquisition_channel", "country", "plan_tier"]).size()
            if not (keys > 1).any():
                errors.append("identity_split_share > 0 but no split identities were generated")
        if c.weekly_seasonality >= 0.5 and c.noise_level == 0 and c.num_days >= 14:
            wd = self._utc(events["timestamp"]).dt.weekday
            by_day = events.groupby(self._utc(events["timestamp"]).dt.date).size()
            days = pd.Series(by_day.index).map(lambda d: d.weekday()).to_numpy()
            weekend, weekday = by_day.to_numpy()[days >= 5], by_day.to_numpy()[days < 5]
            if len(weekend) >= 2 and weekend.mean() >= 0.95 * weekday.mean():
                errors.append("weekly seasonality configured but weekend volume is not lower than weekday volume")
        return errors

    def check_phase3_scenario_effects(self) -> List[str]:
        errors: List[str] = []
        sc, p = self.scenario, self.scenario.parameters
        orders = self.tables.get("orders", pd.DataFrame())
        events = self.tables.get("events", pd.DataFrame())
        users = self.tables.get("users", pd.DataFrame())
        if orders.empty:
            return errors
        o_ts = self._utc(orders["timestamp"])

        if sc.category == "pricing_change":
            cut = pd.Timestamp(p["conversion_effects"][0]["start"])
            starts = events[events["event_name"] == "checkout_started"]
            e_ts = self._utc(starts["timestamp"])
            days_pre = max(1.0, (cut - pd.Timestamp(self.config.start_time)).days) if self.config else 15
            days_post = max(1.0, (pd.Timestamp(self.config.start_time) + pd.Timedelta(days=self.config.num_days) - cut).days) if self.config else 15
            if (e_ts >= cut).sum() / days_post > 0.85 * (e_ts < cut).sum() / days_pre:
                errors.append("Pricing check failed: checkout starts per day did not fall after the price change")
        if sc.category in ("overlapping_causes",):
            if int((orders["failure_reason"] == "payment_gateway_timeout").sum()) == 0:
                errors.append("Overlapping check failed: no mobile gateway failures")
            if not events.empty and not (events["browser"] == "Safari").any():
                errors.append("Overlapping check failed: no Safari events")
        if sc.category == "third_party_outage":
            eff = p["order_failure_effects"][0]
            out = orders[orders["failure_reason"] == eff["reason"]]
            if out.empty:
                errors.append("Outage check failed: no outage failures were generated")
            else:
                ts = self._utc(out["timestamp"])
                if ts.min() < pd.Timestamp(eff["start"]) or ts.max() >= pd.Timestamp(eff["end"]):
                    errors.append("Outage check failed: outage failures fall outside the configured window")
                versions = events.loc[events["user_id"].isin(out["user_id"]), "app_version"].nunique()
                if versions < 2:
                    errors.append("Outage check failed: failures are not spread across app versions")
        if sc.category == "denominator_effect":
            cut = pd.Timestamp(p["paid_social_cut_start"])
            chan = events["user_id"].map(dict(zip(users["user_id"], users["acquisition_channel"])))
            ev_ts = self._utc(events["timestamp"])
            pre = (chan[ev_ts < cut] == "paid_social").mean()
            post = (chan[ev_ts >= cut] == "paid_social").mean()
            if not post < 0.6 * pre:
                errors.append(f"Denominator check failed: paid_social share of traffic did not fall ({pre:.2%} -> {post:.2%})")
        if sc.category == "gradual_or_cohort_effect":
            ts = self._utc(events["timestamp"])
            n = self.config.num_days if self.config else 45
            third = pd.Timedelta(days=n / 3)
            start = pd.Timestamp(self.config.start_time) if self.config else ts.min()
            early = ((ts >= start + third) & (ts < start + 2 * third)).sum()
            late = (ts >= start + 2 * third).sum()
            if not late < early:
                errors.append("Gradual-decay check failed: activity did not decline over time")
        if sc.category == "seasonality_or_calendar" and self.config:
            hol = pd.Timestamp(p["holiday"]).date()
            by_day = events.groupby(self._utc(events["timestamp"]).dt.date).size()
            same_wd = [d for d in by_day.index if d.weekday() == hol.weekday() and d != hol]
            if hol in by_day.index and same_wd and by_day[hol] >= 0.8 * by_day[same_wd].mean():
                errors.append("Seasonal check failed: the holiday shows no dip versus other same-weekday days")
        if sc.category == "acquisition_shift" and "decoy_release_id" in p:
            if p["decoy_release_id"] not in set(self.tables["releases"]["release_id"]):
                errors.append("Decoy check failed: decoy release missing from the release log")
        return errors
