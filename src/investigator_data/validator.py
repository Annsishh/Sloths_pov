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
        if self.scenario:
            errors.extend(self.check_scenario_effects())
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

            # Check lower bound (inclusive): timestamp >= start_dt
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

        # 1. Events cannot occur before user signup
        events_df = self.tables.get("events", pd.DataFrame())
        if not events_df.empty:
            for _, row in events_df.iterrows():
                u_signup = user_signups.get(row["user_id"])
                if u_signup and row["timestamp"] < u_signup:
                    errors.append(
                        f"Temporal violation: event {row['event_id']} timestamp {row['timestamp']} "
                        f"is earlier than user {row['user_id']} signup {u_signup}"
                    )
                    break

        # 2. Orders cannot occur before user signup
        orders_df = self.tables.get("orders", pd.DataFrame())
        if not orders_df.empty:
            for _, row in orders_df.iterrows():
                u_signup = user_signups.get(row["user_id"])
                if u_signup and row["timestamp"] < u_signup:
                    errors.append(
                        f"Temporal violation: order {row['order_id']} timestamp {row['timestamp']} "
                        f"is earlier than user {row['user_id']} signup {u_signup}"
                    )
                    break

        # 3. User feedback cannot occur before user signup
        feedback_df = self.tables.get("user_feedback", pd.DataFrame())
        if not feedback_df.empty:
            for _, row in feedback_df.iterrows():
                u_signup = user_signups.get(row["user_id"])
                if u_signup and row["timestamp"] < u_signup:
                    errors.append(
                        f"Temporal violation: feedback {row['feedback_id']} timestamp {row['timestamp']} "
                        f"is earlier than user {row['user_id']} signup {u_signup}"
                    )
                    break

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
                orders_df["browser"] = orders_df["user_id"].map(user_browser)
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
