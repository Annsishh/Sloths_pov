"""Investigator data schemas and entity column definitions."""

from typing import Dict, List

TABLE_COLUMNS: Dict[str, List[str]] = {
    "users": [
        "user_id",
        "signup_timestamp",
        "acquisition_channel",
        "country",
        "device_type",
        "browser",
        "plan_tier",
    ],
    "releases": [
        "release_id",
        "version",
        "deployed_timestamp",
        "component",
        "description",
        "status",
    ],
    "feature_exposures": [
        "exposure_id",
        "user_id",
        "feature_key",
        "variant",
        "first_exposed_timestamp",
    ],
    "events": [
        "event_id",
        "user_id",
        "session_id",
        "timestamp",
        "event_name",
        "page",
        "device_type",
        "browser",
        "app_version",
    ],
    "orders": [
        "order_id",
        "user_id",
        "timestamp",
        "amount",
        "currency",
        "status",
        "failure_reason",
    ],
    "user_feedback": [
        "feedback_id",
        "user_id",
        "timestamp",
        "rating",
        "category",
        "comment",
    ],
    "daily_product_metrics": [
        "date",
        "daily_active_users",
        "new_signups",
        "total_events",
        "completed_orders",
        "failed_orders",
        "gross_revenue",
        "avg_feedback_rating",
    ],
}

