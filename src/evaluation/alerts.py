"""Turn an alert spec (metric + windows) into the Observation that starts an investigation.

Uses ONLY investigator-accessible tables. The alert is what a monitoring system would show; it contains no cause.
"""

from __future__ import annotations

from typing import Any, Dict

from analytics.loader import DataLoader
from analytics.models import TimeWindow
from analytics.utils import apply_time_filter
from investigation.models import Observation

METRIC_DESCRIPTIONS = {
    "completed_orders_per_day": "completed backend orders per day",
    "order_success_rate": "share of payment attempts that completed",
    "orders_per_100_sessions": "completed backend orders per 100 sessions",
    "tracked_orders_per_100_sessions": "order_completed telemetry events per 100 sessions",
}


def metric_value(loader: DataLoader, metric: str, window: TimeWindow) -> float:
    orders = apply_time_filter(loader.load("orders"), "timestamp", window)
    if metric == "completed_orders_per_day":
        return float((orders["status"] == "completed").sum() / window.duration_days())
    if metric == "order_success_rate":
        return float((orders["status"] == "completed").mean()) if len(orders) else float("nan")
    events = apply_time_filter(loader.load("events"), "timestamp", window)
    sessions = events["session_id"].nunique()
    if sessions == 0:
        return float("nan")
    if metric == "orders_per_100_sessions":
        return float((orders["status"] == "completed").sum() / sessions * 100)
    if metric == "tracked_orders_per_100_sessions":
        return float((events["event_name"] == "order_completed").sum() / sessions * 100)
    raise ValueError(f"Unknown alert metric '{metric}'")


def build_observation(loader: DataLoader, alert: Dict[str, Any]) -> Observation:
    metric = alert["metric"]
    base = TimeWindow.from_iso(*alert["baseline_window"])
    comp = TimeWindow.from_iso(*alert["comparison_window"])
    b, c = metric_value(loader, metric, base), metric_value(loader, metric, comp)
    if b != b or c != c:
        raise ValueError(f"Metric '{metric}' is undefined in one of the alert windows.")
    return Observation(
        metric_name=metric, baseline_window=base, comparison_window=comp,
        baseline_value=round(b, 6), comparison_value=round(c, 6),
        absolute_change=round(c - b, 6), relative_change=round((c - b) / b, 6) if b != 0 else None,
        description=f"Alert: {METRIC_DESCRIPTIONS.get(metric, metric)} changed between {base} and {comp}.",
    )
