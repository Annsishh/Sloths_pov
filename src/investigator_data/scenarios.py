"""Scenario definitions and ground-truth specifications.

Separation Principle:
The ScenarioDefinition holds ground-truth parameters, injected causes,
affected segments, and expected effects. This is used by the simulation engine
and saved strictly into the `ground_truth` folder for benchmarking.
The investigator agent must NEVER inspect the `ground_truth` folder.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

@dataclass
class ScenarioDefinition:
    scenario_id: str
    name: str
    category: str  # e.g., 'baseline', 'product_regression', 'acquisition_shift', 'tracking_failure'
    description: str
    ground_truth_cause: str
    start_date: datetime
    end_date: datetime
    affected_components: List[str] = field(default_factory=list)
    affected_dimensions: Dict[str, Any] = field(default_factory=dict)
    expected_observable_effects: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_ground_truth_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "ground_truth_cause": self.ground_truth_cause,
            "simulation_window": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "affected_components": self.affected_components,
            "affected_dimensions": self.affected_dimensions,
            "expected_observable_effects": self.expected_observable_effects,
            "parameters": self.parameters,
        }


def get_baseline_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="baseline",
        name="Healthy Baseline SaaS Operations",
        category="baseline",
        description="Normal, steady-state user activity, conversion rates, and feedback.",
        ground_truth_cause="None. Normal business operations with typical variance.",
        start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        affected_components=[],
        affected_dimensions={},
        expected_observable_effects={
            "order_completion_rate": "Stable around 90-95% of attempted orders",
            "daily_active_users": "Stable weekly seasonality without persistent drops",
            "feedback_rating": "Average rating ~4.1 to 4.4",
            "event_tracking": "Consistent session-to-order funnel across all browsers and devices",
        },
        parameters={},
    )


def get_product_regression_scenario() -> ScenarioDefinition:
    """Scenario 1: Checkout payment failure regression on mobile app v2.4.0 deployed on Sept 15."""
    return ScenarioDefinition(
        scenario_id="product_regression",
        name="Mobile Checkout Gateway Bug",
        category="product_regression",
        description=(
            "Release v2.4.0 deployed on 2026-09-15 introduced a bug in the mobile checkout payment "
            "gateway SDK, causing ~75% of checkout attempts on mobile to fail with payment_gateway_timeout."
        ),
        ground_truth_cause=(
            "Release v2.4.0 introduced an unhandled network timeout in the mobile checkout SDK. "
            "Mobile orders fail at 75% rate starting from deployment timestamp 2026-09-15 10:00:00 UTC."
        ),
        start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        affected_components=["mobile_checkout", "payment_sdk", "release_v2.4.0"],
        affected_dimensions={
            "device_type": "mobile",
            "release_version": "v2.4.0",
            "effective_from": "2026-09-15T10:00:00Z",
        },
        expected_observable_effects={
            "orders": "Sharp spike in status='failed' with failure_reason='payment_gateway_timeout' exclusively on mobile",
            "daily_product_metrics": "Drop in completed_orders and gross_revenue after Sept 15; failed_orders surges",
            "user_feedback": "Spike in negative feedback (rating 1-2) with category='billing'/'checkout' on mobile",
            "releases": "Correlates directly with deployment of release_id 'rel_240' on 2026-09-15",
        },
        parameters={
            "release_id": "rel_240",
            "release_version": "v2.4.0",
            "deploy_time": "2026-09-15T10:00:00Z",
            "mobile_order_failure_prob": 0.75,
            "negative_feedback_boost": 0.25,
        },
    )


def get_acquisition_shift_scenario() -> ScenarioDefinition:
    """Scenario 2: Low-quality traffic surge from low-intent social ad campaign starting Sept 12."""
    return ScenarioDefinition(
        scenario_id="acquisition_shift",
        name="Paid Social Low-Intent Traffic Surge",
        category="acquisition_shift",
        description=(
            "Starting 2026-09-12, an aggressive paid social advertising campaign tripes new user signups "
            "from channel 'paid_social', but these users have extremely low intent, high bounce rates, and "
            "negligible order conversion (~1% vs baseline 8%)."
        ),
        ground_truth_cause=(
            "Paid acquisition team launched low-intent ad campaigns on social media. Overall signup volume surges, "
            "causing aggregate conversion rate and average engagement per user to plunge despite healthy core product."
        ),
        start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        affected_components=["acquisition_funnel", "paid_social"],
        affected_dimensions={
            "acquisition_channel": "paid_social",
            "effective_from": "2026-09-12T00:00:00Z",
        },
        expected_observable_effects={
            "users": "Surge in new signups with acquisition_channel='paid_social' from Sept 12 onwards",
            "events": "Paid social users generate fewer pageviews per session (1-2 pages, high bounce)",
            "orders": "Overall conversion drops, but segmented by channel, organic/direct/search remain steady",
            "daily_product_metrics": "New signups up significantly; average revenue per user down",
        },
        parameters={
            "surge_start": "2026-09-12T00:00:00Z",
            "paid_social_volume_multiplier": 3.0,
            "paid_social_conversion_rate": 0.012,
            "paid_social_events_multiplier": 0.4,
            "paid_social_checkout_start_prob": 0.03,  # lower checkout initiation
        },
    )


def get_tracking_failure_scenario() -> ScenarioDefinition:
    """Scenario 3: Analytics event tracking drops checkout_completed on Safari browsers from Sept 18."""
    return ScenarioDefinition(
        scenario_id="tracking_failure",
        name="Safari Event Tracking Dropped on Checkout",
        category="tracking_failure",
        description=(
            "A JavaScript tracking script update deployed on 2026-09-18 introduced an issue where "
            "'order_completed' analytics events are dropped on Safari browsers (~90% missing in event stream), "
            "even though the actual backend orders and revenue succeed normally."
        ),
        ground_truth_cause=(
            "Analytics client-side SDK telemetry dropped tracking events on Safari (ITP/cookie or script syntax bug). "
            "Actual backend orders succeeded, causing a discrepancy between analytics events and financial orders."
        ),
        start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        affected_components=["analytics_sdk", "event_tracking"],
        affected_dimensions={
            "browser": "Safari",
            "event_name": "order_completed",
            "effective_from": "2026-09-18T00:00:00Z",
        },
        expected_observable_effects={
            "events": "order_completed events on Safari drop by ~90% after Sept 18",
            "orders": "Real backend orders remain normal on Safari, creating a divergence between events and orders tables",
            "daily_product_metrics": "Total tracked events drop slightly on Safari funnel; financial revenue unaffected",
            "future_investigation_hook": (
                "Investigator comparing orders table vs events tracking will detect that backend payment logs "
                "are intact while telemetry events dropped, pinpointing an instrumentation/analytics bug."
            ),
        },
        parameters={
            "tracking_drop_start": "2026-09-18T00:00:00Z",
            "dropped_event": "order_completed",
            "affected_browser": "Safari",
            "drop_probability": 0.90,
        },
    )


SCENARIO_REGISTRY = {
    "baseline": get_baseline_scenario,
    "product_regression": get_product_regression_scenario,
    "acquisition_shift": get_acquisition_shift_scenario,
    "tracking_failure": get_tracking_failure_scenario,
}

