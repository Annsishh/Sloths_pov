"""Scenario definitions and ground-truth specifications.

Separation Principle:
The ScenarioDefinition holds ground-truth parameters, injected causes,
affected segments, and expected effects. This is used by the simulation engine
and saved strictly into the `ground_truth` folder for benchmarking.
The investigator agent must NEVER inspect the `ground_truth` folder.

Two pieces of a scenario are deliberately NOT ground truth:
  * `alert`: the metric and windows that triggered the investigation (what a monitoring system would show);
    written next to, not inside, `ground_truth/`.
  * `releases`: part of the observable `releases` table.
Everything else (causes, parameters, expected answer, shares) is ground truth.

Engine parameter keys (presence enables the mechanism, so causes can be combined):
  surge_start, paid_social_*            acquisition surge / low-intent social traffic
  deploy_time, mobile_order_failure_prob  mobile checkout bug in releases >= `release_version`
  tracking_drop_start, drop_probability   Safari order_completed events dropped
  order_failure_effects                 [{start, end, prob, reason, device?}] payment failures
  conversion_effects                    [{start, multiplier, channel?}] checkout-start multipliers
  tenure_decay_rate                     per-day churn hazard applied to every user's sessions
  paid_social_low_intent, paid_social_cut_start, paid_social_cut_fraction   denominator scenario
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Observable metrics an alert can reference (computed by the evaluation harness from investigator data).
ALERT_METRICS = (
    "completed_orders_per_day",        # backend completed orders / day
    "order_success_rate",              # completed / (completed + failed) backend orders
    "orders_per_100_sessions",         # backend completed orders per 100 sessions (conversion)
    "tracked_orders_per_100_sessions", # order_completed telemetry events per 100 sessions
)

UTC = timezone.utc


@dataclass
class ScenarioDefinition:
    scenario_id: str
    name: str
    category: str  # ground-truth cause family, e.g. 'product_regression', 'baseline', 'natural_variation'
    description: str
    ground_truth_cause: str
    start_date: datetime
    end_date: datetime
    affected_components: List[str] = field(default_factory=list)
    affected_dimensions: Dict[str, Any] = field(default_factory=dict)
    expected_observable_effects: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    # --- Phase 3 ---
    config_overrides: Dict[str, Any] = field(default_factory=dict)   # world knobs (GeneratorConfig fields)
    releases: Optional[List[Dict[str, Any]]] = None                  # observable release log; None = default four
    alert: Dict[str, Any] = field(default_factory=dict)              # NOT ground truth (see module docstring)
    expected_answer: Dict[str, Any] = field(default_factory=dict)    # what a correct investigation concludes

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
            "config_overrides": self.config_overrides,
            "expected_answer": self.expected_answer,
        }

    def to_alert_dict(self) -> Dict[str, Any]:
        return dict(self.alert)


def _d(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=UTC)


_START, _END = _d(2026, 9, 1), _d(2026, 10, 1)


def _release(release_id, version, when: datetime, component, description, status="success"):
    return {"release_id": release_id, "version": version, "deployed_timestamp": when.isoformat(),
            "component": component, "description": description, "status": status}


def default_releases() -> List[Dict[str, Any]]:
    """The standard four-release log (also the observable release table of most scenarios)."""
    return [
        _release("rel_238", "v2.3.8", _d(2026, 9, 3, 10), "core_web", "Performance improvements and caching updates for dashboard."),
        _release("rel_239", "v2.3.9", _d(2026, 9, 9, 14), "notifications_service", "Email delivery and in-app alert latency optimizations."),
        _release("rel_240", "v2.4.0", _d(2026, 9, 15, 10), "mobile_checkout", "Upgraded mobile payment processing SDK & one-click checkout."),
        _release("rel_241", "v2.4.1", _d(2026, 9, 23, 16), "reporting_engine", "Export CSV/PDF feature overhaul and analytics widgets."),
    ]


def _alert(metric, base, comp):
    return {"metric": metric, "baseline_window": list(base), "comparison_window": list(comp)}


# ---------------------------------------------------------------------------
# Original four scenarios
# ---------------------------------------------------------------------------

def get_baseline_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="baseline",
        name="Healthy Baseline SaaS Operations",
        category="baseline",
        description="Normal, steady-state user activity, conversion rates, and feedback, with weekly seasonality.",
        ground_truth_cause="None. Normal business operations with typical variance.",
        start_date=_START,
        end_date=_END,
        expected_observable_effects={
            "order_completion_rate": "Stable around 90-95% of attempted orders",
            "daily_active_users": "Weekly seasonality (quieter weekends) without persistent drops",
            "feedback_rating": "Average rating ~4.1 to 4.4",
            "event_tracking": "Consistent session-to-order funnel across all browsers and devices",
        },
        alert=_alert("completed_orders_per_day", ("2026-09-01", "2026-09-15"), ("2026-09-15", "2026-10-01")),
        expected_answer={"primary_cause": "natural_variation", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_product_regression_scenario() -> ScenarioDefinition:
    """Checkout payment failure regression on mobile app v2.4.0 deployed on Sept 15."""
    return ScenarioDefinition(
        scenario_id="product_regression",
        name="Mobile Checkout Gateway Bug",
        category="product_regression",
        description=(
            "Release v2.4.0 deployed on 2026-09-15 introduced a bug in the mobile checkout payment "
            "gateway SDK, causing ~75% of checkout attempts on mobile sessions running v2.4.0 or later to fail "
            "with payment_gateway_timeout. Mobile users adopt the new app version gradually."
        ),
        ground_truth_cause=(
            "Release v2.4.0 introduced an unhandled network timeout in the mobile checkout SDK. "
            "Mobile orders from app version >= v2.4.0 fail at 75% rate; users upgrade over ~2 days after 2026-09-15 10:00 UTC."
        ),
        start_date=_START,
        end_date=_END,
        affected_components=["mobile_checkout", "payment_sdk", "release_v2.4.0"],
        affected_dimensions={"device_type": "mobile", "release_version": "v2.4.0", "effective_from": "2026-09-15T10:00:00Z"},
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
            "effect_size": 0.75,
            "negative_feedback_boost": 0.25,
        },
        alert=_alert("order_success_rate", ("2026-09-01", "2026-09-15"), ("2026-09-15", "2026-10-01")),
        expected_answer={"primary_cause": "code_regression", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_acquisition_shift_scenario() -> ScenarioDefinition:
    """Low-quality traffic surge from low-intent social ad campaign starting Sept 12."""
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
        start_date=_START,
        end_date=_END,
        affected_components=["acquisition_funnel", "paid_social"],
        affected_dimensions={"acquisition_channel": "paid_social", "effective_from": "2026-09-12T00:00:00Z"},
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
            "paid_social_checkout_start_prob": 0.03,
        },
        alert=_alert("orders_per_100_sessions", ("2026-09-01", "2026-09-12"), ("2026-09-12", "2026-10-01")),
        expected_answer={"primary_cause": "traffic_mix_shift", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_tracking_failure_scenario() -> ScenarioDefinition:
    """Analytics event tracking drops order_completed on Safari browsers from Sept 18."""
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
        start_date=_START,
        end_date=_END,
        affected_components=["analytics_sdk", "event_tracking"],
        affected_dimensions={"browser": "Safari", "event_name": "order_completed", "effective_from": "2026-09-18T00:00:00Z"},
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
            "effect_size": 0.90,
        },
        alert=_alert("tracked_orders_per_100_sessions", ("2026-09-01", "2026-09-18"), ("2026-09-18", "2026-10-01")),
        expected_answer={"primary_cause": "tracking_loss", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


# ---------------------------------------------------------------------------
# Phase 3 scenarios
# ---------------------------------------------------------------------------

def get_noise_only_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="noise_only",
        name="Noise Only: No Real Cause",
        category="natural_variation",
        description="Nothing changed. High day-to-day noise produced an apparent drop between two adjacent weeks.",
        ground_truth_cause="None. The apparent decline is random day-to-day variation.",
        start_date=_START,
        end_date=_END,
        expected_observable_effects={"all": "No table contains a persistent shift; apparent drop is within normal variance."},
        parameters={},
        config_overrides={"noise_level": 0.35, "existing_user_share": 0.75},
        # chosen at generation time as the worst adjacent 7-day decline (mode below), still pure noise
        alert={"metric": "completed_orders_per_day", "mode": "worst_adjacent_week"},
        expected_answer={"primary_cause": "natural_variation", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_seasonal_dip_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="seasonal_dip",
        name="Weekend Seasonality Dip",
        category="seasonality_or_calendar",
        description=(
            "Strong weekly seasonality (weekends ~30% of weekdays) plus a public-holiday dip on 2026-09-21. "
            "An alert fires comparing a weekend/holiday window with the preceding Monday-Thursday."
        ),
        ground_truth_cause="Calendar effect: the comparison window covers a weekend and a holiday; same-weekday history shows it is normal.",
        start_date=_START,
        end_date=_d(2026, 10, 6),
        expected_observable_effects={"events/orders": "Volume dips on Saturdays/Sundays every week and on the holiday."},
        parameters={"holiday": "2026-09-21"},
        config_overrides={"weekly_seasonality": 1.0, "num_days": 35, "holiday_effects": [["2026-09-21", 0.4]], "existing_user_share": 0.75},
        alert=_alert("completed_orders_per_day", ("2026-09-14", "2026-09-19"), ("2026-09-19", "2026-09-22")),
        expected_answer={"primary_cause": "seasonality_or_calendar", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_pricing_change_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="pricing_change",
        name="Unobserved Price Increase",
        category="pricing_change",
        description=(
            "On 2026-09-16 prices were raised. Conversion fell ~30% across every segment, version and browser. "
            "No table records pricing, so the cause is not observable."
        ),
        ground_truth_cause="A price increase on 2026-09-16 reduced checkout starts uniformly. Pricing is absent from all observable tables.",
        start_date=_START,
        end_date=_END,
        affected_components=["pricing"],
        affected_dimensions={"effective_from": "2026-09-16T00:00:00Z", "scope": "all users"},
        expected_observable_effects={"events/orders": "Uniform drop in checkout starts and orders after Sept 16; no release, mix or tracking change."},
        parameters={
            "conversion_effects": [{"start": "2026-09-16T00:00:00+00:00", "multiplier": 0.55}],
            "effect_size": 0.45,
        },
        config_overrides={"existing_user_share": 0.75},
        alert=_alert("orders_per_100_sessions", ("2026-09-01", "2026-09-16"), ("2026-09-16", "2026-10-01")),
        expected_answer={
            "primary_cause": None, "secondary_causes": [], "decoys": [], "insufficient_evidence": True,
            "acceptable_data_request_keywords": ["pricing", "price", "policy"],
            "untestable_categories": ["pricing_or_policy_change"],
        },
    )


def get_decoy_release_scenario() -> ScenarioDefinition:
    rels = default_releases() + [
        _release("rel_250", "v2.5.0", _d(2026, 9, 14, 12), "reporting_engine", "Dashboard widget refresh and export tweaks."),
    ]
    return ScenarioDefinition(
        scenario_id="decoy_release",
        name="Decoy Release During Traffic Mix Shift",
        category="acquisition_shift",
        description=(
            "A low-intent paid-social surge starts 2026-09-14 00:00; release v2.5.0 (reporting only) ships at 12:00 the "
            "same day and is NOT the cause. The conversion drop is a traffic mix shift."
        ),
        ground_truth_cause="Low-intent paid-social traffic surge from 2026-09-14. Release rel_250 is coincidental and harmless.",
        start_date=_START,
        end_date=_END,
        affected_components=["acquisition_funnel", "paid_social"],
        affected_dimensions={"acquisition_channel": "paid_social", "effective_from": "2026-09-14T00:00:00Z"},
        expected_observable_effects={"users/orders": "paid_social share jumps; segment conversion unchanged; no release effect."},
        parameters={
            "surge_start": "2026-09-14T00:00:00Z",
            "paid_social_volume_multiplier": 3.0,
            "paid_social_conversion_rate": 0.012,
            "paid_social_events_multiplier": 0.4,
            "paid_social_checkout_start_prob": 0.03,
            "decoy_release_id": "rel_250",
        },
        releases=rels,
        alert=_alert("orders_per_100_sessions", ("2026-09-01", "2026-09-14"), ("2026-09-14", "2026-10-01")),
        expected_answer={"primary_cause": "traffic_mix_shift", "secondary_causes": [], "decoys": ["code_regression"], "insufficient_evidence": False},
    )


def get_overlapping_causes_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="overlapping_causes",
        name="Mobile Regression Plus Safari Tracking Gap",
        category="overlapping_causes",
        description=(
            "From 2026-09-15 two things go wrong together: release v2.4.0 breaks mobile checkout for part of the orders "
            "(real revenue loss), and a tracking bug drops order_completed events on Safari (measurement loss only). "
            "Both depress tracked orders."
        ),
        ground_truth_cause="(1) mobile checkout regression in v2.4.0 (real loss); (2) Safari telemetry loss (measurement artifact).",
        start_date=_START,
        end_date=_END,
        affected_components=["mobile_checkout", "analytics_sdk"],
        affected_dimensions={"device_type": "mobile", "browser": "Safari", "effective_from": "2026-09-15T10:00:00Z"},
        expected_observable_effects={
            "orders": "Mobile failures with payment_gateway_timeout on v2.4.0",
            "events": "Safari order_completed events missing while Safari backend orders are fine",
        },
        parameters={
            "release_id": "rel_240", "release_version": "v2.4.0", "deploy_time": "2026-09-15T10:00:00Z",
            "mobile_order_failure_prob": 0.5,
            "tracking_drop_start": "2026-09-15T10:00:00Z", "dropped_event": "order_completed",
            "affected_browser": "Safari", "drop_probability": 0.9,
        },
        alert=_alert("tracked_orders_per_100_sessions", ("2026-09-01", "2026-09-15"), ("2026-09-15", "2026-10-01")),
        expected_answer={
            "primary_cause": "code_regression", "secondary_causes": ["tracking_loss"], "decoys": [], "insufficient_evidence": False,
            "shares_source": "measured_effects",   # approximate shares are measured during generation
        },
    )


def get_gradual_decay_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="gradual_decay",
        name="Gradual Retention Decay",
        category="gradual_or_cohort_effect",
        description=(
            "Engagement erodes continuously: every user's chance of coming back falls with tenure (about 5% per day). "
            "There is no release, mix change or tracking issue and no single onset date."
        ),
        ground_truth_cause="Retention-driven decline: older cohorts steadily disengage. No single onset.",
        start_date=_START,
        end_date=_d(2026, 10, 16),
        expected_observable_effects={"events/orders": "Slow monotonic decline in sessions and orders; older cohorts retain worse."},
        parameters={"tenure_decay_rate": 0.05},
        config_overrides={"num_days": 45, "num_users": 1500, "existing_user_share": 0.75},
        alert=_alert("completed_orders_per_day", ("2026-09-15", "2026-09-30"), ("2026-09-30", "2026-10-15")),
        expected_answer={"primary_cause": "gradual_or_cohort_effect", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_denominator_effect_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="denominator_effect",
        name="Conversion Rises Because Low-Intent Traffic Fell",
        category="denominator_effect",
        description=(
            "Paid-social users are low-intent. On 2026-09-15 the paid-social campaign was cut by ~75%. Per-segment "
            "conversion is unchanged, but overall conversion rises because the denominator lost its low converters."
        ),
        ground_truth_cause="Fewer low-intent paid-social signups after 2026-09-15 raised the overall rate (mix/denominator effect); the product did not improve.",
        start_date=_START,
        end_date=_END,
        affected_components=["acquisition_funnel", "paid_social"],
        affected_dimensions={"acquisition_channel": "paid_social", "effective_from": "2026-09-15T00:00:00Z"},
        expected_observable_effects={"users/orders": "paid_social share falls; each segment's own conversion is unchanged; overall conversion rises."},
        parameters={
            "paid_social_low_intent": True,
            "paid_social_share": 0.35,
            "paid_social_conversion_rate": 0.03,
            "paid_social_events_multiplier": 0.4,
            "paid_social_checkout_start_prob": 0.075,
            "paid_social_cut_start": "2026-09-15T00:00:00+00:00",
            "paid_social_cut_fraction": 0.75,
        },
        # more purchase intent per session keeps order counts large enough for a clear signal
        config_overrides={"baseline_checkout_start_prob": 0.30, "baseline_order_conversion_rate": 0.20, "existing_user_share": 0.75},
        alert=_alert("orders_per_100_sessions", ("2026-09-01", "2026-09-15"), ("2026-09-15", "2026-10-01")),
        expected_answer={"primary_cause": "denominator_effect", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


def get_third_party_outage_scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id="third_party_outage",
        name="Payment Processor Outage",
        category="third_party_outage",
        description=(
            "On 2026-09-20 between 10:00 and 22:00 UTC the external payment processor was unavailable: ~90% of payment "
            "attempts failed with payment_processor_unavailable, on every app version, device and browser."
        ),
        ground_truth_cause="External payment processor outage 2026-09-20 10:00-22:00 UTC.",
        start_date=_START,
        end_date=_END,
        affected_components=["payment_processor"],
        affected_dimensions={"effective_from": "2026-09-20T10:00:00Z", "effective_until": "2026-09-20T22:00:00Z", "scope": "all versions"},
        expected_observable_effects={"orders": "Failed orders with payment_processor_unavailable concentrated in a 12-hour window across all versions."},
        parameters={
            "order_failure_effects": [{
                "start": "2026-09-20T10:00:00+00:00", "end": "2026-09-20T22:00:00+00:00",
                "prob": 0.9, "reason": "payment_processor_unavailable",
            }],
            "effect_size": 0.9,
        },
        config_overrides={"baseline_checkout_start_prob": 0.30, "baseline_order_conversion_rate": 0.20, "existing_user_share": 0.75},
        alert=_alert("order_success_rate", ("2026-09-19", "2026-09-20"), ("2026-09-20", "2026-09-21")),
        expected_answer={"primary_cause": "third_party_outage", "secondary_causes": [], "decoys": [], "insufficient_evidence": False},
    )


SCENARIO_REGISTRY = {
    "baseline": get_baseline_scenario,
    "product_regression": get_product_regression_scenario,
    "acquisition_shift": get_acquisition_shift_scenario,
    "tracking_failure": get_tracking_failure_scenario,
    "noise_only": get_noise_only_scenario,
    "seasonal_dip": get_seasonal_dip_scenario,
    "pricing_change": get_pricing_change_scenario,
    "decoy_release": get_decoy_release_scenario,
    "overlapping_causes": get_overlapping_causes_scenario,
    "gradual_decay": get_gradual_decay_scenario,
    "denominator_effect": get_denominator_effect_scenario,
    "third_party_outage": get_third_party_outage_scenario,
}

ORIGINAL_SCENARIOS = ("baseline", "product_regression", "acquisition_shift", "tracking_failure")


def build_config(scenario: "ScenarioDefinition", seed: int = 42, num_users: Optional[int] = None):
    """GeneratorConfig for a scenario: defaults + the scenario's world overrides (+ explicit num_users)."""
    from dataclasses import replace
    from investigator_data.config import GeneratorConfig
    cfg = GeneratorConfig(seed=seed).with_overrides(scenario.config_overrides)
    return replace(cfg, num_users=num_users) if num_users is not None else cfg
