"""Controlled catalogue of analytics capabilities exposed to the LLM investigator.

Enforces a strict boundary: the LLM may select only from predefined, validated
analytical operations with controlled parameter schemas. Arbitrary SQL, Python,
or system execution is prohibited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class OperationSpec:
    """Specification of an allowed analytical operation."""
    name: str
    description: str
    allowed_params: Dict[str, type]
    required_params: List[str]
    allowed_dimensions: List[str]
    example: Dict[str, Any]
    notes: str = ""
    # Tables the operation reads and the analytical capabilities it provides. Hypothesis
    # testability (investigation.testability) is derived from these.
    requires_tables: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    # Operations that compare two periods always receive the baseline window.
    needs_comparison_window: bool = False
    # Extra per-operation parameter checks: params -> error message or None.
    extra_validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None


def _int_in(params: Dict[str, Any], key: str, lo: int, hi: int) -> Optional[str]:
    if key in params:
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            return f"Parameter '{key}' must be an integer in [{lo}, {hi}]."
    return None


def _validate_statistical_test(p: Dict[str, Any]) -> Optional[str]:
    t = p.get("test_type", "proportions")
    if t not in ("proportions", "means"):
        return "test_type must be 'proportions' or 'means'."
    if t == "proportions":
        for k in ("k1", "n1", "k2", "n2"):
            if k not in p:
                return f"test_type='proportions' requires '{k}'."
        for k in ("k1", "n1", "k2", "n2"):
            if isinstance(p[k], bool) or not isinstance(p[k], int) or p[k] < 0:
                return f"'{k}' must be a non-negative integer."
        if p["n1"] == 0 or p["n2"] == 0:
            return "n1 and n2 must be positive."
        if p["k1"] > p["n1"] or p["k2"] > p["n2"]:
            return "k cannot exceed n."
    else:
        for k in ("values1", "values2"):
            v = p.get(k)
            if not isinstance(v, list) or not 2 <= len(v) <= 10000 or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                return f"'{k}' must be a list of 2-10000 numbers."
    return None


def _validate_seasonal(p: Dict[str, Any]) -> Optional[str]:
    return _int_in(p, "baseline_weeks", 1, 52) or _int_in(p, "min_weeks", 1, 52)


def _validate_retention(p: Dict[str, Any]) -> Optional[str]:
    days = p.get("retention_days")
    if days is not None and (not days or len(days) > 8 or not all(isinstance(d, int) and not isinstance(d, bool) and 1 <= d <= 365 for d in days)):
        return "retention_days must be 1-8 integers in [1, 365]."
    ev = p.get("qualifying_events")
    if ev is not None and (len(ev) > 20 or not all(isinstance(e, str) for e in ev)):
        return "qualifying_events must be up to 20 strings."
    if p.get("cohort_period") not in (None, "day", "week", "month"):
        return "cohort_period must be 'day', 'week' or 'month'."
    return None


# Bounded catalogue of operations exposed to the investigation controller
CONTROLLED_OPERATIONS: Dict[str, OperationSpec] = {
    "tracking_gap_check": OperationSpec(
        name="tracking_gap_check",
        description="Compare backend financial completed orders against client telemetry order_completed events to detect dropped tracking or instrumentation failure.",
        allowed_params={"breakdown": list, "grace_period_seconds": int, "order_status": str, "completed_event": str},
        required_params=[],
        allowed_dimensions=["browser", "device_type", "app_version"],
        example={"breakdown": ["browser"], "grace_period_seconds": 60},
        notes="Crucial for diagnosing instrumentation drops vs. real transaction declines.",
        requires_tables=['events', 'orders', 'users'],
        provides=['telemetry_vs_backend_reconciliation'],
    ),
    "funnel": OperationSpec(
        name="funnel",
        description="Chronological forward-ordered step conversion funnel (e.g. checkout_started -> order_completed).",
        allowed_params={"funnel_steps": list, "breakdown": list},
        required_params=["funnel_steps"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel"],
        example={"funnel_steps": ["checkout_started", "order_completed"], "breakdown": ["device_type"]},
        requires_tables=['events', 'users'],
        provides=['funnel_conversion'],
    ),
    "funnel_comparison": OperationSpec(
        name="funnel_comparison",
        description="Period-over-period comparison of funnel drop-offs between baseline and comparison periods.",
        allowed_params={"funnel_steps": list, "breakdown": list},
        required_params=["funnel_steps"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel"],
        example={"funnel_steps": ["checkout_started", "order_completed"], "breakdown": ["browser"]},
        requires_tables=['events', 'users'],
        provides=['funnel_conversion'],
        needs_comparison_window=True,
    ),
    "segment": OperationSpec(
        name="segment",
        description="Multi-dimensional breakdown of counts, sums, means, or rates for orders or events.",
        allowed_params={"table": str, "metric_column": str, "aggregation": str, "breakdown": list, "metric_value": (str, int, float, bool)},
        required_params=["breakdown"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel", "country", "plan_tier"],
        example={"table": "orders", "breakdown": ["acquisition_channel"], "aggregation": "count"},
        requires_tables=['orders', 'users'],
        provides=['segment_breakdown'],
    ),
    "segment_comparison": OperationSpec(
        name="segment_comparison",
        description="Compare segmented metric values between baseline and comparison periods.",
        allowed_params={"table": str, "metric_column": str, "aggregation": str, "breakdown": list, "metric_value": (str, int, float, bool)},
        required_params=["breakdown"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel", "country", "plan_tier"],
        example={"table": "orders", "breakdown": ["device_type"], "aggregation": "rate", "metric_column": "status", "metric_value": "completed"},
        requires_tables=['orders', 'users'],
        provides=['segment_comparison'],
        needs_comparison_window=True,
    ),
    "period_comparison": OperationSpec(
        name="period_comparison",
        description="Mathematically pooled period-over-period comparison of counts, sums, means, or rates.",
        allowed_params={"table": str, "metric_column": str, "aggregation": str, "metric_value": (str, int, float, bool), "time_column": str},
        required_params=["aggregation"],
        allowed_dimensions=[],
        example={"table": "orders", "aggregation": "mean", "metric_column": "amount"},
        requires_tables=['events', 'users'],
        provides=['period_comparison'],
        needs_comparison_window=True,
    ),
    "time_series": OperationSpec(
        name="time_series",
        description="Daily or weekly chronological time series of a metric.",
        allowed_params={"table": str, "time_column": str, "metric_column": str, "aggregation": str, "freq": str, "metric_value": (str, int, float, bool)},
        required_params=["aggregation"],
        allowed_dimensions=[],
        example={"table": "events", "aggregation": "count", "freq": "D"},
        requires_tables=['events', 'users'],
        provides=['time_series'],
    ),
    "release_impact": OperationSpec(
        name="release_impact",
        description="Compare metrics in symmetric pre- and post-deployment windows for a software release.",
        allowed_params={"release_id": str, "window_days": int, "table": str, "metric_column": str, "aggregation": str, "time_column": str},
        required_params=["release_id"],
        allowed_dimensions=[],
        example={"release_id": "rel_v1_1", "window_days": 7, "table": "orders", "aggregation": "count"},
        notes="Reports non-causal temporal correlation caveats.",
        requires_tables=['releases', 'orders', 'users'],
        provides=['release_before_after'],
    ),
    "exposure_effect": OperationSpec(
        name="exposure_effect",
        description="Compare outcome conversion between control and treatment groups for a feature flag with complete follow-up.",
        allowed_params={"feature_key": str, "outcome_event": str, "post_exposure_days": int, "max_observation_date": str},
        required_params=["feature_key"],
        allowed_dimensions=[],
        example={"feature_key": "new_checkout", "outcome_event": "order_completed", "post_exposure_days": 7},
        notes="Conducts two-proportion z-test and enforces user deduplication.",
        requires_tables=['feature_exposures', 'events', 'users'],
        provides=['experiment_exposure_comparison'],
    ),
    "event_volume_check": OperationSpec(
        name="event_volume_check",
        description="Daily or weekly telemetry event counts segmented by dimension.",
        allowed_params={"event_names": list, "breakdown": list, "freq": str},
        required_params=[],
        allowed_dimensions=["browser", "device_type", "app_version"],
        example={"event_names": ["order_completed", "checkout_started"], "breakdown": ["browser"]},
        requires_tables=['events', 'users'],
        provides=['event_volume_by_segment'],
    ),
    "error_analysis": OperationSpec(
        name="error_analysis",
        description="Breakdown of transaction and order failures by failure_reason across dimensions.",
        allowed_params={"breakdown": list},
        required_params=[],
        allowed_dimensions=["device_type", "browser", "acquisition_channel"],
        example={"breakdown": ["device_type"]},
        requires_tables=['orders', 'users'],
        provides=['failure_reason_breakdown'],
    ),
    "seasonal_baseline": OperationSpec(
        name="seasonal_baseline",
        description="Compare each day of the window with the SAME WEEKDAY in prior weeks (expected value and prediction range) to separate calendar seasonality from a real shift.",
        allowed_params={"table": str, "time_column": str, "metric_column": str, "aggregation": str, "metric_value": (str, int, float, bool), "baseline_weeks": int, "min_weeks": int},
        required_params=["aggregation"],
        allowed_dimensions=[],
        example={"table": "orders", "aggregation": "count", "baseline_weeks": 4},
        notes="Needs at least min_weeks (default 2) prior weeks of data before each day; otherwise returns insufficient_data.",
        requires_tables=["events", "users"],
        provides=["seasonal_baseline"],
        extra_validator=_validate_seasonal,
    ),
    "mix_shift_decomposition": OperationSpec(
        name="mix_shift_decomposition",
        description="Split the change in a rate between baseline and comparison windows into a MIX effect (who is in the population), a within-segment RATE effect and an interaction; they sum exactly to the total change.",
        allowed_params={"table": str, "metric_column": str, "metric_value": (str, int, float, bool), "breakdown": list, "time_column": str},
        required_params=["metric_column", "metric_value", "breakdown"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel", "country", "plan_tier"],
        example={"table": "orders", "metric_column": "status", "metric_value": "completed", "breakdown": ["acquisition_channel"]},
        requires_tables=["orders", "users"],
        provides=["mix_decomposition"],
        needs_comparison_window=True,
    ),
    "retention": OperationSpec(
        name="retention",
        description="Cohort retention (share of signup cohorts active N days later) with complete-follow-up enforcement.",
        allowed_params={"retention_days": list, "qualifying_events": list, "cohort_period": str, "max_observation_date": str},
        required_params=[],
        allowed_dimensions=[],
        example={"retention_days": [1, 7], "cohort_period": "week"},
        requires_tables=["users", "events"],
        provides=["retention_curve"],
        extra_validator=_validate_retention,
    ),
    "retention_comparison": OperationSpec(
        name="retention_comparison",
        description="Compare cohort retention between baseline and comparison windows, with a two-proportion test per retention horizon.",
        allowed_params={"retention_days": list, "qualifying_events": list, "cohort_period": str, "max_observation_date": str},
        required_params=[],
        allowed_dimensions=[],
        example={"retention_days": [1, 7], "cohort_period": "week"},
        requires_tables=["users", "events"],
        provides=["cohort_retention"],
        needs_comparison_window=True,
        extra_validator=_validate_retention,
    ),
    "duplicate_event_check": OperationSpec(
        name="duplicate_event_check",
        description="Detect inflated or duplicated telemetry events per session.",
        allowed_params={"event_name": str, "max_per_session": int},
        required_params=[],
        allowed_dimensions=[],
        example={"event_name": "order_completed", "max_per_session": 3},
        requires_tables=["events"],
        provides=["duplicate_event_detection"],
        extra_validator=lambda p: _int_in(p, "max_per_session", 1, 1000),
    ),
    "metric_aggregate": OperationSpec(
        name="metric_aggregate",
        description="Compute a standard, explicitly defined product metric (e.g. order_conversion_rate, gross_revenue) with its definition.",
        allowed_params={"metric_name": str},
        required_params=["metric_name"],
        allowed_dimensions=[],
        example={"metric_name": "order_conversion_rate"},
        requires_tables=["users", "events", "orders"],
        provides=["metric_definition_lookup"],
    ),
    "statistical_test": OperationSpec(
        name="statistical_test",
        description="Standalone two-proportion z-test (conversion/failure rates) or Welch t-test (continuous metrics) on caller-supplied counts/values.",
        allowed_params={"test_type": str, "k1": int, "n1": int, "k2": int, "n2": int, "values1": list, "values2": list, "label1": str, "label2": str, "target_effect": (int, float)},
        required_params=["test_type"],
        allowed_dimensions=[],
        example={"test_type": "proportions", "k1": 30, "n1": 400, "k2": 55, "n2": 410},
        notes="Inputs are NOT read from the data, so its p-value is excluded from the multiple-testing family and the SUPPORTED rule. Use it for sanity checks of numbers already returned by other tools.",
        requires_tables=[],
        provides=["adhoc_statistical_test"],
        extra_validator=_validate_statistical_test,
    ),
    "feedback_analysis": OperationSpec(
        name="feedback_analysis",
        description="Customer feedback ratings, sentiment distribution, and category breakdowns.",
        allowed_params={"breakdown": list},
        required_params=[],
        allowed_dimensions=["device_type", "acquisition_channel"],
        example={"breakdown": ["device_type"]},
        requires_tables=['user_feedback', 'users'],
        provides=['feedback_sentiment'],
    ),
}


# Columns a filter may reference, with the value types they accept.
ALLOWED_FILTER_COLUMNS = {
    "acquisition_channel", "country", "device_type", "browser", "plan_tier",   # users
    "app_version", "event_name", "page",                                       # events
    "status", "failure_reason", "currency",                                    # orders
    "category", "rating",                                                      # feedback
    "feature_key", "variant",                                                  # exposures
}
MAX_FILTER_VALUES = 50
_SCALAR_TYPES = (str, int, float, bool)


def validate_filters(filters: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return an error message if filters are not plain column == value / column in [values] predicates.

    Only equality and membership are supported (that is all the analytics layer implements);
    operator syntax (dicts, `col__gt`, `col>`) is rejected explicitly rather than ignored.
    """
    for col, val in (filters or {}).items():
        if not isinstance(col, str) or col not in ALLOWED_FILTER_COLUMNS:
            return (f"Filter column '{col}' is not permitted. Allowed filter columns: "
                    f"{sorted(ALLOWED_FILTER_COLUMNS)}")
        if val is None or isinstance(val, _SCALAR_TYPES):
            continue
        if isinstance(val, (list, tuple, set)):
            if not val:
                return f"Filter '{col}' has an empty value list."
            if len(val) > MAX_FILTER_VALUES:
                return f"Filter '{col}' has {len(val)} values; at most {MAX_FILTER_VALUES} allowed."
            if not all(isinstance(v, _SCALAR_TYPES) for v in val):
                return f"Filter '{col}' list values must be str/int/float/bool."
            continue
        if isinstance(val, dict):
            return (f"Filter '{col}' uses operator syntax {sorted(val.keys())}; only equality "
                    "(column: value) and membership (column: [values]) filters are supported.")
        return f"Filter '{col}' has unsupported value type {type(val).__name__}."
    return None


def validate_analysis_plan(
    operation: str,
    params: Dict[str, Any],
    dimensions: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate a proposed analysis against the controlled capability catalogue.

    Checks the operation, required/allowed parameters and their types, breakdown dimensions
    (both `dimensions` and `params["breakdown"]`) and filters. An operation with no allowed
    dimensions rejects any dimension rather than skipping the check.

    Returns (is_valid, error_message).
    """
    op_clean = operation.strip().lower()
    if op_clean not in CONTROLLED_OPERATIONS:
        return False, (
            f"Operation '{operation}' is not permitted. Allowed operations are: "
            f"{list(CONTROLLED_OPERATIONS.keys())}"
        )

    spec = CONTROLLED_OPERATIONS[op_clean]
    params = params or {}

    # Check required params
    for req_key in spec.required_params:
        if req_key not in params or params[req_key] is None:
            return False, f"Operation '{op_clean}' requires parameter '{req_key}'."

    # Check allowed params
    for key, val in params.items():
        if key not in spec.allowed_params:
            return False, f"Parameter '{key}' is not recognized for operation '{op_clean}'."
        expected_type = spec.allowed_params[key]
        if not isinstance(val, expected_type):
            return False, (
                f"Parameter '{key}' has invalid type {type(val).__name__}; "
                f"expected {expected_type}."
            )

    if spec.extra_validator is not None:
        extra_err = spec.extra_validator(params)
        if extra_err:
            return False, f"Operation '{op_clean}': {extra_err}"

    # Check dimensions: every dimension (explicit or via `breakdown`) must be allowed for this op.
    requested = list(dimensions or [])
    breakdown = params.get("breakdown")
    if isinstance(breakdown, list):
        if not all(isinstance(d, str) for d in breakdown):
            return False, "Parameter 'breakdown' must be a list of dimension names."
        requested += breakdown
    for d in requested:
        if d not in spec.allowed_dimensions:
            if not spec.allowed_dimensions:
                return False, f"Operation '{op_clean}' does not support breakdown dimensions; got '{d}'."
            return False, f"Dimension '{d}' is not supported for operation '{op_clean}'."

    filt_err = validate_filters(filters)
    if filt_err:
        return False, filt_err

    return True, None


def describe_capabilities() -> str:
    """Format the controlled capability catalogue for inclusion in model prompts."""
    lines = ["Available Analytical Capabilities:"]
    for op_name, spec in CONTROLLED_OPERATIONS.items():
        lines.append(f"- **{op_name}**: {spec.description}")
        if spec.required_params:
            lines.append(f"  Required params: {spec.required_params}")
        if spec.allowed_params:
            lines.append(f"  Allowed params: {list(spec.allowed_params.keys())}")
        if spec.allowed_dimensions:
            lines.append(f"  Allowed breakdown dimensions: {spec.allowed_dimensions}")
        lines.append(f"  Example parameters: {spec.example}")
    return "\n".join(lines)
