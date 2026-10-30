"""Controlled catalogue of analytics capabilities exposed to the LLM investigator.

Enforces a strict boundary: the LLM may select only from predefined, validated
analytical operations with controlled parameter schemas. Arbitrary SQL, Python,
or system execution is prohibited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


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
    ),
    "funnel": OperationSpec(
        name="funnel",
        description="Chronological forward-ordered step conversion funnel (e.g. checkout_started -> order_completed).",
        allowed_params={"funnel_steps": list, "breakdown": list},
        required_params=["funnel_steps"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel"],
        example={"funnel_steps": ["checkout_started", "order_completed"], "breakdown": ["device_type"]},
    ),
    "funnel_comparison": OperationSpec(
        name="funnel_comparison",
        description="Period-over-period comparison of funnel drop-offs between baseline and comparison periods.",
        allowed_params={"funnel_steps": list, "breakdown": list},
        required_params=["funnel_steps"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel"],
        example={"funnel_steps": ["checkout_started", "order_completed"], "breakdown": ["browser"]},
    ),
    "segment": OperationSpec(
        name="segment",
        description="Multi-dimensional breakdown of counts, sums, means, or rates for orders or events.",
        allowed_params={"table": str, "metric_column": str, "aggregation": str, "breakdown": list, "metric_value": (str, int, float, bool)},
        required_params=["breakdown"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel", "country", "plan_tier"],
        example={"table": "orders", "breakdown": ["acquisition_channel"], "aggregation": "count"},
    ),
    "segment_comparison": OperationSpec(
        name="segment_comparison",
        description="Compare segmented metric values between baseline and comparison periods.",
        allowed_params={"table": str, "metric_column": str, "aggregation": str, "breakdown": list, "metric_value": (str, int, float, bool)},
        required_params=["breakdown"],
        allowed_dimensions=["browser", "device_type", "acquisition_channel", "country", "plan_tier"],
        example={"table": "orders", "breakdown": ["device_type"], "aggregation": "rate", "metric_column": "status", "metric_value": "completed"},
    ),
    "period_comparison": OperationSpec(
        name="period_comparison",
        description="Mathematically pooled period-over-period comparison of counts, sums, means, or rates.",
        allowed_params={"table": str, "metric_column": str, "aggregation": str, "metric_value": (str, int, float, bool), "time_column": str},
        required_params=["aggregation"],
        allowed_dimensions=[],
        example={"table": "orders", "aggregation": "mean", "metric_column": "amount"},
    ),
    "time_series": OperationSpec(
        name="time_series",
        description="Daily or weekly chronological time series of a metric.",
        allowed_params={"table": str, "time_column": str, "metric_column": str, "aggregation": str, "freq": str, "metric_value": (str, int, float, bool)},
        required_params=["aggregation"],
        allowed_dimensions=[],
        example={"table": "events", "aggregation": "count", "freq": "D"},
    ),
    "release_impact": OperationSpec(
        name="release_impact",
        description="Compare metrics in symmetric pre- and post-deployment windows for a software release.",
        allowed_params={"release_id": str, "window_days": int, "table": str, "metric_column": str, "aggregation": str, "time_column": str},
        required_params=["release_id"],
        allowed_dimensions=[],
        example={"release_id": "rel_v1_1", "window_days": 7, "table": "orders", "aggregation": "count"},
        notes="Reports non-causal temporal correlation caveats.",
    ),
    "exposure_effect": OperationSpec(
        name="exposure_effect",
        description="Compare outcome conversion between control and treatment groups for a feature flag with complete follow-up.",
        allowed_params={"feature_key": str, "outcome_event": str, "post_exposure_days": int, "max_observation_date": str},
        required_params=["feature_key"],
        allowed_dimensions=[],
        example={"feature_key": "new_checkout", "outcome_event": "order_completed", "post_exposure_days": 7},
        notes="Conducts two-proportion z-test and enforces user deduplication.",
    ),
    "event_volume_check": OperationSpec(
        name="event_volume_check",
        description="Daily or weekly telemetry event counts segmented by dimension.",
        allowed_params={"event_names": list, "breakdown": list, "freq": str},
        required_params=[],
        allowed_dimensions=["browser", "device_type", "app_version"],
        example={"event_names": ["order_completed", "checkout_started"], "breakdown": ["browser"]},
    ),
    "error_analysis": OperationSpec(
        name="error_analysis",
        description="Breakdown of transaction and order failures by failure_reason across dimensions.",
        allowed_params={"breakdown": list},
        required_params=[],
        allowed_dimensions=["device_type", "browser", "acquisition_channel"],
        example={"breakdown": ["device_type"]},
    ),
    "feedback_analysis": OperationSpec(
        name="feedback_analysis",
        description="Customer feedback ratings, sentiment distribution, and category breakdowns.",
        allowed_params={"breakdown": list},
        required_params=[],
        allowed_dimensions=["device_type", "acquisition_channel"],
        example={"breakdown": ["device_type"]},
    ),
}


def validate_analysis_plan(
    operation: str,
    params: Dict[str, Any],
    dimensions: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate a proposed analysis against the controlled capability catalogue.
    
    Returns (is_valid, error_message).
    """
    op_clean = operation.strip().lower()
    if op_clean not in CONTROLLED_OPERATIONS:
        return False, (
            f"Operation '{operation}' is not permitted. Allowed operations are: "
            f"{list(CONTROLLED_OPERATIONS.keys())}"
        )

    spec = CONTROLLED_OPERATIONS[op_clean]

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

    # Check dimensions
    dims = dimensions or []
    if spec.allowed_dimensions:
        for d in dims:
            if d not in spec.allowed_dimensions:
                return False, f"Dimension '{d}' is not supported for operation '{op_clean}'."

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
