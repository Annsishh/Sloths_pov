"""Which hypothesis categories can be tested with the available tools and tables.

Testability is DERIVED, not hard-coded per scenario: each category needs a set of
capabilities; an operation in the capability catalogue provides capabilities and needs
tables; a capability is available iff some operation providing it has all its tables.
Capabilities no operation provides (e.g. latency logs) make a category UNTESTABLE, and
the category's `data_request` says what to ask for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from investigation.enums import HypothesisCategory as C


@dataclass(frozen=True)
class CategoryRequirement:
    capabilities: Tuple[str, ...] = ()
    data_request: str = ""
    # Set for categories that can never be tested directly, whatever tools exist.
    ineligible_reason: str = ""


CATEGORY_REQUIREMENTS: Dict[C, CategoryRequirement] = {
    C.CODE_REGRESSION: CategoryRequirement(("release_before_after",), "release log with deployment timestamps"),
    C.FEATURE_OR_EXPERIMENT_EFFECT: CategoryRequirement(("experiment_exposure_comparison",), "feature-flag / experiment exposure log"),
    C.CONFIG_OR_FLAG_CHANGE: CategoryRequirement(("config_change_log",), "configuration and feature-flag change audit log"),
    C.PERFORMANCE_DEGRADATION: CategoryRequirement(("latency_metrics",), "page/API latency and error-rate logs (p50/p95)"),
    C.PRICING_OR_POLICY_CHANGE: CategoryRequirement(("pricing_change_log",), "pricing and policy change log"),
    C.TRAFFIC_MIX_SHIFT: CategoryRequirement(("mix_decomposition",), "user acquisition and segment attributes"),
    C.BOT_OR_INVALID_TRAFFIC: CategoryRequirement(("bot_traffic_flags",), "bot / invalid-traffic classification (user agent, IP reputation, traffic-source validity flags)"),
    C.TRACKING_LOSS: CategoryRequirement(("telemetry_vs_backend_reconciliation",), "client telemetry and backend order records"),
    C.PIPELINE_OR_DEFINITION_CHANGE: CategoryRequirement(("pipeline_run_log",), "ETL/pipeline run log and metric-definition change history"),
    C.SEASONALITY_OR_CALENDAR: CategoryRequirement(("seasonal_baseline",), "several prior weeks of history for a same-weekday baseline, plus a holiday/event calendar"),
    C.THIRD_PARTY_OUTAGE: CategoryRequirement(("failure_reason_breakdown", "time_series"), "order failure reasons over time"),
    C.MARKET_OR_COMPETITOR: CategoryRequirement(("external_market_data",), "external market, competitor and campaign data"),
    C.GRADUAL_OR_COHORT_EFFECT: CategoryRequirement(("cohort_retention",), "user signup cohorts with long-run activity"),
    C.DENOMINATOR_EFFECT: CategoryRequirement(("funnel_conversion", "mix_decomposition"), "funnel and segment data"),
    C.NATURAL_VARIATION: CategoryRequirement(("time_series",), "daily history of the metric"),
    C.MULTIPLE_CAUSES: CategoryRequirement(ineligible_reason=(
        "composite: expressed by a conclusion with two or more supported causes, never supported directly")),
    C.UNSPECIFIED: CategoryRequirement(ineligible_reason=(
        "placeholder category: refine it into a specific cause family before it can be tested or supported")),
    C.EXTERNAL_UNSPECIFIED: CategoryRequirement(ineligible_reason=(
        "placeholder category: refine it into a specific external cause (e.g. third_party_outage, "
        "market_or_competitor, seasonality_or_calendar) before it can be tested or supported")),
}


# Categories whose underlying cause the available tools can measure directly. Everything else is only ever
# "consistent with" the evidence: tools can show a pattern but cannot confirm the external/causal mechanism.
CONFIRMABLE_CATEGORIES = frozenset({C.TRACKING_LOSS.value, C.TRAFFIC_MIX_SHIFT.value, C.DENOMINATOR_EFFECT.value})


def attribution_strength(category: str) -> str:
    return "confirmed" if category in CONFIRMABLE_CATEGORIES else "consistent_with"


@dataclass
class TestabilityAssessment:
    __test__ = False  # not a pytest class

    category: str
    testable: bool
    reason: str
    missing_capabilities: List[str] = field(default_factory=list)
    data_request: str = ""
    provided_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category, "testable": self.testable, "reason": self.reason,
            "missing_capabilities": list(self.missing_capabilities),
            "data_request": self.data_request, "provided_by": list(self.provided_by),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestabilityAssessment":
        return cls(d["category"], d["testable"], d["reason"], list(d.get("missing_capabilities", [])),
                   d.get("data_request", ""), list(d.get("provided_by", [])))


def available_capabilities(operations: Mapping[str, Any], available_tables: Iterable[str]) -> Dict[str, List[str]]:
    """capability -> operations that provide it, restricted to ops whose tables are all available."""
    tables = set(available_tables)
    caps: Dict[str, List[str]] = {}
    for name, spec in operations.items():
        if set(getattr(spec, "requires_tables", [])) <= tables:
            for cap in getattr(spec, "provides", []):
                caps.setdefault(cap, []).append(name)
    return caps


def assess_testability(
    operations: Mapping[str, Any],
    available_tables: Iterable[str],
) -> Dict[str, TestabilityAssessment]:
    """Assess every category against the operation catalogue and the tables present."""
    caps = available_capabilities(operations, available_tables)
    out: Dict[str, TestabilityAssessment] = {}
    for cat, req in CATEGORY_REQUIREMENTS.items():
        if req.ineligible_reason:
            out[cat.value] = TestabilityAssessment(cat.value, False, req.ineligible_reason)
            continue
        missing = [c for c in req.capabilities if c not in caps]
        if missing:
            out[cat.value] = TestabilityAssessment(
                cat.value, False,
                f"no available analysis provides: {', '.join(missing)}",
                missing_capabilities=missing, data_request=req.data_request,
            )
        else:
            providers = sorted({op for c in req.capabilities for op in caps[c]})
            out[cat.value] = TestabilityAssessment(cat.value, True, "testable with available tools", provided_by=providers)
    return out
