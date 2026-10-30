"""Shared builders for Phase 1 investigation tests."""

import pytest

from analytics.models import AnalysisResult, AnalysisStatus, TimeWindow
from investigation import EvidenceStance, InvestigationState, Observation, SupportPolicy


@pytest.fixture
def observation():
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    return Observation("order_conversion_rate", bw, cw, 0.15, 0.08, -0.07, -0.4667, "Conversion drop.")


@pytest.fixture
def make_state(observation):
    def _make(policy=None, testability=None):
        return InvestigationState.create(
            "Phase 1 test", observation, investigation_id="inv_p1",
            support_policy=policy or SupportPolicy(), testability=testability,
        )
    return _make


def tool_result(operation, req_id, p_value=None, p_values=None, effect=None, se=0.005, method="test_method", mde=None):
    """An AnalysisResult as a deterministic tool would return it (the only source of p-values/effects/MDE)."""
    meta = {}
    if p_value is not None:
        meta["p_value"] = p_value
    if p_values:
        meta["p_values"] = p_values
    if mde is not None:
        meta["mde"] = mde
    if effect is not None:
        meta["effect_estimate"] = {
            "value": effect, "method": method, "se": se,
            "window": {"start": "2026-09-10T00:00:00+00:00", "end": "2026-09-20T00:00:00+00:00"},
        }
    return AnalysisResult(request_id=req_id, operation=operation, status=AnalysisStatus.OK, data={}, metadata=meta)


def add_support(state, hyp_id, ev_id, operation, p_value=None, effect=None, stance=EvidenceStance.SUPPORTS,
                p_values=None, mde=None, se=0.005, method="test_method"):
    """Record one tool-produced evidence item and link it to a hypothesis."""
    result = tool_result(operation, f"req_{ev_id}", p_value, p_values, effect, se, method, mde)
    state.record_evidence(
        finding=f"finding {ev_id}", source_tables=["orders"], analysis_result=result, evidence_id=ev_id,
    )
    state.link_evidence(ev_id, hyp_id, stance, f"rationale for {ev_id}")
