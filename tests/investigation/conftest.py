"""Shared builders for Phase 1 investigation tests."""

import pytest

from analytics.models import TimeWindow
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


def add_support(state, hyp_id, ev_id, operation, p_value=None, effect=None, stance=EvidenceStance.SUPPORTS):
    """Record one evidence item and link it to a hypothesis."""
    state.record_evidence(
        finding=f"finding {ev_id}", source_tables=["orders"], operation=operation,
        analysis_request_id=f"req_{ev_id}", evidence_id=ev_id, p_value=p_value, estimated_effect=effect,
    )
    state.link_evidence(ev_id, hyp_id, stance, f"rationale for {ev_id}")
