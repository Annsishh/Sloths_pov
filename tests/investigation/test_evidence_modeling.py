"""Tests for empirical evidence recording, hypothesis linking, and stance evaluation."""

import pandas as pd
import pytest

from analytics.models import AnalysisResult, AnalysisStatus, TimeWindow
from investigation import (
    EvidenceStance,
    InferenceType,
    InvestigationState,
    Observation,
)


@pytest.fixture
def active_state():
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    obs = Observation(
        metric_name="order_conversion_rate",
        baseline_window=bw,
        comparison_window=cw,
        baseline_value=0.15,
        comparison_value=0.08,
        absolute_change=-0.07,
        relative_change=-0.4667,
        description="Conversion drop trigger.",
    )
    state = InvestigationState.create(title="Active Conversion Investigation", observation=obs)
    state.add_hypothesis(
        title="Tracking failure on Safari",
        explanation="Client events are not received on Safari.",
        category="tracking_failure",
        hypothesis_id="hyp_safari",
    )
    state.add_hypothesis(
        title="Product bug on Chrome",
        explanation="Chrome users cannot complete checkout.",
        category="product_regression",
        hypothesis_id="hyp_chrome",
    )
    return state


def test_record_evidence_from_analysis_result(active_state):
    # Simulated AnalysisResult from tracking_gap_check
    result = AnalysisResult(
        request_id="req_track_check",
        operation="tracking_gap_check",
        status=AnalysisStatus.OK,
        data=pd.DataFrame([
            {"browser": "Chrome", "tracking_ratio": 1.0, "matched_orders": 100},
            {"browser": "Safari", "tracking_ratio": 0.05, "matched_orders": 5},
        ]),
        limitations=["60s grace period used."],
        warnings=["Severe tracking failure detected in Safari."],
    )

    ev = active_state.record_evidence(
        finding="Safari telemetry match rate collapsed to 5.0% (5/100 orders), while Chrome tracking stayed at 100%.",
        source_tables=["events", "orders", "users"],
        analysis_result=result,
        population={"browser": "Safari"},
        inference_type=InferenceType.MEASUREMENT_ARTIFACT,
        evidence_id="ev_safari_gap",
    )

    assert ev.evidence_id == "ev_safari_gap"
    assert ev.operation == "tracking_gap_check"
    assert ev.inference_type == InferenceType.MEASUREMENT_ARTIFACT
    assert "60s grace period used." in ev.limitations
    assert "ev_safari_gap" in active_state.investigation.evidence_store
    assert active_state.validate() == []


def test_evidence_supports_one_and_contradicts_another(active_state):
    # Record evidence
    ev = active_state.record_evidence(
        finding="Chrome conversion rate was stable at 15.2% before and 15.0% after release; Safari telemetry dropped 95%.",
        source_tables=["orders", "events"],
        inference_type=InferenceType.MEASUREMENT_ARTIFACT,
        evidence_id="ev_browser_comp",
    )

    # 1. Link to hyp_safari as SUPPORTS
    link_safari = active_state.link_evidence(
        evidence_id="ev_browser_comp",
        hypothesis_id="hyp_safari",
        stance=EvidenceStance.SUPPORTS,
        rationale="The discrepancy is isolated to Safari, directly substantiating the Safari tracking failure hypothesis.",
    )

    # 2. Link to hyp_chrome as CONTRADICTS
    link_chrome = active_state.link_evidence(
        evidence_id="ev_browser_comp",
        hypothesis_id="hyp_chrome",
        stance=EvidenceStance.CONTRADICTS,
        rationale="Chrome conversion rate did not decline (15.2% vs 15.0%), refuting the Chrome bug hypothesis.",
    )

    hyp_safari = active_state.investigation.hypotheses["hyp_safari"]
    hyp_chrome = active_state.investigation.hypotheses["hyp_chrome"]

    assert "ev_browser_comp" in hyp_safari.supporting_evidence_ids
    assert "ev_browser_comp" not in hyp_safari.contradicting_evidence_ids

    assert "ev_browser_comp" in hyp_chrome.contradicting_evidence_ids
    assert "ev_browser_comp" not in hyp_chrome.supporting_evidence_ids

    assert len(active_state.investigation.evidence_links) == 2
    assert active_state.validate() == []


def test_link_evidence_validation(active_state):
    with pytest.raises(KeyError, match="Evidence 'ev_nonexistent' does not exist"):
        active_state.link_evidence(
            evidence_id="ev_nonexistent",
            hypothesis_id="hyp_safari",
            stance=EvidenceStance.SUPPORTS,
            rationale="Valid rationale",
        )

    with pytest.raises(KeyError, match="Hypothesis 'hyp_nonexistent' does not exist"):
        active_state.record_evidence(
            finding="Some finding",
            source_tables=["orders"],
            evidence_id="ev_valid",
        )
        active_state.link_evidence(
            evidence_id="ev_valid",
            hypothesis_id="hyp_nonexistent",
            stance=EvidenceStance.SUPPORTS,
            rationale="Valid rationale",
        )

    with pytest.raises(ValueError, match="requires an explicit rationale"):
        active_state.link_evidence(
            evidence_id="ev_valid",
            hypothesis_id="hyp_safari",
            stance=EvidenceStance.SUPPORTS,
            rationale="   ",
        )
