"""Tests for concluding investigations vs. halting due to insufficient evidence."""

import pytest

from analytics.models import TimeWindow
from investigation import (
    ConfidenceLevel,
    EvidenceStance,
    HypothesisStatus,
    InferenceType,
    InvestigationState,
    InvestigationStatus,
    Observation,
)


@pytest.fixture
def investigation_with_evidence():
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
    state = InvestigationState.create(title="Conclusion Test Investigation", observation=obs)
    state.add_hypothesis(
        title="Safari tracking failure",
        explanation="Client events are dropped on Safari.",
        category="tracking_failure",
        hypothesis_id="hyp_safari",
    )
    state.add_hypothesis(
        title="Mobile checkout gateway failure",
        explanation="Mobile users could not process payment.",
        category="product_regression",
        hypothesis_id="hyp_mobile",
    )
    return state


def test_conclude_investigation_successfully(investigation_with_evidence):
    state = investigation_with_evidence

    # Record supporting evidence
    state.record_evidence(
        finding="Safari telemetry match rate was 0.0% while backend orders succeeded.",
        source_tables=["orders", "events"],
        inference_type=InferenceType.MEASUREMENT_ARTIFACT,
        evidence_id="ev_safari_match",
    )
    state.link_evidence(
        evidence_id="ev_safari_match",
        hypothesis_id="hyp_safari",
        stance=EvidenceStance.SUPPORTS,
        rationale="Matches telemetry tracking failure pattern.",
    )

    # Update hypothesis status to SUPPORTED
    state.update_hypothesis_status(
        hypothesis_id="hyp_safari",
        new_status=HypothesisStatus.SUPPORTED,
        reason="Verified against backend order records.",
        triggered_by_evidence_id="ev_safari_match",
    )

    # Conclude
    state.conclude(
        conclusion_summary=(
            "The observed 46.7% drop in reported conversion is a measurement artifact caused by "
            "dropped client-side order_completed events on Safari. Backend completed orders remained healthy."
        ),
        supported_hypothesis_id="hyp_safari",
        confidence_level=ConfidenceLevel.HIGH,
        confidence_justification=(
            "Direct comparison between backend payment records and client event streams showed "
            "100% telemetry loss specifically isolated to Safari browsers while actual completed transactions were unaffected."
        ),
        assumptions=["Backend payment records represent true financial transactions."],
        remaining_uncertainties=["Root cause of Safari client-side SDK event loss (ITP vs. script error)."],
    )

    assert state.investigation.status == InvestigationStatus.CONCLUDED
    assert state.investigation.confidence is not None
    assert state.investigation.confidence.level == ConfidenceLevel.HIGH
    assert "measurement artifact" in state.investigation.stopping_reason
    assert state.validate() == []


def test_cannot_conclude_with_unsupported_hypothesis(investigation_with_evidence):
    state = investigation_with_evidence

    # Attempting to conclude with a hypothesis still in PROPOSED status must fail
    with pytest.raises(ValueError, match="not 'supported'"):
        state.conclude(
            conclusion_summary="Premature conclusion",
            supported_hypothesis_id="hyp_safari",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_justification="No justification",
        )


def test_stop_due_to_insufficient_evidence(investigation_with_evidence):
    state = investigation_with_evidence

    # Stop because external logs are not available
    state.stop_insufficient_evidence(
        reason="Unable to determine whether marketing mix changed or ad network tracking shifted.",
        missing_information_summary="UTM campaign parameter logging was disabled in the target window.",
        confidence_justification="Cannot distinguish ad network shift from seasonal organic traffic drop without UTM data.",
        remaining_uncertainties=["Campaign-level acquisition breakdown"],
    )

    assert state.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert state.investigation.confidence.level == ConfidenceLevel.LOW
    assert "UTM campaign parameter logging was disabled" in state.investigation.stopping_reason
    assert state.validate() == []


def test_concluded_vs_stopped_are_distinct(investigation_with_evidence):
    assert InvestigationStatus.CONCLUDED != InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert InvestigationStatus.CONCLUDED.value == "concluded"
    assert InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE.value == "stopped_insufficient_evidence"
