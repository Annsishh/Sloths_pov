"""Tests for state integrity validation and strict ground-truth separation."""

import pytest

from analytics.models import TimeWindow
from investigation import (
    EvidenceStance,
    HypothesisEvidenceLink,
    InvestigationState,
    Observation,
)


@pytest.fixture
def clean_state():
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
    state = InvestigationState.create(title="Integrity Test Investigation", observation=obs)
    state.add_hypothesis(
        title="Valid hypothesis",
        explanation="Valid empirical explanation.",
        category="product_regression",
        hypothesis_id="hyp_valid",
    )
    return state


def test_validation_detects_broken_evidence_link(clean_state):
    # Manually append a broken link with a non-existent evidence ID
    broken_link = HypothesisEvidenceLink(
        evidence_id="ev_missing",
        hypothesis_id="hyp_valid",
        stance=EvidenceStance.SUPPORTS,
        rationale="Broken link test.",
    )
    clean_state.investigation.evidence_links.append(broken_link)

    errors = clean_state.validate()
    assert len(errors) > 0
    assert any("evidence 'ev_missing' does not exist" in e for e in errors)


def test_validation_detects_broken_hypothesis_link(clean_state):
    # Record valid evidence
    clean_state.record_evidence(
        finding="Sample finding",
        source_tables=["orders"],
        evidence_id="ev_valid",
    )
    # Manually append a broken link with a non-existent hypothesis ID
    broken_link = HypothesisEvidenceLink(
        evidence_id="ev_valid",
        hypothesis_id="hyp_missing",
        stance=EvidenceStance.SUPPORTS,
        rationale="Broken link test.",
    )
    clean_state.investigation.evidence_links.append(broken_link)

    errors = clean_state.validate()
    assert len(errors) > 0
    assert any("hypothesis 'hyp_missing' does not exist" in e for e in errors)


def test_ground_truth_leakage_detection(clean_state):
    # If privileged scenario ground-truth information is injected into hypothesis explanation
    clean_state.add_hypothesis(
        title="Leaked hypothesis",
        explanation="This matches the scenario_ground_truth json parameter.",
        category="leak",
        hypothesis_id="hyp_leak",
    )

    errors = clean_state.validate()
    assert len(errors) > 0
    assert any("Ground-truth leakage detected" in e for e in errors)


def test_ground_truth_leakage_in_evidence_finding(clean_state):
    # If privileged ground-truth marker is present in an evidence finding
    clean_state.record_evidence(
        finding="Verified against injected_parameters in the synthetic simulation.",
        source_tables=["orders"],
        evidence_id="ev_leak",
    )

    errors = clean_state.validate()
    assert len(errors) > 0
    assert any("Ground-truth leakage detected" in e for e in errors)
