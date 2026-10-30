"""End-to-end tests for the InvestigationController loop using real scenario data."""

from pathlib import Path
import pytest

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import TimeWindow
from controller.controller import InvestigationController
from controller.llm_client import MockLLMClient
from controller.schemas import (
    AnalysisDecision,
    EvidenceInterpretation,
    HypothesesGenerationOutput,
    HypothesisProposal,
)
from investigation import (
    ConfidenceLevel,
    EvidenceStance,
    HypothesisStatus,
    InvestigationStatus,
    Observation,
)


@pytest.fixture
def real_tracking_failure_controller():
    # Load real tracking_failure scenario investigator data
    data_dir = Path("data/tracking_failure/investigator_data")
    if not data_dir.exists():
        pytest.skip("Tracking failure dataset not generated on disk.")

    loader = DataLoader(data_dir=data_dir)
    executor = AnalyticsExecutor(loader)

    # Programmed sequential responses for MockLLMClient
    canned = [
        # Response 1: Hypothesis Formulation
        HypothesesGenerationOutput(
            hypotheses=[
                HypothesisProposal(
                    title="Safari tracking failure",
                    explanation="Client telemetry on Safari dropped order_completed events.",
                    category="tracking_failure",
                    suggested_analysis="Run tracking_gap_check segmented by browser.",
                ),
                HypothesisProposal(
                    title="Payment gateway rejection",
                    explanation="Payment processor rejected checkout attempts.",
                    category="external_factor",
                    suggested_analysis="Run error_analysis.",
                ),
            ],
            reasoning="Need to distinguish client event loss from real payment transaction failure.",
        ),
        # Response 2: Analysis Selection
        AnalysisDecision(
            hypothesis_id_to_test="hyp_1",  # Will map to first created hypothesis
            operation="tracking_gap_check",
            parameters={"breakdown": ["browser"], "grace_period_seconds": 60},
            dimensions=["browser"],
            filters={},
            comparison_window_needed=False,
            rationale="Verify whether backend completed orders match client telemetry events across browser segments.",
        ),
        # Response 3: Evidence Interpretation
        EvidenceInterpretation(
            hypothesis_id="hyp_1",
            stance="supports",
            finding_summary=(
                "Backend orders on Safari succeeded normally, but client-side telemetry captured "
                "0% of order_completed events, while Chrome maintained a ~100% tracking ratio."
            ),
            rationale="The drop in reported conversion is isolated to Safari event tracking loss.",
            inference_type="measurement_artifact",
            recommended_hypothesis_status="supported",
            status_update_reason="Confirmed zero telemetry events captured for valid backend orders on Safari.",
            unresolved_questions=["Root cause of Safari client SDK event drops (ITP vs JavaScript blocker)."],
            is_investigation_complete=True,
            stopping_reason="The conversion decline is entirely an instrumentation failure on Safari; financial revenue was unaffected.",
            confidence_level="high",
            confidence_justification="Direct cross-table comparison confirmed backend orders were intact while telemetry was dropped exclusively on Safari.",
        ),
        # Phase 1: the model declared victory after ONE analysis; the deterministic support rule
        # (>=2 independent analyses) defers completion, so the loop asks for a second, independent check.
        AnalysisDecision(
            hypothesis_id_to_test="hyp_1",
            operation="event_volume_check",
            parameters={"event_names": ["order_completed"], "breakdown": ["browser"]},
            dimensions=["browser"],
            rationale="Independently confirm that Safari order_completed telemetry volume collapsed.",
        ),
        EvidenceInterpretation(
            hypothesis_id="hyp_1",
            stance="supports",
            finding_summary="Safari order_completed event volume collapsed after the shift while other browsers were stable.",
            rationale="A second, independent view of the event stream agrees with the tracking gap check.",
            inference_type="measurement_artifact",
            recommended_hypothesis_status="supported",
            status_update_reason="Independent telemetry volume check agrees with the reconciliation.",
            is_investigation_complete=True,
            stopping_reason="The conversion decline is entirely an instrumentation failure on Safari; financial revenue was unaffected.",
            confidence_level="high",
            confidence_justification="Two independent analyses agree; reconciliation test is significant after correction.",
        ),
    ]

    mock_llm = MockLLMClient(canned_responses=canned)
    return InvestigationController(executor=executor, llm_client=mock_llm, max_iterations=4)


def test_investigation_loop_tracking_failure_scenario(real_tracking_failure_controller):
    controller = real_tracking_failure_controller

    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-15T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-15T00:00:00Z", "2026-09-30T00:00:00Z")
    obs = Observation(
        metric_name="order_conversion_rate",
        baseline_window=bw,
        comparison_window=cw,
        baseline_value=0.038,
        comparison_value=0.024,
        absolute_change=-0.014,
        relative_change=-0.368,
        description="Reported order conversion rate declined by 36.8% in second half of simulation.",
    )

    state = controller.start_investigation(obs, title="Investigating Tracking Failure Scenario")
    assert state.investigation.status == InvestigationStatus.OPEN

    # Run the investigation loop
    final_state = controller.run_investigation_loop(state)

    # Assert investigation concluded successfully
    assert final_state.investigation.status == InvestigationStatus.CONCLUDED
    assert final_state.investigation.confidence is not None
    assert final_state.investigation.confidence.level == ConfidenceLevel.HIGH

    # Assert evidence was recorded and linked
    assert len(final_state.investigation.evidence_store) >= 1
    ev = list(final_state.investigation.evidence_store.values())[0]
    assert ev.operation == "tracking_gap_check"
    assert "Safari" in ev.finding

    # Assert hypothesis was updated to SUPPORTED
    supported_hyps = [
        h for h in final_state.investigation.hypotheses.values()
        if h.status == HypothesisStatus.SUPPORTED
    ]
    assert len(supported_hyps) == 1
    assert "Safari" in supported_hyps[0].title
    assert len(supported_hyps[0].supporting_evidence_ids) == 2  # two independent analyses (Phase 1 rule)

    # Assert unresolved question was recorded
    assert len(final_state.investigation.unresolved_questions) >= 1

    # Assert history recorded all key milestones
    event_types = [e.event_type.value for e in final_state.investigation.history]
    assert "investigation_triggered" in event_types
    assert "hypothesis_added" in event_types
    assert "analysis_dispatched" in event_types
    assert "evidence_recorded" in event_types
    assert "evidence_linked" in event_types
    assert "investigation_concluded" in event_types

    # Full integrity validation
    assert final_state.validate() == []
