"""Tests for contradictory evidence, hypothesis refutation, and investigative revision."""

import pandas as pd
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
from investigation.support import SupportPolicy
from investigation import (
    ConfidenceLevel,
    EvidenceStance,
    HypothesisStatus,
    InvestigationStatus,
    Observation,
)


@pytest.fixture
def mock_dataset_controller():
    # Synthetic orders with no gateway errors
    users_df = pd.DataFrame([{"user_id": f"u{i}", "device_type": "mobile"} for i in range(10)])
    orders_df = pd.DataFrame([
        {"order_id": f"o{i}", "user_id": f"u{i}", "status": "failed", "failure_reason": "card_declined", "timestamp": pd.Timestamp("2026-09-12T10:00:00Z")}
        for i in range(5)
    ])
    # Phase 1: `time_series` (needs events) must be available for third_party_outage to be testable.
    events_df = pd.DataFrame([
        {"event_id": f"e{i}", "user_id": f"u{i}", "session_id": f"s{i}", "timestamp": pd.Timestamp("2026-09-12T10:00:00Z"),
         "event_name": "checkout_failed", "device_type": "mobile", "browser": "Chrome", "app_version": "v2.3.8"}
        for i in range(5)
    ])
    loader = DataLoader.from_dataframes({"users": users_df, "orders": orders_df, "events": events_df})
    executor = AnalyticsExecutor(loader)

    # Sequence of responses:
    # 1. Propose Hypotheses: Gateway Outage (hyp_1) vs. User Card Declines (hyp_2)
    # 2. Test hyp_1 with error_analysis
    # 3. Interpret error_analysis: 0 gateway timeouts -> CONTRADICTS hyp_1, refute hyp_1, continue
    # 4. Test hyp_2 with error_analysis
    # 5. Interpret error_analysis: card declines confirmed -> SUPPORTS hyp_2, conclude
    canned = [
        HypothesesGenerationOutput(
            hypotheses=[
                HypothesisProposal(
                    title="Payment gateway timeout outage",
                    explanation="Payment processor had network timeouts causing failed checkouts.",
                    category="external_factor",
                    suggested_analysis="Run error_analysis.",
                ),
                HypothesisProposal(
                    title="Card decline surge",
                    explanation="Issuing banks declined user cards.",
                    category="user_behavior",
                    suggested_analysis="Run error_analysis.",
                ),
            ],
            reasoning="Failed checkouts could be infrastructure timeouts or user credit card declines.",
        ),
        # Iteration 1: Test Gateway Outage
        AnalysisDecision(
            hypothesis_id_to_test="hyp_1",
            operation="error_analysis",
            parameters={"breakdown": ["device_type"]},
            dimensions=["device_type"],
            rationale="Check if orders failed due to gateway_timeout.",
        ),
        # Interpretation 1: Contradicts hyp_1 -> Refuted
        EvidenceInterpretation(
            hypothesis_id="hyp_1",
            stance="contradicts",
            finding_summary="Error analysis found 0 gateway_timeout errors (all 5 failed orders were card_declined).",
            rationale="No gateway timeout errors occurred, directly disproving the gateway outage hypothesis.",
            inference_type="observational_correlation",
            recommended_hypothesis_status="refuted",
            status_update_reason="Empirical evidence found 0 gateway timeouts.",
            is_investigation_complete=False,
        ),
        # Iteration 2: Pivot to Card Decline
        AnalysisDecision(
            hypothesis_id_to_test="hyp_2",
            operation="error_analysis",
            parameters={"breakdown": ["device_type"]},
            dimensions=["device_type"],
            rationale="Verify that failed orders were predominantly card_declined.",
        ),
        # Interpretation 2: Supports hyp_2, but ONE analysis is not enough for the deterministic
        # support rule (Phase 1), so the model's "complete" claim is deferred. The model also refines the
        # placeholder category external_unspecified into a specific, testable one.
        EvidenceInterpretation(
            hypothesis_id="hyp_2",
            stance="supports",
            finding_summary="100% of failed orders (5/5) were card_declined.",
            rationale="Substantiates user card decline hypothesis.",
            inference_type="observational_correlation",
            refined_category="third_party_outage",
            recommended_hypothesis_status="supported",
            status_update_reason="Confirmed card_declined was the exclusive failure reason.",
            is_investigation_complete=True,
        ),
        # Iteration 3: independent second analysis of the same hypothesis
        AnalysisDecision(
            hypothesis_id_to_test="hyp_2",
            operation="time_series",
            parameters={"table": "orders", "aggregation": "count", "freq": "D"},
            rationale="Check that failed orders cluster in time consistent with a payment-side event.",
        ),
        EvidenceInterpretation(
            hypothesis_id="hyp_2",
            stance="supports",
            finding_summary="All 5 failed orders occurred on 2026-09-12, concentrated in a single day.",
            rationale="Concentration in one window is consistent with an external card-network event.",
            inference_type="observational_correlation",
            recommended_hypothesis_status="supported",
            status_update_reason="Second independent analysis agrees.",
            is_investigation_complete=True,
            stopping_reason="The conversion drop was caused by user payment declines rather than infrastructure failure.",
            confidence_level="high",
            confidence_justification="All failed orders were verified to have card_declined failure reason.",
        ),
    ]

    mock_llm = MockLLMClient(canned_responses=canned)
    # The tiny mock dataset has no inferential test, so require 0 significant items (other thresholds stay default).
    return InvestigationController(
        executor=executor, llm_client=mock_llm, max_iterations=3,
        support_policy=SupportPolicy(min_significant_supporting=0),
    )


def test_contradiction_and_hypothesis_revision_loop(mock_dataset_controller):
    controller = mock_dataset_controller

    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    obs = Observation(
        metric_name="order_conversion_rate",
        baseline_window=bw,
        comparison_window=cw,
        baseline_value=0.20,
        comparison_value=0.10,
        absolute_change=-0.10,
        relative_change=-0.50,
        description="Conversion halved.",
    )

    state = controller.start_investigation(obs)
    final_state = controller.run_investigation_loop(state)

    # Verify hyp_1 was REFUTED
    hyp_1 = final_state.investigation.hypotheses["hyp_1"]
    assert hyp_1.status == HypothesisStatus.REFUTED
    assert len(hyp_1.contradicting_evidence_ids) == 1
    assert len(hyp_1.history) >= 1
    assert hyp_1.history[-1].new_status == HypothesisStatus.REFUTED
    assert "0 gateway timeouts" in hyp_1.history[-1].reason

    # Verify hyp_2 was SUPPORTED
    hyp_2 = final_state.investigation.hypotheses["hyp_2"]
    assert hyp_2.status == HypothesisStatus.SUPPORTED
    assert len(hyp_2.supporting_evidence_ids) == 2
    assert hyp_2.category == "third_party_outage"

    # Verify investigation concluded with hyp_2
    assert final_state.investigation.status == InvestigationStatus.CONCLUDED
    assert final_state.investigation.confidence.level == ConfidenceLevel.HIGH

    # Verify state validity
    assert final_state.validate() == []
