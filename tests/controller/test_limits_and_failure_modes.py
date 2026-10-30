"""Tests for controller limits, failure modes, malformed outputs, and insufficient evidence."""

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
from investigation import (
    ConfidenceLevel,
    InvestigationStatus,
    Observation,
)


@pytest.fixture
def mock_controller_setup():
    loader = DataLoader.from_dataframes({})
    executor = AnalyticsExecutor(loader)
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    obs = Observation(
        metric_name="daily_active_users",
        baseline_window=bw,
        comparison_window=cw,
        baseline_value=1000.0,
        comparison_value=800.0,
        absolute_change=-200.0,
        relative_change=-0.20,
        description="DAU dropped 20%.",
    )
    return executor, obs


def test_max_iterations_stops_with_insufficient_evidence(mock_controller_setup):
    executor, obs = mock_controller_setup

    # Continuous loop that never concludes
    canned = [
        HypothesesGenerationOutput(
            hypotheses=[
                HypothesisProposal(title="Hypothesis 1", explanation="Explanation 1 of 15 chars", category="product_regression"),
                HypothesisProposal(title="Hypothesis 2", explanation="Explanation 2 of 15 chars", category="user_behavior"),
            ],
            reasoning="Testing maximum iteration boundary limit.",
        ),
        # Iteration 1
        AnalysisDecision(hypothesis_id_to_test="hyp_1", operation="error_analysis", rationale="Testing iter 1"),
        EvidenceInterpretation(
            hypothesis_id="hyp_1",
            stance="inconclusive",
            finding_summary="Tool returned insufficient rows to determine significance.",
            rationale="Still inconclusive.",
            is_investigation_complete=False,
        ),
        # Iteration 2
        AnalysisDecision(hypothesis_id_to_test="hyp_2", operation="error_analysis", rationale="Testing iter 2"),
        EvidenceInterpretation(
            hypothesis_id="hyp_2",
            stance="inconclusive",
            finding_summary="Tool returned inconclusive results on second iteration.",
            rationale="Still inconclusive.",
            is_investigation_complete=False,
        ),
    ]

    mock_llm = MockLLMClient(canned_responses=canned)
    controller = InvestigationController(executor=executor, llm_client=mock_llm, max_iterations=2)

    state = controller.start_investigation(obs)
    final_state = controller.run_investigation_loop(state)

    # Must be STOPPED_INSUFFICIENT_EVIDENCE, not CONCLUDED or OPEN
    assert final_state.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert "Maximum investigation iterations reached" in final_state.investigation.stopping_reason
    assert final_state.investigation.confidence.level == ConfidenceLevel.LOW
    assert final_state.validate() == []


def test_max_analyses_budget_enforced(mock_controller_setup):
    executor, obs = mock_controller_setup

    canned = [
        HypothesesGenerationOutput(
            hypotheses=[
                HypothesisProposal(title="Hypothesis A", explanation="Explanation A of 15 chars", category="product_regression"),
                HypothesisProposal(title="Hypothesis B", explanation="Explanation B of 15 chars", category="user_behavior"),
            ],
            reasoning="Testing analysis budget.",
        ),
        # Iteration 1
        AnalysisDecision(hypothesis_id_to_test="hyp_1", operation="error_analysis", rationale="Run analysis 1"),
        EvidenceInterpretation(
            hypothesis_id="hyp_1",
            stance="inconclusive",
            finding_summary="First analysis completed but inconclusive.",
            rationale="Need more data.",
            is_investigation_complete=False,
        ),
    ]

    # Limit to max_analyses = 1
    mock_llm = MockLLMClient(canned_responses=canned)
    controller = InvestigationController(executor=executor, llm_client=mock_llm, max_iterations=5, max_analyses=1)

    state = controller.start_investigation(obs)
    final_state = controller.run_investigation_loop(state)

    assert final_state.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert "Analysis execution budget limit reached" in final_state.investigation.stopping_reason
    assert controller.analyses_executed_count == 1


def test_invalid_tool_call_rejected_cleanly(mock_controller_setup):
    executor, obs = mock_controller_setup

    canned = [
        HypothesesGenerationOutput(
            hypotheses=[
                HypothesisProposal(title="Hypothesis A", explanation="Explanation A of 15 chars", category="product_regression"),
                HypothesisProposal(title="Hypothesis B", explanation="Explanation B of 15 chars", category="user_behavior"),
            ],
            reasoning="Testing invalid tool call.",
        ),
        # Propose unpermitted operation
        AnalysisDecision(
            hypothesis_id_to_test="hyp_1",
            operation="unpermitted_arbitrary_query",
            rationale="Attempting to call unpermitted operation.",
        ),
    ]

    mock_llm = MockLLMClient(canned_responses=canned)
    controller = InvestigationController(executor=executor, llm_client=mock_llm, max_iterations=1)

    state = controller.start_investigation(obs)
    final_state = controller.run_investigation_loop(state)

    # Controller should reject unpermitted operation and halt cleanly
    assert final_state.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert any("Analysis Plan Rejected" in e.title for e in final_state.investigation.history)
    assert final_state.validate() == []


def test_insufficient_evidence_decision_by_model(mock_controller_setup):
    executor, obs = mock_controller_setup

    canned = [
        HypothesesGenerationOutput(
            hypotheses=[
                HypothesisProposal(title="Hypothesis A", explanation="Explanation A of 15 chars", category="external_factor"),
                HypothesisProposal(title="Hypothesis B", explanation="Explanation B of 15 chars", category="user_behavior"),
            ],
            reasoning="Testing insufficient evidence decision.",
        ),
        AnalysisDecision(hypothesis_id_to_test="hyp_1", operation="error_analysis", rationale="Checking logs"),
        EvidenceInterpretation(
            hypothesis_id="hyp_1",
            stance="inconclusive",
            finding_summary="Logs are missing for external partner API calls.",
            rationale="Without partner API logs, this hypothesis cannot be validated or refuted.",
            unresolved_questions=["Missing partner webhook logs."],
            is_investigation_complete=True,
            stopping_reason="Partner API logs are uninstrumented; investigation cannot proceed.",
            confidence_level="low",
            confidence_justification="Core dependency lacks observability.",
        ),
    ]

    mock_llm = MockLLMClient(canned_responses=canned)
    controller = InvestigationController(executor=executor, llm_client=mock_llm, max_iterations=2)

    state = controller.start_investigation(obs)
    final_state = controller.run_investigation_loop(state)

    assert final_state.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert "Partner API logs are uninstrumented" in final_state.investigation.stopping_reason
    assert final_state.investigation.confidence.level == ConfidenceLevel.LOW
    assert len(final_state.investigation.unresolved_questions) == 1
    assert final_state.validate() == []
