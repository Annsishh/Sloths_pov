"""Tests for competing hypothesis generation by the InvestigationController."""

import pytest

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import TimeWindow
from controller.controller import InvestigationController
from controller.llm_client import MockLLMClient
from controller.schemas import HypothesesGenerationOutput, HypothesisProposal
from investigation import HypothesisStatus, Observation


@pytest.fixture
def sample_controller_and_state():
    loader = DataLoader.from_dataframes({})
    executor = AnalyticsExecutor(loader)

    mock_llm = MockLLMClient(
        canned_responses=[
            HypothesesGenerationOutput(
                hypotheses=[
                    HypothesisProposal(
                        title="Safari client-side tracking failure",
                        explanation="Client telemetry on Safari stopped logging order_completed events due to ITP restriction.",
                        category="tracking_failure",
                        suggested_analysis="Run tracking_gap_check segmented by browser.",
                    ),
                    HypothesisProposal(
                        title="Mobile checkout payment gateway crash",
                        explanation="A recent deployment caused the mobile checkout button to fail on submission.",
                        category="product_regression",
                        suggested_analysis="Run error_analysis segmented by device_type.",
                    ),
                ],
                reasoning="The reported conversion decline could be a data tracking loss or a genuine product breakage.",
            )
        ]
    )

    controller = InvestigationController(executor=executor, llm_client=mock_llm)
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
        description="Conversion dropped by 46.7% after release.",
    )
    state = controller.start_investigation(obs)
    return controller, state


def test_step_formulate_hypotheses_creates_competing_hypotheses(sample_controller_and_state):
    controller, state = sample_controller_and_state

    created = controller.step_formulate_hypotheses(state)
    assert len(created) == 2
    assert len(state.investigation.hypotheses) == 2

    h1 = created[0]
    h2 = created[1]

    assert h1.title == "Safari client-side tracking failure"
    assert h1.category == "tracking_failure"
    assert h1.status == HypothesisStatus.PROPOSED

    assert h2.title == "Mobile checkout payment gateway crash"
    assert h2.category == "product_regression"
    assert h2.status == HypothesisStatus.PROPOSED

    # Check history events
    added_events = [e for e in state.investigation.history if "Hypothesis Added" in e.title]
    assert len(added_events) == 2
    assert state.validate() == []
