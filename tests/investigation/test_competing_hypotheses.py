"""Tests for managing multiple competing hypotheses and preserving audit trails."""

import pytest

from analytics.models import TimeWindow
from investigation import (
    HypothesisStatus,
    InvestigationState,
    InvestigationStatus,
    Observation,
)


@pytest.fixture
def base_state():
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
    return InvestigationState.create(
        title="Conversion Drop Investigation",
        observation=obs,
    )


def test_add_competing_hypotheses(base_state):
    h1 = base_state.add_hypothesis(
        title="Checkout button bug on mobile",
        explanation="Release v1.2 introduced a JavaScript error preventing mobile checkout submission.",
        category="product_regression",
        hypothesis_id="hyp_mobile_bug",
    )
    h2 = base_state.add_hypothesis(
        title="Paid social acquisition mix surge",
        explanation="Surge in low-intent paid social traffic diluted overall conversion rate.",
        category="acquisition_shift",
        hypothesis_id="hyp_mix_shift",
    )
    h3 = base_state.add_hypothesis(
        title="Safari telemetry drop",
        explanation="Client-side order_completed events were dropped on Safari browsers.",
        category="tracking_failure",
        hypothesis_id="hyp_safari_drop",
    )

    assert len(base_state.investigation.hypotheses) == 3
    assert base_state.investigation.status == InvestigationStatus.IN_PROGRESS
    assert h1.status == HypothesisStatus.PROPOSED
    assert h2.status == HypothesisStatus.PROPOSED
    assert h3.status == HypothesisStatus.PROPOSED
    assert base_state.validate() == []


def test_update_hypothesis_preserves_history(base_state):
    h = base_state.add_hypothesis(
        title="Payment gateway outage",
        explanation="Gateway was rejecting transactions.",
        category="external_factor",
        hypothesis_id="hyp_gateway",
    )

    # 1. Move to ACTIVE
    base_state.update_hypothesis_status(
        hypothesis_id="hyp_gateway",
        new_status=HypothesisStatus.ACTIVE,
        reason="Prioritized for initial verification against error logs.",
    )
    assert h.status == HypothesisStatus.ACTIVE
    assert len(h.history) == 1
    assert h.history[0].previous_status == HypothesisStatus.PROPOSED
    assert h.history[0].new_status == HypothesisStatus.ACTIVE

    # 2. Refute hypothesis
    base_state.update_hypothesis_status(
        hypothesis_id="hyp_gateway",
        new_status=HypothesisStatus.REFUTED,
        reason="Error analysis showed 0 gateway timeout errors in orders table.",
    )
    assert h.status == HypothesisStatus.REFUTED
    assert len(h.history) == 2
    assert h.history[1].previous_status == HypothesisStatus.ACTIVE
    assert h.history[1].new_status == HypothesisStatus.REFUTED
    assert "0 gateway timeout errors" in h.history[1].reason
    assert base_state.validate() == []


def test_add_hypothesis_validation_failures(base_state):
    with pytest.raises(ValueError, match="title cannot be empty"):
        base_state.add_hypothesis(title="", explanation="Valid explanation", category="cat")

    with pytest.raises(ValueError, match="explanation cannot be empty"):
        base_state.add_hypothesis(title="Valid title", explanation="   ", category="cat")

    base_state.add_hypothesis(title="Title 1", explanation="Expl 1", category="cat", hypothesis_id="hyp_dup")
    with pytest.raises(ValueError, match="already exists"):
        base_state.add_hypothesis(title="Title 2", explanation="Expl 2", category="cat", hypothesis_id="hyp_dup")


def test_update_nonexistent_hypothesis(base_state):
    with pytest.raises(KeyError, match="not found"):
        base_state.update_hypothesis_status(
            hypothesis_id="hyp_unknown",
            new_status=HypothesisStatus.ACTIVE,
            reason="Testing unknown ID",
        )
