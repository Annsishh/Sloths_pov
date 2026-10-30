"""Tests for Investigation creation, observation recording, and serialization."""

import json
from datetime import datetime, timezone
import pytest

from analytics.models import TimeWindow
from investigation import (
    InvestigationState,
    InvestigationStatus,
    Observation,
)


@pytest.fixture
def sample_observation():
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    return Observation(
        metric_name="order_conversion_rate",
        baseline_window=bw,
        comparison_window=cw,
        baseline_value=0.15,
        comparison_value=0.08,
        absolute_change=-0.07,
        relative_change=-0.4667,
        description="Conversion dropped significantly from 15.0% to 8.0% following Release v1.2.",
    )


def test_investigation_initialization(sample_observation):
    state = InvestigationState.create(
        title="Investigation of Conversion Rate Drop post-v1.2",
        observation=sample_observation,
        investigation_id="inv_test_01",
    )

    assert state.investigation.investigation_id == "inv_test_01"
    assert state.investigation.status == InvestigationStatus.OPEN
    assert state.investigation.title == "Investigation of Conversion Rate Drop post-v1.2"
    assert state.investigation.observation.metric_name == "order_conversion_rate"
    assert state.investigation.observation.absolute_change == -0.07
    assert len(state.investigation.history) == 1
    assert "Investigation Initiated" in state.investigation.history[0].title
    assert state.validate() == []


def test_serialization_roundtrip(sample_observation):
    state = InvestigationState.create(
        title="Serialization Test Investigation",
        observation=sample_observation,
    )
    state.add_hypothesis(
        title="Safari tracking failure",
        explanation="Client events on Safari are dropped by ITP.",
        category="tracking_failure",
        hypothesis_id="hyp_safari",
    )

    # Dictionary roundtrip
    data_dict = state.to_dict()
    reconstructed = InvestigationState.from_dict(data_dict)

    assert reconstructed.investigation.investigation_id == state.investigation.investigation_id
    assert reconstructed.investigation.title == state.investigation.title
    assert reconstructed.investigation.observation.baseline_value == 0.15
    assert "hyp_safari" in reconstructed.investigation.hypotheses
    assert reconstructed.investigation.hypotheses["hyp_safari"].title == "Safari tracking failure"
    assert reconstructed.validate() == []

    # JSON roundtrip
    json_str = state.to_json()
    parsed_json = json.loads(json_str)
    reconstructed_json = InvestigationState.from_dict(parsed_json)
    assert reconstructed_json.investigation.investigation_id == state.investigation.investigation_id
