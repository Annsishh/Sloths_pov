"""Tests for unresolved questions, data gap tracking, and investigative direction pivots."""

import pytest

from analytics.models import TimeWindow
from investigation import (
    HistoryEventType,
    InvestigationState,
    Observation,
    QuestionStatus,
)


@pytest.fixture
def investigation_state():
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    obs = Observation(
        metric_name="daily_active_users",
        baseline_window=bw,
        comparison_window=cw,
        baseline_value=1200.0,
        comparison_value=950.0,
        absolute_change=-250.0,
        relative_change=-0.2083,
        description="DAU dropped by 20.8%.",
    )
    state = InvestigationState.create(title="DAU Drop Investigation", observation=obs)
    state.add_hypothesis(
        title="App crash surge on iOS v2.1",
        explanation="Release v2.1 caused a crash on launch.",
        category="product_regression",
        hypothesis_id="hyp_crash",
    )
    return state


def test_unresolved_question_lifecycle(investigation_state):
    q = investigation_state.add_unresolved_question(
        question_text="Did the mobile crash reporting service register an error spike on iOS v2.1?",
        hypothesis_id="hyp_crash",
        context_or_limitation="Mobile crash telemetry is managed in an external vendor dashboard, not in primary database.",
        suggested_action="Request user inspect Crashlytics console for v2.1 stack traces.",
        question_id="q_crashlytics",
    )

    assert q.question_id == "q_crashlytics"
    assert q.status == QuestionStatus.OPEN
    assert q.resolution is None
    assert investigation_state.validate() == []

    # Resolve question
    resolved_q = investigation_state.answer_question(
        question_id="q_crashlytics",
        resolution="User checked Crashlytics: 0 crash reports logged for iOS v2.1 during the target window.",
    )

    assert resolved_q.status == QuestionStatus.ANSWERED
    assert resolved_q.resolved_at is not None
    assert "0 crash reports" in resolved_q.resolution

    # Verify history event logged
    answered_events = [
        e for e in investigation_state.investigation.history
        if e.event_type == HistoryEventType.QUESTION_ANSWERED
    ]
    assert len(answered_events) == 1
    assert investigation_state.validate() == []


def test_investigative_direction_pivot(investigation_state):
    event = investigation_state.record_direction_change(
        from_focus="iOS client crash analysis",
        to_focus="Marketing acquisition channel traffic breakdown",
        rationale=(
            "Crash reports showed 0 errors on iOS v2.1, eliminating the app crash hypothesis. "
            "Traffic telemetry indicates a 30% drop in organic search visits, requiring acquisition shift analysis."
        ),
    )

    assert event.event_type == HistoryEventType.DIRECTION_CHANGED
    assert "Investigative Pivot" in event.title
    assert "0 errors on iOS v2.1" in event.description
    assert event.metadata["from_focus"] == "iOS client crash analysis"
    assert event.metadata["to_focus"] == "Marketing acquisition channel traffic breakdown"
    assert investigation_state.validate() == []
