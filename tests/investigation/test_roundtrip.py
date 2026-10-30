"""Serialization round-trip covering every field of every investigation model."""

import dataclasses
import json
from datetime import datetime, timezone

import pytest

from analytics.models import TimeWindow
from investigation import (
    ConfidenceLevel,
    EvidenceStance,
    HypothesisStatus,
    InferenceType,
    InvestigationState,
    Observation,
    SupportPolicy,
)
from investigation import models as M

EVIDENCE_WINDOW = TimeWindow.from_iso("2026-09-10T06:30:00+00:00", "2026-09-17T18:45:00+00:00")


def _full_state() -> InvestigationState:
    """An investigation in which every optional field of every model is populated."""
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    obs = Observation("conv", bw, cw, 0.15, 0.08, -0.07, -0.4667, "Conversion drop.")
    state = InvestigationState.create(
        "Round trip", obs, investigation_id="inv_rt", support_policy=SupportPolicy(alpha=0.1, min_significant_supporting=1)
    )
    state.add_hypothesis("Safari tracking loss", "Events dropped on Safari.", "tracking_failure", "hyp_a")
    state.add_hypothesis("Mobile regression", "Release broke mobile checkout.", "product_regression", "hyp_b")
    for i, (op, p) in enumerate([("tracking_gap_check", 0.0001), ("event_volume_check", None)]):
        state.record_evidence(
            finding=f"finding {i}", source_tables=["orders", "events"], operation=op,
            analysis_request_id=f"req_{i}", population={"browser": "Safari"}, window=EVIDENCE_WINDOW,
            result_summary={"rows": 2, "metric_values": [0.1, 0.2]}, limitations=["lim"],
            inference_type=InferenceType.MEASUREMENT_ARTIFACT, evidence_id=f"ev_{i}",
            p_value=p, estimated_effect=-0.05 if i == 0 else None,
        )
        state.link_evidence(f"ev_{i}", "hyp_a", EvidenceStance.SUPPORTS, "supports it")
    state.link_evidence("ev_0", "hyp_b", EvidenceStance.CONTRADICTS, "refutes it")
    state.update_hypothesis_status("hyp_b", HypothesisStatus.REFUTED, "contradicted", "ev_0")
    state.update_hypothesis_status("hyp_a", HypothesisStatus.SUPPORTED, "rule met", "ev_0")
    q = state.add_unresolved_question("Why Safari?", "hyp_a", "ctx", "check SDK", "q_1")
    state.answer_question(q.question_id, "ITP")
    state.add_unresolved_question("Still open?", "hyp_a", "ctx2", "ask", "q_2")
    state.record_direction_change("a", "b", "because")
    state.conclude(
        "Tracking loss on Safari.", "hyp_a", ConfidenceLevel.HIGH, "two analyses",
        assumptions=["backend is truth"], remaining_uncertainties=["SDK root cause"],
    )
    return state


def test_roundtrip_dataclass_equality():
    state = _full_state()
    rebuilt = InvestigationState.from_dict(json.loads(state.to_json()))
    assert rebuilt.investigation == state.investigation  # dataclass eq compares every field recursively
    assert rebuilt.to_dict() == state.to_dict()


def test_evidence_window_survives_roundtrip_with_time_of_day():
    state = _full_state()
    rebuilt = InvestigationState.from_dict(state.to_dict())
    assert rebuilt.investigation.evidence_store["ev_0"].window == EVIDENCE_WINDOW


def test_fixture_populates_every_field_of_every_model():
    """Guards the round-trip test: no model field may be left at its empty default."""
    inv = _full_state().investigation
    instances = [
        inv, inv.observation, inv.confidence, inv.conclusion,
        *inv.hypotheses.values(), *inv.evidence_store.values(), *inv.evidence_links,
        *inv.unresolved_questions.values(), *inv.history,
        *[t for h in inv.hypotheses.values() for t in h.history],
        *inv.conclusion.supported_causes, *inv.conclusion.ruled_out,
    ]
    # fields legitimately empty for some instances of a class; require that SOME instance populates them
    seen = {}
    for obj in instances:
        for f in dataclasses.fields(obj):
            populated = getattr(obj, f.name) not in (None, "", [], {}, ())
            key = (type(obj).__name__, f.name)
            seen[key] = seen.get(key, False) or populated
    unpopulated = sorted(k for k, ok in seen.items() if not ok)
    # untestable / inconclusive lists are exercised in test_conclusions_phase1; resolution fields in questions are covered
    allowed_empty = {("Conclusion", "untestable"), ("Conclusion", "inconclusive_hypothesis_ids"),
                     ("Conclusion", "warnings"), ("Conclusion", "data_requests")}
    assert [k for k in unpopulated if k not in allowed_empty] == []


def test_legacy_string_windows_still_load():
    """Older serialised states stored windows as '[a – b)' strings."""
    d = _full_state().to_dict()
    d["observation"]["baseline_window"] = "[2026-09-01 – 2026-09-10)"
    d["evidence_store"]["ev_0"]["window"] = "[2026-09-10 – 2026-09-17)"
    rebuilt = InvestigationState.from_dict(d)
    assert rebuilt.investigation.observation.baseline_window == TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    assert rebuilt.investigation.evidence_store["ev_0"].window.start == datetime(2026, 9, 10, tzinfo=timezone.utc)
