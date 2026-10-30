"""Phase 2a bookkeeping: BH family registry, audit events, natural-variation power, tool-only effect estimates."""

import pytest
from pydantic import ValidationError

from analytics.models import AnalysisResult, AnalysisStatus
from controller.schemas import EvidenceInterpretation
from investigation import (
    EvidenceStance,
    HypothesisStatus,
    SupportPolicy,
    SupportRuleViolation,
)
from investigation.enums import HistoryEventType
from investigation.models import EffectEstimate

from conftest import add_support, tool_result
from analytics.models import TimeWindow

WIN = TimeWindow.from_iso("2026-09-10", "2026-09-20")


def _hyp(state, hid="h", category="tracking_failure"):
    return state.add_hypothesis(f"Hypothesis {hid}", "Some plausible mechanism.", category, hid)


# ------------------------------------------------------------------ BH family bookkeeping

def test_every_segment_p_value_from_one_tool_enters_the_family_raw(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_values={"seg=Safari": 0.001, "seg=Chrome": 0.2, "seg=Firefox": 0.9})
    assert s.tests_run == 3
    assert sorted(s.investigation.test_registry.values()) == [0.001, 0.2, 0.9]       # stored RAW
    adj = s.adjusted_test_p_values()
    assert adj["req_e1|seg=Safari"] == pytest.approx(0.003)                           # 0.001 * 3 / 1
    assert s.adjusted_p_values()["e1"] == pytest.approx(0.003)                        # evidence takes its best test


def test_tests_from_uninterpreted_results_still_count(make_state):
    """Dropping unfavourable results must not shrink the correction family."""
    s = make_state()
    _hyp(s)
    s.register_analysis_result(tool_result("segment_comparison", "ghost_req", p_values={"a": 0.9, "b": 0.8}))
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.03)
    assert s.tests_run == 3
    assert s.adjusted_p_values()["e1"] == pytest.approx(0.09)                         # 0.03 * 3 / 1


def test_registration_is_idempotent_when_the_same_result_is_recorded_as_evidence(make_state):
    s = make_state()
    _hyp(s)
    res = tool_result("tracking_gap_check", "r1", p_value=0.01)
    s.register_analysis_result(res)
    s.record_evidence("finding", ["orders"], analysis_result=res, evidence_id="e1")
    assert s.tests_run == 1


def test_caller_supplied_p_values_are_excluded_from_family_and_support(make_state):
    s = make_state()
    _hyp(s)
    res = tool_result("statistical_test", "r_stat", p_value=1e-9)
    res.metadata["p_value_provenance"] = "caller_supplied"
    s.record_evidence("LLM-supplied counts look significant", ["orders"], analysis_result=res, evidence_id="e_stat")
    s.link_evidence("e_stat", "h", EvidenceStance.SUPPORTS, "r")
    assert s.tests_run == 0 and s.adjusted_p_values() == {}
    add_support(s, "h", "e2", "tracking_gap_check")
    d = s.evaluate_support("h")
    assert not d.eligible and any("significant" in r for r in d.reasons)


def test_p_value_out_of_range_is_rejected(make_state):
    s = make_state()
    with pytest.raises(ValueError):
        s.register_tests("x", {"a": 1.5})


# ------------------------------------------------------------------ audit events

def test_bh_driven_support_withdrawal_is_logged_with_before_and_after(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.04)
    add_support(s, "h", "e2", "event_volume_check")
    s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "rule met", "e1")
    assert s.investigation.hypotheses["h"].support_snapshot == {"e1": pytest.approx(0.04)}

    _hyp(s, "o", "product_regression")
    for i, p in enumerate([0.5, 0.6, 0.7]):
        add_support(s, "o", f"x{i}", "segment_comparison", p_value=p, stance=EvidenceStance.INCONCLUSIVE)
    assert s.reassess_supported() == ["h"]

    ev = [e for e in s.investigation.history if e.event_type == HistoryEventType.SUPPORT_WITHDRAWN]
    assert len(ev) == 1
    md = ev[0].metadata
    assert md["hypothesis_id"] == "h"
    assert md["adjusted_p_before"] == {"e1": pytest.approx(0.04)}
    assert md["adjusted_p_after"] == {"e1": pytest.approx(0.16)}
    assert md["tests_in_family"] == 4 and md["alpha"] == 0.05
    assert any("significant" in r for r in md["reasons"]) and "no longer satisfied" in ev[0].description


def test_no_withdrawal_event_when_support_still_holds(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.001)
    add_support(s, "h", "e2", "event_volume_check")
    s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "rule met", "e1")
    assert s.reassess_supported() == []
    assert not [e for e in s.investigation.history if e.event_type == HistoryEventType.SUPPORT_WITHDRAWN]


def test_unknown_category_coercion_logs_a_warning_event(make_state):
    s = make_state()
    h = s.add_hypothesis("Weird cause", "Something unusual happened.", "gremlins", "h")
    assert h.category == "unspecified"
    w = [e for e in s.investigation.history if e.event_type == HistoryEventType.WARNING]
    assert len(w) == 1 and w[0].metadata == {
        "hypothesis_id": "h", "original_category": "gremlins", "coerced_to": "unspecified"}
    assert "gremlins" in w[0].description


@pytest.mark.parametrize("category", ["unspecified", "other", "external_factor", "product_regression", "code_regression"])
def test_known_and_legacy_categories_log_no_warning(make_state, category):
    s = make_state()
    s.add_hypothesis("Cause", "Something plausible happened.", category, "h")
    assert not [e for e in s.investigation.history if e.event_type == HistoryEventType.WARNING]


# ------------------------------------------------------------------ natural variation needs power

def _nv(state):
    state.add_hypothesis("Just noise", "The change is within normal variation.", "natural_variation", "nv")


def test_underpowered_nonsignificant_result_is_inconclusive_not_support(make_state):
    s = make_state()          # |observed change| = 0.07 -> MDE must be <= 0.07
    _nv(s)
    add_support(s, "nv", "n1", "period_comparison", p_value=0.60, mde=0.20)     # could not have seen a 0.07 change
    add_support(s, "nv", "n2", "time_series", p_value=0.70, mde=0.15)
    d = s.evaluate_support("nv")
    assert not d.eligible and d.needs_more_evidence
    assert any("underpowered" in r and "inconclusive" in r for r in d.reasons)
    with pytest.raises(SupportRuleViolation):
        s.update_hypothesis_status("nv", HypothesisStatus.SUPPORTED, "no significant change", "n1")


def test_result_without_an_mde_cannot_establish_power(make_state):
    s = make_state()
    _nv(s)
    add_support(s, "nv", "n1", "period_comparison", p_value=0.60)
    add_support(s, "nv", "n2", "time_series", p_value=0.70)
    assert not s.evaluate_support("nv").eligible


def test_adequately_powered_nonsignificant_result_supports_natural_variation(make_state):
    s = make_state()
    _nv(s)
    add_support(s, "nv", "n1", "period_comparison", p_value=0.60, mde=0.05)
    add_support(s, "nv", "n2", "time_series")
    assert s.evaluate_support("nv").eligible


def test_mde_threshold_is_configurable(make_state):
    s = make_state(SupportPolicy(nv_max_mde_ratio=2.0))      # allow MDE up to 2 x 0.07
    _nv(s)
    add_support(s, "nv", "n1", "period_comparison", p_value=0.60, mde=0.12)
    add_support(s, "nv", "n2", "time_series")
    assert s.evaluate_support("nv").eligible
    strict = make_state(SupportPolicy(nv_max_mde_ratio=0.5))
    _nv(strict)
    add_support(strict, "nv", "n1", "period_comparison", p_value=0.60, mde=0.05)
    add_support(strict, "nv", "n2", "time_series")
    assert not strict.evaluate_support("nv").eligible


# ------------------------------------------------------------------ effect estimates: tools only

def test_effect_estimate_requires_method_window_and_uncertainty():
    ok = dict(value=-0.01, method="m", window=WIN, se=0.002)
    assert EffectEstimate(**ok).uncertainty_width() == 0.002
    assert EffectEstimate(value=-0.01, method="m", window=WIN, ci_lower=-0.02, ci_upper=0.0).uncertainty_width() == pytest.approx(0.02 / 3.92)
    with pytest.raises(ValueError, match="method"):
        EffectEstimate(**{**ok, "method": " "})
    with pytest.raises(ValueError, match="uncertainty"):
        EffectEstimate(value=-0.01, method="m", window=WIN)
    with pytest.raises(ValueError, match="uncertainty"):
        EffectEstimate(value=-0.01, method="m", window=WIN, ci_lower=-0.02)
    with pytest.raises(ValueError, match="window"):
        EffectEstimate(value=-0.01, method="m", window=None, se=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        EffectEstimate(**{**ok, "se": -1})


def test_a_tool_result_with_an_incomplete_estimate_is_rejected_not_silently_accepted(make_state):
    s = make_state()
    res = AnalysisResult("r", "mix_shift_decomposition", AnalysisStatus.OK, {},
                         metadata={"effect_estimate": {"value": -0.01, "method": "m"}})
    with pytest.raises((ValueError, KeyError, TypeError)):
        s.record_evidence("f", ["orders"], analysis_result=res)


@pytest.mark.parametrize("key", ["estimated_effect", "effect_estimate", "explained_share", "explained_change", "share", "estimated_change"])
def test_llm_output_that_sets_effects_or_shares_is_rejected(key):
    base = dict(hypothesis_id="h", stance="supports", finding_summary="Finding with numbers.", rationale="Because of the numbers.")
    EvidenceInterpretation(**base)                                  # fine without
    with pytest.raises(ValidationError, match="deterministic"):
        EvidenceInterpretation(**base, **{key: -0.5})


def test_llm_json_with_an_effect_is_rejected_at_parse_time():
    raw = ('{"hypothesis_id":"h","stance":"supports","finding_summary":"Finding with numbers.",'
           '"rationale":"Because of the numbers.","estimated_effect":-0.03}')
    with pytest.raises(ValidationError):
        EvidenceInterpretation.model_validate_json(raw)
