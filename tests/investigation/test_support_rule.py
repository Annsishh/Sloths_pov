"""The deterministic SUPPORTED rule (not an LLM judgement)."""

import pytest

from investigation import (
    EvidenceStance,
    HypothesisStatus,
    SupportPolicy,
    SupportRuleViolation,
)

from conftest import add_support


def _hyp(state, hid="h", category="tracking_failure"):
    return state.add_hypothesis(f"Hypothesis {hid}", "Some plausible mechanism.", category, hid)


def test_eligible_with_two_independent_analyses_one_significant(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.001)
    add_support(s, "h", "e2", "event_volume_check")
    assert s.evaluate_support("h").eligible
    s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "rule met", "e1")
    assert s.investigation.hypotheses["h"].status == HypothesisStatus.SUPPORTED


def test_single_evidence_is_not_enough(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.001)
    d = s.evaluate_support("h")
    assert not d.eligible and d.needs_more_evidence
    with pytest.raises(SupportRuleViolation, match="need 2"):
        s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "LLM says so", "e1")
    assert s.investigation.hypotheses["h"].status == HypothesisStatus.PROPOSED


def test_two_items_from_the_same_analysis_type_do_not_count_as_independent(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "time_series", p_value=0.001)
    add_support(s, "h", "e2", "time_series", p_value=0.002)
    d = s.evaluate_support("h")
    assert not d.eligible
    assert any("distinct analysis" in r for r in d.reasons)


def test_any_contradicting_evidence_blocks_support(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.001)
    add_support(s, "h", "e2", "event_volume_check")
    add_support(s, "h", "e3", "funnel", stance=EvidenceStance.CONTRADICTS)
    d = s.evaluate_support("h")
    assert not d.eligible and not d.needs_more_evidence  # more evidence cannot fix a contradiction
    assert any("contradicting" in r for r in d.reasons)


def test_requires_significance_after_correction(make_state):
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.20)
    add_support(s, "h", "e2", "event_volume_check")
    d = s.evaluate_support("h")
    assert not d.eligible
    assert any("significant" in r for r in d.reasons)


def test_bh_correction_across_all_tests_in_the_investigation(make_state):
    """p=0.04 is significant alone, but not once three other tests were run (q = 0.04*4/1 = 0.16)."""
    s = make_state()
    _hyp(s)
    add_support(s, "h", "e1", "tracking_gap_check", p_value=0.04)
    add_support(s, "h", "e2", "event_volume_check")
    assert s.evaluate_support("h").eligible
    s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "rule met", "e1")

    _hyp(s, "other", "product_regression")
    for i, p in enumerate([0.5, 0.6, 0.7]):
        add_support(s, "other", f"x{i}", "segment_comparison", p_value=p, stance=EvidenceStance.INCONCLUSIVE)
    assert s.tests_run == 4
    assert s.adjusted_p_values()["e1"] == pytest.approx(0.16)
    assert not s.evaluate_support("h").eligible
    # ...so support is withdrawn deterministically
    assert s.reassess_supported() == ["h"]
    assert s.investigation.hypotheses["h"].status == HypothesisStatus.ACTIVE


def test_thresholds_are_configurable(make_state):
    lenient = make_state(SupportPolicy(min_supporting_evidence=1, min_distinct_analyses=1, min_significant_supporting=0))
    _hyp(lenient)
    add_support(lenient, "h", "e1", "time_series")
    assert lenient.evaluate_support("h").eligible

    strict = make_state(SupportPolicy(min_supporting_evidence=3, min_distinct_analyses=3))
    _hyp(strict)
    add_support(strict, "h", "e1", "tracking_gap_check", p_value=0.001)
    add_support(strict, "h", "e2", "event_volume_check")
    assert not strict.evaluate_support("h").eligible
    add_support(strict, "h", "e3", "funnel")
    assert strict.evaluate_support("h").eligible

    uncorrected = make_state(SupportPolicy(correction="none"))
    _hyp(uncorrected)
    add_support(uncorrected, "h", "e1", "tracking_gap_check", p_value=0.04)
    add_support(uncorrected, "h", "e2", "event_volume_check")
    _hyp(uncorrected, "o", "product_regression")
    for i, p in enumerate([0.5, 0.6, 0.7]):
        add_support(uncorrected, "o", f"x{i}", "segment_comparison", p_value=p, stance=EvidenceStance.INCONCLUSIVE)
    assert uncorrected.evaluate_support("h").eligible


def test_policy_validation_and_serialization_roundtrip():
    with pytest.raises(ValueError):
        SupportPolicy(correction="holm")
    with pytest.raises(ValueError):
        SupportPolicy(alpha=1.5)
    p = SupportPolicy(alpha=0.01, min_supporting_evidence=3)
    assert SupportPolicy.from_dict(p.to_dict()) == p


def test_policy_is_stored_in_the_investigation(make_state):
    s = make_state(SupportPolicy(alpha=0.01))
    assert s.investigation.support_policy["alpha"] == 0.01
    assert s.policy.alpha == 0.01


@pytest.mark.parametrize("category", ["unspecified", "external_unspecified", "other", "external_factor"])
def test_placeholder_categories_can_never_be_supported(make_state, category):
    s = make_state()
    _hyp(s, category=category)
    add_support(s, "h", "e1", "error_analysis", p_value=0.0001)
    add_support(s, "h", "e2", "time_series", p_value=0.0001)
    d = s.evaluate_support("h")
    assert not d.eligible and not d.needs_more_evidence
    assert any("placeholder" in r for r in d.reasons)
    with pytest.raises(SupportRuleViolation):
        s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "claim", "e1")


def test_refining_a_placeholder_unlocks_support(make_state):
    s = make_state()
    _hyp(s, category="external_factor")
    add_support(s, "h", "e1", "error_analysis", p_value=0.0001)
    add_support(s, "h", "e2", "time_series")
    assert not s.evaluate_support("h").eligible
    s.refine_category("h", "third_party_outage", "failures cluster in one window")
    assert s.evaluate_support("h").eligible


def test_refinement_must_name_a_specific_cause(make_state):
    s = make_state()
    _hyp(s, category="other")
    for bad in ("unspecified", "external_unspecified", "multiple_causes"):
        with pytest.raises(ValueError, match="not a specific"):
            s.refine_category("h", bad, "r")
    with pytest.raises(ValueError, match="Unknown"):
        s.refine_category("h", "gremlins", "r")
    with pytest.raises(ValueError, match="reason"):
        s.refine_category("h", "code_regression", " ")


def test_multiple_causes_is_composite_and_never_directly_supported(make_state):
    s = make_state()
    _hyp(s, category="multiple_causes")
    add_support(s, "h", "e1", "error_analysis", p_value=0.0001)
    add_support(s, "h", "e2", "time_series")
    assert not s.evaluate_support("h").eligible


def test_untestable_category_cannot_be_supported_even_with_evidence_attached(make_state):
    s = make_state()
    _hyp(s, category="pricing_change")
    add_support(s, "h", "e1", "segment_comparison", p_value=0.0001)
    add_support(s, "h", "e2", "funnel_comparison")
    d = s.evaluate_support("h")
    assert not d.eligible and any("UNTESTABLE" in r for r in d.reasons)


def test_natural_variation_needs_non_significant_evidence(make_state):
    s = make_state()
    _hyp(s, "nv", "natural_variation")
    add_support(s, "nv", "e1", "time_series", p_value=0.60)
    add_support(s, "nv", "e2", "period_comparison")
    assert s.evaluate_support("nv").eligible

    sig = make_state()
    _hyp(sig, "nv", "natural_variation")
    add_support(sig, "nv", "e1", "time_series", p_value=0.001)
    add_support(sig, "nv", "e2", "period_comparison")
    d = sig.evaluate_support("nv")
    assert not d.eligible and any("contradicts natural variation" in r for r in d.reasons)


def test_natural_variation_excludes_and_is_excluded_by_real_causes(make_state):
    s = make_state()
    _hyp(s, "cause", "tracking_failure")
    add_support(s, "cause", "c1", "tracking_gap_check", p_value=0.0001)
    add_support(s, "cause", "c2", "event_volume_check")
    s.update_hypothesis_status("cause", HypothesisStatus.SUPPORTED, "rule met", "c1")

    _hyp(s, "nv", "natural_variation")
    add_support(s, "nv", "n1", "time_series", p_value=0.8)
    add_support(s, "nv", "n2", "period_comparison")
    d = s.evaluate_support("nv")
    assert not d.eligible and any("alongside supported cause" in r for r in d.reasons)
