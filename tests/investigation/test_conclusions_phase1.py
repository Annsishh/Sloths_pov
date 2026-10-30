"""Explained-change accounting, INSUFFICIENT_EVIDENCE reports, and the untestable-promotion guardrail."""

import pytest

from investigation import (
    ConfidenceLevel,
    EvidenceStance,
    HypothesisStatus,
    InvestigationStatus,
    SupportRuleViolation,
)

from conftest import add_support


def _supported(state, hid, category, ops, effect, p=0.0001, method="test_method"):
    state.add_hypothesis(f"Cause {hid}", "A plausible mechanism for the drop.", category, hid)
    add_support(state, hid, f"{hid}_1", ops[0], p_value=p, effect=effect, method=method)
    add_support(state, hid, f"{hid}_2", ops[1])
    state.update_hypothesis_status(hid, HypothesisStatus.SUPPORTED, "rule met", f"{hid}_1")


def _conclude(state, ids, **kw):
    return state.conclude("Concluded.", ids[0], ConfidenceLevel.MEDIUM, "justified", additional_supported_ids=ids[1:], **kw)


def test_two_causes_with_hand_computed_shares_and_remainder(make_state):
    s = make_state()  # observed change = -0.07
    _supported(s, "a", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=-0.042)
    _supported(s, "b", "product_regression", ["release_impact", "time_series"], effect=-0.021)
    c = _conclude(s, ["a", "b"])

    shares = {x.hypothesis_id: x for x in c.supported_causes}
    assert shares["a"].share == pytest.approx(0.6)
    assert shares["b"].share == pytest.approx(0.3)
    assert c.explained_change == pytest.approx(-0.063)
    assert c.unexplained_change == pytest.approx(-0.007)
    assert c.unexplained_share == pytest.approx(0.1)
    assert c.warnings == []
    assert c.outcome == "concluded"
    assert s.investigation.conclusion is c
    assert shares["a"].basis_evidence_ids == ["a_1"]


def test_conclude_has_no_way_to_inject_explained_changes(make_state):
    """Shares come only from tool estimates: conclude() takes no caller-supplied numbers."""
    import inspect
    assert "explained_changes" not in inspect.signature(make_state().conclude).parameters
    assert "estimated_effect" not in inspect.signature(make_state().record_evidence).parameters


def test_unquantified_cause_is_reported_not_guessed(make_state):
    s = make_state()
    _supported(s, "a", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=None)
    c = _conclude(s, ["a"])
    cause = c.supported_causes[0]
    assert cause.estimated_change is None and cause.share is None
    assert c.explained_change == 0
    assert c.unexplained_change == pytest.approx(-0.07) and c.unexplained_share == pytest.approx(1.0)
    assert any("no quantified effect" in w for w in c.warnings)


def test_over_attribution_is_flagged(make_state):
    s = make_state()
    _supported(s, "a", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=-0.06)
    _supported(s, "b", "product_regression", ["release_impact", "time_series"], effect=-0.05)
    c = _conclude(s, ["a", "b"])
    assert c.unexplained_change == pytest.approx(0.04)
    assert c.unexplained_share == pytest.approx(-0.5714285714)
    assert any("exceed the observed change" in w for w in c.warnings)


def test_opposite_direction_cause_is_flagged(make_state):
    s = make_state()
    _supported(s, "a", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=+0.02)
    c = _conclude(s, ["a"])
    assert any("opposite" in w for w in c.warnings)


def test_most_precise_estimate_is_used_and_method_se_window_recorded(make_state):
    s = make_state()
    s.add_hypothesis("Cause", "A plausible mechanism for the drop.", "tracking_failure", "a")
    add_support(s, "a", "a_1", "tracking_gap_check", p_value=0.0001, effect=-0.02, se=0.010, method="m_coarse")
    add_support(s, "a", "a_2", "event_volume_check", effect=-0.04, se=0.002, method="m_precise")
    s.update_hypothesis_status("a", HypothesisStatus.SUPPORTED, "rule met", "a_1")
    c = _conclude(s, ["a"])
    cause = c.supported_causes[0]
    assert cause.estimated_change == pytest.approx(-0.04)
    assert (cause.method, cause.se) == ("m_precise", 0.002)
    assert cause.window is not None and cause.window.start.isoformat().startswith("2026-09-10")
    assert cause.basis_evidence_ids == ["a_2"]


def test_estimates_from_different_methods_flag_possible_double_counting(make_state):
    s = make_state()
    _supported(s, "a", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=-0.03, method="method_x")
    _supported(s, "b", "product_regression", ["release_impact", "time_series"], effect=-0.02, method="method_y")
    c = _conclude(s, ["a", "b"])
    assert any("double counting" in w for w in c.warnings)


def test_same_method_does_not_flag_double_counting(make_state):
    s = make_state()
    _supported(s, "a", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=-0.03, method="same")
    _supported(s, "b", "product_regression", ["release_impact", "time_series"], effect=-0.02, method="same")
    assert not any("double counting" in w for w in _conclude(s, ["a", "b"]).warnings)


def test_conclusion_wording_is_hedged_unless_the_tools_can_confirm(make_state):
    s = make_state()
    _supported(s, "trk", "tracking_failure", ["tracking_gap_check", "event_volume_check"], effect=-0.035)
    _supported(s, "ext", "third_party_outage", ["error_analysis", "time_series"], effect=-0.02)
    c = _conclude(s, ["trk", "ext"])
    by = {x.hypothesis_id: x for x in c.supported_causes}
    assert by["trk"].attribution == "confirmed" and by["ext"].attribution == "consistent_with"
    assert "consistent with" in by["ext"].statement and "cannot confirm" in by["ext"].statement
    assert "consistent with" not in by["trk"].statement
    assert "method test_method" in by["ext"].statement and "SE" in by["ext"].statement
    assert c.statements == [x.statement for x in c.supported_causes]
    assert c.tests_in_family == 2 and c.fdr_alpha == 0.05


def test_natural_variation_attributes_the_whole_change_to_noise(make_state):
    s = make_state()
    s.add_hypothesis("Just noise", "The change is within normal variation.", "natural_variation", "nv")
    add_support(s, "nv", "n1", "time_series", p_value=0.7, mde=0.02)
    add_support(s, "nv", "n2", "period_comparison")
    s.update_hypothesis_status("nv", HypothesisStatus.SUPPORTED, "rule met", "n1")
    c = _conclude(s, ["nv"])
    assert c.supported_causes[0].share == pytest.approx(1.0)
    assert c.unexplained_change == pytest.approx(0.0)


def test_conclude_rejects_cause_no_longer_satisfying_rule(make_state):
    s = make_state()
    s.add_hypothesis("Cause", "A plausible mechanism for the drop.", "tracking_failure", "a")
    add_support(s, "a", "a_1", "tracking_gap_check", p_value=0.04)
    add_support(s, "a", "a_2", "event_volume_check")
    s.update_hypothesis_status("a", HypothesisStatus.SUPPORTED, "rule met", "a_1")
    s.add_hypothesis("Other", "Another mechanism for the drop.", "product_regression", "o")
    for i, p in enumerate([0.5, 0.6, 0.7]):
        add_support(s, "o", f"o{i}", "segment_comparison", p_value=p, stance=EvidenceStance.INCONCLUSIVE)
    with pytest.raises(ValueError, match="no longer satisfy"):
        _conclude(s, ["a"])


# ----------------------------------------------------------------- INSUFFICIENT_EVIDENCE


def _insufficient_state(make_state):
    s = make_state()
    s.add_hypothesis("Mobile checkout bug", "A release broke mobile checkout.", "product_regression", "reg")
    s.add_hypothesis("Price increase", "A pricing change reduced conversion.", "pricing_change", "price")
    s.add_hypothesis("Unknown external", "Something outside the product.", "external_factor", "ext")
    add_support(s, "reg", "r1", "release_impact", p_value=0.9, stance=EvidenceStance.CONTRADICTS)
    s.update_hypothesis_status("reg", HypothesisStatus.REFUTED, "No change in mobile checkout after the release.", "r1")
    return s


def test_insufficient_evidence_report_lists_ruled_out_untestable_and_data_requests(make_state):
    s = _insufficient_state(make_state)
    c = s.stop_insufficient_evidence("Cannot decide.", "No pricing log available.", "Nothing explains it.")

    assert s.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert c.outcome == "insufficient_evidence"
    assert c.supported_causes == []
    assert c.unexplained_share == pytest.approx(1.0)

    assert [r.hypothesis_id for r in c.ruled_out] == ["reg"]
    assert c.ruled_out[0].reason == "No change in mobile checkout after the release."
    assert c.ruled_out[0].evidence_ids == ["r1"]

    untestable = {u.hypothesis_id: u for u in c.untestable}
    assert set(untestable) == {"price", "ext"}
    assert untestable["price"].category == "pricing_or_policy_change"
    assert "pricing and policy change log" in untestable["price"].data_request
    # concrete window, inclusive dates
    assert "2026-09-10..2026-09-19 (comparison window)" in untestable["price"].data_request
    assert "2026-09-01..2026-09-09 (baseline window)" in untestable["price"].data_request
    assert "Clarify the specific cause family" in untestable["ext"].data_request
    assert len(c.data_requests) == 2


def test_insufficient_evidence_with_nothing_untestable_still_gives_a_data_request(make_state):
    s = make_state()
    s.add_hypothesis("Release bug", "A release broke checkout.", "product_regression", "reg")
    c = s.stop_insufficient_evidence("Inconclusive.", "Results were ambiguous.", "No signal.")
    assert c.inconclusive_hypothesis_ids == ["reg"]
    assert c.untestable == []
    assert len(c.data_requests) == 1 and "No specific missing dataset" in c.data_requests[0]


def test_open_question_actions_become_data_requests(make_state):
    s = make_state()
    s.add_hypothesis("Release bug", "A release broke checkout.", "product_regression", "reg")
    s.add_unresolved_question("Was there a config change?", "reg", suggested_action="Export the flag change log for Sep 10-19")
    c = s.stop_insufficient_evidence("Inconclusive.", "Config history missing.", "No signal.")
    assert "Export the flag change log for Sep 10-19" in c.data_requests


def test_insufficient_conclusion_is_serialized(make_state):
    s = _insufficient_state(make_state)
    s.stop_insufficient_evidence("Cannot decide.", "No pricing log available.", "Nothing explains it.")
    d = s.to_dict()["conclusion"]
    assert d["outcome"] == "insufficient_evidence" and d["ruled_out"] and d["untestable"] and d["data_requests"]


# ----------------------------------------------------------------- guardrail


def test_ruling_out_every_testable_hypothesis_does_not_promote_an_untestable_one(make_state):
    s = make_state()
    s.add_hypothesis("Release bug", "A release broke checkout.", "product_regression", "reg")
    s.add_hypothesis("Tracking loss", "Events were dropped.", "tracking_failure", "trk")
    s.add_hypothesis("Price increase", "A pricing change reduced conversion.", "pricing_change", "price")
    add_support(s, "reg", "r1", "release_impact", p_value=0.9, stance=EvidenceStance.CONTRADICTS)
    add_support(s, "trk", "t1", "tracking_gap_check", p_value=0.8, stance=EvidenceStance.CONTRADICTS)
    s.update_hypothesis_status("reg", HypothesisStatus.REFUTED, "No release effect.", "r1")
    s.update_hypothesis_status("trk", HypothesisStatus.REFUTED, "Tracking intact.", "t1")

    # every testable hypothesis is ruled out; the untestable one is NOT supported by elimination
    assert s.investigation.hypotheses["price"].status == HypothesisStatus.PROPOSED
    with pytest.raises(SupportRuleViolation):
        s.update_hypothesis_status("price", HypothesisStatus.SUPPORTED, "It must be pricing by elimination.")
    with pytest.raises(ValueError, match="not 'supported'"):
        s.conclude("By elimination.", "price", ConfidenceLevel.HIGH, "ruled everything else out")

    c = s.stop_insufficient_evidence("Only an untestable cause remains.", "No pricing log.", "Elimination is not proof.")
    assert c.supported_causes == []
    assert [u.hypothesis_id for u in c.untestable] == ["price"]
    assert {r.hypothesis_id for r in c.ruled_out} == {"reg", "trk"}
    assert s.investigation.hypotheses["price"].status != HypothesisStatus.SUPPORTED


def test_untestable_hypothesis_stays_unsupported_even_with_supporting_links_and_significant_p(make_state):
    s = make_state()
    s.add_hypothesis("Price increase", "A pricing change reduced conversion.", "pricing_change", "price")
    add_support(s, "price", "p1", "segment_comparison", p_value=0.00001)
    add_support(s, "price", "p2", "funnel_comparison")
    with pytest.raises(SupportRuleViolation, match="UNTESTABLE"):
        s.update_hypothesis_status("price", HypothesisStatus.SUPPORTED, "Looks like it.", "p1")
