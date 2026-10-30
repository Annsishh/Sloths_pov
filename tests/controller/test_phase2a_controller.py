"""Phase 2a controller tests, including the two named acceptance tests for the SUPPORTED rule."""

import pandas as pd
import pytest

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import TimeWindow
from controller.capability_catalogue import CONTROLLED_OPERATIONS, validate_analysis_plan
from controller.controller import InvestigationController
from controller.llm_client import MockLLMClient
from controller.schemas import (
    AnalysisDecision,
    EvidenceInterpretation,
    HypothesesGenerationOutput,
    HypothesisProposal,
)
from investigation import HypothesisStatus, InvestigationState, InvestigationStatus, Observation, SupportPolicy


@pytest.fixture
def executor():
    users = pd.DataFrame([{"user_id": f"u{i}", "device_type": "mobile", "browser": "Chrome"} for i in range(10)])
    orders = pd.DataFrame([
        {"order_id": f"o{i}", "user_id": f"u{i}", "status": "failed", "failure_reason": "card_declined",
         "timestamp": pd.Timestamp("2026-09-12T10:00:00Z")} for i in range(5)])
    events = pd.DataFrame([
        {"event_id": f"e{i}", "user_id": f"u{i}", "session_id": f"s{i}", "timestamp": pd.Timestamp("2026-09-12T10:00:00Z"),
         "event_name": "checkout_failed", "device_type": "mobile", "browser": "Chrome", "app_version": "v2.3.8"}
        for i in range(5)])
    return AnalyticsExecutor(DataLoader.from_dataframes({"users": users, "orders": orders, "events": events}))


@pytest.fixture
def obs():
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    return Observation("order_conversion_rate", bw, cw, 0.20, 0.10, -0.10, -0.50, "Conversion halved.")


def _proposals():
    return HypothesesGenerationOutput(
        hypotheses=[HypothesisProposal(title="Tracking dropped", explanation="Telemetry events were dropped.", category="tracking_failure"),
                    HypothesisProposal(title="Release bug", explanation="A release broke checkout.", category="product_regression")],
        reasoning="Two competing explanations.")


def _decision(hyp, op="error_analysis", **params):
    return AnalysisDecision(hypothesis_id_to_test=hyp, operation=op, parameters=params, rationale="Check the breakdown.")


def _interp(hyp, **kw):
    kw.setdefault("stance", "supports")
    return EvidenceInterpretation(
        hypothesis_id=hyp, finding_summary="All 5 failed orders were card_declined.",
        rationale="This is consistent with the hypothesis.", status_update_reason="Because of the result.", **kw)


# ------------------------------------------------------------------ (a) single-evidence SUPPORTED is refused

def test_single_evidence_supported_recommendation_is_refused(executor, obs):
    """(a) A model recommending SUPPORTED after ONE piece of evidence is refused, whatever it claims."""
    canned = [
        _proposals(),
        _decision("hyp_1", "tracking_gap_check", breakdown=["browser"]),
        _interp("hyp_1", recommended_hypothesis_status="supported", is_investigation_complete=True,
                stopping_reason="Definitely tracking.", confidence_level="high", confidence_justification="Sure."),
    ]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=1)  # DEFAULT policy
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))

    h = final.investigation.hypotheses["hyp_1"]
    assert h.status != HypothesisStatus.SUPPORTED
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert final.investigation.conclusion.supported_causes == []
    refusal = [e for e in final.investigation.history if e.title == "Support Recommendation Not Accepted"]
    assert len(refusal) == 1 and "1 supporting evidence item(s); need 2" in refusal[0].description


def test_single_evidence_cannot_be_forced_through_the_state_api_either(executor, obs):
    state = InvestigationState.create("t", obs)
    state.add_hypothesis("Tracking dropped", "Telemetry events were dropped.", "tracking_failure", "h")
    state.record_evidence("f", ["orders"], operation="tracking_gap_check", p_value=1e-9, evidence_id="e1")
    from investigation import EvidenceStance, SupportRuleViolation
    state.link_evidence("e1", "h", EvidenceStance.SUPPORTS, "r")
    with pytest.raises(SupportRuleViolation):
        state.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "force", "e1")


# ------------------------------------------------------------------ (b) the DEFAULT policy is what is enforced

def test_default_support_policy_is_enforced(executor, obs):
    """(b) The documented defaults, with no policy passed anywhere, are what the controller and state enforce."""
    d = SupportPolicy()
    assert (d.min_supporting_evidence, d.min_distinct_analyses, d.max_contradicting_evidence,
            d.min_significant_supporting, d.alpha, d.correction) == (2, 2, 0, 1, 0.05, "bh")

    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient())      # no policy argument
    assert ctrl.support_policy == d
    state = ctrl.start_investigation(obs)
    assert state.investigation.support_policy == d.to_dict() and state.policy == d
    assert InvestigationState.create("t", obs).policy == d                             # default state too

    state.add_hypothesis("Tracking dropped", "Telemetry events were dropped.", "tracking_failure", "h")
    from investigation import EvidenceStance
    for i, op in enumerate(["tracking_gap_check", "event_volume_check"]):
        state.record_evidence("f", ["orders"], operation=op, evidence_id=f"e{i}", p_value=0.2)   # 2 analyses, NOT significant
        state.link_evidence(f"e{i}", "h", EvidenceStance.SUPPORTS, "r")
    assert not state.evaluate_support("h").eligible                                    # default alpha 0.05 enforced
    state.record_evidence("f", ["orders"], operation="funnel", evidence_id="e9", p_value=0.001)
    state.link_evidence("e9", "h", EvidenceStance.SUPPORTS, "r")
    assert state.evaluate_support("h").eligible
    state.record_evidence("f", ["orders"], operation="time_series", evidence_id="c1")
    state.link_evidence("c1", "h", EvidenceStance.CONTRADICTS, "r")
    assert not state.evaluate_support("h").eligible                                    # default: zero contradictions


# ------------------------------------------------------------------ catalogue exposure

@pytest.mark.parametrize("op", ["retention", "retention_comparison", "duplicate_event_check", "metric_aggregate",
                                "statistical_test", "seasonal_baseline", "mix_shift_decomposition"])
def test_phase2a_operations_are_exposed_and_their_examples_validate(op):
    assert op in CONTROLLED_OPERATIONS
    ok, err = validate_analysis_plan(op, CONTROLLED_OPERATIONS[op].example)
    assert ok, err


@pytest.mark.parametrize("op,params,fragment", [
    ("statistical_test", {"test_type": "anova"}, "test_type"),
    ("statistical_test", {"test_type": "proportions", "k1": 5, "n1": 10}, "requires 'k2'"),
    ("statistical_test", {"test_type": "proportions", "k1": 11, "n1": 10, "k2": 1, "n2": 5}, "cannot exceed"),
    ("statistical_test", {"test_type": "proportions", "k1": 1, "n1": 0, "k2": 1, "n2": 5}, "positive"),
    ("statistical_test", {"test_type": "means", "values1": [1.0], "values2": [1.0, 2.0]}, "2-10000"),
    ("statistical_test", {"test_type": "means", "values1": [1, "x"], "values2": [1.0, 2.0]}, "numbers"),
    ("seasonal_baseline", {"aggregation": "count", "baseline_weeks": 0}, "baseline_weeks"),
    ("seasonal_baseline", {"aggregation": "count", "baseline_weeks": 99}, "baseline_weeks"),
    ("retention", {"retention_days": [0]}, "retention_days"),
    ("retention", {"retention_days": list(range(1, 12))}, "retention_days"),
    ("retention_comparison", {"cohort_period": "decade"}, "cohort_period"),
    ("duplicate_event_check", {"max_per_session": 0}, "max_per_session"),
    ("mix_shift_decomposition", {"metric_column": "status", "metric_value": "completed"}, "requires parameter 'breakdown'"),
    ("metric_aggregate", {}, "requires parameter 'metric_name'"),
])
def test_phase2a_parameter_validation(op, params, fragment):
    ok, err = validate_analysis_plan(op, params)
    assert ok is False and fragment in err


def test_mix_shift_rejects_unknown_dimensions_and_filters():
    p = {"metric_column": "status", "metric_value": "completed", "breakdown": ["user_id"]}
    assert validate_analysis_plan("mix_shift_decomposition", p)[0] is False
    p["breakdown"] = ["device_type"]
    assert validate_analysis_plan("mix_shift_decomposition", p, filters={"ground_truth": "x"})[0] is False


def test_comparison_operations_always_receive_the_baseline_window(executor, obs):
    canned = [_proposals(), _decision("hyp_1", "retention_comparison"), _interp("hyp_1")]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=1)
    state = ctrl.start_investigation(obs)
    ctrl.step_formulate_hypotheses(state)
    dec = ctrl.step_select_analysis(state, 1)
    seen = []
    orig = executor.run
    executor.run = lambda req: (seen.append(req), orig(req))[1]
    ctrl.execute_analysis(state, dec)
    assert seen[0].comparison_window == obs.baseline_window and seen[0].window == obs.comparison_window


def test_executed_tests_join_the_family_even_if_interpretation_fails(executor, obs):
    canned = [_proposals(), _decision("hyp_1", "tracking_gap_check", breakdown=["browser"])]  # no interpretation available
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=1)
    state = ctrl.start_investigation(obs)
    ctrl.step_formulate_hypotheses(state)
    dec = ctrl.step_select_analysis(state, 1)
    result = ctrl.execute_analysis(state, dec)
    ctrl.step_interpret_result(state, dec, result)           # mock runs out -> interpretation fails
    assert state.investigation.evidence_store == {}
    assert state.tests_run == len(result.metadata.get("p_values", {}))
