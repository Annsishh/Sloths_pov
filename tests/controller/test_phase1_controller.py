"""Controller behaviour under the deterministic support rule (mock LLM, no API keys)."""

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
from investigation import HypothesisStatus, InvestigationStatus, Observation, SupportPolicy

# The tiny mock dataset has no inferential test, so only the significance threshold is relaxed.
NO_SIGNIFICANCE = SupportPolicy(min_significant_supporting=0)


@pytest.fixture
def executor():
    users = pd.DataFrame([{"user_id": f"u{i}", "device_type": "mobile", "browser": "Chrome"} for i in range(10)])
    orders = pd.DataFrame([
        {"order_id": f"o{i}", "user_id": f"u{i}", "status": "failed", "failure_reason": "card_declined",
         "timestamp": pd.Timestamp("2026-09-12T10:00:00Z")} for i in range(5)
    ])
    events = pd.DataFrame([
        {"event_id": f"e{i}", "user_id": f"u{i}", "session_id": f"s{i}", "timestamp": pd.Timestamp("2026-09-12T10:00:00Z"),
         "event_name": "checkout_failed", "device_type": "mobile", "browser": "Chrome", "app_version": "v2.3.8"}
        for i in range(5)
    ])
    return AnalyticsExecutor(DataLoader.from_dataframes({"users": users, "orders": orders, "events": events}))


@pytest.fixture
def obs():
    bw = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    cw = TimeWindow.from_iso("2026-09-10T00:00:00Z", "2026-09-20T00:00:00Z")
    return Observation("order_conversion_rate", bw, cw, 0.20, 0.10, -0.10, -0.50, "Conversion halved.")


def _proposals(*specs):
    return HypothesesGenerationOutput(
        hypotheses=[HypothesisProposal(title=t, explanation=f"{t}: a plausible explanation.", category=c) for t, c in specs],
        reasoning="Testing support rule handling.",
    )


def _decision(hyp, op="error_analysis", **params):
    return AnalysisDecision(hypothesis_id_to_test=hyp, operation=op, parameters=params,
                            rationale="Check the failed order breakdown.")


def _interp(hyp, stance="supports", status=None, complete=False, refined=None, **kw):
    return EvidenceInterpretation(
        hypothesis_id=hyp, stance=stance, finding_summary="All 5 failed orders were card_declined.",
        rationale="This is consistent with the hypothesis.", recommended_hypothesis_status=status,
        status_update_reason="Because of the result.", is_investigation_complete=complete,
        refined_category=refined, **kw,
    )


def test_start_investigation_derives_testability_from_available_tables(executor, obs):
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient())
    state = ctrl.start_investigation(obs)
    t = state.testability
    assert t["third_party_outage"].testable and t["tracking_loss"].testable
    assert not t["code_regression"].testable   # no releases table in this dataset
    assert not t["pricing_or_policy_change"].testable
    assert state.investigation.support_policy == SupportPolicy().to_dict()


def test_model_recommendation_of_supported_is_not_accepted_for_a_placeholder_category(executor, obs):
    canned = [
        _proposals(("Card network outage", "external_factor"), ("Release bug", "product_regression")),
        _decision("hyp_1"),
        _interp("hyp_1", status="supported", complete=True),     # placeholder category + 1 analysis
        _decision("hyp_1", "time_series", table="orders", aggregation="count"),
        _interp("hyp_1", status="supported", complete=True),     # 2 analyses but still a placeholder
    ]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=3,
                                   support_policy=NO_SIGNIFICANCE)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))

    h = final.investigation.hypotheses["hyp_1"]
    assert h.category == "external_unspecified"
    assert h.status != HypothesisStatus.SUPPORTED
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    titles = [e.title for e in final.investigation.history]
    assert titles.count("Support Recommendation Not Accepted") == 2
    assert any("placeholder" in e.description for e in final.investigation.history if e.title == "Support Recommendation Not Accepted")
    assert "hyp_1" in [u.hypothesis_id for u in final.investigation.conclusion.untestable]
    assert final.validate() == []


def test_refining_the_placeholder_lets_the_rule_grant_support_and_conclude(executor, obs):
    canned = [
        _proposals(("Card network outage", "external_factor"), ("Release bug", "product_regression")),
        _decision("hyp_1"),
        _interp("hyp_1", status="supported", complete=True),
        _decision("hyp_1", "time_series", table="orders", aggregation="count"),
        _interp("hyp_1", status="supported", complete=True, refined="third_party_outage",
                stopping_reason="Card declines at the payment network explain the drop.",
                confidence_level="medium"),
    ]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=3,
                                   support_policy=NO_SIGNIFICANCE)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))

    h = final.investigation.hypotheses["hyp_1"]
    assert h.category == "third_party_outage" and h.status == HypothesisStatus.SUPPORTED
    assert final.investigation.status == InvestigationStatus.CONCLUDED
    assert final.investigation.conclusion.outcome == "concluded"
    assert [c.hypothesis_id for c in final.investigation.conclusion.supported_causes] == ["hyp_1"]
    assert final.investigation.conclusion.unexplained_share == pytest.approx(1.0)  # nothing quantified
    assert "Completion Deferred" in [e.title for e in final.investigation.history]
    assert final.validate() == []


def test_untestable_hypothesis_is_not_promoted_when_the_testable_one_is_refuted(executor, obs):
    canned = [
        _proposals(("Price increase", "pricing_change"), ("Card network outage", "external_factor")),
        _decision("hyp_1"),
        _interp("hyp_1", status="supported", complete=True),    # model claims victory for an untestable cause
        _decision("hyp_2"),
        _interp("hyp_2", stance="contradicts", status="refuted"),
    ]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=3,
                                   support_policy=NO_SIGNIFICANCE)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))

    assert final.investigation.hypotheses["hyp_1"].status != HypothesisStatus.SUPPORTED
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    c = final.investigation.conclusion
    assert c.supported_causes == []
    assert [r.hypothesis_id for r in c.ruled_out] == ["hyp_2"]
    price = next(u for u in c.untestable if u.hypothesis_id == "hyp_1")
    assert "pricing and policy change log" in price.data_request
    assert "2026-09-10..2026-09-19" in price.data_request


def test_unsupported_recommendation_does_not_end_the_loop_early(executor, obs):
    """One analysis + 'complete' from the model must not conclude; a second analysis can."""
    canned = [
        _proposals(("Card network outage", "third_party_outage"), ("Release bug", "product_regression")),
        _decision("hyp_1"),
        _interp("hyp_1", status="supported", complete=True),
    ]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=2,
                                   support_policy=NO_SIGNIFICANCE)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))
    # the model ran out of canned replies in iteration 2 -> halted as insufficient, never silently concluded
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert final.investigation.hypotheses["hyp_1"].status == HypothesisStatus.ACTIVE
    assert final.investigation.conclusion.supported_causes == []


def test_evidence_source_tables_come_from_the_catalogue(executor, obs):
    canned = [
        _proposals(("Card network outage", "third_party_outage"), ("Release bug", "product_regression")),
        _decision("hyp_1"),
        _interp("hyp_1"),
    ]
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), max_iterations=1)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))
    ev = next(iter(final.investigation.evidence_store.values()))
    assert ev.source_tables == ["orders", "users"]


def test_proposal_categories_are_normalized_by_the_schema():
    p = HypothesisProposal(title="Some hypothesis", explanation="A plausible explanation text.", category="Tracking_Failure")
    assert p.category == "tracking_loss"
    # Unknown names are kept so the investigation state can coerce AND log a warning (see test_audit_events).
    assert HypothesisProposal(title="Some hypothesis", explanation="A plausible explanation text.", category="nonsense").category == "nonsense"
    assert HypothesisProposal(title="Some hypothesis", explanation="A plausible explanation text.").category == "unspecified"
