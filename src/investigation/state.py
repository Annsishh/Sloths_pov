"""InvestigationState controller and lifecycle manager.

Provides a robust, validated, immutable-audit-trail state machine for:
- Initializing an investigation from a triggering observation
- Registering and iteratively updating competing hypotheses
- Recording empirical evidence with strict provenance and limitations
- Disentangling empirical findings from hypothesis interpretations
- Logging shifts in investigative direction and decisions
- Distinguishing definitive conclusions from halts due to insufficient evidence
- Validating referential integrity and preventing ground-truth leakage
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from analytics.models import AnalysisResult, TimeWindow
from investigation.enums import (
    PLACEHOLDER_CATEGORIES,
    ConfidenceLevel,
    HypothesisCategory,
    normalize_category,
    EvidenceStance,
    HistoryEventType,
    HypothesisStatus,
    InferenceType,
    InvestigationStatus,
    QuestionStatus,
)
from investigation.models import (
    ConfidenceAssessment,
    Conclusion,
    Evidence,
    ExplainedShare,
    RuledOutHypothesis,
    UntestableHypothesis,
    EffectEstimate,
    extract_tests,
    window_to_dict,
    HistoryEvent,
    Hypothesis,
    HypothesisEvidenceLink,
    HypothesisTransition,
    Investigation,
    Observation,
    UnresolvedQuestion,
    _utc_now,
)
from investigation.support import (
    SupportDecision,
    SupportPolicy,
    adjusted_p_values,
    adjusted_test_p_values,
    evaluate_support,
    evidence_test_keys,
)
from investigation.testability import TestabilityAssessment, assess_testability, attribution_strength


# Known privileged terms that must never leak from synthetic ground truth into the investigation state
GROUND_TRUTH_FORBIDDEN_MARKERS = [
    "scenario_ground_truth",
    "ground_truth.json",
    "injected_parameters",
    "root_cause_label",
    "expected_scenario_outcome",
]


class SupportRuleViolation(ValueError):
    """Raised when a hypothesis is marked SUPPORTED without satisfying the deterministic rule."""

    def __init__(self, hypothesis_id: str, reasons: List[str]) -> None:
        self.hypothesis_id = hypothesis_id
        self.reasons = list(reasons)
        super().__init__(f"Hypothesis '{hypothesis_id}' cannot be SUPPORTED: " + "; ".join(reasons))


class InvestigationState:
    """Manages the lifecycle and state transitions of an active or concluded Investigation."""

    def __init__(self, investigation: Investigation) -> None:
        self.investigation = investigation

    @property
    def policy(self) -> SupportPolicy:
        return SupportPolicy.from_dict(self.investigation.support_policy)

    @property
    def testability(self) -> Dict[str, TestabilityAssessment]:
        return self.investigation.testability

    @property
    def tests_run(self) -> int:
        """Size of the investigation-wide multiple-testing family (every registered test)."""
        return len(self.investigation.test_registry)

    def register_tests(self, source_id: str, tests: Dict[str, float], provenance: str = "computed") -> int:
        """Add a tool's raw p-values to the BH family under `source_id`. Idempotent per (source, label).

        Caller-supplied p-values (unverifiable inputs) are never registered.
        """
        if provenance != "computed":
            return 0
        added = 0
        for label, p in tests.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p-value for '{label}' must lie in [0, 1]; got {p}")
            key = f"{source_id}|{label}"
            if key not in self.investigation.test_registry:
                self.investigation.test_registry[key] = float(p)
                added += 1
        return added

    def register_analysis_result(self, result: AnalysisResult) -> int:
        """Register a result's tests as soon as it is executed, even if it is never interpreted
        (dropping unfavourable tests must not shrink the correction family)."""
        meta = result.metadata if isinstance(result.metadata, dict) else {}
        return self.register_tests(result.request_id, extract_tests(result), meta.get("p_value_provenance", "computed"))

    def adjusted_test_p_values(self) -> Dict[str, float]:
        return adjusted_test_p_values(self.investigation, self.policy)

    def adjusted_p_values(self) -> Dict[str, float]:
        return adjusted_p_values(self.investigation, self.policy)

    @staticmethod
    def default_testability() -> Dict[str, TestabilityAssessment]:
        """Testability against the full capability catalogue and every known table."""
        from analytics.loader import CATALOGUE
        from controller.capability_catalogue import CONTROLLED_OPERATIONS
        return assess_testability(CONTROLLED_OPERATIONS, CATALOGUE.keys())

    # ------------------------------------------------------------------ factory
    @classmethod
    def create(
        cls,
        title: str,
        observation: Observation,
        investigation_id: Optional[str] = None,
        support_policy: Optional[SupportPolicy] = None,
        testability: Optional[Dict[str, TestabilityAssessment]] = None,
    ) -> "InvestigationState":
        """Initialize a new investigation triggered by an observed metric deviation."""
        inv_id = investigation_id or f"inv_{uuid.uuid4().hex[:8]}"
        now = _utc_now()

        inv = Investigation(
            investigation_id=inv_id,
            title=title,
            observation=observation,
            status=InvestigationStatus.OPEN,
            created_at=now,
            updated_at=now,
            support_policy=(support_policy or SupportPolicy()).to_dict(),
            testability=dict(testability) if testability is not None else cls.default_testability(),
        )

        state = cls(inv)
        state._record_event(
            event_type=HistoryEventType.INVESTIGATION_TRIGGERED,
            title="Investigation Initiated",
            description=(
                f"Investigation triggered for metric '{observation.metric_name}' "
                f"exhibiting a change of {observation.absolute_change:+.2f} "
                f"({observation.relative_change:+.1%})"
                if observation.relative_change is not None
                else f"change of {observation.absolute_change:+.2f}"
            ),
            metadata={"observation": observation.to_dict()},
        )
        return state

    # ------------------------------------------------------------------ hypotheses
    def add_hypothesis(
        self,
        title: str,
        explanation: str,
        category: str,
        hypothesis_id: Optional[str] = None,
    ) -> Hypothesis:
        """Formulate a new competing hypothesis."""
        if not title or not title.strip():
            raise ValueError("Hypothesis title cannot be empty.")
        if not explanation or not explanation.strip():
            raise ValueError("Hypothesis explanation cannot be empty.")

        hyp_id = hypothesis_id or f"hyp_{len(self.investigation.hypotheses) + 1}"
        if hyp_id in self.investigation.hypotheses:
            raise ValueError(f"Hypothesis with ID '{hyp_id}' already exists.")

        now = _utc_now()
        norm_cat = normalize_category(category)
        raw_clean = (category or "").strip().lower()
        coerced = norm_cat == HypothesisCategory.UNSPECIFIED and raw_clean not in {"unspecified", "other"}
        hyp = Hypothesis(
            hypothesis_id=hyp_id,
            title=title.strip(),
            explanation=explanation.strip(),
            category=norm_cat.value,
            status=HypothesisStatus.PROPOSED,
            created_at=now,
            updated_at=now,
        )

        self.investigation.hypotheses[hyp_id] = hyp
        self.investigation.updated_at = now

        # Transition status to IN_PROGRESS if open
        if self.investigation.status == InvestigationStatus.OPEN:
            self.investigation.status = InvestigationStatus.IN_PROGRESS

        self._record_event(
            event_type=HistoryEventType.HYPOTHESIS_ADDED,
            title=f"Hypothesis Added: {title}",
            description=explanation,
            metadata={"hypothesis_id": hyp_id, "category": norm_cat.value, "original_category": category.strip()},
        )
        if coerced:
            self._record_event(
                event_type=HistoryEventType.WARNING,
                title="Warning: unknown category coerced to 'unspecified'",
                description=(f"Hypothesis '{hyp_id}' was proposed with unknown category '{category.strip()}'; "
                             "it is stored as 'unspecified' and must be refined before it can be supported."),
                metadata={"hypothesis_id": hyp_id, "original_category": category.strip(), "coerced_to": "unspecified"},
            )
        return hyp

    def refine_category(self, hypothesis_id: str, new_category: str, reason: str) -> Hypothesis:
        """Re-categorise a hypothesis (typically a placeholder) into a specific cause family."""
        if hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found in investigation.")
        if not reason or not reason.strip():
            raise ValueError("Category refinement requires an explicit reason.")
        try:
            new_cat = HypothesisCategory(new_category.strip().lower())
        except ValueError:
            raise ValueError(f"Unknown category '{new_category}'.") from None
        if new_cat in PLACEHOLDER_CATEGORIES or new_cat == HypothesisCategory.MULTIPLE_CAUSES:
            raise ValueError(f"'{new_cat.value}' is not a specific cause family; refinement must name one.")

        hyp = self.investigation.hypotheses[hypothesis_id]
        old = hyp.category
        if old == new_cat.value:
            return hyp
        hyp.category = new_cat.value
        hyp.updated_at = _utc_now()
        self._record_event(
            event_type=HistoryEventType.HYPOTHESIS_STATUS_UPDATED,
            title=f"Hypothesis Category Refined: {hyp.title}: {old} -> {new_cat.value}",
            description=reason.strip(),
            metadata={"hypothesis_id": hypothesis_id, "previous_category": old, "new_category": new_cat.value},
        )
        return hyp

    # ------------------------------------------------------------------ support rule
    def evaluate_support(self, hypothesis_id: str) -> SupportDecision:
        """Apply the deterministic SUPPORTED rule (see investigation.support) without changing state."""
        if hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found in investigation.")
        return evaluate_support(self.investigation, hypothesis_id, self.policy, self.testability)

    def reassess_supported(self) -> List[str]:
        """Demote SUPPORTED hypotheses that no longer satisfy the rule (e.g. BH-adjusted p-values
        rise as more tests accumulate). Every withdrawal is logged with before/after adjusted p-values."""
        demoted: List[str] = []
        for hid, hyp in list(self.investigation.hypotheses.items()):
            if hyp.status != HypothesisStatus.SUPPORTED:
                continue
            decision = self.evaluate_support(hid)
            if decision.eligible:
                continue
            after_all = self.adjusted_p_values()
            before = dict(hyp.support_snapshot)
            after = {e: after_all[e] for e in hyp.supporting_evidence_ids if e in after_all}
            reason = "Support rule no longer satisfied: " + "; ".join(decision.reasons)
            self._set_status(hyp, HypothesisStatus.ACTIVE, reason, None)
            self._record_event(
                event_type=HistoryEventType.SUPPORT_WITHDRAWN,
                title=f"Support Withdrawn: {hyp.title}",
                description=reason,
                metadata={
                    "hypothesis_id": hid,
                    "adjusted_p_before": before,
                    "adjusted_p_after": after,
                    "tests_in_family": self.tests_run,
                    "alpha": self.policy.alpha,
                    "reasons": list(decision.reasons),
                },
            )
            demoted.append(hid)
        return demoted

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: HypothesisStatus,
        reason: str,
        triggered_by_evidence_id: Optional[str] = None,
    ) -> Hypothesis:
        """Update a hypothesis's status with an explicit transition audit record."""
        if hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found in investigation.")
        if not reason or not reason.strip():
            raise ValueError("Status transition requires an explicit explanatory reason.")

        if triggered_by_evidence_id and triggered_by_evidence_id not in self.investigation.evidence_store:
            raise KeyError(f"Evidence '{triggered_by_evidence_id}' referenced in transition does not exist.")

        hyp = self.investigation.hypotheses[hypothesis_id]
        if hyp.status == new_status:
            return hyp  # No-op

        if new_status == HypothesisStatus.SUPPORTED:
            decision = self.evaluate_support(hypothesis_id)
            if not decision.eligible:
                raise SupportRuleViolation(hypothesis_id, decision.reasons)

        self._set_status(hyp, new_status, reason.strip(), triggered_by_evidence_id)
        return hyp

    def _set_status(
        self,
        hyp: Hypothesis,
        new_status: HypothesisStatus,
        reason: str,
        triggered_by_evidence_id: Optional[str],
    ) -> None:
        now = _utc_now()
        transition = HypothesisTransition(
            transition_id=f"trans_{uuid.uuid4().hex[:8]}",
            previous_status=hyp.status,
            new_status=new_status,
            reason=reason,
            triggered_by_evidence_id=triggered_by_evidence_id,
            timestamp=now,
        )
        hyp.history.append(transition)
        hyp.status = new_status
        if new_status == HypothesisStatus.SUPPORTED:
            adj = self.adjusted_p_values()
            hyp.support_snapshot = {e: adj[e] for e in hyp.supporting_evidence_ids if e in adj}
        hyp.updated_at = now
        self.investigation.updated_at = now
        self._record_event(
            event_type=HistoryEventType.HYPOTHESIS_STATUS_UPDATED,
            title=f"Hypothesis Status Changed: {hyp.title} -> {new_status.value}",
            description=reason,
            metadata={
                "hypothesis_id": hyp.hypothesis_id,
                "previous_status": transition.previous_status.value,
                "new_status": new_status.value,
                "triggered_by_evidence_id": triggered_by_evidence_id,
            },
        )

    # ------------------------------------------------------------------ evidence
    def record_evidence(
        self,
        finding: str,
        source_tables: List[str],
        analysis_result: Optional[AnalysisResult] = None,
        analysis_request_id: Optional[str] = None,
        operation: Optional[str] = None,
        population: Optional[Dict[str, Any]] = None,
        window: Optional[TimeWindow] = None,
        result_summary: Optional[Dict[str, Any]] = None,
        limitations: Optional[List[str]] = None,
        inference_type: InferenceType = InferenceType.OBSERVATIONAL_CORRELATION,
        evidence_id: Optional[str] = None,
        p_value: Optional[float] = None,
        mde: Optional[float] = None,
    ) -> Evidence:
        """Record an empirical evidence item in the investigation state."""
        if not finding or not finding.strip():
            raise ValueError("Evidence finding statement cannot be empty.")
        if not source_tables:
            raise ValueError("Evidence must specify at least one source table.")

        ev_id = evidence_id or f"ev_{len(self.investigation.evidence_store) + 1}_{uuid.uuid4().hex[:4]}"
        if ev_id in self.investigation.evidence_store:
            raise ValueError(f"Evidence with ID '{ev_id}' already exists.")

        if analysis_result is not None:
            ev = Evidence.from_analysis_result(
                evidence_id=ev_id,
                finding=finding.strip(),
                result=analysis_result,
                source_tables=source_tables,
                population=population,
                window=window,
                inference_type=inference_type,
            )
            if p_value is not None:
                ev.p_value = p_value
            if mde is not None:
                ev.mde = mde
        else:
            ev = Evidence(
                evidence_id=ev_id,
                finding=finding.strip(),
                source_tables=list(source_tables),
                analysis_request_id=analysis_request_id,
                operation=operation,
                population=population or {},
                window=window,
                result_summary=result_summary or {},
                limitations=list(limitations or []),
                inference_type=inference_type,
                recorded_at=_utc_now(),
                p_value=p_value,
                mde=mde,
            )
        if ev.p_value is not None and not (0.0 <= ev.p_value <= 1.0):
            raise ValueError(f"p_value must lie in [0, 1]; got {ev.p_value}")

        self.investigation.evidence_store[ev_id] = ev
        self.register_tests(ev.analysis_request_id or ev.evidence_id, ev.all_tests(), ev.p_value_provenance)
        self.investigation.updated_at = _utc_now()

        self._record_event(
            event_type=HistoryEventType.EVIDENCE_RECORDED,
            title=f"Evidence Recorded: {ev_id}",
            description=finding,
            metadata={
                "evidence_id": ev_id,
                "source_tables": source_tables,
                "operation": ev.operation,
                "inference_type": ev.inference_type.value,
                "p_value": ev.p_value,
                "p_values": dict(ev.p_values),
                "tests_in_family": self.tests_run,
            },
        )
        return ev

    def link_evidence(
        self,
        evidence_id: str,
        hypothesis_id: str,
        stance: EvidenceStance,
        rationale: str,
    ) -> HypothesisEvidenceLink:
        """Link an evidence item to a hypothesis with an explicit investigative interpretation."""
        if evidence_id not in self.investigation.evidence_store:
            raise KeyError(f"Evidence '{evidence_id}' does not exist.")
        if hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' does not exist.")
        if not rationale or not rationale.strip():
            raise ValueError("Linking evidence requires an explicit rationale explaining the interpretation.")

        hyp = self.investigation.hypotheses[hypothesis_id]
        link = HypothesisEvidenceLink(
            evidence_id=evidence_id,
            hypothesis_id=hypothesis_id,
            stance=stance,
            rationale=rationale.strip(),
            recorded_at=_utc_now(),
        )

        # Update hypothesis direct index
        if stance == EvidenceStance.SUPPORTS:
            if evidence_id not in hyp.supporting_evidence_ids:
                hyp.supporting_evidence_ids.append(evidence_id)
        elif stance == EvidenceStance.CONTRADICTS:
            if evidence_id not in hyp.contradicting_evidence_ids:
                hyp.contradicting_evidence_ids.append(evidence_id)

        self.investigation.evidence_links.append(link)
        hyp.updated_at = _utc_now()
        self.investigation.updated_at = _utc_now()

        self._record_event(
            event_type=HistoryEventType.EVIDENCE_LINKED,
            title=f"Evidence {evidence_id} linked to {hypothesis_id} ({stance.value})",
            description=rationale,
            metadata={
                "evidence_id": evidence_id,
                "hypothesis_id": hypothesis_id,
                "stance": stance.value,
            },
        )
        return link

    # ------------------------------------------------------------------ unresolved questions
    def add_unresolved_question(
        self,
        question_text: str,
        hypothesis_id: Optional[str] = None,
        context_or_limitation: str = "",
        suggested_action: str = "",
        question_id: Optional[str] = None,
    ) -> UnresolvedQuestion:
        """Record an open information gap, missing data issue, or verification question."""
        if not question_text or not question_text.strip():
            raise ValueError("Question text cannot be empty.")
        if hypothesis_id and hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' does not exist.")

        q_id = question_id or f"q_{len(self.investigation.unresolved_questions) + 1}_{uuid.uuid4().hex[:4]}"
        now = _utc_now()
        q = UnresolvedQuestion(
            question_id=q_id,
            question_text=question_text.strip(),
            hypothesis_id=hypothesis_id,
            status=QuestionStatus.OPEN,
            context_or_limitation=context_or_limitation.strip(),
            suggested_action=suggested_action.strip(),
            created_at=now,
        )

        self.investigation.unresolved_questions[q_id] = q
        self.investigation.updated_at = now

        self._record_event(
            event_type=HistoryEventType.QUESTION_ADDED,
            title=f"Unresolved Question Added: {q_id}",
            description=question_text,
            metadata={"question_id": q_id, "hypothesis_id": hypothesis_id},
        )
        return q

    def answer_question(
        self,
        question_id: str,
        resolution: str,
    ) -> UnresolvedQuestion:
        """Resolve a previously open question with an answer or verification result."""
        if question_id not in self.investigation.unresolved_questions:
            raise KeyError(f"Question '{question_id}' not found.")
        if not resolution or not resolution.strip():
            raise ValueError("Question resolution cannot be empty.")

        q = self.investigation.unresolved_questions[question_id]
        now = _utc_now()
        q.resolution = resolution.strip()
        q.status = QuestionStatus.ANSWERED
        q.resolved_at = now
        self.investigation.updated_at = now

        self._record_event(
            event_type=HistoryEventType.QUESTION_ANSWERED,
            title=f"Question Resolved: {question_id}",
            description=resolution,
            metadata={"question_id": question_id},
        )
        return q

    # ------------------------------------------------------------------ direction pivots
    def record_direction_change(
        self,
        from_focus: str,
        to_focus: str,
        rationale: str,
    ) -> HistoryEvent:
        """Record an intentional shift in investigative focus or hypothesis prioritization."""
        if not rationale or not rationale.strip():
            raise ValueError("Direction changes require an explicit rationale explaining the pivot.")

        return self._record_event(
            event_type=HistoryEventType.DIRECTION_CHANGED,
            title=f"Investigative Pivot: {from_focus} -> {to_focus}",
            description=rationale.strip(),
            metadata={"from_focus": from_focus, "to_focus": to_focus},
        )

    # ------------------------------------------------------------------ conclusion & stopping
    def supported_hypothesis_ids(self) -> List[str]:
        return [h.hypothesis_id for h in self.investigation.hypotheses.values()
                if h.status == HypothesisStatus.SUPPORTED]

    def conclude(
        self,
        conclusion_summary: str,
        supported_hypothesis_id: str,
        confidence_level: ConfidenceLevel,
        confidence_justification: str,
        assumptions: Optional[List[str]] = None,
        remaining_uncertainties: Optional[List[str]] = None,
        additional_supported_ids: Optional[List[str]] = None,
    ) -> Conclusion:
        """Conclude the investigation with a substantiated conclusion.

        Explained shares come ONLY from deterministic tool estimates (`Evidence.effect_estimate`):
        the most precise supporting estimate is used; a cause with none is reported as unquantified.
        """
        if not conclusion_summary or not conclusion_summary.strip():
            raise ValueError("Conclusion summary cannot be empty.")
        if supported_hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Supported hypothesis '{supported_hypothesis_id}' not found.")

        ids = [supported_hypothesis_id] + [i for i in (additional_supported_ids or []) if i != supported_hypothesis_id]
        for hid in ids:
            if hid not in self.investigation.hypotheses:
                raise KeyError(f"Supported hypothesis '{hid}' not found.")
            hyp = self.investigation.hypotheses[hid]
            if hyp.status != HypothesisStatus.SUPPORTED:
                raise ValueError(
                    f"Cannot conclude with hypothesis '{hid}' because its status "
                    f"is '{hyp.status.value}', not 'supported'."
                )
            if not hyp.supporting_evidence_ids:
                raise ValueError(
                    f"Cannot conclude with hypothesis '{hid}' because it has no supporting evidence."
                )

        # Adjusted p-values shift as tests accumulate; make sure the rule still holds at conclusion time.
        demoted = [d for d in self.reassess_supported() if d in ids]
        if demoted:
            raise ValueError(f"Cannot conclude: hypothesis(es) {demoted} no longer satisfy the support rule.")

        conclusion = self._build_conclusion("concluded", ids)

        now = _utc_now()
        self.investigation.status = InvestigationStatus.CONCLUDED
        self.investigation.stopping_reason = conclusion_summary.strip()
        self.investigation.conclusion = conclusion
        self.investigation.confidence = ConfidenceAssessment(
            level=confidence_level,
            justification=confidence_justification.strip(),
            competing_hypotheses_addressed=len(self.investigation.hypotheses) > 1,
            assumptions=list(assumptions or []),
            remaining_uncertainties=list(remaining_uncertainties or []),
            assessed_at=now,
        )
        self.investigation.updated_at = now

        self._record_event(
            event_type=HistoryEventType.INVESTIGATION_CONCLUDED,
            title="Investigation Concluded",
            description=conclusion_summary,
            metadata={
                "supported_hypothesis_id": supported_hypothesis_id,
                "supported_hypothesis_ids": ids,
                "confidence_level": confidence_level.value,
                "unexplained_share": conclusion.unexplained_share,
            },
        )
        return conclusion

    def stop_insufficient_evidence(
        self,
        reason: str,
        missing_information_summary: str,
        confidence_justification: str,
        remaining_uncertainties: Optional[List[str]] = None,
    ) -> Conclusion:
        """Halt because essential data is missing, unobserved, or inconclusive.

        The resulting Conclusion lists what was ruled out and why, what remains untestable with
        the available data, and a concrete data request. Ruling out the testable hypotheses never
        promotes an untestable one: SUPPORTED always requires positive, rule-satisfying evidence.
        """
        if not reason or not reason.strip():
            raise ValueError("Stopping reason cannot be empty.")
        if not missing_information_summary or not missing_information_summary.strip():
            raise ValueError("Must specify what missing information prevented further investigation.")

        self.reassess_supported()
        conclusion = self._build_conclusion(
            "insufficient_evidence", self.supported_hypothesis_ids(),
            missing_summary=missing_information_summary.strip(),
        )

        now = _utc_now()
        self.investigation.status = InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
        self.investigation.stopping_reason = f"{reason.strip()} [Missing: {missing_information_summary.strip()}]"
        self.investigation.conclusion = conclusion
        self.investigation.confidence = ConfidenceAssessment(
            level=ConfidenceLevel.LOW,
            justification=confidence_justification.strip(),
            competing_hypotheses_addressed=False,
            assumptions=[],
            remaining_uncertainties=list(remaining_uncertainties or [missing_information_summary]),
            assessed_at=now,
        )
        self.investigation.updated_at = now

        self._record_event(
            event_type=HistoryEventType.INVESTIGATION_STOPPED,
            title="Investigation Halted (Insufficient Evidence)",
            description=self.investigation.stopping_reason,
            metadata={
                "missing_information": missing_information_summary,
                "ruled_out": [r.hypothesis_id for r in conclusion.ruled_out],
                "untestable": [u.hypothesis_id for u in conclusion.untestable],
                "data_requests": list(conclusion.data_requests),
            },
        )
        return conclusion

    # ------------------------------------------------------------------ conclusion accounting
    @staticmethod
    def _fmt_window(w: TimeWindow) -> str:
        last_day = (w.end - timedelta(microseconds=1)).date()
        return f"{w.start.date()}..{last_day}"

    def _data_request_text(self, what: str) -> str:
        obs = self.investigation.observation
        return (f"{what} for {self._fmt_window(obs.comparison_window)} (comparison window) "
                f"and {self._fmt_window(obs.baseline_window)} (baseline window)")

    def _estimate_for(self, hyp: Hypothesis):
        """Best deterministic estimate for a cause: the supporting evidence's most precise EffectEstimate."""
        total = self.investigation.observation.absolute_change
        ests = [(i, self.investigation.evidence_store[i].effect_estimate) for i in hyp.supporting_evidence_ids
                if i in self.investigation.evidence_store and self.investigation.evidence_store[i].effect_estimate is not None]
        if ests:
            eid, est = min(ests, key=lambda t: t[1].uncertainty_width())
            note = "most precise of %d estimate(s)" % len(ests) if len(ests) > 1 else "single tool estimate"
            return est, [eid], note
        if normalize_category(hyp.category) == HypothesisCategory.NATURAL_VARIATION:
            return None, list(hyp.supporting_evidence_ids), "change attributed to chance variation (not a tool estimate)"
        return None, list(hyp.supporting_evidence_ids), "no quantified effect estimate available"

    @staticmethod
    def _statement(title: str, category: str, attribution: str, est: Optional[EffectEstimate], share: Optional[float]) -> str:
        lead = (f"{title} ({category}) is supported by the evidence" if attribution == "confirmed"
                else f"The evidence is consistent with {title} ({category}); the available tools cannot confirm the underlying cause")
        if est is None or share is None:
            return lead + "; its contribution to the observed change could not be quantified."
        unc = (f" +/- {est.se:.3g} SE" if est.se is not None else f" (CI {est.ci_lower:.3g} to {est.ci_upper:.3g})")
        return lead + f", accounting for about {share:.0%} of the observed change (estimate {est.value:+.4g}{unc}, method {est.method})."

    def _build_conclusion(
        self,
        outcome: str,
        supported_ids: List[str],
        missing_summary: str = "",
    ) -> Conclusion:
        inv = self.investigation
        total = inv.observation.absolute_change
        warnings: List[str] = []

        causes: List[ExplainedShare] = []
        estimates: Dict[str, EffectEstimate] = {}
        for hid in supported_ids:
            hyp = inv.hypotheses[hid]
            est, basis, note = self._estimate_for(hyp)
            if normalize_category(hyp.category) == HypothesisCategory.NATURAL_VARIATION:
                value, share = total, (1.0 if total != 0 else None)
            else:
                value = est.value if est else None
                share = (est.value / total) if (est is not None and total != 0) else None
            attribution = attribution_strength(hyp.category)
            if est is not None:
                estimates[hid] = est
            causes.append(ExplainedShare(
                hid, hyp.category, value, share, basis, note, attribution,
                method=est.method if est else "", se=est.se if est else None,
                ci_lower=est.ci_lower if est else None, ci_upper=est.ci_upper if est else None,
                window=est.window if est else None,
                statement=self._statement(hyp.title, hyp.category, attribution, est, share),
            ))

        quantified = [c for c in causes if c.estimated_change is not None]
        explained = sum(c.estimated_change for c in quantified)
        unexplained = total - explained
        unexplained_share = (unexplained / total) if total != 0 else None

        unquantified = [c.hypothesis_id for c in causes if c.estimated_change is None]
        if unquantified:
            warnings.append(
                f"Supported cause(s) {unquantified} have no quantified effect; the unexplained remainder "
                "may include their contribution."
            )
        methods = sorted({e.method for e in estimates.values()})
        if len(methods) > 1:
            warnings.append(
                "Possible double counting: shares come from different methods "
                f"({methods}) whose attributions may overlap; do not assume they add up independently."
            )
        if total == 0:
            warnings.append("Observed change is zero; shares are undefined.")
        elif abs(explained) > abs(total) * (1 + 1e-9):
            warnings.append(
                f"Attributed changes ({explained:+.4g}) exceed the observed change ({total:+.4g}); "
                "estimates overlap or are overstated."
            )
        if total != 0 and any(c.estimated_change is not None and c.estimated_change * total < 0 for c in causes):
            warnings.append("At least one cause is estimated to move the metric opposite to the observed change.")

        ruled_out: List[RuledOutHypothesis] = []
        untestable: List[UntestableHypothesis] = []
        inconclusive: List[str] = []
        requests: List[str] = []
        for hid, hyp in inv.hypotheses.items():
            if hyp.status == HypothesisStatus.SUPPORTED:
                continue
            if hyp.status == HypothesisStatus.REFUTED:
                last = next((t for t in reversed(hyp.history) if t.new_status == HypothesisStatus.REFUTED), None)
                ruled_out.append(RuledOutHypothesis(
                    hid, hyp.title, hyp.category,
                    last.reason if last else "marked refuted",
                    list(hyp.contradicting_evidence_ids),
                ))
                continue
            ta = inv.testability.get(normalize_category(hyp.category).value)
            if ta is not None and not ta.testable:
                if ta.data_request:
                    request = self._data_request_text(ta.data_request)
                else:
                    request = (f"Clarify the specific cause family of hypothesis '{hyp.title}' "
                               "(its category is not specific enough to test)")
                untestable.append(UntestableHypothesis(hid, hyp.title, hyp.category, ta.reason, request))
                if request not in requests:
                    requests.append(request)
            else:
                inconclusive.append(hid)

        if outcome == "insufficient_evidence":
            for q in inv.unresolved_questions.values():
                if q.status.value != "answered" and q.suggested_action and q.suggested_action not in requests:
                    requests.append(q.suggested_action)
            if not requests:
                requests.append(
                    "No specific missing dataset was identified from the hypothesis set"
                    + (f" ({missing_summary})" if missing_summary else "")
                    + "; ask domain owners which additional logs could distinguish the remaining hypotheses."
                )

        return Conclusion(
            outcome=outcome,
            total_observed_change=total,
            supported_causes=causes,
            explained_change=explained,
            unexplained_change=unexplained,
            unexplained_share=unexplained_share,
            ruled_out=ruled_out,
            untestable=untestable,
            inconclusive_hypothesis_ids=inconclusive,
            data_requests=requests,
            warnings=warnings,
            tests_in_family=self.tests_run,
            fdr_alpha=self.policy.alpha,
            statements=[c.statement for c in causes],
        )

    # ------------------------------------------------------------------ validation
    def validate(self) -> List[str]:
        """Perform a comprehensive integrity and ground-truth isolation audit."""
        errors: List[str] = []

        # 1. Referential integrity: evidence links
        for link in self.investigation.evidence_links:
            if link.evidence_id not in self.investigation.evidence_store:
                errors.append(f"Broken link: evidence '{link.evidence_id}' does not exist in evidence_store.")
            if link.hypothesis_id not in self.investigation.hypotheses:
                errors.append(f"Broken link: hypothesis '{link.hypothesis_id}' does not exist in hypotheses.")

        # 2. Referential integrity: hypotheses evidence lists
        for hyp_id, hyp in self.investigation.hypotheses.items():
            for ev_id in hyp.supporting_evidence_ids:
                if ev_id not in self.investigation.evidence_store:
                    errors.append(f"Hypothesis '{hyp_id}' references non-existent supporting evidence '{ev_id}'.")
            for ev_id in hyp.contradicting_evidence_ids:
                if ev_id not in self.investigation.evidence_store:
                    errors.append(f"Hypothesis '{hyp_id}' references non-existent contradicting evidence '{ev_id}'.")
            for trans in hyp.history:
                if trans.triggered_by_evidence_id and trans.triggered_by_evidence_id not in self.investigation.evidence_store:
                    errors.append(
                        f"Hypothesis '{hyp_id}' transition '{trans.transition_id}' references "
                        f"non-existent evidence '{trans.triggered_by_evidence_id}'."
                    )

        # 3. Referential integrity: unresolved questions
        for q_id, q in self.investigation.unresolved_questions.items():
            if q.hypothesis_id and q.hypothesis_id not in self.investigation.hypotheses:
                errors.append(f"Question '{q_id}' references non-existent hypothesis '{q.hypothesis_id}'.")

        # 4. Status consistency
        if self.investigation.status == InvestigationStatus.CONCLUDED:
            if not self.investigation.stopping_reason:
                errors.append("Concluded investigation must have an explicit stopping_reason.")
            if not self.investigation.confidence:
                errors.append("Concluded investigation must have a ConfidenceAssessment.")

        elif self.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE:
            if not self.investigation.stopping_reason:
                errors.append("Stopped investigation must have an explicit stopping_reason explaining the data gap.")

        # 5. Ground-truth leakage audit
        # Verify that no text contains privileged labels from synthetic generation
        all_text_blocks = [
            self.investigation.title,
            self.investigation.stopping_reason or "",
            self.investigation.observation.description,
        ]
        for hyp in self.investigation.hypotheses.values():
            all_text_blocks.extend([hyp.title, hyp.explanation])
            for t in hyp.history:
                all_text_blocks.append(t.reason)

        for ev in self.investigation.evidence_store.values():
            all_text_blocks.append(ev.finding)
            all_text_blocks.extend(ev.limitations)

        for link in self.investigation.evidence_links:
            all_text_blocks.append(link.rationale)

        for q in self.investigation.unresolved_questions.values():
            all_text_blocks.extend([q.question_text, q.context_or_limitation, q.suggested_action, q.resolution or ""])

        for marker in GROUND_TRUTH_FORBIDDEN_MARKERS:
            for text in all_text_blocks:
                if marker.lower() in text.lower():
                    errors.append(f"Ground-truth leakage detected: Found privileged marker '{marker}' in investigation text.")
                    break

        return errors

    # ------------------------------------------------------------------ internal helpers & serialization
    def _record_event(
        self,
        event_type: HistoryEventType,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEvent:
        event = HistoryEvent(
            event_id=f"evt_{len(self.investigation.history) + 1}_{uuid.uuid4().hex[:4]}",
            event_type=event_type,
            title=title,
            description=description,
            metadata=metadata or {},
            timestamp=_utc_now(),
        )
        self.investigation.history.append(event)
        self.investigation.updated_at = event.timestamp
        return event

    def to_dict(self) -> Dict[str, Any]:
        return self.investigation.to_dict()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationState":
        obs = Observation.from_dict(data["observation"])

        # Reconstruct hypotheses
        hypotheses: Dict[str, Hypothesis] = {}
        for h_id, h_data in data.get("hypotheses", {}).items():
            trans_list = [
                HypothesisTransition(
                    transition_id=t["transition_id"],
                    previous_status=HypothesisStatus(t["previous_status"]),
                    new_status=HypothesisStatus(t["new_status"]),
                    reason=t["reason"],
                    triggered_by_evidence_id=t.get("triggered_by_evidence_id"),
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                )
                for t in h_data.get("history", [])
            ]
            hypotheses[h_id] = Hypothesis(
                hypothesis_id=h_data["hypothesis_id"],
                title=h_data["title"],
                explanation=h_data["explanation"],
                category=h_data["category"],
                status=HypothesisStatus(h_data["status"]),
                supporting_evidence_ids=list(h_data.get("supporting_evidence_ids", [])),
                contradicting_evidence_ids=list(h_data.get("contradicting_evidence_ids", [])),
                history=trans_list,
                created_at=datetime.fromisoformat(h_data["created_at"]),
                updated_at=datetime.fromisoformat(h_data["updated_at"]),
                support_snapshot=dict(h_data.get("support_snapshot", {})),
            )

        # Reconstruct evidence
        evidence_store: Dict[str, Evidence] = {
            e_id: Evidence.from_dict(e_data) for e_id, e_data in data.get("evidence_store", {}).items()
        }

        # Reconstruct links
        evidence_links = [
            HypothesisEvidenceLink(
                evidence_id=l["evidence_id"],
                hypothesis_id=l["hypothesis_id"],
                stance=EvidenceStance(l["stance"]),
                rationale=l["rationale"],
                recorded_at=datetime.fromisoformat(l["recorded_at"]),
            )
            for l in data.get("evidence_links", [])
        ]

        # Reconstruct questions
        unresolved_questions: Dict[str, UnresolvedQuestion] = {}
        for q_id, q_data in data.get("unresolved_questions", {}).items():
            unresolved_questions[q_id] = UnresolvedQuestion(
                question_id=q_data["question_id"],
                question_text=q_data["question_text"],
                hypothesis_id=q_data.get("hypothesis_id"),
                status=QuestionStatus(q_data["status"]),
                context_or_limitation=q_data.get("context_or_limitation", ""),
                suggested_action=q_data.get("suggested_action", ""),
                resolution=q_data.get("resolution"),
                created_at=datetime.fromisoformat(q_data["created_at"]),
                resolved_at=datetime.fromisoformat(q_data["resolved_at"]) if q_data.get("resolved_at") else None,
            )

        # Reconstruct confidence
        conf_data = data.get("confidence")
        confidence = None
        if conf_data:
            confidence = ConfidenceAssessment(
                level=ConfidenceLevel(conf_data["level"]),
                justification=conf_data["justification"],
                competing_hypotheses_addressed=conf_data["competing_hypotheses_addressed"],
                assumptions=list(conf_data.get("assumptions", [])),
                remaining_uncertainties=list(conf_data.get("remaining_uncertainties", [])),
                assessed_at=datetime.fromisoformat(conf_data["assessed_at"]),
            )

        # Reconstruct history
        history = [
            HistoryEvent(
                event_id=ev["event_id"],
                event_type=HistoryEventType(ev["event_type"]),
                title=ev["title"],
                description=ev["description"],
                metadata=dict(ev.get("metadata", {})),
                timestamp=datetime.fromisoformat(ev["timestamp"]),
            )
            for ev in data.get("history", [])
        ]

        inv = Investigation(
            investigation_id=data["investigation_id"],
            title=data["title"],
            observation=obs,
            status=InvestigationStatus(data["status"]),
            stopping_reason=data.get("stopping_reason"),
            hypotheses=hypotheses,
            evidence_store=evidence_store,
            evidence_links=evidence_links,
            unresolved_questions=unresolved_questions,
            confidence=confidence,
            history=history,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            conclusion=Conclusion.from_dict(data["conclusion"]) if data.get("conclusion") else None,
            support_policy=dict(data.get("support_policy", {})),
            testability={k: TestabilityAssessment.from_dict(v) for k, v in data.get("testability", {}).items()},
            test_registry={k: float(v) for k, v in data.get("test_registry", {}).items()},
        )
        return cls(inv)
