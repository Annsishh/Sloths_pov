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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from analytics.models import AnalysisResult, TimeWindow
from investigation.enums import (
    ConfidenceLevel,
    EvidenceStance,
    HistoryEventType,
    HypothesisStatus,
    InferenceType,
    InvestigationStatus,
    QuestionStatus,
)
from investigation.models import (
    ConfidenceAssessment,
    Evidence,
    HistoryEvent,
    Hypothesis,
    HypothesisEvidenceLink,
    HypothesisTransition,
    Investigation,
    Observation,
    UnresolvedQuestion,
    _utc_now,
)


# Known privileged terms that must never leak from synthetic ground truth into the investigation state
GROUND_TRUTH_FORBIDDEN_MARKERS = [
    "scenario_ground_truth",
    "ground_truth.json",
    "injected_parameters",
    "root_cause_label",
    "expected_scenario_outcome",
]


class InvestigationState:
    """Manages the lifecycle and state transitions of an active or concluded Investigation."""

    def __init__(self, investigation: Investigation) -> None:
        self.investigation = investigation

    # ------------------------------------------------------------------ factory
    @classmethod
    def create(
        cls,
        title: str,
        observation: Observation,
        investigation_id: Optional[str] = None,
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
        hyp = Hypothesis(
            hypothesis_id=hyp_id,
            title=title.strip(),
            explanation=explanation.strip(),
            category=category.strip(),
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
            metadata={"hypothesis_id": hyp_id, "category": category},
        )
        return hyp

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

        now = _utc_now()
        transition = HypothesisTransition(
            transition_id=f"trans_{uuid.uuid4().hex[:8]}",
            previous_status=hyp.status,
            new_status=new_status,
            reason=reason.strip(),
            triggered_by_evidence_id=triggered_by_evidence_id,
            timestamp=now,
        )

        hyp.history.append(transition)
        hyp.status = new_status
        hyp.updated_at = now
        self.investigation.updated_at = now

        self._record_event(
            event_type=HistoryEventType.HYPOTHESIS_STATUS_UPDATED,
            title=f"Hypothesis Status Changed: {hyp.title} -> {new_status.value}",
            description=reason,
            metadata={
                "hypothesis_id": hypothesis_id,
                "previous_status": transition.previous_status.value,
                "new_status": new_status.value,
                "triggered_by_evidence_id": triggered_by_evidence_id,
            },
        )
        return hyp

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
            )

        self.investigation.evidence_store[ev_id] = ev
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
    def conclude(
        self,
        conclusion_summary: str,
        supported_hypothesis_id: str,
        confidence_level: ConfidenceLevel,
        confidence_justification: str,
        assumptions: Optional[List[str]] = None,
        remaining_uncertainties: Optional[List[str]] = None,
    ) -> None:
        """Conclude the investigation with a substantiated conclusion."""
        if not conclusion_summary or not conclusion_summary.strip():
            raise ValueError("Conclusion summary cannot be empty.")
        if supported_hypothesis_id not in self.investigation.hypotheses:
            raise KeyError(f"Supported hypothesis '{supported_hypothesis_id}' not found.")

        hyp = self.investigation.hypotheses[supported_hypothesis_id]
        if hyp.status != HypothesisStatus.SUPPORTED:
            raise ValueError(
                f"Cannot conclude with hypothesis '{supported_hypothesis_id}' because its status "
                f"is '{hyp.status.value}', not 'supported'."
            )
        if not hyp.supporting_evidence_ids:
            raise ValueError(
                f"Cannot conclude with hypothesis '{supported_hypothesis_id}' because it has no supporting evidence."
            )

        now = _utc_now()
        self.investigation.status = InvestigationStatus.CONCLUDED
        self.investigation.stopping_reason = conclusion_summary.strip()
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
                "confidence_level": confidence_level.value,
            },
        )

    def stop_insufficient_evidence(
        self,
        reason: str,
        missing_information_summary: str,
        confidence_justification: str,
        remaining_uncertainties: Optional[List[str]] = None,
    ) -> None:
        """Halt the investigation because essential data is missing, unobserved, or inconclusive."""
        if not reason or not reason.strip():
            raise ValueError("Stopping reason cannot be empty.")
        if not missing_information_summary or not missing_information_summary.strip():
            raise ValueError("Must specify what missing information prevented further investigation.")

        now = _utc_now()
        self.investigation.status = InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
        self.investigation.stopping_reason = f"{reason.strip()} [Missing: {missing_information_summary.strip()}]"
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
            metadata={"missing_information": missing_information_summary},
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
        return json.dumps(self.to_dict(), indent=indent)

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
            )

        # Reconstruct evidence
        evidence_store: Dict[str, Evidence] = {}
        for e_id, e_data in data.get("evidence_store", {}).items():
            evidence_store[e_id] = Evidence(
                evidence_id=e_data["evidence_id"],
                finding=e_data["finding"],
                source_tables=list(e_data.get("source_tables", [])),
                analysis_request_id=e_data.get("analysis_request_id"),
                operation=e_data.get("operation"),
                population=dict(e_data.get("population", {})),
                result_summary=dict(e_data.get("result_summary", {})),
                limitations=list(e_data.get("limitations", [])),
                inference_type=InferenceType(e_data.get("inference_type", InferenceType.OBSERVATIONAL_CORRELATION.value)),
                recorded_at=datetime.fromisoformat(e_data["recorded_at"]),
            )

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
        )
        return cls(inv)
