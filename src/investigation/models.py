"""Core data models for the Investigation State and Evidence Model.

These models provide explicit, typed, validated structures for representing:
- The triggering metric observation
- Competing hypotheses and their audit histories
- Empirical evidence records with provenance and limitations
- Disentangled links between evidence findings and hypothesis interpretations
- Unresolved questions and data gaps
- Calibrated confidence assessments
- Chronological history of investigation decisions and pivots
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Observation:
    """The empirical metric deviation that triggered the investigation.
    
    Distinguishes the triggering observation from candidate explanatory hypotheses.
    """
    metric_name: str
    baseline_window: TimeWindow
    comparison_window: TimeWindow
    baseline_value: float
    comparison_value: float
    absolute_change: float
    relative_change: Optional[float]
    description: str
    timestamp: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_window": str(self.baseline_window),
            "comparison_window": str(self.comparison_window),
            "baseline_value": self.baseline_value,
            "comparison_value": self.comparison_value,
            "absolute_change": self.absolute_change,
            "relative_change": self.relative_change,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        # Supports reconstructing TimeWindow if ISO strings or objects are passed
        bw = data["baseline_window"]
        cw = data["comparison_window"]
        if isinstance(bw, str) and " – " in bw:
            # parse string format e.g. [2026-09-01 – 2026-09-10)
            parts = bw.strip("[]()").split(" – ")
            bw_obj = TimeWindow.from_iso(parts[0], parts[1])
        elif isinstance(bw, dict):
            bw_obj = TimeWindow.from_iso(bw["start"], bw["end"])
        else:
            bw_obj = bw

        if isinstance(cw, str) and " – " in cw:
            parts = cw.strip("[]()").split(" – ")
            cw_obj = TimeWindow.from_iso(parts[0], parts[1])
        elif isinstance(cw, dict):
            cw_obj = TimeWindow.from_iso(cw["start"], cw["end"])
        else:
            cw_obj = cw

        ts = data.get("timestamp")
        ts_dt = datetime.fromisoformat(ts).astimezone(timezone.utc) if isinstance(ts, str) else _utc_now()

        return cls(
            metric_name=data["metric_name"],
            baseline_window=bw_obj,
            comparison_window=cw_obj,
            baseline_value=float(data["baseline_value"]),
            comparison_value=float(data["comparison_value"]),
            absolute_change=float(data["absolute_change"]),
            relative_change=float(data["relative_change"]) if data.get("relative_change") is not None else None,
            description=data.get("description", ""),
            timestamp=ts_dt,
        )


@dataclass
class Evidence:
    """An objective empirical finding from an analytical operation or data check.
    
    Holds factual results and provenance, strictly distinct from hypotheses
    and interpretations.
    """
    evidence_id: str
    finding: str
    source_tables: List[str]
    analysis_request_id: Optional[str] = None
    operation: Optional[str] = None
    population: Dict[str, Any] = field(default_factory=dict)
    window: Optional[TimeWindow] = None
    result_summary: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    inference_type: InferenceType = InferenceType.OBSERVATIONAL_CORRELATION
    recorded_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "finding": self.finding,
            "source_tables": list(self.source_tables),
            "analysis_request_id": self.analysis_request_id,
            "operation": self.operation,
            "population": dict(self.population),
            "window": str(self.window) if self.window else None,
            "result_summary": dict(self.result_summary),
            "limitations": list(self.limitations),
            "inference_type": self.inference_type.value,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_analysis_result(
        cls,
        evidence_id: str,
        finding: str,
        result: AnalysisResult,
        source_tables: List[str],
        population: Optional[Dict[str, Any]] = None,
        window: Optional[TimeWindow] = None,
        inference_type: InferenceType = InferenceType.OBSERVATIONAL_CORRELATION,
    ) -> "Evidence":
        """Construct an Evidence item directly from a verified AnalysisResult."""
        summary_data: Dict[str, Any] = {}
        if isinstance(result.data, dict):
            summary_data = result.data
        elif hasattr(result.data, "to_dict"):
            # e.g. DataFrame
            summary_data = {"rows": len(result.data)}
            if "metric_value" in result.data.columns:
                summary_data["metric_values"] = result.data["metric_value"].dropna().tolist()

        return cls(
            evidence_id=evidence_id,
            finding=finding,
            source_tables=source_tables,
            analysis_request_id=result.request_id,
            operation=result.operation,
            population=population or {},
            window=window,
            result_summary=summary_data,
            limitations=list(result.limitations),
            inference_type=inference_type,
            recorded_at=_utc_now(),
        )


@dataclass
class HypothesisEvidenceLink:
    """Explicit relationship connecting a piece of Evidence to a Hypothesis.
    
    Separates the empirical finding (Evidence) from its investigative
    interpretation (HypothesisEvidenceLink.rationale). Allows a single evidence
    item to simultaneously support one hypothesis and contradict another.
    """
    evidence_id: str
    hypothesis_id: str
    stance: EvidenceStance
    rationale: str
    recorded_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "hypothesis_id": self.hypothesis_id,
            "stance": self.stance.value,
            "rationale": self.rationale,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass
class HypothesisTransition:
    """Historical record of a status transition for a hypothesis."""
    transition_id: str
    previous_status: HypothesisStatus
    new_status: HypothesisStatus
    reason: str
    triggered_by_evidence_id: Optional[str] = None
    timestamp: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "previous_status": self.previous_status.value,
            "new_status": self.new_status.value,
            "reason": self.reason,
            "triggered_by_evidence_id": self.triggered_by_evidence_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Hypothesis:
    """A candidate explanation for the observed anomaly.
    
    Tracks its full lifecycle, supporting and contradicting evidence references,
    and status transition history.
    """
    hypothesis_id: str
    title: str
    explanation: str
    category: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    history: List[HypothesisTransition] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "explanation": self.explanation,
            "category": self.category,
            "status": self.status.value,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "history": [t.to_dict() for t in self.history],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class UnresolvedQuestion:
    """A specific information gap, pending verification, or hypothesis differentiator."""
    question_id: str
    question_text: str
    hypothesis_id: Optional[str] = None
    status: QuestionStatus = QuestionStatus.OPEN
    context_or_limitation: str = ""
    suggested_action: str = ""
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=_utc_now)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "hypothesis_id": self.hypothesis_id,
            "status": self.status.value,
            "context_or_limitation": self.context_or_limitation,
            "suggested_action": self.suggested_action,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class ConfidenceAssessment:
    """Calibrated uncertainty evaluation for an investigation finding or conclusion."""
    level: ConfidenceLevel
    justification: str
    competing_hypotheses_addressed: bool
    assumptions: List[str] = field(default_factory=list)
    remaining_uncertainties: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "justification": self.justification,
            "competing_hypotheses_addressed": self.competing_hypotheses_addressed,
            "assumptions": list(self.assumptions),
            "remaining_uncertainties": list(self.remaining_uncertainties),
            "assessed_at": self.assessed_at.isoformat(),
        }


@dataclass
class HistoryEvent:
    """An immutable audit record in the investigation's chronological log."""
    event_id: str
    event_type: HistoryEventType
    title: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Investigation:
    """The aggregate root representing an active or concluded product investigation.
    
    Encapsulates the full state: triggering observation, competing hypotheses,
    empirical evidence store, evidence-hypothesis links, unresolved questions,
    calibrated confidence, and chronological event history.
    """
    investigation_id: str
    title: str
    observation: Observation
    status: InvestigationStatus = InvestigationStatus.OPEN
    stopping_reason: Optional[str] = None
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    evidence_store: Dict[str, Evidence] = field(default_factory=dict)
    evidence_links: List[HypothesisEvidenceLink] = field(default_factory=list)
    unresolved_questions: Dict[str, UnresolvedQuestion] = field(default_factory=dict)
    confidence: Optional[ConfidenceAssessment] = None
    history: List[HistoryEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "title": self.title,
            "status": self.status.value,
            "stopping_reason": self.stopping_reason,
            "observation": self.observation.to_dict(),
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            "evidence_store": {k: v.to_dict() for k, v in self.evidence_store.items()},
            "evidence_links": [link.to_dict() for link in self.evidence_links],
            "unresolved_questions": {k: v.to_dict() for k, v in self.unresolved_questions.items()},
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "history": [ev.to_dict() for ev in self.history],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
