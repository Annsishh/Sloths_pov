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
from investigation.testability import TestabilityAssessment
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


def window_to_dict(w: Optional[TimeWindow]) -> Optional[Dict[str, str]]:
    """Lossless serialisation of a TimeWindow (full ISO timestamps, not date-only)."""
    return {"start": w.start.isoformat(), "end": w.end.isoformat()} if w else None


def window_from_any(raw: Any) -> Optional[TimeWindow]:
    """Rebuild a TimeWindow from a TimeWindow, {start,end} dict, or legacy '[a – b)' string."""
    if raw is None or isinstance(raw, TimeWindow):
        return raw
    if isinstance(raw, dict):
        return TimeWindow.from_iso(raw["start"], raw["end"])
    if isinstance(raw, str) and " – " in raw:
        a, b = raw.strip("[]()").split(" – ")
        return TimeWindow.from_iso(a, b)
    raise ValueError(f"Cannot parse time window from {raw!r}")


@dataclass
class EffectEstimate:
    """A deterministic tool's estimate of how much of the observed metric change a cause accounts for.

    Only analysis tools may produce these (via result.metadata['effect_estimate']); LLM output is rejected.
    Every estimate carries its method, an uncertainty (SE and/or CI) and the window it applies to.
    """
    value: float                       # absolute change in the metric
    method: str
    window: TimeWindow
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    metric: str = ""

    def __post_init__(self) -> None:
        if not self.method or not self.method.strip():
            raise ValueError("EffectEstimate requires a method.")
        if self.window is None:
            raise ValueError("EffectEstimate requires a window.")
        if self.se is None and (self.ci_lower is None or self.ci_upper is None):
            raise ValueError("EffectEstimate requires an uncertainty: se, or both ci_lower and ci_upper.")
        if self.se is not None and self.se < 0:
            raise ValueError("EffectEstimate se must be non-negative.")

    def uncertainty_width(self) -> float:
        """Comparable spread: se, else a CI-derived se. Used to pick the most precise estimate."""
        if self.se is not None:
            return self.se
        return (self.ci_upper - self.ci_lower) / 3.92

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "method": self.method, "window": window_to_dict(self.window),
                "se": self.se, "ci_lower": self.ci_lower, "ci_upper": self.ci_upper, "metric": self.metric}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EffectEstimate":
        return cls(value=float(d["value"]), method=d.get("method", ""), window=window_from_any(d.get("window")),
                   se=d.get("se"), ci_lower=d.get("ci_lower"), ci_upper=d.get("ci_upper"), metric=d.get("metric", ""))


def extract_tests(result: AnalysisResult) -> Dict[str, float]:
    """Labelled RAW p-values reported by a tool (metadata `p_values` dict and/or scalar `p_value`)."""
    out: Dict[str, float] = {}
    meta = result.metadata if isinstance(result.metadata, dict) else {}
    for label, p in (meta.get("p_values") or {}).items():
        if p is not None:
            out[str(label)] = float(p)
    if meta.get("p_value") is not None:
        out.setdefault("overall", float(meta["p_value"]))
    return out


def extract_effect_estimate(result: AnalysisResult) -> Optional[EffectEstimate]:
    meta = result.metadata if isinstance(result.metadata, dict) else {}
    raw = meta.get("effect_estimate")
    return EffectEstimate.from_dict(raw) if raw else None


def extract_p_value(result: AnalysisResult) -> Optional[float]:
    """Scalar `p_value` from a tool's metadata, if it reports exactly one test that way."""
    meta = result.metadata.get("p_value") if isinstance(result.metadata, dict) else None
    return float(meta) if meta is not None else None


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
            "baseline_window": window_to_dict(self.baseline_window),
            "comparison_window": window_to_dict(self.comparison_window),
            "baseline_value": self.baseline_value,
            "comparison_value": self.comparison_value,
            "absolute_change": self.absolute_change,
            "relative_change": self.relative_change,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        bw_obj = window_from_any(data["baseline_window"])
        cw_obj = window_from_any(data["comparison_window"])

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
    # Raw (uncorrected) p-value of a single test behind this evidence, if any.
    p_value: Optional[float] = None
    # Labelled raw p-values when a tool runs several tests (e.g. one per segment); each joins the BH family.
    p_values: Dict[str, float] = field(default_factory=dict)
    # Smallest effect the tests could have detected (80% power), in the analysis metric's units.
    mde: Optional[float] = None
    # "computed" from data, or "caller_supplied" (unverifiable inputs; excluded from BH family and support).
    p_value_provenance: str = "computed"
    # Deterministic tool estimate of the share of the observed change this evidence explains.
    effect_estimate: Optional[EffectEstimate] = None

    def all_tests(self) -> Dict[str, float]:
        out = dict(self.p_values)
        if self.p_value is not None:
            out.setdefault("overall", self.p_value)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "finding": self.finding,
            "source_tables": list(self.source_tables),
            "analysis_request_id": self.analysis_request_id,
            "operation": self.operation,
            "population": dict(self.population),
            "window": window_to_dict(self.window),
            "result_summary": dict(self.result_summary),
            "limitations": list(self.limitations),
            "inference_type": self.inference_type.value,
            "recorded_at": self.recorded_at.isoformat(),
            "p_value": self.p_value,
            "p_values": dict(self.p_values),
            "mde": self.mde,
            "p_value_provenance": self.p_value_provenance,
            "effect_estimate": self.effect_estimate.to_dict() if self.effect_estimate else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(
            evidence_id=d["evidence_id"],
            finding=d["finding"],
            source_tables=list(d.get("source_tables", [])),
            analysis_request_id=d.get("analysis_request_id"),
            operation=d.get("operation"),
            population=dict(d.get("population", {})),
            window=window_from_any(d.get("window")),
            result_summary=dict(d.get("result_summary", {})),
            limitations=list(d.get("limitations", [])),
            inference_type=InferenceType(d.get("inference_type", InferenceType.OBSERVATIONAL_CORRELATION.value)),
            recorded_at=datetime.fromisoformat(d["recorded_at"]),
            p_value=d.get("p_value"),
            p_values=dict(d.get("p_values", {})),
            mde=d.get("mde"),
            p_value_provenance=d.get("p_value_provenance", "computed"),
            effect_estimate=EffectEstimate.from_dict(d["effect_estimate"]) if d.get("effect_estimate") else None,
        )

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
            p_value=extract_p_value(result),
            p_values=extract_tests(result),
            mde=(float(result.metadata["mde"]) if isinstance(result.metadata, dict) and result.metadata.get("mde") is not None else None),
            p_value_provenance=(result.metadata.get("p_value_provenance", "computed") if isinstance(result.metadata, dict) else "computed"),
            effect_estimate=extract_effect_estimate(result),
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
    # BH-adjusted p-values of the supporting evidence at the moment SUPPORTED was granted (audit "before" values).
    support_snapshot: Dict[str, float] = field(default_factory=dict)

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
            "support_snapshot": dict(self.support_snapshot),
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
class ExplainedShare:
    """Share of the observed metric change attributed to one supported cause."""
    hypothesis_id: str
    category: str
    estimated_change: Optional[float]   # absolute change in the observed metric; None = not quantified
    share: Optional[float]              # estimated_change / total observed change
    basis_evidence_ids: List[str] = field(default_factory=list)
    note: str = ""
    attribution: str = "consistent_with"  # "confirmed" | "consistent_with"
    method: str = ""
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    window: Optional[TimeWindow] = None
    statement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "category": self.category,
                "estimated_change": self.estimated_change, "share": self.share,
                "basis_evidence_ids": list(self.basis_evidence_ids), "note": self.note,
                "attribution": self.attribution, "method": self.method, "se": self.se,
                "ci_lower": self.ci_lower, "ci_upper": self.ci_upper,
                "window": window_to_dict(self.window), "statement": self.statement}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplainedShare":
        return cls(d["hypothesis_id"], d["category"], d.get("estimated_change"), d.get("share"),
                   list(d.get("basis_evidence_ids", [])), d.get("note", ""), d.get("attribution", "consistent_with"),
                   d.get("method", ""), d.get("se"), d.get("ci_lower"), d.get("ci_upper"),
                   window_from_any(d.get("window")), d.get("statement", ""))


@dataclass
class RuledOutHypothesis:
    hypothesis_id: str
    title: str
    category: str
    reason: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "title": self.title, "category": self.category,
                "reason": self.reason, "evidence_ids": list(self.evidence_ids)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuledOutHypothesis":
        return cls(d["hypothesis_id"], d["title"], d["category"], d["reason"], list(d.get("evidence_ids", [])))


@dataclass
class UntestableHypothesis:
    hypothesis_id: str
    title: str
    category: str
    reason: str
    data_request: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "title": self.title, "category": self.category,
                "reason": self.reason, "data_request": self.data_request}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UntestableHypothesis":
        return cls(d["hypothesis_id"], d["title"], d["category"], d["reason"], d.get("data_request", ""))


@dataclass
class Conclusion:
    """Final structured outcome: supported causes with explained shares, the unexplained
    remainder, and (when evidence is insufficient) what was ruled out, what could not be
    tested, and what data to request."""
    outcome: str                                  # "concluded" | "insufficient_evidence"
    total_observed_change: float
    supported_causes: List[ExplainedShare] = field(default_factory=list)
    explained_change: float = 0.0                 # sum of quantified estimates
    unexplained_change: float = 0.0               # total - explained (signed)
    unexplained_share: Optional[float] = None
    ruled_out: List[RuledOutHypothesis] = field(default_factory=list)
    untestable: List[UntestableHypothesis] = field(default_factory=list)
    inconclusive_hypothesis_ids: List[str] = field(default_factory=list)
    data_requests: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tests_in_family: int = 0                      # tests in the investigation-wide BH family
    fdr_alpha: float = 0.05
    statements: List[str] = field(default_factory=list)   # deterministic, hedged wording per supported cause

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "total_observed_change": self.total_observed_change,
            "supported_causes": [c.to_dict() for c in self.supported_causes],
            "explained_change": self.explained_change,
            "unexplained_change": self.unexplained_change,
            "unexplained_share": self.unexplained_share,
            "ruled_out": [r.to_dict() for r in self.ruled_out],
            "untestable": [u.to_dict() for u in self.untestable],
            "inconclusive_hypothesis_ids": list(self.inconclusive_hypothesis_ids),
            "data_requests": list(self.data_requests),
            "warnings": list(self.warnings),
            "tests_in_family": self.tests_in_family,
            "fdr_alpha": self.fdr_alpha,
            "statements": list(self.statements),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Conclusion":
        return cls(
            outcome=d["outcome"],
            total_observed_change=d["total_observed_change"],
            supported_causes=[ExplainedShare.from_dict(x) for x in d.get("supported_causes", [])],
            explained_change=d.get("explained_change", 0.0),
            unexplained_change=d.get("unexplained_change", 0.0),
            unexplained_share=d.get("unexplained_share"),
            ruled_out=[RuledOutHypothesis.from_dict(x) for x in d.get("ruled_out", [])],
            untestable=[UntestableHypothesis.from_dict(x) for x in d.get("untestable", [])],
            inconclusive_hypothesis_ids=list(d.get("inconclusive_hypothesis_ids", [])),
            data_requests=list(d.get("data_requests", [])),
            warnings=list(d.get("warnings", [])),
            tests_in_family=d.get("tests_in_family", 0),
            fdr_alpha=d.get("fdr_alpha", 0.05),
            statements=list(d.get("statements", [])),
        )


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
    conclusion: Optional[Conclusion] = None
    support_policy: Dict[str, Any] = field(default_factory=dict)
    testability: Dict[str, TestabilityAssessment] = field(default_factory=dict)
    # Every statistical test run in this investigation, "<source>|<label>" -> raw p-value. This is the BH family.
    test_registry: Dict[str, float] = field(default_factory=dict)

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
            "conclusion": self.conclusion.to_dict() if self.conclusion else None,
            "support_policy": dict(self.support_policy),
            "testability": {k: v.to_dict() for k, v in self.testability.items()},
            "test_registry": dict(self.test_registry),
        }
