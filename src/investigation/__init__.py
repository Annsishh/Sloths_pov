"""Investigation State and Evidence Model Package.

Exports:
- Enums:
    InvestigationStatus, HypothesisStatus, EvidenceStance,
    InferenceType, ConfidenceLevel, QuestionStatus, HistoryEventType
- Models:
    Observation, Evidence, HypothesisEvidenceLink, HypothesisTransition,
    Hypothesis, UnresolvedQuestion, ConfidenceAssessment, HistoryEvent,
    Investigation
- State Management:
    InvestigationState
"""

from investigation.enums import (
    HypothesisCategory,
    LEGACY_CATEGORY_ALIASES,
    normalize_category,
    ConfidenceLevel,
    EvidenceStance,
    HistoryEventType,
    HypothesisStatus,
    InferenceType,
    InvestigationStatus,
    QuestionStatus,
)
from investigation.models import (
    Conclusion,
    ExplainedShare,
    RuledOutHypothesis,
    UntestableHypothesis,
    ConfidenceAssessment,
    Evidence,
    HistoryEvent,
    Hypothesis,
    HypothesisEvidenceLink,
    HypothesisTransition,
    Investigation,
    Observation,
    UnresolvedQuestion,
)
from investigation.state import InvestigationState, SupportRuleViolation
from investigation.support import SupportDecision, SupportPolicy
from investigation.testability import TestabilityAssessment, assess_testability

__all__ = [
    # Enums
    "InvestigationStatus",
    "HypothesisStatus",
    "EvidenceStance",
    "InferenceType",
    "ConfidenceLevel",
    "QuestionStatus",
    "HistoryEventType",
    # Models
    "Observation",
    "Evidence",
    "HypothesisEvidenceLink",
    "HypothesisTransition",
    "Hypothesis",
    "UnresolvedQuestion",
    "ConfidenceAssessment",
    "HistoryEvent",
    "Investigation",
    # State
    "InvestigationState",
    "SupportRuleViolation",
    # Phase 1
    "HypothesisCategory", "LEGACY_CATEGORY_ALIASES", "normalize_category",
    "Conclusion", "ExplainedShare", "RuledOutHypothesis", "UntestableHypothesis",
    "SupportPolicy", "SupportDecision", "TestabilityAssessment", "assess_testability",
]
