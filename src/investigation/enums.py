"""Enumerations for the Investigation State and Evidence Model.

Defines explicit statuses, evidence stances, inference types, and event classifications
to preserve critical distinctions between:
- Observations vs. Explanations
- Hypotheses vs. Established findings
- Supporting vs. Contradictory evidence
- Statistical association vs. Causal inference
- Concluded investigations vs. Those stopped due to insufficient evidence
"""

from enum import Enum


class InvestigationStatus(str, Enum):
    """Lifecycle status of an investigation."""
    OPEN = "open"                                       # Initial anomaly observed, investigation starting
    IN_PROGRESS = "in_progress"                         # Actively investigating hypotheses and gathering evidence
    CONCLUDED = "concluded"                             # Reached a supported, defensible conclusion
    STOPPED_INSUFFICIENT_EVIDENCE = "stopped_insufficient_evidence"  # Halted because needed data is absent or inconclusive
    PAUSED = "paused"                                   # Paused awaiting user response or manual check


class HypothesisStatus(str, Enum):
    """Lifecycle status of a competing hypothesis."""
    PROPOSED = "proposed"         # Plausible candidate explanation formulated
    ACTIVE = "active"             # Currently under active analysis
    SUPPORTED = "supported"       # Empirical evidence substantiates this explanation
    REFUTED = "refuted"           # Contradicted by empirical evidence
    DEPRIORITIZED = "deprioritized" # Deprioritized in favor of stronger alternatives


class EvidenceStance(str, Enum):
    """The relation between an empirical finding and a hypothesis."""
    SUPPORTS = "supports"         # Empirical finding provides positive evidence for the hypothesis
    CONTRADICTS = "contradicts"   # Empirical finding provides contradictory/incompatible evidence
    INCONCLUSIVE = "inconclusive" # Empirical finding does not distinguish or is neutral


class InferenceType(str, Enum):
    """The epistemological nature of an evidence finding."""
    OBSERVATIONAL_CORRELATION = "observational_correlation"  # Metric shifted alongside event; no counterfactual
    STATISTICAL_ASSOCIATION = "statistical_association"      # Statistically significant difference; unobserved confounders possible
    MEASUREMENT_ARTIFACT = "measurement_artifact"            # Instrumentation error, tracking failure, data quality bug
    CAUSAL_INFERENCE = "causal_inference"                    # Controlled randomized experiment with confirmed covariate balance


class ConfidenceLevel(str, Enum):
    """Calibrated confidence in an investigation's conclusion or finding."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QuestionStatus(str, Enum):
    """Status of an unresolved question or missing information item."""
    OPEN = "open"
    ANSWERED = "answered"
    BLOCKED = "blocked"


class HistoryEventType(str, Enum):
    """Type of action recorded in the investigation history."""
    INVESTIGATION_TRIGGERED = "investigation_triggered"
    HYPOTHESIS_ADDED = "hypothesis_added"
    HYPOTHESIS_STATUS_UPDATED = "hypothesis_status_updated"
    ANALYSIS_DISPATCHED = "analysis_dispatched"
    EVIDENCE_RECORDED = "evidence_recorded"
    EVIDENCE_LINKED = "evidence_linked"
    DIRECTION_CHANGED = "direction_changed"
    QUESTION_ADDED = "question_added"
    QUESTION_ANSWERED = "question_answered"
    INVESTIGATION_CONCLUDED = "investigation_concluded"
    INVESTIGATION_STOPPED = "investigation_stopped"


class HypothesisCategory(str, Enum):
    """Cause families a hypothesis can belong to.

    The last two members are placeholders, not causes: a hypothesis in either must be
    refined into a specific category before it can be SUPPORTED.
    """
    CODE_REGRESSION = "code_regression"
    FEATURE_OR_EXPERIMENT_EFFECT = "feature_or_experiment_effect"
    CONFIG_OR_FLAG_CHANGE = "config_or_flag_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PRICING_OR_POLICY_CHANGE = "pricing_or_policy_change"
    TRAFFIC_MIX_SHIFT = "traffic_mix_shift"
    BOT_OR_INVALID_TRAFFIC = "bot_or_invalid_traffic"
    TRACKING_LOSS = "tracking_loss"
    PIPELINE_OR_DEFINITION_CHANGE = "pipeline_or_definition_change"
    SEASONALITY_OR_CALENDAR = "seasonality_or_calendar"
    THIRD_PARTY_OUTAGE = "third_party_outage"
    MARKET_OR_COMPETITOR = "market_or_competitor"
    GRADUAL_OR_COHORT_EFFECT = "gradual_or_cohort_effect"
    DENOMINATOR_EFFECT = "denominator_effect"
    NATURAL_VARIATION = "natural_variation"
    MULTIPLE_CAUSES = "multiple_causes"
    # Placeholders (legacy `other` / `external_factor`); never SUPPORTED directly.
    UNSPECIFIED = "unspecified"
    EXTERNAL_UNSPECIFIED = "external_unspecified"


# Only unambiguous legacy names are migrated. `other` and `external_factor` deliberately do
# NOT map to natural_variation / third_party_outage: those are specific claims.
LEGACY_CATEGORY_ALIASES = {
    "product_regression": HypothesisCategory.CODE_REGRESSION,
    "tracking_failure": HypothesisCategory.TRACKING_LOSS,
    "acquisition_shift": HypothesisCategory.TRAFFIC_MIX_SHIFT,
    "pricing_change": HypothesisCategory.PRICING_OR_POLICY_CHANGE,
    "user_behavior": HypothesisCategory.GRADUAL_OR_COHORT_EFFECT,
    "other": HypothesisCategory.UNSPECIFIED,
    "external_factor": HypothesisCategory.EXTERNAL_UNSPECIFIED,
}

PLACEHOLDER_CATEGORIES = frozenset({HypothesisCategory.UNSPECIFIED, HypothesisCategory.EXTERNAL_UNSPECIFIED})


def normalize_category(raw: str) -> HypothesisCategory:
    """Map a raw/legacy category string to a HypothesisCategory; unknown values become UNSPECIFIED."""
    clean = (raw or "").strip().lower()
    try:
        return HypothesisCategory(clean)
    except ValueError:
        return LEGACY_CATEGORY_ALIASES.get(clean, HypothesisCategory.UNSPECIFIED)
