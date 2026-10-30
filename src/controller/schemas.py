"""Pydantic schemas for validated structured LLM outputs in the investigation loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from investigation.enums import HypothesisCategory, normalize_category


FORBIDDEN_MODEL_KEYS = frozenset({
    "estimated_effect", "effect_estimate", "explained_share", "explained_change", "share", "estimated_change",
})


class HypothesisProposal(BaseModel):
    """A proposed candidate explanation for an observed metric anomaly."""
    hypothesis_id: Optional[str] = Field(default=None, description="Optional identifier, e.g. hyp_1, hyp_2")
    title: str = Field(..., min_length=5, description="Clear, concise hypothesis title")
    explanation: str = Field(..., min_length=15, description="Plausible causal mechanism explaining the metric change")
    category: str = Field(
        default="unspecified",
        description=(
            "Cause family, one of: " + " | ".join(c.value for c in HypothesisCategory)
            + ". Use unspecified / external_unspecified only if unsure; they must be refined before the hypothesis can be supported."
        ),
    )
    suggested_analysis: str = Field(
        default="",
        description="Proposed analytical operation or check to evaluate this hypothesis"
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Known and legacy names are normalised. Unknown names are kept (lower-cased) so that the investigation
        state can coerce them to 'unspecified' AND log a warning audit event; coercing here would hide it."""
        clean = (v or "").strip().lower()
        norm = normalize_category(clean)
        if norm == HypothesisCategory.UNSPECIFIED and clean not in ("unspecified", "other", ""):
            return clean
        return norm.value


class HypothesesGenerationOutput(BaseModel):
    """Validated structured output from the LLM when proposing hypotheses."""
    hypotheses: List[HypothesisProposal] = Field(
        ...,
        min_length=2,
        description="At least two distinct, competing candidate hypotheses"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Initial reasoning considering multiple possible explanations"
    )


class AnalysisDecision(BaseModel):
    """Validated decision specifying an analytical operation to execute."""
    hypothesis_id_to_test: str = Field(..., description="ID of the hypothesis this analysis intends to evaluate")
    operation: str = Field(..., description="Analytical operation name from the controlled capability catalogue")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to the operation")
    dimensions: List[str] = Field(default_factory=list, description="Dimensions to segment by")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
    comparison_window_needed: bool = Field(
        default=False,
        description="Whether to supply the observation comparison window to the analysis"
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Why this analysis was chosen and what expected outcome would support or refute the hypothesis"
    )


class EvidenceInterpretation(BaseModel):
    """Validated structured output interpreting an actual executed AnalysisResult."""
    hypothesis_id: str = Field(..., description="ID of the hypothesis evaluated")
    stance: str = Field(
        ...,
        description="Relation to hypothesis: supports | contradicts | inconclusive"
    )
    finding_summary: str = Field(
        ...,
        min_length=10,
        description="Empirical finding quoting actual tool-returned metrics, not invented numbers"
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Detailed explanation of why the evidence supports or contradicts the hypothesis"
    )
    inference_type: str = Field(
        default="observational_correlation",
        description="observational_correlation | statistical_association | measurement_artifact | causal_inference"
    )
    recommended_hypothesis_status: Optional[str] = Field(
        default=None,
        description="supported | refuted | active | deprioritized | none"
    )
    refined_category: Optional[str] = Field(
        default=None,
        description="If the hypothesis category is unspecified/external_unspecified (or wrong), the specific cause family it belongs to",
    )
    status_update_reason: Optional[str] = Field(
        default=None,
        description="Explicit evidentiary justification if recommending a status transition"
    )
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="New open questions or unobserved information gaps identified from this result"
    )
    is_investigation_complete: bool = Field(
        default=False,
        description="True if evidence is sufficient to conclude or if investigation cannot proceed further"
    )
    stopping_reason: Optional[str] = Field(
        default=None,
        description="Conclusion summary or explanation of why investigation must halt"
    )
    confidence_level: Optional[str] = Field(
        default=None,
        description="Confidence in current conclusion: very_low | low | medium | high"
    )
    confidence_justification: Optional[str] = Field(
        default=None,
        description="Explicit explanation of confidence level, addressed alternatives, and remaining limitations"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_model_supplied_effects(cls, data: Any) -> Any:
        """Effect estimates / explained shares may come only from deterministic tools, never from the model."""
        if isinstance(data, dict):
            bad = sorted(k for k in data if k in FORBIDDEN_MODEL_KEYS)
            if bad:
                raise ValueError(
                    f"LLM output may not set {bad}: effect estimates and explained shares come only from "
                    "deterministic analysis tools."
                )
        return data

    @field_validator("stance")
    @classmethod
    def validate_stance(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in {"supports", "contradicts", "inconclusive"}:
            raise ValueError(f"Invalid stance '{v}'. Must be 'supports', 'contradicts', or 'inconclusive'.")
        return clean

    @field_validator("inference_type")
    @classmethod
    def validate_inference_type(cls, v: str) -> str:
        clean = v.strip().lower()
        allowed = {
            "observational_correlation",
            "statistical_association",
            "measurement_artifact",
            "causal_inference",
        }
        return clean if clean in allowed else "observational_correlation"
