"""Deterministic rule deciding whether a hypothesis may be marked SUPPORTED.

The LLM may *recommend* a status; this module decides. Defaults:
  * the hypothesis category is testable with the available tools and is not a placeholder;
  * at least `min_supporting_evidence` supporting items, from at least `min_distinct_analyses`
    different analysis operations;
  * at most `max_contradicting_evidence` contradicting items (default zero);
  * at least `min_significant_supporting` supporting items significant after
    Benjamini–Hochberg correction across every test in the investigation (alpha configurable).
`natural_variation` is the mirror image: it needs p-valued evidence that is NOT significant,
and it excludes (and is excluded by) any other supported cause.
All thresholds live in SupportPolicy so they can be configured per run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from analytics.multiple_testing import benjamini_hochberg
from investigation.enums import (
    PLACEHOLDER_CATEGORIES,
    HypothesisCategory,
    HypothesisStatus,
    normalize_category,
)
from investigation.models import Evidence, Hypothesis, Investigation
from investigation.testability import TestabilityAssessment


@dataclass(frozen=True)
class SupportPolicy:
    min_supporting_evidence: int = 2
    min_distinct_analyses: int = 2
    max_contradicting_evidence: int = 0
    min_significant_supporting: int = 1
    alpha: float = 0.05
    correction: str = "bh"  # "bh" | "none"

    def __post_init__(self) -> None:
        if self.correction not in ("bh", "none"):
            raise ValueError("correction must be 'bh' or 'none'")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "SupportPolicy":
        return cls(**d) if d else cls()


@dataclass
class SupportDecision:
    hypothesis_id: str
    eligible: bool
    reasons: List[str] = field(default_factory=list)  # why NOT eligible (empty when eligible)
    needs_more_evidence: bool = False  # True when the only blockers are evidence thresholds


def adjusted_p_values(investigation: Investigation, policy: SupportPolicy) -> Dict[str, float]:
    """evidence_id -> adjusted p-value over every p-valued evidence item in the investigation."""
    items = [(e.evidence_id, e.p_value) for e in investigation.evidence_store.values() if e.p_value is not None]
    if not items:
        return {}
    ps = [p for _, p in items]
    adj = benjamini_hochberg(ps) if policy.correction == "bh" else ps
    return {eid: q for (eid, _), q in zip(items, adj)}


def evaluate_support(
    investigation: Investigation,
    hypothesis_id: str,
    policy: SupportPolicy,
    testability: Mapping[str, TestabilityAssessment],
) -> SupportDecision:
    hyp: Hypothesis = investigation.hypotheses[hypothesis_id]
    cat = normalize_category(hyp.category)
    reasons: List[str] = []  # hard blockers: more evidence cannot fix these
    soft: List[str] = []  # evidence-threshold shortfalls: more analyses could fix these

    if cat in PLACEHOLDER_CATEGORIES:
        reasons.append(f"category '{cat.value}' is a placeholder; refine it into a specific cause family first")
    elif cat == HypothesisCategory.MULTIPLE_CAUSES:
        reasons.append("'multiple_causes' is composite; support each constituent cause instead")
    else:
        ta = testability.get(cat.value)
        if ta is not None and not ta.testable:
            reasons.append(f"category '{cat.value}' is UNTESTABLE with current tools/data ({ta.reason})")
    if reasons:
        return SupportDecision(hypothesis_id, False, reasons)

    supporting: List[Evidence] = [
        investigation.evidence_store[i] for i in hyp.supporting_evidence_ids if i in investigation.evidence_store
    ]
    adj = adjusted_p_values(investigation, policy)

    if len(supporting) < policy.min_supporting_evidence:
        soft.append(f"{len(supporting)} supporting evidence item(s); need {policy.min_supporting_evidence}")
    analyses = {e.operation or e.evidence_id for e in supporting}
    if len(analyses) < policy.min_distinct_analyses:
        soft.append(f"supporting evidence comes from {len(analyses)} distinct analysis type(s); need {policy.min_distinct_analyses}")
    if len(hyp.contradicting_evidence_ids) > policy.max_contradicting_evidence:
        reasons.append(f"{len(hyp.contradicting_evidence_ids)} contradicting evidence item(s); at most {policy.max_contradicting_evidence} allowed")

    p_valued = [e for e in supporting if e.evidence_id in adj]
    if cat == HypothesisCategory.NATURAL_VARIATION:
        if not p_valued:
            soft.append("natural_variation needs at least one p-valued supporting analysis showing the change is not significant")
        elif any(adj[e.evidence_id] < policy.alpha for e in p_valued):
            reasons.append("a supporting analysis shows a statistically significant change, which contradicts natural variation")
        others = [h.hypothesis_id for h in investigation.hypotheses.values()
                  if h.hypothesis_id != hypothesis_id and h.status == HypothesisStatus.SUPPORTED]
        if others:
            reasons.append(f"cannot be supported alongside supported cause(s): {others}")
    else:
        n_sig = sum(1 for e in p_valued if adj[e.evidence_id] < policy.alpha)
        if n_sig < policy.min_significant_supporting:
            soft.append(
                f"{n_sig} supporting item(s) significant at alpha={policy.alpha} after "
                f"{'BH correction' if policy.correction == 'bh' else 'no correction'}; need {policy.min_significant_supporting}"
            )
        nv = [h.hypothesis_id for h in investigation.hypotheses.values()
              if h.status == HypothesisStatus.SUPPORTED and normalize_category(h.category) == HypothesisCategory.NATURAL_VARIATION]
        if nv:
            reasons.append(f"cannot be supported alongside supported natural_variation hypothesis {nv}")

    all_reasons = reasons + soft
    return SupportDecision(hypothesis_id, not all_reasons, all_reasons, needs_more_evidence=bool(soft) and not reasons)
