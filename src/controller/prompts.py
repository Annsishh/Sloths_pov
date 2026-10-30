"""Prompt construction for the LLM Investigation Controller.

Enforces strict analytical objectivity, ground-truth isolation, provenance requirements,
and calibrated uncertainty.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from analytics.models import AnalysisResult
from controller.capability_catalogue import describe_capabilities
from investigation.enums import HypothesisCategory
from investigation.models import Evidence, Hypothesis, Observation


SYSTEM_INVESTIGATION_PROMPT = """You are an AI Product Investigator: a rigorous, evidence-driven system that investigates product metric deviations.

Core Behavioral Principles:
1. Ground Truth Isolation: You are investigating real telemetry and backend tables. Never speculate about synthetic scenario parameters or injected labels.
2. Evidence Integrity: You MUST NEVER invent numbers, fabricate metrics, or claim an analysis succeeded when it failed. Quote only real tool-returned values.
3. Plausibility is Not Evidence: An explanation is only a hypothesis until backed by empirical findings.
4. Competing Explanations: Always consider multiple plausible competing explanations (e.g., instrumentation bug vs. real product regression vs. mix shift vs. external factor).
5. Association vs. Causation: Statistical significance (p < 0.05) or temporal correlation does not prove causality. Respect stated limitations.
6. Honest Uncertainty: If data is missing or inconclusive, explicitly state that evidence is insufficient rather than guessing.
"""


def format_hypotheses_prompt(observation: Observation, testability: Optional[Dict[str, Any]] = None) -> str:
    """Prompt asking the LLM to generate multiple competing hypotheses for an anomaly."""
    rel_str = f" ({observation.relative_change:+.1%})" if observation.relative_change is not None else ""
    cat_lines = []
    for c in HypothesisCategory:
        ta = (testability or {}).get(c.value)
        tag = "" if ta is None else (" [testable]" if ta.testable else f" [UNTESTABLE: {ta.reason}]")
        cat_lines.append(f"- {c.value}{tag}")
    categories = "\n".join(cat_lines)
    return f"""A product metric deviation has been observed:
- Metric Name: {observation.metric_name}
- Baseline Window: {observation.baseline_window} (Value: {observation.baseline_value})
- Comparison Window: {observation.comparison_window} (Value: {observation.comparison_value})
- Observed Change: {observation.absolute_change:+.2f}{rel_str}
- Description: {observation.description}

Task:
Formulate at least TWO distinct, competing candidate hypotheses that could explain this deviation.
Assign each hypothesis exactly one category from this list:
{categories}

You may propose untestable causes when they are plausible, but they can never be marked supported without data the
tools cannot provide. Avoid unspecified / external_unspecified; if unavoidable, refine the category (refined_category)
once evidence reveals it.

Return your output matching the requested HypothesesGenerationOutput schema.
"""


def format_analysis_selection_prompt(
    observation: Observation,
    hypotheses: List[Hypothesis],
    evidence_list: List[Evidence],
    iteration: int,
    max_iterations: int,
    testability: Optional[Dict[str, Any]] = None,
) -> str:
    """Prompt asking the LLM to select an analytical operation to test a hypothesis."""
    hyp_text = []
    for h in hypotheses:
        ta = (testability or {}).get(h.category)
        note = "" if ta is None or ta.testable else f" <{'UNTESTABLE' if h.category not in ('unspecified', 'external_unspecified') else 'NEEDS CATEGORY REFINEMENT'}>"
        hyp_text.append(f"- [{h.hypothesis_id}] ({h.status.value.upper()}) [{h.category}]{note} {h.title}: {h.explanation}")
    hyp_summary = "\n".join(hyp_text)

    ev_text = []
    if evidence_list:
        for ev in evidence_list:
            ev_text.append(f"- [{ev.evidence_id}] ({ev.inference_type.value}) {ev.finding}")
    else:
        ev_text.append("  (No empirical evidence recorded yet)")
    ev_summary = "\n".join(ev_text)

    capabilities = describe_capabilities()

    return f"""Investigation Context (Iteration {iteration} of {max_iterations}):
Metric: {observation.metric_name} (Shift: {observation.absolute_change:+.2f})
Baseline Window: {observation.baseline_window}
Comparison Window: {observation.comparison_window}

Current Competing Hypotheses:
{hyp_summary}

Empirical Evidence Gathered So Far:
{ev_summary}

{capabilities}

Task:
Select the single most informative analytical operation from the capabilities list to evaluate one of the active hypotheses.
You MUST choose an operation from the available capabilities list and provide valid parameters.
Do not generate arbitrary SQL or Python code.
Return your decision matching the requested AnalysisDecision schema.
"""


def format_evidence_interpretation_prompt(
    hypothesis: Hypothesis,
    decision_rationale: str,
    result: AnalysisResult,
) -> str:
    """Prompt asking the LLM to interpret the actual tool-returned AnalysisResult."""
    # Serialize data cleanly
    data_summary = ""
    if result.data is not None:
        if hasattr(result.data, "to_dict"):
            try:
                data_summary = json.dumps(result.data.to_dict(), default=str, indent=2)
            except Exception:
                data_summary = str(result.data)
        elif isinstance(result.data, (dict, list)):
            data_summary = json.dumps(result.data, default=str, indent=2)
        else:
            data_summary = str(result.data)
    else:
        data_summary = "None (No data returned)"

    metadata_str = json.dumps(result.metadata, default=str, indent=2) if result.metadata else "{}"
    limitations_str = "\n".join(f"- {l}" for l in result.limitations) if result.limitations else "None noted."
    warnings_str = "\n".join(f"- {w}" for w in result.warnings) if result.warnings else "None noted."

    return f"""The analytical tool has executed. Interpret the actual result for the following hypothesis:

Hypothesis [{hypothesis.hypothesis_id}]: {hypothesis.title}
Explanation: {hypothesis.explanation}
Test Rationale: {decision_rationale}

Executed Tool Result:
- Operation: {result.operation}
- Execution Status: {result.status.value.upper()}
- Returned Data:
{data_summary}
- Metadata:
{metadata_str}
- Tool Limitations:
{limitations_str}
- Tool Warnings:
{warnings_str}
- Error Message: {result.error_message or 'None'}

Task:
Interpret this result honestly based STRICTLY on the numbers returned above.
1. Determine whether this finding SUPPORTS, CONTRADICTS, or is INCONCLUSIVE regarding hypothesis [{hypothesis.hypothesis_id}].
2. Summarize the empirical finding concisely, quoting exact values from the tool output.
3. Explain your rationale.
4. If the evidence supports or refutes this hypothesis, recommend an updated status. Recommending "supported" is only a
   recommendation: the system grants it only after multiple independent analyses, statistically significant after
   multiple-comparison correction, with no contradicting evidence. If the hypothesis category is unspecified or wrong,
   set refined_category to the specific cause family.
5. If the evidence is sufficient to conclude the overall investigation (or halt due to missing data), set is_investigation_complete=True and supply stopping_reason and confidence_level.

Return your interpretation matching the requested EvidenceInterpretation schema.
"""
