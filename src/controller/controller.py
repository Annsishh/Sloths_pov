"""InvestigationController: Central AI investigation loop and orchestration engine.

Coordinates:
- The LLM Client (structured reasoning & interpretation)
- The Investigation State Model (hypotheses, evidence, history, validation)
- The Deterministic Analytics Executor (tool execution against investigator data)
- The Controlled Analytics Capability Catalogue (parameter validation & sandboxing)
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from analytics.executor import AnalyticsExecutor
from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus, TimeWindow
from controller.capability_catalogue import CONTROLLED_OPERATIONS, validate_analysis_plan
from controller.llm_client import LLMClient, create_llm_client
from controller.prompts import (
    SYSTEM_INVESTIGATION_PROMPT,
    format_analysis_selection_prompt,
    format_evidence_interpretation_prompt,
    format_hypotheses_prompt,
)
from controller.schemas import (
    AnalysisDecision,
    EvidenceInterpretation,
    HypothesesGenerationOutput,
)
from investigation.enums import (
    ConfidenceLevel,
    EvidenceStance,
    HistoryEventType,
    HypothesisStatus,
    InferenceType,
    InvestigationStatus,
)
from investigation.models import Evidence, Hypothesis, Observation
from investigation.state import InvestigationState, SupportRuleViolation
from investigation.support import SupportPolicy
from investigation.testability import assess_testability

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when the LLM-call or analysis budget would be exceeded."""


class InvestigationController:
    """Orchestrates the AI investigation loop over an investigation state."""

    def __init__(
        self,
        executor: AnalyticsExecutor,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 5,
        max_analyses: int = 10,
        max_llm_calls: Optional[int] = None,
        support_policy: Optional[SupportPolicy] = None,
    ) -> None:
        self.executor = executor
        self.support_policy = support_policy or SupportPolicy()
        self.llm_client = llm_client or create_llm_client()
        self.max_iterations = max_iterations
        self.max_analyses = max_analyses
        self.analyses_executed_count = 0
        self.max_llm_calls = max_llm_calls if max_llm_calls is not None else 4 * max_analyses + 10
        self.llm_calls = 0

    def _llm(self, **kwargs: Any) -> Any:
        """Every model call goes through here so the call budget is enforced in one place."""
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.max_llm_calls}).")
        self.llm_calls += 1
        return self.llm_client.generate_structured(**kwargs)

    # ------------------------------------------------------------------ data context
    def build_data_context(self) -> str:
        """Observable facts the model needs to parameterise tools: tables + coverage, release log, feature flags,
        app versions. Built only from investigator-accessible tables (never ground truth)."""
        loader = self.executor.loader
        lines = ["Available data (investigator tables only):"]
        for name in loader.available_tables():
            try:
                cov = loader.time_coverage(name)
                span = ""
                if cov:
                    lo = min(v for k, v in cov.items() if k.endswith("_min"))
                    hi = max(v for k, v in cov.items() if k.endswith("_max"))
                    span = f", {lo.date()} to {hi.date()}"
                lines.append(f"- {name}: {loader.row_count(name)} rows{span}")
            except Exception:
                lines.append(f"- {name}")
        tables = set(loader.available_tables())
        if "releases" in tables:
            rel = loader.load("releases")
            lines.append("Release log (release_id | version | deployed | component):")
            for _, r in rel.sort_values("deployed_timestamp").iterrows():
                lines.append(f"  {r['release_id']} | {r['version']} | {r['deployed_timestamp']:%Y-%m-%d %H:%M} UTC | {r['component']}")
        if "feature_exposures" in tables:
            keys = sorted(loader.load("feature_exposures")["feature_key"].unique())
            lines.append(f"Feature flags: {keys}")
        if "events" in tables:
            ev = loader.load("events")
            lines.append(f"Event names: {sorted(ev['event_name'].unique())}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ entry point
    def start_investigation(
        self,
        observation: Observation,
        title: Optional[str] = None,
    ) -> InvestigationState:
        """Initialize a new investigation state from a metric observation."""
        inv_title = title or f"Investigation of {observation.metric_name} deviation"
        testability = assess_testability(CONTROLLED_OPERATIONS, self.executor.loader.available_tables())
        return InvestigationState.create(
            title=inv_title,
            observation=observation,
            support_policy=self.support_policy,
            testability=testability,
        )

    # ------------------------------------------------------------------ individual loop steps
    def step_formulate_hypotheses(self, state: InvestigationState) -> List[Hypothesis]:
        """Ask the LLM to propose at least two competing hypotheses for the observation."""
        obs = state.investigation.observation
        prompt = format_hypotheses_prompt(obs, state.testability)

        try:
            output: HypothesesGenerationOutput = self._llm(
                prompt=prompt,
                schema=HypothesesGenerationOutput,
                system_prompt=SYSTEM_INVESTIGATION_PROMPT,
            )
        except Exception as e:
            logger.error(f"Failed to generate hypotheses: {str(e)}")
            state.investigation.history.append(
                state._record_event(
                    event_type=HistoryEventType.INVESTIGATION_STOPPED,
                    title="Hypothesis Generation Failed",
                    description=f"LLM call failed: {str(e)}",
                )
            )
            raise

        created = []
        for prop in output.hypotheses:
            h_id = prop.hypothesis_id or f"hyp_{len(state.investigation.hypotheses) + 1}"
            hyp = state.add_hypothesis(
                title=prop.title,
                explanation=prop.explanation,
                category=prop.category,
                hypothesis_id=h_id,
            )
            created.append(hyp)

        return created

    def step_select_analysis(
        self,
        state: InvestigationState,
        iteration: int,
    ) -> Optional[AnalysisDecision]:
        """Ask the LLM to select an analytical operation from the controlled catalogue."""
        active_hyps = [
            h for h in state.investigation.hypotheses.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.ACTIVE)
        ]
        if not active_hyps:
            return None

        evidence_list = list(state.investigation.evidence_store.values())
        prompt = format_analysis_selection_prompt(
            observation=state.investigation.observation,
            hypotheses=active_hyps,
            evidence_list=evidence_list,
            iteration=iteration,
            max_iterations=self.max_iterations,
            testability=state.testability,
            data_context=self.build_data_context(),
        )

        try:
            decision: AnalysisDecision = self._llm(
                prompt=prompt,
                schema=AnalysisDecision,
                system_prompt=SYSTEM_INVESTIGATION_PROMPT,
            )
        except Exception as e:
            logger.error(f"Failed to select analysis: {str(e)}")
            state._record_event(
                event_type=HistoryEventType.INVESTIGATION_STOPPED,
                title="Analysis Selection Failed",
                description=f"LLM call failed: {str(e)}",
            )
            return None

        # Verify hypothesis ID exists
        if decision.hypothesis_id_to_test not in state.investigation.hypotheses:
            logger.warning(f"Model referenced non-existent hypothesis: {decision.hypothesis_id_to_test}")
            state._record_event(
                event_type=HistoryEventType.DIRECTION_CHANGED,
                title="Invalid Hypothesis Reference in Analysis Decision",
                description=f"Model suggested testing non-existent hypothesis '{decision.hypothesis_id_to_test}'.",
            )
            return None

        # Validate against controlled capability catalogue
        is_valid, err_msg = validate_analysis_plan(
            operation=decision.operation,
            params=decision.parameters,
            dimensions=decision.dimensions,
            filters=decision.filters,
        )
        if not is_valid:
            logger.warning(f"Invalid analysis plan from LLM: {err_msg}")
            state._record_event(
                event_type=HistoryEventType.DIRECTION_CHANGED,
                title="Analysis Plan Rejected",
                description=f"Model proposed invalid operation or parameters: {err_msg}",
            )
            return None

        return decision

    def execute_analysis(
        self,
        state: InvestigationState,
        decision: AnalysisDecision,
    ) -> AnalysisResult:
        """Construct an AnalysisRequest from the decision and execute via AnalyticsExecutor."""
        obs = state.investigation.observation
        if self.analyses_executed_count >= self.max_analyses:       # also guards callers that bypass the loop
            state._record_event(event_type=HistoryEventType.DIRECTION_CHANGED, title="Analysis Budget Exceeded",
                                description=f"Refused to run '{decision.operation}': {self.max_analyses} analyses already executed.")
            return AnalysisResult(request_id="budget", operation=decision.operation, status=AnalysisStatus.INVALID_PARAMS,
                                  data=None, error_message="Analysis budget exhausted.")
        decision = decision.model_copy(update={"operation": decision.operation.strip().lower()})

        # Select window (period-comparison operations always get the baseline window)
        spec = CONTROLLED_OPERATIONS.get(decision.operation.strip().lower())
        if decision.comparison_window_needed or (spec is not None and spec.needs_comparison_window):
            win = obs.comparison_window
            comp_win = obs.baseline_window
        else:
            win = obs.comparison_window
            comp_win = None

        lookback = decision.parameters.get("lookback_days")
        if spec is not None and spec.supports_lookback and isinstance(lookback, int) and lookback > 0:
            win = TimeWindow(win.start - timedelta(days=lookback), win.end)

        request = AnalysisRequest(
            operation=decision.operation,
            window=win,
            comparison_window=comp_win,
            params=dict(decision.parameters),
            dimensions=list(decision.dimensions),
            filters=dict(decision.filters),
        )

        self.analyses_executed_count += 1
        result = self.executor.run(request)
        # Every test the tool ran joins the BH family now, whether or not it is later interpreted.
        state.register_analysis_result(result)

        state._record_event(
            event_type=HistoryEventType.ANALYSIS_DISPATCHED,
            title=f"Analysis Executed: {decision.operation}",
            description=decision.rationale,
            metadata={
                "operation": decision.operation,
                "parameters": dict(decision.parameters),
                "dimensions": list(decision.dimensions),
                "filters": dict(decision.filters),
                "window": str(win),
                "comparison_window": str(comp_win) if comp_win else None,
                "request_id": result.request_id,
                "status": result.status.value,
                "hypothesis_id": decision.hypothesis_id_to_test,
            },
        )
        return result

    def step_interpret_result(
        self,
        state: InvestigationState,
        decision: AnalysisDecision,
        result: AnalysisResult,
    ) -> Optional[EvidenceInterpretation]:
        """Ask the LLM to interpret the actual tool-returned result and update state."""
        hyp = state.investigation.hypotheses[decision.hypothesis_id_to_test]

        prompt = format_evidence_interpretation_prompt(
            hypothesis=hyp,
            decision_rationale=decision.rationale,
            result=result,
        )

        try:
            interp: EvidenceInterpretation = self._llm(
                prompt=prompt,
                schema=EvidenceInterpretation,
                system_prompt=SYSTEM_INVESTIGATION_PROMPT,
            )
        except Exception as e:
            logger.error(f"Failed to interpret result: {str(e)}")
            state._record_event(
                event_type=HistoryEventType.DIRECTION_CHANGED,
                title="Result Interpretation Failed",
                description=f"LLM call failed to parse interpretation: {str(e)}",
            )
            return None

        # 0. Category refinement (placeholders must become specific before support is possible)
        if interp.refined_category:
            try:
                state.refine_category(
                    hyp.hypothesis_id,
                    interp.refined_category,
                    interp.status_update_reason or interp.rationale,
                )
            except ValueError as e:
                state._record_event(
                    event_type=HistoryEventType.DIRECTION_CHANGED,
                    title="Category Refinement Rejected",
                    description=str(e),
                    metadata={"hypothesis_id": hyp.hypothesis_id},
                )

        # 1. Record Evidence in state
        try:
            inf_type = InferenceType(interp.inference_type)
        except ValueError:
            inf_type = InferenceType.OBSERVATIONAL_CORRELATION

        op_spec = CONTROLLED_OPERATIONS.get(result.operation.lower())
        ev = state.record_evidence(
            finding=interp.finding_summary,
            source_tables=list(op_spec.requires_tables) if op_spec and op_spec.requires_tables else ["orders", "events"],
            analysis_result=result,
            population=decision.filters,
            window=result.metadata.get("window"),
            inference_type=inf_type,
        )

        # 2. Link Evidence to Hypothesis
        try:
            stance = EvidenceStance(interp.stance)
        except ValueError:
            stance = EvidenceStance.INCONCLUSIVE

        state.link_evidence(
            evidence_id=ev.evidence_id,
            hypothesis_id=decision.hypothesis_id_to_test,
            stance=stance,
            rationale=interp.rationale,
        )

        # 3. Status. The LLM may only recommend; SUPPORTED is granted by the deterministic rule.
        support_blocked = False
        if interp.recommended_hypothesis_status:
            rec_status = interp.recommended_hypothesis_status.strip().lower()
            if rec_status in {"supported", "refuted", "active", "deprioritized"}:
                new_status = HypothesisStatus(rec_status)
                reason = interp.status_update_reason or f"Updated based on evidence {ev.evidence_id} ({stance.value})"
                if new_status == HypothesisStatus.SUPPORTED:
                    pass  # decided below by the rule, never directly by the model
                else:
                    state.update_hypothesis_status(
                        hypothesis_id=decision.hypothesis_id_to_test,
                        new_status=new_status,
                        reason=reason,
                        triggered_by_evidence_id=ev.evidence_id,
                    )
        # Re-evaluate everything: adjusted p-values shift as tests accumulate.
        state.reassess_supported()  # logs a SUPPORT_WITHDRAWN audit event with before/after adjusted p-values
        llm_wanted_support = bool(interp.recommended_hypothesis_status
                                  and interp.recommended_hypothesis_status.strip().lower() == "supported")
        if hyp.status in (HypothesisStatus.PROPOSED, HypothesisStatus.ACTIVE):
            decision_rule = state.evaluate_support(hyp.hypothesis_id)
            if decision_rule.eligible:
                state.update_hypothesis_status(
                    hypothesis_id=hyp.hypothesis_id,
                    new_status=HypothesisStatus.SUPPORTED,
                    reason=(interp.status_update_reason or "Support rule satisfied")
                           + f" [rule: {state.policy.min_supporting_evidence}+ supporting items from "
                           f"{state.policy.min_distinct_analyses}+ analyses, significant after correction]",
                    triggered_by_evidence_id=ev.evidence_id,
                )
            else:
                if llm_wanted_support:
                    support_blocked = True
                    state._record_event(
                        event_type=HistoryEventType.DIRECTION_CHANGED,
                        title="Support Recommendation Not Accepted",
                        description=f"Model recommended SUPPORTED for '{hyp.hypothesis_id}' but the deterministic "
                                    "rule is not met: " + "; ".join(decision_rule.reasons),
                        metadata={"hypothesis_id": hyp.hypothesis_id, "needs_more_evidence": decision_rule.needs_more_evidence},
                    )
                if hyp.status == HypothesisStatus.PROPOSED and stance != EvidenceStance.CONTRADICTS:
                    state.update_hypothesis_status(
                        hypothesis_id=hyp.hypothesis_id,
                        new_status=HypothesisStatus.ACTIVE,
                        reason=f"Under active analysis; evidence {ev.evidence_id} recorded ({stance.value}).",
                        triggered_by_evidence_id=ev.evidence_id,
                    )

        # 4. Handle newly identified unresolved questions
        for q_text in interp.unresolved_questions:
            state.add_unresolved_question(
                question_text=q_text,
                hypothesis_id=decision.hypothesis_id_to_test,
                context_or_limitation=f"Identified during analysis {result.operation}",
            )

        # 5. Check if investigation is complete
        if interp.is_investigation_complete:
            supported_ids = state.supported_hypothesis_ids()
            if supported_ids:
                conf_level = ConfidenceLevel.MEDIUM
                if interp.confidence_level:
                    try:
                        conf_level = ConfidenceLevel(interp.confidence_level.lower())
                    except ValueError:
                        pass
                primary = hyp.hypothesis_id if hyp.hypothesis_id in supported_ids else supported_ids[0]
                state.conclude(
                    conclusion_summary=interp.stopping_reason or interp.finding_summary,
                    supported_hypothesis_id=primary,
                    confidence_level=conf_level,
                    confidence_justification=interp.confidence_justification or interp.rationale,
                    remaining_uncertainties=interp.unresolved_questions,
                    additional_supported_ids=[i for i in supported_ids if i != primary],
                )
            elif support_blocked:
                # The model declared victory but the rule needs more evidence: keep investigating.
                state._record_event(
                    event_type=HistoryEventType.DIRECTION_CHANGED,
                    title="Completion Deferred",
                    description="Model declared the investigation complete, but no hypothesis satisfies the support rule yet.",
                )
            else:
                # Stop due to insufficient evidence
                state.stop_insufficient_evidence(
                    reason=interp.stopping_reason or "Evidence was insufficient to substantiate hypotheses.",
                    missing_information_summary="; ".join(interp.unresolved_questions) or "Missing conclusive instrumentation.",
                    confidence_justification=interp.confidence_justification or "Data gaps prevent distinguishing explanations.",
                )

        return interp

    def _conclude_with_supported(self, state: InvestigationState) -> bool:
        """Conclude from hypotheses the rule has already marked SUPPORTED; False if there are none."""
        supported_ids = state.supported_hypothesis_ids()
        if not supported_ids:
            return False
        titles = "; ".join(state.investigation.hypotheses[i].title for i in supported_ids)
        state.conclude(
            conclusion_summary=f"The evidence is consistent with: {titles} (support rule satisfied; see conclusion statements for what tools can confirm).",
            supported_hypothesis_id=supported_ids[0],
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_justification="Support rule satisfied (multiple independent significant analyses, no refuting evidence).",
            additional_supported_ids=supported_ids[1:],
        )
        return True

    # ------------------------------------------------------------------ full investigation loop
    def run_investigation_loop(self, state: InvestigationState) -> InvestigationState:
        """Run the complete investigation loop until concluded, stopped, or budget exhausted."""
        # 1. Formulate hypotheses if none exist
        if not state.investigation.hypotheses:
            self.step_formulate_hypotheses(state)

        # 2. Iterate investigation cycles
        for iteration in range(1, self.max_iterations + 1):
            if state.investigation.status in (InvestigationStatus.CONCLUDED, InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE):
                break

            if self.llm_calls >= self.max_llm_calls or self.analyses_executed_count >= self.max_analyses:
                state.stop_insufficient_evidence(
                    reason="Analysis execution budget limit reached.",
                    missing_information_summary=f"Reached maximum allowed analyses ({self.max_analyses}) before all hypotheses were resolved.",
                    confidence_justification="Investigation halted by budget constraint.",
                )
                break

            # Select analysis
            decision = self.step_select_analysis(state, iteration)
            if not decision:
                if self._conclude_with_supported(state):
                    break
                state.stop_insufficient_evidence(
                    reason="No further analytical actions could be selected.",
                    missing_information_summary="Available tools exhausted without distinguishing competing hypotheses.",
                    confidence_justification="Cannot proceed without new data sources.",
                )
                break

            # Execute analysis
            result = self.execute_analysis(state, decision)

            # Interpret result
            self.step_interpret_result(state, decision, result)
            if state.investigation.status in (InvestigationStatus.CONCLUDED, InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE):
                break

        # If loop finishes without terminal status, halt with explicit insufficient evidence status
        if state.investigation.status in (InvestigationStatus.OPEN, InvestigationStatus.IN_PROGRESS):
            if self._conclude_with_supported(state):
                return state
            state.stop_insufficient_evidence(
                reason="Maximum investigation iterations reached without conclusive proof.",
                missing_information_summary="Cycle limit reached with competing hypotheses still unresolved.",
                confidence_justification="Investigation reached iteration boundary without conclusive empirical distinction.",
            )

        return state
