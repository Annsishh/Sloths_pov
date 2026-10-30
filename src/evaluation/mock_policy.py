"""A rule-based stand-in for the LLM, used ONLY to verify the evaluation harness offline.

It reads nothing but the prompt text (observation, hypotheses, data context, tool results) — never ground truth —
and decides with fixed rules, so it is deterministic. Its scores say nothing about how a real model performs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from controller.llm_client import MockLLMClient
from controller.schemas import (
    AnalysisDecision,
    EvidenceInterpretation,
    HypothesesGenerationOutput,
    HypothesisProposal,
)

HYPOTHESES: List[Tuple[str, str, str]] = [
    ("Telemetry tracking loss", "Client-side event tracking dropped events so reported metrics fell.", "tracking_loss"),
    ("Code regression in a release", "A recent release introduced a bug that broke part of checkout.", "code_regression"),
    ("Traffic mix shift", "The mix of arriving users shifted toward lower-converting segments.", "traffic_mix_shift"),
    ("Third-party outage", "An external payment dependency failed for a short period.", "third_party_outage"),
    ("Seasonality or calendar effect", "The comparison window covers weekends or holidays that are normally quieter.", "seasonality_or_calendar"),
    ("Natural variation", "The change is within normal day-to-day variation of the metric.", "natural_variation"),
    ("Pricing or policy change", "A price or policy change reduced conversion.", "pricing_or_policy_change"),
    ("Gradual cohort decay", "Retention is eroding gradually across cohorts.", "gradual_or_cohort_effect"),
]

RELEASE_LINE = re.compile(r"^\s+(rel_\w+) \| (v[\d.]+) \| (\d{4}-\d{2}-\d{2}) [\d:]+ UTC \| (\w+)", re.M)


def _json_after(text: str, marker: str, end_marker: str) -> Any:
    a = text.find(marker)
    if a < 0:
        return None
    a += len(marker)
    b = text.find(end_marker, a)
    try:
        return json.loads(text[a:b if b > 0 else None].strip())
    except (json.JSONDecodeError, ValueError):
        return None


class HeuristicInvestigator:
    """Callable for MockLLMClient(generator_fn=...). One instance per investigation run."""

    def __init__(self) -> None:
        self.steps_done: Dict[str, int] = {}
        self.last_stance: Dict[str, str] = {}
        self.pending: Dict[str, Tuple[str, str]] = {}      # hyp_id -> (category, operation) just requested
        self.prompts_seen: List[str] = []

    # ------------------------------------------------------------------ plans
    @staticmethod
    def _plan(category: str, prompt: str) -> List[Tuple[str, Dict[str, Any]]]:
        if category == "tracking_loss":
            return [("tracking_gap_check", {"breakdown": ["browser"]}),
                    ("event_volume_check", {"event_names": ["order_completed"], "breakdown": ["browser"]})]
        if category == "code_regression":
            m = re.search(r"Comparison Window: \[(\d{4}-\d{2}-\d{2})", prompt)
            comp_start = m.group(1) if m else "0000-00-00"
            rels = [r for r in RELEASE_LINE.findall(prompt) if r[2] >= comp_start]   # releases inside/after the comparison window start
            release = sorted(rels, key=lambda r: r[2])[0][0] if rels else (RELEASE_LINE.findall(prompt) or [("rel_238",)])[-1][0]
            return [("release_did", {"release_id": release, "window_days": 7, "table": "orders",
                                     "metric_column": "status", "metric_value": "completed"}),
                    ("error_analysis", {"breakdown": ["failure_reason"]})]
        if category == "traffic_mix_shift":
            return [("mix_shift_decomposition", {"table": "sessions", "metric_column": "converted",
                                                 "metric_value": True, "breakdown": ["acquisition_channel"]}),
                    ("funnel_comparison", {"funnel_steps": ["checkout_started", "order_completed"], "breakdown": ["acquisition_channel"]})]
        if category == "third_party_outage":
            return [("error_analysis", {"breakdown": ["failure_reason"]}),
                    ("spike_check", {"table": "orders", "metric_column": "status", "metric_value": "failed", "lookback_days": 14})]
        if category == "seasonality_or_calendar":
            return [("seasonal_baseline", {"table": "orders", "aggregation": "count", "baseline_weeks": 4}),
                    ("change_point", {"table": "orders", "aggregation": "count", "lookback_days": 14})]
        if category == "natural_variation":
            return [("change_point", {"table": "orders", "aggregation": "count", "lookback_days": 14}),
                    ("period_comparison", {"table": "orders", "aggregation": "rate", "metric_column": "status", "metric_value": "completed"})]
        if category == "gradual_or_cohort_effect":
            return [("retention_comparison", {"retention_days": [1, 7]}),
                    ("change_point", {"table": "events", "aggregation": "count"})]
        return []

    # ------------------------------------------------------------------ dispatch
    def __call__(self, prompt: str, schema: Any) -> Any:
        self.prompts_seen.append(prompt)
        if schema is HypothesesGenerationOutput:
            return HypothesesGenerationOutput(
                hypotheses=[HypothesisProposal(title=t, explanation=e, category=c) for t, e, c in HYPOTHESES],
                reasoning="Covering the main cause families the available tools can test, plus untestable ones.",
            )
        if schema is AnalysisDecision:
            return self._decide(prompt)
        if schema is EvidenceInterpretation:
            return self._interpret(prompt)
        raise RuntimeError(f"HeuristicInvestigator cannot answer schema {getattr(schema, '__name__', schema)}")

    def _decide(self, prompt: str) -> AnalysisDecision:
        rows = re.findall(r"- \[(hyp_\d+)\] \((\w+)\) \[(\w+)\](?: <([A-Z ]+)>)?", prompt)
        candidates = []
        for hid, status, category, flag in rows:
            if status not in ("PROPOSED", "ACTIVE") or flag:
                continue
            plan = self._plan(category, prompt)
            done = self.steps_done.get(hid, 0)
            if done >= len(plan):
                continue
            if done > 0 and self.last_stance.get(hid) != "supports":
                continue                                # follow up only on hypotheses that looked supportive
            candidates.append((done, int(hid.split("_")[1]), hid, category, plan[done]))
        if not candidates:
            raise RuntimeError("no further informative analysis")
        candidates.sort()                              # untested hypotheses first, in proposal order
        _, _, hid, category, (op, params) = candidates[0]
        self.steps_done[hid] = self.steps_done.get(hid, 0) + 1
        self.pending[hid] = (category, op)
        dims = list(params.get("breakdown", []))
        # App releases ship through the mobile app, so the policy looks at mobile users first (a standard segmentation).
        filters = {"device_type": "mobile"} if op == "release_did" else {}
        return AnalysisDecision(hypothesis_id_to_test=hid, operation=op, parameters=params, dimensions=dims, filters=filters,
                                rationale=f"Test the {category} hypothesis with {op}.")

    def _interpret(self, prompt: str) -> EvidenceInterpretation:
        hid = re.search(r"Hypothesis \[(hyp_\d+)\]", prompt).group(1)
        category, op = self.pending.get(hid, ("", ""))
        meta = _json_after(prompt, "- Metadata:\n", "\n- Tool Limitations:") or {}
        data = _json_after(prompt, "- Returned Data:\n", "\n- Metadata:") or {}
        status_ok = "Execution Status: OK" in prompt
        p = {k: v for k, v in (meta.get("p_values") or {}).items() if v is not None}
        if meta.get("p_value") is not None:
            p["overall"] = meta["p_value"]
        pmin = min(p.values()) if p else None

        stance, detail = "inconclusive", "The tool returned no usable result."
        if status_ok:
            stance, detail = self._stance(category, op, meta, data, p, pmin, hid)
        self.last_stance[hid] = stance
        return EvidenceInterpretation(
            hypothesis_id=hid, stance=stance,
            finding_summary=f"{op}: {detail}"[:400] if len(detail) > 3 else f"{op} returned a result.",
            rationale=f"Rule-based reading of {op} output for {category}: stance={stance}.",
            inference_type="statistical_association" if p else "observational_correlation",
            recommended_hypothesis_status={"supports": "supported", "contradicts": "refuted"}.get(stance),
            status_update_reason=f"{op} result {stance} the hypothesis ({detail})"[:300],
        )

    @staticmethod
    def _stance(category, op, meta, data, p, pmin, hid) -> Tuple[str, str]:
        fmt = lambda v: "n/a" if v is None else f"{v:.3g}"
        if op == "tracking_gap_check":
            if pmin is not None and pmin < 0.01:
                return "supports", f"segment capture rates differ (min raw p={fmt(pmin)}); backend orders exceed telemetry events"
            return "contradicts", f"capture rates are consistent across segments (min p={fmt(pmin)})"
        if op == "release_did":
            did, pv = meta.get("did"), p.get("did_interaction")
            if did is not None and pv is not None and pv < 0.05 and did < 0:
                return "supports", f"users on the new version fell by {did:+.3f} relative to others (DiD p={fmt(pv)})"
            return "contradicts", f"no significant difference-in-differences (DiD={fmt(did)}, p={fmt(pv)})"
        if op == "mix_shift_decomposition":
            mix, total, pv = meta.get("mix_effect"), meta.get("total_change"), p.get("mix_homogeneity")
            if pv is not None and pv < 0.05 and total and mix * total > 0 and abs(mix) >= 0.3 * abs(total):
                return "supports", f"mix effect {mix:+.4f} of total {total:+.4f} (composition p={fmt(pv)})"
            return "contradicts", f"mix effect {fmt(mix)} explains little of {fmt(total)}"
        if op == "seasonal_baseline":
            dev, pat = p.get("seasonal_deviation"), p.get("weekday_pattern")
            if pat is not None and pat < 0.05 and (dev is None or dev >= 0.05):
                return "supports", f"clear weekday pattern (p={fmt(pat)}) and no unexplained deviation (p={fmt(dev)})"
            return "contradicts", f"deviation from same-weekday baseline remains (p={fmt(dev)}) or no weekday pattern (p={fmt(pat)})"
        if op == "change_point":
            pv = p.get("change_point_permutation")
            if category == "natural_variation":
                return ("supports", f"no change point (p={fmt(pv)})") if (pv is not None and pv >= 0.05) else ("contradicts", f"a change point exists (p={fmt(pv)})")
            if category == "gradual_or_cohort_effect":
                return ("supports", "gradual decline, no single step") if (pv is not None and pv < 0.05 and meta.get("onset_shape") == "gradual") else ("contradicts", "shift is a step or absent")
            return "supports", f"consistent with the earlier finding (p={fmt(pv)})"
        if op == "period_comparison":
            pv = p.get("rate_change")
            return ("supports", f"rate change not significant (p={fmt(pv)})") if (pv is not None and pv >= 0.05) else ("contradicts", f"rate changed (p={fmt(pv)})")
        if op == "spike_check":
            top, rate, overall = meta.get("top_period"), meta.get("top_rate"), meta.get("overall_rate")
            sig = [d for d, v in p.items() if v < 0.001]
            if len(sig) == 1 and rate is not None and rate >= 0.3:
                return "supports", f"failure rate spikes to {rate:.2f} on {top} versus {overall:.2f} overall (single abnormal day, raw p={fmt(p[sig[0]])})"
            return "contradicts", f"no single abnormal day ({len(sig)} day(s) with raw p<0.001)"
        if op == "error_analysis":
            text = json.dumps(data)
            if category == "third_party_outage":
                return ("supports", "failures with a non-card reason are present") if re.search(r"unavailable|gateway", text) else ("contradicts", "only routine card declines")
            return "supports", "failed orders concentrate on one device segment"
        if op == "retention_comparison":
            return ("supports", f"retention changed (min p={fmt(pmin)})") if (pmin is not None and pmin < 0.05) else ("contradicts", "retention is stable")
        if op in ("event_volume_check", "funnel_comparison"):
            return "supports", "independent volume view agrees with the first finding"
        return "inconclusive", "no rule for this operation"


def make_mock_client() -> Tuple[MockLLMClient, HeuristicInvestigator]:
    policy = HeuristicInvestigator()
    return MockLLMClient(generator_fn=policy), policy
