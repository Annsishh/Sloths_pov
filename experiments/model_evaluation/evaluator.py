"""Lightweight evaluator. EVALUATOR-ONLY: this is the one module here that reads ground truth and the rubric.

Every criterion is a 0..1 score (higher is better) or None when it cannot be measured for that run. Cost and latency are
reported as raw numbers, not scored. Most criteria are deterministic proxies; see INCOMPLETE_CRITERIA for what they miss.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from evaluation.scoring import load_truth, score_run

PLACEHOLDER_CATEGORIES = {"unspecified", "external_unspecified", "multiple_causes"}
CONFIDENCE = {"very_low": 0.1, "low": 0.3, "medium": 0.6, "high": 0.9}
QUALITY_CRITERIA = ("hypothesis_quality", "evidence_relevance", "analysis_selection", "contradiction_handling",
                    "uncertainty_calibration", "conclusion_quality", "schema_compliance")

INCOMPLETE_CRITERIA = {
    "hypothesis_quality": "Category-level proxy: recall of expected cause categories plus share of non-placeholder categories. Does not judge whether the written explanation/mechanism is sound.",
    "evidence_relevance": "Share of executed operations in a hand-written per-scenario relevance list (rubric.json, author judgement). Does not judge whether the parameters/filters/segments chosen were the right ones, and cannot see operations the model never requested.",
    "analysis_selection": "Coverage of rubric 'core' operation groups plus duplicate and guardrail-block penalties. Same rubric limitation; a different valid strategy can score lower.",
    "contradiction_handling": "Proxy from final hypothesis statuses: wrong-category hypotheses should end refuted/unsupported, and no supported hypothesis should carry contradicting evidence. Does not check whether the model's stated stance on each result was right. None when no wrong-category hypothesis was proposed.",
    "uncertainty_calibration": "1 - Brier score of the model's LAST stated confidence_level vs. run correctness. Very noisy per run (one sample); meaningful only when averaged over many runs. None when no confidence was stated.",
    "conclusion_quality": "Reuses src/evaluation/scoring.score_run (primary cause, false causes, INSUFFICIENT_EVIDENCE outcome, data requests, secondary recall). The final conclusion is assembled by the deterministic controller from the model's hypotheses/analyses/stances, so this partly measures the pipeline, not only the model.",
    "schema_compliance": "Valid structured responses / (valid + schema-invalid). API errors are counted separately and excluded.",
    "not_measured": "No LLM-judge or human grading of explanation text; no repeated-run variance unless --runs > 1; cost is None unless prices are configured; tokens are provider-reported where available.",
}


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return round(mean(v), 4) if v else None


def _hypothesis_quality(hyps: List[Dict[str, Any]], exp: Dict[str, Any]) -> Optional[float]:
    expected = ({exp.get("primary_cause")} | set(exp.get("secondary_causes", []))) - {None}
    if not expected:
        expected = set(exp.get("untestable_categories", []))
    if not hyps:
        return 0.0
    cats = {h["category"] for h in hyps}
    recall = len(expected & cats) / len(expected) if expected else 1.0
    specific = sum(1 for h in hyps if h["category"] not in PLACEHOLDER_CATEGORIES) / len(hyps)
    return round(0.7 * recall + 0.3 * specific, 4)


def _analysis_scores(ops: List[str], blocks: int, rub: Dict[str, Any]) -> Dict[str, Optional[float]]:
    if not ops:
        return {"evidence_relevance": 0.0, "analysis_selection": 0.0}
    relevant = set(rub["relevant"])
    relevance = sum(1 for o in ops if o in relevant) / len(ops)
    groups = rub["core"]
    coverage = sum(1 for g in groups if set(g) & set(ops)) / len(groups) if groups else 1.0
    distinct = len(set(ops)) / len(ops)
    blocked = blocks / (len(ops) + blocks)
    return {"evidence_relevance": round(relevance, 4),
            "analysis_selection": round(0.6 * coverage + 0.2 * distinct + 0.2 * (1 - blocked), 4)}


def _contradiction_handling(hyps: Dict[str, Any], exp: Dict[str, Any]) -> Optional[float]:
    expected = ({exp.get("primary_cause")} | set(exp.get("secondary_causes", []))) - {None}
    wrong = [h for h in hyps.values() if h["category"] not in expected]
    if not wrong:
        return None
    not_supported = sum(1 for h in wrong if h["status"] != "supported") / len(wrong)
    refuted = sum(1 for h in wrong if h["status"] == "refuted") / len(wrong)
    base = 0.5 * not_supported + 0.5 * refuted
    sup = [h for h in hyps.values() if h["status"] == "supported"]
    flagged = sum(1 for h in sup if h["contradicting_evidence_ids"]) / len(sup) if sup else 0.0
    return round(base * (1 - 0.5 * flagged), 4)


def _calibration(confidences: List[str], correct: bool) -> Optional[float]:
    if not confidences or confidences[-1] not in CONFIDENCE:
        return None
    return round(1 - (CONFIDENCE[confidences[-1]] - float(correct)) ** 2, 4)


def _schema_compliance(calls: List[Dict[str, Any]]) -> Optional[float]:
    valid = sum(1 for c in calls if c["valid"])
    bad = sum(1 for c in calls if c["error_kind"] == "schema")
    return round(valid / (valid + bad), 4) if valid + bad else None


def evaluate_run(res: Dict[str, Any], data_root: Path, rubric: Dict[str, Any]) -> Dict[str, Any]:
    truth = load_truth(data_root, res["data_scenario"])
    exp, calls, outputs = truth["expected_answer"], res["calls"], res["outputs"]
    api_errors = sum(1 for c in calls if c["error_kind"] == "api")
    schema_errors = sum(1 for c in calls if c["error_kind"] == "schema")
    ev: Dict[str, Any] = {"model_id": res["model_id"], "scenario": res["scenario"], "run": res["run"], "is_mock": res["is_mock"],
                          "completed": res["record"] is not None, "run_error": res["error"],
                          "api_errors": api_errors, "schema_errors": schema_errors, "n_calls": len(calls)}
    ev.update({k: None for k in QUALITY_CRITERIA})
    ev["schema_compliance"] = _schema_compliance(calls)
    rec = res["record"]
    if rec is None:                               # crashed before any investigation state existed
        ev.update(conclusion_quality=0.0, correct=False, outcome="none", supported=[], tokens_in=0, tokens_out=0,
                  cost_usd=None, latency_sec=round(sum(c["latency_sec"] for c in calls), 4), wall_sec=res.get("wall_sec"))
        return ev
    sc = score_run(rec, truth)
    hyps = rec.state.get("hypotheses", {})
    ev["hypothesis_quality"] = _hypothesis_quality([{"category": h["category"]} for h in hyps.values()], exp)
    ev.update(_analysis_scores(rec.analyses, sc["guardrail_blocks"], rubric[res["scenario"]]))
    ev["contradiction_handling"] = _contradiction_handling(hyps, exp)
    ev["uncertainty_calibration"] = _calibration(outputs.get("stated_confidences", []), sc["run_correct"])
    parts = [float(sc["primary_correct"]), float(not sc["false_causes"]), float(sc["outcome_matches_expectation"]),
             sc["secondary_recall"], None if sc["data_request_ok"] is None else float(sc["data_request_ok"])]
    ev["conclusion_quality"] = _mean(parts)
    ev.update(correct=sc["run_correct"], outcome=sc["outcome"], supported=sc["supported"], expected_primary=sc["expected_primary"],
              integrity_errors=len(sc["integrity_errors"]), guardrail_blocks=sc["guardrail_blocks"], n_analyses=sc["n_analyses"],
              tokens_in=sc["input_tokens"], tokens_out=sc["output_tokens"], tokens_estimated=rec.llm.get("tokens_estimated"),
              cost_usd=sc["cost_usd"], latency_sec=sc["latency_sec"], wall_sec=res.get("wall_sec"))
    return ev


def summarise(evals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per model x scenario rows plus one ALL row per model (means of the criteria that were measurable)."""
    rows = []
    for model in sorted({e["model_id"] for e in evals}):
        mine = [e for e in evals if e["model_id"] == model]
        for scen in [*sorted({e["scenario"] for e in mine}), "ALL"]:
            sub = mine if scen == "ALL" else [e for e in mine if e["scenario"] == scen]
            row: Dict[str, Any] = {"model_id": model, "scenario": scen, "runs": len(sub), "is_mock": any(e["is_mock"] for e in sub),
                                   "completed_runs": sum(1 for e in sub if e["completed"]),
                                   "correct_rate": _mean([float(e["correct"]) for e in sub if "correct" in e])}
            for k in QUALITY_CRITERIA:
                row[k] = _mean([e[k] for e in sub])
            row["quality_mean"] = _mean([row[k] for k in QUALITY_CRITERIA])
            row["api_errors"] = sum(e["api_errors"] for e in sub)
            row["schema_errors"] = sum(e["schema_errors"] for e in sub)
            row["llm_calls"] = sum(e["n_calls"] for e in sub)
            row["tokens_in"] = sum(e.get("tokens_in") or 0 for e in sub)
            row["tokens_out"] = sum(e.get("tokens_out") or 0 for e in sub)
            costs = [e["cost_usd"] for e in sub if e.get("cost_usd") is not None]
            row["cost_usd"] = round(sum(costs), 6) if costs else None
            row["latency_sec"] = round(sum(e.get("latency_sec") or 0 for e in sub), 3)
            rows.append(row)
    return rows
