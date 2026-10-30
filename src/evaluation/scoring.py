"""Score investigation runs against ground truth. This module (and only this one) reads ground-truth files."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from evaluation.runner import RunRecord
from investigation.state import InvestigationState

# History-event titles that mean the controller blocked something the model tried to do.
BLOCK_TITLES = ("Support Recommendation Not Accepted", "Analysis Plan Rejected", "Invalid Hypothesis Reference",
                "Category Refinement Rejected", "Completion Deferred")


def load_truth(data_root: Path, scenario: str) -> Dict[str, Any]:
    return json.loads((Path(data_root) / scenario / "ground_truth" / "scenario_ground_truth.json").read_text(encoding="utf-8"))


def measured_code_share(truth: Dict[str, Any]) -> Optional[float]:
    """For overlapping causes: share of the lost tracked orders due to the real regression (measured while generating)."""
    m = truth.get("measured_effects") or {}
    reg, trk = m.get("regression_failed_orders", 0), m.get("tracking_dropped_events", 0)
    if truth.get("expected_answer", {}).get("shares_source") == "measured_effects" and (reg + trk) > 0:
        return reg / (reg + trk)
    return None


def score_run(run: RunRecord, truth: Dict[str, Any]) -> Dict[str, Any]:
    exp = truth["expected_answer"]
    st = run.state
    concl = st.get("conclusion") or {}
    supported = [c["category"] for c in concl.get("supported_causes", [])]
    outcome = concl.get("outcome", "none")
    primary, secondary, decoys = exp.get("primary_cause"), list(exp.get("secondary_causes", [])), list(exp.get("decoys", []))
    expect_insufficient = bool(exp.get("insufficient_evidence"))

    if expect_insufficient:
        primary_correct = outcome == "insufficient_evidence" and not supported
    else:
        primary_correct = primary in supported
    expected_set = {primary, *secondary} - {None}
    false_causes = [c for c in supported if c not in expected_set]

    secondary_recall = (len([c for c in secondary if c in supported]) / len(secondary)) if secondary else None

    hyps = st.get("hypotheses", {}).values()
    decoys_not_supported = (sum(1 for d in decoys if d not in supported) / len(decoys)) if decoys else None
    decoys_refuted = (sum(1 for d in decoys if any(h["category"] == d and h["status"] == "refuted" for h in hyps)) / len(decoys)) if decoys else None

    outcome_matches = (outcome == "insufficient_evidence") == expect_insufficient
    data_request_ok = None
    if expect_insufficient:
        kws = [k.lower() for k in exp.get("acceptable_data_request_keywords", [])]
        reqs = " ".join(concl.get("data_requests", [])).lower()
        data_request_ok = any(k in reqs for k in kws) if kws else bool(concl.get("data_requests"))

    share_error = None
    true_code = measured_code_share(truth)
    if true_code is not None:
        by = {c["category"]: c for c in concl.get("supported_causes", [])}
        a, b = by.get("code_regression", {}).get("estimated_change"), by.get("tracking_loss", {}).get("estimated_change")
        if a is not None and b is not None and (abs(a) + abs(b)) > 0:
            share_error = abs(abs(a) / (abs(a) + abs(b)) - true_code)

    events = st.get("history", [])
    blocks = sum(1 for e in events if any(e["title"].startswith(t) for t in BLOCK_TITLES))
    integrity = InvestigationState.from_dict(st).validate()
    hard = list(integrity)
    for h in st.get("hypotheses", {}).values():                # a SUPPORTED placeholder/untestable category would be a rule breach
        ta = st.get("testability", {}).get(h["category"], {})
        if h["status"] == "supported" and (h["category"] in ("unspecified", "external_unspecified", "multiple_causes") or not ta.get("testable", True)):
            hard.append(f"supported hypothesis {h['hypothesis_id']} in non-testable category {h['category']}")

    run_correct = bool(primary_correct and not false_causes and outcome_matches and (data_request_ok in (None, True)))
    return {
        "scenario": run.scenario, "run": run.run, "expected_primary": primary, "supported": supported, "outcome": outcome,
        "primary_correct": bool(primary_correct), "secondary_recall": secondary_recall,
        "false_causes": false_causes, "decoys_not_supported": decoys_not_supported, "decoys_actively_refuted": decoys_refuted,
        "insufficient_expected": expect_insufficient, "outcome_matches_expectation": bool(outcome_matches),
        "data_request_ok": data_request_ok, "explained_share_error": share_error,
        "explained_change_share": concl.get("explained_change") and concl.get("total_observed_change")
        and concl["explained_change"] / concl["total_observed_change"],
        "analyses": run.analyses, "n_analyses": len(run.analyses), "n_distinct_analyses": len(set(run.analyses)),
        "guardrail_blocks": blocks, "integrity_errors": hard,
        "run_correct": run_correct, "error": run.error,
        "llm_calls": run.llm.get("calls"), "input_tokens": run.llm.get("input_tokens"), "output_tokens": run.llm.get("output_tokens"),
        "latency_sec": run.llm.get("latency_sec"), "cost_usd": run.llm.get("cost_usd"), "wall_sec": run.wall_sec,
        "is_mock": run.is_mock,
    }


def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(float(v) for v in vals), 4) if vals else None


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scenario in sorted({s["scenario"] for s in scores}):
        rows = [s for s in scores if s["scenario"] == scenario]
        out[scenario] = {
            "runs": len(rows),
            "run_correct_rate": _avg([{**r, "x": float(r["run_correct"])} for r in rows], "x"),
            "primary_correct_rate": _avg([{**r, "x": float(r["primary_correct"])} for r in rows], "x"),
            "false_cause_rate": _avg([{**r, "x": float(bool(r["false_causes"]))} for r in rows], "x"),
            "secondary_recall": _avg(rows, "secondary_recall"),
            "decoys_not_supported": _avg(rows, "decoys_not_supported"),
            "decoys_actively_refuted": _avg(rows, "decoys_actively_refuted"),
            "insufficient_expected": rows[0]["insufficient_expected"],
            "outcome_matches_rate": _avg([{**r, "x": float(r["outcome_matches_expectation"])} for r in rows], "x"),
            "data_request_ok_rate": _avg([{**r, "x": float(r["data_request_ok"])} for r in rows if r["data_request_ok"] is not None], "x"),
            "explained_share_error": _avg(rows, "explained_share_error"),
            "mean_analyses": _avg(rows, "n_analyses"),
            "mean_guardrail_blocks": _avg(rows, "guardrail_blocks"),
            "integrity_errors": sum(len(r["integrity_errors"]) for r in rows),
            "mean_llm_calls": _avg(rows, "llm_calls"),
            "mean_latency_sec": _avg(rows, "latency_sec"),
            "mean_cost_usd": _avg(rows, "cost_usd"),
            "failed_runs": sum(1 for r in rows if r["error"]),
        }
    return out


def overall(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "runs": len(scores),
        "run_correct_rate": _avg([{**r, "x": float(r["run_correct"])} for r in scores], "x"),
        "primary_correct_rate": _avg([{**r, "x": float(r["primary_correct"])} for r in scores], "x"),
        "total_llm_calls": sum(r["llm_calls"] or 0 for r in scores),
        "total_cost_usd": round(sum(r["cost_usd"] for r in scores if r["cost_usd"] is not None), 6) if any(r["cost_usd"] is not None for r in scores) else None,
        "integrity_errors": sum(len(r["integrity_errors"]) for r in scores),
        "is_mock": any(r["is_mock"] for r in scores),
    }
