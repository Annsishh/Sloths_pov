"""Write evaluation results as JSON plus a readable markdown summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.scoring import aggregate, overall


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.0%}"


def _num(v: Optional[float], nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def build_summary(config: Dict[str, Any], scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"config": config, "overall": overall(scores), "by_scenario": aggregate(scores)}


def summary_markdown(summary: Dict[str, Any]) -> str:
    cfg, ov, by = summary["config"], summary["overall"], summary["by_scenario"]
    lines = ["# Evaluation summary", ""]
    if ov["is_mock"]:
        lines += ["> **MOCK RUN - NOT A MODEL EVALUATION.** These numbers come from a deterministic rule-based reference policy and "
                  "only verify that the harness works end to end.", ""]
    lines += [f"- Provider / model: `{cfg.get('provider')}` / `{cfg.get('model')}`",
              f"- Runs per scenario: {cfg.get('runs')}  |  total runs: {ov['runs']}  |  LLM calls: {ov['total_llm_calls']}",
              f"- Cost (USD): {_num(ov['total_cost_usd'], 4) if ov['total_cost_usd'] is not None else 'not computed (no price table given)'}",
              f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
              f"**Overall: run correct {_pct(ov['run_correct_rate'])}, primary cause correct {_pct(ov['primary_correct_rate'])}, "
              f"integrity errors {ov['integrity_errors']}.**", "",
              "| scenario | runs | correct | primary | false cause | secondary recall | decoys ok | INSUFF. ok | data request | share err | analyses | blocks | calls | latency s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in by.items():
        lines.append(
            f"| {name} | {a['runs']} | {_pct(a['run_correct_rate'])} | {_pct(a['primary_correct_rate'])} | {_pct(a['false_cause_rate'])} | "
            f"{_pct(a['secondary_recall'])} | {_pct(a['decoys_not_supported'])} | {_pct(a['outcome_matches_rate'])} | "
            f"{_pct(a['data_request_ok_rate'])} | {_num(a['explained_share_error'], 3)} | {_num(a['mean_analyses'], 1)} | "
            f"{_num(a['mean_guardrail_blocks'], 1)} | {_num(a['mean_llm_calls'], 1)} | {_num(a['mean_latency_sec'], 2)} |")
    lines += ["", "Columns: *correct* = right primary cause (or right INSUFFICIENT_EVIDENCE) with no unexpected supported cause; "
              "*decoys ok* = decoy categories not supported; *INSUFF. ok* = outcome matches whether the data can answer; "
              "*blocks* = times a guardrail refused something the model tried; *share err* = |estimated - measured| share for overlapping causes."]
    bad = {k: v for k, v in by.items() if v["integrity_errors"] or v["failed_runs"]}
    if bad:
        lines += ["", "## Problems", ""] + [f"- {k}: {v['integrity_errors']} integrity error(s), {v['failed_runs']} failed run(s)" for k, v in bad.items()]
    return "\n".join(lines) + "\n"


def write_results(out_dir: Path, config: Dict[str, Any], records: List[Any], scores: List[Dict[str, Any]],
                  sweep: Optional[Dict[str, Any]] = None, sweep_md: Optional[str] = None, sweep_svg: Optional[str] = None,
                  include_prompts: bool = False) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(config, scores)
    paths = {
        "json": out_dir / "results.json", "markdown": out_dir / "summary.md",
    }
    paths["json"].write_text(json.dumps({"summary": summary, "scores": scores,
                                         "runs": [r.to_dict(include_prompts) for r in records],
                                         "sweep": sweep}, indent=2, default=str), encoding="utf-8")
    paths["markdown"].write_text(summary_markdown(summary), encoding="utf-8")
    if sweep is not None:
        (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        paths["sweep_json"] = out_dir / "sweep.json"
        if sweep_md:
            (out_dir / "sweep.md").write_text(sweep_md, encoding="utf-8"); paths["sweep_md"] = out_dir / "sweep.md"
        if sweep_svg:
            (out_dir / "sweep.svg").write_text(sweep_svg, encoding="utf-8"); paths["sweep_svg"] = out_dir / "sweep.svg"
    return paths


def save_run_trace(out_dir: Path, record: Any, score: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Path]:
    """Write <out_dir>/<scenario>/run_<n>.json and .md: the Phase 6 trace plus scenario/run/provider/model, score and error.

    Never depends on the run succeeding: if the trace cannot be built the files still record why."""
    from investigation.state import InvestigationState
    from investigation.trace import build_trace, trace_markdown

    d = Path(out_dir) / record.scenario
    d.mkdir(parents=True, exist_ok=True)
    stem = f"run_{record.run}"
    meta = {"scenario": record.scenario, "run": record.run, "provider": record.provider, "model": record.model,
            "is_mock": record.is_mock, "error": error or record.error, "score": score, "trace": None, "trace_error": None}
    md = f"# {record.scenario} / run {record.run} ({record.provider}/{record.model})\n\n"
    if meta["error"]:
        md += f"**Error:** {meta['error']}\n\n"
    try:
        state = InvestigationState.from_dict(record.state)
        meta["trace"] = build_trace(state)
        md += trace_markdown(state)
    except Exception as e:
        meta["trace_error"] = f"{type(e).__name__}: {e}"
        md += f"_Trace could not be built: {meta['trace_error']}_\n"
    (d / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (d / f"{stem}.md").write_text(md, encoding="utf-8")
    return {"json": d / f"{stem}.json", "markdown": d / f"{stem}.md"}


def save_failed_run(out_dir: Path, scenario: str, run: int, provider: str, model: Optional[str], error: str) -> Path:
    """A run that crashed before any state existed still leaves a file naming scenario, run, provider, model and error."""
    d = Path(out_dir) / scenario
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run_{run}.json"
    p.write_text(json.dumps({"scenario": scenario, "run": run, "provider": provider, "model": model, "error": error,
                             "score": None, "trace": None}, indent=2), encoding="utf-8")
    return p
