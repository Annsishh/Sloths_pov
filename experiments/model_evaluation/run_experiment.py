"""CLI: run the benchmark across configured models and write a comparative report.

    python -m experiments.model_evaluation.run_experiment --config experiments/model_evaluation/models.example.json
    python -m experiments.model_evaluation.run_experiment --config experiments/model_evaluation/models.demo_mock.json --allow-mock

Outputs (default experiments/model_evaluation/output/<timestamp>/):
    results.json     config, per-model summary, per-run evaluations, incomplete-criteria notes
    comparison.csv   one row per model x scenario, plus an ALL row per model
    runs/*.json      raw model responses, hypotheses, analyses, evidence requests, conclusion, errors for every run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.clients import EvalConfigError  # noqa: E402
from experiments.model_evaluation.evaluator import INCOMPLETE_CRITERIA, QUALITY_CRITERIA, evaluate_run, summarise  # noqa: E402
from experiments.model_evaluation.providers import build_client  # noqa: E402
from experiments.model_evaluation.runner import DEFAULT_DATA_ROOT, HERE, load_json, run_experiment  # noqa: E402

CSV_FIELDS = ["model_id", "scenario", "runs", "is_mock", "completed_runs", "correct_rate", *QUALITY_CRITERIA, "quality_mean",
              "api_errors", "schema_errors", "llm_calls", "tokens_in", "tokens_out", "cost_usd", "latency_sec"]


def run_artifact(res: Dict[str, Any], ev: Dict[str, Any]) -> Dict[str, Any]:
    """Everything captured for one run, for later inspection. Prompts are not stored; raw responses are."""
    rec = res["record"]
    st = rec.state if rec else {}
    return {
        "model_id": res["model_id"], "provider": res["provider"], "model": res["model"], "scenario": res["scenario"], "run": res["run"],
        "error": res["error"], "api_errors": ev["api_errors"], "schema_errors": ev["schema_errors"],
        "model_outputs": res["outputs"],
        "investigation": {"hypotheses": st.get("hypotheses", {}), "analyses_executed": rec.analyses if rec else [],
                          "conclusion": st.get("conclusion")},
        "llm_usage": rec.llm if rec else None,
        "calls": res["calls"],                      # includes raw_response, valid, error, error_kind, usage, latency per call
        "evaluation": ev,
    }


def write_report(out: Path, config: Dict[str, Any], results: List[Dict[str, Any]], evals: List[Dict[str, Any]], rubric: Dict[str, Any]) -> Dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    for res, ev in zip(results, evals):
        (out / "runs" / f"{res['model_id']}__{res['scenario']}__run{res['run']}.json").write_text(
            json.dumps(run_artifact(res, ev), indent=2, default=str), encoding="utf-8")
    rows = summarise(evals)
    mock = any(e["is_mock"] for e in evals)
    (out / "results.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "WARNING": "MOCK RUN - NOT A MODEL EVALUATION" if mock else None,
        "config": config, "scenarios_in_rubric": sorted(rubric),
        "summary": rows, "runs": evals, "incomplete_criteria": INCOMPLETE_CRITERIA}, indent=2, default=str), encoding="utf-8")
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"json": out / "results.json", "csv": out / "comparison.csv", "runs": out / "runs"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default="all", help="comma-separated model ids from the config, or 'all'")
    ap.add_argument("--scenarios", default="all", help="comma-separated scenario ids from scenarios.json, or 'all'")
    ap.add_argument("--runs", type=int, default=None, help="runs per model x scenario (default: config value)")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="print how many investigations/LLM calls would run, then exit")
    args = ap.parse_args(argv)

    cfg = load_json(Path(args.config))
    models = cfg["models"] if args.models == "all" else [m for m in cfg["models"] if m["id"] in args.models.split(",")]
    scenarios = load_json(HERE / "scenarios.json")["scenarios"]
    if args.scenarios != "all":
        scenarios = [s for s in scenarios if s["id"] in args.scenarios.split(",")]
    runs = args.runs or cfg.get("runs", 1)
    max_analyses = cfg.get("max_analyses", 10)
    n = len(models) * len(scenarios) * runs
    print(f"{len(models)} model(s) x {len(scenarios)} scenario(s) x {runs} run(s) = {n} investigations; "
          f"up to ~{n * (2 + 2 * max_analyses)} LLM calls (typically ~{n * 19}).")
    if args.estimate:
        return 0
    try:                                           # fail BEFORE spending anything if any model is misconfigured
        for m in models:
            build_client(m, args.allow_mock)
    except EvalConfigError as e:
        print(f"REFUSING TO RUN: {e}", file=sys.stderr)
        return 3

    def progress(r: Dict[str, Any]) -> None:
        n_err = sum(1 for c in r["calls"] if c["error"])
        print(f"[{r['model_id']} / {r['scenario']} #{r['run']}] calls={len(r['calls'])} call_errors={n_err} "
              f"{'ERROR: ' + r['error'][:120] if r['error'] else 'ok'}")

    results = run_experiment(models, scenarios, runs, Path(args.data_dir), max_analyses, args.allow_mock, progress)
    rubric = load_json(HERE / "rubric.json")             # evaluator-only files are read only after every run finished
    evals = [evaluate_run(r, Path(args.data_dir), rubric) for r in results]
    out = Path(args.out) if args.out else HERE / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = write_report(out, {"models": [{k: v for k, v in m.items() if k != "api_key_env"} for m in models],
                               "scenarios": [s["id"] for s in scenarios], "runs": runs, "max_analyses": max_analyses}, results, evals, rubric)
    print("Wrote:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
