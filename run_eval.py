"""Evaluation CLI. Runs the full investigation against every scenario, N times, and scores it vs ground truth.

Real evaluation (PAID API calls - you run this yourself):
    set ANTHROPIC_API_KEY=...            (PowerShell: $env:ANTHROPIC_API_KEY="...")
    python run_eval.py --provider anthropic --model <model-id> --runs 3 --sweep --out eval_results/run1

Offline harness check (no API, rule-based reference policy; results are NOT a model evaluation):
    python run_eval.py --provider mock --allow-mock --runs 1 --out eval_results/mock

Preview the number of LLM calls without running anything:
    python run_eval.py --estimate --runs 3 --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from evaluation.clients import EvalConfigError, resolve_client  # noqa: E402
from evaluation.metered import PriceTable  # noqa: E402
from evaluation.mock_policy import make_mock_client  # noqa: E402
from evaluation.report import write_results  # noqa: E402
from evaluation.runner import DEFAULT_MAX_ANALYSES, estimate_llm_calls, run_scenario_dir  # noqa: E402
from evaluation.scoring import load_truth, score_run  # noqa: E402
from evaluation.sweep import run_effect_sweep, sweep_markdown, sweep_svg  # noqa: E402
from investigator_data.scenarios import SCENARIO_REGISTRY  # noqa: E402

DEFAULT_EFFECTS = (0.1, 0.2, 0.3, 0.5, 0.75)
DEFAULT_NOISES = (0.0, 0.2, 0.4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="anthropic", help="anthropic | openai | mock (mock needs --allow-mock)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--allow-mock", action="store_true", help="permit the rule-based reference policy (not a model evaluation)")
    ap.add_argument("--runs", type=int, default=3, help="runs per scenario")
    ap.add_argument("--scenarios", default="all", help="comma-separated scenario ids, or 'all'")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "eval_results" / "latest"))
    ap.add_argument("--max-analyses", type=int, default=DEFAULT_MAX_ANALYSES)
    ap.add_argument("--sweep", action="store_true", help="also run the product_regression effect-size x noise sweep")
    ap.add_argument("--sweep-runs", type=int, default=3)
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens (enables cost)")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--estimate", action="store_true", help="print the estimated LLM call count and exit")
    ap.add_argument("--save-prompts", action="store_true")
    args = ap.parse_args(argv)

    names = sorted(SCENARIO_REGISTRY) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",")]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        print(f"Unknown scenario(s): {unknown}", file=sys.stderr)
        return 2
    cells = len(DEFAULT_EFFECTS) * len(DEFAULT_NOISES) if args.sweep else 0
    est = estimate_llm_calls(args.runs, len(names), cells, args.sweep_runs, args.max_analyses)
    print(f"Estimated LLM calls: ~{est['calls_typical']} typical, <= {est['calls_max']} maximum "
          f"({est['runs']} runs x {est['calls_per_run_typical']} typical / {est['calls_per_run_max']} max calls per run)")
    if args.estimate:
        return 0

    try:
        inner = resolve_client(args.provider, args.model, args.allow_mock)
    except EvalConfigError as e:
        print(f"REFUSING TO RUN: {e}", file=sys.stderr)
        return 3
    is_mock = args.provider == "mock"
    price = PriceTable(args.price_in, args.price_out) if args.price_in is not None and args.price_out is not None else None

    def factory():
        return make_mock_client() if is_mock else (inner, None)

    records, scores = [], []
    for name in names:
        truth = load_truth(args.data_dir, name)
        for r in range(args.runs):
            rec = run_scenario_dir(Path(args.data_dir), name, r, factory, provider=args.provider, model=args.model,
                                   allow_mock=args.allow_mock, price=price, max_analyses=args.max_analyses)
            sc = score_run(rec, truth)
            records.append(rec); scores.append(sc)
            print(f"[{name} #{r}] outcome={sc['outcome']} supported={sc['supported']} correct={sc['run_correct']} calls={sc['llm_calls']}")

    sweep = sweep_md = sweep_chart = None
    if args.sweep:
        sweep = run_effect_sweep(DEFAULT_EFFECTS, DEFAULT_NOISES, args.sweep_runs, factory, provider=args.provider,
                                 model=args.model, allow_mock=args.allow_mock, price=price, max_analyses=args.max_analyses)
        sweep_md, sweep_chart = sweep_markdown(sweep, is_mock), sweep_svg(sweep)
    cfg = {"provider": args.provider, "model": args.model, "runs": args.runs, "scenarios": names, "max_analyses": args.max_analyses}
    paths = write_results(Path(args.out), cfg, records, scores, sweep, sweep_md, sweep_chart, args.save_prompts)
    print("Wrote:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
