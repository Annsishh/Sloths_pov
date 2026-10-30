"""CLI script to generate synthetic product datasets for the AI Product Investigator."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import SCENARIO_REGISTRY, build_config
from investigator_data.validator import DataValidator


def generate_scenario_dataset(scenario_key: str, output_base: Path, seed: int, num_users=None) -> None:
    if scenario_key not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario: {scenario_key}. Supported: {list(SCENARIO_REGISTRY.keys())}")

    scenario_def = SCENARIO_REGISTRY[scenario_key]()
    config = build_config(scenario_def, seed=seed, num_users=num_users)

    print(f"\n==================================================")
    print(f"Generating Scenario: [{scenario_key}] - {scenario_def.name}")
    print(f"Seed: {seed} | Users: {config.num_users}")
    print(f"==================================================")

    engine = SyntheticDataEngine(config=config, scenario=scenario_def)
    tables = engine.generate_all()
    config = engine.config

    # Validate generated tables
    validator = DataValidator(tables=tables, scenario=scenario_def, config=config)
    is_valid, errors = validator.validate_all()
    if not is_valid:
        print(f"[!] Validation warnings for {scenario_key}:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("[OK] Data validation passed (referential, temporal, and schema consistency verified).")

    # Strict separation of Investigator-accessible data vs Ground Truth
    scenario_dir = output_base / scenario_key
    investigator_dir = scenario_dir / "investigator_data"
    ground_truth_dir = scenario_dir / "ground_truth"

    investigator_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Investigator-accessible tabular data
    print(f"\nSaving investigator data to: {investigator_dir}")
    for table_name, df in tables.items():
        csv_path = investigator_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  - {table_name}.csv ({len(df):,} rows)")

    # 2. Save Ground Truth evaluation metadata (ISOLATED from investigator)
    print(f"\nSaving ground-truth metadata to: {ground_truth_dir}")
    gt_payload = scenario_def.to_ground_truth_dict()
    gt_payload["measured_effects"] = engine.truth      # counts measured while generating (approximate-share basis)
    gt_payload["generator_config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()
                                      if k not in ("start_time",)}
    gt_file = ground_truth_dir / "scenario_ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_payload, f, indent=2)
    print(f"  - scenario_ground_truth.json (Root cause & injected parameters)")

    # 3. The alert that triggered the investigation: observable-metric spec, NOT ground truth, so it lives beside both folders
    alert = scenario_def.to_alert_dict()
    if alert.get("mode") == "worst_adjacent_week":
        alert = worst_adjacent_week_alert(tables, alert["metric"])
    with open(scenario_dir / "alert.json", "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2)
    print("  - alert.json (metric + windows that fired the alert)")


def worst_adjacent_week_alert(tables, metric):
    """Pick the adjacent 7-day pair with the largest decline in completed orders per day (a seeded, data-driven alert)."""
    import pandas as pd
    orders = tables["orders"]
    ts = pd.to_datetime(orders.loc[orders["status"] == "completed", "timestamp"], utc=True)
    daily = ts.dt.floor("D").value_counts().sort_index()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(days, fill_value=0)
    best, best_drop = None, 0.0
    for i in range(7, len(daily) - 6):
        before, after = daily.iloc[i - 7:i].mean(), daily.iloc[i:i + 7].mean()
        if before > 0 and (before - after) / before > best_drop:
            best, best_drop = i, (before - after) / before
    i = best if best is not None else 7
    f = lambda d: d.strftime("%Y-%m-%d")
    return {"metric": metric, "baseline_window": [f(days[i - 7]), f(days[i])], "comparison_window": [f(days[i]), f(days[min(i + 7, len(days) - 1)] + pd.Timedelta(days=1))]}


def main():
    parser = argparse.ArgumentParser(description="Synthetic SaaS Data Generator for AI Product Investigator")
    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        choices=list(SCENARIO_REGISTRY.keys()),
        help="Specific scenario to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all supported scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory path (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=None,
        help="Number of users to simulate (default: the scenario's own world, normally 1000)",
    )

    args = parser.parse_args()
    output_path = Path(args.output_dir)

    if args.all:
        for scen in SCENARIO_REGISTRY.keys():
            generate_scenario_dataset(scen, output_path, seed=args.seed, num_users=args.users)
    else:
        generate_scenario_dataset(args.scenario, output_path, seed=args.seed, num_users=args.users)

    print("\n[DONE] Data generation complete.")


if __name__ == "__main__":
    main()
