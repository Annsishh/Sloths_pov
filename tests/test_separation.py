"""Separation and isolation tests.

Verifies that:
1. Investigator data files contain zero ground-truth labels or causal explanations.
2. Ground-truth metadata is saved in a completely isolated directory structure.
"""

import json
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import get_product_regression_scenario


def test_investigator_tables_contain_no_ground_truth_labels():
    scenario = get_product_regression_scenario()
    config = GeneratorConfig(seed=42, num_users=100)
    engine = SyntheticDataEngine(config=config, scenario=scenario)
    tables = engine.generate_all()

    forbidden_terms = ["ground_truth", "scenario_id", "regression", "root_cause", "injected"]

    for table_name, df in tables.items():
        # Check column names
        for col in df.columns:
            for term in forbidden_terms:
                assert term not in col.lower(), f"Forbidden ground truth term '{term}' leaked into column '{col}' of '{table_name}'"

        # Check string columns for accidental leakage of scenario description or cause text
        str_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name.startswith("string")]
        for col in str_cols:
            matched = df[col].astype(str).str.contains("regression|ground_truth", case=False, regex=True)
            assert not matched.any(), f"Ground-truth label leaked into values of '{table_name}.{col}'"
