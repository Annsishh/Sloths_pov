"""Reproducibility test suite.

Ensures that identical seeds and configurations produce bit-for-bit identical datasets.
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import SCENARIO_REGISTRY


def test_reproducibility_baseline():
    config1 = GeneratorConfig(seed=12345, num_users=100)
    config2 = GeneratorConfig(seed=12345, num_users=100)

    engine1 = SyntheticDataEngine(config=config1)
    engine2 = SyntheticDataEngine(config=config2)

    tables1 = engine1.generate_all()
    tables2 = engine2.generate_all()

    for table_name in tables1.keys():
        pd.testing.assert_frame_equal(
            tables1[table_name],
            tables2[table_name],
            check_exact=True,
            obj=f"Table {table_name} mismatch across identical seeds",
        )


def test_different_seeds_produce_different_data():
    config1 = GeneratorConfig(seed=100, num_users=100)
    config2 = GeneratorConfig(seed=200, num_users=100)

    engine1 = SyntheticDataEngine(config=config1)
    engine2 = SyntheticDataEngine(config=config2)

    tables1 = engine1.generate_all()
    tables2 = engine2.generate_all()

    # Signups or user distributions should differ
    assert not tables1["users"].equals(tables2["users"])

