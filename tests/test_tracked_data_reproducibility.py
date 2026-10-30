"""The tracked datasets under data/ must equal a fresh regeneration from their seed.

Fails if the generator, its dependencies, or the tracked files drift apart.
Regenerate the tracked files with: python generate_data.py --all --output-dir data
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import SCENARIO_REGISTRY

DATA = Path(__file__).resolve().parents[1] / "data"
SEED, USERS = 42, 1000  # generate_data.py defaults used for the tracked files


@pytest.mark.parametrize("scenario", sorted(SCENARIO_REGISTRY))
def test_tracked_dataset_matches_regeneration(scenario):
    base = DATA / scenario
    if not base.is_dir():
        pytest.fail(f"tracked dataset missing for scenario '{scenario}'")
    scen = SCENARIO_REGISTRY[scenario]()
    tables = SyntheticDataEngine(GeneratorConfig(seed=SEED, num_users=USERS), scen).generate_all()

    for name, fresh in tables.items():
        on_disk = pd.read_csv(base / "investigator_data" / f"{name}.csv")
        # compare exactly as written: round-trip the fresh frame through CSV the same way the generator did
        roundtrip = pd.read_csv(pd.io.common.StringIO(fresh.to_csv(index=False)))
        assert_frame_equal(on_disk, roundtrip, check_exact=False, rtol=1e-9, obj=f"{scenario}/{name}")

    gt = json.loads((base / "ground_truth" / "scenario_ground_truth.json").read_text(encoding="utf-8"))
    assert gt == json.loads(json.dumps(scen.to_ground_truth_dict()))
