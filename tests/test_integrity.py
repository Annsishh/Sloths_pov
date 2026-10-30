"""Referential integrity, temporal consistency, funnel stages, and boundary tests."""

from datetime import datetime, timezone
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.validator import DataValidator


def test_referential_and_temporal_integrity():
    config = GeneratorConfig(seed=42, num_users=300)
    engine = SyntheticDataEngine(config=config)
    tables = engine.generate_all()

    validator = DataValidator(tables=tables, config=config)
    is_valid, errors = validator.validate_all()

    assert is_valid, f"Validation failed with errors: {errors}"
    assert len(errors) == 0


def test_no_orphan_records():
    config = GeneratorConfig(seed=777, num_users=200)
    engine = SyntheticDataEngine(config=config)
    tables = engine.generate_all()

    user_ids = set(tables["users"]["user_id"])

    for entity in ["events", "orders", "user_feedback", "feature_exposures"]:
        df = tables[entity]
        if not df.empty:
            assert set(df["user_id"]).issubset(user_ids), f"Orphan records found in {entity}"


def test_simulation_window_boundaries():
    """Verify that all timestamps across all tables are within [start_time, end_time)

    Uses parsed datetimes and verifies against generator's actual configured simulation window
    with an exclusive upper boundary.
    """
    config = GeneratorConfig(
        seed=101,
        num_users=300,
        start_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
        num_days=30,
    )
    engine = SyntheticDataEngine(config=config)
    tables = engine.generate_all()

    validator = DataValidator(tables=tables, config=config)
    boundary_errors = validator.check_simulation_window_bounds()
    assert len(boundary_errors) == 0, f"Boundary errors detected: {boundary_errors}"


def test_checkout_funnel_independence():
    """Verify checkout funnel:
    1. checkout_started events can occur without leading to an order (abandonment).
    2. checkout attempts can fail (e.g. status='failed') independently.
    3. checkout_abandoned events occur when users drop off before payment attempt.
    """
    config = GeneratorConfig(seed=42, num_users=500)
    engine = SyntheticDataEngine(config=config)
    tables = engine.generate_all()

    events_df = tables["events"]
    orders_df = tables["orders"]

    checkout_starts = events_df[events_df["event_name"] == "checkout_started"]
    completed_orders = orders_df[orders_df["status"] == "completed"]
    failed_orders = orders_df[orders_df["status"] == "failed"]

    # Checkout starts should strictly exceed completed orders
    assert len(checkout_starts) > len(completed_orders), (
        f"Checkout starts ({len(checkout_starts)}) must exceed completed orders ({len(completed_orders)})"
    )

    # Some checkout starts should abandon
    abandoned_events = events_df[events_df["event_name"] == "checkout_abandoned"]
    assert len(abandoned_events) > 0, "Expected abandoned checkout events"

    # Both completed and failed orders should be represented in baseline
    assert len(failed_orders) > 0, "Expected some baseline order failures (e.g. card declined)"
    assert len(completed_orders) > 0, "Expected completed orders in baseline"
