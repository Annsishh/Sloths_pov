"""Tests for DataLoader and DataCatalogue."""

from datetime import datetime, timezone
from pathlib import Path
import pytest
import pandas as pd

from analytics.loader import DataLoader, DataCatalogue, CATALOGUE
from analytics.models import TimeWindow


def test_loader_from_dataframes():
    users_df = pd.DataFrame([
        {"user_id": "u1", "signup_timestamp": "2026-09-01T10:00:00Z", "acquisition_channel": "direct"},
        {"user_id": "u2", "signup_timestamp": "2026-09-02T12:00:00Z", "acquisition_channel": "paid_social"},
    ])
    loader = DataLoader.from_dataframes({"users": users_df})

    loaded = loader.load("users")
    assert len(loaded) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded["signup_timestamp"])
    assert loaded["signup_timestamp"].dt.tz is not None


def test_loader_caching():
    users_df = pd.DataFrame([{"user_id": "u1", "signup_timestamp": "2026-09-01T10:00:00Z"}])
    loader = DataLoader.from_dataframes({"users": users_df})

    df1 = loader.load("users")
    df2 = loader.load("users")
    assert df1 is df2  # cached reference


def test_loader_real_dataset():
    data_dir = Path("C:/Users/ANISHA/OneDrive/Desktop/sloths_pov/data/baseline/investigator_data")
    if not data_dir.exists():
        pytest.skip("Baseline data directory not found")

    loader = DataLoader(data_dir=data_dir)
    tables = loader.available_tables()
    assert "users" in tables
    assert "events" in tables
    assert "orders" in tables

    events_df = loader.load("events")
    assert not events_df.empty
    assert pd.api.types.is_datetime64_any_dtype(events_df["timestamp"])


def test_data_catalogue():
    users_df = pd.DataFrame([{"user_id": "u1", "signup_timestamp": "2026-09-01T10:00:00Z"}])
    loader = DataLoader.from_dataframes({"users": users_df})
    catalogue = DataCatalogue(loader)

    desc = catalogue.describe()
    assert "users" in desc
    assert desc["users"]["primary_key"] == "user_id"
    assert catalogue.supports_analysis(["users"])
    assert not catalogue.supports_analysis(["users", "orders"])

