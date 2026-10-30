"""Data loading layer and source catalogue for the analytics package.

DataLoader
----------
Loads investigator-accessible CSV tables with consistent, timezone-aware datetime parsing.
Caches tables in-memory to avoid redundant I/O. Supports constructing directly from
in-memory DataFrames for unit testing with small controlled datasets.

DataCatalogue
-------------
Describes available tables, fields, known relationships, and time coverage.
The future investigation controller uses this to determine which analyses the data supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Table metadata
# ---------------------------------------------------------------------------

@dataclass
class FieldInfo:
    name: str
    dtype: str              # "string" | "datetime" | "float" | "int" | "category"
    description: str
    nullable: bool = False


@dataclass
class TableInfo:
    """Catalogue entry for one data table."""

    name: str
    filename: str
    description: str
    primary_key: str
    timestamp_columns: List[str]
    fields: List[FieldInfo]
    foreign_keys: Dict[str, str] = field(default_factory=dict)   # field → table.field
    notes: str = ""

    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]


# ---------------------------------------------------------------------------
# Static catalogue definition
# ---------------------------------------------------------------------------

CATALOGUE: Dict[str, TableInfo] = {
    "users": TableInfo(
        name="users",
        filename="users.csv",
        description="One row per user. Contains acquisition channel, device, browser, plan tier.",
        primary_key="user_id",
        timestamp_columns=["signup_timestamp"],
        fields=[
            FieldInfo("user_id", "string", "Unique user identifier"),
            FieldInfo("signup_timestamp", "datetime", "UTC timestamp when user signed up"),
            FieldInfo("acquisition_channel", "category", "Channel through which the user was acquired"),
            FieldInfo("country", "category", "Two-letter ISO country code"),
            FieldInfo("device_type", "category", "desktop | mobile | tablet"),
            FieldInfo("browser", "category", "Chrome | Safari | Firefox | Edge"),
            FieldInfo("plan_tier", "category", "starter | pro | enterprise"),
        ],
        notes="Each user appears exactly once. Use signup_timestamp to define cohorts.",
    ),
    "events": TableInfo(
        name="events",
        filename="events.csv",
        description=(
            "Telemetry event stream. Each row is a single user action within a session. "
            "Contains page views, feature clicks, checkout funnel events, and order outcomes."
        ),
        primary_key="event_id",
        timestamp_columns=["timestamp"],
        foreign_keys={"user_id": "users.user_id"},
        fields=[
            FieldInfo("event_id", "string", "Unique event identifier"),
            FieldInfo("user_id", "string", "FK → users.user_id"),
            FieldInfo("session_id", "string", "Groups events within a single session"),
            FieldInfo("timestamp", "datetime", "UTC event time"),
            FieldInfo("event_name", "category",
                      "page_view | feature_click | checkout_started | checkout_abandoned | "
                      "checkout_failed | order_completed"),
            FieldInfo("page", "category", "Page where the event occurred"),
            FieldInfo("device_type", "category", "Denormalised from user session context"),
            FieldInfo("browser", "category", "Denormalised from user session context"),
            FieldInfo("app_version", "category", "App/web version active during the session"),
        ],
        notes=(
            "order_completed events may be subject to client-side tracking failures. "
            "Always cross-reference with the orders table for financial accuracy."
        ),
    ),
    "orders": TableInfo(
        name="orders",
        filename="orders.csv",
        description=(
            "Backend payment records. Each row is a payment attempt. "
            "Status is 'completed' or 'failed'. Not subject to client-side tracking issues."
        ),
        primary_key="order_id",
        timestamp_columns=["timestamp"],
        foreign_keys={"user_id": "users.user_id"},
        fields=[
            FieldInfo("order_id", "string", "Unique order identifier"),
            FieldInfo("user_id", "string", "FK → users.user_id"),
            FieldInfo("timestamp", "datetime", "UTC time the payment was attempted"),
            FieldInfo("amount", "float", "Payment amount in default currency"),
            FieldInfo("currency", "string", "ISO currency code"),
            FieldInfo("status", "category", "completed | failed"),
            FieldInfo("failure_reason", "string", "Non-null only for failed orders", nullable=True),
        ],
        notes="Source of truth for revenue. Correlate with events for funnel analysis.",
    ),
    "user_feedback": TableInfo(
        name="user_feedback",
        filename="user_feedback.csv",
        description="User-submitted feedback. Each row is one feedback submission.",
        primary_key="feedback_id",
        timestamp_columns=["timestamp"],
        foreign_keys={"user_id": "users.user_id"},
        fields=[
            FieldInfo("feedback_id", "string", "Unique feedback identifier"),
            FieldInfo("user_id", "string", "FK → users.user_id"),
            FieldInfo("timestamp", "datetime", "UTC submission time"),
            FieldInfo("rating", "int", "Integer 1–5 star rating"),
            FieldInfo("category", "category", "billing | product | general"),
            FieldInfo("comment", "string", "Free-text feedback", nullable=True),
        ],
    ),
    "releases": TableInfo(
        name="releases",
        filename="releases.csv",
        description="Product release log. Each row is one deployment.",
        primary_key="release_id",
        timestamp_columns=["deployed_timestamp"],
        fields=[
            FieldInfo("release_id", "string", "Unique release identifier"),
            FieldInfo("version", "string", "Semantic version string"),
            FieldInfo("deployed_timestamp", "datetime", "UTC deployment time"),
            FieldInfo("component", "category", "Product component affected"),
            FieldInfo("description", "string", "Human-readable release notes"),
            FieldInfo("status", "category", "success | failed | rolled_back"),
        ],
        notes="Cross-reference with metric time series to identify correlated changes.",
    ),
    "feature_exposures": TableInfo(
        name="feature_exposures",
        filename="feature_exposures.csv",
        description=(
            "A/B experiment exposure log. Each row records the first time "
            "a user was assigned to a feature flag variant."
        ),
        primary_key="exposure_id",
        timestamp_columns=["first_exposed_timestamp"],
        foreign_keys={"user_id": "users.user_id"},
        fields=[
            FieldInfo("exposure_id", "string", "Unique exposure identifier"),
            FieldInfo("user_id", "string", "FK → users.user_id"),
            FieldInfo("feature_key", "category", "Name of the feature flag"),
            FieldInfo("variant", "category", "control | treatment"),
            FieldInfo("first_exposed_timestamp", "datetime", "UTC time of first exposure"),
        ],
        notes=(
            "Only first exposure is recorded per (user, feature_key). "
            "Exposure does not guarantee interaction with the feature."
        ),
    ),
    "daily_product_metrics": TableInfo(
        name="daily_product_metrics",
        filename="daily_product_metrics.csv",
        description=(
            "Pre-aggregated daily product metrics. Useful for quick trend inspection. "
            "Always verify against raw event and order tables for investigation conclusions."
        ),
        primary_key="date",
        timestamp_columns=["date"],
        fields=[
            FieldInfo("date", "datetime", "Calendar date (UTC)"),
            FieldInfo("daily_active_users", "int", "Users with at least one event on this date"),
            FieldInfo("new_signups", "int", "Users whose signup_timestamp falls on this date"),
            FieldInfo("total_events", "int", "Total telemetry events on this date"),
            FieldInfo("completed_orders", "int", "Backend completed orders on this date"),
            FieldInfo("failed_orders", "int", "Backend failed orders on this date"),
            FieldInfo("gross_revenue", "float", "Sum of completed order amounts"),
            FieldInfo("avg_feedback_rating", "float", "Average user feedback rating"),
        ],
        notes=(
            "Pre-aggregated. Does not support segment-level breakdown. "
            "Use raw events and orders tables for segmented analysis."
        ),
    ),
}


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Loads investigator-accessible CSV tables with consistent datetime parsing.

    Parameters
    ----------
    data_dir : str or Path, optional
        Path to the `investigator_data` directory.
    tables : Dict[str, pd.DataFrame], optional
        Pre-loaded in-memory tables (useful for unit tests with controlled datasets).
    eager : bool
        If True and data_dir provided, load all known tables on construction.
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        tables: Optional[Dict[str, pd.DataFrame]] = None,
        eager: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else None
        if self.data_dir is not None and not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._cache: Dict[str, pd.DataFrame] = {}
        self.catalogue: Dict[str, TableInfo] = CATALOGUE

        # If in-memory tables provided, parse and cache them
        if tables is not None:
            for name, df in tables.items():
                ts_cols = CATALOGUE[name].timestamp_columns if name in CATALOGUE else []
                self._cache[name] = self._parse_timestamps(df.copy(), ts_cols)

        if eager and self.data_dir is not None:
            for table_name in CATALOGUE:
                self.load(table_name)

    @classmethod
    def from_dataframes(cls, tables: Dict[str, pd.DataFrame]) -> "DataLoader":
        """Factory method to construct DataLoader directly from a dictionary of DataFrames."""
        return cls(tables=tables)

    # ------------------------------------------------------------------ core
    def load(self, table_name: str) -> pd.DataFrame:
        """Return the requested table as a DataFrame.

        Datetime columns are parsed to UTC-aware timestamps.
        Results are cached; subsequent calls are O(1).
        """
        if table_name in self._cache:
            return self._cache[table_name]

        if table_name not in CATALOGUE:
            raise KeyError(
                f"Unknown table '{table_name}'. Available: {list(CATALOGUE.keys())}"
            )

        if self.data_dir is None:
            raise FileNotFoundError(
                f"Table '{table_name}' not in cache and no data_dir configured."
            )

        info = CATALOGUE[table_name]
        csv_path = self.data_dir / info.filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected table file not found: {csv_path}"
            )

        df = pd.read_csv(csv_path)
        df = self._parse_timestamps(df, info.timestamp_columns)
        self._cache[table_name] = df
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load and return all available tables as a dict."""
        available = self.available_tables()
        return {name: self.load(name) for name in available}

    # ------------------------------------------------------------------ inspection helpers
    def available_tables(self) -> List[str]:
        """Return names of tables available either in memory or on disk."""
        if self.data_dir is None:
            return list(self._cache.keys())
        tables_on_disk = [
            name
            for name, info in CATALOGUE.items()
            if (self.data_dir / info.filename).exists()
        ]
        return sorted(list(set(tables_on_disk) | set(self._cache.keys())))

    def table_info(self, table_name: str) -> TableInfo:
        if table_name not in CATALOGUE:
            raise KeyError(f"Unknown table: {table_name}")
        return CATALOGUE[table_name]

    def time_coverage(self, table_name: str) -> Dict[str, datetime]:
        """Return {start, end} of the timestamp range for a table."""
        df = self.load(table_name)
        ts_cols = CATALOGUE[table_name].timestamp_columns if table_name in CATALOGUE else []
        result = {}
        for col in ts_cols:
            if col in df.columns and not df[col].dropna().empty:
                result[f"{col}_min"] = df[col].min()
                result[f"{col}_max"] = df[col].max()
        return result

    def row_count(self, table_name: str) -> int:
        return len(self.load(table_name))

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ private
    @staticmethod
    def _parse_timestamps(df: pd.DataFrame, ts_cols: List[str]) -> pd.DataFrame:
        for col in ts_cols:
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], utc=True)
                else:
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize("UTC")
                    else:
                        df[col] = df[col].dt.tz_convert("UTC")
        return df


# ---------------------------------------------------------------------------
# DataCatalogue
# ---------------------------------------------------------------------------

class DataCatalogue:
    """Describes the available data sources for a given scenario.

    The investigation controller uses this to discover which tables,
    fields, and relationships are available before constructing analyses.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def describe(self) -> Dict[str, Dict]:
        """Return a structured description of every available table."""
        result = {}
        for name in self._loader.available_tables():
            info = CATALOGUE.get(name)
            if info:
                result[name] = {
                    "description": info.description,
                    "primary_key": info.primary_key,
                    "fields": {f.name: {"dtype": f.dtype, "description": f.description} for f in info.fields},
                    "timestamp_columns": info.timestamp_columns,
                    "foreign_keys": info.foreign_keys,
                    "notes": info.notes,
                }
        return result

    def relationships(self) -> List[Dict[str, str]]:
        """Return list of FK relationships as {from_table, from_field, to_table, to_field}."""
        rels = []
        for name, info in CATALOGUE.items():
            for field_name, ref in info.foreign_keys.items():
                to_table, to_field = ref.split(".")
                rels.append({
                    "from_table": name,
                    "from_field": field_name,
                    "to_table": to_table,
                    "to_field": to_field,
                })
        return rels

    def supports_analysis(self, required_tables: List[str]) -> bool:
        """Return True if all required tables are available."""
        available = set(self._loader.available_tables())
        return all(t in available for t in required_tables)
