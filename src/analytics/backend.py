"""The data-backend interface the analytics layer depends on.

Every analytical operation, the executor, the controller's data context and the SQL sandbox read data ONLY through
this interface. The default implementation is the pandas/CSV `DataLoader`; a warehouse backend (BigQuery, Snowflake)
would implement the same few methods - it only has to return the canonical tables as DataFrames, in the internal
vocabulary (see analytics.loader.CATALOGUE). No warehouse backend is provided here: this module only fixes the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

import pandas as pd

DERIVED_SESSIONS = "sessions"


def derive_sessions(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per session built from events: first-event time, user/device/browser/version, and whether the session
    contained a checkout start / a tracked order_completed. Lets rate tools use sessions as the unit."""
    cols = ["session_id", "user_id", "timestamp", "device_type", "browser", "app_version", "checkout_started", "converted"]
    if ev.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    g = ev.sort_values("timestamp").groupby("session_id", sort=False)
    sess = g.agg(user_id=("user_id", "first"), timestamp=("timestamp", "first"), device_type=("device_type", "first"),
                 browser=("browser", "first"), app_version=("app_version", "first")).reset_index()
    flags = ev.assign(cs=ev["event_name"] == "checkout_started", cv=ev["event_name"] == "order_completed") \
              .groupby("session_id").agg(checkout_started=("cs", "any"), converted=("cv", "any")).reset_index()
    return sess.merge(flags, on="session_id")[cols]


class DataBackend(ABC):
    """Read-only access to the investigator-accessible tables, in the internal vocabulary.

    Contract:
      * `load(name)` returns a DataFrame with the columns declared in analytics.loader.CATALOGUE, timestamp columns as
        timezone-aware UTC datetimes. Callers must not mutate it.
      * `load("sessions")` returns the derived session table (see derive_sessions); it is not listed by available_tables().
      * Implementations must only expose investigator-accessible data (never scenario labels or injected parameters).
    """

    @abstractmethod
    def load(self, table_name: str) -> pd.DataFrame: ...

    @abstractmethod
    def available_tables(self) -> List[str]: ...

    # --- conveniences with default implementations (a backend may override them with cheaper, pushed-down versions)
    def load_all(self) -> Dict[str, pd.DataFrame]:
        return {name: self.load(name) for name in self.available_tables()}

    def row_count(self, table_name: str) -> int:
        return len(self.load(table_name))

    def time_coverage(self, table_name: str) -> Dict[str, datetime]:
        from analytics.loader import CATALOGUE
        df = self.load(table_name)
        out: Dict[str, datetime] = {}
        for col in (CATALOGUE[table_name].timestamp_columns if table_name in CATALOGUE else []):
            if col in df.columns and not df[col].dropna().empty:
                out[f"{col}_min"], out[f"{col}_max"] = df[col].min(), df[col].max()
        return out

    def clear_cache(self) -> None:
        """Drop any cached tables (no-op for backends that do not cache)."""
