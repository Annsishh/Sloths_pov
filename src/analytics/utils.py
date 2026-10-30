"""Shared filtering, join, and validation utilities used across the analytics layer.

All analytical operations work with timezone-aware datetime columns.
apply_time_filter() and apply_filters() provide the canonical entry-points
for scoping DataFrames to a TimeWindow and filter predicates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analytics.models import AnalysisResult, AnalysisStatus, TimeWindow


# ---------------------------------------------------------------------------
# Time filtering — half-open interval [start, end)
# ---------------------------------------------------------------------------

def apply_time_filter(
    df: pd.DataFrame,
    column: str,
    window: TimeWindow,
) -> pd.DataFrame:
    """Return rows where ``window.start <= df[column] < window.end``.

    Ensures parsed timezone-aware UTC datetime comparison.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame. Available: {list(df.columns)}")

    if df.empty:
        return df.copy()

    ts_series = df[column]
    if not pd.api.types.is_datetime64_any_dtype(ts_series):
        ts_series = pd.to_datetime(ts_series, utc=True)

    mask = (ts_series >= window.start) & (ts_series < window.end)
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Dimension / category filtering
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply equality / membership filters to a DataFrame.

    Parameters
    ----------
    filters : dict mapping column name to a single value, list of values, or None.
    """
    if not filters or df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            raise KeyError(f"Filter column '{col}' not found in DataFrame columns: {list(df.columns)}")
        if val is None:
            mask &= df[col].isna()
        elif isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(val)
        else:
            mask &= df[col] == val
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Safe join helpers (no column collisions, no denominator distortion)
# ---------------------------------------------------------------------------

def enrich_with_user_attrs(
    df: pd.DataFrame,
    users_df: pd.DataFrame,
    columns: Sequence[str],
    how: str = "left",
) -> pd.DataFrame:
    """Left-join user attribute columns onto a table that contains user_id.

    Guarantees:
    1. Users table is deduplicated on user_id so row counts in df are never multiplied.
    2. Only columns that are in users_df AND NOT already in df are joined,
       preventing duplicate suffix columns (_x, _y).
    """
    if "user_id" not in df.columns:
        raise KeyError("DataFrame must contain 'user_id' to enrich with user attributes")
    if "user_id" not in users_df.columns:
        raise KeyError("users_df must contain 'user_id'")

    if df.empty:
        return df.copy()

    # Only join attributes that don't already exist in df
    cols_to_join = [c for c in columns if c in users_df.columns and c not in df.columns]
    if not cols_to_join:
        return df.copy()

    subset = users_df[["user_id"] + cols_to_join].drop_duplicates(subset=["user_id"])
    return df.merge(subset, on="user_id", how=how)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

SUPPORTED_AGGREGATIONS = {"count", "sum", "mean", "rate", "std", "median"}

def validate_dimensions(df: pd.DataFrame, dimensions: List[str]) -> Optional[str]:
    """Return error message if any requested dimension is missing, else None."""
    missing = [d for d in dimensions if d not in df.columns]
    if missing:
        return f"Requested dimensions not found in data: {missing}. Available: {list(df.columns)}"
    return None


def validate_aggregation(agg: str) -> Optional[str]:
    """Return error message if aggregation is unsupported, else None."""
    if agg not in SUPPORTED_AGGREGATIONS:
        return f"Unsupported aggregation '{agg}'. Supported: {sorted(SUPPORTED_AGGREGATIONS)}"
    return None


def validate_numeric_column(df: pd.DataFrame, column: str) -> Optional[str]:
    """Verify that column exists and is numeric."""
    if column not in df.columns:
        return f"Column '{column}' not found in data."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numeric (dtype: {df[column].dtype})."
    return None


def check_min_sample(
    n: int,
    min_sample: int,
    context: str = "",
) -> List[str]:
    """Return warning list if sample size is below threshold."""
    if n < min_sample:
        ctx = f" ({context})" if context else ""
        return [f"Sample size {n} is below minimum {min_sample}{ctx}; result may have high statistical variance."]
    return []
