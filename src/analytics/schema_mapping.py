"""Schema mapping: translate a real company's table, column, event and value names into the internal concepts.

The analytics layer, the capability catalogue and the prompts speak ONE vocabulary (orders, events, users, `order_completed`,
status `completed`, ...). A mapping file says how a company's warehouse names correspond to it, e.g. `purchase_success`
-> `order_completed`. `MappedBackend` applies the mapping while loading, so nothing downstream depends on any company's -
or the synthetic generator's - names.

Mapping file (JSON):
{
  "tables":      {"events": "raw.web_events", "orders": "billing.payments"},           # internal -> external table
  "columns":     {"events": {"user_id": "customer_id", "timestamp": "event_ts"}},      # internal -> external column
  "event_names": {"order_completed": "purchase_success"},                               # internal -> external event name
  "values":      {"orders": {"status": {"completed": "SUCCESS", "failed": "DECLINED"}}} # internal -> external value
}
Tables that are not mapped keep their internal name; tables absent from the raw source are simply unavailable (testability
is then derived from what remains).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import pandas as pd

from analytics.backend import DERIVED_SESSIONS, DataBackend, derive_sessions
from analytics.loader import CATALOGUE, DataLoader
from analytics.path_guard import assert_investigator_path

REQUIRED_COLUMNS = {name: [f.name for f in info.fields if not f.nullable] for name, info in CATALOGUE.items()}


class SchemaMappingError(ValueError):
    """The mapping file is malformed or does not fit the raw data."""


@dataclass
class SchemaMapping:
    tables: Dict[str, str] = field(default_factory=dict)
    columns: Dict[str, Dict[str, str]] = field(default_factory=dict)
    event_names: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SchemaMapping":
        unknown = set(d) - {"tables", "columns", "event_names", "values", "comment"}
        if unknown:
            raise SchemaMappingError(f"Unknown mapping section(s): {sorted(unknown)}")
        m = cls(tables=dict(d.get("tables", {})), columns={k: dict(v) for k, v in d.get("columns", {}).items()},
                event_names=dict(d.get("event_names", {})),
                values={t: {c: dict(v) for c, v in cols.items()} for t, cols in d.get("values", {}).items()})
        m.validate_shape()
        return m

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SchemaMapping":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            raise SchemaMappingError(f"Mapping file is not valid JSON: {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"tables": self.tables, "columns": self.columns, "event_names": self.event_names, "values": self.values}

    # ------------------------------------------------------------------ validation
    def validate_shape(self) -> None:
        errs: List[str] = []
        for internal in list(self.tables) + list(self.columns) + list(self.values):
            if internal not in CATALOGUE:
                errs.append(f"unknown internal table '{internal}'")
        for section, data in (("columns", self.columns), ("values", self.values)):
            for t, cols in data.items():
                if t in CATALOGUE:
                    bad = [c for c in cols if c not in CATALOGUE[t].field_names()]
                    if bad:
                        errs.append(f"table '{t}' has no internal column(s) {bad} ({section} section)")
        for label, d in (("tables", self.tables), ("event_names", self.event_names)):
            ext = list(d.values())
            if len(set(ext)) != len(ext):
                errs.append(f"'{label}' maps two internal names to the same external name")
        for t, cols in self.columns.items():
            ext = list(cols.values())
            if len(set(ext)) != len(ext):
                errs.append(f"table '{t}': two internal columns map to the same external column")
        if errs:
            raise SchemaMappingError("; ".join(errs))

    def external_table(self, internal: str) -> str:
        return self.tables.get(internal, internal)

    # ------------------------------------------------------------------ apply
    def to_internal(self, internal_table: str, raw: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and translate event names / values of one raw table into the internal vocabulary."""
        col_map = self.columns.get(internal_table, {})
        missing = [ext for ext in col_map.values() if ext not in raw.columns]
        if missing:
            raise SchemaMappingError(f"raw table for '{internal_table}' lacks mapped column(s) {missing}")
        df = raw.rename(columns={ext: internal for internal, ext in col_map.items()}).copy()
        if internal_table == "events" and self.event_names and "event_name" in df.columns:
            ext_to_int = {ext: internal for internal, ext in self.event_names.items()}
            df["event_name"] = df["event_name"].map(lambda v: ext_to_int.get(v, v))
        for col, vmap in self.values.get(internal_table, {}).items():
            if col in df.columns:
                inv = {ext: internal for internal, ext in vmap.items()}
                df[col] = df[col].map(lambda v: inv.get(v, v))
        absent = [c for c in REQUIRED_COLUMNS[internal_table] if c not in df.columns]
        if absent:
            raise SchemaMappingError(f"after mapping, table '{internal_table}' lacks required column(s) {absent}")
        return df[[c for c in CATALOGUE[internal_table].field_names() if c in df.columns]]

    def validate_against(self, raw_tables: Mapping[str, pd.DataFrame]) -> List[str]:
        """Dry-run the mapping on raw tables; returns human-readable problems (empty = fits)."""
        problems: List[str] = []
        for internal in CATALOGUE:
            ext = self.external_table(internal)
            if ext in raw_tables:
                try:
                    self.to_internal(internal, raw_tables[ext])
                except SchemaMappingError as e:
                    problems.append(str(e))
        return problems


# ---------------------------------------------------------------------------
# Raw sources and the mapped backend
# ---------------------------------------------------------------------------

class FrameSource:
    """Raw tables held in memory, keyed by EXTERNAL table name."""

    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        self.frames = dict(frames)

    def tables(self) -> List[str]:
        return list(self.frames)

    def read(self, external: str) -> pd.DataFrame:
        return self.frames[external]


class CsvDirectorySource:
    """Raw CSV files `<external_table>.csv` in an investigator-accessible directory."""

    def __init__(self, directory: Union[str, Path]) -> None:
        assert_investigator_path(directory)
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.dir}")

    def tables(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.csv"))

    def read(self, external: str) -> pd.DataFrame:
        path = self.dir / f"{external}.csv"
        assert_investigator_path(path)
        return pd.read_csv(path)


class MappedBackend(DataBackend):
    """DataBackend over a raw source plus a SchemaMapping: presents the internal vocabulary to the analytics layer."""

    def __init__(self, source: Any, mapping: SchemaMapping) -> None:
        self.source, self.mapping = source, mapping
        self._cache: Dict[str, pd.DataFrame] = {}

    def available_tables(self) -> List[str]:
        raw = set(self.source.tables())
        return sorted(t for t in CATALOGUE if self.mapping.external_table(t) in raw)

    def load(self, table_name: str) -> pd.DataFrame:
        if table_name in self._cache:
            return self._cache[table_name]
        if table_name == DERIVED_SESSIONS:
            self._cache[table_name] = derive_sessions(self.load("events"))
            return self._cache[table_name]
        if table_name not in CATALOGUE:
            raise KeyError(f"Unknown table '{table_name}'. Available: {list(CATALOGUE)}")
        if table_name not in self.available_tables():
            raise FileNotFoundError(f"Table '{table_name}' (external '{self.mapping.external_table(table_name)}') is not in the source.")
        df = self.mapping.to_internal(table_name, self.source.read(self.mapping.external_table(table_name)))
        df = DataLoader._parse_timestamps(df, CATALOGUE[table_name].timestamp_columns)
        self._cache[table_name] = df
        return df

    def clear_cache(self) -> None:
        self._cache.clear()
