"""Read-only SQL sandbox for the custom_query extension.

Defence in depth, strongest layer first:
  1. DATABASE LEVEL: tables are exported to a temporary SQLite *file* that is then opened with the URI parameter
     `mode=ro`, so even a statement that slipped past every other check cannot modify anything (SQLite refuses with
     "attempt to write a readonly database"). `PRAGMA query_only = ON` is set as well.
  2. ENGINE LEVEL: an authorizer callback allows only SELECT/READ/RECURSIVE, whitelisted SQL functions, and reads of the
     exported investigator tables (not sqlite_master or anything else); ATTACH is additionally disabled with
     SQLITE_LIMIT_ATTACHED = 0 and extension loading is off.
  3. TEXT LEVEL: comments are replaced by whitespace (exactly as SQLite tokenises them) and the SAME cleaned text that was
     checked is the text that is executed; string literals are ignored by the keyword scan; one statement only; length cap.
  4. RESOURCE LEVEL: row cap, wall-clock timeout via the progress handler, and SQLite length limits.
Only investigator tables are ever exported, so ground truth cannot be reached through SQL either.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

MAX_QUERY_CHARS = 10_000
HARD_MAX_ROWS = 5_000
HARD_MAX_TIMEOUT = 10.0

ALLOWED_FUNCTIONS = frozenset({
    "count", "sum", "avg", "min", "max", "total", "abs", "round", "coalesce", "ifnull", "nullif", "iif", "length",
    "lower", "upper", "substr", "substring", "trim", "ltrim", "rtrim", "replace", "instr", "strftime", "date", "time",
    "datetime", "julianday", "unixepoch", "typeof", "like", "glob", "group_concat", "sqrt", "floor", "ceil", "ceiling",
    "ln", "log", "log10", "exp", "pow", "power", "sign",
})

# Verbs and objects that can never appear in a read-only query (checked outside string literals).
FORBIDDEN_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|ANALYZE|TRIGGER|"
    r"LOAD_EXTENSION|READFILE|WRITEFILE|FTS3_TOKENIZER|SAVEPOINT|RELEASE|ROLLBACK|COMMIT|BEGIN)\b", re.IGNORECASE)

_TOKEN = re.compile(r"""
    (?P<sq>'(?:[^']|'')*')            # string literal
  | (?P<dq>"(?:[^"]|"")*")            # quoted identifier
  | (?P<bt>`[^`]*`)                   # backtick identifier
  | (?P<br>\[[^\]]*\])                # bracket identifier
  | (?P<lc>--[^\n]*)                  # line comment
  | (?P<bc>/\*.*?(?:\*/|\Z))          # block comment (an unterminated one runs to the end, as in SQLite)
""", re.VERBOSE | re.DOTALL)


class SQLPolicyError(ValueError):
    """The query was rejected by the sandbox policy before execution."""


def clean_sql(raw: str) -> Tuple[str, str]:
    """Return (executable_text, text_for_keyword_scan).

    Comments become a single space, matching SQLite's tokenizer (so `SEL/**/ECT` stays two tokens and is not
    mistaken for SELECT). For the keyword scan, string literals and quoted identifiers are blanked out as well.
    """
    exec_parts: List[str] = []
    scan_parts: List[str] = []
    pos = 0
    for m in _TOKEN.finditer(raw):
        plain = raw[pos:m.start()]
        exec_parts.append(plain)
        scan_parts.append(plain)
        kind = m.lastgroup
        if kind in ("lc", "bc"):
            exec_parts.append(" ")
            scan_parts.append(" ")
        else:
            exec_parts.append(m.group(0))
            scan_parts.append(" " * 2)
        pos = m.end()
    tail = raw[pos:]
    exec_parts.append(tail)
    scan_parts.append(tail)
    return "".join(exec_parts).strip(), "".join(scan_parts).strip()


def validate_query(raw_query: str) -> str:
    """Apply the text-level policy and return the exact text that will be executed."""
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise SQLPolicyError("Parameter 'query' is required for custom_query operation.")
    if len(raw_query) > MAX_QUERY_CHARS:
        raise SQLPolicyError(f"Query rejected by security policy: longer than {MAX_QUERY_CHARS} characters.")
    if "\x00" in raw_query:
        raise SQLPolicyError("Query rejected by security policy: NUL byte.")
    executable, scan = clean_sql(raw_query)
    body = scan.rstrip().rstrip(";").rstrip()
    if ";" in body:
        raise SQLPolicyError("Query rejected by security policy: Multiple statements are not permitted.")
    if not re.match(r"^(SELECT|WITH)\b", body, re.IGNORECASE):
        raise SQLPolicyError("Query rejected by security policy: Only read-only SELECT queries are permitted.")
    m = FORBIDDEN_WORDS.search(body)
    if m:
        raise SQLPolicyError(f"Query rejected by security policy: Only read-only SELECT queries are permitted ('{m.group(1).upper()}' is not allowed).")
    return executable.rstrip().rstrip(";").rstrip()


class SQLSandbox:
    """Builds the read-only database file once per set of tables and runs validated queries against it."""

    def __init__(self, tables: Dict[str, pd.DataFrame]) -> None:
        self._tables = {n: df for n, df in tables.items()}
        self._dir = tempfile.mkdtemp(prefix="sloths_sql_")
        self._finalizer = weakref.finalize(self, shutil.rmtree, self._dir, True)
        self.path = Path(self._dir) / "sandbox.db"
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        rw = sqlite3.connect(str(self.path))
        try:
            for name, df in self._tables.items():
                export = df.copy()
                for col in export.columns:
                    if pd.api.types.is_datetime64_any_dtype(export[col]):
                        export[col] = export[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                export.to_sql(name, rw, index=False)
            rw.commit()
        finally:
            rw.close()

    @property
    def table_names(self) -> List[str]:
        return sorted(self._tables)

    # ------------------------------------------------------------------ connections
    def raw_readonly_connection(self) -> sqlite3.Connection:
        """Connection with ONLY the database-level protection (mode=ro); used by tests to prove it works alone."""
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def _guarded_connection(self, timeout_sec: float) -> sqlite3.Connection:
        conn = self.raw_readonly_connection()
        conn.execute("PRAGMA query_only = ON")
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.NotSupportedError):
            pass
        for limit_name, value in (("SQLITE_LIMIT_ATTACHED", 0), ("SQLITE_LIMIT_LENGTH", 5_000_000),
                                  ("SQLITE_LIMIT_SQL_LENGTH", MAX_QUERY_CHARS * 2), ("SQLITE_LIMIT_COMPOUND_SELECT", 20),
                                  ("SQLITE_LIMIT_FUNCTION_ARG", 16), ("SQLITE_LIMIT_EXPR_DEPTH", 200)):
            if hasattr(sqlite3, limit_name) and hasattr(conn, "setlimit"):
                conn.setlimit(getattr(sqlite3, limit_name), value)
        allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}

        def authorizer(action, arg1, arg2, db_name, trigger):
            if action in allowed_actions:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                # CTE names also arrive here, so block by name: schema tables (sqlite_*) are never readable; the database
                # file itself only contains the exported investigator tables.
                return sqlite3.SQLITE_DENY if (arg1 or "").lower().startswith("sqlite_") else sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_DENY

        conn.set_authorizer(authorizer)
        start = time.perf_counter()
        conn.set_progress_handler(lambda: 1 if (time.perf_counter() - start) > timeout_sec else 0, 500)
        return conn

    # ------------------------------------------------------------------ run
    def run(self, raw_query: str, max_rows: int = 1000, timeout_sec: float = 5.0) -> Tuple[pd.DataFrame, List[str]]:
        query = validate_query(raw_query)
        max_rows = max(1, min(int(max_rows), HARD_MAX_ROWS))
        timeout_sec = max(0.1, min(float(timeout_sec), HARD_MAX_TIMEOUT))
        conn = self._guarded_connection(timeout_sec)
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchmany(max_rows + 1)
            cols = [d[0] for d in cur.description] if cur.description else []
        finally:
            conn.close()
        warnings: List[str] = []
        if len(rows) > max_rows:
            rows = rows[:max_rows]
            warnings.append(f"Result truncated to max_rows={max_rows}.")
        return pd.DataFrame(rows, columns=cols), warnings

    def close(self) -> None:
        self._finalizer()
