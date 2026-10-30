"""Central analytics execution engine and composable dispatcher.

AnalyticsExecutor coordinates access to the data layer, validates analysis requests,
dispatches them to deterministic analytical modules, and returns standardized AnalysisResult
objects.

Provides an extensibility mechanism for custom analyses using in-memory SQLite
with strict read-only query validation.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from analytics.decomposition import mix_shift_decomposition
from analytics.seasonality import seasonal_baseline
from analytics.data_quality import duplicate_event_check, event_volume_check, tracking_gap_check
from analytics.feedback_analysis import analyze_errors, analyze_feedback
from analytics.funnel import compare_funnel, compute_funnel
from analytics.loader import DataCatalogue, DataLoader
from analytics.metrics import compute_metric
from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.release_analysis import exposure_effect, release_impact
from analytics.retention import compare_retention, compute_retention
from analytics.segmentation import compare_segments, segment_metric
from analytics.statistical import run_statistical_test
from analytics.time_series import compare_periods, compute_time_series


class AnalyticsExecutor:
    """Executes analytical operations against investigator data."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.catalogue = DataCatalogue(loader)
        self._custom_handlers: Dict[str, Callable[[AnalysisRequest], AnalysisResult]] = {}

    # ------------------------------------------------------------------ source discovery
    def discover_sources(self) -> Dict[str, Any]:
        """Return catalogue of available tables and schemas."""
        return self.catalogue.describe()

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Return a copy of a table from the data loader."""
        return self.loader.load(table_name).copy()

    # ------------------------------------------------------------------ dispatch
    def run(self, request: AnalysisRequest) -> AnalysisResult:
        """Dispatch and execute an analytical request."""
        start_time = time.perf_counter()
        op = request.operation.lower()

        # Check for registered custom extensions
        if op in self._custom_handlers:
            try:
                res = self._custom_handlers[op](request)
                res.metadata["execution_duration_sec"] = round(time.perf_counter() - start_time, 4)
                return res
            except Exception as e:
                return AnalysisResult(
                    request_id=request.request_id,
                    operation=request.operation,
                    status=AnalysisStatus.ERROR,
                    data=None,
                    error_message=f"Custom handler error: {str(e)}",
                )

        try:
            # 1. Retention
            if op == "retention":
                users = self.loader.load("users")
                events = self.loader.load("events")
                res = compute_retention(users, events, request)

            elif op == "retention_comparison":
                users = self.loader.load("users")
                events = self.loader.load("events")
                res = compare_retention(users, events, request)

            # 2. Funnel
            elif op == "funnel":
                events = self.loader.load("events")
                users = self.loader.load("users")
                res = compute_funnel(events, users, request)

            elif op == "funnel_comparison":
                events = self.loader.load("events")
                users = self.loader.load("users")
                res = compare_funnel(events, users, request)

            # 3. Segmentation
            elif op == "segment":
                primary_table = request.params.get("table", "orders")
                primary = self.loader.load(primary_table)
                users = self.loader.load("users")
                res = segment_metric(primary, users, request)

            elif op == "segment_comparison":
                primary_table = request.params.get("table", "orders")
                primary = self.loader.load(primary_table)
                users = self.loader.load("users")
                res = compare_segments(primary, users, request)

            # 4. Time Series & Period Comparison
            elif op == "time_series":
                primary_table = request.params.get("table", "events")
                primary = self.loader.load(primary_table)
                users = self.loader.load("users")
                res = compute_time_series(primary, users, request)

            elif op == "period_comparison":
                primary_table = request.params.get("table", "events")
                primary = self.loader.load(primary_table)
                users = self.loader.load("users")
                res = compare_periods(primary, users, request)

            elif op == "seasonal_baseline":
                primary_table = request.params.get("table", "events")
                res = seasonal_baseline(self.loader.load(primary_table), self.loader.load("users"), request)

            elif op == "mix_shift_decomposition":
                primary_table = request.params.get("table", "orders")
                res = mix_shift_decomposition(self.loader.load(primary_table), self.loader.load("users"), request)

            # 5. Release & Exposure Analysis
            elif op == "release_impact":
                releases = self.loader.load("releases")
                primary_table = request.params.get("table", "orders")
                primary = self.loader.load(primary_table)
                users = self.loader.load("users")
                res = release_impact(releases, primary, users, request)

            elif op == "exposure_effect":
                exposures = self.loader.load("feature_exposures")
                primary_table = request.params.get("table", "events")
                primary = self.loader.load(primary_table)
                users = self.loader.load("users")
                res = exposure_effect(exposures, primary, users, request)

            # 6. Data Quality & Tracking
            elif op == "tracking_gap_check":
                events = self.loader.load("events")
                orders = self.loader.load("orders")
                users = self.loader.load("users")
                res = tracking_gap_check(events, orders, users, request)

            elif op == "event_volume_check":
                events = self.loader.load("events")
                users = self.loader.load("users")
                res = event_volume_check(events, users, request)

            elif op == "duplicate_event_check":
                events = self.loader.load("events")
                res = duplicate_event_check(events, request)

            # 7. Metrics & Feedback
            elif op == "metric_aggregate":
                tables = self.loader.load_all()
                res = compute_metric(tables, request)

            elif op == "feedback_analysis":
                feedback = self.loader.load("user_feedback")
                users = self.loader.load("users")
                res = analyze_feedback(feedback, users, request)

            elif op == "error_analysis":
                orders = self.loader.load("orders")
                users = self.loader.load("users")
                res = analyze_errors(orders, users, request)

            # 8. Statistical Testing
            elif op == "statistical_test":
                res = run_statistical_test(request)

            # 9. Custom Safe SQL Extensibility
            elif op == "custom_query":
                res = self._execute_sql_query(request)

            else:
                return AnalysisResult(
                    request_id=request.request_id,
                    operation=request.operation,
                    status=AnalysisStatus.INVALID_PARAMS,
                    data=None,
                    error_message=f"Unsupported analytical operation: '{request.operation}'.",
                )

            res.metadata["execution_duration_sec"] = round(time.perf_counter() - start_time, 4)
            return res

        except KeyError as e:
            return AnalysisResult(
                request_id=request.request_id,
                operation=request.operation,
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=f"Missing required parameter or key: {str(e)}",
            )
        except Exception as e:
            return AnalysisResult(
                request_id=request.request_id,
                operation=request.operation,
                status=AnalysisStatus.ERROR,
                data=None,
                error_message=f"Execution error: {str(e)}",
            )

    # ------------------------------------------------------------------ extensibility
    def register_operation(
        self,
        name: str,
        handler: Callable[[AnalysisRequest], AnalysisResult],
    ) -> None:
        """Register a new composable analytical operation handler."""
        self._custom_handlers[name.lower()] = handler

    def _execute_sql_query(self, request: AnalysisRequest) -> AnalysisResult:
        """Execute a safe, read-only SQL query against in-memory SQLite tables."""
        raw_query = request.params.get("query", "").strip()
        max_rows = int(request.params.get("max_rows", 5000))
        timeout_sec = float(request.params.get("timeout_seconds", 5.0))

        if not raw_query:
            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message="Parameter 'query' is required for custom_query operation.",
            )

        # Strip SQL comments
        cleaned_query = re.sub(r"--.*?$|/\*.*?\*/", "", raw_query, flags=re.DOTALL | re.MULTILINE).strip()

        # Reject multiple statements (e.g. separated by semicolons)
        statements = [s.strip() for s in cleaned_query.split(";") if s.strip()]
        if len(statements) > 1:
            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message="Query rejected by security policy: Multiple statements are not permitted.",
            )

        target_query = statements[0] if statements else ""
        if not re.match(r"^(SELECT|WITH)\b", target_query, re.IGNORECASE):
            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message="Query rejected by security policy: Only read-only SELECT queries are permitted.",
            )

        # Reject mutation or management keywords
        forbidden = re.compile(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|PRAGMA|VACUUM|EXEC|TRANSACTION|BEGIN|COMMIT|ROLLBACK)\b",
            re.IGNORECASE,
        )
        if forbidden.search(target_query):
            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message="Query rejected by security policy: Only read-only SELECT queries are permitted.",
            )

        # Build in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        try:
            for table_name in self.loader.available_tables():
                df = self.loader.load(table_name)
                # Convert timestamps to ISO strings for SQLite compatibility
                export_df = df.copy()
                for col in export_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(export_df[col]):
                        export_df[col] = export_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                export_df.to_sql(table_name, conn, index=False)

            # SQLite engine-level read-only authorizer
            def _authorizer(action, arg1, arg2, db_name, trigger_name):
                allowed = {
                    sqlite3.SQLITE_SELECT,
                    sqlite3.SQLITE_READ,
                    sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE,
                }
                if action in allowed:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            conn.set_authorizer(_authorizer)

            # Execution time bound via SQLite progress handler
            start_exec = time.perf_counter()
            conn.set_progress_handler(
                lambda: 1 if (time.perf_counter() - start_exec) > timeout_sec else 0,
                500,
            )

            # Bounded memory cursor fetch
            cursor = conn.cursor()
            cursor.execute(target_query)
            fetched_rows = cursor.fetchmany(max_rows + 1)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            if len(fetched_rows) > max_rows:
                fetched_rows = fetched_rows[:max_rows]
                warnings = [f"Result truncated to max_rows={max_rows}."]
            else:
                warnings = []

            result_df = pd.DataFrame(fetched_rows, columns=columns)

            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.OK,
                data=result_df,
                metadata={
                    "query": raw_query,
                    "row_count": len(result_df),
                    "window": str(request.window),
                    "timeout_seconds": timeout_sec,
                },
                limitations=["Custom SQL execution; verify query logic independently."],
                warnings=warnings,
            )
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                return AnalysisResult(
                    request_id=request.request_id,
                    operation="custom_query",
                    status=AnalysisStatus.ERROR,
                    data=None,
                    error_message=f"Query execution timed out after {timeout_sec}s.",
                )
            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.ERROR,
                data=None,
                error_message=f"SQL error: {str(e)}",
            )
        except Exception as e:
            return AnalysisResult(
                request_id=request.request_id,
                operation="custom_query",
                status=AnalysisStatus.ERROR,
                data=None,
                error_message=f"SQL error: {str(e)}",
            )
        finally:
            conn.close()

