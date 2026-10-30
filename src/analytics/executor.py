"""Central analytics execution engine and composable dispatcher.

AnalyticsExecutor coordinates access to the data layer, validates analysis requests,
dispatches them to deterministic analytical modules, and returns standardized AnalysisResult
objects.

Provides an extensibility mechanism for custom analyses using in-memory SQLite
with strict read-only query validation.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from analytics.changepoint import change_point
from analytics.spikes import spike_check
from analytics.decomposition import mix_shift_decomposition
from analytics.ratio_metrics import ratio_metric_comparison
from analytics.seasonality import seasonal_baseline
from analytics.data_quality import duplicate_event_check, event_volume_check, tracking_gap_check
from analytics.feedback_analysis import analyze_errors, analyze_feedback
from analytics.funnel import compare_funnel, compute_funnel
from analytics.loader import DataCatalogue, DataLoader
from analytics.metrics import compute_metric
from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.release_analysis import exposure_effect, release_did, release_impact, sample_ratio_check
from analytics.retention import compare_retention, compute_retention
from analytics.sql_sandbox import SQLPolicyError, SQLSandbox
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
        request = self._normalise(request, op)

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

            elif op == "change_point":
                primary_table = request.params.get("table", "orders")
                res = change_point(self.loader.load(primary_table), self.loader.load("users"), request)

            elif op == "spike_check":
                primary_table = request.params.get("table", "orders")
                res = spike_check(self.loader.load(primary_table), self.loader.load("users"), request)

            elif op == "ratio_metric_comparison":
                res = ratio_metric_comparison(self.loader.load("orders"), self.loader.load("events"), request)

            elif op == "release_did":
                primary_table = request.params.get("table", "orders")
                res = release_did(self.loader.load("releases"), self.loader.load("events"),
                                  self.loader.load(primary_table), self.loader.load("users"), request)

            elif op == "sample_ratio_check":
                res = sample_ratio_check(self.loader.load("feature_exposures"), request)

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

    # Operations whose tools group by `request.dimensions`; the catalogue exposes the same idea as params["breakdown"].
    _DIMENSION_OPS = {"funnel", "funnel_comparison", "segment", "segment_comparison"}

    @classmethod
    def _normalise(cls, request: AnalysisRequest, op: str) -> AnalysisRequest:
        """Reconcile catalogue parameter names with what the tools read (funnel_steps -> steps, breakdown -> dimensions)."""
        params = dict(request.params)
        dims = list(request.dimensions)
        if op in ("funnel", "funnel_comparison") and "steps" not in params and "funnel_steps" in params:
            params["steps"] = params["funnel_steps"]
        if op in cls._DIMENSION_OPS and not dims and isinstance(params.get("breakdown"), list):
            dims = list(params["breakdown"])
        if params == request.params and dims == request.dimensions:
            return request
        from dataclasses import replace
        return replace(request, params=params, dimensions=dims)

    # ------------------------------------------------------------------ extensibility
    def register_operation(
        self,
        name: str,
        handler: Callable[[AnalysisRequest], AnalysisResult],
    ) -> None:
        """Register a new composable analytical operation handler."""
        self._custom_handlers[name.lower()] = handler

    def _sql_sandbox(self) -> SQLSandbox:
        """The read-only database file for the current tables (rebuilt if the loaded tables changed)."""
        tables = {name: self.loader.load(name) for name in self.loader.available_tables()}
        signature = tuple((n, len(df)) for n, df in sorted(tables.items()))
        if getattr(self, "_sandbox", None) is None or self._sandbox_signature != signature:
            if getattr(self, "_sandbox", None) is not None:
                self._sandbox.close()
            self._sandbox, self._sandbox_signature = SQLSandbox(tables), signature
        return self._sandbox

    def _execute_sql_query(self, request: AnalysisRequest) -> AnalysisResult:
        """Execute a validated, read-only SQL query (see analytics.sql_sandbox for the layers of protection)."""
        def reply(status, error=None, **kw):
            return AnalysisResult(request_id=request.request_id, operation="custom_query", status=status,
                                  data=kw.pop("data", None), error_message=error, **kw)

        raw_query = request.params.get("query", "")
        try:
            max_rows = int(request.params.get("max_rows", 5000))
            timeout_sec = float(request.params.get("timeout_seconds", 5.0))
        except (TypeError, ValueError):
            return reply(AnalysisStatus.INVALID_PARAMS, "max_rows and timeout_seconds must be numbers.")
        try:
            df, warnings = self._sql_sandbox().run(raw_query, max_rows=max_rows, timeout_sec=timeout_sec)
        except SQLPolicyError as e:
            return reply(AnalysisStatus.INVALID_PARAMS, str(e))
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                return reply(AnalysisStatus.ERROR, f"Query execution timed out after {timeout_sec}s.")
            return reply(AnalysisStatus.ERROR, f"SQL error: {e}")
        except Exception as e:
            return reply(AnalysisStatus.ERROR, f"SQL error: {e}")
        return reply(
            AnalysisStatus.OK, data=df,
            metadata={"query": raw_query, "row_count": len(df), "window": str(request.window), "timeout_seconds": timeout_sec},
            limitations=["Custom SQL execution; verify query logic independently."], warnings=warnings,
        )
