"""Tests for AnalyticsExecutor dispatcher, catalogue discovery, and custom query execution."""

import pandas as pd
import pytest

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow


@pytest.fixture
def sample_executor():
    users_df = pd.DataFrame([
        {"user_id": "u1", "country": "US", "signup_timestamp": pd.Timestamp("2026-09-01T10:00:00Z")},
        {"user_id": "u2", "country": "CA", "signup_timestamp": pd.Timestamp("2026-09-02T10:00:00Z")},
    ])
    orders_df = pd.DataFrame([
        {"order_id": "o1", "user_id": "u1", "amount": 100.0, "status": "completed", "timestamp": pd.Timestamp("2026-09-03T10:00:00Z")},
        {"order_id": "o2", "user_id": "u2", "amount": 50.0, "status": "completed", "timestamp": pd.Timestamp("2026-09-03T11:00:00Z")},
    ])
    loader = DataLoader.from_dataframes({"users": users_df, "orders": orders_df})
    return AnalyticsExecutor(loader)


def test_executor_discover_sources(sample_executor):
    sources = sample_executor.discover_sources()
    assert "users" in sources
    assert "orders" in sources
    assert "fields" in sources["users"]
    assert "user_id" in sources["users"]["fields"]


def test_executor_dispatch_metric(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="metric_aggregate",
        window=window,
        params={"metric_name": "gross_revenue"},
    )
    res = sample_executor.run(req)
    assert res.ok()
    df = res.data
    assert df["value"].iloc[0] == 150.0


def test_executor_custom_query_select(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="custom_query",
        window=window,
        params={"query": "SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id ORDER BY user_id"},
    )
    res = sample_executor.run(req)
    assert res.ok()
    df = res.data
    assert len(df) == 2
    assert df.loc[df["user_id"] == "u1", "total"].iloc[0] == 100.0


def test_executor_custom_query_rejects_mutation(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="custom_query",
        window=window,
        params={"query": "DROP TABLE orders"},
    )
    res = sample_executor.run(req)
    assert res.status == AnalysisStatus.INVALID_PARAMS
    assert "security policy" in res.error_message


def test_executor_unknown_operation(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="unsupported_op",
        window=window,
    )
    res = sample_executor.run(req)
    assert res.status == AnalysisStatus.INVALID_PARAMS
    assert "Unsupported analytical operation" in res.error_message


def test_executor_custom_query_rejects_multiple_statements(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="custom_query",
        window=window,
        params={"query": "SELECT * FROM users; SELECT * FROM orders;"},
    )
    res = sample_executor.run(req)
    assert res.status == AnalysisStatus.INVALID_PARAMS
    assert "Multiple statements" in res.error_message


def test_executor_custom_query_cte_support(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="custom_query",
        window=window,
        params={"query": "WITH active AS (SELECT user_id FROM orders WHERE status='completed') SELECT COUNT(*) as cnt FROM active"},
    )
    res = sample_executor.run(req)
    assert res.ok()
    assert res.data["cnt"].iloc[0] == 2


def test_executor_custom_query_bounds_rows(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="custom_query",
        window=window,
        params={"query": "SELECT * FROM orders", "max_rows": 1},
    )
    res = sample_executor.run(req)
    assert res.ok()
    assert len(res.data) == 1
    assert any("Result truncated" in w for w in res.warnings)


def test_executor_custom_query_timeout(sample_executor):
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    # Recursive runaway query with small timeout
    req = AnalysisRequest(
        operation="custom_query",
        window=window,
        params={
            "query": "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r) SELECT MAX(i) FROM r",
            "timeout_seconds": 0.2,
        },
    )
    res = sample_executor.run(req)
    assert res.status == AnalysisStatus.ERROR
    assert "timed out" in res.error_message
