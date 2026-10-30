"""Tests for time series and period-over-period comparisons."""

import numpy as np
import pandas as pd
import pytest

from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.time_series import compare_periods, compute_time_series


@pytest.fixture
def sample_orders_and_users():
    users_df = pd.DataFrame([
        {"user_id": f"u{i}", "country": "US" if i % 2 == 0 else "CA", "plan": "pro"}
        for i in range(10)
    ])

    # Orders across 10 days (2026-09-01 to 2026-09-10)
    orders = []
    # 5 orders per day in Period 1 (Sept 1 to Sept 5), each $100
    for day in range(1, 6):
        for o in range(5):
            orders.append({
                "order_id": f"o_{day}_{o}",
                "user_id": f"u{o}",
                "timestamp": pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z"),
                "total_amount": 100.0,
                "status": "completed",
            })
    # 10 orders per day in Period 2 (Sept 6 to Sept 10), each $50
    for day in range(6, 11):
        for o in range(10):
            orders.append({
                "order_id": f"o_{day}_{o}",
                "user_id": f"u{o % 10}",
                "timestamp": pd.Timestamp(f"2026-09-{day:02d}T10:00:00Z"),
                "total_amount": 50.0,
                "status": "completed" if o % 2 == 0 else "refunded",
            })

    orders_df = pd.DataFrame(orders)
    return orders_df, users_df


def test_compute_time_series_daily_counts(sample_orders_and_users):
    orders_df, users_df = sample_orders_and_users
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-06T00:00:00Z")

    req = AnalysisRequest(
        operation="time_series",
        window=window,
        params={"aggregation": "count", "freq": "D"},
    )
    res = compute_time_series(orders_df, users_df, req)
    assert res.ok()
    df = res.data
    # 5 days in window (Sept 1 through Sept 5)
    assert len(df) == 5
    assert (df["metric_value"] == 5).all()


def test_compute_time_series_sum(sample_orders_and_users):
    orders_df, users_df = sample_orders_and_users
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-06T00:00:00Z")

    req = AnalysisRequest(
        operation="time_series",
        window=window,
        params={"aggregation": "sum", "metric_column": "total_amount", "freq": "D"},
    )
    res = compute_time_series(orders_df, users_df, req)
    assert res.ok()
    df = res.data
    assert (df["metric_value"] == 500.0).all()


def test_compare_periods_pooled_counts_and_sums(sample_orders_and_users):
    orders_df, users_df = sample_orders_and_users
    # Baseline: Sept 1 to Sept 6 (5 days, 25 orders, $2500 total)
    base_win = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-06T00:00:00Z")
    # Comparison: Sept 6 to Sept 11 (5 days, 50 orders, $2500 total)
    comp_win = TimeWindow.from_iso("2026-09-06T00:00:00Z", "2026-09-11T00:00:00Z")

    # 1. Compare Counts
    req_cnt = AnalysisRequest(
        operation="period_comparison",
        window=comp_win,
        comparison_window=base_win,
        params={"aggregation": "count"},
    )
    res_cnt = compare_periods(orders_df, users_df, req_cnt)
    assert res_cnt.ok()
    df_cnt = res_cnt.data
    assert df_cnt.loc[df_cnt["period"] == "baseline", "metric_value"].iloc[0] == 25
    assert df_cnt.loc[df_cnt["period"] == "comparison", "metric_value"].iloc[0] == 50
    assert df_cnt.loc[df_cnt["period"] == "change", "metric_value"].iloc[0] == 25
    assert df_cnt.loc[df_cnt["period"] == "change", "relative_change"].iloc[0] == 1.0  # +100%

    # 2. Compare Means (Baseline mean = 100, Comp mean = 50)
    req_mean = AnalysisRequest(
        operation="period_comparison",
        window=comp_win,
        comparison_window=base_win,
        params={"aggregation": "mean", "metric_column": "total_amount"},
    )
    res_mean = compare_periods(orders_df, users_df, req_mean)
    assert res_mean.ok()
    df_mean = res_mean.data
    assert df_mean.loc[df_mean["period"] == "baseline", "metric_value"].iloc[0] == 100.0
    assert df_mean.loc[df_mean["period"] == "comparison", "metric_value"].iloc[0] == 50.0
    assert df_mean.loc[df_mean["period"] == "change", "metric_value"].iloc[0] == -50.0
    assert df_mean.loc[df_mean["period"] == "change", "relative_change"].iloc[0] == -0.5  # -50%


def test_compare_periods_rates(sample_orders_and_users):
    orders_df, users_df = sample_orders_and_users
    # Baseline: 25 completed out of 25 (rate 1.0)
    base_win = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-06T00:00:00Z")
    # Comp: 25 completed out of 50 (rate 0.5)
    comp_win = TimeWindow.from_iso("2026-09-06T00:00:00Z", "2026-09-11T00:00:00Z")

    req = AnalysisRequest(
        operation="period_comparison",
        window=comp_win,
        comparison_window=base_win,
        params={
            "aggregation": "rate",
            "metric_column": "status",
            "metric_value": "completed",
        },
    )
    res = compare_periods(orders_df, users_df, req)
    assert res.ok()
    df = res.data
    assert df.loc[df["period"] == "baseline", "metric_value"].iloc[0] == 1.0
    assert df.loc[df["period"] == "comparison", "metric_value"].iloc[0] == 0.5
    assert df.loc[df["period"] == "change", "metric_value"].iloc[0] == -0.5
    assert df.loc[df["period"] == "change", "relative_change"].iloc[0] == -0.5


def test_compare_periods_missing_comparison_window(sample_orders_and_users):
    orders_df, users_df = sample_orders_and_users
    win = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-06T00:00:00Z")
    req = AnalysisRequest(
        operation="period_comparison",
        window=win,
        comparison_window=None,
    )
    res = compare_periods(orders_df, users_df, req)
    assert res.status == AnalysisStatus.INVALID_PARAMS

