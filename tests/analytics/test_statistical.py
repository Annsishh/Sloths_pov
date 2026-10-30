"""Tests for statistical hypothesis testing (proportions and means)."""

import numpy as np
import pytest

from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.statistical import compare_means, compare_proportions, run_statistical_test


def test_compare_proportions_known_values():
    # 50/100 vs 30/100
    res = compare_proportions(k1=50, n1=100, k2=30, n2=100)
    assert res["rate_sample_1"] == 0.50
    assert res["rate_sample_2"] == 0.30
    assert res["absolute_difference"] == 0.20
    assert res["z_statistic"] == pytest.approx(2.8868, abs=1e-3)
    assert res["p_value"] == pytest.approx(0.00389, abs=1e-4)
    assert res["is_significant_p05"] is True
    assert res["is_significant_p01"] is True
    # 95% CI lower/upper
    assert res["ci_lower_95"] == pytest.approx(0.0671, abs=1e-3)
    assert res["ci_upper_95"] == pytest.approx(0.3329, abs=1e-3)
    assert len(res["warnings"]) == 0


def test_compare_proportions_small_sample_warning():
    # k2 = 2 (< 5) -> triggers warning
    res = compare_proportions(k1=20, n1=100, k2=2, n2=100)
    assert len(res["warnings"]) > 0
    assert "Low expected count" in res["warnings"][0]


def test_compare_proportions_invalid_inputs():
    with pytest.raises(ValueError, match="Sample sizes must be positive"):
        compare_proportions(k1=5, n1=0, k2=5, n2=10)

    with pytest.raises(ValueError, match="Success counts must be between 0 and n"):
        compare_proportions(k1=15, n1=10, k2=5, n2=10)


def test_compare_means_welch():
    # Sample 1: [10, 12, 14, 16, 18] -> mean 14.0, var 10.0
    # Sample 2: [20, 22, 24, 26, 28] -> mean 24.0, var 10.0
    s1 = [10.0, 12.0, 14.0, 16.0, 18.0]
    s2 = [20.0, 22.0, 24.0, 26.0, 28.0]

    res = compare_means(s1, s2)
    assert res["mean_sample_1"] == 14.0
    assert res["mean_sample_2"] == 24.0
    assert res["absolute_difference"] == -10.0
    assert res["t_statistic"] == pytest.approx(-5.0, abs=1e-2)
    assert res["p_value"] < 0.01
    assert res["is_significant_p05"] is True


def test_run_statistical_test_entrypoint():
    window = TimeWindow.from_iso("2026-09-01T00:00:00Z", "2026-09-10T00:00:00Z")
    req = AnalysisRequest(
        operation="statistical_test",
        window=window,
        params={
            "test_type": "proportions",
            "k1": 40,
            "n1": 100,
            "k2": 20,
            "n2": 100,
        },
    )
    res = run_statistical_test(req)
    assert res.ok()
    df = res.data
    assert df["absolute_difference"].iloc[0] == 0.20
    assert df["is_significant_p05"].iloc[0] == True


def test_compare_means_zero_variance_identical():
    s1 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    s2 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    res = compare_means(s1, s2)
    assert res["absolute_difference"] == 0.0
    assert res["t_statistic"] == 0.0
    assert res["p_value"] == 1.0
    assert any("Zero variance in both samples" in w for w in res["warnings"])


def test_compare_means_zero_variance_unequal():
    s1 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    s2 = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    res = compare_means(s1, s2)
    assert res["absolute_difference"] == -10.0
    assert res["t_statistic"] is None
    assert res["p_value"] is None
    assert any("unequal means" in w for w in res["warnings"])


def test_compare_means_small_sample_warning():
    # n = 4 < 10
    s1 = [1.0, 2.0, 3.0, 4.0]
    s2 = [2.0, 3.0, 4.0, 5.0]
    res = compare_means(s1, s2)
    assert any("Small sample size" in w for w in res["warnings"])
