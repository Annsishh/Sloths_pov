"""Tests for the controlled analytics capability catalogue and parameter validation."""

import pytest

from controller.capability_catalogue import (
    CONTROLLED_OPERATIONS,
    describe_capabilities,
    validate_analysis_plan,
)


def test_capability_catalogue_contains_expected_operations():
    assert "tracking_gap_check" in CONTROLLED_OPERATIONS
    assert "funnel" in CONTROLLED_OPERATIONS
    assert "release_impact" in CONTROLLED_OPERATIONS
    assert "exposure_effect" in CONTROLLED_OPERATIONS
    assert "segment" in CONTROLLED_OPERATIONS
    assert "error_analysis" in CONTROLLED_OPERATIONS

    # Prohibited operations must NOT be exposed
    assert "custom_query" not in CONTROLLED_OPERATIONS
    assert "raw_sql" not in CONTROLLED_OPERATIONS
    assert "execute" not in CONTROLLED_OPERATIONS


def test_validate_analysis_plan_success():
    # Valid tracking_gap_check
    is_valid, err = validate_analysis_plan(
        operation="tracking_gap_check",
        params={"breakdown": ["browser"], "grace_period_seconds": 60},
        dimensions=["browser"],
    )
    assert is_valid is True
    assert err is None

    # Valid funnel
    is_valid, err = validate_analysis_plan(
        operation="funnel",
        params={"funnel_steps": ["checkout_started", "order_completed"]},
    )
    assert is_valid is True
    assert err is None


def test_validate_analysis_plan_rejects_unpermitted_operation():
    is_valid, err = validate_analysis_plan(
        operation="custom_query",
        params={"query": "SELECT * FROM orders"},
    )
    assert is_valid is False
    assert "not permitted" in err


def test_validate_analysis_plan_rejects_missing_required_params():
    # funnel requires funnel_steps
    is_valid, err = validate_analysis_plan(
        operation="funnel",
        params={},
    )
    assert is_valid is False
    assert "requires parameter 'funnel_steps'" in err


def test_validate_analysis_plan_rejects_invalid_param_type():
    # grace_period_seconds must be int
    is_valid, err = validate_analysis_plan(
        operation="tracking_gap_check",
        params={"grace_period_seconds": "not_an_int"},
    )
    assert is_valid is False
    assert "invalid type" in err


def test_validate_analysis_plan_rejects_unsupported_dimension():
    is_valid, err = validate_analysis_plan(
        operation="tracking_gap_check",
        params={},
        dimensions=["unsupported_dimension_xyz"],
    )
    assert is_valid is False
    assert "not supported" in err


def test_describe_capabilities():
    desc = describe_capabilities()
    assert "Available Analytical Capabilities:" in desc
    assert "tracking_gap_check" in desc
    assert "funnel" in desc
