"""Filter and dimension validation in validate_analysis_plan."""

import pytest

from controller.capability_catalogue import (
    ALLOWED_FILTER_COLUMNS,
    CONTROLLED_OPERATIONS,
    validate_analysis_plan,
)


def _ok(op, params=None, dims=None, filters=None):
    return validate_analysis_plan(op, params or {}, dims, filters)


@pytest.mark.parametrize("filters", [
    {"browser": "Safari"},
    {"device_type": ["mobile", "tablet"]},
    {"status": "failed", "failure_reason": None},
    {"rating": 1},
    {"app_version": ["v2.3.8"]},
])
def test_valid_filters_pass(filters):
    assert _ok("tracking_gap_check", filters=filters) == (True, None)


@pytest.mark.parametrize("filters,fragment", [
    ({"ground_truth_label": "x"}, "not permitted"),
    ({"user_id": "u1"}, "not permitted"),                          # not an allowed filter column
    ({"browser; DROP TABLE": "x"}, "not permitted"),
    ({"browser__ne": "Safari"}, "not permitted"),                  # operator-suffix syntax
    ({"browser": {"$ne": "Safari"}}, "operator syntax"),
    ({"browser": {"op": "in", "value": ["a"]}}, "operator syntax"),
    ({"browser": []}, "empty"),
    ({"browser": ["Safari", {"x": 1}]}, "must be str/int/float/bool"),
    ({"browser": ["v"] * 51}, "at most"),
    ({"browser": object()}, "unsupported value type"),
])
def test_invalid_filters_are_rejected(filters, fragment):
    ok, err = _ok("tracking_gap_check", filters=filters)
    assert ok is False and fragment in err


def test_filter_columns_cover_user_event_and_order_dimensions():
    assert {"browser", "device_type", "acquisition_channel", "app_version", "status"} <= ALLOWED_FILTER_COLUMNS


@pytest.mark.parametrize("op", ["period_comparison", "time_series", "release_impact", "exposure_effect"])
def test_operations_without_allowed_dimensions_reject_dimensions(op):
    params = {"aggregation": "count", "release_id": "r", "feature_key": "f"}
    params = {k: v for k, v in params.items() if k in CONTROLLED_OPERATIONS[op].allowed_params}
    assert CONTROLLED_OPERATIONS[op].allowed_dimensions == []
    ok, err = _ok(op, params, dims=["browser"])
    assert ok is False and "does not support breakdown dimensions" in err
    assert _ok(op, params)[0] is True  # without dimensions the same plan is fine


def test_breakdown_param_is_checked_like_dimensions():
    ok, err = _ok("segment", {"breakdown": ["not_a_dimension"]})
    assert ok is False and "not supported" in err
    ok, err = _ok("tracking_gap_check", {"breakdown": ["country"]})  # valid dimension, but not for this op
    assert ok is False and "not supported" in err
    ok, err = _ok("segment", {"breakdown": [3]})
    assert ok is False and "list of dimension names" in err
    assert _ok("segment", {"breakdown": ["device_type", "country"]})[0] is True


def test_every_catalogue_example_still_validates():
    for name, spec in CONTROLLED_OPERATIONS.items():
        ok, err = validate_analysis_plan(name, spec.example)
        assert ok, f"{name}: {err}"
