"""Hypothesis categories, legacy migration, and derived TESTABLE/UNTESTABLE classification."""

from types import SimpleNamespace

import pytest

from analytics.loader import CATALOGUE
from controller.capability_catalogue import CONTROLLED_OPERATIONS
from investigation import HypothesisCategory, assess_testability, normalize_category

REQUIRED = {
    "code_regression", "feature_or_experiment_effect", "config_or_flag_change", "performance_degradation",
    "pricing_or_policy_change", "traffic_mix_shift", "bot_or_invalid_traffic", "tracking_loss",
    "pipeline_or_definition_change", "seasonality_or_calendar", "third_party_outage", "market_or_competitor",
    "gradual_or_cohort_effect", "denominator_effect", "natural_variation", "multiple_causes",
}


def test_all_required_categories_exist_plus_placeholders():
    values = {c.value for c in HypothesisCategory}
    assert REQUIRED <= values
    assert values - REQUIRED == {"unspecified", "external_unspecified"}


@pytest.mark.parametrize("legacy,expected", [
    ("product_regression", "code_regression"),
    ("tracking_failure", "tracking_loss"),
    ("acquisition_shift", "traffic_mix_shift"),
    ("pricing_change", "pricing_or_policy_change"),
    ("user_behavior", "gradual_or_cohort_effect"),
    ("other", "unspecified"),
    ("external_factor", "external_unspecified"),
])
def test_legacy_category_migration(legacy, expected):
    assert normalize_category(legacy).value == expected
    assert normalize_category(legacy.upper()).value == expected  # case-insensitive


def test_ambiguous_legacy_names_do_not_become_specific_claims():
    assert normalize_category("other") != HypothesisCategory.NATURAL_VARIATION
    assert normalize_category("external_factor") != HypothesisCategory.THIRD_PARTY_OUTAGE


def test_unknown_category_becomes_unspecified():
    assert normalize_category("gremlins") == HypothesisCategory.UNSPECIFIED
    assert normalize_category("") == HypothesisCategory.UNSPECIFIED


def test_state_normalizes_category_and_keeps_original_in_history(make_state):
    state = make_state()
    h = state.add_hypothesis("Release bug", "A release broke checkout.", "product_regression")
    assert h.category == "code_regression"
    evt = state.investigation.history[-1]
    assert evt.metadata["original_category"] == "product_regression"
    assert evt.metadata["category"] == "code_regression"


@pytest.fixture
def full_assessment():
    return assess_testability(CONTROLLED_OPERATIONS, CATALOGUE.keys())


def test_every_category_has_an_assessment(full_assessment):
    assert set(full_assessment) == {c.value for c in HypothesisCategory}


def test_testable_set_with_current_catalogue_and_all_tables(full_assessment):
    testable = {k for k, v in full_assessment.items() if v.testable}
    assert testable == {
        "code_regression", "feature_or_experiment_effect", "traffic_mix_shift", "tracking_loss",
        "third_party_outage", "denominator_effect", "natural_variation",
        # Phase 2a: seasonal_baseline and retention_comparison are now exposed through the catalogue
        "seasonality_or_calendar", "gradual_or_cohort_effect",
    }


def test_untestable_categories_say_what_data_is_missing(full_assessment):
    for cat in ("config_or_flag_change", "performance_degradation", "pricing_or_policy_change",
                "bot_or_invalid_traffic", "pipeline_or_definition_change", "market_or_competitor"):
        ta = full_assessment[cat]
        assert not ta.testable and ta.data_request and ta.missing_capabilities
    # Phase 2a: these two became testable once their tools were exposed.
    assert full_assessment["seasonality_or_calendar"].provided_by == ["seasonal_baseline"]
    assert full_assessment["gradual_or_cohort_effect"].provided_by == ["retention_comparison"]


def test_composite_and_placeholder_categories_are_never_testable(full_assessment):
    for cat in ("multiple_causes", "unspecified", "external_unspecified"):
        assert not full_assessment[cat].testable
        assert full_assessment[cat].reason


def test_testability_is_derived_from_available_tables():
    no_releases = [t for t in CATALOGUE if t != "releases"]
    ta = assess_testability(CONTROLLED_OPERATIONS, no_releases)
    assert not ta["code_regression"].testable
    assert ta["code_regression"].missing_capabilities == ["release_before_after"]
    assert ta["tracking_loss"].testable  # unaffected


def test_testability_is_derived_from_the_catalogue_not_hard_coded():
    extra = dict(CONTROLLED_OPERATIONS)
    extra["latency_probe"] = SimpleNamespace(requires_tables=["events"], provides=["latency_metrics"])
    assert not assess_testability(CONTROLLED_OPERATIONS, CATALOGUE.keys())["performance_degradation"].testable
    ta = assess_testability(extra, CATALOGUE.keys())["performance_degradation"]
    assert ta.testable and ta.provided_by == ["latency_probe"]


def test_catalogue_operations_declare_real_tables_and_capabilities():
    for name, spec in CONTROLLED_OPERATIONS.items():
        assert spec.provides, name
        # statistical_test works on caller-supplied numbers and reads no table
        assert spec.requires_tables or "adhoc_statistical_test" in spec.provides, name
        assert set(spec.requires_tables) <= set(CATALOGUE), name
