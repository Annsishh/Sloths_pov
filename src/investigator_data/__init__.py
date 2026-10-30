"""Investigator Data Package."""

from investigator_data.config import GeneratorConfig
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import (
    SCENARIO_REGISTRY,
    ScenarioDefinition,
    get_acquisition_shift_scenario,
    get_baseline_scenario,
    get_product_regression_scenario,
    get_tracking_failure_scenario,
)
from investigator_data.validator import DataValidator

__all__ = [
    "GeneratorConfig",
    "SyntheticDataEngine",
    "DataValidator",
    "ScenarioDefinition",
    "SCENARIO_REGISTRY",
    "get_baseline_scenario",
    "get_product_regression_scenario",
    "get_acquisition_shift_scenario",
    "get_tracking_failure_scenario",
]

