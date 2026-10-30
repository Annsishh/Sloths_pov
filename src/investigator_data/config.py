"""Global configuration defaults for the synthetic data generator."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class GeneratorConfig:
    """Simulation parameters. Every realism knob is seeded and reproducible.

    Knobs
    -----
    noise_level               day-to-day lognormal traffic noise (sd of log multiplier); 0 = none.
    weekly_seasonality        strength of the weekday pattern (0 = flat, 1 = weekends at ~30% of weekdays).
    holiday_effects           ((iso_date, traffic_multiplier), ...) calendar spikes/dips.
    effect_size               overrides the scenario's primary effect size (failure/drop probability, conversion cut).
    effect_onset              "step" or "ramp" for scenario effects; ramp length in `effect_ramp_days`.
    late_arrival_rate         share of events that arrive late; those arriving after the extraction cut-off are missing.
    duplicate_event_rate      share of events logged twice (new event_id, identical content).
    bot_traffic_share         share of ALL users that are bots (HeadlessChrome, many page views, never purchase).
    identity_split_share      share of people who appear as two user_ids (two devices).
    event_renames             ((from_version, old_event_name, new_event_name), ...) rename events from a release on.
    app_adoption_days_mobile  mean days a mobile user takes to update to a new release (0 = instant).
    existing_user_share       share of users who signed up BEFORE the window (stationary user base). 0 = legacy world where
                              every user signs up inside the window, so activity grows ~linearly (confounds pre/post).
    """
    seed: int = 42
    start_time: datetime = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    num_days: int = 30
    num_users: int = 1000
    baseline_events_per_user_day: float = 3.5
    baseline_checkout_start_prob: float = 0.12
    baseline_order_conversion_rate: float = 0.08
    baseline_feedback_rate: float = 0.05
    default_currency: str = "USD"

    # --- Phase 3 realism knobs (defaults describe the standard world) ---
    noise_level: float = 0.0
    weekly_seasonality: float = 0.5
    holiday_effects: Tuple[Tuple[str, float], ...] = ()
    effect_size: Optional[float] = None
    effect_onset: str = "step"
    effect_ramp_days: float = 7.0
    late_arrival_rate: float = 0.0
    late_arrival_mean_hours: float = 12.0
    duplicate_event_rate: float = 0.0
    bot_traffic_share: float = 0.0
    identity_split_share: float = 0.0
    event_renames: Tuple[Tuple[str, str, str], ...] = ()
    app_adoption_days_mobile: float = 2.0
    existing_user_share: float = 0.0

    def __post_init__(self) -> None:
        if self.effect_onset not in ("step", "ramp"):
            raise ValueError("effect_onset must be 'step' or 'ramp'")
        for name in ("late_arrival_rate", "duplicate_event_rate", "bot_traffic_share", "identity_split_share", "existing_user_share"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.noise_level < 0 or not 0.0 <= self.weekly_seasonality <= 1.0:
            raise ValueError("noise_level must be >= 0 and weekly_seasonality in [0, 1]")
        if self.effect_ramp_days <= 0 or self.app_adoption_days_mobile < 0:
            raise ValueError("effect_ramp_days must be positive and app_adoption_days_mobile non-negative")
        if self.effect_size is not None and not 0.0 <= self.effect_size <= 1.0:
            raise ValueError("effect_size must be in [0, 1]")

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "GeneratorConfig":
        """Return a copy with scenario-level world overrides applied (lists become tuples)."""
        if not overrides:
            return self
        clean = {k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                 for k, v in overrides.items()}
        return replace(self, **clean)
