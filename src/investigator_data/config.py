"""Global configuration defaults for the synthetic data generator."""

from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass(frozen=True)
class GeneratorConfig:
    """Default simulation parameters."""
    seed: int = 42
    start_time: datetime = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    num_days: int = 30
    num_users: int = 1000
    baseline_events_per_user_day: float = 3.5
    baseline_checkout_start_prob: float = 0.12
    baseline_order_conversion_rate: float = 0.08
    baseline_feedback_rate: float = 0.05
    default_currency: str = "USD"

