"""Shared data models for the analytics layer.

AnalysisRequest  — parameterised description of an analysis to run.
AnalysisResult   — structured result with data, metadata, limitations, and provenance.
MetricDefinition — explicit metric semantics (counting unit, population, denominator, etc.).
TimeWindow       — a half-open interval [start, end) used consistently throughout.
AnalysisStatus   — outcome of executing an analysis.

These models are the contract between the future investigation controller
and the deterministic analytical engine. All analytical operations accept an
AnalysisRequest and return an AnalysisResult.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AnalysisStatus(str, Enum):
    OK = "ok"                        # Analysis ran and returned usable results
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough rows / events for a reliable result
    INVALID_PARAMS = "invalid_params"        # Bad input detected before execution
    DATA_QUALITY = "data_quality"            # Execution revealed a data-quality problem
    ERROR = "error"                          # Unexpected execution error


@dataclass
class TimeWindow:
    """Half-open interval [start, end) expressed in UTC.

    All analytics treat timestamps as UTC ISO-8601 strings and
    filter using  start <= timestamp < end  (exclusive upper bound).
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            raise ValueError("TimeWindow.start must be timezone-aware")
        if self.end.tzinfo is None:
            raise ValueError("TimeWindow.end must be timezone-aware")
        if self.start >= self.end:
            raise ValueError(f"TimeWindow start ({self.start}) must be before end ({self.end})")

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_iso(cls, start: str, end: str) -> "TimeWindow":
        def _parse(raw: str) -> datetime:
            dt = datetime.fromisoformat(raw)
            # Naive / date-only strings are UTC by contract; astimezone() would assume local time.
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

        return cls(start=_parse(start), end=_parse(end))

    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400.0

    def overlaps(self, other: "TimeWindow") -> bool:
        return self.start < other.end and other.start < self.end

    def __str__(self) -> str:
        return f"[{self.start.date()} – {self.end.date()})"


@dataclass
class MetricDefinition:
    """Explicit specification of what a metric measures and how it is computed.

    Every field is required so that the investigation controller and the
    analyst can reason about what the metric *means*, not just what number
    it produces.

    Fields
    ------
    name:             Human-readable metric name.
    description:      Plain-English explanation of business meaning.
    counting_unit:    What each row / event represents (e.g. "user", "session", "order").
    numerator_events: Event names (or table+filter) that contribute to the numerator.
    denominator:      What divides the numerator ("users", "sessions", "orders", or "none").
    eligible_filter:  Column→value pairs that define the eligible population.
    time_column:      Name of the timestamp column used to scope the population.
    min_sample_size:  Minimum n below which the result is flagged as insufficient.
    notes:            Known caveats, data-quality considerations, or assumptions.
    """

    name: str
    description: str
    counting_unit: str
    numerator_events: List[str] = field(default_factory=list)
    denominator: str = "none"   # "users" | "sessions" | "orders" | "events" | "none"
    eligible_filter: Dict[str, Any] = field(default_factory=dict)
    time_column: str = "timestamp"
    min_sample_size: int = 30
    notes: str = ""


@dataclass
class AnalysisRequest:
    """Parameterised description of an analysis to run.

    The investigation controller constructs these and passes them to
    AnalyticsExecutor.run(). All fields are keyword-only to prevent
    positional-argument confusion.

    Fields
    ------
    operation:    Name of the analytical operation (e.g. "retention", "funnel", "segment").
    window:       Primary time window. Operations that compare two periods also accept
                  a comparison_window.
    params:       Operation-specific parameters (see each analytical module).
    dimensions:   Columns to slice results by (e.g. ["device_type", "browser"]).
    filters:      Column→value / column→[values] pairs applied before analysis.
    comparison_window: Optional second period for before/after comparisons.
    metric:       Optional MetricDefinition if the operation needs explicit metric semantics.
    request_id:   Stable identifier; auto-generated if omitted.
    """

    operation: str
    window: TimeWindow
    params: Dict[str, Any] = field(default_factory=dict)
    dimensions: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    comparison_window: Optional[TimeWindow] = None
    metric: Optional[MetricDefinition] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AnalysisResult:
    """Structured output of an analytical operation.

    Includes the actual data, provenance metadata, limitations, and
    enough context to trace every material claim back to its source.
    """

    request_id: str
    operation: str
    status: AnalysisStatus
    data: Optional[Any]                    # DataFrame, dict, list, or scalar — operation-dependent
    metadata: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ helpers
    def ok(self) -> bool:
        return self.status == AnalysisStatus.OK

    def summary(self) -> str:
        lines = [f"[{self.status.value.upper()}] {self.operation}"]
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        if self.limitations:
            lines.append("  Limitations: " + "; ".join(self.limitations))
        if self.warnings:
            lines.append("  Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)

