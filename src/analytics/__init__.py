"""Analytics package for the AI Product Investigator.

Provides a reusable, composable analytical foundation covering:
  - Data loading and source catalogue
  - Metric definitions and aggregation
  - Retention and cohort analysis
  - Funnel conversion and drop-off
  - Segmentation and population comparisons
  - Time-series and period comparisons
  - Release and feature-exposure analysis
  - Event tracking and data-quality investigation
  - Error analysis and customer feedback analysis
  - Statistical comparisons and uncertainty estimates
  - Composable operation executor

The future investigation controller calls these modules through the
AnalyticsExecutor, passing AnalysisRequest objects and receiving
AnalysisResult objects that include results, metadata, limitations,
and provenance information.
"""

from analytics.loader import DataLoader, DataCatalogue, TableInfo
from analytics.models import (
    AnalysisRequest,
    AnalysisResult,
    MetricDefinition,
    TimeWindow,
    AnalysisStatus,
)
from analytics.executor import AnalyticsExecutor
from analytics.metrics import STANDARD_METRICS, compute_metric
from analytics.retention import compute_retention, compare_retention
from analytics.funnel import compute_funnel, compare_funnel
from analytics.segmentation import segment_metric, compare_segments
from analytics.time_series import compute_time_series, compare_periods
from analytics.release_analysis import release_impact, exposure_effect
from analytics.data_quality import event_volume_check, tracking_gap_check, duplicate_event_check
from analytics.feedback_analysis import analyze_feedback, analyze_errors
from analytics.statistical import compare_proportions, compare_means, run_statistical_test

__all__ = [
    "DataLoader",
    "DataCatalogue",
    "TableInfo",
    "AnalysisRequest",
    "AnalysisResult",
    "MetricDefinition",
    "TimeWindow",
    "AnalysisStatus",
    "AnalyticsExecutor",
    "STANDARD_METRICS",
    "compute_metric",
    "compute_retention",
    "compare_retention",
    "compute_funnel",
    "compare_funnel",
    "segment_metric",
    "compare_segments",
    "compute_time_series",
    "compare_periods",
    "release_impact",
    "exposure_effect",
    "event_volume_check",
    "tracking_gap_check",
    "duplicate_event_check",
    "analyze_feedback",
    "analyze_errors",
    "compare_proportions",
    "compare_means",
    "run_statistical_test",
]
