"""Funnel conversion and drop-off analysis.

Computes ordered step conversion rates through a sequence of event names.

Methodology
-----------
- Sequential Chronological Funnel: A session (or user) converts through Step 0 -> Step 1 -> ... -> Step K
  if and only if Step 0 occurs, Step 1 occurs at or after Step 0, Step 2 occurs at or after Step 1, etc.
- Scoping: Evaluates events within the specified TimeWindow.
- Grouping unit: Default is session-scoped (one funnel attempt per session_id).
  Can be user-scoped (one attempt per user_id) via user_scoped=True.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.utils import apply_time_filter, apply_filters, enrich_with_user_attrs, check_min_sample


def compute_funnel(
    events_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compute chronological conversion rates through an ordered event funnel.

    Parameters in ``request.params``
    ---------------------------------
    steps : List[str]
        Ordered list of event names defining funnel steps (min 2).
    user_scoped : bool, optional
        If True count unique users; if False (default) count sessions.
    min_sample : int, optional
        Minimum count at the top of funnel (default 10).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    steps: List[str] = request.params.get("steps", [])
    user_scoped: bool = request.params.get("user_scoped", False)
    min_sample: int = request.params.get("min_sample", 10)

    if not isinstance(steps, list) or len(steps) < 2:
        return AnalysisResult(
            request_id=request.request_id,
            operation="funnel",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="funnel requires 'steps' parameter with at least 2 event names.",
        )

    # Scope events to window
    scoped_events = apply_time_filter(events_df, "timestamp", request.window)

    # Safely enrich with user attributes if required by filters or dimensions
    user_cols_needed = [c for c in list(request.filters.keys()) + request.dimensions if c in users_df.columns]
    if user_cols_needed:
        scoped_events = enrich_with_user_attrs(scoped_events, users_df, user_cols_needed)

    try:
        scoped_events = apply_filters(scoped_events, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="funnel",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if scoped_events.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="funnel",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No events in the specified window after applying filters.",
        )

    group_col = "user_id" if user_scoped else "session_id"
    counting_unit = "unique users" if user_scoped else "sessions"

    # Filter to events that are part of the funnel
    relevant_events = scoped_events[scoped_events["event_name"].isin(steps)].copy()
    if relevant_events.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="funnel",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No funnel events found matching the specified steps.",
        )

    # Sort events by timestamp for chronological progression
    relevant_events.sort_values(by="timestamp", inplace=True)

    # Track sequential stage reached by each group
    # A group starts at stage 0 if it has steps[0]. It advances to stage 1 if it has steps[1] at t >= t0, etc.
    group_max_stage: Dict[Any, int] = {}
    group_last_ts: Dict[Any, pd.Timestamp] = {}

    for gid, group in relevant_events.groupby(group_col, sort=False):
        curr_step_idx = 0
        last_t = None

        for _, ev in group.iterrows():
            ev_name = ev["event_name"]
            ev_t = ev["timestamp"]

            if curr_step_idx < len(steps) and ev_name == steps[curr_step_idx]:
                if last_t is None or ev_t >= last_t:
                    curr_step_idx += 1
                    last_t = ev_t
                    if curr_step_idx == len(steps):
                        break  # reached end of funnel

        if curr_step_idx > 0:
            group_max_stage[gid] = curr_step_idx - 1  # 0-indexed highest step completed

    # Compute counts per step
    step_counts = []
    top_count = sum(1 for stage in group_max_stage.values() if stage >= 0)

    for i, step_name in enumerate(steps):
        # Count groups that reached at least step i
        count_at_step = sum(1 for stage in group_max_stage.values() if stage >= i)
        step_counts.append(count_at_step)

    counts_df_rows = []
    for i, step_name in enumerate(steps):
        n = step_counts[i]
        prev_n = step_counts[i - 1] if i > 0 else n

        conv_from_top = round(n / top_count, 4) if top_count > 0 else 0.0
        conv_from_prev = round(n / prev_n, 4) if prev_n > 0 else (1.0 if i == 0 else 0.0)
        drop_from_prev = round(1.0 - conv_from_prev, 4) if i > 0 else 0.0
        drop_count = prev_n - n if i > 0 else 0

        counts_df_rows.append({
            "step_index": i,
            "step_name": step_name,
            "count": n,
            "conversion_from_top": conv_from_top,
            "conversion_from_prev": conv_from_prev,
            "drop_off_from_prev": drop_from_prev,
            "drop_count_from_prev": drop_count,
        })

    result_df = pd.DataFrame(counts_df_rows)
    warnings.extend(check_min_sample(top_count, min_sample, context=f"top step '{steps[0]}'"))

    return AnalysisResult(
        request_id=request.request_id,
        operation="funnel",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "steps": steps,
            "counting_unit": counting_unit,
            "window": str(request.window),
            "top_funnel_count": top_count,
            "total_events_in_window": len(scoped_events),
        },
        limitations=limitations,
        warnings=warnings,
    )


def compare_funnel(
    events_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare funnel conversion across two time windows (baseline vs comparison)."""
    if request.comparison_window is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="funnel_comparison",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="compare_funnel requires request.comparison_window to be set.",
        )

    # Baseline run
    base_req = AnalysisRequest(
        operation="funnel",
        window=request.comparison_window,
        params=request.params,
        filters=request.filters,
        dimensions=request.dimensions,
        request_id=request.request_id + "_baseline",
    )
    base_res = compute_funnel(events_df, users_df, base_req)

    # Comparison run
    comp_res = compute_funnel(events_df, users_df, request)

    if not base_res.ok() or not comp_res.ok():
        return AnalysisResult(
            request_id=request.request_id,
            operation="funnel_comparison",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message=(
                f"Baseline: {base_res.error_message}; "
                f"Comparison: {comp_res.error_message}"
            ),
        )

    b_df = base_res.data
    c_df = comp_res.data

    merged = b_df.merge(
        c_df,
        on=["step_index", "step_name"],
        suffixes=("_baseline", "_comparison"),
    )

    merged["abs_change_conv_from_top"] = (
        merged["conversion_from_top_comparison"] - merged["conversion_from_top_baseline"]
    ).round(4)
    merged["abs_change_conv_from_prev"] = (
        merged["conversion_from_prev_comparison"] - merged["conversion_from_prev_baseline"]
    ).round(4)

    return AnalysisResult(
        request_id=request.request_id,
        operation="funnel_comparison",
        status=AnalysisStatus.OK,
        data=merged,
        metadata={
            "baseline_window": str(request.comparison_window),
            "comparison_window": str(request.window),
            "steps": request.params.get("steps", []),
        },
        limitations=base_res.limitations + comp_res.limitations,
        warnings=base_res.warnings + comp_res.warnings,
    )
