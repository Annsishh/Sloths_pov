"""Event tracking and data-quality investigation.

Provides diagnostics for instrumentation discrepancies, tracking gaps,
event volume anomalies, and duplicate event bursts.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.utils import (
    apply_filters,
    apply_time_filter,
    check_min_sample,
    enrich_with_user_attrs,
    validate_dimensions,
)


def event_volume_check(
    events_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compute periodic event volumes, optionally broken down by dimension.

    Parameters in ``request.params``
    ---------------------------------
    event_names : List[str], optional
        Restrict to these event names.
    breakdown : List[str], optional
        Dimensions to slice by (e.g. ["browser", "device_type"]).
    freq : str
        Period grouping frequency — "D" (default), "W".
    """
    limitations: List[str] = []
    warnings: List[str] = []

    event_names: Optional[List[str]] = request.params.get("event_names")
    breakdown: List[str] = request.params.get("breakdown", [])
    freq: str = request.params.get("freq", "D")

    scoped = apply_time_filter(events_df, "timestamp", request.window)

    needed_user_cols = [c for c in list(request.filters.keys()) + breakdown if c in users_df.columns]
    if needed_user_cols:
        scoped = enrich_with_user_attrs(scoped, users_df, needed_user_cols)

    try:
        scoped = apply_filters(scoped, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="event_volume_check",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if event_names:
        scoped = scoped[scoped["event_name"].isin(event_names)]

    if scoped.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="event_volume_check",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No events found matching filters in the specified window.",
        )

    dim_err = validate_dimensions(scoped, breakdown)
    if dim_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="event_volume_check",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=dim_err,
        )

    scoped = scoped.copy()
    ts_utc = scoped["timestamp"].dt.tz_convert("UTC") if scoped["timestamp"].dt.tz is not None else scoped["timestamp"]
    if freq.upper() == "D":
        scoped["_period"] = ts_utc.dt.floor("D")
    elif freq.upper() == "W":
        scoped["_period"] = ts_utc.dt.floor("D") - pd.to_timedelta(ts_utc.dt.weekday, unit="D")
    else:
        with pd.option_context("mode.chained_assignment", None):
            scoped["_period"] = ts_utc.dt.to_period(freq).dt.start_time

    group_cols = ["_period"] + breakdown + ["event_name"]
    result = scoped.groupby(group_cols, sort=True).size().reset_index(name="event_count")
    result = result.rename(columns={"_period": "period"})
    result["period"] = result["period"].dt.strftime("%Y-%m-%d")

    return AnalysisResult(
        request_id=request.request_id,
        operation="event_volume_check",
        status=AnalysisStatus.OK,
        data=result,
        metadata={
            "event_names": event_names or "all",
            "breakdown": breakdown,
            "freq": freq,
            "window": str(request.window),
            "total_events": len(scoped),
        },
        limitations=limitations,
        warnings=warnings,
    )


def tracking_gap_check(
    events_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare backend completed orders with telemetry order_completed events.

    Detects instrumentation failures where orders succeed financially in backend
    tables but are dropped by client-side event tracking (e.g. Safari ITP/SDK bugs).

    Parameters in ``request.params``
    ---------------------------------
    breakdown : List[str]
        Dimensions to compare (e.g. ["browser"]).
    completed_event : str
        Telemetry event name (default "order_completed").
    order_status : str
        Backend status to evaluate (default "completed").
    grace_period_seconds : int
        Allowed timestamp offset between order and telemetry event (default 60s).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    breakdown: List[str] = request.params.get("breakdown", ["browser"])
    completed_event: str = request.params.get("completed_event", "order_completed")
    order_status: str = request.params.get("order_status", "completed")
    grace_sec: int = request.params.get("grace_period_seconds", 60)

    # Scope orders to window
    scoped_orders = apply_time_filter(orders_df, "timestamp", request.window)
    # Scope events with grace margin to avoid boundary artifacts
    expanded_window_end = request.window.end + timedelta(seconds=grace_sec)
    events_mask = (events_df["timestamp"] >= request.window.start) & (events_df["timestamp"] < expanded_window_end)
    scoped_events = events_df.loc[events_mask].copy()

    # Safely enrich dimensions from users_df without duplicate column collisions
    needed_user_cols = [c for c in list(request.filters.keys()) + breakdown if c in users_df.columns]
    if needed_user_cols:
        scoped_orders = enrich_with_user_attrs(scoped_orders, users_df, needed_user_cols)
        scoped_events = enrich_with_user_attrs(scoped_events, users_df, needed_user_cols)

    try:
        scoped_orders = apply_filters(scoped_orders, request.filters)
        scoped_events = apply_filters(scoped_events, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="tracking_gap_check",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    # Filter to target statuses
    backend_orders = scoped_orders[scoped_orders["status"] == order_status].copy()
    telemetry_events = scoped_events[scoped_events["event_name"] == completed_event].copy()

    def _match_1to1(orders: pd.DataFrame, events: pd.DataFrame) -> int:
        """Greedy nearest-timestamp 1-to-1 matching of orders to telemetry events.
        
        Guarantees that one event matches at most one order and duplicate events
        cannot artificially inflate order matching.
        """
        if orders.empty or events.empty:
            return 0

        user_events: Dict[str, List[Tuple[Any, pd.Timestamp]]] = {}
        for idx, row in events.iterrows():
            uid = row["user_id"]
            if uid not in user_events:
                user_events[uid] = []
            user_events[uid].append((idx, row["timestamp"]))

        matched_count = 0
        consumed_events = set()

        # Match chronologically
        for _, ord_row in orders.sort_values("timestamp").iterrows():
            uid = ord_row["user_id"]
            o_time = ord_row["timestamp"]
            candidates = user_events.get(uid, [])

            best_idx = None
            best_diff = None

            for ev_idx, ev_time in candidates:
                if ev_idx in consumed_events:
                    continue
                diff_sec = abs((ev_time - o_time).total_seconds())
                if diff_sec <= grace_sec:
                    if best_diff is None or diff_sec < best_diff:
                        best_diff = diff_sec
                        best_idx = ev_idx

            if best_idx is not None:
                consumed_events.add(best_idx)
                matched_count += 1

        return matched_count

    # Dimensions check
    valid_dims = [d for d in breakdown if d in backend_orders.columns or d in telemetry_events.columns]

    rows = []
    if not valid_dims:
        n_orders = len(backend_orders)
        n_events = len(telemetry_events)
        n_matched = _match_1to1(backend_orders, telemetry_events)

        ratio = round(n_events / n_orders, 4) if n_orders > 0 else None
        match_rate = round(n_matched / n_orders, 4) if n_orders > 0 else None

        rows.append({
            "backend_orders": n_orders,
            "telemetry_events": n_events,
            "matched_orders": n_matched,
            "unmatched_orders": max(0, n_orders - n_matched),
            "tracking_ratio": ratio,
            "match_rate": match_rate,
        })
    else:
        # Collect distinct dimension tuples
        dim_values_set = set()
        for _, r in backend_orders.iterrows():
            dim_values_set.add(tuple(r[d] for d in valid_dims))
        for _, r in telemetry_events.iterrows():
            dim_values_set.add(tuple(r[d] for d in valid_dims))

        for dim_tuple in sorted(list(dim_values_set)):
            dim_dict = dict(zip(valid_dims, dim_tuple))

            # Filter backend orders for this segment
            ord_mask = pd.Series(True, index=backend_orders.index)
            for d, val in dim_dict.items():
                ord_mask &= (backend_orders[d] == val)
            seg_orders = backend_orders.loc[ord_mask]

            # Filter telemetry events for this segment
            ev_mask = pd.Series(True, index=telemetry_events.index)
            for d, val in dim_dict.items():
                ev_mask &= (telemetry_events[d] == val)
            seg_events = telemetry_events.loc[ev_mask]

            n_ord = len(seg_orders)
            n_ev = len(seg_events)
            n_match = _match_1to1(seg_orders, seg_events)

            ratio = round(n_ev / n_ord, 4) if n_ord > 0 else None
            match_rate = round(n_match / n_ord, 4) if n_ord > 0 else None

            row_data = dict(dim_dict)
            row_data.update({
                "backend_orders": n_ord,
                "telemetry_events": n_ev,
                "matched_orders": n_match,
                "unmatched_orders": max(0, n_ord - n_match),
                "tracking_ratio": ratio,
                "match_rate": match_rate,
            })
            rows.append(row_data)

    result_df = pd.DataFrame(rows)

    # Warnings for severe tracking drop or inflation
    for _, r in result_df.iterrows():
        ratio = r.get("tracking_ratio")
        orders_cnt = r.get("backend_orders", 0)
        seg_name = {d: r[d] for d in valid_dims} if valid_dims else "overall"
        if ratio is not None and orders_cnt >= 5 and ratio < 0.70:
            warnings.append(
                f"Severe tracking failure detected in {seg_name}: "
                f"only {ratio:.1%} tracking ratio ({r['telemetry_events']} events for {orders_cnt} orders)."
            )
        elif ratio is not None and orders_cnt >= 5 and ratio > 1.30:
            warnings.append(
                f"Tracking event inflation / duplication detected in {seg_name}: "
                f"{ratio:.1%} tracking ratio ({r['telemetry_events']} events for {orders_cnt} orders)."
            )

    limitations.append(
        f"Event matching uses a {grace_sec}s grace period to accommodate asynchronous client network latency. "
        "Tracking gaps identify telemetry loss rather than real transaction volume loss."
    )

    return AnalysisResult(
        request_id=request.request_id,
        operation="tracking_gap_check",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "breakdown": breakdown,
            "completed_event": completed_event,
            "order_status": order_status,
            "window": str(request.window),
            "total_backend_orders": len(backend_orders),
            "total_telemetry_events": len(telemetry_events),
        },
        limitations=limitations,
        warnings=warnings,
    )


def duplicate_event_check(
    events_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Detect sessions with an unusually high repetition of a specific event."""
    event_name: str = request.params.get("event_name", "order_completed")
    max_per_session: int = request.params.get("max_per_session", 3)

    scoped = apply_time_filter(events_df, "timestamp", request.window)
    target = scoped[scoped["event_name"] == event_name]

    if target.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="duplicate_event_check",
            status=AnalysisStatus.OK,
            data=pd.DataFrame(columns=["session_id", "event_count"]),
            metadata={
                "event_name": event_name,
                "max_per_session": max_per_session,
                "total_sessions_with_event": 0,
                "flagged_sessions": 0,
                "window": str(request.window),
            },
        )

    per_session = target.groupby("session_id").size().reset_index(name="event_count")
    flagged = per_session[per_session["event_count"] > max_per_session].copy()

    warnings = []
    if not flagged.empty:
        warnings.append(
            f"{len(flagged)} session(s) fired '{event_name}' more than {max_per_session} times. "
            "Possible duplicate event logging bug."
        )

    return AnalysisResult(
        request_id=request.request_id,
        operation="duplicate_event_check",
        status=AnalysisStatus.OK,
        data=flagged,
        metadata={
            "event_name": event_name,
            "max_per_session": max_per_session,
            "total_sessions_with_event": len(per_session),
            "flagged_sessions": len(flagged),
            "window": str(request.window),
        },
        limitations=[],
        warnings=warnings,
    )
