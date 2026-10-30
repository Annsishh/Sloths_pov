"""Retention and cohort analysis.

Computes day-N retention for user cohorts defined by their signup_timestamp.

Counting conventions & methodology
-----------------------------------
- Cohort definition: Users who signed up within the cohort period (daily or weekly).
- Observation window: For each user and each retention day N, observation is
  complete if ``signup_timestamp + (N + 1) days <= max_observation_date``.
  Users with incomplete observation for day N are excluded from day N's denominator
  (eligible_dN), preventing incomplete-window bias without discarding earlier valid days.
- Retained status: An eligible user is counted as retained on day N if they produced at
  least one qualifying event during ``[signup + N days, signup + (N + 1) days)``.
- Cohort aggregation: When aggregating across cohorts or periods, pooled weighted
  retention (sum of retained / sum of eligible) is used—never an unweighted average of rates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus, TimeWindow
from analytics.utils import apply_time_filter, apply_filters, check_min_sample


MIN_COHORT_SIZE = 10


def compute_retention(
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compute day-N retention for user cohorts.

    Parameters in ``request.params``
    ---------------------------------
    retention_days : List[int]
        Days at which to measure retention (e.g. [1, 7, 14, 30]).
    qualifying_events : List[str], optional
        Event names that count as active. Defaults to all events.
    cohort_period : str, optional
        "day" | "week" (default "week").
    max_observation_date : str | datetime, optional
        Upper bound for the observation window (defaults to request.window.end).
    """
    limitations: List[str] = []
    warnings: List[str] = []

    retention_days: List[int] = request.params.get("retention_days", [1, 7, 14, 30])
    qualifying_events: Optional[List[str]] = request.params.get("qualifying_events")
    cohort_period: str = request.params.get("cohort_period", "week")
    max_obs_input = request.params.get("max_observation_date")

    if not isinstance(retention_days, list) or not retention_days:
        return AnalysisResult(
            request_id=request.request_id,
            operation="retention",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="retention_days must be a non-empty list of integers.",
        )

    for d in retention_days:
        if not isinstance(d, int) or d < 0:
            return AnalysisResult(
                request_id=request.request_id,
                operation="retention",
                status=AnalysisStatus.INVALID_PARAMS,
                data=None,
                error_message=f"All retention_days must be non-negative integers. Found: {d}",
            )

    # Sort retention days
    retention_days = sorted(list(set(retention_days)))

    if max_obs_input is not None:
        if isinstance(max_obs_input, str):
            max_observation_date = pd.Timestamp(max_obs_input, tz="UTC")
        elif isinstance(max_obs_input, pd.Timestamp):
            max_observation_date = max_obs_input.tz_convert("UTC") if max_obs_input.tz is not None else max_obs_input.tz_localize("UTC")
        else:
            max_observation_date = pd.Timestamp(max_obs_input).tz_convert("UTC")
    else:
        max_observation_date = pd.Timestamp(request.window.end)

    # Scope users to signup window
    users_in_window = apply_time_filter(users_df, "signup_timestamp", request.window)
    users_in_window = apply_filters(users_in_window, request.filters)

    if users_in_window.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="retention",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No users found in the specified window after applying filters.",
        )

    # Scope events up to max_observation_date
    events_in_obs = events_df[events_df["timestamp"] < max_observation_date].copy()
    if qualifying_events:
        events_in_obs = events_in_obs[events_in_obs["event_name"].isin(qualifying_events)]

    # Fast event lookup: user_id -> sorted list of timestamps
    user_event_times: Dict[str, List[pd.Timestamp]] = {}
    if not events_in_obs.empty:
        user_event_times = (
            events_in_obs.groupby("user_id")["timestamp"]
            .apply(lambda s: sorted(s.tolist()))
            .to_dict()
        )

    if cohort_period == "day":
        cohort_key_fn = lambda ts: ts.date().isoformat()
    elif cohort_period == "week":
        cohort_key_fn = lambda ts: f"{ts.year}-W{ts.isocalendar()[1]:02d}"
    else:
        return AnalysisResult(
            request_id=request.request_id,
            operation="retention",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=f"Unknown cohort_period '{cohort_period}'. Supported: 'day', 'week'.",
        )

    # Track users by cohort
    cohort_data: Dict[str, List[Dict[str, Any]]] = {}
    incomplete_exclusions_per_day: Dict[int, int] = {d: 0 for d in retention_days}

    for _, user in users_in_window.iterrows():
        uid = user["user_id"]
        signup = user["signup_timestamp"]
        c_key = cohort_key_fn(signup)

        if c_key not in cohort_data:
            cohort_data[c_key] = []

        user_evs = user_event_times.get(uid, [])
        user_retention: Dict[str, Any] = {"user_id": uid, "signup": signup}

        for day_n in retention_days:
            obs_end_for_day = signup + pd.Timedelta(days=day_n + 1)
            if obs_end_for_day > max_observation_date:
                # Incomplete follow-up for this day
                incomplete_exclusions_per_day[day_n] += 1
                user_retention[f"eligible_d{day_n}"] = False
                user_retention[f"retained_d{day_n}"] = False
            else:
                user_retention[f"eligible_d{day_n}"] = True
                day_start = signup + pd.Timedelta(days=day_n)
                # Check for activity in [signup + day_n, signup + day_n + 1)
                has_event = any(day_start <= t < obs_end_for_day for t in user_evs)
                user_retention[f"retained_d{day_n}"] = has_event

        cohort_data[c_key].append(user_retention)

    # Build cohort summary rows
    cohort_rows = []
    for c_key in sorted(cohort_data.keys()):
        u_list = cohort_data[c_key]
        n_cohort = len(u_list)
        warns = check_min_sample(n_cohort, MIN_COHORT_SIZE, context=f"cohort {c_key}")
        warnings.extend(warns)

        row: Dict[str, Any] = {"cohort": c_key, "cohort_size": n_cohort}
        for day_n in retention_days:
            eligible_count = sum(1 for u in u_list if u[f"eligible_d{day_n}"])
            retained_count = sum(1 for u in u_list if u[f"eligible_d{day_n}"] and u[f"retained_d{day_n}"])

            row[f"eligible_d{day_n}"] = eligible_count
            row[f"retained_d{day_n}"] = retained_count
            row[f"retention_d{day_n}"] = round(retained_count / eligible_count, 4) if eligible_count > 0 else None

        cohort_rows.append(row)

    result_df = pd.DataFrame(cohort_rows)

    # Document incomplete window exclusions
    for day_n, count in incomplete_exclusions_per_day.items():
        if count > 0:
            limitations.append(
                f"Day {day_n}: {count} user(s) excluded from denominator due to incomplete observation window."
            )

    return AnalysisResult(
        request_id=request.request_id,
        operation="retention",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "retention_days": retention_days,
            "cohort_period": cohort_period,
            "qualifying_events": qualifying_events or "all",
            "window": str(request.window),
            "max_observation_date": max_observation_date.isoformat(),
            "total_users_in_window": len(users_in_window),
            "cohort_count": len(cohort_rows),
        },
        limitations=limitations,
        warnings=warnings,
    )


def compare_retention(
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare retention between two time windows (baseline vs comparison).

    Uses weighted aggregation (sum of retained / sum of eligible) across cohorts
    to eliminate Simpson's paradox and unweighted-average distortions.
    """
    if request.comparison_window is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="retention_comparison",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="compare_retention requires request.comparison_window to be set.",
        )

    # Baseline period
    baseline_req = AnalysisRequest(
        operation="retention",
        window=request.comparison_window,
        params=request.params,
        filters=request.filters,
        request_id=request.request_id + "_baseline",
    )
    baseline_res = compute_retention(users_df, events_df, baseline_req)

    # Comparison period
    comp_res = compute_retention(users_df, events_df, request)

    if not baseline_res.ok() or not comp_res.ok():
        return AnalysisResult(
            request_id=request.request_id,
            operation="retention_comparison",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message=(
                f"Baseline: {baseline_res.error_message}; "
                f"Comparison: {comp_res.error_message}"
            ),
        )

    b_df = baseline_res.data
    c_df = comp_res.data
    retention_days = request.params.get("retention_days", [1, 7, 14, 30])

    def _weighted_period_summary(df: pd.DataFrame, label: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "period": label,
            "cohort_size": int(df["cohort_size"].sum()),
        }
        for day_n in retention_days:
            elig_col = f"eligible_d{day_n}"
            ret_col = f"retained_d{day_n}"
            rate_col = f"retention_d{day_n}"

            tot_elig = int(df[elig_col].sum()) if elig_col in df.columns else 0
            tot_ret = int(df[ret_col].sum()) if ret_col in df.columns else 0

            row[elig_col] = tot_elig
            row[ret_col] = tot_ret
            row[rate_col] = round(tot_ret / tot_elig, 4) if tot_elig > 0 else None
        return row

    b_summary = _weighted_period_summary(b_df, "baseline")
    c_summary = _weighted_period_summary(c_df, "comparison")

    # Change row
    delta_row: Dict[str, Any] = {
        "period": "change",
        "cohort_size": c_summary["cohort_size"] - b_summary["cohort_size"],
    }
    for day_n in retention_days:
        rate_col = f"retention_d{day_n}"
        b_rate = b_summary.get(rate_col)
        c_rate = c_summary.get(rate_col)
        if b_rate is not None and c_rate is not None:
            delta_row[rate_col] = round(c_rate - b_rate, 4)
        else:
            delta_row[rate_col] = None

    result_df = pd.DataFrame([b_summary, c_summary, delta_row])

    # Raw two-proportion z-tests per retention horizon (corrected once, investigation-wide).
    from analytics.statistical import compare_proportions
    p_values: Dict[str, float] = {}
    for day_n in retention_days:
        n_b, k_b = b_summary.get(f"eligible_d{day_n}", 0), b_summary.get(f"retained_d{day_n}", 0)
        n_c, k_c = c_summary.get(f"eligible_d{day_n}", 0), c_summary.get(f"retained_d{day_n}", 0)
        if n_b > 0 and n_c > 0:
            p_values[f"retention_d{day_n}"] = compare_proportions(k_c, n_c, k_b, n_b)["p_value"]

    return AnalysisResult(
        request_id=request.request_id,
        operation="retention_comparison",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "baseline_window": str(request.comparison_window),
            "comparison_window": str(request.window),
            "retention_days": retention_days,
            "cohort_period": request.params.get("cohort_period", "week"),
            "p_values": p_values,
        },
        limitations=baseline_res.limitations + comp_res.limitations,
        warnings=baseline_res.warnings + comp_res.warnings,
    )
