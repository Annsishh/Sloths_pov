"""Release impact and feature-exposure analysis.

Methodology
-----------
1. Release Impact:
   - Compares a pre-deployment window [deploy_time - window_days, deploy_time)
     with an identical post-deployment window [deploy_time, deploy_time + window_days).
   - Computes pooled metrics (not unweighted averages) and computes absolute and relative changes.
   - Explicitly notes non-causal temporal correlation limitations.

2. Feature Exposure Effect:
   - Evaluates conversion/outcomes in an individualised post-exposure window
     [first_exposed_timestamp, first_exposed_timestamp + post_exposure_days).
   - Enforces complete follow-up: Users whose follow-up window extends beyond data observation
     are excluded from denominators to eliminate incomplete follow-up bias.
   - Computes statistical comparison (two-proportion z-test) between control and treatment.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus, TimeWindow
from analytics.utils import (
    apply_filters,
    apply_time_filter,
    check_min_sample,
    enrich_with_user_attrs,
    validate_aggregation,
)


def release_impact(
    releases_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare a metric immediately before and after a release deployment."""
    limitations: List[str] = []
    warnings: List[str] = []

    release_id: Optional[str] = request.params.get("release_id")
    window_days: int = request.params.get("window_days", 7)
    metric_column: Optional[str] = request.params.get("metric_column")
    aggregation: str = request.params.get("aggregation", "count").lower()
    metric_value: Optional[Any] = request.params.get("metric_value")
    time_column: str = request.params.get("time_column", "timestamp")

    if release_id is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="release_impact",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="release_id parameter is required.",
        )

    agg_err = validate_aggregation(aggregation)
    if agg_err:
        return AnalysisResult(
            request_id=request.request_id,
            operation="release_impact",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=agg_err,
        )

    # Locate release
    release_row = releases_df[releases_df["release_id"] == release_id]
    if release_row.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="release_impact",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=f"Release '{release_id}' not found in releases table.",
        )

    deploy_ts = release_row.iloc[0]["deployed_timestamp"]
    if pd.isna(deploy_ts):
        return AnalysisResult(
            request_id=request.request_id,
            operation="release_impact",
            status=AnalysisStatus.DATA_QUALITY,
            data=None,
            error_message=f"Release '{release_id}' has a null deployed_timestamp.",
        )

    deploy_ts = pd.Timestamp(deploy_ts)
    deploy_ts = deploy_ts.tz_localize("UTC") if deploy_ts.tz is None else deploy_ts.tz_convert("UTC")

    pre_window = TimeWindow(
        start=deploy_ts - timedelta(days=window_days),
        end=deploy_ts,
    )
    post_window = TimeWindow(
        start=deploy_ts,
        end=deploy_ts + timedelta(days=window_days),
    )

    def _calc_window(window: TimeWindow, label: str) -> Dict[str, Any]:
        scoped = apply_time_filter(primary_df, time_column, window)
        needed_cols = [c for c in request.filters if c in users_df.columns]
        if needed_cols:
            scoped = enrich_with_user_attrs(scoped, users_df, needed_cols)
        scoped = apply_filters(scoped, request.filters)

        n = len(scoped)
        warnings.extend(check_min_sample(n, 10, context=f"{label} window"))

        if n == 0:
            return {"period": label, "window": str(window), "count": 0, "value": 0.0}

        if aggregation == "count" or metric_column is None:
            val = n
        elif aggregation == "sum":
            val = float(scoped[metric_column].sum())
        elif aggregation == "mean":
            val = float(scoped[metric_column].mean())
        elif aggregation == "rate":
            hits = (scoped[metric_column] == metric_value).sum()
            val = float(hits / n)
        else:
            val = n

        return {"period": label, "window": str(window), "count": n, "value": round(val, 4)}

    try:
        pre_row = _calc_window(pre_window, "pre_release")
        post_row = _calc_window(post_window, "post_release")
    except Exception as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="release_impact",
            status=AnalysisStatus.ERROR,
            data=None,
            error_message=str(e),
        )

    pre_val = pre_row["value"]
    post_val = post_row["value"]
    abs_change = round(post_val - pre_val, 4)
    rel_change = round(abs_change / pre_val, 4) if pre_val != 0 else None

    change_row = {
        "period": "change",
        "window": "—",
        "count": post_row["count"] - pre_row["count"],
        "value": abs_change,
        "relative_change": rel_change,
    }

    result_df = pd.DataFrame([pre_row, post_row, change_row])
    release_info = release_row.iloc[0]

    limitations.append(
        "Temporal correlation only. A metric change coinciding with a release does not establish "
        "that the release caused the change. Concurrent shifts (e.g. traffic mix, external outages) "
        "must be evaluated before drawing causal conclusions."
    )

    return AnalysisResult(
        request_id=request.request_id,
        operation="release_impact",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "release_id": release_id,
            "version": release_info.get("version"),
            "component": release_info.get("component"),
            "deployed_timestamp": deploy_ts.isoformat(),
            "window_days": window_days,
            "aggregation": aggregation,
            "metric_column": metric_column,
            "absolute_change": abs_change,
            "relative_change": rel_change,
        },
        limitations=limitations,
        warnings=warnings,
    )


def exposure_effect(
    feature_exposures_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Compare an outcome metric between control and treatment groups of a feature."""
    limitations: List[str] = []
    warnings: List[str] = []

    feature_key: Optional[str] = request.params.get("feature_key")
    outcome_event: Optional[str] = request.params.get("outcome_event")
    post_days: int = request.params.get("post_exposure_days", 7)
    max_obs_input = request.params.get("max_observation_date")

    if feature_key is None:
        return AnalysisResult(
            request_id=request.request_id,
            operation="exposure_effect",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message="feature_key parameter is required.",
        )

    # Determine max observation date
    if max_obs_input is not None:
        max_observation_date = pd.Timestamp(max_obs_input, tz="UTC")
    else:
        max_observation_date = pd.Timestamp(request.window.end)

    # Filter exposures
    exps = feature_exposures_df[feature_exposures_df["feature_key"] == feature_key].copy()
    if exps.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="exposure_effect",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message=f"No exposure records found for feature '{feature_key}'.",
        )

    # Optional filter on exposure window
    if request.window is not None:
        exps = apply_time_filter(exps, "first_exposed_timestamp", request.window)

    # Apply user filters if any
    needed_user_cols = [c for c in request.filters if c in users_df.columns]
    if needed_user_cols:
        exps = enrich_with_user_attrs(exps, users_df, needed_user_cols)
    try:
        exps = apply_filters(exps, request.filters)
    except KeyError as e:
        return AnalysisResult(
            request_id=request.request_id,
            operation="exposure_effect",
            status=AnalysisStatus.INVALID_PARAMS,
            data=None,
            error_message=str(e),
        )

    if exps.empty:
        return AnalysisResult(
            request_id=request.request_id,
            operation="exposure_effect",
            status=AnalysisStatus.INSUFFICIENT_DATA,
            data=None,
            error_message="No exposures remain after applying filters and window constraints.",
        )

    # Prepare events lookup for fast vectorized evaluation
    if not primary_df.empty and "timestamp" in primary_df.columns:
        p_df = primary_df.copy()
        if outcome_event and "event_name" in p_df.columns:
            p_df = p_df[p_df["event_name"] == outcome_event]
        # Dict: user_id -> sorted list of event timestamps
        user_outcome_times = (
            p_df.groupby("user_id")["timestamp"]
            .apply(lambda s: sorted(s.tolist()))
            .to_dict()
        )
    else:
        user_outcome_times = {}

    # Detect crossover contamination (user exposed to both control and treatment)
    ctrl_uids = set(exps[exps["variant"] == "control"]["user_id"])
    treat_uids = set(exps[exps["variant"] == "treatment"]["user_id"])
    crossover = ctrl_uids.intersection(treat_uids)
    if crossover:
        warnings.append(
            f"{len(crossover)} user(s) were assigned to both control and treatment (crossover contamination). "
            "These users were excluded to preserve group independence."
        )
        exps = exps[~exps["user_id"].isin(crossover)]

    # Deduplicate repeated exposure records for the same user to their earliest exposure
    n_raw_exps = len(exps)
    exps = exps.sort_values("first_exposed_timestamp").drop_duplicates(subset=["user_id"], keep="first")
    if len(exps) < n_raw_exps:
        limitations.append(
            f"{n_raw_exps - len(exps)} repeated exposure record(s) were deduplicated to the user's earliest exposure."
        )

    # Sample ratio mismatch: an A/B result is only trustworthy if assignment matched the intended split.
    from analytics.experiment_checks import sample_ratio_mismatch
    srm = None
    assigned = exps.groupby("variant").size()
    if len(assigned) >= 2:
        ratios = request.params.get("expected_ratio")
        if ratios is not None and len(ratios) != len(assigned):
            ratios = None
            warnings.append("expected_ratio length does not match the number of variants; an equal split was assumed.")
        srm = sample_ratio_mismatch(assigned.to_numpy(), ratios)
        if srm["mismatch"]:
            warnings.append(
                f"SAMPLE RATIO MISMATCH (chi-square p={srm['p_value']:.2g}): assignment {dict(zip(assigned.index, assigned.values))} "
                "deviates from the intended split. The A/B result is untrusted and no p-value is reported."
            )

    variant_results = []
    incomplete_follow_up_count = 0

    for variant in ["control", "treatment"]:
        v_users = exps[exps["variant"] == variant]
        total_assigned = len(v_users)

        eligible_count = 0
        converted_count = 0

        for _, row in v_users.iterrows():
            uid = row["user_id"]
            exp_time = row["first_exposed_timestamp"]
            followup_end = exp_time + timedelta(days=post_days)

            # Incomplete follow-up check
            if followup_end > max_observation_date:
                incomplete_follow_up_count += 1
                continue

            eligible_count += 1
            user_events = user_outcome_times.get(uid, [])
            has_outcome = any(exp_time <= t < followup_end for t in user_events)
            if has_outcome:
                converted_count += 1

        warnings.extend(check_min_sample(eligible_count, 15, context=f"variant '{variant}'"))
        conv_rate = round(converted_count / eligible_count, 4) if eligible_count > 0 else 0.0

        variant_results.append({
            "variant": variant,
            "total_assigned": total_assigned,
            "eligible_users": eligible_count,
            "converted_users": converted_count,
            "conversion_rate": conv_rate,
        })

    result_df = pd.DataFrame(variant_results)

    if incomplete_follow_up_count > 0:
        limitations.append(
            f"{incomplete_follow_up_count} user(s) had incomplete follow-up windows "
            f"(follow-up extends past {max_observation_date.isoformat()}) and were excluded from conversion analysis."
        )

    # Statistical comparison between Control (index 0) and Treatment (index 1)
    if len(result_df) == 2:
        ctrl = result_df[result_df["variant"] == "control"].iloc[0]
        treat = result_df[result_df["variant"] == "treatment"].iloc[0]

        n_c, k_c = ctrl["eligible_users"], ctrl["converted_users"]
        n_t, k_t = treat["eligible_users"], treat["converted_users"]

        p_c = k_c / n_c if n_c > 0 else 0.0
        p_t = k_t / n_t if n_t > 0 else 0.0
        diff = round(p_t - p_c, 4)
        rel_diff = round(diff / p_c, 4) if p_c != 0 else None

        # Two-proportion z-test
        p_val = None
        z_stat = None
        if n_c > 0 and n_t > 0:
            p_pool = (k_c + k_t) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            if se > 0:
                from scipy import stats
                z_stat = float((p_t - p_c) / se)
                p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        comp_row = {
            "variant": "difference (treatment - control)",
            "total_assigned": None,
            "eligible_users": None,
            "converted_users": None,
            "conversion_rate": diff,
        }
        result_df = pd.concat([result_df, pd.DataFrame([comp_row])], ignore_index=True)
        result_df["relative_difference"] = [None, None, rel_diff]
    else:
        diff = None
        rel_diff = None
        z_stat = None
        p_val = None

    if srm is not None and srm["mismatch"]:
        p_val = None
        z_stat = None
        limitations.append("Effect test withheld because of a sample ratio mismatch.")

    limitations.append(
        "Observational distinction: Exposure records reflect assignment, but true causal inference "
        "requires confirming balance on pre-exposure covariates (device, channel, tenure) "
        "and absence of concurrent experiments."
    )

    return AnalysisResult(
        request_id=request.request_id,
        operation="exposure_effect",
        status=AnalysisStatus.OK,
        data=result_df,
        metadata={
            "feature_key": feature_key,
            "outcome_event": outcome_event,
            "post_exposure_days": post_days,
            "absolute_difference": diff,
            "relative_difference": rel_diff,
            "z_statistic": round(z_stat, 4) if z_stat is not None else None,
            "p_value": round(p_val, 4) if p_val is not None else None,
            "is_significant_p05": bool(p_val < 0.05) if p_val is not None else None,
            "srm": srm,
            "trusted": not (srm is not None and srm["mismatch"]),
            "statistical_inference_caveat": "Statistical significance indicates sample association, not causal proof.",
        },
        limitations=limitations,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Difference-in-differences: users on the new version vs users still on the old one, same window
# ---------------------------------------------------------------------------

def release_did(
    releases_df: pd.DataFrame,
    events_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    users_df: pd.DataFrame,
    request: AnalysisRequest,
) -> AnalysisResult:
    """Difference-in-differences for a release.

    "Exposed" users are those with at least one event on the release's version after deployment (within the
    post window); everyone else is the comparison group. For each group the rate of
    `metric_column == metric_value` is measured in the pre and post windows; the DiD is
    (post_exposed - pre_exposed) - (post_other - pre_other). Unlike a plain pre/post comparison it nets out
    anything that hit both groups at the same time (seasonality, campaigns, outages across versions).
    """
    def bad(msg, status=AnalysisStatus.INVALID_PARAMS):
        return AnalysisResult(request.request_id, "release_did", status, None, error_message=msg)

    from scipy import stats as _stats

    p = request.params
    release_id = p.get("release_id")
    window_days = int(p.get("window_days", 7))
    metric_column, metric_value = p.get("metric_column", "status"), p.get("metric_value", "completed")
    time_column = p.get("time_column", "timestamp")
    exposure_column = p.get("exposure_column", "app_version")
    if not release_id:
        return bad("release_id parameter is required.")
    if window_days < 1:
        return bad("window_days must be >= 1.")
    row = releases_df[releases_df["release_id"] == release_id]
    if row.empty:
        return bad(f"Release '{release_id}' not found in releases table.")
    deploy = pd.Timestamp(row.iloc[0]["deployed_timestamp"])
    deploy = deploy.tz_localize("UTC") if deploy.tzinfo is None else deploy.tz_convert("UTC")
    version = row.iloc[0]["version"]
    if exposure_column not in events_df.columns:
        return bad(f"exposure_column '{exposure_column}' not found in events.")
    if metric_column not in primary_df.columns:
        return bad(f"metric_column '{metric_column}' not found in the analysed table.")

    pre = TimeWindow(deploy - timedelta(days=window_days), deploy)
    post = TimeWindow(deploy, deploy + timedelta(days=window_days))
    post_ev = apply_time_filter(events_df, "timestamp", post)
    exposed_users = set(post_ev.loc[post_ev[exposure_column] == version, "user_id"])
    if not exposed_users:
        return bad(f"No users have events on version '{version}' in the post window.", AnalysisStatus.INSUFFICIENT_DATA)

    cells = {}
    for period, win in (("pre", pre), ("post", post)):
        scoped = apply_time_filter(primary_df, time_column, win)
        need = [c for c in request.filters if c in users_df.columns]
        if need:
            scoped = enrich_with_user_attrs(scoped, users_df, need)
        scoped = apply_filters(scoped, request.filters)
        is_exp = scoped["user_id"].isin(exposed_users)
        for group, part in (("exposed", scoped[is_exp]), ("other", scoped[~is_exp])):
            n = len(part)
            k = int((part[metric_column] == metric_value).sum())
            cells[(group, period)] = (n, k)
    empty = [f"{g}/{per}" for (g, per), (n, _) in cells.items() if n == 0]
    if empty:
        return bad(f"Empty cell(s) {empty}: difference-in-differences needs observations in all four group x period cells.",
                   AnalysisStatus.INSUFFICIENT_DATA)

    rate = {key: k / n for key, (n, k) in cells.items()}
    var = sum(rate[key] * (1 - rate[key]) / cells[key][0] for key in cells)
    se = math.sqrt(var)
    did = (rate[("exposed", "post")] - rate[("exposed", "pre")]) - (rate[("other", "post")] - rate[("other", "pre")])
    z = did / se if se > 0 else (0.0 if did == 0 else math.inf)
    pval = float(2 * _stats.norm.sf(abs(z))) if math.isfinite(z) else 0.0
    zc = float(_stats.norm.ppf(0.975))
    mde = float((zc + _stats.norm.ppf(0.8)) * se)

    post_n = cells[("exposed", "post")][0] + cells[("other", "post")][0]
    exposed_share = cells[("exposed", "post")][0] / post_n
    warnings: List[str] = []
    small = [f"{g}/{per}" for (g, per), (n, _) in cells.items() if n < 30]
    if small:
        warnings.append(f"Small cell(s) {small} (< 30 rows): the normal approximation is unreliable.")
    pre_gap = rate[("exposed", "pre")] - rate[("other", "pre")]

    df = pd.DataFrame([
        {"group": g, "period": per, "n": cells[(g, per)][0], "hits": cells[(g, per)][1], "rate": round(rate[(g, per)], 4)}
        for g in ("exposed", "other") for per in ("pre", "post")
    ])
    return AnalysisResult(
        request_id=request.request_id, operation="release_did", status=AnalysisStatus.OK, data=df,
        metadata={
            "release_id": release_id, "version": version, "deployed_timestamp": deploy.isoformat(),
            "exposure_definition": f"users with >=1 event where {exposure_column} == '{version}' in the post window",
            "did": did, "se": se, "z": z, "ci_lower": did - zc * se, "ci_upper": did + zc * se,
            "exposed_share_post": exposed_share, "pre_period_gap": pre_gap, "mde": mde,
            "p_values": {"did_interaction": pval},
            "effect_estimate": {
                # change in the OVERALL post-period rate attributable to the release = DiD x share of rows exposed
                "value": did * exposed_share, "method": "difference_in_differences:exposed_share_weighted",
                "se": se * exposed_share,
                "window": {"start": post.start.isoformat(), "end": post.end.isoformat()},
                "metric": f"rate({metric_column}=={metric_value})",
            },
            "window": str(post),
        },
        limitations=[
            "Assumes parallel trends: absent the release, exposed and other users would have moved alike.",
            "Exposure is self-selected (who upgrades), so groups may differ; the pre-period gap is reported for inspection.",
            "A DiD estimate is consistent with, not proof of, a release effect.",
        ],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Sample ratio mismatch
# ---------------------------------------------------------------------------

def sample_ratio_check(feature_exposures_df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
    """Chi-square check that assignment counts match the intended allocation (default equal split)."""
    from analytics.experiment_checks import sample_ratio_mismatch

    def bad(msg, status=AnalysisStatus.INVALID_PARAMS):
        return AnalysisResult(request.request_id, "sample_ratio_check", status, None, error_message=msg)

    feature_key = request.params.get("feature_key")
    if not feature_key:
        return bad("feature_key parameter is required.")
    exps = feature_exposures_df[feature_exposures_df["feature_key"] == feature_key]
    if request.window is not None and not exps.empty:
        exps = apply_time_filter(exps, "first_exposed_timestamp", request.window)
    exps = exps.drop_duplicates(subset=["user_id"])
    counts = exps.groupby("variant").size()
    if len(counts) < 2:
        return bad(f"Fewer than two variants observed for '{feature_key}'.", AnalysisStatus.INSUFFICIENT_DATA)
    ratios = request.params.get("expected_ratio")
    if ratios is not None and len(ratios) != len(counts):
        return bad(f"expected_ratio needs {len(counts)} entries (variants: {list(counts.index)}).")
    srm = sample_ratio_mismatch(counts.to_numpy(), ratios)
    df = pd.DataFrame({"variant": counts.index, "assigned": counts.values, "expected": srm["expected"]})
    warnings = ["SAMPLE RATIO MISMATCH: assignment is not as intended; do not trust A/B results from this experiment."] if srm["mismatch"] else []
    return AnalysisResult(
        request.request_id, "sample_ratio_check", AnalysisStatus.OK, df,
        metadata={"feature_key": feature_key, "srm": srm, "p_values": {"srm_chi_square": srm["p_value"]}, "window": str(request.window)},
        limitations=["A mismatch indicates broken assignment or logging; a pass does not prove randomisation."],
        warnings=warnings,
    )
