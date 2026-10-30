"""Mix-shift decomposition of a rate change.

For an overall rate R = sum_i w_i * r_i (w = segment share of the denominator, r = segment rate):

    total       = R1 - R0
    mix         = sum_i (w1_i - w0_i) * r0_i        # change in who is in the population
    rate        = sum_i  w0_i * (r1_i - r0_i)       # change in behaviour within segments
    interaction = sum_i (w1_i - w0_i) * (r1_i - r0_i)

mix + rate + interaction == total exactly. A segment present in only one period gets its
undefined rate replaced by the other period's rate (or the overall baseline rate), which keeps the
identity exact: that segment then contributes only through mix/interaction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analytics.models import AnalysisRequest, AnalysisResult, AnalysisStatus
from analytics.statistical import compare_proportions
from analytics.utils import apply_filters, apply_time_filter, enrich_with_user_attrs

Counts = Dict[Tuple, Tuple[int, int]]  # segment -> (denominator n, numerator k)


def decompose_rate_change(base: Counts, comp: Counts) -> Dict[str, Any]:
    """Exact mix / rate / interaction split of the change in overall rate. Returns segment rows + totals."""
    n0, n1 = sum(n for n, _ in base.values()), sum(n for n, _ in comp.values())
    if n0 <= 0 or n1 <= 0:
        raise ValueError("Both periods need a positive denominator for mix-shift decomposition.")
    R0 = sum(k for _, k in base.values()) / n0
    R1 = sum(k for _, k in comp.values()) / n1

    rows: List[Dict[str, Any]] = []
    mix = rate = inter = 0.0
    for seg in sorted(set(base) | set(comp), key=str):
        bn, bk = base.get(seg, (0, 0))
        cn, ck = comp.get(seg, (0, 0))
        w0, w1 = bn / n0, cn / n1
        r0 = bk / bn if bn > 0 else (ck / cn if cn > 0 else R0)
        r1 = ck / cn if cn > 0 else r0
        if bn == 0:           # new segment: no baseline behaviour; use overall baseline rate as reference
            r0 = R0
        m, r, i = (w1 - w0) * r0, w0 * (r1 - r0), (w1 - w0) * (r1 - r0)
        mix, rate, inter = mix + m, rate + r, inter + i
        rows.append({
            "segment": seg, "base_n": bn, "comp_n": cn, "base_share": w0, "comp_share": w1,
            "base_rate": r0 if bn > 0 else None, "comp_rate": (ck / cn) if cn > 0 else None,
            "mix_effect": m, "rate_effect": r, "interaction": i, "contribution": m + r + i,
        })
    total = R1 - R0
    return {"rows": rows, "base_rate": R0, "comp_rate": R1, "total_change": total,
            "mix_effect": mix, "rate_effect": rate, "interaction": inter, "n0": n0, "n1": n1}


def bootstrap_se(base: Counts, comp: Counts, draws: int = 300, seed: int = 0) -> Dict[str, float]:
    """Seeded parametric bootstrap standard errors of the mix / rate / interaction effects."""
    rng = np.random.default_rng(seed)
    out: Dict[str, List[float]] = {"mix_effect": [], "rate_effect": [], "interaction": []}

    def _resample(c: Counts) -> Counts:
        segs = list(c)
        n = sum(v[0] for v in c.values())
        w = np.array([c[s][0] for s in segs], dtype=float) / n
        ns = rng.multinomial(n, w)
        return {s: (int(m), int(rng.binomial(m, (c[s][1] / c[s][0]) if c[s][0] else 0.0))) for s, m in zip(segs, ns)}

    for _ in range(draws):
        d = decompose_rate_change(_resample(base), _resample(comp))
        for key in out:
            out[key].append(d[key])
    return {k: float(np.std(v, ddof=1)) for k, v in out.items()}


def mix_shift_decomposition(
    primary_df: pd.DataFrame, users_df: pd.DataFrame, request: AnalysisRequest
) -> AnalysisResult:
    """Decompose the change in a rate (share of rows where metric_column == metric_value) by segment.

    Params: metric_column, metric_value (the numerator condition), breakdown (user/row dimensions),
    time_column (default "timestamp"). `window` is the comparison period, `comparison_window` the baseline.
    """
    def bad(msg: str, status: AnalysisStatus = AnalysisStatus.INVALID_PARAMS) -> AnalysisResult:
        return AnalysisResult(request.request_id, "mix_shift_decomposition", status, None, error_message=msg)

    if request.comparison_window is None:
        return bad("mix_shift_decomposition requires a comparison_window (baseline period).")
    metric_column = request.params.get("metric_column")
    metric_value = request.params.get("metric_value")
    breakdown: List[str] = list(request.params.get("breakdown") or request.dimensions)
    time_column = request.params.get("time_column", "timestamp")
    if not metric_column or metric_value is None:
        return bad("Parameters 'metric_column' and 'metric_value' (the rate numerator) are required.")
    if not breakdown:
        return bad("At least one breakdown dimension is required.")

    def _counts(window) -> Counts:
        scoped = apply_time_filter(primary_df, time_column, window)
        need = [c for c in breakdown + list(request.filters) if c in users_df.columns]
        if need:
            scoped = enrich_with_user_attrs(scoped, users_df, need)
        scoped = apply_filters(scoped, request.filters)
        if scoped.empty:
            return {}
        missing = [d for d in breakdown if d not in scoped.columns]
        if missing:
            raise KeyError(f"Breakdown dimension(s) not found: {missing}")
        out: Counts = {}
        for key, grp in scoped.groupby(breakdown, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            out[key] = (len(grp), int((grp[metric_column] == metric_value).sum()))
        return out

    try:
        base, comp = _counts(request.comparison_window), _counts(request.window)
    except KeyError as e:
        return bad(str(e))
    if not base or not comp:
        return bad("Both periods need data for the decomposition.", AnalysisStatus.INSUFFICIENT_DATA)

    d = decompose_rate_change(base, comp)
    se = bootstrap_se(base, comp)
    warnings: List[str] = []

    # Did the composition change at all? Chi-square test of homogeneity of segment counts across periods.
    segs = sorted(set(base) | set(comp), key=str)
    table = np.array([[base.get(s, (0, 0))[0] for s in segs], [comp.get(s, (0, 0))[0] for s in segs]])
    table = table[:, table.sum(axis=0) > 0]
    p_values: Dict[str, float] = {}
    if table.shape[1] >= 2:
        p_values["mix_homogeneity"] = float(stats.chi2_contingency(table)[1])
    else:
        warnings.append("Only one segment present; the mix cannot change.")
    k0, k1 = sum(k for _, k in base.values()), sum(k for _, k in comp.values())
    overall = compare_proportions(k1, d["n1"], k0, d["n0"])
    p_values["overall_rate_change"] = overall["p_value"]

    df = pd.DataFrame(d["rows"])
    for i, dim in enumerate(breakdown):
        df.insert(i, dim, [r["segment"][i] for r in d["rows"]])
    df = df.drop(columns=["segment"])

    small = [str(r["segment"]) for r in d["rows"] if 0 < min(r["base_n"] or 10**9, r["comp_n"] or 10**9) < 10]
    if small:
        warnings.append(f"Segments with fewer than 10 rows in a period have unstable rates: {small}")

    return AnalysisResult(
        request_id=request.request_id,
        operation="mix_shift_decomposition",
        status=AnalysisStatus.OK,
        data=df,
        metadata={
            "base_rate": d["base_rate"], "comp_rate": d["comp_rate"], "total_change": d["total_change"],
            "mix_effect": d["mix_effect"], "rate_effect": d["rate_effect"], "interaction": d["interaction"],
            "identity_residual": d["total_change"] - (d["mix_effect"] + d["rate_effect"] + d["interaction"]),
            "bootstrap_se": se,
            "p_values": p_values,
            "effect_estimate": {
                "value": d["mix_effect"], "method": "mix_rate_decomposition:mix_effect",
                "se": se["mix_effect"],
                "window": {"start": request.window.start.isoformat(), "end": request.window.end.isoformat()},
                "metric": f"rate({metric_column}=={metric_value})",
            },
            "window": str(request.window),
        },
        limitations=[
            "Interaction (joint mix x rate) is reported separately and is NOT included in the mix estimate.",
            "Shares attribute arithmetic, not causes: a mix change says who arrived, not why.",
            "Standard errors are from a seeded parametric bootstrap (300 draws).",
        ],
        warnings=warnings,
    )
