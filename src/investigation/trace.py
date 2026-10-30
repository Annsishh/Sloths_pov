"""Investigation trace: a chronological, human-readable record of what was hypothesised, which analyses were chosen,
what evidence came back, how every hypothesis changed status, and the final conclusion (explained shares, unexplained
remainder, ruled-out and untestable hypotheses, data requests). Rendered as JSON (machine) and Markdown (people).

Built purely from the Investigation's own audit history, so it can be regenerated from a saved state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from investigation.enums import HistoryEventType as T
from investigation.state import InvestigationState


def _fmt_p(p: Optional[float]) -> str:
    return "n/a" if p is None else (f"{p:.2e}" if p < 0.001 else f"{p:.4f}")


def build_trace(state: InvestigationState) -> Dict[str, Any]:
    inv = state.investigation
    adj = state.adjusted_p_values()
    steps: List[Dict[str, Any]] = []
    for ev in inv.history:                                   # history is append-only, hence chronological
        md = ev.metadata
        step: Dict[str, Any] = {"time": ev.timestamp.isoformat(), "type": ev.event_type.value, "title": ev.title, "description": ev.description}
        if ev.event_type == T.HYPOTHESIS_ADDED:
            h = inv.hypotheses.get(md.get("hypothesis_id"))
            ta = inv.testability.get(h.category) if h else None
            step["hypothesis"] = {"id": md.get("hypothesis_id"), "category": md.get("category"),
                                  "original_category": md.get("original_category"),
                                  "testable": ta.testable if ta else None, "testability_reason": ta.reason if ta else None}
        elif ev.event_type == T.ANALYSIS_DISPATCHED:
            step["analysis"] = {k: md.get(k) for k in ("operation", "parameters", "dimensions", "filters", "window",
                                                       "comparison_window", "request_id", "status", "hypothesis_id")}
            step["analysis"]["rationale"] = ev.description
        elif ev.event_type == T.EVIDENCE_RECORDED:
            e = inv.evidence_store.get(md.get("evidence_id"))
            if e:
                step["evidence"] = {
                    "id": e.evidence_id, "operation": e.operation, "finding": e.finding, "inference_type": e.inference_type.value,
                    "p_values_raw": e.all_tests(), "p_value_provenance": e.p_value_provenance,
                    "adjusted_p_min": adj.get(e.evidence_id), "mde": e.mde,
                    "effect_estimate": e.effect_estimate.to_dict() if e.effect_estimate else None,
                    "limitations": list(e.limitations), "tests_in_family_at_that_moment": md.get("tests_in_family"),
                }
        elif ev.event_type == T.EVIDENCE_LINKED:
            step["link"] = {k: md.get(k) for k in ("evidence_id", "hypothesis_id", "stance")}
            step["link"]["rationale"] = ev.description
        elif ev.event_type == T.HYPOTHESIS_STATUS_UPDATED and "new_status" in md:
            step["status_change"] = {k: md.get(k) for k in ("hypothesis_id", "previous_status", "new_status", "triggered_by_evidence_id")}
            step["status_change"]["reason"] = ev.description
        elif ev.event_type == T.SUPPORT_WITHDRAWN:
            step["support_withdrawn"] = {k: md.get(k) for k in ("hypothesis_id", "adjusted_p_before", "adjusted_p_after",
                                                                "tests_in_family", "alpha", "reasons")}
        elif ev.event_type in (T.WARNING, T.DIRECTION_CHANGED):
            step["note"] = md or None
        steps.append(step)

    c = inv.conclusion
    return {
        "investigation_id": inv.investigation_id, "title": inv.title, "status": inv.status.value,
        "observation": inv.observation.to_dict(),
        "support_policy": dict(inv.support_policy),
        "hypotheses_final": {h.hypothesis_id: {"title": h.title, "category": h.category, "status": h.status.value,
                                               "supporting": list(h.supporting_evidence_ids), "contradicting": list(h.contradicting_evidence_ids)}
                             for h in inv.hypotheses.values()},
        "steps": steps,
        "tests_in_family": state.tests_run,
        "conclusion": c.to_dict() if c else None,
        "stopping_reason": inv.stopping_reason,
        "confidence": inv.confidence.to_dict() if inv.confidence else None,
    }


def trace_markdown(state: InvestigationState) -> str:
    t = build_trace(state)
    o = t["observation"]
    rel = f", {o['relative_change']:+.1%}" if o["relative_change"] is not None else ""
    L: List[str] = [f"# Investigation trace: {t['title']}", "",
                    f"- Status: **{t['status']}**  |  investigation `{t['investigation_id']}`",
                    f"- Trigger: `{o['metric_name']}` {o['baseline_value']} -> {o['comparison_value']} ({o['absolute_change']:+.4g}{rel})",
                    f"- Windows: baseline {o['baseline_window']['start'][:10]}..{o['baseline_window']['end'][:10]}, "
                    f"comparison {o['comparison_window']['start'][:10]}..{o['comparison_window']['end'][:10]}",
                    f"- Support policy: {t['support_policy']}", "", "## Timeline", ""]
    for n, s in enumerate(t["steps"], 1):
        ts = s["time"][11:19]
        if "hypothesis" in s:
            h = s["hypothesis"]
            tag = "testable" if h["testable"] else f"UNTESTABLE ({h['testability_reason']})"
            orig = f" (proposed as `{h['original_category']}`)" if h.get("original_category") and h["original_category"] != h["category"] else ""
            L.append(f"{n}. `{ts}` **Hypothesis {h['id']}** [{h['category']}]{orig} - {s['title'].split(': ', 1)[-1]} - {tag}")
        elif "analysis" in s:
            a = s["analysis"]
            L.append(f"{n}. `{ts}` **Analysis** `{a['operation']}` for {a['hypothesis_id']} - status {a['status']}; params {a['parameters']}"
                     + (f", filters {a['filters']}" if a["filters"] else "") + f"; window {a['window']}")
            L.append(f"    - why: {a['rationale']}")
        elif "evidence" in s:
            e = s["evidence"]
            tests = ", ".join(f"{k}={_fmt_p(v)}" for k, v in e["p_values_raw"].items()) or "none"
            L.append(f"{n}. `{ts}` **Evidence {e['id']}** ({e['operation']}): {e['finding']}")
            L.append(f"    - raw p-values: {tests}; best adjusted p: {_fmt_p(e['adjusted_p_min'])}; MDE: {e['mde']}; family size then: {e['tests_in_family_at_that_moment']}")
            if e["effect_estimate"]:
                ee = e["effect_estimate"]
                L.append(f"    - effect estimate {ee['value']:+.4g} (se {ee['se']}), method `{ee['method']}`")
        elif "link" in s:
            k = s["link"]
            L.append(f"{n}. `{ts}` Evidence {k['evidence_id']} **{k['stance']}** {k['hypothesis_id']}: {k['rationale']}")
        elif "status_change" in s:
            k = s["status_change"]
            L.append(f"{n}. `{ts}` **{k['hypothesis_id']}: {k['previous_status']} -> {k['new_status']}** - {k['reason']}")
        elif "support_withdrawn" in s:
            k = s["support_withdrawn"]
            L.append(f"{n}. `{ts}` **SUPPORT WITHDRAWN for {k['hypothesis_id']}**: adjusted p {k['adjusted_p_before']} -> {k['adjusted_p_after']} "
                     f"(family of {k['tests_in_family']} tests, alpha {k['alpha']})")
        else:
            L.append(f"{n}. `{ts}` {s['title']}" + (f" - {s['description']}" if s["description"] else ""))

    c = t["conclusion"]
    L += ["", "## Conclusion", ""]
    if c is None:
        L.append("_The investigation has not concluded._")
    else:
        share = f" ({c['unexplained_share']:.0%})" if c["unexplained_share"] is not None else ""
        L.append(f"**Outcome: {c['outcome'].replace('_', ' ').upper()}** (stopping reason: {t['stopping_reason']})")
        L.append(f"- Observed change: {c['total_observed_change']:+.4g}; explained {c['explained_change']:+.4g}; "
                 f"**unexplained remainder {c['unexplained_change']:+.4g}**{share}")
        L.append(f"- Multiple-testing family: {c['tests_in_family']} tests (Benjamini-Hochberg, alpha {c['fdr_alpha']})")
        if c["supported_causes"]:
            L += ["", "### Supported causes", ""]
            for s in c["supported_causes"]:
                sh = f"{s['share']:.0%}" if s["share"] is not None else "not quantified"
                L.append(f"- {s['statement']}  _[{s['hypothesis_id']}; share {sh}; {s['attribution']}]_")
        if c["ruled_out"]:
            L += ["", "### Ruled out", ""] + [f"- {r['hypothesis_id']} ({r['category']}): {r['reason']}" for r in c["ruled_out"]]
        if c["untestable"]:
            L += ["", "### Untestable with current data", ""] + [f"- {u['hypothesis_id']} ({u['category']}): {u['reason']}" for u in c["untestable"]]
        if c["inconclusive_hypothesis_ids"]:
            L += ["", f"### Inconclusive: {', '.join(c['inconclusive_hypothesis_ids'])}"]
        if c["data_requests"]:
            L += ["", "### Data requests", ""] + [f"- {r}" for r in c["data_requests"]]
        if c["warnings"]:
            L += ["", "### Warnings", ""] + [f"- {w}" for w in c["warnings"]]
    return "\n".join(L) + "\n"


def write_trace(state: InvestigationState, out_dir: Path, stem: str = "trace") -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / f"{stem}.json", out / f"{stem}.md"
    jp.write_text(json.dumps(build_trace(state), indent=2, default=str), encoding="utf-8")
    mp.write_text(trace_markdown(state), encoding="utf-8")
    return {"json": jp, "markdown": mp}
