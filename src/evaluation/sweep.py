"""Effect-size x noise sweep for product_regression: how small an effect is reliably detected?"""

from __future__ import annotations

import html
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from analytics.loader import DataLoader
from evaluation.runner import ClientFactory, run_investigation
from investigator_data.engine import SyntheticDataEngine
from investigator_data.scenarios import SCENARIO_REGISTRY, build_config

RELIABLE_RATE = 0.8


def run_effect_sweep(
    effects: Sequence[float],
    noises: Sequence[float],
    runs: int,
    client_factory: ClientFactory,
    *,
    base_seed: int = 1000,
    scenario: str = "product_regression",
    target_category: str = "code_regression",
    num_users: Optional[int] = None,
    **run_kwargs: Any,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for noise in noises:
        for effect in effects:
            hits, calls = 0, 0
            for r in range(runs):
                scn = SCENARIO_REGISTRY[scenario]()
                seed = base_seed + r
                cfg = replace(build_config(scn, seed=seed, num_users=num_users), effect_size=float(effect), noise_level=float(noise))
                tables = SyntheticDataEngine(cfg, scn).generate_all()
                record = run_investigation(DataLoader.from_dataframes(tables), scn.alert, client_factory,
                                           scenario=f"{scenario}@{effect}/{noise}", run=r, seed=seed, **run_kwargs)
                supported = [c["category"] for c in (record.state.get("conclusion") or {}).get("supported_causes", [])]
                hits += int(target_category in supported)
                calls += record.llm.get("calls", 0)
            rows.append({"effect_size": float(effect), "noise_level": float(noise), "runs": runs,
                         "detected": hits, "detection_rate": hits / runs, "llm_calls": calls})
    return {"scenario": scenario, "target_category": target_category, "rows": rows,
            "smallest_reliable_effect": smallest_reliable_effect(rows)}


def smallest_reliable_effect(rows: List[Dict[str, Any]], threshold: float = RELIABLE_RATE) -> Dict[str, Optional[float]]:
    """Per noise level: smallest effect whose detection rate, and every larger effect's, is >= threshold."""
    out: Dict[str, Optional[float]] = {}
    for noise in sorted({r["noise_level"] for r in rows}):
        cells = sorted((r for r in rows if r["noise_level"] == noise), key=lambda r: r["effect_size"])
        best = None
        for i in range(len(cells)):
            if all(c["detection_rate"] >= threshold for c in cells[i:]):
                best = cells[i]["effect_size"]
                break
        out[str(noise)] = best
    return out


def sweep_markdown(sweep: Dict[str, Any], is_mock: bool) -> str:
    rows = sweep["rows"]
    effects = sorted({r["effect_size"] for r in rows})
    noises = sorted({r["noise_level"] for r in rows})
    lines = ["# Effect-size sweep", ""]
    if is_mock:
        lines += ["> **MOCK RUN** - produced by the rule-based reference policy, not a language model. It verifies the harness only.", ""]
    lines += [f"Scenario `{sweep['scenario']}`, target `{sweep['target_category']}`. Detection rate (detected/runs) per cell.", "",
              "| noise \\ effect | " + " | ".join(f"{e:g}" for e in effects) + " |", "|---|" + "---|" * len(effects)]
    for n in noises:
        cells = []
        for e in effects:
            r = next(x for x in rows if x["noise_level"] == n and x["effect_size"] == e)
            cells.append(f"{r['detection_rate']:.0%} ({r['detected']}/{r['runs']})")
        lines.append(f"| {n:g} | " + " | ".join(cells) + " |")
    lines += ["", f"Smallest reliably detected effect (rate >= {RELIABLE_RATE:.0%} and for all larger effects):", ""]
    for n, e in sweep["smallest_reliable_effect"].items():
        lines.append(f"- noise {n}: " + (f"**{e:g}**" if e is not None else "none of the tested effects"))
    return "\n".join(lines) + "\n"


def sweep_svg(sweep: Dict[str, Any], title: str = "Detection rate vs effect size") -> str:
    """Self-contained SVG line chart (no plotting dependency). One line per noise level, distinct markers."""
    rows = sweep["rows"]
    effects = sorted({r["effect_size"] for r in rows})
    noises = sorted({r["noise_level"] for r in rows})
    W, H, L, R, T, B = 640, 380, 64, 150, 48, 56
    pw, ph = W - L - R, H - T - B
    xs = lambda e: L + (effects.index(e) / max(1, len(effects) - 1)) * pw
    ys = lambda v: T + (1 - v) * ph
    colors = ["#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626"]
    markers = ["circle", "square", "diamond", "triangle", "cross"]

    def marker(kind, x, y, c):
        if kind == "circle":
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c}"/>'
        if kind == "square":
            return f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{c}"/>'
        if kind == "diamond":
            return f'<polygon points="{x:.1f},{y - 5.5:.1f} {x + 5.5:.1f},{y:.1f} {x:.1f},{y + 5.5:.1f} {x - 5.5:.1f},{y:.1f}" fill="{c}"/>'
        if kind == "triangle":
            return f'<polygon points="{x:.1f},{y - 5.5:.1f} {x + 5:.1f},{y + 4:.1f} {x - 5:.1f},{y + 4:.1f}" fill="{c}"/>'
        return f'<path d="M{x - 4:.1f} {y - 4:.1f}L{x + 4:.1f} {y + 4:.1f}M{x + 4:.1f} {y - 4:.1f}L{x - 4:.1f} {y + 4:.1f}" stroke="{c}" stroke-width="2.5"/>'

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" font-family="system-ui, sans-serif" font-size="12">',
           f'<title>{html.escape(title)}</title><rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{L}" y="26" font-size="15" font-weight="600" fill="#111827">{html.escape(title)}</text>']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" y1="{ys(v):.1f}" x2="{L + pw}" y2="{ys(v):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 8}" y="{ys(v) + 4:.1f}" text-anchor="end" fill="#4b5563">{v:.0%}</text>')
    out.append(f'<line x1="{L}" y1="{ys(RELIABLE_RATE):.1f}" x2="{L + pw}" y2="{ys(RELIABLE_RATE):.1f}" stroke="#6b7280" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{L + pw - 4}" y="{ys(RELIABLE_RATE) - 5:.1f}" text-anchor="end" fill="#6b7280">reliable (80%)</text>')
    for e in effects:
        out.append(f'<text x="{xs(e):.1f}" y="{T + ph + 20}" text-anchor="middle" fill="#4b5563">{e:g}</text>')
    out.append(f'<text x="{L + pw / 2}" y="{H - 12}" text-anchor="middle" fill="#374151">effect size (injected failure probability)</text>')
    out.append(f'<text transform="translate(16 {T + ph / 2}) rotate(-90)" text-anchor="middle" fill="#374151">detection rate</text>')
    for i, n in enumerate(noises):
        c, m = colors[i % len(colors)], markers[i % len(markers)]
        pts = [(xs(e), ys(next(r for r in rows if r["noise_level"] == n and r["effect_size"] == e)["detection_rate"])) for e in effects]
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{c}" stroke-width="2"/>')
        out += [marker(m, x, y, c) for x, y in pts]
        ly = T + 18 * i
        out.append(marker(m, L + pw + 22, ly + 4, c))
        out.append(f'<text x="{L + pw + 34}" y="{ly + 8}" fill="#111827">noise {n:g}</text>')
    out.append("</svg>")
    return "\n".join(out)
