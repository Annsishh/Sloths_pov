"""Run identical scenarios independently across configured models and collect raw artefacts.

MODEL-FACING: this module reads only the scenario manifest and (through the existing evaluation.runner.run_scenario_dir)
data/<scenario>/investigator_data + alert.json. It never touches ground truth or the rubric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evaluation.metered import PriceTable
from evaluation.runner import RunRecord, run_scenario_dir

from .providers import MOCK_PROVIDERS, RecordingClient, build_client

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parent.parent / "data"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_find(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a model response as JSON (tolerating code fences); None if it is not JSON."""
    if not raw:
        return None
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


def extract_model_outputs(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Hypotheses proposed, analyses requested, evidence requests and stated confidence, read from the VALID raw responses."""
    hyps, analyses, questions, confidences = [], [], [], []
    for c in calls:
        if not c["valid"]:
            continue
        d = _json_find(c["raw_response"])
        if not d:
            continue
        if c["schema"] == "HypothesesGenerationOutput":
            hyps += [{"title": h.get("title"), "category": h.get("category")} for h in d.get("hypotheses", [])]
        elif c["schema"] == "AnalysisDecision":
            analyses.append({"operation": d.get("operation"), "hypothesis": d.get("hypothesis_id_to_test"),
                             "dimensions": d.get("dimensions", []), "rationale": d.get("rationale")})
        elif c["schema"] == "EvidenceInterpretation":
            questions += [q for q in d.get("unresolved_questions", []) if q not in questions]
            if d.get("confidence_level"):
                confidences.append(d["confidence_level"])
    return {"proposed_hypotheses": hyps, "requested_analyses": analyses, "evidence_requests": questions,
            "stated_confidences": confidences}


def run_one(model_cfg: Dict[str, Any], scenario: Dict[str, Any], run: int, data_root: Path, max_analyses: int,
            allow_mock: bool) -> Dict[str, Any]:
    """One independent investigation: fresh client, fresh state. Never raises; failures are recorded as data."""
    is_mock = model_cfg["provider"] in MOCK_PROVIDERS
    price = (PriceTable(model_cfg["price_in_per_mtok"], model_cfg["price_out_per_mtok"])
             if model_cfg.get("price_in_per_mtok") is not None and model_cfg.get("price_out_per_mtok") is not None else None)
    out: Dict[str, Any] = {"model_id": model_cfg["id"], "provider": model_cfg["provider"], "model": model_cfg.get("model"),
                           "scenario": scenario["id"], "slot": scenario["slot"], "data_scenario": scenario["data_scenario"],
                           "run": run, "is_mock": is_mock, "error": None, "record": None, "calls": [], "outputs": {}}
    rec_client: Optional[RecordingClient] = None
    t0 = time.perf_counter()

    def factory():
        nonlocal rec_client
        rec_client = RecordingClient(build_client(model_cfg, allow_mock))
        return rec_client, None

    try:
        record: RunRecord = run_scenario_dir(data_root, scenario["data_scenario"], run, factory, provider=model_cfg["provider"],
                                             model=model_cfg.get("model"), allow_mock=allow_mock, price=price, max_analyses=max_analyses)
        out["record"] = record
        out["error"] = record.error
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["wall_sec"] = round(time.perf_counter() - t0, 4)
    if rec_client is not None:
        out["calls"] = rec_client.calls
        out["outputs"] = extract_model_outputs(rec_client.calls)
    return out


def run_experiment(models: List[Dict[str, Any]], scenarios: List[Dict[str, Any]], runs: int, data_root: Path,
                   max_analyses: int, allow_mock: bool, on_run: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    results = []
    for m in models:
        for s in scenarios:
            for r in range(runs):
                res = run_one(m, s, r, data_root, max_analyses, allow_mock)
                results.append(res)
                if on_run:
                    on_run(res)
    return results
