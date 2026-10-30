"""Run full investigations against scenario data and collect everything the scorer needs.

The investigator sees ONLY `investigator_data/` tables and the alert (metric + windows). Ground truth is read
later, by scoring.py, and never passed into an investigation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from controller.controller import InvestigationController
from controller.llm_client import LLMClient
from evaluation.alerts import build_observation
from evaluation.clients import assert_real
from evaluation.metered import MeteredClient, PriceTable
from investigation.enums import HistoryEventType
from investigation.support import SupportPolicy

# (client, optional policy object exposing .prompts_seen) -- a fresh one per run
ClientFactory = Callable[[], Tuple[LLMClient, Any]]

DEFAULT_MAX_ITERATIONS = 16
DEFAULT_MAX_ANALYSES = 16


@dataclass
class RunRecord:
    scenario: str
    run: int
    seed: Optional[int]
    is_mock: bool
    provider: str
    model: Optional[str]
    observation: Dict[str, Any]
    state: Dict[str, Any]
    analyses: List[str]
    llm: Dict[str, Any]
    wall_sec: float
    error: Optional[str] = None
    prompts: List[str] = field(default_factory=list, repr=False)

    def to_dict(self, include_prompts: bool = False) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in ("scenario", "run", "seed", "is_mock", "provider", "model", "observation",
                                           "state", "analyses", "llm", "wall_sec", "error")}
        if include_prompts:
            d["prompts"] = self.prompts
        return d


def run_investigation(
    loader: DataLoader,
    alert: Dict[str, Any],
    client_factory: ClientFactory,
    *,
    scenario: str = "adhoc",
    run: int = 0,
    seed: Optional[int] = None,
    provider: str = "mock",
    model: Optional[str] = None,
    allow_mock: bool = False,
    price: Optional[PriceTable] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_analyses: int = DEFAULT_MAX_ANALYSES,
    support_policy: Optional[SupportPolicy] = None,
) -> RunRecord:
    inner, policy = client_factory()
    assert_real(inner, allow_mock)                       # a "real" run can never be a mock in disguise
    is_mock = provider == "mock"
    client = MeteredClient(inner, price=price, is_mock=is_mock)
    started = time.perf_counter()
    error = None
    obs = build_observation(loader, alert)
    controller = InvestigationController(
        executor=AnalyticsExecutor(loader), llm_client=client,
        max_iterations=max_iterations, max_analyses=max_analyses, support_policy=support_policy,
    )
    state = controller.start_investigation(obs, title=f"{scenario} / run {run}")
    try:
        controller.run_investigation_loop(state)
    except Exception as e:                               # a failed run is data, not a crash of the whole evaluation
        error = f"{type(e).__name__}: {e}"
        if state.investigation.status.value in ("open", "in_progress"):
            state.stop_insufficient_evidence(
                reason=f"Run aborted: {error}", missing_information_summary="The investigation could not complete.",
                confidence_justification="Run failed.")
    analyses = [e.metadata.get("operation") for e in state.investigation.history
                if e.event_type == HistoryEventType.ANALYSIS_DISPATCHED]
    return RunRecord(
        scenario=scenario, run=run, seed=seed, is_mock=is_mock, provider=provider, model=model,
        observation=obs.to_dict(), state=state.to_dict(), analyses=analyses, llm=client.summary(),
        wall_sec=round(time.perf_counter() - started, 4), error=error,
        prompts=list(getattr(policy, "prompts_seen", []) or []),
    )


def run_scenario_dir(
    data_root: Path, scenario: str, run: int, client_factory: ClientFactory, **kw: Any
) -> RunRecord:
    """Investigate one on-disk scenario. Only `investigator_data/` and `alert.json` are read here."""
    base = Path(data_root) / scenario
    loader = DataLoader(data_dir=base / "investigator_data")
    alert = json.loads((base / "alert.json").read_text(encoding="utf-8"))
    return run_investigation(loader, alert, client_factory, scenario=scenario, run=run, **kw)


def estimate_llm_calls(runs: int, scenarios: int, sweep_cells: int = 0, sweep_runs: int = 0,
                       max_analyses: int = DEFAULT_MAX_ANALYSES) -> Dict[str, Any]:
    """Upper/typical LLM call counts. Per run: 1 hypothesis call + 2 calls per analysis (select + interpret)
    + 1 final selection attempt that may fail."""
    per_run_max = 1 + 2 * max_analyses + 1
    per_run_typical = 1 + 2 * 9 + 1          # ~9 analyses observed with the reference mock policy
    n_runs = runs * scenarios + sweep_cells * sweep_runs
    return {"runs": n_runs, "calls_per_run_typical": per_run_typical, "calls_per_run_max": per_run_max,
            "calls_typical": n_runs * per_run_typical, "calls_max": n_runs * per_run_max}
