"""Phase 6: schema mapping, backend interface, investigation trace."""

import json
from pathlib import Path

import pandas as pd
import pytest

from analytics.backend import DataBackend
from analytics.executor import AnalyticsExecutor
from analytics.loader import CATALOGUE, DataLoader
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.schema_mapping import (CsvDirectorySource, FrameSource, MappedBackend, SchemaMapping,
                                      SchemaMappingError)
from controller.controller import InvestigationController
from controller.llm_client import MockLLMClient
from evaluation.mock_policy import make_mock_client
from investigation.models import Observation
from investigation.testability import assess_testability
from investigation.trace import build_trace, trace_markdown, write_trace
from controller.capability_catalogue import CONTROLLED_OPERATIONS

ROOT = Path(__file__).resolve().parents[1]
W = TimeWindow.from_iso
EXAMPLE = ROOT / "schema_mapping.example.json"


def company_tables():
    """A 'real company' warehouse: different table, column, event and value names."""
    ts = lambda d, h=10: pd.Timestamp(f"2026-09-{d:02d}T{h:02d}:00:00Z")
    customers = pd.DataFrame([{"customer_id": f"c{i}", "created_at": ts(1), "utm_channel": "organic", "country": "US",
                               "device": "mobile", "browser": "Safari" if i < 5 else "Chrome", "plan": "pro"} for i in range(10)])
    ev, pay = [], []
    for i in range(10):
        for d in (3, 13):
            sid = f"v{i}_{d}"
            ev.append({"id": f"e{i}{d}a", "customer_id": f"c{i}", "visit_id": sid, "event_ts": ts(d), "name": "begin_checkout",
                       "page": "checkout", "device": "mobile", "browser": "Safari" if i < 5 else "Chrome", "build": "1.0"})
            ok = not (d == 13 and i % 2 == 0)
            pay.append({"payment_id": f"p{i}{d}", "customer_id": f"c{i}", "attempted_at": ts(d), "amount_usd": 10.0, "currency": "USD",
                        "outcome": "SUCCESS" if ok else "DECLINED", "decline_code": None if ok else "card"})
            if ok and not (d == 13 and i < 5):          # Safari tracking loss on the second period
                ev.append({"id": f"e{i}{d}b", "customer_id": f"c{i}", "visit_id": sid, "event_ts": ts(d) + pd.Timedelta(seconds=2), "name": "purchase_success",
                           "page": "checkout", "device": "mobile", "browser": "Safari" if i < 5 else "Chrome", "build": "1.0"})
    return {"dim_customers": customers, "raw_web_events": pd.DataFrame(ev), "fct_payments": pd.DataFrame(pay)}


def mapped():
    return MappedBackend(FrameSource(company_tables()), SchemaMapping.from_file(EXAMPLE))


# ------------------------------------------------------------------ schema mapping

def test_example_mapping_file_is_valid_and_round_trips():
    m = SchemaMapping.from_file(EXAMPLE)
    assert SchemaMapping.from_dict(m.to_dict()) == m
    assert m.event_names["order_completed"] == "purchase_success"


def test_mapped_backend_presents_the_internal_vocabulary():
    b = mapped()
    assert b.available_tables() == ["events", "orders", "users"]
    ev, od = b.load("events"), b.load("orders")
    assert list(ev.columns) == ["event_id", "user_id", "session_id", "timestamp", "event_name", "page", "device_type", "browser", "app_version"]
    assert set(ev["event_name"]) == {"checkout_started", "order_completed"}                # purchase_success -> order_completed
    assert set(od["status"]) == {"completed", "failed"}                                    # SUCCESS/DECLINED -> completed/failed
    assert pd.api.types.is_datetime64_any_dtype(ev["timestamp"]) and str(ev["timestamp"].dt.tz) == "UTC"
    assert set(b.load("sessions")["converted"]) == {True, False}
    assert b.load("events") is b.load("events")


def test_analytics_run_unchanged_on_mapped_company_data():
    ex = AnalyticsExecutor(mapped())
    res = ex.run(AnalysisRequest("tracking_gap_check", W("2026-09-10", "2026-09-20"), params={"breakdown": ["browser"]}))
    assert res.ok(), res.error_message
    rows = res.data.set_index("browser")
    assert rows.loc["Safari", "telemetry_events"] == 0 and rows.loc["Safari", "backend_orders"] > 0     # tracking loss found
    assert rows.loc["Chrome", "match_rate"] == 1.0


def test_unmapped_names_pass_through_and_unknown_values_are_kept():
    m = SchemaMapping.from_dict({"event_names": {"order_completed": "buy"}})
    raw = company_tables()["raw_web_events"].rename(columns={"customer_id": "user_id", "visit_id": "session_id", "event_ts": "timestamp",
                                                              "name": "event_name", "device": "device_type", "build": "app_version", "id": "event_id"})
    out = m.to_internal("events", raw)
    assert "begin_checkout" in set(out["event_name"]) and "purchase_success" in set(out["event_name"])    # not mapped -> unchanged


@pytest.mark.parametrize("bad,msg", [
    ({"tables": {"nonsense": "x"}}, "unknown internal table"),
    ({"columns": {"events": {"bogus": "x"}}}, "no internal column"),
    ({"values": {"orders": {"bogus": {"a": "b"}}}}, "no internal column"),
    ({"tables": {"events": "same", "orders": "same"}}, "same external name"),
    ({"event_names": {"a": "x", "b": "x"}}, "same external name"),
    ({"columns": {"events": {"user_id": "c", "session_id": "c"}}}, "same external column"),
    ({"surprise": {}}, "Unknown mapping section"),
])
def test_malformed_mappings_are_rejected(bad, msg):
    with pytest.raises(SchemaMappingError, match=msg):
        SchemaMapping.from_dict(bad)


def test_mapping_that_does_not_fit_the_raw_data_is_reported():
    m = SchemaMapping.from_file(EXAMPLE)
    raw = company_tables()
    assert m.validate_against(raw) == []
    raw["fct_payments"] = raw["fct_payments"].drop(columns=["outcome"])
    problems = m.validate_against(raw)
    assert problems and "outcome" in problems[0]
    with pytest.raises(SchemaMappingError):
        MappedBackend(FrameSource(raw), m).load("orders")


def test_invalid_json_file_is_reported(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaMappingError, match="not valid JSON"):
        SchemaMapping.from_file(p)


def test_missing_raw_tables_make_testability_shrink_not_crash():
    full = mapped()
    assert "releases" not in full.available_tables()
    ta = assess_testability(CONTROLLED_OPERATIONS, full.available_tables())
    assert not ta["code_regression"].testable and ta["tracking_loss"].testable        # no release log -> no regression test
    with pytest.raises(FileNotFoundError):
        full.load("releases")
    with pytest.raises(KeyError):
        full.load("made_up")


def test_csv_source_refuses_ground_truth_directories(tmp_path):
    from analytics.path_guard import ForbiddenDataPathError
    d = tmp_path / "ground_truth"
    d.mkdir()
    with pytest.raises(ForbiddenDataPathError):
        CsvDirectorySource(d)
    ok = tmp_path / "raw"
    ok.mkdir()
    for name, df in company_tables().items():
        df.to_csv(ok / f"{name}.csv", index=False)
    b = MappedBackend(CsvDirectorySource(ok), SchemaMapping.from_file(EXAMPLE))
    assert b.load("orders")["status"].isin(["completed", "failed"]).all()


def test_catalogue_stays_in_the_internal_vocabulary():
    src = (ROOT / "src" / "controller" / "capability_catalogue.py").read_text(encoding="utf-8")
    assert "purchase_success" not in src and "fct_payments" not in src          # nothing company-specific is hard-coded


# ------------------------------------------------------------------ backend interface

class TinyBackend(DataBackend):
    """The smallest possible backend: implements ONLY the two abstract methods."""
    def __init__(self, tables): self._t = tables
    def available_tables(self): return list(self._t)
    def load(self, name): return self._t[name] if name != "sessions" else __import__("analytics.backend", fromlist=["x"]).derive_sessions(self._t["events"])


def internal_tables():
    b = mapped()
    return {n: b.load(n) for n in b.available_tables()}


def test_dataloader_and_mapped_backend_implement_the_interface():
    assert issubclass(DataLoader, DataBackend) and issubclass(MappedBackend, DataBackend)
    with pytest.raises(TypeError):
        DataBackend()


def test_a_minimal_backend_runs_the_executor_and_the_controller_context():
    backend = TinyBackend(internal_tables())
    ex = AnalyticsExecutor(backend)
    assert ex.run(AnalysisRequest("tracking_gap_check", W("2026-09-10", "2026-09-20"), params={})).ok()
    assert ex.run(AnalysisRequest("custom_query", W("2026-09-01", "2026-10-01"), params={"query": "SELECT COUNT(*) AS n FROM orders"})).ok()
    assert backend.row_count("orders") == 20 and "timestamp_min" in backend.time_coverage("orders")
    assert set(backend.load_all()) == {"events", "orders", "users"}
    ctx = InvestigationController(executor=ex, llm_client=MockLLMClient()).build_data_context()
    assert "orders: 20 rows" in ctx and "Event names" in ctx
    backend.clear_cache()


def test_default_backend_is_still_pandas():
    ex = AnalyticsExecutor(DataLoader.from_dataframes({"users": pd.DataFrame(columns=list(CATALOGUE["users"].field_names()))}))
    assert isinstance(ex.loader, DataLoader)


def test_analytics_modules_do_not_reach_around_the_backend_interface():
    """Only the loader module may read files; operations receive DataFrames from the backend."""
    for path in (ROOT / "src" / "analytics").glob("*.py"):
        if path.name in {"loader.py", "schema_mapping.py", "sql_sandbox.py", "path_guard.py"}:
            continue
        assert "read_csv" not in path.read_text(encoding="utf-8"), path.name


# ------------------------------------------------------------------ trace

@pytest.fixture(scope="module")
def finished_state():
    data = ROOT / "data" / "tracking_failure"
    if not data.exists():
        pytest.skip("tracked data missing")
    from evaluation.runner import run_scenario_dir
    from investigation.state import InvestigationState
    rec = run_scenario_dir(ROOT / "data", "tracking_failure", 0, make_mock_client, provider="mock", allow_mock=True)
    return InvestigationState.from_dict(rec.state)


def test_trace_contains_every_required_element(finished_state):
    t = build_trace(finished_state)
    kinds = {k for s in t["steps"] for k in ("hypothesis", "analysis", "evidence", "link", "status_change") if k in s}
    assert kinds == {"hypothesis", "analysis", "evidence", "link", "status_change"}
    a = next(s["analysis"] for s in t["steps"] if "analysis" in s)
    assert a["operation"] and a["parameters"] is not None and a["window"] and a["rationale"]
    e = next(s["evidence"] for s in t["steps"] if "evidence" in s)
    assert e["finding"] and "p_values_raw" in e and "adjusted_p_min" in e and e["tests_in_family_at_that_moment"] is not None
    c = t["conclusion"]
    assert c["outcome"] == "concluded" and c["supported_causes"] and c["statements"] and c["tests_in_family"] > 0
    assert "unexplained_change" in c and "data_requests" in c and t["support_policy"]


def test_trace_is_chronological(finished_state):
    times = [s["time"] for s in build_trace(finished_state)["steps"]]
    assert times == sorted(times)
    order = [s["type"] for s in build_trace(finished_state)["steps"]]
    assert order[0] == "investigation_triggered" and order[-1] == "investigation_concluded"


def test_trace_markdown_sections_and_json_roundtrip(finished_state, tmp_path):
    md = trace_markdown(finished_state)
    for heading in ("# Investigation trace", "## Timeline", "**Analysis**", "**Evidence", "## Conclusion", "unexplained remainder", "### Supported causes"):
        assert heading in md, heading
    paths = write_trace(finished_state, tmp_path, "run1")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["conclusion"]["outcome"] == "concluded"
    assert paths["markdown"].read_text(encoding="utf-8") == md


def test_trace_of_an_insufficient_evidence_run_lists_requests_ruled_out_and_untestable():
    from evaluation.runner import run_scenario_dir
    from investigation.state import InvestigationState
    if not (ROOT / "data" / "pricing_change").exists():
        pytest.skip("tracked data missing")
    rec = run_scenario_dir(ROOT / "data", "pricing_change", 0, make_mock_client, provider="mock", allow_mock=True)
    st = InvestigationState.from_dict(rec.state)
    md = trace_markdown(st)
    assert "INSUFFICIENT EVIDENCE" in md and "### Data requests" in md and "### Untestable with current data" in md
    assert "pricing" in md.lower() and "### Ruled out" in md


def test_trace_records_support_withdrawal_and_category_coercion(make_state_for_trace):
    s = make_state_for_trace
    md = trace_markdown(s)
    assert "SUPPORT WITHDRAWN" in md and "adjusted p" in md
    assert any(x["type"] == "warning" for x in build_trace(s)["steps"])


@pytest.fixture
def make_state_for_trace():
    from investigation import EvidenceStance, HypothesisStatus, InvestigationState
    from analytics.models import AnalysisResult
    obs = Observation("m", W("2026-09-01", "2026-09-10"), W("2026-09-10", "2026-09-20"), 0.2, 0.1, -0.1, -0.5, "drop")
    s = InvestigationState.create("t", obs)
    s.add_hypothesis("Odd", "Something odd happened here.", "gremlins", "g")
    s.add_hypothesis("Tracking", "Events dropped by the SDK.", "tracking_failure", "h")
    s.add_hypothesis("Other", "Another mechanism entirely.", "product_regression", "o")
    mk = lambda op, rid, p: AnalysisResult(rid, op, AnalysisStatus.OK, {}, metadata={"p_value": p} if p is not None else {})
    for ev_id, op, p in (("e1", "tracking_gap_check", 0.04), ("e2", "event_volume_check", None)):
        s.record_evidence("finding " + ev_id, ["orders"], analysis_result=mk(op, "r" + ev_id, p), evidence_id=ev_id)
        s.link_evidence(ev_id, "h", EvidenceStance.SUPPORTS, "supports")
    s.update_hypothesis_status("h", HypothesisStatus.SUPPORTED, "rule met", "e1")
    for i, p in enumerate([0.5, 0.6, 0.7]):
        s.record_evidence("noise", ["orders"], analysis_result=mk("segment_comparison", f"x{i}", p), evidence_id=f"x{i}")
        s.link_evidence(f"x{i}", "o", EvidenceStance.INCONCLUSIVE, "n")
    s.reassess_supported()
    s.stop_insufficient_evidence("Cannot decide.", "Nothing more.", "Weak.")
    return s
