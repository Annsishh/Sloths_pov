"""Evaluation harness: refusal rules, isolation, scoring, metering, sweep, reports. No API calls anywhere."""

import ast
import io
import json
import sys
import xml.dom.minidom
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # run_eval.py lives at the repo root

import run_eval                                                              # noqa: E402
from analytics.loader import DataLoader                                      # noqa: E402
from analytics.models import TimeWindow                                      # noqa: E402
from analytics.path_guard import ForbiddenDataPathError                      # noqa: E402
from controller.llm_client import MockLLMClient                              # noqa: E402
from evaluation.alerts import build_observation, metric_value                # noqa: E402
from evaluation.clients import AnthropicClient, EvalConfigError, assert_real, resolve_client  # noqa: E402
from evaluation.metered import MeteredClient, PriceTable                     # noqa: E402
from evaluation.mock_policy import make_mock_client                          # noqa: E402
from evaluation.report import summary_markdown, write_results                # noqa: E402
from evaluation.runner import RunRecord, estimate_llm_calls, run_scenario_dir  # noqa: E402
from evaluation.scoring import aggregate, load_truth, measured_code_share, overall, score_run  # noqa: E402
from evaluation.sweep import run_effect_sweep, smallest_reliable_effect, sweep_markdown, sweep_svg  # noqa: E402

DATA = ROOT / "data"
needs_data = pytest.mark.skipif(not (DATA / "tracking_failure" / "alert.json").exists(), reason="tracked data missing")


# ------------------------------------------------------------------ refusal to run a fake "real" evaluation

@pytest.fixture
def no_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)


def test_mock_provider_requires_explicit_opt_in(no_keys):
    with pytest.raises(EvalConfigError, match="--allow-mock"):
        resolve_client("mock")
    assert isinstance(resolve_client("mock", allow_mock=True), MockLLMClient)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_real_providers_without_keys_refuse_instead_of_falling_back(no_keys, provider):
    with pytest.raises(EvalConfigError, match="refusing to fall back to the mock"):
        resolve_client(provider, model="some-model")


def test_real_provider_requires_a_model(no_keys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(EvalConfigError, match="--model"):
        resolve_client("anthropic")
    assert isinstance(resolve_client("anthropic", model="m"), AnthropicClient)


def test_unknown_provider_is_rejected(no_keys):
    with pytest.raises(EvalConfigError, match="Unknown provider"):
        resolve_client("gemini-ultra")


def test_assert_real_blocks_a_mock_in_disguise():
    with pytest.raises(EvalConfigError):
        assert_real(MockLLMClient(), allow_mock=False)
    with pytest.raises(EvalConfigError):
        assert_real(MeteredClient(MockLLMClient()), allow_mock=False)       # wrapped mock
    assert_real(MockLLMClient(), allow_mock=True)


def test_cli_refuses_real_run_that_would_fall_back_to_mock(no_keys, capsys):
    assert run_eval.main(["--provider", "anthropic", "--model", "m", "--runs", "1"]) == 3
    assert "REFUSING TO RUN" in capsys.readouterr().err
    assert run_eval.main(["--provider", "mock", "--runs", "1"]) == 3                # mock without --allow-mock
    assert run_eval.main(["--provider", "bogus", "--allow-mock"]) == 3


def test_cli_estimate_makes_no_calls(no_keys, capsys):
    assert run_eval.main(["--estimate", "--runs", "3", "--sweep"]) == 0
    out = capsys.readouterr().out
    assert "Estimated LLM calls" in out and "maximum" in out


def test_call_estimate_formula():
    e = estimate_llm_calls(runs=3, scenarios=12, sweep_cells=15, sweep_runs=3, max_analyses=16)
    assert e["runs"] == 3 * 12 + 15 * 3 == 81
    assert e["calls_per_run_max"] == 1 + 2 * 16 + 1 == 34
    assert e["calls_max"] == 81 * 34 and e["calls_typical"] == 81 * 20


# ------------------------------------------------------------------ the real client (network mocked)

def test_anthropic_client_request_shape_and_usage(monkeypatch):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return Resp(json.dumps({"content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 11, "output_tokens": 3}}).encode())

    monkeypatch.setattr("evaluation.clients.urllib.request.urlopen", fake_urlopen)
    c = AnthropicClient(api_key="secret", model="m-1")
    assert c.generate("hi", "be brief") == "hello"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["X-api-key"] == "secret" and seen["headers"]["Anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "m-1" and seen["body"]["system"] == "be brief"
    assert c.last_usage == {"input_tokens": 11, "output_tokens": 3}
    m = MeteredClient(c, PriceTable(3.0, 15.0))
    m.generate("hi")
    s = m.summary()
    assert (s["input_tokens"], s["output_tokens"], s["tokens_estimated"]) == (11, 3, False)
    assert s["cost_usd"] == pytest.approx(11 / 1e6 * 3 + 3 / 1e6 * 15)


# ------------------------------------------------------------------ metering

def test_metering_estimates_tokens_and_counts_errors_without_prices():
    inner = MockLLMClient(simulate_error="boom")
    m = MeteredClient(inner)
    with pytest.raises(Exception):
        m.generate("x" * 400)
    assert m.summary()["calls"] == 1 and m.summary()["errors"] == 1
    assert m.summary()["cost_usd"] is None and m.summary()["tokens_estimated"] is True
    assert m.calls[0].input_tokens == 100


# ------------------------------------------------------------------ alert -> observation (hand computed)

def _tiny():
    o = lambda i, d, st: {"order_id": f"o{i}", "user_id": "u", "timestamp": pd.Timestamp(f"2026-09-{d:02d}T10:00:00Z"), "status": st}
    e = lambda i, d, name, s: {"event_id": f"e{i}", "user_id": "u", "session_id": s, "timestamp": pd.Timestamp(f"2026-09-{d:02d}T10:00:00Z"), "event_name": name}
    orders = pd.DataFrame([o(1, 2, "completed"), o(2, 2, "completed"), o(3, 3, "failed"), o(4, 12, "completed"), o(5, 12, "failed"),
                           o(6, 13, "failed"), o(7, 13, "failed")])
    events = pd.DataFrame([e(1, 2, "page_view", "a"), e(2, 2, "order_completed", "a"), e(3, 3, "page_view", "b"), e(4, 12, "page_view", "c"),
                           e(5, 12, "page_view", "d"), e(6, 13, "page_view", "e"), e(7, 13, "order_completed", "e"), e(8, 13, "page_view", "f")])
    return DataLoader.from_dataframes({"orders": orders, "events": events, "users": pd.DataFrame({"user_id": ["u"]})})


def test_metric_values_hand_computed():
    ld = _tiny()
    base, comp = TimeWindow.from_iso("2026-09-01", "2026-09-11"), TimeWindow.from_iso("2026-09-11", "2026-09-15")
    assert metric_value(ld, "completed_orders_per_day", base) == pytest.approx(2 / 10)
    assert metric_value(ld, "order_success_rate", base) == pytest.approx(2 / 3)
    assert metric_value(ld, "order_success_rate", comp) == pytest.approx(1 / 4)
    assert metric_value(ld, "orders_per_100_sessions", base) == pytest.approx(2 / 2 * 100)     # sessions a, b
    assert metric_value(ld, "tracked_orders_per_100_sessions", comp) == pytest.approx(1 / 4 * 100)
    with pytest.raises(ValueError):
        metric_value(ld, "nonsense", base)


def test_build_observation_signs_and_relative_change():
    obs = build_observation(_tiny(), {"metric": "order_success_rate", "baseline_window": ["2026-09-01", "2026-09-11"],
                                      "comparison_window": ["2026-09-11", "2026-09-15"]})
    assert obs.absolute_change == pytest.approx(0.25 - 2 / 3, abs=1e-5)
    assert obs.relative_change == pytest.approx((0.25 - 2 / 3) / (2 / 3), abs=1e-5)
    with pytest.raises(ValueError, match="undefined"):
        build_observation(_tiny(), {"metric": "order_success_rate", "baseline_window": ["2027-01-01", "2027-01-02"],
                                    "comparison_window": ["2026-09-11", "2026-09-15"]})


# ------------------------------------------------------------------ end-to-end with the reference policy

@needs_data
def test_end_to_end_on_disk_scenario_and_prompts_contain_no_ground_truth():
    rec = run_scenario_dir(DATA, "tracking_failure", 0, make_mock_client, provider="mock", allow_mock=True)
    assert rec.error is None and rec.is_mock and rec.analyses
    truth = load_truth(DATA, "tracking_failure")
    assert "tracking_loss" in [c["category"] for c in rec.state["conclusion"]["supported_causes"]]
    forbidden = [truth["ground_truth_cause"], "expected_answer", "scenario_ground_truth", "measured_effects", "ground_truth"]
    for prompt in rec.prompts:
        for f in forbidden:
            assert f not in prompt
    assert json.dumps(rec.state).find(truth["ground_truth_cause"]) == -1
    sc = score_run(rec, truth)
    assert sc["primary_correct"] and sc["run_correct"] and sc["integrity_errors"] == [] and sc["is_mock"]
    assert sc["llm_calls"] == rec.llm["calls"] and sc["n_analyses"] == len(rec.analyses)


@needs_data
def test_investigator_side_loader_refuses_the_ground_truth_directory():
    with pytest.raises(ForbiddenDataPathError):
        DataLoader(data_dir=DATA / "tracking_failure" / "ground_truth")


def test_investigation_packages_never_import_the_evaluation_harness():
    for pkg in ("controller", "investigation", "analytics"):
        for path in (ROOT / "src" / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert not any(n.split(".")[0] == "evaluation" for n in names), f"{path} imports evaluation"


@needs_data
def test_runs_are_deterministic_with_the_reference_policy():
    a = run_scenario_dir(DATA, "tracking_failure", 0, make_mock_client, provider="mock", allow_mock=True)
    b = run_scenario_dir(DATA, "tracking_failure", 0, make_mock_client, provider="mock", allow_mock=True)
    assert a.analyses == b.analyses
    cats = lambda r: [(c["category"], c["statement"]) for c in r.state["conclusion"]["supported_causes"]]
    assert cats(a) == cats(b)                      # evidence ids carry random suffixes, so compare content


# ------------------------------------------------------------------ scoring (hand built records)

def _record(supported=(), outcome="concluded", data_requests=(), hyps=None, causes=None, scenario="s", analyses=("a",), history=()):
    causes = causes or [{"category": c, "estimated_change": None, "hypothesis_id": f"h_{c}"} for c in supported]
    state = {"conclusion": {"outcome": outcome, "supported_causes": causes, "data_requests": list(data_requests),
                            "explained_change": 0.0, "total_observed_change": -1.0},
             "hypotheses": hyps or {}, "history": list(history), "testability": {}, "evidence_store": {},
             "evidence_links": [], "unresolved_questions": {}, "status": "concluded", "stopping_reason": "x", "confidence": None,
             "investigation_id": "i", "title": "t", "observation": None}
    r = RunRecord(scenario=scenario, run=0, seed=None, is_mock=False, provider="p", model="m", observation={}, state=state,
                  analyses=list(analyses), llm={"calls": 5, "input_tokens": 100, "output_tokens": 10, "latency_sec": 1.5, "cost_usd": 0.01}, wall_sec=2.0)
    return r


def _score(rec, expected):
    import evaluation.scoring as sc
    sc.InvestigationState = type("S", (), {"from_dict": staticmethod(lambda d: type("V", (), {"validate": lambda self: []})())})
    try:
        return sc.score_run(rec, {"expected_answer": expected, "measured_effects": {}})
    finally:
        from investigation.state import InvestigationState
        sc.InvestigationState = InvestigationState


def test_scoring_primary_secondary_and_false_causes():
    exp = {"primary_cause": "code_regression", "secondary_causes": ["tracking_loss"], "decoys": [], "insufficient_evidence": False}
    ok = _score(_record(["code_regression", "tracking_loss"]), exp)
    assert ok["primary_correct"] and ok["secondary_recall"] == 1.0 and ok["false_causes"] == [] and ok["run_correct"]
    half = _score(_record(["code_regression"]), exp)
    assert half["primary_correct"] and half["secondary_recall"] == 0.0
    extra = _score(_record(["code_regression", "tracking_loss", "third_party_outage"]), exp)
    assert extra["false_causes"] == ["third_party_outage"] and not extra["run_correct"]
    wrong = _score(_record(["traffic_mix_shift"]), exp)
    assert not wrong["primary_correct"]


def test_scoring_decoys():
    exp = {"primary_cause": "traffic_mix_shift", "secondary_causes": [], "decoys": ["code_regression"], "insufficient_evidence": False}
    hyps = {"h1": {"category": "code_regression", "status": "refuted", "hypothesis_id": "h1"}}
    good = _score(_record(["traffic_mix_shift"], hyps=hyps), exp)
    assert good["decoys_not_supported"] == 1.0 and good["decoys_actively_refuted"] == 1.0
    never_tested = _score(_record(["traffic_mix_shift"]), exp)
    assert never_tested["decoys_not_supported"] == 1.0 and never_tested["decoys_actively_refuted"] == 0.0
    fooled = _score(_record(["code_regression"]), exp)
    assert fooled["decoys_not_supported"] == 0.0 and not fooled["primary_correct"]


def test_scoring_insufficient_evidence_and_data_request():
    exp = {"primary_cause": None, "secondary_causes": [], "decoys": [], "insufficient_evidence": True,
           "acceptable_data_request_keywords": ["pricing"]}
    good = _score(_record([], "insufficient_evidence", ["Pricing change log for 2026-09-16.."]), exp)
    assert good["primary_correct"] and good["outcome_matches_expectation"] and good["data_request_ok"] and good["run_correct"]
    vague = _score(_record([], "insufficient_evidence", ["Please send more data"]), exp)
    assert vague["data_request_ok"] is False and not vague["run_correct"]
    overconfident = _score(_record(["code_regression"], "concluded"), exp)
    assert not overconfident["primary_correct"] and not overconfident["outcome_matches_expectation"]
    # and the reverse: stopping when the data could answer
    answerable = {"primary_cause": "code_regression", "secondary_causes": [], "decoys": [], "insufficient_evidence": False}
    gave_up = _score(_record([], "insufficient_evidence", ["more"]), answerable)
    assert not gave_up["outcome_matches_expectation"] and not gave_up["run_correct"]


def test_explained_share_error_against_measured_truth():
    truth = {"expected_answer": {"primary_cause": "code_regression", "secondary_causes": ["tracking_loss"], "decoys": [],
                                 "insufficient_evidence": False, "shares_source": "measured_effects"},
             "measured_effects": {"regression_failed_orders": 60, "tracking_dropped_events": 40}}
    assert measured_code_share(truth) == pytest.approx(0.6)
    causes = [{"category": "code_regression", "estimated_change": -0.06, "hypothesis_id": "a"},
              {"category": "tracking_loss", "estimated_change": -0.02, "hypothesis_id": "b"}]       # estimated share 0.75
    import evaluation.scoring as sc
    rec = _record(causes=causes)
    sc.InvestigationState = type("S", (), {"from_dict": staticmethod(lambda d: type("V", (), {"validate": lambda self: []})())})
    try:
        out = sc.score_run(rec, truth)
    finally:
        from investigation.state import InvestigationState
        sc.InvestigationState = InvestigationState
    assert out["explained_share_error"] == pytest.approx(abs(0.75 - 0.6))
    assert measured_code_share({"expected_answer": {}, "measured_effects": {"regression_failed_orders": 5}}) is None


def test_guardrail_blocks_are_counted():
    hist = [{"title": "Support Recommendation Not Accepted"}, {"title": "Analysis Plan Rejected"}, {"title": "Evidence Recorded: ev_1"}]
    s = _score(_record(["code_regression"], history=hist),
               {"primary_cause": "code_regression", "secondary_causes": [], "decoys": [], "insufficient_evidence": False})
    assert s["guardrail_blocks"] == 2


def test_aggregate_and_overall():
    exp = {"primary_cause": "code_regression", "secondary_causes": [], "decoys": [], "insufficient_evidence": False}
    scores = [_score(_record(["code_regression"], scenario="a"), exp), _score(_record(["traffic_mix_shift"], scenario="a"), exp),
              _score(_record(["code_regression"], scenario="b"), exp)]
    agg = aggregate(scores)
    assert agg["a"]["runs"] == 2 and agg["a"]["primary_correct_rate"] == 0.5 and agg["b"]["primary_correct_rate"] == 1.0
    assert agg["a"]["mean_llm_calls"] == 5 and agg["a"]["mean_cost_usd"] == 0.01
    ov = overall(scores)
    assert ov["runs"] == 3 and ov["total_llm_calls"] == 15 and ov["total_cost_usd"] == pytest.approx(0.03)
    assert ov["primary_correct_rate"] == pytest.approx(2 / 3, abs=1e-3)


# ------------------------------------------------------------------ sweep

def test_smallest_reliable_effect_requires_monotone_reliability():
    rows = [{"noise_level": 0.0, "effect_size": e, "detection_rate": r} for e, r in
            [(0.1, 0.0), (0.2, 1.0), (0.3, 0.5), (0.5, 1.0), (0.75, 1.0)]]
    assert smallest_reliable_effect(rows)["0.0"] == 0.5          # 0.2 is reliable but 0.3 is not, so it does not count
    rows2 = [{"noise_level": 0.2, "effect_size": e, "detection_rate": r} for e, r in [(0.1, 0.9), (0.2, 0.8), (0.3, 1.0)]]
    assert smallest_reliable_effect(rows2)["0.2"] == 0.1
    rows3 = [{"noise_level": 0.4, "effect_size": 0.1, "detection_rate": 0.2}, {"noise_level": 0.4, "effect_size": 0.5, "detection_rate": 0.7}]
    assert smallest_reliable_effect(rows3)["0.4"] is None


def test_effect_sweep_runs_end_to_end_and_renders():
    sweep = run_effect_sweep([0.05, 0.9], [0.0], 1, make_mock_client, num_users=400, provider="mock", allow_mock=True, max_analyses=16)
    assert len(sweep["rows"]) == 2 and all(r["runs"] == 1 for r in sweep["rows"])
    assert sweep["rows"][0]["llm_calls"] > 0
    md = sweep_markdown(sweep, is_mock=True)
    assert "MOCK RUN" in md and "| noise" in md and "Smallest reliably detected effect" in md
    svg = sweep_svg(sweep)
    dom = xml.dom.minidom.parseString(svg)                    # well-formed XML
    assert dom.documentElement.tagName == "svg" and "detection rate" in svg and "noise 0" in svg


# ------------------------------------------------------------------ reports & CLI

@needs_data
def test_cli_mock_run_writes_results_and_flags_itself_as_mock(tmp_path, no_keys, capsys):
    out = tmp_path / "res"
    rc = run_eval.main(["--provider", "mock", "--allow-mock", "--runs", "1", "--scenarios", "tracking_failure,baseline",
                        "--out", str(out), "--save-prompts"])
    assert rc == 0
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["summary"]["overall"]["is_mock"] is True and data["summary"]["overall"]["runs"] == 2
    assert {s["scenario"] for s in data["scores"]} == {"tracking_failure", "baseline"}
    assert data["runs"][0]["prompts"]                                         # prompts saved only on request
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "MOCK RUN - NOT A MODEL EVALUATION" in md and "| tracking_failure |" in md


def test_unknown_scenario_is_rejected(capsys):
    assert run_eval.main(["--scenarios", "nope", "--allow-mock", "--provider", "mock"]) == 2


def test_markdown_for_real_runs_has_no_mock_banner():
    scores = [_score(_record(["code_regression"]), {"primary_cause": "code_regression", "secondary_causes": [], "decoys": [],
                                                    "insufficient_evidence": False})]
    from evaluation.report import build_summary
    md = summary_markdown(build_summary({"provider": "anthropic", "model": "m", "runs": 1}, scores))
    assert "MOCK" not in md and "anthropic" in md
