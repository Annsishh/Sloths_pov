"""Tests for experiments/model_evaluation: isolation, provider parsing, recording, and an offline end-to-end run."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.model_evaluation import providers, run_experiment  # noqa: E402
from experiments.model_evaluation.evaluator import _calibration, _contradiction_handling, _schema_compliance  # noqa: E402
from evaluation.clients import EvalConfigError  # noqa: E402

EXP = ROOT / "experiments" / "model_evaluation"
needs_data = pytest.mark.skipif(not (ROOT / "data" / "tracking_failure").exists(), reason="data/ not generated")


def test_model_facing_modules_never_reference_ground_truth_or_rubric():
    for name in ("providers.py", "runner.py"):
        text = (EXP / name).read_text(encoding="utf-8")
        assert not re.search(r"ground_truth|load_truth|expected_answer|evaluator|rubric\.json|scoring", text), name


def test_src_does_not_import_the_experiment():
    for p in (ROOT / "src").rglob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+experiments", p.read_text(encoding="utf-8"), re.M), p


def test_scenario_manifest_has_the_five_required_slots_and_all_exist_with_rubric():
    sc = json.loads((EXP / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    rub = json.loads((EXP / "rubric.json").read_text(encoding="utf-8"))
    assert len(sc) == 5 and {s["id"] for s in sc} <= set(rub)
    assert {s["slot"] for s in sc} == {"Product regression", "Acquisition shift", "Tracking failure", "Competing hypotheses", "Insufficient evidence"}


def test_build_client_refuses_mock_without_flag_and_missing_keys_or_placeholder_models(monkeypatch):
    with pytest.raises(EvalConfigError):
        providers.build_client({"id": "m", "provider": "mock"}, allow_mock=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EvalConfigError):
        providers.build_client({"id": "g", "provider": "gemini", "model": "m1"})
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(EvalConfigError):
        providers.build_client({"id": "g", "provider": "gemini", "model": providers.PLACEHOLDER_MODEL})
    assert isinstance(providers.build_client({"id": "g", "provider": "gemini", "model": "m1"}), providers.GeminiClient)


def test_gemini_and_openai_parse_text_and_usage(monkeypatch):
    seen = {}

    def fake_post(url, headers, payload, timeout, name):
        seen[name] = (url, headers, payload)
        if name == "Gemini":
            return {"candidates": [{"content": {"parts": [{"text": "he"}, {"text": "llo"}]}}],
                    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 2}}
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 4}}

    monkeypatch.setattr(providers, "_post_json", fake_post)
    g = providers.GeminiClient("k", "gm")
    assert g.generate("p", "sys") == "hello" and g.last_usage == {"input_tokens": 7, "output_tokens": 5}
    assert seen["Gemini"][0].endswith("/models/gm:generateContent") and seen["Gemini"][2]["systemInstruction"]
    o = providers.OpenAIClient("k", "om")
    assert o.generate("p") == "hi" and o.last_usage == {"input_tokens": 5, "output_tokens": 4}


def test_recording_client_keeps_raw_text_of_invalid_and_valid_responses_and_classifies_errors():
    from controller.llm_client import MockLLMClient
    from controller.schemas import HypothesesGenerationOutput

    class Bad(MockLLMClient):
        def generate(self, prompt, system_prompt=None):
            self.last_usage = {"input_tokens": 3, "output_tokens": 2}
            return "not json at all"
        generate_structured = providers.GenericHTTPClient.generate_structured

    rc = providers.RecordingClient(Bad())
    with pytest.raises(ValueError):
        rc.generate_structured("p", HypothesesGenerationOutput)
    c = rc.calls[0]
    assert c["raw_response"] == "not json at all" and not c["valid"] and c["error_kind"] == "schema" and c["usage"]["input_tokens"] == 3

    class Down(MockLLMClient):
        def generate_structured(self, *a, **k):
            raise ConnectionError("down")
    rc2 = providers.RecordingClient(Down())
    with pytest.raises(ConnectionError):
        rc2.generate_structured("p", HypothesesGenerationOutput)
    assert rc2.calls[0]["error_kind"] == "api"


def test_evaluator_helpers():
    assert _schema_compliance([{"valid": True, "error_kind": None}, {"valid": False, "error_kind": "schema"}, {"valid": False, "error_kind": "api"}]) == 0.5
    assert _calibration(["high"], True) > _calibration(["high"], False) and _calibration([], True) is None
    hyps = {"a": {"category": "code_regression", "status": "supported", "contradicting_evidence_ids": []},
            "b": {"category": "tracking_loss", "status": "refuted", "contradicting_evidence_ids": ["e1"]}}
    assert _contradiction_handling(hyps, {"primary_cause": "code_regression"}) == 1.0
    assert _contradiction_handling({"a": hyps["a"]}, {"primary_cause": "code_regression"}) is None


@needs_data
def test_end_to_end_offline_run_writes_json_csv_and_raw_runs(tmp_path):
    out = tmp_path / "o"
    rc = run_experiment.main(["--config", str(EXP / "models.demo_mock.json"), "--allow-mock", "--scenarios", "tracking_failure,insufficient_evidence",
                              "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["WARNING"] and {r["model_id"] for r in res["runs"]} == {"mock_reference", "mock_faulty"}
    ref = {r["scenario"]: r for r in res["runs"] if r["model_id"] == "mock_reference"}
    assert ref["tracking_failure"]["correct"] and ref["insufficient_evidence"]["outcome"] == "insufficient_evidence"
    faulty = [r for r in res["runs"] if r["model_id"] == "mock_faulty"]
    assert all(r["schema_errors"] >= 1 and r["schema_compliance"] == 0.0 for r in faulty)
    assert (out / "comparison.csv").read_text(encoding="utf-8").startswith("model_id,scenario")
    raw = json.loads((out / "runs" / "mock_reference__tracking_failure__run0.json").read_text(encoding="utf-8"))
    assert raw["calls"][0]["raw_response"] and raw["model_outputs"]["proposed_hypotheses"] and raw["model_outputs"]["requested_analyses"]


def test_cli_refuses_to_run_real_provider_without_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"models": [{"id": "o", "provider": "openai", "model": "x"}]}), encoding="utf-8")
    assert run_experiment.main(["--config", str(cfg), "--out", str(tmp_path / "o")]) == 3
    assert "REFUSING" in capsys.readouterr().err
