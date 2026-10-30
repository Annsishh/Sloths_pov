"""Phase 5: SQL sandbox bypass attempts and adversarial LLM behaviour. No network, no keys."""

import sqlite3

import pandas as pd
import pytest

from analytics.executor import AnalyticsExecutor
from analytics.loader import DataLoader
from analytics.models import AnalysisRequest, AnalysisStatus, TimeWindow
from analytics.sql_sandbox import SQLPolicyError, SQLSandbox, clean_sql, validate_query
from controller.capability_catalogue import CONTROLLED_OPERATIONS, unsafe_string_reason, validate_analysis_plan
from controller.controller import InvestigationController
from controller.llm_client import MockLLMClient
from controller.schemas import AnalysisDecision, EvidenceInterpretation, HypothesesGenerationOutput, HypothesisProposal
from investigation import InvestigationStatus, Observation

W = TimeWindow.from_iso


@pytest.fixture
def executor():
    users = pd.DataFrame([{"user_id": f"u{i}", "device_type": "mobile", "browser": "Chrome"} for i in range(6)])
    orders = pd.DataFrame([{"order_id": f"o{i}", "user_id": f"u{i}", "status": "failed" if i % 2 else "completed",
                            "failure_reason": "update" if i % 2 else None, "amount": 10.0,
                            "timestamp": pd.Timestamp("2026-09-12T10:00:00Z")} for i in range(6)])
    events = pd.DataFrame([{"event_id": f"e{i}", "user_id": f"u{i}", "session_id": f"s{i}", "timestamp": pd.Timestamp("2026-09-12T10:00:00Z"),
                            "event_name": "page_view", "device_type": "mobile", "browser": "Chrome", "app_version": "v1"} for i in range(6)])
    return AnalyticsExecutor(DataLoader.from_dataframes({"users": users, "orders": orders, "events": events}))


def q(ex, sql, **p):
    return ex.run(AnalysisRequest("custom_query", W("2026-09-01", "2026-10-01"), params={"query": sql, **p}))


# ------------------------------------------------------------------ SQL: bypass attempts are all refused

@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "drop table orders", "DrOp TaBlE orders", "DELETE FROM orders", "UPDATE orders SET amount = 0",
    "INSERT INTO orders VALUES (1)", "CREATE TABLE x (a int)", "ALTER TABLE orders ADD COLUMN z int",
    "SELECT 1; DROP TABLE orders", "SELECT 1;DELETE FROM orders", "SELECT 1; SELECT 2",
    "SELECT 1 -- harmless\n; DROP TABLE orders", "/* x */ DROP TABLE orders",
    "PRAGMA table_info(orders)", "pragma writable_schema = 1", "SELECT 1; PRAGMA query_only = OFF",
    "ATTACH DATABASE '/tmp/x.db' AS x", "attach database ':memory:' as m", "SELECT 1 FROM orders; ATTACH DATABASE 'a' AS b",
    "WITH t AS (SELECT 1) DELETE FROM orders", "WITH t AS (SELECT 1) INSERT INTO orders SELECT * FROM t",
    "VACUUM", "REINDEX", "EXPLAIN DROP TABLE orders", "  \n\t DROP TABLE orders",
    "SELECT load_extension('x')", "SELECT readfile('/etc/passwd')", "SELECT writefile('/tmp/x', 'y')",
    "SELECT * FROM sqlite_master", "SELECT name FROM sqlite_schema", "SELECT * FROM sqlite_temp_master",
    "", "   ", ";", "-- only a comment",
])
def test_bypass_attempts_are_rejected(executor, sql):
    res = q(executor, sql)
    assert res.status in (AnalysisStatus.INVALID_PARAMS, AnalysisStatus.ERROR), (sql, res.status)
    assert res.data is None
    # and nothing changed
    assert len(q(executor, "SELECT * FROM orders").data) == 6


def test_comment_tricks_do_not_hide_or_fake_keywords(executor):
    # a comment is whitespace to SQLite, so SEL/**/ECT is NOT SELECT
    assert q(executor, "SEL/**/ECT 1").status != AnalysisStatus.OK
    assert q(executor, "SELECT 1 /* DROP TABLE orders */").status == AnalysisStatus.OK         # inert text in a comment
    assert q(executor, "SELECT 1 -- DROP TABLE orders").status == AnalysisStatus.OK
    assert q(executor, "SELECT /*! DROP TABLE orders */ 1").status == AnalysisStatus.OK
    exe, scan = clean_sql("SELECT 'a;b' /* c */ FROM orders -- d")
    assert exe == "SELECT 'a;b'   FROM orders" and ";" not in scan


def test_keywords_inside_string_literals_are_not_false_positives(executor):
    res = q(executor, "SELECT COUNT(*) AS n FROM orders WHERE failure_reason = 'update' OR failure_reason = 'DROP; DELETE'")
    assert res.ok() and int(res.data["n"][0]) == 3
    assert q(executor, "SELECT replace('abc', 'b', 'x') AS r").data["r"][0] == "axc"            # REPLACE() is just a function
    assert q(executor, "SELECT 1;").ok()                                                         # a trailing semicolon is fine


def test_case_tricks_on_select_are_fine_but_not_on_blocked_verbs(executor):
    assert q(executor, "sElEcT 1 AS x").ok()
    assert q(executor, "wITh t As (SELECT 1 AS x) SELECT x FROM t").ok()
    assert q(executor, "DeLeTe FROM orders").status == AnalysisStatus.INVALID_PARAMS


def test_unlisted_functions_are_denied_by_the_authorizer(executor):
    for fn in ("randomblob(8)", "hex('a')", "zeroblob(10)", "sqlite_version()", "json('{}')"):
        assert q(executor, f"SELECT {fn}").status == AnalysisStatus.ERROR, fn
    assert q(executor, "SELECT ROUND(AVG(amount), 1) AS a FROM orders").ok()


def test_only_exported_investigator_tables_are_readable(executor):
    assert q(executor, "SELECT COUNT(*) FROM orders").ok()
    assert q(executor, "SELECT * FROM ground_truth").status == AnalysisStatus.ERROR
    assert q(executor, "SELECT * FROM nonexistent").status == AnalysisStatus.ERROR


def test_limits_are_clamped_regardless_of_what_the_caller_asks(executor, monkeypatch):
    import analytics.sql_sandbox as sb
    assert sb.HARD_MAX_TIMEOUT == 10.0 and sb.HARD_MAX_ROWS == 5000
    monkeypatch.setattr(sb, "HARD_MAX_TIMEOUT", 1.0)          # keep the test fast; the clamp logic is what is under test
    big = q(executor, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 20000) SELECT x FROM c", max_rows=10**9)
    assert big.ok() and len(big.data) == 5000 and any("truncated" in w for w in big.warnings)
    slow = q(executor, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c", timeout_seconds=10**6)
    assert slow.status == AnalysisStatus.ERROR and "timed out" in slow.error_message             # timeout is capped
    assert q(executor, "SELECT 1", max_rows="lots").status == AnalysisStatus.INVALID_PARAMS
    assert q(executor, "SELECT '" + "x" * 11000 + "'").status == AnalysisStatus.INVALID_PARAMS


def test_database_level_read_only_holds_even_without_authorizer_or_text_filters(executor):
    """Layer 1 alone: a raw mode=ro connection cannot be written to."""
    box = executor._sql_sandbox()
    conn = box.raw_readonly_connection()
    try:
        for stmt in ("DELETE FROM orders", "UPDATE orders SET amount = 0", "DROP TABLE orders", "CREATE TABLE t (a)",
                     "INSERT INTO orders (order_id) VALUES ('x')"):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute(stmt)
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 6
    finally:
        conn.close()


def test_attach_is_blocked_at_the_engine_even_if_text_filters_were_bypassed(executor):
    box = executor._sql_sandbox()
    conn = box._guarded_connection(2.0)
    try:
        for stmt in ("ATTACH DATABASE ':memory:' AS x", "PRAGMA query_only = OFF", "DELETE FROM orders"):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(stmt)
    finally:
        conn.close()


def test_the_sandbox_file_is_cleaned_up_and_sees_only_the_given_tables(executor):
    box = SQLSandbox({"t": pd.DataFrame({"a": [1, 2]})})
    path = box.path
    assert box.table_names == ["t"] and path.exists()
    box.close()
    assert not path.exists()


def test_sandbox_rebuilds_when_the_data_changes(executor):
    first = executor._sql_sandbox()
    assert executor._sql_sandbox() is first
    executor.loader._cache["orders"] = executor.loader._cache["orders"].iloc[:3]
    assert executor._sql_sandbox() is not first
    assert int(q(executor, "SELECT COUNT(*) AS n FROM orders").data["n"][0]) == 3


def test_validate_query_returns_the_exact_text_that_is_executed():
    assert validate_query("SELECT 1 /* c */ AS x ;") == "SELECT 1   AS x"
    with pytest.raises(SQLPolicyError):
        validate_query("SELECT 1; SELECT 2")


def test_custom_query_is_not_in_the_catalogue():
    assert "custom_query" not in CONTROLLED_OPERATIONS
    assert not validate_analysis_plan("custom_query", {"query": "SELECT 1"})[0]


# ------------------------------------------------------------------ catalogue: vocabularies, paths, types

@pytest.mark.parametrize("op,params", [
    ("segment", {"table": "user_feedback", "breakdown": ["device_type"]}),
    ("segment", {"table": "../ground_truth/scenario_ground_truth", "breakdown": ["device_type"]}),
    ("segment", {"table": "data/x/ground_truth", "breakdown": ["device_type"]}),
    ("time_series", {"aggregation": "count", "table": "orders; DROP TABLE orders"}),
    ("time_series", {"aggregation": "median_of_medians"}),
    ("time_series", {"aggregation": "count", "freq": "M"}),
    ("time_series", {"aggregation": "count", "time_column": "user_id"}),
    ("release_impact", {"release_id": "../../ground_truth/x"}),
    ("release_impact", {"release_id": "rel_240", "metric_column": "C:\\secrets"}),
    ("release_impact", {"release_id": "rel_240", "window_days": True}),
    ("exposure_effect", {"feature_key": "ground_truth"}),
    ("exposure_effect", {"feature_key": "GROUND_TRUTH_flag"}),
    ("exposure_effect", {"feature_key": "x" * 300}),
    ("exposure_effect", {"feature_key": "a\x00b"}),
    ("funnel", {"funnel_steps": ["a", "b"] * 40}),
    ("funnel", {"funnel_steps": ["a", {"nested": "dict"}]}),
    ("segment", {"breakdown": ["device_type"], "metric_value": float("nan")}),
])
def test_dangerous_or_out_of_vocabulary_parameters_are_rejected(op, params):
    ok, err = validate_analysis_plan(op, params)
    assert ok is False, (op, params)


def test_unsafe_string_reasons():
    assert unsafe_string_reason("orders") is None and unsafe_string_reason("2026-09-15T10:00:00Z") is None
    assert "ground-truth" in unsafe_string_reason("my_ground_truth")
    assert "path" in unsafe_string_reason("a/b") and "path" in unsafe_string_reason("..") and "path" in unsafe_string_reason("a\\b")
    assert "characters" in unsafe_string_reason("a;b") and "characters" in unsafe_string_reason("$(rm)")


def test_every_catalogue_example_still_passes_hardening():
    for name, spec in CONTROLLED_OPERATIONS.items():
        assert validate_analysis_plan(name, spec.example)[0], name


# ------------------------------------------------------------------ adversarial mock LLM

@pytest.fixture
def obs():
    return Observation("order_conversion_rate", W("2026-09-01", "2026-09-10"), W("2026-09-10", "2026-09-20"), 0.2, 0.1, -0.1, -0.5, "drop")


def hyps():
    return HypothesesGenerationOutput(
        hypotheses=[HypothesisProposal(title="Tracking dropped", explanation="Telemetry events were dropped.", category="tracking_loss"),
                    HypothesisProposal(title="Release bug", explanation="A release broke checkout.", category="code_regression")],
        reasoning="two candidates")


def decision(**kw):
    kw.setdefault("hypothesis_id_to_test", "hyp_1")
    kw.setdefault("rationale", "attempting something it should not")
    return AnalysisDecision(**kw)


def run_with(executor, obs, canned, **ctl):
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(canned), **ctl)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))
    return ctrl, final


@pytest.mark.parametrize("bad", [
    decision(operation="custom_query", parameters={"query": "SELECT 1"}),
    decision(operation="__import__('os').system('x')"),
    decision(operation="funnel; DROP TABLE orders", parameters={"funnel_steps": ["a", "b"]}),
    decision(operation="tracking_gap_check", dimensions=["user_id"]),
    decision(operation="tracking_gap_check", parameters={"breakdown": ["password_hash"]}),
    decision(operation="tracking_gap_check", filters={"ground_truth_label": "x"}),
    decision(operation="tracking_gap_check", filters={"browser": {"$ne": "Safari"}}),
    decision(operation="segment", parameters={"table": "../ground_truth/scenario_ground_truth.json", "breakdown": ["device_type"]}),
    decision(operation="release_impact", parameters={"release_id": "rel_240", "table": "C:/secret"}),
    decision(operation="time_series", parameters={"aggregation": "count"}, dimensions=["browser"]),
])
def test_adversarial_decisions_never_reach_the_executor(executor, obs, bad):
    calls = []
    orig = executor.run
    executor.run = lambda r: (calls.append(r.operation), orig(r))[1]
    _, final = run_with(executor, obs, [hyps(), bad], max_iterations=2)
    assert calls == []
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert any(e.title == "Analysis Plan Rejected" for e in final.investigation.history)
    assert final.validate() == []


def test_references_to_unknown_hypotheses_are_refused(executor, obs):
    _, final = run_with(executor, obs, [hyps(), decision(hypothesis_id_to_test="hyp_99", operation="error_analysis")], max_iterations=2)
    assert any("Invalid Hypothesis Reference" in e.title for e in final.investigation.history)
    assert final.investigation.evidence_store == {}


def test_operation_name_tricks_are_normalised_not_bypassed(executor, obs):
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient([hyps(), decision(operation="  ERROR_ANALYSIS ", parameters={"breakdown": ["device_type"]}, dimensions=["device_type"])]))
    state = ctrl.start_investigation(obs)
    ctrl.step_formulate_hypotheses(state)
    res = ctrl.execute_analysis(state, ctrl.step_select_analysis(state, 1))
    assert res.ok() and res.operation == "error_analysis"


def test_analysis_budget_cannot_be_exceeded_even_when_the_loop_is_bypassed(executor, obs):
    ctrl = InvestigationController(executor=executor, llm_client=MockLLMClient(), max_analyses=2)
    state = ctrl.start_investigation(obs)
    d = decision(operation="error_analysis", parameters={"breakdown": ["device_type"]}, dimensions=["device_type"])
    assert ctrl.execute_analysis(state, d).ok() and ctrl.execute_analysis(state, d).ok()
    refused = ctrl.execute_analysis(state, d)
    assert refused.status == AnalysisStatus.INVALID_PARAMS and ctrl.analyses_executed_count == 2
    assert any(e.title == "Analysis Budget Exceeded" for e in state.investigation.history)


def test_a_model_that_requests_analyses_forever_is_stopped_by_the_budgets(executor, obs):
    forever = lambda prompt, schema: (hyps() if schema is HypothesesGenerationOutput else
                                      decision(operation="error_analysis", parameters={"breakdown": ["device_type"]}, dimensions=["device_type"])
                                      if schema is AnalysisDecision else
                                      EvidenceInterpretation(hypothesis_id="hyp_1", stance="inconclusive", finding_summary="Nothing conclusive here.",
                                                             rationale="The numbers do not settle it."))
    client = MockLLMClient(generator_fn=forever)
    ctrl = InvestigationController(executor=executor, llm_client=client, max_iterations=50, max_analyses=3)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE
    assert ctrl.analyses_executed_count == 3
    assert len(client.call_history) <= 1 + 2 * 3 + 1


def test_llm_call_budget_is_enforced(executor, obs):
    forever = lambda prompt, schema: (hyps() if schema is HypothesesGenerationOutput else
                                      decision(operation="error_analysis", parameters={"breakdown": ["device_type"]}, dimensions=["device_type"])
                                      if schema is AnalysisDecision else
                                      EvidenceInterpretation(hypothesis_id="hyp_1", stance="inconclusive", finding_summary="Nothing conclusive here.",
                                                             rationale="The numbers do not settle it."))
    client = MockLLMClient(generator_fn=forever)
    ctrl = InvestigationController(executor=executor, llm_client=client, max_iterations=50, max_analyses=50, max_llm_calls=5)
    final = ctrl.run_investigation_loop(ctrl.start_investigation(obs))
    assert len(client.call_history) == 5 and ctrl.llm_calls == 5
    assert final.investigation.status == InvestigationStatus.STOPPED_INSUFFICIENT_EVIDENCE


def test_default_llm_call_budget_scales_with_the_analysis_budget(executor):
    assert InvestigationController(executor=executor, llm_client=MockLLMClient(), max_analyses=10).max_llm_calls == 50
    assert InvestigationController(executor=executor, llm_client=MockLLMClient(), max_analyses=3, max_llm_calls=7).max_llm_calls == 7


def test_ground_truth_text_in_model_output_is_flagged_by_the_state_audit(executor, obs):
    bad_interp = EvidenceInterpretation(hypothesis_id="hyp_1", stance="supports", finding_summary="See scenario_ground_truth for the answer.",
                                        rationale="The injected_parameters say so, clearly enough.")
    _, final = run_with(executor, obs, [hyps(), decision(operation="error_analysis", parameters={"breakdown": ["device_type"]}, dimensions=["device_type"]), bad_interp], max_iterations=1)
    errors = final.validate()
    assert any("Ground-truth leakage" in e for e in errors)


def test_model_cannot_smuggle_effect_numbers(executor, obs):
    with pytest.raises(Exception, match="deterministic"):
        EvidenceInterpretation(hypothesis_id="hyp_1", stance="supports", finding_summary="Finding with numbers.",
                               rationale="Because of the numbers.", explained_share=0.9)
