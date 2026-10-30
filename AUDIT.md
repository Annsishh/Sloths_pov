# AUDIT.md — Phase 0

Status of every gap listed in Phases 1–6, verified by reading the full source (not summaries).
Legend: **CONFIRMED** = gap is real · **PARTIAL** = some of it exists · **ALREADY HANDLED** = done, no work needed.

## Baseline facts

- Tests (Python 3.14, `pytest -v`): **93 collected → 92 passed, 1 skipped** at the start; the skip was a real bug (see below). After the Phase 0 fixes: **97 passed, 0 skipped** (93 + 4 new isolation tests).
- Slowest test: 0.95 s. **No test exceeds 5 s.** Full suite 6–17 s.
- Scenarios on disk: `baseline`, `product_regression`, `acquisition_shift`, `tracking_failure` (all four present, ~2.6 MB each).
- Ground truth is stored at `data/<scenario>/ground_truth/scenario_ground_truth.json`, a **sibling directory** of `data/<scenario>/investigator_data/`. It is physically separate from the investigator CSVs. Nothing physically stops an investigator process from reading it, though: `DataLoader` takes any directory, and the only guard is convention plus the static test added in Phase 0.
- Environment: `python` on PATH is 3.12 without dependencies; the dependencies live in 3.14. `requirements.txt` was missing `scipy` and `pydantic`, both imported by `src/` (fixed).

## Phase 1 — Hypothesis space and honest conclusions

| Gap | Verdict | Evidence |
|---|---|---|
| Expand category enum (16 families) | **PARTIAL** | `Hypothesis.category` is a free `str` (`investigation/models.py`); `state.add_hypothesis` does no validation. The only vocabulary is `HypothesisProposal.validate_category` (`controller/schemas.py`): 7 values (`tracking_failure, product_regression, acquisition_shift, external_factor, pricing_change, user_behavior, other`), and unknown values are silently coerced to `other`. The prompt (`controller/prompts.py`) lists a third, different list. Scenario `category` in `investigator_data/scenarios.py` is a fourth. |
| TESTABLE / UNTESTABLE from the capability catalogue | **CONFIRMED** | No notion of testability. `DataCatalogue.supports_analysis` only checks table availability; `CONTROLLED_OPERATIONS` has no link from operation to hypothesis category. |
| Deterministic SUPPORTED rule | **CONFIRMED** | `InvestigationController.step_interpret_result` (steps 3, 5) applies the LLM's `recommended_hypothesis_status` directly; `InvestigationState.update_hypothesis_status` accepts any transition; `conclude()` only checks `status == SUPPORTED` and that at least one supporting evidence id exists. |
| Explained-change accounting | **CONFIRMED** | `conclude()` takes one `supported_hypothesis_id`; no share estimate or unexplained remainder anywhere. `Observation` holds the totals but nothing apportions them. |
| INSUFFICIENT_EVIDENCE must list ruled-out / untestable / data request | **PARTIAL** | `stop_insufficient_evidence` records a reason, a `missing_information_summary` string and `remaining_uncertainties`. No ruled-out list, no untestable list, no structured data request. |
| Guardrail: ruling out testables must not promote an untestable | **CONFIRMED (absent)** | No testability concept, and the LLM can set SUPPORTED unconditionally, so the guardrail can't exist yet. |

## Phase 2 — Statistical correctness

| Gap | Verdict | Evidence |
|---|---|---|
| Benjamini–Hochberg + per-investigation test count | **CONFIRMED** | No FDR/Bonferroni code anywhere. Tests are judged individually at fixed 0.05/0.01 (`compare_proportions`). `InvestigationState` tracks no test count (the controller only counts analyses via `analyses_executed_count`). |
| Correct test selection / flag misuse | **PARTIAL** | Both tests exist (`analytics/statistical.py`: `compare_proportions`, `compare_means`), chosen by the caller's `test_type`. No misuse detection (e.g. means test on a 0/1 column). `exposure_effect` re-implements the z-test inline instead of calling `compare_proportions`. `period_comparison`/`release_impact` report no significance at all. |
| Seasonality-aware baseline (same weekday, N prior weeks) | **CONFIRMED** | Baseline is whatever `baseline_window` the Observation carries. The only `weekday` use is week bucketing in `time_series`/`data_quality`. |
| Change-point detection | **CONFIRMED** | None. |
| Mix-shift decomposition | **CONFIRMED** | `segmentation.compare_segments` reports per-segment before/after only; no mix / rate / interaction split. |
| Difference-in-differences for releases | **CONFIRMED** | `release_analysis.release_impact` is symmetric pre/post only. `exposure_effect` compares exposed/unexposed, but only post-exposure. |
| Sample ratio mismatch | **CONFIRMED** | `exposure_effect` checks crossover contamination only; no chi-square on assignment ratio. |
| Delta-method ratio metrics | **CONFIRMED** | None. |
| MDE / power warning | **PARTIAL** | `utils.check_min_sample` and the small-cell warning in `compare_proportions` exist, but they are fixed thresholds, not power for the observed effect. |
| Edge cases (empty segments, zero denominators, single-day windows) | **PARTIAL** | Handled in places: `compare_proportions` rejects n≤0 and guards se=0; `compare_means` handles zero variance; `time_series` returns `INSUFFICIENT_DATA` on empty; `compare_periods` floors duration at 1 day; zero-baseline relative change returns `None`. Not systematically covered by tests. |
| Expose new tools via catalogue with validation | **CONFIRMED** | See next row for what is and isn't exposed today. |
| Retention and feedback exposed? | **PARTIAL** | **`feedback_analysis` and `error_analysis` are exposed.** **`retention` and `retention_comparison` are implemented in the executor but NOT in `CONTROLLED_OPERATIONS`**, so the LLM cannot call them. Also unexposed: `duplicate_event_check`, `metric_aggregate`, `statistical_test`. `custom_query` is deliberately unexposed (keep it that way). |

## Phase 3 — Synthetic data realism

`GeneratorConfig` (`investigator_data/config.py`) has only: seed, start_time, num_days, num_users and four baseline rates. Hard-coded values live in `engine.py`.

| Knob | Verdict | Evidence |
|---|---|---|
| Noise level | **CONFIRMED** | None beyond Poisson/Bernoulli sampling. |
| Weekly seasonality | **CONFIRMED** | None in the engine. The baseline scenario text claims "stable weekly seasonality" (`scenarios.py`), but sessions are drawn uniformly over time (`engine.py`, session_offset). The description is inaccurate. |
| Holiday spikes/dips | **CONFIRMED** | None. |
| Effect size | **PARTIAL** | Exists as per-scenario `parameters` (e.g. `mobile_order_failure_prob`), not as a config knob. |
| Onset (step vs ramp) | **CONFIRMED** | Step only. The release cutoff `2026-09-15 10:00` is hard-coded in `engine._generate_events_and_orders` for **every** scenario (app_version v2.4.0), not read from the scenario. |
| Late-arriving data | **CONFIRMED** | None. |
| Duplicate events | **CONFIRMED** (generator) | `duplicate_event_check` exists in analytics, but the generator never produces duplicates. |
| Bot traffic share | **CONFIRMED** | None. |
| Multi-device identity splits | **CONFIRMED** | None; one user_id = one device/browser. |
| Event-name changes across versions | **CONFIRMED** | None; `app_version` varies but event names are constant. |
| 8 new scenarios | **CONFIRMED** | Registry holds only the 4 original scenarios. |
| Validators cover new knobs | n/a yet | `DataValidator` has schema, referential, window-bound, temporal and scenario-effect checks. |

## Phase 4 — Evaluation harness

| Gap | Verdict | Evidence |
|---|---|---|
| Real-LLM runner, scoring, sweep | **CONFIRMED** | Nothing exists. |
| Real LLM client | **PARTIAL** | `GenericHTTPClient` is OpenAI-compatible `chat/completions` only. No Anthropic client, no token/cost accounting (`LLMResponse` is defined but the client methods return plain strings or models). **Hazard:** `create_llm_client()` silently returns `MockLLMClient` when no key is set, so an eval run could report "results" from the mock. The harness must refuse to start without an explicit real provider. |

## Phase 5 — Safety hardening

| Gap | Verdict | Evidence |
|---|---|---|
| SQL read-only at DB level | **PARTIAL** | `AnalyticsExecutor._execute_sql_query` already sets a `sqlite3` authorizer that allows only SELECT/READ/FUNCTION/RECURSIVE, and a progress-handler timeout. The DB is a fresh `:memory:` copy per call, so there is nothing on disk to mutate. Missing: `PRAGMA query_only`/`mode=ro` (not applicable to `:memory:` as it stands), and bypass tests. Text filters (comment stripping, `;` split, keyword regex) are also present. Weak spots in the text layer: the split on `;` misfires on literals; the keyword list rejects legitimate uses such as `replace()`. Existing tests cover mutation, multi-statement, CTE, row bound and timeout, **not** comments, case tricks, PRAGMA or ATTACH. |
| LLM can't call ops/dimensions outside the catalogue | **PARTIAL** | `validate_analysis_plan` rejects unknown operations, unknown params, wrong types, and unsupported dimensions, **but only when `allowed_dimensions` is non-empty**. **`filters` are never validated** (any column name passes). `table` is an unrestricted `str` (the loader raises on unknown tables, so it can't reach other files). `custom_query` is not in the catalogue. |
| Budgets enforced | **PARTIAL** | `max_analyses` and `max_iterations` are enforced at the top of the loop (`test_max_analyses_budget_enforced`, `test_max_iterations_stops_with_insufficient_evidence`). `execute_analysis` itself is not guarded, and LLM calls are unbudgeted. |
| Cannot reference ground-truth paths | **PARTIAL** | Static guard added in Phase 0 (`tests/test_source_isolation.py`) and the `GROUND_TRUTH_FORBIDDEN_MARKERS` leakage check in `InvestigationState.validate`. `DataLoader` will still load any directory it is given. |
| Adversarial mock-LLM tests | **PARTIAL** | `test_invalid_tool_call_rejected_cleanly` and `test_capability_catalogue.py` cover the basics; no adversarial tests for filters, ground-truth paths, or budget overrun. |

## Phase 6 — Portability and output

| Gap | Verdict | Evidence |
|---|---|---|
| Schema mapping layer | **CONFIRMED** | Table/event names are hard-coded in `loader.CATALOGUE`, `data_quality.py` (`order_completed`, `checkout_started`), `capability_catalogue.py` examples, and the engine. |
| Backend interface | **CONFIRMED**, with a correction to the premise | There is **no SQLite default backend today**. Analytics modules take pandas DataFrames from `DataLoader`, which reads CSVs. SQLite is used only inside `custom_query`. "Keep SQLite as default" therefore means introducing a SQLite implementation, or the interface wraps the pandas loader first. Decision needed in Phase 6. |
| Investigation trace (md + JSON) | **PARTIAL** | `Investigation.history` is a chronological `HistoryEvent` log and `to_json()` exists. No markdown rendering, no explained shares or data requests (those arrive in Phase 1). |

## Duplicated or conflicting concepts

1. **Table/column schema, two sources of truth:** `investigator_data/schemas.TABLE_COLUMNS` (generator) vs `analytics/loader.CATALOGUE` (`FieldInfo`). The same columns are defined twice with no consistency test between them.
2. **Category vocabularies, four copies:** `HypothesisProposal.validate_category` (7), the list in `controller/prompts.py`, `Hypothesis.category` (unconstrained), and `ScenarioDefinition.category` (4 scenario names). Nothing ties them together.
3. **Status vocabularies overlap in name only:** `AnalysisStatus` (analytics) vs `InvestigationStatus` / `HypothesisStatus` (investigation). Distinct concepts, fine, but `step_interpret_result` re-parses status strings by hand with its own set literal.
4. **Time windows:** `TimeWindow` objects in `analytics/models.py`, but `Observation.to_dict` serialises them to `"[2026-09-01 – 2026-09-10)"` strings and `from_dict` re-parses that format (lossy: date-only). `Evidence.window` is serialised to a string and **dropped on `InvestigationState.from_dict`** (a round-trip bug; evidence windows are lost).
5. **Z-test implemented twice:** `statistical.compare_proportions` and the inline version in `release_analysis.exposure_effect`.
6. **Ground-truth marker lists:** `GROUND_TRUTH_FORBIDDEN_MARKERS` (`investigation/state.py`), a separate term list in `tests/test_separation.py`, and now the new static test. Should converge on one constant.
7. **Metric semantics:** `MetricDefinition` (`analytics/models.py`, used by `metrics.py`) vs `Observation.metric_name` (free string). No link between the observed metric and a defined metric.
8. **Hard-coded release cutoff:** `2026-09-15 10:00` appears in `engine.py` and independently in `scenarios.py` parameters.
9. **Controller hard-codes `source_tables=["orders", "events"]`** for every evidence item (`step_interpret_result`), regardless of which tables the operation actually read.

## Phase 0 changes made

- `.gitignore`: extended (env variants, sqlite/db/log, eval outputs, build caches). `data/` is deliberately **not** ignored (see report).
- `requirements.txt`: added `scipy`, `pydantic`.
- `tests/analytics/test_loader.py::test_loader_real_dataset`: replaced a hard-coded `C:/Users/ANISHA/OneDrive/Desktop/...` path with a repo-relative one. It had been silently skipping.
- New `tests/test_source_isolation.py` (4 tests): static guard that `controller/`, `investigation/`, `analytics/` neither import generator/scenario modules nor reference ground-truth paths.
