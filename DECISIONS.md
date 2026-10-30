# DECISIONS.md

Autonomous-run decision log. Format: phase · question · options · choice · why. `[REVIEW]` = unsure, please check.

## Phase 2a

1. **Raw p-values from `tracking_gap_check`.** Options: keep tool-level Bonferroni (old) / emit raw per-segment p-values.
   Chose raw, labelled (`p_values` dict), per instruction. Rewrote `test_pvalue_matches_hand_computed_...` (it asserted Bonferroni).
2. **BH family = registry, not evidence.** Every p-value a tool returns is registered when the analysis *executes*
   (`Investigation.test_registry`), so uninterpreted results still count. Chosen because dropping unfavourable results would
   otherwise shrink the family. Caller-supplied inputs (`statistical_test`) are excluded entirely: bogus small p-values would
   inflate BH ranks and help *other* hypotheses.
3. **"Underpowered" needs a target.** First version flagged `observed < MDE`; that is vacuously true for every non-significant
   result. Replaced by `mde > target_effect` (target given by caller, e.g. the observed change). For natural_variation the
   target is `nv_max_mde_ratio` (default 1.0) x |observed change|. `[REVIEW]` MDE units assumed equal to the observation metric's
   units (true when the same metric is analysed; no unit check exists).
4. **`nv_max_mde_ratio` added to `SupportPolicy`.** New field for correction 2 only; the defaults of the SUPPORTED rule
   (2/2/0/1/0.05/BH) are unchanged.
5. **Support-withdrawal audit.** `Hypothesis.support_snapshot` stores adjusted p-values at grant time to provide "before".
6. **Unknown-category warning.** The pydantic schema used to coerce unknown categories silently, hiding them from the state.
   Schema now keeps unknown names; the state coerces and logs a WARNING event. Updated my own Phase 1 assertion
   (`category="nonsense"`) accordingly.
7. **"Consistent with" wording.** `CONFIRMABLE_CATEGORIES = {tracking_loss, traffic_mix_shift, denominator_effect}` (measured
   directly by reconciliation / exact arithmetic). Everything else, including code_regression (a before/after or DiD cannot
   rule out confounders) and third_party_outage, is "consistent with". `[REVIEW]` code_regression classification.
8. **Effect estimates only from tools.** `record_evidence(estimated_effect=)` and `conclude(explained_changes=)` were removed.
   Estimates arrive via `result.metadata["effect_estimate"]` (method, SE or CI, window mandatory) and LLM output containing
   estimate/share keys is rejected by `EvidenceInterpretation`. Phase 1 tests that injected numbers now build tool-style
   `AnalysisResult`s. When several estimates support one cause the most precise (lowest SE) is used; different methods across
   causes raise a double-counting warning. `[REVIEW]` only `mix_shift_decomposition` produces estimates so far.
9. **Mix decomposition drives traffic_mix_shift and denominator_effect** (capability `mix_decomposition`), replacing the weaker
   `segment_comparison`. Test `test_testable_set...` updated: seasonality and cohort categories became testable.
10. **Tests of the mix/seasonal tools produce p-values** (`mix_homogeneity` chi-square, `overall_rate_change`, window-level
    seasonal deviation) so those categories can meet the significance requirement. `[REVIEW]` the seasonal p-value treats days as
    independent.
11. **`statistical_test` exposed but inert for support.** Its inputs are model-supplied numbers; its p-value is flagged
    `caller_supplied` and ignored by the support rule and the BH family.
12. **Seasonality effect estimate not produced** (how much of a change is "seasonal" depends on the baseline definition);
    seasonality causes stay unquantified. `[REVIEW]`
13. Junction/symlink test skip messages now state the reason.

## Phase 2b

1. **code_regression now needs `release_diff_in_diff`** (was `release_before_after`). `release_impact` has no significance
   test, so a code_regression could never meet the SUPPORTED rule. Updated one Phase 1 assertion (missing capability name).
2. **DiD exposure definition.** "Exposed" = user with >=1 event on the release version in the post window (a user-level,
   self-selected group). Alternatives (session-level, assigned-by-flag) rejected: pre-release nobody has the new version at
   session level. Caveats (parallel trends, selection) are in the tool's limitations. The effect estimate is
   `DiD x exposed share of post rows` so it is in units of the overall metric. `[REVIEW]` selection bias of upgraders.
3. **Change point.** Single split scan + seeded permutation test of the *max* statistic (naive per-split t-test p-values are
   anti-conservative). Ignores autocorrelation/weekly seasonality (stated in limitations). Step-vs-ramp flag by SSE.
4. **SRM threshold 0.001** (industry convention); on mismatch `exposure_effect` withholds its p-value (so it cannot enter the
   BH family or support a hypothesis) and sets `trusted=False`.
5. **Delta method** per-user clusters (`user` = independent unit). Only `revenue_per_session` / `orders_per_session` are named
   ratio metrics; the pure function `delta_method_ratio` is general.
6. **Edge sweep** is a parametrised test over EVERY catalogue operation x {empty tables, single-day window, no-data window}
   asserting a structured status instead of an exception. It passed without needing further fixes beyond those made in
   Phase 1/2a.

## Phase 3

1. **Weekly seasonality is ON by default for every scenario** (`weekly_seasonality=0.5`), not only baseline. Baseline claimed
   seasonality it did not have; a world where only baseline has it would make baseline-vs-scenario comparisons meaningless.
   Consequence: all tracked datasets were regenerated; the original four scenarios' cause parameters are unchanged and their
   quantitative tests still pass. `[REVIEW]` "Defaults reproduce current scenarios" holds for causes, not byte-for-byte data.
2. **Release timing derives from the scenario's release log** (`ScenarioDefinition.releases`); app_version of a session =
   latest release deployed at least `adoption_lag` earlier. This removed the hard-coded 2026-09-15 10:00 (and 09-08 14:00) cutoffs
   and exposed an existing inconsistency (the generator switched to v2.3.9 on Sep 8 while the release table said Sep 9).
   Feature rollout `one_click_checkout` now follows the scenario's mobile_checkout release.
3. **Mobile users adopt new versions gradually** (`app_adoption_days_mobile`, exponential, default 2 d; web = instant). Without
   partial adoption there is no unexposed group and difference-in-differences is impossible. The regression is now tied to
   *app version >= release*, so it is also inherited by v2.4.1 (as a real bug would be). `[REVIEW]` changes regression dynamics.
4. **Hidden generator artifact found: legacy world has every user sign up inside the window**, so activity grows ~50x over a month
   and confounds any pre/post comparison (a 45% price-driven drop was invisible). New knob `existing_user_share`; the six new
   scenarios that need a stationary world set 0.75; the original four, decoy_release and overlapping_causes keep the legacy world.
   `[REVIEW]` consider enabling it everywhere.
5. **Separate RNG stream (`aux`) for realism knobs** so toggling a knob does not reshuffle unrelated draws.
6. **Alert spec lives beside, not inside, ground_truth** (`data/<scn>/alert.json`): the metric and windows that fired the alert
   are what monitoring would show, and the harness needs them to build the Observation. A test asserts it carries no cause info.
7. **Ground truth gains `expected_answer`, `measured_effects` (counters recorded while generating, the basis for the
   overlapping-causes approximate shares) and the resolved `generator_config`.** The reproducibility test was updated for these.
8. **Scenario-specific volume boosts** (`baseline_checkout_start_prob` 0.30 / conversion 0.20 for denominator_effect and
   third_party_outage) so order counts are large enough for the effect to be statistically visible; outage window widened to
   12 h at 90% after 6 failures proved too few to detect.
9. **Denominator effect implemented as campaign traffic switched off** (sessions of paid_social users thinned after the date),
   not as fewer signups: with a stationary user base most paid-social users pre-exist the cut.
10. **Validator**: now also flags unexpected columns (the leak found in Phase 1), release/version consistency, and each configured
    knob that left no fingerprint (and vice versa). The temporal check was vectorised (pure speed-up, same rule).
11. Test runtime: the Phase 3 module shares one 14 s fixture (generates all 12 scenarios once). `[REVIEW]` slow setup.

## Phase 4

1. **Offline verification uses a rule-based reference policy** (`evaluation/mock_policy.py`) behind `MockLLMClient(generator_fn=...)`.
   It sees only prompt text and tool output (never ground truth) and is deterministic. Its accuracy (5/12 scenarios fully right
   with 1 run each) is NOT an estimate of any model's ability; every report from it carries a "MOCK RUN" banner. `[REVIEW]` do not
   quote its numbers.
2. **A real evaluation can never fall back to the mock.** `resolve_client` raises `EvalConfigError` for `mock` without `--allow-mock`,
   for a missing API key, or a missing model; `assert_real` re-checks the resolved client (including inside the metering wrapper).
   The CLI exits with code 3 and prints "REFUSING TO RUN".
3. **Anthropic client added** (`evaluation/clients.py`, standard library only) alongside the existing OpenAI-compatible client.
   Token usage comes from the API response; if a client reports none, tokens are estimated as chars/4 and flagged
   `tokens_estimated`. Cost is computed only when `--price-in/--price-out` are given (never guessed).
5. **Alert -> Observation** is computed by the harness from investigator tables only (`evaluation/alerts.py`); ground truth is read
   only by `scoring.py`. A test asserts no prompt contains any ground-truth text and that controller/investigation/analytics never
   import `evaluation`.
6. **Scoring.** run_correct = right primary (or right INSUFFICIENT_EVIDENCE with an on-topic data request) AND no unexpected supported
   cause. Overlapping causes' share error compares the model's normalised code_regression share with the share measured while
   generating (regression-failed orders vs dropped events). `[REVIEW]` that "measured share" definition is approximate.
7. **Sweep.** Smallest *reliably* detected effect = smallest effect whose detection rate AND every larger effect's rate is >= 80%
   (a lucky cell in the middle does not count). Chart is a dependency-free SVG.
8. **Bugs found by running the harness and fixed:**
   - `Evidence.window` held a *string* from the controller, which broke serialisation (now coerced to `TimeWindow`);
   - the catalogue advertised `funnel_steps`/`breakdown` but the funnel tools read `steps`/`dimensions`, so **funnel and
     funnel_comparison never worked through the catalogue**; funnel also did not implement breakdowns at all (now it does);
   - `error_analysis` could not break down by `failure_reason` (the tool's headline use) because that dimension was not allowed;
   - no tool returned a p-value for third_party_outage evidence, so that category could never be SUPPORTED: added `spike_check`
     (raw per-day p-values) and re-based the category on `spike_detection`;
   - the model was never shown release ids/tables, so it could not call `release_did`: prompts now include a data context
     (tables+coverage, release log, flags, event names);
   - 1-day alert windows made time-based tools useless: `lookback_days` (controller-applied) for time_series/change_point/spike_check;
   - mix decomposition had no session-level unit: added a derived `sessions` table (not listed as on-disk data).
9. **Seasonality semantics.** seasonal_baseline emits two p-values: `weekday_pattern` (LOW = real weekday pattern; this is what meets
   the significance requirement for the seasonality category) and `seasonal_deviation` (LOW = NOT explained by seasonality).
   Supporting evidence for seasonality therefore needs the first low AND the second high; the SUPPORTED rule only enforces the first,
   the second is the investigator's job. `[REVIEW]` consider encoding this in the rule.
10. **BH across a broad investigation is demanding.** An investigator that tests all categories enters ~11-25 tests into one family,
    so borderline effects (e.g. weekday pattern p=0.02) cannot become SUPPORTED. This is the specified behaviour; noted because it
    lowers recall for investigators that test broadly. `[REVIEW]`
11. Scenario tweaks to make effects answerable: seasonal_dip alert now covers the weekend only (the holiday dip stays a world
    feature), 49 days of history; decoy_release gets the stationary, higher-volume world.

## Phase 5

1. **SQL: read-only enforced at the database level.** Tables are exported to a temp SQLite *file* opened with URI `mode=ro` (plus
   `PRAGMA query_only`, an authorizer, `SQLITE_LIMIT_ATTACHED=0`). Previously the DB was an in-memory copy guarded by an authorizer
   and regexes. A test proves the `mode=ro` connection alone refuses every write.
2. **The text that is checked is the text that is executed** (comments replaced by a space, as SQLite tokenises them), and string
   literals are ignored by the keyword scan, removing false positives such as `WHERE failure_reason = 'update'` and `replace()`.
3. **Authorizer reads:** schema tables (`sqlite_*`) are denied by name. Whitelisting exported table names broke recursive CTEs
   (CTE names reach the authorizer as table names); the file only contains investigator tables so this is equivalent. Functions are
   whitelisted (e.g. `hex`, `randomblob`, `load_extension`, `readfile` denied). `[REVIEW]` function whitelist breadth.
4. **Limits are clamped inside the sandbox** (rows <= 5000, timeout <= 10 s, query <= 10 000 chars) whatever the caller passes.
5. **Catalogue:** closed vocabularies for `table`/`aggregation`/`freq`/`time_column`/`order_status`; every string parameter must match
   `[A-Za-z0-9_ .:+-]{0,200}` with no path separators, `..` or ground-truth names; lists <= 50 items; bool is not an int; NaN/inf
   rejected. `custom_query` stays outside the catalogue.
6. **Budgets:** new LLM-call budget (default `4*max_analyses + 10`) enforced in one `_llm` choke point; `execute_analysis` itself now
   refuses past `max_analyses`; operation names are normalised before dispatch.
7. `tests/test_source_isolation.py`: `capability_catalogue.py` and `sql_sandbox.py` join the guard-file allowlist (they must name
   "ground_truth" to block it; the test still forbids them from reading files).
8. Test speed: the infinite-recursion timeout test monkeypatches the hard cap to 1 s (the real cap, 10 s, is asserted separately).

## Phase 6

1. **Backend interface wraps the pandas loader.** `analytics.backend.DataBackend` (abstract: `load`, `available_tables`; defaults for
   `load_all`, `row_count`, `time_coverage`, `clear_cache`). `DataLoader` is the default implementation; the executor depends only on
   the interface. No SQLite or warehouse backend was built (SQLite remains only inside the `custom_query` sandbox).
2. **Mapping is JSON, not YAML** (no new dependency). Sections: tables, columns, event_names, values. `MappedBackend` applies it at
   load time so the catalogue, tools and prompts keep one internal vocabulary; tables missing from the raw source are unavailable and
   testability shrinks accordingly. `[REVIEW]` mapping handles renames and value/event translation only, not derived columns
   or unit conversions.
3. **Trace** is built from the investigation's own append-only history (so it can be regenerated from a saved state). Analysis events
   now record parameters, dimensions, filters and windows. `[REVIEW]` `run_eval.py` does not yet save traces automatically.
