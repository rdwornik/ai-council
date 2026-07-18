# Code-Quality Audit — ai-council `src/` (implementation ground truth)

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — SEED-1/2/3/4 shipped (#16/#22/#25/#26); open: #45–#48 (SEED-5/utcnow/dead-code/RunPolicy), #20. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-06 · **Model:** Fable (severity calibration + whole-codebase judgment) · **Mode:** execution, READ-ONLY (this report is the only repo artifact) · **Effort:** high · **Verified against:** live `main` @ `796aa50` (branch point), first-hand reads + `radon`/`vulture`/`mypy`/`ast` on the checked-out tree · **Status:** committed record

> **Companion to the intake.** The technical-architect intake (`2026-07-06-technical-architect-intake.md`) tells you WHAT to build. This tells you what you are building ON: is the code clean, consistent, readable, simple, efficient — or accreted and patched? **Zero refactoring was performed.** Findings are refactor *seeds* (§5) for the architect to fold into the backlog. This audit judges IMPLEMENTATION quality only; functional/design gaps live in the 07-04/07-05/07-06 lane docs and are referenced, never refiled (§6).

---

## 1. Executive verdict

**Overall grade: B+ — engineered, not patched.**

**Is this codebase patched or engineered? Engineered.** It is disciplined, well-documented, and consistently styled — its dominant debt is **duplication, not accretion**: two provider families at ~85% copy-paste with zero template-method leverage, and a single **277-line, F-rank-72 `cli.py:main`** carrying two partially-drifted request paths. Both sit directly under the planned Wave-C builds (CliProvider v1, D2 parity, doctor).

The "engineered" call is evidence-backed, not vibes: **zero bare `except:`, zero commented-out code, zero `TODO`/`FIXME`/`HACK` markers** across 4,744 LOC; `logging.getLogger(__name__)` used in every module; pure dataclasses for the domain layer with `TYPE_CHECKING` import discipline; clean package layering with no hard import cycles; **every maintainability-index score is rank A**; average cyclomatic complexity is **B (5.14)**; the provider ABCs are minimal and correct; parallelism (`asyncio.gather` / `create_task`) is used everywhere with no sync-in-async and no N+1 API calls; and the 432-test suite mocks at the right seam (the `AIProvider` ABC) and asserts mostly on behavior. This is a codebase someone thought about.

The debt is concentrated and nameable, not diffuse. **Top-5 findings by risk to the upcoming builds:**

| # | Finding | Evidence | Risk to Wave-C | Verdict |
|---|---|---|---|---|
| 1 | `cli.py:main` is a 277-LOC / F(72) function with two partially-duplicated request-building paths (inbox loop vs single-question) | `cli.py:359-635`; inbox path `445-562` vs single path `565-635` each independently resolve mode/panel/synthesizer, build `RunRequest`, and call `run_research` | D2 parity (R1) + doctor v1 (main → subcommand) land **here**; the known D2 `--file` gap is a *symptom* of this split | **REFACTOR-FIRST** |
| 2 | Provider families ~85% boilerplate, **no template method**, manual registries | Family A ~85% / Family B ~72-88% copy-paste; `xai.py`≡`deepseek.py` (12-line diff, zero logic); both bases pure-interface | CliProvider v1 (#16) extends Family A → a **6th wholesale copy** + hand-edited registry | **WORKABLE + strong seed** |
| 3 | Dual, half-wired error/retry policy; retry path reaches through the ABC | `policy.should_retry` wired (`debate.py:50`) but `should_abort`/`max_retries_per_provider`/`base.is_retryable` are **tested-but-dead**; retry mutates private `provider._config.timeout_sec` in place (`debate.py:52-56`) — a contract the ABC never declares | CliProvider must replicate the implicit `_config.timeout_sec` contract; #18/#19 hook policy | **seed (SEED-3)** |
| 4 | `save_to_file` is E(33) / 138 LOC — the 2nd-most-complex function — and growing | `output.py:179-316` mixes metadata derivation + markdown assembly + routing + metrics trigger | verdict package (DRAFT-INT-1) + `seats[]`/`synthesis` namespaces pile onto `output.py` | **WORKABLE** (clean `_write_routed`/`_save_metrics_json` seams mitigate) |
| 5 | `orchestrator`↔`runner` backward-compat re-export shim (soft cycle) muddies the eviction/synthesis seam | `runner.py:79` re-exports `CouncilRunner` from `orchestrator` (`noqa: E402`); `orchestrator.py:20` imports utilities from `runner` | eviction/synthesis is where #18/#19 (baseline-gated) will hook | **WORKABLE** (seam REFACTOR-FIRST) |

**Refactor seeds produced: 5** (§5). All five load-bearing surfaces have verdicts (§4). Nothing outside this report was changed.

---

## 2. Mechanical evidence (Phase 1)

### 2.1 Inventory

| Package | Modules (code) | LOC | Share | Note |
|---|---|---|---|---|
| `ai_council/` (core) | 13 | 2,435 | 51% | cli, output, debate, orchestrator, inbox, synthesis, models, runner, routing, healthcheck, metrics, mode_detector, policy |
| `ai_council/providers/` (debate) | 5 + base | 543 | 11% | ABC + 5 providers |
| `ai_council/research/` (core) | 8 | 1,013 | 21% | runner, merger, output, display, cache, provider, models |
| `ai_council/research/providers/` | 5 + base-in-core | 753 | 16% | 5 research providers |
| **Total `src`** | **30 code (34 files)** | **4,744** | | 143 functions/methods |
| `tests/` | 20 files | 6,583 | | **432 tests** collected (test:src ratio ≈ 1.39:1) |

**The research subsystem is ~37% of `src` and is a structural parallel of the debate half** (its own `runner.py`, `output.py`, `providers/`, `models.py`, ABC) — a deliberate separate code path (ADR-05), but it means the debate half's smells are mirrored on the research side (see §2.3, §3).

### 2.2 Longest functions (by source LOC — `ast`)

| LOC | Function | File:line | CC rank |
|---|---|---|---|
| 277 | `main` | `cli.py:359` | **F (72)** |
| 166 | `CouncilRunner.run` | `orchestrator.py:34` | D (22) |
| 138 | `save_to_file` | `output.py:179` | **E (33)** |
| 110 | `run_debate` | `debate.py:186` | C (17) |
| 97 | `synthesize` | `synthesis.py:51` | B |
| 94 | `run_research` | `research/runner.py:133` | C (14) |
| 86 | `save_research_to_file` | `research/output.py:29` | **E (33)** |
| 79 | `run_research_with_display` | `research/display.py:87` | C (16) |
| 72 | `_call_provider` | `debate.py:37` | C (12) |
| 68 | `print_research_summary` | `research/output.py:117` | C (20) |

**20 of 143 functions exceed 50 LOC (14%); 41 exceed 30 LOC (29%).** The tail is fine; the head (top ~4) is genuinely oversized. `research/output.py` carries two of the top-10 — the research output layer mirrors `output.py`'s hotspots.

### 2.3 Complexity & maintainability (`radon cc -a`, `radon mi`)

- **Average complexity: B (5.14)** across 170 blocks. Distribution: **114 A · 37 B · 15 C · 1 D · 2 E · 1 F**. Only 4 blocks are worse than C — the codebase is overwhelmingly simple; the hotspots are few and named.
- **CC offenders:** `main` **F(72)** ‖ `save_to_file` **E(33)** ‖ `classify_error` **E(33)** ‖ `CouncilRunner.run` D(22) ‖ `print_research_summary` C(20) ‖ `run_debate`/`save_research_to_file` C(17) ‖ `run_research_with_display` C(16) ‖ `run_research` C(14) ‖ `_extract_content`×3 C(11-12) (the research-provider duplication showing up as identical complexity).
- **Maintainability index — all rank A, but the low end is the story:**

| Lowest MI (danger) | Highest MI (healthiest) |
|---|---|
| `cli.py` **22.95** | `healthcheck.py` 79.51 |
| `output.py` 33.86 | `policy.py` 78.77 |
| `research/providers/grok_research.py` 40.20 | `mode_detector.py` 78.70 |
| `research/output.py` 44.39 | `research/models.py` 78.17 |
| `openai_deep_research.py` 45.27 | `routing.py` 76.03 |

  The six debate providers cluster tightly at **MI 47-48** (`anthropic` 47.31, `deepseek` 47.78, `xai` 47.78, `openai` 48.67, `gemini` 59.05) — a **duplication fingerprint**: near-identical files score near-identical MI.

### 2.4 Duplication (`diff` + first-hand reads)

**Family A — debate providers: ~85% boilerplate.** ~66 of 436 LOC is genuinely provider-specific. The `name()`, `model_string()`, `time.monotonic()` timing, `asyncio.wait_for(..., timeout=self._config.timeout_sec)` wrapper, both `except → ProviderError` blocks ("Request timed out after…", "API call failed:…"), empty-response check, `logger.info(...)`, and the 9-field `return ModelResponse(...)` block are **byte-identical across all five** (e.g. the `ModelResponse` block at `anthropic.py:79`, `gemini.py:89`, `openai_provider.py:74`, `xai.py:76`, `deepseek.py:78`). **`xai.py` ≡ `deepseek.py` — a 12-line diff with zero logic delta** (docstring, class name, one error-noun, logger label). The only legitimately-divergent code is client construction + the SDK call + token extraction.

**Family B — research providers: ~72-88% boilerplate.** `openai_deep_research.py` and `openai_mini_research.py` are **>90% identical — one genuine logic line differs** (`reasoning={"effort": …}` at `openai_deep_research.py:108`). `_extract_content` / `_extract_sources` / `_collect_annotations` are **triplicated verbatim (~150 LOC)** across grok/deep/mini. `gemini_research.py` (Interactions API poll-loop) is the one legitimate outlier (~90 unique LOC). Copy-paste hazards pasted 5×: the deprecated `datetime.utcnow()` timestamp (§2.7) and the cost formula with magic literal `1_000_000`.

**Both ABCs are pure interface — no template method.** `AIProvider` (`base.py:80-107`) and `ResearchProvider` (`research/provider.py:16-42`) declare only `name`/`model_string`/`generate`|`research` with **zero shared implementation**. Registries are manual: `PROVIDER_CLASSES` dict (`cli.py:38-45`) and the `if/elif` chain (`research/runner.py:68-119`). **Adding a provider today = copy a whole file + hand-edit a registry branch.**

### 2.5 Dead / unwired code (`vulture` @ 60%, cross-checked against callers)

`vulture` flags cross-checked against `grep` for real callers (production vs tests-only):

| Symbol | Site | Status |
|---|---|---|
| `is_retryable` | `base.py:67` | **Unwired** — tested (`test_base_provider.py`), **0 production callers** |
| `should_abort` | `policy.py:23` | **Unwired** — tested (`test_policy.py`), **0 production callers** |
| `max_retries_per_provider` | `policy.py:12` | **Dead knob** — never read; retry count is hardcoded to "once" (`debate.py:43-94`), and `RunPolicy` is always `.default()` (never configurable) |
| `model_string` (×12) | ABCs + 10 providers | **Abstract + implemented everywhere, called only by tests** — 0 production callers. Every provider must implement a method nothing in the app calls |
| `cache_invalidate` | `research/cache.py:133` | Public util, tests-only — acceptable library surface |
| `_target_projects` | `routing.py:26` | Assigned in `__init__`, **never read** — write-only state |
| `done_tasks` / `snippet` | `research/display.py:135,149` / `research/models.py:12` | Genuine unused locals — trivial |

`classify_error` (`base.py:16`) **is** wired (`healthcheck.py:52`, `synthesis.py:101`). The theme: **a policy/retry/abort layer built and unit-tested ahead of wiring.** See SEED-3.

### 2.6 Import structure — clean, one soft cycle

Layering is correct and one-directional almost everywhere: `models ← policy` (via `TYPE_CHECKING` to avoid the runtime edge, `models.py:9`) · `providers ← models + base` · `orchestrator ← debate/output/synthesis/runner` · `cli` is the composition root importing everything. The research subtree is a clean tree rooted at `research/runner`. **The debate and research provider families never cross-import** (ADR "keep separate" honored). **No hard cycles.** The one blemish: `runner.py:79` re-exports `CouncilRunner` from `orchestrator` for backward-compat (`noqa: E402, F401`) while `orchestrator.py:20` imports two utilities *from* `runner` — a soft cycle left by a historical split (both module docstrings narrate it). New code still importing `CouncilRunner` from `runner` (as `cli.py:26` does) perpetuates it. See SEED-5.

### 2.7 Consistency scans

| Dimension | Finding |
|---|---|
| **Error handling** | **0 bare `except:`**. 25 broad `except Exception` — all the *wrap-and-classify* pattern (SDK error → `ProviderError`), alongside specific catches (`APITimeoutError`, `APIError`, `RoutingError`, `ProviderError`, `asyncio.TimeoutError`, `ValueError`). Appropriate for a resilient multi-provider tool. |
| **Logging vs print** | `logging.getLogger(__name__)` in **every** module — consistent. Rich `console.print` for human UX. 3 raw `print(json…, file=sys.stdout)` (`orchestrator.py:197`, `research/runner.py:174,224`) — intentional machine-readable `--json` output. Clean separation. |
| **Data modeling** | `@dataclass` for the domain (`models.py` 9, `research/models.py` 3, `policy.py`); `dict[str,X]` only for keyed collections (provider maps, statuses). No raw-dict-as-object smell. No Pydantic — plain dataclasses. |
| **Debt markers** | **0** `TODO`/`FIXME`/`HACK`/`XXX`; **0** commented-out code blocks. |
| **Comments** | `radon raw`: 259 docstring blocks, ~3-4% inline-comment ratio — docstring-heavy with genuine WHY-comments (e.g. the gemini per-call-client rationale, `gemini.py:26-30`). Healthy. |
| **Naming** | snake_case throughout; private helpers `_`-prefixed consistently. |

### 2.8 Types (`mypy --strict`)

**21 errors (19 in `src/ai_council/`, 2 in `config/`).** The **known #20 six** are the SDK-typing errors in the three research providers (`openai_mini_research.py:156`, `openai_deep_research.py:101,154`, `grok_research.py:65,151` — `.create()` overloads, object-not-iterable, TypedDict `misc`). The remaining ~13 are trivial and tolerated by the project's own gate: bare-generic `type-arg` (`dict`/`list`/`Task`, 8×) and `no-untyped-def` missing annotations (5×, incl. `orchestrator.py:34` `run`'s `output_dir` param). Type-hint coverage is otherwise high. *(#20 not refiled — §6.)*

### 2.9 `datetime` usage

- **`datetime.utcnow()` — 5 sites, all in research providers** (`gemini_research.py:55`, `grok_research.py:53`, `openai_deep_research.py:56`, `openai_mini_research.py:54`, `perplexity.py:48`). Deprecated in 3.12 (repo pins ≥3.12); the copy-paste means one fix touches 5 files. *(Known deferred sweep per the audit brief — but see §6: it has no doc citation anywhere.)*
- **Naive `datetime.now()` — 7 sites** for filename/header timestamps (`inbox.py:153`, `output.py:196,239,421,429`, `research/output.py:39,47`). `research/cache.py` correctly uses tz-aware `datetime.now(timezone.utc)`. Minor inconsistency, not elevated.

### 2.10 Config discipline

**Good.** Models, prompts, personas, per-provider `timeout_sec`/`max_tokens`, and the research `min_successful_providers` all live in `settings.yaml`. **Exceptions worth the architect's eye:**
- **`RunPolicy` is always `.default()`** (`cli.py:443`, `debate.py:214`) — never loaded from YAML. So `min_panel_size`, `abort_if_round1_below`, `max_retries_per_provider`, and the `retryable_errors`/`non_retryable_errors` lists are **code-only and unconfigurable**.
- **Two "3-provider quality floor" numbers:** `_MIN_QUALITY_RESPONSES = 3` hardcoded in `debate.py:16` vs `min_successful_providers: 3` in `settings.yaml:397` (research). Same concept, two homes.
- `healthcheck._DEFAULT_TIMEOUT_SEC`/`_MAX_TIMEOUT_SEC` are hardcoded safety caps — fine.

### 2.11 Efficiency

**No material findings.** All three subsystems run providers concurrently — `asyncio.gather` (`debate.py:244`, `healthcheck.py:67`) and `asyncio.create_task` + `as_completed` (`research/display.py:130`). **No sync-in-async** (grep: zero `time.sleep`/`requests`/`urllib` anywhere). No N+1 API loops; no redundant full-transcript re-passes (the `save_to_file` panel/synth derivations are trivial in-memory scans). `gemini.py` constructs a client per `generate()` call — deliberate and documented (event-loop binding across `asyncio.run()` boundaries, `gemini.py:26-30`). No micro-optimization theater warranted.

---

## 3. Package-by-package assessment

**`ai_council/` core — B.** The orchestration spine is correct and the domain layer is clean. `models.py` (**A-**): pure dataclasses, `TYPE_CHECKING` discipline, every field documented (`provider_statuses` is mildly duplicated across `DebateOutcome`/`DebateResult`; `DebateResult` has grown to 15 fields but each is defaulted). `debate.py` (**B+**): `run_debate` is a readable linear round-loop; blind-voting `_anonymize_responses` is isolated (ADR-03 contract intact); the retry `_call_provider` is the one wart (reaches through the ABC into `_config.timeout_sec`). `synthesis.py`/`metrics.py`/`mode_detector.py`/`routing.py`/`policy.py`/`inbox.py` (**A-/B+**): small, cohesive, well-tested. `cli.py` (**C+**) and `output.py` (**B**) carry the core's debt — see §4.

**`ai_council/providers/` — B-.** Correct, behaviorally-tested ABC implementations with a **clean contract** but **~85% copy-paste and no base leverage**. The base ships a 9-category `classify_error` taxonomy the providers ignore in favor of a cruder generic `except`. Grade held down purely by duplication, not correctness.

**`ai_council/research/` core — B.** Well-separated (cache/merger/display/output/runner/models), parallelized, cache-keyed (`make_cache_key` correctly lives in `merger.py` per the known gotcha). Mirrors the debate half's structure *and* its complexity hotspots: `save_research_to_file` is E(33), `print_research_summary` C(20) — the same "long save/print function" smell as `output.py`.

**`ai_council/research/providers/` — B-.** Same duplication story as Family A, slightly worse: triplicated `_extract_*` helpers (~150 LOC), `utcnow()`×5, cost-formula×5. `gemini_research.py` legitimately bespoke.

**`tests/` — B+.** A strong, behavior-oriented suite mocking at the `AIProvider` ABC seam (`conftest.py` `MockProvider` shadow-then-`AsyncMock`, matching CLAUDE.md §10). All six load-bearing seams have coverage (§4 table). Central fixtures (`conftest.py:79-150`) are well-reused. **Brittleness is concentrated, not pervasive:** positional-arg assertions on `save_to_file` (`test_runner.py:275,305,334`), exact SDK poll-count/`call_args.kwargs` (`test_research.py:553,846,…`), and pokes at the undeclared private `_config` (`test_debate.py:275`). `test_research.py` (1,760 LOC) is **organized** (~22 `TestXxx` classes) but heavy on per-provider SDK mock scaffolding (six `_make_provider` helpers, a ~200-line Gemini mock duplicated between two classes) — it should be split into `test_research_<provider>.py` before it grows further. **Known flake W1 root cause identified — §4.5.**

---

## 4. Load-bearing surfaces verdict (Phase 3 — the centerpiece)

For each surface a planned Wave-C build will touch: **SOLID** (extend as-is) / **WORKABLE** (extend with care; watch the named risk) / **REFACTOR-FIRST** (a minimal pre-refactor should precede the build).

### 4.1 `cli.py` — **REFACTOR-FIRST**
*Wave-C here: D2 parity (R1), secrets-loading rule (DRAFT-DOC-3), doctor v1 (main → subcommand).*

`main` is a **277-LOC / F(72)** function (`cli.py:359-635`, lowest MI in the repo at 22.95). It inlines two large, partially-duplicated request-building paths: the **inbox loop** (`445-562`) and the **single-question path** (`565-635`), which independently resolve mode/panel/synthesizer precedence, build a `RunRequest`, and invoke `run_research`. They have already **drifted**: the inbox path parses frontmatter via `parse_file` (`:468`) while the single `--file` path does a raw `read_text` (`:565-566`) — this *is* the known D2 `--file` gap, and it exists **because** the two paths are copy-pasted rather than shared. Doctor v1 additionally needs `main` to become a subcommand dispatcher (today it is a single `@click.command`). The helper functions above `main` (`_check_and_filter_providers`, `_select_health_check_targets`, `_interactive_confirm_mode`, …) are clean and single-purpose — the debt is entirely the un-extracted orchestration in `main`. → **SEED-1.**

### 4.2 `providers/` + `research/providers/` — **WORKABLE** (ABC clean) **+ strong seed**
*Wave-C here: CliProvider v1 / backend axis (#16).*

The `AIProvider` **contract is a clean seam** — three correct abstract methods; a new backend behind it needs no surgery to the *interface*. But the family has **zero template-method leverage** (§2.4): CliProvider v1 will be a **6th wholesale copy** of the ~85%-boilerplate skeleton plus a hand-added `PROVIDER_CLASSES` branch. It will *work* — hence WORKABLE, not REFACTOR-FIRST — but it deepens the copy-paste. There is also an **undeclared implicit contract**: the runtime reads `provider._config.timeout_sec` via `getattr` in both `debate.py:52` and `healthcheck.py:34`, which the ABC never declares — CliProvider must replicate it blindly. → **SEED-2** (template base) **+ SEED-3** (declare the `_config` contract). *Note: SEED-2 keeps each provider a distinct class/file — it is NOT a provider merge, so it does not touch the "keep xai/deepseek separate" ADR; but because it sits on an ADR boundary, the architect ratifies it.*

### 4.3 `output.py` — **WORKABLE**
*Wave-C here: verdict package (DRAFT-INT-1), `seats[]` + `synthesis` metrics namespaces.*

The two seams the builds actually extend are **clean and additive**: `_write_routed` (`:121-176`, the canonical+secondary+return+targets router) and `_save_metrics_json` (`:450-487`, already namespaced with `calls[]`/`synthesis`). The verdict package can be a sibling emitter calling `_write_routed`; `seats[]`/`synthesis` slot into `_save_metrics_json` without restructuring. **The watch item:** `save_to_file` is already **E(33) / 138 LOC** (`:179-316`) — the 2nd-most-complex function — mixing metadata derivation, markdown assembly, routing, and the metrics trigger. Adding verdict emission into its body would push it past readability. → **SEED-4** (decompose header assembly first; keep new emission out of `save_to_file`).

### 4.4 `healthcheck.py` — **SOLID**
*Wave-C here: doctor v1.*

68 LOC, **highest MI in the repo (79.51)**, no offenders. `run_health_checks` (`:58`, parallel `gather`) and `classify_error` (the 9-class taxonomy) are exactly what L-DOC specifies the doctor reuse — mode-proven, clean return contract `dict[name → (ok, msg)]`. Doctor v1 grows from this **without a pre-refactor**. The only implicit dependency is the same `getattr(provider,"_config")` timeout probe (`:34`) addressed by SEED-3. No seed of its own.

### 4.5 `orchestrator.py` / `runner.py` — **WORKABLE** (eviction correct; **seam REFACTOR-FIRST**)
*Wave-C here: eviction/synthesis path; #18/#19 hook later (baseline-gated).*

The **eviction/synthesis mechanism itself is correct** (`exclude_synthesizer_from_panel` → `pick_synthesizer`, `orchestrator.py:61-74` → `runner.py:44-75`) and behaviorally tested (`test_runner.py:108-154`). `CouncilRunner.run` is long (166 LOC, D-22) but a readable linear pipeline. **The seam is the debt:** the `runner`↔`orchestrator` backward-compat re-export (§2.6) means the module that *should* hold only panel utilities also re-exports the orchestrator, and `cli.py` imports `CouncilRunner` through the deprecated path. Before #18/#19 add policy hooks here, the seam should be made one-directional. → **SEED-5.** *(The synthesizer-identity concern is EPI-2's, not a code-quality issue — the mechanism is sound.)*

---

## 5. Refactor seeds

Seeds only — **no backlog items, no execution.** Each `done-when` is a testable hard end-state (per the intake's standing rule), not "tests pass."

### SEED-1 — Extract the per-question request build/dispatch from `cli.py:main`
- **Finding:** 277-LOC / F(72) `main` with two drifted request-building paths (§4.1).
- **Seed:** Extract a single `build_request(question_text, meta, config, cli_overrides) → RunRequest` and a `dispatch(request, …)` used by **both** the inbox loop and the single-question path; route `--file` and inbox through **one** frontmatter-parse.
- **Done-when:** `radon cc` rank of `main` ≤ C (≤15); `main` ≤ 60 LOC; exactly **one** `parse_file` call site feeds question-building (grep); exactly **one** `run_research` call site; `--file` and inbox produce identical `RunRequest` fields for identical frontmatter (a parity test, not "tests pass").
- **Precedes:** D2 parity (R1) — the `--file` gap becomes a one-line fix in the shared path; **enables** doctor v1 (main → `@click.group` dispatcher).

### SEED-2 — Template-method base for the provider families (optional; ADR-gated)
- **Finding:** ~85% (A) / ~72-88% (B) copy-paste, no base leverage; a new provider is a wholesale copy (§2.4).
- **Seed:** A base `generate()`/`research()` that owns timing + error-wrapping + `ModelResponse`/`ResearchResult` construction and calls abstract hooks (`_invoke(prompt)` / `_parse(response)`); hoist the triplicated `_extract_*` helpers (Family B) and the cost formula into the base. **Each provider stays its own class/file** (not a merge).
- **Done-when:** adding a new OpenAI-compatible provider is < 30 LOC of provider-specific code; `xai.py`/`deepseek.py` shrink to ≤ ~15 LOC each; a new `ModelResponse` field is wired in **one** place, not five; the `utcnow()` timestamp and cost literal `1_000_000` each appear **once** (grep count = 1).
- **Accompanies:** CliProvider v1 (#16). **FLAG:** sits on the "keep providers separate" ADR boundary — architect ratifies before build.

### SEED-3 — Unify error classification; wire-or-delete the dead policy layer; declare the `_config` contract
- **Finding:** two classification systems (`policy.should_retry` substring-match vs `base.classify_error` taxonomy); `should_abort`/`max_retries_per_provider`/`base.is_retryable` tested-but-dead; retry mutates private `_config.timeout_sec` through the ABC (§2.5, §4.2).
- **Seed:** Pick one classifier (the canonical `classify_error`→`is_retryable` pathway is the better one) and route the retry decision through it; either wire `should_abort` + `max_retries_per_provider` into the runtime or delete them; add the timeout-override / `_config` expectation to the `AIProvider` contract (or a base attribute) so the runtime stops reaching in via `getattr`.
- **Done-when:** exactly one error-classification code path (grep: `should_retry` and `is_retryable` not both live); **zero** tested-but-unwired policy methods (`vulture` clean on `policy.py`/`base.py` for these symbols); the runtime reads no provider attribute the ABC doesn't declare.
- **Precedes/accompanies:** CliProvider v1 (implicit contract) and any #18/#19 policy work.

### SEED-4 — Decompose `save_to_file` before the verdict package lands
- **Finding:** `save_to_file` E(33) / 138 LOC, growing (§4.3).
- **Seed:** Extract header-metadata assembly (`output.py:196-271`) into `_build_header(result) → list[str]`; keep verdict-package emission a **sibling** function calling `_write_routed`, not new lines inside `save_to_file`.
- **Done-when:** `save_to_file` `radon cc` ≤ C (≤15); verdict-package artifact is emitted by a function that does not also assemble the transcript; `_write_routed` remains the single write primitive (one definition).
- **Accompanies:** verdict package (DRAFT-INT-1, R2).

### SEED-5 — Dissolve the `runner`→`orchestrator` re-export shim
- **Finding:** soft import cycle from the backward-compat re-export (§2.6, §4.5).
- **Seed:** Make imports one-directional — `runner.py` holds only panel utilities; all `CouncilRunner` importers (incl. `cli.py:26`) import from `ai_council.orchestrator`; remove the `noqa: E402` re-export.
- **Done-when:** `runner.py` contains no `from ai_council.orchestrator import` (grep); no module-level cycle (an `import-linter`/`pydeps` contract passes); `CouncilRunner` has exactly one import source.
- **Precedes:** #18/#19 (baseline-gated) hooking the eviction/synthesis path.

---

## 6. Explicitly not findings (known items referenced, not refiled)

These are already captured elsewhere; listed so the architect sees they were *seen*, not missed. **None is a new discovery.**

| Known item | Where it lives | This audit's note |
|---|---|---|
| **#20 mypy six** | `BACKLOG.md:48` (Epic C) | The six are the SDK-typing errors in `openai_mini`/`openai_deep`/`grok` research (§2.8). Not refiled. `--strict` surfaces ~13 *more* trivial bare-generic/annotation errors the project gate tolerates — noted as coverage evidence only, not refiled. |
| **`utcnow()` sweep** | *(no doc citation exists)* | 5 sites counted (§2.9). **Transparency flag:** the audit brief treats this as a "known deferred sweep," but a repo-wide search finds it recorded in **no** doc/JOURNAL/LESSONS/audit — only in code. Not refiled as a discovery; the architect may want to formally file it, since there is currently nothing to defer *to*. |
| **D2 parity gaps** | CONTRACT §7; fable-audit D2; L-INT | `cli.py:565-567` (`--file` raw read, frontmatter leak) + `research/runner.py:133-143` (`--return-dir` no-op). Referenced as the *root* SEED-1 also resolves. Not refiled. |
| **`cli.py:388` secrets hazard** | L-DOC §7.6 / DRAFT-DOC-3 | `load_dotenv(override=False)` at `:388-389` — empty-string credential wins; DRAFT-DOC-3 fixes. Confirmed present first-hand; not refiled. |
| **Flake W1** | L-GOV W1; JOURNAL | `test_inbox_exits_3_when_any_batch_run_degraded` — **lives in `test_research.py:1632`** (not `test_inbox.py`). **Root cause (verified, §3/§4.5):** the test is non-hermetic — it invokes real `cli.main` without disabling `scan_downloads` (`settings.yaml:31`) or overriding `downloads_dir` (`~/Downloads`), and `load_config` is uncached — so it scans the operator's real `~/Downloads`, explaining the n=1 order-dependence **and** implying a latent side effect (processed downloads files get archived). This is the requested root-cause assessment, not a fix and not a refile — but it upgrades W1 from "ordering flake" to "test-isolation defect." |
| Other deferred (exit-0 under-reporting; minority-report heading-heuristic D13 `output.py:381,403`; F12 stale research pin; `claude-sonnet` unused seat; ADR-03 blind-critique measurement M3; #9/#18/#19 baseline-gated) | Lane docs / BACKLOG | All design/functional items, already captured. Not implementation-quality defects. Referenced, not refiled. |

---

*End of audit. Zero code, config, or test files were modified. `git status` clean apart from this report.*
